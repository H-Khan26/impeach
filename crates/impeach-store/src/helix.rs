//! [`HelixStore`]: [`GraphStore`] over HelixDB v3 through the `helix-db` Rust SDK.
//!
//! Python origin: `storage/helix/{client,connection,queries}.py`.
//!
//! - Modes: `embedded` (on disk under `<data_dir>/<namespace>`), `memory` (in process,
//!   discarded on close) and `server` (HTTP client of a HelixDB v3 server at `url`).
//! - One process holds at most one handle per `(mode, location, namespace)`:
//!   [`HelixStore::open`] returns the live handle when one exists (an embedded database
//!   accepts a single writer).
//! - An embedded handle holds an exclusive, non-blocking OS lock on
//!   `<data_dir>/<namespace>.lock` (via `fs4`), so a second writer process fails at open
//!   instead of fencing the first.
//! - Opening creates the vector and text indexes (idempotent) and waits up to
//!   [`INDEX_BUILD_TIMEOUT`] for pending builds to activate, then reads or creates the
//!   `StoreInfo` node holding the store id.
//! - Every request is one transaction. Transaction conflicts (and index backpressure, which
//!   the SDK marks retryable) are retried up to [`TRANSACTION_CONFLICT_ATTEMPTS`] times with
//!   capped, jittered exponential backoff. A fenced or closed handle fails with
//!   [`StoreError::DatabaseLost`] now and on every later call.
//!
//! The SDK is async; the store owns a small tokio runtime (worker threads with a 16 MiB stack,
//! which the embedded engine needs in debug builds) and blocks the calling thread on a
//! channel, so it is safe to call from inside another async runtime as well.

use std::collections::{BTreeSet, HashMap};
use std::fs::{File, OpenOptions};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, OnceLock, PoisonError, RwLock, Weak};

use impeach_core::util::lock;
use std::time::{Duration, Instant};

use helix_db::dsl::prelude::*;
use helix_db::{Client, HelixDbSource, HelixError};
use impeach_core::{ChunkId, EntityId, FactId, NodeId, SpanId};
use impeach_text::retry::{Jitter, RandomJitter};
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use crate::error::{Result, StoreError};
use crate::graph::{
    clamp_limit, merge_hits, node_properties, or_default, parse_node_id, query_names,
    validate_confidence, validate_vector, ChunkNode, DocumentNode, EmbeddingWrite, EntityHit,
    EntityNode, FactHit, FactNode, SearchHit, SpanNode, Stored, StoredChunk, StoredDocument,
    StoredEntity, StoredFact, StoredSpan, VectorLabel, CHUNK_OF_DOCUMENT, DEFAULT_FACT_ENTITY_ROLE,
    DEFAULT_FACT_SPAN_ROLE, DEFAULT_SPAN_ENTITY_METHOD, DOCUMENT, DOCUMENT_CHUNK,
    EMBEDDING_PROPERTY, ENTITY, ENTITY_NAME_SEARCH_K, FACT, FACT_ABOUT_ENTITY,
    FACT_DERIVED_FROM_SPAN, FACT_IN_DOCUMENT, SPAN, SPAN_OF_CHUNK, SPAN_REFERS_TO_ENTITY,
    STORE_ID_PROPERTY, STORE_INFO, TEXT_INDEXES, VECTOR_INDEXED_LABELS,
};
use crate::store::GraphStore;

/// How long opening waits for pending index builds before logging a warning and continuing.
pub const INDEX_BUILD_TIMEOUT: Duration = Duration::from_secs(60);
/// Poll interval while waiting for index builds.
const INDEX_POLL_INTERVAL: Duration = Duration::from_millis(50);
/// Attempts per request while it fails with a transaction conflict.
pub const TRANSACTION_CONFLICT_ATTEMPTS: u32 = 20;
/// First conflict backoff, seconds.
const CONFLICT_BACKOFF_BASE_S: f64 = 0.005;
/// Largest conflict backoff, seconds.
const CONFLICT_BACKOFF_CAP_S: f64 = 0.5;
/// Items written per transaction by the batch methods.
pub const BATCH_WRITE_SIZE: usize = 100;
/// Stack size of the runtime's worker threads.
const WORKER_STACK_BYTES: usize = 16 * 1024 * 1024;

/// Where the database lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HelixMode {
    /// On disk under `<data_dir>/<namespace>`.
    Embedded,
    /// In process, discarded when the store closes.
    Memory,
    /// A remote HelixDB v3 server.
    Server,
}

impl HelixMode {
    /// Config value (`embedded`, `memory`, `server`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Embedded => "embedded",
            Self::Memory => "memory",
            Self::Server => "server",
        }
    }
}

impl From<impeach_config::pipeline::HelixMode> for HelixMode {
    fn from(mode: impeach_config::pipeline::HelixMode) -> Self {
        match mode {
            impeach_config::pipeline::HelixMode::Embedded => Self::Embedded,
            impeach_config::pipeline::HelixMode::Memory => Self::Memory,
            impeach_config::pipeline::HelixMode::Server => Self::Server,
        }
    }
}

/// Settings for [`HelixStore::open`] (the pipeline's `helix:` block).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HelixOptions {
    /// Database mode.
    pub mode: HelixMode,
    /// Parent directory of embedded databases (relative paths resolve against the working
    /// directory).
    pub data_dir: PathBuf,
    /// Server URL (server mode).
    pub url: Option<String>,
    /// Database name; in server mode only a label of the connection.
    pub namespace: String,
    /// Bearer token for server mode; empty means none.
    pub api_key: Option<String>,
    /// Dimension of the vector indexes.
    pub embedding_dim: usize,
}

impl HelixOptions {
    /// In-memory database `namespace`.
    pub fn memory(namespace: impl Into<String>, embedding_dim: usize) -> Self {
        Self {
            mode: HelixMode::Memory,
            data_dir: PathBuf::from("data/helix"),
            url: None,
            namespace: namespace.into(),
            api_key: None,
            embedding_dim,
        }
    }

    /// On-disk database `<data_dir>/<namespace>`.
    pub fn embedded(
        data_dir: impl Into<PathBuf>,
        namespace: impl Into<String>,
        embedding_dim: usize,
    ) -> Self {
        Self {
            mode: HelixMode::Embedded,
            data_dir: data_dir.into(),
            ..Self::memory(namespace, embedding_dim)
        }
    }

    /// Server at `url`.
    pub fn server(
        url: impl Into<String>,
        api_key: Option<String>,
        namespace: impl Into<String>,
        embedding_dim: usize,
    ) -> Self {
        Self {
            mode: HelixMode::Server,
            url: Some(url.into()),
            api_key,
            ..Self::memory(namespace, embedding_dim)
        }
    }

    /// Options from the pipeline config.
    #[must_use]
    pub fn from_config(config: &impeach_config::pipeline::HelixConfig) -> Self {
        Self {
            mode: config.mode.into(),
            data_dir: PathBuf::from(&config.data_dir),
            url: config.url.clone(),
            namespace: config.namespace.clone(),
            api_key: Some(config.api_key.clone()).filter(|key| !key.is_empty()),
            embedding_dim: config.embedding_dim,
        }
    }

    /// Canonical data directory (embedded; created if missing), `memory`, or the server URL
    /// without a trailing slash.
    fn location(&self) -> Result<String> {
        match self.mode {
            HelixMode::Embedded => {
                let dir = if self.data_dir.as_os_str().is_empty() {
                    PathBuf::from("data/helix")
                } else {
                    self.data_dir.clone()
                };
                // Canonical so every spelling of one directory shares a registry entry.
                std::fs::create_dir_all(&dir)?;
                let canonical = strip_verbatim_prefix(std::fs::canonicalize(&dir)?);
                Ok(canonical.to_string_lossy().into_owned())
            }
            HelixMode::Memory => Ok("memory".to_owned()),
            HelixMode::Server => match self.url.as_deref().filter(|url| !url.is_empty()) {
                Some(url) => Ok(url.trim_end_matches('/').to_owned()),
                None => Err(StoreError::Connection("Helix server mode requires a url".into())),
            },
        }
    }
}

type RegistryKey = (HelixMode, String, String);

/// Windows verbatim path prefix: backslash, backslash, question mark, backslash.
const VERBATIM_PREFIX: &str = concat!('\\', '\\', '?', '\\');

/// Drops the verbatim prefix (`\\?\`) that `canonicalize` adds on Windows when the rest is a
/// plain drive path (`C:\...`), so the location reads like an ordinary path.
fn strip_verbatim_prefix(path: PathBuf) -> PathBuf {
    let text = path.to_string_lossy();
    match text.strip_prefix(VERBATIM_PREFIX) {
        Some(rest) if rest.as_bytes().get(1) == Some(&b':') => PathBuf::from(rest),
        _ => path,
    }
}

static REGISTRY: LazyLock<Mutex<HashMap<RegistryKey, Weak<HelixStore>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Graph store backed by HelixDB.
pub struct HelixStore {
    key: RegistryKey,
    embedding_dim: usize,
    client: Client,
    runtime: Option<tokio::runtime::Runtime>,
    /// Set once the handle is closed or fenced; every later request fails with it.
    lost: OnceLock<String>,
    /// Held for the duration of one request so requests run one at a time.
    request_lock: Mutex<()>,
    /// Indexes already reported as missing.
    warned_indexes: Mutex<BTreeSet<String>>,
    store_id: RwLock<String>,
    file_lock: Mutex<Option<File>>,
}

impl std::fmt::Debug for HelixStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HelixStore")
            .field("mode", &self.key.0)
            .field("location", &self.key.1)
            .field("namespace", &self.key.2)
            .field("embedding_dim", &self.embedding_dim)
            .finish_non_exhaustive()
    }
}

impl HelixStore {
    /// Returns the shared store for this database, opening and preparing it on first use.
    ///
    /// # Errors
    /// [`StoreError::Connection`] when the mode is incomplete, the lock is held by another
    /// process, the database cannot be opened, an index build is blocked or aborted, or the
    /// database is already open here with another `embedding_dim`; [`StoreError::Io`] for
    /// data-directory failures.
    pub fn open(options: &HelixOptions) -> Result<Arc<Self>> {
        let key = (options.mode, options.location()?, options.namespace.clone());
        let mut registry = lock(&REGISTRY);
        if let Some(store) = registry.get(&key).and_then(Weak::upgrade) {
            if store.lost.get().is_none() {
                if store.embedding_dim != options.embedding_dim {
                    return Err(StoreError::Connection(format!(
                        "Helix database {} is already open with embedding_dim={}, requested {}",
                        store.describe(),
                        store.embedding_dim,
                        options.embedding_dim
                    )));
                }
                return Ok(store);
            }
        }
        let store = Arc::new(Self::open_new(key.clone(), options)?);
        registry.insert(key, Arc::downgrade(&store));
        tracing::info!(
            "Opened Helix database {} (store_id={})",
            store.describe(),
            store.store_id()
        );
        Ok(store)
    }

    /// [`HelixStore::open`] with options from the pipeline config.
    ///
    /// # Errors
    /// As [`HelixStore::open`].
    pub fn from_config(config: &impeach_config::pipeline::HelixConfig) -> Result<Arc<Self>> {
        Self::open(&HelixOptions::from_config(config))
    }

    fn open_new(key: RegistryKey, options: &HelixOptions) -> Result<Self> {
        let dim = NonZeroUsize::new(options.embedding_dim).ok_or_else(|| {
            StoreError::validation("embedding_dim", "0", "helix.embedding_dim must be positive")
        })?;
        let file_lock = match key.0 {
            HelixMode::Embedded => Some(acquire_lock(Path::new(&key.1), &key.2)?),
            HelixMode::Memory | HelixMode::Server => None,
        };
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("helix-store")
            .thread_stack_size(WORKER_STACK_BYTES)
            .enable_all()
            .build()?;
        let client = open_client(&runtime, &key, options.api_key.as_deref())?;
        let store = Self {
            key,
            embedding_dim: dim.get(),
            client,
            runtime: Some(runtime),
            lost: OnceLock::new(),
            request_lock: Mutex::new(()),
            warned_indexes: Mutex::new(BTreeSet::new()),
            store_id: RwLock::new(String::new()),
            file_lock: Mutex::new(file_lock),
        };
        let prepared = store.ensure_indexes(dim).and_then(|()| store.ensure_store_info());
        match prepared {
            Ok(store_id) => {
                *store.store_id.write().unwrap_or_else(PoisonError::into_inner) = store_id;
                Ok(store)
            }
            Err(error) => {
                if let Err(close_error) = store.close() {
                    tracing::debug!(
                        "closing a Helix database that failed to prepare: {close_error}"
                    );
                }
                Err(match error {
                    StoreError::Connection(_) | StoreError::DatabaseLost(_) => error,
                    other => StoreError::Connection(format!(
                        "Failed to prepare Helix database {}: {other}",
                        store.describe()
                    )),
                })
            }
        }
    }

    /// Database mode.
    #[must_use]
    pub fn mode(&self) -> HelixMode {
        self.key.0
    }

    /// Absolute data directory (embedded), `memory`, or the server URL.
    #[must_use]
    pub fn location(&self) -> &str {
        &self.key.1
    }

    /// Database name.
    #[must_use]
    pub fn namespace(&self) -> &str {
        &self.key.2
    }

    fn describe(&self) -> String {
        format!("({}, {}, {})", self.key.0.as_str(), self.key.1, self.key.2)
    }

    /// Flushes and closes the database and releases the file lock. Every handle sharing this
    /// store then fails with [`StoreError::DatabaseLost`]; a later [`HelixStore::open`] opens
    /// a fresh handle. Closing twice is a no-op.
    ///
    /// # Errors
    /// [`StoreError::Connection`] when the embedded database fails to flush.
    pub fn close(&self) -> Result<()> {
        if self.lost.set(format!("Helix database {} was closed", self.describe())).is_err() {
            return Ok(());
        }
        let result = if self.key.0 == HelixMode::Server {
            Ok(())
        } else {
            let client = self.client.clone();
            self.run(async move { client.close().await }).and_then(|closed| {
                closed.map_err(|error| {
                    StoreError::Connection(format!(
                        "Failed to close Helix database {}: {error}",
                        self.describe()
                    ))
                })
            })
        };
        if let Some(file) = lock(&self.file_lock).take() {
            if let Err(error) = fs4::FileExt::unlock(&file) {
                tracing::debug!("unlocking the Helix lock file failed: {error}");
            }
        }
        result
    }

    /// Runs `future` on the store's runtime and waits for its output.
    fn run<T, F>(&self, future: F) -> Result<T>
    where
        T: Send + 'static,
        F: std::future::Future<Output = T> + Send + 'static,
    {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        self.runtime()?.spawn(async move {
            // The receiver only disappears when the caller is gone.
            let _ = sender.send(future.await);
        });
        receiver.recv().map_err(|_| {
            StoreError::Connection(format!("Helix request on {} was cancelled", self.describe()))
        })
    }

    /// Sends one request (one transaction), retrying transaction conflicts. The request lock
    /// is held only while a request runs, not during the backoff between attempts.
    fn execute(&self, query: &str, request: &QueryRequest) -> Result<Value> {
        let mut jitter = RandomJitter::default();
        let mut attempt = 0;
        loop {
            if let Some(lost) = self.lost.get() {
                return Err(StoreError::DatabaseLost(lost.clone()));
            }
            let outcome = {
                let _request = lock(&self.request_lock);
                let client = self.client.clone();
                let request = request.clone();
                self.run(async move { client.query::<Value>(request).send().await })?
            };
            let error = match outcome {
                Ok(value) => return Ok(value),
                Err(error) => error,
            };
            if is_database_lost(&error) {
                let message = format!(
                    "Helix database {} is no longer usable ({error}); it was closed or another \
                     process opened the same embedded database",
                    self.describe()
                );
                tracing::error!("{message}");
                let _ = self.lost.set(message.clone());
                return Err(StoreError::DatabaseLost(message));
            }
            if !is_retryable(&error) {
                if error.error_code() == Some("index_not_found")
                    && lock(&self.warned_indexes).insert(query.to_owned())
                {
                    tracing::warn!(
                        "Helix index for {query} is missing or not active yet; searches on it \
                         fail until it is built: {error}"
                    );
                }
                return Err(map_error(query, &error));
            }
            if attempt + 1 >= TRANSACTION_CONFLICT_ATTEMPTS {
                return Err(StoreError::Conflict {
                    attempts: TRANSACTION_CONFLICT_ATTEMPTS,
                    message: error.to_string(),
                });
            }
            std::thread::sleep(conflict_backoff(attempt, &mut jitter));
            attempt += 1;
        }
    }

    fn read(&self, query: &str, batch: ReadBatch) -> Result<Map<String, Value>> {
        object(self.execute(query, &QueryRequest::read(batch))?)
    }

    fn write(&self, query: &str, batch: WriteBatch) -> Result<Map<String, Value>> {
        object(self.execute(query, &QueryRequest::write(batch))?)
    }

    /// Creates every index (idempotent) and waits until pending builds activate.
    fn ensure_indexes(&self, dim: NonZeroUsize) -> Result<()> {
        let mut batch = write_batch();
        let mut names = Vec::new();
        for label in VECTOR_INDEXED_LABELS {
            let name = format!("vector_{label}");
            batch = batch.var_as(
                &name,
                g().create_vector_index_nodes(
                    label,
                    EMBEDDING_PROPERTY,
                    dim,
                    VectorDistanceMetric::Cosine,
                    None::<&str>,
                ),
            );
            names.push(name);
        }
        for (label, property) in TEXT_INDEXES {
            let name = format!("text_{label}_{property}");
            batch = batch.var_as(&name, g().create_text_index_nodes(label, property, None::<&str>));
            names.push(name);
        }
        let receipts = self.write("CreateIndexes", batch.returning(names))?;
        let mut pending: Vec<(String, String)> = receipts
            .iter()
            .filter(|(_, receipt)| {
                matches!(receipt["kind"].as_str(), Some("accepted" | "existing_operation"))
            })
            .filter_map(|(name, receipt)| {
                receipt["operation_id"].as_str().map(|id| (name.clone(), id.to_owned()))
            })
            .collect();
        let deadline = Instant::now() + INDEX_BUILD_TIMEOUT;
        while !pending.is_empty() {
            let mut still_pending = Vec::new();
            for (name, operation_id) in pending {
                let status = self.read(
                    "GetIndexOperation",
                    read_batch()
                        .var_as("s", g().get_index_operation(operation_id.as_str()))
                        .returning(["s"]),
                )?;
                let status = &status["s"];
                match status["status"].as_str() {
                    Some("succeeded") => {}
                    Some(state @ ("blocked" | "aborted")) => {
                        return Err(StoreError::Connection(
                            format!(
                                "Helix index build '{name}' is {state}: {} {}",
                                status["blocker_code"].as_str().unwrap_or_default(),
                                status["message"].as_str().unwrap_or_default()
                            )
                            .trim()
                            .to_owned(),
                        ));
                    }
                    _ => still_pending.push((name, operation_id)),
                }
            }
            pending = still_pending;
            if pending.is_empty() {
                break;
            }
            if Instant::now() > deadline {
                let names: Vec<&str> = pending.iter().map(|(name, _)| name.as_str()).collect();
                tracing::warn!(
                    "Helix index builds still running after {}s: {}; searches may fail until they \
                     finish",
                    INDEX_BUILD_TIMEOUT.as_secs(),
                    names.join(", ")
                );
                break;
            }
            std::thread::sleep(INDEX_POLL_INTERVAL);
        }
        Ok(())
    }

    /// Returns this database's store id, creating the `StoreInfo` node on first open.
    fn ensure_store_info(&self) -> Result<String> {
        let new_id = uuid::Uuid::new_v4().simple().to_string();
        let result = self.write(
            "EnsureStoreInfo",
            write_batch()
                .var_as(
                    "found",
                    g().n_with_label(STORE_INFO)
                        .limit(1_usize)
                        .value_map(Some(vec![STORE_ID_PROPERTY])),
                )
                .var_as_if(
                    "created",
                    BatchCondition::VarEmpty("found".to_owned()),
                    g().add_n(STORE_INFO, vec![(STORE_ID_PROPERTY, new_id.as_str())])
                        .value_map(Some(vec![STORE_ID_PROPERTY])),
                )
                .returning(["found", "created"]),
        )?;
        Ok(rows(&result, "found")
            .iter()
            .chain(rows(&result, "created"))
            .find_map(|row| row[STORE_ID_PROPERTY].as_str().map(str::to_owned))
            .unwrap_or(new_id))
    }

    fn create_node<I: From<String>, T: DeserializeOwned>(
        &self,
        query: &str,
        label: &'static str,
        properties: Vec<(&'static str, PropertyValue)>,
    ) -> Result<Stored<I, T>> {
        let result = self.write(
            query,
            write_batch()
                .var_as("node", g().add_n(label, properties).value_map(Some(fields(label))))
                .returning(["node"]),
        )?;
        decode(single(&result, "node", label, "(new)")?)
    }

    fn update_node<I: From<String>, T: DeserializeOwned>(
        &self,
        query: &str,
        label: &'static str,
        field: &str,
        id: &str,
        properties: Vec<(&'static str, PropertyValue)>,
    ) -> Result<Stored<I, T>> {
        let node_id = parse_node_id(field, id)?;
        let mut properties = properties.into_iter();
        let (name, value) = properties
            .next()
            .ok_or_else(|| StoreError::validation(field, id, "no properties to update"))?;
        let traversal = properties.fold(
            g().n(NodeRef::id(node_id)).has_label(label).set_property(name, value),
            |traversal, (name, value)| traversal.set_property(name, value),
        );
        let result = self.write(
            query,
            write_batch()
                .var_as("node", traversal.value_map(Some(fields(label))))
                .returning(["node"]),
        )?;
        decode(single(&result, "node", label, id)?)
    }

    /// Creates a node and its edge to an existing parent atomically; nothing is written when
    /// the parent is missing.
    #[allow(clippy::too_many_arguments)]
    fn create_under<I: From<String>, T: DeserializeOwned>(
        &self,
        query: &str,
        label: &'static str,
        properties: Vec<(&'static str, PropertyValue)>,
        edge_label: &'static str,
        parent_label: &'static str,
        parent_field: &str,
        parent_id: &str,
        edge_properties: Vec<(&'static str, PropertyValue)>,
    ) -> Result<Stored<I, T>> {
        let parent = parse_node_id(parent_field, parent_id)?;
        let batch =
            write_batch().var_as("parent", g().n(NodeRef::id(parent)).has_label(parent_label));
        let batch = add_child(batch, "parent", 0, label, properties, edge_label, edge_properties)
            .returning(["parent", "props0"]);
        let result = self.write(query, batch)?;
        single(&result, "parent", parent_label, parent_id)?;
        decode(single(&result, "props0", label, "(new)")?)
    }

    /// Ids among `ids` that are `label` nodes.
    fn existing(&self, query: &str, label: &str, ids: &[u64]) -> Result<BTreeSet<u64>> {
        if ids.is_empty() {
            return Ok(BTreeSet::new());
        }
        let result = self.read(
            query,
            read_batch()
                .var_as(
                    "found",
                    g().n(NodeRef::ids(ids.iter().copied()))
                        .has_label(label)
                        .value_map(Some(vec!["$id"])),
                )
                .returning(["found"]),
        )?;
        Ok(rows(&result, "found").iter().filter_map(|row| row["$id"].as_u64()).collect())
    }

    fn set_embedding(
        &self,
        query: &str,
        label: VectorLabel,
        id: &str,
        vector: &[f32],
    ) -> Result<()> {
        validate_vector(vector, self.embedding_dim)?;
        let node_id = parse_node_id(label.id_field(), id)?;
        let result = self.write(
            query,
            write_batch()
                .var_as(
                    "node",
                    g().n(NodeRef::id(node_id))
                        .has_label(label.as_str())
                        .set_property(EMBEDDING_PROPERTY, PropertyValue::from(vector.to_vec()))
                        .value_map(Some(vec!["$id"])),
                )
                .returning(["node"]),
        )?;
        single(&result, "node", label.as_str(), id)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn link(
        &self,
        query: &str,
        (from_label, from_field, from_id): (&'static str, &str, &str),
        (to_label, to_field, to_id): (&'static str, &str, &str),
        edge_label: &'static str,
        edge_properties: Vec<(&'static str, PropertyValue)>,
    ) -> Result<()> {
        let from = parse_node_id(from_field, from_id)?;
        let to = parse_node_id(to_field, to_id)?;
        let result = self.write(
            query,
            write_batch()
                .var_as("src", g().n(NodeRef::id(from)).has_label(from_label))
                .var_as("dst", g().n(NodeRef::id(to)).has_label(to_label))
                .var_as_if(
                    "edge",
                    BatchCondition::VarNotEmpty("dst".to_owned()),
                    g().n(NodeRef::var("src"))
                        .add_e(edge_label, NodeRef::var("dst"), edge_properties)
                        .count(),
                )
                .returning(["src", "dst", "edge"]),
        )?;
        single(&result, "src", from_label, from_id)?;
        single(&result, "dst", to_label, to_id)?;
        Ok(())
    }

    fn vector_search<T: DeserializeOwned, I: From<String>>(
        &self,
        query: &str,
        label: &'static str,
        vector: &[f32],
        limit: usize,
    ) -> Result<Vec<SearchHit<Stored<I, T>>>> {
        validate_vector(vector, self.embedding_dim)?;
        let limit = clamp_limit(limit);
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut projection = fields(label);
        projection.push("$distance");
        let result = self.read(
            query,
            read_batch()
                .var_as(
                    "hits",
                    g().vector_search_nodes(
                        label,
                        EMBEDDING_PROPERTY,
                        vector.to_vec(),
                        limit,
                        None,
                    )
                    .value_map(Some(projection)),
                )
                .returning(["hits"]),
        )?;
        rows(&result, "hits")
            .iter()
            .map(|row| {
                // Cosine distance is (1 - cos) / 2; callers expect cosine similarity.
                let distance = row["$distance"].as_f64().unwrap_or(1.0);
                let score = (1.0 - 2.0 * distance).clamp(0.0, 1.0);
                Ok(SearchHit { node: decode(row)?, score })
            })
            .collect()
    }

    /// BM25 search over one or more text-indexed properties, merged per node (best score
    /// wins), best `limit` first.
    fn text_search<T: DeserializeOwned, I: From<String>>(
        &self,
        query: &str,
        label: &'static str,
        properties: &[&'static str],
        text: &str,
        limit: usize,
    ) -> Result<Vec<SearchHit<Stored<I, T>>>> {
        let text = text.trim();
        if text.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let mut projection = fields(label);
        projection.push("$score");
        let mut batch = read_batch();
        let mut names = Vec::new();
        for (index, property) in properties.iter().enumerate() {
            let name = format!("hits{index}");
            batch = batch.var_as(
                &name,
                g().text_search_nodes(label, *property, text, limit, None)
                    .value_map(Some(projection.clone())),
            );
            names.push(name);
        }
        let result = self.read(query, batch.returning(names.clone()))?;
        let mut by_id: HashMap<u64, &Value> = HashMap::new();
        let lists = names
            .iter()
            .map(|name| {
                rows(&result, name)
                    .iter()
                    .filter_map(|row| {
                        let id = row["$id"].as_u64()?;
                        by_id.entry(id).or_insert(row);
                        Some((id, row["$score"].as_f64().unwrap_or(0.0)))
                    })
                    .collect()
            })
            .collect();
        merge_hits(lists, limit)
            .into_iter()
            .filter_map(|(id, score)| by_id.get(&id).map(|row| (*row, score)))
            .map(|(row, score)| Ok(SearchHit { node: decode(row)?, score }))
            .collect()
    }

    fn runtime(&self) -> Result<&tokio::runtime::Runtime> {
        self.runtime.as_ref().ok_or_else(|| {
            StoreError::DatabaseLost(format!("Helix database {} was dropped", self.describe()))
        })
    }
}

impl Drop for HelixStore {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            tracing::warn!("closing Helix database {} failed: {error}", self.describe());
        }
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}

/// Appends the entries creating child `index` under the node in variable `parent`:
/// `node{index}` (the node), `edge{index}` (edge count) and `props{index}` (its properties).
fn add_child(
    batch: WriteBatch,
    parent: &str,
    index: usize,
    label: &'static str,
    properties: Vec<(&'static str, PropertyValue)>,
    edge_label: &'static str,
    edge_properties: Vec<(&'static str, PropertyValue)>,
) -> WriteBatch {
    let node = format!("node{index}");
    let condition = || BatchCondition::VarNotEmpty(parent.to_owned());
    batch
        .var_as_if(&node, condition(), g().add_n(label, properties))
        .var_as_if(
            &format!("edge{index}"),
            condition(),
            g().n(NodeRef::var(node.as_str()))
                .add_e(edge_label, NodeRef::var(parent), edge_properties)
                .count(),
        )
        .var_as_if(
            &format!("props{index}"),
            condition(),
            g().n(NodeRef::var(node.as_str())).value_map(Some(fields(label))),
        )
}

fn acquire_lock(dir: &Path, namespace: &str) -> Result<File> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{namespace}.lock"));
    let file =
        OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&path)?;
    match fs4::FileExt::try_lock(&file) {
        Ok(()) => Ok(file),
        Err(fs4::TryLockError::WouldBlock) => Err(StoreError::Connection(format!(
            "Helix database lock {} is held by another process; an embedded database accepts \
             one writer process at a time (stop the other process, use a different \
             data_dir/namespace, or run a Helix server)",
            path.display()
        ))),
        Err(fs4::TryLockError::Error(error)) => Err(StoreError::Io(error)),
    }
}

fn open_client(
    runtime: &tokio::runtime::Runtime,
    key: &RegistryKey,
    api_key: Option<&str>,
) -> Result<Client> {
    let (mode, location, namespace) = key;
    let failed = |error: &dyn std::fmt::Display| {
        StoreError::Connection(format!(
            "Failed to open Helix database ({}: {location}, namespace={namespace}): {error}",
            mode.as_str()
        ))
    };
    let source = match mode {
        HelixMode::Server => {
            return Client::new(Some(location))
                .map(|client| client.with_api_key(api_key.filter(|key| !key.is_empty())))
                .map_err(|error| failed(&error));
        }
        HelixMode::Embedded => {
            std::fs::create_dir_all(location)?;
            HelixDbSource::Disk { root: PathBuf::from(location), database: namespace.clone() }
        }
        HelixMode::Memory => HelixDbSource::InMemory { database: namespace.clone() },
    };
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    runtime.spawn(async move {
        let _ = sender.send(Client::open(source).await);
    });
    receiver.recv().map_err(|error| failed(&error))?.map_err(|error| failed(&error))
}

/// Capped exponential backoff with jitter: half fixed, half random.
fn conflict_backoff(attempt: u32, jitter: &mut dyn Jitter) -> Duration {
    let delay =
        (CONFLICT_BACKOFF_BASE_S * 2_f64.powi(attempt.min(30) as i32)).min(CONFLICT_BACKOFF_CAP_S);
    Duration::from_secs_f64(delay / 2.0 + jitter.sample(delay / 2.0))
}

/// Transaction conflicts (embedded or HTTP 409) and index backpressure: the request did not
/// commit and can be resent unchanged.
fn is_retryable(error: &HelixError) -> bool {
    error.error_code() == Some("transaction_conflict")
        || error.is_conflict()
        || error.is_index_backpressure()
        || error.to_string().to_lowercase().contains("transaction conflict")
}

/// The embedded handle was fenced by a newer client or closed underneath us.
fn is_database_lost(error: &HelixError) -> bool {
    let HelixError::EmbeddedError { code, details } = error else {
        return false;
    };
    let text = details.to_lowercase();
    matches!(code.as_str(), "database_closed" | "writer_fenced_commit_outcome_unknown")
        || text.contains("detected newer db client")
        || text.contains("closed error")
}

fn map_error(query: &str, error: &HelixError) -> StoreError {
    match error {
        HelixError::ReqwestError(_) | HelixError::InvalidURL(_) => {
            StoreError::Connection(format!("Query '{query}' could not reach Helix: {error}"))
        }
        _ if error.error_code() == Some("zero_norm_cosine_vector") => {
            StoreError::validation("embedding_vector", "", error.to_string())
        }
        _ => StoreError::Query { query: query.to_owned(), message: error.to_string() },
    }
}

fn object(value: Value) -> Result<Map<String, Value>> {
    match value {
        Value::Object(map) => Ok(map),
        Value::Null => Ok(Map::new()),
        other => Err(StoreError::Query {
            query: String::new(),
            message: format!("unexpected Helix response: {other}"),
        }),
    }
}

/// Rows of a returned variable; missing, `null` and non-array values read as no rows.
fn rows<'a>(result: &'a Map<String, Value>, var: &str) -> &'a [Value] {
    result.get(var).and_then(Value::as_array).map_or(&[], Vec::as_slice)
}

fn single<'a>(
    result: &'a Map<String, Value>,
    var: &str,
    label: &str,
    id: &str,
) -> Result<&'a Value> {
    rows(result, var).first().ok_or_else(|| StoreError::not_found(label, id))
}

/// `$id`, `$label` and the label's properties.
fn fields(label: &str) -> Vec<&'static str> {
    let mut fields = vec!["$id", "$label"];
    fields.extend_from_slice(node_properties(label).unwrap_or_default());
    fields
}

/// Decodes a `value_map` row into a stored node; `$`-prefixed keys other than `$id` are
/// dropped and missing properties take their defaults.
fn decode<I: From<String>, T: DeserializeOwned>(row: &Value) -> Result<Stored<I, T>> {
    let malformed = |message: String| StoreError::Query { query: String::new(), message };
    let object =
        row.as_object().ok_or_else(|| malformed(format!("expected a node row, got {row}")))?;
    let id = match &object.get("$id") {
        Some(Value::Number(number)) => number.to_string(),
        Some(Value::String(text)) => text.clone(),
        _ => return Err(malformed(format!("node row without $id: {row}"))),
    };
    let properties: Map<String, Value> = object
        .iter()
        .filter(|(key, _)| !key.starts_with('$'))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let node = serde_json::from_value(Value::Object(properties))
        .map_err(|error| malformed(format!("cannot decode node {id}: {error}")))?;
    Ok(Stored { id: I::from(id), node })
}

fn document_properties(document: &DocumentNode) -> Vec<(&'static str, PropertyValue)> {
    vec![
        ("DocType", document.doc_type.as_str().into()),
        ("Title", document.title.as_str().into()),
        ("DateCreated", document.date_created.as_str().into()),
        ("FilePath", document.file_path.as_str().into()),
        ("TotalPages", document.total_pages.into()),
        ("Metadata", document.metadata.as_str().into()),
    ]
}

fn chunk_properties(chunk: &ChunkNode) -> Vec<(&'static str, PropertyValue)> {
    vec![
        ("Text", chunk.text.as_str().into()),
        ("PageNumber", chunk.page_number.into()),
        ("StartChar", chunk.start_char.into()),
        ("EndChar", chunk.end_char.into()),
        ("Section", chunk.section.as_str().into()),
        ("SequenceNumber", chunk.sequence_number.into()),
        ("Metadata", chunk.metadata.as_str().into()),
    ]
}

fn span_properties(span: &SpanNode, chunk_id: u64) -> Vec<(&'static str, PropertyValue)> {
    vec![
        ("DocId", span.doc_id.as_str().into()),
        ("ChunkId", chunk_id.to_string().into()),
        ("StartChar", span.start_char.into()),
        ("EndChar", span.end_char.into()),
        ("SpanType", span.span_type.as_str().into()),
        ("SpanLabel", span.span_label.as_str().into()),
        ("SurfaceText", span.surface_text.as_str().into()),
        ("Metadata", span.metadata.as_str().into()),
    ]
}

fn entity_properties(entity: &EntityNode) -> Vec<(&'static str, PropertyValue)> {
    vec![
        ("Name", entity.name.as_str().into()),
        ("EntityType", entity.entity_type.as_str().into()),
        ("AliasesJson", entity.aliases_json.as_str().into()),
        ("Description", entity.description.as_str().into()),
        ("Metadata", entity.metadata.as_str().into()),
    ]
}

fn fact_properties(fact: &FactNode) -> Vec<(&'static str, PropertyValue)> {
    vec![
        ("FactText", fact.fact_text.as_str().into()),
        ("FactOrigin", fact.fact_origin.as_str().into()),
        ("FactType", fact.fact_type.as_str().into()),
        ("Status", fact.status.as_str().into()),
        ("AssertionTriple", fact.assertion_triple.as_str().into()),
        ("TemporalValidity", fact.temporal_validity.as_str().into()),
        ("Certainty", fact.certainty.into()),
        ("Metadata", fact.metadata.as_str().into()),
        ("ValueType", fact.value_type.as_str().into()),
        ("PropertyType", fact.property_type.as_str().into()),
    ]
}

impl GraphStore for HelixStore {
    fn store_id(&self) -> String {
        self.store_id.read().unwrap_or_else(PoisonError::into_inner).clone()
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn reset(&self) -> Result<()> {
        self.write(
            "Reset",
            write_batch().var_as("dropped", g().n(NodeRef::all()).drop()).returning(["dropped"]),
        )?;
        let store_id = self.ensure_store_info()?;
        *self.store_id.write().unwrap_or_else(PoisonError::into_inner) = store_id;
        Ok(())
    }

    fn create_document(&self, document: &DocumentNode) -> Result<StoredDocument> {
        self.create_node(query_names::CREATE_DOCUMENT, DOCUMENT, document_properties(document))
    }

    fn update_document(&self, doc_id: &NodeId, document: &DocumentNode) -> Result<StoredDocument> {
        self.update_node(
            query_names::UPDATE_DOCUMENT,
            DOCUMENT,
            "doc_id",
            doc_id.as_str(),
            document_properties(document),
        )
    }

    fn get_document_chunks(&self, doc_id: &NodeId) -> Result<Vec<StoredChunk>> {
        let id = parse_node_id("doc_id", doc_id.as_str())?;
        let result = self.read(
            query_names::GET_DOCUMENT_CHUNKS,
            read_batch()
                .var_as(
                    "chunks",
                    g().n(NodeRef::id(id))
                        .has_label(DOCUMENT)
                        .in_(Some(CHUNK_OF_DOCUMENT))
                        .value_map(Some(fields(DOCUMENT_CHUNK))),
                )
                .returning(["chunks"]),
        )?;
        rows(&result, "chunks").iter().map(decode).collect()
    }

    fn create_document_chunk(&self, doc_id: &NodeId, chunk: &ChunkNode) -> Result<StoredChunk> {
        self.create_under(
            query_names::CREATE_DOCUMENT_CHUNK,
            DOCUMENT_CHUNK,
            chunk_properties(chunk),
            CHUNK_OF_DOCUMENT,
            DOCUMENT,
            "doc_id",
            doc_id.as_str(),
            vec![("SequenceNumber", chunk.sequence_number.into())],
        )
    }

    fn update_document_chunk(&self, chunk_id: &ChunkId, chunk: &ChunkNode) -> Result<StoredChunk> {
        self.update_node(
            query_names::UPDATE_DOCUMENT_CHUNK,
            DOCUMENT_CHUNK,
            "chunk_id",
            chunk_id.as_str(),
            chunk_properties(chunk),
        )
    }

    fn create_chunk_embedding(&self, chunk_id: &ChunkId, embedding: &[f32]) -> Result<()> {
        self.set_embedding(
            query_names::CREATE_CHUNK_EMBEDDING,
            VectorLabel::DocumentChunk,
            chunk_id.as_str(),
            embedding,
        )
    }

    fn create_span_for_chunk(&self, span: &SpanNode) -> Result<StoredSpan> {
        let chunk_id = parse_node_id("chunk_id", span.chunk_id.as_str())?;
        self.create_under(
            query_names::CREATE_SPAN_FOR_CHUNK,
            SPAN,
            span_properties(span, chunk_id),
            SPAN_OF_CHUNK,
            DOCUMENT_CHUNK,
            "chunk_id",
            span.chunk_id.as_str(),
            vec![("SequenceNumber", span.start_char.into())],
        )
    }

    fn create_entity(&self, entity: &EntityNode) -> Result<StoredEntity> {
        let entity = entity.validated()?;
        self.create_node(query_names::CREATE_ENTITY, ENTITY, entity_properties(&entity))
    }

    fn create_entity_embedding(&self, entity_id: &EntityId, embedding: &[f32]) -> Result<()> {
        self.set_embedding(
            query_names::CREATE_ENTITY_EMBEDDING,
            VectorLabel::Entity,
            entity_id.as_str(),
            embedding,
        )
    }

    fn find_entity_by_name(&self, name: &str) -> Result<Vec<EntityHit>> {
        self.text_search(
            query_names::FIND_ENTITY_BY_NAME,
            ENTITY,
            &["Name", "AliasesJson"],
            name,
            ENTITY_NAME_SEARCH_K,
        )
    }

    fn find_similar_entities(&self, embedding: &[f32], limit: usize) -> Result<Vec<EntityHit>> {
        self.vector_search(query_names::FIND_SIMILAR_ENTITIES, ENTITY, embedding, limit)
    }

    fn get_all_entities(&self, limit: usize) -> Result<Vec<StoredEntity>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let result = self.read(
            query_names::GET_ALL_ENTITIES,
            read_batch()
                .var_as(
                    "entities",
                    g().n_with_label(ENTITY).range(0_usize, limit).value_map(Some(fields(ENTITY))),
                )
                .returning(["entities"]),
        )?;
        rows(&result, "entities").iter().map(decode).collect()
    }

    fn link_span_to_entity(
        &self,
        span_id: &SpanId,
        entity_id: &EntityId,
        confidence: f64,
        method: &str,
    ) -> Result<()> {
        let confidence = validate_confidence(confidence)?;
        self.link(
            query_names::LINK_SPAN_TO_ENTITY,
            (SPAN, "span_id", span_id.as_str()),
            (ENTITY, "entity_id", entity_id.as_str()),
            SPAN_REFERS_TO_ENTITY,
            vec![
                ("Confidence", confidence.into()),
                ("Method", or_default(method, DEFAULT_SPAN_ENTITY_METHOD).into()),
            ],
        )
    }

    fn create_fact(&self, fact: &FactNode) -> Result<StoredFact> {
        fact.validate()?;
        self.create_node(query_names::CREATE_FACT, FACT, fact_properties(fact))
    }

    fn create_fact_embedding(&self, fact_id: &FactId, embedding: &[f32]) -> Result<()> {
        self.set_embedding(
            query_names::CREATE_FACT_EMBEDDING,
            VectorLabel::Fact,
            fact_id.as_str(),
            embedding,
        )
    }

    fn link_fact_to_document(
        &self,
        fact_id: &FactId,
        doc_id: &NodeId,
        page_number: i64,
        section: &str,
    ) -> Result<()> {
        self.link(
            query_names::LINK_FACT_TO_DOCUMENT,
            (FACT, "fact_id", fact_id.as_str()),
            (DOCUMENT, "doc_id", doc_id.as_str()),
            FACT_IN_DOCUMENT,
            vec![("PageNumber", page_number.into()), ("Section", section.into())],
        )
    }

    fn link_fact_to_entity(
        &self,
        fact_id: &FactId,
        entity_id: &EntityId,
        role: &str,
    ) -> Result<()> {
        self.link(
            query_names::LINK_FACT_TO_ENTITY,
            (FACT, "fact_id", fact_id.as_str()),
            (ENTITY, "entity_id", entity_id.as_str()),
            FACT_ABOUT_ENTITY,
            vec![("Role", or_default(role, DEFAULT_FACT_ENTITY_ROLE).into())],
        )
    }

    fn link_fact_to_span(
        &self,
        fact_id: &FactId,
        span_id: &SpanId,
        role: &str,
        confidence: f64,
    ) -> Result<()> {
        let confidence = validate_confidence(confidence)?;
        self.link(
            query_names::LINK_FACT_TO_SPAN,
            (FACT, "fact_id", fact_id.as_str()),
            (SPAN, "span_id", span_id.as_str()),
            FACT_DERIVED_FROM_SPAN,
            vec![
                ("Role", or_default(role, DEFAULT_FACT_SPAN_ROLE).into()),
                ("Confidence", confidence.into()),
            ],
        )
    }

    fn get_entity_facts(&self, entity_id: &EntityId) -> Result<Vec<StoredFact>> {
        let id = parse_node_id("entity_id", entity_id.as_str())?;
        let result = self.read(
            query_names::GET_ENTITY_FACTS,
            read_batch()
                .var_as(
                    "facts",
                    g().n(NodeRef::id(id))
                        .has_label(ENTITY)
                        .in_(Some(FACT_ABOUT_ENTITY))
                        .value_map(Some(fields(FACT))),
                )
                .returning(["facts"]),
        )?;
        rows(&result, "facts").iter().map(decode).collect()
    }

    fn get_fact_entities(&self, fact_id: &FactId) -> Result<Vec<EntityId>> {
        let id = parse_node_id("fact_id", fact_id.as_str())?;
        let result = self.read(
            query_names::GET_FACT_ENTITIES,
            read_batch()
                .var_as(
                    "entities",
                    g().n(NodeRef::id(id))
                        .has_label(FACT)
                        .out(Some(FACT_ABOUT_ENTITY))
                        .has_label(ENTITY)
                        .value_map(Some(fields(ENTITY))),
                )
                .returning(["entities"]),
        )?;
        let mut ids: Vec<EntityId> = Vec::new();
        for row in rows(&result, "entities") {
            let entity: StoredEntity = decode(row)?;
            if !ids.contains(&entity.id) {
                ids.push(entity.id);
            }
        }
        Ok(ids)
    }

    fn find_facts_by_text(&self, keywords: &str, limit: usize) -> Result<Vec<FactHit>> {
        self.text_search(
            query_names::FIND_FACTS_BY_TEXT,
            FACT,
            &["FactText"],
            keywords,
            clamp_limit(limit),
        )
    }

    fn find_similar_facts(&self, embedding: &[f32], limit: usize) -> Result<Vec<FactHit>> {
        self.vector_search(query_names::FIND_SIMILAR_FACTS, FACT, embedding, limit)
    }

    fn create_chunks_batch(
        &self,
        doc_id: &NodeId,
        chunks: &[ChunkNode],
    ) -> Result<Vec<StoredChunk>> {
        let parent = parse_node_id("doc_id", doc_id.as_str())?;
        let mut stored = Vec::with_capacity(chunks.len());
        for group in chunks.chunks(BATCH_WRITE_SIZE) {
            let mut batch =
                write_batch().var_as("parent", g().n(NodeRef::id(parent)).has_label(DOCUMENT));
            let mut names = vec!["parent".to_owned()];
            for (index, chunk) in group.iter().enumerate() {
                batch = add_child(
                    batch,
                    "parent",
                    index,
                    DOCUMENT_CHUNK,
                    chunk_properties(chunk),
                    CHUNK_OF_DOCUMENT,
                    vec![("SequenceNumber", chunk.sequence_number.into())],
                );
                names.push(format!("props{index}"));
            }
            let result = self.write("CreateChunksBatch", batch.returning(names))?;
            single(&result, "parent", DOCUMENT, doc_id.as_str())?;
            for index in 0..group.len() {
                stored.push(decode(single(
                    &result,
                    &format!("props{index}"),
                    DOCUMENT_CHUNK,
                    "(new)",
                )?)?);
            }
        }
        Ok(stored)
    }

    fn delete_document_chunks(&self, doc_id: &NodeId) -> Result<usize> {
        let id = parse_node_id("doc_id", doc_id.as_str())?;
        // Dropping a node also drops its incident edges and its vector.
        let result = self.write(
            "DeleteDocumentChunks",
            write_batch()
                .var_as(
                    "chunks",
                    g().n(NodeRef::id(id))
                        .has_label(DOCUMENT)
                        .in_(Some(CHUNK_OF_DOCUMENT))
                        .has_label(DOCUMENT_CHUNK),
                )
                .var_as(
                    "spans",
                    g().n(NodeRef::var("chunks")).in_(Some(SPAN_OF_CHUNK)).has_label(SPAN),
                )
                .var_as("chunk_count", g().n(NodeRef::var("chunks")).count())
                .var_as("dropped_spans", g().n(NodeRef::var("spans")).drop())
                .var_as("dropped_chunks", g().n(NodeRef::var("chunks")).drop())
                .returning(["chunk_count"]),
        )?;
        let count = result.get("chunk_count").and_then(Value::as_u64).unwrap_or(0);
        Ok(usize::try_from(count).unwrap_or(usize::MAX))
    }

    fn create_spans_batch(&self, spans: &[SpanNode]) -> Result<Vec<StoredSpan>> {
        let parents = spans
            .iter()
            .map(|span| parse_node_id("chunk_id", span.chunk_id.as_str()))
            .collect::<Result<Vec<u64>>>()?;
        let distinct: Vec<u64> =
            parents.iter().copied().collect::<BTreeSet<u64>>().into_iter().collect();
        let found = self.existing("CreateSpansBatch", DOCUMENT_CHUNK, &distinct)?;
        if let Some(missing) = spans.iter().zip(&parents).find(|(_, id)| !found.contains(id)) {
            return Err(StoreError::not_found(DOCUMENT_CHUNK, missing.0.chunk_id.as_str()));
        }
        let mut stored = Vec::with_capacity(spans.len());
        for (group, group_parents) in
            spans.chunks(BATCH_WRITE_SIZE).zip(parents.chunks(BATCH_WRITE_SIZE))
        {
            let mut batch = write_batch();
            let mut names = Vec::new();
            let mut declared = BTreeSet::new();
            for parent in group_parents {
                if declared.insert(*parent) {
                    let var = format!("parent{parent}");
                    batch =
                        batch.var_as(&var, g().n(NodeRef::id(*parent)).has_label(DOCUMENT_CHUNK));
                    names.push(var);
                }
            }
            for (index, (span, parent)) in group.iter().zip(group_parents).enumerate() {
                batch = add_child(
                    batch,
                    &format!("parent{parent}"),
                    index,
                    SPAN,
                    span_properties(span, *parent),
                    SPAN_OF_CHUNK,
                    vec![("SequenceNumber", span.start_char.into())],
                );
                names.push(format!("props{index}"));
            }
            let result = self.write("CreateSpansBatch", batch.returning(names))?;
            for (index, span) in group.iter().enumerate() {
                stored.push(decode(single(
                    &result,
                    &format!("props{index}"),
                    DOCUMENT_CHUNK,
                    span.chunk_id.as_str(),
                )?)?);
            }
        }
        Ok(stored)
    }

    fn create_embeddings_batch(
        &self,
        label: VectorLabel,
        items: &[EmbeddingWrite<'_>],
    ) -> Result<usize> {
        let mut ids = Vec::with_capacity(items.len());
        for item in items {
            validate_vector(item.vector, self.embedding_dim)?;
            ids.push(parse_node_id(label.id_field(), item.node_id)?);
        }
        let found = self.existing("CreateEmbeddingsBatch", label.as_str(), &ids)?;
        if let Some(missing) = items.iter().zip(&ids).find(|(_, id)| !found.contains(id)) {
            return Err(StoreError::not_found(label.as_str(), missing.0.node_id));
        }
        for (group, group_ids) in items.chunks(BATCH_WRITE_SIZE).zip(ids.chunks(BATCH_WRITE_SIZE)) {
            let mut batch = write_batch();
            let mut names = Vec::new();
            for (index, (item, id)) in group.iter().zip(group_ids).enumerate() {
                let name = format!("node{index}");
                batch = batch.var_as(
                    &name,
                    g().n(NodeRef::id(*id))
                        .has_label(label.as_str())
                        .set_property(EMBEDDING_PROPERTY, PropertyValue::from(item.vector.to_vec()))
                        .value_map(Some(vec!["$id"])),
                );
                names.push(name);
            }
            let result = self.write("CreateEmbeddingsBatch", batch.returning(names))?;
            for (index, item) in group.iter().enumerate() {
                single(&result, &format!("node{index}"), label.as_str(), item.node_id)?;
            }
        }
        Ok(items.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use impeach_text::retry::FixedJitter;

    #[test]
    fn conflict_backoff_is_capped_and_half_jittered() {
        let mut none = FixedJitter(0.0);
        let mut full = FixedJitter(1.0);
        assert_eq!(conflict_backoff(0, &mut none), Duration::from_secs_f64(0.0025));
        assert_eq!(conflict_backoff(0, &mut full), Duration::from_secs_f64(0.005));
        assert_eq!(conflict_backoff(3, &mut none), Duration::from_secs_f64(0.02));
        assert_eq!(conflict_backoff(19, &mut none), Duration::from_secs_f64(0.25));
        assert_eq!(conflict_backoff(19, &mut full), Duration::from_secs_f64(0.5));
    }

    #[test]
    fn errors_are_classified_like_python() {
        let conflict = HelixError::EmbeddedError {
            code: "transaction_conflict".into(),
            details: "Transaction conflict: x".into(),
        };
        assert!(is_retryable(&conflict));
        assert!(!is_database_lost(&conflict));
        let fenced = HelixError::EmbeddedError {
            code: "storage_error".into(),
            details: "Detected newer DB client".into(),
        };
        assert!(is_database_lost(&fenced));
        assert!(!is_retryable(&fenced));
        let missing = HelixError::EmbeddedError {
            code: "index_not_found".into(),
            details: "missing index".into(),
        };
        assert!(matches!(map_error("FindFactsByText", &missing), StoreError::Query { .. }));
        let zero = HelixError::EmbeddedError {
            code: "zero_norm_cosine_vector".into(),
            details: "norm must be non-zero".into(),
        };
        assert!(matches!(map_error("x", &zero), StoreError::Validation { .. }));
    }

    #[test]
    fn decode_drops_virtual_fields_and_defaults_missing_properties() {
        let row = serde_json::json!({"$id": 7, "$label": "Entity", "$score": 1.5, "Name": "Acme"});
        let stored: StoredEntity = decode(&row).expect("decodes");
        assert_eq!(stored.id, EntityId::new("7"));
        assert_eq!(stored.node.name, "Acme");
        assert_eq!(stored.node.aliases_json, "[]");
        assert!(decode::<EntityId, EntityNode>(&serde_json::json!({"Name": "x"})).is_err());
    }

    #[test]
    fn options_resolve_locations() {
        let memory = HelixOptions::memory("ns", 3);
        assert_eq!(memory.location().expect("memory"), "memory");
        let server = HelixOptions::server("http://localhost:6969/", None, "ns", 3);
        assert_eq!(server.location().expect("server"), "http://localhost:6969");
        let mut no_url = server;
        no_url.url = None;
        assert!(matches!(no_url.location(), Err(StoreError::Connection(_))));
        let base = tempfile::tempdir().expect("tempdir");
        let embedded = HelixOptions::embedded(base.path().join("data/helix"), "ns", 3);
        let location = embedded.location().expect("embedded");
        assert!(Path::new(&location).is_absolute());
        assert!(Path::new(&location).is_dir());
        for spelling in ["data/./helix", "data/other/../helix", "data//helix"] {
            let same = HelixOptions::embedded(base.path().join(spelling), "ns", 3);
            assert_eq!(same.location().expect("embedded"), location, "{spelling}");
        }
    }

    #[test]
    fn verbatim_prefix_is_dropped_for_drive_paths_only() {
        let drive = format!("{VERBATIM_PREFIX}C:/data");
        assert_eq!(strip_verbatim_prefix(PathBuf::from(&drive)), PathBuf::from("C:/data"));
        let unc = format!("{VERBATIM_PREFIX}UNC/server/share");
        assert_eq!(strip_verbatim_prefix(PathBuf::from(&unc)), PathBuf::from(&unc));
        assert_eq!(strip_verbatim_prefix(PathBuf::from("/srv/data")), PathBuf::from("/srv/data"));
    }
}
