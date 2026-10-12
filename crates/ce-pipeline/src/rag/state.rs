//! Persistent document-identity cache (Python `services/rag/state.py`).

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use ce_core::{Metadata, NodeId};
use serde_json::Value;

/// Cache key of a document: `scope|db_id|case_id|doc_id`.
///
/// `doc_id` alone is not unique (cases reuse ids such as `001`), so the key also holds the
/// document's `db_id` and `case_id` from its metadata and a `scope` naming the database
/// (`<store_id>#<namespace>`). Missing, null or empty identifiers are empty strings.
#[must_use]
pub fn document_identity_key(doc_id: &str, metadata: Option<&Metadata>, scope: &str) -> String {
    let field =
        |key: &str| metadata.and_then(|m| m.get(key)).map(py_str_or_empty).unwrap_or_default();
    [scope.to_owned(), field("db_id"), field("case_id"), doc_id.to_owned()].join("|")
}

/// Python `str(value or "")`.
fn py_str_or_empty(value: &Value) -> String {
    match value {
        Value::Null | Value::Bool(false) => String::new(),
        Value::String(s) => s.clone(),
        Value::Number(n) if n.as_f64() == Some(0.0) => String::new(),
        Value::Bool(true) => "True".to_owned(),
        Value::Array(a) if a.is_empty() => String::new(),
        Value::Object(o) if o.is_empty() => String::new(),
        other => other.to_string(),
    }
}

/// Default cache file: `~/.cache/contradiction_engine/rag_documents.json` (the Python path).
#[must_use]
pub fn default_document_cache_path() -> PathBuf {
    let home = std::env::var_os("USERPROFILE")
        .filter(|v| !v.is_empty())
        .or_else(|| std::env::var_os("HOME").filter(|v| !v.is_empty()))
        .map_or_else(|| PathBuf::from("."), PathBuf::from);
    home.join(".cache").join("contradiction_engine").join("rag_documents.json")
}

/// Document identity -> graph `Document` id, persisted as a JSON object (sorted keys, two-space
/// indent) so later runs update the same node instead of creating a new one.
#[derive(Debug)]
pub struct DocumentStateStore {
    path: PathBuf,
    scope: String,
    cache: Mutex<BTreeMap<String, String>>,
}

impl DocumentStateStore {
    /// Opens the cache at `path` (missing or unreadable files start empty) for `scope`.
    #[must_use]
    pub fn open(path: impl Into<PathBuf>, scope: impl Into<String>) -> Self {
        let path = path.into();
        let cache = Mutex::new(load(&path));
        Self { path, scope: scope.into(), cache }
    }

    /// The cache file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The scope prefix of every key.
    #[must_use]
    pub fn scope(&self) -> &str {
        &self.scope
    }

    /// Key of a document in this cache.
    #[must_use]
    pub fn key_for(&self, doc_id: &str, metadata: Option<&Metadata>) -> String {
        document_identity_key(doc_id, metadata, &self.scope)
    }

    /// The cached graph id of a document.
    #[must_use]
    pub fn lookup(&self, doc_id: &str, metadata: Option<&Metadata>) -> Option<NodeId> {
        if doc_id.is_empty() {
            return None;
        }
        self.lock().get(&self.key_for(doc_id, metadata)).map(|id| NodeId::from(id.as_str()))
    }

    /// Records the graph id of a document and persists the cache when it changed.
    pub fn record(&self, doc_id: &str, helix_id: &NodeId, metadata: Option<&Metadata>) {
        if doc_id.is_empty() || helix_id.as_str().is_empty() {
            return;
        }
        let key = self.key_for(doc_id, metadata);
        let mut cache = self.lock();
        if cache.get(&key).is_some_and(|id| id == helix_id.as_str()) {
            return;
        }
        cache.insert(key, helix_id.as_str().to_owned());
        persist(&self.path, &cache);
    }

    /// Forgets a document and persists the cache when it changed.
    pub fn delete(&self, doc_id: &str, metadata: Option<&Metadata>) {
        if doc_id.is_empty() {
            return;
        }
        let key = self.key_for(doc_id, metadata);
        let mut cache = self.lock();
        if cache.remove(&key).is_some() {
            persist(&self.path, &cache);
        }
    }

    fn lock(&self) -> MutexGuard<'_, BTreeMap<String, String>> {
        ce_core::util::lock(&self.cache)
    }
}

fn load(path: &Path) -> BTreeMap<String, String> {
    let Ok(raw) = std::fs::read(path) else {
        return BTreeMap::new();
    };
    match serde_json::from_slice::<Value>(&raw) {
        Ok(Value::Object(map)) => map
            .into_iter()
            .filter_map(|(key, value)| {
                let value = match value {
                    Value::String(s) => s,
                    Value::Null => return None,
                    other => other.to_string(),
                };
                (!key.is_empty() && !value.is_empty()).then_some((key, value))
            })
            .collect(),
        Ok(_) => BTreeMap::new(),
        Err(error) => {
            tracing::warn!(path = %path.display(), "failed to load document state cache: {error}");
            BTreeMap::new()
        }
    }
}

/// Writes the cache to a temporary file next to `path` and renames it over `path`, so a crash
/// leaves either the old or the new file, never a truncated one.
fn persist(path: &Path, cache: &BTreeMap<String, String>) {
    let write = || -> std::io::Result<()> {
        let dir = match path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent,
            _ => Path::new("."),
        };
        std::fs::create_dir_all(dir)?;
        let json = serde_json::to_string_pretty(cache).map_err(std::io::Error::other)?;
        let mut file = tempfile::NamedTempFile::new_in(dir)?;
        file.write_all(json.as_bytes())?;
        file.as_file().sync_all()?;
        file.persist(path).map_err(|error| error.error)?;
        Ok(())
    };
    if let Err(error) = write() {
        tracing::warn!(path = %path.display(), "failed to persist document state cache: {error}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn meta(value: serde_json::Value) -> Metadata {
        match value {
            Value::Object(map) => map,
            _ => Metadata::new(),
        }
    }

    #[test]
    fn identity_key_includes_db_id_and_case_id() {
        let m = meta(json!({"case_id": "C", "db_id": "D"}));
        assert_eq!(document_identity_key("001", Some(&m), "s"), "s|D|C|001");
        assert_eq!(document_identity_key("001", None, ""), "|||001");
        let numeric = meta(json!({"case_id": 7, "db_id": null}));
        assert_eq!(document_identity_key("1", Some(&numeric), ""), "||7|1");
    }

    #[test]
    fn default_path_uses_contradiction_engine_cache_dir() {
        let path = default_document_cache_path();
        assert_eq!(path.file_name().and_then(|n| n.to_str()), Some("rag_documents.json"));
        assert_eq!(
            path.parent().and_then(Path::file_name).and_then(|n| n.to_str()),
            Some("contradiction_engine")
        );
    }

    #[test]
    fn same_doc_id_different_case_is_a_different_key() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = DocumentStateStore::open(dir.path().join("cache.json"), "");
        let case_a = meta(json!({"case_id": "CASE-A"}));
        let case_b = meta(json!({"case_id": "CASE-B"}));
        store.record("001", &NodeId::from("node-a"), Some(&case_a));
        assert_eq!(store.lookup("001", Some(&case_a)), Some(NodeId::from("node-a")));
        assert_eq!(store.lookup("001", Some(&case_b)), None);
        assert_eq!(store.lookup("001", None), None);
    }

    #[test]
    fn scope_separates_databases_and_persists() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("nested").join("cache.json");
        let c = meta(json!({"case_id": "C"}));
        DocumentStateStore::open(&path, "store-1#ns").record("001", &NodeId::from("5"), Some(&c));
        assert_eq!(
            DocumentStateStore::open(&path, "store-1#ns").lookup("001", Some(&c)),
            Some(NodeId::from("5"))
        );
        assert_eq!(DocumentStateStore::open(&path, "store-2#ns").lookup("001", Some(&c)), None);

        let reopened = DocumentStateStore::open(&path, "store-1#ns");
        reopened.delete("001", Some(&c));
        assert_eq!(DocumentStateStore::open(&path, "store-1#ns").lookup("001", Some(&c)), None);
        let written = std::fs::read_to_string(&path).expect("cache file");
        assert_eq!(written, "{}");
    }

    #[test]
    fn persist_replaces_the_file_atomically_without_leftovers() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("cache.json");
        std::fs::write(&path, "{\"old\": \"1\"}").expect("write");
        let store = DocumentStateStore::open(&path, "s");
        store.record("001", &NodeId::from("7"), None);
        store.record("002", &NodeId::from("8"), None);
        let entries: Vec<String> = std::fs::read_dir(dir.path())
            .expect("read dir")
            .map(|entry| entry.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(entries, ["cache.json"], "no temporary files are left behind");
        let reopened = DocumentStateStore::open(&path, "s");
        assert_eq!(reopened.lookup("001", None), Some(NodeId::from("7")));
        assert_eq!(reopened.lookup("002", None), Some(NodeId::from("8")));
        let written: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read")).expect("json");
        assert_eq!(written, json!({"old": "1", "s|||001": "7", "s|||002": "8"}));
    }

    #[test]
    fn corrupt_cache_files_start_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("cache.json");
        std::fs::write(&path, "not json").expect("write");
        let store = DocumentStateStore::open(&path, "");
        assert_eq!(store.lookup("001", None), None);
        store.record("001", &NodeId::from("3"), None);
        assert!(std::fs::read_to_string(&path).expect("read").contains("\"|||001\": \"3\""));
    }
}
