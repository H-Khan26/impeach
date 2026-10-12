//! `RAGIndexStage`: chunk the document, write Document / Chunk / Span nodes, embed the chunks
//! (Python `stages/rag_index.py` and `services/rag/pipeline.py`).
//!
//! Online flow (`offline: false`):
//! 1. With `use_document_cache: true` and no `helix_id` on the input, look the document up in
//!    the [`DocumentStateStore`] (key `<store_id>#<namespace>|db_id|case_id|doc_id`) and reuse
//!    the cached `Document` node. Off by default: every run creates a fresh node.
//! 2. Create the `Document` node, or update it when the document carries a `helix_id`. When a
//!    cached id turns out to be stale (the update fails with `NotFound`), drop it and create a
//!    new node; any other update error fails the stage.
//! 3. Chunk (rules in [`crate::rag::chunker`]); a reused document first loses its old chunks,
//!    spans and chunk vectors ([`GraphStore::delete_document_chunks`]); write chunks with
//!    [`GraphStore::create_chunks_batch`]; with `include_sentence_spans`, write sentence spans
//!    with [`GraphStore::create_spans_batch`].
//! 4. When there are chunks, resolve the registry's `<pipeline.name>_bge` embedder, embed
//!    every chunk text and store the vectors with [`GraphStore::create_embeddings_batch`].
//!
//! `offline: true` writes nothing and embeds nothing: ids are fabricated (`doc:<doc_id>`,
//! `<doc_id>:chunk-00000`, `<doc_id>:sent-00000`) and the context gets no embedder.
//!
//! Context writes: `document` (with `helix_id`; restored on error as well), `helix_doc_id`,
//! `doc_id`, `rag_index_result`, `doc_text`, `index_persisted` (true online, false offline),
//! and `embedder` in online mode when the document has chunks.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use ce_config::pipeline::RagIndexStageConfig;
use ce_config::StageConfig;
use ce_core::{
    ChunkWriteResult, DocumentPayload, DocumentWriteResult, EmbeddingWriteResult, IndexingResult,
    NodeId, SpanWriteResult,
};
use ce_models::{Embedder, ModelKind, ModelRegistry};
use ce_store::{GraphStore, StoreError};
use indexmap::IndexMap;

use crate::context::{PipelineContext, SharedEmbedder, StageBuildContext};
use crate::error::{PipelineError, Result};
use crate::rag::{default_document_cache_path, writer, Chunker, DocumentStateStore};
use crate::stage::Stage;

/// The RAG index stage.
#[derive(Debug)]
pub struct RagIndexStage {
    name: String,
    offline: bool,
    chunker: Chunker,
    store: Arc<dyn GraphStore>,
    registry: Arc<ModelRegistry>,
    embedder_key: String,
    cache_scope: String,
    document_cache: Option<DocumentStateStore>,
}

impl RagIndexStage {
    /// Config `type` of this stage.
    pub const TYPE_NAME: &'static str = "RAGIndexStage";

    /// Builds the stage from its config block and the shared services. With
    /// `use_document_cache` the cache file is [`default_document_cache_path`].
    ///
    /// # Errors
    /// [`PipelineError::InvalidStageConfig`] for an unusable `rag.chunking` block.
    pub fn new(config: &RagIndexStageConfig, services: &StageBuildContext) -> Result<Self> {
        let chunker = Chunker::new(&services.config.rag.chunking)?;
        let cache_scope =
            format!("{}#{}", services.store.store_id(), services.config.helix_namespace());
        let document_cache = (config.use_document_cache && !config.offline)
            .then(|| DocumentStateStore::open(default_document_cache_path(), cache_scope.clone()));
        Ok(Self {
            name: config.name.clone(),
            offline: config.offline,
            chunker,
            store: Arc::clone(&services.store),
            registry: Arc::clone(&services.registry),
            embedder_key: services.registry.key(ModelKind::Bge),
            cache_scope,
            document_cache,
        })
    }

    /// [`crate::StageConstructor`] of this stage.
    ///
    /// # Errors
    /// [`PipelineError::InvalidStageConfig`] for another stage type; errors of
    /// [`RagIndexStage::new`].
    pub fn construct(stage: &StageConfig, services: &StageBuildContext) -> Result<Box<dyn Stage>> {
        match stage {
            StageConfig::RagIndex(config) => Ok(Box::new(Self::new(config, services)?)),
            other => Err(PipelineError::InvalidStageConfig {
                stage: other.name().to_owned(),
                message: format!(
                    "expected a {} config, got {}",
                    Self::TYPE_NAME,
                    other.type_name()
                ),
            }),
        }
    }

    /// Enables the document cache with the file at `path` (scoped to this store and
    /// namespace), whatever `use_document_cache` says. Ignored in offline mode.
    #[must_use]
    pub fn with_document_cache_at(mut self, path: impl Into<PathBuf>) -> Self {
        if !self.offline {
            self.document_cache = Some(DocumentStateStore::open(path, self.cache_scope.clone()));
        }
        self
    }

    /// Disables the document cache.
    #[must_use]
    pub fn without_document_cache(mut self) -> Self {
        self.document_cache = None;
        self
    }

    /// The document cache, when enabled.
    #[must_use]
    pub fn document_cache(&self) -> Option<&DocumentStateStore> {
        self.document_cache.as_ref()
    }

    /// Whether the stage runs offline.
    #[must_use]
    pub fn is_offline(&self) -> bool {
        self.offline
    }

    /// Indexes one document online (steps 1-4 of the module docs) and returns the indexing
    /// result; `document.helix_id` is set to the `Document` node id.
    ///
    /// # Errors
    /// Store and model errors; [`PipelineError::EmbeddingCount`] /
    /// [`PipelineError::EmbeddingDimension`] when the embedder output does not fit.
    pub fn index_document(&self, document: &mut DocumentPayload) -> Result<IndexingResult> {
        self.index_online(document).map(|(result, _)| result)
    }

    /// [`RagIndexStage::index_document`], also returning the embedder when one was resolved
    /// (only documents with chunks resolve it).
    fn index_online(
        &self,
        document: &mut DocumentPayload,
    ) -> Result<(IndexingResult, Option<Arc<dyn Embedder>>)> {
        let mut timings = IndexMap::new();

        let started = Instant::now();
        self.attach_cached_helix_id(document);
        let doc_write = self.ensure_document_with_retry(document)?;
        timings.insert("document".to_owned(), started.elapsed().as_secs_f64());
        if let Some(cache) = &self.document_cache {
            cache.record(document.doc_id.as_str(), &doc_write.helix_id, Some(&document.metadata));
        }

        let started = Instant::now();
        let chunking = self.chunker.chunk(document);
        let chunk_writes = writer::write_chunks(&*self.store, &doc_write, chunking.chunks)?;
        let span_writes =
            if self.chunker.include_sentence_spans() && !chunking.sentence_spans.is_empty() {
                writer::write_sentence_spans(
                    &*self.store,
                    &doc_write.helix_id,
                    &chunk_writes,
                    &chunking.sentence_spans,
                )?
            } else {
                Vec::new()
            };
        timings.insert("chunking".to_owned(), started.elapsed().as_secs_f64());

        let started = Instant::now();
        let (embeddings, embedder) = self.embed_chunks(&chunk_writes)?;
        timings.insert("embeddings".to_owned(), started.elapsed().as_secs_f64());

        tracing::info!(
            doc_id = %document.doc_id,
            helix_doc_id = %doc_write.helix_id,
            chunks = chunk_writes.len(),
            spans = span_writes.len(),
            embeddings = embeddings.count,
            "indexed document"
        );
        let result = IndexingResult {
            document: doc_write,
            chunks: chunk_writes,
            spans: span_writes,
            embeddings,
            doc_text: Some(document.text.clone()),
            timings,
        };
        Ok((result, embedder))
    }

    /// Offline indexing: chunks with fabricated ids, no store writes, no embeddings, no
    /// timings.
    #[must_use]
    pub fn index_offline(&self, document: &DocumentPayload) -> IndexingResult {
        let doc_id = document.doc_id.as_str();
        let chunking = self.chunker.chunk(document);
        let chunks = chunking
            .chunks
            .into_iter()
            .map(|mut chunk| {
                let helix_id = chunk.helix_id.clone().unwrap_or_else(|| {
                    NodeId::from(format!("{doc_id}:chunk-{:05}", chunk.sequence_number))
                });
                chunk.helix_id = Some(helix_id.clone());
                ChunkWriteResult { chunk, helix_id, created: true }
            })
            .collect();
        let spans = chunking
            .sentence_spans
            .iter()
            .enumerate()
            .map(|(i, span)| SpanWriteResult {
                span_id: format!("{doc_id}:sent-{i:05}").into(),
                chunk_sequence: span.chunk_sequence,
                created: true,
            })
            .collect();
        IndexingResult {
            document: DocumentWriteResult {
                doc_id: document.doc_id.clone(),
                helix_id: format!("doc:{doc_id}").into(),
                created: true,
            },
            chunks,
            spans,
            embeddings: EmbeddingWriteResult::default(),
            doc_text: Some(document.text.clone()),
            timings: IndexMap::new(),
        }
    }

    fn attach_cached_helix_id(&self, document: &mut DocumentPayload) {
        let Some(cache) = &self.document_cache else {
            return;
        };
        if document.helix_id.is_some() {
            return;
        }
        if let Some(cached) = cache.lookup(document.doc_id.as_str(), Some(&document.metadata)) {
            document.helix_id = Some(cached);
        }
    }

    fn ensure_document_with_retry(
        &self,
        document: &mut DocumentPayload,
    ) -> Result<DocumentWriteResult> {
        match writer::ensure_document(&*self.store, document) {
            Err(PipelineError::Store(error @ StoreError::NotFound { .. }))
                if document.helix_id.is_some() =>
            {
                let error = PipelineError::Store(error);
                let Some(cache) = &self.document_cache else {
                    return Err(error);
                };
                tracing::warn!(
                    doc_id = %document.doc_id,
                    helix_id = ?document.helix_id,
                    "cached Document id appears stale ({error}); creating a new node"
                );
                cache.delete(document.doc_id.as_str(), Some(&document.metadata));
                document.helix_id = None;
                writer::ensure_document(&*self.store, document)
            }
            other => other,
        }
    }

    /// Embeds and stores the chunk vectors; resolves the embedder only when there are chunks
    /// and returns it.
    fn embed_chunks(
        &self,
        chunks: &[ChunkWriteResult],
    ) -> Result<(EmbeddingWriteResult, Option<Arc<dyn Embedder>>)> {
        if chunks.is_empty() {
            return Ok((EmbeddingWriteResult::default(), None));
        }
        let embedder = self.registry.embedder(&self.embedder_key)?;
        let texts: Vec<&str> = chunks.iter().map(|w| w.chunk.text.as_str()).collect();
        let vectors = embedder.embed(&texts)?;
        if vectors.len() != chunks.len() {
            return Err(PipelineError::EmbeddingCount {
                embeddings: vectors.len(),
                chunks: chunks.len(),
            });
        }
        let expected = self.store.embedding_dim();
        if let Some(bad) = vectors.iter().find(|v| v.len() != expected) {
            return Err(PipelineError::EmbeddingDimension { actual: bad.len(), expected });
        }
        let written = writer::upsert_chunk_embeddings(&*self.store, chunks, &vectors)?;
        Ok((written, Some(embedder)))
    }
}

impl Stage for RagIndexStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<()> {
        let mut document = ctx.data.document.take().ok_or_else(|| {
            PipelineError::MissingContext { stage: self.name.clone(), key: "document" }
        })?;
        let indexed = if self.offline {
            Ok((self.index_offline(&document), None))
        } else {
            self.index_online(&mut document)
        };
        ctx.data.document = Some(document);
        let (result, embedder) = indexed?;
        if let Some(embedder) = embedder {
            ctx.data.embedder = Some(SharedEmbedder(embedder));
        }
        ctx.data.index_persisted = !self.offline;
        ctx.data.helix_doc_id = Some(result.document.helix_id.clone());
        ctx.data.doc_id = Some(result.document.doc_id.clone());
        ctx.data.doc_text = result.doc_text.clone();
        ctx.data.rag_index_result = Some(result);
        Ok(())
    }
}
