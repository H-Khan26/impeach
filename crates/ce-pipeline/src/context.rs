//! The shared pipeline context: services every stage uses plus typed per-run data.
//!
//! Python origin: `pipeline/runner.py` (`PipelineContext`, a string-keyed dict). Here every
//! context key of PROCESSING.md's "Context keys" table is a typed `Option` slot of
//! [`ContextData`]; `None` means "not written by any stage yet" (Python: key absent).

use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

use ce_config::pipeline::ExtractionMode;
use ce_config::PipelineConfig;
use ce_core::processing::{DefinedTerm, ExtractedRelation, FactLinkingStats};
use ce_core::{
    ChunkId, ContradictionReport, CorefClusterContext, DocId, DocumentPayload, EdResult, EntityId,
    ExtractedFact, FactId, IndexingResult, Mention, MentionId, Metadata, NodeId, SpanId,
};
use ce_models::{Embedder, ModelRegistry};
use ce_store::{FactNode, GraphStore};
use indexmap::IndexMap;

/// Shared services a pipeline is built with and runs against: the validated config, the model
/// registry and the graph store. Cheap to clone (three `Arc`s).
#[derive(Clone, Debug)]
pub struct StageBuildContext {
    /// Validated pipeline config.
    pub config: Arc<PipelineConfig>,
    /// Model registry keyed `<pipeline.name>_<type>`.
    pub registry: Arc<ModelRegistry>,
    /// Graph store shared by every stage.
    pub store: Arc<dyn GraphStore>,
}

impl StageBuildContext {
    /// Bundles the three services.
    pub fn new(
        config: Arc<PipelineConfig>,
        registry: Arc<ModelRegistry>,
        store: Arc<dyn GraphStore>,
    ) -> Self {
        Self { config, registry, store }
    }
}

/// Context of one pipeline run (Python `PipelineContext`): the shared services and the data
/// stages hand to each other.
#[derive(Debug)]
pub struct PipelineContext {
    /// Model registry.
    pub registry: Arc<ModelRegistry>,
    /// Pipeline config.
    pub config: Arc<PipelineConfig>,
    /// Graph store.
    pub store: Arc<dyn GraphStore>,
    /// Per-run data written by the caller and the stages.
    pub data: ContextData,
}

impl PipelineContext {
    /// An empty context over the given services.
    #[must_use]
    pub fn new(services: &StageBuildContext) -> Self {
        Self {
            registry: Arc::clone(&services.registry),
            config: Arc::clone(&services.config),
            store: Arc::clone(&services.store),
            data: ContextData::default(),
        }
    }
}

/// An embedder shared through the context (Python context key `embedding_service`).
#[derive(Clone)]
pub struct SharedEmbedder(pub Arc<dyn Embedder>);

impl fmt::Debug for SharedEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SharedEmbedder").field(&self.0.model_name()).finish()
    }
}

impl Deref for SharedEmbedder {
    type Target = dyn Embedder;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

/// Typed per-run data. Field names are the Python context keys (`_all_mentions_by_id` is
/// `all_mentions_by_id`); the writer of each slot is noted on the field.
///
/// Maps are insertion ordered like Python dicts.
#[derive(Debug, Clone, Default)]
pub struct ContextData {
    /// Runner: the input document, coerced from the [`crate::PipelineInput`]. RAGIndexStage
    /// writes it back with its graph `helix_id` set. (Python passes it as the stage `data`.)
    pub document: Option<DocumentPayload>,

    /// `run_processing`: the document's metadata (`case_id`, `db_id`, ...), when not empty.
    pub doc_metadata: Option<Metadata>,
    /// `run_processing`: the document's `case_id`, when present.
    pub case_id: Option<String>,

    /// RAGIndexStage: graph id of the `Document` node.
    pub helix_doc_id: Option<NodeId>,
    /// RAGIndexStage: logical document id.
    pub doc_id: Option<DocId>,
    /// RAGIndexStage: document, chunk, span and embedding writes with timings.
    pub rag_index_result: Option<IndexingResult>,
    /// RAGIndexStage: full document text.
    pub doc_text: Option<String>,
    /// RAGIndexStage (online mode, documents with chunks): the chunk embedder (Python
    /// `embedding_service`). Later stages treat `None` as "no embeddings available" (offline
    /// runs and empty documents).
    pub embedder: Option<SharedEmbedder>,
    /// RAGIndexStage: `true` once the document, its chunks and spans were written to the graph
    /// store (online mode); `false` before indexing and for offline runs, whose ids are
    /// fabricated. The counterpart of Python's `helix_client` being in the context: later
    /// stages write graph nodes under the indexed chunks only when it is `true`.
    pub index_persisted: bool,

    /// MentionStage (augmented by CoreferenceStage): mentions per chunk graph id.
    pub mentions_by_chunk: Option<IndexMap<ChunkId, Vec<Mention>>>,
    /// MentionStage (augmented by CoreferenceStage): every mention of the document.
    pub doc_mentions: Option<Vec<Mention>>,
    /// MentionStage: mention id -> graph id of its mention `Span`.
    pub mention_to_span_id: Option<IndexMap<MentionId, SpanId>>,

    /// DefinitionStage: role terms defined in the document.
    pub defined_terms: Option<Vec<DefinedTerm>>,

    /// CoreferenceStage: coreference clusters mapped onto mentions.
    pub coref_clusters: Option<Vec<CorefClusterContext>>,
    /// CoreferenceStage: cluster id -> canonical mention id.
    pub canonical_mention_for_cluster: Option<IndexMap<String, MentionId>>,
    /// CoreferenceStage: mention id -> canonical text.
    pub canonical_text_for_mention: Option<IndexMap<MentionId, String>>,
    /// CoreferenceStage: synthetic pronoun mention id -> canonical mention id.
    pub synthetic_pronoun_mentions: Option<IndexMap<MentionId, MentionId>>,

    /// EDStage: disambiguation results.
    pub ed_results: Option<Vec<EdResult>>,
    /// EDStage: mention id -> chosen entity id.
    pub resolved_mentions: Option<IndexMap<MentionId, EntityId>>,
    /// EDStage: normalised role term (e.g. `PLAINTIFF`) -> entity id.
    pub role_map: Option<IndexMap<String, EntityId>>,
    /// EDStage: every mention by id (Python `_all_mentions_by_id`).
    pub all_mentions_by_id: Option<IndexMap<MentionId, Mention>>,

    /// RelationStage: relations across all chunks.
    pub extracted_relations: Option<Vec<ExtractedRelation>>,
    /// RelationStage: relation label -> count.
    pub relations_by_type: Option<IndexMap<String, usize>>,

    /// FactStage: every extracted fact.
    pub extracted_facts: Option<Vec<ExtractedFact>>,
    /// FactStage: extraction mode.
    pub extraction_mode: Option<ExtractionMode>,
    /// FactStage: entity-linking counts.
    pub fact_linking_stats: Option<FactLinkingStats>,
    /// FactStage: chunk graph id -> text the facts were extracted from.
    pub resolved_text_by_chunk: Option<IndexMap<ChunkId, String>>,
    /// FactStage: number of facts written to the store.
    pub indexed_facts_count: Option<usize>,
    /// FactStage: graph ids of the written facts, aligned with `extracted_facts` (`None` for a
    /// fact that was not written).
    pub fact_helix_ids: Option<Vec<Option<FactId>>>,
    /// FactStage (`commit_facts: false`): fact nodes left for the contradiction stage to commit.
    pub fact_nodes_pending: Option<Vec<FactNode>>,

    /// ContradictionCheckStage: the contradiction report.
    pub contradiction_report: Option<ContradictionReport>,
    /// ContradictionCheckStage: facts that passed every check.
    pub clean_facts: Option<Vec<ExtractedFact>>,
    /// ContradictionCheckStage: graph ids of the committed facts.
    pub committed_fact_ids: Option<Vec<FactId>>,
}
