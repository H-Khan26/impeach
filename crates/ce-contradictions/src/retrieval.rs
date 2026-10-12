//! Similar-fact retrieval for the cross-document check (Python
//! `services/contradictions/retrieval.py`).
//!
//! Sources, in order: graph (facts linked to the given entities by `FactAboutEntity`), BM25
//! over fact text and vector search over fact embeddings; BM25 / vector run according to
//! `fact_retrieval_strategy`. Candidates are deduplicated by fact id (first source wins the
//! insertion slot, later sources are appended to `sources`), scored by the highest weight of
//! their sources times `multi_source_boost` when found by several, filtered by `min_score`
//! (when positive), sorted by score (stable, descending) and capped at `max_candidates`.

use std::fmt;
use std::sync::Arc;

use ce_config::services::SimilarFactRetrieverConfig;
use ce_core::{EntityId, ExtractedFact, FactId, RetrievedFact};
use ce_models::Embedder;
use ce_store::{GraphStore, StoredFact};

use crate::error::Result;
use crate::prefilter::CandidateEntityMap;

/// Retrieval source name of graph (shared entity) hits.
pub const SOURCE_GRAPH: &str = "graph";
/// Retrieval source name of BM25 hits.
pub const SOURCE_BM25: &str = "bm25";
/// Retrieval source name of vector hits.
pub const SOURCE_VECTOR: &str = "vector";

/// Retrieves facts already in the graph that might contradict a new fact.
#[derive(Clone)]
pub struct SimilarFactRetriever {
    store: Arc<dyn GraphStore>,
    embedder: Arc<dyn Embedder>,
    config: SimilarFactRetrieverConfig,
}

impl fmt::Debug for SimilarFactRetriever {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimilarFactRetriever")
            .field("store", &self.store)
            .field("embedder", &self.embedder.model_name())
            .field("config", &self.config)
            .finish()
    }
}

impl SimilarFactRetriever {
    /// A retriever over `store`, embedding queries with `embedder`.
    #[must_use]
    pub fn new(
        store: Arc<dyn GraphStore>,
        embedder: Arc<dyn Embedder>,
        config: SimilarFactRetrieverConfig,
    ) -> Self {
        Self { store, embedder, config }
    }

    /// The retrieval settings.
    #[must_use]
    pub fn config(&self) -> &SimilarFactRetrieverConfig {
        &self.config
    }

    /// Candidates for `new_fact`: graph hits for each of `entity_ids`, then BM25 / vector hits
    /// for its text, minus `exclude_fact_ids`, ranked and capped (module docs).
    ///
    /// # Errors
    /// [`crate::ContradictionError::Store`] for a fatal store error; other store and embedding
    /// failures are logged and that source contributes nothing.
    pub fn retrieve_similar_facts(
        &self,
        new_fact: &ExtractedFact,
        entity_ids: &[EntityId],
        exclude_fact_ids: &[FactId],
    ) -> Result<Vec<RetrievedFact>> {
        todo!("worker A")
    }

    /// Entity ids each candidate is linked to by `FactAboutEntity` edges
    /// ([`GraphStore::get_fact_entities`]), keyed by candidate fact id in candidate order.
    /// Candidates without links are omitted. This is the `candidate_entity_map` the gating
    /// receives (Python always passed an empty map).
    ///
    /// # Errors
    /// [`crate::ContradictionError::Store`] for a fatal store error; other failures are
    /// logged and leave that candidate out.
    pub fn candidate_entity_map(&self, candidates: &[RetrievedFact]) -> Result<CandidateEntityMap> {
        todo!("worker A")
    }
}

/// A stored fact node as a [`RetrievedFact`] with no sources and score 0.0: the parsed
/// `AssertionTriple` (subject / relationship / object), the parsed `Metadata` JSON object
/// plus the node's `FactType`, `FactOrigin`, `Status` and `Certainty` under those keys, and
/// the side from metadata `side` or `er.side` ([`ce_text::normalize_side`]). `None` for a
/// node without text.
#[must_use]
pub fn retrieved_fact_from_node(stored: &StoredFact) -> Option<RetrievedFact> {
    todo!("worker A")
}
