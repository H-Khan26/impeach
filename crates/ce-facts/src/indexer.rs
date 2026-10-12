//! Fact indexing (Python `services/facts/indexer.py`): `Fact` nodes, fact embeddings and the
//! edges linking a fact to its document, entities and source spans.

use std::fmt;
use std::sync::Arc;

use ce_core::{EntityId, ExtractedFact, FactId, NodeId, SpanId, TypeVocabulary};
use ce_models::Embedder;
use ce_store::{FactNode, GraphStore};

use crate::error::Result;

/// Number of fact texts embedded per embedder call.
pub const FACT_EMBED_BATCH_SIZE: usize = 64;

/// Outcome of [`FactIndexer::index_facts`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndexResult {
    /// Facts written with their embedding.
    pub count: usize,
    /// Graph id per input fact, in input order (`None` when the node was not written).
    pub fact_ids: Vec<Option<FactId>>,
}

/// Edges to create for one written fact ([`FactIndexer::link_fact_context`]).
#[derive(Debug, Clone, Copy, Default)]
pub struct FactContextLinks<'a> {
    /// Graph id of the source `Document` (`FactInDocument`).
    pub doc_id: Option<&'a NodeId>,
    /// Subject entity (`FactAboutEntity`, role `SUBJECT`).
    pub subject_entity_id: Option<&'a EntityId>,
    /// Object entity (`FactAboutEntity`, role `OBJECT`).
    pub object_entity_id: Option<&'a EntityId>,
    /// Source spans (`FactDerivedFromSpan`, role `SOURCE_SPAN`, confidence 1.0).
    pub source_span_ids: &'a [SpanId],
    /// Page number of the `FactInDocument` edge.
    pub page_number: i64,
    /// Section of the `FactInDocument` edge.
    pub section: &'a str,
}

/// The `Fact` node of an extracted fact: text, JSON triple / validity / metadata, certainty
/// and the taxonomy-normalised value and property types.
#[must_use]
pub fn fact_node(fact: &ExtractedFact, vocab: &dyn TypeVocabulary) -> FactNode {
    let _ = (fact, vocab);
    todo!("worker B")
}

/// Writes facts, their embeddings and context edges.
pub struct FactIndexer {
    store: Arc<dyn GraphStore>,
    embedder: Arc<dyn Embedder>,
    batch_size: usize,
}

impl fmt::Debug for FactIndexer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FactIndexer")
            .field("embedder", &self.embedder.model_name())
            .field("batch_size", &self.batch_size)
            .finish_non_exhaustive()
    }
}

impl FactIndexer {
    /// Indexer over a store and an embedder, embedding [`FACT_EMBED_BATCH_SIZE`] texts per call.
    #[must_use]
    pub fn new(store: Arc<dyn GraphStore>, embedder: Arc<dyn Embedder>) -> Self {
        Self { store, embedder, batch_size: FACT_EMBED_BATCH_SIZE }
    }

    /// Sets the embedding batch size (at least 1).
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    /// Writes the nodes, then embeds the written ones and stores their vectors.
    ///
    /// # Errors
    /// [`crate::FactError::Store`] for a fatal store error; other failures are logged.
    pub fn index_facts(&self, nodes: &[FactNode]) -> Result<IndexResult> {
        let _ = (&self.store, nodes);
        todo!("worker B")
    }

    /// Creates the context edges of one written fact.
    ///
    /// # Errors
    /// [`crate::FactError::Store`] for a fatal store error; other failures are logged.
    pub fn link_fact_context(&self, fact_id: &FactId, links: &FactContextLinks<'_>) -> Result<()> {
        let _ = (fact_id, links);
        todo!("worker B")
    }

    /// Commits extracted facts: [`fact_node`] + [`FactIndexer::index_facts`], then the context
    /// edges of each written fact (`source_doc_id`, subject / object entity ids, the
    /// `source_span_ids` metadata list) and `metadata.helix_fact_id`.
    ///
    /// # Errors
    /// [`crate::FactError::Store`] for a fatal store error; other failures are logged.
    pub fn commit_facts(
        &self,
        facts: &mut [ExtractedFact],
        vocab: &dyn TypeVocabulary,
    ) -> Result<IndexResult> {
        let _ = (facts, vocab);
        todo!("worker B")
    }
}
