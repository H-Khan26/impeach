//! The [`GraphStore`] trait: one method per query of the Python Helix client, plus batch writes
//! and the chunk cleanup of re-indexed documents.

use impeach_core::{ChunkId, EntityId, FactId, NodeId, SpanId};

use crate::error::Result;
use crate::graph::{
    ChunkNode, DocumentNode, EmbeddingWrite, EntityHit, EntityNode, FactHit, FactNode, SpanNode,
    StoredChunk, StoredDocument, StoredEntity, StoredFact, StoredSpan, VectorLabel,
};

/// Graph, vector and text storage used by the pipeline.
///
/// Every method is one atomic transaction. Methods take `&self` (backends use interior
/// mutability) so one store is shared as `Arc<dyn GraphStore>`.
///
/// Errors common to every method: [`crate::StoreError::Validation`] for malformed input
/// (including ids that are not graph node ids), [`crate::StoreError::NotFound`] when a
/// referenced node does not exist or has another label, and, for database-backed stores,
/// [`crate::StoreError::Connection`], [`crate::StoreError::Conflict`] and
/// [`crate::StoreError::DatabaseLost`] (fatal; never swallow it).
///
/// Searches return hits best first. Vector scores are cosine similarity clamped to `[0, 1]`;
/// text scores are raw BM25 scores. Search limits are clamped to
/// [`crate::graph::MAX_SEARCH_K`]; a limit of 0 or an empty text query returns no hits.
pub trait GraphStore: Send + Sync + std::fmt::Debug {
    /// Random id kept in the database; changes when the database is reset. Client-side caches
    /// of node ids are keyed on it.
    fn store_id(&self) -> String;

    /// Dimension of every vector index.
    fn embedding_dim(&self) -> usize;

    /// Deletes every node and edge and assigns a new [`GraphStore::store_id`].
    ///
    /// # Errors
    /// Database failures.
    fn reset(&self) -> Result<()>;

    /// `CreateDocument`: adds a `Document` node.
    ///
    /// # Errors
    /// Database failures.
    fn create_document(&self, document: &DocumentNode) -> Result<StoredDocument>;

    /// `UpdateDocument`: overwrites every property of an existing `Document`.
    ///
    /// # Errors
    /// `NotFound` when `doc_id` is not a document.
    fn update_document(&self, doc_id: &NodeId, document: &DocumentNode) -> Result<StoredDocument>;

    /// `GetDocumentChunks`: chunks linked to a document by `ChunkOfDocument`, each once
    /// (empty when the document does not exist).
    ///
    /// # Errors
    /// `Validation` for a malformed id; database failures.
    fn get_document_chunks(&self, doc_id: &NodeId) -> Result<Vec<StoredChunk>>;

    /// `CreateDocumentChunk`: adds a `DocumentChunk` and its `ChunkOfDocument` edge
    /// (`SequenceNumber` = the chunk's sequence number).
    ///
    /// # Errors
    /// `NotFound` when `doc_id` is not a document (nothing is written).
    fn create_document_chunk(&self, doc_id: &NodeId, chunk: &ChunkNode) -> Result<StoredChunk>;

    /// `UpdateDocumentChunk`: overwrites every property of an existing chunk.
    ///
    /// # Errors
    /// `NotFound` when `chunk_id` is not a chunk.
    fn update_document_chunk(&self, chunk_id: &ChunkId, chunk: &ChunkNode) -> Result<StoredChunk>;

    /// `CreateChunkEmbedding`: sets the chunk's vector (replacing any previous one).
    ///
    /// # Errors
    /// `Validation` for a malformed vector; `NotFound` when `chunk_id` is not a chunk.
    fn create_chunk_embedding(&self, chunk_id: &ChunkId, embedding: &[f32]) -> Result<()>;

    /// `CreateSpanForChunk`: adds a `Span` and its `SpanOfChunk` edge to `span.chunk_id`
    /// (`SequenceNumber` = the span's start offset).
    ///
    /// # Errors
    /// `NotFound` when `span.chunk_id` is not a chunk (nothing is written).
    fn create_span_for_chunk(&self, span: &SpanNode) -> Result<StoredSpan>;

    /// `CreateEntity`: validates and adds an `Entity`.
    ///
    /// # Errors
    /// `Validation` per [`EntityNode::validated`].
    fn create_entity(&self, entity: &EntityNode) -> Result<StoredEntity>;

    /// `CreateEntityEmbedding`: sets the entity's vector.
    ///
    /// # Errors
    /// `Validation` for a malformed vector; `NotFound` when `entity_id` is not an entity.
    fn create_entity_embedding(&self, entity_id: &EntityId, embedding: &[f32]) -> Result<()>;

    /// `FindEntityByName`: BM25 search over `Name` and `AliasesJson`, merged per entity
    /// (best score wins), top [`crate::graph::ENTITY_NAME_SEARCH_K`].
    ///
    /// # Errors
    /// Database failures.
    fn find_entity_by_name(&self, name: &str) -> Result<Vec<EntityHit>>;

    /// `FindSimilarEntities`: cosine search over entity vectors.
    ///
    /// # Errors
    /// `Validation` for a malformed query vector.
    fn find_similar_entities(&self, embedding: &[f32], limit: usize) -> Result<Vec<EntityHit>>;

    /// `GetAllEntities`: the first `limit` entities in id order.
    ///
    /// # Errors
    /// Database failures.
    fn get_all_entities(&self, limit: usize) -> Result<Vec<StoredEntity>>;

    /// `LinkSpanToEntity`: adds `SpanRefersToEntity` (`Confidence`, `Method`; an empty method
    /// becomes `NER+ED`).
    ///
    /// # Errors
    /// `Validation` when `confidence` is outside `[0, 1]`; `NotFound` for a missing endpoint.
    fn link_span_to_entity(
        &self,
        span_id: &SpanId,
        entity_id: &EntityId,
        confidence: f64,
        method: &str,
    ) -> Result<()>;

    /// `CreateFact`: validates and adds a `Fact`.
    ///
    /// # Errors
    /// `Validation` per [`FactNode::validate`].
    fn create_fact(&self, fact: &FactNode) -> Result<StoredFact>;

    /// `CreateFactEmbedding`: sets the fact's vector.
    ///
    /// # Errors
    /// `Validation` for a malformed vector; `NotFound` when `fact_id` is not a fact.
    fn create_fact_embedding(&self, fact_id: &FactId, embedding: &[f32]) -> Result<()>;

    /// `LinkFactToDocument`: adds `FactInDocument` (`PageNumber`, `Section`).
    ///
    /// # Errors
    /// `NotFound` for a missing endpoint.
    fn link_fact_to_document(
        &self,
        fact_id: &FactId,
        doc_id: &NodeId,
        page_number: i64,
        section: &str,
    ) -> Result<()>;

    /// `LinkFactToEntity`: adds `FactAboutEntity` (`Role`; empty becomes `RELATED`).
    ///
    /// # Errors
    /// `NotFound` for a missing endpoint.
    fn link_fact_to_entity(&self, fact_id: &FactId, entity_id: &EntityId, role: &str)
        -> Result<()>;

    /// `LinkFactToSpan`: adds `FactDerivedFromSpan` (`Role`, empty becomes `SOURCE_SPAN`;
    /// `Confidence`).
    ///
    /// # Errors
    /// `Validation` when `confidence` is outside `[0, 1]`; `NotFound` for a missing endpoint.
    fn link_fact_to_span(
        &self,
        fact_id: &FactId,
        span_id: &SpanId,
        role: &str,
        confidence: f64,
    ) -> Result<()>;

    /// `GetEntityFacts`: facts linked to the entity by `FactAboutEntity`, each once (empty
    /// when the entity does not exist).
    ///
    /// # Errors
    /// `Validation` for a malformed id; database failures.
    fn get_entity_facts(&self, entity_id: &EntityId) -> Result<Vec<StoredFact>>;

    /// `GetFactEntities`: entities the fact is linked to by `FactAboutEntity`, each once, in
    /// first-edge order (empty when the fact does not exist).
    ///
    /// # Errors
    /// `Validation` for a malformed id; database failures.
    fn get_fact_entities(&self, fact_id: &FactId) -> Result<Vec<EntityId>>;

    /// `FindFactsByText`: BM25 search over `FactText`.
    ///
    /// # Errors
    /// Database failures.
    fn find_facts_by_text(&self, keywords: &str, limit: usize) -> Result<Vec<FactHit>>;

    /// `FindSimilarFacts`: cosine search over fact vectors.
    ///
    /// # Errors
    /// `Validation` for a malformed query vector.
    fn find_similar_facts(&self, embedding: &[f32], limit: usize) -> Result<Vec<FactHit>>;

    /// Batch `CreateDocumentChunk`: adds every chunk under one document, results in input
    /// order. All or nothing per transaction; backends may split very large inputs into
    /// several transactions. An empty input writes nothing and does not look up `doc_id`.
    ///
    /// # Errors
    /// `Validation` for a malformed `doc_id`; `NotFound` when `chunks` is not empty and
    /// `doc_id` is not a document.
    fn create_chunks_batch(
        &self,
        doc_id: &NodeId,
        chunks: &[ChunkNode],
    ) -> Result<Vec<StoredChunk>>;

    /// Deletes every chunk linked to the document by `ChunkOfDocument`, the spans linked to
    /// those chunks by `SpanOfChunk`, their vectors and every edge touching them, in one
    /// transaction; the `Document` node itself is kept. Returns the number of chunks deleted
    /// (0 when the document does not exist).
    ///
    /// # Errors
    /// `Validation` for a malformed id; database failures.
    fn delete_document_chunks(&self, doc_id: &NodeId) -> Result<usize>;

    /// Batch `CreateSpanForChunk`: results in input order; transactions as in
    /// [`GraphStore::create_chunks_batch`].
    ///
    /// # Errors
    /// `NotFound` when any `chunk_id` is not a chunk.
    fn create_spans_batch(&self, spans: &[SpanNode]) -> Result<Vec<StoredSpan>>;

    /// Batch `Create*Embedding` for one label; returns the number of vectors written.
    /// Every vector is validated before anything is written.
    ///
    /// # Errors
    /// `Validation` for a malformed id or vector; `NotFound` when an id is not a `label` node.
    fn create_embeddings_batch(
        &self,
        label: VectorLabel,
        items: &[EmbeddingWrite<'_>],
    ) -> Result<usize>;
}
