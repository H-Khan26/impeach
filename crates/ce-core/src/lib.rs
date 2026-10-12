//! Data records shared by every contradiction-engine crate.
//!
//! Field names serialise exactly as the Python prototype's records so that JSONL payloads,
//! facts and contradiction reports diff cleanly across implementations: every field is always
//! written (`None` as `null`), fields keep the Python declaration order, and fields with a
//! Python default may be omitted on input.
//!
//! Modules mirror the Python `schema` package: [`entities`] (mentions, relationships,
//! coreference clusters), [`facts`], [`contradictions`], [`rag`] (documents, chunks, indexing
//! results) and [`ed`] (entity disambiguation). [`ids`] holds the string id newtypes,
//! [`vocab`] the value / property type vocabulary contract and [`pyjson`] the Python
//! `json.dumps` byte format used for every diffable output.

pub mod contradictions;
pub mod ed;
pub mod entities;
pub mod error;
pub mod facts;
pub mod ids;
pub mod pyjson;
pub mod rag;
pub mod vocab;

/// Free-form JSON object (Python `Dict[str, Any]`), insertion ordered.
pub type Metadata = serde_json::Map<String, serde_json::Value>;

pub use contradictions::{
    ContradictionPair, ContradictionReport, ContradictionReviewItem, ContradictionSource,
    ContradictionStatus, FactContradictionResult, NliLabel, RetrievedFact, ReviewClassification,
    ReviewStatus,
};
pub use ed::{DocContext, EdCandidate, EdContext, EdResult, LocalContext};
pub use entities::{
    CorefClusterContext, DetectionSource, Evidence, Mention, Relationship, RelationshipResponse,
    RelationshipType, SegmentEr, Span,
};
pub use error::{CoreError, Result};
pub use facts::{
    AssertionTriple, ExtractedFact, PropertyType, TemporalContext, TemporalValidity, TimeWindow,
    ValueType,
};
pub use ids::{ChunkId, DocId, EntityId, FactId, MentionId, NodeId, SpanId};
pub use rag::{
    Chunk, ChunkEmbeddingInput, ChunkWriteResult, ChunkingResult, DocumentPayload,
    DocumentWriteResult, EmbeddingWriteResult, IndexingResult, SentenceSpan, SpanWriteResult,
};
pub use vocab::{FallbackVocabulary, TypeVocabulary};
