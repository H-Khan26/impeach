//! Graph storage: the [`GraphStore`] trait with HelixDB and in-memory backends, and the SQLite
//! contradiction review queue.
//!
//! Python origin: `storage/helix/{client,connection,queries,models,errors}.py` and
//! `storage/sqlite/review_store.py`.
//!
//! - [`graph`]: the graph model (labels, PascalCase properties, edges, indexes), typed nodes
//!   ([`DocumentNode`], [`ChunkNode`], [`SpanNode`], [`EntityNode`], [`FactNode`]) with the
//!   Python parameter validation, and the result types [`Stored`] and [`SearchHit`].
//! - [`GraphStore`]: one method per query of the Python Helix client, plus `store_id`,
//!   `reset`, batch writes for chunks, spans, embeddings, facts, span-entity and fact edges,
//!   `create_entity_with_embedding`, and `delete_document_chunks`
//!   (clears a document's chunks before it is re-indexed). Synchronous and object safe; share
//!   one store as `Arc<dyn GraphStore>`.
//! - [`MemoryStore`]: complete in-process backend (brute-force cosine, Okapi BM25 from
//!   [`bm25`]); used by tests and model-free runs.
//! - `HelixStore` (cargo feature `helix`, on by default): embedded HelixDB v3 on disk or in
//!   memory, or a HelixDB server, with the single-writer file lock, index creation and
//!   transaction-conflict retry of the Python connection layer.
//! - [`SqliteReviewStore`]: the `contradiction_review_queue` table.
//!
//! Search scores: vector hits carry cosine similarity clamped to `[0, 1]`; text hits carry the
//! raw BM25 score, which differs between backends (see [`bm25`]).
//!
//! ```
//! use impeach_store::{EntityNode, GraphStore, MemoryStore};
//!
//! let store = MemoryStore::new(3);
//! let acme = store.create_entity(&EntityNode::new("Acme Corp.", "ORG"))?;
//! store.create_entity_embedding(&acme.id, &[1.0, 0.0, 0.0])?;
//! let hits = store.find_similar_entities(&[0.9, 0.1, 0.0], 5)?;
//! assert_eq!(hits[0].node.id, acme.id);
//! assert!(hits[0].score > 0.9);
//! # Ok::<(), impeach_store::StoreError>(())
//! ```

pub mod bm25;
pub mod error;
pub mod graph;
#[cfg(feature = "helix")]
pub mod helix;
pub mod memory;
pub mod review_store;
mod store;

pub use error::{Result, StoreError};
pub use graph::{
    ChunkNode, DocumentNode, EmbeddingWrite, EntityHit, EntityNode, FactHit, FactLink, FactNode,
    SearchHit, SpanEntityLink, SpanNode, Stored, StoredChunk, StoredDocument, StoredEntity,
    StoredFact, StoredSpan, VectorLabel,
};
#[cfg(feature = "helix")]
pub use helix::{HelixMode, HelixOptions, HelixStore};
pub use memory::MemoryStore;
pub use review_store::{ReviewStoreConfig, SqliteReviewStore};
pub use store::GraphStore;

/// Embedding dimension of `BAAI/bge-small-en-v1.5`, the Python client default.
pub const DEFAULT_EMBEDDING_DIM: usize = 384;
