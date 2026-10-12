//! The RAG substrate used by RAGIndexStage: chunking, the document-identity cache and graph
//! writes (Python `services/rag/`).

pub mod chunker;
pub mod state;
pub mod writer;

pub use chunker::Chunker;
pub use state::{default_document_cache_path, document_identity_key, DocumentStateStore};
