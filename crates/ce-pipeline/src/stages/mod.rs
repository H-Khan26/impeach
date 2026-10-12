//! Pipeline stages. [`crate::StageRegistry::builtin`] registers each one under its config
//! `type` name.

pub mod rag_index;

pub use rag_index::RagIndexStage;
