//! Pipeline stages. [`crate::StageRegistry::builtin`] registers each one under its config
//! `type` name.

pub mod coreference;
pub mod definition;
pub mod mention;
pub mod rag_index;

pub use coreference::CoreferenceStage;
pub use definition::DefinitionStage;
pub use mention::MentionStage;
pub use rag_index::RagIndexStage;
