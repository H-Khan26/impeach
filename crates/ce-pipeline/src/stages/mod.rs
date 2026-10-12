//! Pipeline stages. [`crate::StageRegistry::builtin`] registers each one under its config
//! `type` name.

pub mod coreference;
pub mod definition;
pub mod ed;
pub mod fact;
pub mod mention;
pub mod rag_index;
pub mod relation;

pub use coreference::CoreferenceStage;
pub use definition::DefinitionStage;
pub use ed::EdStage;
pub use fact::FactStage;
pub use mention::MentionStage;
pub use rag_index::RagIndexStage;
pub use relation::RelationStage;
