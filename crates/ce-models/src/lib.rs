//! Model traits, the model registry and model backends.
//!
//! - [`traits`]: [`Embedder`], [`NliJudge`], [`Llm`], [`NerModel`], [`Coref`] and their
//!   records. Every text offset is a character offset (Python `str` semantics).
//! - [`registry`]: [`ModelRegistry`], keyed `<pipeline.name>_<type>`, lazy construction.
//! - [`mock`]: deterministic mocks of every trait (always compiled).
//! - [`json`]: lenient JSON extraction from LLM output.

pub mod error;
pub mod json;
pub mod mock;
pub mod registry;
pub mod traits;

pub use error::{ModelError, Result};
pub use mock::{MockCoref, MockEmbedder, MockLlm, MockNer, MockNli};
pub use registry::{model_key, ModelKind, ModelRegistry};
pub use traits::{
    Coref, CorefCluster, Embedder, Llm, LlmRequest, NerModel, NerSpan, NliJudge, NliScores,
    RelationHit,
};
