//! Model traits, the model registry and model backends.
//!
//! - [`traits`]: [`Embedder`], [`NliJudge`], [`Llm`], [`NerModel`], [`Coref`] and their
//!   records. Every text offset is a character offset (Python `str` semantics); backends that
//!   work in bytes or tokens convert at their boundary.
//! - [`registry`]: [`ModelRegistry`], keyed `<pipeline.name>_<type>` like the Python registry,
//!   constructs each configured model lazily on first access and shares it as `Arc<dyn Trait>`.
//!   Injected models (`insert_*`) take precedence over configured backends.
//! - [`mock`]: deterministic mocks of every trait (always compiled); every test uses these.
//! - [`json`]: lenient JSON extraction from LLM output (the Python code-fence stripping plus
//!   recovery from prose, trailing commas and raw control characters).
//! - [`sidecar`]: JSON-lines-over-stdio child processes (always compiled) and
//!   [`SidecarCoref`], the `fastcoref` coreference backend (fastcoref has no Rust port).
//! - [`rule_coref`]: [`RuleCoref`], deterministic name / alias matching and pronoun linking
//!   over the detected mentions (`models.coref.backend: rule`).
//! - [`llm_coref`]: [`LlmCoref`], clusters returned by an [`Llm`] as JSON
//!   (`models.coref.backend: llm`).
//!
//! # Backends and cargo features
//!
//! | Feature | Default | Module | Backend |
//! | --- | --- | --- | --- |
//! | `fastembed` | on | `fastembed` | `FastEmbedder`: `BAAI/bge-small-en-v1.5`, `all-MiniLM-L6-v2` through fastembed / ONNX Runtime |
//! | `ort-nli` | on | `ort_nli` | `OrtNli`: `cross-encoder/nli-deberta-v3-large` ONNX export from the Hugging Face hub, run with ort |
//! | `ollama` | on | `ollama` | `OllamaLlm`: Ollama HTTP API through ollama-rs |
//! | `gliner2` | on | `gliner2` | `Gliner2Ner`: GLiNER2 entities and relations through the gliner2 crate (candle) |
//!
//! ONNX Runtime binaries are downloaded at build time (`ort` `download-binaries`). Model files
//! are downloaded on first use: fastembed models into `.fastembed_cache/` (or
//! `FASTEMBED_CACHE_DIR`), Hugging Face models into the standard Hugging Face cache.
//! With a feature off, the registry reports [`ModelError::Unavailable`] naming the feature.

pub mod error;
#[cfg(feature = "fastembed")]
pub mod fastembed;
#[cfg(feature = "gliner2")]
pub mod gliner2;
pub mod json;
pub mod llm_coref;
pub mod mock;
#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(feature = "ort-nli")]
pub mod ort_nli;
pub mod registry;
pub mod rule_coref;
pub mod sidecar;
pub mod traits;

pub use error::{ModelError, Result};
#[cfg(feature = "fastembed")]
pub use fastembed::{FastEmbedOptions, FastEmbedder};
#[cfg(feature = "gliner2")]
pub use gliner2::{Gliner2Ner, Gliner2Options};
pub use llm_coref::{LlmCoref, LLM_COREF_PROMPT_PATH};
pub use mock::{MockCoref, MockEmbedder, MockLlm, MockNer, MockNli};
#[cfg(feature = "ollama")]
pub use ollama::{OllamaLlm, OllamaOptions};
#[cfg(feature = "ort-nli")]
pub use ort_nli::{OrtNli, OrtNliOptions};
pub use registry::{model_key, ModelKind, ModelRegistry};
pub use rule_coref::{RuleCoref, PRONOUN_SENTENCE_WINDOW};
pub use sidecar::{SidecarCommand, SidecarCoref, SidecarProcess};
pub use traits::{
    is_pronoun, normalize_l2, Coref, CorefCluster, CorefMention, Embedder, Llm, LlmRequest,
    NerModel, NerSpan, NliJudge, NliScores, RelationHit, PRONOUNS,
};
