//! Strict pipeline configuration, mappings registry, prompt loading and asset resolution.
//!
//! Every config struct rejects unknown fields, and YAML field names match the Python
//! prototype so the same files load in both implementations.
//!
//! - [`pipeline`]: `PipelineConfig` and every stage / model / infrastructure block, with
//!   [`PipelineConfig::validate`] reproducing the Pydantic validators.
//! - [`services`]: contradiction-check service configs built from a validated stage block.
//! - [`mappings`]: `MappingsRegistry` over the ten `config/mappings/*.yaml` files.
//! - [`prompts`]: prompt JSON files.
//! - [`assets`]: embedded assets and path resolution (working directory first, then the
//!   embedded copy, with `impeach/` mapped to the asset root).
//!
//! Parse errors name the YAML path of the offending node, e.g.
//! `stages[7].retrieval: unknown field `bm25_limt``.

pub mod assets;
mod de;
pub mod error;
pub mod mappings;
pub mod pipeline;
pub mod prompts;
pub mod services;

pub use assets::{
    load_config, load_config_in, load_default_config, resolve_asset, resolve_asset_in,
    ResolvedAsset,
};
pub use error::{ConfigError, Result};
pub use mappings::MappingsRegistry;
pub use pipeline::{PipelineConfig, StageConfig};
pub use prompts::PromptTemplate;
