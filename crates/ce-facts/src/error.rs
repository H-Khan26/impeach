//! Error type of the fact crate.

use ce_config::ConfigError;
use ce_models::ModelError;
use ce_store::StoreError;

/// Result alias for fact extraction and indexing.
pub type Result<T, E = FactError> = std::result::Result<T, E>;

/// Failure of fact extraction or indexing.
///
/// LLM failures and malformed LLM output are logged and yield no facts for that chunk or pass;
/// non-fatal store failures are logged and leave the affected fact unwritten. Only fatal store
/// errors ([`StoreError::is_fatal`]) and configuration errors propagate.
#[derive(Debug, thiserror::Error)]
pub enum FactError {
    /// A prompt file or mapping could not be loaded.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// A model failure that cannot be skipped.
    #[error(transparent)]
    Model(#[from] ModelError),
    /// A fatal graph store failure.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// LLM output that is not a fact graph (`{"nodes": [...]}`).
    #[error("malformed fact extraction output: {0}")]
    Parse(String),
}

impl FactError {
    /// Whether the error must stop the pipeline (a lost database).
    #[must_use]
    pub fn is_fatal(&self) -> bool {
        matches!(self, Self::Store(error) if error.is_fatal())
    }
}
