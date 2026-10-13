//! Error type of the contradiction crate.

use impeach_config::ConfigError;
use impeach_facts::FactError;
use impeach_store::StoreError;

/// Result alias for the contradiction check.
pub type Result<T, E = ContradictionError> = std::result::Result<T, E>;

/// Failure of the contradiction check.
///
/// Judge (NLI / LLM) failures are logged and score the pair as neutral with confidence 0.0;
/// non-fatal store failures during retrieval are logged and yield no candidates. Fatal store
/// errors ([`StoreError::is_fatal`]), configuration errors and commit errors propagate.
#[derive(Debug, thiserror::Error)]
pub enum ContradictionError {
    /// A prompt file or mapping could not be loaded, or the settings are inconsistent.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// A fatal graph store failure, or a review-store (SQLite) failure.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// Committing clean facts failed.
    #[error(transparent)]
    Fact(#[from] FactError),
}

impl ContradictionError {
    /// Whether the error must stop the pipeline (a lost database).
    #[must_use]
    pub fn is_fatal(&self) -> bool {
        match self {
            Self::Store(error) => error.is_fatal(),
            Self::Fact(error) => error.is_fatal(),
            Self::Config(_) => false,
        }
    }
}
