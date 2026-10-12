//! Error type of the entity disambiguation crate.

use ce_config::ConfigError;
use ce_models::ModelError;
use ce_store::StoreError;

/// Result alias for entity disambiguation.
pub type Result<T, E = EdError> = std::result::Result<T, E>;

/// Failure of entity disambiguation.
///
/// Retrieval and entity creation follow the Python error policy: store failures are logged and
/// treated as "no candidates" / "no entity", except fatal ones ([`StoreError::is_fatal`], a lost
/// database), which always propagate. Model failures of the LLM tie-break are logged and leave
/// the mention unresolved.
#[derive(Debug, thiserror::Error)]
pub enum EdError {
    /// Configuration or asset loading failed (mappings, roles, prompt file).
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// A fatal graph store failure.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// A model failure that cannot be skipped.
    #[error(transparent)]
    Model(#[from] ModelError),
    /// The ED prompt is missing or malformed.
    #[error("ED prompt: {0}")]
    Prompt(String),
}

impl EdError {
    /// Whether the error must stop the pipeline (a lost database).
    #[must_use]
    pub fn is_fatal(&self) -> bool {
        matches!(self, Self::Store(error) if error.is_fatal())
    }
}

/// Applies the store error policy: a fatal error propagates, any other error is logged with
/// `what` and replaced by `T::default()`.
///
/// # Errors
/// [`EdError::Store`] for a fatal store error.
pub fn recover_store<T: Default>(result: Result<T, StoreError>, what: &str) -> Result<T> {
    match result {
        Ok(value) => Ok(value),
        Err(error) if error.is_fatal() => Err(error.into()),
        Err(error) => {
            tracing::warn!(error = %error, "{what} failed");
            Ok(T::default())
        }
    }
}
