//! Error type of mention detection.

use ce_config::ConfigError;
use ce_models::ModelError;

/// Result alias for mention detection.
pub type Result<T, E = NerError> = std::result::Result<T, E>;

/// Failure while building or running a detector.
#[derive(Debug, thiserror::Error)]
pub enum NerError {
    /// Mappings or roles config could not be loaded or is invalid.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// The NER model failed or is unavailable.
    #[error(transparent)]
    Model(#[from] ModelError),
    /// A detector was given settings it cannot use.
    #[error("invalid detector configuration: {0}")]
    InvalidConfig(String),
}
