//! Error type for `ce-core`.

use thiserror::Error;

/// Errors raised when building or decoding core records.
#[derive(Debug, Error)]
pub enum CoreError {
    /// A record failed to (de)serialise.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// A numeric field is outside its allowed range.
    #[error("{field} must be in [{min}, {max}], got {value}")]
    OutOfRange {
        /// Field name.
        field: &'static str,
        /// Offending value.
        value: f64,
        /// Inclusive lower bound.
        min: f64,
        /// Inclusive upper bound.
        max: f64,
    },
}

/// Result alias for `ce-core`.
pub type Result<T, E = CoreError> = std::result::Result<T, E>;
