//! Evaluator error type.

use std::path::PathBuf;

use impeach_models::ModelError;

/// Errors raised while loading inputs, embedding or writing the report.
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    /// A file could not be read or written.
    #[error("{path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// The I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The pipeline output is not valid JSON of the expected shape.
    #[error("{path}: invalid e2e output JSON: {source}")]
    Json {
        /// The file.
        path: PathBuf,
        /// The parse error.
        #[source]
        source: serde_json::Error,
    },
    /// The report could not be serialised.
    #[error("failed to serialise the report: {0}")]
    Serialize(#[source] serde_json::Error),
    /// The key has no `C-*` or `D*` rows.
    #[error("No contradictions parsed from key: {}", path.display())]
    EmptyKey {
        /// The key file.
        path: PathBuf,
    },
    /// The embedder failed.
    #[error(transparent)]
    Model(#[from] ModelError),
    /// The embedder returned a different number of vectors than texts.
    #[error("embedder returned {got} vectors for {expected} texts")]
    EmbeddingCount {
        /// Texts sent.
        expected: usize,
        /// Vectors received.
        got: usize,
    },
    /// The score matrix holds NaN (scipy raises `ValueError` here).
    #[error("score matrix contains NaN")]
    InvalidScores,
}

/// Result alias for evaluator operations.
pub type Result<T, E = EvalError> = std::result::Result<T, E>;
