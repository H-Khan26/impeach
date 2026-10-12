//! Model error type.

/// Errors raised by model backends and the registry.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// A model could not be loaded (missing files, bad format, runtime init failure).
    #[error("failed to load model `{model}`: {message}")]
    Load { model: String, message: String },
    /// Model files could not be downloaded.
    #[error("failed to download `{resource}`: {message}")]
    Download { resource: String, message: String },
    /// The model ran but inference failed.
    #[error("inference failed: {0}")]
    Inference(String),
    /// A backend returned output that does not follow its protocol (bad JSON, wrong shape).
    #[error("protocol error: {0}")]
    Protocol(String),
    /// The backend is not compiled in or not configured.
    #[error("model backend `{backend}` is unavailable: {reason}")]
    Unavailable { backend: String, reason: String },
    /// No model is registered under the key.
    #[error("unknown model key `{key}`; known keys: {known:?}")]
    UnknownModel { key: String, known: Vec<String> },
    /// I/O failure (sidecar pipes, cache directories).
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Result alias for model operations.
pub type Result<T, E = ModelError> = std::result::Result<T, E>;

/// `error` and its sources joined with `: ` (many backend errors keep the cause in `source`).
#[cfg_attr(not(any(feature = "fastembed", feature = "ollama")), allow(dead_code))]
pub(crate) fn error_chain(error: &dyn std::error::Error) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(e) = source {
        let text = e.to_string();
        if !message.ends_with(&text) {
            message.push_str(": ");
            message.push_str(&text);
        }
        source = e.source();
    }
    message
}
