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
    /// The backend is not compiled in, not configured, or its server is not reachable when the
    /// model is resolved.
    #[error("model backend `{backend}` is unavailable: {reason}")]
    Unavailable { backend: String, reason: String },
    /// A request to a model server failed to connect or timed out.
    #[error("connection failed: {0}")]
    Connection(String),
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

/// [`ModelError::Download`] for a model that is not in its cache and could not be downloaded:
/// names the model, the cache directory that was searched and how to fix it (`setting`).
#[cfg_attr(
    not(any(feature = "fastembed", feature = "ort-nli", feature = "gliner2")),
    allow(dead_code)
)]
pub(crate) fn missing_model(
    model: &str,
    cache_dir: &std::path::Path,
    setting: &str,
    cause: &str,
) -> ModelError {
    ModelError::Download {
        resource: model.to_owned(),
        message: format!(
            "the model is not in the model cache {} and could not be downloaded ({cause}). \
             Run once with network access to download it, copy the model files into that \
             directory, or {setting}",
            cache_dir.display()
        ),
    }
}

/// The Hugging Face hub cache: `$HF_HOME/hub` (when `honour_hf_home`), else
/// `<home>/.cache/huggingface/hub` with the home directory `hf-hub` uses (`dirs::home_dir`:
/// the user profile folder on Windows, `$HOME` elsewhere).
#[cfg_attr(not(any(feature = "ort-nli", feature = "gliner2")), allow(dead_code))]
pub(crate) fn hf_hub_cache_dir(honour_hf_home: bool) -> std::path::PathBuf {
    if honour_hf_home {
        if let Some(home) = std::env::var_os("HF_HOME") {
            return std::path::PathBuf::from(home).join("hub");
        }
    }
    let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("~"));
    home.join(".cache").join("huggingface").join("hub")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_model_names_model_cache_and_fix() {
        let error = missing_model(
            "BAAI/bge-small-en-v1.5",
            std::path::Path::new("cache-dir"),
            "set FASTEMBED_CACHE_DIR",
            "connection refused",
        );
        let text = error.to_string();
        assert!(text.starts_with("failed to download `BAAI/bge-small-en-v1.5`: "), "{text}");
        for part in ["cache-dir", "connection refused", "network access", "FASTEMBED_CACHE_DIR"] {
            assert!(text.contains(part), "`{part}` missing from: {text}");
        }
    }

    #[test]
    fn hub_cache_ends_with_huggingface_hub() {
        let dir = hf_hub_cache_dir(false);
        assert!(dir.ends_with(".cache/huggingface/hub"), "{}", dir.display());
    }
}
