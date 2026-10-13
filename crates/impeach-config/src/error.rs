//! Error type for config, mappings, prompt and asset loading.

use thiserror::Error;

/// Result alias used throughout `impeach-config`.
pub type Result<T, E = ConfigError> = std::result::Result<T, E>;

/// Errors raised while resolving, parsing or validating configuration files.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// The asset exists neither under the base directory nor in the embedded assets.
    #[error("asset not found: {path} (looked in the working directory and the embedded assets)")]
    AssetNotFound {
        /// Path as given by the caller.
        path: String,
    },
    /// Reading a file from disk failed.
    #[error("failed to read {path}: {source}")]
    Io {
        /// File that could not be read.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The asset is not valid UTF-8.
    #[error("{path} is not valid UTF-8")]
    NotUtf8 {
        /// Offending asset.
        path: String,
    },
    /// The YAML document is empty or contains only comments.
    #[error("config file is empty: {path}")]
    Empty {
        /// Empty file.
        path: String,
    },
    /// YAML syntax, type, missing-field or unknown-field error. The message names the
    /// dotted path of the offending node (e.g. `stages[7].retrieval: unknown field ...`).
    #[error("failed to parse {path}: {source}")]
    Yaml {
        /// File being parsed.
        path: String,
        /// Underlying parser error, including the node path and line/column.
        #[source]
        source: yaml_serde::Error,
    },
    /// JSON syntax or schema error.
    #[error("failed to parse {path}: {source}")]
    Json {
        /// File being parsed.
        path: String,
        /// Underlying parser error.
        #[source]
        source: serde_json::Error,
    },
    /// A value parsed but failed a validation rule.
    #[error("invalid config at `{path}`: {message}")]
    Validation {
        /// Dotted / indexed path of the offending field, e.g. `stages[3].min_score`.
        path: String,
        /// Human-readable reason, naming the bad value.
        message: String,
    },
    /// A regex in a mapping file does not compile.
    #[error("invalid regex `{name}` in {file}: {source}")]
    Regex {
        /// Mapping file containing the pattern.
        file: String,
        /// Pattern name or location inside the file.
        name: String,
        /// Compilation error.
        #[source]
        source: Box<regex::Error>,
    },
    /// A GLiNER2 label set name is not defined in `gliner2.yaml`.
    #[error("{kind} label set '{name}' not found. Available: {available:?}")]
    UnknownLabelSet {
        /// `Entity` or `Relation`.
        kind: &'static str,
        /// Requested name.
        name: String,
        /// Defined names, in file order.
        available: Vec<String>,
    },
    /// A prompt file does not have the shape the caller asked for.
    #[error("{path} is not {expected} prompt")]
    PromptKind {
        /// Prompt file.
        path: String,
        /// Expected kind (`an ED`, `an extraction` or `a judge`).
        expected: &'static str,
    },
}

impl ConfigError {
    /// Builds a [`ConfigError::Validation`].
    pub fn validation(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Validation { path: path.into(), message: message.into() }
    }
}
