//! Pipeline error type.

use std::path::PathBuf;

use ce_config::ConfigError;
use ce_ed::EdError;
use ce_facts::FactError;
use ce_ingest::IngestError;
use ce_models::ModelError;
use ce_store::StoreError;

/// Result alias for pipeline operations.
pub type Result<T, E = PipelineError> = std::result::Result<T, E>;

/// Failure while building or running a pipeline.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// The pipeline config could not be loaded or is invalid.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// The input JSONL could not be read (empty path, missing file, empty file, bad JSON,
    /// missing `text`).
    #[error(transparent)]
    Ingest(#[from] IngestError),
    /// A graph-store operation failed.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// A model could not be resolved or failed.
    #[error(transparent)]
    Model(#[from] ModelError),
    /// Entity disambiguation failure.
    #[error(transparent)]
    Ed(#[from] EdError),
    /// Fact extraction or indexing failure.
    #[error(transparent)]
    Fact(#[from] FactError),
    /// No constructor is registered for a stage type.
    #[error("unknown stage type `{stage_type}`; registered stage types: {known:?}")]
    UnknownStage {
        /// The `type` of the stage config.
        stage_type: String,
        /// Registered stage types, in registration order.
        known: Vec<String>,
    },
    /// A stage was given a config it cannot use.
    #[error("invalid configuration for stage `{stage}`: {message}")]
    InvalidStageConfig {
        /// Stage name or type.
        stage: String,
        /// What is wrong.
        message: String,
    },
    /// A stage needs a context value that no earlier stage or input provided.
    #[error("stage `{stage}` needs `{key}` in the pipeline context")]
    MissingContext {
        /// Stage name.
        stage: String,
        /// Context slot name.
        key: &'static str,
    },
    /// An input document has the wrong shape.
    #[error("invalid document {path}: {message}")]
    InvalidDocument {
        /// Source file.
        path: PathBuf,
        /// What is wrong.
        message: String,
    },
    /// An input file could not be read.
    #[error("failed to read {path}: {source}")]
    Io {
        /// File path.
        path: PathBuf,
        /// Underlying error.
        source: std::io::Error,
    },
    /// The embedder returned a different number of vectors than chunks.
    #[error("embedding count ({embeddings}) does not match chunk count ({chunks})")]
    EmbeddingCount {
        /// Vectors returned.
        embeddings: usize,
        /// Chunks embedded.
        chunks: usize,
    },
    /// The embedder's vectors do not fit the store's vector indexes.
    #[error("embedding dimension {actual} does not match the store's embedding_dim {expected}")]
    EmbeddingDimension {
        /// Dimension of the vectors returned.
        actual: usize,
        /// The store's dimension.
        expected: usize,
    },
    /// A stage failed; `source` is the stage's error.
    #[error("stage `{stage}` failed: {source}")]
    Stage {
        /// Stage name.
        stage: String,
        /// The stage's error.
        #[source]
        source: Box<PipelineError>,
    },
}

impl PipelineError {
    /// Whether the error means the database handle is gone for good.
    #[must_use]
    pub fn is_fatal(&self) -> bool {
        match self {
            Self::Store(error) => error.is_fatal(),
            Self::Ed(error) => error.is_fatal(),
            Self::Fact(error) => error.is_fatal(),
            Self::Stage { source, .. } => source.is_fatal(),
            _ => false,
        }
    }
}
