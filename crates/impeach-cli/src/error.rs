//! Error reporting: the Python CLI's stderr prefixes and the message text.
//!
//! Python prints `Error:` for `FileNotFoundError`, `Configuration Error:` for `ValueError`
//! (invalid config, invalid input document) and `Pipeline Error:` for anything raised while
//! the pipeline runs. Errors outside the pipeline (eval, store) use `Error:`.

use std::io;

use impeach_config::ConfigError;
use impeach_ingest::IngestError;
use impeach_pipeline::PipelineError;

/// Category of a failure, selecting the stderr prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// A file is missing, or a failure outside the pipeline.
    General,
    /// Invalid config or input document.
    Configuration,
    /// A failure while the pipeline runs.
    Pipeline,
}

impl ErrorKind {
    /// The stderr prefix, without the colon.
    pub fn prefix(self) -> &'static str {
        match self {
            Self::General => "Error",
            Self::Configuration => "Configuration Error",
            Self::Pipeline => "Pipeline Error",
        }
    }
}

/// Classifies an error by the first known error type in its chain.
pub fn classify(error: &anyhow::Error) -> ErrorKind {
    if let Some(e) = error.downcast_ref::<PipelineError>() {
        return pipeline_kind(e);
    }
    if let Some(e) = error.downcast_ref::<ConfigError>() {
        return config_kind(e);
    }
    if let Some(e) = error.downcast_ref::<IngestError>() {
        return ingest_kind(e);
    }
    ErrorKind::General
}

fn pipeline_kind(error: &PipelineError) -> ErrorKind {
    match error {
        PipelineError::Config(e) => config_kind(e),
        PipelineError::Ingest(e) => ingest_kind(e),
        PipelineError::Io { source, .. } if source.kind() == io::ErrorKind::NotFound => {
            ErrorKind::General
        }
        PipelineError::UnknownStage { .. }
        | PipelineError::InvalidStageConfig { .. }
        | PipelineError::InvalidDocument { .. } => ErrorKind::Configuration,
        PipelineError::Stage { source, .. } => match pipeline_kind(source) {
            ErrorKind::Configuration => ErrorKind::Configuration,
            _ => ErrorKind::Pipeline,
        },
        _ => ErrorKind::Pipeline,
    }
}

fn config_kind(error: &ConfigError) -> ErrorKind {
    match error {
        ConfigError::AssetNotFound { .. } => ErrorKind::General,
        ConfigError::Io { source, .. } if source.kind() == io::ErrorKind::NotFound => {
            ErrorKind::General
        }
        _ => ErrorKind::Configuration,
    }
}

fn ingest_kind(error: &IngestError) -> ErrorKind {
    match error {
        IngestError::NotFound(_) => ErrorKind::General,
        IngestError::Io { source, .. } if source.kind() == io::ErrorKind::NotFound => {
            ErrorKind::General
        }
        IngestError::Io { .. } | IngestError::Pdf { .. } => ErrorKind::Pipeline,
        _ => ErrorKind::Configuration,
    }
}

/// The error message: the top-level message followed by each cause whose text it does not
/// already contain (the library errors embed their source in their own message).
pub fn message(error: &anyhow::Error) -> String {
    let mut text = error.to_string();
    for cause in error.chain().skip(1) {
        let cause = cause.to_string();
        if !text.contains(&cause) {
            text.push_str(": ");
            text.push_str(&cause);
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn missing_files_are_general_errors() {
        let error = anyhow::Error::new(PipelineError::Ingest(IngestError::NotFound(
            PathBuf::from("missing.jsonl"),
        )));
        assert_eq!(classify(&error), ErrorKind::General);
        let error = anyhow::Error::new(ConfigError::AssetNotFound { path: "x.yaml".into() });
        assert_eq!(classify(&error), ErrorKind::General);
    }

    #[test]
    fn invalid_config_and_documents_are_configuration_errors() {
        let error = anyhow::Error::new(PipelineError::Config(ConfigError::validation(
            "helix.namespace",
            "must not be empty",
        )));
        assert_eq!(classify(&error), ErrorKind::Configuration);
        let error = anyhow::Error::new(IngestError::MissingText(PathBuf::from("doc.jsonl")));
        assert_eq!(classify(&error), ErrorKind::Configuration);
    }

    #[test]
    fn stage_failures_are_pipeline_errors() {
        let error = anyhow::Error::new(PipelineError::Stage {
            stage: "rag_index".into(),
            source: Box::new(PipelineError::EmbeddingCount { embeddings: 1, chunks: 2 }),
        });
        assert_eq!(classify(&error), ErrorKind::Pipeline);
        assert_eq!(
            message(&error),
            "stage `rag_index` failed: embedding count (1) does not match chunk count (2)"
        );
    }

    #[test]
    fn context_is_joined_with_its_cause() {
        let error = anyhow::Error::new(io::Error::other("disk full")).context("writing out.json");
        assert_eq!(message(&error), "writing out.json: disk full");
        assert_eq!(classify(&error), ErrorKind::General);
    }
}
