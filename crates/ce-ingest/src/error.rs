//! Ingestion errors.

use std::path::PathBuf;

/// Errors raised while loading, parsing or writing documents.
#[derive(Debug, thiserror::Error)]
pub enum IngestError {
    /// The input path is empty.
    #[error("file path cannot be empty")]
    EmptyPath,
    /// The input file does not exist.
    #[error("source file does not exist: {}", .0.display())]
    NotFound(PathBuf),
    /// A standard document has an extension other than `.pdf`, `.txt` or `.json` (the Python
    /// suffix, with its dot, or empty).
    #[error("unsupported file type: {0:?}")]
    UnsupportedExtension(String),
    /// A transcript has an extension other than `.json`, `.csv`, `.txt` or `.text`.
    #[error("unsupported transcript format: {0:?}")]
    UnsupportedTranscriptFormat(String),
    /// A feature that is not available in this build: `llm_cleaning`, `vision_ocr`, or `pdf`
    /// when the `pdf` cargo feature is off.
    #[error("{0} is not supported")]
    Unsupported(&'static str),
    /// Reading or writing a file failed.
    #[error("I/O error on {}: {source}", path.display())]
    Io {
        /// File being read or written.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },
    /// The file is not valid UTF-8.
    #[error("invalid UTF-8 in {}: {source}", path.display())]
    Encoding {
        /// File being read.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: std::str::Utf8Error,
    },
    /// Malformed JSON, or a record that cannot be serialised.
    #[error("invalid JSON in {}: {source}", path.display())]
    Json {
        /// File being read or written.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: serde_json::Error,
    },
    /// Malformed CSV transcript.
    #[error("invalid CSV in {}: {source}", path.display())]
    Csv {
        /// File being read.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: csv::Error,
    },
    /// PDF text extraction failed.
    #[error("error reading PDF file {}: {message}", path.display())]
    Pdf {
        /// File being read.
        path: PathBuf,
        /// Extractor message.
        message: String,
    },
    /// A JSON document or JSONL record has the wrong shape (missing `doc_id` / `text`, wrong
    /// field types).
    #[error("invalid document {}: {reason}", path.display())]
    InvalidDocument {
        /// File being read.
        path: PathBuf,
        /// What is wrong.
        reason: String,
    },
    /// A transcript cannot be flattened (non-object utterances, unhashable or unsortable
    /// speakers).
    #[error("invalid transcript {}: {reason}", path.display())]
    InvalidTranscript {
        /// File being read.
        path: PathBuf,
        /// What is wrong.
        reason: String,
    },
    /// The first line of a JSONL file is empty.
    #[error("JSONL file is empty: {}", .0.display())]
    EmptyJsonl(PathBuf),
    /// The first JSONL record is not an object with a non-empty `text` field.
    #[error("JSONL record in {} must contain a non-empty 'text' field", .0.display())]
    MissingText(PathBuf),
}

/// Result alias for this crate.
pub type Result<T, E = IngestError> = std::result::Result<T, E>;
