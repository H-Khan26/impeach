//! Document ingestion: file loaders (pdf, txt, json), the transcript parser and JSONL
//! `DocumentPayload` output (Python `api/ingestion.py`, `services/rag/document_loader.py`,
//! `transcript/transcript_parser.py`, and the JSONL reader of `api/processing.py`).
//!
//! [`run_ingestion`] turns one source file into a one-line JSONL file holding a
//! [`ce_core::DocumentPayload`]; [`read_first_record`] reads it back for processing.
//!
//! ```no_run
//! use ce_ingest::{run_ingestion, IngestOptions};
//!
//! let options = IngestOptions {
//!     doc_type: Some("complaint".to_owned()),
//!     ..IngestOptions::new("CASE-123", "DB-456")
//! };
//! let result = run_ingestion("documents/complaint.pdf", "output/complaint.jsonl", &options)?;
//! let payload = ce_ingest::read_first_record(&result.output_path)?;
//! # Ok::<(), ce_ingest::IngestError>(())
//! ```
//!
//! Output matches the Python prototype byte for byte apart from `date_created` (UTC RFC 3339
//! with `Z` here, naive local time in Python) and path separators: same key order, `", "` /
//! `": "` separators, unescaped non-ASCII text, Python float formatting.
//!
//! Differences from the Python prototype:
//! - PDF text comes from `pdf-extract` instead of pdfplumber, so PDF output is not
//!   byte-comparable (the header / footer cleaning and line-break normalisation are exact
//!   ports). The `pdf` cargo feature (default on) enables it; without it `.pdf` files fail with
//!   [`IngestError::Unsupported`]`("pdf")`.
//! - LLM cleaning and vision OCR are not implemented: requesting either for a standard document
//!   fails with [`IngestError::Unsupported`] before anything is written. Transcripts ignore
//!   both flags, as in Python.
//! - A leading UTF-8 byte-order mark is ignored in every input (Python keeps it as text, which
//!   breaks JSON and CSV headers).
//! - JSON documents need string `doc_id` / `text` and an integer `total_pages`; JSONL records
//!   need a string `text`. Python passes other types through untyped.
//! - Extra top-level JSONL keys move into `metadata` in record order (Python: set order).

pub mod error;
pub mod ingest;
pub mod jsonl;
pub mod loader;
pub mod pdf;
mod py;
pub mod transcript;

pub use error::{IngestError, Result};
pub use ingest::{run_ingestion, IngestOptions, IngestionResult};
pub use jsonl::{read_first_record, to_python_json};
pub use loader::{DocumentLoader, LoadedDocument};
pub use pdf::{normalize_pdf_linebreaks, PdfCleaning};
pub use transcript::{flatten_utterances, parse_file, FlattenedTranscript, Utterance};
