//! `run_ingestion`: one source file to a one-line `DocumentPayload` JSONL file.

use std::path::{Path, PathBuf};

use ce_core::{DocId, DocumentPayload, Metadata};
use chrono::{SubsecRound, Utc};
use serde::Serialize;
use serde_json::Value;

use crate::error::{IngestError, Result};
use crate::jsonl::write_payload;
use crate::loader::{stem, suffix, DocumentLoader};
use crate::transcript;

/// Ingestion options (the keyword arguments of the Python `run_ingestion`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestOptions {
    /// Case identifier, stored in `metadata.case_id`.
    pub case_id: String,
    /// Database identifier, stored in `metadata.db_id`.
    pub db_id: String,
    /// Parse the file as a transcript (`.json`, `.csv`, `.txt`, `.text`).
    pub transcript_doc: bool,
    /// Document type; `None` or empty means `UNKNOWN` (`transcript` for transcripts).
    pub doc_type: Option<String>,
    /// LLM text cleaning of standard documents; not available yet
    /// ([`IngestError::Unsupported`]`("llm_cleaning")`).
    pub use_llm_cleaning: bool,
    /// Vision-model OCR of standard documents; not available
    /// ([`IngestError::Unsupported`]`("vision_ocr")`).
    pub use_vision_ocr: bool,
    /// With vision OCR, use it only for PDFs without a text layer. Kept for API parity; it has
    /// no effect while vision OCR is unavailable.
    pub auto_detect_scanned: bool,
}

impl IngestOptions {
    /// Options with the Python defaults: a standard document, no doc type, no LLM cleaning,
    /// no vision OCR, `auto_detect_scanned` on.
    pub fn new(case_id: impl Into<String>, db_id: impl Into<String>) -> Self {
        Self {
            case_id: case_id.into(),
            db_id: db_id.into(),
            transcript_doc: false,
            doc_type: None,
            use_llm_cleaning: false,
            use_vision_ocr: false,
            auto_detect_scanned: true,
        }
    }

    fn doc_type_or(&self, default: &str) -> String {
        self.doc_type
            .as_deref()
            .filter(|doc_type| !doc_type.is_empty())
            .unwrap_or(default)
            .to_owned()
    }
}

/// Result of [`run_ingestion`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IngestionResult {
    /// Document id written to the JSONL record.
    pub doc_id: DocId,
    /// The JSONL file written.
    pub output_path: PathBuf,
    /// The record's metadata.
    pub metadata: Metadata,
}

/// Ingests one document and writes it as a single-line JSONL [`DocumentPayload`] to
/// `output_path` (parent directories are created, an existing file is replaced).
///
/// Standard documents (`.pdf`, `.txt`, `.json`) go through [`DocumentLoader`] with the
/// default PDF cleaning; `doc_id` is the file stem (or the JSON `doc_id`), `title` the file
/// stem, and the metadata holds `case_id`, `db_id`, `transcript_doc`, `source_path`,
/// `used_llm_cleaning`, `used_vision_ocr`. Transcripts are parsed with [`crate::transcript`]
/// and flattened to `SPEAKER (role): text` lines; their metadata holds `case_id`, `db_id`,
/// `transcript_doc`, `source_path`, `utterance_count`, `unique_speakers`. `date_created` is the
/// current UTC time (microsecond precision). Paths are recorded as given.
///
/// # Errors
/// [`IngestError::EmptyPath`], [`IngestError::NotFound`], [`IngestError::Unsupported`] when LLM
/// cleaning or vision OCR is requested for a standard document,
/// [`IngestError::UnsupportedExtension`] / [`IngestError::UnsupportedTranscriptFormat`], and the
/// read, parse and write errors of the selected loader.
pub fn run_ingestion(
    file_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    options: &IngestOptions,
) -> Result<IngestionResult> {
    let file_path = file_path.as_ref();
    let output_path = output_path.as_ref();
    if file_path.as_os_str().is_empty() {
        return Err(IngestError::EmptyPath);
    }
    if !file_path.exists() {
        return Err(IngestError::NotFound(file_path.to_path_buf()));
    }
    if !options.transcript_doc {
        if options.use_llm_cleaning {
            return Err(IngestError::Unsupported("llm_cleaning"));
        }
        if options.use_vision_ocr {
            return Err(IngestError::Unsupported("vision_ocr"));
        }
    }
    if let Some(parent) = output_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .map_err(|source| IngestError::Io { path: parent.to_path_buf(), source })?;
    }

    let payload = if options.transcript_doc {
        ingest_transcript(file_path, options)?
    } else {
        ingest_standard_document(file_path, options)?
    };
    write_payload(&payload, output_path)?;

    Ok(IngestionResult {
        doc_id: payload.doc_id,
        output_path: output_path.to_path_buf(),
        metadata: payload.metadata,
    })
}

fn base_metadata(file_path: &Path, options: &IngestOptions, transcript_doc: bool) -> Metadata {
    let mut metadata = Metadata::new();
    metadata.insert("case_id".to_owned(), Value::from(options.case_id.as_str()));
    metadata.insert("db_id".to_owned(), Value::from(options.db_id.as_str()));
    metadata.insert("transcript_doc".to_owned(), Value::Bool(transcript_doc));
    metadata.insert("source_path".to_owned(), Value::from(path_string(file_path)));
    metadata
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn payload(
    doc_id: String,
    text: String,
    doc_type: String,
    file_path: &Path,
    total_pages: Option<u32>,
    metadata: Metadata,
) -> DocumentPayload {
    DocumentPayload {
        doc_id: DocId::from(doc_id),
        text,
        doc_type,
        title: stem(file_path),
        file_path: Some(path_string(file_path)),
        total_pages,
        date_created: Some(Utc::now().trunc_subsecs(6)),
        metadata,
        helix_id: None,
    }
}

fn ingest_standard_document(file_path: &Path, options: &IngestOptions) -> Result<DocumentPayload> {
    let loaded = DocumentLoader::default().load_document(file_path)?;
    let mut metadata = base_metadata(file_path, options, false);
    metadata.insert("used_llm_cleaning".to_owned(), Value::Bool(options.use_llm_cleaning));
    metadata.insert("used_vision_ocr".to_owned(), Value::Bool(options.use_vision_ocr));
    Ok(payload(
        loaded.doc_id,
        loaded.text,
        options.doc_type_or("UNKNOWN"),
        file_path,
        loaded.total_pages,
        metadata,
    ))
}

fn ingest_transcript(file_path: &Path, options: &IngestOptions) -> Result<DocumentPayload> {
    let ext = suffix(file_path);
    let utterances = match ext.to_lowercase().as_str() {
        ".json" => transcript::parse_json(file_path)?,
        ".csv" => transcript::parse_csv(file_path)?,
        ".txt" | ".text" => transcript::parse_plain_text(file_path)?,
        _ => return Err(IngestError::UnsupportedTranscriptFormat(ext)),
    };
    let flat = transcript::flatten_utterances(&utterances).map_err(|reason| {
        IngestError::InvalidTranscript { path: file_path.to_path_buf(), reason }
    })?;
    let mut metadata = base_metadata(file_path, options, true);
    metadata.insert("utterance_count".to_owned(), Value::from(utterances.len()));
    metadata.insert("unique_speakers".to_owned(), Value::Array(flat.unique_speakers));
    Ok(payload(
        stem(file_path),
        flat.text,
        options.doc_type_or("transcript"),
        file_path,
        None,
        metadata,
    ))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;

    fn write(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, content).expect("write");
        path
    }

    fn read_line(path: &Path) -> Value {
        let content = std::fs::read_to_string(path).expect("read output");
        assert_eq!(content.matches('\n').count(), 1, "exactly one line");
        serde_json::from_str(&content).expect("valid JSON")
    }

    #[test]
    fn ingests_txt_with_api_metadata() {
        let dir = tempfile::tempdir().expect("tempdir");
        let input = write(dir.path(), "complaint.txt", "Plaintiff alleges.\n");
        let output = dir.path().join("nested/out/complaint.jsonl");
        let result = run_ingestion(&input, &output, &IngestOptions::new("CASE-123", "DB-456"))
            .expect("ingest");
        assert_eq!(result.doc_id.as_str(), "complaint");
        assert_eq!(result.output_path, output);
        let record = read_line(&output);
        let keys: Vec<&str> =
            record.as_object().expect("object").keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "doc_id",
                "text",
                "doc_type",
                "title",
                "file_path",
                "total_pages",
                "date_created",
                "metadata",
                "helix_id"
            ]
        );
        let source = input.to_string_lossy();
        assert_eq!(
            record["metadata"],
            json!({
                "case_id": "CASE-123",
                "db_id": "DB-456",
                "transcript_doc": false,
                "source_path": source,
                "used_llm_cleaning": false,
                "used_vision_ocr": false,
            })
        );
        assert_eq!(Value::Object(result.metadata), record["metadata"]);
        assert_eq!(record["doc_type"], "UNKNOWN");
        assert_eq!(record["title"], "complaint");
        assert_eq!(record["text"], "Plaintiff alleges.\n");
        assert_eq!(record["total_pages"], Value::Null);
        assert_eq!(record["helix_id"], Value::Null);
        let created = record["date_created"].as_str().expect("date string");
        assert!(created.ends_with('Z'), "{created}");
        assert!(ce_core::rag::parse_iso_datetime(created).is_some());
    }

    #[test]
    fn ingests_transcript_with_transcript_metadata() {
        let dir = tempfile::tempdir().expect("tempdir");
        let input = write(
            dir.path(),
            "depo.json",
            r#"[{"speaker": "Jane Doe", "role": "witness", "text": "No."}, {"speaker": "MR. A", "text": "Why?"}]"#,
        );
        let output = dir.path().join("depo.jsonl");
        let options = IngestOptions {
            transcript_doc: true,
            use_llm_cleaning: true,
            ..IngestOptions::new("C", "D")
        };
        let result = run_ingestion(&input, &output, &options).expect("ingest");
        let record = read_line(&output);
        assert_eq!(record["text"], "Jane Doe (witness): No.\nMR. A: Why?");
        assert_eq!(record["doc_type"], "transcript");
        let metadata = record["metadata"].as_object().expect("metadata");
        let keys: Vec<&str> = metadata.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "case_id",
                "db_id",
                "transcript_doc",
                "source_path",
                "utterance_count",
                "unique_speakers"
            ]
        );
        assert_eq!(metadata["utterance_count"], 2);
        assert_eq!(metadata["unique_speakers"], json!(["Jane Doe", "MR. A"]));
        assert_eq!(result.doc_id.as_str(), "depo");
    }

    #[test]
    fn doc_type_override_and_empty_default() {
        let dir = tempfile::tempdir().expect("tempdir");
        let input = write(dir.path(), "a.txt", "x");
        let output = dir.path().join("a.jsonl");
        let mut options = IngestOptions::new("C", "D");
        options.doc_type = Some("complaint".to_owned());
        run_ingestion(&input, &output, &options).expect("ingest");
        assert_eq!(read_line(&output)["doc_type"], "complaint");
        options.doc_type = Some(String::new());
        run_ingestion(&input, &output, &options).expect("ingest");
        assert_eq!(read_line(&output)["doc_type"], "UNKNOWN");
    }

    #[test]
    fn rejects_bad_inputs() {
        let dir = tempfile::tempdir().expect("tempdir");
        let output = dir.path().join("o.jsonl");
        let options = IngestOptions::new("C", "D");
        assert!(matches!(run_ingestion("", &output, &options), Err(IngestError::EmptyPath)));
        assert!(matches!(
            run_ingestion(dir.path().join("missing.txt"), &output, &options),
            Err(IngestError::NotFound(_))
        ));
        let docx = write(dir.path(), "brief.docx", "x");
        assert!(matches!(
            run_ingestion(&docx, &output, &options),
            Err(IngestError::UnsupportedExtension(ext)) if ext == ".docx"
        ));
        let transcript = IngestOptions { transcript_doc: true, ..options.clone() };
        assert!(matches!(
            run_ingestion(&docx, &output, &transcript),
            Err(IngestError::UnsupportedTranscriptFormat(ext)) if ext == ".docx"
        ));
        assert!(!output.exists());
    }

    #[test]
    fn llm_cleaning_and_vision_ocr_are_unsupported() {
        let dir = tempfile::tempdir().expect("tempdir");
        let input = write(dir.path(), "a.txt", "x");
        let output = dir.path().join("sub/a.jsonl");
        let llm = IngestOptions { use_llm_cleaning: true, ..IngestOptions::new("C", "D") };
        assert!(matches!(
            run_ingestion(&input, &output, &llm),
            Err(IngestError::Unsupported("llm_cleaning"))
        ));
        let ocr = IngestOptions {
            use_vision_ocr: true,
            auto_detect_scanned: false,
            ..IngestOptions::new("C", "D")
        };
        assert!(matches!(
            run_ingestion(&input, &output, &ocr),
            Err(IngestError::Unsupported("vision_ocr"))
        ));
        assert!(!output.parent().expect("parent").exists());
    }

    #[test]
    fn output_round_trips_through_read_first_record() {
        let dir = tempfile::tempdir().expect("tempdir");
        let input = write(
            dir.path(),
            "memo.json",
            r#"{"doc_id": "m-1", "text": "Body", "total_pages": 2}"#,
        );
        let output = dir.path().join("memo.jsonl");
        run_ingestion(&input, &output, &IngestOptions::new("C", "D")).expect("ingest");
        let doc = crate::read_first_record(&output).expect("read back");
        assert_eq!(doc.doc_id.as_str(), "m-1");
        assert_eq!(doc.title, "memo");
        assert_eq!(doc.total_pages, Some(2));
        assert_eq!(doc.metadata["case_id"], "C");
        assert!(doc.date_created.is_some());
    }
}
