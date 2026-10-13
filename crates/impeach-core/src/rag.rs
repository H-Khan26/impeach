//! Document, chunk and indexing records of the RAG substrate (Python `schema/rag.py`).
//!
//! The chunking / embedding configuration dataclasses of that module are configuration and live
//! in `impeach-config`.

use chrono::{DateTime, NaiveDate, NaiveTime, SecondsFormat, Utc};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::entities::Span;
use crate::ids::{ChunkId, DocId, NodeId, SpanId};
use crate::Metadata;

/// Parses an ISO-8601 timestamp the way the Python loader does (`datetime.fromisoformat`, with a
/// trailing `Z` accepted). Accepts extended (`2025-01-15`) and basic (`20250115`) dates, any
/// single-character separator, times `HH`, `HH:MM`, `HH:MM:SS[.f]`, `HHMM`, `HHMMSS[.f]`, and offsets
/// `Z`, `+HH`, `+HHMM`, `+HH:MM`, `+HH:MM:SS`. Timestamps without an offset are taken as UTC;
/// date-only values map to midnight. Returns `None` for unparseable input.
#[must_use]
pub fn parse_iso_datetime(value: &str) -> Option<DateTime<Utc>> {
    let text = value.trim();
    let (date_part, time_part) = split_date_time(text)?;
    let date = NaiveDate::parse_from_str(date_part, "%Y-%m-%d")
        .or_else(|_| NaiveDate::parse_from_str(date_part, "%Y%m%d"))
        .ok()?;
    let Some(time_part) = time_part else {
        return date.and_hms_opt(0, 0, 0).map(|naive| naive.and_utc());
    };
    let (clock, offset_seconds) = split_offset(time_part)?;
    let time = parse_clock(clock)?;
    let naive = date.and_time(time);
    let utc = naive.checked_sub_signed(chrono::TimeDelta::seconds(offset_seconds))?;
    Some(utc.and_utc())
}

/// Splits `text` into the date and the optional time (with offset) at the separator, which is
/// any single character as in Python 3.11+; the date is 10 (`YYYY-MM-DD`) or 8 (`YYYYMMDD`)
/// characters.
fn split_date_time(text: &str) -> Option<(&str, Option<&str>)> {
    let date_len = if text.as_bytes().get(4) == Some(&b'-') { 10 } else { 8 };
    if text.len() == date_len {
        return Some((text, None));
    }
    let date = text.get(..date_len)?;
    let rest = text.get(date_len..)?;
    let mut chars = rest.chars();
    chars.next()?;
    Some((date, Some(chars.as_str())))
}

/// Splits a time into the clock part and its UTC offset in seconds (0 when absent).
fn split_offset(time: &str) -> Option<(&str, i64)> {
    if let Some(clock) = time.strip_suffix(['Z', 'z']) {
        return Some((clock, 0));
    }
    let Some(sign_at) = time.find(['+', '-']) else {
        return Some((time, 0));
    };
    let sign = if time.as_bytes()[sign_at] == b'-' { -1 } else { 1 };
    let offset = &time[sign_at + 1..];
    let offset = offset.split_once('.').map_or(offset, |(whole, _)| whole);
    let digits: String = offset.chars().filter(|c| *c != ':').collect();
    if !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let field = |i: usize| digits.get(i..i + 2).and_then(|d| d.parse::<i64>().ok());
    let (hours, minutes, seconds) = match digits.len() {
        2 => (field(0)?, 0, 0),
        4 => (field(0)?, field(2)?, 0),
        6 => (field(0)?, field(2)?, field(4)?),
        _ => return None,
    };
    if hours >= 24 || minutes >= 60 || seconds >= 60 {
        return None;
    }
    Some((&time[..sign_at], sign * (hours * 3600 + minutes * 60 + seconds)))
}

/// Parses `HH`, `HH:MM`, `HH:MM:SS[.f]`, `HHMM` or `HHMMSS[.f]`.
fn parse_clock(clock: &str) -> Option<NaiveTime> {
    ["%H:%M:%S%.f", "%H:%M", "%H%M%S%.f", "%H%M"]
        .iter()
        .find_map(|fmt| NaiveTime::parse_from_str(clock, fmt).ok())
        .or_else(|| {
            let hour = (clock.len() == 2).then(|| clock.parse::<u32>().ok()).flatten()?;
            NaiveTime::from_hms_opt(hour, 0, 0)
        })
}

/// Formats a timestamp as RFC 3339 in UTC with a `Z` suffix and only as many fractional digits
/// as needed.
#[must_use]
pub fn format_datetime(value: &DateTime<Utc>) -> String {
    value.to_rfc3339_opts(SecondsFormat::AutoSi, true)
}

/// Serde adapter for `Option<DateTime<Utc>>`: writes RFC 3339 (`Z`) or `null`; reads anything
/// [`parse_iso_datetime`] accepts, and maps unparseable or non-string values to `None` as the
/// Python loader does.
mod optional_datetime {
    use chrono::{DateTime, Utc};
    use serde::{Deserialize, Deserializer, Serializer};

    pub(super) fn serialize<S: Serializer>(
        value: &Option<DateTime<Utc>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            Some(dt) => serializer.serialize_str(&super::format_datetime(dt)),
            None => serializer.serialize_none(),
        }
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<DateTime<Utc>>, D::Error> {
        let raw = serde_json::Value::deserialize(deserializer)?;
        Ok(raw.as_str().and_then(super::parse_iso_datetime))
    }
}

fn default_doc_type() -> String {
    "UNKNOWN".to_owned()
}

/// Normalised document input (one JSONL line of ingestion output).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentPayload {
    /// Document id.
    pub doc_id: DocId,
    /// Full document text.
    pub text: String,
    /// Document type, e.g. `complaint`; `UNKNOWN` by default.
    #[serde(default = "default_doc_type")]
    pub doc_type: String,
    /// Title.
    #[serde(default)]
    pub title: String,
    /// Original file path.
    #[serde(default)]
    pub file_path: Option<String>,
    /// Page count.
    #[serde(default)]
    pub total_pages: Option<u32>,
    /// Creation timestamp.
    #[serde(default, with = "optional_datetime")]
    pub date_created: Option<DateTime<Utc>>,
    /// Free-form metadata (`case_id`, `db_id`, ...).
    #[serde(default)]
    pub metadata: Metadata,
    /// Graph node id once written.
    #[serde(default)]
    pub helix_id: Option<NodeId>,
}

impl DocumentPayload {
    /// Creates a payload with the Python defaults for every optional field.
    pub fn new(doc_id: impl Into<DocId>, text: impl Into<String>) -> Self {
        Self {
            doc_id: doc_id.into(),
            text: text.into(),
            doc_type: default_doc_type(),
            title: String::new(),
            file_path: None,
            total_pages: None,
            date_created: None,
            metadata: Metadata::new(),
            helix_id: None,
        }
    }
}

/// A chunk of document text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    /// Document id.
    pub doc_id: DocId,
    /// Chunk text.
    pub text: String,
    /// Position of the chunk in the document.
    pub sequence_number: usize,
    /// Start offset in the document.
    pub start_char: usize,
    /// End offset in the document.
    pub end_char: usize,
    /// Page number.
    #[serde(default)]
    pub page_number: Option<u32>,
    /// Section name.
    #[serde(default)]
    pub section: Option<String>,
    /// Free-form metadata.
    #[serde(default)]
    pub metadata: Metadata,
    /// Graph node id once written.
    #[serde(default)]
    pub helix_id: Option<NodeId>,
    /// Sentence spans (`{"start", "end"}` document offsets).
    #[serde(default)]
    pub sentence_spans: Vec<Span>,
    /// Neighbouring chunk sequence numbers (`prev_chunk_seq`, `next_chunk_seq`).
    #[serde(default)]
    pub neighbors: IndexMap<String, i64>,
    /// Section hierarchy.
    #[serde(default)]
    pub section_path: Vec<String>,
}

impl Chunk {
    /// Creates a chunk with the Python defaults for every optional field.
    pub fn new(
        doc_id: impl Into<DocId>,
        text: impl Into<String>,
        sequence_number: usize,
        start_char: usize,
        end_char: usize,
    ) -> Self {
        Self {
            doc_id: doc_id.into(),
            text: text.into(),
            sequence_number,
            start_char,
            end_char,
            page_number: None,
            section: None,
            metadata: Metadata::new(),
            helix_id: None,
            sentence_spans: Vec::new(),
            neighbors: IndexMap::new(),
            section_path: Vec::new(),
        }
    }
}

/// Sentence span derived from a chunk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SentenceSpan {
    /// Sequence number of the owning chunk.
    pub chunk_sequence: usize,
    /// Start offset in the document.
    pub start_char: usize,
    /// End offset in the document.
    pub end_char: usize,
    /// Sentence text.
    pub text: String,
    /// Free-form metadata.
    #[serde(default)]
    pub metadata: Metadata,
}

/// Output of a chunker.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChunkingResult {
    /// Chunks in document order.
    pub chunks: Vec<Chunk>,
    /// Sentence spans across all chunks.
    #[serde(default)]
    pub sentence_spans: Vec<SentenceSpan>,
}

/// A chunk embedding ready to be written.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkEmbeddingInput {
    /// Chunk id.
    pub chunk_id: ChunkId,
    /// Embedding vector.
    pub embedding: Vec<f32>,
    /// Document id.
    #[serde(default)]
    pub doc_id: Option<DocId>,
    /// Free-form metadata.
    #[serde(default)]
    pub metadata: Metadata,
}

/// Result of a document upsert.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentWriteResult {
    /// Document id.
    pub doc_id: DocId,
    /// Graph node id.
    pub helix_id: NodeId,
    /// Whether the node was newly created.
    pub created: bool,
}

/// Result of writing one chunk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkWriteResult {
    /// The chunk written.
    pub chunk: Chunk,
    /// Graph node id.
    pub helix_id: NodeId,
    /// Whether the node was newly created.
    pub created: bool,
}

/// Result of writing one sentence span.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpanWriteResult {
    /// Span id (e.g. `doc-1:sent-00000`).
    pub span_id: SpanId,
    /// Sequence number of the owning chunk.
    pub chunk_sequence: usize,
    /// Whether the node was newly created.
    pub created: bool,
}

/// Summary of embedding upserts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingWriteResult {
    /// Number of embeddings written.
    pub count: usize,
    /// Error messages.
    #[serde(default)]
    pub errors: Vec<String>,
}

/// Summary of indexing one document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexingResult {
    /// Document write.
    pub document: DocumentWriteResult,
    /// Chunk writes.
    pub chunks: Vec<ChunkWriteResult>,
    /// Span writes.
    pub spans: Vec<SpanWriteResult>,
    /// Embedding writes.
    pub embeddings: EmbeddingWriteResult,
    /// Document text.
    #[serde(default)]
    pub doc_text: Option<String>,
    /// Step timings in seconds (`document`, `chunking`, `embeddings`).
    #[serde(default)]
    pub timings: IndexMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    fn utc(y: i32, mo: u32, d: u32, h: u32, mi: u32, s: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, mo, d, h, mi, s).single().expect("valid date")
    }

    #[test]
    fn parses_python_isoformat_variants() {
        let base = utc(2025, 1, 15, 10, 30, 0);
        assert_eq!(parse_iso_datetime("2025-01-15T10:30:00"), Some(base));
        assert_eq!(parse_iso_datetime("2025-01-15T10:30:00Z"), Some(base));
        assert_eq!(parse_iso_datetime("2025-01-15T10:30:00z"), Some(base));
        assert_eq!(parse_iso_datetime("2025-01-15T10:30:00+00:00"), Some(base));
        assert_eq!(parse_iso_datetime("2025-01-15T12:30:00+02:00"), Some(base));
        assert_eq!(parse_iso_datetime("2025-01-15 10:30:00"), Some(base));
        assert_eq!(parse_iso_datetime("2025-01-15T10:30"), Some(base));
        assert_eq!(parse_iso_datetime(" 2025-01-15T10:30:00 "), Some(base));
        assert_eq!(parse_iso_datetime("2025-01-15"), Some(utc(2025, 1, 15, 0, 0, 0)));
        let micro = parse_iso_datetime("2025-01-15T10:30:00.123456").expect("parses");
        assert_eq!(format_datetime(&micro), "2025-01-15T10:30:00.123456Z");
        assert_eq!(parse_iso_datetime("yesterday"), None);
        assert_eq!(parse_iso_datetime("2025-01-15T10:30:00+0000"), Some(base));
        assert_eq!(parse_iso_datetime("2025-01-15T12:30:00+0200"), Some(base));
        assert_eq!(parse_iso_datetime("2025-01-15T12:30:00+02"), Some(base));
        assert_eq!(parse_iso_datetime("2025-01-15T08:30:00-02"), Some(base));
        assert_eq!(parse_iso_datetime("2025-01-15T05:00:00-05:30"), Some(base));
        assert_eq!(parse_iso_datetime("2025-01-15T12:30:00+02:00:00"), Some(base));
        assert_eq!(parse_iso_datetime("2025-01-15T10:30:00.000+00:00"), Some(base));
        assert_eq!(parse_iso_datetime("2025-01-15t10:30:00"), Some(base));
        assert_eq!(parse_iso_datetime("2025-01-15T1030"), Some(base));
        assert_eq!(parse_iso_datetime("2025-01-15T103000"), Some(base));
        assert_eq!(parse_iso_datetime("20250115T103000"), Some(base));
        assert_eq!(parse_iso_datetime("20250115"), Some(utc(2025, 1, 15, 0, 0, 0)));
        assert_eq!(parse_iso_datetime("2025-01-15T10"), Some(utc(2025, 1, 15, 10, 0, 0)));
        assert_eq!(parse_iso_datetime("2025-01-15T10:30:00+2"), None);
        assert_eq!(parse_iso_datetime("2025-01-15T10:30:00+24:00"), None);
        assert_eq!(parse_iso_datetime("2025-01-15T10:30:00+ab"), None);
        assert_eq!(parse_iso_datetime("2025-01-15X10:30"), Some(base));
        assert_eq!(parse_iso_datetime("2025-01-15TT10:30"), None);
        assert_eq!(parse_iso_datetime("2025-13-15"), None);
        assert_eq!(parse_iso_datetime(""), None);
    }

    #[test]
    fn formats_rfc3339_utc() {
        assert_eq!(format_datetime(&utc(2025, 1, 15, 10, 30, 0)), "2025-01-15T10:30:00Z");
    }

    #[test]
    fn payload_defaults_match_python() {
        let payload: DocumentPayload =
            serde_json::from_value(json!({"doc_id": "d1", "text": "Body"})).expect("payload");
        assert_eq!(payload, DocumentPayload::new("d1", "Body"));
        assert_eq!(payload.doc_type, "UNKNOWN");
        assert_eq!(
            serde_json::to_value(&payload).expect("serialises"),
            json!({
                "doc_id": "d1",
                "text": "Body",
                "doc_type": "UNKNOWN",
                "title": "",
                "file_path": null,
                "total_pages": null,
                "date_created": null,
                "metadata": {},
                "helix_id": null
            })
        );
    }

    #[test]
    fn payload_reads_python_jsonl_line() {
        let line = r#"{"doc_id": "test-doc-1", "text": "This is test document text for processing.", "doc_type": "complaint", "title": "Test Document", "file_path": "original/path.pdf", "total_pages": 1, "date_created": "2025-11-20T14:32:01.123456", "metadata": {"case_id": "CASE-123", "db_id": "DB-456"}, "helix_id": null}"#;
        let payload: DocumentPayload = serde_json::from_str(line).expect("python line parses");
        assert_eq!(payload.doc_type, "complaint");
        assert_eq!(payload.total_pages, Some(1));
        assert_eq!(
            payload.date_created.as_ref().map(format_datetime).as_deref(),
            Some("2025-11-20T14:32:01.123456Z")
        );
        assert_eq!(payload.metadata.get("case_id"), Some(&json!("CASE-123")));
    }

    #[test]
    fn payload_unparseable_date_becomes_none() {
        let payload: DocumentPayload = serde_json::from_value(json!({
            "doc_id": "d1", "text": "", "date_created": "not a date"
        }))
        .expect("payload");
        assert!(payload.date_created.is_none());
        let payload: DocumentPayload =
            serde_json::from_value(json!({"doc_id": "d1", "text": "", "date_created": 5}))
                .expect("payload");
        assert!(payload.date_created.is_none());
    }

    #[test]
    fn chunk_defaults_and_python_shapes() {
        let chunk: Chunk = serde_json::from_value(json!({
            "doc_id": "d1",
            "text": "Hello.",
            "sequence_number": 0,
            "start_char": 0,
            "end_char": 6,
            "sentence_spans": [{"start": 0, "end": 6}],
            "neighbors": {"next_chunk_seq": 1}
        }))
        .expect("chunk");
        let mut expected = Chunk::new("d1", "Hello.", 0, 0, 6);
        expected.sentence_spans = vec![Span::new(0, 6)];
        expected.neighbors.insert("next_chunk_seq".into(), 1);
        assert_eq!(chunk, expected);
    }

    #[test]
    fn write_results_round_trip() {
        let result = IndexingResult {
            document: DocumentWriteResult {
                doc_id: "d1".into(),
                helix_id: "h1".into(),
                created: true,
            },
            chunks: vec![ChunkWriteResult {
                chunk: Chunk::new("d1", "Hello.", 0, 0, 6),
                helix_id: "h2".into(),
                created: true,
            }],
            spans: vec![SpanWriteResult {
                span_id: "d1:sent-00000".into(),
                chunk_sequence: 0,
                created: true,
            }],
            embeddings: EmbeddingWriteResult { count: 1, errors: vec![] },
            doc_text: None,
            timings: IndexMap::from([("document".to_owned(), 0.5)]),
        };
        let json = serde_json::to_string(&result).expect("serialises");
        let back: IndexingResult = serde_json::from_str(&json).expect("deserialises");
        assert_eq!(back, result);
    }
}
