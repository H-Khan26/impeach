//! JSONL output in the byte format of Python `json.dumps(obj, ensure_ascii=False)` and reading
//! of the first JSONL record.

use std::io::{self, BufRead, Write};
use std::path::Path;

use ce_core::{DocId, DocumentPayload, Metadata};
use serde::Serialize;
use serde_json::ser::Formatter;
use serde_json::Value;

use crate::error::{IngestError, Result};
use crate::loader::stem;
use crate::py;

/// `serde_json` formatter that writes `", "` and `": "` separators and Python float `repr`.
#[derive(Debug, Clone, Copy, Default)]
struct PythonFormatter;

impl Formatter for PythonFormatter {
    fn begin_array_value<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }

    fn write_f64<W: ?Sized + Write>(&mut self, writer: &mut W, value: f64) -> io::Result<()> {
        writer.write_all(py::float_repr(value).as_bytes())
    }

    fn write_f32<W: ?Sized + Write>(&mut self, writer: &mut W, value: f32) -> io::Result<()> {
        writer.write_all(py::float_repr(f64::from(value)).as_bytes())
    }
}

/// Serialises `value` as Python `json.dumps(value, ensure_ascii=False)` does: `", "` / `": "`
/// separators, non-ASCII text unescaped, floats as Python `repr`. Field order is the serde
/// order.
///
/// # Errors
/// The serialisation error of `value`.
pub fn to_python_json<T: Serialize + ?Sized>(value: &T) -> serde_json::Result<String> {
    let mut out = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut out, PythonFormatter);
    value.serialize(&mut serializer)?;
    // serde_json only writes valid UTF-8.
    Ok(String::from_utf8_lossy(&out).into_owned())
}

/// Writes `payload` as the only line of `path` (truncating it), newline-terminated.
pub(crate) fn write_payload(payload: &DocumentPayload, path: &Path) -> Result<()> {
    let mut line = to_python_json(payload)
        .map_err(|source| IngestError::Json { path: path.to_path_buf(), source })?;
    line.push('\n');
    std::fs::write(path, line)
        .map_err(|source| IngestError::Io { path: path.to_path_buf(), source })
}

/// Top-level keys of a `DocumentPayload` record; other keys are folded into `metadata`.
const PAYLOAD_FIELDS: [&str; 9] = [
    "doc_id",
    "text",
    "doc_type",
    "title",
    "file_path",
    "total_pages",
    "date_created",
    "metadata",
    "helix_id",
];

/// Reads the first record of a JSONL file as a [`DocumentPayload`], with the rules of the
/// Python `run_processing` loader and the document coercion of its RAG index stage:
/// - `text` must be a non-empty string;
/// - a missing or empty `doc_id` defaults to the file stem; non-string ids are formatted with
///   Python `str()`;
/// - top-level keys that are not payload fields (e.g. `case_id`, `db_id`) move into
///   `metadata`, in record order, without overwriting keys already there;
/// - a missing `file_path` becomes the JSONL path; an empty `doc_type` becomes `UNKNOWN` and an
///   empty `title` the `doc_id`; `date_created` accepts every ISO-8601 form (unparseable ->
///   `None`).
///
/// A leading byte-order mark is ignored.
///
/// # Errors
/// [`IngestError::EmptyPath`], [`IngestError::NotFound`], [`IngestError::EmptyJsonl`] for an
/// empty first line, [`IngestError::Json`] for malformed JSON, [`IngestError::MissingText`]
/// and [`IngestError::InvalidDocument`] for records of the wrong shape.
pub fn read_first_record(jsonl_path: impl AsRef<Path>) -> Result<DocumentPayload> {
    let path = jsonl_path.as_ref();
    if path.as_os_str().is_empty() {
        return Err(IngestError::EmptyPath);
    }
    if !path.exists() {
        return Err(IngestError::NotFound(path.to_path_buf()));
    }
    let io_error = |source| IngestError::Io { path: path.to_path_buf(), source };
    let file = std::fs::File::open(path).map_err(io_error)?;
    let mut raw = Vec::new();
    io::BufReader::new(file).read_until(b'\n', &mut raw).map_err(io_error)?;
    let raw = raw.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&raw);
    // Python's text mode ends a line at `\n`, `\r\n` or a lone `\r`.
    let raw = raw.iter().position(|b| *b == b'\r').map_or(raw, |end| &raw[..end]);
    let line = std::str::from_utf8(raw)
        .map_err(|source| IngestError::Encoding { path: path.to_path_buf(), source })?;
    let line = ce_text::normalize::py_strip(line);
    if line.is_empty() {
        return Err(IngestError::EmptyJsonl(path.to_path_buf()));
    }
    let record: Value = serde_json::from_str(line)
        .map_err(|source| IngestError::Json { path: path.to_path_buf(), source })?;
    match record {
        Value::Object(map) if map.get("text").is_some_and(py::truthy) => {
            record_to_payload(map, path)
        }
        _ => Err(IngestError::MissingText(path.to_path_buf())),
    }
}

fn record_to_payload(mut record: Metadata, path: &Path) -> Result<DocumentPayload> {
    let invalid = |reason: &str| IngestError::InvalidDocument {
        path: path.to_path_buf(),
        reason: reason.to_owned(),
    };
    let Some(Value::String(text)) = record.remove("text") else {
        return Err(invalid("'text' must be a string"));
    };
    let doc_id = match record.get("doc_id") {
        Some(value) if py::truthy(value) => py::py_str(value),
        _ => stem(path),
    };

    let mut metadata = match record.remove("metadata") {
        Some(Value::Object(map)) => map,
        Some(value) if py::truthy(&value) => return Err(invalid("'metadata' must be an object")),
        _ => Metadata::new(),
    };
    for (key, value) in &record {
        if !PAYLOAD_FIELDS.contains(&key.as_str()) && !metadata.contains_key(key) {
            metadata.insert(key.clone(), value.clone());
        }
    }

    let optional_string = |key: &str| match record.get(key) {
        None | Some(Value::Null) => None,
        Some(value) => Some(py::py_str(value)),
    };
    let truthy_string =
        |key: &str| record.get(key).filter(|value| py::truthy(value)).map(py::py_str);
    let file_path = if record.contains_key("file_path") {
        optional_string("file_path")
    } else {
        Some(path.to_string_lossy().into_owned())
    };
    let total_pages = match record.get("total_pages") {
        None | Some(Value::Null) => None,
        Some(value) => Some(
            value
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| invalid("'total_pages' must be a non-negative integer"))?,
        ),
    };
    let date_created = record
        .get("date_created")
        .and_then(Value::as_str)
        .and_then(ce_core::rag::parse_iso_datetime);

    Ok(DocumentPayload {
        doc_type: truthy_string("doc_type").unwrap_or_else(|| "UNKNOWN".to_owned()),
        title: truthy_string("title").unwrap_or_else(|| doc_id.clone()),
        doc_id: DocId::from(doc_id),
        text,
        file_path,
        total_pages,
        date_created,
        metadata,
        helix_id: optional_string("helix_id").map(Into::into),
    })
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;

    #[test]
    fn python_json_format() {
        let value =
            json!({"a": [1, 2.5, 1e20, "é\n\u{1}"], "b": {}, "c": [], "d": null, "e": true});
        assert_eq!(
            to_python_json(&value).expect("serialize"),
            "{\"a\": [1, 2.5, 1e+20, \"é\\n\\u0001\"], \"b\": {}, \"c\": [], \"d\": null, \"e\": true}"
        );
    }

    fn write(dir: &Path, name: &str, content: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, content).expect("write");
        path
    }

    #[test]
    fn reads_full_record() {
        let dir = tempfile::tempdir().expect("tempdir");
        let line = json!({
            "doc_id": "test-doc-1",
            "text": "This is test document text for processing.",
            "doc_type": "complaint",
            "title": "Test Document",
            "file_path": "original/path.pdf",
            "total_pages": 1,
            "date_created": "2025-01-15T10:30:00",
            "metadata": {"case_id": "CASE-123", "db_id": "DB-456"},
            "helix_id": null,
        });
        let path = write(dir.path(), "test_doc.jsonl", &format!("{line}\n{{\"ignored\": 1}}\n"));
        let doc = read_first_record(&path).expect("read");
        assert_eq!(doc.doc_id.as_str(), "test-doc-1");
        assert_eq!(doc.text, "This is test document text for processing.");
        assert_eq!(doc.doc_type, "complaint");
        assert_eq!(doc.title, "Test Document");
        assert_eq!(doc.file_path.as_deref(), Some("original/path.pdf"));
        assert_eq!(doc.total_pages, Some(1));
        assert_eq!(
            doc.date_created.map(|d| ce_core::rag::format_datetime(&d)),
            Some("2025-01-15T10:30:00Z".to_owned())
        );
        assert_eq!(Value::Object(doc.metadata), json!({"case_id": "CASE-123", "db_id": "DB-456"}));
        assert_eq!(doc.helix_id, None);
    }

    #[test]
    fn accepts_top_level_identifiers() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(
            dir.path(),
            "minimal.jsonl",
            "{\"case_id\": \"mason_v_billy\", \"db_id\": \"depo_1\", \"doc_id\": \"002\", \"text\": \"Mason was here.\"}\r\n",
        );
        let doc = read_first_record(&path).expect("read");
        assert_eq!(doc.doc_id.as_str(), "002");
        assert_eq!(doc.title, "002");
        assert_eq!(doc.doc_type, "UNKNOWN");
        assert_eq!(doc.file_path, Some(path.to_string_lossy().into_owned()));
        assert_eq!(
            Value::Object(doc.metadata),
            json!({"case_id": "mason_v_billy", "db_id": "depo_1"})
        );
    }

    #[test]
    fn folds_extra_keys_without_overwriting_and_defaults_doc_id() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(
            dir.path(),
            "deposition_7.jsonl",
            "\u{feff}{\"text\": \"T\", \"doc_id\": \"\", \"case_id\": \"top\", \"metadata\": {\"case_id\": \"nested\"}, \"extra\": [1], \"file_path\": null}\n",
        );
        let doc = read_first_record(&path).expect("read");
        assert_eq!(doc.doc_id.as_str(), "deposition_7");
        assert_eq!(doc.file_path, None);
        assert_eq!(Value::Object(doc.metadata), json!({"case_id": "nested", "extra": [1]}));

        let numeric = write(dir.path(), "n.jsonl", "{\"doc_id\": 2, \"text\": \"T\"}");
        assert_eq!(read_first_record(&numeric).expect("read").doc_id.as_str(), "2");
    }

    #[test]
    fn errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(matches!(read_first_record(""), Err(IngestError::EmptyPath)));
        assert!(matches!(
            read_first_record(dir.path().join("does_not_exist.jsonl")),
            Err(IngestError::NotFound(_))
        ));
        let empty = write(dir.path(), "empty.jsonl", "");
        assert!(matches!(read_first_record(&empty), Err(IngestError::EmptyJsonl(_))));
        let blank_first = write(dir.path(), "blank.jsonl", "\n{\"text\": \"T\"}\n");
        assert!(matches!(read_first_record(&blank_first), Err(IngestError::EmptyJsonl(_))));
        let invalid = write(dir.path(), "invalid.jsonl", "not valid json {");
        assert!(matches!(read_first_record(&invalid), Err(IngestError::Json { .. })));
        for (name, line) in [
            ("no_text.jsonl", "{\"doc_id\": \"a\"}"),
            ("empty_text.jsonl", "{\"text\": \"\"}"),
            ("array.jsonl", "[1]"),
        ] {
            let path = write(dir.path(), name, line);
            assert!(matches!(read_first_record(&path), Err(IngestError::MissingText(_))), "{name}");
        }
        let numeric_text = write(dir.path(), "num.jsonl", "{\"text\": 5}");
        assert!(matches!(
            read_first_record(&numeric_text),
            Err(IngestError::InvalidDocument { .. })
        ));
    }
}
