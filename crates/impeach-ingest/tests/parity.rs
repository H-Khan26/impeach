//! Parity of `run_ingestion` with the Python prototype: the JSONL files under
//! `fixtures/reference/ingest/` were written by the Python `run_ingestion` (case `CASE-PARITY`,
//! db `DB-PARITY`) from the inputs under `fixtures/input/`. Apart from `date_created` and the
//! recorded paths (compared by file name), the Rust line must equal the Python line byte for
//! byte.

use std::path::{Path, PathBuf};

use impeach_ingest::{run_ingestion, IngestOptions};
use pretty_assertions::assert_eq;
use serde_json::Value;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// File name of a recorded path, whatever its separators.
fn file_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

fn first_line(path: &Path) -> String {
    let content = std::fs::read_to_string(path).expect("read JSONL");
    content.lines().next().expect("one line").trim_end_matches('\r').to_owned()
}

fn json_string(text: &str) -> String {
    serde_json::to_string(text).expect("serialize string")
}

fn check(input: &str, reference: &str, transcript_doc: bool) {
    let root = workspace_root();
    let input = root.join(input);
    let reference_path = root.join("fixtures/reference/ingest").join(reference);
    let out_dir = tempfile::tempdir().expect("tempdir");
    let output = out_dir.path().join(reference);
    let options =
        IngestOptions { transcript_doc, ..IngestOptions::new("CASE-PARITY", "DB-PARITY") };
    run_ingestion(&input, &output, &options).expect("run_ingestion");

    let rust_line = first_line(&output);
    let python_line = first_line(&reference_path);
    let rust: Value = serde_json::from_str(&rust_line).expect("Rust JSON");
    let python: Value = serde_json::from_str(&python_line).expect("Python JSON");

    let rust_path = rust["file_path"].as_str().expect("Rust file_path");
    let python_path = python["file_path"].as_str().expect("Python file_path");
    assert_eq!(file_name(rust_path), file_name(python_path));
    assert_eq!(rust["metadata"]["source_path"].as_str(), Some(rust_path));
    assert_eq!(python["metadata"]["source_path"].as_str(), Some(python_path));
    let rust_date = rust["date_created"].as_str().expect("Rust date_created");
    assert!(impeach_core::rag::parse_iso_datetime(rust_date).is_some(), "{rust_date}");
    let python_date = python["date_created"].as_str().expect("Python date_created");

    let aligned = rust_line
        .replace(&json_string(rust_date), &json_string(python_date))
        .replace(&json_string(rust_path), &json_string(python_path));
    assert_eq!(aligned, python_line, "{reference}");
}

#[test]
fn txt_document_matches_python() {
    check("fixtures/input/ingest/sample_note.txt", "sample_note.txt.jsonl", false);
}

#[test]
fn json_document_matches_python() {
    check("fixtures/input/ingest/sample_memo.json", "sample_memo.json.jsonl", false);
}

#[test]
fn json_transcript_matches_python() {
    check(
        "fixtures/input/transcripts/example_deposition.json",
        "example_deposition.json.jsonl",
        true,
    );
}

#[test]
fn plain_text_transcript_matches_python() {
    check("fixtures/input/ingest/sample_transcript.txt", "sample_transcript.txt.jsonl", true);
}

#[test]
fn csv_transcript_matches_python() {
    check("fixtures/input/ingest/sample_transcript.csv", "sample_transcript.csv.jsonl", true);
}
