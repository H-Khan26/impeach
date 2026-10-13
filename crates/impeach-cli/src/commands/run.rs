//! `run`: ingest a document to a temporary JSONL file, then process it (Python `__main__`).

use anyhow::Context as _;
use impeach_ingest::{run_ingestion, IngestOptions};

use crate::cli::RunArgs;
use crate::commands::process;
use crate::output::println;

/// Ingests `args.file` into a temporary directory and processes the JSONL record.
///
/// # Errors
/// Ingestion, processing and summary write errors.
pub fn execute(args: &RunArgs) -> anyhow::Result<()> {
    let run = {
        let tmp_dir = tempfile::tempdir().context("failed to create a temporary directory")?;
        let jsonl_path = tmp_dir.path().join("ingested.jsonl");

        println(&format!("Ingesting: {}", args.file.display()));
        let options = IngestOptions::new(&args.ids.case_id, &args.ids.db_id);
        let ingested = run_ingestion(&args.file, &jsonl_path, &options)?;
        println(&format!("Ingested document: {}", ingested.doc_id));

        process::process_jsonl(&jsonl_path, &args.processing)?
    };
    process::report(&args.file, &args.processing, &run)
}
