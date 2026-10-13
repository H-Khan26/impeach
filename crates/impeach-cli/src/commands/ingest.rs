//! `ingest`: raw document -> one-line JSONL.

use impeach_ingest::{run_ingestion, IngestOptions};

use crate::cli::IngestArgs;
use crate::output::println;

/// Ingests `args.file` into `args.output`.
///
/// # Errors
/// The ingestion error.
pub fn execute(args: &IngestArgs) -> anyhow::Result<()> {
    let mut options = IngestOptions::new(&args.ids.case_id, &args.ids.db_id);
    options.transcript_doc = args.transcript;
    println(&format!("Ingesting: {}", args.file.display()));
    let result = run_ingestion(&args.file, &args.output, &options)?;
    println(&format!("Ingested document: {}", result.doc_id));
    println(&format!("JSONL written to: {}", result.output_path.display()));
    Ok(())
}
