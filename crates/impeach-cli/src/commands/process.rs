//! `process`: pre-ingested JSONL -> processing pipeline, console report and JSON summary.

use std::fmt::Write as _;
use std::path::Path;

use anyhow::Context as _;
use impeach_pipeline::{run_processing_with, ProcessingEnv, ProcessingOptions, ProcessingRun};
use serde_json::json;

use crate::cli::{ProcessArgs, ProcessingArgs};
use crate::output::{eprintln, print, println};

/// Processes `args.jsonl`.
///
/// # Errors
/// Config, input, store, model and stage errors; summary write errors.
pub fn execute(args: &ProcessArgs) -> anyhow::Result<()> {
    let run = process_jsonl(&args.jsonl, &args.processing)?;
    report(&args.jsonl, &args.processing, &run)
}

/// Runs the pipeline on `jsonl` with the `processing` options, announcing the config first.
///
/// # Errors
/// As [`run_processing_with`].
pub fn process_jsonl(jsonl: &Path, processing: &ProcessingArgs) -> anyhow::Result<ProcessingRun> {
    let config_label = processing
        .config
        .as_ref()
        .map_or_else(|| "packaged default".to_owned(), |path| path.display().to_string());
    println(&format!("Processing with config: {config_label}"));
    let options = ProcessingOptions {
        config_path: processing.config.clone(),
        db_path: processing.db_path.clone(),
        ..ProcessingOptions::default()
    };
    Ok(run_processing_with(jsonl, &options, ProcessingEnv::default())?)
}

/// Prints the console report for `document` (and, with `--timings`, the stage timing summary
/// to stderr) and writes the `-o` summary when requested.
///
/// # Errors
/// Summary serialisation and write errors.
pub fn report(
    document: &Path,
    processing: &ProcessingArgs,
    run: &ProcessingRun,
) -> anyhow::Result<()> {
    print(&render_report(document, run));
    if processing.timings {
        eprintln(&crate::timings::render_run(&document.display().to_string(), &run.report));
    }
    if let Some(output) = &processing.output {
        let summary = summary_json(document, processing.config.as_deref(), run)?;
        std::fs::write(output, summary)
            .with_context(|| format!("failed to write results to {}", output.display()))?;
        println(&format!("Results written to: {}", output.display()));
    }
    Ok(())
}

/// The console report: the Python report block (fact and contradiction counts when the
/// contradiction stage ran), chunk / embedding counts of the RAG index stage and per-stage
/// wall times.
pub fn render_report(document: &Path, run: &ProcessingRun) -> String {
    let rule = "=".repeat(60);
    let result = &run.result;
    let mut out = String::new();
    // Writing to a String cannot fail.
    let _ = writeln!(out, "\n{rule}");
    let _ = writeln!(out, "Contradiction Detection Report");
    let _ = writeln!(out, "{rule}");
    let _ = writeln!(out, "Document: {}", document.display());
    if let Some(report) = &result.contradiction_report {
        let _ = writeln!(out, "Total facts extracted: {}", report.total_facts_extracted);
        let _ = writeln!(out, "Clean facts: {}", result.clean_facts.len());
        let _ = writeln!(out, "Contradictions: {}", result.contradiction_count);
        let _ = writeln!(out, "Needs review: {}", result.needs_review_count);
        let _ = writeln!(out, "Committed facts: {}", result.committed_fact_ids.len());
    }
    if let Some(index) = &run.data.rag_index_result {
        let _ =
            writeln!(out, "Chunks: {}, Embeddings: {}", index.chunks.len(), index.embeddings.count);
        if !index.embeddings.errors.is_empty() {
            let _ = writeln!(out, "Embedding errors: {}", index.embeddings.errors.len());
        }
    }
    if run.report.stage_timings.is_empty() {
        let _ = writeln!(out, "Stages: none ran");
    }
    for timing in &run.report.stage_timings {
        let _ = writeln!(out, "Stage {}: {:.3} s", timing.stage, timing.elapsed.as_secs_f64());
    }
    let _ = writeln!(out, "Total time: {:.3} s", run.report.total.as_secs_f64());
    if let Some(db_path) = &result.sqlite_db_path {
        let _ = writeln!(out, "SQLite output: {db_path}");
    }
    let _ = writeln!(out, "{rule}\n");
    out
}

/// The Python `-o` summary (`json.dump(..., indent=2)`, keys in Python order).
///
/// # Errors
/// Serialisation errors (not expected for these values).
pub fn summary_json(
    document: &Path,
    config: Option<&Path>,
    run: &ProcessingRun,
) -> anyhow::Result<String> {
    let result = &run.result;
    let total_facts_extracted =
        result.contradiction_report.as_ref().map_or(0, |report| report.total_facts_extracted);
    let summary = json!({
        "file_path": document.display().to_string(),
        "config_path": config.map(|path| path.display().to_string()),
        "total_facts_extracted": total_facts_extracted,
        "clean_facts_count": result.clean_facts.len(),
        "contradiction_count": result.contradiction_count,
        "needs_review_count": result.needs_review_count,
        "committed_fact_ids": result.committed_fact_ids,
        "sqlite_db_path": result.sqlite_db_path,
    });
    Ok(impeach_eval::report::to_python_json_indent2(&summary)?)
}
