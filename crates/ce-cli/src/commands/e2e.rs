//! `run-e2e`: sequential end-to-end run over pre-ingested JSONL documents (Python
//! `tests/e2e/run_e2e.py`).

use std::path::{Path, PathBuf};

use ce_pipeline::e2e::{render_summary, run_e2e, scenario, E2eEnv, E2eOptions, Scenario};

use crate::cli::RunE2eArgs;
use crate::output::{print, println};

/// Default output directory of `run-e2e`.
const DEFAULT_OUTPUT_DIR: &str = "data/output";

/// Runs the scenario or the given inputs, prints the summary (and the evaluation summary).
///
/// # Errors
/// Missing arguments or files, config / store / write / evaluation errors, and a run whose
/// status is `failed`.
pub fn execute(args: &RunE2eArgs) -> anyhow::Result<()> {
    let preset: Option<Scenario> =
        args.scenario.as_deref().and_then(|name| scenario(name, &args.fixtures_dir));
    let inputs: Vec<PathBuf> = if args.inputs.is_empty() {
        preset.as_ref().map(|preset| preset.inputs.clone()).unwrap_or_default()
    } else {
        args.inputs.clone()
    };
    if inputs.is_empty() {
        anyhow::bail!("provide --scenario or --inputs");
    }
    let missing: Vec<String> = inputs
        .iter()
        .chain(args.config.as_ref())
        .filter(|path| !path.exists())
        .map(|path| path.display().to_string())
        .collect();
    if !missing.is_empty() {
        anyhow::bail!("file(s) not found: {}", missing.join(", "));
    }
    let key = if args.no_eval {
        None
    } else {
        args.key.clone().or_else(|| preset.as_ref().map(|preset| preset.key.clone()))
    };
    if let Some(key) = key.as_deref().filter(|key| !key.exists()) {
        anyhow::bail!("key not found: {}", key.display());
    }

    let label = args.scenario.as_deref().unwrap_or("custom");
    let output = args
        .output
        .clone()
        .unwrap_or_else(|| Path::new(DEFAULT_OUTPUT_DIR).join(format!("e2e_{label}.json")));
    let options = E2eOptions {
        config_path: args.config.clone(),
        output,
        db_path: args.db_path.clone(),
        append_db: args.append_db,
        key,
        no_eval: args.no_eval,
        threshold: args.threshold,
        fixtures_dir: args.fixtures_dir.clone(),
    };
    let run = run_e2e(&inputs, args.scenario.as_deref(), &options, E2eEnv::default())?;

    print(&render_summary(&run.output));
    println(&format!("Output written to {}", run.output_path.display()));
    if let Some(report) = &run.eval_report {
        let path = run.eval_report_path.as_ref().map(|path| path.display().to_string());
        print(&report.summary(path.as_deref()));
    }
    if run.output.status != "ok" {
        let error = run
            .output
            .documents
            .iter()
            .find_map(|document| {
                document.error.as_ref().map(|error| format!("{}: {error}", document.doc_path))
            })
            .unwrap_or_default();
        anyhow::bail!("e2e run failed ({error})");
    }
    Ok(())
}
