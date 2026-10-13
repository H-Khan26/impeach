//! `contradiction-engine` command-line binary: `ingest`, `process`, `run`, `eval`, `run-e2e`
//! and `store reset` (Python `python -m contradiction_engine` and `tests/e2e/evaluate_output.py`).

mod cli;
mod commands;
mod error;
mod output;

use std::io::IsTerminal;
use std::process::ExitCode;

use clap::Parser;
use tracing_subscriber::EnvFilter;

use crate::cli::Cli;

/// Default log filter: this workspace's crates at info, dependencies at warn.
const DEFAULT_FILTER: &str = "warn,contradiction_engine=info,ce_core=info,ce_config=info,\
                              ce_text=info,ce_ingest=info,ce_store=info,ce_models=info,\
                              ce_pipeline=info,ce_eval=info";

/// Filter of `-v`: this workspace's crates at debug, dependencies at info.
const VERBOSE_FILTER: &str = "info,contradiction_engine=debug,ce_core=debug,ce_config=debug,\
                              ce_text=debug,ce_ingest=debug,ce_store=debug,ce_models=debug,\
                              ce_pipeline=debug,ce_eval=debug";

fn main() -> ExitCode {
    let cli = Cli::parse();
    init_tracing(cli.verbose);
    match commands::dispatch(cli.command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let kind = error::classify(&error);
            output::eprintln(&format!("{}: {}", kind.prefix(), error::message(&error)));
            if cli.verbose {
                output::eprintln(&format!("{error:?}"));
            }
            ExitCode::FAILURE
        }
    }
}

/// Logs to stderr (ANSI colours only on a terminal): `-v` selects [`VERBOSE_FILTER`],
/// otherwise `RUST_LOG` or [`DEFAULT_FILTER`].
fn init_tracing(verbose: bool) {
    let filter = if verbose {
        EnvFilter::new(VERBOSE_FILTER)
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER))
    };
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal())
        .with_env_filter(filter)
        .init();
}
