//! `impeach` command-line binary: `ingest`, `process`, `run`, `eval`, `run-e2e`
//! and `store reset` (the Python CLI entry point and `tests/e2e/evaluate_output.py`).

mod cli;
mod commands;
mod error;
mod output;
mod timings;

use std::io::IsTerminal;
use std::process::ExitCode;

use clap::Parser;
use tracing_subscriber::EnvFilter;

use crate::cli::Cli;

/// Default log filter: this workspace's crates at info, dependencies at warn.
const DEFAULT_FILTER: &str = "warn,impeach=info,impeach_core=info,impeach_config=info,\
                              impeach_text=info,impeach_ingest=info,impeach_store=info,\
                              impeach_models=info,impeach_pipeline=info,impeach_eval=info";

/// Filter of `-v`: this workspace's crates at debug, dependencies at info.
const VERBOSE_FILTER: &str = "info,impeach=debug,impeach_core=debug,impeach_config=debug,\
                              impeach_text=debug,impeach_ingest=debug,impeach_store=debug,\
                              impeach_models=debug,impeach_pipeline=debug,impeach_eval=debug";

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
