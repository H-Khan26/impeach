//! Subcommand implementations.

pub mod e2e;
pub mod eval;
pub mod ingest;
pub mod process;
pub mod run;
pub mod store;

use crate::cli::{Command, StoreCommand};

/// Runs one subcommand.
///
/// # Errors
/// The subcommand's error.
pub fn dispatch(command: Command) -> anyhow::Result<()> {
    match command {
        Command::Ingest(args) => ingest::execute(&args),
        Command::Process(args) => process::execute(&args),
        Command::Run(args) => run::execute(&args),
        Command::Eval(args) => eval::execute(&args),
        Command::RunE2e(args) => e2e::execute(&args),
        Command::Store(StoreCommand::Reset(args)) => store::reset(&args),
    }
}
