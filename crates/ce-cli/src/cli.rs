//! Command-line arguments (clap derive).

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

/// Default case identifier written to ingested metadata.
pub const DEFAULT_CASE_ID: &str = "CLI-CASE";
/// Default database identifier written to ingested metadata.
pub const DEFAULT_DB_ID: &str = "CLI-DB";

/// Contradiction detection over legal documents: ingest, process, evaluate.
#[derive(Debug, Parser)]
#[command(name = "contradiction-engine", version, about, long_about = None)]
pub struct Cli {
    /// Enable verbose (debug) logging.
    #[arg(short, long, global = true)]
    pub verbose: bool,

    #[command(subcommand)]
    pub command: Command,
}

/// Subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Ingest a raw document (.pdf, .txt, .json, or a transcript) into a one-line JSONL file.
    Ingest(IngestArgs),
    /// Run the processing pipeline on a pre-ingested JSONL file.
    Process(ProcessArgs),
    /// Ingest a document to a temporary JSONL file, then process it.
    Run(RunArgs),
    /// Evaluate e2e pipeline output against a ground-truth contradiction key.
    Eval(EvalArgs),
    /// Manage the graph store.
    #[command(subcommand)]
    Store(StoreCommand),
}

/// Metadata identifiers shared by `ingest` and `run`.
#[derive(Debug, Args)]
pub struct IdArgs {
    /// Case identifier for metadata.
    #[arg(long, default_value = DEFAULT_CASE_ID)]
    pub case_id: String,
    /// Database identifier for metadata.
    #[arg(long, default_value = DEFAULT_DB_ID)]
    pub db_id: String,
}

/// `ingest` arguments.
#[derive(Debug, Args)]
pub struct IngestArgs {
    /// Document to ingest.
    pub file: PathBuf,
    /// JSONL file to write.
    #[arg(short, long)]
    pub output: PathBuf,
    #[command(flatten)]
    pub ids: IdArgs,
    /// Parse the file as a transcript (.json, .csv, .txt, .text).
    #[arg(long)]
    pub transcript: bool,
}

/// `process` arguments.
#[derive(Debug, Args)]
pub struct ProcessArgs {
    /// Pre-ingested JSONL file (the first record is processed).
    pub jsonl: PathBuf,
    #[command(flatten)]
    pub processing: ProcessingArgs,
}

/// `run` arguments.
#[derive(Debug, Args)]
pub struct RunArgs {
    /// Document to process (.pdf, .txt, or .json).
    pub file: PathBuf,
    #[command(flatten)]
    pub ids: IdArgs,
    #[command(flatten)]
    pub processing: ProcessingArgs,
}

/// Processing options shared by `process` and `run`.
#[derive(Debug, Args)]
pub struct ProcessingArgs {
    /// Pipeline config YAML (default: the packaged processing config).
    #[arg(short, long)]
    pub config: Option<PathBuf>,
    /// SQLite database for storing contradictions.
    #[arg(long)]
    pub db_path: Option<PathBuf>,
    /// Write a JSON summary of the results to this path.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

/// `eval` arguments.
#[derive(Debug, Args)]
pub struct EvalArgs {
    /// Pipeline output JSON (e2e runner output).
    #[arg(short, long)]
    pub input: PathBuf,
    /// Ground-truth markdown key.
    #[arg(short, long)]
    pub key: PathBuf,
    /// Report JSON path (default: <input stem>_eval.json next to the input).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Cosine-similarity threshold for a fact match.
    #[arg(short, long, default_value_t = ce_eval::DEFAULT_THRESHOLD)]
    pub threshold: f64,
    /// Embedding model (fastembed: all-MiniLM-L6-v2 or BAAI/bge-small-en-v1.5).
    #[arg(short, long, default_value = ce_eval::DEFAULT_MODEL)]
    pub model: String,
    /// Disable the number/date/amount compatibility guard.
    #[arg(long)]
    pub no_value_guard: bool,
}

/// `store` subcommands.
#[derive(Debug, Subcommand)]
pub enum StoreCommand {
    /// Delete the embedded Helix database of the configured namespace.
    Reset(StoreResetArgs),
}

/// `store reset` arguments.
#[derive(Debug, Args)]
pub struct StoreResetArgs {
    /// Pipeline config YAML (default: the packaged processing config).
    #[arg(short, long)]
    pub config: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn command_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn ingest_defaults_match_python() {
        let cli = Cli::parse_from(["contradiction-engine", "ingest", "a.txt", "-o", "a.jsonl"]);
        let Command::Ingest(args) = cli.command else { panic!("expected ingest") };
        assert_eq!(args.ids.case_id, "CLI-CASE");
        assert_eq!(args.ids.db_id, "CLI-DB");
        assert!(!args.transcript);
        assert!(!cli.verbose);
    }

    #[test]
    fn verbose_is_global() {
        let cli = Cli::parse_from(["contradiction-engine", "run", "a.txt", "-v", "-c", "x.yaml"]);
        assert!(cli.verbose);
        let Command::Run(args) = cli.command else { panic!("expected run") };
        assert_eq!(args.processing.config, Some(PathBuf::from("x.yaml")));
    }

    #[test]
    fn eval_defaults_match_python() {
        let cli = Cli::parse_from(["contradiction-engine", "eval", "-i", "o.json", "-k", "k.md"]);
        let Command::Eval(args) = cli.command else { panic!("expected eval") };
        assert!((args.threshold - 0.75).abs() < f64::EPSILON);
        assert_eq!(args.model, "all-MiniLM-L6-v2");
        assert!(args.output.is_none());
        assert!(!args.no_value_guard);
    }
}
