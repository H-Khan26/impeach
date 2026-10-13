//! Command-line arguments (clap derive).

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

/// Default case identifier written to ingested metadata.
pub const DEFAULT_CASE_ID: &str = "CLI-CASE";
/// Default database identifier written to ingested metadata.
pub const DEFAULT_DB_ID: &str = "CLI-DB";

/// Contradiction detection over legal documents: ingest, process, evaluate.
#[derive(Debug, Parser)]
#[command(name = "impeach", version, about, long_about = None)]
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
    /// Run pre-ingested JSONL documents sequentially into one store, write the e2e output
    /// JSON and evaluate it against a key.
    #[command(name = "run-e2e", after_help = SCENARIO_HELP)]
    RunE2e(RunE2eArgs),
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
    #[arg(short, long, default_value_t = impeach_eval::DEFAULT_THRESHOLD)]
    pub threshold: f64,
    /// Embedding model (fastembed: all-MiniLM-L6-v2 or BAAI/bge-small-en-v1.5).
    #[arg(short, long, default_value = impeach_eval::DEFAULT_MODEL)]
    pub model: String,
    /// Disable the number/date/amount compatibility guard.
    #[arg(long)]
    pub no_value_guard: bool,
}

/// Preset scenarios listed after the `run-e2e` help.
pub const SCENARIO_HELP: &str = "Scenarios (paths under --fixtures-dir):
  green_valley: input/green_valley_incident_report.jsonl -> input/lopez_v_green_valley_complaint.jsonl (key: keys/GREEN_VALLEY_CONTRA_KEY.md)
  mason: input/Mason_billy.jsonl -> input/Mason_watson.jsonl (key: keys/MASON_KEY.md)";

/// `run-e2e` arguments.
#[derive(Debug, Args)]
pub struct RunE2eArgs {
    /// Preset inputs and key.
    #[arg(long, value_parser = clap::builder::PossibleValuesParser::new(impeach_pipeline::e2e::SCENARIO_NAMES))]
    pub scenario: Option<String>,
    /// JSONL files, processed in the given order (overrides the scenario inputs).
    #[arg(long, num_args = 1.., value_name = "JSONL")]
    pub inputs: Vec<PathBuf>,
    /// Pipeline config YAML (default: the packaged processing config).
    #[arg(short, long)]
    pub config: Option<PathBuf>,
    /// Output JSON (default: data/output/e2e_<scenario|custom>.json).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// SQLite review-store path (default: <output>.sqlite3 next to the output).
    #[arg(long)]
    pub db_path: Option<PathBuf>,
    /// Keep an existing SQLite file at the review-store path and append to it (default:
    /// delete it first).
    #[arg(long)]
    pub append_db: bool,
    /// Contradiction key markdown; enables evaluation (a scenario supplies a default).
    #[arg(long)]
    pub key: Option<PathBuf>,
    /// Skip evaluation even if a key is available.
    #[arg(long)]
    pub no_eval: bool,
    /// Evaluator similarity threshold (default: the evaluator default, 0.75).
    #[arg(short, long)]
    pub threshold: Option<f64>,
    /// Root of the scenario fixtures (`input/`, `keys/`).
    #[arg(long, default_value = "fixtures")]
    pub fixtures_dir: PathBuf,
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
        let cli = Cli::parse_from(["impeach", "ingest", "a.txt", "-o", "a.jsonl"]);
        let Command::Ingest(args) = cli.command else { panic!("expected ingest") };
        assert_eq!(args.ids.case_id, "CLI-CASE");
        assert_eq!(args.ids.db_id, "CLI-DB");
        assert!(!args.transcript);
        assert!(!cli.verbose);
    }

    #[test]
    fn verbose_is_global() {
        let cli = Cli::parse_from(["impeach", "run", "a.txt", "-v", "-c", "x.yaml"]);
        assert!(cli.verbose);
        let Command::Run(args) = cli.command else { panic!("expected run") };
        assert_eq!(args.processing.config, Some(PathBuf::from("x.yaml")));
    }

    #[test]
    fn run_e2e_defaults() {
        let cli = Cli::parse_from(["impeach", "run-e2e", "--scenario", "mason"]);
        let Command::RunE2e(args) = cli.command else { panic!("expected run-e2e") };
        assert_eq!(args.scenario.as_deref(), Some("mason"));
        assert!(args.inputs.is_empty());
        assert!(args.config.is_none() && args.output.is_none() && args.db_path.is_none());
        assert!(!args.append_db && !args.no_eval);
        assert!(args.key.is_none() && args.threshold.is_none());
        assert_eq!(args.fixtures_dir, PathBuf::from("fixtures"));
    }

    #[test]
    fn run_e2e_takes_several_inputs_and_short_flags() {
        let cli = Cli::parse_from([
            "impeach",
            "run-e2e",
            "--inputs",
            "a.jsonl",
            "b.jsonl",
            "-c",
            "x.yaml",
            "-o",
            "out.json",
            "-t",
            "0.8",
            "--no-eval",
            "--append-db",
        ]);
        let Command::RunE2e(args) = cli.command else { panic!("expected run-e2e") };
        assert_eq!(args.inputs, [PathBuf::from("a.jsonl"), PathBuf::from("b.jsonl")]);
        assert_eq!(args.config, Some(PathBuf::from("x.yaml")));
        assert_eq!(args.output, Some(PathBuf::from("out.json")));
        assert_eq!(args.threshold, Some(0.8));
        assert!(args.no_eval && args.append_db);
    }

    #[test]
    fn run_e2e_rejects_unknown_scenarios() {
        let error = Cli::try_parse_from(["impeach", "run-e2e", "--scenario", "x"])
            .expect_err("unknown scenario");
        assert_eq!(error.kind(), clap::error::ErrorKind::InvalidValue);
    }

    #[test]
    fn eval_defaults_match_python() {
        let cli = Cli::parse_from(["impeach", "eval", "-i", "o.json", "-k", "k.md"]);
        let Command::Eval(args) = cli.command else { panic!("expected eval") };
        assert!((args.threshold - 0.75).abs() < f64::EPSILON);
        assert_eq!(args.model, "all-MiniLM-L6-v2");
        assert!(args.output.is_none());
        assert!(!args.no_value_guard);
    }
}
