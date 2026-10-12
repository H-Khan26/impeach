//! End-to-end runner (Python `tests/e2e/run_e2e.py`): pre-ingested JSONL documents processed
//! sequentially through [`crate::run_processing_with`] into one graph store, so later
//! documents are checked against facts committed by earlier ones; writes the stage-based
//! [`E2eOutput`] JSON that `ce-eval` reads and evaluates it when a key is available.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ce_core::e2e::{
    ContradictionCheckSummary, E2eAggregates, E2eDocument, E2eOutput, EdSummary,
    FactExtractionSummary,
};
use ce_core::processing::ProcessingResult;
use ce_core::{EdResult, ExtractedFact};
use ce_eval::EvalReport;
use ce_models::{Embedder, ModelRegistry};
use ce_store::GraphStore;

use crate::error::Result;
use crate::factory::StageRegistry;

/// Preset scenario names.
pub const SCENARIO_NAMES: &[&str] = &["green_valley", "mason"];

/// A preset scenario: inputs in processing order and its contradiction key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scenario {
    /// Scenario name.
    pub name: String,
    /// JSONL inputs, in processing order.
    pub inputs: Vec<PathBuf>,
    /// Markdown contradiction key.
    pub key: PathBuf,
}

/// The preset scenario `name` with paths under `fixtures_dir` (`input/` and `keys/`).
#[must_use]
pub fn scenario(name: &str, fixtures_dir: &Path) -> Option<Scenario> {
    todo!("worker C")
}

/// Options of an e2e run.
#[derive(Debug, Clone)]
pub struct E2eOptions {
    /// Pipeline config YAML; `None` uses the packaged `processing_v17.yaml`.
    pub config_path: Option<PathBuf>,
    /// Output JSON path.
    pub output: PathBuf,
    /// SQLite review store; `None` uses `<output>.sqlite3`.
    pub db_path: Option<PathBuf>,
    /// Keep an existing SQLite file and append to it (default: delete it first).
    pub append_db: bool,
    /// Contradiction key; `None` uses the scenario's key (if any).
    pub key: Option<PathBuf>,
    /// Skip evaluation even when a key is available.
    pub no_eval: bool,
    /// Evaluator similarity threshold; `None` uses [`ce_eval::DEFAULT_THRESHOLD`].
    pub threshold: Option<f64>,
    /// Root of the scenario fixtures (`input/`, `keys/`).
    pub fixtures_dir: PathBuf,
}

/// Services injected into an e2e run; `None` fields use the production default.
#[derive(Default)]
pub struct E2eEnv {
    /// Model registry shared by every document.
    pub registry: Option<Arc<ModelRegistry>>,
    /// Graph store shared by every document.
    pub store: Option<Arc<dyn GraphStore>>,
    /// Stage constructors.
    pub stages: Option<StageRegistry>,
    /// Embedder of the evaluator; `None` loads [`ce_eval::DEFAULT_MODEL`] (feature
    /// `fastembed`; without it evaluation is skipped with a warning).
    pub eval_embedder: Option<Arc<dyn Embedder>>,
}

impl std::fmt::Debug for E2eEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("E2eEnv")
            .field("registry", &self.registry)
            .field("store", &self.store)
            .field("stages", &self.stages)
            .field("eval_embedder", &self.eval_embedder.as_ref().map(|e| e.model_name().to_owned()))
            .finish()
    }
}

/// What an e2e run produced.
#[derive(Debug)]
pub struct E2eRun {
    /// The output written to [`E2eRun::output_path`].
    pub output: E2eOutput,
    /// Output JSON path.
    pub output_path: PathBuf,
    /// Evaluation report, when a key was evaluated.
    pub eval_report: Option<EvalReport>,
    /// Evaluation report path, when written.
    pub eval_report_path: Option<PathBuf>,
}

/// Runs `inputs` sequentially (stopping at the first failed document), writes the output
/// JSON and evaluates it against the key when one is available and the run succeeded.
/// `scenario` labels the run (`e2e_<scenario|custom>_<timestamp>`).
///
/// # Errors
/// Config, store-open, output-write and evaluation errors. A failing document does not fail
/// the run: it is recorded with its `error` and the status becomes `failed`.
pub fn run_e2e(
    inputs: &[PathBuf],
    scenario: Option<&str>,
    options: &E2eOptions,
    env: E2eEnv,
) -> Result<E2eRun> {
    todo!("worker C")
}

/// [`run_e2e`] over the preset scenario `name` (its key unless `options.key` is set).
///
/// # Errors
/// [`crate::PipelineError::Config`] (validation) for an unknown scenario; errors of
/// [`run_e2e`].
pub fn run_scenario(name: &str, options: &E2eOptions, env: E2eEnv) -> Result<E2eRun> {
    todo!("worker C")
}

/// Entity disambiguation summary (`summarize_ed`).
#[must_use]
pub fn summarize_ed(ed_results: &[EdResult]) -> EdSummary {
    todo!("worker C")
}

/// Fact extraction summary (`summarize_facts`).
#[must_use]
pub fn summarize_facts(facts: &[ExtractedFact]) -> FactExtractionSummary {
    todo!("worker C")
}

/// Contradiction check summary (`summarize_contradictions`).
#[must_use]
pub fn summarize_contradictions(result: &ProcessingResult) -> ContradictionCheckSummary {
    todo!("worker C")
}

/// Totals over the successfully processed documents (`build_aggregates`).
#[must_use]
pub fn build_aggregates(documents: &[E2eDocument]) -> E2eAggregates {
    todo!("worker C")
}
