//! `run_processing`: process one pre-ingested JSONL document (Python `api/processing.py`).
//!
//! [`run_processing`] is the public entry point. [`run_processing_with`] is the same flow with
//! injectable services (model registry, graph store, stage registry) for tests and the CLI, and
//! returns the full context and stage timings as well.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ce_config::pipeline::{ContradictionReviewStoreConfig, HelixConfig};
use ce_config::{PipelineConfig, StageConfig};
use ce_core::processing::ProcessingResult;
use ce_models::ModelRegistry;
use ce_store::GraphStore;
use serde_json::Value;

use crate::context::{ContextData, StageBuildContext};
use crate::error::Result;
use crate::factory::{StageRegistry, UnregisteredStages};
use crate::runner::{PipelineInput, RunReport};

/// Options of [`run_processing`].
#[derive(Debug, Clone)]
pub struct ProcessingOptions {
    /// Pipeline config YAML; `None` uses the packaged `processing_v17.yaml`.
    pub config_path: Option<PathBuf>,
    /// SQLite review database; when set, the first `ContradictionCheckStage` gets an enabled
    /// `review_store` at this path. The result reports it only when that stage ran.
    pub db_path: Option<PathBuf>,
    /// Whether SQLite write errors fail the run (only with `db_path`); default true.
    pub fail_on_sqlite_error: bool,
}

impl Default for ProcessingOptions {
    fn default() -> Self {
        Self { config_path: None, db_path: None, fail_on_sqlite_error: true }
    }
}

/// Services injected into [`run_processing_with`]; `None` fields use the production default.
#[derive(Debug, Default)]
pub struct ProcessingEnv {
    /// Model registry; default [`ModelRegistry::from_config`].
    pub registry: Option<Arc<ModelRegistry>>,
    /// Graph store; default [`open_store`] on the config's `helix` block.
    pub store: Option<Arc<dyn GraphStore>>,
    /// Stage constructors; default [`StageRegistry::builtin`].
    pub stages: Option<StageRegistry>,
    /// Handling of stage types without a constructor; default
    /// [`UnregisteredStages::Skip`] (stage types not ported yet are left out with a warning).
    pub unregistered: Option<UnregisteredStages>,
}

/// Everything a processing run produced.
#[derive(Debug)]
pub struct ProcessingRun {
    /// The `run_processing` result.
    pub result: ProcessingResult,
    /// The final pipeline context data (every intermediate value).
    pub data: ContextData,
    /// Per-stage timings.
    pub report: RunReport,
    /// Names of the stages that ran, in order.
    pub stages: Vec<String>,
}

/// Runs the processing pipeline on the first record of a pre-ingested JSONL file.
///
/// 1. Reads the record ([`ce_ingest::read_first_record`]: `text` required, `doc_id` defaults
///    to the file stem, extra top-level keys move into `metadata`).
/// 2. Loads the config ([`load_processing_config`]).
/// 3. Opens the graph store and builds the registered stages.
/// 4. Seeds the context with `doc_metadata` (when not empty) and `case_id` (when present),
///    then runs the pipeline on the record.
/// 5. Collects the result from the context. `sqlite_db_path` is `options.db_path` only when a
///    `ContradictionCheckStage` with the review store was built and ran; otherwise `None`.
///
/// # Errors
/// [`crate::PipelineError::Ingest`] for path / JSONL problems (empty path, missing or empty
/// file, invalid JSON, missing `text`), [`crate::PipelineError::Config`] for config problems,
/// store, model and stage errors.
pub fn run_processing(
    jsonl_path: impl AsRef<Path>,
    options: &ProcessingOptions,
) -> Result<ProcessingResult> {
    run_processing_with(jsonl_path, options, ProcessingEnv::default()).map(|run| run.result)
}

/// [`run_processing`] with injected services; returns the context and timings as well.
///
/// # Errors
/// As [`run_processing`].
pub fn run_processing_with(
    jsonl_path: impl AsRef<Path>,
    options: &ProcessingOptions,
    env: ProcessingEnv,
) -> Result<ProcessingRun> {
    let document = ce_ingest::read_first_record(jsonl_path.as_ref())?;
    let config = Arc::new(load_processing_config(options)?);
    let registry = env.registry.unwrap_or_else(|| Arc::new(ModelRegistry::from_config(&config)));
    let store = match env.store {
        Some(store) => store,
        None => open_store(&config.helix)?,
    };
    let stage_registry = env.stages.unwrap_or_else(StageRegistry::builtin);
    let review_store_attached = review_store_stage_is_built(&config, &stage_registry);
    let services = StageBuildContext::new(config, registry, store);
    let pipeline =
        stage_registry.build(services, env.unregistered.unwrap_or(UnregisteredStages::Skip))?;

    let mut ctx = pipeline.new_context();
    if !document.metadata.is_empty() {
        ctx.data.doc_metadata = Some(document.metadata.clone());
    }
    ctx.data.case_id = document.metadata.get("case_id").and_then(case_id_string);

    let report = pipeline.run(&mut ctx, PipelineInput::from(document))?;

    let data = ctx.data;
    let result = ProcessingResult::new(
        data.contradiction_report.clone(),
        data.clean_facts.clone().unwrap_or_default(),
        data.committed_fact_ids.clone().unwrap_or_default(),
        data.extracted_facts.clone().unwrap_or_default(),
        data.ed_results.clone().unwrap_or_default(),
        options
            .db_path
            .as_ref()
            .filter(|_| review_store_attached)
            .map(|p| p.to_string_lossy().into_owned()),
    );
    let stages = pipeline.stage_names().into_iter().map(str::to_owned).collect();
    Ok(ProcessingRun { result, data, report, stages })
}

/// Whether the pipeline built from `config` includes a stage that writes the SQLite review
/// store: an enabled `ContradictionCheckStage` with an enabled `review_store` whose type has a
/// constructor in `stages`. Unregistered stage types are skipped or fail the build, so a stage
/// counted here is built and, when the run succeeds, has run.
fn review_store_stage_is_built(config: &PipelineConfig, stages: &StageRegistry) -> bool {
    config.stages.iter().any(|stage| match stage {
        StageConfig::ContradictionCheck(check) => {
            stage.enabled()
                && check.review_store.as_ref().is_some_and(|review| review.enabled)
                && stages.is_registered(stage.type_name())
        }
        _ => false,
    })
}

/// A truthy `case_id` as text (Python `if case_id:`).
fn case_id_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        Value::Number(n) if n.as_f64() != Some(0.0) => Some(n.to_string()),
        _ => None,
    }
}

/// Loads `options.config_path` (default: the packaged `processing_v17.yaml`) and, when
/// `options.db_path` is set, enables the SQLite review store of the first
/// `ContradictionCheckStage` at that path (a warning is logged when there is none).
///
/// # Errors
/// [`crate::PipelineError::Config`].
pub fn load_processing_config(options: &ProcessingOptions) -> Result<PipelineConfig> {
    let mut config = match &options.config_path {
        Some(path) => ce_config::load_config(path)?,
        None => ce_config::load_default_config()?,
    };
    if let Some(db_path) = &options.db_path {
        enable_review_store(&mut config, db_path, options.fail_on_sqlite_error);
    }
    Ok(config)
}

fn enable_review_store(config: &mut PipelineConfig, db_path: &Path, fail_on_sqlite_error: bool) {
    let stage = config.stages.iter_mut().find_map(|stage| match stage {
        StageConfig::ContradictionCheck(check) => Some(check),
        _ => None,
    });
    match stage {
        Some(check) => {
            check.review_store = Some(ContradictionReviewStoreConfig {
                db_path: db_path.to_string_lossy().into_owned(),
                enabled: true,
                fail_on_sqlite_error,
            });
            tracing::info!(db_path = %db_path.display(), "enabled SQLite review store for ContradictionCheckStage");
        }
        None => tracing::warn!(
            "no ContradictionCheckStage in the pipeline config; SQLite output will not be enabled"
        ),
    }
}

/// Opens the graph store of a `helix` config block: `ce_store::HelixStore::from_config` with
/// the `helix` feature; without it an in-memory [`ce_store::MemoryStore`] of
/// `helix.embedding_dim` dimensions (nothing persists; a warning is logged).
///
/// # Errors
/// Store open errors (`helix` feature only).
pub fn open_store(helix: &HelixConfig) -> Result<Arc<dyn GraphStore>> {
    #[cfg(feature = "helix")]
    {
        let store: Arc<dyn GraphStore> = ce_store::HelixStore::from_config(helix)?;
        Ok(store)
    }
    #[cfg(not(feature = "helix"))]
    {
        tracing::warn!(
            mode = helix.mode.as_str(),
            "built without the `helix` feature; using an in-memory graph store (nothing persists)"
        );
        Ok(Arc::new(ce_store::MemoryStore::new(helix.embedding_dim)))
    }
}
