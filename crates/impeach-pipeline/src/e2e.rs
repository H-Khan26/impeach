//! End-to-end runner (Python `tests/e2e/run_e2e.py`): pre-ingested JSONL documents processed
//! sequentially through [`crate::run_processing_with`] into one graph store, so later
//! documents are checked against facts committed by earlier ones; writes the stage-based
//! [`E2eOutput`] JSON that `impeach-eval` reads and evaluates it when a key is available.
//!
//! [`run_e2e`]:
//! 1. Loads the config once and opens one model registry and one graph store (the injected
//!    ones of [`E2eEnv`], else [`ModelRegistry::from_config`] and [`crate::open_store`]),
//!    shared by every document.
//! 2. Starts a fresh SQLite review store (`db_path`, default `<output>.sqlite3`): an existing
//!    file is deleted unless `append_db`.
//! 3. Processes the inputs in order with the review store enabled (`fail_on_sqlite_error`).
//!    Each document entry starts with the `doc_id` / `case_id` / `db_id` / `title` of its first
//!    JSONL record (top-level key, else the same key in `metadata`; scalars rendered as text).
//!    The first failing document is recorded with its `error` (the error message) and stops
//!    the run (`status: "failed"`): later documents depend on earlier ones being committed.
//! 4. Writes the output JSON (Python `json.dumps(indent=2)`), creating parent directories.
//! 5. When the run succeeded, `no_eval` is off and a key is set: evaluates the output with
//!    [`impeach_eval::run_evaluation`] and writes the report next to it
//!    ([`impeach_eval::default_report_path`]). The evaluator embedder is [`E2eEnv::eval_embedder`],
//!    else [`impeach_eval::DEFAULT_MODEL`] through fastembed (without the `fastembed` feature the
//!    evaluation is skipped with a warning).
//!
//! [`render_summary`] renders the per-document tables and aggregates the Python runner printed.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use chrono::Local;
use impeach_core::e2e::{
    CleanFactRecord, ContradictionCheckSummary, CountMap, E2eAggregates, E2eDocument, E2eOutput,
    EdSummary, EntityLinkingSummary, FactExtractionSummary, FactRecord, FilteringStats, NliStats,
    PairRecord, PipelineSummary, RetrievalStats,
};
use impeach_core::processing::ProcessingResult;
use impeach_core::util::python_isoformat;
use impeach_core::{ContradictionPair, ContradictionSource, EdResult, ExtractedFact};
use impeach_eval::{EvalOptions, EvalReport};
use impeach_models::{Embedder, ModelRegistry};
use impeach_store::GraphStore;
use serde_json::Value;

use crate::api::{
    load_processing_config, open_store, run_processing_with, ProcessingEnv, ProcessingOptions,
};
use crate::error::{PipelineError, Result};
use crate::factory::StageRegistry;

/// Preset scenario names.
pub const SCENARIO_NAMES: &[&str] = &["green_valley", "mason"];

/// Config path reported when the packaged default config is used.
pub const DEFAULT_CONFIG_LABEL: &str = "impeach/config/processing/processing_v17.yaml";

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
    let (inputs, key): (&[&str], &str) = match name {
        "green_valley" => (
            &["green_valley_incident_report.jsonl", "lopez_v_green_valley_complaint.jsonl"],
            "GREEN_VALLEY_CONTRA_KEY.md",
        ),
        "mason" => (&["Mason_billy.jsonl", "Mason_watson.jsonl"], "MASON_KEY.md"),
        _ => return None,
    };
    Some(Scenario {
        name: name.to_owned(),
        inputs: inputs.iter().map(|input| fixtures_dir.join("input").join(input)).collect(),
        key: fixtures_dir.join("keys").join(key),
    })
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
    /// Evaluator similarity threshold; `None` uses [`impeach_eval::DEFAULT_THRESHOLD`].
    pub threshold: Option<f64>,
    /// Root of the scenario fixtures (`input/`, `keys/`).
    pub fixtures_dir: PathBuf,
}

impl E2eOptions {
    /// Options writing to `output` with every other setting at its default (packaged config,
    /// `<output>.sqlite3`, fresh database, evaluation when a key is available, fixtures under
    /// `fixtures/`).
    pub fn new(output: impl Into<PathBuf>) -> Self {
        Self {
            config_path: None,
            output: output.into(),
            db_path: None,
            append_db: false,
            key: None,
            no_eval: false,
            threshold: None,
            fixtures_dir: PathBuf::from("fixtures"),
        }
    }

    /// The SQLite review store path: `db_path`, else the output path with extension
    /// `.sqlite3`.
    #[must_use]
    pub fn sqlite_path(&self) -> PathBuf {
        self.db_path.clone().unwrap_or_else(|| self.output.with_extension("sqlite3"))
    }
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
    /// Embedder of the evaluator; `None` loads [`impeach_eval::DEFAULT_MODEL`] (feature
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
    let key = options.key.as_ref().filter(|_| !options.no_eval);
    if let Some(key) = key {
        if !key.exists() {
            return Err(PipelineError::Io {
                path: key.clone(),
                source: std::io::Error::new(std::io::ErrorKind::NotFound, "key not found"),
            });
        }
    }

    let db_path = options.sqlite_path();
    let processing = ProcessingOptions {
        config_path: options.config_path.clone(),
        db_path: Some(db_path.clone()),
        fail_on_sqlite_error: true,
    };
    let config = load_processing_config(&processing)?;
    let registry = env.registry.unwrap_or_else(|| Arc::new(ModelRegistry::from_config(&config)));
    let store = match env.store {
        Some(store) => store,
        None => open_store(&config.helix)?,
    };
    prepare_review_db(&db_path, options.append_db)?;

    let run_id =
        format!("e2e_{}_{}", scenario.unwrap_or("custom"), Local::now().format("%Y%m%d_%H%M%S"));
    tracing::info!(
        run_id = run_id.as_str(),
        pipeline = config.name(),
        helix_mode = config.helix.mode.as_str(),
        namespace = config.helix_namespace(),
        sqlite = %db_path.display(),
        inputs = inputs.len(),
        "e2e run started"
    );

    let mut documents: Vec<E2eDocument> = Vec::with_capacity(inputs.len());
    let mut failed = false;
    for (index, path) in inputs.iter().enumerate() {
        let header = read_jsonl_header(path);
        tracing::info!(
            "Document {}: {} (doc_id={}, case_id={})",
            index + 1,
            path.display(),
            header.doc_id.as_deref().unwrap_or("None"),
            header.case_id.as_deref().unwrap_or("None")
        );
        let doc_env = ProcessingEnv {
            registry: Some(Arc::clone(&registry)),
            store: Some(Arc::clone(&store)),
            stages: env.stages.clone(),
            unregistered: None,
        };
        let start = Instant::now();
        match run_processing_with(path, &processing, doc_env) {
            Ok(run) => {
                let result = &run.result;
                documents.push(E2eDocument {
                    processing_time_seconds: Some(start.elapsed().as_secs_f64()),
                    stage_4_entity_disambiguation: Some(summarize_ed(&result.ed_results)),
                    stage_5_fact_extraction: Some(summarize_facts(&result.extracted_facts)),
                    stage_6_contradiction_check: Some(summarize_contradictions(result)),
                    ..header
                });
            }
            Err(error) => {
                tracing::error!(document = %path.display(), "FAILED: {error}");
                documents.push(E2eDocument { error: Some(error.to_string()), ..header });
                failed = true;
                break;
            }
        }
    }

    let total_time: f64 =
        documents.iter().filter_map(|document| document.processing_time_seconds).sum();
    #[allow(clippy::cast_precision_loss)]
    let average = if documents.is_empty() { 0.0 } else { total_time / documents.len() as f64 };
    let output = E2eOutput {
        run_id,
        timestamp: python_isoformat(&Local::now().naive_local()),
        scenario: scenario.map(str::to_owned),
        pipeline_name: config.name().to_owned(),
        helix_namespace: config.helix_namespace().to_owned(),
        helix_url: config.helix.url.clone(),
        helix_mode: config.helix.mode.as_str().to_owned(),
        helix_data_dir: Some(config.helix.data_dir.clone()),
        config_path: options
            .config_path
            .as_ref()
            .map_or_else(|| DEFAULT_CONFIG_LABEL.to_owned(), |path| path.display().to_string()),
        sqlite_db_path: db_path.display().to_string(),
        inputs: inputs.iter().map(|path| path.display().to_string()).collect(),
        status: if failed { "failed" } else { "ok" }.to_owned(),
        pipeline_summary: PipelineSummary {
            total_documents: documents.len() as u64,
            total_processing_time_seconds: total_time,
            avg_processing_time_per_doc: average,
        },
        aggregates: build_aggregates(&documents),
        documents,
    };
    write_output(&options.output, &output)?;
    tracing::info!(output = %options.output.display(), status = output.status.as_str(), "e2e output written");

    let mut run = E2eRun {
        output,
        output_path: options.output.clone(),
        eval_report: None,
        eval_report_path: None,
    };
    if let (Some(key), false) = (key, failed) {
        let Some(embedder) = eval_embedder(env.eval_embedder)? else {
            return Ok(run);
        };
        let report_path = impeach_eval::default_report_path(&options.output);
        let eval_options = EvalOptions {
            threshold: options.threshold.unwrap_or(impeach_eval::DEFAULT_THRESHOLD),
            ..EvalOptions::default()
        };
        let report = impeach_eval::run_evaluation(
            &options.output,
            key,
            Some(&report_path),
            embedder.as_ref(),
            &eval_options,
        )?;
        run.eval_report = Some(report);
        run.eval_report_path = Some(report_path);
    }
    Ok(run)
}

/// [`run_e2e`] over the preset scenario `name` (its key unless `options.key` is set).
///
/// # Errors
/// [`crate::PipelineError::Config`] (validation) for an unknown scenario; errors of
/// [`run_e2e`].
pub fn run_scenario(name: &str, options: &E2eOptions, env: E2eEnv) -> Result<E2eRun> {
    let preset = scenario(name, &options.fixtures_dir).ok_or_else(|| {
        impeach_config::ConfigError::validation(
            "scenario",
            format!("unknown scenario `{name}`; expected one of {SCENARIO_NAMES:?}"),
        )
    })?;
    let mut options = options.clone();
    if options.key.is_none() {
        options.key = Some(preset.key.clone());
    }
    run_e2e(&preset.inputs, Some(name), &options, env)
}

/// Deletes an existing review database unless `append`, and creates its directory.
fn prepare_review_db(db_path: &Path, append: bool) -> Result<()> {
    let io_error = |source| PipelineError::Io { path: db_path.to_path_buf(), source };
    if db_path.exists() {
        if append {
            tracing::info!(
                "appending to existing SQLite review store {} (append_db)",
                db_path.display()
            );
        } else {
            std::fs::remove_file(db_path).map_err(io_error)?;
        }
    }
    if let Some(parent) = db_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(io_error)?;
    }
    Ok(())
}

/// Writes the output as Python `json.dumps(indent=2)`, creating parent directories.
fn write_output(path: &Path, output: &E2eOutput) -> Result<()> {
    let io_error = |source| PipelineError::Io { path: path.to_path_buf(), source };
    let text = impeach_eval::report::to_python_json_indent2(output)
        .map_err(|error| PipelineError::Eval(impeach_eval::EvalError::Serialize(error)))?;
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(io_error)?;
    }
    std::fs::write(path, text).map_err(io_error)
}

/// The injected evaluator embedder, else the default evaluator model (feature `fastembed`).
fn eval_embedder(injected: Option<Arc<dyn Embedder>>) -> Result<Option<Arc<dyn Embedder>>> {
    if let Some(embedder) = injected {
        return Ok(Some(embedder));
    }
    #[cfg(feature = "fastembed")]
    {
        use impeach_models::{FastEmbedOptions, FastEmbedder};
        tracing::info!(model = impeach_eval::DEFAULT_MODEL, "loading evaluator embedding model");
        let embedder = FastEmbedder::new(&FastEmbedOptions::new(impeach_eval::DEFAULT_MODEL))?;
        Ok(Some(Arc::new(embedder)))
    }
    #[cfg(not(feature = "fastembed"))]
    {
        tracing::warn!(
            "evaluation needs the embedding model {}, but this build has no `fastembed` \
             feature; skipping evaluation",
            impeach_eval::DEFAULT_MODEL
        );
        Ok(None)
    }
}

/// `doc_id` / `case_id` / `db_id` / `title` of the first JSONL record (empty header when the
/// file cannot be read; the processing error reports why).
fn read_jsonl_header(path: &Path) -> E2eDocument {
    let mut document =
        E2eDocument { doc_path: path.display().to_string(), ..E2eDocument::default() };
    let Ok(text) = std::fs::read_to_string(path) else {
        return document;
    };
    let Some(Ok(Value::Object(record))) =
        text.lines().find(|line| !line.trim().is_empty()).map(serde_json::from_str::<Value>)
    else {
        return document;
    };
    let metadata = record.get("metadata").and_then(Value::as_object);
    let field = |key: &str| -> Option<String> {
        let value = match record.get(key) {
            Some(value) => value,
            None => metadata.and_then(|meta| meta.get(key))?,
        };
        match value {
            Value::Null => None,
            Value::String(text) => Some(text.clone()),
            other => Some(other.to_string()),
        }
    };
    document.doc_id = field("doc_id");
    document.case_id = field("case_id");
    document.db_id = field("db_id");
    document.title = field("title");
    document
}

/// Increments `key` of a counter (insertion ordered like Python `Counter`).
fn count(map: &mut CountMap, key: &str) {
    *map.entry(key.to_owned()).or_insert(0) += 1;
}

/// `part / whole * 100`, 0.0 when `whole` is 0.
#[allow(clippy::cast_precision_loss)]
fn percent(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64 * 100.0
    }
}

/// Entity disambiguation summary (`summarize_ed`).
#[must_use]
pub fn summarize_ed(ed_results: &[EdResult]) -> EdSummary {
    let mut summary = EdSummary::default();
    for result in ed_results {
        if result.chosen_entity_id.as_ref().is_some_and(|id| !id.is_empty()) {
            let strategy =
                if result.strategy_used.is_empty() { "unknown" } else { &result.strategy_used };
            summary.total_resolved += 1;
            count(&mut summary.resolution_by_strategy, strategy);
            if strategy.contains("_create") {
                summary.new_entities_created_count += 1;
            } else {
                summary.entities_reused_count += 1;
            }
        }
    }
    summary.total_unresolved = ed_results.len() as u64 - summary.total_resolved;
    summary
}

fn non_empty(text: Option<&str>) -> Option<&str> {
    text.filter(|text| !text.is_empty())
}

/// One extracted fact (`fact_record`).
fn fact_record(fact: &ExtractedFact) -> FactRecord {
    FactRecord {
        fact_id: fact.id.to_string(),
        text: fact.text.clone(),
        value_type: fact.value_type.clone(),
        property_type: fact.property_type.clone(),
        subject_entity_id: fact.subject_entity_id.as_ref().map(ToString::to_string),
        object_entity_id: fact.object_entity_id.as_ref().map(ToString::to_string),
        is_boilerplate: fact.is_boilerplate,
    }
}

/// Fact extraction summary (`summarize_facts`).
#[must_use]
pub fn summarize_facts(facts: &[ExtractedFact]) -> FactExtractionSummary {
    let records: Vec<FactRecord> = facts.iter().map(fact_record).collect();
    let mut summary = FactExtractionSummary {
        total_facts: records.len() as u64,
        ..FactExtractionSummary::default()
    };
    let mut linking = EntityLinkingSummary::default();
    for record in &records {
        let subject = non_empty(record.subject_entity_id.as_deref()).is_some();
        let object = non_empty(record.object_entity_id.as_deref()).is_some();
        linking.linked_subjects += u64::from(subject);
        linking.linked_objects += u64::from(object);
        linking.linked_both += u64::from(subject && object);
        linking.linked_either += u64::from(subject || object);
        summary.boilerplate_count += u64::from(record.is_boilerplate);
        count(
            &mut summary.facts_by_value_type,
            non_empty(record.value_type.as_deref()).unwrap_or("UNKNOWN"),
        );
        count(
            &mut summary.facts_by_property_type,
            non_empty(record.property_type.as_deref()).unwrap_or("UNKNOWN"),
        );
    }
    linking.linking_rate_percent = percent(linking.linked_either, summary.total_facts);
    summary.entity_linking = linking;
    summary.all_facts = Some(records);
    summary
}

/// One detected pair (`pair_record`).
fn pair_record(pair: &ContradictionPair) -> PairRecord {
    PairRecord {
        new_fact_text: pair.new_fact_text.clone(),
        new_fact_id: pair.new_fact_id.to_string(),
        existing_fact_text: pair.existing_fact_text.clone(),
        existing_fact_id: pair.existing_fact_id.to_string(),
        nli_label: pair.nli_label.as_str().to_owned(),
        nli_confidence: pair.nli_confidence,
        retrieval_sources: pair.retrieval_sources.clone(),
    }
}

/// Contradiction check summary (`summarize_contradictions`).
#[must_use]
pub fn summarize_contradictions(result: &ProcessingResult) -> ContradictionCheckSummary {
    let report = result.contradiction_report.as_ref();
    let pairs: &[ContradictionPair] =
        report.map_or(&[], |report| report.unique_contradiction_pairs.as_slice());
    let (intra, cross): (Vec<&ContradictionPair>, Vec<&ContradictionPair>) = pairs
        .iter()
        .partition(|pair| pair.contradiction_source == ContradictionSource::IntraDocument);
    let mut sources = CountMap::new();
    let mut labels = CountMap::new();
    for pair in pairs {
        for source in &pair.retrieval_sources {
            count(&mut sources, source);
        }
        count(&mut labels, pair.nli_label.as_str());
    }
    ContradictionCheckSummary {
        total_checked: result.extracted_facts.len() as u64,
        contradiction_count: result.contradiction_count as u64,
        needs_review_count: result.needs_review_count as u64,
        clean_count: result.clean_facts.len() as u64,
        committed_fact_count: result.committed_fact_ids.len() as u64,
        all_clean_facts: result
            .clean_facts
            .iter()
            .map(|fact| {
                let record = fact_record(fact);
                CleanFactRecord {
                    fact_id: record.fact_id,
                    text: record.text,
                    value_type: record.value_type,
                    property_type: record.property_type,
                    subject_entity_id: record.subject_entity_id,
                    object_entity_id: record.object_entity_id,
                }
            })
            .collect(),
        intra_doc_contradictions: intra.into_iter().map(pair_record).collect(),
        cross_doc_contradictions: cross.into_iter().map(pair_record).collect(),
        retrieval_stats: RetrievalStats {
            total_candidates_retrieved: report
                .map_or(0, |report| report.total_candidates_retrieved as u64),
            retrieval_sources_breakdown: sources,
        },
        filtering_stats: FilteringStats {
            pairs_after_prefilter: report
                .map_or(0, |report| report.total_pairs_after_prefilter as u64),
            total_nli_calls: report.map_or(0, |report| report.total_nli_calls as u64),
        },
        nli_stats: NliStats {
            nli_label_breakdown: labels,
            contradiction_threshold: report.map(|report| report.contradiction_threshold),
            review_threshold: report.map(|report| report.review_threshold),
        },
    }
}

/// Totals over the successfully processed documents (`build_aggregates`).
#[must_use]
pub fn build_aggregates(documents: &[E2eDocument]) -> E2eAggregates {
    let mut aggregates = E2eAggregates::default();
    let mut reused = 0;
    let mut linked = 0;
    for document in documents.iter().filter(|document| document.error.is_none()) {
        if let Some(ed) = &document.stage_4_entity_disambiguation {
            aggregates.total_entities_resolved += ed.total_resolved;
            aggregates.total_entities_created += ed.new_entities_created_count;
            reused += ed.entities_reused_count;
        }
        if let Some(facts) = &document.stage_5_fact_extraction {
            aggregates.total_facts_extracted += facts.total_facts;
            linked += facts.entity_linking.linked_either;
        }
        if let Some(check) = &document.stage_6_contradiction_check {
            aggregates.total_facts_committed += check.committed_fact_count;
            aggregates.total_contradictions += check.contradiction_count;
            aggregates.total_needs_review += check.needs_review_count;
            aggregates.total_intra_doc_pairs += check.intra_doc_contradictions.len() as u64;
            aggregates.total_cross_doc_pairs += check.cross_doc_contradictions.len() as u64;
        }
    }
    aggregates.entity_reuse_rate_percent = percent(reused, aggregates.total_entities_resolved);
    aggregates.fact_linking_rate_percent = percent(linked, aggregates.total_facts_extracted);
    aggregates
}

/// `text` cut to `max_chars` characters with a `...` suffix (Python `truncate`).
fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_owned()
    } else {
        let mut cut: String = text.chars().take(max_chars.saturating_sub(3)).collect();
        cut.push_str("...");
        cut
    }
}

/// A two-column ASCII table with a centred title.
fn ascii_table(out: &mut String, title: &str, header: [&str; 2], rows: &[(String, String)]) {
    let width = |column: usize, head: &str| {
        rows.iter()
            .map(|row| if column == 0 { &row.0 } else { &row.1 })
            .map(|cell| cell.chars().count())
            .chain(std::iter::once(head.chars().count()))
            .max()
            .unwrap_or(0)
    };
    let (left, right) = (width(0, header[0]), width(1, header[1]));
    let total = left + right + 7;
    let rule = format!("+{}+{}+", "-".repeat(left + 2), "-".repeat(right + 2));
    // Writing to a String cannot fail.
    let _ = writeln!(out, "{title:^total$}");
    let _ = writeln!(out, "{rule}");
    let _ = writeln!(out, "| {:<left$} | {:<right$} |", header[0], header[1]);
    let _ = writeln!(out, "|{}+{}|", "-".repeat(left + 2), "-".repeat(right + 2));
    for (key, value) in rows {
        let _ = writeln!(out, "| {key:<left$} | {value:<right$} |");
    }
    let _ = writeln!(out, "{rule}");
}

/// The console summary of a run: one table per document (with its top three pairs per
/// kind), failed documents with their error, and the aggregates.
#[must_use]
pub fn render_summary(output: &E2eOutput) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "{}", output.run_id);
    let _ = writeln!(out, "config: {} (pipeline {})", output.config_path, output.pipeline_name);
    let _ = writeln!(
        out,
        "helix:  {} {} namespace={}",
        output.helix_mode,
        output.helix_url.as_deref().or(output.helix_data_dir.as_deref()).unwrap_or(""),
        output.helix_namespace
    );
    let _ = writeln!(out, "sqlite: {}", output.sqlite_db_path);
    let names: Vec<String> = output.inputs.iter().map(|input| file_name(input)).collect();
    let _ = writeln!(out, "inputs: {}", names.join(", "));

    for (index, document) in output.documents.iter().enumerate() {
        let _ = writeln!(
            out,
            "\nDocument {}: {} (doc_id={}, case_id={})",
            index + 1,
            file_name(&document.doc_path),
            document.doc_id.as_deref().unwrap_or("None"),
            document.case_id.as_deref().unwrap_or("None")
        );
        if let Some(error) = &document.error {
            let _ = writeln!(out, "  FAILED: {error}");
            continue;
        }
        let ed = document.stage_4_entity_disambiguation.clone().unwrap_or_default();
        let facts = document.stage_5_fact_extraction.clone().unwrap_or_default();
        let check = document.stage_6_contradiction_check.clone().unwrap_or_default();
        let rows = [
            (
                "Entity disambiguation".to_owned(),
                format!(
                    "{} resolved, {} created, {} reused, {} unresolved",
                    ed.total_resolved,
                    ed.new_entities_created_count,
                    ed.entities_reused_count,
                    ed.total_unresolved
                ),
            ),
            (
                "Facts".to_owned(),
                format!(
                    "{} extracted, {:.1}% entity-linked",
                    facts.total_facts, facts.entity_linking.linking_rate_percent
                ),
            ),
            (
                "Contradictions".to_owned(),
                format!(
                    "{} contradiction, {} needs review, {} clean, {} committed",
                    check.contradiction_count,
                    check.needs_review_count,
                    check.clean_count,
                    check.committed_fact_count
                ),
            ),
            (
                "Pairs".to_owned(),
                format!(
                    "{} intra-doc, {} cross-doc",
                    check.intra_doc_contradictions.len(),
                    check.cross_doc_contradictions.len()
                ),
            ),
        ];
        let title = format!(
            "{} ({:.1}s)",
            file_name(&document.doc_path),
            document.processing_time_seconds.unwrap_or(0.0)
        );
        ascii_table(&mut out, &title, ["Stage", "Key metrics"], &rows);
        for (label, pairs) in [
            ("Intra-doc", &check.intra_doc_contradictions),
            ("Cross-doc", &check.cross_doc_contradictions),
        ] {
            if pairs.is_empty() {
                continue;
            }
            let _ = writeln!(out, "  {label} pairs ({}), top 3:", pairs.len());
            let mut sorted: Vec<&PairRecord> = pairs.iter().collect();
            sorted.sort_by(|a, b| b.nli_confidence.total_cmp(&a.nli_confidence));
            for pair in sorted.into_iter().take(3) {
                let _ = writeln!(out, "    - {}", truncate(&pair.new_fact_text, 70));
                let _ = writeln!(
                    out,
                    "      vs {}  [{} {:.2}]",
                    truncate(&pair.existing_fact_text, 70),
                    pair.nli_label,
                    pair.nli_confidence
                );
            }
        }
    }

    let aggregates = &output.aggregates;
    let failed = if output.status == "failed" { " (FAILED)" } else { "" };
    let rows = [
        ("Documents processed".to_owned(), format!("{}{failed}", output.documents.len())),
        (
            "Total time".to_owned(),
            format!("{:.1}s", output.pipeline_summary.total_processing_time_seconds),
        ),
        (
            "Entities resolved / created".to_owned(),
            format!(
                "{} / {}",
                aggregates.total_entities_resolved, aggregates.total_entities_created
            ),
        ),
        ("Entity reuse rate".to_owned(), format!("{:.1}%", aggregates.entity_reuse_rate_percent)),
        (
            "Facts extracted / committed".to_owned(),
            format!("{} / {}", aggregates.total_facts_extracted, aggregates.total_facts_committed),
        ),
        ("Fact linking rate".to_owned(), format!("{:.1}%", aggregates.fact_linking_rate_percent)),
        (
            "Contradictions / needs review".to_owned(),
            format!("{} / {}", aggregates.total_contradictions, aggregates.total_needs_review),
        ),
        (
            "Pairs intra / cross".to_owned(),
            format!("{} / {}", aggregates.total_intra_doc_pairs, aggregates.total_cross_doc_pairs),
        ),
    ];
    out.push('\n');
    ascii_table(&mut out, "Aggregates", ["Metric", "Value"], &rows);
    out
}

/// The file name of a path string (the whole string when it has none).
fn file_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map_or_else(|| path.to_owned(), |name| name.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenarios_resolve_under_the_fixtures_dir() {
        let preset = scenario("mason", Path::new("fx")).expect("mason");
        assert_eq!(
            preset.inputs,
            [Path::new("fx/input/Mason_billy.jsonl"), Path::new("fx/input/Mason_watson.jsonl")]
        );
        assert_eq!(preset.key, Path::new("fx/keys/MASON_KEY.md"));
        assert!(scenario("other", Path::new("fx")).is_none());
        for name in SCENARIO_NAMES {
            assert!(scenario(name, Path::new(".")).is_some());
        }
    }

    #[test]
    fn truncate_counts_characters() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("abcdefghijk", 8), "abcde...");
        assert_eq!(truncate("ééééé", 4), "é...");
    }

    #[test]
    fn default_sqlite_path_sits_next_to_the_output() {
        let options = E2eOptions::new("out/e2e_mason.json");
        assert_eq!(options.sqlite_path(), Path::new("out/e2e_mason.sqlite3"));
    }

    #[test]
    fn ed_summary_counts_strategies() {
        let results: Vec<EdResult> = serde_json::from_value(serde_json::json!([
            {"chosen_entity_id": "e1", "candidates": [], "resolved_by_llm": false,
             "confidence": 1.0, "strategy_used": "rule_create"},
            {"chosen_entity_id": "e2", "candidates": [], "resolved_by_llm": false,
             "confidence": 1.0, "strategy_used": "bm25"},
            {"chosen_entity_id": null, "candidates": [], "resolved_by_llm": false,
             "confidence": 0.0, "strategy_used": "none"},
            {"chosen_entity_id": "e3", "candidates": [], "resolved_by_llm": false,
             "confidence": 1.0, "strategy_used": ""}
        ]))
        .expect("ed results");
        let summary = summarize_ed(&results);
        assert_eq!(summary.total_resolved, 3);
        assert_eq!(summary.total_unresolved, 1);
        assert_eq!(summary.new_entities_created_count, 1);
        assert_eq!(summary.entities_reused_count, 2);
        let strategies: Vec<(&str, u64)> =
            summary.resolution_by_strategy.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        assert_eq!(strategies, [("rule_create", 1), ("bm25", 1), ("unknown", 1)]);
    }
}
