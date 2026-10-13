//! ContradictionStage: build (judge prompt, review store), the skip paths (no facts, offline,
//! no judge), the NLI and LLM judges, an injected judge, review rows in SQLite, facts the fact
//! stage already committed, and the `contradiction_test` scenarios run through the stage.

mod common;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use impeach_config::pipeline::{
    ContradictionCheckStageConfig, ContradictionReviewStoreConfig, JudgeModel,
};
use impeach_config::services::ContradictionCheckConfig;
use impeach_config::{MappingsRegistry, PipelineConfig, StageConfig};
use impeach_contradictions::{ContradictionJudge, JudgeVerdict};
use impeach_core::{
    AssertionTriple, ContradictionSource, ContradictionStatus, DocId, ExtractedFact, FactId,
    NliLabel, ReviewClassification, ReviewStatus, TemporalValidity,
};
use impeach_facts::FactIndexer;
use impeach_models::{MockEmbedder, MockLlm, MockNli, ModelKind, ModelRegistry, NliScores};
use impeach_pipeline::{
    ContradictionStage, PipelineContext, PipelineError, SharedEmbedder, Stage, StageBuildContext,
};
use impeach_store::{GraphStore, MemoryStore, ReviewStoreConfig, SqliteReviewStore};
use pretty_assertions::assert_eq;
use serde_json::Value;

const DIM: usize = common::DIM;

fn contradiction(confidence: f32) -> NliScores {
    let rest = (1.0 - confidence) / 2.0;
    NliScores { contradiction: confidence, entailment: rest, neutral: rest }
}

fn stage_config(config: &PipelineConfig) -> ContradictionCheckStageConfig {
    config
        .stages
        .iter()
        .find_map(|stage| match stage {
            StageConfig::ContradictionCheck(check) => Some(check.clone()),
            _ => None,
        })
        .expect("config has a ContradictionCheckStage")
}

/// The packaged config with an edited contradiction stage block.
fn config_with(edit: impl FnOnce(&mut ContradictionCheckStageConfig)) -> PipelineConfig {
    let mut config = common::default_config();
    let check = config.stages.iter_mut().find_map(|stage| match stage {
        StageConfig::ContradictionCheck(check) => Some(check),
        _ => None,
    });
    edit(check.expect("config has a ContradictionCheckStage"));
    config
}

fn speed_fact(id: &str, mph: u32) -> ExtractedFact {
    let mut fact = ExtractedFact::new(
        id,
        format!("Daniel Reed drove at {mph} miles per hour."),
        AssertionTriple::new("Daniel Reed", "DROVE_AT_SPEED", format!("{mph} mph")),
        0.9,
        TemporalValidity::default(),
    );
    fact.slot_name = Some("speed".to_owned());
    fact.value_type = Some("NUMERIC".to_owned());
    fact.property_type = Some("QUANTITATIVE".to_owned());
    fact
}

/// NLI: the 45 / 30 mph pair contradicts with `confidence`.
fn speed_nli(confidence: f32) -> Arc<MockNli> {
    Arc::new(MockNli::new().with_rule(move |premise, hypothesis| {
        let pair = |a: &str, b: &str| premise.contains(a) && hypothesis.contains(b);
        (pair("45 miles", "30 miles") || pair("30 miles", "45 miles"))
            .then(|| contradiction(confidence))
    }))
}

struct Env {
    store: Arc<MemoryStore>,
    embedder: Arc<MockEmbedder>,
    registry: Arc<ModelRegistry>,
    services: StageBuildContext,
}

impl Env {
    fn new(config: PipelineConfig) -> Self {
        Self::with_embedder(config, MockEmbedder::new(DIM))
    }

    fn with_embedder(config: PipelineConfig, embedder: MockEmbedder) -> Self {
        let store = Arc::new(MemoryStore::new(DIM));
        let registry = Arc::new(ModelRegistry::empty(config.name()));
        let dyn_store: Arc<dyn GraphStore> = store.clone();
        let services = StageBuildContext::new(Arc::new(config), Arc::clone(&registry), dyn_store);
        Self { store, embedder: Arc::new(embedder), registry, services }
    }

    fn with_nli(self, nli: Arc<MockNli>) -> Self {
        self.registry.insert_nli(self.registry.key(ModelKind::DebertaNli), nli);
        self
    }

    fn stage(&self) -> ContradictionStage {
        ContradictionStage::new(&stage_config(&self.services.config), &self.services)
            .expect("stage builds")
    }

    /// A context after an online fact stage: `facts`, the embedder, `index_persisted`.
    fn context(&self, facts: Vec<ExtractedFact>) -> PipelineContext {
        let mut ctx = PipelineContext::new(&self.services);
        ctx.data.doc_id = Some(DocId::new("doc-1"));
        ctx.data.case_id = Some("CASE-9".to_owned());
        ctx.data.embedder = Some(SharedEmbedder(self.embedder.clone()));
        ctx.data.index_persisted = true;
        ctx.data.extracted_facts = Some(facts);
        ctx
    }

    fn fact_count(&self) -> usize {
        (0..1000_u64)
            .filter_map(|id| self.store.node_snapshot(&id.to_string()))
            .filter(|(label, _)| *label == impeach_store::graph::FACT)
            .count()
    }
}

fn two_speeds() -> Vec<ExtractedFact> {
    vec![speed_fact("STATED_FACT_0", 45), speed_fact("STATED_FACT_1", 30)]
}

// --- build ---

#[test]
fn construct_rejects_other_stage_types() {
    let env = Env::new(common::default_config());
    let error = ContradictionStage::construct(&env.services.config.stages[0], &env.services)
        .err()
        .expect("RAG config is rejected");
    assert!(matches!(error, PipelineError::InvalidStageConfig { .. }), "{error}");
    let stage = ContradictionStage::construct(
        &StageConfig::ContradictionCheck(stage_config(&env.services.config)),
        &env.services,
    )
    .expect("builds");
    assert_eq!(stage.name(), "contradiction_check");
}

#[test]
fn qwen_judge_without_prompt_file_fails_the_build() {
    let config = config_with(|check| {
        let verification = check.verification.as_mut().expect("verification block");
        verification.judge_model = JudgeModel::Qwen;
        verification.qwen_prompt_file = None;
    });
    let env = Env::new(config);
    let error = ContradictionStage::new(&stage_config(&env.services.config), &env.services)
        .expect_err("missing prompt file");
    assert!(matches!(error, PipelineError::Config(_)), "{error}");
    assert!(error.to_string().contains("qwen_prompt_file"), "{error}");

    let config = config_with(|check| {
        let verification = check.verification.as_mut().expect("verification block");
        verification.judge_model = JudgeModel::Qwen;
        verification.qwen_prompt_file = Some("impeach/prompts/nope.json".to_owned());
    });
    let env = Env::new(config);
    let error = ContradictionStage::new(&stage_config(&env.services.config), &env.services)
        .expect_err("missing prompt");
    assert!(matches!(error, PipelineError::Config(_)), "{error}");
}

#[test]
fn review_store_initialisation_follows_fail_on_sqlite_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let bad_path = dir.path().join("missing_dir").join("review.sqlite3");
    let review = |fail_on_sqlite_error: bool| ContradictionReviewStoreConfig {
        db_path: bad_path.to_string_lossy().into_owned(),
        enabled: true,
        fail_on_sqlite_error,
    };

    let env = Env::new(config_with(|check| check.review_store = Some(review(true))));
    let error = ContradictionStage::new(&stage_config(&env.services.config), &env.services)
        .expect_err("unopenable database");
    assert!(matches!(error, PipelineError::Store(_)), "{error}");

    let env = Env::new(config_with(|check| check.review_store = Some(review(false))));
    assert!(!env.stage().review_store_enabled(), "writes disabled after a failed init");

    // Disabled, or enabled without a path: no review store.
    let env = Env::new(common::default_config());
    assert!(!env.stage().review_store_enabled());
    let env = Env::new(config_with(|check| {
        check.review_store =
            Some(ContradictionReviewStoreConfig { db_path: String::new(), ..review(true) });
    }));
    assert!(!env.stage().review_store_enabled());
}

// --- skip paths ---

#[test]
fn no_facts_gives_an_empty_report() {
    let env = Env::new(common::default_config()).with_nli(speed_nli(0.95));
    let mut ctx = env.context(Vec::new());
    ctx.data.extracted_facts = None;
    env.stage().run(&mut ctx).expect("runs");
    let report = ctx.data.contradiction_report.as_ref().expect("report");
    assert_eq!(report.doc_id, "doc-1");
    assert_eq!(report.total_facts_extracted, 0);
    assert_eq!(ctx.data.clean_facts, Some(Vec::new()));
    assert_eq!(ctx.data.committed_fact_ids, Some(Vec::new()));
}

#[test]
fn offline_run_reports_every_fact_clean() {
    let nli = speed_nli(0.95);
    let env = Env::new(common::default_config()).with_nli(nli.clone());
    let mut ctx = env.context(two_speeds());
    ctx.data.index_persisted = false;
    env.stage().run(&mut ctx).expect("runs");
    let report = ctx.data.contradiction_report.as_ref().expect("report");
    assert_eq!(report.total_facts_extracted, 2);
    assert_eq!(report.clean_facts, two_speeds());
    assert!(report.contradiction_results.is_empty());
    assert_eq!(ctx.data.clean_facts, Some(two_speeds()));
    assert_eq!(ctx.data.committed_fact_ids, Some(Vec::new()));
    assert!(nli.calls().is_empty());
    assert_eq!(env.fact_count(), 0);

    // No embedder at all behaves the same.
    let mut ctx = env.context(two_speeds());
    ctx.data.embedder = None;
    env.stage().run(&mut ctx).expect("runs");
    assert_eq!(ctx.data.clean_facts, Some(two_speeds()));
}

#[test]
fn missing_nli_model_reports_every_fact_clean() {
    let env = Env::new(common::default_config());
    let mut ctx = env.context(two_speeds());
    env.stage().run(&mut ctx).expect("runs");
    let report = ctx.data.contradiction_report.as_ref().expect("report");
    assert_eq!(report.clean_facts.len(), 2);
    assert!(report.debug_trace.is_empty());
    // The all-clean report keeps the Python default thresholds.
    assert!((report.contradiction_threshold - 0.85).abs() < f64::EPSILON);
    assert_eq!(ctx.data.committed_fact_ids, Some(Vec::new()));
    assert_eq!(env.fact_count(), 0, "nothing is committed without a check");
}

// --- judges ---

#[test]
fn nli_judge_finds_an_intra_document_contradiction() {
    let nli = speed_nli(0.95);
    let env = Env::new(common::default_config()).with_nli(nli.clone());
    let mut ctx = env.context(two_speeds());
    env.stage().run(&mut ctx).expect("runs");
    let report = ctx.data.contradiction_report.as_ref().expect("report");
    // processing_v17 thresholds.
    assert!((report.contradiction_threshold - 0.80).abs() < f64::EPSILON);
    assert_eq!(report.contradiction_count(), 1);
    let pair = &report.unique_contradiction_pairs[0];
    assert_eq!(pair.contradiction_source, ContradictionSource::IntraDocument);
    assert_eq!(ctx.data.clean_facts, Some(Vec::new()));
    assert_eq!(ctx.data.committed_fact_ids, Some(Vec::new()));
    assert_eq!(nli.calls().len(), 1);
}

#[test]
fn clean_facts_are_committed_with_their_graph_ids() {
    let env = Env::new(common::default_config()).with_nli(Arc::new(MockNli::new()));
    let mut ctx = env.context(two_speeds());
    env.stage().run(&mut ctx).expect("runs");
    let committed = ctx.data.committed_fact_ids.clone().expect("committed ids");
    assert_eq!(committed.len(), 2);
    assert_eq!(env.fact_count(), 2);
    let clean = ctx.data.clean_facts.as_ref().expect("clean facts");
    for (fact, id) in clean.iter().zip(&committed) {
        assert_eq!(fact.metadata["helix_fact_id"], Value::String(id.to_string()));
        assert_eq!(
            env.embedder.vector_for(&fact.text),
            env.store.embedding(id.as_str()).expect("vector")
        );
    }
}

#[test]
fn qwen_judge_uses_the_registry_llm_and_prompt() {
    let config = config_with(|check| {
        check.verification.as_mut().expect("verification").judge_model = JudgeModel::Qwen;
    });
    let llm = Arc::new(MockLlm::with_responder(|request| {
        let verdict = if request.prompt.contains("45 miles") && request.prompt.contains("30 miles")
        {
            r#"{"verdict": "contradiction", "confidence": 0.9, "reasoning": "different speeds"}"#
        } else {
            r#"{"verdict": "neutral", "confidence": 0.8}"#
        };
        Ok(verdict.to_owned())
    }));
    let nli = speed_nli(0.95);
    let env = Env::new(config).with_nli(nli.clone());
    env.registry.insert_llm(env.registry.key(ModelKind::Ollama), llm.clone());
    let mut ctx = env.context(two_speeds());
    env.stage().run(&mut ctx).expect("runs");

    let report = ctx.data.contradiction_report.as_ref().expect("report");
    assert_eq!(report.contradiction_count(), 1);
    assert!((report.unique_contradiction_pairs[0].nli_confidence - 0.9).abs() < 1e-9);
    let prompts = llm.prompts();
    assert_eq!(prompts.len(), 1);
    assert!(prompts[0].contains("legal fact contradiction detector"), "{}", prompts[0]);
    assert!(prompts[0].contains("Fact A: "), "{}", prompts[0]);
    assert!(nli.calls().is_empty(), "the NLI model is not used in qwen mode");
}

#[test]
fn qwen_judge_falls_back_to_nli_without_an_llm() {
    let config = config_with(|check| {
        check.verification.as_mut().expect("verification").judge_model = JudgeModel::Qwen;
    });
    let nli = speed_nli(0.95);
    let env = Env::new(config).with_nli(nli.clone());
    let mut ctx = env.context(two_speeds());
    env.stage().run(&mut ctx).expect("runs");
    assert_eq!(nli.calls().len(), 1);
    assert_eq!(ctx.data.contradiction_report.as_ref().expect("report").contradiction_count(), 1);

    // Neither model: every fact is clean.
    let config = config_with(|check| {
        check.verification.as_mut().expect("verification").judge_model = JudgeModel::Qwen;
    });
    let env = Env::new(config);
    let mut ctx = env.context(two_speeds());
    env.stage().run(&mut ctx).expect("runs");
    assert_eq!(ctx.data.clean_facts.as_ref().expect("clean").len(), 2);
}

/// A judge calling every pair a contradiction (0.99) and counting its calls.
#[derive(Debug, Default)]
struct AlwaysContradiction {
    pairs: Mutex<usize>,
}

impl ContradictionJudge for AlwaysContradiction {
    fn name(&self) -> &str {
        "always"
    }

    fn judge_batch(&self, pairs: &[(&str, &str)]) -> Vec<JudgeVerdict> {
        *self.pairs.lock().expect("lock") += pairs.len();
        pairs
            .iter()
            .map(|_| JudgeVerdict {
                label: NliLabel::Contradiction,
                confidence: 0.99,
                reasoning: None,
            })
            .collect()
    }
}

#[test]
fn injected_judge_replaces_the_registry_models() {
    let nli = speed_nli(0.1);
    let env = Env::new(common::default_config()).with_nli(nli.clone());
    let judge = Arc::new(AlwaysContradiction::default());
    let stage = env.stage().with_judge(judge.clone());
    let mut ctx = env.context(two_speeds());
    stage.run(&mut ctx).expect("runs");
    assert_eq!(*judge.pairs.lock().expect("lock"), 1);
    assert!(nli.calls().is_empty());
    assert_eq!(ctx.data.contradiction_report.as_ref().expect("report").contradiction_count(), 1);
}

#[test]
fn judge_is_resolved_once_per_stage() {
    let nli = speed_nli(0.95);
    let env = Env::new(common::default_config()).with_nli(nli);
    let stage = env.stage();
    stage.run(&mut env.context(two_speeds())).expect("first run");
    // Replacing the registry model does not change the cached judge.
    let other = speed_nli(0.1);
    env.registry.insert_nli(env.registry.key(ModelKind::DebertaNli), other.clone());
    let mut ctx = env.context(two_speeds());
    stage.run(&mut ctx).expect("second run");
    assert!(other.calls().is_empty());
    assert_eq!(ctx.data.contradiction_report.as_ref().expect("report").contradiction_count(), 1);
}

// --- review store ---

#[test]
fn review_rows_are_written_for_contradictions() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("review.sqlite3");
    let config = config_with(|check| {
        check.review_store = Some(ContradictionReviewStoreConfig {
            db_path: db_path.to_string_lossy().into_owned(),
            enabled: true,
            fail_on_sqlite_error: true,
        });
    });
    let env = Env::new(config).with_nli(speed_nli(0.7));
    let stage = env.stage();
    assert!(stage.review_store_enabled());
    let mut ctx = env.context(two_speeds());
    stage.run(&mut ctx).expect("runs");

    let store = SqliteReviewStore::new(&ReviewStoreConfig::new(&db_path));
    let rows = store.get_by_status(ReviewStatus::Pending, None).expect("rows");
    assert_eq!(rows.len(), 1);
    let row = &rows[0];
    assert_eq!(row.case_id, "CASE-9");
    assert_eq!(row.session_id, "doc-1");
    assert_eq!(row.segment_id, "doc-1");
    assert_eq!(row.segment_index, 0);
    assert_eq!(row.doc_id, "doc-1");
    assert_eq!(row.stage, "PROCESSING");
    assert_eq!(row.source_stream.as_deref(), Some("processing_pipeline"));
    assert_eq!(row.source_pipeline.as_deref(), Some("processing_v17"));
    assert_eq!(row.classification, ReviewClassification::NeedsReview);
    assert_eq!(row.contradiction_source, ContradictionSource::IntraDocument);
    assert_eq!(row.new_fact_id, "STATED_FACT_0");
    assert_eq!(row.existing_fact_id, "STATED_FACT_1");
    assert_eq!(row.nli_label, Some(NliLabel::Contradiction));
    assert!(row.dedupe_key.is_some());

    // A second run of the same document inserts nothing new (dedupe key).
    stage.run(&mut env.context(two_speeds())).expect("runs again");
    assert_eq!(store.get_by_status(ReviewStatus::Pending, None).expect("rows").len(), 1);

    // Without a contradiction nothing is written; the case id falls back to the doc id.
    let env2 = Env::new(config_with(|check| {
        check.review_store = Some(ContradictionReviewStoreConfig {
            db_path: dir.path().join("clean.sqlite3").to_string_lossy().into_owned(),
            enabled: true,
            fail_on_sqlite_error: true,
        });
    }))
    .with_nli(Arc::new(MockNli::new()));
    env2.stage().run(&mut env2.context(two_speeds())).expect("runs");
    let clean = SqliteReviewStore::new(&ReviewStoreConfig::new(dir.path().join("clean.sqlite3")));
    assert!(clean.get_by_status(ReviewStatus::Pending, None).expect("rows").is_empty());
}

#[test]
fn review_case_id_falls_back_to_the_doc_id() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("review.sqlite3");
    let env = Env::new(config_with(|check| {
        check.review_store = Some(ContradictionReviewStoreConfig {
            db_path: db_path.to_string_lossy().into_owned(),
            enabled: true,
            fail_on_sqlite_error: true,
        });
    }))
    .with_nli(speed_nli(0.95));
    let mut ctx = env.context(two_speeds());
    ctx.data.case_id = None;
    env.stage().run(&mut ctx).expect("runs");
    let store = SqliteReviewStore::new(&ReviewStoreConfig::new(&db_path));
    let rows = store.get_by_status(ReviewStatus::Pending, None).expect("rows");
    assert_eq!(rows[0].case_id, "doc-1");
    assert_eq!(rows[0].classification, ReviewClassification::Contradiction);
}

// --- facts committed by the fact stage ---

#[test]
fn facts_already_committed_are_excluded_and_not_written_twice() {
    let env = Env::new(common::default_config())
        .with_nli(Arc::new(MockNli::new().with_default(contradiction(0.99))));
    // The fact stage committed the facts (commit_facts: true).
    let mut facts = vec![speed_fact("STATED_FACT_0", 45)];
    let store: Arc<dyn GraphStore> = env.store.clone();
    let mappings = MappingsRegistry::load().expect("mappings");
    let result = FactIndexer::new(store, env.embedder.clone())
        .commit_facts(&mut facts, &mappings.taxonomy)
        .expect("commit");
    let mut ctx = env.context(facts);
    ctx.data.fact_helix_ids = Some(result.fact_ids.clone());

    env.stage().run(&mut ctx).expect("runs");
    let report = ctx.data.contradiction_report.as_ref().expect("report");
    assert!(report.contradiction_results.is_empty(), "the fact is not checked against itself");
    let ids: Vec<FactId> = result.fact_ids.into_iter().flatten().collect();
    assert_eq!(ctx.data.committed_fact_ids, Some(ids));
    assert_eq!(env.fact_count(), 1);
}

// --- contradiction_test scenarios ---

fn scenario(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/input/contradiction_test")
        .join(format!("scenario_{name}.json"));
    serde_json::from_str(&std::fs::read_to_string(path).expect("scenario reads"))
        .expect("scenario JSON")
}

fn text_fact(entry: &Value) -> ExtractedFact {
    ExtractedFact::new(
        entry["id"].as_str().expect("id"),
        entry["text"].as_str().expect("text"),
        AssertionTriple::new("", "", ""),
        0.9,
        TemporalValidity::default(),
    )
}

/// Unit vector along axis `axis`.
fn axis(axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; DIM];
    vector[axis] = 1.0;
    vector
}

/// Runs a scenario through the stage: ground truth committed first, then the test facts as one
/// document. A truth and the test fact it pairs with (by `expected_contradiction_with`, else
/// by position) share an embedding; the NLI scores each such pair by the test fact's expected
/// status (contradiction 0.95, needs_review 0.7 between the processing_v17 thresholds,
/// clean: entailment) and every other pair as neutral. Returns (id, status) per test fact.
fn run_scenario(name: &str) -> Vec<(String, String, String)> {
    let scenario = scenario(name);
    let truths = scenario["ground_truth"].as_array().expect("ground_truth");
    let tests = scenario["test_facts"].as_array().expect("test_facts");

    let partner = |index: usize, test: &Value| -> Option<&Value> {
        let id = test["id"].as_str()?;
        truths
            .iter()
            .find(|t| t["expected_contradiction_with"].as_str() == Some(id))
            .or_else(|| truths.get(index).filter(|t| t["expected_contradiction_with"].is_null()))
    };
    let mut embedder = MockEmbedder::new(DIM);
    let mut scripted: Vec<(String, String, String)> = Vec::new();
    for (index, test) in tests.iter().enumerate() {
        let test_text = test["text"].as_str().expect("text").to_owned();
        let vector = axis(index);
        embedder = embedder.with_vector(test_text.clone(), vector.clone());
        if let Some(truth) = partner(index, test) {
            let truth_text = truth["text"].as_str().expect("text").to_owned();
            embedder = embedder.with_vector(truth_text.clone(), vector);
            let status = test["expected_status"].as_str().expect("status").to_owned();
            scripted.push((truth_text, test_text, status));
        }
    }
    for (index, truth) in truths.iter().enumerate() {
        let text = truth["text"].as_str().expect("text");
        if !scripted.iter().any(|(t, _, _)| t == text) {
            embedder = embedder.with_vector(text, axis(tests.len() + index));
        }
    }
    let nli = MockNli::new().with_rule(move |premise, hypothesis| {
        scripted.iter().find_map(|(truth, test, status)| {
            (premise.contains(truth.as_str()) && hypothesis.contains(test.as_str())).then(|| {
                match status.as_str() {
                    "contradiction" => contradiction(0.95),
                    "needs_review" => contradiction(0.7),
                    _ => NliScores { contradiction: 0.02, entailment: 0.9, neutral: 0.08 },
                }
            })
        })
    });

    let env = Env::with_embedder(common::default_config(), embedder).with_nli(Arc::new(nli));
    let mut ground_truth: Vec<ExtractedFact> = truths.iter().map(text_fact).collect();
    let store: Arc<dyn GraphStore> = env.store.clone();
    let mappings = MappingsRegistry::load().expect("mappings");
    FactIndexer::new(store, env.embedder.clone())
        .commit_facts(&mut ground_truth, &mappings.taxonomy)
        .expect("commit ground truth");

    let test_facts: Vec<ExtractedFact> = tests.iter().map(text_fact).collect();
    let mut ctx = env.context(test_facts);
    ctx.data.doc_id = Some(DocId::new(scenario["doc_id"].as_str().expect("doc_id")));
    env.stage().run(&mut ctx).expect("stage runs");
    let report = ctx.data.contradiction_report.expect("report");
    assert_eq!(report.doc_id, scenario["doc_id"].as_str().expect("doc_id"));

    tests
        .iter()
        .map(|test| {
            let id = test["id"].as_str().expect("id");
            let status = report
                .contradiction_results
                .iter()
                .find(|result| result.new_fact.id == id)
                .map_or_else(
                    || {
                        assert!(report.clean_facts.iter().any(|f| f.id == id), "{id} is clean");
                        "clean".to_owned()
                    },
                    |result| result.status.as_str().to_owned(),
                );
            (id.to_owned(), test["expected_status"].as_str().expect("status").to_owned(), status)
        })
        .collect()
}

fn assert_scenario(name: &str) {
    let results = run_scenario(name);
    assert!(!results.is_empty());
    for (id, expected, actual) in &results {
        assert_eq!(actual, expected, "{name}: {id}");
    }
}

#[test]
fn scenario_entailment() {
    assert_scenario("entailment");
}

#[test]
fn scenario_negation() {
    assert_scenario("negation");
}

#[test]
fn scenario_quantitative() {
    assert_scenario("quantitative");
}

#[test]
fn scenario_temporal() {
    assert_scenario("temporal");
    let statuses: Vec<String> = run_scenario("temporal").into_iter().map(|r| r.2).collect();
    assert!(statuses.contains(&ContradictionStatus::NeedsReview.as_str().to_owned()));
}

#[test]
fn scenario_unrelated() {
    assert_scenario("unrelated");
}

#[test]
fn service_config_comes_from_the_stage_block() {
    let config = common::default_config();
    let check = ContradictionCheckConfig::from_stage(&stage_config(&config));
    assert!((check.contradiction_threshold - 0.80).abs() < f64::EPSILON);
    assert!(check.pre_filter.enable_text_overlap_fallback);
}
