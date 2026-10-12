//! `run_processing` (port of `tests/unit/test_api_processing.py`) and the RAG-only flow.

mod common;
mod ed_common;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ce_config::StageConfig;
use ce_core::{ContradictionReport, DocumentPayload, EdResult, ExtractedFact};
use ce_ingest::IngestError;
use ce_models::{MockEmbedder, ModelKind, ModelRegistry};
use ce_pipeline::{
    load_processing_config, run_processing, run_processing_with, PipelineContext, PipelineError,
    ProcessingEnv, ProcessingOptions, ProcessingResult, RagIndexStage, Stage, StageBuildContext,
    StageRegistry,
};
use ce_store::{GraphStore, MemoryStore};
use common::{fixture, DIM};
use pretty_assertions::assert_eq;
use serde_json::json;

fn fact() -> ExtractedFact {
    serde_json::from_value(json!({
        "id": "STATED_FACT_0",
        "text": "A REL B",
        "assertion_triple": {"subject": "A", "relationship": "REL", "object": "B"},
        "certainty": 0.9,
        "temporal_validity": {}
    }))
    .expect("fact")
}

fn report() -> ContradictionReport {
    let pair = json!({
        "new_fact_text": "A REL B",
        "new_fact_id": "STATED_FACT_0",
        "existing_fact_id": "EXISTING_FACT_1",
        "existing_fact_text": "Existing fact text",
        "nli_label": "contradiction",
        "nli_confidence": 0.95
    });
    serde_json::from_value(json!({
        "doc_id": "doc-1",
        "total_facts_extracted": 1,
        "clean_facts": [serde_json::to_value(fact()).expect("fact json")],
        "committed_fact_ids": ["EXISTING_FACT_1"],
        "contradiction_results": [{
            "new_fact": serde_json::to_value(fact()).expect("fact json"),
            "status": "contradiction",
            "contradicting_pairs": [pair.clone()]
        }],
        "unique_contradiction_pairs": [pair]
    }))
    .expect("report")
}

/// What the dummy stage saw: the document, `case_id` and `doc_metadata`.
type Seen = Arc<Mutex<Option<(DocumentPayload, Option<String>, Option<ce_core::Metadata>)>>>;

/// Populates the context keys of a full processing run and records what it saw.
#[derive(Debug, Default)]
struct DummyProcessingStage {
    seen: Seen,
}

impl Stage for DummyProcessingStage {
    fn name(&self) -> &str {
        "dummy"
    }

    fn run(&self, ctx: &mut PipelineContext) -> ce_pipeline::Result<()> {
        let document = ctx.data.document.clone().expect("runner sets the document");
        *self.seen.lock().expect("lock") =
            Some((document, ctx.data.case_id.clone(), ctx.data.doc_metadata.clone()));
        let ed: EdResult = serde_json::from_value(json!({
            "chosen_entity_id": "entity-1",
            "candidates": [],
            "resolved_by_llm": false,
            "confidence": 0.9,
            "strategy_used": "rule"
        }))
        .expect("ed result");
        ctx.data.contradiction_report = Some(report());
        ctx.data.clean_facts = Some(vec![fact()]);
        ctx.data.committed_fact_ids = Some(vec!["EXISTING_FACT_1".into()]);
        ctx.data.extracted_facts = Some(vec![fact()]);
        ctx.data.ed_results = Some(vec![ed]);
        Ok(())
    }
}

/// A stage that does nothing.
#[derive(Debug)]
struct NoopStage;

impl Stage for NoopStage {
    fn name(&self) -> &str {
        "noop"
    }

    fn run(&self, _: &mut PipelineContext) -> ce_pipeline::Result<()> {
        Ok(())
    }
}

/// An env whose only stage is the dummy (registered for the first config stage type).
fn dummy_env() -> (ProcessingEnv, Seen) {
    let seen: Seen = Arc::default();
    let shared = Arc::clone(&seen);
    let stages = StageRegistry::empty().with(
        "RAGIndexStage",
        move |_: &StageConfig, _: &StageBuildContext| {
            Ok(Box::new(DummyProcessingStage { seen: Arc::clone(&shared) }) as Box<dyn Stage>)
        },
    );
    let env = ProcessingEnv {
        registry: Some(Arc::new(ModelRegistry::empty("processing_v17"))),
        store: Some(Arc::new(MemoryStore::new(DIM))),
        stages: Some(stages),
        unregistered: None,
    };
    (env, seen)
}

fn mock_env() -> (ProcessingEnv, Arc<MemoryStore>) {
    let registry = ModelRegistry::empty("processing_v17");
    registry.insert_embedder(registry.key(ModelKind::Bge), Arc::new(MockEmbedder::new(DIM)));
    let store = Arc::new(MemoryStore::new(DIM));
    let dyn_store: Arc<dyn GraphStore> = store.clone();
    let env = ProcessingEnv {
        registry: Some(Arc::new(registry)),
        store: Some(dyn_store),
        stages: Some(rag_index_only()),
        unregistered: None,
    };
    (env, store)
}

/// A stage registry with only `RAGIndexStage`; later stage types are skipped.
fn rag_index_only() -> StageRegistry {
    StageRegistry::empty().with(RagIndexStage::TYPE_NAME, RagIndexStage::construct)
}

fn create_test_jsonl(dir: &Path) -> PathBuf {
    let path = dir.join("test_doc.jsonl");
    let record = json!({
        "doc_id": "test-doc-1",
        "text": "This is test document text for processing.",
        "doc_type": "complaint",
        "title": "Test Document",
        "file_path": "original/path.pdf",
        "total_pages": 1,
        "date_created": "2025-11-20T14:32:01.123456",
        "metadata": {"case_id": "CASE-123", "db_id": "DB-456"},
        "helix_id": null,
    });
    std::fs::write(&path, format!("{record}\n")).expect("write jsonl");
    path
}

#[test]
fn run_processing_populates_processing_result() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (env, _) = dummy_env();
    let run =
        run_processing_with(create_test_jsonl(dir.path()), &ProcessingOptions::default(), env)
            .expect("runs");
    let result = run.result;
    let report = result.contradiction_report.as_ref().expect("report");
    assert_eq!(report.total_facts_extracted, 1);
    assert_eq!(result.clean_facts.len(), 1);
    assert_eq!(result.committed_fact_ids.len(), 1);
    assert_eq!(result.extracted_facts.len(), 1);
    assert_eq!(result.ed_results.len(), 1);
    assert_eq!(result.contradiction_count, 1);
    assert_eq!(result.needs_review_count, 0);
    assert_eq!(result.sqlite_db_path, None);

    let value = result.to_json().expect("json");
    assert!(value["contradiction_report"].is_object());
    assert!(value["clean_facts"].is_array());
    let back: ProcessingResult = serde_json::from_value(value).expect("round trip");
    assert_eq!(back, result);
}

#[test]
fn run_processing_preserves_document_identity() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (env, seen) = dummy_env();
    run_processing_with(create_test_jsonl(dir.path()), &ProcessingOptions::default(), env)
        .expect("runs");
    let (doc, case_id, doc_metadata) = seen.lock().expect("lock").take().expect("stage ran");
    assert_eq!(doc.doc_id.as_str(), "test-doc-1");
    assert_eq!(doc.text, "This is test document text for processing.");
    assert_eq!(doc.title, "Test Document");
    assert_eq!(doc.metadata["case_id"], json!("CASE-123"));
    assert_eq!(doc.metadata["db_id"], json!("DB-456"));
    assert_eq!(case_id.as_deref(), Some("CASE-123"));
    assert_eq!(
        doc_metadata.map(serde_json::Value::Object),
        Some(json!({"case_id": "CASE-123", "db_id": "DB-456"}))
    );
}

#[test]
fn run_processing_accepts_top_level_identifiers() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("minimal.jsonl");
    std::fs::write(
        &path,
        "{\"case_id\": \"mason_v_billy\", \"db_id\": \"depo_1\", \"doc_id\": \"002\", \"text\": \"Mason was here.\"}\n",
    )
    .expect("write");
    let (env, seen) = dummy_env();
    run_processing_with(&path, &ProcessingOptions::default(), env).expect("runs");
    let (doc, case_id, _) = seen.lock().expect("lock").take().expect("stage ran");
    assert_eq!(doc.doc_id.as_str(), "002");
    assert_eq!(doc.metadata["case_id"], json!("mason_v_billy"));
    assert_eq!(doc.metadata["db_id"], json!("depo_1"));
    assert_eq!(case_id.as_deref(), Some("mason_v_billy"));
}

#[test]
fn run_processing_without_identifiers_seeds_nothing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("bare.jsonl");
    std::fs::write(&path, "{\"text\": \"Bare.\"}\n").expect("write");
    let (env, seen) = dummy_env();
    run_processing_with(&path, &ProcessingOptions::default(), env).expect("runs");
    let (doc, case_id, doc_metadata) = seen.lock().expect("lock").take().expect("stage ran");
    assert_eq!(doc.doc_id.as_str(), "bare");
    assert_eq!(case_id, None);
    assert_eq!(doc_metadata, None);
}

#[test]
fn run_processing_db_path_enables_review_store() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("review.sqlite3");
    let options =
        ProcessingOptions { db_path: Some(db_path.clone()), ..ProcessingOptions::default() };

    let config = load_processing_config(&options).expect("config");
    let review_store = config
        .stages
        .iter()
        .find_map(|stage| match stage {
            StageConfig::ContradictionCheck(check) => check.review_store.clone(),
            _ => None,
        })
        .expect("default config has a ContradictionCheckStage with a review store");
    assert!(review_store.enabled);
    assert!(review_store.fail_on_sqlite_error);
    assert_eq!(review_store.db_path, db_path.to_string_lossy());

    let options_lenient = ProcessingOptions { fail_on_sqlite_error: false, ..options.clone() };
    let lenient = load_processing_config(&options_lenient).expect("config");
    assert!(lenient.stages.iter().any(|stage| matches!(
        stage,
        StageConfig::ContradictionCheck(check)
            if check.review_store.as_ref().is_some_and(|r| !r.fail_on_sqlite_error)
    )));

    // Only a built ContradictionCheckStage writes the review store.
    let (env, _) = dummy_env();
    let result = run_processing_with(create_test_jsonl(dir.path()), &options, env).expect("runs");
    assert_eq!(result.result.sqlite_db_path, None);

    let (mut env, _) = dummy_env();
    env.stages = env.stages.map(|stages| {
        stages.with("ContradictionCheckStage", |_: &StageConfig, _: &StageBuildContext| {
            Ok(Box::new(NoopStage) as Box<dyn Stage>)
        })
    });
    let result = run_processing_with(create_test_jsonl(dir.path()), &options, env).expect("runs");
    assert_eq!(result.stages, ["dummy", "noop"]);
    assert_eq!(result.result.sqlite_db_path, Some(db_path.to_string_lossy().into_owned()));
}

#[test]
fn db_path_is_not_reported_when_only_rag_index_runs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let options = ProcessingOptions {
        db_path: Some(dir.path().join("review.sqlite3")),
        ..ProcessingOptions::default()
    };
    let (env, _) = mock_env();
    let run = run_processing_with(fixture("Mason_billy.jsonl"), &options, env).expect("runs");
    assert_eq!(run.stages, ["rag_index"]);
    assert_eq!(run.result.sqlite_db_path, None);
    assert!(!dir.path().join("review.sqlite3").exists());
}

#[test]
fn run_processing_raises_for_missing_jsonl() {
    let dir = tempfile::tempdir().expect("tempdir");
    let error =
        run_processing(dir.path().join("does_not_exist.jsonl"), &ProcessingOptions::default())
            .expect_err("missing");
    assert!(matches!(error, PipelineError::Ingest(IngestError::NotFound(_))), "{error}");
}

#[test]
fn run_processing_raises_for_empty_jsonl() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("empty.jsonl");
    std::fs::write(&path, "").expect("write");
    let error = run_processing(&path, &ProcessingOptions::default()).expect_err("empty");
    assert!(matches!(error, PipelineError::Ingest(IngestError::EmptyJsonl(_))), "{error}");
    assert!(error.to_string().contains("empty"), "{error}");
}

#[test]
fn run_processing_raises_for_invalid_json() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("invalid.jsonl");
    std::fs::write(&path, "not valid json {").expect("write");
    let error = run_processing(&path, &ProcessingOptions::default()).expect_err("invalid");
    assert!(matches!(error, PipelineError::Ingest(IngestError::Json { .. })), "{error}");
}

#[test]
fn run_processing_raises_for_empty_path() {
    let error = run_processing("", &ProcessingOptions::default()).expect_err("empty path");
    assert!(matches!(error, PipelineError::Ingest(IngestError::EmptyPath)), "{error}");
}

#[test]
fn run_processing_reports_bad_config_paths() {
    let dir = tempfile::tempdir().expect("tempdir");
    let options = ProcessingOptions {
        config_path: Some(dir.path().join("nope.yaml")),
        ..ProcessingOptions::default()
    };
    let (env, _) = dummy_env();
    let error =
        run_processing_with(create_test_jsonl(dir.path()), &options, env).expect_err("config");
    assert!(matches!(error, PipelineError::Config(_)), "{error}");
}

#[test]
fn run_processing_indexes_fixtures_with_mocks() {
    for path in common::jsonl_fixtures() {
        let (env, store) = mock_env();
        let run = run_processing_with(&path, &ProcessingOptions::default(), env).expect("runs");
        assert_eq!(run.stages, ["rag_index"]);
        assert_eq!(run.report.stage_timings.len(), 1);
        let indexed = run.data.rag_index_result.as_ref().expect("indexed");
        assert_eq!(indexed.embeddings.count, indexed.chunks.len());
        assert_eq!(store.node_count(), 1 + indexed.chunks.len() + indexed.spans.len());
        assert!(run.data.doc_metadata.is_some());
        assert!(run.data.case_id.is_some());
        // Later stages are not ported yet: the result is empty.
        let result = run.result;
        assert_eq!(result.contradiction_report, None);
        assert!(result.extracted_facts.is_empty() && result.clean_facts.is_empty());
        assert_eq!((result.contradiction_count, result.needs_review_count), (0, 0));
    }
    let (env, _) = mock_env();
    let run = run_processing_with(fixture("Mason_billy.jsonl"), &ProcessingOptions::default(), env)
        .expect("runs");
    assert_eq!(run.data.case_id.as_deref(), Some("mason_v_billy"));
    let document = run.data.document.expect("document");
    assert!(document.file_path.is_some_and(|p| p.ends_with("Mason_billy.jsonl")));
}

#[cfg(not(feature = "helix"))]
#[test]
fn open_store_without_helix_is_in_memory() {
    let config = common::default_config();
    let store = ce_pipeline::open_store(&config.helix).expect("store");
    assert_eq!(store.embedding_dim(), config.helix.embedding_dim);
}

#[cfg(feature = "helix")]
#[test]
fn open_store_with_helix_memory_mode() {
    let mut config = common::default_config();
    config.helix.mode = ce_config::pipeline::HelixMode::Memory;
    config.helix.namespace = "ce_pipeline_api_test".into();
    let store = ce_pipeline::open_store(&config.helix).expect("store");
    assert_eq!(store.embedding_dim(), config.helix.embedding_dim);

    // The built-in stages against helix memory mode: mock embedder, a gazetteer NER model of
    // the Green Valley names and rule coreference.
    let registry = ModelRegistry::empty("processing_v17");
    registry.insert_embedder(registry.key(ModelKind::Bge), Arc::new(MockEmbedder::new(DIM)));
    let ner = ed_common::GazetteerNer::new(ed_common::GREEN_VALLEY_NAMES);
    registry.insert_ner(registry.key(ModelKind::Gliner2), Arc::new(ner));
    registry.insert_coref(registry.key(ModelKind::Coref), Arc::new(ce_models::RuleCoref::new()));
    let env = ProcessingEnv {
        registry: Some(Arc::new(registry)),
        store: Some(Arc::clone(&store)),
        stages: None,
        unregistered: None,
    };
    let run = run_processing_with(
        fixture("lopez_v_green_valley_complaint.jsonl"),
        &ProcessingOptions::default(),
        env,
    )
    .expect("runs");
    assert_eq!(
        run.stages,
        [
            "rag_index",
            "mention_detection",
            "definition_extraction",
            "coreference",
            "ed_resolution",
            "relation_extraction",
            "fact_extraction"
        ]
    );
    // No LLM is registered: the fact stage skips without writing anything.
    assert!(run.data.extracted_facts.is_none());
    let indexed = run.data.rag_index_result.as_ref().expect("indexed");
    assert_eq!(indexed.chunks.len(), 6);
    assert_eq!(indexed.embeddings.count, 6);
    let chunks = store.get_document_chunks(&indexed.document.helix_id).expect("chunks");
    assert_eq!(chunks.len(), 6);

    let mentions = run.data.doc_mentions.as_ref().expect("mentions");
    let span_ids = run.data.mention_to_span_id.as_ref().expect("mention spans written");
    assert!(!span_ids.is_empty());
    assert!(span_ids.len() <= mentions.len());

    // ED against helix: mentions resolved to created entities.
    let resolved = run.data.resolved_mentions.as_ref().expect("resolved_mentions");
    assert!(!resolved.is_empty());
    assert!(!run.data.ed_results.as_ref().expect("ed_results").is_empty());
    assert!(!store.get_all_entities(1000).expect("entities").is_empty());
    assert_eq!(run.data.extracted_relations.as_deref(), Some(&[][..]));
}
