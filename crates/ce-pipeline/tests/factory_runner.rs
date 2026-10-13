//! Stage registry, pipeline construction and the runner's inputs.

mod common;

use std::sync::{Arc, Mutex};

use ce_config::StageConfig;
use ce_core::DocumentPayload;
use ce_pipeline::{
    build_pipeline, Pipeline, PipelineContext, PipelineError, PipelineInput, RagIndexStage, Stage,
    StageBuildContext, StageRegistry, UnregisteredStages,
};
use common::{config_with, default_config, fixture, Harness};
use pretty_assertions::assert_eq;
use serde_json::json;

/// Records the document each run sees and optionally fails.
#[derive(Debug)]
struct Probe {
    name: String,
    seen: Arc<Mutex<Vec<DocumentPayload>>>,
    fail: bool,
}

impl Stage for Probe {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, ctx: &mut PipelineContext) -> ce_pipeline::Result<()> {
        if let Some(document) = &ctx.data.document {
            self.seen.lock().expect("lock").push(document.clone());
        }
        if self.fail {
            return Err(PipelineError::MissingContext {
                stage: self.name.clone(),
                key: "doc_text",
            });
        }
        ctx.data.case_id = Some(format!("seen-by-{}", self.name));
        Ok(())
    }
}

fn probe(name: &str, fail: bool) -> (Probe, Arc<Mutex<Vec<DocumentPayload>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    (Probe { name: name.to_owned(), seen: Arc::clone(&seen), fail }, seen)
}

const BUILTIN_TYPES: [&str; 8] = [
    "RAGIndexStage",
    "MentionStage",
    "DefinitionStage",
    "CoreferenceStage",
    "EDStage",
    "RelationStage",
    "FactStage",
    "ContradictionCheckStage",
];

#[test]
fn builtin_registry_builds_every_stage() {
    let harness = Harness::new(default_config());
    let pipeline = build_pipeline(
        Arc::clone(&harness.services.config),
        Arc::clone(&harness.services.registry),
        Arc::clone(&harness.services.store),
    )
    .expect("builds");
    assert_eq!(
        pipeline.stage_names(),
        [
            "rag_index",
            "mention_detection",
            "definition_extraction",
            "coreference",
            "ed_resolution",
            "relation_extraction",
            "fact_extraction",
            "contradiction_check"
        ]
    );
    assert_eq!(StageRegistry::builtin().type_names(), BUILTIN_TYPES);
}

#[test]
fn unregistered_stage_types_error_with_the_known_types() {
    let harness = Harness::new(default_config());
    let registry = StageRegistry::empty()
        .with(RagIndexStage::TYPE_NAME, RagIndexStage::construct)
        .with("MentionStage", ce_pipeline::MentionStage::construct);
    let error = registry
        .build(harness.services.clone(), UnregisteredStages::Error)
        .expect_err("DefinitionStage has no constructor");
    match error {
        PipelineError::UnknownStage { stage_type, known } => {
            assert_eq!(stage_type, "DefinitionStage");
            assert_eq!(known, ["RAGIndexStage", "MentionStage"]);
        }
        other => panic!("unexpected error {other}"),
    }
    StageRegistry::builtin()
        .build(harness.services.clone(), UnregisteredStages::Error)
        .expect("every stage type is registered");
    let message = StageRegistry::empty()
        .build_stage(&harness.services.config.stages[0], &harness.services)
        .err()
        .map(|e| e.to_string())
        .expect("no constructors");
    assert_eq!(message, "unknown stage type `RAGIndexStage`; registered stage types: []");
}

#[test]
fn disabled_stages_are_not_constructed() {
    let harness = Harness::new(config_with(|rag| rag.enabled = false, |_| {}));
    let built = Arc::new(Mutex::new(0));
    let counter = Arc::clone(&built);
    let registry = StageRegistry::empty().with(RagIndexStage::TYPE_NAME, move |stage, services| {
        *counter.lock().expect("lock") += 1;
        RagIndexStage::construct(stage, services)
    });
    let pipeline =
        registry.build(harness.services.clone(), UnregisteredStages::Skip).expect("builds");
    assert!(pipeline.stage_names().is_empty());
    assert_eq!(*built.lock().expect("lock"), 0);
}

#[test]
fn custom_constructors_follow_config_order() {
    let harness = Harness::new(default_config());
    let mut registry = StageRegistry::builtin();
    for type_name in ["MentionStage", "EDStage"] {
        registry.register(type_name, |stage: &StageConfig, _: &StageBuildContext| {
            Ok(Box::new(probe(stage.name(), false).0) as Box<dyn Stage>)
        });
    }
    assert!(registry.is_registered("EDStage"));
    let pipeline =
        registry.build(harness.services.clone(), UnregisteredStages::Skip).expect("builds");
    assert_eq!(
        pipeline.stage_names(),
        [
            "rag_index",
            "mention_detection",
            "definition_extraction",
            "coreference",
            "ed_resolution",
            "relation_extraction",
            "fact_extraction",
            "contradiction_check"
        ]
    );
}

#[test]
fn construct_rejects_other_stage_configs() {
    let harness = Harness::new(default_config());
    let mention = harness
        .services
        .config
        .stages
        .iter()
        .find(|s| s.type_name() == "MentionStage")
        .expect("mention stage")
        .clone();
    let error = RagIndexStage::construct(&mention, &harness.services).err().expect("wrong type");
    assert!(matches!(error, PipelineError::InvalidStageConfig { .. }), "{error}");
}

#[test]
fn sentence_chunking_rejects_overlap_not_below_size() {
    let harness = Harness::new(config_with(|_| {}, |c| c.chunk_overlap = c.chunk_size));
    let error = RagIndexStage::new(&harness.rag_config(), &harness.services).expect_err("invalid");
    assert!(matches!(error, PipelineError::InvalidStageConfig { .. }), "{error}");
}

#[test]
fn runner_records_stage_timings_in_order() {
    let harness = Harness::new(default_config());
    let (a, seen_a) = probe("a", false);
    let (b, _) = probe("b", false);
    let pipeline = Pipeline::new(vec![Box::new(a), Box::new(b)], harness.services.clone());
    let mut ctx = pipeline.new_context();
    let report = pipeline
        .run(&mut ctx, PipelineInput::from(DocumentPayload::new("d1", "Text.")))
        .expect("runs");
    let names: Vec<_> = report.stage_timings.iter().map(|t| t.stage.as_str()).collect();
    assert_eq!(names, ["a", "b"]);
    assert!(
        report.total >= report.stage_timings.iter().map(|t| t.elapsed).sum::<std::time::Duration>()
    );
    assert_eq!(ctx.data.case_id.as_deref(), Some("seen-by-b"));
    assert_eq!(seen_a.lock().expect("lock")[0].doc_id.as_str(), "d1");
}

#[test]
fn runner_stops_at_the_first_failing_stage() {
    let harness = Harness::new(default_config());
    let (a, _) = probe("a", true);
    let (b, seen_b) = probe("b", false);
    let pipeline = Pipeline::new(vec![Box::new(a), Box::new(b)], harness.services.clone());
    let mut ctx = pipeline.new_context();
    let error = pipeline
        .run(&mut ctx, PipelineInput::from(DocumentPayload::new("d1", "Text.")))
        .expect_err("stage a fails");
    assert_eq!(
        error.to_string(),
        "stage `a` failed: stage `a` needs `doc_text` in the pipeline context"
    );
    assert!(seen_b.lock().expect("lock").is_empty());
}

fn seen_document(input: PipelineInput) -> DocumentPayload {
    let harness = Harness::new(default_config());
    let (stage, seen) = probe("probe", false);
    let pipeline = Pipeline::new(vec![Box::new(stage)], harness.services.clone());
    let mut ctx = pipeline.new_context();
    pipeline.run(&mut ctx, input).expect("runs");
    let document = seen.lock().expect("lock").remove(0);
    document
}

#[test]
fn jsonl_input_uses_the_run_processing_reader() {
    let path = fixture("Mason_watson.jsonl");
    assert!(matches!(PipelineInput::from_path(&path), PipelineInput::Jsonl(_)));
    let document = seen_document(PipelineInput::from_path(&path));
    assert_eq!(document.doc_id.as_str(), "002");
    assert_eq!(document.title, "002");
    assert_eq!(
        serde_json::Value::Object(document.metadata),
        json!({"case_id": "mason_v_billy", "db_id": "depo_1"})
    );
}

#[test]
fn json_file_input_follows_the_document_loader() {
    let path = fixture("ingest/sample_memo.json");
    assert!(matches!(PipelineInput::from_path(&path), PipelineInput::JsonFile(_)));
    let document = seen_document(PipelineInput::from_path(&path));
    assert_eq!(document.doc_id.as_str(), "memo-007");
    assert_eq!(document.title, "memo-007");
    assert_eq!(document.doc_type, "UNKNOWN");
    assert_eq!(document.total_pages, Some(2));
    assert_eq!(document.file_path.as_deref(), Some(path.to_string_lossy().as_ref()));
    assert!(document.metadata.is_empty(), "unknown keys such as author are dropped");
    assert!(document.text.starts_with("Memo: the Lopez lease"));

    let dir = tempfile::tempdir().expect("tempdir");
    let promoted = dir.path().join("doc.json");
    std::fs::write(
        &promoted,
        r#"{"doc_id": 7, "text": "T.", "case_id": "C-1", "db_id": "D-1", "metadata": {"db_id": "kept"}, "title": "", "date_created": "2024-03-01T10:00:00Z"}"#,
    )
    .expect("write");
    let document = seen_document(PipelineInput::JsonFile(promoted));
    assert_eq!(document.doc_id.as_str(), "7");
    assert_eq!(document.title, "7");
    assert_eq!(
        serde_json::Value::Object(document.metadata),
        json!({"db_id": "kept", "case_id": "C-1"})
    );
    assert!(document.date_created.is_some());
}

#[test]
fn json_file_input_errors() {
    let dir = tempfile::tempdir().expect("tempdir");
    let missing = PipelineInput::JsonFile(dir.path().join("missing.json")).into_document();
    assert!(matches!(missing, Err(PipelineError::Io { .. })));
    for (name, content) in [
        ("bad.json", "not json"),
        ("array.json", "[1]"),
        ("no_text.json", r#"{"doc_id": "a"}"#),
        ("bad_text.json", r#"{"doc_id": "a", "text": 5}"#),
    ] {
        let path = dir.path().join(name);
        std::fs::write(&path, content).expect("write");
        let result = PipelineInput::JsonFile(path).into_document();
        assert!(matches!(result, Err(PipelineError::InvalidDocument { .. })), "{name}");
    }
    let jsonl = PipelineInput::Jsonl(dir.path().join("missing.jsonl")).into_document();
    assert!(matches!(jsonl, Err(PipelineError::Ingest(ce_ingest::IngestError::NotFound(_)))));
}

#[test]
fn rag_index_pipeline_runs_on_json_input() {
    let harness = Harness::new(default_config());
    let pipeline = StageRegistry::empty()
        .with(RagIndexStage::TYPE_NAME, RagIndexStage::construct)
        .build(harness.services.clone(), UnregisteredStages::Skip)
        .expect("builds");
    let mut ctx = pipeline.new_context();
    let report = pipeline
        .run(&mut ctx, PipelineInput::from_path(fixture("ingest/sample_memo.json")))
        .expect("runs");
    assert_eq!(report.stage_timings.len(), 1);
    let result = ctx.data.rag_index_result.expect("indexed");
    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.embeddings.count, 1);
    // The em dash is one character: offsets are character offsets.
    let text = ctx.data.doc_text.expect("text");
    assert_eq!(result.chunks[0].chunk.end_char, text.trim_end().chars().count());
}
