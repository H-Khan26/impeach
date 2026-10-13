//! The e2e runner with injected mocks: the Green Valley scenario into one `MemoryStore`, the
//! output JSON and its evaluation, a failing document, the fresh review database and the
//! argument errors.

mod common;
mod contradiction_common;
mod ed_common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ce_core::e2e::E2eOutput;
use ce_models::{MockEmbedder, ModelRegistry};
use ce_pipeline::e2e::{
    build_aggregates, render_summary, run_e2e, run_scenario, E2eEnv, E2eOptions,
    DEFAULT_CONFIG_LABEL,
};
use ce_pipeline::PipelineError;
use ce_store::{GraphStore, MemoryStore};
use contradiction_common::{add_models, green_valley_nli, COMPLAINT_SPEED, INCIDENT_SPEED};
use pretty_assertions::assert_eq;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn registry() -> Arc<ModelRegistry> {
    let ner = Arc::new(ed_common::GazetteerNer::new(ed_common::GREEN_VALLEY_NAMES));
    let registry = ed_common::registry(ner);
    add_models(&registry, green_valley_nli());
    registry
}

fn env(store: &Arc<MemoryStore>) -> E2eEnv {
    let dyn_store: Arc<dyn GraphStore> = store.clone();
    E2eEnv {
        registry: Some(registry()),
        store: Some(dyn_store),
        stages: None,
        eval_embedder: Some(Arc::new(MockEmbedder::new(ed_common::DIM))),
    }
}

fn options(output: &Path) -> E2eOptions {
    E2eOptions { fixtures_dir: fixtures_dir(), ..E2eOptions::new(output) }
}

#[test]
fn green_valley_scenario_with_mocks() {
    let dir = tempfile::tempdir().expect("tempdir");
    let output = dir.path().join("out").join("e2e_green_valley.json");
    let store = Arc::new(MemoryStore::new(ed_common::DIM));
    let run = run_scenario("green_valley", &options(&output), env(&store)).expect("run");

    let out = &run.output;
    assert_eq!(out.status, "ok");
    assert_eq!(out.scenario.as_deref(), Some("green_valley"));
    assert!(out.run_id.starts_with("e2e_green_valley_"), "{}", out.run_id);
    assert_eq!(out.pipeline_name, "processing_v17");
    assert_eq!(out.helix_namespace, "processing_v17");
    assert_eq!(out.helix_mode, "embedded");
    assert_eq!(out.config_path, DEFAULT_CONFIG_LABEL);
    let sqlite = dir.path().join("out").join("e2e_green_valley.sqlite3");
    assert_eq!(out.sqlite_db_path, sqlite.display().to_string());
    assert!(sqlite.exists());
    assert_eq!(out.inputs.len(), 2);
    assert!(out.inputs[0].ends_with("green_valley_incident_report.jsonl"));
    assert_eq!(out.pipeline_summary.total_documents, 2);
    assert!(out.pipeline_summary.total_processing_time_seconds > 0.0);

    // Per-document blocks.
    assert_eq!(out.documents.len(), 2);
    let (first, second) = (&out.documents[0], &out.documents[1]);
    assert_eq!(first.doc_id.as_deref(), Some("001"));
    assert_eq!(second.doc_id.as_deref(), Some("002"));
    assert_eq!(second.case_id.as_deref(), Some("lopez_v_green_valley"));
    assert!(first.error.is_none() && first.processing_time_seconds.is_some());
    let facts = first.stage_5_fact_extraction.as_ref().expect("facts");
    assert_eq!(facts.total_facts, 7);
    assert_eq!(facts.all_facts.as_ref().map(Vec::len), Some(7));
    let ed = first.stage_4_entity_disambiguation.as_ref().expect("ed");
    assert!(ed.total_resolved > 0 && ed.new_entities_created_count > 0);
    let check = second.stage_6_contradiction_check.as_ref().expect("check");
    assert_eq!((check.contradiction_count, check.needs_review_count), (1, 1));
    assert_eq!(check.total_checked, 7);
    assert_eq!(check.committed_fact_count, 5);
    assert!(check.intra_doc_contradictions.is_empty());
    assert_eq!(check.cross_doc_contradictions.len(), 2);
    let speed = check
        .cross_doc_contradictions
        .iter()
        .find(|p| p.new_fact_text == COMPLAINT_SPEED)
        .expect("speed pair");
    assert_eq!(speed.existing_fact_text, INCIDENT_SPEED);
    assert_eq!(speed.nli_label, "contradiction");
    assert!(check.retrieval_stats.total_candidates_retrieved > 0);
    assert_eq!(check.nli_stats.nli_label_breakdown.get("contradiction"), Some(&2));
    assert_eq!(check.nli_stats.contradiction_threshold, Some(0.80));

    // Aggregates over both documents.
    let aggregates = &out.aggregates;
    assert_eq!(aggregates, &build_aggregates(&out.documents));
    assert_eq!(aggregates.total_facts_extracted, 14);
    assert_eq!(aggregates.total_facts_committed, 12);
    assert_eq!((aggregates.total_contradictions, aggregates.total_needs_review), (1, 1));
    assert_eq!((aggregates.total_intra_doc_pairs, aggregates.total_cross_doc_pairs), (0, 2));
    assert!(aggregates.entity_reuse_rate_percent > 0.0);

    // The file holds the same output (Python json.dumps indent=2).
    let text = std::fs::read_to_string(&output).expect("output written");
    assert!(text.starts_with("{\n  \"run_id\": "));
    let back: E2eOutput = serde_json::from_str(&text).expect("output parses");
    assert_eq!(&back, out);
    assert_eq!(run.output_path, output);

    // Evaluated against the scenario key with the injected embedder.
    let report = run.eval_report.as_ref().expect("evaluated");
    assert!(report.ground_truth_file.ends_with("GREEN_VALLEY_CONTRA_KEY.md"));
    let report_path = run.eval_report_path.as_ref().expect("report path");
    assert_eq!(report_path, &dir.path().join("out").join("e2e_green_valley_eval.json"));
    assert!(report_path.exists());

    let summary = render_summary(out);
    assert!(summary.contains("Document 2: lopez_v_green_valley_complaint.jsonl"), "{summary}");
    assert!(summary.contains("1 contradiction, 1 needs review, 5 clean, 5 committed"), "{summary}");
    assert!(summary.contains("Cross-doc pairs (2), top 3:"), "{summary}");
    assert!(summary.contains("| Facts extracted / committed   | 14 / 12"), "{summary}");
    assert!(summary.is_ascii());
}

#[test]
fn failing_document_stops_the_run() {
    let dir = tempfile::tempdir().expect("tempdir");
    let bad = dir.path().join("bad.jsonl");
    std::fs::write(
        &bad,
        "{\"doc_id\": \"bad-1\", \"metadata\": {\"case_id\": \"C-1\", \"db_id\": 4}}\n",
    )
    .expect("write");
    let good = common::fixture("green_valley_incident_report.jsonl");
    let output = dir.path().join("failed.json");
    let mut options = options(&output);
    options.key = Some(fixtures_dir().join("keys/GREEN_VALLEY_CONTRA_KEY.md"));
    let store = Arc::new(MemoryStore::new(ed_common::DIM));

    let run = run_e2e(&[bad.clone(), good], None, &options, env(&store)).expect("run completes");
    let out = &run.output;
    assert_eq!(out.status, "failed");
    assert!(out.run_id.starts_with("e2e_custom_"));
    assert_eq!(out.documents.len(), 1, "later documents are not run");
    let document = &out.documents[0];
    assert_eq!(document.doc_id.as_deref(), Some("bad-1"));
    assert_eq!(document.case_id.as_deref(), Some("C-1"));
    assert_eq!(document.db_id.as_deref(), Some("4"));
    assert_eq!(document.doc_path, bad.display().to_string());
    assert!(document.error.as_deref().is_some_and(|e| e.contains("text")), "{document:?}");
    assert!(document.stage_6_contradiction_check.is_none());
    assert_eq!(out.aggregates, build_aggregates(&[]));
    // No evaluation of a failed run.
    assert!(run.eval_report.is_none());
    assert!(!dir.path().join("failed_eval.json").exists());
    let back: E2eOutput =
        serde_json::from_str(&std::fs::read_to_string(&output).expect("written")).expect("JSON");
    assert_eq!(back.status, "failed");
    assert!(render_summary(out).contains("FAILED"));
}

#[test]
fn review_database_is_fresh_unless_appending() {
    let dir = tempfile::tempdir().expect("tempdir");
    let output = dir.path().join("run.json");
    let db_path = dir.path().join("review.sqlite3");
    let input = [common::fixture("green_valley_incident_report.jsonl")];
    let mut options = options(&output);
    options.db_path = Some(db_path.clone());
    options.no_eval = true;

    // A stale (corrupt) file is replaced.
    std::fs::write(&db_path, "not a database").expect("write");
    let store = Arc::new(MemoryStore::new(ed_common::DIM));
    let run = run_e2e(&input, None, &options, env(&store)).expect("run");
    assert_eq!(run.output.status, "ok");
    assert_eq!(run.output.sqlite_db_path, db_path.display().to_string());

    // Appending keeps the file, so the corrupt one now fails the document.
    std::fs::write(&db_path, "not a database").expect("write");
    options.append_db = true;
    let store = Arc::new(MemoryStore::new(ed_common::DIM));
    let run = run_e2e(&input, None, &options, env(&store)).expect("run");
    assert_eq!(run.output.status, "failed");
    assert!(run.output.documents[0].error.is_some());
}

#[test]
fn argument_errors() {
    let dir = tempfile::tempdir().expect("tempdir");
    let output = dir.path().join("x.json");
    let store = Arc::new(MemoryStore::new(ed_common::DIM));
    let error = run_scenario("nope", &options(&output), env(&store)).expect_err("unknown");
    assert!(matches!(error, PipelineError::Config(_)), "{error}");
    assert!(error.to_string().contains("unknown scenario"), "{error}");

    let mut missing_key = options(&output);
    missing_key.key = Some(dir.path().join("missing.md"));
    let error = run_e2e(&[common::fixture("Mason_billy.jsonl")], None, &missing_key, env(&store))
        .expect_err("missing key");
    assert!(matches!(error, PipelineError::Io { .. }), "{error}");
    assert!(!output.exists(), "nothing runs without the key");

    // `no_eval` ignores the key.
    missing_key.no_eval = true;
    let run = run_e2e(&[common::fixture("Mason_billy.jsonl")], None, &missing_key, env(&store))
        .expect("runs");
    assert!(run.eval_report.is_none());
}
