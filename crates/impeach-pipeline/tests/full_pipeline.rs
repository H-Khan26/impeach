//! `run_processing_with` through all eight stages with mock models (embedder, gazetteer NER,
//! rule coreference, canned LLM facts, scripted NLI) and one `MemoryStore`: the incident
//! report first, then the complaint, which contradicts a committed fact of the first.

mod common;
mod contradiction_common;
mod ed_common;

use std::sync::Arc;

use contradiction_common::{
    add_models, green_valley_nli, ALL_STAGES, COMPLAINT_SIGNAL, COMPLAINT_SPEED, INCIDENT_SPEED,
};
use impeach_core::{ContradictionSource, ContradictionStatus, FactId, ReviewStatus};
use impeach_pipeline::{run_processing_with, ProcessingOptions, ProcessingRun};
use impeach_store::graph::{FACT, FACT_ABOUT_ENTITY, FACT_IN_DOCUMENT};
use impeach_store::{MemoryStore, ReviewStoreConfig, SqliteReviewStore};
use pretty_assertions::assert_eq;

fn fact_texts(store: &MemoryStore) -> Vec<String> {
    (0..5000_u64)
        .filter_map(|id| store.node_snapshot(&id.to_string()))
        .filter(|(label, _)| *label == FACT)
        .filter_map(|(_, node)| node["FactText"].as_str().map(str::to_owned))
        .collect()
}

fn run(
    store: &Arc<MemoryStore>,
    nli: &Arc<impeach_models::MockNli>,
    file: &str,
    options: &ProcessingOptions,
) -> ProcessingRun {
    let ner = Arc::new(ed_common::GazetteerNer::new(ed_common::GREEN_VALLEY_NAMES));
    let mut env = ed_common::env(store, ner);
    let registry = env.registry.take().expect("registry");
    add_models(&registry, Arc::clone(nli));
    env.registry = Some(registry);
    run_processing_with(common::fixture(file), options, env).expect("pipeline runs")
}

#[test]
fn two_documents_through_every_stage() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("review.sqlite3");
    let options =
        ProcessingOptions { db_path: Some(db_path.clone()), ..ProcessingOptions::default() };
    let store = Arc::new(MemoryStore::new(ed_common::DIM));
    let nli = green_valley_nli();

    // Document 1: every fact is clean and committed by the contradiction stage.
    let first = run(&store, &nli, "green_valley_incident_report.jsonl", &options);
    assert_eq!(first.stages, ALL_STAGES);
    assert_eq!(first.report.stage_timings.len(), 8);
    let result = &first.result;
    let report = result.contradiction_report.as_ref().expect("report");
    assert!(!result.extracted_facts.is_empty());
    assert_eq!(report.total_facts_extracted, result.extracted_facts.len());
    assert_eq!(result.clean_facts.len(), result.extracted_facts.len());
    assert_eq!(result.committed_fact_ids.len(), result.clean_facts.len());
    assert_eq!((result.contradiction_count, result.needs_review_count), (0, 0));
    assert!(!result.ed_results.is_empty());
    assert_eq!(result.sqlite_db_path, Some(db_path.to_string_lossy().into_owned()));
    // Intra-document pairs were considered (none passes the pre-filter); the store was empty.
    assert!(report.debug_trace.iter().any(|e| e["phase"] == "intra_document"));
    assert_eq!(report.total_candidates_retrieved, 0);
    let committed_first: Vec<FactId> = result.committed_fact_ids.clone();
    let texts = fact_texts(&store);
    assert_eq!(texts.len(), committed_first.len());
    assert!(texts.iter().any(|t| t == INCIDENT_SPEED));
    // The deferred commit writes the context edges.
    let edges = store.edges();
    assert_eq!(edges.iter().filter(|e| e.label == FACT_IN_DOCUMENT).count(), committed_first.len());
    assert!(edges.iter().any(|e| e.label == FACT_ABOUT_ENTITY));
    let first_facts = result.extracted_facts.len();

    // Document 2: the complaint's speed contradicts the incident report, its red light needs
    // review against Reed's green light.
    let second = run(&store, &nli, "lopez_v_green_valley_complaint.jsonl", &options);
    assert_eq!(second.stages, ALL_STAGES);
    let result = &second.result;
    let report = result.contradiction_report.as_ref().expect("report");
    assert_eq!((result.contradiction_count, result.needs_review_count), (1, 1));
    assert!(report.total_candidates_retrieved > 0);

    let speed = report
        .contradiction_results
        .iter()
        .find(|r| r.new_fact.text == COMPLAINT_SPEED)
        .expect("speed result");
    assert_eq!(speed.status, ContradictionStatus::Contradiction);
    let pair = &speed.contradicting_pairs[0];
    assert_eq!(pair.existing_fact_text, INCIDENT_SPEED);
    assert_eq!(pair.contradiction_source, ContradictionSource::CrossDocument);
    assert!(committed_first.contains(&pair.existing_fact_id), "{pair:?}");
    let signal = report
        .contradiction_results
        .iter()
        .find(|r| r.new_fact.text == COMPLAINT_SIGNAL)
        .expect("signal result");
    assert_eq!(signal.status, ContradictionStatus::NeedsReview);

    // Contradicting facts are not committed; the clean ones are.
    let texts = fact_texts(&store);
    assert!(!texts.iter().any(|t| t == COMPLAINT_SPEED || t == COMPLAINT_SIGNAL), "{texts:?}");
    assert_eq!(result.clean_facts.len(), result.extracted_facts.len() - 2);
    assert_eq!(result.committed_fact_ids.len(), result.clean_facts.len());
    assert_eq!(texts.len(), committed_first.len() + result.committed_fact_ids.len());
    assert!(result.clean_facts.iter().all(|f| f.text != COMPLAINT_SPEED));
    // Seven canned facts per document (the complaint repeats some incident phrases).
    assert_eq!((first_facts, committed_first.len()), (7, 7));
    assert_eq!((result.extracted_facts.len(), result.committed_fact_ids.len()), (7, 5));

    // Both pairs are in the review queue.
    let review = SqliteReviewStore::new(&ReviewStoreConfig::new(&db_path));
    let rows = review.get_by_status(ReviewStatus::Pending, None).expect("rows");
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|row| row.contradiction_source == ContradictionSource::CrossDocument));
    assert!(rows.iter().all(|row| row.case_id == "lopez_v_green_valley" && row.doc_id == "002"));
    assert!(rows.iter().all(|row| row.source_pipeline.as_deref() == Some("processing_v17")));

    // The result serialises in the Python field order.
    let value = result.to_json().expect("json");
    let keys: Vec<&str> = value.as_object().expect("object").keys().map(String::as_str).collect();
    assert_eq!(
        keys,
        [
            "contradiction_report",
            "clean_facts",
            "committed_fact_ids",
            "extracted_facts",
            "ed_results",
            "sqlite_db_path",
            "contradiction_count",
            "needs_review_count"
        ]
    );
}
