//! Review-queue items built from a contradiction report.

mod common_b;

use ce_contradictions::{build_review_items_from_report, ReviewProvenance};
use ce_core::{
    ContradictionPair, ContradictionReport, ContradictionSource, ContradictionStatus,
    FactContradictionResult, NliLabel, ReviewClassification, ReviewStatus,
};
use common_b::{assert_close, fact};
use pretty_assertions::assert_eq;
use regex::Regex;
use serde_json::{json, Value};

fn provenance() -> ReviewProvenance {
    ReviewProvenance {
        case_id: "case-1".into(),
        session_id: "session-1".into(),
        segment_id: "doc-1".into(),
        segment_index: 3,
        doc_id: "doc-1".into(),
        stage: "PROCESSING".into(),
        witness_name: Some("Maria Lopez".into()),
        source_stream: None,
        segment_start_ts: Some("00:01".into()),
        segment_end_ts: None,
        source_pipeline: Some("processing_v17".into()),
    }
}

fn pair(new_id: &str, existing_id: &str, confidence: f64) -> ContradictionPair {
    let mut pair = ContradictionPair::new(
        format!("new {new_id}"),
        new_id,
        existing_id,
        format!("existing {existing_id}"),
        NliLabel::Contradiction,
        confidence,
    );
    pair.retrieval_sources = vec!["graph".into(), "vector".into()];
    pair.priority_score = 1.5;
    pair.shared_entity_ids = vec!["e1".into()];
    pair
}

fn report() -> ContradictionReport {
    let mut report = ContradictionReport::new("doc-1");
    let mut first = FactContradictionResult::new(
        fact("STATED_FACT_0", "Speed was 35 mph", "car", "HAD_SPEED", "35 mph"),
        ContradictionStatus::Contradiction,
    );
    first.contradicting_pairs = vec![pair("STATED_FACT_0", "fact-9", 0.9)];
    // A second result for the same id must not replace the first payload.
    let second = FactContradictionResult::new(
        fact("STATED_FACT_0", "Duplicate", "x", "R", "y"),
        ContradictionStatus::NeedsReview,
    );
    report.contradiction_results = vec![first, second];
    let mut intra = pair("STATED_FACT_1", "STATED_FACT_2", 0.7);
    intra.nli_label = NliLabel::Neutral;
    intra.contradiction_source = ContradictionSource::IntraDocument;
    report.unique_contradiction_pairs = vec![pair("STATED_FACT_0", "fact-9", 0.9), intra];
    report
}

#[test]
fn no_pairs_no_items() {
    let mut report = report();
    report.unique_contradiction_pairs.clear();
    assert!(build_review_items_from_report(&report, &provenance(), Some("ts")).is_empty());
}

#[test]
fn one_item_per_unique_pair() {
    let report = report();
    let items = build_review_items_from_report(&report, &provenance(), Some("2024-01-02T03:04:05"));
    assert_eq!(items.len(), 2);

    let first = &items[0];
    assert_eq!(first.id, None);
    assert_eq!(first.case_id, "case-1");
    assert_eq!(first.session_id, "session-1");
    assert_eq!(first.segment_id, "doc-1");
    assert_eq!(first.segment_index, 3);
    assert_eq!(first.doc_id.as_str(), "doc-1");
    assert_eq!(first.stage, "PROCESSING");
    assert_eq!(first.witness_name.as_deref(), Some("Maria Lopez"));
    assert_eq!(first.source_stream, None);
    assert_eq!(first.source_pipeline.as_deref(), Some("processing_v17"));
    assert_eq!(first.segment_start_ts.as_deref(), Some("00:01"));
    assert_eq!(first.segment_end_ts, None);
    assert_eq!(first.new_fact_id.as_str(), "STATED_FACT_0");
    assert_eq!(first.helix_new_fact_id, None);
    assert_eq!(first.existing_fact_id.as_str(), "fact-9");
    assert_close(first.nli_score, 0.9);
    assert_eq!(first.nli_label, Some(NliLabel::Contradiction));
    assert_eq!(first.classification, ReviewClassification::Contradiction);
    assert_eq!(first.contradiction_source, ContradictionSource::CrossDocument);
    assert_eq!(first.status, ReviewStatus::Pending);
    assert_eq!(first.created_at, "2024-01-02T03:04:05");
    assert_eq!(first.updated_at, "2024-01-02T03:04:05");
    assert_eq!(first.resolved_by, None);
    assert_eq!(first.resolution_note, None);
    assert_eq!(first.dedupe_key, None);

    let expected_new =
        serde_json::to_value(&report.contradiction_results[0].new_fact).expect("fact serialises");
    assert_eq!(Value::Object(first.new_fact_payload.clone()), expected_new);
    assert_eq!(
        Value::Object(first.existing_fact_payload.clone()),
        json!({
            "fact_id": "fact-9",
            "fact_text": "existing fact-9",
            "retrieval_sources": ["graph", "vector"],
            "priority_score": 1.5,
            "shared_entity_ids": ["e1"],
        })
    );
    let keys: Vec<&str> = first.existing_fact_payload.keys().map(String::as_str).collect();
    assert_eq!(
        keys,
        ["fact_id", "fact_text", "retrieval_sources", "priority_score", "shared_entity_ids"]
    );

    let second = &items[1];
    assert_eq!(second.classification, ReviewClassification::NeedsReview);
    assert_eq!(second.nli_label, Some(NliLabel::Neutral));
    assert_eq!(second.contradiction_source, ContradictionSource::IntraDocument);
    // No result for this fact id: the payload is empty.
    assert!(second.new_fact_payload.is_empty());
}

#[test]
fn classification_uses_the_report_threshold() {
    let mut report = report();
    report.contradiction_threshold = 0.7;
    let items = build_review_items_from_report(&report, &provenance(), Some("ts"));
    let classes: Vec<_> = items.iter().map(|i| i.classification).collect();
    assert_eq!(classes, [ReviewClassification::Contradiction, ReviewClassification::Contradiction]);

    report.contradiction_threshold = 0.95;
    let items = build_review_items_from_report(&report, &provenance(), Some("ts"));
    assert!(items.iter().all(|i| i.classification == ReviewClassification::NeedsReview));
}

#[test]
fn missing_timestamp_uses_utc_now_in_python_isoformat() {
    let pattern = Regex::new(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{6})?$").expect("regex");
    for created in [None, Some("")] {
        let items = build_review_items_from_report(&report(), &provenance(), created);
        let created_at = &items[0].created_at;
        assert!(pattern.is_match(created_at), "{created_at}");
        assert!(items.iter().all(|i| &i.created_at == created_at && &i.updated_at == created_at));
    }
}
