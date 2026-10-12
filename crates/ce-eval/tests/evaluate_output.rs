//! Cases of the Python `tests/unit/test_evaluate_output.py`, scripted-vector matching with
//! `MockEmbedder`, and byte parity with reports written by the Python evaluator on the
//! synthetic e2e output in `fixtures/input/eval/`.

mod common;

use std::collections::BTreeSet;

use ce_core::e2e::E2eOutput;
use ce_eval::{
    assign_one_to_one, evaluate, load_e2e_output, parse_key_file, run_evaluation, AssignmentMethod,
    EvalError, EvalOptions, EvalReport, GroundTruthContradiction, GroundTruthFact, Key,
    PipelineOutput,
};
use ce_models::MockEmbedder;
use ce_text::values::{extract_values, values_compatible, Number, Value};
use chrono::NaiveDateTime;
use common::{read_fixture, workspace_root, TokenEmbedder};
use serde_json::json;

// ---------------------------------------------------------------------------------------------
// Value extraction / guard
// ---------------------------------------------------------------------------------------------

fn num(value: f64) -> Value {
    Value::Num(Number::from(value))
}

#[test]
fn extract_values_cases() {
    let cases: [(&str, Vec<Value>); 7] = [
        (
            "Collision date: March 5, 2023",
            vec![Value::Date { year: Some(2023), month: 3, day: Some(5) }],
        ),
        ("Approximately 5:30 p.m.", vec![Value::Time { hour: 5, minute: 30, meridiem: Some('p') }]),
        ("expenses exceed $28,000.00", vec![num(28000.0)]),
        ("unable to work for four (4) weeks", vec![num(4.0)]),
        ("acknowledges the 2018 collision", vec![Value::Year(2018)]),
        ("Reed traveling northbound on 14th Ave", vec![]),
        ("Lopez was the sole occupant", vec![]),
    ];
    for (text, expected) in cases {
        let expected: BTreeSet<Value> = expected.into_iter().collect();
        assert_eq!(extract_values(text), expected, "{text}");
    }
}

#[test]
fn values_compatible_cases() {
    let cases = [
        ("March 5, 2023", "June 10, 2018", false),
        ("March 5, 2023", "on March 5, 2023 at the intersection", true),
        ("in 2018", "June 10, 2018", true),
        ("$300", "300 dollars", true),
        ("$3200", "$300", false),
        ("5:30 p.m.", "9:20 p.m.", false),
        ("Lopez was properly restrained", "June 10, 2018", true),
    ];
    for (key, cand, compatible) in cases {
        assert_eq!(
            values_compatible(&extract_values(key), &extract_values(cand)),
            compatible,
            "{key} vs {cand}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Assignment
// ---------------------------------------------------------------------------------------------

#[test]
fn assignment_is_one_to_one() {
    let scores = vec![vec![0.95, 0.80], vec![0.90, 0.10]];
    assert_eq!(
        assign_one_to_one(&scores, 0.75, AssignmentMethod::Optimal).unwrap(),
        [(0, 1), (1, 0)]
    );
}

#[test]
fn assignment_respects_threshold() {
    let scores = vec![vec![0.95, 0.10], vec![0.60, 0.20]];
    assert_eq!(assign_one_to_one(&scores, 0.75, AssignmentMethod::Optimal).unwrap(), [(0, 0)]);
}

#[test]
fn assignment_greedy_fallback() {
    let scores = vec![vec![0.95, 0.80], vec![0.90, 0.10]];
    assert_eq!(assign_one_to_one(&scores, 0.75, AssignmentMethod::Greedy).unwrap(), [(0, 0)]);
}

// ---------------------------------------------------------------------------------------------
// evaluate() on synthetic outputs
// ---------------------------------------------------------------------------------------------

fn gt(id: &str, a: &str, b: &str) -> GroundTruthContradiction {
    let fact = |text: &str, source: &str| GroundTruthFact {
        text: text.to_owned(),
        source: source.to_owned(),
        paragraph_ref: None,
    };
    GroundTruthContradiction {
        id: id.to_owned(),
        contradiction_type: "t".to_owned(),
        fact_a: fact(a, "Doc 1"),
        fact_b: fact(b, "Doc 2"),
        notes: String::new(),
        is_inter_doc: true,
    }
}

fn output(cross: &[(&str, &str)], all_facts: Option<&[&str]>) -> E2eOutput {
    let pairs: Vec<_> = cross
        .iter()
        .map(|(n, e)| json!({"new_fact_text": n, "existing_fact_text": e, "nli_label": "contradiction", "nli_confidence": 0.9}))
        .collect();
    let s5 = match all_facts {
        Some(facts) => json!({"all_facts": facts.iter().enumerate()
            .map(|(i, t)| json!({"fact_id": format!("F{i}"), "text": t})).collect::<Vec<_>>()}),
        None => json!({}),
    };
    serde_json::from_value(json!({"documents": [{
        "doc_id": "002",
        "stage_5_fact_extraction": s5,
        "stage_6_contradiction_check": {
            "all_clean_facts": [],
            "intra_doc_contradictions": [],
            "cross_doc_contradictions": pairs,
        },
    }]}))
    .expect("synthetic output is valid")
}

fn options(threshold: f64, use_value_guard: bool) -> EvalOptions {
    EvalOptions { threshold, use_value_guard, ..EvalOptions::default() }
}

#[test]
fn evaluate_matches_counts_and_guard() {
    let key = Key {
        contradictions: vec![
            gt(
                "C-01",
                "collision occurred on March 5, 2023",
                "collision occurred on March 4, 2023",
            ),
            gt("C-02", "Lopez was the sole occupant", "two people were in the vehicle"),
            gt("C-03", "hospital expenses totaled $3200", "hospital expenses totaled $300"),
        ],
    };
    let out = output(
        &[
            ("the collision occurred on March 4, 2023", "the collision occurred on March 5, 2023"),
            ("hospital expenses totaled $500", "hospital expenses totaled $900"),
            ("Reed was northbound", "Reed was southbound"),
        ],
        None,
    );
    let report = evaluate(&out, &key, &TokenEmbedder::new(), &options(0.6, true)).unwrap();

    let overall = &report.metrics.overall;
    assert_eq!((overall.true_positives, overall.ground_truth, overall.detected), (1, 3, 3));
    assert!((overall.recall - 1.0 / 3.0).abs() < 1e-12);
    assert!((overall.precision - 1.0 / 3.0).abs() < 1e-12);
    assert_eq!((overall.false_positives, overall.false_negatives), (2, 2));
    assert_eq!(report.metrics.inter_doc.true_positives, 1);
    assert_eq!(report.metrics.intra_doc.ground_truth, 0);

    let [matched] = report.matched_contradictions.as_slice() else {
        panic!("expected one match, got {:?}", report.matched_contradictions);
    };
    assert_eq!(matched.ground_truth.ground_truth_id, "C-01");
    assert!(matched.detected_pair.existing_fact_text.contains("March 5, 2023"));
    assert_eq!(matched.scores.orientation, "a=existing,b=new");
    let missed: Vec<&str> = report
        .missed_contradictions
        .iter()
        .map(|m| m.ground_truth.ground_truth_id.as_str())
        .collect();
    assert_eq!(missed, ["C-02", "C-03"]);
    let c03 = &report.missed_contradictions[1];
    assert!(c03.closest_detected_pair.as_ref().unwrap().value_guard_rejected);
    assert_eq!(report.false_positives.len(), 2);

    let unguarded = evaluate(&out, &key, &TokenEmbedder::new(), &options(0.6, false)).unwrap();
    assert_eq!(unguarded.metrics.overall.true_positives, 2);
}

#[test]
fn fact_coverage_prefers_all_facts() {
    let key = Key {
        contradictions: vec![gt(
            "C-01",
            "Lopez was the sole occupant",
            "two people were in the vehicle",
        )],
    };
    let out = output(&[], Some(&["Lopez was the sole occupant of the car"]));
    assert!(PipelineOutput::from_e2e(&out).has_all_facts);
    let report = evaluate(&out, &key, &TokenEmbedder::new(), &options(0.6, true)).unwrap();
    let cov = &report.metrics.fact_coverage;
    assert_eq!((cov.covered, cov.total), (1, 2));
    assert_eq!(cov.fact_source, "all_extracted_facts");
}

#[test]
fn run_e2e_output_is_readable_by_evaluator() {
    let fact = |id: &str, text: &str, subject: Option<&str>| {
        json!({"fact_id": id, "text": text, "value_type": null, "property_type": null,
               "subject_entity_id": subject, "object_entity_id": null, "is_boilerplate": false})
    };
    let doc = |doc_id: &str| {
        let facts = [
            fact("F0", "Collision on March 5, 2023", None),
            fact("F1", "Lopez was alone", Some("E1")),
        ];
        json!({
            "doc_id": doc_id, "case_id": "c1", "db_id": "d1", "title": null,
            "doc_path": format!("{doc_id}.jsonl"), "processing_time_seconds": 0.1,
            "stage_4_entity_disambiguation": {"total_resolved": 0, "total_unresolved": 0,
                "resolution_by_strategy": {}, "new_entities_created_count": 0, "entities_reused_count": 0},
            "stage_5_fact_extraction": {"total_facts": 2, "boilerplate_count": 0,
                "facts_by_value_type": {"UNKNOWN": 2}, "facts_by_property_type": {"UNKNOWN": 2},
                "entity_linking": {"linked_subjects": 1, "linked_objects": 0, "linked_both": 0,
                    "linked_either": 1, "linking_rate_percent": 50.0},
                "all_facts": facts},
            "stage_6_contradiction_check": {"total_checked": 2, "contradiction_count": 1,
                "needs_review_count": 0, "clean_count": 1, "committed_fact_count": 1,
                "all_clean_facts": [{"fact_id": "F1", "text": "Lopez was alone", "value_type": null,
                    "property_type": null, "subject_entity_id": "E1", "object_entity_id": null}],
                "intra_doc_contradictions": [],
                "cross_doc_contradictions": [{"new_fact_text": "Collision on March 5, 2023",
                    "new_fact_id": "F0", "existing_fact_text": "Collision on March 4, 2023",
                    "existing_fact_id": "H1", "nli_label": "contradiction", "nli_confidence": 0.97,
                    "retrieval_sources": []}],
                "retrieval_stats": {"total_candidates_retrieved": 0, "retrieval_sources_breakdown": {}},
                "filtering_stats": {"pairs_after_prefilter": 0, "total_nli_calls": 0},
                "nli_stats": {"nli_label_breakdown": {"contradiction": 1},
                    "contradiction_threshold": null, "review_threshold": null}},
        })
    };
    let payload: E2eOutput =
        serde_json::from_value(json!({"status": "ok", "documents": [doc("001"), doc("002")]}))
            .unwrap();
    let ids: Vec<_> = payload.documents.iter().map(|d| d.doc_id.as_deref()).collect();
    assert_eq!(ids, [Some("001"), Some("002")]);
    assert_eq!(payload.documents[0].case_id.as_deref(), Some("c1"));

    let loaded = PipelineOutput::from_e2e(&payload);
    assert_eq!(loaded.cross_doc.len(), 2);
    assert!(loaded.intra_doc.is_empty());
    assert!(loaded.has_all_facts);
    let texts: BTreeSet<&str> = loaded.facts.iter().map(|f| f.text.as_str()).collect();
    assert_eq!(texts, BTreeSet::from(["Collision on March 5, 2023", "Lopez was alone"]));
}

/// Scripted vectors: both key rows prefer detection 0; only the optimal assignment matches
/// both rows.
#[test]
fn scripted_vectors_optimal_versus_greedy() {
    let b = vec![0.0, 0.0, 0.0, 0.0, 1.0];
    let embedder = MockEmbedder::new(5)
        .with_vector("new zero", vec![1.0, 0.0, 0.0, 0.0, 0.0])
        .with_vector("new one", vec![0.0, 1.0, 0.0, 0.0, 0.0])
        .with_vector("key a one", vec![0.8, 0.55, 0.0575_f32.sqrt(), 0.0, 0.0])
        .with_vector("key a two", vec![0.75, 0.1, 0.0, 0.4275_f32.sqrt(), 0.0])
        .with_vector("old zero", b.clone())
        .with_vector("old one", b.clone())
        .with_vector("key b one", b.clone())
        .with_vector("key b two", b);
    let key = Key {
        contradictions: vec![
            gt("K-1", "key a one", "key b one"),
            gt("K-2", "key a two", "key b two"),
        ],
    };
    let out = output(&[("new zero", "old zero"), ("new one", "old one")], None);

    let optimal = evaluate(&out, &key, &embedder, &options(0.5, true)).unwrap();
    assert_eq!(optimal.assignment, "optimal (scipy linear_sum_assignment)");
    assert_eq!(optimal.embedding_model, "mock-embedder");
    let pairs: Vec<(&str, &str)> = optimal
        .matched_contradictions
        .iter()
        .map(|m| (m.ground_truth.ground_truth_id.as_str(), m.detected_pair.new_fact_text.as_str()))
        .collect();
    assert_eq!(pairs, [("K-1", "new one"), ("K-2", "new zero")]);
    let k1 = &optimal.matched_contradictions[0].scores;
    assert!((k1.combined - 0.55).abs() < 1e-6 && (k1.fact_b - 1.0).abs() < 1e-6);
    assert_eq!(k1.orientation, "a=new,b=existing");

    let greedy_options = EvalOptions { assignment: AssignmentMethod::Greedy, ..options(0.5, true) };
    let greedy = evaluate(&out, &key, &embedder, &greedy_options).unwrap();
    assert_eq!(greedy.assignment, "greedy (scipy not installed)");
    assert_eq!(greedy.metrics.overall.true_positives, 1);
    let missed = &greedy.missed_contradictions[0];
    assert_eq!(missed.ground_truth.ground_truth_id, "K-2");
    let closest = missed.closest_detected_pair.as_ref().unwrap();
    assert_eq!(closest.pair.new_fact_text, "new zero");
    assert_eq!(closest.taken_by.as_deref(), Some("K-1"));
}

// ---------------------------------------------------------------------------------------------
// Parity with the Python evaluator (reports generated from the same inputs with an identical
// token-cosine similarity)
// ---------------------------------------------------------------------------------------------

fn synthetic_report(use_value_guard: bool, strip_stage_5: bool) -> EvalReport {
    let root = workspace_root();
    let mut output = load_e2e_output(&root.join("fixtures/input/eval/e2e_synthetic.json"))
        .expect("e2e fixture loads");
    if strip_stage_5 {
        for doc in &mut output.documents {
            doc.stage_5_fact_extraction = None;
        }
    }
    let key = parse_key_file(&root.join("fixtures/keys/GREEN_VALLEY_CONTRA_KEY.md"))
        .expect("key fixture parses");
    let options = EvalOptions {
        threshold: 0.6,
        use_value_guard,
        assignment: AssignmentMethod::Optimal,
        embedding_model: None,
        input_file: "fixtures/input/eval/e2e_synthetic.json".to_owned(),
        ground_truth_file: "fixtures/keys/GREEN_VALLEY_CONTRA_KEY.md".to_owned(),
        timestamp: Some(
            NaiveDateTime::parse_from_str("2026-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
                .expect("valid timestamp"),
        ),
    };
    evaluate(&output, &key, &TokenEmbedder::new(), &options).expect("evaluation succeeds")
}

fn assert_parity(report: &EvalReport, name: &str) {
    let expected = read_fixture(&format!("fixtures/input/eval/expected_report_{name}.json"));
    pretty_assertions::assert_eq!(report.to_json().expect("report serialises"), expected);
    let back: EvalReport = serde_json::from_str(&expected).expect("expected report is a report");
    assert_eq!(&back, report);
    let summary = read_fixture(&format!("fixtures/input/eval/expected_summary_{name}.txt"));
    pretty_assertions::assert_eq!(report.summary(Some("out/report_eval.json")), summary);
}

#[test]
fn parity_with_python_value_guard_on() {
    assert_parity(&synthetic_report(true, false), "guard");
}

#[test]
fn parity_with_python_value_guard_off() {
    assert_parity(&synthetic_report(false, false), "no_guard");
}

#[test]
fn parity_with_python_fallback_facts() {
    assert_parity(&synthetic_report(true, true), "fallback_facts");
}

// ---------------------------------------------------------------------------------------------
// run_evaluation
// ---------------------------------------------------------------------------------------------

#[test]
fn run_evaluation_writes_report() {
    let root = workspace_root();
    let dir = tempfile::tempdir().unwrap();
    let report_path = dir.path().join("nested").join("e2e_synthetic_eval.json");
    let report = run_evaluation(
        &root.join("fixtures/input/eval/e2e_synthetic.json"),
        &root.join("fixtures/keys/GREEN_VALLEY_CONTRA_KEY.md"),
        Some(&report_path),
        &TokenEmbedder::new(),
        &EvalOptions { threshold: 0.6, ..EvalOptions::default() },
    )
    .unwrap();
    assert!(report.evaluation_id.starts_with("eval_"));
    assert!(report.input_file.ends_with("e2e_synthetic.json"));
    assert_eq!(report.metrics.overall.true_positives, 5);
    let written = std::fs::read_to_string(&report_path).unwrap();
    assert_eq!(written, report.to_json().unwrap());
}

#[test]
fn run_evaluation_rejects_empty_key() {
    let dir = tempfile::tempdir().unwrap();
    let key_path = dir.path().join("empty.md");
    std::fs::write(&key_path, "# nothing\n## Entities\n| a | b |\n").unwrap();
    let err = run_evaluation(
        &workspace_root().join("fixtures/input/eval/e2e_synthetic.json"),
        &key_path,
        None,
        &TokenEmbedder::new(),
        &EvalOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(err, EvalError::EmptyKey { .. }));
    assert!(err.to_string().starts_with("No contradictions parsed from key: "));
}
