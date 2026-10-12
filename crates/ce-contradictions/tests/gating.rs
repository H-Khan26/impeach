//! ContradictionGating: dual_path delegation and permissive hard filters / triggers.

mod common_b;

use std::sync::Arc;

use ce_config::pipeline::VerificationStrategyConfig;
use ce_config::services::PreFilterConfig;
use ce_config::MappingsRegistry;
use ce_contradictions::{CandidateEntityMap, ContradictionGating, FactPairPreFilter};
use ce_core::{EntityId, ExtractedFact, FactId, RetrievedFact};
use ce_models::{Embedder, MockEmbedder};
use common_b::{assert_close, fact, retrieved};
use pretty_assertions::assert_eq;
use serde_json::{json, Value};

fn config(overrides: Value) -> VerificationStrategyConfig {
    let mut base = json!({"judge_model": "nli", "gating_strategy": "permissive"});
    if let (Some(base), Value::Object(extra)) = (base.as_object_mut(), overrides) {
        base.extend(extra);
    }
    serde_json::from_value(base).expect("verification config deserialises")
}

fn prefilter(embedder: Option<Arc<dyn Embedder>>) -> Arc<FactPairPreFilter> {
    let mappings = MappingsRegistry::load().expect("mappings load");
    Arc::new(FactPairPreFilter::new(PreFilterConfig::default(), embedder, mappings))
}

fn permissive(overrides: Value) -> ContradictionGating {
    ContradictionGating::new(config(overrides), None, prefilter(None))
}

fn ids(scored: &[(&RetrievedFact, f64)]) -> Vec<String> {
    scored.iter().map(|(c, _)| c.fact_id.as_str().to_owned()).collect()
}

fn new_fact(text: &str) -> ExtractedFact {
    fact("STATED_FACT_0", text, "the car", "HAD_COLOR", "red")
}

fn with_value_type(mut candidate: RetrievedFact, value_type: &str) -> RetrievedFact {
    candidate.metadata.insert("value_type".into(), json!(value_type));
    candidate
}

#[test]
fn keyword_trigger_scores_jaccard() {
    let gating = permissive(json!({}));
    // {the, car, was, red} vs {the, car, was, blue}: 3 / 5 (stopwords are kept).
    let candidates = [retrieved("c1", "The car was blue", 0.5)];
    let scored = gating.filter_pairs(
        &new_fact("The car was red"),
        &candidates,
        &[],
        &CandidateEntityMap::new(),
    );
    assert_eq!(ids(&scored), ["c1"]);
    assert_close(scored[0].1, 0.5 * 1.6);
}

#[test]
fn keyword_tokens_skip_single_characters_and_fold_case() {
    let gating = permissive(json!({}));
    let candidates = [retrieved("c1", "CAFÉ owner x", 1.0), retrieved("c2", "a b c d", 1.0)];
    let scored = gating.filter_pairs(
        &new_fact("Café Owner y z"),
        &candidates,
        &[],
        &CandidateEntityMap::new(),
    );
    assert_eq!(ids(&scored), ["c1"]);
    assert_close(scored[0].1, 2.0);
}

#[test]
fn pairs_without_a_trigger_are_dropped() {
    let gating = permissive(json!({}));
    // {car, red, today} vs {car, blue, yesterday, and}: 1 / 6 < 0.30.
    let candidates = [retrieved("c1", "car blue yesterday and", 1.0)];
    let scored = gating.filter_pairs(
        &new_fact("car red today"),
        &candidates,
        &[],
        &CandidateEntityMap::new(),
    );
    assert!(scored.is_empty());

    let lowered = permissive(json!({"keyword_overlap_threshold": 0.1}));
    assert_eq!(
        lowered
            .filter_pairs(&new_fact("car red today"), &candidates, &[], &CandidateEntityMap::new())
            .len(),
        1
    );
}

#[test]
fn shared_entity_trigger_fires_from_the_candidate_entity_map() {
    let gating = permissive(json!({}));
    let candidates =
        [retrieved("c1", "Weather was sunny", 0.5), retrieved("c2", "Unrelated words here", 0.5)];
    let new_ids = [EntityId::new("e1")];
    let mut map = CandidateEntityMap::new();
    map.insert(FactId::new("c1"), vec![EntityId::new("e2"), EntityId::new("e1")]);
    map.insert(FactId::new("c2"), vec![EntityId::new("e3")]);

    let scored = gating.filter_pairs(&new_fact("Maria ran the light"), &candidates, &new_ids, &map);
    assert_eq!(ids(&scored), ["c1"]);
    assert_close(scored[0].1, 0.5 * 3.0);

    // Python passed an empty map, so the trigger never fired.
    let empty = gating.filter_pairs(
        &new_fact("Maria ran the light"),
        &candidates,
        &new_ids,
        &CandidateEntityMap::new(),
    );
    assert!(empty.is_empty());
}

#[test]
fn triggers_add_up() {
    let gating = permissive(json!({}));
    let candidates = [retrieved("c1", "The car was blue", 1.0)];
    let mut map = CandidateEntityMap::new();
    map.insert(FactId::new("c1"), vec![EntityId::new("e1")]);
    let scored = gating.filter_pairs(
        &new_fact("The car was red"),
        &candidates,
        &[EntityId::new("e1")],
        &map,
    );
    assert_close(scored[0].1, 1.0 + 2.0 + 0.6);
}

#[test]
fn value_type_mismatch_is_rejected() {
    let gating = permissive(json!({}));
    let mut f = new_fact("The car was red");
    f.value_type = Some("numeric".into());
    let candidates = [
        with_value_type(retrieved("location", "The car was red", 1.0), "LOCATION"),
        with_value_type(retrieved("numeric", "The car was red", 1.0), "NUMERIC"),
        with_value_type(retrieved("text", "The car was red", 1.0), "text"),
        // Unknown types normalise to the default (TEXT).
        with_value_type(retrieved("bogus", "The car was red", 1.0), "bogus"),
        retrieved("untyped", "The car was red", 1.0),
    ];
    let scored = gating.filter_pairs(&f, &candidates, &[], &CandidateEntityMap::new());
    assert_eq!(ids(&scored), ["numeric", "text", "bogus", "untyped"]);

    let strict = permissive(json!({"allow_text_value_type_mismatch": false}));
    let scored = strict.filter_pairs(&f, &candidates, &[], &CandidateEntityMap::new());
    assert_eq!(ids(&scored), ["numeric", "untyped"]);

    // A TEXT new fact pairs with anything when allowed.
    f.value_type = Some("TEXT".into());
    let scored = gating.filter_pairs(&f, &candidates, &[], &CandidateEntityMap::new());
    assert_eq!(scored.len(), candidates.len());
}

#[test]
fn subject_entity_id_mismatch_is_rejected_when_required() {
    let mut f = new_fact("The car was red");
    f.subject_entity_id = Some("e1".into());
    let with_subject = |id: &str, subject: Value| {
        let mut c = retrieved(id, "The car was red", 1.0);
        c.metadata.insert("subject_entity_id".into(), subject);
        c
    };
    let candidates = [
        with_subject("same", json!("e1")),
        with_subject("other", json!("e2")),
        with_subject("empty", json!("")),
        with_subject("null", json!(null)),
        retrieved("absent", "The car was red", 1.0),
    ];
    let required = permissive(json!({"require_entity_id_match": true}));
    let scored = required.filter_pairs(&f, &candidates, &[], &CandidateEntityMap::new());
    assert_eq!(ids(&scored), ["same", "empty", "null", "absent"]);

    let lenient = permissive(json!({}));
    assert_eq!(lenient.filter_pairs(&f, &candidates, &[], &CandidateEntityMap::new()).len(), 5);

    f.subject_entity_id = None;
    assert_eq!(required.filter_pairs(&f, &candidates, &[], &CandidateEntityMap::new()).len(), 5);
}

#[test]
fn results_are_sorted_stably_by_priority() {
    let gating = permissive(json!({}));
    let candidates = [
        retrieved("low-a", "The car was blue", 0.5),
        retrieved("high", "The car was blue", 1.0),
        retrieved("zero", "The car was blue", 0.0),
        retrieved("low-b", "The car was blue", 0.5),
    ];
    let scored = gating.filter_pairs(
        &new_fact("The car was red"),
        &candidates,
        &[],
        &CandidateEntityMap::new(),
    );
    assert_eq!(ids(&scored), ["high", "low-a", "low-b"]);
}

#[test]
fn semantic_trigger_uses_cached_embeddings() {
    let embedder = Arc::new(
        MockEmbedder::new(2)
            .with_vector("Maria ran the light", vec![1.0, 0.0])
            .with_vector("Signal violation by driver", vec![0.8, 0.6])
            .with_vector("Weather report", vec![0.5, 0.866]),
    );
    let gating =
        ContradictionGating::new(config(json!({})), Some(embedder.clone()), prefilter(None));
    let candidates = [
        retrieved("near", "Signal violation by driver", 1.0),
        retrieved("far", "Weather report", 1.0),
    ];
    let scored = gating.filter_pairs(
        &new_fact("Maria ran the light"),
        &candidates,
        &[],
        &CandidateEntityMap::new(),
    );
    assert_eq!(ids(&scored), ["near"]);
    assert!((scored[0].1 - (1.0 + 0.8 * 1.5)).abs() < 1e-6, "{}", scored[0].1);
    // The new fact's text is embedded once.
    assert_eq!(embedder.calls(), 3);

    let again = gating.filter_pairs(
        &new_fact("Maria ran the light"),
        &candidates,
        &[],
        &CandidateEntityMap::new(),
    );
    assert_eq!(ids(&again), ["near"]);
    assert_eq!(embedder.calls(), 3, "embeddings are cached");

    // Without an embedder the semantic trigger never fires.
    let plain = permissive(json!({}));
    assert!(plain
        .filter_pairs(
            &new_fact("Maria ran the light"),
            &candidates,
            &[],
            &CandidateEntityMap::new()
        )
        .is_empty());
}

#[test]
fn dual_path_delegates_to_the_prefilter() {
    let prefilter = prefilter(None);
    let gating = ContradictionGating::new(
        config(json!({"gating_strategy": "dual_path"})),
        None,
        prefilter.clone(),
    );
    assert_eq!(gating.config().gating_strategy, ce_config::pipeline::GatingStrategy::DualPath);

    let mut f = fact(
        "STATED_FACT_0",
        "Maria Lopez drove at 35 mph",
        "Maria Lopez",
        "DROVE_AT_SPEED",
        "35 mph",
    );
    f.value_type = Some("NUMERIC".into());
    f.property_type = Some("QUANTITATIVE".into());
    let mut same = retrieved("same", "Maria Lopez drove at 45 mph", 0.8);
    same.assertion_subject = Some("Maria Lopez".into());
    same.assertion_relationship = Some("DROVE_AT_SPEED".into());
    same.assertion_object = Some("45 mph".into());
    same.metadata.insert("value_type".into(), json!("NUMERIC"));
    same.metadata.insert("property_type".into(), json!("QUANTITATIVE"));
    let unrelated = retrieved("unrelated", "The weather was sunny", 0.9);
    let candidates = [unrelated, same];
    let new_ids = [EntityId::new("e1")];
    let mut map = CandidateEntityMap::new();
    map.insert(FactId::new("same"), vec![EntityId::new("e1")]);

    let gated = gating.filter_pairs(&f, &candidates, &new_ids, &map);
    let direct = prefilter.filter_pairs(&f, &candidates, &new_ids, &map);
    assert_eq!(ids(&gated), ids(&direct));
    for ((_, a), (_, b)) in gated.iter().zip(&direct) {
        assert_close(*a, *b);
    }
    assert_eq!(ids(&gated), ["same"]);
}
