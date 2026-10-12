//! `IntraDocFilter`: bucketed pairing, the corrected temporal-context handling,
//! `require_assertion_overlap`, Path A / Path B / slot match and the Jaccard gate.

use ce_config::services::{IntraDocFilterConfig, PreFilterConfig};
use ce_config::MappingsRegistry;
use ce_contradictions::{
    FactPairPreFilter, IntraDocCandidate, IntraDocFilter, IntraDocOutcome,
    TEMPORAL_MISMATCH_PRIORITY_FACTOR,
};
use ce_core::{AssertionTriple, EntityId, ExtractedFact, TemporalContext, TemporalValidity};
use pretty_assertions::assert_eq;

fn prefilter() -> FactPairPreFilter {
    FactPairPreFilter::new(
        PreFilterConfig::default(),
        None,
        MappingsRegistry::load().expect("embedded mappings load"),
    )
}

fn filter() -> IntraDocFilter {
    IntraDocFilter::new(IntraDocFilterConfig::default())
}

fn fact(id: &str, text: &str, triple: (&str, &str, &str)) -> ExtractedFact {
    ExtractedFact::new(
        id,
        text,
        AssertionTriple::new(triple.0, triple.1, triple.2),
        1.0,
        TemporalValidity::default(),
    )
}

fn typed(mut f: ExtractedFact, value_type: &str, property_type: &str) -> ExtractedFact {
    f.value_type = Some(value_type.into());
    f.property_type = Some(property_type.into());
    f
}

fn slotted(mut f: ExtractedFact, slot: &str) -> ExtractedFact {
    f.slot_name = Some(slot.into());
    f
}

fn linked(mut f: ExtractedFact, subject: Option<&str>, object: Option<&str>) -> ExtractedFact {
    f.subject_entity_id = subject.map(EntityId::new);
    f.object_entity_id = object.map(EntityId::new);
    f
}

fn with_context(mut f: ExtractedFact, context: TemporalContext) -> ExtractedFact {
    f.temporal_context = context;
    f
}

/// Pairs as `(first id, second id, same temporal context)`.
fn pair_ids(facts: &[ExtractedFact]) -> Vec<(String, String, bool)> {
    filter()
        .candidate_pairs(facts)
        .into_iter()
        .map(|IntraDocCandidate { first, second, same_temporal_context }| {
            (facts[first].id.to_string(), facts[second].id.to_string(), same_temporal_context)
        })
        .collect()
}

fn p(a: &str, b: &str, same: bool) -> (String, String, bool) {
    (a.to_owned(), b.to_owned(), same)
}

// --- Pairing ---

#[test]
fn buckets_pair_slot_entity_and_struct_groups() {
    let facts = vec![
        slotted(fact("F3", "speed 45", ("Truck", "speed", "45")), "speed"),
        slotted(fact("F1", "speed 30", ("Truck", "speed", "30")), "speed"),
        // Structural bucket TEXT:ATTRIBUTE (missing types), not shared with slotted facts.
        fact("F2", "color red", ("Truck", "color", "red")),
        typed(fact("F5", "date one", ("Crash", "on", "May 1")), "TEMPORAL_POINT", "TEMPORAL"),
        fact("F4", "color blue", ("Truck", "color", "blue")),
        typed(fact("F6", "date two", ("Crash", "on", "May 2")), "TEMPORAL_POINT", "TEMPORAL"),
        // Alone in its structural bucket.
        typed(fact("F7", "count", ("Cars", "count", "3")), "NUMERIC", "QUANTITY"),
    ];
    assert_eq!(
        pair_ids(&facts),
        vec![p("F1", "F3", true), p("F2", "F4", true), p("F5", "F6", true)]
    );
}

#[test]
fn entity_bucket_crosses_slot_and_struct_buckets_once() {
    let facts = vec![
        linked(slotted(fact("B", "b", ("x", "r", "y")), "speed"), Some("E1"), None),
        linked(fact("A", "a", ("x", "r", "y")), Some("E1"), None),
        linked(fact("C", "c", ("x", "r", "y")), Some("E1"), Some("E2")),
        // Linked only by object id: no entity bucket, but shares the struct bucket with A, C.
        linked(fact("D", "d", ("x", "r", "y")), None, Some("E1")),
    ];
    assert_eq!(
        pair_ids(&facts),
        vec![
            p("A", "B", true),
            p("A", "C", true),
            p("A", "D", true),
            p("B", "C", true),
            p("C", "D", true),
        ]
    );
}

#[test]
fn empty_slot_and_entity_ids_are_ignored() {
    let facts = vec![
        linked(slotted(fact("A", "a", ("x", "r", "y")), ""), Some(""), None),
        typed(fact("B", "b", ("x", "r", "y")), "", ""),
    ];
    // Both fall back to the TEXT:ATTRIBUTE structural bucket.
    assert_eq!(pair_ids(&facts), vec![p("A", "B", true)]);
}

#[test]
fn temporal_mismatch_pairs_are_kept_after_same_context_pairs() {
    let facts = vec![
        with_context(fact("A", "a", ("x", "r", "y")), TemporalContext::PriorIncident),
        fact("B", "b", ("x", "r", "y")),
        fact("C", "c", ("x", "r", "y")),
        with_context(fact("D", "d", ("x", "r", "y")), TemporalContext::PriorIncident),
    ];
    assert_eq!(
        pair_ids(&facts),
        vec![
            p("A", "D", true),
            p("B", "C", true),
            p("A", "B", false),
            p("A", "C", false),
            p("B", "D", false),
            p("C", "D", false),
        ]
    );
}

#[test]
fn single_fact_has_no_pairs() {
    assert!(filter().candidate_pairs(&[fact("A", "a", ("x", "r", "y"))]).is_empty());
    assert!(filter().candidate_pairs(&[]).is_empty());
}

// --- Assertion overlap (require_assertion_overlap, handoff bug 3) ---

#[test]
fn assertion_overlap_by_name_or_linked_entity() {
    let (f, pre) = (filter(), prefilter());
    let base = fact("A", "a", ("The Truck", "hit", "Fence"));
    let same_subject = fact("B", "b", ("truck", "drove", "Road"));
    let same_object = fact("B", "b", ("Car", "hit", "fence"));
    let crossed = fact("B", "b", ("Fence", "was hit by", "Car"));
    let unrelated = fact("B", "b", ("Car", "drove", "Road"));
    assert!(f.has_assertion_overlap(&base, &same_subject, &pre));
    assert!(f.has_assertion_overlap(&base, &same_object, &pre));
    assert!(f.has_assertion_overlap(&base, &crossed, &pre));
    assert!(!f.has_assertion_overlap(&base, &unrelated, &pre));
    // Linked subject of one = linked object of the other.
    let a = linked(base, Some("E1"), None);
    let b = linked(unrelated, None, Some("E1"));
    assert!(f.has_assertion_overlap(&a, &b, &pre));
    // Empty names never match.
    let empty_a = fact("A", "a", ("", "r", ""));
    let empty_b = fact("B", "b", ("", "r", ""));
    assert!(!f.has_assertion_overlap(&empty_a, &empty_b, &pre));
}

#[test]
fn require_assertion_overlap_skips_or_keeps_pair() {
    let pre = prefilter();
    let a = slotted(fact("A", "Truck speed was 45 mph", ("Truck", "speed", "45 mph")), "speed");
    let b = slotted(fact("B", "Car speed was 30 mph", ("Car", "speed", "30 mph")), "speed");

    let strict = filter().evaluate_pair(&a, &b, &pre);
    assert_eq!(strict.outcome, IntraDocOutcome::SkipNoAssertionOverlap);
    assert!(!strict.assertion_overlap);
    assert_eq!(strict.priority, 0.0);

    let lenient = IntraDocFilter::new(IntraDocFilterConfig {
        require_assertion_overlap: false,
        ..IntraDocFilterConfig::default()
    });
    let decision = lenient.evaluate_pair(&a, &b, &pre);
    assert_eq!(decision.outcome, IntraDocOutcome::Pass);
    assert!(decision.slot_match);
    assert!(!decision.assertion_overlap);
}

// --- Path A / Path B / slot match / Jaccard gate ---

#[test]
fn path_a_needs_types_subject_and_predicate() {
    let (f, pre) = (filter(), prefilter());
    let a = typed(
        fact("A", "Meridian was founded in 2010", ("Meridian", "FOUNDED_IN", "2010")),
        "TEMPORAL_POINT",
        "TEMPORAL",
    );
    let b = typed(
        fact("B", "Meridian was founded in 2005", ("meridian", "founded_in", "2005")),
        "temporal_point",
        "temporal",
    );
    assert!(f.check_path_a(&a, &b, &pre));
    let other_subject = typed(
        fact("B", "Acme was founded in 2005", ("Acme", "FOUNDED_IN", "2005")),
        "TEMPORAL_POINT",
        "TEMPORAL",
    );
    assert!(!f.check_path_a(&a, &other_subject, &pre));
    let other_type = typed(b.clone(), "NUMERIC", "TEMPORAL");
    assert!(!f.check_path_a(&a, &other_type, &pre));
    let untyped = fact("B", "Meridian was founded in 2005", ("Meridian", "FOUNDED_IN", "2005"));
    assert!(!f.check_path_a(&a, &untyped, &pre));
    let other_predicate = typed(
        fact("B", "Meridian dissolved in 2005", ("Meridian", "DISSOLVED_IN", "2005")),
        "TEMPORAL_POINT",
        "TEMPORAL",
    );
    assert!(!f.check_path_a(&a, &other_predicate, &pre));
}

#[test]
fn path_b_needs_shared_entity_and_fuzzy_triple() {
    let (f, pre) = (filter(), prefilter());
    let a = linked(fact("A", "a", ("James Carter", "is", "CEO")), Some("E1"), Some("E2"));
    let b = linked(fact("B", "b", ("CEO", "is", "James Carter")), Some("E2"), None);
    assert!(f.check_path_b(&a, &b, &pre));
    let unlinked = fact("B", "b", ("CEO", "is", "James Carter"));
    assert!(!f.check_path_b(&a, &unlinked, &pre));
    let other_object = linked(fact("B", "b", ("James Carter", "is", "Driver")), Some("E1"), None);
    assert!(!f.check_path_b(&a, &other_object, &pre));
}

#[test]
fn incompatible_category_is_skipped() {
    let decision = filter().evaluate_pair(
        &fact("A", "Truck hit the fence", ("Truck", "hit", "fence")),
        &fact("B", "Truck was red", ("Truck", "color", "red")),
        &prefilter(),
    );
    assert_eq!(decision.outcome, IntraDocOutcome::SkipIncompatibleCategory);
    assert!(decision.assertion_overlap);
    assert_eq!(decision.text_overlap, 0.0);
}

#[test]
fn jaccard_gate_applies_only_to_path_a_alone() {
    let pre = prefilter();
    let a = typed(
        fact("A", "Meridian was founded in 2010 downtown", ("Meridian", "FOUNDED_IN", "2010")),
        "TEMPORAL_POINT",
        "TEMPORAL",
    );
    let b = typed(
        fact("B", "Records show establishment year 2005", ("Meridian", "FOUNDED_IN", "2005")),
        "TEMPORAL_POINT",
        "TEMPORAL",
    );
    let decision = filter().evaluate_pair(&a, &b, &pre);
    assert!(decision.path_a_valid && !decision.path_b_valid && !decision.slot_match);
    assert_eq!(decision.text_overlap, 0.0);
    assert_eq!(decision.outcome, IntraDocOutcome::SkipLowJaccard);

    // The same slot bypasses the gate.
    let decision = filter().evaluate_pair(
        &slotted(a.clone(), "founded"),
        &slotted(b.clone(), "founded"),
        &pre,
    );
    assert_eq!(decision.outcome, IntraDocOutcome::Pass);

    // So does Path B.
    let a_linked = linked(a.clone(), Some("E1"), None);
    let b_linked =
        linked(fact("B", "Founded 2010 per records", ("Meridian", "x", "2010")), Some("E1"), None);
    let decision = filter().evaluate_pair(&a_linked, &b_linked, &pre);
    assert!(decision.path_b_valid);
    assert_eq!(decision.outcome, IntraDocOutcome::Pass);

    // A high enough overlap passes on Path A alone.
    let close = typed(
        fact("B", "Meridian was founded in 2005 downtown", ("Meridian", "FOUNDED_IN", "2005")),
        "TEMPORAL_POINT",
        "TEMPORAL",
    );
    let decision = filter().evaluate_pair(&a, &close, &pre);
    assert_eq!(decision.outcome, IntraDocOutcome::Pass);
    // {meridian, founded, downtown} shared of 5 tokens.
    assert!((decision.text_overlap - 0.6).abs() < 1e-12);
    assert!((decision.priority - 0.6).abs() < 1e-12);
}

#[test]
fn temporal_mismatch_lowers_priority_but_keeps_pair() {
    let pre = prefilter();
    let a = slotted(
        fact("A", "No prior accidents in ten years", ("Driver", "had", "no prior accidents")),
        "prior_accidents",
    );
    let b = with_context(
        slotted(
            fact("B", "Prior collision in 2018 on record", ("Driver", "had", "collision")),
            "prior_accidents",
        ),
        TemporalContext::PriorIncident,
    );
    let mismatched = filter().evaluate_pair(&a, &b, &pre);
    assert_eq!(mismatched.outcome, IntraDocOutcome::Pass);
    let same = filter().evaluate_pair(&a, &with_context(b, TemporalContext::CurrentIncident), &pre);
    assert_eq!(same.outcome, IntraDocOutcome::Pass);
    assert!(same.text_overlap > 0.0);
    assert_eq!(mismatched.text_overlap, same.text_overlap);
    assert_eq!(mismatched.priority, same.priority * TEMPORAL_MISMATCH_PRIORITY_FACTOR);
    assert!(mismatched.priority < same.priority);
}

// --- Relationship compatibility helper ---

#[test]
fn relationships_compatible_uses_registry() {
    let (f, pre) = (filter(), prefilter());
    let with_rel = |rel: &str| fact("A", "a", ("x", rel, "y"));
    assert!(f.relationships_compatible(&with_rel(""), &with_rel("located_in"), &pre));
    assert!(f.relationships_compatible(&with_rel("FOUNDED_IN"), &with_rel("founded_in"), &pre));
    let registry = pre.mappings();
    for (a, b) in [("hired_in", "located_in"), ("EMPLOYED_BY", "born_in"), ("x", "y")] {
        assert_eq!(
            f.relationships_compatible(&with_rel(a), &with_rel(b), &pre),
            registry.relationship_types.are_compatible(&a.to_lowercase(), &b.to_lowercase()),
            "{a} / {b}"
        );
    }
}
