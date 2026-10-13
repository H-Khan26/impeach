//! `FactPairPreFilter`: cases of Python `tests/unit/test_prefilter.py` plus the corrected
//! `require_entity_overlap` / `use_assertion_boost` behaviour and the embedding paths.

use std::collections::HashSet;
use std::sync::Arc;

use impeach_config::services::PreFilterConfig;
use impeach_config::MappingsRegistry;
use impeach_contradictions::{CandidateEntityMap, FactPairPreFilter, ASSERTION_BOOST};
use impeach_core::{
    AssertionTriple, EntityId, ExtractedFact, FactId, RetrievedFact, TemporalValidity,
};
use impeach_models::{Embedder, MockEmbedder, ModelError};
use serde_json::json;

fn mappings() -> &'static MappingsRegistry {
    MappingsRegistry::load().expect("embedded mappings load")
}

fn filter_with(config: PreFilterConfig) -> FactPairPreFilter {
    FactPairPreFilter::new(config, None, mappings())
}

fn default_filter() -> FactPairPreFilter {
    filter_with(PreFilterConfig::default())
}

/// Token-overlap fallback enabled with threshold 0.1.
fn fallback_filter() -> FactPairPreFilter {
    filter_with(PreFilterConfig {
        min_text_overlap: 0.1,
        enable_text_overlap_fallback: true,
        ..PreFilterConfig::default()
    })
}

fn fact(text: &str, subject: &str, relationship: &str, object: &str) -> ExtractedFact {
    ExtractedFact::new(
        "fact_001",
        text,
        AssertionTriple::new(subject, relationship, object),
        1.0,
        TemporalValidity::default(),
    )
}

fn simple_fact(text: &str) -> ExtractedFact {
    fact(text, "Subject", "RELATES_TO", "Object")
}

fn typed_fact(text: &str, subject: &str, relationship: &str, object: &str) -> ExtractedFact {
    let mut f = fact(text, subject, relationship, object);
    f.value_type = Some("TEMPORAL_POINT".into());
    f.property_type = Some("TEMPORAL".into());
    f
}

fn retrieved(text: &str) -> RetrievedFact {
    let mut r = RetrievedFact::new("existing_001", text);
    r.sources = vec!["vector".into()];
    r.relevance_score = 0.8;
    r
}

fn retrieved_triple(text: &str, subject: &str, object: &str) -> RetrievedFact {
    let mut r = retrieved(text);
    r.assertion_subject = Some(subject.into());
    r.assertion_object = Some(object.into());
    r
}

/// A structurally typed new fact and candidate (TEMPORAL_POINT / TEMPORAL).
fn founding_pair(candidate_subject: &str) -> (ExtractedFact, RetrievedFact) {
    let new_fact = typed_fact(
        "Meridian Construction was founded in 2010",
        "Meridian Construction",
        "FOUNDED_IN",
        "2010",
    );
    let mut candidate = retrieved(&format!("{candidate_subject} was founded in 2005"));
    candidate.assertion_subject = Some(candidate_subject.into());
    candidate.assertion_relationship = Some("FOUNDED_IN".into());
    candidate.assertion_object = Some("2005".into());
    candidate.metadata.insert("value_type".into(), json!("TEMPORAL_POINT"));
    candidate.metadata.insert("property_type".into(), json!("TEMPORAL"));
    (new_fact, candidate)
}

fn meridian_pair() -> (ExtractedFact, RetrievedFact) {
    founding_pair("Meridian Construction")
}

fn ids(items: &[&str]) -> HashSet<EntityId> {
    items.iter().map(|s| EntityId::new(*s)).collect()
}

fn none() -> HashSet<EntityId> {
    HashSet::new()
}

fn carter_pair() -> (ExtractedFact, RetrievedFact) {
    (
        fact("James Carter is CEO", "James Carter", "RELATES_TO", "CEO"),
        retrieved_triple("James Carter became CEO", "James Carter", "CEO"),
    )
}

fn approx(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
}

// --- Jaccard similarity (Python `_jaccard_similarity`, here `impeach_text::jaccard`) ---

fn set(items: &[&str]) -> HashSet<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

#[test]
fn jaccard_cases() {
    let tokens = set(&["the", "quick", "brown", "fox"]);
    assert_eq!(impeach_text::jaccard(&tokens, &tokens), 1.0);
    assert_eq!(impeach_text::jaccard(&set(&["apple", "banana"]), &set(&["dog", "cat"])), 0.0);
    assert_eq!(impeach_text::jaccard(&set(&["a", "b", "c"]), &set(&["b", "c", "d"])), 0.5);
    assert_eq!(impeach_text::jaccard(&set(&[]), &set(&["a", "b"])), 0.0);
    assert_eq!(impeach_text::jaccard(&set(&["a", "b"]), &set(&[])), 0.0);
    assert_eq!(impeach_text::jaccard(&set(&[]), &set(&[])), 0.0);
    assert_eq!(impeach_text::jaccard(&set(&["a", "b"]), &set(&["a", "b", "c", "d"])), 0.5);
}

// --- Tokenisation ---

#[test]
fn basic_tokenization() {
    let tokens = default_filter().tokenize("The quick brown fox jumps");
    assert!(set(&["quick", "brown", "fox", "jumps"]).is_subset(&tokens));
    assert!(!tokens.contains("the"));
}

#[test]
fn stopword_removal() {
    let tokens = default_filter().tokenize("The cat is by the mat and it was sleeping");
    for stopword in ["the", "is", "by", "and", "it", "was"] {
        assert!(!tokens.contains(stopword), "{stopword}");
    }
    assert!(set(&["cat", "mat", "sleeping"]).is_subset(&tokens));
}

#[test]
fn yaml_boolean_like_stopwords_removed() {
    let tokens = default_filter().tokenize("Yes the cat sat on no mat");
    assert!(tokens.is_disjoint(&set(&["yes", "on", "no"])));
}

#[test]
fn single_char_removal() {
    let tokens = default_filter().tokenize("A cat sat on a mat");
    assert!(!tokens.contains("a"));
    assert!(set(&["cat", "sat", "mat"]).is_subset(&tokens));
}

#[test]
fn punctuation_handling() {
    let tokens = default_filter().tokenize("Hello, world! This is a test.");
    assert!(set(&["hello", "world", "test"]).is_subset(&tokens));
    assert!(tokens.is_disjoint(&set(&[",", "!", "."])));
}

#[test]
fn numbers_preserved() {
    let tokens = default_filter().tokenize("Founded in 2005 with 100 employees");
    assert!(set(&["2005", "100", "founded", "employees"]).is_subset(&tokens));
}

#[test]
fn lowercase_conversion() {
    let tokens = default_filter().tokenize("James Carter became CEO of Meridian");
    assert!(set(&["james", "carter", "ceo", "meridian"]).is_subset(&tokens));
    assert!(!tokens.contains("James") && !tokens.contains("CEO"));
}

#[test]
fn empty_and_stopword_only_text() {
    let filter = default_filter();
    assert!(filter.tokenize("").is_empty());
    assert!(filter.tokenize("the a an is are was were").is_empty());
}

// --- Entity name normalisation ---

#[test]
fn entity_normalization() {
    let filter = default_filter();
    for (raw, expected) in [
        ("JAMES CARTER", "james carter"),
        ("  Meridian Construction  ", "meridian construction"),
        ("The Meridian Group", "meridian group"),
        ("Meridian Inc", "meridian"),
        ("Meridian Corporation", "meridian"),
        ("Meridian LLC", "meridian"),
        ("", ""),
    ] {
        assert_eq!(filter.normalize_entity_name(raw), expected, "{raw:?}");
    }
}

// --- Assertion triple matching ---

#[test]
fn assertion_exact_match() {
    let (new_fact, cand) = carter_pair();
    assert!(default_filter().same_assertion_entities(&new_fact, &cand));
}

#[test]
fn assertion_swapped_entities() {
    let new_fact = fact("CEO is James Carter", "CEO", "RELATES_TO", "James Carter");
    let cand = retrieved_triple("James Carter became CEO", "James Carter", "CEO");
    assert!(default_filter().same_assertion_entities(&new_fact, &cand));
}

#[test]
fn assertion_partial_subject_match() {
    let new_fact = fact("James Carter resigned", "James Carter", "RELATES_TO", "Resignation");
    let cand = retrieved_triple("James Carter became CEO", "James Carter", "CEO");
    assert!(default_filter().same_assertion_entities(&new_fact, &cand));
}

#[test]
fn assertion_no_match() {
    let new_fact = fact("Sarah Miller resigned", "Sarah Miller", "RELATES_TO", "Resignation");
    let cand = retrieved_triple("James Carter became CEO", "James Carter", "CEO");
    assert!(!default_filter().same_assertion_entities(&new_fact, &cand));
}

#[test]
fn assertion_missing_candidate_triple() {
    let new_fact = fact("James Carter is CEO", "James Carter", "RELATES_TO", "CEO");
    let cand = retrieved("James Carter became CEO");
    assert!(!default_filter().same_assertion_entities(&new_fact, &cand));
}

#[test]
fn assertion_normalized_matching() {
    let new_fact =
        fact("Meridian hired Sarah", "Meridian Construction", "RELATES_TO", "Sarah Miller");
    let cand = retrieved_triple(
        "Meridian Construction hired Sarah Miller",
        "meridian construction",
        "sarah miller",
    );
    assert!(default_filter().same_assertion_entities(&new_fact, &cand));
}

// --- Priority computation (dual-path semantics) ---

#[test]
fn unrelated_pair_scores_zero() {
    let new_fact = simple_fact("The sky is blue today");
    let cand = retrieved("Meridian Construction was founded in 2005");
    assert_eq!(default_filter().compute_priority(&new_fact, &cand, &none(), &none()), 0.0);
}

#[test]
fn path_a_structural_match_passes() {
    let (new_fact, cand) = meridian_pair();
    assert!(default_filter().compute_priority(&new_fact, &cand, &none(), &none()) > 0.0);
    assert_eq!(default_filter().check_path_a(&new_fact, &cand), (true, 1.0));
}

#[test]
fn path_a_requires_subject_overlap() {
    let (new_fact, cand) = founding_pair("Other Company");
    assert_eq!(default_filter().compute_priority(&new_fact, &cand, &none(), &none()), 0.0);
}

#[test]
fn path_a_requires_matching_value_type() {
    let (new_fact, mut cand) = meridian_pair();
    cand.metadata.insert("value_type".into(), json!("LOCATION"));
    assert_eq!(default_filter().compute_priority(&new_fact, &cand, &none(), &none()), 0.0);
}

#[test]
fn path_a_compares_taxonomy_normalised_types() {
    let (mut new_fact, cand) = meridian_pair();
    new_fact.value_type = Some("temporal_point".into());
    assert!(default_filter().check_path_a(&new_fact, &cand).0);
}

#[test]
fn path_a_requires_structural_types() {
    let (new_fact, mut cand) = meridian_pair();
    cand.metadata.clear();
    // No shared entities either, so neither path validates.
    assert_eq!(default_filter().compute_priority(&new_fact, &cand, &none(), &none()), 0.0);
}

#[test]
fn path_b_entity_triple_match_passes() {
    let (new_fact, cand) = carter_pair();
    let priority =
        default_filter().compute_priority(&new_fact, &cand, &ids(&["ent_001"]), &ids(&["ent_001"]));
    assert!(priority > 0.0);
}

#[test]
fn path_b_requires_shared_entity() {
    let (new_fact, cand) = carter_pair();
    let priority =
        default_filter().compute_priority(&new_fact, &cand, &ids(&["ent_001"]), &ids(&["ent_002"]));
    assert_eq!(priority, 0.0);
}

#[test]
fn path_b_graph_source_counts_as_entity_link() {
    let (new_fact, mut cand) = carter_pair();
    cand.sources = vec!["graph".into()];
    assert!(default_filter().compute_priority(&new_fact, &cand, &ids(&["ent_001"]), &none()) > 0.0);
    // The implicit link counts as one shared entity.
    assert_eq!(
        default_filter().check_path_b(&new_fact, &cand, &ids(&["ent_001"]), &none()),
        (true, 1.1)
    );
}

#[test]
fn more_shared_entities_boost_priority() {
    let (new_fact, cand) = carter_pair();
    let filter = default_filter();
    let one = filter.compute_priority(&new_fact, &cand, &ids(&["e1"]), &ids(&["e1"]));
    let two = filter.compute_priority(&new_fact, &cand, &ids(&["e1", "e2"]), &ids(&["e1", "e2"]));
    assert!(two > one);
}

#[test]
fn text_overlap_fallback_disabled_by_default() {
    let new_fact = simple_fact("Meridian Construction was founded in 2010");
    let cand = retrieved("Meridian Construction was founded in 2005");
    assert_eq!(default_filter().compute_priority(&new_fact, &cand, &none(), &none()), 0.0);
}

#[test]
fn text_overlap_fallback_when_enabled() {
    let new_fact = simple_fact("Meridian Construction was founded in 2010");
    let cand = retrieved("Meridian Construction was founded in 2005");
    // Overlap 3/5: 0.8 * (1 + 0.6 * 0.8) * (1 + 0.6 * 0.3); no assertion boost (no triple).
    approx(
        fallback_filter().compute_priority(&new_fact, &cand, &none(), &none()),
        0.8 * 1.48 * 1.18,
    );
}

#[test]
fn fallback_still_rejects_low_overlap() {
    let new_fact = simple_fact("The sky is blue today");
    let cand = retrieved("Meridian Construction was founded in 2005");
    assert_eq!(fallback_filter().compute_priority(&new_fact, &cand, &none(), &none()), 0.0);
}

#[test]
fn higher_relevance_gives_higher_priority() {
    let (new_fact, mut low) = meridian_pair();
    let (_, mut high) = meridian_pair();
    low.relevance_score = 0.4;
    high.relevance_score = 0.9;
    let filter = default_filter();
    assert!(
        filter.compute_priority(&new_fact, &high, &none(), &none())
            > filter.compute_priority(&new_fact, &low, &none(), &none())
    );
}

// --- filter_pairs ---

#[test]
fn filter_pairs_empty_candidates() {
    let result = default_filter().filter_pairs(
        &simple_fact("Some fact text"),
        &[],
        &[],
        &CandidateEntityMap::new(),
    );
    assert!(result.is_empty());
}

#[test]
fn filter_pairs_all_filtered_out() {
    let new_fact = simple_fact("The sky is blue and clear today");
    let mut a = retrieved("Meridian Construction was founded");
    a.fact_id = FactId::new("1");
    let mut b = retrieved("James Carter became CEO");
    b.fact_id = FactId::new("2");
    let candidates = [a, b];
    assert!(default_filter()
        .filter_pairs(&new_fact, &candidates, &[], &CandidateEntityMap::new())
        .is_empty());
}

#[test]
fn filter_pairs_some_pass() {
    let (new_fact, mut good) = meridian_pair();
    good.fact_id = FactId::new("1");
    let mut bad = retrieved("The sky is blue today");
    bad.fact_id = FactId::new("2");
    let candidates = [good, bad];
    let result =
        default_filter().filter_pairs(&new_fact, &candidates, &[], &CandidateEntityMap::new());
    let kept: Vec<&str> = result.iter().map(|(c, _)| c.fact_id.as_str()).collect();
    assert_eq!(kept, ["1"]);
}

#[test]
fn filter_pairs_sorted_by_priority() {
    let (new_fact, mut low) = meridian_pair();
    let (_, mut high) = meridian_pair();
    low.fact_id = FactId::new("low_priority");
    low.relevance_score = 0.5;
    high.fact_id = FactId::new("high_priority");
    high.relevance_score = 0.9;
    let candidates = [low, high];
    let result =
        default_filter().filter_pairs(&new_fact, &candidates, &[], &CandidateEntityMap::new());
    let kept: Vec<&str> = result.iter().map(|(c, _)| c.fact_id.as_str()).collect();
    assert_eq!(kept, ["high_priority", "low_priority"]);
    assert!(result[0].1 >= result[1].1);
}

#[test]
fn filter_pairs_sort_is_stable_for_equal_priorities() {
    let (new_fact, mut first) = meridian_pair();
    let (_, mut second) = meridian_pair();
    first.fact_id = FactId::new("b");
    second.fact_id = FactId::new("a");
    let candidates = [first, second];
    let result =
        default_filter().filter_pairs(&new_fact, &candidates, &[], &CandidateEntityMap::new());
    let kept: Vec<&str> = result.iter().map(|(c, _)| c.fact_id.as_str()).collect();
    assert_eq!(kept, ["b", "a"]);
}

#[test]
fn filter_pairs_returns_priority_scores() {
    let (new_fact, cand) = meridian_pair();
    let candidates = [cand];
    let result =
        default_filter().filter_pairs(&new_fact, &candidates, &[], &CandidateEntityMap::new());
    assert_eq!(result.len(), 1);
    assert!(result[0].1 > 0.0);
}

#[test]
fn filter_pairs_uses_entity_map() {
    let (new_fact, mut cand) = carter_pair();
    cand.fact_id = FactId::new("c1");
    let candidates = [cand];
    let filter = default_filter();
    let new_ids = [EntityId::new("e1")];
    let map = |id: &str| CandidateEntityMap::from([(FactId::new("c1"), vec![EntityId::new(id)])]);
    assert!(filter.filter_pairs(&new_fact, &candidates, &new_ids, &map("e2")).is_empty());
    assert_eq!(filter.filter_pairs(&new_fact, &candidates, &new_ids, &map("e1")).len(), 1);
}

// --- shared_entities ---

#[test]
fn shared_entities_cases() {
    let filter = default_filter();
    let entity_ids = |items: &[&str]| items.iter().map(|s| EntityId::new(*s)).collect::<Vec<_>>();
    let map = CandidateEntityMap::from([(FactId::new("f1"), entity_ids(&["e3", "e4"]))]);
    assert!(filter
        .shared_entities(&entity_ids(&["e1", "e2"]), &map, &FactId::new("f1"))
        .is_empty());
    let map = CandidateEntityMap::from([(FactId::new("f1"), entity_ids(&["e2", "e3"]))]);
    assert_eq!(
        filter.shared_entities(&entity_ids(&["e1", "e2", "e2"]), &map, &FactId::new("f1")),
        entity_ids(&["e2"])
    );
    let map = CandidateEntityMap::from([(FactId::new("f2"), entity_ids(&["e1"]))]);
    assert!(filter.shared_entities(&entity_ids(&["e1"]), &map, &FactId::new("f1")).is_empty());
}

// --- Corrected config keys (handoff bug 3) ---

#[test]
fn exact_priority_with_and_without_assertion_boost() {
    let (new_fact, cand) = meridian_pair();
    // Path A score 1.0, overlap 3/5: 0.8 * (1 + 1.5) * (1 + 0.6 * 0.3) = 2.36.
    let no_boost =
        filter_with(PreFilterConfig { use_assertion_boost: false, ..Default::default() });
    approx(no_boost.compute_priority(&new_fact, &cand, &none(), &none()), 2.36);
    // Default config boosts it: the subjects line up.
    approx(
        default_filter().compute_priority(&new_fact, &cand, &none(), &none()),
        2.36 * ASSERTION_BOOST,
    );
}

#[test]
fn assertion_boost_needs_aligned_assertions() {
    let new_fact = simple_fact("Meridian Construction was founded in 2010");
    let cand = retrieved_triple("Meridian Construction was founded in 2005", "Other", "Thing");
    let on = fallback_filter();
    let off = filter_with(PreFilterConfig { use_assertion_boost: false, ..on.config().clone() });
    let priority_on = on.compute_priority(&new_fact, &cand, &none(), &none());
    assert!(priority_on > 0.0);
    approx(priority_on, off.compute_priority(&new_fact, &cand, &none(), &none()));
}

#[test]
fn require_entity_overlap_skips_disjoint_entity_sets() {
    let (new_fact, cand) = meridian_pair();
    let lenient = default_filter();
    let strict =
        filter_with(PreFilterConfig { require_entity_overlap: true, ..Default::default() });
    // Path A passes regardless of entities unless overlap is required.
    assert!(lenient.compute_priority(&new_fact, &cand, &ids(&["e1"]), &ids(&["e2"])) > 0.0);
    assert_eq!(strict.compute_priority(&new_fact, &cand, &ids(&["e1"]), &ids(&["e2"])), 0.0);
    // Applies only when both sides carry entity ids.
    assert!(strict.compute_priority(&new_fact, &cand, &ids(&["e1"]), &none()) > 0.0);
    assert!(strict.compute_priority(&new_fact, &cand, &none(), &ids(&["e2"])) > 0.0);
    assert!(strict.compute_priority(&new_fact, &cand, &ids(&["e1", "e2"]), &ids(&["e2"])) > 0.0);
}

// --- Embedding paths ---

#[derive(Debug)]
struct FailingEmbedder;

impl Embedder for FailingEmbedder {
    fn embed(&self, _texts: &[&str]) -> Result<Vec<Vec<f32>>, ModelError> {
        Err(ModelError::Inference("offline".into()))
    }

    fn dimension(&self) -> usize {
        3
    }

    fn model_name(&self) -> &str {
        "failing"
    }
}

fn embedding_filter(embedder: Arc<dyn Embedder>, config: PreFilterConfig) -> FactPairPreFilter {
    FactPairPreFilter::new(config, Some(embedder), mappings())
}

#[test]
fn predicate_similarity_without_embedder_uses_jaccard() {
    let filter = default_filter();
    assert_eq!(filter.predicate_similarity("", "founded"), 0.0);
    assert_eq!(filter.predicate_similarity("FOUNDED_IN", " founded in "), 1.0);
    // {"founded", "year"} vs {"founded"}.
    assert_eq!(filter.predicate_similarity("founded_year", "founded"), 0.5);
}

#[test]
fn predicate_similarity_uses_cached_embeddings() {
    let embedder = Arc::new(
        MockEmbedder::new(2)
            .with_vector("founded in", vec![1.0, 0.0])
            .with_vector("established in", vec![1.0, 1.0]),
    );
    let filter = embedding_filter(embedder.clone(), PreFilterConfig::default());
    approx(filter.predicate_similarity("FOUNDED_IN", "established_in"), 0.5_f64.sqrt());
    assert_eq!(embedder.calls(), 2);
    approx(filter.predicate_similarity("founded_in", "Established_In"), 0.5_f64.sqrt());
    assert_eq!(embedder.calls(), 2);
}

#[test]
fn embedding_failure_scores_zero() {
    let filter = embedding_filter(Arc::new(FailingEmbedder), PreFilterConfig::default());
    assert_eq!(filter.predicate_similarity("founded_in", "established_in"), 0.0);
    assert_eq!(filter.text_embedding_similarity("a b", "c d"), 0.0);
    assert_eq!(default_filter().text_embedding_similarity("a b", "c d"), 0.0);
}

#[test]
fn semantic_fallback_uses_text_embeddings() {
    let new_text = "The truck ran the red light";
    let cand_text = "Signal was ignored by the lorry";
    let embedder = Arc::new(
        MockEmbedder::new(2)
            .with_vector(new_text, vec![1.0, 0.0])
            .with_vector(cand_text, vec![0.9, 0.1]),
    );
    let config =
        PreFilterConfig { use_semantic_similarity_fallback: true, ..PreFilterConfig::default() };
    let filter = embedding_filter(embedder, config.clone());
    let new_fact = simple_fact(new_text);
    let cand = retrieved(cand_text);
    let similarity = filter.text_embedding_similarity(new_text, cand_text);
    assert!(similarity > 0.45);
    approx(
        filter.compute_priority(&new_fact, &cand, &none(), &none()),
        0.8 * (1.0 + similarity * 0.8),
    );
    // No embedder: the semantic fallback is off.
    assert_eq!(filter_with(config).compute_priority(&new_fact, &cand, &none(), &none()), 0.0);
}
