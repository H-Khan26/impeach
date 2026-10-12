//! `EdRetriever` over `MemoryStore` and `MockEmbedder`: strategies, scores, thresholds,
//! cache and structured-key lookups.

use std::collections::BTreeSet;
use std::sync::Arc;

use ce_config::pipeline::EntityRetrievalStrategy;
use ce_config::MappingsRegistry;
use ce_core::{EdCandidate, EntityId};
use ce_ed::retrieval::{EdRetrieval, EdRetriever, RetrieverSettings, StructuredKeyQuery};
use ce_ed::{CaseCitationKey, NameSimilarity, StatuteKey};
use ce_models::{Embedder, MockEmbedder};
use ce_store::{EntityNode, GraphStore, MemoryStore};
use pretty_assertions::assert_eq;

const DIM: usize = 3;

fn similarity() -> NameSimilarity {
    NameSimilarity::from_mappings(MappingsRegistry::load().expect("embedded mappings load"))
}

fn retriever(
    store: &Arc<MemoryStore>,
    embedder: Option<MockEmbedder>,
    settings: RetrieverSettings,
) -> EdRetriever {
    let store: Arc<dyn GraphStore> = store.clone();
    let embedder = embedder.map(|e| Arc::new(e) as Arc<dyn Embedder>);
    EdRetriever::new(store, embedder, settings, similarity())
}

fn add(store: &MemoryStore, name: &str, entity_type: &str) -> EntityId {
    store.create_entity(&EntityNode::new(name, entity_type)).expect("entity created").id
}

fn add_full(
    store: &MemoryStore,
    name: &str,
    entity_type: &str,
    aliases: &str,
    metadata: &str,
    vector: Option<[f32; DIM]>,
) -> EntityId {
    let node = EntityNode {
        aliases_json: aliases.to_owned(),
        metadata: metadata.to_owned(),
        ..EntityNode::new(name, entity_type)
    };
    let id = store.create_entity(&node).expect("entity created").id;
    if let Some(vector) = vector {
        store.create_entity_embedding(&id, &vector).expect("embedding stored");
    }
    id
}

fn add_vec(store: &MemoryStore, name: &str, entity_type: &str, vector: [f32; DIM]) -> EntityId {
    add_full(store, name, entity_type, "[]", "{}", Some(vector))
}

fn ids(candidates: &[EdCandidate]) -> Vec<&str> {
    candidates.iter().map(|c| c.entity_id.as_str()).collect()
}

fn id_set(candidates: &[EdCandidate]) -> BTreeSet<&str> {
    candidates.iter().map(|c| c.entity_id.as_str()).collect()
}

fn assert_close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
}

fn embedder_for(text: &str, vector: [f32; DIM]) -> MockEmbedder {
    MockEmbedder::new(DIM).with_vector(text, vector.to_vec())
}

#[test]
fn ignores_case_id_and_searches_globally() {
    let store = Arc::new(MemoryStore::new(DIM));
    let name = "Green Valley Logistics, Inc.";
    let aliases = r#"["Green Valley Logistics"]"#;
    let e1 =
        add_full(&store, name, "ORG", aliases, r#"{"case_id": "CASE-1"}"#, Some([1.0, 0.0, 0.0]));
    let e2 =
        add_full(&store, name, "ORG", aliases, r#"{"case_id": "CASE-2"}"#, Some([0.9, 0.1, 0.0]));
    let e3 = add_full(&store, name, "ORG", "[]", "{}", Some([0.8, 0.2, 0.0]));
    let all: BTreeSet<&str> = [e1.as_str(), e2.as_str(), e3.as_str()].into();
    let r =
        retriever(&store, Some(embedder_for(name, [1.0, 0.0, 0.0])), RetrieverSettings::default());

    let exact = r.find_by_exact_match(name, Some("ORG")).expect("exact");
    assert_eq!(id_set(&exact), all);

    let by_vector = r.find_by_embedding(name, 5, 0.0).expect("embedding");
    assert_eq!(id_set(&by_vector), all);

    let first = exact.iter().find(|c| c.entity_id == e1).expect("e1");
    assert_eq!(first.aliases, vec!["Green Valley Logistics".to_owned()]);
    assert_eq!(first.metadata.get("case_id").and_then(|v| v.as_str()), Some("CASE-1"));
}

#[test]
fn bm25_exact_match_filters_by_type() {
    let store = Arc::new(MemoryStore::new(DIM));
    let org = add(&store, "Acme Corp", "ORG");
    let person = add(&store, "Acme Corp", "PERSON");
    add(&store, "Unrelated Name", "ORG");
    let r = retriever(&store, None, RetrieverSettings::default());

    let orgs = r.find_by_exact_match("Acme Corp", Some("ORG")).expect("exact");
    assert_eq!(ids(&orgs), vec![org.as_str()]);
    assert_eq!(orgs[0].source, "exact");
    assert_close(orgs[0].score, 0.0);
    assert_eq!(orgs[0].canonical_name, "Acme Corp");
    assert_eq!(orgs[0].label, "ORG");

    let expected: BTreeSet<&str> = [org.as_str(), person.as_str()].into();
    assert_eq!(id_set(&r.find_by_exact_match("Acme Corp", None).expect("exact")), expected);
    // An empty type is no filter (Python truthiness).
    assert_eq!(id_set(&r.find_by_exact_match("Acme Corp", Some("")).expect("exact")), expected);
    assert!(r.find_by_exact_match("Nothing", Some("ORG")).expect("exact").is_empty());
}

#[test]
fn malformed_aliases_and_metadata_are_tolerated() {
    let store = Arc::new(MemoryStore::new(DIM));
    let id = add_full(&store, "Acme Corp", "ORG", r#"["Acme", 7]"#, "[1, 2]", None);
    let r = retriever(&store, None, RetrieverSettings::default());
    let found = r.find_by_exact_match("Acme Corp", None).expect("exact");
    assert_eq!(ids(&found), vec![id.as_str()]);
    assert_eq!(found[0].aliases, vec!["Acme".to_owned()]);
    assert!(found[0].metadata.is_empty());
}

#[test]
fn embedding_scores_are_real_cosines() {
    let store = Arc::new(MemoryStore::new(DIM));
    let high = add_vec(&store, "Alpha", "ORG", [1.0, 0.0, 0.0]);
    let low = add_vec(&store, "Beta", "ORG", [0.1, 1.0, 0.0]);
    let zero = add_vec(&store, "Gamma", "ORG", [0.0, 0.0, 1.0]);
    let r = retriever(
        &store,
        Some(embedder_for("query", [1.0, 0.0, 0.0])),
        RetrieverSettings::default(),
    );

    // Every candidate keeps its genuine score, even a low or zero one.
    let found = r.find_by_embedding("query", 5, 0.0).expect("embedding");
    assert_eq!(ids(&found), vec![high.as_str(), low.as_str(), zero.as_str()]);
    let hits = store.find_similar_entities(&[1.0, 0.0, 0.0], 5).expect("hits");
    for (candidate, hit) in found.iter().zip(&hits) {
        assert_eq!(candidate.entity_id, hit.node.id);
        assert_close(candidate.score, hit.score);
        assert_eq!(candidate.source, "embedding");
    }
    assert_close(found[0].score, 1.0);
    assert!(found[1].score > 0.0 && found[1].score < 0.2);
    assert_close(found[2].score, 0.0);

    // A positive threshold drops weaker candidates.
    let strong = r.find_by_embedding("query", 5, 0.5).expect("embedding");
    assert_eq!(ids(&strong), vec![high.as_str()]);

    // The limit applies after sorting.
    let top = r.find_by_embedding("query", 2, 0.0).expect("embedding");
    assert_eq!(ids(&top), vec![high.as_str(), low.as_str()]);
}

#[test]
fn embedding_without_embedder_or_with_bad_vector_finds_nothing() {
    let store = Arc::new(MemoryStore::new(DIM));
    add_vec(&store, "Alpha", "ORG", [1.0, 0.0, 0.0]);
    let none = retriever(&store, None, RetrieverSettings::default());
    assert!(none.find_by_embedding("Alpha", 5, 0.0).expect("embedding").is_empty());
    // A vector of the wrong dimension is a non-fatal store error.
    let wrong_dim =
        retriever(&store, Some(MockEmbedder::new(DIM + 1)), RetrieverSettings::default());
    assert!(wrong_dim.find_by_embedding("Alpha", 5, 0.0).expect("embedding").is_empty());
}

#[test]
fn equal_embedding_scores_keep_store_order() {
    let store = Arc::new(MemoryStore::new(DIM));
    let first = add_vec(&store, "One", "ORG", [1.0, 0.0, 0.0]);
    let second = add_vec(&store, "Two", "ORG", [1.0, 0.0, 0.0]);
    let r =
        retriever(&store, Some(embedder_for("q", [1.0, 0.0, 0.0])), RetrieverSettings::default());
    let found = r.find_by_embedding("q", 5, 0.0).expect("embedding");
    let hits = store.find_similar_entities(&[1.0, 0.0, 0.0], 5).expect("hits");
    assert_eq!(ids(&found), hits.iter().map(|h| h.node.id.as_str()).collect::<Vec<_>>());
    assert_eq!(id_set(&found), [first.as_str(), second.as_str()].into());
}

#[test]
fn vector_fuzzy_combines_cosine_and_name_similarity() {
    let store = Arc::new(MemoryStore::new(DIM));
    let lopez = add_vec(&store, "Maria Lopez", "PERSON", [1.0, 0.0, 0.0]);
    let alias = add_full(
        &store,
        "M. L. Unrelated",
        "PERSON",
        r#"["Maria Lopez"]"#,
        "{}",
        Some([0.6, 0.8, 0.0]),
    );
    let other = add_vec(&store, "Zebra Holdings Group", "PERSON", [0.0, 0.0, 1.0]);
    let query = "Maria Lopez";
    let r =
        retriever(&store, Some(embedder_for(query, [0.6, 0.8, 0.0])), RetrieverSettings::default());
    let sim = similarity();

    let found = r.find_by_vector_with_fuzzy(query, None, 10, None).expect("vector_fuzzy");
    let combined = |id: &EntityId, best: f64| {
        let cosine = store
            .find_similar_entities(&[0.6, 0.8, 0.0], 10)
            .expect("hits")
            .into_iter()
            .find(|h| &h.node.id == id)
            .expect("hit")
            .score;
        0.4 * cosine + 0.6 * best
    };
    let lopez_score =
        combined(&lopez, sim.enhanced_string_similarity(query, "Maria Lopez", "PERSON"));
    let alias_best = sim
        .enhanced_string_similarity(query, "M. L. Unrelated", "PERSON")
        .max(sim.enhanced_string_similarity(query, "Maria Lopez", "PERSON"));
    let alias_score = combined(&alias, alias_best);
    let other_score =
        combined(&other, sim.enhanced_string_similarity(query, "Zebra Holdings Group", "PERSON"));
    assert!(other_score < 0.5, "precondition: {other_score}");

    let mut expected = vec![(alias.as_str(), alias_score), (lopez.as_str(), lopez_score)];
    expected.sort_by(|a, b| b.1.total_cmp(&a.1));
    assert_eq!(ids(&found), expected.iter().map(|e| e.0).collect::<Vec<_>>());
    for (candidate, (_, score)) in found.iter().zip(&expected) {
        assert_close(candidate.score, *score);
        assert_eq!(candidate.source, "vector_fuzzy");
    }

    // Threshold: inclusive at the lower score, exclusive above it.
    let lowest = expected[1].1;
    let at = r.find_by_vector_with_fuzzy(query, None, 10, Some(lowest)).expect("vector_fuzzy");
    assert_eq!(at.len(), 2);
    let above = r.find_by_vector_with_fuzzy(query, None, 10, Some(lowest + 1e-9)).expect("vf");
    assert_eq!(ids(&above), vec![expected[0].0]);
    assert!(expected[0].1 > lowest + 1e-9, "precondition: distinct combined scores");
    // Type filter.
    assert!(r.find_by_vector_with_fuzzy(query, Some("ORG"), 10, None).expect("vf").is_empty());
    // Empty surface form.
    assert!(r.find_by_vector_with_fuzzy("", None, 10, None).expect("vf").is_empty());
}

#[test]
fn vector_fuzzy_overfetch_reaches_past_filtered_candidates() {
    let store = Arc::new(MemoryStore::new(DIM));
    add_vec(&store, "Acme", "PERSON", [1.0, 0.0, 0.0]);
    add_vec(&store, "Acme", "PERSON", [0.9, 0.1, 0.0]);
    let org = add_vec(&store, "Acme", "ORG", [0.8, 0.2, 0.0]);
    let embedder = || Some(embedder_for("Acme", [1.0, 0.0, 0.0]));

    let overfetching = retriever(&store, embedder(), RetrieverSettings::default());
    let found =
        overfetching.find_by_vector_with_fuzzy("Acme", Some("ORG"), 1, Some(0.0)).expect("vf");
    assert_eq!(ids(&found), vec![org.as_str()]);

    for multiplier in [0, 1] {
        let settings = RetrieverSettings {
            vector_fuzzy_overfetch_multiplier: multiplier,
            ..Default::default()
        };
        let single = retriever(&store, embedder(), settings);
        let found =
            single.find_by_vector_with_fuzzy("Acme", Some("ORG"), 1, Some(0.0)).expect("vf");
        assert!(found.is_empty(), "multiplier {multiplier}: {found:?}");
    }
}

#[test]
fn vector_fuzzy_initial_threshold_drops_weak_vectors() {
    let store = Arc::new(MemoryStore::new(DIM));
    add_vec(&store, "Acme", "ORG", [0.0, 1.0, 0.0]);
    let embedder = || Some(embedder_for("Acme", [1.0, 0.0, 0.0]));
    let open = retriever(&store, embedder(), RetrieverSettings::default());
    assert_eq!(open.find_by_vector_with_fuzzy("Acme", None, 5, Some(0.0)).expect("vf").len(), 1);
    let settings = RetrieverSettings { initial_vector_min_similarity: 0.5, ..Default::default() };
    let strict = retriever(&store, embedder(), settings);
    assert!(strict.find_by_vector_with_fuzzy("Acme", None, 5, Some(0.0)).expect("vf").is_empty());
}

#[test]
fn configured_strategy_routes_exact_match() {
    let store = Arc::new(MemoryStore::new(DIM));
    let id = add_vec(&store, "Maria Lopez", "PERSON", [1.0, 0.0, 0.0]);
    let embedder = || Some(embedder_for("Maria Lopez", [1.0, 0.0, 0.0]));

    let vector = retriever(
        &store,
        embedder(),
        RetrieverSettings { strategy: EntityRetrievalStrategy::VectorFuzzy, ..Default::default() },
    );
    let found = vector.find_by_exact_match("Maria Lopez", Some("PERSON")).expect("exact");
    assert_eq!(ids(&found), vec![id.as_str()]);
    assert_eq!(found[0].source, "vector_fuzzy");

    let all = retriever(
        &store,
        None,
        RetrieverSettings {
            strategy: EntityRetrievalStrategy::AllEntitiesFuzzy,
            ..Default::default()
        },
    );
    let found = all.find_by_exact_match("Maria Lopez", Some("PERSON")).expect("exact");
    assert_eq!(ids(&found), vec![id.as_str()]);
    assert_eq!(found[0].source, "all_entities_fuzzy");

    // find_candidates picks the strategy per call.
    let bm25 = retriever(&store, embedder(), RetrieverSettings::default());
    let exact = bm25.find_candidates("Maria Lopez", None, 10, None, None).expect("candidates");
    assert_eq!(exact[0].source, "exact");
    let fuzzy = bm25
        .find_candidates("Maria Lopez", None, 10, None, Some(EntityRetrievalStrategy::VectorFuzzy))
        .expect("candidates");
    assert_eq!(fuzzy[0].source, "vector_fuzzy");
}

#[test]
fn all_entities_fuzzy_caches_until_invalidated() {
    let store = Arc::new(MemoryStore::new(DIM));
    let first = add(&store, "Maria Lopez", "PERSON");
    add(&store, "Zebra Holdings Group", "ORG");
    let r = retriever(&store, None, RetrieverSettings::default());
    let expected = similarity().enhanced_string_similarity("Maria Lopez", "Maria Lopez", "PERSON");

    let found =
        r.find_all_entities_with_fuzzy("Maria Lopez", Some("PERSON"), 10, None).expect("all");
    assert_eq!(ids(&found), vec![first.as_str()]);
    assert_close(found[0].score, expected);
    assert_eq!(found[0].source, "all_entities_fuzzy");

    let second = add(&store, "Maria Lopez", "PERSON");
    let cached =
        r.find_all_entities_with_fuzzy("Maria Lopez", Some("PERSON"), 10, None).expect("all");
    assert_eq!(ids(&cached), vec![first.as_str()]);
    assert_close(cached[0].score, expected);

    r.invalidate_cache();
    let fresh =
        r.find_all_entities_with_fuzzy("Maria Lopez", Some("PERSON"), 10, None).expect("all");
    assert_eq!(id_set(&fresh), [first.as_str(), second.as_str()].into());
    let limited =
        r.find_all_entities_with_fuzzy("Maria Lopez", Some("PERSON"), 1, None).expect("all");
    assert_eq!(limited.len(), 1);
    assert!(r.find_all_entities_with_fuzzy("", None, 10, None).expect("all").is_empty());
    assert!(r
        .find_all_entities_with_fuzzy("Maria Lopez", Some("PERSON"), 10, Some(expected + 0.01))
        .expect("all")
        .is_empty());
}

#[test]
fn find_by_similarity_applies_threshold_type_and_limit() {
    let store = Arc::new(MemoryStore::new(DIM));
    let exact = add(&store, "Maria Lopez", "PERSON");
    let partial = add(&store, "Lopez Trucking", "ORG");
    let distant = add(&store, "Lopez", "PERSON");
    let r = retriever(&store, None, RetrieverSettings::default());
    let sim = similarity();
    let query = "Maria Lopez";

    let scores = [
        (exact.as_str(), sim.enhanced_string_similarity(query, "Maria Lopez", "PERSON")),
        (partial.as_str(), sim.enhanced_string_similarity(query, "Lopez Trucking", "ORG")),
        (distant.as_str(), sim.enhanced_string_similarity(query, "Lopez", "PERSON")),
    ];
    for min in [0.0, 0.75, 0.95] {
        let found = r.find_by_similarity(query, None, 10, min).expect("similarity");
        let expected: BTreeSet<&str> =
            scores.iter().filter(|(_, s)| *s >= min).map(|(id, _)| *id).collect();
        assert_eq!(id_set(&found), expected, "min {min}");
        assert!(found.windows(2).all(|w| w[0].score >= w[1].score));
        assert!(found.iter().all(|c| c.source == "similarity" && c.score >= min));
    }
    let people = r.find_by_similarity(query, Some("PERSON"), 10, 0.0).expect("similarity");
    assert!(people.iter().all(|c| c.label == "PERSON"));
    assert_eq!(people.len(), 2);
    assert_eq!(ids(&r.find_by_similarity(query, None, 1, 0.0).expect("similarity")).len(), 1);
    assert_eq!(ids(&r.find_by_similarity(query, None, 1, 0.0).expect("s"))[0], exact.as_str());
    assert!(r.find_by_similarity("", None, 10, 0.0).expect("similarity").is_empty());
}

#[test]
fn docket_lookup_searches_docket_entries_then_cases() {
    let store = Arc::new(MemoryStore::new(DIM));
    let case = add(&store, "Case No. 2:23-cv-01492", "CASE");
    let docket = add(&store, "Docket Entry 2:23-cv-01492", "DOCKET_ENTRY");
    add(&store, "No. 2:23-cv-01492", "ORG");
    let r = retriever(&store, None, RetrieverSettings::default());

    let found = r
        .find_by_structured_key(&StructuredKeyQuery::DocketEntry { number: "2:23-cv-01492" }, None)
        .expect("docket");
    assert_eq!(ids(&found), vec![docket.as_str(), case.as_str()]);
    assert!(found.iter().all(|c| c.source == "structured_docket"));
    assert!(r
        .find_by_structured_key(&StructuredKeyQuery::DocketEntry { number: "" }, None)
        .expect("docket")
        .is_empty());
}

#[test]
fn exhibit_lookup_uses_exhibit_types() {
    let store = Arc::new(MemoryStore::new(DIM));
    let exhibit = add(&store, "Exhibit 12A", "EXHIBIT");
    let exhibit_id = add(&store, "Ex. 12A", "EXHIBIT_ID");
    add(&store, "Exhibit 12A", "DOCUMENT");
    let r = retriever(&store, None, RetrieverSettings::default());

    let query = StructuredKeyQuery::Exhibit { id: "12a", party_attribution: Some("Plaintiff") };
    let found = r.find_by_structured_key(&query, Some("CASE-1")).expect("exhibit");
    assert_eq!(id_set(&found), [exhibit.as_str(), exhibit_id.as_str()].into());
    assert_eq!(found.len(), 2);
    assert!(found.iter().all(|c| c.source == "structured_exhibit"));
    let empty = StructuredKeyQuery::Exhibit { id: "", party_attribution: None };
    assert!(r.find_by_structured_key(&empty, None).expect("exhibit").is_empty());
}

#[test]
fn statute_lookup_uses_statute_type() {
    let store = Arc::new(MemoryStore::new(DIM));
    let statute = add(&store, "42 U.S.C. 1983", "STATUTE");
    add(&store, "42 U.S.C. 1983", "CASE");
    let r = retriever(&store, None, RetrieverSettings::default());

    let key = StatuteKey {
        title: Some("42".into()),
        section: Some("1983".into()),
        state: Some("ignored".into()),
        ..StatuteKey::default()
    };
    let found =
        r.find_by_structured_key(&StructuredKeyQuery::Statute(&key), None).expect("statute");
    assert_eq!(ids(&found), vec![statute.as_str()]);
    assert_eq!(found[0].source, "structured_statute");

    let no_section = StatuteKey { title: Some("42".into()), ..StatuteKey::default() };
    assert!(r
        .find_by_structured_key(&StructuredKeyQuery::Statute(&no_section), None)
        .expect("statute")
        .is_empty());
}

#[test]
fn case_citation_lookup_uses_case_type() {
    let store = Arc::new(MemoryStore::new(DIM));
    let caption = add(&store, "Smith v. Jones", "CASE");
    let citation = add(&store, "123 F.3d 456", "CASE");
    add(&store, "Smith", "PERSON");
    let r = retriever(&store, None, RetrieverSettings::default());

    let by_caption = CaseCitationKey {
        plaintiff: Some("Smith".into()),
        defendant: Some("Jones".into()),
        ..CaseCitationKey::default()
    };
    let found = r
        .find_by_structured_key(&StructuredKeyQuery::CaseCitation(&by_caption), None)
        .expect("caption");
    assert_eq!(ids(&found), vec![caption.as_str()]);
    assert_eq!(found[0].source, "structured_case");

    let by_citation = CaseCitationKey {
        volume: Some("123".into()),
        reporter: Some("F.3d".into()),
        page: Some("456".into()),
        court: Some("9th Cir.".into()),
        ..CaseCitationKey::default()
    };
    let found = r
        .find_by_structured_key(&StructuredKeyQuery::CaseCitation(&by_citation), None)
        .expect("citation");
    assert_eq!(found[0].entity_id, citation);
    assert!(found.iter().all(|c| c.label == "CASE" && c.source == "structured_case"));

    assert!(r
        .find_by_structured_key(
            &StructuredKeyQuery::CaseCitation(&CaseCitationKey::default()),
            None
        )
        .expect("empty")
        .is_empty());
}

#[test]
fn settings_from_config_map_retrieval_fields() {
    let config = ce_config::load_default_config().expect("default config loads");
    let Some(ce_config::StageConfig::Ed(ed)) = config.get_stage_config("EDStage") else {
        panic!("EDStage config expected");
    };
    let mut ed = ed.clone();
    ed.entity_retrieval_strategy = EntityRetrievalStrategy::VectorFuzzy;
    ed.max_candidates = 7;
    ed.min_embedding_similarity = 0.3;
    ed.vector_fuzzy_overfetch_multiplier = 5;
    ed.vector_fuzzy_min_similarity = 0.6;
    ed.vector_fuzzy_vector_weight = 0.3;
    ed.vector_fuzzy_fuzzy_weight = 0.7;
    assert_eq!(
        RetrieverSettings::from_config(&ed),
        RetrieverSettings {
            strategy: EntityRetrievalStrategy::VectorFuzzy,
            max_candidates: 7,
            vector_fuzzy_overfetch_multiplier: 5,
            vector_fuzzy_min_similarity: 0.6,
            vector_fuzzy_vector_weight: 0.3,
            vector_fuzzy_fuzzy_weight: 0.7,
            initial_vector_min_similarity: 0.3,
        }
    );
}
