//! `SimilarFactRetriever` over `MemoryStore` and `MockEmbedder`: sources, fusion, strategy,
//! `min_score`, cap, exclusion, failure handling and the candidate entity map.

use std::sync::Arc;

use impeach_config::pipeline::FactRetrievalStrategy;
use impeach_config::services::SimilarFactRetrieverConfig;
use impeach_contradictions::{
    CandidateEntityMap, GraphLookupCache, SimilarFactRetriever, SOURCE_BM25, SOURCE_GRAPH,
    SOURCE_VECTOR,
};
use impeach_core::{
    AssertionTriple, EntityId, ExtractedFact, FactId, RetrievedFact, TemporalValidity,
};
use impeach_models::{Embedder, MockEmbedder, ModelError};
use impeach_store::{EntityNode, FactNode, GraphStore, MemoryStore};
use pretty_assertions::assert_eq;
use serde_json::json;

const DIM: usize = 2;
const QUERY: &str = "truck speeding";

/// The store contents shared by the tests.
struct World {
    store: Arc<MemoryStore>,
    entity: EntityId,
    other_entity: EntityId,
    /// Linked to `entity` and a BM25 hit.
    multi: FactId,
    /// Linked to `entity` only.
    graph: FactId,
    /// BM25 hit only.
    text: FactId,
    /// Vector hit only (the only fact with an embedding).
    vector: FactId,
}

fn add_fact(store: &MemoryStore, text: &str, metadata: &str) -> FactId {
    let mut node = FactNode::new(
        text,
        r#"{"subject": "Truck", "relationship": "WAS", "object": "speeding"}"#,
        "{}",
        0.75,
    );
    node.metadata = metadata.to_owned();
    store.create_fact(&node).expect("fact created").id
}

fn world() -> World {
    let store = Arc::new(MemoryStore::new(DIM));
    let entity = store.create_entity(&EntityNode::new("Truck", "VEHICLE")).expect("entity").id;
    let other_entity =
        store.create_entity(&EntityNode::new("Main Street", "LOCATION")).expect("entity").id;
    let multi = add_fact(&store, "The truck was speeding downtown", "{}");
    let graph = add_fact(&store, "The driver stopped at the light", r#"{"side": "plaintiff"}"#);
    let text = add_fact(&store, "A speeding truck hit the fence", "{}");
    let vector = add_fact(&store, "Velocity exceeded posted limits", "{}");
    store.create_fact_embedding(&vector, &[1.0, 0.0]).expect("embedding");
    store.link_fact_to_entity(&multi, &entity, "SUBJECT").expect("link");
    store.link_fact_to_entity(&multi, &other_entity, "OBJECT").expect("link");
    store.link_fact_to_entity(&graph, &entity, "SUBJECT").expect("link");
    World { store, entity, other_entity, multi, graph, text, vector }
}

fn retriever(world: &World, config: SimilarFactRetrieverConfig) -> SimilarFactRetriever {
    let embedder = MockEmbedder::new(DIM).with_vector(QUERY, vec![1.0, 0.0]);
    retriever_with(world, Arc::new(embedder), config)
}

fn retriever_with(
    world: &World,
    embedder: Arc<dyn Embedder>,
    config: SimilarFactRetrieverConfig,
) -> SimilarFactRetriever {
    let store: Arc<dyn GraphStore> = world.store.clone();
    SimilarFactRetriever::new(store, embedder, config)
}

fn new_fact() -> ExtractedFact {
    ExtractedFact::new(
        "STATED_FACT_0",
        QUERY,
        AssertionTriple::new("Truck", "WAS", "speeding"),
        0.9,
        TemporalValidity::default(),
    )
}

fn summary(found: &[RetrievedFact]) -> Vec<(FactId, Vec<String>, f64)> {
    found.iter().map(|f| (f.fact_id.clone(), f.sources.clone(), f.relevance_score)).collect()
}

fn sources(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

fn ids(found: &[RetrievedFact]) -> Vec<FactId> {
    found.iter().map(|f| f.fact_id.clone()).collect()
}

#[test]
fn all_sources_are_fused_and_ranked() {
    let w = world();
    let found = retriever(&w, SimilarFactRetrieverConfig::default())
        .retrieve_similar_facts(
            &new_fact(),
            std::slice::from_ref(&w.entity),
            &[],
            &mut GraphLookupCache::default(),
        )
        .expect("retrieval");
    assert_eq!(
        summary(&found),
        vec![
            (w.multi.clone(), sources(&[SOURCE_GRAPH, SOURCE_BM25]), 1.2),
            (w.graph.clone(), sources(&[SOURCE_GRAPH]), 1.0),
            (w.text.clone(), sources(&[SOURCE_BM25]), 0.9),
            (w.vector.clone(), sources(&[SOURCE_VECTOR]), 0.8),
        ]
    );
}

#[test]
fn retrieved_facts_carry_node_fields() {
    let w = world();
    let found = retriever(&w, SimilarFactRetrieverConfig::default())
        .retrieve_similar_facts(
            &new_fact(),
            std::slice::from_ref(&w.entity),
            &[],
            &mut GraphLookupCache::default(),
        )
        .expect("retrieval");
    let graph = found.iter().find(|f| f.fact_id == w.graph).expect("graph fact found");
    assert_eq!(graph.fact_text, "The driver stopped at the light");
    assert_eq!(graph.assertion_subject.as_deref(), Some("Truck"));
    assert_eq!(graph.assertion_relationship.as_deref(), Some("WAS"));
    assert_eq!(graph.assertion_object.as_deref(), Some("speeding"));
    assert_eq!(graph.side.as_deref(), Some("PLAINTIFF"));
    assert_eq!(
        serde_json::Value::Object(graph.metadata.clone()),
        json!({"side": "plaintiff", "FactType": "STATED", "FactOrigin": "STATED",
               "Status": "RAW", "Certainty": 0.75})
    );
    // Types are not copied from the node (as in Python).
    assert!(!graph.metadata.contains_key("value_type"));
}

#[test]
fn excluded_facts_are_dropped_from_every_source() {
    let w = world();
    let found = retriever(&w, SimilarFactRetrieverConfig::default())
        .retrieve_similar_facts(
            &new_fact(),
            std::slice::from_ref(&w.entity),
            &[w.multi.clone(), w.vector.clone()],
            &mut GraphLookupCache::default(),
        )
        .expect("retrieval");
    assert_eq!(ids(&found), vec![w.graph.clone(), w.text.clone()]);
}

#[test]
fn strategy_selects_text_sources() {
    let w = world();
    let with = |strategy| SimilarFactRetrieverConfig {
        fact_retrieval_strategy: strategy,
        ..SimilarFactRetrieverConfig::default()
    };
    let bm25 = retriever(&w, with(FactRetrievalStrategy::Bm25))
        .retrieve_similar_facts(
            &new_fact(),
            std::slice::from_ref(&w.entity),
            &[],
            &mut GraphLookupCache::default(),
        )
        .expect("retrieval");
    assert_eq!(ids(&bm25), vec![w.multi.clone(), w.graph.clone(), w.text.clone()]);

    let vector = retriever(&w, with(FactRetrievalStrategy::Vector))
        .retrieve_similar_facts(
            &new_fact(),
            std::slice::from_ref(&w.entity),
            &[],
            &mut GraphLookupCache::default(),
        )
        .expect("retrieval");
    assert_eq!(
        summary(&vector),
        vec![
            (w.multi.clone(), sources(&[SOURCE_GRAPH]), 1.0),
            (w.graph.clone(), sources(&[SOURCE_GRAPH]), 1.0),
            (w.vector.clone(), sources(&[SOURCE_VECTOR]), 0.8),
        ]
    );
}

#[test]
fn min_score_and_cap() {
    let w = world();
    let config =
        SimilarFactRetrieverConfig { min_score: 0.85, ..SimilarFactRetrieverConfig::default() };
    let found = retriever(&w, config)
        .retrieve_similar_facts(
            &new_fact(),
            std::slice::from_ref(&w.entity),
            &[],
            &mut GraphLookupCache::default(),
        )
        .expect("retrieval");
    assert_eq!(ids(&found), vec![w.multi.clone(), w.graph.clone(), w.text.clone()]);

    let config =
        SimilarFactRetrieverConfig { max_candidates: 2, ..SimilarFactRetrieverConfig::default() };
    let found = retriever(&w, config)
        .retrieve_similar_facts(
            &new_fact(),
            std::slice::from_ref(&w.entity),
            &[],
            &mut GraphLookupCache::default(),
        )
        .expect("retrieval");
    assert_eq!(ids(&found), vec![w.multi.clone(), w.graph.clone()]);
}

#[test]
fn custom_weights_reorder_sources() {
    let w = world();
    let config = SimilarFactRetrieverConfig {
        graph_weight: 0.1,
        bm25_weight: 0.2,
        vector_weight: 0.9,
        multi_source_boost: 1.0,
        ..SimilarFactRetrieverConfig::default()
    };
    let found = retriever(&w, config)
        .retrieve_similar_facts(
            &new_fact(),
            std::slice::from_ref(&w.entity),
            &[],
            &mut GraphLookupCache::default(),
        )
        .expect("retrieval");
    assert_eq!(
        summary(&found),
        vec![
            (w.vector.clone(), sources(&[SOURCE_VECTOR]), 0.9),
            (w.multi.clone(), sources(&[SOURCE_GRAPH, SOURCE_BM25]), 0.2),
            (w.text.clone(), sources(&[SOURCE_BM25]), 0.2),
            (w.graph.clone(), sources(&[SOURCE_GRAPH]), 0.1),
        ]
    );
}

#[test]
fn entity_lookup_errors_are_skipped() {
    let w = world();
    // A malformed id is a validation error for the memory store: logged, no graph hits.
    let found = retriever(&w, SimilarFactRetrieverConfig::default())
        .retrieve_similar_facts(
            &new_fact(),
            &[EntityId::new("not-a-node"), w.other_entity.clone()],
            &[],
            &mut GraphLookupCache::default(),
        )
        .expect("retrieval");
    assert_eq!(
        summary(&found),
        vec![
            (w.multi.clone(), sources(&[SOURCE_GRAPH, SOURCE_BM25]), 1.2),
            (w.text.clone(), sources(&[SOURCE_BM25]), 0.9),
            (w.vector.clone(), sources(&[SOURCE_VECTOR]), 0.8),
        ]
    );
}

#[derive(Debug)]
struct FailingEmbedder;

impl Embedder for FailingEmbedder {
    fn embed(&self, _texts: &[&str]) -> Result<Vec<Vec<f32>>, ModelError> {
        Err(ModelError::Inference("offline".into()))
    }

    fn dimension(&self) -> usize {
        DIM
    }

    fn model_name(&self) -> &str {
        "failing"
    }
}

#[test]
fn embedding_failure_drops_only_vector_hits() {
    let w = world();
    let found =
        retriever_with(&w, Arc::new(FailingEmbedder), SimilarFactRetrieverConfig::default())
            .retrieve_similar_facts(&new_fact(), &[], &[], &mut GraphLookupCache::default())
            .expect("retrieval");
    assert_eq!(ids(&found), vec![w.multi.clone(), w.text.clone()]);
}

#[test]
fn wrong_dimension_query_is_a_non_fatal_vector_failure() {
    let w = world();
    let embedder = Arc::new(MockEmbedder::new(DIM + 1));
    let found = retriever_with(&w, embedder, SimilarFactRetrieverConfig::default())
        .retrieve_similar_facts(&new_fact(), &[], &[], &mut GraphLookupCache::default())
        .expect("retrieval");
    assert_eq!(ids(&found), vec![w.multi.clone(), w.text.clone()]);
}

#[test]
fn candidate_entity_map_follows_fact_about_entity_edges() {
    let w = world();
    let retriever = retriever(&w, SimilarFactRetrieverConfig::default());
    let found = retriever
        .retrieve_similar_facts(
            &new_fact(),
            std::slice::from_ref(&w.entity),
            &[],
            &mut GraphLookupCache::default(),
        )
        .expect("retrieval");
    let mut candidates = found;
    candidates.push(RetrievedFact::new("not-a-node", "unknown fact"));
    let map = retriever
        .candidate_entity_map(&candidates, &mut GraphLookupCache::default())
        .expect("entity map");
    let expected = CandidateEntityMap::from([
        (w.multi.clone(), vec![w.entity.clone(), w.other_entity.clone()]),
        (w.graph.clone(), vec![w.entity.clone()]),
    ]);
    assert_eq!(map, expected);
    assert!(retriever
        .candidate_entity_map(&[], &mut GraphLookupCache::default())
        .expect("empty map")
        .is_empty());
}
