//! Fact indexing against a `MemoryStore`: Python-format node properties, embeddings aligned
//! with their facts across embedder batches (the 64+ facts crash), context edges and the
//! commit path used by the fact and contradiction stages.

use std::sync::{Arc, Mutex};

use ce_config::MappingsRegistry;
use ce_core::{
    AssertionTriple, EntityId, ExtractedFact, FactId, NodeId, SpanId, TemporalValidity,
    TypeVocabulary,
};
use ce_facts::{fact_node, FactContextLinks, FactIndexer, FACT_EMBED_BATCH_SIZE};
use ce_models::{Embedder, MockEmbedder, ModelError};
use ce_store::graph::{FACT, FACT_ABOUT_ENTITY, FACT_DERIVED_FROM_SPAN, FACT_IN_DOCUMENT};
use ce_store::{ChunkNode, DocumentNode, EntityNode, FactNode, GraphStore, MemoryStore, SpanNode};
use pretty_assertions::assert_eq;
use serde_json::{json, Value};

const DIM: usize = 16;

fn vocab() -> &'static dyn TypeVocabulary {
    &MappingsRegistry::load().expect("mappings load").taxonomy
}

/// Embedder recording the size of every batch; vectors come from a `MockEmbedder`. Batches
/// listed in `short_batches` (0-based call index) return one vector too few; batches in
/// `failing_batches` fail.
#[derive(Debug)]
struct CountingEmbedder {
    inner: MockEmbedder,
    batches: Mutex<Vec<usize>>,
    short_batches: Vec<usize>,
    failing_batches: Vec<usize>,
}

impl CountingEmbedder {
    fn new() -> Self {
        Self {
            inner: MockEmbedder::new(DIM),
            batches: Mutex::new(Vec::new()),
            short_batches: Vec::new(),
            failing_batches: Vec::new(),
        }
    }

    fn batches(&self) -> Vec<usize> {
        self.batches.lock().expect("lock").clone()
    }
}

impl Embedder for CountingEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, ModelError> {
        let call = {
            let mut batches = self.batches.lock().expect("lock");
            batches.push(texts.len());
            batches.len() - 1
        };
        if self.failing_batches.contains(&call) {
            return Err(ModelError::Inference("embedder down".to_owned()));
        }
        let mut vectors = self.inner.embed(texts)?;
        if self.short_batches.contains(&call) {
            vectors.pop();
        }
        Ok(vectors)
    }

    fn dimension(&self) -> usize {
        DIM
    }

    fn model_name(&self) -> &str {
        "counting"
    }
}

struct Setup {
    store: Arc<MemoryStore>,
    embedder: Arc<CountingEmbedder>,
    indexer: FactIndexer,
}

fn setup_with(embedder: CountingEmbedder) -> Setup {
    let store = Arc::new(MemoryStore::new(DIM));
    let embedder = Arc::new(embedder);
    let dyn_store: Arc<dyn GraphStore> = store.clone();
    let indexer = FactIndexer::new(dyn_store, embedder.clone());
    Setup { store, embedder, indexer }
}

fn setup() -> Setup {
    setup_with(CountingEmbedder::new())
}

fn fact(index: usize) -> ExtractedFact {
    ExtractedFact::new(
        format!("STATED_FACT_{index}"),
        format!("Fact number {index} about the collision on Mission Street."),
        AssertionTriple::new("Daniel Reed", "DROVE", format!("vehicle {index}")),
        0.9,
        TemporalValidity::default(),
    )
}

fn facts(count: usize) -> Vec<ExtractedFact> {
    (0..count).map(fact).collect()
}

/// Asserts every written fact carries the vector of its own text.
fn assert_aligned(setup: &Setup, facts: &[ExtractedFact], ids: &[Option<FactId>]) {
    for (fact, id) in facts.iter().zip(ids) {
        let id = id.as_ref().expect("fact written");
        let stored = setup.store.embedding(id.as_str()).expect("fact has a vector");
        assert_eq!(stored, setup.embedder.inner.vector_for(&fact.text), "fact {}", fact.id);
    }
}

#[test]
fn fact_node_uses_python_json_and_normalised_types() {
    let mut fact = ExtractedFact::new(
        "STATED_FACT_0",
        "José drove the truck",
        AssertionTriple::new("José", "DROVE", "truck"),
        0.8,
        TemporalValidity { t_valid_from: Some("2024-03-14".to_owned()), t_valid_until: None },
    );
    fact.metadata.insert("source_chunk_id".to_owned(), json!("12"));
    fact.metadata.insert("is_denial".to_owned(), json!(false));
    fact.metadata.insert("source_span_ids".to_owned(), json!(["3", "4"]));
    fact.value_type = Some("text".to_owned());
    fact.property_type = Some("EXISTS".to_owned());

    let node = fact_node(&fact, vocab());
    assert_eq!(
        node,
        FactNode {
            fact_text: "José drove the truck".to_owned(),
            fact_origin: "STATED".to_owned(),
            fact_type: "STATED".to_owned(),
            status: "RAW".to_owned(),
            assertion_triple:
                r#"{"subject": "Jos\u00e9", "relationship": "DROVE", "object": "truck"}"#.to_owned(),
            temporal_validity: r#"{"t_valid_from": "2024-03-14", "t_valid_until": null}"#
                .to_owned(),
            certainty: 0.8,
            metadata:
                r#"{"source_chunk_id": "12", "is_denial": false, "source_span_ids": ["3", "4"]}"#
                    .to_owned(),
            value_type: "TEXT".to_owned(),
            property_type: "EXISTENCE".to_owned(),
        }
    );
}

#[test]
fn fact_node_defaults_unknown_types_to_the_taxonomy_defaults() {
    let mut fact = fact(0);
    fact.value_type = Some("NOT_A_TYPE".to_owned());
    fact.property_type = None;
    let node = fact_node(&fact, vocab());
    assert_eq!(node.value_type, vocab().default_value_type());
    assert_eq!(node.property_type, vocab().default_property_type());
    assert_eq!(node.metadata, "{}");
}

#[test]
fn index_facts_writes_nodes_and_searchable_embeddings() {
    let setup = setup();
    let facts = facts(3);
    let nodes: Vec<FactNode> = facts.iter().map(|f| fact_node(f, vocab())).collect();
    let result = setup.indexer.index_facts(&nodes).expect("indexing succeeds");
    assert_eq!(result.count, 3);
    assert_eq!(result.fact_ids.len(), 3);
    for (fact, id) in facts.iter().zip(&result.fact_ids) {
        let id = id.as_ref().expect("written");
        let (label, properties) = setup.store.node_snapshot(id.as_str()).expect("node exists");
        assert_eq!(label, FACT);
        assert_eq!(properties["FactText"], Value::String(fact.text.clone()));
        assert_eq!(properties["Status"], "RAW");
        let hits = setup
            .store
            .find_similar_facts(&setup.embedder.inner.vector_for(&fact.text), 1)
            .expect("search works");
        assert_eq!(&hits[0].node.id, id);
        assert!(hits[0].score > 0.99, "score {}", hits[0].score);
    }
    assert_eq!(setup.embedder.batches(), vec![3]);
}

#[test]
fn index_facts_of_nothing_writes_nothing() {
    let setup = setup();
    let result = setup.indexer.index_facts(&[]).expect("indexing succeeds");
    assert_eq!(result.count, 0);
    assert!(result.fact_ids.is_empty());
    assert!(setup.embedder.batches().is_empty());
    let mut none: Vec<ExtractedFact> = Vec::new();
    assert_eq!(setup.indexer.commit_facts(&mut none, vocab()).expect("commit").count, 0);
}

#[test]
fn invalid_node_is_skipped_without_stopping_the_others() {
    let setup = setup();
    let mut nodes: Vec<FactNode> = facts(3).iter().map(|f| fact_node(f, vocab())).collect();
    nodes[1].fact_text = String::new();
    let result = setup.indexer.index_facts(&nodes).expect("indexing succeeds");
    assert_eq!(result.count, 2);
    assert!(result.fact_ids[0].is_some());
    assert_eq!(result.fact_ids[1], None);
    assert!(result.fact_ids[2].is_some());
    assert_eq!(setup.embedder.batches(), vec![2]);
}

#[test]
fn hundred_facts_commit_in_aligned_batches() {
    // Regression: Python crashed on documents with 64+ facts.
    let setup = setup();
    let mut facts = facts(100);
    let result = setup.indexer.commit_facts(&mut facts, vocab()).expect("commit succeeds");
    assert_eq!(result.count, 100);
    assert_eq!(result.fact_ids.iter().flatten().count(), 100);
    assert_eq!(setup.embedder.batches(), vec![FACT_EMBED_BATCH_SIZE, 36]);
    assert_aligned(&setup, &facts, &result.fact_ids);
    for (fact, id) in facts.iter().zip(&result.fact_ids) {
        let id = id.as_ref().expect("written");
        assert_eq!(fact.metadata["helix_fact_id"], Value::String(id.to_string()));
    }
}

#[test]
fn batch_size_boundaries() {
    for (count, expected) in [(63, vec![63]), (64, vec![64]), (65, vec![64, 1])] {
        let setup = setup();
        let mut facts = facts(count);
        let result = setup.indexer.commit_facts(&mut facts, vocab()).expect("commit succeeds");
        assert_eq!(result.count, count, "{count} facts");
        assert_eq!(setup.embedder.batches(), expected, "{count} facts");
        assert_aligned(&setup, &facts, &result.fact_ids);
    }
}

#[test]
fn custom_batch_size_is_at_least_one() {
    let setup = setup();
    let indexer = FactIndexer::new(setup.store.clone(), setup.embedder.clone()).with_batch_size(0);
    let nodes: Vec<FactNode> = facts(3).iter().map(|f| fact_node(f, vocab())).collect();
    let result = indexer.index_facts(&nodes).expect("indexing succeeds");
    assert_eq!(result.count, 3);
    assert_eq!(setup.embedder.batches(), vec![1, 1, 1]);
}

#[test]
fn wrong_vector_count_leaves_that_batch_unembedded() {
    let mut embedder = CountingEmbedder::new();
    embedder.short_batches = vec![1];
    let setup = setup_with(embedder);
    let mut facts = facts(100);
    let result = setup.indexer.commit_facts(&mut facts, vocab()).expect("commit succeeds");
    assert_eq!(result.count, 64);
    assert_eq!(result.fact_ids.iter().flatten().count(), 100, "nodes are still written");
    assert_aligned(&setup, &facts[..64], &result.fact_ids[..64]);
    for id in result.fact_ids[64..].iter().flatten() {
        assert_eq!(setup.store.embedding(id.as_str()), None);
    }
}

#[test]
fn embedder_error_leaves_facts_unembedded() {
    let mut embedder = CountingEmbedder::new();
    embedder.failing_batches = vec![0];
    let setup = setup_with(embedder);
    let mut facts = facts(70);
    let result = setup.indexer.commit_facts(&mut facts, vocab()).expect("commit succeeds");
    // The first batch fails; the second (6 facts) is embedded.
    assert_eq!(result.count, 6);
    assert_eq!(setup.embedder.batches(), vec![64, 6]);
    assert_aligned(&setup, &facts[64..], &result.fact_ids[64..]);
}

/// A document with one chunk, two sentence spans and two entities.
struct Graph {
    doc_id: NodeId,
    spans: Vec<SpanId>,
    maria: EntityId,
    acme: EntityId,
}

fn graph(store: &MemoryStore) -> Graph {
    let doc_id = store.create_document(&DocumentNode::default()).expect("document").id;
    let chunk = ChunkNode { text: "Maria Lopez sued Acme.".to_owned(), ..ChunkNode::default() };
    let chunk_id = store.create_document_chunk(&doc_id, &chunk).expect("chunk").id;
    let spans = (0..2)
        .map(|i| {
            let span = SpanNode {
                doc_id: doc_id.clone(),
                chunk_id: chunk_id.clone(),
                start_char: i * 10,
                end_char: i * 10 + 5,
                span_type: "SENTENCE".to_owned(),
                ..SpanNode::default()
            };
            store.create_span_for_chunk(&span).expect("span").id
        })
        .collect();
    let maria = store.create_entity(&EntityNode::new("Maria Lopez", "PER")).expect("entity").id;
    let acme = store.create_entity(&EntityNode::new("Acme", "ORG")).expect("entity").id;
    Graph { doc_id, spans, maria, acme }
}

fn edges(store: &MemoryStore, label: &str, from: &FactId) -> Vec<(String, Value)> {
    store
        .edges()
        .into_iter()
        .filter(|edge| edge.label == label && edge.from == from.as_str())
        .map(|edge| (edge.to, Value::Object(edge.properties)))
        .collect()
}

#[test]
fn commit_facts_writes_context_edges() {
    let setup = setup();
    let graph = graph(&setup.store);
    let mut linked = fact(0);
    linked.source_doc_id = Some(graph.doc_id.as_str().into());
    linked.subject_entity_id = Some(graph.maria.clone());
    linked.object_entity_id = Some(graph.acme.clone());
    let span_ids: Vec<Value> = graph.spans.iter().map(|s| json!(s.as_str())).collect();
    linked.metadata.insert("source_span_ids".to_owned(), Value::Array(span_ids));
    // Span ids encoded as a JSON string (tolerated like Python).
    let mut encoded = fact(1);
    encoded.source_doc_id = Some(graph.doc_id.as_str().into());
    encoded.metadata.insert(
        "source_span_ids".to_owned(),
        Value::String(format!("[\"{}\"]", graph.spans[1].as_str())),
    );
    let mut facts = vec![linked, encoded];

    let result = setup.indexer.commit_facts(&mut facts, vocab()).expect("commit succeeds");
    let first = result.fact_ids[0].clone().expect("written");
    let second = result.fact_ids[1].clone().expect("written");

    assert_eq!(
        edges(&setup.store, FACT_IN_DOCUMENT, &first),
        vec![(graph.doc_id.to_string(), json!({"PageNumber": 0, "Section": ""}))]
    );
    assert_eq!(
        edges(&setup.store, FACT_ABOUT_ENTITY, &first),
        vec![
            (graph.maria.to_string(), json!({"Role": "SUBJECT"})),
            (graph.acme.to_string(), json!({"Role": "OBJECT"})),
        ]
    );
    assert_eq!(
        edges(&setup.store, FACT_DERIVED_FROM_SPAN, &first),
        graph
            .spans
            .iter()
            .map(|s| (s.to_string(), json!({"Role": "SOURCE_SPAN", "Confidence": 1.0})))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        edges(&setup.store, FACT_DERIVED_FROM_SPAN, &second),
        vec![(graph.spans[1].to_string(), json!({"Role": "SOURCE_SPAN", "Confidence": 1.0}))]
    );
    assert!(edges(&setup.store, FACT_ABOUT_ENTITY, &second).is_empty());

    let maria_facts = setup.store.get_entity_facts(&graph.maria).expect("query works");
    assert_eq!(maria_facts.len(), 1);
    assert_eq!(maria_facts[0].id, first);
    assert_eq!(facts[0].metadata["helix_fact_id"], json!(first.as_str()));
    // The stored node holds the metadata as it was before the commit.
    let (_, properties) = setup.store.node_snapshot(first.as_str()).expect("node");
    let stored: Value =
        serde_json::from_str(properties["Metadata"].as_str().expect("string")).expect("json");
    assert!(stored.get("helix_fact_id").is_none());
}

#[test]
fn missing_edge_targets_are_logged_not_fatal() {
    let setup = setup();
    let mut orphan = fact(0);
    orphan.source_doc_id = Some("no-such-doc".into());
    orphan.subject_entity_id = Some("999".into());
    orphan.metadata.insert("source_span_ids".to_owned(), json!(["998", 5]));
    let mut facts = vec![orphan, fact(1)];
    let result = setup.indexer.commit_facts(&mut facts, vocab()).expect("commit succeeds");
    assert_eq!(result.count, 2);
    assert!(setup.store.edges().iter().all(|edge| edge.label != FACT_IN_DOCUMENT));

    let id = result.fact_ids[1].clone().expect("written");
    let links = FactContextLinks {
        doc_id: Some(&NodeId::new("424242")),
        object_entity_id: Some(&EntityId::new("424243")),
        ..FactContextLinks::default()
    };
    setup.indexer.link_fact_context(&id, &links).expect("missing targets are not fatal");
}

#[test]
fn facts_that_fail_validation_get_no_edges_or_helix_id() {
    let setup = setup();
    let graph = graph(&setup.store);
    let mut invalid = fact(0);
    invalid.certainty = 1.5;
    invalid.subject_entity_id = Some(graph.maria.clone());
    let mut facts = vec![invalid, fact(1)];
    let result = setup.indexer.commit_facts(&mut facts, vocab()).expect("commit succeeds");
    assert_eq!(result.fact_ids[0], None);
    assert_eq!(result.count, 1);
    assert!(!facts[0].metadata.contains_key("helix_fact_id"));
    assert!(facts[1].metadata.contains_key("helix_fact_id"));
    assert!(setup.store.get_entity_facts(&graph.maria).expect("query").is_empty());
}
