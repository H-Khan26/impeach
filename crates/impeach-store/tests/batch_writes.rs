//! Contract of the batch write methods (`create_entity_with_embedding`,
//! `link_spans_to_entities_batch`, `create_facts_batch`, `link_facts_batch`), run against
//! `MemoryStore`, the trait's default implementations and (feature `helix`) `HelixStore` in
//! memory mode.

use std::sync::Arc;

use impeach_core::{ChunkId, EntityId, FactId, NodeId, SpanId};
use impeach_store::graph::{
    EntityHit, FactHit, FactLink, SpanEntityLink, StoredChunk, StoredDocument, StoredEntity,
    StoredFact, StoredSpan,
};
use impeach_store::memory::EdgeRecord;
use impeach_store::{
    ChunkNode, DocumentNode, EmbeddingWrite, EntityNode, FactNode, GraphStore, MemoryStore,
    SpanNode, StoreError, VectorLabel,
};
use pretty_assertions::assert_eq;

const DIM: usize = 4;
const X: [f32; DIM] = [1.0, 0.0, 0.0, 0.0];
const Y: [f32; DIM] = [0.0, 1.0, 0.0, 0.0];

fn memory_store() -> Arc<dyn GraphStore> {
    Arc::new(MemoryStore::new(DIM))
}

fn default_methods_store() -> Arc<dyn GraphStore> {
    Arc::new(DefaultMethods(MemoryStore::new(DIM)))
}

#[cfg(feature = "helix")]
fn helix_store() -> Arc<dyn GraphStore> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let namespace = format!("batch-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::SeqCst));
    // A small batch size makes the batches below span several transactions.
    let options = impeach_store::HelixOptions {
        batch_size: 3,
        ..impeach_store::HelixOptions::memory(namespace, DIM)
    };
    impeach_store::HelixStore::open(&options).expect("in-memory Helix database opens")
}

fn is_validation(error: &StoreError, expected_field: &str) -> bool {
    matches!(error, StoreError::Validation { field, .. } if field == expected_field)
}

fn is_not_found(error: &StoreError) -> bool {
    matches!(error, StoreError::NotFound { .. })
}

fn fact(text: &str) -> FactNode {
    FactNode::new(text, r#"{"subject": "A"}"#, r#"{"t_valid_from": null}"#, 0.9)
}

/// A document with one chunk and `spans` sentence spans.
fn document_with_spans(store: &dyn GraphStore, spans: usize) -> (NodeId, Vec<SpanId>) {
    let doc = store.create_document(&DocumentNode::default()).expect("document");
    let chunk = store
        .create_document_chunk(&doc.id, &ChunkNode { text: "text".into(), ..ChunkNode::default() })
        .expect("chunk");
    let span_ids = (0..spans)
        .map(|i| {
            let node = SpanNode {
                doc_id: doc.id.clone(),
                chunk_id: chunk.id.clone(),
                start_char: i as i64 * 10,
                ..SpanNode::default()
            };
            store.create_span_for_chunk(&node).expect("span").id
        })
        .collect();
    (doc.id, span_ids)
}

fn entity_with_embedding(store: &dyn GraphStore) {
    let mut acme = EntityNode::new("Acme Corp", "ORG");
    acme.aliases_json = r#"["Acme"]"#.into();
    let stored = store.create_entity_with_embedding(&acme, &X).expect("entity with vector");
    assert_eq!(stored.node, acme);
    let hits = store.find_similar_entities(&X, 5).expect("vector search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].node.id, stored.id);
    assert!((hits[0].score - 1.0).abs() < 1e-3, "{}", hits[0].score);
    let by_name = store.find_entity_by_name("acme").expect("text search");
    assert_eq!(by_name[0].node.id, stored.id);

    // Empty aliases are stored as `[]`, as in `create_entity`.
    let mut bare = EntityNode::new("Bare", "PER");
    bare.aliases_json = String::new();
    let bare = store.create_entity_with_embedding(&bare, &Y).expect("bare entity");
    assert_eq!(bare.node.aliases_json, "[]");

    let error = store
        .create_entity_with_embedding(&EntityNode::new("Short", "ORG"), &[1.0, 0.0])
        .expect_err("wrong dimension");
    assert!(is_validation(&error, "embedding_vector"), "{error:?}");
    let error = store
        .create_entity_with_embedding(&EntityNode::new("Zero", "ORG"), &[0.0; DIM])
        .expect_err("zero vector");
    assert!(is_validation(&error, "embedding_vector"), "{error:?}");
    let error =
        store.create_entity_with_embedding(&EntityNode::new("", "ORG"), &X).expect_err("no name");
    assert!(is_validation(&error, "Name"), "{error:?}");
    assert_eq!(store.get_all_entities(100).expect("entities").len(), 2, "failures write nothing");
}

fn link<'a>(span: &'a SpanId, entity: &'a EntityId, confidence: f64) -> SpanEntityLink<'a> {
    SpanEntityLink { span_id: span, entity_id: entity, confidence, method: "NER+ED" }
}

fn span_entity_links(store: &dyn GraphStore) {
    let (_, spans) = document_with_spans(store, 3);
    let first = store.create_entity(&EntityNode::new("First", "ORG")).expect("entity").id;
    let second = store.create_entity(&EntityNode::new("Second", "ORG")).expect("entity").id;

    assert_eq!(store.link_spans_to_entities_batch(&[]).expect("empty"), 0);
    let missing_span = SpanId::new("987654");
    let error = store
        .link_spans_to_entities_batch(&[
            link(&spans[0], &first, 0.85),
            link(&missing_span, &first, 0.85),
        ])
        .expect_err("missing span");
    assert!(is_not_found(&error), "{error:?}");
    let as_span = SpanId::new(first.as_str());
    let error = store
        .link_spans_to_entities_batch(&[link(&as_span, &second, 0.85)])
        .expect_err("an entity is not a span");
    assert!(is_not_found(&error), "{error:?}");
    let error = store
        .link_spans_to_entities_batch(&[link(&spans[0], &first, 1.5)])
        .expect_err("confidence");
    assert!(is_validation(&error, "confidence"), "{error:?}");
    let bad_id = EntityId::new("not an id");
    let error = store
        .link_spans_to_entities_batch(&[link(&spans[0], &bad_id, 0.5)])
        .expect_err("malformed id");
    assert!(is_validation(&error, "entity_id"), "{error:?}");

    let links = [
        link(&spans[0], &first, 0.85),
        link(&spans[1], &second, 0.5),
        link(&spans[2], &first, 0.85),
        link(&spans[0], &second, 0.85),
    ];
    assert_eq!(store.link_spans_to_entities_batch(&links).expect("links"), 4);
}

fn facts_batch(store: &dyn GraphStore) {
    assert!(store.create_facts_batch(&[]).expect("empty").is_empty());
    let mut invalid = fact("zebra crossing");
    invalid.certainty = 2.0;
    let error = store.create_facts_batch(&[fact("zebra stripes"), invalid]).expect_err("certainty");
    assert!(is_validation(&error, "Certainty"), "{error:?}");
    assert!(
        store.find_facts_by_text("zebra", 10).expect("search").is_empty(),
        "a failed batch writes nothing"
    );

    let inputs: Vec<FactNode> = (0..8).map(|i| fact(&format!("fact number {i} paid"))).collect();
    let stored = store.create_facts_batch(&inputs).expect("facts");
    assert_eq!(stored.iter().map(|s| s.node.clone()).collect::<Vec<_>>(), inputs, "input order");
    let ids: std::collections::BTreeSet<&FactId> = stored.iter().map(|s| &s.id).collect();
    assert_eq!(ids.len(), inputs.len(), "distinct ids");
    let hits = store.find_facts_by_text("number 3", 10).expect("text index");
    assert!(hits.iter().any(|hit| hit.node.id == stored[3].id));
    store.create_fact_embedding(&stored[0].id, &X).expect("facts accept vectors");
}

/// `atomic`: a failed batch writes nothing (the trait defaults write link by link).
fn fact_links_batch(store: &dyn GraphStore, atomic: bool) {
    let (doc, spans) = document_with_spans(store, 1);
    let first = store.create_entity(&EntityNode::new("First", "ORG")).expect("entity").id;
    let second = store.create_entity(&EntityNode::new("Second", "ORG")).expect("entity").id;
    let facts = store.create_facts_batch(&[fact("one"), fact("two")]).expect("facts");
    let (f0, f1) = (&facts[0].id, &facts[1].id);

    assert_eq!(store.link_facts_batch(&[]).expect("empty"), 0);
    let links = [
        FactLink::Document { fact_id: f0, doc_id: &doc, page_number: 3, section: "Facts" },
        FactLink::Entity { fact_id: f0, entity_id: &first, role: "SUBJECT" },
        FactLink::Entity { fact_id: f0, entity_id: &second, role: "OBJECT" },
        FactLink::Span { fact_id: f0, span_id: &spans[0], role: "", confidence: 1.0 },
        FactLink::Entity { fact_id: f1, entity_id: &first, role: "" },
    ];
    assert_eq!(store.link_facts_batch(&links).expect("links"), 5);
    assert_eq!(store.get_fact_entities(f0).expect("entities"), vec![first.clone(), second.clone()]);
    let mut linked: Vec<FactId> =
        store.get_entity_facts(&first).expect("facts").into_iter().map(|f| f.id).collect();
    linked.sort();
    let mut expected = vec![f0.clone(), f1.clone()];
    expected.sort();
    assert_eq!(linked, expected);

    let missing = EntityId::new("987654");
    let error = store
        .link_facts_batch(&[
            FactLink::Entity { fact_id: f1, entity_id: &second, role: "OBJECT" },
            FactLink::Entity { fact_id: f1, entity_id: &missing, role: "OBJECT" },
        ])
        .expect_err("missing entity");
    assert!(is_not_found(&error), "{error:?}");
    if atomic {
        assert_eq!(
            store.get_fact_entities(f1).expect("entities"),
            vec![first.clone()],
            "a failed batch writes nothing"
        );
    }
    let missing_doc = NodeId::new("987654");
    let error = store
        .link_facts_batch(&[FactLink::Document {
            fact_id: f1,
            doc_id: &missing_doc,
            page_number: 0,
            section: "",
        }])
        .expect_err("missing document");
    assert!(is_not_found(&error), "{error:?}");
    let error = store
        .link_facts_batch(&[FactLink::Span {
            fact_id: f1,
            span_id: &spans[0],
            role: "",
            confidence: f64::NAN,
        }])
        .expect_err("confidence");
    assert!(is_validation(&error, "confidence"), "{error:?}");
    let as_fact = FactId::new(first.as_str());
    let error = store
        .link_facts_batch(&[FactLink::Entity { fact_id: &as_fact, entity_id: &second, role: "" }])
        .expect_err("an entity is not a fact");
    assert!(is_not_found(&error), "{error:?}");
}

macro_rules! batch_contract {
    ($module:ident, $factory:path, $atomic:expr) => {
        mod $module {
            #[test]
            fn entity_with_embedding() {
                super::entity_with_embedding(&*$factory());
            }

            #[test]
            fn span_entity_links() {
                super::span_entity_links(&*$factory());
            }

            #[test]
            fn facts_batch() {
                super::facts_batch(&*$factory());
            }

            #[test]
            fn fact_links_batch() {
                super::fact_links_batch(&*$factory(), $atomic);
            }
        }
    };
}

batch_contract!(memory, super::memory_store, true);
batch_contract!(default_methods, super::default_methods_store, false);
#[cfg(feature = "helix")]
batch_contract!(helix, super::helix_store, true);

/// Edges written by `write`, from a fresh memory store with one document, two spans, two
/// entities and two facts.
fn memory_edges(
    write: impl Fn(&dyn GraphStore, &[SpanId], &[EntityId], &[FactId], &NodeId),
) -> Vec<EdgeRecord> {
    let store = MemoryStore::new(DIM);
    let (doc, spans) = document_with_spans(&store, 2);
    let entities: Vec<EntityId> = ["A", "B"]
        .iter()
        .map(|name| store.create_entity(&EntityNode::new(*name, "ORG")).expect("entity").id)
        .collect();
    let facts: Vec<FactId> =
        ["x", "y"].iter().map(|text| store.create_fact(&fact(text)).expect("fact").id).collect();
    write(&store, &spans, &entities, &facts, &doc);
    store.edges().into_iter().skip(3).collect()
}

#[test]
fn memory_batch_edges_match_single_writes() {
    let single = memory_edges(|store, spans, entities, facts, doc| {
        store.link_span_to_entity(&spans[0], &entities[0], 0.85, "").expect("link");
        store.link_span_to_entity(&spans[1], &entities[1], 0.5, "NER+ED").expect("link");
        store.link_fact_to_document(&facts[0], doc, 2, "Intro").expect("link");
        store.link_fact_to_entity(&facts[0], &entities[0], "").expect("link");
        store.link_fact_to_span(&facts[1], &spans[1], "", 1.0).expect("link");
    });
    let batched = memory_edges(|store, spans, entities, facts, doc| {
        let links = [
            SpanEntityLink {
                span_id: &spans[0],
                entity_id: &entities[0],
                confidence: 0.85,
                method: "",
            },
            SpanEntityLink {
                span_id: &spans[1],
                entity_id: &entities[1],
                confidence: 0.5,
                method: "NER+ED",
            },
        ];
        store.link_spans_to_entities_batch(&links).expect("span links");
        let links = [
            FactLink::Document {
                fact_id: &facts[0],
                doc_id: doc,
                page_number: 2,
                section: "Intro",
            },
            FactLink::Entity { fact_id: &facts[0], entity_id: &entities[0], role: "" },
            FactLink::Span { fact_id: &facts[1], span_id: &spans[1], role: "", confidence: 1.0 },
        ];
        store.link_facts_batch(&links).expect("fact links");
    });
    assert_eq!(batched, single);
    assert_eq!(single.len(), 5);
}

#[test]
fn memory_entity_with_embedding_matches_two_calls() {
    let store = MemoryStore::new(DIM);
    let one = store.create_entity_with_embedding(&EntityNode::new("One", "ORG"), &X).expect("one");
    let two = store.create_entity(&EntityNode::new("Two", "ORG")).expect("two");
    store.create_entity_embedding(&two.id, &X).expect("vector");
    assert_eq!(store.embedding(one.id.as_str()), store.embedding(two.id.as_str()));
    let (label, properties) = store.node_snapshot(one.id.as_str()).expect("node");
    assert_eq!(label, "Entity");
    assert_eq!(properties["AliasesJson"], "[]");
}

/// A store that implements only the required methods, so the batch methods use the trait's
/// default implementations.
#[derive(Debug)]
struct DefaultMethods(MemoryStore);

type R<T> = impeach_store::Result<T>;

impl GraphStore for DefaultMethods {
    fn store_id(&self) -> String {
        self.0.store_id()
    }
    fn embedding_dim(&self) -> usize {
        self.0.embedding_dim()
    }
    fn reset(&self) -> R<()> {
        self.0.reset()
    }
    fn create_document(&self, document: &DocumentNode) -> R<StoredDocument> {
        self.0.create_document(document)
    }
    fn update_document(&self, doc_id: &NodeId, document: &DocumentNode) -> R<StoredDocument> {
        self.0.update_document(doc_id, document)
    }
    fn get_document_chunks(&self, doc_id: &NodeId) -> R<Vec<StoredChunk>> {
        self.0.get_document_chunks(doc_id)
    }
    fn create_document_chunk(&self, doc_id: &NodeId, chunk: &ChunkNode) -> R<StoredChunk> {
        self.0.create_document_chunk(doc_id, chunk)
    }
    fn update_document_chunk(&self, chunk_id: &ChunkId, chunk: &ChunkNode) -> R<StoredChunk> {
        self.0.update_document_chunk(chunk_id, chunk)
    }
    fn create_chunk_embedding(&self, chunk_id: &ChunkId, embedding: &[f32]) -> R<()> {
        self.0.create_chunk_embedding(chunk_id, embedding)
    }
    fn create_span_for_chunk(&self, span: &SpanNode) -> R<StoredSpan> {
        self.0.create_span_for_chunk(span)
    }
    fn create_entity(&self, entity: &EntityNode) -> R<StoredEntity> {
        self.0.create_entity(entity)
    }
    fn create_entity_embedding(&self, entity_id: &EntityId, embedding: &[f32]) -> R<()> {
        self.0.create_entity_embedding(entity_id, embedding)
    }
    fn find_entity_by_name(&self, name: &str) -> R<Vec<EntityHit>> {
        self.0.find_entity_by_name(name)
    }
    fn find_similar_entities(&self, embedding: &[f32], limit: usize) -> R<Vec<EntityHit>> {
        self.0.find_similar_entities(embedding, limit)
    }
    fn get_all_entities(&self, limit: usize) -> R<Vec<StoredEntity>> {
        self.0.get_all_entities(limit)
    }
    fn link_span_to_entity(
        &self,
        span_id: &SpanId,
        entity_id: &EntityId,
        confidence: f64,
        method: &str,
    ) -> R<()> {
        self.0.link_span_to_entity(span_id, entity_id, confidence, method)
    }
    fn create_fact(&self, fact: &FactNode) -> R<StoredFact> {
        self.0.create_fact(fact)
    }
    fn create_fact_embedding(&self, fact_id: &FactId, embedding: &[f32]) -> R<()> {
        self.0.create_fact_embedding(fact_id, embedding)
    }
    fn link_fact_to_document(
        &self,
        fact_id: &FactId,
        doc_id: &NodeId,
        page_number: i64,
        section: &str,
    ) -> R<()> {
        self.0.link_fact_to_document(fact_id, doc_id, page_number, section)
    }
    fn link_fact_to_entity(&self, fact_id: &FactId, entity_id: &EntityId, role: &str) -> R<()> {
        self.0.link_fact_to_entity(fact_id, entity_id, role)
    }
    fn link_fact_to_span(
        &self,
        fact_id: &FactId,
        span_id: &SpanId,
        role: &str,
        confidence: f64,
    ) -> R<()> {
        self.0.link_fact_to_span(fact_id, span_id, role, confidence)
    }
    fn get_entity_facts(&self, entity_id: &EntityId) -> R<Vec<StoredFact>> {
        self.0.get_entity_facts(entity_id)
    }
    fn get_fact_entities(&self, fact_id: &FactId) -> R<Vec<EntityId>> {
        self.0.get_fact_entities(fact_id)
    }
    fn find_facts_by_text(&self, keywords: &str, limit: usize) -> R<Vec<FactHit>> {
        self.0.find_facts_by_text(keywords, limit)
    }
    fn find_similar_facts(&self, embedding: &[f32], limit: usize) -> R<Vec<FactHit>> {
        self.0.find_similar_facts(embedding, limit)
    }
    fn create_chunks_batch(&self, doc_id: &NodeId, chunks: &[ChunkNode]) -> R<Vec<StoredChunk>> {
        self.0.create_chunks_batch(doc_id, chunks)
    }
    fn delete_document_chunks(&self, doc_id: &NodeId) -> R<usize> {
        self.0.delete_document_chunks(doc_id)
    }
    fn create_spans_batch(&self, spans: &[SpanNode]) -> R<Vec<StoredSpan>> {
        self.0.create_spans_batch(spans)
    }
    fn create_embeddings_batch(
        &self,
        label: VectorLabel,
        items: &[EmbeddingWrite<'_>],
    ) -> R<usize> {
        self.0.create_embeddings_batch(label, items)
    }
}
