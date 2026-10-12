//! Shared `GraphStore` contract, run against `MemoryStore` and (feature `helix`) `HelixStore`
//! in memory mode.

use std::sync::Arc;

use ce_core::{ChunkId, EntityId, FactId, NodeId, SpanId};
use ce_store::graph::{ENTITY_NAME_SEARCH_K, MAX_SEARCH_K};
use ce_store::{
    ChunkNode, DocumentNode, EmbeddingWrite, EntityNode, FactNode, GraphStore, MemoryStore,
    SpanNode, StoreError, VectorLabel,
};
use pretty_assertions::assert_eq;

const DIM: usize = 4;

fn memory_store() -> Arc<dyn GraphStore> {
    Arc::new(MemoryStore::new(DIM))
}

#[cfg(feature = "helix")]
fn helix_store() -> Arc<dyn GraphStore> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let namespace =
        format!("contract-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::SeqCst));
    ce_store::HelixStore::open(&ce_store::HelixOptions::memory(namespace, DIM))
        .expect("in-memory Helix database opens")
}

fn document(title: &str) -> DocumentNode {
    DocumentNode {
        doc_type: "complaint".into(),
        title: title.into(),
        date_created: "2024-01-02T03:04:05Z".into(),
        file_path: "docs/complaint.pdf".into(),
        total_pages: 3,
        metadata: r#"{"doc_id": "doc-1"}"#.into(),
    }
}

fn chunk(sequence: i64, text: &str) -> ChunkNode {
    ChunkNode {
        text: text.into(),
        page_number: 1,
        start_char: sequence * 100,
        end_char: sequence * 100 + 99,
        section: String::new(),
        sequence_number: sequence,
        metadata: format!(r#"{{"chunk_index": {sequence}}}"#),
    }
}

fn span(doc: &NodeId, chunk: &ChunkId, start: i64, text: &str) -> SpanNode {
    SpanNode {
        doc_id: doc.clone(),
        chunk_id: chunk.clone(),
        start_char: start,
        end_char: start + text.len() as i64,
        span_type: "SENTENCE".into(),
        span_label: "SENTENCE".into(),
        surface_text: text.into(),
        metadata: "{}".into(),
    }
}

fn entity(name: &str, entity_type: &str, aliases: &[&str]) -> EntityNode {
    EntityNode {
        aliases_json: serde_json::to_string(aliases).expect("aliases serialise"),
        description: format!("{name} description"),
        ..EntityNode::new(name, entity_type)
    }
}

fn fact(text: &str) -> FactNode {
    FactNode::new(
        text,
        r#"{"subject": "A", "relationship": "PAID", "object": "B"}"#,
        r#"{"t_valid_from": null, "t_valid_until": null}"#,
        0.9,
    )
}

fn is_validation(result: &StoreError, expected_field: &str) -> bool {
    matches!(result, StoreError::Validation { field, .. } if field == expected_field)
}

fn is_not_found(result: &StoreError) -> bool {
    matches!(result, StoreError::NotFound { .. })
}

fn assert_close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-3, "expected {expected}, got {actual}");
}

fn store_id_and_reset(store: &dyn GraphStore) {
    let first = store.store_id();
    assert!(!first.is_empty());
    assert_eq!(store.store_id(), first, "store id is stable");
    assert_eq!(store.embedding_dim(), DIM);
    let doc = store.create_document(&document("before reset")).expect("document");
    let acme = store.create_entity(&entity("Acme", "ORG", &[])).expect("entity");
    store.reset().expect("reset");
    assert_ne!(store.store_id(), first, "reset assigns a new store id");
    assert!(store.get_all_entities(10).expect("entities").is_empty());
    assert!(store.find_entity_by_name("Acme").expect("search").is_empty());
    assert!(is_not_found(&store.update_document(&doc.id, &document("x")).expect_err("gone")));
    assert!(is_not_found(
        &store.create_entity_embedding(&acme.id, &[1.0, 0.0, 0.0, 0.0]).expect_err("gone")
    ));
}

fn documents(store: &dyn GraphStore) {
    let created = store.create_document(&document("Complaint")).expect("document");
    assert!(created.id.as_str().parse::<u64>().is_ok(), "ids are integers: {}", created.id);
    assert_eq!(created.node, document("Complaint"));
    let mut changed = document("Amended complaint");
    changed.total_pages = 7;
    let updated = store.update_document(&created.id, &changed).expect("update");
    assert_eq!(updated.id, created.id);
    assert_eq!(updated.node, changed);
    let missing = store.update_document(&NodeId::new("987654"), &changed).expect_err("missing");
    assert!(is_not_found(&missing), "{missing:?}");
    let malformed = store.update_document(&NodeId::new("doc-1"), &changed).expect_err("malformed");
    assert!(is_validation(&malformed, "doc_id"), "{malformed:?}");
    assert!(store.get_document_chunks(&created.id).expect("no chunks").is_empty());
    assert!(store.get_document_chunks(&NodeId::new("987654")).expect("missing doc").is_empty());
    let entity = store.create_entity(&entity("Acme", "ORG", &[])).expect("entity");
    let wrong_label = store
        .update_document(&NodeId::new(entity.id.as_str()), &changed)
        .expect_err("an entity is not a document");
    assert!(is_not_found(&wrong_label));
}

fn chunks_and_embeddings(store: &dyn GraphStore) {
    let doc = store.create_document(&document("Doc")).expect("document");
    let first =
        store.create_document_chunk(&doc.id, &chunk(0, "The tenant paid rent.")).expect("chunk 0");
    let second =
        store.create_document_chunk(&doc.id, &chunk(1, "The landlord refused.")).expect("chunk 1");
    assert_eq!(first.node, chunk(0, "The tenant paid rent."));
    assert_ne!(first.id, second.id);
    let missing =
        store.create_document_chunk(&NodeId::new("987654"), &chunk(2, "x")).expect_err("no doc");
    assert!(is_not_found(&missing));

    let mut revised = chunk(1, "The landlord refused the rent.");
    revised.section = "Facts".into();
    let updated = store.update_document_chunk(&second.id, &revised).expect("update chunk");
    assert_eq!(updated.node, revised);
    assert!(is_not_found(
        &store.update_document_chunk(&ChunkId::new("987654"), &revised).expect_err("missing")
    ));

    let mut chunks = store.get_document_chunks(&doc.id).expect("chunks");
    chunks.sort_by_key(|stored| stored.node.sequence_number);
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].id, first.id);
    assert_eq!(chunks[1].node, revised);

    store.create_chunk_embedding(&first.id, &[0.1, 0.2, 0.3, 0.4]).expect("embedding");
    store.create_chunk_embedding(&first.id, &[0.4, 0.3, 0.2, 0.1]).expect("embedding replaced");
    let wrong_dim = store.create_chunk_embedding(&first.id, &[1.0, 0.0]).expect_err("dim");
    assert!(is_validation(&wrong_dim, "embedding_vector"));
    let zero = store.create_chunk_embedding(&first.id, &[0.0; DIM]).expect_err("zero vector");
    assert!(is_validation(&zero, "embedding_vector"));
    let nan = store.create_chunk_embedding(&first.id, &[f32::NAN, 0.0, 0.0, 1.0]).expect_err("nan");
    assert!(is_validation(&nan, "embedding_vector"));
    let missing = store
        .create_chunk_embedding(&ChunkId::new("987654"), &[1.0, 0.0, 0.0, 0.0])
        .expect_err("missing");
    assert!(is_not_found(&missing));
    let doc_as_chunk = store
        .create_chunk_embedding(&ChunkId::new(doc.id.as_str()), &[1.0, 0.0, 0.0, 0.0])
        .expect_err("a document is not a chunk");
    assert!(is_not_found(&doc_as_chunk));
}

fn spans(store: &dyn GraphStore) {
    let doc = store.create_document(&document("Doc")).expect("document");
    let chunk =
        store.create_document_chunk(&doc.id, &chunk(0, "Maria Lopez paid.")).expect("chunk");
    let input = span(&doc.id, &chunk.id, 0, "Maria Lopez paid.");
    let created = store.create_span_for_chunk(&input).expect("span");
    assert_eq!(created.node, input);
    let orphan = span(&doc.id, &ChunkId::new("987654"), 0, "x");
    assert!(is_not_found(&store.create_span_for_chunk(&orphan).expect_err("missing chunk")));
    let malformed = span(&doc.id, &ChunkId::new("chunk-a"), 0, "x");
    assert!(is_validation(
        &store.create_span_for_chunk(&malformed).expect_err("bad id"),
        "chunk_id"
    ));
}

fn entities(store: &dyn GraphStore) {
    let error = store.create_entity(&EntityNode::new("", "ORG")).expect_err("name required");
    assert!(is_validation(&error, "Name"));
    let error = store.create_entity(&EntityNode::new("Acme", "")).expect_err("type required");
    assert!(is_validation(&error, "EntityType"));
    let mut bad_aliases = EntityNode::new("Acme", "ORG");
    bad_aliases.aliases_json = r#"{"not": "a list"}"#.into();
    assert!(is_validation(&store.create_entity(&bad_aliases).expect_err("aliases"), "AliasesJson"));

    let mut no_aliases = EntityNode::new("Acme", "ORG");
    no_aliases.aliases_json = String::new();
    let acme = store.create_entity(&no_aliases).expect("entity");
    assert_eq!(acme.node.aliases_json, "[]", "empty aliases are stored as []");
    let maria =
        store.create_entity(&entity("Maria Lopez", "PERSON", &["Ms. Lopez"])).expect("entity");
    let court = store.create_entity(&entity("Superior Court", "ORG", &[])).expect("entity");

    let all = store.get_all_entities(10).expect("all");
    let ids: Vec<&EntityId> = all.iter().map(|stored| &stored.id).collect();
    assert_eq!(ids, vec![&acme.id, &maria.id, &court.id], "id order");
    assert_eq!(all[1].node, maria.node);
    assert_eq!(store.get_all_entities(2).expect("limited").len(), 2);
    assert!(store.get_all_entities(0).expect("zero").is_empty());
}

fn find_entity_by_name_covers_aliases(store: &dyn GraphStore) {
    let acme = store
        .create_entity(&entity("Acme Corporation", "ORG", &["Acme", "Defendant"]))
        .expect("acme");
    let gvl = store
        .create_entity(&entity(
            "Green Valley Logistics, Inc.",
            "ORG",
            &["Green Valley Logistics", "GVL"],
        ))
        .expect("gvl");
    let maria = store.create_entity(&entity("Maria Lopez", "PERSON", &[])).expect("maria");
    store.create_entity(&entity("Valley Farm Supply", "ORG", &[])).expect("farm");

    let hits = store.find_entity_by_name("Defendant").expect("alias search");
    assert_eq!(hits.len(), 1, "only the alias matches");
    assert_eq!(hits[0].node.id, acme.id);
    assert!(hits[0].score > 0.0);

    let hits = store.find_entity_by_name("GVL").expect("alias search");
    assert_eq!(hits.iter().map(|hit| &hit.node.id).collect::<Vec<_>>(), vec![&gvl.id]);

    let hits = store.find_entity_by_name("maria").expect("case-insensitive");
    assert_eq!(hits[0].node.id, maria.id);
    assert_eq!(hits[0].node.node.name, "Maria Lopez");

    let hits = store.find_entity_by_name("Green Valley Logistics").expect("name search");
    assert_eq!(hits[0].node.id, gvl.id, "best match first");
    assert!(hits.len() >= 2, "partial matches are returned too");
    assert!(hits.windows(2).all(|pair| pair[0].score >= pair[1].score), "sorted by score");
    let ids: std::collections::BTreeSet<&EntityId> = hits.iter().map(|hit| &hit.node.id).collect();
    assert_eq!(ids.len(), hits.len(), "one hit per entity although Name and aliases both match");

    assert!(store.find_entity_by_name("").expect("empty").is_empty());
    assert!(store.find_entity_by_name("   ").expect("blank").is_empty());
    assert!(store.find_entity_by_name("nonexistentterm").expect("no match").is_empty());

    for index in 0..ENTITY_NAME_SEARCH_K + 5 {
        store.create_entity(&entity(&format!("Witness {index}"), "PERSON", &[])).expect("witness");
    }
    assert_eq!(store.find_entity_by_name("Witness").expect("capped").len(), ENTITY_NAME_SEARCH_K);
}

/// Regression: the Python retriever read a score key the store never wrote, so every vector
/// candidate scored 0 and was dropped. Scores here must be the real cosine similarity.
fn vector_scores_are_real_cosine(store: &dyn GraphStore) {
    let exact = store.create_entity(&entity("Exact", "ORG", &[])).expect("entity");
    let near = store.create_entity(&entity("Near", "ORG", &[])).expect("entity");
    let orthogonal = store.create_entity(&entity("Orthogonal", "ORG", &[])).expect("entity");
    let opposite = store.create_entity(&entity("Opposite", "ORG", &[])).expect("entity");
    let unindexed = store.create_entity(&entity("No vector", "ORG", &[])).expect("entity");
    store.create_entity_embedding(&exact.id, &[2.0, 0.0, 0.0, 0.0]).expect("vector");
    store.create_entity_embedding(&near.id, &[0.6, 0.8, 0.0, 0.0]).expect("vector");
    store.create_entity_embedding(&orthogonal.id, &[0.0, 0.0, 1.0, 0.0]).expect("vector");
    store.create_entity_embedding(&opposite.id, &[-1.0, 0.0, 0.0, 0.0]).expect("vector");

    let hits = store.find_similar_entities(&[1.0, 0.0, 0.0, 0.0], 10).expect("search");
    let ids: Vec<&EntityId> = hits.iter().map(|hit| &hit.node.id).collect();
    assert_eq!(ids[..2], [&exact.id, &near.id], "ranked by cosine");
    assert!(
        hits.iter().all(|hit| hit.node.id != unindexed.id),
        "nodes without a vector are not candidates"
    );
    assert_close(hits[0].score, 1.0);
    assert_close(hits[1].score, 0.6);
    assert!(hits[1].score > 0.5, "vector candidates keep their real score");
    assert!(hits.iter().all(|hit| (0.0..=1.0).contains(&hit.score)), "scores clamped to [0, 1]");
    let by_id = |id: &EntityId| hits.iter().find(|hit| &hit.node.id == id).map(|hit| hit.score);
    assert_close(by_id(&orthogonal.id).expect("orthogonal hit"), 0.0);
    // HelixDB may leave the antipodal vector out of the candidate set; when present it scores 0.
    if let Some(score) = by_id(&opposite.id) {
        assert_close(score, 0.0);
    }
    assert_eq!(hits[0].node.node.name, "Exact");

    assert_eq!(store.find_similar_entities(&[1.0, 0.0, 0.0, 0.0], 2).expect("limited").len(), 2);
    assert!(store.find_similar_entities(&[1.0, 0.0, 0.0, 0.0], 0).expect("zero").is_empty());
    let clamped =
        store.find_similar_entities(&[1.0, 0.0, 0.0, 0.0], MAX_SEARCH_K * 10).expect("clamped");
    assert!((3..=4).contains(&clamped.len()), "a huge limit is clamped, not an error");
    assert_eq!(clamped[0].node.id, exact.id);
    let error = store.find_similar_entities(&[1.0, 0.0], 5).expect_err("dimension");
    assert!(is_validation(&error, "embedding_vector"));
    let error = store.find_similar_entities(&[0.0; DIM], 5).expect_err("zero query");
    assert!(is_validation(&error, "embedding_vector"));
}

fn facts_and_searches(store: &dyn GraphStore) {
    assert!(is_validation(&store.create_fact(&fact("")).expect_err("text"), "FactText"));
    let mut uncertain = fact("x");
    uncertain.certainty = 1.5;
    assert!(is_validation(&store.create_fact(&uncertain).expect_err("certainty"), "Certainty"));
    let mut bad_json = fact("x");
    bad_json.assertion_triple = "{oops".into();
    assert!(is_validation(&store.create_fact(&bad_json).expect_err("triple"), "AssertionTriple"));

    let paid = store.create_fact(&fact("The tenant paid the March rent invoice")).expect("fact");
    assert_eq!(paid.node, fact("The tenant paid the March rent invoice"));
    let refused = store.create_fact(&fact("The landlord refused the rent payment")).expect("fact");
    let unrelated = store.create_fact(&fact("The hearing was adjourned")).expect("fact");

    let hits = store.find_facts_by_text("rent invoice", 10).expect("text search");
    assert_eq!(hits[0].node.id, paid.id);
    assert_eq!(hits.len(), 2);
    assert!(hits.iter().all(|hit| hit.score > 0.0));
    assert!(hits.iter().all(|hit| hit.node.id != unrelated.id));
    assert_eq!(store.find_facts_by_text("rent", 1).expect("limited").len(), 1);
    assert!(store.find_facts_by_text("rent", 0).expect("zero").is_empty());
    assert!(store.find_facts_by_text("", 10).expect("empty").is_empty());

    store.create_fact_embedding(&paid.id, &[0.0, 1.0, 0.0, 0.0]).expect("vector");
    store.create_fact_embedding(&refused.id, &[0.0, 0.8, 0.6, 0.0]).expect("vector");
    let missing = store
        .create_fact_embedding(&FactId::new("987654"), &[0.0, 1.0, 0.0, 0.0])
        .expect_err("missing");
    assert!(is_not_found(&missing));
    let hits = store.find_similar_facts(&[0.0, 1.0, 0.0, 0.0], 5).expect("vector search");
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].node.id, paid.id);
    assert_close(hits[0].score, 1.0);
    assert_close(hits[1].score, 0.8);
    assert_eq!(hits[1].node.node, fact("The landlord refused the rent payment"));
}

fn links_and_entity_facts(store: &dyn GraphStore) {
    let doc = store.create_document(&document("Doc")).expect("document");
    let chunk = store.create_document_chunk(&doc.id, &chunk(0, "Acme paid.")).expect("chunk");
    let span = store.create_span_for_chunk(&span(&doc.id, &chunk.id, 0, "Acme")).expect("span");
    let acme = store.create_entity(&entity("Acme", "ORG", &[])).expect("entity");
    let other = store.create_entity(&entity("Other", "ORG", &[])).expect("entity");
    let first = store.create_fact(&fact("Acme paid the invoice")).expect("fact");
    let second = store.create_fact(&fact("Acme signed the lease")).expect("fact");

    store.link_span_to_entity(&span.id, &acme.id, 0.8, "").expect("span link");
    let error =
        store.link_span_to_entity(&span.id, &acme.id, 1.2, "NER+ED").expect_err("confidence");
    assert!(is_validation(&error, "confidence"));
    let error =
        store.link_span_to_entity(&SpanId::new("987654"), &acme.id, 0.5, "x").expect_err("span");
    assert!(is_not_found(&error));
    let error = store
        .link_span_to_entity(&span.id, &EntityId::new("987654"), 0.5, "x")
        .expect_err("entity");
    assert!(is_not_found(&error));
    let error = store
        .link_span_to_entity(&span.id, &EntityId::new(doc.id.as_str()), 0.5, "x")
        .expect_err("a document is not an entity");
    assert!(is_not_found(&error));

    store.link_fact_to_document(&first.id, &doc.id, 2, "Facts").expect("fact-doc");
    assert!(is_not_found(
        &store.link_fact_to_document(&first.id, &NodeId::new("987654"), 0, "").expect_err("doc")
    ));
    store.link_fact_to_span(&first.id, &span.id, "", 1.0).expect("fact-span");
    assert!(is_validation(
        &store.link_fact_to_span(&first.id, &span.id, "", -0.1).expect_err("c"),
        "confidence"
    ));
    assert!(is_not_found(
        &store.link_fact_to_span(&FactId::new("987654"), &span.id, "", 1.0).expect_err("fact")
    ));

    store.link_fact_to_entity(&first.id, &acme.id, "SUBJECT").expect("fact-entity");
    store.link_fact_to_entity(&first.id, &acme.id, "").expect("second edge, same pair");
    store.link_fact_to_entity(&second.id, &acme.id, "OBJECT").expect("fact-entity");
    assert!(is_not_found(
        &store.link_fact_to_entity(&first.id, &EntityId::new("987654"), "").expect_err("entity")
    ));
    assert!(is_validation(
        &store.link_fact_to_entity(&FactId::new(""), &acme.id, "").expect_err("id"),
        "fact_id"
    ));

    let mut facts = store.get_entity_facts(&acme.id).expect("facts");
    facts.sort_by(|a, b| a.id.as_str().len().cmp(&b.id.as_str().len()).then(a.id.cmp(&b.id)));
    let ids: Vec<&FactId> = facts.iter().map(|stored| &stored.id).collect();
    assert_eq!(ids, vec![&first.id, &second.id], "each linked fact once");
    assert_eq!(facts[0].node, first.node);
    assert!(store.get_entity_facts(&other.id).expect("no facts").is_empty());
    assert!(store.get_entity_facts(&EntityId::new("987654")).expect("missing").is_empty());
}

fn batches(store: &dyn GraphStore) {
    let doc = store.create_document(&document("Doc")).expect("document");
    let inputs: Vec<ChunkNode> = (0..150).map(|i| chunk(i, &format!("chunk number {i}"))).collect();
    let created = store.create_chunks_batch(&doc.id, &inputs).expect("chunks batch");
    assert_eq!(created.len(), inputs.len());
    assert!(
        created.iter().zip(&inputs).all(|(stored, input)| &stored.node == input),
        "input order"
    );
    let ids: std::collections::BTreeSet<&ChunkId> =
        created.iter().map(|stored| &stored.id).collect();
    assert_eq!(ids.len(), inputs.len(), "distinct ids");
    assert_eq!(store.get_document_chunks(&doc.id).expect("chunks").len(), inputs.len());
    assert!(store.create_chunks_batch(&doc.id, &[]).expect("empty").is_empty());
    assert!(store.create_chunks_batch(&NodeId::new("987654"), &[]).expect("empty").is_empty());
    let missing = store.create_chunks_batch(&NodeId::new("987654"), &inputs[..2]).expect_err("doc");
    assert!(is_not_found(&missing));

    let spans: Vec<SpanNode> = created
        .iter()
        .take(3)
        .flat_map(|stored| {
            [span(&doc.id, &stored.id, 0, "first"), span(&doc.id, &stored.id, 10, "second")]
        })
        .collect();
    let stored_spans = store.create_spans_batch(&spans).expect("spans batch");
    assert_eq!(stored_spans.iter().map(|stored| stored.node.clone()).collect::<Vec<_>>(), spans);
    assert!(store.create_spans_batch(&[]).expect("empty").is_empty());
    let mut with_orphan = spans[..1].to_vec();
    with_orphan.push(span(&doc.id, &ChunkId::new("987654"), 0, "orphan"));
    assert!(is_not_found(&store.create_spans_batch(&with_orphan).expect_err("orphan")));

    let first = store.create_entity(&entity("First", "ORG", &[])).expect("entity");
    let second = store.create_entity(&entity("Second", "ORG", &[])).expect("entity");
    let third = store.create_entity(&entity("Third", "ORG", &[])).expect("entity");
    let vectors = [[1.0_f32, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
    let missing_item = [
        EmbeddingWrite { node_id: third.id.as_str(), vector: &vectors[2] },
        EmbeddingWrite { node_id: "987654", vector: &vectors[0] },
    ];
    let error =
        store.create_embeddings_batch(VectorLabel::Entity, &missing_item).expect_err("missing");
    assert!(is_not_found(&error));
    let bad_vector = [EmbeddingWrite { node_id: third.id.as_str(), vector: &[1.0, 0.0] }];
    let error = store.create_embeddings_batch(VectorLabel::Entity, &bad_vector).expect_err("dim");
    assert!(is_validation(&error, "embedding_vector"));
    assert!(
        store.find_similar_entities(&vectors[2], 5).expect("search").is_empty(),
        "a failed batch writes nothing"
    );

    let items = [
        EmbeddingWrite { node_id: first.id.as_str(), vector: &vectors[0] },
        EmbeddingWrite { node_id: second.id.as_str(), vector: &vectors[1] },
    ];
    assert_eq!(store.create_embeddings_batch(VectorLabel::Entity, &items).expect("batch"), 2);
    let hits = store.find_similar_entities(&vectors[1], 5).expect("search");
    assert_eq!(hits[0].node.id, second.id);
    assert_close(hits[0].score, 1.0);
    let wrong_label = [EmbeddingWrite { node_id: first.id.as_str(), vector: &vectors[0] }];
    let error = store.create_embeddings_batch(VectorLabel::Fact, &wrong_label).expect_err("label");
    assert!(is_not_found(&error));

    let chunk_items: Vec<EmbeddingWrite<'_>> = created
        .iter()
        .take(2)
        .map(|stored| EmbeddingWrite { node_id: stored.id.as_str(), vector: &vectors[0] })
        .collect();
    assert_eq!(
        store.create_embeddings_batch(VectorLabel::DocumentChunk, &chunk_items).expect("chunks"),
        2
    );
    assert_eq!(store.create_embeddings_batch(VectorLabel::Fact, &[]).expect("empty"), 0);
}

fn shared_across_threads(store: &Arc<dyn GraphStore>) {
    let handles: Vec<_> = (0..4)
        .map(|worker| {
            let store = Arc::clone(store);
            std::thread::spawn(move || {
                for index in 0..5 {
                    store
                        .create_entity(&entity(
                            &format!("Worker {worker} entity {index}"),
                            "ORG",
                            &[],
                        ))
                        .expect("concurrent write");
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().expect("worker finishes");
    }
    assert_eq!(store.get_all_entities(100).expect("entities").len(), 20);
}

macro_rules! contract_tests {
    ($module:ident, $factory:path) => {
        mod $module {

            #[test]
            fn store_id_and_reset() {
                super::store_id_and_reset(&*$factory());
            }

            #[test]
            fn documents() {
                super::documents(&*$factory());
            }

            #[test]
            fn chunks_and_embeddings() {
                super::chunks_and_embeddings(&*$factory());
            }

            #[test]
            fn spans() {
                super::spans(&*$factory());
            }

            #[test]
            fn entities() {
                super::entities(&*$factory());
            }

            #[test]
            fn find_entity_by_name_covers_aliases() {
                super::find_entity_by_name_covers_aliases(&*$factory());
            }

            #[test]
            fn vector_scores_are_real_cosine() {
                super::vector_scores_are_real_cosine(&*$factory());
            }

            #[test]
            fn facts_and_searches() {
                super::facts_and_searches(&*$factory());
            }

            #[test]
            fn links_and_entity_facts() {
                super::links_and_entity_facts(&*$factory());
            }

            #[test]
            fn batches() {
                super::batches(&*$factory());
            }

            #[test]
            fn shared_across_threads() {
                super::shared_across_threads(&$factory());
            }
        }
    };
}

contract_tests!(memory, super::memory_store);
#[cfg(feature = "helix")]
contract_tests!(helix, super::helix_store);
