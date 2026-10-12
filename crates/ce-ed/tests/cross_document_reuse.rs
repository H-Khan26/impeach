//! Cross-document reuse at the entity-creator level (port of
//! `tests/unit/ed/test_ed_cross_document_reuse.py`): type-aware reuse of name variations,
//! alias reuse, entity embeddings and types that must not merge. The `_collect_cluster_aliases`
//! cases are unit tests of `ce_ed::resolver`.

mod ed_support;

use std::sync::Arc;

use ce_models::MockEmbedder;
use ed_support::{creator, ed_context, entities, seed_entity, store, DIM};
use pretty_assertions::assert_eq;

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|v| (*v).to_owned()).collect()
}

#[test]
fn reuses_entity_by_person_name_variation() {
    let (memory, store) = store();
    let existing = seed_entity(
        &memory,
        "Maria Lopez",
        "PERSON",
        "[]",
        "{\"normalized_name\": \"maria lopez\"}",
    );
    let creator = creator(&store, Some(Arc::new(MockEmbedder::new(DIM))));
    let id = creator
        .create_entity_for_mention(
            &ed_context("Ms. Lopez", "PERSON", "CASE-1", "doc-2", "m1"),
            &[],
            None,
        )
        .expect("runs");
    assert_eq!(id, Some(existing));
    assert_eq!(entities(&memory).len(), 1);
}

#[test]
fn reuses_entity_by_org_suffix_variation() {
    let (memory, store) = store();
    let existing = seed_entity(
        &memory,
        "Green Valley Logistics, Inc.",
        "ORG",
        "[]",
        "{\"normalized_name\": \"green valley logistics inc\"}",
    );
    let creator = creator(&store, Some(Arc::new(MockEmbedder::new(DIM))));
    let id = creator
        .create_entity_for_mention(
            &ed_context("Green Valley Logistics", "ORG", "CASE-1", "doc-2", "m1"),
            &[],
            None,
        )
        .expect("runs");
    assert_eq!(id, Some(existing));
    assert_eq!(entities(&memory).len(), 1);
}

#[test]
fn creates_new_entity_when_similarity_too_low() {
    let (memory, store) = store();
    let unrelated = seed_entity(&memory, "John Smith", "PERSON", "[]", "{}");
    let creator = creator(&store, Some(Arc::new(MockEmbedder::new(DIM))));
    let id = creator
        .create_entity_for_mention(
            &ed_context("Maria Lopez", "PERSON", "CASE-1", "doc-2", "m1"),
            &[],
            None,
        )
        .expect("runs")
        .expect("entity");
    assert_ne!(id, unrelated);
    assert_eq!(entities(&memory).len(), 2);
}

#[test]
fn second_document_reuses_first_document_entity() {
    let (memory, store) = store();
    let creator = creator(&store, Some(Arc::new(MockEmbedder::new(DIM))));
    let first = creator
        .create_entity_for_mention(
            &ed_context("Maria Lopez", "PERSON", "CASE-1", "doc-1", "m1"),
            &strings(&["Ms. Lopez"]),
            None,
        )
        .expect("runs")
        .expect("entity");
    let stored = entities(&memory);
    assert_eq!(stored.len(), 1);
    assert!(stored[0].node.aliases().contains(&"Ms. Lopez".to_owned()));

    let second = creator
        .create_entity_for_mention(
            &ed_context("Ms. Lopez", "PERSON", "CASE-1", "doc-2", "m2"),
            &[],
            None,
        )
        .expect("runs")
        .expect("entity");
    assert_eq!(second, first);
    assert_eq!(entities(&memory).len(), 1);
}

#[test]
fn second_document_reuses_via_alias_match() {
    let (memory, store) = store();
    let creator = creator(&store, Some(Arc::new(MockEmbedder::new(DIM))));
    let first = creator
        .create_entity_for_mention(
            &ed_context("Robert Johnson", "PERSON", "CASE-1", "doc-1", "m1"),
            &strings(&["Bob Johnson", "Mr. Johnson"]),
            None,
        )
        .expect("runs")
        .expect("entity");
    let second = creator
        .create_entity_for_mention(
            &ed_context("Bob Johnson", "PERSON", "CASE-1", "doc-2", "m2"),
            &[],
            None,
        )
        .expect("runs")
        .expect("entity");
    assert_eq!(second, first);
    assert_eq!(entities(&memory).len(), 1);
}

#[test]
fn different_entity_types_not_confused() {
    let (memory, store) = store();
    let creator = creator(&store, Some(Arc::new(MockEmbedder::new(DIM))));
    let person = creator
        .create_entity_for_mention(
            &ed_context("Mercury", "PERSON", "CASE-1", "doc-1", "m1"),
            &[],
            None,
        )
        .expect("runs")
        .expect("entity");
    let org = creator
        .create_entity_for_mention(
            &ed_context("Mercury", "ORG", "CASE-1", "doc-2", "m2"),
            &[],
            None,
        )
        .expect("runs")
        .expect("entity");
    assert_ne!(person, org);
    assert_eq!(entities(&memory).len(), 2);
}

#[test]
fn creates_embedding_for_new_entity() {
    let (memory, store) = store();
    let embedder = Arc::new(MockEmbedder::new(DIM));
    let creator = creator(&store, Some(embedder.clone()));
    let id = creator
        .create_entity_for_mention(
            &ed_context("Maria Lopez", "PERSON", "CASE-1", "doc-1", "m1"),
            &[],
            None,
        )
        .expect("runs")
        .expect("entity");
    assert_eq!(entities(&memory).len(), 1);
    let vector = memory.embedding(id.as_str()).expect("embedding stored");
    assert_eq!(vector.len(), DIM);
    assert_eq!(embedder.calls(), 1);
}

#[test]
fn no_embedding_without_embedding_service() {
    let (memory, store) = store();
    let creator = creator(&store, None);
    let id = creator
        .create_entity_for_mention(
            &ed_context("Maria Lopez", "PERSON", "CASE-1", "doc-1", "m1"),
            &[],
            None,
        )
        .expect("runs")
        .expect("entity");
    assert_eq!(entities(&memory).len(), 1);
    assert_eq!(memory.embedding(id.as_str()), None);
}
