//! Cross-document entity reuse: the two Green Valley documents run through MentionStage,
//! DefinitionStage, CoreferenceStage (rule backend) and EDStage in sequence against one
//! MemoryStore with a MockEmbedder (GLiNER2 replaced by a gazetteer of the documents' names).
//!
//! Observed: the incident report creates 8 entities (6 hybrid, 2 LLM-first labels), the rest
//! of its mentions match within the document; the complaint creates none and reuses them
//! (5 `hybrid_heuristic` exact matches, 2 `llm_skipped_reuse`, 1 intra-document match).

mod common;
mod ed_common;

use std::collections::BTreeSet;
use std::sync::Arc;

use ce_core::{EntityId, MentionId};
use ce_pipeline::{run_processing_with, ProcessingOptions, ProcessingRun};
use ce_store::{GraphStore, MemoryStore};
use common::fixture;
use ed_common::{env, GazetteerNer, DIM, GREEN_VALLEY_NAMES};
use pretty_assertions::assert_eq;

fn run(store: &Arc<MemoryStore>, name: &str) -> ProcessingRun {
    let ner = Arc::new(GazetteerNer::new(GREEN_VALLEY_NAMES));
    run_processing_with(fixture(name), &ProcessingOptions::default(), env(store, ner))
        .expect("runs")
}

fn entity_ids(store: &MemoryStore) -> BTreeSet<EntityId> {
    store.get_all_entities(1000).expect("entities").into_iter().map(|e| e.id).collect()
}

fn span_links(store: &MemoryStore) -> usize {
    store.edges().iter().filter(|e| e.label == "SpanRefersToEntity").count()
}

fn count(run: &ProcessingRun, predicate: impl Fn(&str) -> bool) -> usize {
    let results = run.data.ed_results.as_ref().expect("ed_results");
    results.iter().filter(|r| predicate(&r.strategy_used)).count()
}

/// Entity resolved for the first mention named `name`.
fn entity_of(run: &ProcessingRun, name: &str) -> Option<EntityId> {
    let mentions = run.data.doc_mentions.as_ref().expect("mentions");
    let resolved = run.data.resolved_mentions.as_ref().expect("resolved");
    let id: &MentionId = &mentions.iter().find(|m| m.name == name)?.id;
    resolved.get(id).cloned()
}

#[test]
fn second_document_reuses_entities_of_the_first() {
    let store = Arc::new(MemoryStore::new(DIM));

    let first = run(&store, "green_valley_incident_report.jsonl");
    assert_eq!(
        first.stages,
        [
            "rag_index",
            "mention_detection",
            "definition_extraction",
            "coreference",
            "ed_resolution",
            "relation_extraction",
            "fact_extraction",
            "contradiction_check"
        ]
    );
    let after_first = entity_ids(&store);
    let created = count(&first, |s| s.ends_with("_create"));
    assert_eq!(created, after_first.len());
    assert_eq!(created, 8);
    assert_eq!(count(&first, |s| s.ends_with("_reuse") || s == "vector_fuzzy"), 0);
    let first_results = first.data.ed_results.as_ref().expect("ed_results");
    assert!(first_results.iter().all(|r| r.chosen_entity_id.is_some()));
    let links_after_first = span_links(&store);
    assert!(links_after_first > 0);
    let maria = entity_of(&first, "Maria Lopez").expect("Maria Lopez resolved");
    let reed = entity_of(&first, "Daniel Reed").expect("Daniel Reed resolved");
    assert_eq!(entity_of(&first, "Lopez"), Some(maria.clone()));
    assert_eq!(entity_of(&first, "Reed"), Some(reed.clone()));
    assert_ne!(maria, reed);

    let second = run(&store, "lopez_v_green_valley_complaint.jsonl");
    // No new entity: every name of the complaint already exists.
    assert_eq!(entity_ids(&store), after_first);
    assert_eq!(count(&second, |s| s.ends_with("_create")), 0);
    let creator_reuse = count(&second, |s| s.ends_with("_reuse") || s == "vector_fuzzy");
    let service_reuse = count(&second, |s| s == "hybrid_heuristic");
    assert_eq!((creator_reuse, service_reuse), (2, 5));
    let second_results = second.data.ed_results.as_ref().expect("ed_results");
    for result in second_results {
        let chosen = result.chosen_entity_id.as_ref().expect("resolved");
        assert!(after_first.contains(chosen), "{result:?}");
    }
    assert_eq!(entity_of(&second, "Reed"), Some(reed));
    assert_eq!(entity_of(&second, "Green Valley"), entity_of(&first, "Green Valley"));
    assert_eq!(entity_of(&second, "Honda Civic"), entity_of(&first, "Honda Civic"));
    assert!(span_links(&store) > links_after_first);
}
