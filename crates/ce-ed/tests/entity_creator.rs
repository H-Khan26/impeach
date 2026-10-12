//! EntityCreator: role blocking (port of `tests/unit/test_entity_creator_role_blocking.py`),
//! strategy annotation, value types, type-safe reuse (`TestCrossTypeEntityReuse` of
//! `test_ed_type_compatibility.py`) and the entity part of `test_ed_helix_reuse.py`.

mod ed_support;

use std::sync::Arc;

use ce_config::pipeline::EntityRetrievalStrategy;
use ce_core::{DocContext, EdContext, EdResult, LocalContext, Mention, Span};
use ce_ed::{EntityCreator, EntityOutcome};
use ce_models::{Embedder, MockEmbedder};
use ce_store::{GraphStore, MemoryStore};
use ed_support::{creator, ed_context, entities, seed_entity, store, DIM};
use pretty_assertions::assert_eq;
use serde_json::{json, Value};

/// The role-blocking fixture: `all_entities_fuzzy`, threshold 0.80, no embedder.
fn role_creator(store: &Arc<dyn GraphStore>) -> EntityCreator {
    creator(store, None)
        .with_reuse_similarity_threshold(0.80)
        .with_retrieval_strategy(EntityRetrievalStrategy::AllEntitiesFuzzy)
}

/// The role-blocking mention context with `name`.
fn role_ctx(name: &str) -> EdContext {
    let mut mention = Mention::new("mention_1", "PER", name, Span::new(100, 110));
    mention.source = Some("test".into());
    mention.doc_id = Some("doc_123".into());
    let mut doc = DocContext::new("doc_123", "complaint");
    doc.case_id = Some("case_456".into());
    doc.jurisdiction = Some("CA".into());
    EdContext {
        mention,
        local_context: LocalContext::new(
            "This is a test sentence.",
            "This is a test chunk with more context.",
        ),
        doc_context: doc,
        span_label: "PER".into(),
        normalized_key: None,
    }
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|v| (*v).to_owned()).collect()
}

fn unresolved(strategy: &str) -> EdResult {
    EdResult {
        mention_id: None,
        chosen_entity_id: None,
        candidates: vec![],
        resolved_by_llm: false,
        confidence: 0.5,
        strategy_used: strategy.to_owned(),
    }
}

fn metadata_of(store: &MemoryStore) -> Value {
    let all = entities(store);
    assert_eq!(all.len(), 1);
    serde_json::from_str(&all[0].node.metadata).expect("metadata JSON")
}

#[test]
fn skip_creation_role_only_no_aliases() {
    let (memory, store) = store();
    let outcome = role_creator(&store)
        .create_or_reuse_entity_for_mention(&role_ctx("Plaintiff"), &[], None)
        .expect("runs");
    assert_eq!(outcome, None);
    assert!(entities(&memory).is_empty());
}

#[test]
fn skip_creation_role_only_with_role_aliases() {
    let (memory, store) = store();
    let outcome = role_creator(&store)
        .create_or_reuse_entity_for_mention(
            &role_ctx("Plaintiff"),
            &strings(&["Defendant", "Appellant"]),
            None,
        )
        .expect("runs");
    assert_eq!(outcome, None);
    assert!(entities(&memory).is_empty());
}

#[test]
fn salvage_from_alias_when_mention_is_role() {
    let (memory, store) = store();
    let outcome = role_creator(&store)
        .create_or_reuse_entity_for_mention(
            &role_ctx("Plaintiff"),
            &strings(&["Maria Lopez", "Ms. Lopez"]),
            None,
        )
        .expect("runs")
        .expect("entity created");
    assert!(outcome.is_new());
    let all = entities(&memory);
    assert_eq!(all.len(), 1);
    assert_eq!(&all[0].id, outcome.entity_id());
    assert_eq!(all[0].node.name, "Maria Lopez");
    assert!(all[0].node.aliases_json.contains("Plaintiff"));
    assert_eq!(all[0].node.aliases(), strings(&["Ms. Lopez", "Plaintiff"]));
}

#[test]
fn role_stored_in_metadata_when_salvaged() {
    let (memory, store) = store();
    role_creator(&store)
        .create_or_reuse_entity_for_mention(&role_ctx("Plaintiff"), &strings(&["John Smith"]), None)
        .expect("runs")
        .expect("entity created");
    let metadata = metadata_of(&memory);
    assert_eq!(metadata["roles"], json!(["PLAINTIFF"]));
    assert_eq!(
        metadata,
        json!({
            "source_doc_id": "doc_123",
            "case_id": "case_456",
            "jurisdiction": "CA",
            "normalized_name": "john smith",
            "roles": ["PLAINTIFF"]
        })
    );
}

#[test]
fn multiple_roles_in_aliases() {
    let (memory, store) = store();
    role_creator(&store)
        .create_or_reuse_entity_for_mention(
            &role_ctx("Maria Lopez"),
            &strings(&["Plaintiff", "Counter-Claimant", "Ms. Lopez"]),
            None,
        )
        .expect("runs")
        .expect("entity created");
    let metadata = metadata_of(&memory);
    assert_eq!(metadata["roles"], json!(["PLAINTIFF", "COUNTER-CLAIMANT"]));
}

#[test]
fn skip_role_strategy_annotation() {
    let (_, store) = store();
    let result = role_creator(&store)
        .create_entity_from_ed_result(
            &role_ctx("Defendant"),
            unresolved("bm25_no_match"),
            &[],
            None,
        )
        .expect("runs");
    assert_eq!(result.chosen_entity_id, None);
    assert_eq!(result.strategy_used, "bm25_no_match_skip_role");
    assert!((result.confidence - 0.5).abs() < f64::EPSILON);
}

#[test]
fn create_strategy_annotation_on_success() {
    let (memory, store) = store();
    let result = role_creator(&store)
        .create_entity_from_ed_result(
            &role_ctx("Maria Lopez"),
            unresolved("bm25_no_match"),
            &[],
            None,
        )
        .expect("runs");
    assert_eq!(result.chosen_entity_id.as_ref(), Some(&entities(&memory)[0].id));
    assert_eq!(result.strategy_used, "bm25_no_match_create");
    assert!((result.confidence - 0.7).abs() < f64::EPSILON);
    assert!(!result.resolved_by_llm);
}

#[test]
fn reuse_strategy_annotation_depends_on_retrieval_strategy() {
    let (memory, store) = store();
    let existing = seed_entity(&memory, "Maria Lopez", "PER", "[]", "{}");

    let bm25 = creator(&store, None);
    let result = bm25
        .create_entity_from_ed_result(
            &role_ctx("Maria Lopez"),
            unresolved("hybrid_unresolved"),
            &[],
            None,
        )
        .expect("runs");
    assert_eq!(result.chosen_entity_id.as_ref(), Some(&existing));
    assert_eq!(result.strategy_used, "hybrid_unresolved_reuse");
    assert!((result.confidence - 0.85).abs() < f64::EPSILON);

    // vector_fuzzy reuse reports `vector_fuzzy`. The entity vector is unrelated to the query
    // vector (cosine about 0): the hit is still a candidate.
    let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(DIM));
    memory
        .create_entity_embedding(&existing, &embedder.embed(&["unrelated"]).expect("vector")[0])
        .expect("embedding stored");
    let vector = creator(&store, Some(embedder))
        .with_retrieval_strategy(EntityRetrievalStrategy::VectorFuzzy);
    let result = vector
        .create_entity_from_ed_result(
            &role_ctx("Maria Lopez"),
            unresolved("hybrid_unresolved"),
            &[],
            None,
        )
        .expect("runs");
    assert_eq!(result.chosen_entity_id.as_ref(), Some(&existing));
    assert_eq!(result.strategy_used, "vector_fuzzy");
    assert_eq!(entities(&memory).len(), 1);
}

#[test]
fn resolved_results_and_value_types_are_not_created() {
    let (memory, store) = store();
    let creator = role_creator(&store);
    let mut chosen = unresolved("rule");
    chosen.chosen_entity_id = Some("42".into());
    let kept = creator
        .create_entity_from_ed_result(&role_ctx("Maria Lopez"), chosen.clone(), &[], None)
        .expect("runs");
    assert_eq!(kept, chosen);

    let mut ctx = role_ctx("March 3, 2021");
    "DATE".clone_into(&mut ctx.span_label);
    let skipped = creator
        .create_entity_from_ed_result(&ctx, unresolved("hybrid_unresolved"), &[], None)
        .expect("runs");
    assert_eq!(skipped.chosen_entity_id, None);
    assert_eq!(skipped.strategy_used, "skip_value_type");
    assert!(skipped.confidence.abs() < f64::EPSILON);
    assert!(entities(&memory).is_empty());
}

#[test]
fn reuses_existing_entity_by_normalized_name_and_records_case() {
    // test_entity_creator_reuses_existing_entity_by_normalized_name_and_case
    let (memory, store) = store();
    let creator = creator(&store, None);
    let first = creator
        .create_entity_for_mention(
            &ed_context("Maria Lopez", "PERSON", "CASE-1", "doc-1", "m1"),
            &[],
            None,
        )
        .expect("runs")
        .expect("created");
    let second = creator
        .create_entity_for_mention(
            &ed_context("Maria Lopez", "PERSON", "CASE-1", "doc-2", "m1"),
            &[],
            None,
        )
        .expect("runs")
        .expect("reused");
    assert_eq!(first, second);
    assert_eq!(entities(&memory).len(), 1);
    let metadata = metadata_of(&memory);
    assert_eq!(metadata["normalized_name"], "maria lopez");
    assert_eq!(metadata["case_id"], "CASE-1");
}

#[test]
fn org_loc_facility_with_same_name_reuse_one_entity() {
    let (memory, store) = store();
    let creator = creator(&store, Some(Arc::new(MockEmbedder::new(DIM))));
    let ids: Vec<_> = ["ORG", "LOC", "FACILITY"]
        .iter()
        .enumerate()
        .map(|(i, label)| {
            let ctx = ed_context(
                "Riverside Arena",
                label,
                "CASE-1",
                &format!("doc-{i}"),
                &format!("m{i}"),
            );
            creator.create_entity_for_mention(&ctx, &[], None).expect("runs").expect("entity")
        })
        .collect();
    assert!(ids.iter().all(|id| *id == ids[0]), "{ids:?}");
    assert_eq!(entities(&memory).len(), 1);
}

#[test]
fn person_and_org_with_same_name_stay_separate() {
    for (first, second) in [("PER", "ORG"), ("PERSON", "ORG"), ("ORG", "PER")] {
        let (memory, store) = store();
        let creator = creator(&store, Some(Arc::new(MockEmbedder::new(DIM))));
        let a = creator
            .create_entity_for_mention(
                &ed_context("Mercury", first, "CASE-1", "doc-1", "m1"),
                &[],
                None,
            )
            .expect("runs")
            .expect("entity");
        let b = creator
            .create_entity_for_mention(
                &ed_context("Mercury", second, "CASE-1", "doc-2", "m2"),
                &[],
                None,
            )
            .expect("runs")
            .expect("entity");
        assert_ne!(a, b, "{first} vs {second}");
        assert_eq!(entities(&memory).len(), 2);
    }
}

#[test]
fn product_and_facility_stay_separate() {
    let (memory, store) = store();
    let creator = creator(&store, None);
    let product = creator
        .create_entity_for_mention(
            &ed_context("Plant 4", "PRODUCT", "CASE-1", "doc-1", "m1"),
            &[],
            None,
        )
        .expect("runs")
        .expect("entity");
    let facility = creator
        .create_entity_for_mention(
            &ed_context("Plant 4", "FACILITY", "CASE-1", "doc-2", "m2"),
            &[],
            None,
        )
        .expect("runs")
        .expect("entity");
    assert_ne!(product, facility);
    assert_eq!(entities(&memory).len(), 2);
}

#[test]
fn incompatible_candidates_are_skipped_on_every_strategy() {
    for strategy in [
        EntityRetrievalStrategy::Bm25,
        EntityRetrievalStrategy::VectorFuzzy,
        EntityRetrievalStrategy::AllEntitiesFuzzy,
    ] {
        let (memory, store) = store();
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(DIM));
        let per =
            seed_entity(&memory, "Mercury", "PER", "[]", "{\"normalized_name\": \"mercury\"}");
        memory
            .create_entity_embedding(&per, &embedder.embed(&["Mercury"]).expect("vector")[0])
            .expect("embedding");
        let creator = creator(&store, Some(embedder)).with_retrieval_strategy(strategy);
        let outcome = creator
            .create_or_reuse_entity_for_mention(
                &ed_context("Mercury", "ORG", "C", "d", "m"),
                &[],
                None,
            )
            .expect("runs")
            .expect("entity");
        assert!(matches!(&outcome, EntityOutcome::Created(id) if *id != per), "{strategy:?}");
    }
}

#[test]
fn cross_type_reuse_needs_a_strong_match() {
    // "Riverside Arena Annex West" scores 0.875 against "Riverside Arena" (ORG similarity).
    let mention = ed_context("Riverside Arena Annex West", "ORG", "C", "d", "m1");

    // Same type: reused at the 0.72 threshold.
    let (memory, store) = store();
    let org = seed_entity(&memory, "Riverside Arena", "ORG", "[]", "{}");
    let creator = creator(&store, None).with_reuse_similarity_threshold(0.72);
    let outcome = creator.create_or_reuse_entity_for_mention(&mention, &[], None).expect("runs");
    assert_eq!(outcome, Some(EntityOutcome::Reused(org)));

    // Compatible other type (LOC): needs 0.90, so a new entity.
    let (memory, store) = ed_support::store();
    let loc = seed_entity(&memory, "Riverside Arena", "LOC", "[]", "{}");
    let creator = ed_support::creator(&store, None).with_reuse_similarity_threshold(0.72);
    let outcome = creator
        .create_or_reuse_entity_for_mention(&mention, &[], None)
        .expect("runs")
        .expect("entity");
    assert!(matches!(&outcome, EntityOutcome::Created(id) if *id != loc));
}

#[test]
fn role_named_entities_are_not_reused_for_real_names() {
    let (memory, store) = store();
    let role_entity = seed_entity(
        &memory,
        "Plaintiff",
        "PER",
        "[\"Maria Lopez\"]",
        "{\"normalized_name\": \"maria lopez\"}",
    );
    let outcome = creator(&store, None)
        .create_or_reuse_entity_for_mention(
            &ed_context("Maria Lopez", "PER", "C", "d", "m"),
            &[],
            None,
        )
        .expect("runs")
        .expect("entity");
    assert!(matches!(&outcome, EntityOutcome::Created(id) if *id != role_entity));
}

#[test]
fn new_entities_get_an_embedding_only_with_an_embedder() {
    let (memory, store) = store();
    let embedder = Arc::new(MockEmbedder::new(DIM));
    let created = creator(&store, Some(embedder.clone()))
        .create_entity_for_mention(
            &ed_context("Maria Lopez", "PERSON", "CASE-1", "doc-1", "m1"),
            &strings(&["Ms. Lopez"]),
            None,
        )
        .expect("runs")
        .expect("entity");
    let expected = embedder
        .vector_for("Maria Lopez: Mentioned in: Maria Lopez did something. (Aliases: Ms. Lopez)");
    assert_eq!(memory.embedding(created.as_str()), Some(expected));

    let (memory, store) = ed_support::store();
    let created = creator(&store, None)
        .create_entity_for_mention(
            &ed_context("Maria Lopez", "PERSON", "CASE-1", "doc-1", "m1"),
            &[],
            None,
        )
        .expect("runs")
        .expect("entity");
    assert_eq!(memory.embedding(created.as_str()), None);
}

#[test]
fn extra_metadata_is_merged_before_roles() {
    let (memory, store) = store();
    let mut extra = ce_core::Metadata::new();
    extra.insert("case_id".into(), json!("OVERRIDE"));
    extra.insert("origin".into(), json!("test"));
    creator(&store, None)
        .create_entity_for_mention(&role_ctx("Plaintiff"), &strings(&["Maria Lopez"]), Some(&extra))
        .expect("runs")
        .expect("entity");
    let raw = entities(&memory)[0].node.metadata.clone();
    assert_eq!(
        raw,
        "{\"source_doc_id\": \"doc_123\", \"case_id\": \"OVERRIDE\", \"jurisdiction\": \"CA\", \
         \"normalized_name\": \"maria lopez\", \"origin\": \"test\", \"roles\": [\"PLAINTIFF\"]}"
    );
}
