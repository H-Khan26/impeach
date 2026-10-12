//! EdService over a MemoryStore: routing, hybrid heuristics, the LLM tie-break (MockLlm) and
//! candidate retrieval with real cosine scores.

mod ed_support;

use std::sync::Arc;

use ce_config::prompts::EdPrompt;
use ce_core::EdContext;
use ce_ed::{
    EdError, EdRetriever, EdService, EdServiceSettings, KeyPatterns, NameSimilarity,
    RetrieverSettings,
};
use ce_models::{Embedder, Llm, MockEmbedder, MockLlm};
use ce_store::{GraphStore, MemoryStore};
use ed_support::{ed_context, mappings, seed_entity, similarity, store, DIM};
use pretty_assertions::assert_eq;

const PROMPT_PATH: &str = "contradiction_engine/prompts/ed/disambiguation.json";

struct Fixture {
    memory: Arc<MemoryStore>,
    retriever: EdRetriever,
    similarity: NameSimilarity,
    patterns: KeyPatterns,
    prompt: EdPrompt,
}

impl Fixture {
    fn new(embedder: Option<Arc<dyn Embedder>>) -> Self {
        let (memory, store) = store();
        let similarity = similarity();
        let retriever = EdRetriever::new(
            store,
            embedder,
            RetrieverSettings {
                initial_vector_min_similarity: 0.55,
                ..RetrieverSettings::default()
            },
            similarity.clone(),
        );
        Self {
            memory,
            retriever,
            similarity,
            patterns: KeyPatterns::from_mappings(&mappings().detection_patterns),
            prompt: EdPrompt::load(PROMPT_PATH).expect("prompt loads"),
        }
    }

    fn service(&self, llm: Option<Arc<dyn Llm>>) -> EdService<'_> {
        EdService::new(
            &self.retriever,
            EdServiceSettings::default(),
            &self.similarity,
            &self.patterns,
        )
        .with_llm(llm)
        .with_prompt(Some(&self.prompt))
    }
}

fn ctx(name: &str, label: &str) -> EdContext {
    ed_context(name, label, "CASE-1", "doc-2", "m1")
}

#[test]
fn unique_true_exact_match_is_a_hybrid_heuristic() {
    let fixture = Fixture::new(None);
    let id = seed_entity(&fixture.memory, "Maria Lopez", "PER", "[]", "{}");
    seed_entity(&fixture.memory, "Maria Lopez-Garcia", "PER", "[]", "{}");
    let result = fixture
        .service(None)
        .resolve_span(&ctx("Maria Lopez", "PER"), false, 10)
        .expect("resolves");
    assert_eq!(result.chosen_entity_id, Some(id));
    assert_eq!(result.strategy_used, "hybrid_heuristic");
    assert!((result.confidence - 0.85).abs() < f64::EPSILON);
    assert!(result.candidates.iter().any(|c| c.source == "exact" && (c.score - 1.0).abs() < 1e-9));
}

#[test]
fn close_name_is_a_hybrid_similarity_match() {
    let fixture = Fixture::new(None);
    let id = seed_entity(
        &fixture.memory,
        "Green Valley Logistics, Inc.",
        "ORG",
        "[]",
        "{\"source_doc_id\": \"doc-1\"}",
    );
    let result = fixture
        .service(None)
        .resolve_span(&ctx("Green Valley Logistics", "ORG"), false, 10)
        .expect("resolves");
    assert_eq!(result.chosen_entity_id, Some(id));
    assert_eq!(result.strategy_used, "hybrid_similarity");
    assert!(result.confidence > 0.72 * 0.9 && result.confidence <= 0.9, "{}", result.confidence);
}

#[test]
fn ambiguous_candidates_go_to_the_llm() {
    let fixture = Fixture::new(None);
    // Both names score 0.675 against the mention: below the hybrid threshold.
    let a = seed_entity(&fixture.memory, "Green Valley Logistics", "ORG", "[]", "{}");
    let b = seed_entity(&fixture.memory, "Valley Medical Center", "ORG", "[]", "{}");
    let answer = format!(
        "```json\n{{\"chosen_entity_id\": \"{b}\", \"confidence\": 0.9, \"rationale\": \"x\"}}\n```"
    );
    let llm = Arc::new(MockLlm::with_responses([answer]));
    let result = fixture
        .service(Some(llm.clone()))
        .resolve_span(&ctx("Valley Transport", "ORG"), true, 10)
        .expect("resolves");
    assert_eq!(result.chosen_entity_id, Some(b.clone()));
    assert_eq!(result.strategy_used, "hybrid_llm");
    assert!(result.resolved_by_llm);
    assert!((result.confidence - 0.9).abs() < f64::EPSILON);
    let ids: Vec<_> = result.candidates.iter().map(|c| c.entity_id.clone()).collect();
    assert!(ids.contains(&a) && ids.contains(&b));

    let prompts = llm.prompts();
    assert_eq!(prompts.len(), 1);
    let prompt = &prompts[0];
    assert!(prompt.starts_with("You are an expert legal entity disambiguation assistant."));
    assert!(prompt.contains("ORGANIZATION Entity Guidelines:"));
    assert!(prompt
        .contains("\n\nPayload:\n{\"mention\": {\"id\": \"m1\", \"text\": \"Valley Transport\""));
    assert!(prompt.contains("\"case_id\": \"CASE-1\""));
}

#[test]
fn llm_without_choice_or_with_errors_leaves_the_mention_unresolved() {
    let fixture = Fixture::new(None);
    seed_entity(&fixture.memory, "Green Valley Logistics", "ORG", "[]", "{}");
    seed_entity(&fixture.memory, "Valley Medical Center", "ORG", "[]", "{}");
    let mention = ctx("Valley Transport", "ORG");

    let null_choice = Arc::new(MockLlm::with_responses([
        "{\"chosen_entity_id\": null, \"confidence\": 0.0, \"rationale\": \"none\"}",
    ]));
    let result = fixture.service(Some(null_choice)).resolve_span(&mention, true, 10).expect("runs");
    assert_eq!(result.chosen_entity_id, None);
    assert_eq!(result.strategy_used, "hybrid_llm");
    assert!(!result.resolved_by_llm);

    for llm in
        [MockLlm::new(), MockLlm::with_responses(["not json"]), MockLlm::with_responses(["[1]"])]
    {
        let result =
            fixture.service(Some(Arc::new(llm))).resolve_span(&mention, true, 10).expect("runs");
        assert_eq!(result.chosen_entity_id, None);
        assert_eq!(result.strategy_used, "hybrid_unresolved");
    }

    // allow_llm = false never calls the model.
    let llm = Arc::new(MockLlm::new());
    let result =
        fixture.service(Some(llm.clone())).resolve_span(&mention, false, 10).expect("runs");
    assert_eq!(result.strategy_used, "hybrid_unresolved");
    assert!(llm.prompts().is_empty());
}

#[test]
fn other_labels_are_llm_first() {
    let fixture = Fixture::new(None);
    let truck = seed_entity(&fixture.memory, "Freightliner truck", "PRODUCT", "[]", "{}");
    let mention = ctx("Freightliner truck", "PRODUCT");
    let result = fixture.service(None).resolve_span(&mention, true, 10).expect("runs");
    assert_eq!(result.chosen_entity_id, None);
    assert_eq!(result.strategy_used, "llm_skipped");
    assert_eq!(result.candidates.len(), 1);

    let answer = format!("{{\"chosen_entity_id\": \"{truck}\", \"confidence\": \"0.8\"}}");
    let llm = Arc::new(MockLlm::with_responses([answer]));
    let result = fixture.service(Some(llm.clone())).resolve_span(&mention, true, 10).expect("runs");
    assert_eq!(result.chosen_entity_id, Some(truck));
    assert_eq!(result.strategy_used, "llm_first");
    assert!((result.confidence - 0.8).abs() < f64::EPSILON);
    // PRODUCT has no type guidance.
    assert!(!llm.prompts()[0].contains("Entity Guidelines"));
}

#[test]
fn llm_only_routes_every_label_through_the_llm() {
    let fixture = Fixture::new(None);
    let id = seed_entity(&fixture.memory, "Maria Lopez", "PER", "[]", "{}");
    let llm: Arc<dyn Llm> =
        Arc::new(MockLlm::with_responses([format!("{{\"chosen_entity_id\": \"{id}\"}}")]));
    let service = EdService::new(
        &fixture.retriever,
        EdServiceSettings { llm_only: true, ..EdServiceSettings::default() },
        &fixture.similarity,
        &fixture.patterns,
    )
    .with_llm(Some(llm))
    .with_prompt(Some(&fixture.prompt));
    let result = service.resolve_span(&ctx("Maria Lopez", "PER"), true, 10).expect("runs");
    assert_eq!(result.strategy_used, "llm_first");
    assert_eq!(result.chosen_entity_id, Some(id));
    assert!(result.resolved_by_llm);
    assert!(result.confidence.abs() < f64::EPSILON);
}

#[test]
fn missing_prompt_is_an_error_when_the_llm_is_needed() {
    let fixture = Fixture::new(None);
    seed_entity(&fixture.memory, "Freightliner truck", "PRODUCT", "[]", "{}");
    let llm: Arc<dyn Llm> = Arc::new(MockLlm::new());
    let service = EdService::new(
        &fixture.retriever,
        EdServiceSettings::default(),
        &fixture.similarity,
        &fixture.patterns,
    )
    .with_llm(Some(llm));
    let error = service
        .resolve_span(&ctx("Freightliner truck", "PRODUCT"), true, 10)
        .expect_err("no prompt");
    assert!(matches!(error, EdError::Prompt(_)), "{error}");
}

#[test]
fn embedding_candidates_carry_cosine_scores_and_respect_type_groups() {
    let query = "Riverside Arena Riverside Arena did something.";
    let embedder = Arc::new(MockEmbedder::new(DIM).with_vector(query, vec![1.0; DIM]));
    let fixture = Fixture::new(Some(embedder.clone()));
    let unit = vec![1.0_f32; DIM];
    let venue = seed_entity(&fixture.memory, "The Pavilion", "LOC", "[]", "{}");
    let person = seed_entity(&fixture.memory, "Pavilion Smith", "PER", "[]", "{}");
    let product = seed_entity(&fixture.memory, "Pavilion 3000", "PRODUCT", "[]", "{}");
    for id in [&venue, &person, &product] {
        fixture.memory.create_entity_embedding(id, &unit).expect("embedding");
    }
    let service = fixture.service(None);
    let candidates =
        service.retrieve_candidates(&ctx("Riverside Arena", "ORG"), 10).expect("candidates");
    let ids: Vec<_> = candidates.iter().map(|c| c.entity_id.clone()).collect();
    assert_eq!(ids, vec![venue]);
    assert_eq!(candidates[0].source, "embedding");
    assert!((candidates[0].score - 1.0).abs() < 1e-6, "{}", candidates[0].score);

    // The vector candidate wins the hybrid route only on name similarity.
    let result = service.resolve_span(&ctx("Riverside Arena", "ORG"), false, 10).expect("runs");
    assert_eq!(result.chosen_entity_id, None);
    assert_eq!(result.strategy_used, "hybrid_unresolved");
}

#[test]
fn vector_candidate_with_a_close_name_is_reported_as_vector_fuzzy() {
    let query = "Riverside Arena LLC Riverside Arena LLC did something.";
    let embedder = Arc::new(MockEmbedder::new(DIM).with_vector(query, vec![1.0; DIM]));
    let fixture = Fixture::new(Some(embedder));
    // A name BM25 cannot find (no shared token) but the vector search can.
    let arena = seed_entity(&fixture.memory, "RiversideArena", "ORG", "[]", "{}");
    fixture.memory.create_entity_embedding(&arena, &[1.0; DIM]).expect("embedding");
    let service = fixture.service(None);
    let result = service.resolve_span(&ctx("Riverside Arena LLC", "ORG"), false, 10).expect("runs");
    let embedding = result.candidates.iter().find(|c| c.entity_id == arena).expect("candidate");
    assert_eq!(embedding.source, "embedding");
    assert_eq!(result.chosen_entity_id, Some(arena));
    assert_eq!(result.strategy_used, "vector_fuzzy");
}

#[test]
fn cross_doc_boost_prefers_entities_from_other_documents() {
    let fixture = Fixture::new(None);
    let service = fixture.service(None);
    let same_doc = seed_entity(
        &fixture.memory,
        "Lopez Freight Co",
        "ORG",
        "[]",
        "{\"source_doc_id\": \"doc-2\"}",
    );
    let other_doc = seed_entity(
        &fixture.memory,
        "Lopez Freight Co",
        "ORG",
        "[]",
        "{\"source_doc_id\": \"doc-1\"}",
    );
    let result = service.resolve_span(&ctx("Lopez Freight", "ORG"), false, 10).expect("runs");
    assert_eq!(result.chosen_entity_id, Some(other_doc), "same-doc entity {same_doc}");
    assert_eq!(result.strategy_used, "hybrid_similarity");
}

#[test]
fn retrieval_is_global_across_cases() {
    // test_helix_ed_retriever_ignores_case_id_and_searches_globally (service view).
    let fixture = Fixture::new(None);
    let ids: Vec<_> = ["CASE-1", "CASE-2"]
        .iter()
        .map(|case| {
            seed_entity(
                &fixture.memory,
                "Green Valley Logistics, Inc.",
                "ORG",
                "[\"Green Valley Logistics\"]",
                &format!("{{\"case_id\": \"{case}\"}}"),
            )
        })
        .chain([seed_entity(&fixture.memory, "Green Valley Logistics, Inc.", "ORG", "[]", "{}")])
        .collect();
    let candidates = fixture
        .service(None)
        .retrieve_candidates(&ctx("Green Valley Logistics, Inc.", "ORG"), 10)
        .expect("candidates");
    let mut found: Vec<_> = candidates.iter().map(|c| c.entity_id.clone()).collect();
    found.sort();
    let mut expected = ids;
    expected.sort();
    assert_eq!(found, expected);
    assert!(candidates.iter().all(|c| c.source == "exact"));
}
