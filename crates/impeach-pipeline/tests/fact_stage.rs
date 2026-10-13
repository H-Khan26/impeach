//! FactStage: skips (no index, offline run, missing LLM), the commit and pending paths, fact
//! annotation and linking, the resolved-text view, prompt entities, the 64+ facts regression
//! through `MockLlm`, multi-pass extraction and a Green Valley smoke run through every stage.

mod common;
mod ed_common;

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use impeach_config::pipeline::{ExtractionMode, FactStageConfig};
use impeach_config::{PipelineConfig, StageConfig};
use impeach_core::{
    AssertionTriple, ChunkId, ChunkWriteResult, DocId, DocumentPayload, EdCandidate, EdResult,
    EntityId, ExtractedFact, Mention, MentionId, Span, TemporalContext, TemporalValidity,
    TimeWindow,
};
use impeach_facts::{FactExtractor, PromptEntity};
use impeach_models::{MockEmbedder, MockLlm, ModelKind, ModelRegistry};
use impeach_pipeline::{
    run_processing_with, FactStage, Pipeline, PipelineContext, PipelineError, PipelineInput,
    ProcessingOptions, RagIndexStage, Stage, StageBuildContext,
};
use impeach_store::graph::{FACT, FACT_ABOUT_ENTITY, FACT_DERIVED_FROM_SPAN, FACT_IN_DOCUMENT};
use impeach_store::{EntityNode, GraphStore, MemoryStore};
use indexmap::IndexMap;
use pretty_assertions::assert_eq;
use serde_json::{json, Value};

const DIM: usize = common::DIM;

const TEXT: &str = "Maria Lopez sued Green Valley Logistics after the collision on Mission \
Street. The Plaintiff alleges that Daniel Reed ran a red light. She was not wearing a seat belt.";

const MULTIPASS_PROMPTS: [&str; 4] = [
    "impeach/prompts/processing/multipass/processing_mullti_v2/pass1_incident.json",
    "impeach/prompts/processing/multipass/processing_mullti_v2/pass2_party.json",
    "impeach/prompts/processing/multipass/processing_mullti_v2/pass3_history.json",
    "impeach/prompts/processing/multipass/processing_mullti_v2/pass4_damages.json",
];

type ExtractFn = dyn Fn(&str, &[PromptEntity]) -> Vec<ExtractedFact> + Send + Sync;

/// Extractor answering through a closure and recording the text and entities of every call.
struct ScriptedExtractor {
    respond: Box<ExtractFn>,
    calls: Mutex<Vec<(String, Vec<PromptEntity>)>>,
}

impl std::fmt::Debug for ScriptedExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScriptedExtractor").finish_non_exhaustive()
    }
}

impl ScriptedExtractor {
    fn new(
        respond: impl Fn(&str, &[PromptEntity]) -> Vec<ExtractedFact> + Send + Sync + 'static,
    ) -> Arc<Self> {
        Arc::new(Self { respond: Box::new(respond), calls: Mutex::new(Vec::new()) })
    }

    fn calls(&self) -> Vec<(String, Vec<PromptEntity>)> {
        self.calls.lock().expect("lock").clone()
    }
}

impl FactExtractor for ScriptedExtractor {
    fn extract_facts(&self, chunk_text: &str, entities: &[PromptEntity]) -> Vec<ExtractedFact> {
        self.calls.lock().expect("lock").push((chunk_text.to_owned(), entities.to_vec()));
        (self.respond)(chunk_text, entities)
    }
}

fn fact(id: &str, text: &str, subject: &str, relationship: &str, object: &str) -> ExtractedFact {
    ExtractedFact::new(
        id,
        text,
        AssertionTriple::new(subject, relationship, object),
        0.9,
        TemporalValidity::default(),
    )
}

/// The four facts the scripted extractor returns for [`TEXT`]: linked by name, by role term
/// (role map), unlinked, and boilerplate.
fn text_facts() -> Vec<ExtractedFact> {
    let mut sued = fact(
        "STATED_FACT_0",
        "Maria Lopez sued Green Valley.",
        "Maria Lopez",
        "SUED",
        "Green Valley Logistics",
    );
    sued.slot_name = Some("lawsuit".to_owned());
    sued.value_raw = Some("sued".to_owned());
    let mut red_light = fact(
        "STATED_FACT_1",
        "Plaintiff alleges Reed ran a red light.",
        "Plaintiff",
        "ALLEGES",
        "Daniel Reed",
    );
    red_light.value_raw = Some(String::new());
    let mut belt = fact(
        "STATED_FACT_2",
        "She was not wearing a seat belt.",
        "seat belt",
        "NOT_WORN_BY",
        "occupant",
    );
    belt.is_denial = true;
    belt.temporal_context = TemporalContext::PriorIncident;
    belt.time_window = Some(TimeWindow { start: Some("2023-03-04".to_owned()), end: None });
    let mut boilerplate =
        fact("STATED_FACT_3", "All rights reserved.", "Plaintiff", "RESERVES", "rights");
    boilerplate.is_boilerplate = true;
    vec![sued, red_light, belt, boilerplate]
}

fn fact_config(edit: impl FnOnce(&mut FactStageConfig)) -> PipelineConfig {
    let mut config = common::default_config();
    let fact = config.stages.iter_mut().find_map(|stage| match stage {
        StageConfig::Fact(fact) => Some(fact),
        _ => None,
    });
    edit(fact.expect("config has a FactStage"));
    config
}

fn stage_config(config: &PipelineConfig) -> FactStageConfig {
    config
        .stages
        .iter()
        .find_map(|stage| match stage {
            StageConfig::Fact(fact) => Some(fact.clone()),
            _ => None,
        })
        .expect("config has a FactStage")
}

/// Services over a fresh `MemoryStore`: a `MockEmbedder` as `bge` and `llm` (if any) as the
/// `ollama` model, in a registry without config-built models.
struct Env {
    store: Arc<MemoryStore>,
    embedder: Arc<MockEmbedder>,
    services: StageBuildContext,
}

fn setup(config: PipelineConfig, llm: Option<Arc<MockLlm>>) -> Env {
    let store = Arc::new(MemoryStore::new(DIM));
    let embedder = Arc::new(MockEmbedder::new(DIM));
    let registry = ModelRegistry::empty(config.name());
    registry.insert_embedder(registry.key(ModelKind::Bge), embedder.clone());
    if let Some(llm) = llm {
        registry.insert_llm(registry.key(ModelKind::Ollama), llm);
    }
    let dyn_store: Arc<dyn GraphStore> = store.clone();
    let services = StageBuildContext::new(Arc::new(config), Arc::new(registry), dyn_store);
    Env { store, embedder, services }
}

impl Env {
    fn stage(&self) -> FactStage {
        FactStage::new(&stage_config(&self.services.config), &self.services).expect("stage builds")
    }

    /// A context after RAGIndexStage on `text`.
    fn indexed(&self, text: &str) -> PipelineContext {
        let rag = self
            .services
            .config
            .stages
            .iter()
            .find_map(|stage| match stage {
                StageConfig::RagIndex(rag) => Some(rag.clone()),
                _ => None,
            })
            .expect("RAGIndexStage config");
        let stage = RagIndexStage::new(&rag, &self.services).expect("rag stage builds");
        let pipeline = Pipeline::new(vec![Box::new(stage)], self.services.clone());
        let mut ctx = pipeline.new_context();
        pipeline
            .run(&mut ctx, PipelineInput::Payload(Box::new(DocumentPayload::new("doc-1", text))))
            .expect("indexing runs");
        assert!(ctx.data.index_persisted);
        ctx
    }

    fn count_label(&self, label: &str) -> usize {
        (0..5000_u64)
            .filter_map(|id| self.store.node_snapshot(&id.to_string()))
            .filter(|(node_label, _)| *node_label == label)
            .count()
    }
}

/// Entities of [`TEXT`] written to the store, with ED results, mentions and the role map in
/// the context. Returns `(maria, green_valley, reed)` entity ids.
fn add_entities(env: &Env, ctx: &mut PipelineContext) -> (EntityId, EntityId, EntityId) {
    let entity = |name: &str, label: &str| {
        env.store.create_entity(&EntityNode::new(name, label)).expect("entity").id
    };
    let maria = entity("Maria Lopez", "PER");
    let gv = entity("Green Valley Logistics, Inc.", "ORG");
    let reed = entity("Daniel Reed", "PER");
    let surfaces: [(&str, &str, &EntityId); 5] = [
        ("Maria Lopez", "PER", &maria),
        ("Green Valley Logistics", "ORG", &gv),
        ("Plaintiff", "PER", &maria),
        ("Daniel Reed", "PER", &reed),
        ("She", "PER", &maria),
    ];

    let index = ctx.data.rag_index_result.clone().expect("indexed");
    let mut by_chunk: IndexMap<ChunkId, Vec<Mention>> = IndexMap::new();
    let mut doc_mentions = Vec::new();
    let mut resolved = IndexMap::new();
    for chunk in &index.chunks {
        let text = &chunk.chunk.text;
        for (i, (surface, label, entity_id)) in surfaces.iter().enumerate() {
            for (byte, _) in text.match_indices(surface) {
                let start = chunk.chunk.start_char + text[..byte].chars().count();
                let id = format!("m-{}-{i}-{start}", chunk.chunk.sequence_number);
                let mention = Mention::new(
                    id.as_str(),
                    *label,
                    *surface,
                    Span::new(start, start + surface.chars().count()),
                );
                resolved.insert(MentionId::new(id), (*entity_id).clone());
                by_chunk
                    .entry(ChunkId::new(chunk.helix_id.as_str()))
                    .or_default()
                    .push(mention.clone());
                doc_mentions.push(mention);
            }
        }
    }
    let result = |id: &EntityId, name: &str, label: &str, aliases: &[&str]| {
        let mut candidate = EdCandidate::new(id.clone(), name, label);
        candidate.aliases = aliases.iter().map(|a| (*a).to_owned()).collect();
        EdResult {
            mention_id: None,
            chosen_entity_id: Some(id.clone()),
            candidates: vec![candidate],
            resolved_by_llm: false,
            confidence: 1.0,
            strategy_used: "rule_create".to_owned(),
        }
    };
    ctx.data.ed_results = Some(vec![
        result(&maria, "Maria Lopez", "PER", &["Lopez", "Ms. Lopez"]),
        result(&gv, "Green Valley Logistics, Inc.", "ORG", &["Green Valley"]),
        result(&reed, "Daniel Reed", "PER", &["Reed"]),
    ]);
    ctx.data.mentions_by_chunk = Some(by_chunk);
    ctx.data.doc_mentions = Some(doc_mentions);
    ctx.data.resolved_mentions = Some(resolved);
    ctx.data.role_map = Some(IndexMap::from([("PLAINTIFF".to_owned(), maria.clone())]));
    (maria, gv, reed)
}

fn edge_targets(store: &MemoryStore, label: &str, fact_id: &str) -> Vec<(String, Value)> {
    store
        .edges()
        .into_iter()
        .filter(|edge| edge.label == label && edge.from == fact_id)
        .map(|edge| (edge.to, Value::Object(edge.properties)))
        .collect()
}

fn assert_untouched(ctx: &PipelineContext) {
    let data = &ctx.data;
    assert!(data.extracted_facts.is_none());
    assert!(data.fact_linking_stats.is_none());
    assert!(data.extraction_mode.is_none());
    assert!(data.indexed_facts_count.is_none());
    assert!(data.fact_helix_ids.is_none());
    assert!(data.fact_nodes_pending.is_none());
    assert!(data.resolved_text_by_chunk.is_none());
}

// --- build and skips ---

#[test]
fn missing_prompt_file_fails_the_build() {
    let config = fact_config(|fact| fact.prompt_files = vec!["no/such/prompt.json".to_owned()]);
    let env = setup(config, None);
    let error = FactStage::new(&stage_config(&env.services.config), &env.services)
        .expect_err("missing prompt file");
    assert!(matches!(error, PipelineError::Config(_)), "{error}");
}

#[test]
fn construct_rejects_other_stage_types() {
    let env = setup(common::default_config(), None);
    let error = FactStage::construct(&env.services.config.stages[0], &env.services)
        .err()
        .expect("RAGIndexStage config is rejected");
    assert!(matches!(error, PipelineError::InvalidStageConfig { .. }), "{error}");
}

#[test]
fn without_index_result_nothing_happens() {
    let extractor = ScriptedExtractor::new(|_, _| text_facts());
    let env = setup(common::default_config(), None);
    let mut ctx = PipelineContext::new(&env.services);
    env.stage().with_extractor(extractor.clone()).run(&mut ctx).expect("runs");
    assert_untouched(&ctx);
    assert!(extractor.calls().is_empty());
}

#[test]
fn offline_run_skips_without_writing() {
    let extractor = ScriptedExtractor::new(|_, _| text_facts());
    let env = setup(common::default_config(), None);
    let stage = env.stage().with_extractor(extractor.clone());

    let mut not_persisted = env.indexed(TEXT);
    not_persisted.data.index_persisted = false;
    stage.run(&mut not_persisted).expect("runs");
    assert_untouched(&not_persisted);

    let mut no_embedder = env.indexed(TEXT);
    no_embedder.data.embedder = None;
    stage.run(&mut no_embedder).expect("runs");
    assert_untouched(&no_embedder);

    assert!(extractor.calls().is_empty());
}

#[test]
fn missing_llm_skips_the_stage() {
    let env = setup(common::default_config(), None);
    let mut ctx = env.indexed(TEXT);
    env.stage().run(&mut ctx).expect("runs");
    assert_untouched(&ctx);

    // A model type without a registry entry.
    let config = fact_config(|fact| fact.model_type = "qwen".to_owned());
    let env = setup(config, Some(Arc::new(MockLlm::new())));
    let mut ctx = env.indexed(TEXT);
    env.stage().run(&mut ctx).expect("runs");
    assert_untouched(&ctx);
}

// --- pending and commit paths ---

#[test]
fn pending_path_leaves_nodes_for_the_contradiction_stage() {
    let extractor = ScriptedExtractor::new(|_, _| text_facts());
    let env = setup(fact_config(|fact| fact.commit_facts = false), None);
    let mut ctx = env.indexed(TEXT);
    let (maria, gv, reed) = add_entities(&env, &mut ctx);
    env.stage().with_extractor(extractor).run(&mut ctx).expect("runs");

    let data = &ctx.data;
    let facts = data.extracted_facts.as_ref().expect("facts");
    assert_eq!(facts.len(), 3, "boilerplate fact dropped");
    assert_eq!(data.indexed_facts_count, Some(0));
    assert_eq!(data.fact_helix_ids, Some(Vec::new()));
    assert_eq!(data.extraction_mode, Some(ExtractionMode::Processing));
    let nodes = data.fact_nodes_pending.as_ref().expect("pending nodes");
    assert_eq!(nodes.len(), 3);
    assert_eq!(env.count_label(FACT), 0, "nothing written to the store");

    // Linking: by name, by the role map (rewritten to the canonical name), pronoun unlinked.
    assert_eq!(facts[0].subject_entity_id.as_ref(), Some(&maria));
    assert_eq!(facts[0].object_entity_id.as_ref(), Some(&gv));
    assert_eq!(facts[1].subject_entity_id.as_ref(), Some(&maria));
    assert_eq!(facts[1].assertion_triple.subject, "Maria Lopez");
    assert_eq!(facts[1].object_entity_id.as_ref(), Some(&reed));
    assert_eq!(facts[2].subject_entity_id, None);
    let stats = data.fact_linking_stats.expect("stats");
    assert_eq!(
        (
            stats.total_facts,
            stats.linked_subjects,
            stats.linked_objects,
            stats.linked_both,
            stats.linked_either
        ),
        (3, 2, 2, 2, 2)
    );

    // Pending nodes hold the linked, annotated facts in Python JSON form.
    let triple: Value = serde_json::from_str(&nodes[1].assertion_triple).expect("json");
    assert_eq!(triple["subject"], "Maria Lopez");
    let metadata: Value = serde_json::from_str(&nodes[0].metadata).expect("json");
    assert_eq!(metadata["extraction_mode"], "processing");
    assert!(nodes[0].metadata.starts_with("{\"source_chunk_id\": "), "{}", nodes[0].metadata);
}

#[test]
fn commit_path_writes_facts_embeddings_and_edges() {
    let extractor = ScriptedExtractor::new(|_, _| text_facts());
    let env = setup(fact_config(|fact| fact.commit_facts = true), None);
    let mut ctx = env.indexed(TEXT);
    let (maria, gv, _) = add_entities(&env, &mut ctx);
    env.stage().with_extractor(extractor).run(&mut ctx).expect("runs");

    let data = &ctx.data;
    let facts = data.extracted_facts.as_ref().expect("facts");
    let ids = data.fact_helix_ids.as_ref().expect("ids");
    assert_eq!(facts.len(), 3);
    assert_eq!(ids.len(), 3);
    assert_eq!(data.indexed_facts_count, Some(3));
    assert!(data.fact_nodes_pending.is_none());
    assert_eq!(env.count_label(FACT), 3);

    let doc_id = data.helix_doc_id.clone().expect("document id");
    let index = data.rag_index_result.as_ref().expect("indexed");
    let sentence_spans: Vec<String> = index.spans.iter().map(|s| s.span_id.to_string()).collect();
    assert!(!sentence_spans.is_empty());
    for (fact, id) in facts.iter().zip(ids) {
        let id = id.as_ref().expect("written").as_str();
        assert_eq!(fact.metadata["helix_fact_id"], json!(id));
        let vector = env.store.embedding(id).expect("fact vector");
        assert_eq!(vector, env.embedder.vector_for(&fact.text));
        assert_eq!(
            edge_targets(&env.store, FACT_IN_DOCUMENT, id),
            vec![(doc_id.to_string(), json!({"PageNumber": 0, "Section": ""}))]
        );
        let spans: Vec<String> = edge_targets(&env.store, FACT_DERIVED_FROM_SPAN, id)
            .into_iter()
            .map(|(to, _)| to)
            .collect();
        assert_eq!(spans, sentence_spans);
    }
    let first = ids[0].as_ref().expect("written").as_str();
    assert_eq!(
        edge_targets(&env.store, FACT_ABOUT_ENTITY, first),
        vec![
            (maria.to_string(), json!({"Role": "SUBJECT"})),
            (gv.to_string(), json!({"Role": "OBJECT"})),
        ]
    );
    let maria_facts = env.store.get_entity_facts(&maria).expect("query");
    assert_eq!(maria_facts.len(), 2);
}

#[test]
fn facts_get_source_ids_and_metadata_in_python_order() {
    let extractor = ScriptedExtractor::new(|_, _| text_facts());
    let env = setup(common::default_config(), None);
    let mut ctx = env.indexed(TEXT);
    add_entities(&env, &mut ctx);
    env.stage().with_extractor(extractor).run(&mut ctx).expect("runs");

    let data = &ctx.data;
    let facts = data.extracted_facts.as_ref().expect("facts");
    let chunk = &data.rag_index_result.as_ref().expect("indexed").chunks[0];
    let doc_id = data.helix_doc_id.as_ref().expect("doc id");
    let span_ids: Vec<Value> = data
        .rag_index_result
        .as_ref()
        .expect("indexed")
        .spans
        .iter()
        .filter(|s| s.chunk_sequence == chunk.chunk.sequence_number)
        .map(|s| json!(s.span_id.as_str()))
        .collect();
    for fact in facts {
        assert_eq!(
            fact.source_chunk_id.as_ref().map(ChunkId::as_str),
            Some(chunk.helix_id.as_str())
        );
        assert_eq!(fact.source_doc_id.as_ref().map(DocId::as_str), Some(doc_id.as_str()));
        assert_eq!(fact.metadata["source_chunk_id"], json!(chunk.helix_id.as_str()));
        assert_eq!(fact.metadata["source_span_ids"], Value::Array(span_ids.clone()));
    }
    let keys = |fact: &ExtractedFact| fact.metadata.keys().cloned().collect::<Vec<_>>();
    assert_eq!(
        keys(&facts[0]),
        [
            "source_chunk_id",
            "source_span_ids",
            "slot_name",
            "temporal_context",
            "is_denial",
            "extraction_mode",
            "value_raw"
        ]
    );
    assert_eq!(facts[0].metadata["slot_name"], "lawsuit");
    assert_eq!(facts[0].metadata["value_raw"], "sued");
    // Empty value_raw and no slot name: those keys are absent.
    assert_eq!(
        keys(&facts[1]),
        ["source_chunk_id", "source_span_ids", "temporal_context", "is_denial", "extraction_mode"]
    );
    assert_eq!(facts[2].metadata["temporal_context"], "prior_incident");
    assert_eq!(facts[2].metadata["is_denial"], true);
    assert_eq!(facts[2].metadata["time_window"], json!({"start": "2023-03-04", "end": null}));
}

#[test]
fn chunks_without_sentence_spans_get_no_span_ids() {
    let extractor = ScriptedExtractor::new(|_, _| text_facts());
    let env = setup(common::default_config(), None);
    let mut ctx = env.indexed(TEXT);
    ctx.data.rag_index_result.as_mut().expect("indexed").spans.clear();
    env.stage().with_extractor(extractor).run(&mut ctx).expect("runs");
    let facts = ctx.data.extracted_facts.as_ref().expect("facts");
    assert!(facts.iter().all(|f| !f.metadata.contains_key("source_span_ids")));
}

#[test]
fn short_chunks_are_skipped() {
    let extractor = ScriptedExtractor::new(|_, _| text_facts());
    let env = setup(common::default_config(), None);
    let stage = env.stage().with_extractor(extractor.clone());

    let mut short = env.indexed("Maria Lopez sued Green Valley.");
    stage.run(&mut short).expect("runs");
    assert!(extractor.calls().is_empty());
    assert_eq!(short.data.extracted_facts, Some(Vec::new()));
    assert_eq!(short.data.extraction_mode, Some(ExtractionMode::Processing));
    assert!(short.data.indexed_facts_count.is_none());
    assert!(short.data.fact_nodes_pending.is_none());
}

#[test]
fn resolved_text_view_is_sent_and_stored() {
    let extractor = ScriptedExtractor::new(|_, _| Vec::new());
    let env = setup(fact_config(|fact| fact.use_resolved_text_view = true), None);
    let mut ctx = env.indexed(TEXT);
    let (maria, gv, reed) = add_entities(&env, &mut ctx);
    env.stage().with_extractor(extractor.clone()).run(&mut ctx).expect("runs");

    let calls = extractor.calls();
    assert_eq!(calls.len(), 1);
    let expected = format!(
        "[{maria}:Maria Lopez] sued [{gv}:Green Valley Logistics, Inc.] after the collision on \
Mission Street. The [{maria}:Maria Lopez] alleges that [{reed}:Daniel Reed] ran a red light. \
[{maria}:Maria Lopez] was not wearing a seat belt."
    );
    assert_eq!(calls[0].0, expected);
    let chunk_id = ctx.data.rag_index_result.as_ref().expect("indexed").chunks[0].helix_id.clone();
    let stored = ctx.data.resolved_text_by_chunk.as_ref().expect("resolved texts");
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[chunk_id.as_str()], expected);
    // No facts: only the always-written keys.
    assert_eq!(ctx.data.extracted_facts, Some(Vec::new()));
    assert!(ctx.data.fact_helix_ids.is_none());

    // Without the view the chunk text is sent and nothing is stored.
    let extractor = ScriptedExtractor::new(|_, _| Vec::new());
    let env = env_without_view();
    let mut ctx = env.indexed(TEXT);
    add_entities(&env, &mut ctx);
    env.stage().with_extractor(extractor.clone()).run(&mut ctx).expect("runs");
    assert_eq!(extractor.calls()[0].0, TEXT);
    assert!(ctx.data.resolved_text_by_chunk.is_none());
}

fn env_without_view() -> Env {
    setup(fact_config(|fact| fact.use_resolved_text_view = false), None)
}

#[test]
fn bracket_tokens_from_the_resolved_view_link_facts() {
    // Answers with a fact whose subject is the first bracket token of the text.
    let extractor = ScriptedExtractor::new(|text, _| {
        let start = text.find('[').expect("token");
        let end = text[start..].find(']').expect("token end") + start + 1;
        vec![fact("STATED_FACT_0", "Token fact.", &text[start..end], "DID", "something")]
    });
    let env = setup(fact_config(|fact| fact.use_resolved_text_view = true), None);
    let mut ctx = env.indexed(TEXT);
    let (maria, _, _) = add_entities(&env, &mut ctx);
    env.stage().with_extractor(extractor).run(&mut ctx).expect("runs");
    let facts = ctx.data.extracted_facts.as_ref().expect("facts");
    assert_eq!(facts[0].subject_entity_id.as_ref(), Some(&maria));
    assert_eq!(facts[0].assertion_triple.subject, "Maria Lopez");
}

#[test]
fn prompt_entities_list_resolved_entities_without_pronouns_or_roles() {
    let extractor = ScriptedExtractor::new(|_, _| Vec::new());
    let env = env_without_view();
    let mut ctx = env.indexed(TEXT);
    let (maria, gv, reed) = add_entities(&env, &mut ctx);
    env.stage().with_extractor(extractor.clone()).run(&mut ctx).expect("runs");
    let entities = &extractor.calls()[0].1;
    let summary: Vec<(&str, &str, Vec<&str>)> = entities
        .iter()
        .map(|e| (e.id.as_str(), e.name.as_str(), e.aliases.iter().map(String::as_str).collect()))
        .collect();
    assert_eq!(
        summary,
        vec![
            (reed.as_str(), "Daniel Reed", vec!["Reed"]),
            (
                gv.as_str(),
                "Green Valley Logistics, Inc.",
                vec!["Green Valley Logistics", "Green Valley"]
            ),
            (maria.as_str(), "Maria Lopez", vec!["Lopez", "Ms. Lopez"]),
        ]
    );
}

// --- through MockLlm ---

/// A single-pass response with `count` facts about Daniel Reed.
fn many_facts_response(count: usize) -> String {
    let nodes: Vec<Value> = (0..count)
        .map(|i| {
            json!({
                "node_type": "STATED_FACT",
                "id": format!("STATED_FACT_{i}"),
                "text": format!("Daniel Reed made delivery number {i} on March 4, 2023."),
                "value_type": "NUMERIC",
                "property_type": "QUANTITATIVE",
                "slot_name": "delivery",
                "value_raw": i.to_string(),
                "assertion_triple": {"subject": "Daniel Reed", "relationship": "MADE_DELIVERY", "object": format!("delivery {i}")},
                "certainty": 0.9,
                "temporal_context": "current_incident",
                "is_denial": false,
                "is_boilerplate": false,
                "temporal_validity": {"t_valid_from": null, "t_valid_until": null},
                "metadata": {}
            })
        })
        .collect();
    json!({ "nodes": nodes }).to_string()
}

#[test]
fn hundred_facts_from_one_chunk_are_all_indexed() {
    // Regression: Python crashed on documents with 64+ facts (more than one embedding batch).
    let llm = Arc::new(MockLlm::with_responses([many_facts_response(100)]));
    let env = setup(fact_config(|fact| fact.commit_facts = true), Some(llm.clone()));
    let mut ctx = env.indexed(TEXT);
    let (_, _, reed) = add_entities(&env, &mut ctx);
    env.stage().run(&mut ctx).expect("runs");

    assert_eq!(llm.prompts().len(), 1);
    let data = &ctx.data;
    let facts = data.extracted_facts.as_ref().expect("facts");
    assert_eq!(facts.len(), 100);
    assert_eq!(data.indexed_facts_count, Some(100));
    let ids = data.fact_helix_ids.as_ref().expect("ids");
    assert_eq!(ids.iter().flatten().count(), 100);
    for (fact, id) in facts.iter().zip(ids) {
        let id = id.as_ref().expect("written");
        assert_eq!(env.store.embedding(id.as_str()), Some(env.embedder.vector_for(&fact.text)));
    }
    assert_eq!(env.store.get_entity_facts(&reed).expect("query").len(), 100);
    assert_eq!(data.fact_linking_stats.expect("stats").linked_subjects, 100);
}

#[test]
fn single_pass_prompt_lists_resolved_entities() {
    let llm = Arc::new(MockLlm::with_responses([many_facts_response(1)]));
    let env = setup(fact_config(|fact| fact.use_resolved_text_view = true), Some(llm.clone()));
    let mut ctx = env.indexed(TEXT);
    let (_, _, reed) = add_entities(&env, &mut ctx);
    env.stage().run(&mut ctx).expect("runs");
    let prompts = llm.prompts();
    assert_eq!(prompts.len(), 1);
    assert!(
        prompts[0].contains(&format!("[{reed}:Daniel Reed] ran a red light")),
        "{}",
        prompts[0]
    );
    assert!(prompts[0].contains("Daniel Reed (ID: "), "{}", prompts[0]);
    assert!(!prompts[0].contains("{segments}"));
    assert_eq!(ctx.data.extracted_facts.as_ref().expect("facts").len(), 1);
    assert!(ctx.data.fact_nodes_pending.is_some());
}

#[test]
fn llm_failure_yields_no_facts() {
    let env = setup(common::default_config(), Some(Arc::new(MockLlm::new())));
    let mut ctx = env.indexed(TEXT);
    env.stage().run(&mut ctx).expect("runs");
    assert_eq!(ctx.data.extracted_facts, Some(Vec::new()));
    assert!(ctx.data.fact_nodes_pending.is_none());
    assert_eq!(ctx.data.extraction_mode, Some(ExtractionMode::Processing));
}

#[test]
fn multi_pass_runs_every_prompt_and_merges_the_passes() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let llm = Arc::new(MockLlm::with_responder(move |_| {
        let call = counter.fetch_add(1, Ordering::SeqCst);
        Ok(json!({"nodes": [{
            "node_type": "STATED_FACT",
            "id": "STATED_FACT_0",
            "text": format!("Daniel Reed fact from call {call}."),
            "assertion_triple": {"subject": "Daniel Reed", "relationship": "DID", "object": format!("thing {call}")},
            "certainty": 0.9,
            "temporal_validity": {}
        }]})
        .to_string())
    }));
    let config = fact_config(|fact| {
        fact.prompt_files = MULTIPASS_PROMPTS.iter().map(|p| (*p).to_owned()).collect();
        fact.parallel = true;
        fact.max_workers = 4;
    });
    let env = setup(config, Some(llm.clone()));
    let mut ctx = env.indexed(TEXT);
    add_entities(&env, &mut ctx);
    env.stage().run(&mut ctx).expect("runs");

    assert_eq!(llm.prompts().len(), 4);
    let facts = ctx.data.extracted_facts.as_ref().expect("facts");
    assert_eq!(facts.len(), 4);
    let mut passes: Vec<&str> =
        facts.iter().map(|f| f.id.as_str().trim_end_matches("_STATED_FACT_0")).collect();
    passes.sort_unstable();
    assert_eq!(passes, ["damages_expenses", "history_employment", "incident_core", "party_state"]);
    assert_eq!(ctx.data.fact_linking_stats.expect("stats").linked_subjects, 4);
    assert_eq!(ctx.data.fact_nodes_pending.as_ref().expect("pending").len(), 4);
}

// --- smoke ---

/// Canned single-pass responses for the Green Valley incident report: each entry's nodes are
/// returned for prompts containing its `when` phrase.
fn green_valley_llm() -> MockLlm {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/llm/facts/green_valley/incident_report_responses.json");
    let raw = std::fs::read_to_string(path).expect("fixture reads");
    let entries: Vec<Value> = serde_json::from_str(&raw).expect("fixture is JSON");
    MockLlm::with_responder(move |request| {
        let nodes: Vec<Value> = entries
            .iter()
            .filter(|entry| request.prompt.contains(entry["when"].as_str().unwrap_or("\u{0}")))
            .flat_map(|entry| entry["response"]["nodes"].as_array().cloned().unwrap_or_default())
            .collect();
        Ok(format!("```json\n{}\n```", json!({ "nodes": nodes })))
    })
}

#[test]
fn smoke_green_valley_through_the_fact_stage() {
    let store = Arc::new(MemoryStore::new(ed_common::DIM));
    let ner = Arc::new(ed_common::GazetteerNer::new(ed_common::GREEN_VALLEY_NAMES));
    let mut env = ed_common::env(&store, ner);
    let registry = env.registry.take().expect("registry");
    let llm = Arc::new(green_valley_llm());
    registry.insert_llm(registry.key(ModelKind::Ollama), llm.clone());
    env.registry = Some(registry);

    let run = run_processing_with(
        common::fixture("green_valley_incident_report.jsonl"),
        &ProcessingOptions::default(),
        env,
    )
    .expect("pipeline runs");
    assert_eq!(
        run.stages,
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
    let data = &run.data;
    let chunks: Vec<&ChunkWriteResult> = data
        .rag_index_result
        .as_ref()
        .expect("indexed")
        .chunks
        .iter()
        .filter(|c| c.chunk.text.chars().count() >= impeach_pipeline::stages::fact::MIN_CHUNK_CHARS)
        .collect();
    assert!(chunks.len() > 1);
    assert_eq!(llm.prompts().len(), chunks.len(), "one LLM call per chunk");

    // Overlapping chunks may repeat a phrase; every canned fact but the boilerplate one appears.
    let facts = data.extracted_facts.as_ref().expect("facts");
    let texts: BTreeSet<&str> = facts.iter().map(|f| f.text.as_str()).collect();
    assert_eq!(texts.len(), 7, "{texts:?}");
    assert!(facts.iter().all(|f| !f.text.contains("reserves all rights")));
    let stats = data.fact_linking_stats.expect("stats");
    assert_eq!(stats.total_facts, facts.len());
    assert!(stats.linked_either > 0, "{stats:?}");

    // The default config (processing_v17) defers the commit to the contradiction stage.
    let pending = data.fact_nodes_pending.as_ref().expect("pending nodes");
    assert_eq!(pending.len(), facts.len());
    assert_eq!(data.indexed_facts_count, Some(0));
    assert_eq!(data.fact_helix_ids, Some(Vec::new()));
    assert_eq!(store.edges().iter().filter(|e| e.label == FACT_IN_DOCUMENT).count(), 0);
    let resolved_texts = data.resolved_text_by_chunk.as_ref().expect("resolved text view");
    assert_eq!(resolved_texts.len(), chunks.len());
    assert!(resolved_texts.values().any(|text| text.contains(":Daniel Reed]")));

    // Daniel Reed is linked by name.
    let reed_facts: Vec<&ExtractedFact> =
        facts.iter().filter(|f| f.assertion_triple.subject == "Daniel Reed").collect();
    assert!(!reed_facts.is_empty());
    assert!(reed_facts.iter().all(|f| f.subject_entity_id.is_some()));
    let denial = facts.iter().find(|f| f.slot_name.as_deref() == Some("prior_accidents"));
    let denial = denial.expect("denial fact");
    assert!(denial.is_denial);
    assert_eq!(denial.metadata["time_window"], json!({"start": "2013-03-22", "end": "2023-03-22"}));
    let seat_belt =
        pending.iter().find(|n| n.fact_text.contains("seat belt")).expect("seat belt node");
    assert_eq!(seat_belt.property_type, "EXISTENCE");
}
