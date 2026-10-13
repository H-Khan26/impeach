//! EDStage on hand-built contexts: offline runs, the input set, coreference propagation with
//! the CoreferenceStage's type groups (port of `TestEDPropagationUsesCorefGroups`), synthetic
//! pronoun mentions, the role map, span links and the LLM lookup.

mod common;

use std::sync::{Arc, Mutex};

use impeach_config::{PipelineConfig, StageConfig};
use impeach_core::processing::DefinedTerm;
use impeach_core::{
    Chunk, ChunkWriteResult, CorefClusterContext, DocumentWriteResult, EdResult, EntityId,
    IndexingResult, Mention, NodeId, Span, SpanId,
};
use impeach_ed::{EdResolver, Resolution, ResolveInput};
use impeach_models::{MockLlm, ModelRegistry};
use impeach_pipeline::{EdStage, PipelineContext, PipelineError, Stage, StageBuildContext};
use impeach_store::{EntityNode, GraphStore, MemoryStore};
use indexmap::IndexMap;
use pretty_assertions::assert_eq;

/// Resolves every input mention to `ent-<mention id>` and records the inputs.
#[derive(Debug, Default)]
struct StubResolver {
    inputs: Arc<Mutex<Vec<String>>>,
    saw_all_mentions: Arc<Mutex<Option<usize>>>,
}

impl EdResolver for StubResolver {
    fn resolve(
        &self,
        mentions: &[Mention],
        input: &ResolveInput<'_>,
    ) -> impeach_ed::Result<Resolution> {
        *self.inputs.lock().expect("lock") = mentions.iter().map(|m| m.id.to_string()).collect();
        *self.saw_all_mentions.lock().expect("lock") = input.all_mentions_by_id.map(IndexMap::len);
        let results = mentions
            .iter()
            .map(|m| {
                let result = EdResult {
                    mention_id: Some(m.id.clone()),
                    chosen_entity_id: Some(format!("ent-{}", m.id).into()),
                    candidates: vec![],
                    resolved_by_llm: false,
                    confidence: 0.0,
                    strategy_used: "stub".into(),
                };
                (m.id.clone(), result)
            })
            .collect();
        Ok(Resolution { results, outcomes: IndexMap::new() })
    }
}

fn services_with(config: PipelineConfig, registry: ModelRegistry) -> StageBuildContext {
    let store: Arc<dyn GraphStore> = Arc::new(MemoryStore::new(common::DIM));
    StageBuildContext::new(Arc::new(config), Arc::new(registry), store)
}

fn services() -> StageBuildContext {
    let config = common::default_config();
    services_with(config, ModelRegistry::empty("processing_v17"))
}

fn ed_block(config: &PipelineConfig) -> impeach_config::pipeline::EdStageConfig {
    config
        .stages
        .iter()
        .find_map(|s| match s {
            StageConfig::Ed(ed) => Some(ed.clone()),
            _ => None,
        })
        .expect("default config has an EDStage")
}

fn stub_stage(services: &StageBuildContext) -> (EdStage, Arc<Mutex<Vec<String>>>) {
    let stub = StubResolver::default();
    let inputs = Arc::clone(&stub.inputs);
    let stage = EdStage::new(&ed_block(&services.config), services)
        .expect("stage builds")
        .with_resolver(Box::new(stub));
    (stage, inputs)
}

fn mention(id: &str, entity_type: &str, name: &str) -> Mention {
    Mention::new(id, entity_type, name, Span::new(0, name.chars().count()))
}

/// A persisted context with `mentions` in one cluster `c1` whose canonical is the first.
fn clustered(services: &StageBuildContext, mentions: Vec<Mention>) -> PipelineContext {
    let mut ctx = PipelineContext::new(services);
    ctx.data.index_persisted = true;
    ctx.data.coref_clusters = Some(vec![CorefClusterContext {
        id: "c1".into(),
        member_mention_ids: mentions.iter().map(|m| m.id.clone()).collect(),
        pronoun_spans: vec![],
    }]);
    ctx.data.canonical_mention_for_cluster =
        Some([("c1".to_owned(), mentions[0].id.clone())].into_iter().collect());
    ctx.data.doc_mentions = Some(mentions);
    ctx
}

fn resolved(ctx: &PipelineContext) -> Vec<(String, String)> {
    ctx.data
        .resolved_mentions
        .as_ref()
        .expect("resolved_mentions")
        .iter()
        .map(|(m, e)| (m.to_string(), e.to_string()))
        .collect()
}

fn pairs(raw: &[(&str, &str)]) -> Vec<(String, String)> {
    raw.iter().map(|(a, b)| ((*a).to_owned(), (*b).to_owned())).collect()
}

#[test]
fn v17_org_loc_cluster_propagates() {
    let services = services();
    let (stage, _) = stub_stage(&services);
    let mut ctx = clustered(
        &services,
        vec![mention("m1", "ORG", "Riverside Arena"), mention("m2", "LOC", "the arena")],
    );
    stage.run(&mut ctx).expect("runs");
    assert_eq!(resolved(&ctx), pairs(&[("m1", "ent-m1"), ("m2", "ent-m1")]));
}

#[test]
fn coref_groups_override_yaml_groups() {
    let services = services();
    assert!(!impeach_ed::entity_types_compatible(
        &impeach_config::MappingsRegistry::load().expect("mappings").entity_types,
        Some("FACILITY"),
        Some("PRODUCT"),
    ));
    let (stage, _) = stub_stage(&services);
    let mut ctx = clustered(
        &services,
        vec![mention("m1", "FACILITY", "Plant 4"), mention("m2", "PRODUCT", "the plant")],
    );
    stage.run(&mut ctx).expect("runs");
    assert_eq!(resolved(&ctx), pairs(&[("m1", "ent-m1"), ("m2", "ent-m1")]));
}

#[test]
fn cross_group_cluster_is_not_propagated() {
    let services = services();
    let (stage, _) = stub_stage(&services);
    let mut ctx = clustered(
        &services,
        vec![mention("m1", "PER", "Maria Lopez"), mention("m2", "PRODUCT", "the Lopez truck")],
    );
    stage.run(&mut ctx).expect("runs");
    assert_eq!(resolved(&ctx), pairs(&[("m1", "ent-m1")]));
}

#[test]
fn falls_back_to_entity_types_yaml_without_coref_groups() {
    let mut config = common::default_config();
    for stage in &mut config.stages {
        if let StageConfig::Coreference(coref) = stage {
            coref.incompatible_type_groups.clear();
        }
    }
    let services = services_with(config, ModelRegistry::empty("processing_v17"));
    let (stage, _) = stub_stage(&services);
    assert!(stage.coref_type_groups().is_none());

    let mut ok =
        clustered(&services, vec![mention("m1", "ORG", "Arena"), mention("m2", "LOC", "arena")]);
    stage.run(&mut ok).expect("runs");
    assert_eq!(resolved(&ok), pairs(&[("m1", "ent-m1"), ("m2", "ent-m1")]));

    let mut blocked = clustered(
        &services,
        vec![mention("m1", "FACILITY", "Plant"), mention("m2", "PRODUCT", "plant")],
    );
    stage.run(&mut blocked).expect("runs");
    assert_eq!(resolved(&blocked), pairs(&[("m1", "ent-m1")]));
}

#[test]
fn input_set_and_synthetic_pronoun_propagation() {
    let services = services();
    let stub = StubResolver::default();
    let inputs = Arc::clone(&stub.inputs);
    let saw_all = Arc::clone(&stub.saw_all_mentions);
    let stage = EdStage::new(&ed_block(&services.config), &services)
        .expect("stage builds")
        .with_resolver(Box::new(stub));

    let mentions = vec![
        mention("m1", "PER", "Maria Lopez"),
        mention("m2", "PER", "Lopez"),
        mention("m3", "ORG", "Green Valley"),
        mention("m4", "DATE", "March 4, 2023"),
        mention("syn1", "PER", "she"),
        mention("syn2", "PER", "her"),
        mention("m5", "PER", "Daniel Reed"),
    ];
    let mut ctx = clustered(&services, mentions);
    let clusters = ctx.data.coref_clusters.as_mut().expect("clusters");
    clusters[0].member_mention_ids = vec!["m1".into(), "m2".into()];
    clusters.push(CorefClusterContext {
        id: "c2".into(),
        member_mention_ids: vec!["m5".into()],
        pronoun_spans: vec![],
    });
    ctx.data
        .canonical_mention_for_cluster
        .as_mut()
        .expect("canonical")
        .insert("c2".into(), "m5".into());
    ctx.data.synthetic_pronoun_mentions = Some(
        [("syn1".into(), "m1".into()), ("syn2".into(), "missing".into())].into_iter().collect(),
    );
    stage.run(&mut ctx).expect("runs");

    // Canonicals first, then unclustered non-synthetic mentions; DATE skips ED.
    assert_eq!(*inputs.lock().expect("lock"), ["m1", "m5", "m3"]);
    assert_eq!(*saw_all.lock().expect("lock"), Some(7));
    assert_eq!(ctx.data.all_mentions_by_id.as_ref().map(IndexMap::len), Some(7));
    assert_eq!(
        resolved(&ctx),
        pairs(&[
            ("m1", "ent-m1"),
            ("m5", "ent-m5"),
            ("m3", "ent-m3"),
            ("m2", "ent-m1"),
            ("syn1", "ent-m1"),
        ])
    );
    let results = ctx.data.ed_results.as_ref().expect("ed_results");
    assert_eq!(results.len(), 3);
}

#[test]
fn without_coref_every_non_value_mention_is_resolved() {
    let services = services();
    let (stage, inputs) = stub_stage(&services);
    let mut ctx = PipelineContext::new(&services);
    ctx.data.index_persisted = true;
    ctx.data.doc_mentions = Some(vec![
        mention("m1", "PER", "Maria Lopez"),
        mention("m2", "MONEY", "$5,000"),
        mention("m3", "ORG", "Green Valley"),
    ]);
    stage.run(&mut ctx).expect("runs");
    assert_eq!(*inputs.lock().expect("lock"), ["m1", "m3"]);
    assert_eq!(resolved(&ctx), pairs(&[("m1", "ent-m1"), ("m3", "ent-m3")]));
    assert!(ctx.data.all_mentions_by_id.is_none());
    assert!(ctx.data.role_map.is_none());
}

#[test]
fn nothing_to_do_without_mentions_or_inputs() {
    let services = services();
    let (stage, _) = stub_stage(&services);
    let mut ctx = PipelineContext::new(&services);
    ctx.data.index_persisted = true;
    stage.run(&mut ctx).expect("runs");
    ctx.data.doc_mentions = Some(vec![]);
    stage.run(&mut ctx).expect("runs");
    assert!(ctx.data.ed_results.is_none() && ctx.data.resolved_mentions.is_none());

    // Coreference path whose only input is a value type: no writes.
    let mut ctx = clustered(&services, vec![mention("m1", "DATE", "March 4, 2023")]);
    stage.run(&mut ctx).expect("runs");
    assert!(ctx.data.ed_results.is_none() && ctx.data.resolved_mentions.is_none());
}

#[test]
fn offline_run_resolves_nothing() {
    let services = services();
    let stage = EdStage::new(&ed_block(&services.config), &services).expect("stage builds");
    let mut ctx = PipelineContext::new(&services);
    ctx.data.doc_mentions = Some(vec![mention("m1", "PER", "Maria Lopez")]);
    stage.run(&mut ctx).expect("runs");
    assert_eq!(ctx.data.ed_results.as_deref(), Some(&[][..]));
    assert_eq!(ctx.data.resolved_mentions.as_ref().map(IndexMap::len), Some(0));
    assert!(services.store.get_all_entities(10).expect("entities").is_empty());
}

#[test]
fn role_map_binds_defined_roles() {
    let services = services();
    let (stage, _) = stub_stage(&services);
    let mut ctx = PipelineContext::new(&services);
    ctx.data.index_persisted = true;
    ctx.data.doc_mentions =
        Some(vec![mention("m1", "PER", "Maria Lopez"), mention("m2", "ORG", "Green Valley")]);
    let term = |mention_id: &str, role: &str, confidence: f64| DefinedTerm {
        entity_mention_id: mention_id.into(),
        defined_term: role.into(),
        original_term: role.into(),
        span: Span::new(0, 5),
        confidence,
        extraction_method: "pattern:parenthetical_quoted".into(),
    };
    ctx.data.defined_terms = Some(vec![
        term("m1", "PLAINTIFF", 0.95),
        term("m2", "DEFENDANT", 0.95),
        term("m9", "WITNESS", 0.95),
    ]);
    stage.run(&mut ctx).expect("runs");
    let role_map: Vec<(String, String)> = ctx
        .data
        .role_map
        .as_ref()
        .expect("role_map")
        .iter()
        .map(|(r, e)| (r.clone(), e.to_string()))
        .collect();
    assert_eq!(role_map, pairs(&[("PLAINTIFF", "ent-m1"), ("DEFENDANT", "ent-m2")]));

    // No binding: no role_map.
    let mut ctx = PipelineContext::new(&services);
    ctx.data.index_persisted = true;
    ctx.data.doc_mentions = Some(vec![mention("m1", "PER", "Maria Lopez")]);
    ctx.data.defined_terms = Some(vec![term("m9", "WITNESS", 0.95)]);
    stage.run(&mut ctx).expect("runs");
    assert!(ctx.data.role_map.is_none());
}

#[test]
fn span_links_skip_missing_spans_and_log_store_errors() {
    let memory = Arc::new(MemoryStore::new(common::DIM));
    let store: Arc<dyn GraphStore> = memory.clone();
    let config = common::default_config();
    let services =
        StageBuildContext::new(Arc::new(config), Arc::new(ModelRegistry::empty("x")), store);
    let (stage, _) = stub_stage(&services);
    let mut ctx = PipelineContext::new(&services);
    ctx.data.index_persisted = true;
    ctx.data.doc_mentions = Some(vec![mention("m1", "PER", "Maria Lopez")]);
    // The span and entity do not exist: the store error is logged, the stage succeeds.
    ctx.data.mention_to_span_id = Some([("m1".into(), SpanId::from("404"))].into_iter().collect());
    stage.run(&mut ctx).expect("runs");
    assert!(memory.edges().is_empty());
    assert_eq!(resolved(&ctx), pairs(&[("m1", "ent-m1")]));
}

const TEXT: &str = "Maria Lopez drove a Honda Civic. The Honda Civic was damaged.";

/// One indexed chunk with graph id `chunk_id` and two mentions in it.
fn indexed_context(services: &StageBuildContext, chunk_id: &str) -> PipelineContext {
    let mut ctx = PipelineContext::new(services);
    ctx.data.index_persisted = true;
    let chunk = Chunk::new("doc-llm", TEXT, 0, 0, TEXT.chars().count());
    ctx.data.rag_index_result = Some(IndexingResult {
        document: DocumentWriteResult {
            doc_id: "doc-llm".into(),
            helix_id: NodeId::new("1"),
            created: true,
        },
        chunks: vec![ChunkWriteResult { chunk, helix_id: NodeId::new(chunk_id), created: true }],
        spans: vec![],
        embeddings: Default::default(),
        doc_text: Some(TEXT.to_owned()),
        timings: IndexMap::new(),
    });
    let mut civic = Mention::new("m2", "PRODUCT", "Honda Civic", Span::new(20, 31));
    civic.chunk_id = Some(chunk_id.into());
    let mut maria = Mention::new("m1", "PER", "Maria Lopez", Span::new(0, 11));
    maria.chunk_id = Some(chunk_id.into());
    ctx.data.doc_mentions = Some(vec![maria, civic]);
    ctx
}

#[test]
fn llm_is_looked_up_under_the_pipeline_key_then_ollama() {
    for key in ["processing_v17_ollama", "ollama"] {
        let registry = ModelRegistry::empty("processing_v17");
        let memory = Arc::new(MemoryStore::new(common::DIM));
        let civic =
            memory.create_entity(&EntityNode::new("Honda Civic", "PRODUCT")).expect("entity").id;
        let answer = format!("{{\"chosen_entity_id\": \"{civic}\", \"confidence\": 0.9}}");
        let llm = Arc::new(MockLlm::with_responses([answer]));
        registry.insert_llm(key, llm.clone());
        let store: Arc<dyn GraphStore> = memory.clone();
        let services =
            StageBuildContext::new(Arc::new(common::default_config()), Arc::new(registry), store);
        let stage = EdStage::new(&ed_block(&services.config), &services).expect("stage builds");
        let mut ctx = indexed_context(&services, "chunk-1");
        stage.run(&mut ctx).expect("runs");
        let results = ctx.data.ed_results.as_ref().expect("ed_results");
        let civic_result =
            results.iter().find(|r| r.mention_id.as_ref().is_some_and(|m| m.as_str() == "m2"));
        let civic_result = civic_result.expect("civic resolved");
        assert_eq!(civic_result.strategy_used, "llm_first", "{key}");
        assert!(civic_result.resolved_by_llm);
        assert_eq!(civic_result.chosen_entity_id.as_ref(), Some(&civic));
        assert_eq!(llm.prompts().len(), 1, "{key}");
    }
}

#[test]
fn llm_errors_leave_mentions_to_entity_creation() {
    let registry = ModelRegistry::empty("processing_v17");
    registry.insert_llm("processing_v17_ollama", Arc::new(MockLlm::new()));
    let memory = Arc::new(MemoryStore::new(common::DIM));
    let civic: EntityId =
        memory.create_entity(&EntityNode::new("Honda Civic", "PRODUCT")).expect("entity").id;
    let store: Arc<dyn GraphStore> = memory.clone();
    let services =
        StageBuildContext::new(Arc::new(common::default_config()), Arc::new(registry), store);
    let stage = EdStage::new(&ed_block(&services.config), &services).expect("stage builds");
    let mut ctx = indexed_context(&services, "chunk-1");
    stage.run(&mut ctx).expect("runs despite the LLM error");
    let results = ctx.data.ed_results.as_ref().expect("ed_results");
    let strategies: Vec<&str> = results.iter().map(|r| r.strategy_used.as_str()).collect();
    assert_eq!(strategies, ["hybrid_unresolved_create", "llm_skipped_reuse"]);
    assert_eq!(results[1].chosen_entity_id.as_ref(), Some(&civic));
}

#[test]
fn missing_prompt_file_fails_the_build() {
    let services = services();
    let mut block = ed_block(&services.config);
    block.prompt_file = "impeach/prompts/ed/missing.json".into();
    let error = EdStage::new(&block, &services).expect_err("missing prompt");
    assert!(matches!(error, PipelineError::Config(_)), "{error}");
}

#[test]
fn construct_rejects_other_stage_types() {
    let services = services();
    let other = services
        .config
        .stages
        .iter()
        .find(|s| s.type_name() == "MentionStage")
        .expect("mention stage")
        .clone();
    let error = EdStage::construct(&other, &services).err().expect("wrong type");
    assert!(matches!(error, PipelineError::InvalidStageConfig { .. }), "{error}");
}
