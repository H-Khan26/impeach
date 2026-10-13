//! CoreferenceStage over RAGIndexStage output and hand-built mentions.

mod common;

use std::sync::Arc;

use common::{default_config, Harness};
use impeach_config::pipeline::{CorefBackend, CoreferenceStageConfig};
use impeach_config::{PipelineConfig, StageConfig};
use impeach_core::{ChunkId, DocumentPayload, Mention, Span};
use impeach_models::{CorefCluster, MockCoref, ModelKind};
use impeach_pipeline::{
    ContextData, CoreferenceStage, Pipeline, PipelineContext, PipelineInput, Stage,
};
use indexmap::IndexMap;
use pretty_assertions::assert_eq;

const TEXT: &str = "Maria Lopez was driving a Honda Civic. She stopped at the light. \
                    Green Valley Logistics owned the truck. Lopez said its driver ran the light.";

/// `(name, type, char offset to search from)` of the mentions the injector creates.
const MENTIONS: [(&str, &str, usize); 4] = [
    ("Maria Lopez", "PER", 0),
    ("Honda Civic", "PRODUCT", 0),
    ("Green Valley Logistics", "ORG", 0),
    ("Lopez", "PER", 20),
];

/// Writes `MENTIONS` into `doc_mentions` and `mentions_by_chunk` (as MentionStage would).
#[derive(Debug)]
struct InjectMentions;

impl Stage for InjectMentions {
    fn name(&self) -> &str {
        "inject_mentions"
    }

    fn run(&self, ctx: &mut PipelineContext) -> impeach_pipeline::Result<()> {
        let text = ctx.data.doc_text.clone().expect("doc_text");
        let chunks = ctx.data.rag_index_result.as_ref().expect("rag result").chunks.clone();
        let mut mentions = Vec::new();
        let mut by_chunk: IndexMap<ChunkId, Vec<Mention>> = IndexMap::new();
        for (i, (name, entity_type, from)) in MENTIONS.iter().enumerate() {
            let skip: usize = text.chars().take(*from).map(char::len_utf8).sum();
            let byte = skip + text[skip..].find(name).expect("mention text occurs");
            let start = text[..byte].chars().count();
            let mut mention = Mention::new(
                format!("m{i}"),
                *entity_type,
                *name,
                Span::new(start, start + name.len()),
            );
            mention.score = Some(0.9);
            let chunk = chunks
                .iter()
                .find(|c| c.chunk.start_char <= start && start < c.chunk.end_char)
                .expect("mention inside a chunk");
            mention.chunk_id = Some(ChunkId::new(chunk.helix_id.as_str()));
            by_chunk
                .entry(ChunkId::new(chunk.helix_id.as_str()))
                .or_default()
                .push(mention.clone());
            mentions.push(mention);
        }
        ctx.data.doc_mentions = Some(mentions);
        ctx.data.mentions_by_chunk = Some(by_chunk);
        Ok(())
    }
}

/// The default config with `models.coref.backend` and an edited CoreferenceStage block.
fn config(backend: CorefBackend, edit: impl FnOnce(&mut CoreferenceStageConfig)) -> PipelineConfig {
    let mut config = default_config();
    config.models.coref.backend = backend;
    let block = config
        .stages
        .iter_mut()
        .find_map(|s| match s {
            StageConfig::Coreference(c) => Some(c),
            _ => None,
        })
        .expect("default config has a CoreferenceStage");
    edit(block);
    config
}

fn coref_config(harness: &Harness) -> CoreferenceStageConfig {
    harness
        .services
        .config
        .stages
        .iter()
        .find_map(|s| match s {
            StageConfig::Coreference(c) => Some(c.clone()),
            _ => None,
        })
        .expect("coreference block")
}

/// RAG index -> (optional) mention injection -> coreference.
fn pipeline(harness: &Harness, inject: bool) -> Pipeline {
    let coref =
        CoreferenceStage::new(&coref_config(harness), &harness.services).expect("stage builds");
    let mut stages: Vec<Box<dyn Stage>> = vec![Box::new(harness.stage())];
    if inject {
        stages.push(Box::new(InjectMentions));
    }
    stages.push(Box::new(coref));
    Pipeline::new(stages, harness.services.clone())
}

fn run(harness: &Harness, inject: bool) -> impeach_pipeline::Result<ContextData> {
    let pipeline = pipeline(harness, inject);
    let mut ctx = pipeline.new_context();
    let input = PipelineInput::Payload(Box::new(DocumentPayload::new("doc-coref", TEXT)));
    pipeline.run(&mut ctx, input)?;
    Ok(ctx.data)
}

fn char_slice(start: usize, end: usize) -> String {
    TEXT.chars().skip(start).take(end - start).collect()
}

#[test]
fn rule_backend_clusters_names_and_synthesises_pronouns() {
    let harness = Harness::new(config(CorefBackend::Rule, |_| {}));
    let data = run(&harness, true).expect("pipeline runs");

    let clusters = data.coref_clusters.as_ref().expect("coref_clusters");
    assert_eq!(clusters.len(), 1, "{clusters:?}");
    let cluster = &clusters[0];
    assert_eq!(cluster.id, "doc-coref_cluster_0");
    let members: Vec<&str> =
        cluster.member_mention_ids.iter().map(impeach_core::MentionId::as_str).collect();
    assert_eq!(members, vec!["m0", "m3"]);
    let pronouns: Vec<String> =
        cluster.pronoun_spans.iter().map(|s| char_slice(s.start, s.end)).collect();
    assert_eq!(pronouns, vec!["She"]);

    let canonical = data.canonical_mention_for_cluster.as_ref().expect("canonical map");
    assert_eq!(canonical["doc-coref_cluster_0"].as_str(), "m0");

    let doc_mentions = data.doc_mentions.as_ref().expect("doc_mentions");
    assert_eq!(doc_mentions.len(), 5);
    let synthetic = doc_mentions.last().expect("synthetic mention");
    let she = cluster.pronoun_spans[0];
    assert_eq!(
        synthetic.id.as_str(),
        format!("synth_pronoun_doc-coref_cluster_0_{}_{}", she.start, she.end)
    );
    assert_eq!(synthetic.name, "Maria Lopez");
    assert_eq!(synthetic.entity_type, "PER");
    assert_eq!(synthetic.score, Some(0.85));
    assert_eq!(synthetic.source.as_deref(), Some("coref_pronoun_synthesis"));
    assert_eq!(synthetic.metadata["original_pronoun_text"], "She");

    let synthetic_map = data.synthetic_pronoun_mentions.as_ref().expect("synthetic map");
    assert_eq!(synthetic_map[&synthetic.id].as_str(), "m0");
    let canonical_text = data.canonical_text_for_mention.as_ref().expect("canonical text");
    for id in ["m0", "m3", synthetic.id.as_str()] {
        assert_eq!(canonical_text[id], "Maria Lopez", "{id}");
    }

    // The synthetic mention is appended to its chunk's list.
    let by_chunk = data.mentions_by_chunk.as_ref().expect("mentions_by_chunk");
    let holding: Vec<&ChunkId> = by_chunk
        .iter()
        .filter(|(_, ms)| ms.iter().any(|m| m.id == synthetic.id))
        .map(|(k, _)| k)
        .collect();
    assert_eq!(holding.len(), 1);
    assert_eq!(by_chunk.values().map(Vec::len).sum::<usize>(), 5);
}

#[test]
fn fastcoref_without_command_falls_back_to_rules() {
    let rule = run(&Harness::new(config(CorefBackend::Rule, |_| {})), true).expect("rule runs");
    let fallback = run(&Harness::new(config(CorefBackend::Fastcoref, |_| {})), true)
        .expect("the shipped fastcoref default runs without a sidecar");
    assert!(fallback.coref_clusters.is_some());
    assert_eq!(fallback.coref_clusters, rule.coref_clusters);
    assert_eq!(fallback.synthetic_pronoun_mentions, rule.synthetic_pronoun_mentions);
}

/// The default config with a fastcoref sidecar command that cannot start.
fn broken_sidecar(edit: impl FnOnce(&mut CoreferenceStageConfig)) -> PipelineConfig {
    let mut config = config(CorefBackend::Fastcoref, edit);
    config.models.coref.command = Some(vec!["impeach-pipeline-no-such-coref-sidecar".to_owned()]);
    config
}

#[test]
fn a_sidecar_that_cannot_start_fails_loudly() {
    let harness = Harness::new(broken_sidecar(|_| {}));
    let err = run(&harness, true).expect_err("coreference must not degrade silently");
    let message = err.to_string();
    assert!(message.contains("impeach-pipeline-no-such-coref-sidecar"), "{message}");
    assert!(message.contains("coreference"), "{message}");
}

#[test]
fn skips_before_resolving_the_model() {
    // No mentions: the broken sidecar is never started.
    let harness = Harness::new(broken_sidecar(|_| {}));
    let data = run(&harness, false).expect("no mentions: skipped");
    assert!(data.coref_clusters.is_none());

    // Too long a document.
    let harness = Harness::new(broken_sidecar(|c| c.max_doc_chars = 20));
    let data = run(&harness, true).expect("long document: skipped");
    assert!(data.coref_clusters.is_none());
    assert_eq!(data.doc_mentions.as_ref().map(Vec::len), Some(4));
}

#[test]
fn unregistered_model_key_skips_with_a_warning() {
    let harness = Harness::new(config(CorefBackend::Fastcoref, |c| {
        c.model_name_suffix = "other_coref".to_owned();
    }));
    let data = run(&harness, true).expect("unknown key: skipped");
    assert!(data.coref_clusters.is_none());
}

#[test]
fn mock_backend_writes_nothing() {
    let harness = Harness::new(config(CorefBackend::Mock, |_| {}));
    let data = run(&harness, true).expect("pipeline runs");
    assert!(data.coref_clusters.is_none());
    assert!(data.canonical_mention_for_cluster.is_none());
    assert!(data.synthetic_pronoun_mentions.is_none());
    assert_eq!(data.doc_mentions.as_ref().map(Vec::len), Some(4));
}

#[test]
fn injected_resolver_and_disabled_synthesis() {
    let harness = Harness::new(config(CorefBackend::Fastcoref, |c| {
        c.synthesize_pronoun_mentions = false;
    }));
    // Maria Lopez, the vehicle and "She": the vehicle is not an eligible type.
    let she = TEXT.find("She").expect("She");
    let lopez = TEXT.find("Lopez said").expect("Lopez");
    let resolver = MockCoref::new(vec![CorefCluster {
        spans: vec![(0, 11), (26, 37), (she, she + 3), (lopez, lopez + 5)],
    }]);
    harness
        .services
        .registry
        .insert_coref(harness.services.registry.key(ModelKind::Coref), Arc::new(resolver));
    let data = run(&harness, true).expect("pipeline runs");
    let clusters = data.coref_clusters.as_ref().expect("clusters");
    let members: Vec<&str> =
        clusters[0].member_mention_ids.iter().map(impeach_core::MentionId::as_str).collect();
    assert_eq!(members, vec!["m0", "m3"]);
    assert_eq!(clusters[0].pronoun_spans.len(), 1);
    assert!(data.synthetic_pronoun_mentions.is_none());
    assert_eq!(data.doc_mentions.as_ref().map(Vec::len), Some(4));
}
