//! MentionStage over RAGIndexStage output with `MemoryStore`, `MockEmbedder` and a scripted
//! NER model (port of `tests/unit/test_mention_split_healing.py::
//! test_mention_stage_run_applies_split_healing_before_merging` plus stage integration).
//!
//! The snapshot test compares the mentions of `fixtures/input/green_valley_incident_report.jsonl`
//! with `fixtures/reference/mentions/green_valley_incident_report.json` (a JSON array, one
//! mention per line, each written with `ce_core::pyjson::to_python_json`). The snapshot is
//! self-generated; regenerate it with `UPDATE_SNAPSHOTS=1 cargo test -p ce-pipeline --test
//! mention_stage`.

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use ce_config::pipeline::{MentionDetector, MentionStageConfig};
use ce_config::{PipelineConfig, StageConfig};
use ce_core::pyjson::to_python_json;
use ce_core::{
    Chunk, ChunkWriteResult, DocumentWriteResult, EmbeddingWriteResult, IndexingResult, Mention,
};
use ce_models::{ModelError, ModelKind, NerModel, NerSpan, RelationHit};
use ce_pipeline::{MentionStage, Pipeline, PipelineContext, PipelineError, PipelineInput, Stage};
use ce_store::graph::SPAN;
use common::{char_slice, default_config, fixture, Harness};
use indexmap::IndexMap;
use pretty_assertions::assert_eq;
use serde_json::{json, Value};

/// Finds every whole-word occurrence of fixed surface strings (no word character on either
/// side), like Python `GLiNER2Service._find_word_occurrences`, and reports them with fixed
/// labels and scores. Spans come back grouped by entry order, offsets in characters.
#[derive(Debug)]
struct DictionaryNer {
    entries: Vec<(&'static str, &'static str, f32)>,
}

impl DictionaryNer {
    fn new(entries: Vec<(&'static str, &'static str, f32)>) -> Self {
        Self { entries }
    }
}

fn is_word(c: Option<char>) -> bool {
    c.is_some_and(|c| c.is_alphanumeric() || c == '_')
}

impl NerModel for DictionaryNer {
    fn extract_entities(
        &self,
        text: &str,
        labels: &[&str],
        threshold: f32,
    ) -> Result<Vec<NerSpan>, ModelError> {
        let mut spans = Vec::new();
        for (surface, label, score) in &self.entries {
            if !labels.contains(label) || *score < threshold {
                continue;
            }
            for (byte, _) in text.match_indices(surface) {
                let end_byte = byte + surface.len();
                if is_word(text[..byte].chars().next_back())
                    || is_word(text[end_byte..].chars().next())
                {
                    continue;
                }
                let start = text[..byte].chars().count();
                spans.push(NerSpan {
                    start,
                    end: start + surface.chars().count(),
                    label: (*label).to_owned(),
                    score: *score,
                    text: (*surface).to_owned(),
                });
            }
        }
        Ok(spans)
    }

    fn extract_relations(
        &self,
        _text: &str,
        _labels: &[&str],
        _threshold: f32,
    ) -> Result<Vec<RelationHit>, ModelError> {
        Ok(Vec::new())
    }
}

/// Entities of the green valley report (scores are exact in f32 and f64).
fn green_valley_ner() -> DictionaryNer {
    DictionaryNer::new(vec![
        ("Maria Lopez", "person", 0.96875),
        ("Lopez", "person", 0.875),
        ("Daniel Reed", "person", 0.9375),
        ("Reed", "person", 0.8125),
        ("Green Valley Logistics, Inc.", "company", 0.90625),
        ("Green Valley", "organization", 0.75),
        ("San Francisco Police Department", "organization", 0.9375),
        ("Mission Street", "location", 0.875),
        ("14th Avenue", "location", 0.8125),
        ("San Francisco", "location", 0.90625),
        ("Honda Civic", "vehicle", 0.6875),
        ("March 4, 2023", "date", 0.8125),
        ("9:20 p.m.", "time", 0.5),
        ("$14,250.00", "money", 0.875),
        ("California", "location", 0.375),
    ])
}

fn mention_config(config: &PipelineConfig) -> MentionStageConfig {
    config
        .stages
        .iter()
        .find_map(|stage| match stage {
            StageConfig::Mention(mention) => Some(mention.clone()),
            _ => None,
        })
        .expect("config has a MentionStage")
}

/// The default config with the MentionStage block edited.
fn config_with_mention(mut edit: impl FnMut(&mut MentionStageConfig)) -> PipelineConfig {
    let mut config = default_config();
    for stage in &mut config.stages {
        if let StageConfig::Mention(mention) = stage {
            edit(mention);
        }
    }
    config
}

fn rules_only() -> PipelineConfig {
    config_with_mention(|m| m.detectors = vec![MentionDetector::Eyecite, MentionDetector::Regex])
}

fn mention_stage(harness: &Harness) -> MentionStage {
    MentionStage::new(&mention_config(&harness.services.config), &harness.services)
        .expect("stage builds")
}

/// Runs RAGIndexStage then MentionStage on a fixture.
fn run_fixture(harness: &Harness, name: &str) -> PipelineContext {
    let stages: Vec<Box<dyn Stage>> =
        vec![Box::new(harness.stage()), Box::new(mention_stage(harness))];
    let pipeline = Pipeline::new(stages, harness.services.clone());
    let mut ctx = pipeline.new_context();
    pipeline.run(&mut ctx, PipelineInput::Jsonl(fixture(name))).expect("pipeline runs");
    ctx
}

fn inject_ner(harness: &Harness, ner: DictionaryNer) {
    let registry = &harness.services.registry;
    registry.insert_ner(registry.key(ModelKind::Gliner2), Arc::new(ner));
}

#[test]
fn builds_from_the_default_config_without_loading_a_model() {
    let harness = Harness::new(default_config());
    let stage = mention_stage(&harness);
    assert_eq!(stage.name(), "mention_detection");
    assert!(!stage.skips_span_writes());
}

#[test]
fn unknown_label_set_fails_to_build() {
    let config = config_with_mention(|m| {
        for detector in &mut m.detectors {
            if let MentionDetector::Gliner2(gliner) = detector {
                gliner.label_set = "no_such_set".into();
            }
        }
    });
    let harness = Harness::new(config);
    let error = MentionStage::new(&mention_config(&harness.services.config), &harness.services)
        .expect_err("unknown label set");
    assert!(matches!(error, PipelineError::Config(_)), "{error}");
    assert!(error.to_string().contains("no_such_set"), "{error}");
}

#[test]
fn missing_ner_model_fails_the_stage() {
    let config = config_with_mention(|m| {
        for detector in &mut m.detectors {
            if let MentionDetector::Gliner2(gliner) = detector {
                gliner.model_name_suffix = "not_registered".into();
            }
        }
    });
    let harness = Harness::new(config);
    let stages: Vec<Box<dyn Stage>> =
        vec![Box::new(harness.stage()), Box::new(mention_stage(&harness))];
    let pipeline = Pipeline::new(stages, harness.services.clone());
    let mut ctx = pipeline.new_context();
    let error = pipeline
        .run(&mut ctx, PipelineInput::Jsonl(fixture("Mason_billy.jsonl")))
        .expect_err("unregistered model");
    let text = error.to_string();
    assert!(text.contains("processing_v17_not_registered"), "{text}");
}

#[test]
fn rule_detectors_write_mentions_and_spans() {
    let harness = Harness::new(rules_only());
    let ctx = run_fixture(&harness, "green_valley_incident_report.jsonl");
    let data = &ctx.data;
    let result = data.rag_index_result.as_ref().expect("rag result");
    let text = data.doc_text.as_deref().expect("doc text");
    let mentions = data.doc_mentions.as_ref().expect("doc_mentions");
    let by_chunk = data.mentions_by_chunk.as_ref().expect("mentions_by_chunk");

    // Three amounts and nine dates, as the Python regex detector finds them.
    let mut found: Vec<(&str, &str)> =
        mentions.iter().map(|m| (m.entity_type.as_str(), m.name.as_str())).collect();
    found.sort_unstable();
    found.dedup();
    assert_eq!(
        found,
        [
            ("DATE", "February 2, 2020"),
            ("DATE", "June 10, 2018"),
            ("DATE", "March 20, 2023"),
            ("DATE", "March 21, 2023"),
            ("DATE", "March 22, 2023"),
            ("DATE", "March 4, 2023"),
            ("DATE", "September 1, 2023"),
            ("MONEY", "$14,250.00"),
            ("MONEY", "$7,400.00"),
            ("MONEY", "$8,500.00"),
        ]
    );
    assert_eq!(by_chunk.values().map(Vec::len).sum::<usize>(), mentions.len());
    for (chunk_id, chunk_mentions) in by_chunk {
        let write = result
            .chunks
            .iter()
            .find(|w| w.helix_id.as_str() == chunk_id.as_str())
            .expect("chunk of the index result");
        for m in chunk_mentions {
            assert_eq!(m.chunk_id.as_ref(), Some(chunk_id));
            assert_eq!(m.doc_id.as_ref().map(ce_core::DocId::as_str), Some("001"));
            assert_eq!(char_slice(text, m.span.start, m.span.end), m.name);
            assert!(write.chunk.start_char <= m.span.start && m.span.end <= write.chunk.end_char);
            assert_eq!(m.sentence_index, Some(0), "sentence chunks have one sentence span");
            assert_eq!(m.source.as_deref(), Some("regex_detector"));
            assert_eq!(m.sources.len(), 1);
            assert!(m.id.as_str().starts_with("span-"));
        }
    }

    let span_ids = data.mention_to_span_id.as_ref().expect("mention_to_span_id");
    assert_eq!(span_ids.len(), mentions.len());
    assert_eq!(harness.count_label(SPAN), result.spans.len() + mentions.len());
    let first = &mentions[0];
    let (label, node) =
        harness.store.node_snapshot(span_ids[&first.id].as_str()).expect("mention span");
    assert_eq!(label, SPAN);
    assert_eq!(node["SpanType"], json!("MENTION"));
    assert_eq!(node["SpanLabel"], json!(first.entity_type));
    assert_eq!(node["SurfaceText"], json!(first.name));
    assert_eq!(node["StartChar"], json!(first.span.start));
    let metadata: Value =
        serde_json::from_str(node["Metadata"].as_str().expect("metadata string")).expect("json");
    assert_eq!(
        metadata,
        json!({
            "ner_source": "regex_detector",
            "confidence": 1.0,
            "entity_type": first.entity_type,
            "mention_id": first.id.as_str(),
            "sentence_index": 0
        })
    );
}

#[test]
fn offline_rag_index_skips_span_writes() {
    let mut config = rules_only();
    for stage in &mut config.stages {
        if let StageConfig::RagIndex(rag) = stage {
            rag.offline = true;
        }
    }
    let harness = Harness::new(config);
    assert!(mention_stage(&harness).skips_span_writes());
    let ctx = run_fixture(&harness, "Mason_intra.jsonl");
    let mentions = ctx.data.doc_mentions.as_ref().expect("doc_mentions");
    assert_eq!(mentions.len(), 4);
    assert!(ctx.data.mention_to_span_id.is_none());
    assert_eq!(harness.count_label(SPAN), 0);
    let chunk_ids: Vec<&str> = ctx
        .data
        .mentions_by_chunk
        .as_ref()
        .expect("by chunk")
        .keys()
        .map(ce_core::ChunkId::as_str)
        .collect();
    assert!(chunk_ids.iter().all(|id| id.starts_with("003:chunk-")), "{chunk_ids:?}");
}

#[test]
fn without_rag_index_result_the_stage_writes_nothing() {
    let harness = Harness::new(rules_only());
    let mut ctx = PipelineContext::new(&harness.services);
    mention_stage(&harness).run(&mut ctx).expect("runs");
    assert!(ctx.data.mentions_by_chunk.is_none());
    assert!(ctx.data.doc_mentions.is_none());
}

// Port of test_mention_stage_run_applies_split_healing_before_merging.
#[test]
fn split_healing_runs_before_merging() {
    let harness = Harness::new(config_with_mention(|m| {
        m.detectors.retain(|d| matches!(d, MentionDetector::Gliner2(_)));
    }));
    inject_ner(
        &harness,
        DictionaryNer::new(vec![("Plainti", "person", 0.95), ("ff", "person", 0.95)]),
    );
    let text = "Plainti ff sued the Defendant.";
    let chunk = Chunk::new("doc-1", text, 0, 0, text.chars().count());
    let result = IndexingResult {
        document: DocumentWriteResult {
            doc_id: "doc-1".into(),
            helix_id: "doc-1".into(),
            created: true,
        },
        chunks: vec![ChunkWriteResult { chunk, helix_id: "chunk-1".into(), created: true }],
        spans: Vec::new(),
        embeddings: EmbeddingWriteResult::default(),
        doc_text: None,
        timings: IndexMap::new(),
    };
    let mut ctx = PipelineContext::new(&harness.services);
    ctx.data.rag_index_result = Some(result);
    mention_stage(&harness).run(&mut ctx).expect("runs");

    let by_chunk = ctx.data.mentions_by_chunk.as_ref().expect("mentions_by_chunk");
    let mentions = by_chunk.get("chunk-1").expect("chunk-1");
    assert_eq!(mentions.len(), 1);
    let m = &mentions[0];
    assert_eq!((m.span.start, m.span.end), (0, 10));
    assert_eq!(m.name, "Plainti ff");
    assert_eq!(m.doc_id.as_ref().map(ce_core::DocId::as_str), Some("doc-1"));
    assert_eq!(m.chunk_id.as_ref().map(ce_core::ChunkId::as_str), Some("chunk-1"));
    assert_eq!(m.id.as_str(), "ent_doc-1_0_0");
    assert_eq!(m.entity_type, "PER");
    // No helix_doc_id in the context: no span writes.
    assert!(ctx.data.mention_to_span_id.is_none());
}

#[test]
fn gliner_mentions_merge_with_rule_mentions() {
    let harness = Harness::new(default_config());
    inject_ner(&harness, green_valley_ner());
    let ctx = run_fixture(&harness, "green_valley_incident_report.jsonl");
    let mentions = ctx.data.doc_mentions.as_ref().expect("doc_mentions");
    let money = mentions.iter().find(|m| m.name == "$14,250.00").expect("money mention");
    assert_eq!(money.source.as_deref(), Some("regex_detector"));
    let sources: Vec<(&str, &str)> =
        money.sources.iter().map(|s| (s.name.as_str(), s.label.as_str())).collect();
    assert_eq!(sources, [("regex_detector", "MONEY"), ("gliner2", "MONEY")]);
    // Below the model threshold (0.4): never reported.
    assert!(mentions.iter().all(|m| m.name != "California"));
    // "Green Valley Logistics, Inc." outranks the overlapping "Green Valley" by length.
    assert!(mentions
        .iter()
        .any(|m| m.name == "Green Valley Logistics, Inc." && m.entity_type == "ORG"));
    let ids = ctx.data.mention_to_span_id.as_ref().expect("span ids");
    assert_eq!(ids.len(), mentions.len());
}

fn snapshot_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/reference/mentions/green_valley_incident_report.json")
}

/// A JSON array with one `to_python_json` mention per line.
fn snapshot_text(mentions: &[Mention]) -> String {
    let lines: Vec<String> =
        mentions.iter().map(|m| to_python_json(m).expect("mention serialises")).collect();
    format!("[\n{}\n]\n", lines.join(",\n"))
}

#[test]
fn green_valley_mentions_match_the_snapshot() {
    let harness = Harness::new(default_config());
    inject_ner(&harness, green_valley_ner());
    let ctx = run_fixture(&harness, "green_valley_incident_report.jsonl");
    let mentions = ctx.data.doc_mentions.as_ref().expect("doc_mentions");
    let actual = snapshot_text(mentions);
    let path = snapshot_path();
    if std::env::var_os("UPDATE_SNAPSHOTS").is_some_and(|v| v == "1") {
        std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
        std::fs::write(&path, &actual).expect("write snapshot");
        return;
    }
    let expected = std::fs::read_to_string(&path)
        .expect("snapshot exists (regenerate with UPDATE_SNAPSHOTS=1)")
        .replace("\r\n", "\n");
    assert_eq!(actual, expected);
    let parsed: Vec<Mention> = serde_json::from_str(&expected).expect("snapshot parses");
    assert_eq!(&parsed, mentions);
}

#[cfg(feature = "gliner2")]
#[test]
#[ignore = "needs fastino/gliner2-base-v1 in the Hugging Face cache"]
fn real_gliner2_on_one_green_valley_chunk() {
    let config = config_with_mention(|m| {
        m.detectors.retain(|d| matches!(d, MentionDetector::Gliner2(_)));
    });
    let harness = Harness::new(config);
    let (mut result, data) =
        harness.index(PipelineInput::Jsonl(fixture("green_valley_incident_report.jsonl")));
    result.chunks.truncate(1);
    let text = data.doc_text.clone().expect("doc text");
    let mut ctx = PipelineContext::new(&harness.services);
    ctx.data.rag_index_result = Some(result);
    ctx.data.helix_doc_id = data.helix_doc_id.clone();
    let stage = mention_stage(&harness);
    let started = std::time::Instant::now();
    stage.run(&mut ctx).expect("stage runs with the real model");
    let elapsed = started.elapsed();
    let mentions = ctx.data.doc_mentions.as_ref().expect("doc_mentions");
    for m in mentions {
        eprintln!(
            "[{}:{}] {} ({}) score={:?} id={}",
            m.span.start, m.span.end, m.name, m.entity_type, m.score, m.id
        );
        // Port of test_gliner2_spans.py: names are the text under the spans.
        assert_eq!(common::char_slice(&text, m.span.start, m.span.end), m.name);
        assert_eq!(m.source.as_deref(), Some("gliner2"));
    }
    eprintln!("{} mentions in {elapsed:?}", mentions.len());
    assert!(mentions.iter().any(|m| m.entity_type == "PER"), "{mentions:?}");
}
