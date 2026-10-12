//! RelationStage over RAGIndexStage output with mock relation models.

mod common;
mod ed_common;

use std::sync::Arc;

use ce_config::pipeline::RelationStageConfig;
use ce_config::StageConfig;
use ce_core::processing::ExtractedRelation;
use ce_core::DocumentPayload;
use ce_models::{MockNer, ModelError, ModelKind, NerModel, NerSpan, RelationHit};
use ce_pipeline::{
    ContextData, Pipeline, PipelineContext, PipelineError, PipelineInput, RelationStage, Stage,
};
use common::{default_config, Harness};
use ed_common::GazetteerNer;
use pretty_assertions::assert_eq;

const TEXT: &str =
    "Daniel Reed is employed by Green Valley. Green Valley is located in San Francisco.";

fn relation_config(harness: &Harness) -> RelationStageConfig {
    harness
        .services
        .config
        .stages
        .iter()
        .find_map(|s| match s {
            StageConfig::Relation(r) => Some(r.clone()),
            _ => None,
        })
        .expect("default config has a RelationStage")
}

fn run(harness: &Harness) -> ContextData {
    let stage = RelationStage::new(&relation_config(harness), &harness.services).expect("builds");
    let stages: Vec<Box<dyn Stage>> = vec![Box::new(harness.stage()), Box::new(stage)];
    let pipeline = Pipeline::new(stages, harness.services.clone());
    let mut ctx = pipeline.new_context();
    let input = PipelineInput::Payload(Box::new(DocumentPayload::new("doc-rel", TEXT)));
    pipeline.run(&mut ctx, input).expect("pipeline runs");
    ctx.data
}

fn span(text: &str, score: f32) -> NerSpan {
    NerSpan { start: 0, end: text.chars().count(), label: "x".into(), score, text: text.into() }
}

fn hit(head: &str, label: &str, tail: &str, score: f32) -> RelationHit {
    RelationHit { head: span(head, score), tail: span(tail, score), label: label.into(), score }
}

#[test]
fn builds_from_the_legal_relations_label_set() {
    let harness = Harness::new(default_config());
    let stage = RelationStage::new(&relation_config(&harness), &harness.services).expect("builds");
    assert_eq!(stage.model_key(), "processing_v17_gliner2");
    assert_eq!(stage.labels().len(), 10);
    assert_eq!(stage.labels()[0], "employed_by");
    assert!((stage.threshold() - 0.4).abs() < 1e-6);

    let mut config = relation_config(&harness);
    config.relation_label_set = "no_such_set".into();
    let error = RelationStage::new(&config, &harness.services).expect_err("unknown label set");
    assert!(matches!(error, PipelineError::Config(_)), "{error}");
}

#[test]
fn extracts_relations_per_chunk_with_canonical_types() {
    let harness = Harness::new(default_config());
    let registry = &harness.services.registry;
    let ner = GazetteerNer::new(&[])
        .with_relation("Daniel Reed", "employed_by", "Green Valley", 0.8)
        .with_relation("Green Valley", "located_in", "San Francisco", 0.6)
        .with_relation("Daniel Reed", "owns", "San Francisco", 0.3)
        .with_relation("Daniel Reed", "married_to", "Green Valley", 0.9);
    registry.insert_ner(registry.key(ModelKind::Gliner2), Arc::new(ner));
    let data = run(&harness);

    let indexed = data.rag_index_result.as_ref().expect("indexed");
    let chunk_id = indexed.chunks[0].helix_id.as_str();
    let relations = data.extracted_relations.expect("extracted_relations");
    let expected: Vec<ExtractedRelation> = [
        ("employed_by", "Daniel Reed", "Green Valley", "EMPLOYED_BY", 0.8_f32),
        ("located_in", "Green Valley", "San Francisco", "LOCATED_IN", 0.6),
    ]
    .iter()
    .map(|(relation, head, tail, canonical, score)| ExtractedRelation {
        relation: (*relation).into(),
        head: (*head).into(),
        tail: (*tail).into(),
        chunk_id: chunk_id.into(),
        canonical_relation: (*canonical).into(),
        score: Some(f64::from(*score)),
    })
    .collect();
    assert_eq!(relations, expected);
    let by_type: Vec<(String, usize)> =
        data.relations_by_type.expect("relations_by_type").into_iter().collect();
    assert_eq!(by_type, [("employed_by".to_owned(), 1), ("located_in".to_owned(), 1)]);
    // Nothing is written to the store beyond the indexed document.
    assert_eq!(harness.count_edges("SpanRefersToEntity"), 0);
}

#[test]
fn relations_without_head_or_tail_text_are_dropped() {
    let harness = Harness::new(default_config());
    let registry = &harness.services.registry;
    let ner = MockNer::new(
        vec![],
        vec![
            hit("", "works_for", "Green Valley", 0.9),
            hit("Daniel Reed", "works_for", "", 0.9),
            hit("Daniel Reed", "works_for", "Green Valley", 0.9),
            hit("Daniel Reed", "works_for", "Green Valley", 0.9),
        ],
    );
    registry.insert_ner(registry.key(ModelKind::Gliner2), Arc::new(ner));
    let data = run(&harness);
    let relations = data.extracted_relations.expect("extracted_relations");
    assert_eq!(relations.len(), 2);
    assert!(relations.iter().all(|r| r.canonical_relation == "EMPLOYED_BY"));
    assert_eq!(data.relations_by_type.expect("counts")["works_for"], 2);
}

#[test]
fn unregistered_model_skips_without_writes() {
    let harness = Harness::new(default_config());
    // The registry knows no `processing_v17_missing` model.
    let mut config = relation_config(&harness);
    config.model_name_suffix = "missing".into();
    let stage = RelationStage::new(&config, &harness.services).expect("builds");
    let stages: Vec<Box<dyn Stage>> = vec![Box::new(harness.stage()), Box::new(stage)];
    let pipeline = Pipeline::new(stages, harness.services.clone());
    let mut ctx = pipeline.new_context();
    let input = PipelineInput::Payload(Box::new(DocumentPayload::new("doc-rel", TEXT)));
    pipeline.run(&mut ctx, input).expect("runs");
    assert!(ctx.data.extracted_relations.is_none());
    assert!(ctx.data.relations_by_type.is_none());
}

/// A model whose relation extraction always fails.
#[derive(Debug)]
struct FailingNer;

impl NerModel for FailingNer {
    fn extract_entities(&self, _: &str, _: &[&str], _: f32) -> Result<Vec<NerSpan>, ModelError> {
        Ok(vec![])
    }

    fn extract_relations(
        &self,
        _: &str,
        _: &[&str],
        _: f32,
    ) -> Result<Vec<RelationHit>, ModelError> {
        Err(ModelError::Inference("boom".into()))
    }
}

#[test]
fn chunk_failures_are_logged_and_skipped() {
    let harness = Harness::new(default_config());
    let registry = &harness.services.registry;
    registry.insert_ner(registry.key(ModelKind::Gliner2), Arc::new(FailingNer));
    let data = run(&harness);
    assert_eq!(data.extracted_relations.as_deref(), Some(&[][..]));
    assert_eq!(data.relations_by_type.map(|m| m.len()), Some(0));
}

#[test]
fn without_indexing_result_nothing_happens() {
    let harness = Harness::new(default_config());
    let stage = RelationStage::new(&relation_config(&harness), &harness.services).expect("builds");
    let mut ctx = PipelineContext::new(&harness.services);
    stage.run(&mut ctx).expect("runs");
    assert!(ctx.data.extracted_relations.is_none());
}
