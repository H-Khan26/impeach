//! `MentionStage`: per-chunk mention detection (Python `stages/mention.py`).
//!
//! Reads `rag_index_result` (without it the stage does nothing, as Python returns early when
//! its input is not an `IndexingResult`) and `helix_doc_id`. Every chunk with a graph id runs
//! through [`MentionDetection::detect_chunk`] with the configured detectors, in config order:
//! `eyecite` -> [`CitationDetector`], `regex` -> [`RegexDetector`], `gliner2` ->
//! [`GlinerDetector`] over the registry's `<pipeline.name>_<model_name_suffix>` NER model.
//! The GLiNER2 label set is checked when the stage is built; the model is resolved on the
//! first chunk and the resolved detectors are kept for later documents; a missing or failing
//! model fails the stage (Python lets the model error propagate) and is retried next run. The model threshold is `models.gliner2.threshold`, or the detector's
//! `min_score` when `models.gliner2` is not configured (an injected model).
//!
//! Context writes: `mentions_by_chunk` (chunk graph id -> merged mentions, chunks with at least
//! one mention), `doc_mentions` (all of them, in chunk order) and `mention_to_span_id`.
//!
//! Mention spans: when there are mentions, a `helix_doc_id` and `index_persisted` is set (the
//! index was written to the store; Python checks for a `helix_client`), each mention becomes a `Span`
//! node under its chunk (`SpanType` `MENTION`, label = mention type, surface text = name,
//! metadata `{"ner_source", "confidence", "entity_type", "mention_id"[, "sentence_index"]}`)
//! written with [`GraphStore::create_spans_batch`]; a failed batch falls back to one write
//! per span, logging and skipping failures (Python writes one by one and skips failures);
//! only fatal store errors fail the stage. `mention_to_span_id` maps each written mention id
//! to its span id. After an offline RAGIndexStage (fabricated ids, nothing in the store) the
//! stage writes no spans and leaves `mention_to_span_id` unset.

use std::sync::{Arc, OnceLock};

use ce_config::pipeline::{MentionDetector as DetectorConfig, MentionStageConfig};
use ce_config::{MappingsRegistry, StageConfig};
use ce_core::pyjson::to_python_json_ascii;
use ce_core::util::saturating_i64;
use ce_core::{ChunkId, IndexingResult, Mention, MentionId, Metadata, NodeId, SpanId};
use ce_models::ModelRegistry;
use ce_ner::{
    ChunkView, CitationDetector, ConfiguredDetector, GlinerDetector, MentionDetection, NerError,
    RegexDetector,
};
use ce_store::{GraphStore, SpanNode};
use indexmap::IndexMap;
use serde_json::Value;

use crate::context::{PipelineContext, StageBuildContext};
use crate::error::{PipelineError, Result};
use crate::stage::Stage;

/// Span type of mention spans.
pub const MENTION_SPAN_TYPE: &str = "MENTION";

/// Mentions per chunk graph id (context slot `mentions_by_chunk`).
pub type MentionsByChunk = IndexMap<ChunkId, Vec<Mention>>;

/// A configured detector; GLiNER2 waits for its model until the first chunk.
#[derive(Debug)]
enum DetectorSlot {
    Ready(ConfiguredDetector),
    Gliner { key: String, label_set: String, threshold: f32, min_score: f64 },
}

/// The mention detection stage.
#[derive(Debug)]
pub struct MentionStage {
    name: String,
    mappings: &'static MappingsRegistry,
    detection: MentionDetection,
    detectors: Vec<DetectorSlot>,
    /// The detectors with their models, resolved on first use.
    resolved: OnceLock<Vec<ConfiguredDetector>>,
    registry: Arc<ModelRegistry>,
    store: Arc<dyn GraphStore>,
}

impl MentionStage {
    /// Config `type` of this stage.
    pub const TYPE_NAME: &'static str = "MentionStage";

    /// Builds the stage from its config block and the shared services.
    ///
    /// # Errors
    /// [`PipelineError::Config`] when the mappings do not load or a GLiNER2 `label_set` is
    /// unknown.
    pub fn new(config: &MentionStageConfig, services: &StageBuildContext) -> Result<Self> {
        let mappings = MappingsRegistry::load()?;
        let pipeline = &services.config;
        let mut detectors = Vec::with_capacity(config.detectors.len());
        for detector in &config.detectors {
            let slot = match detector {
                DetectorConfig::Eyecite => {
                    DetectorSlot::Ready(ConfiguredDetector::new(Arc::new(CitationDetector), 0.0))
                }
                DetectorConfig::Regex => DetectorSlot::Ready(ConfiguredDetector::new(
                    Arc::new(RegexDetector::from_mappings(&mappings.detection_patterns)),
                    0.0,
                )),
                DetectorConfig::Gliner2(gliner) => {
                    mappings.gliner2.get_entity_label_set(&gliner.label_set)?;
                    let threshold = pipeline
                        .models
                        .gliner2
                        .as_ref()
                        .map_or(gliner.min_score, |model| model.threshold);
                    DetectorSlot::Gliner {
                        key: format!("{}_{}", pipeline.name(), gliner.model_name_suffix),
                        label_set: gliner.label_set.clone(),
                        threshold: threshold as f32,
                        min_score: gliner.min_score,
                    }
                }
            };
            detectors.push(slot);
        }
        Ok(Self {
            name: config.name.clone(),
            mappings,
            detection: MentionDetection::new(mappings, config.normalize_names),
            detectors,
            resolved: OnceLock::new(),
            registry: Arc::clone(&services.registry),
            store: Arc::clone(&services.store),
        })
    }

    /// [`crate::StageConstructor`] of this stage.
    ///
    /// # Errors
    /// [`PipelineError::InvalidStageConfig`] for another stage type; errors of
    /// [`MentionStage::new`].
    pub fn construct(stage: &StageConfig, services: &StageBuildContext) -> Result<Box<dyn Stage>> {
        match stage {
            StageConfig::Mention(config) => Ok(Box::new(Self::new(config, services)?)),
            other => Err(PipelineError::InvalidStageConfig {
                stage: other.name().to_owned(),
                message: format!(
                    "expected a {} config, got {}",
                    Self::TYPE_NAME,
                    other.type_name()
                ),
            }),
        }
    }

    /// The resolved detectors, built on the first call; a failed resolution is not kept.
    fn active_detectors(&self) -> Result<&[ConfiguredDetector]> {
        if let Some(resolved) = self.resolved.get() {
            return Ok(resolved);
        }
        let built = self.resolve_detectors()?;
        Ok(self.resolved.get_or_init(|| built))
    }

    /// The detectors in config order, resolving GLiNER2 models from the registry.
    fn resolve_detectors(&self) -> Result<Vec<ConfiguredDetector>> {
        self.detectors
            .iter()
            .map(|slot| match slot {
                DetectorSlot::Ready(detector) => Ok(detector.clone()),
                DetectorSlot::Gliner { key, label_set, threshold, min_score } => {
                    let model = self.registry.ner(key)?;
                    let detector = GlinerDetector::new(model, self.mappings, label_set, *threshold)
                        .map_err(|e| self.ner_error(e))?;
                    Ok(ConfiguredDetector::new(Arc::new(detector), *min_score))
                }
            })
            .collect()
    }

    fn ner_error(&self, error: NerError) -> PipelineError {
        match error {
            NerError::Config(e) => PipelineError::Config(e),
            NerError::Model(e) => PipelineError::Model(e),
            NerError::InvalidConfig(message) => {
                PipelineError::InvalidStageConfig { stage: self.name.clone(), message }
            }
        }
    }

    /// Mentions of every chunk with a graph id: per chunk (chunks without mentions left out)
    /// and in document order.
    ///
    /// # Errors
    /// Model resolution and detector failures.
    pub fn detect(&self, result: &IndexingResult) -> Result<(MentionsByChunk, Vec<Mention>)> {
        let mut by_chunk = IndexMap::new();
        let mut all = Vec::new();
        for write in &result.chunks {
            if write.helix_id.as_str().is_empty() {
                continue;
            }
            let active = self.active_detectors()?;
            let chunk = &write.chunk;
            let chunk_id = ChunkId::from(write.helix_id.as_str());
            let view = ChunkView {
                text: &chunk.text,
                doc_id: &chunk.doc_id,
                chunk_id: &chunk_id,
                sequence_number: chunk.sequence_number,
                start_char: chunk.start_char,
                sentence_spans: &chunk.sentence_spans,
            };
            let mentions =
                self.detection.detect_chunk(active, &view).map_err(|e| self.ner_error(e))?;
            if !mentions.is_empty() {
                all.extend(mentions.iter().cloned());
                by_chunk.insert(chunk_id, mentions);
            }
        }
        Ok((by_chunk, all))
    }

    /// Writes one `MENTION` span per mention and returns mention id -> span id for the spans
    /// that were written.
    ///
    /// # Errors
    /// Fatal store errors only.
    pub fn write_mention_spans(
        &self,
        mentions: &[Mention],
        doc_node_id: &NodeId,
    ) -> Result<IndexMap<MentionId, SpanId>> {
        let mut nodes = Vec::with_capacity(mentions.len());
        let mut ids = Vec::with_capacity(mentions.len());
        for mention in mentions {
            match mention_span_node(mention, doc_node_id) {
                Some(node) => {
                    nodes.push(node);
                    ids.push(mention.id.clone());
                }
                None => tracing::debug!(name = %mention.name, "mention has no chunk_id; no span"),
            }
        }
        let mut written = IndexMap::new();
        if nodes.is_empty() {
            return Ok(written);
        }
        match self.store.create_spans_batch(&nodes) {
            Ok(stored) => {
                for (id, span) in ids.into_iter().zip(stored) {
                    written.insert(id, span.id);
                }
            }
            Err(error) if error.is_fatal() => return Err(error.into()),
            Err(error) => {
                tracing::error!("mention span batch failed ({error}); writing spans one by one");
                for (id, node) in ids.into_iter().zip(&nodes) {
                    match self.store.create_span_for_chunk(node) {
                        Ok(stored) => {
                            written.insert(id, stored.id);
                        }
                        Err(error) if error.is_fatal() => return Err(error.into()),
                        Err(error) => tracing::warn!(
                            mention = %node.surface_text,
                            "failed to write mention span: {error}"
                        ),
                    }
                }
            }
        }
        Ok(written)
    }
}

/// The `MENTION` span node of a mention under document `doc_node_id`; `None` without a
/// `chunk_id`.
#[must_use]
pub fn mention_span_node(mention: &Mention, doc_node_id: &NodeId) -> Option<SpanNode> {
    let chunk_id = mention.chunk_id.clone()?;
    let mut metadata = Metadata::new();
    metadata.insert(
        "ner_source".into(),
        Value::from(mention.source.as_deref().filter(|s| !s.is_empty()).unwrap_or("unknown")),
    );
    metadata.insert("confidence".into(), Value::from(mention.score.unwrap_or(0.0)));
    metadata.insert("entity_type".into(), Value::from(mention.entity_type.as_str()));
    metadata.insert("mention_id".into(), Value::from(mention.id.as_str()));
    if let Some(index) = mention.sentence_index {
        metadata.insert("sentence_index".into(), Value::from(index));
    }
    Some(SpanNode {
        doc_id: doc_node_id.clone(),
        chunk_id,
        start_char: saturating_i64(mention.span.start),
        end_char: saturating_i64(mention.span.end),
        span_type: MENTION_SPAN_TYPE.to_owned(),
        span_label: mention.entity_type.clone(),
        surface_text: mention.name.clone(),
        metadata: to_python_json_ascii(&metadata).unwrap_or_else(|_| "{}".to_owned()),
    })
}

impl Stage for MentionStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<()> {
        let Some(result) = ctx.data.rag_index_result.as_ref() else {
            tracing::warn!(stage = %self.name, "no rag_index_result in the context; skipping");
            return Ok(());
        };
        let (by_chunk, all) = self.detect(result)?;
        tracing::info!(
            stage = %self.name,
            chunks = by_chunk.len(),
            mentions = all.len(),
            "detected mentions"
        );
        if ctx.data.index_persisted && !all.is_empty() {
            if let Some(doc_node_id) = ctx.data.helix_doc_id.clone() {
                let written = self.write_mention_spans(&all, &doc_node_id)?;
                tracing::info!(
                    written = written.len(),
                    mentions = all.len(),
                    "wrote mention spans"
                );
                ctx.data.mention_to_span_id = Some(written);
            }
        }
        ctx.data.mentions_by_chunk = Some(by_chunk);
        ctx.data.doc_mentions = Some(all);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ce_core::Span;
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn span_node_metadata_matches_python_key_order() {
        let mut mention = Mention::new("span-0a1b2c3d", "MONEY", "$300", Span::new(10, 14));
        mention.score = Some(1.0);
        mention.source = Some("regex_detector".into());
        mention.chunk_id = Some("5".into());
        mention.sentence_index = Some(2);
        let node = mention_span_node(&mention, &NodeId::from("1")).expect("node");
        assert_eq!(node.span_type, "MENTION");
        assert_eq!(node.span_label, "MONEY");
        assert_eq!(node.surface_text, "$300");
        assert_eq!((node.start_char, node.end_char), (10, 14));
        assert_eq!(node.chunk_id.as_str(), "5");
        assert_eq!(
            node.metadata,
            r#"{"ner_source": "regex_detector", "confidence": 1.0, "entity_type": "MONEY", "mention_id": "span-0a1b2c3d", "sentence_index": 2}"#
        );

        mention.score = None;
        mention.source = None;
        mention.sentence_index = None;
        mention.name = "Zoë".into();
        let node = mention_span_node(&mention, &NodeId::from("1")).expect("node");
        assert_eq!(
            node.metadata,
            r#"{"ner_source": "unknown", "confidence": 0.0, "entity_type": "MONEY", "mention_id": "span-0a1b2c3d"}"#
        );
        mention.chunk_id = None;
        assert!(mention_span_node(&mention, &NodeId::from("1")).is_none());
    }
}
