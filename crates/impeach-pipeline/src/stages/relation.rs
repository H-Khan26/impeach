//! `RelationStage`: GLiNER2 relations per chunk, kept in the context only (Python
//! `stages/relation.py`).
//!
//! The labels are the `relation_label_set` of `gliner2.yaml` (an unknown set fails the stage
//! build); each relation label maps to a canonical relationship type through
//! `relation_to_relationship_type` (`OTHER` when unmapped). The model is
//! `<pipeline.name>_<model_name_suffix>`; a key the registry does not know skips the stage with
//! a warning (context unchanged). The score threshold is `models.gliner2.threshold`, or
//! [`DEFAULT_RELATION_THRESHOLD`] without a `models.gliner2` block (an injected model).
//!
//! Only chunks with a graph id are processed. A model failure (loading or per chunk) is logged
//! and its chunks contribute no relations. Relations without head or tail text are dropped.
//! Writes `extracted_relations` and `relations_by_type` (label -> count, in first-seen order);
//! nothing is written to the store.

use std::sync::Arc;

use impeach_config::pipeline::RelationStageConfig;
use impeach_config::{MappingsRegistry, StageConfig};
use impeach_core::processing::ExtractedRelation;
use impeach_core::ChunkId;
use impeach_models::{ModelError, ModelRegistry, NerModel};
use indexmap::IndexMap;

use crate::context::{PipelineContext, StageBuildContext};
use crate::error::{PipelineError, Result};
use crate::stage::Stage;

/// Relation score threshold when the config has no `models.gliner2` block.
pub const DEFAULT_RELATION_THRESHOLD: f32 = 0.5;

/// Canonical relationship type of unmapped relation labels.
pub const DEFAULT_CANONICAL_RELATION: &str = "OTHER";

/// The relation extraction stage.
#[derive(Debug)]
pub struct RelationStage {
    name: String,
    registry: Arc<ModelRegistry>,
    model_key: String,
    labels: Vec<String>,
    relation_to_canonical: IndexMap<String, String>,
    threshold: f32,
}

impl RelationStage {
    /// Config `type` of this stage.
    pub const TYPE_NAME: &'static str = "RelationStage";

    /// Builds the stage from its config block and the shared services.
    ///
    /// # Errors
    /// [`PipelineError::Config`] when the mappings cannot be loaded or the relation label set
    /// does not exist.
    pub fn new(config: &RelationStageConfig, services: &StageBuildContext) -> Result<Self> {
        let mappings = MappingsRegistry::load()?;
        let labels = mappings.gliner2.get_relation_label_set(&config.relation_label_set)?.to_vec();
        tracing::info!(
            relations = labels.len(),
            label_set = config.relation_label_set.as_str(),
            "RelationStage: loaded relation labels"
        );
        let threshold = services
            .config
            .models
            .gliner2
            .as_ref()
            .map_or(DEFAULT_RELATION_THRESHOLD, |model| model.threshold as f32);
        Ok(Self {
            name: config.name.clone(),
            registry: Arc::clone(&services.registry),
            model_key: format!("{}_{}", services.config.name(), config.model_name_suffix),
            labels,
            relation_to_canonical: mappings.gliner2.relation_to_relationship_type.clone(),
            threshold,
        })
    }

    /// [`crate::StageConstructor`] of this stage.
    ///
    /// # Errors
    /// [`PipelineError::InvalidStageConfig`] for another stage type; errors of
    /// [`RelationStage::new`].
    pub fn construct(stage: &StageConfig, services: &StageBuildContext) -> Result<Box<dyn Stage>> {
        match stage {
            StageConfig::Relation(config) => Ok(Box::new(Self::new(config, services)?)),
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

    /// Registry key of the relation model.
    #[must_use]
    pub fn model_key(&self) -> &str {
        &self.model_key
    }

    /// The relation labels.
    #[must_use]
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// The relation score threshold.
    #[must_use]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Normalised relations of one chunk.
    fn chunk_relations(
        &self,
        model: &dyn NerModel,
        labels: &[&str],
        chunk_id: &ChunkId,
        text: &str,
    ) -> std::result::Result<Vec<ExtractedRelation>, ModelError> {
        let hits = model.extract_relations(text, labels, self.threshold)?;
        Ok(hits
            .into_iter()
            .filter_map(|hit| {
                if hit.head.text.is_empty() || hit.tail.text.is_empty() {
                    tracing::debug!(relation = hit.label.as_str(), "skipping incomplete relation");
                    return None;
                }
                let canonical_relation = self
                    .relation_to_canonical
                    .get(&hit.label)
                    .map_or(DEFAULT_CANONICAL_RELATION, String::as_str)
                    .to_owned();
                Some(ExtractedRelation {
                    relation: hit.label,
                    head: hit.head.text,
                    tail: hit.tail.text,
                    chunk_id: chunk_id.clone(),
                    canonical_relation,
                    score: Some(f64::from(hit.score)),
                })
            })
            .collect())
    }
}

impl Stage for RelationStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<()> {
        let Some(indexing) = ctx.data.rag_index_result.as_ref() else {
            return Ok(());
        };
        let model = match self.registry.ner(&self.model_key) {
            Ok(model) => Some(model),
            Err(ModelError::UnknownModel { .. }) => {
                tracing::warn!(
                    key = self.model_key.as_str(),
                    "RelationStage: model service not found; skipping"
                );
                return Ok(());
            }
            Err(error) => {
                tracing::error!(
                    key = self.model_key.as_str(),
                    error = %error,
                    "RelationStage: model unavailable; no relations extracted"
                );
                None
            }
        };

        let labels: Vec<&str> = self.labels.iter().map(String::as_str).collect();
        let mut relations: Vec<ExtractedRelation> = Vec::new();
        let mut relations_by_type: IndexMap<String, usize> = IndexMap::new();
        if let Some(model) = &model {
            for write in &indexing.chunks {
                if write.helix_id.as_str().is_empty() {
                    continue;
                }
                let chunk_id = ChunkId::from(write.helix_id.as_str());
                match self.chunk_relations(model.as_ref(), &labels, &chunk_id, &write.chunk.text) {
                    Ok(found) => {
                        for relation in &found {
                            *relations_by_type.entry(relation.relation.clone()).or_default() += 1;
                        }
                        relations.extend(found);
                    }
                    Err(error) => tracing::error!(
                        chunk = %chunk_id,
                        error = %error,
                        "RelationStage: failed to extract relations"
                    ),
                }
            }
        }
        tracing::info!(
            relations = relations.len(),
            chunks = indexing.chunks.len(),
            types = relations_by_type.len(),
            "RelationStage: extracted relations"
        );
        ctx.data.extracted_relations = Some(relations);
        ctx.data.relations_by_type = Some(relations_by_type);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use impeach_models::{MockNer, NerSpan, RelationHit};

    fn span(text: &str) -> NerSpan {
        NerSpan { start: 0, end: text.len(), label: "x".into(), score: 0.9, text: text.into() }
    }

    #[test]
    fn unmapped_labels_become_other() {
        let stage = RelationStage {
            name: "relation_extraction".into(),
            registry: Arc::new(ModelRegistry::empty("p")),
            model_key: "p_gliner2".into(),
            labels: vec!["sued".into(), "employed_by".into()],
            relation_to_canonical: [("employed_by".to_owned(), "EMPLOYED_BY".to_owned())]
                .into_iter()
                .collect(),
            threshold: 0.5,
        };
        let model = MockNer::new(
            vec![],
            ["sued", "employed_by"]
                .iter()
                .map(|label| RelationHit {
                    head: span("Lopez"),
                    tail: span("Reed"),
                    label: (*label).into(),
                    score: 0.75,
                })
                .collect(),
        );
        let relations = stage
            .chunk_relations(&model, &["sued", "employed_by"], &ChunkId::new("7"), "text")
            .expect("relations");
        let canonical: Vec<&str> =
            relations.iter().map(|r| r.canonical_relation.as_str()).collect();
        assert_eq!(canonical, ["OTHER", "EMPLOYED_BY"]);
        assert_eq!(relations[0].score, Some(0.75));
        assert_eq!(relations[0].chunk_id.as_str(), "7");
    }
}
