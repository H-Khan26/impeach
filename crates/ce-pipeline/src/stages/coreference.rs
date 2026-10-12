//! `CoreferenceStage`: coreference clusters, canonical mentions and pronoun mentions
//! (Python `stages/coreference.py`).
//!
//! Flow:
//! 1. Skip (context unchanged) without `rag_index_result`, without mentions, or when
//!    `doc_text` is longer than `max_doc_chars` characters.
//! 2. The text is `doc_text`, or the chunk texts joined in sequence order; an empty text
//!    skips.
//! 3. Resolve the `<pipeline.name>_<model_name_suffix>` coreference model. A key the registry
//!    does not know skips with a warning (Python: service not registered); any other model
//!    error (a sidecar that cannot start, a load or inference failure) fails the stage:
//!    coreference never degrades silently to "no clusters". (`fastcoref` without a sidecar
//!    command resolves to the rule backend with a warning; see [`ce_models::registry`].)
//! 4. Call [`ce_models::Coref::resolve_with_mentions`] with the mentions of eligible types,
//!    sorted by start, and post-process the clusters with [`ce_coref::CorefPostProcessor`].
//!
//! Context writes, only when a cluster survives: `coref_clusters`,
//! `canonical_mention_for_cluster`, `canonical_text_for_mention`; with synthetic pronoun
//! mentions also `synthetic_pronoun_mentions`, and the mentions are appended to `doc_mentions`
//! and to `mentions_by_chunk` under their chunk's graph id. Like the Python stage, synthetic
//! pronoun mentions get no `Span` node in the store.

use std::borrow::Cow;
use std::sync::Arc;

use ce_config::pipeline::CoreferenceStageConfig;
use ce_config::StageConfig;
use ce_coref::{reconstruct_doc_text, CorefPostProcessor, CorefSettings, RoleConfig};
use ce_models::{CorefMention, ModelError, ModelRegistry};

use crate::context::{PipelineContext, StageBuildContext};
use crate::error::{PipelineError, Result};
use crate::stage::Stage;

/// The coreference stage.
#[derive(Debug)]
pub struct CoreferenceStage {
    name: String,
    registry: Arc<ModelRegistry>,
    model_key: String,
    processor: CorefPostProcessor,
}

impl CoreferenceStage {
    /// Config `type` of this stage.
    pub const TYPE_NAME: &'static str = "CoreferenceStage";

    /// Builds the stage from its config block and the shared services. Role terms for
    /// canonical mention selection come from the shipped `roles.yaml`, like the Python stage.
    ///
    /// # Errors
    /// [`PipelineError::Config`] when `roles.yaml` cannot be loaded.
    pub fn new(config: &CoreferenceStageConfig, services: &StageBuildContext) -> Result<Self> {
        let roles = RoleConfig::load(None)?;
        Ok(Self {
            name: config.name.clone(),
            registry: Arc::clone(&services.registry),
            model_key: format!("{}_{}", services.config.name(), config.model_name_suffix),
            processor: CorefPostProcessor::new(CorefSettings::from_config(config), roles),
        })
    }

    /// [`crate::StageConstructor`] of this stage.
    ///
    /// # Errors
    /// [`PipelineError::InvalidStageConfig`] for another stage type; errors of
    /// [`CoreferenceStage::new`].
    pub fn construct(stage: &StageConfig, services: &StageBuildContext) -> Result<Box<dyn Stage>> {
        match stage {
            StageConfig::Coreference(config) => Ok(Box::new(Self::new(config, services)?)),
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

    /// Registry key of the coreference model.
    pub fn model_key(&self) -> &str {
        &self.model_key
    }
}

impl Stage for CoreferenceStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<()> {
        let data = &ctx.data;
        let Some(indexing) = data.rag_index_result.as_ref() else {
            return Ok(());
        };
        let mentions = match data.doc_mentions.as_deref() {
            Some(mentions) if !mentions.is_empty() => mentions,
            _ => return Ok(()),
        };
        let doc_id = data.doc_id.clone().unwrap_or_else(|| indexing.document.doc_id.clone());
        if let Some(text) = &data.doc_text {
            if self.processor.exceeds_max_doc_chars(text) {
                tracing::info!(
                    doc_id = %doc_id,
                    chars = text.chars().count(),
                    max_doc_chars = ?self.processor.settings().max_doc_chars,
                    "coreference: document too long; skipping"
                );
                return Ok(());
            }
        }
        let text: Cow<'_, str> = match &data.doc_text {
            Some(text) => Cow::Borrowed(text),
            None => Cow::Owned(reconstruct_doc_text(indexing)),
        };
        if text.is_empty() {
            tracing::info!(doc_id = %doc_id, "coreference: empty document text; skipping");
            return Ok(());
        }
        let coref = match self.registry.coref(&self.model_key) {
            Ok(coref) => coref,
            Err(ModelError::UnknownModel { .. }) => {
                tracing::warn!(
                    key = %self.model_key,
                    "coreference: model is not registered; skipping coreference"
                );
                return Ok(());
            }
            Err(error) => return Err(error.into()),
        };

        let settings = self.processor.settings();
        let mut eligible: Vec<CorefMention<'_>> = mentions
            .iter()
            .filter(|m| settings.is_eligible(&m.entity_type))
            .map(|m| CorefMention {
                start: m.span.start,
                end: m.span.end,
                text: &m.name,
                entity_type: &m.entity_type,
            })
            .collect();
        eligible.sort_by_key(|m| m.start);
        let raw = coref.resolve_with_mentions(&text, &eligible)?;
        let Some(output) = self.processor.process(&doc_id, &text, &raw, mentions, &indexing.chunks)
        else {
            return Ok(());
        };
        tracing::info!(
            doc_id = %doc_id,
            clusters = output.clusters.len(),
            synthetic_mentions = output.synthetic_mentions.len(),
            "coreference: clusters resolved"
        );

        let data = &mut ctx.data;
        data.coref_clusters = Some(output.clusters);
        data.canonical_mention_for_cluster = Some(output.canonical_mention_for_cluster);
        data.canonical_text_for_mention = Some(output.canonical_text_for_mention);
        if let Some(synthetic) = output.synthetic_pronoun_mentions {
            data.doc_mentions.get_or_insert_with(Vec::new).extend(output.synthetic_mentions);
            let by_chunk = data.mentions_by_chunk.get_or_insert_with(Default::default);
            for (chunk, mentions) in output.synthetic_by_chunk {
                by_chunk.entry(chunk).or_default().extend(mentions);
            }
            data.synthetic_pronoun_mentions = Some(synthetic);
        }
        Ok(())
    }
}
