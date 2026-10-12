//! `DefinitionStage`: defined role terms (Python `stages/definition.py`).
//!
//! Reads `doc_mentions`, `doc_text` and `doc_id`; writes `defined_terms` (pattern definitions
//! followed by mention-metadata definitions) when at least one is found. Without mentions or
//! document text the stage writes nothing. The role map is built later, once entity ids exist
//! (see [`ce_ner::definitions::DefinitionExtractor::build_role_map`]).

use ce_config::pipeline::DefinitionStageConfig;
use ce_config::StageConfig;
use ce_ner::definitions::{DefinitionExtractor, RoleConfig};
use indexmap::IndexMap;

use crate::context::{PipelineContext, StageBuildContext};
use crate::error::{PipelineError, Result};
use crate::stage::Stage;

/// The definition stage.
#[derive(Debug)]
pub struct DefinitionStage {
    name: String,
    enabled: bool,
    extractor: DefinitionExtractor,
}

impl DefinitionStage {
    /// Config `type` of this stage.
    pub const TYPE_NAME: &'static str = "DefinitionStage";

    /// Builds the stage from its config block and the shared services. Loads `roles.yaml` from
    /// `role_config_path`, or the shipped file when unset.
    ///
    /// # Errors
    /// [`PipelineError::Config`] when the roles file is missing or invalid.
    pub fn new(config: &DefinitionStageConfig, _services: &StageBuildContext) -> Result<Self> {
        let roles = RoleConfig::load(config.role_config_path.as_deref())?;
        Ok(Self::with_role_config(config, roles))
    }

    /// Builds the stage over an already loaded role config.
    #[must_use]
    pub fn with_role_config(config: &DefinitionStageConfig, roles: RoleConfig) -> Self {
        Self {
            name: config.name.clone(),
            enabled: config.enabled,
            extractor: DefinitionExtractor::new(roles),
        }
    }

    /// The extractor the stage runs.
    #[must_use]
    pub fn extractor(&self) -> &DefinitionExtractor {
        &self.extractor
    }

    /// [`crate::StageConstructor`] of this stage.
    ///
    /// # Errors
    /// [`PipelineError::InvalidStageConfig`] for another stage type; errors of
    /// [`DefinitionStage::new`].
    pub fn construct(stage: &StageConfig, services: &StageBuildContext) -> Result<Box<dyn Stage>> {
        match stage {
            StageConfig::Definition(config) => Ok(Box::new(Self::new(config, services)?)),
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
}

impl Stage for DefinitionStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<()> {
        if !self.enabled {
            tracing::debug!("DefinitionStage is disabled, skipping");
            return Ok(());
        }
        let data = &ctx.data;
        let Some(mentions) = data.doc_mentions.as_deref().filter(|m| !m.is_empty()) else {
            tracing::debug!("DefinitionStage: no mentions found, skipping");
            return Ok(());
        };
        let Some(doc_text) = data.doc_text.as_deref().filter(|t| !t.is_empty()) else {
            tracing::debug!("DefinitionStage: no doc_text in context, skipping");
            return Ok(());
        };
        let doc_id = data.doc_id.as_ref().map_or("unknown", |id| id.as_str());

        let mut definitions = self.extractor.extract_from_text(doc_text, mentions, doc_id);
        definitions.extend(self.extractor.extract_from_metadata(mentions));

        if definitions.is_empty() {
            tracing::info!("DefinitionStage: No role definitions found for doc {doc_id}");
            return Ok(());
        }

        let mut role_summary: IndexMap<&str, usize> = IndexMap::new();
        for definition in &definitions {
            *role_summary.entry(definition.defined_term.as_str()).or_default() += 1;
        }
        tracing::info!(
            "DefinitionStage: Extracted {} definitions for {} unique roles in doc {doc_id}: {:?}",
            definitions.len(),
            role_summary.len(),
            role_summary.keys().collect::<Vec<_>>()
        );

        ctx.data.defined_terms = Some(definitions);
        Ok(())
    }
}
