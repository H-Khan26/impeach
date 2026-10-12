//! `FactStage`: LLM fact extraction per chunk, entity linking and optional commit (Python
//! `stages/facts.py`).

use std::sync::Arc;

use ce_config::pipeline::FactStageConfig;
use ce_config::StageConfig;
use ce_facts::FactExtractor;

use crate::context::{PipelineContext, StageBuildContext};
use crate::error::{PipelineError, Result};
use crate::stage::Stage;

/// The fact extraction stage.
#[derive(Debug)]
pub struct FactStage {
    name: String,
    config: FactStageConfig,
    pipeline_name: String,
    extractor: Option<Arc<dyn FactExtractor>>,
}

impl FactStage {
    /// Config `type` of this stage.
    pub const TYPE_NAME: &'static str = "FactStage";

    /// Builds the stage from its config block and the shared services.
    ///
    /// # Errors
    /// [`PipelineError::Config`] when a prompt file, the mappings or `roles.yaml` cannot be
    /// loaded.
    pub fn new(config: &FactStageConfig, services: &StageBuildContext) -> Result<Self> {
        Ok(Self {
            name: config.name.clone(),
            config: config.clone(),
            pipeline_name: services.config.name().to_owned(),
            extractor: None,
        })
    }

    /// [`crate::StageConstructor`] of this stage.
    ///
    /// # Errors
    /// [`PipelineError::InvalidStageConfig`] for another stage type; errors of
    /// [`FactStage::new`].
    pub fn construct(stage: &StageConfig, services: &StageBuildContext) -> Result<Box<dyn Stage>> {
        match stage {
            StageConfig::Fact(config) => Ok(Box::new(Self::new(config, services)?)),
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

    /// Replaces the extractor built from the registry's LLM and the prompt files.
    #[must_use]
    pub fn with_extractor(mut self, extractor: Arc<dyn FactExtractor>) -> Self {
        self.extractor = Some(extractor);
        self
    }
}

impl Stage for FactStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<()> {
        let _ = (ctx, &self.config, &self.pipeline_name, &self.extractor);
        todo!("worker B")
    }
}
