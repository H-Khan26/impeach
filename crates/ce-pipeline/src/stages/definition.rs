//! `DefinitionStage`: defined role terms (Python `stages/definition.py`).

use ce_config::pipeline::DefinitionStageConfig;
use ce_config::StageConfig;

use crate::context::{PipelineContext, StageBuildContext};
use crate::error::{PipelineError, Result};
use crate::stage::Stage;

/// The definition stage.
#[derive(Debug)]
pub struct DefinitionStage {
    name: String,
}

impl DefinitionStage {
    /// Config `type` of this stage.
    pub const TYPE_NAME: &'static str = "DefinitionStage";

    /// Builds the stage from its config block and the shared services.
    ///
    /// # Errors
    /// Invalid stage settings.
    pub fn new(config: &DefinitionStageConfig, _services: &StageBuildContext) -> Result<Self> {
        Ok(Self { name: config.name.clone() })
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

    fn run(&self, _ctx: &mut PipelineContext) -> Result<()> {
        Ok(())
    }
}
