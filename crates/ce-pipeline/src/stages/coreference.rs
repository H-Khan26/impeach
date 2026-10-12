//! `CoreferenceStage`: coreference clusters, canonical mentions and pronoun mentions (Python `stages/coreference.py`).

use ce_config::pipeline::CoreferenceStageConfig;
use ce_config::StageConfig;

use crate::context::{PipelineContext, StageBuildContext};
use crate::error::{PipelineError, Result};
use crate::stage::Stage;

/// The coreference stage.
#[derive(Debug)]
pub struct CoreferenceStage {
    name: String,
}

impl CoreferenceStage {
    /// Config `type` of this stage.
    pub const TYPE_NAME: &'static str = "CoreferenceStage";

    /// Builds the stage from its config block and the shared services.
    ///
    /// # Errors
    /// Invalid stage settings.
    pub fn new(config: &CoreferenceStageConfig, _services: &StageBuildContext) -> Result<Self> {
        Ok(Self { name: config.name.clone() })
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
}

impl Stage for CoreferenceStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, _ctx: &mut PipelineContext) -> Result<()> {
        Ok(())
    }
}
