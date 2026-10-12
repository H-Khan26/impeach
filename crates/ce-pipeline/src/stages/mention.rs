//! `MentionStage`: per-chunk mention detection (Python `stages/mention.py`).

use ce_config::pipeline::MentionStageConfig;
use ce_config::StageConfig;

use crate::context::{PipelineContext, StageBuildContext};
use crate::error::{PipelineError, Result};
use crate::stage::Stage;

/// The mention detection stage.
#[derive(Debug)]
pub struct MentionStage {
    name: String,
}

impl MentionStage {
    /// Config `type` of this stage.
    pub const TYPE_NAME: &'static str = "MentionStage";

    /// Builds the stage from its config block and the shared services.
    ///
    /// # Errors
    /// Invalid stage settings.
    pub fn new(config: &MentionStageConfig, _services: &StageBuildContext) -> Result<Self> {
        Ok(Self { name: config.name.clone() })
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
}

impl Stage for MentionStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, _ctx: &mut PipelineContext) -> Result<()> {
        Ok(())
    }
}
