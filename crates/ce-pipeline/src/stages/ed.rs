//! `EDStage`: resolves mentions to entities (Python `stages/ed.py`).

use ce_config::pipeline::EdStageConfig;
use ce_config::StageConfig;

use crate::context::{PipelineContext, StageBuildContext};
use crate::error::{PipelineError, Result};
use crate::stage::Stage;

/// The entity disambiguation stage.
#[derive(Debug)]
pub struct EdStage {
    name: String,
}

impl EdStage {
    /// Config `type` of this stage.
    pub const TYPE_NAME: &'static str = "EDStage";

    /// Builds the stage from its config block and the shared services.
    ///
    /// # Errors
    /// Configuration errors.
    pub fn new(config: &EdStageConfig, services: &StageBuildContext) -> Result<Self> {
        let _ = services;
        Ok(Self { name: config.name.clone() })
    }

    /// [`crate::StageConstructor`] of this stage.
    ///
    /// # Errors
    /// [`PipelineError::InvalidStageConfig`] for another stage type; errors of
    /// [`EdStage::new`].
    pub fn construct(stage: &StageConfig, services: &StageBuildContext) -> Result<Box<dyn Stage>> {
        match stage {
            StageConfig::Ed(config) => Ok(Box::new(Self::new(config, services)?)),
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

impl Stage for EdStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<()> {
        let _ = ctx;
        todo!("worker C")
    }
}
