//! `RelationStage`: GLiNER2 relations per chunk, kept in the context only (Python
//! `stages/relation.py`).

use ce_config::pipeline::RelationStageConfig;
use ce_config::StageConfig;

use crate::context::{PipelineContext, StageBuildContext};
use crate::error::{PipelineError, Result};
use crate::stage::Stage;

/// The relation extraction stage.
#[derive(Debug)]
pub struct RelationStage {
    name: String,
}

impl RelationStage {
    /// Config `type` of this stage.
    pub const TYPE_NAME: &'static str = "RelationStage";

    /// Builds the stage from its config block and the shared services.
    ///
    /// # Errors
    /// Configuration errors.
    pub fn new(config: &RelationStageConfig, services: &StageBuildContext) -> Result<Self> {
        let _ = services;
        Ok(Self { name: config.name.clone() })
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
}

impl Stage for RelationStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<()> {
        let _ = ctx;
        todo!("worker C")
    }
}
