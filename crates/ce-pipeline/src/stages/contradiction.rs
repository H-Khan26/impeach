//! `ContradictionStage`: the contradiction check before facts are committed (Python
//! `stages/contradiction_check.py`).
//!
//! Reads `extracted_facts`, `resolved_mentions`, `ed_results`, `doc_id`, `case_id` and the
//! context `embedder`; writes `contradiction_report`, `clean_facts` and `committed_fact_ids`,
//! and, with an enabled `review_store`, the SQLite review queue.

use std::sync::Arc;

use ce_config::pipeline::ContradictionCheckStageConfig;
use ce_config::StageConfig;
use ce_contradictions::ContradictionJudge;

use crate::context::{PipelineContext, StageBuildContext};
use crate::error::{PipelineError, Result};
use crate::stage::Stage;

/// The contradiction check stage (config type `ContradictionCheckStage`).
#[derive(Debug)]
pub struct ContradictionStage {
    name: String,
    config: ContradictionCheckStageConfig,
}

impl ContradictionStage {
    /// Config `type` of this stage.
    pub const TYPE_NAME: &'static str = "ContradictionCheckStage";

    /// Builds the stage from its config block and the shared services.
    ///
    /// # Errors
    /// [`PipelineError::Config`] when the mappings or the judge prompt cannot be loaded;
    /// review-store initialisation errors when `fail_on_sqlite_error` is set.
    pub fn new(
        config: &ContradictionCheckStageConfig,
        services: &StageBuildContext,
    ) -> Result<Self> {
        todo!("worker C")
    }

    /// [`crate::StageConstructor`] of this stage.
    ///
    /// # Errors
    /// [`PipelineError::InvalidStageConfig`] for another stage type; errors of
    /// [`ContradictionStage::new`].
    pub fn construct(stage: &StageConfig, services: &StageBuildContext) -> Result<Box<dyn Stage>> {
        match stage {
            StageConfig::ContradictionCheck(config) => Ok(Box::new(Self::new(config, services)?)),
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

    /// Replaces the judge resolved from the model registry.
    #[must_use]
    pub fn with_judge(self, judge: Arc<dyn ContradictionJudge>) -> Self {
        todo!("worker C")
    }
}

impl Stage for ContradictionStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<()> {
        todo!("worker C")
    }
}
