//! The [`Stage`] trait (Python `stages/base.py`).

use crate::context::PipelineContext;
use crate::error::Result;

/// One processing step of a pipeline.
///
/// Stages communicate only through [`PipelineContext::data`]: each reads the slots earlier
/// stages wrote and writes its own (the Python `data` hand-off is only ever the
/// `IndexingResult`, which lives in `rag_index_result`). Stages are immutable while running
/// (`&self`); state that must change across runs uses interior mutability.
pub trait Stage: Send + Sync {
    /// Stage instance name from the config (e.g. `rag_index`).
    fn name(&self) -> &str;

    /// Runs the stage on the context.
    ///
    /// # Errors
    /// Whatever the stage cannot recover from; the runner stops at the first error.
    fn run(&self, ctx: &mut PipelineContext) -> Result<()>;
}
