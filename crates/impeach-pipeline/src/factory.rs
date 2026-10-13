//! Building pipelines from config (Python `pipeline/factory.py` and the stage registry of
//! `stages/__init__.py`).
//!
//! A [`StageRegistry`] maps stage `type` names to constructors of the stable signature
//! [`StageConstructor`]. [`build_pipeline`] walks `config.stages` in order:
//! - a disabled stage (`enabled: false`) is skipped without being constructed (as in Python);
//! - an enabled stage whose type has a constructor is built and appended;
//! - an enabled stage whose type has no constructor follows [`UnregisteredStages`]:
//!   [`UnregisteredStages::Error`] fails with [`PipelineError::UnknownStage`] listing the
//!   registered types, [`UnregisteredStages::Skip`] logs a warning and leaves it out.
//!
//! [`StageRegistry::builtin`] registers all eight stage types of the config schema, so with
//! it every enabled stage is built. "Unregistered" stage types arise with a custom registry
//! ([`StageRegistry::empty`] plus selected constructors, as tests use); the default is
//! [`UnregisteredStages::Error`] and [`UnregisteredStages::Skip`] opts into leaving them out.

use std::fmt;
use std::sync::Arc;

use impeach_config::pipeline::{CorefBackend, JudgeModel};
use impeach_config::{PipelineConfig, StageConfig};
use impeach_models::{ModelError, ModelRegistry};
use impeach_store::GraphStore;
use indexmap::IndexMap;

use crate::context::StageBuildContext;
use crate::error::{PipelineError, Result};
use crate::runner::Pipeline;
use crate::stage::Stage;
use crate::stages::{
    ContradictionStage, CoreferenceStage, DefinitionStage, EdStage, FactStage, MentionStage,
    RagIndexStage, RelationStage,
};

/// Constructor of one stage type: receives the stage's config block and the shared services.
pub type StageConstructor =
    dyn Fn(&StageConfig, &StageBuildContext) -> Result<Box<dyn Stage>> + Send + Sync;

/// What [`StageRegistry::build`] does with an enabled stage whose type has no constructor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum UnregisteredStages {
    /// Fail with [`PipelineError::UnknownStage`].
    #[default]
    Error,
    /// Leave the stage out and log a warning.
    Skip,
}

/// Stage type name -> constructor, in registration order.
#[derive(Clone, Default)]
pub struct StageRegistry {
    constructors: IndexMap<String, Arc<StageConstructor>>,
}

impl fmt::Debug for StageRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StageRegistry").field("types", &self.type_names()).finish()
    }
}

impl StageRegistry {
    /// A registry without constructors.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Every stage: `RAGIndexStage`, `MentionStage`, `DefinitionStage`, `CoreferenceStage`,
    /// `EDStage`, `RelationStage`, `FactStage`, `ContradictionCheckStage`.
    #[must_use]
    pub fn builtin() -> Self {
        let mut registry = Self::empty();
        registry
            .register(RagIndexStage::TYPE_NAME, RagIndexStage::construct)
            .register(MentionStage::TYPE_NAME, MentionStage::construct)
            .register(DefinitionStage::TYPE_NAME, DefinitionStage::construct)
            .register(CoreferenceStage::TYPE_NAME, CoreferenceStage::construct)
            .register(EdStage::TYPE_NAME, EdStage::construct)
            .register(RelationStage::TYPE_NAME, RelationStage::construct)
            .register(FactStage::TYPE_NAME, FactStage::construct)
            .register(ContradictionStage::TYPE_NAME, ContradictionStage::construct);
        registry
    }

    /// Registers (or replaces) the constructor of `type_name`.
    pub fn register<F>(&mut self, type_name: impl Into<String>, constructor: F) -> &mut Self
    where
        F: Fn(&StageConfig, &StageBuildContext) -> Result<Box<dyn Stage>> + Send + Sync + 'static,
    {
        self.constructors.insert(type_name.into(), Arc::new(constructor));
        self
    }

    /// Builder form of [`StageRegistry::register`].
    #[must_use]
    pub fn with<F>(mut self, type_name: impl Into<String>, constructor: F) -> Self
    where
        F: Fn(&StageConfig, &StageBuildContext) -> Result<Box<dyn Stage>> + Send + Sync + 'static,
    {
        self.register(type_name, constructor);
        self
    }

    /// Whether `type_name` has a constructor.
    #[must_use]
    pub fn is_registered(&self, type_name: &str) -> bool {
        self.constructors.contains_key(type_name)
    }

    /// Registered type names, in registration order.
    #[must_use]
    pub fn type_names(&self) -> Vec<String> {
        self.constructors.keys().cloned().collect()
    }

    /// Builds one stage, whether or not it is enabled.
    ///
    /// # Errors
    /// [`PipelineError::UnknownStage`] when the type has no constructor; constructor errors.
    pub fn build_stage(
        &self,
        stage: &StageConfig,
        services: &StageBuildContext,
    ) -> Result<Box<dyn Stage>> {
        let type_name = stage.type_name();
        let constructor =
            self.constructors.get(type_name).ok_or_else(|| PipelineError::UnknownStage {
                stage_type: type_name.to_owned(),
                known: self.type_names(),
            })?;
        constructor(stage, services)
    }

    /// Builds the enabled stages of `services.config` in config order (rules in the module
    /// docs).
    ///
    /// # Errors
    /// As [`StageRegistry::build_stage`]; unregistered types only with
    /// [`UnregisteredStages::Error`].
    pub fn build_stages(
        &self,
        services: &StageBuildContext,
        unregistered: UnregisteredStages,
    ) -> Result<Vec<Box<dyn Stage>>> {
        let mut stages = Vec::new();
        for stage in &services.config.stages {
            if !stage.enabled() {
                tracing::debug!(stage = stage.name(), "stage disabled; skipped");
                continue;
            }
            if unregistered == UnregisteredStages::Skip && !self.is_registered(stage.type_name()) {
                tracing::warn!(
                    stage = stage.name(),
                    stage_type = stage.type_name(),
                    "stage type has no registered constructor; skipped"
                );
                continue;
            }
            stages.push(self.build_stage(stage, services)?);
        }
        Ok(stages)
    }

    /// Builds a [`Pipeline`] from the enabled stages of `services.config`.
    ///
    /// # Errors
    /// As [`StageRegistry::build_stages`].
    pub fn build(
        &self,
        services: StageBuildContext,
        unregistered: UnregisteredStages,
    ) -> Result<Pipeline> {
        let stages = self.build_stages(&services, unregistered)?;
        probe_llm(&services);
        Ok(Pipeline::new(stages, services))
    }
}

/// Checks once that the `<pipeline>_ollama` server answers when an enabled stage may call it, so
/// an unreachable server is reported (with how to fix it) before any document is processed.
/// The stages still get a client; their calls fail with a short connection error.
fn probe_llm(services: &StageBuildContext) {
    let config = &services.config;
    let stage_needs_llm = |stage: &StageConfig| match stage {
        StageConfig::Fact(_) => true,
        StageConfig::Ed(ed) => ed.allow_llm,
        StageConfig::ContradictionCheck(check) => {
            check.verification.as_ref().is_some_and(|v| v.judge_model == JudgeModel::Qwen)
        }
        _ => false,
    };
    let needs_llm = config.stages.iter().any(|stage| stage.enabled() && stage_needs_llm(stage))
        || config.models.coref.backend == CorefBackend::Llm;
    if !needs_llm {
        return;
    }
    let key = format!("{}_ollama", config.name());
    match services.registry.check_llm_reachable(&key) {
        Ok(_) | Err(ModelError::UnknownModel { .. }) => {}
        Err(error) => tracing::warn!("{error}"),
    }
}

/// Builds the pipeline of `config` with [`StageRegistry::builtin`] (every stage type is
/// registered; [`UnregisteredStages::Error`]).
///
/// # Errors
/// Stage constructor errors.
pub fn build_pipeline(
    config: Arc<PipelineConfig>,
    registry: Arc<ModelRegistry>,
    store: Arc<dyn GraphStore>,
) -> Result<Pipeline> {
    StageRegistry::builtin()
        .build(StageBuildContext::new(config, registry, store), UnregisteredStages::Error)
}
