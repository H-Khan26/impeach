//! `ContradictionStage`: the contradiction check before facts are committed (Python
//! `stages/contradiction_check.py`).
//!
//! Reads `extracted_facts`, `resolved_mentions`, `ed_results`, `fact_helix_ids`, `doc_id`,
//! `case_id` and the context `embedder`; writes `contradiction_report`, `clean_facts` and
//! `committed_fact_ids`, and, with an enabled `review_store`, the SQLite review queue.
//!
//! Build: the mappings, the service config ([`ContradictionCheckConfig::from_stage`]), the
//! judge prompt when `verification.judge_model` is `qwen` (its `qwen_prompt_file` must be set
//! and load), and the SQLite review store when `review_store` is enabled with a non-empty
//! `db_path` (an initialisation failure fails the build when `fail_on_sqlite_error`, else
//! disables review writes).
//!
//! Run:
//! 1. No facts: an empty report for `doc_id`, no clean facts, nothing committed.
//! 2. Offline run (`index_persisted == false` or no context embedder; Python: no Helix client
//!    or embedding service): every fact is reported clean, nothing is committed.
//! 3. The judge, resolved once and cached for the stage's lifetime (an injected judge wins):
//!    `nli` mode uses the registry's `<pipeline>_<nli.model_name_suffix>` NLI model, else
//!    `<nli.model_name_suffix>`; `qwen` mode the `<pipeline>_ollama` LLM, else `ollama`, with
//!    the loaded prompt, falling back to NLI when no LLM is available. Without a judge every
//!    fact is reported clean (warning).
//! 4. [`ContradictionCheckService::check_facts_excluding`] with the entity names of the ED
//!    results ([`build_entity_name_map`]), the document's entity ids (unique
//!    `resolved_mentions` values in first-seen order) and, as exclusions, the facts the fact
//!    stage already committed (`commit_facts: true`; `fact_helix_ids`): they are not retrieved
//!    as candidates of their own document and not written twice (Python had no such guard and
//!    could compare a document's facts with themselves).
//! 5. Review rows for the report's unique pairs (only when there are contradiction results),
//!    with `PROCESSING` provenance, inserted under the `fail_on_sqlite_error` policy.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use ce_config::pipeline::{
    ContradictionCheckStageConfig, GatingStrategy, JudgeModel, VerificationStrategyConfig,
};
use ce_config::prompts::JudgePrompt;
use ce_config::services::ContradictionCheckConfig;
use ce_config::{ConfigError, MappingsRegistry, StageConfig};
use ce_contradictions::{
    build_entity_name_map, build_review_items_from_report, ContradictionCheckService,
    ContradictionJudge, LlmJudge, NliJudgeAdapter, ReviewProvenance,
};
use ce_core::{ContradictionReport, DocId, EntityId, ExtractedFact, FactId};
use ce_models::{Embedder, ModelRegistry};
use ce_store::{ReviewStoreConfig, SqliteReviewStore};
use indexmap::IndexSet;

use crate::context::{ContextData, PipelineContext, StageBuildContext};
use crate::error::{PipelineError, Result};
use crate::stage::Stage;

/// The contradiction check stage (config type `ContradictionCheckStage`).
#[derive(Debug)]
pub struct ContradictionStage {
    name: String,
    config: ContradictionCheckStageConfig,
    check_config: ContradictionCheckConfig,
    pipeline_name: String,
    mappings: &'static MappingsRegistry,
    /// Loaded judge prompt (`qwen` mode only).
    judge_prompt: Option<JudgePrompt>,
    /// Review store, when enabled and initialised.
    review_store: Option<SqliteReviewStore>,
    judge: OnceLock<Arc<dyn ContradictionJudge>>,
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
        let mappings = MappingsRegistry::load()?;
        let judge_prompt = match &config.verification {
            Some(verification) if verification.judge_model == JudgeModel::Qwen => {
                let path = verification.qwen_prompt_file.as_deref().ok_or_else(|| {
                    ConfigError::validation(
                        format!("stages.{}.verification.qwen_prompt_file", config.name),
                        "verification.judge_model='qwen' requires verification.qwen_prompt_file",
                    )
                })?;
                Some(JudgePrompt::load(path)?)
            }
            _ => None,
        };
        let review_store = match &config.review_store {
            Some(review) if review.enabled && !review.db_path.is_empty() => {
                SqliteReviewStore::open(&ReviewStoreConfig {
                    db_path: PathBuf::from(&review.db_path),
                    fail_on_sqlite_error: review.fail_on_sqlite_error,
                })?
            }
            _ => None,
        };
        Ok(Self {
            name: config.name.clone(),
            config: config.clone(),
            check_config: ContradictionCheckConfig::from_stage(config),
            pipeline_name: services.config.name().to_owned(),
            mappings,
            judge_prompt,
            review_store,
            judge: OnceLock::new(),
        })
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
    pub fn with_judge(mut self, judge: Arc<dyn ContradictionJudge>) -> Self {
        self.judge = OnceLock::from(judge);
        self
    }

    /// Whether review rows are written (an enabled and initialised review store).
    #[must_use]
    pub fn review_store_enabled(&self) -> bool {
        self.review_store.is_some()
    }

    fn verification(&self) -> Option<&VerificationStrategyConfig> {
        self.config.verification.as_ref()
    }

    /// The injected or cached judge, else one resolved from the registry (module docs);
    /// `None` (logged) when no model is available.
    fn judge(&self, registry: &ModelRegistry) -> Option<Arc<dyn ContradictionJudge>> {
        if let Some(judge) = self.judge.get() {
            return Some(Arc::clone(judge));
        }
        let resolved = match (self.verification(), &self.judge_prompt) {
            (Some(verification), Some(prompt)) if verification.judge_model == JudgeModel::Qwen => {
                self.llm_judge(registry, prompt, verification.qwen_batch_size).or_else(|| {
                    tracing::warn!(
                        "ContradictionCheckStage: Qwen judge requested but Ollama service not \
                         available. Falling back to NLI."
                    );
                    self.nli_judge(registry)
                })
            }
            _ => self.nli_judge(registry),
        }?;
        tracing::info!(judge = resolved.name(), "ContradictionCheckStage: judge ready");
        Some(Arc::clone(self.judge.get_or_init(|| resolved)))
    }

    fn nli_judge(&self, registry: &ModelRegistry) -> Option<Arc<dyn ContradictionJudge>> {
        let suffix = self.config.nli.model_name_suffix.as_str();
        let key = format!("{}_{suffix}", self.pipeline_name);
        let nli = registry.nli(&key).or_else(|error| {
            registry.nli(suffix).map_err(|_| {
                tracing::warn!(
                    key = key.as_str(),
                    error = %error,
                    "ContradictionCheckStage: NLI service not found in registry"
                );
            })
        });
        nli.ok().map(|nli| Arc::new(NliJudgeAdapter::new(nli)) as Arc<dyn ContradictionJudge>)
    }

    fn llm_judge(
        &self,
        registry: &ModelRegistry,
        prompt: &JudgePrompt,
        batch_size: usize,
    ) -> Option<Arc<dyn ContradictionJudge>> {
        let key = format!("{}_ollama", self.pipeline_name);
        let llm = registry.llm(&key).or_else(|error| {
            registry.llm("ollama").map_err(|_| {
                tracing::warn!(
                    key = key.as_str(),
                    error = %error,
                    "ContradictionCheckStage: Ollama service not found in registry"
                );
            })
        });
        llm.ok().map(|llm| {
            Arc::new(LlmJudge::new(llm, prompt.clone(), batch_size)) as Arc<dyn ContradictionJudge>
        })
    }

    /// Writes review rows for the report's pairs (module docs).
    fn write_review_items(&self, report: &ContradictionReport, data: &ContextData) -> Result<()> {
        let Some(store) = &self.review_store else {
            return Ok(());
        };
        if report.contradiction_results.is_empty() {
            return Ok(());
        }
        let doc_id = if report.doc_id.is_empty() {
            data.doc_id.as_ref().map(DocId::to_string).unwrap_or_default()
        } else {
            report.doc_id.to_string()
        };
        let case_id =
            data.case_id.clone().filter(|case_id| !case_id.is_empty()).unwrap_or(doc_id.clone());
        let provenance = ReviewProvenance {
            case_id,
            session_id: doc_id.clone(),
            segment_id: doc_id.clone(),
            segment_index: 0,
            doc_id: doc_id.clone(),
            stage: "PROCESSING".to_owned(),
            witness_name: None,
            source_stream: Some("processing_pipeline".to_owned()),
            segment_start_ts: None,
            segment_end_ts: None,
            source_pipeline: Some(self.pipeline_name.clone()),
        };
        let mut items = build_review_items_from_report(report, &provenance, None);
        if items.is_empty() {
            return Ok(());
        }
        let inserted = store.try_insert_review_items(&mut items)?;
        tracing::info!(
            "ContradictionCheckStage: inserted {inserted} review row(s) into SQLite for doc {doc_id}"
        );
        Ok(())
    }
}

/// Unique `resolved_mentions` values, in first-seen order.
fn document_entity_ids(data: &ContextData) -> Vec<EntityId> {
    data.resolved_mentions
        .as_ref()
        .map(|resolved| resolved.values().cloned().collect::<IndexSet<EntityId>>())
        .unwrap_or_default()
        .into_iter()
        .collect()
}

/// Graph ids of the facts the fact stage already committed.
fn already_committed(data: &ContextData) -> Vec<FactId> {
    data.fact_helix_ids.as_deref().unwrap_or_default().iter().flatten().cloned().collect()
}

/// Writes the stage outputs.
fn write_outputs(data: &mut ContextData, report: ContradictionReport) {
    data.clean_facts = Some(report.clean_facts.clone());
    data.committed_fact_ids = Some(report.committed_fact_ids.clone());
    data.contradiction_report = Some(report);
}

/// A report that lists every fact as clean (no check was possible).
fn all_clean_report(doc_id: DocId, facts: Vec<ExtractedFact>) -> ContradictionReport {
    let mut report = ContradictionReport::new(doc_id);
    report.total_facts_extracted = facts.len();
    report.clean_facts = facts;
    report
}

impl Stage for ContradictionStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<()> {
        let doc_id = ctx.data.doc_id.clone().unwrap_or_default();
        let facts = ctx.data.extracted_facts.clone().unwrap_or_default();
        if facts.is_empty() {
            tracing::info!("ContradictionCheckStage: No facts to check, skipping");
            write_outputs(&mut ctx.data, ContradictionReport::new(doc_id));
            return Ok(());
        }

        let embedder: Arc<dyn Embedder> = match &ctx.data.embedder {
            Some(embedder) if ctx.data.index_persisted => Arc::clone(&embedder.0),
            _ => {
                tracing::warn!(
                    "ContradictionCheckStage: index not persisted or no embedder (offline run). \
                     Treating all facts as clean (no contradiction check)."
                );
                write_outputs(&mut ctx.data, all_clean_report(doc_id, facts));
                return Ok(());
            }
        };

        let verification = self.verification();
        tracing::info!(
            "ContradictionCheckStage: verification judge={} gating={}",
            verification.map_or("nli", |v| match v.judge_model {
                JudgeModel::Nli => "nli",
                JudgeModel::Qwen => "qwen",
            }),
            verification.map_or("dual_path", |v| match v.gating_strategy {
                GatingStrategy::DualPath => "dual_path",
                GatingStrategy::Permissive => "permissive",
            })
        );
        let existing = already_committed(&ctx.data);
        let Some(judge) = self.judge(&ctx.registry) else {
            tracing::warn!(
                "ContradictionCheckStage: no judge model available. Treating all facts as clean \
                 (no contradiction check)."
            );
            let mut report = all_clean_report(doc_id, facts);
            report.committed_fact_ids = existing;
            write_outputs(&mut ctx.data, report);
            return Ok(());
        };

        let entity_names =
            build_entity_name_map(ctx.data.ed_results.as_deref().unwrap_or_default());
        tracing::debug!(
            "ContradictionCheckStage: Using {} entity names for NLI context enrichment",
            entity_names.len()
        );
        let entity_ids = document_entity_ids(&ctx.data);
        let service = ContradictionCheckService::new(
            Arc::clone(&ctx.store),
            embedder,
            judge,
            self.check_config.clone(),
            self.config.verification.clone(),
            self.mappings,
        )
        .with_entity_names(entity_names);
        let report = service.check_facts_excluding(&facts, &entity_ids, &doc_id, &existing)?;

        if report.has_contradictions() {
            tracing::warn!(
                "ContradictionCheckStage: Found {} contradictions and {} facts needing review in \
                 doc {doc_id}",
                report.contradiction_count(),
                report.needs_review_count()
            );
        } else {
            tracing::info!(
                "ContradictionCheckStage: All {} facts are clean for doc {doc_id}",
                report.clean_facts.len()
            );
        }
        let review = self.write_review_items(&report, &ctx.data);
        write_outputs(&mut ctx.data, report);
        review
    }
}
