//! `EDStage`: resolves mentions to entities (Python `stages/ed.py`).
//!
//! Flow:
//! 1. Nothing to do (context unchanged) without `doc_mentions`.
//! 2. Input mentions, without types in `skip_ed_types` (`entity_types.yaml`):
//!    - with `coref_clusters` and `canonical_mention_for_cluster`: the canonical mention of each
//!      cluster plus every mention in no cluster (synthetic pronoun mentions excluded); when
//!      none remains the context is unchanged. `all_mentions_by_id` is written for alias
//!      collection;
//!    - otherwise every mention.
//! 3. Resolve with the [`EdResolver`] ([`CoreEdResolver`] by default). An offline run
//!    (`index_persisted == false`, Python: no `helix_client`) resolves nothing.
//! 4. Coreference path: each canonical decision is propagated to the cluster members (unless a
//!    member's type is incompatible with the canonical mention's) and then to synthetic pronoun
//!    mentions of resolved canonicals. Type compatibility uses the CoreferenceStage's
//!    `incompatible_type_groups` when configured (types in different groups are incompatible,
//!    ungrouped types compatible with all), else [`entity_types_compatible`].
//! 5. Writes `ed_results` and `resolved_mentions`; `role_map` from `defined_terms` when it
//!    binds a role; `SpanRefersToEntity` edges (confidence 0.85, method `NER+ED`) for resolved
//!    mentions with a mention span, when the index is persisted.
//!
//! The LLM is `<pipeline.name>_ollama`, else `ollama`, looked up only when `allow_llm` is set;
//! a missing or failing model disables the tie-break. Store errors while linking spans are
//! logged, except fatal ones, which fail the stage.

use std::sync::Arc;

use impeach_config::mappings::{types_compatible_in_groups, EntityTypes};
use impeach_config::pipeline::EdStageConfig;
use impeach_config::prompts::EdPrompt;
use impeach_config::{MappingsRegistry, StageConfig};
use impeach_core::{CorefClusterContext, EdResult, EntityId, Mention, MentionId, Metadata};
use impeach_ed::{entity_types_compatible, CoreEdResolver, EdResolver, Resolution, ResolveInput};
use impeach_models::{Llm, ModelError, ModelRegistry};
use impeach_ner::definitions::{DefinitionExtractor, RoleConfig};
use indexmap::{IndexMap, IndexSet};

use crate::context::{PipelineContext, StageBuildContext};
use crate::error::{PipelineError, Result};
use crate::stage::Stage;
use crate::stages::CoreferenceStage;

/// Confidence of a `SpanRefersToEntity` edge written by the stage.
pub const SPAN_LINK_CONFIDENCE: f64 = 0.85;

/// Registry key of the tie-break LLM when `<pipeline.name>_ollama` is not registered.
const FALLBACK_LLM_KEY: &str = "ollama";

/// Method of a `SpanRefersToEntity` edge written by the stage.
pub const SPAN_LINK_METHOD: &str = "NER+ED";

/// The entity disambiguation stage.
#[derive(Debug)]
pub struct EdStage {
    name: String,
    allow_llm: bool,
    pipeline_name: String,
    resolver: Box<dyn EdResolver>,
    entity_types: &'static EntityTypes,
    coref_type_groups: Option<Vec<IndexSet<String>>>,
    definitions: DefinitionExtractor,
}

/// Counts of one run, for the summary log line.
#[derive(Debug, Default)]
struct EdSummary {
    inputs: usize,
    created: usize,
    reused: usize,
    matched: usize,
    intra_doc: usize,
    unresolved: usize,
}

impl EdSummary {
    /// Counts from the entity creator's outcomes (created / reused) and the strategies of the
    /// other resolved results (intra-document matches vs rule, hybrid or LLM matches).
    fn new(inputs: usize, resolution: &Resolution) -> Self {
        let mut summary = Self { inputs, ..Self::default() };
        for (mention_id, result) in &resolution.results {
            if result.chosen_entity_id.is_none() {
                summary.unresolved += 1;
            } else if let Some(outcome) = resolution.outcomes.get(mention_id) {
                if outcome.is_new() {
                    summary.created += 1;
                } else {
                    summary.reused += 1;
                }
            } else if result.strategy_used.starts_with("intra_doc") {
                summary.intra_doc += 1;
            } else {
                summary.matched += 1;
            }
        }
        summary
    }
}

impl EdStage {
    /// Config `type` of this stage.
    pub const TYPE_NAME: &'static str = "EDStage";

    /// Builds the stage from its config block and the shared services: loads the mappings,
    /// the prompt file and the shipped `roles.yaml`.
    ///
    /// # Errors
    /// [`PipelineError::Config`] when the mappings, the prompt file or `roles.yaml` cannot be
    /// loaded.
    pub fn new(config: &EdStageConfig, services: &StageBuildContext) -> Result<Self> {
        let mappings = MappingsRegistry::load()?;
        let prompt = EdPrompt::load(&config.prompt_file)?;
        let roles = RoleConfig::load(None)?;
        let coref_type_groups = match services.config.get_stage_config(CoreferenceStage::TYPE_NAME)
        {
            Some(StageConfig::Coreference(coref)) if !coref.incompatible_type_groups.is_empty() => {
                Some(
                    coref
                        .incompatible_type_groups
                        .iter()
                        .map(|group| group.iter().cloned().collect())
                        .collect(),
                )
            }
            _ => None,
        };
        Ok(Self {
            name: config.name.clone(),
            allow_llm: config.allow_llm,
            pipeline_name: services.config.name().to_owned(),
            resolver: Box::new(CoreEdResolver::new(
                config,
                mappings,
                Some(prompt),
                Some(roles.clone()),
            )),
            entity_types: &mappings.entity_types,
            coref_type_groups,
            definitions: DefinitionExtractor::new(roles),
        })
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

    /// Replaces the resolver.
    #[must_use]
    pub fn with_resolver(mut self, resolver: Box<dyn EdResolver>) -> Self {
        self.resolver = resolver;
        self
    }

    /// The CoreferenceStage's `incompatible_type_groups` used for propagation, if configured.
    #[must_use]
    pub fn coref_type_groups(&self) -> Option<&[IndexSet<String>]> {
        self.coref_type_groups.as_deref()
    }

    /// The tie-break LLM: `<pipeline>_ollama`, else `ollama`; `None` when neither is
    /// registered or the model cannot be built.
    fn lookup_llm(&self, registry: &ModelRegistry) -> Option<Arc<dyn Llm>> {
        let prefixed = format!("{}_ollama", self.pipeline_name);
        let (key, found) = match registry.llm(&prefixed) {
            Err(ModelError::UnknownModel { .. }) => {
                tracing::debug!(key = prefixed.as_str(), "ED LLM not registered; trying `ollama`");
                (FALLBACK_LLM_KEY, registry.llm(FALLBACK_LLM_KEY))
            }
            other => (prefixed.as_str(), other),
        };
        match found {
            Ok(llm) => {
                tracing::info!(key, "ED LLM service loaded");
                Some(llm)
            }
            Err(ModelError::UnknownModel { .. }) => {
                tracing::debug!("ED LLM lookup found no model; tie-break disabled");
                None
            }
            Err(error) => {
                tracing::warn!(error = %error, "ED LLM unavailable; tie-break disabled");
                None
            }
        }
    }

    /// Runs the resolver, or resolves nothing for an offline run.
    fn resolve(
        &self,
        ctx: &PipelineContext,
        mentions: &[Mention],
        all_mentions_by_id: Option<&IndexMap<MentionId, Mention>>,
    ) -> Result<Resolution> {
        let data = &ctx.data;
        if !data.index_persisted {
            tracing::debug!("EDStage: index not persisted (offline run); no resolution");
            return Ok(Resolution::default());
        }
        let llm = if self.allow_llm { self.lookup_llm(&ctx.registry) } else { None };
        let embedder = data.embedder.as_ref().map(|e| Arc::clone(&e.0));
        let empty_metadata = Metadata::new();
        let input = ResolveInput {
            store: &ctx.store,
            embedder: embedder.as_ref(),
            llm: llm.as_ref(),
            chunks: data.rag_index_result.as_ref().map_or(&[], |r| r.chunks.as_slice()),
            doc_metadata: data.doc_metadata.as_ref().unwrap_or(&empty_metadata),
            coref_clusters: data.coref_clusters.as_deref(),
            canonical_for_cluster: data.canonical_mention_for_cluster.as_ref(),
            all_mentions_by_id,
        };
        Ok(self.resolver.resolve(mentions, &input)?)
    }

    /// Whether every member of `cluster` has a type compatible with the canonical mention.
    fn cluster_type_safe(
        &self,
        cluster: &CorefClusterContext,
        canonical: &Mention,
        mention_by_id: &IndexMap<MentionId, Mention>,
    ) -> bool {
        let canonical_type = canonical.entity_type.as_str();
        for member_id in &cluster.member_mention_ids {
            let Some(member) = mention_by_id.get(member_id) else {
                continue;
            };
            let member_type = member.entity_type.as_str();
            let compatible = match &self.coref_type_groups {
                Some(groups) => types_compatible_in_groups(canonical_type, member_type, groups),
                None => entity_types_compatible(
                    self.entity_types,
                    Some(canonical_type),
                    Some(member_type),
                ),
            };
            if !compatible {
                tracing::warn!(
                    cluster = cluster.id.as_str(),
                    canonical = %canonical.id,
                    canonical_type,
                    member = %member_id,
                    member_type,
                    "EDStage: type incompatibility in cluster"
                );
                return false;
            }
        }
        true
    }

    /// Writes `SpanRefersToEntity` edges; returns the number written.
    fn link_spans_to_entities(
        &self,
        ctx: &PipelineContext,
        resolved: &IndexMap<MentionId, EntityId>,
    ) -> Result<usize> {
        if !ctx.data.index_persisted {
            tracing::debug!("EDStage: index not persisted; skipping span-entity linking");
            return Ok(0);
        }
        let Some(span_ids) = ctx.data.mention_to_span_id.as_ref().filter(|m| !m.is_empty()) else {
            tracing::debug!("EDStage: no mention_to_span_id; skipping span-entity linking");
            return Ok(0);
        };
        if resolved.is_empty() {
            tracing::debug!("EDStage: no resolved mentions; skipping span-entity linking");
            return Ok(0);
        }
        let mut linked = 0;
        let mut skipped_no_span = 0;
        let mut errors: Vec<String> = Vec::new();
        for (mention_id, entity_id) in resolved {
            let Some(span_id) = span_ids.get(mention_id) else {
                skipped_no_span += 1;
                continue;
            };
            match ctx.store.link_span_to_entity(
                span_id,
                entity_id,
                SPAN_LINK_CONFIDENCE,
                SPAN_LINK_METHOD,
            ) {
                Ok(()) => linked += 1,
                Err(error) if error.is_fatal() => return Err(error.into()),
                Err(error) => errors.push(format!("mention {mention_id}: {error}")),
            }
        }
        tracing::info!(
            linked,
            skipped_no_span,
            errors = errors.len(),
            "EDStage: created SpanRefersToEntity edges"
        );
        if errors.len() <= 5 {
            for error in &errors {
                tracing::warn!("SpanRefersToEntity edge creation failed: {error}");
            }
        } else {
            tracing::warn!(
                "SpanRefersToEntity edge creation had {} errors (showing first 3): {:?}",
                errors.len(),
                &errors[..3]
            );
        }
        Ok(linked)
    }

    /// Writes the results, the role map and the span links, and logs the summary.
    fn finish(
        &self,
        ctx: &mut PipelineContext,
        inputs: usize,
        resolution: Resolution,
        resolved: IndexMap<MentionId, EntityId>,
    ) -> Result<()> {
        let summary = EdSummary::new(inputs, &resolution);
        let results = resolution.results;
        if let Some(defined_terms) = ctx.data.defined_terms.as_deref().filter(|t| !t.is_empty()) {
            let role_map = self.definitions.build_role_map(defined_terms, &resolved);
            if !role_map.is_empty() {
                tracing::info!(
                    bindings = role_map.len(),
                    roles = ?role_map.keys().collect::<Vec<_>>(),
                    "EDStage: built role map"
                );
                ctx.data.role_map = Some(role_map);
            }
        }
        let span_links = self.link_spans_to_entities(ctx, &resolved)?;
        let doc_id = ctx.data.doc_id.as_ref().map(ToString::to_string).unwrap_or_default();
        tracing::info!(
            doc_id = doc_id.as_str(),
            mentions = summary.inputs,
            results = results.len(),
            created = summary.created,
            reused = summary.reused,
            matched = summary.matched,
            intra_doc = summary.intra_doc,
            unresolved = summary.unresolved,
            resolved_mentions = resolved.len(),
            span_links,
            "EDStage summary"
        );
        ctx.data.ed_results = Some(results.into_values().collect());
        ctx.data.resolved_mentions = Some(resolved);
        Ok(())
    }
}

/// Mention id -> chosen entity id of the results with a chosen entity.
fn resolved_map(results: &IndexMap<MentionId, EdResult>) -> IndexMap<MentionId, EntityId> {
    results
        .iter()
        .filter_map(|(mention_id, result)| {
            result
                .chosen_entity_id
                .as_ref()
                .filter(|id| !id.as_str().is_empty())
                .map(|id| (mention_id.clone(), id.clone()))
        })
        .collect()
}

impl Stage for EdStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<()> {
        let all_mentions: Vec<Mention> = match ctx.data.doc_mentions.as_deref() {
            Some(mentions) if !mentions.is_empty() => mentions.to_vec(),
            _ => return Ok(()),
        };
        let skip_types = &self.entity_types.skip_ed_types;
        let has_coref = ctx.data.coref_clusters.as_ref().is_some_and(|c| !c.is_empty())
            && ctx.data.canonical_mention_for_cluster.as_ref().is_some_and(|c| !c.is_empty());

        if !has_coref {
            let inputs: Vec<Mention> = all_mentions
                .into_iter()
                .filter(|m| !skip_types.contains(m.entity_type.as_str()))
                .collect();
            let resolution = self.resolve(ctx, &inputs, ctx.data.all_mentions_by_id.as_ref())?;
            let resolved = resolved_map(&resolution.results);
            return self.finish(ctx, inputs.len(), resolution, resolved);
        }

        let clusters: Vec<CorefClusterContext> =
            ctx.data.coref_clusters.clone().unwrap_or_default();
        let canonical_for_cluster =
            ctx.data.canonical_mention_for_cluster.clone().unwrap_or_default();
        let synthetic = ctx.data.synthetic_pronoun_mentions.clone().unwrap_or_default();

        let mention_by_id: IndexMap<MentionId, Mention> =
            all_mentions.into_iter().map(|m| (m.id.clone(), m)).collect();
        let mut mention_to_cluster: IndexMap<&MentionId, &str> = IndexMap::new();
        let mut cluster_ids: IndexSet<&str> = IndexSet::new();
        for cluster in &clusters {
            cluster_ids.insert(cluster.id.as_str());
            for member in &cluster.member_mention_ids {
                mention_to_cluster.insert(member, cluster.id.as_str());
            }
        }

        let mut input_ids: Vec<&MentionId> = canonical_for_cluster
            .iter()
            .filter(|(cluster_id, canonical)| {
                mention_by_id.contains_key(*canonical) && cluster_ids.contains(cluster_id.as_str())
            })
            .map(|(_, canonical)| canonical)
            .collect();
        input_ids.extend(
            mention_by_id
                .keys()
                .filter(|id| !mention_to_cluster.contains_key(id) && !synthetic.contains_key(*id)),
        );
        let inputs: Vec<Mention> = input_ids
            .iter()
            .filter_map(|id| mention_by_id.get(*id))
            .filter(|m| !skip_types.contains(m.entity_type.as_str()))
            .cloned()
            .collect();
        if inputs.is_empty() {
            return Ok(());
        }

        let resolution = self.resolve(ctx, &inputs, Some(&mention_by_id))?;
        let mut resolved = resolved_map(&resolution.results);

        for cluster in &clusters {
            let Some(canonical_id) = canonical_for_cluster.get(&cluster.id) else {
                continue;
            };
            let Some(entity_id) = resolution
                .results
                .get(canonical_id)
                .and_then(|r| r.chosen_entity_id.as_ref())
                .filter(|id| !id.as_str().is_empty())
            else {
                continue;
            };
            if let Some(canonical) = mention_by_id.get(canonical_id) {
                if !self.cluster_type_safe(cluster, canonical, &mention_by_id) {
                    tracing::warn!(
                        cluster = cluster.id.as_str(),
                        "EDStage: skipping propagation due to type incompatibility"
                    );
                    continue;
                }
            }
            for member in &cluster.member_mention_ids {
                resolved.insert(member.clone(), entity_id.clone());
            }
        }

        let mut synthetic_propagated = 0;
        for (synthetic_id, canonical_id) in &synthetic {
            if let Some(entity_id) = resolved.get(canonical_id).cloned() {
                resolved.insert(synthetic_id.clone(), entity_id);
                synthetic_propagated += 1;
            }
        }
        if synthetic_propagated > 0 {
            tracing::info!(
                synthetic_propagated,
                "EDStage: propagated entity ids to synthetic pronoun mentions"
            );
        }

        ctx.data.all_mentions_by_id = Some(mention_by_id);
        self.finish(ctx, inputs.len(), resolution, resolved)
    }
}
