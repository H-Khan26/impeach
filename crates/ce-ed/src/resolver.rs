//! The core ED resolver (Python `services/ed/resolver.py`, `interfaces.py`): intra-document
//! matching, the ED service per mention, entity creation and coreference alias propagation.
//!
//! Per mention (in input order): build the [`ce_core::EdContext`] (mentions outside the
//! indexed chunks are skipped); try the [`IntraDocumentMatcher`] (`intra_doc_exact`,
//! `intra_doc_fuzzy`, `intra_doc_alias`, confidence 0.85); else run
//! [`EdService::resolve_span`] and, when unresolved and `create_if_missing` is set,
//! [`EntityCreator::create_entity_from_ed_result`] with the coreference cluster aliases of a
//! canonical mention. Every resolution is registered with the matcher together with the
//! cluster aliases.
//!
//! Entity type compatibility uses [`entity_types_compatible`] everywhere (Python's alias step
//! used a group-membership rule that called ungrouped types incompatible).

use std::sync::Arc;

use ce_config::mappings::EntityTypes;
use ce_config::pipeline::EdStageConfig;
use ce_config::prompts::EdPrompt;
use ce_config::MappingsRegistry;
use ce_core::{
    ChunkWriteResult, CorefClusterContext, EdResult, EntityId, Mention, MentionId, Metadata,
};
use ce_models::{Embedder, Llm};
use ce_ner::definitions::RoleConfig;
use ce_store::GraphStore;
use ce_text::normalize::py_strip;
use indexmap::{IndexMap, IndexSet};

use crate::context_builder::ContextBuilder;
use crate::entity_creator::{EntityCreator, EntityOutcome};
use crate::error::Result;
use crate::name_matching::{entity_types_compatible, names_match, normalize_for_matching};
use crate::name_similarity::NameSimilarity;
use crate::retrieval::{EdRetriever, RetrieverSettings};
use crate::service::{EdService, EdServiceSettings};
use crate::structured_keys::KeyPatterns;

/// Confidence of an intra-document match.
pub const INTRA_DOC_CONFIDENCE: f64 = 0.85;

/// Entities resolved earlier in the same document run, for matching later mentions without
/// the store (Python `IntraDocumentMatcher`).
///
/// Maps keep insertion order (Python dicts; the alias sets were Python sets, iterated here in
/// insertion order).
#[derive(Debug)]
pub struct IntraDocumentMatcher<'a> {
    resolved: IndexMap<String, (EntityId, String, String)>,
    aliases: IndexMap<EntityId, IndexSet<String>>,
    entity_creator: Option<&'a EntityCreator>,
    entity_types: &'a EntityTypes,
}

impl<'a> IntraDocumentMatcher<'a> {
    /// An empty matcher; role-term guards apply only with an entity creator.
    #[must_use]
    pub fn new(entity_creator: Option<&'a EntityCreator>, entity_types: &'a EntityTypes) -> Self {
        Self { resolved: IndexMap::new(), aliases: IndexMap::new(), entity_creator, entity_types }
    }

    fn is_role_term(&self, name: &str) -> bool {
        self.entity_creator.is_some_and(|c| c.is_role_term(name))
    }

    /// Registers `name` (of `entity_type`) as resolved to `entity_id`.
    pub fn register_resolution(&mut self, name: &str, entity_type: &str, entity_id: &EntityId) {
        let normalized = normalize_for_matching(name);
        if normalized.is_empty() {
            return;
        }
        self.resolved.insert(
            normalized.clone(),
            (entity_id.clone(), name.to_owned(), entity_type.to_owned()),
        );
        self.aliases.entry(entity_id.clone()).or_default().insert(normalized);
    }

    /// The entity of a previously resolved name matching `mention`, with the strategy:
    /// `intra_doc_exact` (equal normalised name, compatible type), `intra_doc_fuzzy` (same
    /// type, [`names_match`]) or `intra_doc_alias` (compatible type, a registered alias
    /// matches; role-term aliases only exactly). Role-named entities never match a real name.
    #[must_use]
    pub fn find_match(&self, mention: &Mention) -> Option<(EntityId, &'static str)> {
        let normalized = normalize_for_matching(&mention.name);
        if normalized.is_empty() {
            return None;
        }
        let searching_for_real_name =
            self.entity_creator.is_some() && !self.is_role_term(&mention.name);

        if let Some((entity_id, original_name, resolved_type)) = self.resolved.get(&normalized) {
            if !entity_types_compatible(
                self.entity_types,
                Some(&mention.entity_type),
                Some(resolved_type),
            ) {
                tracing::debug!(
                    mention = mention.name.as_str(),
                    mention_type = mention.entity_type.as_str(),
                    resolved = original_name.as_str(),
                    resolved_type = resolved_type.as_str(),
                    "skipping intra-doc exact match: incompatible types"
                );
            } else if searching_for_real_name && self.is_role_term(original_name) {
                tracing::debug!(
                    mention = mention.name.as_str(),
                    resolved = original_name.as_str(),
                    "skipping intra-doc match: role term"
                );
            } else {
                return Some((entity_id.clone(), "intra_doc_exact"));
            }
        }

        for (entity_id, original_name, entity_type) in self.resolved.values() {
            if mention.entity_type != *entity_type {
                continue;
            }
            if searching_for_real_name && self.is_role_term(original_name) {
                continue;
            }
            if names_match(&mention.name, original_name) {
                return Some((entity_id.clone(), "intra_doc_fuzzy"));
            }
        }

        for (entity_id, aliases) in &self.aliases {
            let canonical = self.resolved.values().find(|(id, _, _)| id == entity_id);
            let canonical_name = canonical.map(|(_, name, _)| name.as_str());
            let entity_type = canonical.map(|(_, _, t)| t.as_str());
            if !entity_types_compatible(self.entity_types, Some(&mention.entity_type), entity_type)
            {
                tracing::debug!(
                    mention = mention.name.as_str(),
                    mention_type = mention.entity_type.as_str(),
                    entity = %entity_id,
                    entity_type = ?entity_type,
                    "skipping intra-doc alias match: incompatible types"
                );
                continue;
            }
            if searching_for_real_name && canonical_name.is_some_and(|n| self.is_role_term(n)) {
                continue;
            }
            for alias in aliases {
                if self.is_role_term(alias) && normalized != normalize_for_matching(alias) {
                    tracing::debug!(
                        alias = alias.as_str(),
                        mention = mention.name.as_str(),
                        "skipping role alias: not an exact match"
                    );
                    continue;
                }
                if names_match(&normalized, alias) {
                    return Some((entity_id.clone(), "intra_doc_alias"));
                }
            }
        }
        None
    }
}

/// Surface texts of the other members of the cluster whose canonical mention is
/// `mention_id` (stripped; those equal to the canonical name, ignoring case, left out), for use
/// as entity aliases. Empty when the mention is not canonical or its members are not in
/// `mention_by_id`.
#[must_use]
pub fn collect_cluster_aliases(
    mention_id: &MentionId,
    coref_clusters: Option<&[CorefClusterContext]>,
    canonical_for_cluster: Option<&IndexMap<String, MentionId>>,
    mention_by_id: &IndexMap<MentionId, Mention>,
) -> Vec<String> {
    let (Some(clusters), Some(canonical_for_cluster)) = (coref_clusters, canonical_for_cluster)
    else {
        return Vec::new();
    };
    if clusters.is_empty() || canonical_for_cluster.is_empty() {
        return Vec::new();
    }
    let Some(cluster_id) = canonical_for_cluster
        .iter()
        .find(|(_, canonical)| *canonical == mention_id)
        .map(|(cluster_id, _)| cluster_id)
        .filter(|cluster_id| !cluster_id.is_empty())
    else {
        return Vec::new();
    };
    let Some(cluster) = clusters.iter().find(|c| &c.id == cluster_id) else {
        return Vec::new();
    };
    let canonical_text =
        mention_by_id.get(mention_id).map(|m| m.name.to_lowercase()).unwrap_or_default();
    cluster
        .member_mention_ids
        .iter()
        .filter(|mid| *mid != mention_id)
        .filter_map(|mid| mention_by_id.get(mid))
        .filter(|member| !member.name.is_empty())
        .map(|member| py_strip(&member.name))
        .filter(|name| name.to_lowercase() != canonical_text)
        .map(str::to_owned)
        .collect()
}

/// Per-run inputs of [`EdResolver::resolve`] (the Python context keys it reads).
#[derive(Clone, Copy)]
pub struct ResolveInput<'a> {
    /// Graph store holding the entities.
    pub store: &'a Arc<dyn GraphStore>,
    /// Embedder for vector retrieval and entity embeddings; `None` disables both.
    pub embedder: Option<&'a Arc<dyn Embedder>>,
    /// LLM for the tie-break; `None` disables it.
    pub llm: Option<&'a Arc<dyn Llm>>,
    /// Chunks written by the RAG index stage.
    pub chunks: &'a [ChunkWriteResult],
    /// Document metadata.
    pub doc_metadata: &'a Metadata,
    /// Coreference clusters.
    pub coref_clusters: Option<&'a [CorefClusterContext]>,
    /// Cluster id -> canonical mention id.
    pub canonical_for_cluster: Option<&'a IndexMap<String, MentionId>>,
    /// Every mention of the document by id (Python `_all_mentions_by_id`); the resolved
    /// mentions when `None`.
    pub all_mentions_by_id: Option<&'a IndexMap<MentionId, Mention>>,
}

impl std::fmt::Debug for ResolveInput<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolveInput")
            .field("chunks", &self.chunks.len())
            .field("has_embedder", &self.embedder.is_some())
            .field("has_llm", &self.llm.is_some())
            .finish_non_exhaustive()
    }
}

/// Output of [`EdResolver::resolve`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Resolution {
    /// Mention id -> result for the mentions that could be resolved or attempted, in input
    /// order.
    pub results: IndexMap<MentionId, EdResult>,
    /// Mention id -> entity created or reused for it by the entity creator.
    pub outcomes: IndexMap<MentionId, EntityOutcome>,
}

/// Resolves a document's mentions to entities (Python `EDResolver` protocol of
/// `interfaces.py`).
pub trait EdResolver: Send + Sync + std::fmt::Debug {
    /// Results per mention and the entity creator's outcomes.
    ///
    /// # Errors
    /// Fatal store errors; prompt errors of the LLM tie-break.
    fn resolve(&self, mentions: &[Mention], input: &ResolveInput<'_>) -> Result<Resolution>;
}

impl EdResolver for CoreEdResolver {
    fn resolve(&self, mentions: &[Mention], input: &ResolveInput<'_>) -> Result<Resolution> {
        CoreEdResolver::resolve(self, mentions, input)
    }
}

/// The default ED resolver (Python `CoreEDResolver`): settings of an `EDStage` block plus the
/// assets loaded once (mappings, prompt, roles).
#[derive(Debug)]
pub struct CoreEdResolver {
    config: EdStageConfig,
    similarity: NameSimilarity,
    patterns: KeyPatterns,
    prompt: Option<EdPrompt>,
    role_config: Option<RoleConfig>,
}

impl CoreEdResolver {
    /// A resolver for `config` over `mappings`; `prompt` is required for LLM tie-breaks,
    /// `role_config` enables role blocking.
    #[must_use]
    pub fn new(
        config: &EdStageConfig,
        mappings: &MappingsRegistry,
        prompt: Option<EdPrompt>,
        role_config: Option<RoleConfig>,
    ) -> Self {
        Self {
            config: config.clone(),
            similarity: NameSimilarity::from_mappings(mappings),
            patterns: KeyPatterns::from_mappings(&mappings.detection_patterns),
            prompt,
            role_config,
        }
    }

    /// The stage settings.
    #[must_use]
    pub fn config(&self) -> &EdStageConfig {
        &self.config
    }

    /// The entity type mappings.
    #[must_use]
    pub fn entity_types(&self) -> &EntityTypes {
        self.similarity.entity_types()
    }

    /// Resolves `mentions`: results by mention id in input order (mentions without a context
    /// have none) and the entity creator's outcomes. Every created entity drops the
    /// `all_entities_fuzzy` caches so later mentions of the document can reuse it.
    ///
    /// # Errors
    /// Fatal store errors; prompt errors of the LLM tie-break.
    pub fn resolve(&self, mentions: &[Mention], input: &ResolveInput<'_>) -> Result<Resolution> {
        let config = &self.config;
        let entity_types = self.similarity.entity_types();
        let retriever = EdRetriever::new(
            Arc::clone(input.store),
            input.embedder.cloned(),
            RetrieverSettings::from_config(config),
            self.similarity.clone(),
        );
        let service = EdService::new(
            &retriever,
            EdServiceSettings::from_config(config),
            &self.similarity,
            &self.patterns,
        )
        .with_llm(input.llm.cloned())
        .with_prompt(self.prompt.as_ref());

        let entity_creator = config.create_if_missing.then(|| {
            EntityCreator::new(
                Arc::clone(input.store),
                input.embedder.cloned(),
                self.similarity.clone(),
                self.role_config.clone(),
            )
            .with_reuse_similarity_threshold(config.reuse_similarity_threshold)
            .with_retrieval_strategy(config.entity_retrieval_strategy)
        });

        let context_builder = ContextBuilder::new(input.chunks, input.doc_metadata, entity_types);
        let own_mentions: IndexMap<MentionId, Mention>;
        let mention_by_id = match input.all_mentions_by_id.filter(|m| !m.is_empty()) {
            Some(all) => all,
            None => {
                own_mentions = mentions.iter().map(|m| (m.id.clone(), m.clone())).collect();
                &own_mentions
            }
        };
        let aliases_of = |mention_id: &MentionId| {
            collect_cluster_aliases(
                mention_id,
                input.coref_clusters,
                input.canonical_for_cluster,
                mention_by_id,
            )
        };

        let mut matcher = config
            .enable_intra_doc_matching
            .then(|| IntraDocumentMatcher::new(entity_creator.as_ref(), entity_types));
        let mut resolution = Resolution::default();

        for mention in mentions {
            let Some(ed_ctx) = context_builder.build_context(mention) else {
                continue;
            };

            if let Some((entity_id, strategy)) =
                matcher.as_ref().and_then(|m| m.find_match(mention))
            {
                tracing::debug!(
                    mention = mention.name.as_str(),
                    entity = %entity_id,
                    strategy,
                    "intra-doc match"
                );
                resolution.results.insert(
                    mention.id.clone(),
                    EdResult {
                        mention_id: Some(mention.id.clone()),
                        chosen_entity_id: Some(entity_id),
                        candidates: Vec::new(),
                        resolved_by_llm: false,
                        confidence: INTRA_DOC_CONFIDENCE,
                        strategy_used: strategy.to_owned(),
                    },
                );
                continue;
            }

            let mut result =
                service.resolve_span(&ed_ctx, config.allow_llm, config.max_candidates)?;
            attach_mention_id(&mut result, mention);

            let aliases = aliases_of(&mention.id);
            if result.chosen_entity_id.is_none() {
                if let Some(creator) = &entity_creator {
                    let (created, outcome) =
                        creator.create_entity_with_outcome(&ed_ctx, result, &aliases, None)?;
                    result = created;
                    attach_mention_id(&mut result, mention);
                    if let Some(outcome) = outcome {
                        if outcome.is_new() {
                            retriever.invalidate_cache();
                        }
                        resolution.outcomes.insert(mention.id.clone(), outcome);
                    }
                }
            }

            if let (Some(entity_id), Some(matcher)) = (&result.chosen_entity_id, matcher.as_mut()) {
                matcher.register_resolution(&mention.name, &mention.entity_type, entity_id);
                for alias in &aliases {
                    matcher.register_resolution(alias, &mention.entity_type, entity_id);
                }
            }
            resolution.results.insert(mention.id.clone(), result);
        }
        Ok(resolution)
    }
}

fn attach_mention_id(result: &mut EdResult, mention: &Mention) {
    if result.mention_id.as_ref().is_none_or(|id| id.as_str().is_empty()) {
        result.mention_id = Some(mention.id.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ce_core::Span;
    use pretty_assertions::assert_eq;

    fn mention(id: &str, entity_type: &str, name: &str) -> Mention {
        Mention::new(id, entity_type, name, Span::new(0, name.chars().count()))
    }

    fn entity_types() -> &'static EntityTypes {
        &MappingsRegistry::load().expect("mappings").entity_types
    }

    #[test]
    fn exact_and_fuzzy_matches() {
        let mut matcher = IntraDocumentMatcher::new(None, entity_types());
        matcher.register_resolution("Maria Lopez", "PER", &"e1".into());
        let exact = matcher.find_match(&mention("m2", "PER", "maria lopez"));
        assert_eq!(exact, Some(("e1".into(), "intra_doc_exact")));
        let fuzzy = matcher.find_match(&mention("m3", "PER", "Lopez"));
        assert_eq!(fuzzy, Some(("e1".into(), "intra_doc_fuzzy")));
        assert_eq!(matcher.find_match(&mention("m4", "PER", "")), None);
    }

    #[test]
    fn exact_match_never_crosses_incompatible_types() {
        let mut matcher = IntraDocumentMatcher::new(None, entity_types());
        matcher.register_resolution("Mercury", "PER", &"e-per".into());
        assert_eq!(matcher.find_match(&mention("m2", "ORG", "Mercury")), None);
        matcher.register_resolution("Riverside Arena", "ORG", &"e-org".into());
        assert_eq!(
            matcher.find_match(&mention("m3", "LOC", "Riverside Arena")),
            Some(("e-org".into(), "intra_doc_exact"))
        );
    }

    #[test]
    fn alias_step_uses_the_single_compatibility_rule() {
        let mut matcher = IntraDocumentMatcher::new(None, entity_types());
        matcher.register_resolution("Riverside Arena Complex", "ORG", &"e1".into());
        // An ungrouped type is compatible with every type (Python's alias rule rejected it).
        assert_eq!(
            matcher.find_match(&mention("m2", "OTHER", "Riverside Arena")),
            Some(("e1".into(), "intra_doc_alias"))
        );
        assert_eq!(
            matcher.find_match(&mention("m3", "CASE", "Riverside Arena")),
            Some(("e1".into(), "intra_doc_alias"))
        );
        // Grouped types in different groups never match.
        assert_eq!(matcher.find_match(&mention("m4", "PER", "Riverside Arena")), None);
        assert_eq!(matcher.find_match(&mention("m5", "PRODUCT", "Riverside Arena")), None);
    }

    #[test]
    fn collect_aliases_from_cluster_members() {
        let all: IndexMap<MentionId, Mention> = [
            mention("m1", "PERSON", "Maria Lopez"),
            mention("m2", "PERSON", "Ms. Lopez"),
            mention("m3", "PERSON", " maria lopez "),
        ]
        .into_iter()
        .map(|m| (m.id.clone(), m))
        .collect();
        let clusters = vec![CorefClusterContext {
            id: "cluster-1".into(),
            member_mention_ids: vec!["m1".into(), "m2".into(), "m3".into()],
            pronoun_spans: vec![],
        }];
        let canonical: IndexMap<String, MentionId> =
            [("cluster-1".to_owned(), MentionId::from("m1"))].into_iter().collect();
        let aliases =
            collect_cluster_aliases(&"m1".into(), Some(&clusters), Some(&canonical), &all);
        assert_eq!(aliases, vec!["Ms. Lopez"]);

        // Members missing from the map give no aliases.
        let reduced: IndexMap<MentionId, Mention> =
            [("m1".into(), mention("m1", "PERSON", "Maria Lopez"))].into_iter().collect();
        assert!(collect_cluster_aliases(&"m1".into(), Some(&clusters), Some(&canonical), &reduced)
            .is_empty());
        // A non-canonical mention gives none.
        assert!(collect_cluster_aliases(&"m2".into(), Some(&clusters), Some(&canonical), &all)
            .is_empty());
        assert!(collect_cluster_aliases(&"m1".into(), None, Some(&canonical), &all).is_empty());
    }
}
