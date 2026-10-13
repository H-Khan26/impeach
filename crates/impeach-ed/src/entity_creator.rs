//! Entity creation for unresolved mentions (Python `services/ed/entity_creator.py`):
//! reuse-before-create lookup (bm25 / `vector_fuzzy` / `all_entities_fuzzy`), role blocking,
//! entity nodes with aliases and metadata, and entity embeddings for cross-document reuse.
//!
//! Reuse never crosses incompatible entity types ([`entity_types_compatible`]): PER "Mercury"
//! and ORG "Mercury" stay two entities, while ORG / LOC / FACILITY share a group. A compatible
//! candidate of another type is reused only at `max(reuse_similarity_threshold, 0.90)`.
//!
//! Role terms (`roles.yaml` seed list) never become canonical names: a role-only mention
//! salvages the first non-role alias or is not created (`<strategy>_skip_role`).
//!
//! Store errors are logged and treated as "no candidates" / "no entity", except fatal ones,
//! which propagate. Embedder failures only skip the entity embedding.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, PoisonError};

use impeach_config::pipeline::EntityRetrievalStrategy;
use impeach_core::pyjson::to_python_json_ascii;
use impeach_core::{EdContext, EdResult, EntityId, Metadata};
use impeach_models::Embedder;
use impeach_ner::definitions::RoleConfig;
use impeach_store::{EntityNode, GraphStore, StoredEntity};
use impeach_text::normalize::py_strip;
use serde_json::Value;

use crate::error::{recover_store, Result};
use crate::name_matching::{entity_types_compatible, normalize_for_matching};
use crate::name_similarity::NameSimilarity;

/// Default minimum similarity for reusing an existing entity.
pub const DEFAULT_REUSE_SIMILARITY_THRESHOLD: f64 = 0.80;

/// Minimum similarity for reusing a compatible entity of another type.
pub const CROSS_TYPE_REUSE_THRESHOLD: f64 = 0.90;

/// Entities fetched by the `all_entities_fuzzy` reuse lookup.
pub const ALL_ENTITIES_LIMIT: usize = 1000;

/// Vector hits fetched by the `vector_fuzzy` reuse lookup.
pub const VECTOR_REUSE_LIMIT: usize = 20;

/// Outcome of [`EntityCreator::create_or_reuse_entity_for_mention`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntityOutcome {
    /// A new entity was created.
    Created(EntityId),
    /// An existing entity was reused.
    Reused(EntityId),
}

impl EntityOutcome {
    /// The entity id.
    #[must_use]
    pub fn entity_id(&self) -> &EntityId {
        match self {
            Self::Created(id) | Self::Reused(id) => id,
        }
    }

    /// Whether the entity was newly created.
    #[must_use]
    pub fn is_new(&self) -> bool {
        matches!(self, Self::Created(_))
    }
}

/// Creates (or reuses) `Entity` nodes for mentions ED could not resolve (Python
/// `EntityCreator`).
pub struct EntityCreator {
    store: Arc<dyn GraphStore>,
    embedder: Option<Arc<dyn Embedder>>,
    similarity: NameSimilarity,
    role_config: Option<RoleConfig>,
    reuse_similarity_threshold: f64,
    retrieval_strategy: EntityRetrievalStrategy,
    all_entities_cache: Mutex<Option<Arc<Vec<StoredEntity>>>>,
}

impl std::fmt::Debug for EntityCreator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EntityCreator")
            .field("store", &self.store)
            .field("embedder", &self.embedder.as_ref().map(|e| e.model_name()))
            .field("has_role_config", &self.role_config.is_some())
            .field("reuse_similarity_threshold", &self.reuse_similarity_threshold)
            .field("retrieval_strategy", &self.retrieval_strategy)
            .finish_non_exhaustive()
    }
}

impl EntityCreator {
    /// A creator over `store`; without an embedder new entities get no vector and the
    /// `vector_fuzzy` lookup finds nothing; without a role config no role blocking applies.
    /// Reuse threshold [`DEFAULT_REUSE_SIMILARITY_THRESHOLD`], strategy `bm25`.
    #[must_use]
    pub fn new(
        store: Arc<dyn GraphStore>,
        embedder: Option<Arc<dyn Embedder>>,
        similarity: NameSimilarity,
        role_config: Option<RoleConfig>,
    ) -> Self {
        Self {
            store,
            embedder,
            similarity,
            role_config,
            reuse_similarity_threshold: DEFAULT_REUSE_SIMILARITY_THRESHOLD,
            retrieval_strategy: EntityRetrievalStrategy::Bm25,
            all_entities_cache: Mutex::new(None),
        }
    }

    /// Sets the minimum similarity for reuse.
    #[must_use]
    pub fn with_reuse_similarity_threshold(mut self, threshold: f64) -> Self {
        self.reuse_similarity_threshold = threshold;
        self
    }

    /// Sets the candidate lookup strategy of reuse-before-create.
    #[must_use]
    pub fn with_retrieval_strategy(mut self, strategy: EntityRetrievalStrategy) -> Self {
        self.retrieval_strategy = strategy;
        self
    }

    /// The candidate lookup strategy.
    #[must_use]
    pub fn retrieval_strategy(&self) -> EntityRetrievalStrategy {
        self.retrieval_strategy
    }

    /// Whether `name` is a role term (`Plaintiff`, `Defendant`, ...); false without a role
    /// config.
    #[must_use]
    pub fn is_role_term(&self, name: &str) -> bool {
        !name.is_empty() && self.role_config.as_ref().is_some_and(|c| c.is_valid_role_term(name))
    }

    /// Whether `name` is a role prefix plus a name (`Defendant Reed`).
    #[must_use]
    pub fn is_hybrid_role_name(&self, name: &str) -> bool {
        !name.is_empty() && self.role_config.as_ref().is_some_and(|c| c.is_hybrid_role_name(name))
    }

    /// The name part of a hybrid role name (`Defendant Reed` -> `Reed`).
    #[must_use]
    pub fn strip_role_prefix<'n>(&self, name: &'n str) -> Option<&'n str> {
        if name.is_empty() {
            return None;
        }
        self.role_config.as_ref()?.strip_role_prefix(name)
    }

    /// An existing entity for the mention, or `None`.
    ///
    /// Candidates come from the configured strategy. Role-named entities are skipped when
    /// `canonical_name` is not a role term; incompatible types are skipped. A candidate whose
    /// `Metadata.normalized_name` equals `normalized_name` is returned at once; otherwise the
    /// best type-aware similarity against name and aliases wins: same type (or a missing type)
    /// at `reuse_similarity_threshold`, another compatible type at
    /// `max(reuse_similarity_threshold, 0.90)`.
    ///
    /// # Errors
    /// Fatal store errors.
    pub fn find_existing_entity_for_mention(
        &self,
        canonical_name: &str,
        entity_type: &str,
        normalized_name: &str,
    ) -> Result<Option<EntityId>> {
        let searching_for_real_name = !self.is_role_term(canonical_name);
        let entities = self.fetch_entity_candidates(canonical_name)?;
        if entities.is_empty() {
            return Ok(None);
        }

        let mut best_same: Option<(&EntityId, f64)> = None;
        let mut best_cross: Option<(&EntityId, f64)> = None;
        let cross_type_threshold = self.reuse_similarity_threshold.max(CROSS_TYPE_REUSE_THRESHOLD);
        let entity_types = self.similarity.entity_types();

        for entity in entities.iter() {
            let node = &entity.node;
            if searching_for_real_name && !node.name.is_empty() && self.is_role_term(&node.name) {
                tracing::debug!(
                    entity = %entity.id,
                    name = node.name.as_str(),
                    canonical_name,
                    "skipping role-named entity for a real name"
                );
                continue;
            }
            if !entity_types_compatible(entity_types, Some(entity_type), Some(&node.entity_type)) {
                tracing::debug!(
                    entity = %entity.id,
                    name = node.name.as_str(),
                    entity_type = node.entity_type.as_str(),
                    mention_type = entity_type,
                    "skipping entity of an incompatible type"
                );
                continue;
            }

            let meta: Option<serde_json::Map<String, Value>> =
                serde_json::from_str(&node.metadata).ok();
            if meta
                .as_ref()
                .and_then(|m| m.get("normalized_name"))
                .and_then(Value::as_str)
                .is_some_and(|n| n == normalized_name)
            {
                tracing::debug!(canonical_name, entity = %entity.id, "entity reuse fast path");
                return Ok(Some(entity.id.clone()));
            }

            let mut best_node_score = 0.0_f64;
            if !node.name.is_empty() {
                let score = self.similarity.enhanced_string_similarity(
                    canonical_name,
                    &node.name,
                    entity_type,
                );
                best_node_score = best_node_score.max(score);
            }
            for alias in node.aliases() {
                let score =
                    self.similarity.enhanced_string_similarity(canonical_name, &alias, entity_type);
                if score > best_node_score {
                    best_node_score = score;
                }
            }
            if best_node_score <= 0.0 {
                continue;
            }

            let best = if !entity_type.is_empty()
                && !node.entity_type.is_empty()
                && node.entity_type != entity_type
            {
                &mut best_cross
            } else {
                &mut best_same
            };
            if best.is_none_or(|(_, score)| best_node_score > score) {
                *best = Some((&entity.id, best_node_score));
            }
        }

        if let Some((id, score)) = best_same {
            if score >= self.reuse_similarity_threshold {
                tracing::debug!(canonical_name, entity = %id, score, "entity reuse match");
                return Ok(Some(id.clone()));
            }
        }
        if let Some((id, score)) = best_cross {
            if score >= cross_type_threshold {
                tracing::debug!(
                    canonical_name,
                    mention_type = entity_type,
                    entity = %id,
                    score,
                    "cross-type entity reuse match"
                );
                return Ok(Some(id.clone()));
            }
        }
        Ok(None)
    }

    /// Reuse candidates by strategy: BM25 name search, vector search over the embedded name
    /// (top [`VECTOR_REUSE_LIMIT`]), or every entity (cached and shared until the next entity
    /// is created). Entities without a name are dropped.
    ///
    /// # Errors
    /// Fatal store errors.
    pub fn fetch_entity_candidates(&self, query_name: &str) -> Result<Arc<Vec<StoredEntity>>> {
        let entities = match self.retrieval_strategy {
            EntityRetrievalStrategy::Bm25 => {
                let hits = recover_store(
                    self.store.find_entity_by_name(query_name),
                    "BM25 entity lookup",
                )?;
                hits.into_iter().map(|hit| hit.node).collect()
            }
            EntityRetrievalStrategy::VectorFuzzy => self.fetch_via_vector(query_name)?,
            EntityRetrievalStrategy::AllEntitiesFuzzy => return self.all_entities_cached(),
        };
        Ok(Arc::new(with_name(entities)))
    }

    fn fetch_via_vector(&self, query_name: &str) -> Result<Vec<StoredEntity>> {
        let Some(embedder) = &self.embedder else {
            tracing::debug!("vector entity lookup unavailable: no embedder");
            return Ok(Vec::new());
        };
        let vector = match embedder.embed(&[query_name]) {
            Ok(mut vectors) if !vectors.is_empty() => vectors.swap_remove(0),
            Ok(_) => return Ok(Vec::new()),
            Err(error) => {
                tracing::warn!(error = %error, "vector entity lookup failed");
                return Ok(Vec::new());
            }
        };
        let hits = recover_store(
            self.store.find_similar_entities(&vector, VECTOR_REUSE_LIMIT),
            "vector entity lookup",
        )?;
        Ok(hits.into_iter().map(|hit| hit.node).collect())
    }

    fn all_entities_cached(&self) -> Result<Arc<Vec<StoredEntity>>> {
        let mut cache = self.all_entities_cache.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(entities) = cache.as_ref() {
            return Ok(Arc::clone(entities));
        }
        let entities = Arc::new(with_name(recover_store(
            self.store.get_all_entities(ALL_ENTITIES_LIMIT),
            "fetching all entities",
        )?));
        tracing::info!(count = entities.len(), "loaded entities into the EntityCreator cache");
        *cache = Some(Arc::clone(&entities));
        Ok(entities)
    }

    /// Drops the `all_entities_fuzzy` cache; called after every entity this creator writes.
    pub fn invalidate_cache(&self) {
        *self.all_entities_cache.lock().unwrap_or_else(PoisonError::into_inner) = None;
    }

    /// The id of a new or reused entity for the mention (Python `create_entity_for_mention`).
    ///
    /// # Errors
    /// Fatal store errors.
    pub fn create_entity_for_mention(
        &self,
        ctx: &EdContext,
        aliases: &[String],
        metadata: Option<&Metadata>,
    ) -> Result<Option<EntityId>> {
        Ok(self.create_or_reuse_entity_for_mention(ctx, aliases, metadata)?.map(|outcome| {
            match outcome {
                EntityOutcome::Created(id) | EntityOutcome::Reused(id) => id,
            }
        }))
    }

    /// Creates or reuses an entity for the mention; `None` when no entity can be made (empty
    /// name, role-only name, store failure).
    ///
    /// The canonical name is the stripped mention name; a role term is replaced by the first
    /// non-role alias. A hybrid role name (`Defendant Reed`) loses its role prefix when an
    /// alias is a pure role term. A new entity gets the sorted, deduplicated aliases (plus the
    /// original mention name when it differs), metadata `source_doc_id`, `case_id`,
    /// `jurisdiction`, `normalized_name`, `metadata`, `roles`, a description from the sentence
    /// and an embedding of `Name: Description (Aliases: ...)`.
    ///
    /// # Errors
    /// Fatal store errors.
    pub fn create_or_reuse_entity_for_mention(
        &self,
        ctx: &EdContext,
        aliases: &[String],
        metadata: Option<&Metadata>,
    ) -> Result<Option<EntityOutcome>> {
        let mention = &ctx.mention;
        let doc_ctx = &ctx.doc_context;
        let mut aliases: Vec<String> = aliases.to_vec();

        let mut canonical_name = py_strip(&mention.name).to_owned();
        if canonical_name.is_empty() {
            tracing::warn!(mention = %mention.id, "cannot create entity: empty name");
            return Ok(None);
        }

        let mut detected_roles: Vec<String> = Vec::new();
        if self.is_role_term(&canonical_name) {
            detected_roles.push(canonical_name.to_uppercase());
            let salvaged = aliases
                .iter()
                .find(|a| !py_strip(a).is_empty() && !self.is_role_term(a))
                .map(|a| py_strip(a).to_owned());
            match salvaged {
                Some(name) => {
                    tracing::info!(
                        mention = %mention.id,
                        role = canonical_name.as_str(),
                        salvaged = name.as_str(),
                        "salvaged canonical name from aliases"
                    );
                    canonical_name = name;
                }
                None => {
                    tracing::warn!(
                        mention = %mention.id,
                        role = canonical_name.as_str(),
                        "skipping entity creation: only a role term is available"
                    );
                    return Ok(None);
                }
            }
        }
        for alias in &aliases {
            if !alias.is_empty() && self.is_role_term(alias) {
                let upper = alias.to_uppercase();
                if !detected_roles.contains(&upper) {
                    detected_roles.push(upper);
                }
            }
        }

        let entity_type = if ctx.span_label.is_empty() {
            if mention.entity_type.is_empty() {
                "OTHER".to_owned()
            } else {
                mention.entity_type.clone()
            }
        } else {
            ctx.span_label.clone()
        };

        if self.is_hybrid_role_name(&canonical_name) && aliases.iter().any(|a| self.is_role_term(a))
        {
            if let Some(stripped) = self.strip_role_prefix(&canonical_name).map(str::to_owned) {
                tracing::info!(
                    from = canonical_name.as_str(),
                    to = stripped.as_str(),
                    "stripped role prefix for entity creation"
                );
                if !aliases.contains(&canonical_name) {
                    aliases.push(canonical_name.clone());
                }
                let prefix_chars =
                    canonical_name.chars().count().saturating_sub(stripped.chars().count());
                let prefix: String = canonical_name.chars().take(prefix_chars).collect();
                let role_prefix = py_strip(&prefix).to_uppercase();
                if !role_prefix.is_empty() && !detected_roles.contains(&role_prefix) {
                    detected_roles.push(role_prefix);
                }
                canonical_name = stripped;
            }
        }

        let normalized_name = normalize_for_matching(&canonical_name);

        if let Some(existing) =
            self.find_existing_entity_for_mention(&canonical_name, &entity_type, &normalized_name)?
        {
            tracing::info!(
                entity = %existing,
                mention = %mention.id,
                name = canonical_name.as_str(),
                "reusing existing entity"
            );
            return Ok(Some(EntityOutcome::Reused(existing)));
        }

        let canonical_lower = canonical_name.to_lowercase();
        let mut alias_set: BTreeSet<String> = aliases
            .iter()
            .map(|a| py_strip(a))
            .filter(|a| !a.is_empty() && a.to_lowercase() != canonical_lower)
            .map(str::to_owned)
            .collect();
        let original_name = py_strip(&mention.name);
        if original_name.to_lowercase() != canonical_lower && !original_name.is_empty() {
            alias_set.insert(original_name.to_owned());
        }
        let alias_list: Vec<String> = alias_set.into_iter().collect();

        let mut meta = Metadata::new();
        if !doc_ctx.doc_id.as_str().is_empty() {
            meta.insert("source_doc_id".into(), Value::from(doc_ctx.doc_id.as_str()));
        }
        if let Some(case_id) = doc_ctx.case_id.as_deref().filter(|c| !c.is_empty()) {
            meta.insert("case_id".into(), Value::from(case_id));
        }
        if let Some(jurisdiction) = doc_ctx.jurisdiction.as_deref().filter(|j| !j.is_empty()) {
            meta.insert("jurisdiction".into(), Value::from(jurisdiction));
        }
        if !normalized_name.is_empty() {
            meta.insert("normalized_name".into(), Value::from(normalized_name.as_str()));
        }
        if let Some(extra) = metadata {
            for (key, value) in extra {
                meta.insert(key.clone(), value.clone());
            }
        }
        if !detected_roles.is_empty() {
            meta.insert("roles".into(), Value::from(detected_roles.clone()));
        }

        let description = if ctx.local_context.sentence_text.is_empty() {
            String::new()
        } else {
            let sentence: String = ctx.local_context.sentence_text.chars().take(200).collect();
            format!("Mentioned in: {sentence}")
        };

        let node = EntityNode {
            name: canonical_name.clone(),
            entity_type,
            aliases_json: json_text(&alias_list, "[]"),
            description: description.clone(),
            metadata: if meta.is_empty() { "{}".to_owned() } else { json_text(&meta, "{}") },
        };

        let created = match self.store.create_entity(&node) {
            Ok(created) => created,
            Err(error) if error.is_fatal() => return Err(error.into()),
            Err(error) => {
                tracing::error!(mention = %mention.id, error = %error, "failed to create entity");
                return Ok(None);
            }
        };
        tracing::info!(
            entity = %created.id,
            mention = %mention.id,
            name = canonical_name.as_str(),
            "created new entity"
        );
        // Later mentions of the document must see the new entity.
        self.invalidate_cache();
        self.create_entity_embedding(&created.id, &canonical_name, &description, &alias_list)?;
        Ok(Some(EntityOutcome::Created(created.id)))
    }

    /// Writes the entity vector of `Name: Description (Aliases: a, b)`; a no-op without an
    /// embedder. Embedder and non-fatal store failures are logged.
    fn create_entity_embedding(
        &self,
        entity_id: &EntityId,
        canonical_name: &str,
        description: &str,
        aliases: &[String],
    ) -> Result<()> {
        let Some(embedder) = &self.embedder else {
            return Ok(());
        };
        let mut text = canonical_name.to_owned();
        if !description.is_empty() {
            text.push_str(": ");
            text.push_str(description);
        }
        if !aliases.is_empty() {
            text.push_str(" (Aliases: ");
            text.push_str(&aliases.join(", "));
            text.push(')');
        }
        let vector = match embedder.embed(&[text.as_str()]) {
            Ok(mut vectors) if !vectors.is_empty() => vectors.swap_remove(0),
            Ok(_) => {
                tracing::warn!(entity = %entity_id, "entity embedding: empty result");
                return Ok(());
            }
            Err(error) => {
                tracing::warn!(entity = %entity_id, error = %error, "entity embedding failed");
                return Ok(());
            }
        };
        match self.store.create_entity_embedding(entity_id, &vector) {
            Ok(()) => {
                tracing::debug!(entity = %entity_id, name = canonical_name, "created entity embedding");
                Ok(())
            }
            Err(error) if error.is_fatal() => Err(error.into()),
            Err(error) => {
                tracing::warn!(entity = %entity_id, error = %error, "failed to store entity embedding");
                Ok(())
            }
        }
    }

    /// `result` unchanged when it has a chosen entity; otherwise a result for a new or reused
    /// entity.
    ///
    /// Types in `skip_ed_types` (DATE, TIME, MONEY) are never created (`skip_value_type`). A
    /// mention that yields no entity keeps its result with strategy `<strategy>_skip_role`.
    /// Created entities get strategy `<strategy>_create` and confidence 0.7; reused ones
    /// `vector_fuzzy` (with that retrieval strategy) or `<strategy>_reuse` and confidence 0.85.
    ///
    /// # Errors
    /// Fatal store errors.
    pub fn create_entity_from_ed_result(
        &self,
        ctx: &EdContext,
        result: EdResult,
        aliases: &[String],
        metadata: Option<&Metadata>,
    ) -> Result<EdResult> {
        Ok(self.create_entity_with_outcome(ctx, result, aliases, metadata)?.0)
    }

    /// [`EntityCreator::create_entity_from_ed_result`] together with whether an entity was
    /// created or reused (`None` when the result was left unchanged or not resolved).
    ///
    /// # Errors
    /// Fatal store errors.
    pub fn create_entity_with_outcome(
        &self,
        ctx: &EdContext,
        result: EdResult,
        aliases: &[String],
        metadata: Option<&Metadata>,
    ) -> Result<(EdResult, Option<EntityOutcome>)> {
        if result.chosen_entity_id.is_some() {
            return Ok((result, None));
        }
        let entity_type =
            if ctx.span_label.is_empty() { &ctx.mention.entity_type } else { &ctx.span_label };
        if self.similarity.entity_types().skip_ed_types.contains(entity_type.as_str()) {
            tracing::debug!(
                entity_type = entity_type.as_str(),
                name = ctx.mention.name.as_str(),
                "skipping entity creation for a value type"
            );
            let result = EdResult {
                resolved_by_llm: false,
                confidence: 0.0,
                strategy_used: "skip_value_type".to_owned(),
                ..result
            };
            return Ok((result, None));
        }

        let Some(outcome) = self.create_or_reuse_entity_for_mention(ctx, aliases, metadata)? else {
            let strategy_used = format!("{}_skip_role", result.strategy_used);
            return Ok((EdResult { strategy_used, ..result }, None));
        };
        let (strategy_used, confidence) = match &outcome {
            EntityOutcome::Created(_) => (format!("{}_create", result.strategy_used), 0.7),
            EntityOutcome::Reused(_)
                if self.retrieval_strategy == EntityRetrievalStrategy::VectorFuzzy =>
            {
                ("vector_fuzzy".to_owned(), 0.85)
            }
            EntityOutcome::Reused(_) => (format!("{}_reuse", result.strategy_used), 0.85),
        };
        let result = EdResult {
            chosen_entity_id: Some(outcome.entity_id().clone()),
            resolved_by_llm: false,
            confidence,
            strategy_used,
            ..result
        };
        Ok((result, Some(outcome)))
    }
}

/// Entities with a non-empty name.
fn with_name(entities: Vec<StoredEntity>) -> Vec<StoredEntity> {
    entities.into_iter().filter(|e| !e.node.name.is_empty()).collect()
}

/// Python `json.dumps(value)` (ASCII-escaped), or `fallback` when serialisation fails.
fn json_text<T: serde::Serialize + ?Sized>(value: &T, fallback: &str) -> String {
    to_python_json_ascii(value).unwrap_or_else(|_| fallback.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use impeach_config::MappingsRegistry;
    use impeach_core::{DocContext, LocalContext, Mention, Span};
    use impeach_store::MemoryStore;
    use pretty_assertions::assert_eq;

    fn creator(store: &Arc<MemoryStore>) -> EntityCreator {
        let mappings = MappingsRegistry::load().expect("mappings");
        let dyn_store: Arc<dyn GraphStore> = store.clone();
        EntityCreator::new(
            dyn_store,
            None,
            NameSimilarity::from_mappings(mappings),
            Some(RoleConfig::load(None).expect("roles")),
        )
    }

    fn ctx(name: &str, label: &str) -> EdContext {
        let mut doc = DocContext::new("doc_123", "complaint");
        doc.case_id = Some("case_456".into());
        doc.jurisdiction = Some("CA".into());
        EdContext {
            mention: Mention::new("mention_1", label, name, Span::new(100, 110)),
            local_context: LocalContext::new(
                "This is a test sentence.",
                "This is a test chunk with more context.",
            ),
            doc_context: doc,
            span_label: label.into(),
            normalized_key: None,
        }
    }

    #[test]
    fn outcome_accessors() {
        let created = EntityOutcome::Created("1".into());
        assert!(created.is_new());
        assert_eq!(created.entity_id().as_str(), "1");
        assert!(!EntityOutcome::Reused("2".into()).is_new());
    }

    #[test]
    fn hybrid_role_name_is_stripped_when_cluster_has_a_role_term() {
        let store = Arc::new(MemoryStore::new(3));
        let creator = creator(&store);
        let outcome = creator
            .create_or_reuse_entity_for_mention(
                &ctx("Defendant Reed", "PER"),
                &["Defendant".to_owned()],
                None,
            )
            .expect("creates")
            .expect("entity");
        let node = store.get_all_entities(10).expect("entities");
        assert_eq!(node.len(), 1);
        assert_eq!(node[0].id, *outcome.entity_id());
        assert_eq!(node[0].node.name, "Reed");
        assert_eq!(node[0].node.aliases(), vec!["Defendant", "Defendant Reed"]);
        let meta: Value = serde_json::from_str(&node[0].node.metadata).expect("json");
        assert_eq!(meta["roles"], serde_json::json!(["DEFENDANT"]));
    }

    #[test]
    fn metadata_and_aliases_use_python_json_formatting() {
        let store = Arc::new(MemoryStore::new(3));
        let creator = creator(&store);
        creator
            .create_entity_for_mention(&ctx("José Ruiz", "PER"), &["Mr. Ruiz".to_owned()], None)
            .expect("creates")
            .expect("entity");
        let entity = &store.get_all_entities(10).expect("entities")[0].node;
        assert_eq!(entity.aliases_json, "[\"Mr. Ruiz\"]");
        assert_eq!(
            entity.metadata,
            "{\"source_doc_id\": \"doc_123\", \"case_id\": \"case_456\", \"jurisdiction\": \
             \"CA\", \"normalized_name\": \"jos\\u00e9 ruiz\"}"
        );
        assert_eq!(entity.description, "Mentioned in: This is a test sentence.");
    }
}
