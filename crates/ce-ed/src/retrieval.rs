//! Candidate entity retrieval for ED (Python `services/ed/retrieval.py`).
//!
//! [`EdRetrieval`] is the retrieval contract the rule resolvers and the ED service use (Python
//! `EDRetrievalService`); [`EdRetriever`] implements it over a [`GraphStore`] and an optional
//! [`Embedder`] (Python `HelixEDRetriever`) with the three strategies of
//! [`EntityRetrievalStrategy`]:
//! - `bm25`: `FindEntityByName` (BM25 over names and aliases);
//! - `vector_fuzzy`: entity vector search (over-fetched by `vector_fuzzy_overfetch_multiplier`),
//!   re-scored as `vector_weight * cosine + fuzzy_weight * best name/alias similarity`;
//! - `all_entities_fuzzy`: name/alias similarity over every entity (cached until
//!   [`EdRetriever::invalidate_cache`]).
//!
//! Vector candidates carry the store's cosine score unchanged; no candidate is dropped because
//! its score is zero. Ranked results are sorted stably by descending score, so equal scores keep
//! retrieval order (Python `list.sort(reverse=True)`); scores are finite, never NaN.
//!
//! Errors: store failures are logged and yield no candidates, except fatal ones
//! ([`crate::error::recover_store`]); embedder failures yield no candidates.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use ce_config::pipeline::{EdStageConfig, EntityRetrievalStrategy};
use ce_core::{EdCandidate, EntityId, Metadata};
use ce_models::Embedder;
use ce_store::{GraphStore, StoredEntity};
use serde_json::Value;

use crate::error::{recover_store, Result};
use crate::name_similarity::NameSimilarity;
use crate::structured_keys::{
    case_citation_key_to_search_terms, exhibit_key_to_search_terms, statute_key_to_search_terms,
    CaseCitationKey, ExhibitKey, StatuteKey,
};

/// Entity limit of the `all_entities_fuzzy` cache (Python `max_entities`).
const ALL_ENTITIES_LIMIT: usize = 1000;

const SOURCE_EMBEDDING: &str = "embedding";
const SOURCE_EXACT: &str = "exact";
const SOURCE_SIMILARITY: &str = "similarity";
const SOURCE_VECTOR_FUZZY: &str = "vector_fuzzy";
const SOURCE_ALL_ENTITIES_FUZZY: &str = "all_entities_fuzzy";
const SOURCE_DOCKET: &str = "structured_docket";
const SOURCE_EXHIBIT: &str = "structured_exhibit";
const SOURCE_STATUTE: &str = "structured_statute";
const SOURCE_CASE: &str = "structured_case";

/// A structured-key lookup (Python `find_by_structured_key(key_type, key_value)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuredKeyQuery<'a> {
    /// `docket_entry`: a docket number such as `2:23-cv-01492`.
    DocketEntry {
        /// Docket number.
        number: &'a str,
    },
    /// `exhibit`: an exhibit id with optional party attribution.
    Exhibit {
        /// Exhibit id, e.g. `12A`.
        id: &'a str,
        /// Party attribution, e.g. `Plaintiff's`.
        party_attribution: Option<&'a str>,
    },
    /// `statute`: title, section, subsection and code type of the key.
    Statute(&'a StatuteKey),
    /// `case_citation`: citation components or caption parties of the key.
    CaseCitation(&'a CaseCitationKey),
}

/// Retrieval of candidate entities (Python `EDRetrievalService`). Retrieval is global over
/// every entity of the store; `case_id` only scopes exhibit keys.
pub trait EdRetrieval: Send + Sync + std::fmt::Debug {
    /// Candidates whose name or alias matches `surface_form`, optionally only of
    /// `entity_type` (labels compared exactly). With the `vector_fuzzy` or `all_entities_fuzzy`
    /// strategy this delegates to that strategy.
    ///
    /// # Errors
    /// Fatal store errors.
    fn find_by_exact_match(
        &self,
        surface_form: &str,
        entity_type: Option<&str>,
    ) -> Result<Vec<EdCandidate>>;

    /// Candidates by vector similarity of `query_text`, best first, each carrying its cosine
    /// score; candidates scoring below a positive `min_similarity` are dropped.
    ///
    /// # Errors
    /// Fatal store errors.
    fn find_by_embedding(
        &self,
        query_text: &str,
        limit: usize,
        min_similarity: f64,
    ) -> Result<Vec<EdCandidate>>;

    /// BM25 candidates re-scored with type-aware name similarity, those at or above
    /// `min_similarity`, best first.
    ///
    /// # Errors
    /// Fatal store errors.
    fn find_by_similarity(
        &self,
        surface_form: &str,
        entity_type: Option<&str>,
        limit: usize,
        min_similarity: f64,
    ) -> Result<Vec<EdCandidate>>;

    /// Candidates for a structured key.
    ///
    /// # Errors
    /// Fatal store errors.
    fn find_by_structured_key(
        &self,
        key: &StructuredKeyQuery<'_>,
        case_id: Option<&str>,
    ) -> Result<Vec<EdCandidate>>;
}

/// Tuning of [`EdRetriever`] (the retrieval fields of `EDStageConfig`).
#[derive(Debug, Clone, PartialEq)]
pub struct RetrieverSettings {
    /// Default strategy.
    pub strategy: EntityRetrievalStrategy,
    /// Candidate limit of strategy-delegated exact matches.
    pub max_candidates: usize,
    /// Vector over-fetch factor of `vector_fuzzy` (at least 1).
    pub vector_fuzzy_overfetch_multiplier: usize,
    /// Minimum combined score of `vector_fuzzy` / minimum similarity of `all_entities_fuzzy`.
    pub vector_fuzzy_min_similarity: f64,
    /// Weight of the cosine score in `vector_fuzzy`.
    pub vector_fuzzy_vector_weight: f64,
    /// Weight of the name similarity in `vector_fuzzy`.
    pub vector_fuzzy_fuzzy_weight: f64,
    /// Minimum cosine score of the initial vector search of `vector_fuzzy`
    /// (`min_embedding_similarity`).
    pub initial_vector_min_similarity: f64,
}

impl Default for RetrieverSettings {
    fn default() -> Self {
        Self {
            strategy: EntityRetrievalStrategy::Bm25,
            max_candidates: 10,
            vector_fuzzy_overfetch_multiplier: 3,
            vector_fuzzy_min_similarity: 0.5,
            vector_fuzzy_vector_weight: 0.4,
            vector_fuzzy_fuzzy_weight: 0.6,
            initial_vector_min_similarity: 0.0,
        }
    }
}

impl RetrieverSettings {
    /// Settings of an `EDStage` config block.
    #[must_use]
    pub fn from_config(config: &EdStageConfig) -> Self {
        Self {
            strategy: config.entity_retrieval_strategy,
            max_candidates: config.max_candidates,
            vector_fuzzy_overfetch_multiplier: config.vector_fuzzy_overfetch_multiplier,
            vector_fuzzy_min_similarity: config.vector_fuzzy_min_similarity,
            vector_fuzzy_vector_weight: config.vector_fuzzy_vector_weight,
            vector_fuzzy_fuzzy_weight: config.vector_fuzzy_fuzzy_weight,
            initial_vector_min_similarity: config.min_embedding_similarity,
        }
    }
}

/// [`EdRetrieval`] over a graph store (Python `HelixEDRetriever`).
pub struct EdRetriever {
    store: Arc<dyn GraphStore>,
    embedder: Option<Arc<dyn Embedder>>,
    settings: RetrieverSettings,
    similarity: NameSimilarity,
    /// Candidates of every entity for `all_entities_fuzzy`; `None` until loaded.
    all_entities: Mutex<Option<Vec<EdCandidate>>>,
}

impl std::fmt::Debug for EdRetriever {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EdRetriever")
            .field("store", &self.store)
            .field("embedder", &self.embedder.as_ref().map(|e| e.model_name()))
            .field("settings", &self.settings)
            .finish_non_exhaustive()
    }
}

impl EdRetriever {
    /// A retriever over `store`; without an embedder vector retrieval returns nothing.
    #[must_use]
    pub fn new(
        store: Arc<dyn GraphStore>,
        embedder: Option<Arc<dyn Embedder>>,
        settings: RetrieverSettings,
        similarity: NameSimilarity,
    ) -> Self {
        Self { store, embedder, settings, similarity, all_entities: Mutex::new(None) }
    }

    /// The settings.
    #[must_use]
    pub fn settings(&self) -> &RetrieverSettings {
        &self.settings
    }

    /// Candidates by `strategy` (default: the configured one); `min_similarity` defaults to
    /// `vector_fuzzy_min_similarity` for the fuzzy strategies.
    ///
    /// # Errors
    /// Fatal store errors.
    pub fn find_candidates(
        &self,
        surface_form: &str,
        entity_type: Option<&str>,
        limit: usize,
        min_similarity: Option<f64>,
        strategy: Option<EntityRetrievalStrategy>,
    ) -> Result<Vec<EdCandidate>> {
        let min_similarity =
            Some(min_similarity.unwrap_or(self.settings.vector_fuzzy_min_similarity));
        match strategy.unwrap_or(self.settings.strategy) {
            EntityRetrievalStrategy::Bm25 => self.find_by_exact_match(surface_form, entity_type),
            EntityRetrievalStrategy::VectorFuzzy => {
                self.find_by_vector_with_fuzzy(surface_form, entity_type, limit, min_similarity)
            }
            EntityRetrievalStrategy::AllEntitiesFuzzy => {
                self.find_all_entities_with_fuzzy(surface_form, entity_type, limit, min_similarity)
            }
        }
    }

    /// The `vector_fuzzy` strategy (source `vector_fuzzy`).
    ///
    /// # Errors
    /// Fatal store errors.
    pub fn find_by_vector_with_fuzzy(
        &self,
        surface_form: &str,
        entity_type: Option<&str>,
        limit: usize,
        min_similarity: Option<f64>,
    ) -> Result<Vec<EdCandidate>> {
        if surface_form.is_empty() {
            return Ok(Vec::new());
        }
        let min_similarity = min_similarity.unwrap_or(self.settings.vector_fuzzy_min_similarity);
        let overfetch = self.settings.vector_fuzzy_overfetch_multiplier.max(1);
        let vector_candidates = self.find_by_embedding(
            surface_form,
            limit.saturating_mul(overfetch),
            self.settings.initial_vector_min_similarity,
        )?;
        if vector_candidates.is_empty() {
            tracing::debug!(surface_form, "vector search returned no candidates");
            return Ok(Vec::new());
        }
        let entity_type = type_filter(entity_type);
        let mut scored = Vec::new();
        for mut candidate in vector_candidates {
            if entity_type.is_some_and(|t| candidate.label != t) {
                continue;
            }
            let best = self.best_name_similarity(surface_form, &candidate, entity_type);
            let combined = self.settings.vector_fuzzy_vector_weight * candidate.score
                + self.settings.vector_fuzzy_fuzzy_weight * best;
            if combined >= min_similarity {
                tracing::debug!(
                    surface_form,
                    entity = %candidate.canonical_name,
                    score = combined,
                    "vector+fuzzy match"
                );
                candidate.score = combined;
                SOURCE_VECTOR_FUZZY.clone_into(&mut candidate.source);
                scored.push(candidate);
            }
        }
        sort_by_score_desc(&mut scored);
        scored.truncate(limit);
        Ok(scored)
    }

    /// The `all_entities_fuzzy` strategy (source `all_entities_fuzzy`).
    ///
    /// # Errors
    /// Fatal store errors.
    pub fn find_all_entities_with_fuzzy(
        &self,
        surface_form: &str,
        entity_type: Option<&str>,
        limit: usize,
        min_similarity: Option<f64>,
    ) -> Result<Vec<EdCandidate>> {
        if surface_form.is_empty() {
            return Ok(Vec::new());
        }
        let min_similarity = min_similarity.unwrap_or(self.settings.vector_fuzzy_min_similarity);
        let entity_type = type_filter(entity_type);
        let mut cache = self.lock_cache();
        if cache.is_none() {
            *cache = Some(self.load_all_entities()?);
        }
        let entities = cache.as_deref().unwrap_or_default();
        if entities.is_empty() {
            tracing::debug!("no entities in the graph for fuzzy matching");
            return Ok(Vec::new());
        }
        let mut scored = Vec::new();
        for candidate in entities {
            if entity_type.is_some_and(|t| candidate.label != t) {
                continue;
            }
            let best = self.best_name_similarity(surface_form, candidate, entity_type);
            if best >= min_similarity {
                let mut scored_candidate = candidate.clone();
                scored_candidate.score = best;
                SOURCE_ALL_ENTITIES_FUZZY.clone_into(&mut scored_candidate.source);
                scored.push(scored_candidate);
            }
        }
        drop(cache);
        sort_by_score_desc(&mut scored);
        scored.truncate(limit);
        Ok(scored)
    }

    /// Drops the entity cache of `all_entities_fuzzy` (call after creating entities).
    pub fn invalidate_cache(&self) {
        *self.lock_cache() = None;
    }

    fn lock_cache(&self) -> MutexGuard<'_, Option<Vec<EdCandidate>>> {
        self.all_entities.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Candidates of the first [`ALL_ENTITIES_LIMIT`] entities; a non-fatal store failure
    /// yields none (cached like a successful load).
    fn load_all_entities(&self) -> Result<Vec<EdCandidate>> {
        let entities =
            recover_store(self.store.get_all_entities(ALL_ENTITIES_LIMIT), "GetAllEntities")?;
        let candidates: Vec<EdCandidate> = entities.iter().filter_map(entity_candidate).collect();
        tracing::info!(count = candidates.len(), "loaded entities into the fuzzy matching cache");
        Ok(candidates)
    }

    /// Best type-aware similarity of `surface_form` to the candidate's name or aliases; the
    /// type is `entity_type`, else the candidate's label.
    fn best_name_similarity(
        &self,
        surface_form: &str,
        candidate: &EdCandidate,
        entity_type: Option<&str>,
    ) -> f64 {
        let entity_type = entity_type.unwrap_or(&candidate.label);
        let mut best = self.similarity.enhanced_string_similarity(
            surface_form,
            &candidate.canonical_name,
            entity_type,
        );
        for alias in &candidate.aliases {
            let score =
                self.similarity.enhanced_string_similarity(surface_form, alias, entity_type);
            if score > best {
                best = score;
            }
        }
        best
    }

    /// Exact-match candidates of every term for every type in order, the first occurrence of
    /// each entity kept and tagged with `source`.
    fn structured_lookup<S: AsRef<str>>(
        &self,
        terms: &[S],
        entity_types: &[&str],
        source: &str,
    ) -> Result<Vec<EdCandidate>> {
        let mut seen: HashSet<EntityId> = HashSet::new();
        let mut candidates = Vec::new();
        for term in terms {
            for entity_type in entity_types {
                for mut candidate in self.find_by_exact_match(term.as_ref(), Some(entity_type))? {
                    if seen.insert(candidate.entity_id.clone()) {
                        source.clone_into(&mut candidate.source);
                        candidates.push(candidate);
                    }
                }
            }
        }
        Ok(candidates)
    }

    fn lookup_docket_entry(&self, number: &str) -> Result<Vec<EdCandidate>> {
        if number.is_empty() {
            return Ok(Vec::new());
        }
        let terms = [
            format!("Docket Entry {number}"),
            format!("Case No. {number}"),
            format!("No. {number}"),
            number.to_owned(),
        ];
        self.structured_lookup(&terms, &["DOCKET_ENTRY", "CASE"], SOURCE_DOCKET)
    }

    fn lookup_exhibit(
        &self,
        id: &str,
        party_attribution: Option<&str>,
        case_id: Option<&str>,
    ) -> Result<Vec<EdCandidate>> {
        if id.is_empty() {
            return Ok(Vec::new());
        }
        let key = ExhibitKey {
            exhibit_id: id.to_uppercase(),
            party_attribution: party_attribution.map(str::to_owned),
            case_id: case_id.map(str::to_owned),
        };
        let terms = exhibit_key_to_search_terms(&key);
        self.structured_lookup(&terms, &["EXHIBIT", "EXHIBIT_ID"], SOURCE_EXHIBIT)
    }

    fn lookup_statute(&self, key: &StatuteKey) -> Result<Vec<EdCandidate>> {
        if key.section.as_deref().unwrap_or_default().is_empty() {
            return Ok(Vec::new());
        }
        // The lookup key keeps only title, section, subsection and code type.
        let key = StatuteKey {
            title: key.title.clone(),
            section: key.section.clone(),
            subsection: key.subsection.clone(),
            code_type: key.code_type.clone(),
            state: None,
            code_name: None,
        };
        let terms = statute_key_to_search_terms(&key);
        self.structured_lookup(&terms, &["STATUTE"], SOURCE_STATUTE)
    }

    fn lookup_case_citation(&self, key: &CaseCitationKey) -> Result<Vec<EdCandidate>> {
        // The lookup key has no court.
        let key = CaseCitationKey { court: None, ..key.clone() };
        let terms = case_citation_key_to_search_terms(&key);
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        self.structured_lookup(&terms, &["CASE"], SOURCE_CASE)
    }
}

impl EdRetrieval for EdRetriever {
    fn find_by_exact_match(
        &self,
        surface_form: &str,
        entity_type: Option<&str>,
    ) -> Result<Vec<EdCandidate>> {
        let strategy = self.settings.strategy;
        if strategy != EntityRetrievalStrategy::Bm25 {
            return self.find_candidates(
                surface_form,
                entity_type,
                self.settings.max_candidates,
                None,
                Some(strategy),
            );
        }
        let hits = recover_store(self.store.find_entity_by_name(surface_form), "FindEntityByName")?;
        let entity_type = type_filter(entity_type);
        Ok(hits
            .iter()
            .filter_map(|hit| entity_candidate(&hit.node))
            .filter(|candidate| entity_type.is_none_or(|t| candidate.label == t))
            .map(|mut candidate| {
                SOURCE_EXACT.clone_into(&mut candidate.source);
                candidate
            })
            .collect())
    }

    fn find_by_embedding(
        &self,
        query_text: &str,
        limit: usize,
        min_similarity: f64,
    ) -> Result<Vec<EdCandidate>> {
        let Some(embedder) = &self.embedder else {
            return Ok(Vec::new());
        };
        let vector = match embedder.embed(&[query_text]) {
            Ok(mut vectors) if !vectors.is_empty() => vectors.swap_remove(0),
            Ok(_) => {
                tracing::warn!("entity query embedding returned no vector");
                return Ok(Vec::new());
            }
            Err(error) => {
                tracing::warn!(error = %error, "entity query embedding failed");
                return Ok(Vec::new());
            }
        };
        let fetch_limit = if min_similarity > 0.0 { limit.saturating_mul(2) } else { limit };
        let hits = recover_store(
            self.store.find_similar_entities(&vector, fetch_limit),
            "FindSimilarEntities",
        )?;
        let mut candidates = Vec::new();
        for hit in &hits {
            let Some(mut candidate) = entity_candidate(&hit.node) else {
                continue;
            };
            candidate.score = hit.score;
            if min_similarity > 0.0 && candidate.score < min_similarity {
                continue;
            }
            candidates.push(candidate);
        }
        sort_by_score_desc(&mut candidates);
        candidates.truncate(limit);
        Ok(candidates)
    }

    fn find_by_similarity(
        &self,
        surface_form: &str,
        entity_type: Option<&str>,
        limit: usize,
        min_similarity: f64,
    ) -> Result<Vec<EdCandidate>> {
        if surface_form.is_empty() {
            return Ok(Vec::new());
        }
        let hits = recover_store(self.store.find_entity_by_name(surface_form), "FindEntityByName")?;
        let entity_type = type_filter(entity_type);
        let mut scored = Vec::new();
        for hit in &hits {
            let Some(mut candidate) = entity_candidate(&hit.node) else {
                continue;
            };
            if entity_type.is_some_and(|t| candidate.label != t) {
                continue;
            }
            let best = self.best_name_similarity(surface_form, &candidate, entity_type);
            if best >= min_similarity {
                tracing::debug!(
                    surface_form,
                    entity = %candidate.canonical_name,
                    entity_id = %candidate.entity_id,
                    score = best,
                    "similarity match"
                );
                candidate.score = best;
                SOURCE_SIMILARITY.clone_into(&mut candidate.source);
                scored.push(candidate);
            }
        }
        sort_by_score_desc(&mut scored);
        scored.truncate(limit);
        Ok(scored)
    }

    fn find_by_structured_key(
        &self,
        key: &StructuredKeyQuery<'_>,
        case_id: Option<&str>,
    ) -> Result<Vec<EdCandidate>> {
        match *key {
            StructuredKeyQuery::DocketEntry { number } => self.lookup_docket_entry(number),
            StructuredKeyQuery::Exhibit { id, party_attribution } => {
                self.lookup_exhibit(id, party_attribution, case_id)
            }
            StructuredKeyQuery::Statute(key) => self.lookup_statute(key),
            StructuredKeyQuery::CaseCitation(key) => self.lookup_case_citation(key),
        }
    }
}

/// Candidate of a stored entity (Python `_map_node_to_candidate`): aliases parsed from
/// `AliasesJson` (invalid JSON -> none), metadata from `Metadata` (invalid -> empty), source
/// `embedding`, score 0.
#[must_use]
pub fn candidate_from_entity(entity: &StoredEntity) -> EdCandidate {
    let node = &entity.node;
    let aliases = match serde_json::from_str::<Value>(&node.aliases_json) {
        Ok(Value::Array(items)) => items
            .into_iter()
            .filter_map(|item| match item {
                Value::String(alias) => Some(alias),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    let metadata = match serde_json::from_str::<Value>(&node.metadata) {
        Ok(Value::Object(map)) => map,
        _ => Metadata::new(),
    };
    EdCandidate {
        entity_id: entity.id.clone(),
        canonical_name: node.name.clone(),
        label: node.entity_type.clone(),
        aliases,
        description: Some(node.description.clone()),
        score: 0.0,
        source: SOURCE_EMBEDDING.to_owned(),
        metadata,
    }
}

/// [`candidate_from_entity`] for entities with an id and a name; `None` otherwise.
fn entity_candidate(entity: &StoredEntity) -> Option<EdCandidate> {
    (!entity.id.as_str().is_empty() && !entity.node.name.is_empty())
        .then(|| candidate_from_entity(entity))
}

/// The type filter: an empty type means no filter (Python truthiness).
fn type_filter(entity_type: Option<&str>) -> Option<&str> {
    entity_type.filter(|t| !t.is_empty())
}

/// Stable sort by descending score: equal scores keep their order. Scores are finite.
fn sort_by_score_desc(candidates: &mut [EdCandidate]) {
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
}
