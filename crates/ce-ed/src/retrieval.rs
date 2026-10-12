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
//! Errors: store failures are logged and yield no candidates, except fatal ones
//! ([`crate::error::recover_store`]); embedder failures yield no candidates.

use std::sync::Arc;

use ce_config::pipeline::{EdStageConfig, EntityRetrievalStrategy};
use ce_core::EdCandidate;
use ce_models::Embedder;
use ce_store::{GraphStore, StoredEntity};

use crate::error::Result;
use crate::name_similarity::NameSimilarity;
use crate::structured_keys::{CaseCitationKey, StatuteKey};

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
        let _ = config;
        todo!("worker B")
    }
}

/// [`EdRetrieval`] over a graph store (Python `HelixEDRetriever`).
pub struct EdRetriever {
    store: Arc<dyn GraphStore>,
    embedder: Option<Arc<dyn Embedder>>,
    settings: RetrieverSettings,
    similarity: NameSimilarity,
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
        let _ = (&store, &embedder, &settings, &similarity);
        todo!("worker B")
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
        let _ = (surface_form, entity_type, limit, min_similarity, strategy);
        todo!("worker B")
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
        let _ = (surface_form, entity_type, limit, min_similarity);
        todo!("worker B")
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
        let _ = (surface_form, entity_type, limit, min_similarity);
        todo!("worker B")
    }

    /// Drops the entity cache of `all_entities_fuzzy` (call after creating entities).
    pub fn invalidate_cache(&self) {
        todo!("worker B")
    }
}

impl EdRetrieval for EdRetriever {
    fn find_by_exact_match(
        &self,
        surface_form: &str,
        entity_type: Option<&str>,
    ) -> Result<Vec<EdCandidate>> {
        let _ = (surface_form, entity_type);
        todo!("worker B")
    }

    fn find_by_embedding(
        &self,
        query_text: &str,
        limit: usize,
        min_similarity: f64,
    ) -> Result<Vec<EdCandidate>> {
        let _ = (query_text, limit, min_similarity);
        todo!("worker B")
    }

    fn find_by_similarity(
        &self,
        surface_form: &str,
        entity_type: Option<&str>,
        limit: usize,
        min_similarity: f64,
    ) -> Result<Vec<EdCandidate>> {
        let _ = (surface_form, entity_type, limit, min_similarity);
        todo!("worker B")
    }

    fn find_by_structured_key(
        &self,
        key: &StructuredKeyQuery<'_>,
        case_id: Option<&str>,
    ) -> Result<Vec<EdCandidate>> {
        let _ = (key, case_id);
        todo!("worker B")
    }
}

/// Candidate of a stored entity (Python `_map_node_to_candidate`): aliases parsed from
/// `AliasesJson` (invalid JSON -> none), metadata from `Metadata` (invalid -> empty), source
/// `embedding`, score 0.
#[must_use]
pub fn candidate_from_entity(entity: &StoredEntity) -> EdCandidate {
    let _ = entity;
    todo!("worker B")
}
