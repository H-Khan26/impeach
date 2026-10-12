//! Cross-document pair pre-filter (Python `services/contradictions/prefilter.py`).
//!
//! A `(new fact, candidate)` pair passes when either path validates (OR logic):
//! - Path A (structural): value type and property type match (taxonomy-normalised), the
//!   subjects overlap (fuzzy similarity >= `subject_overlap_threshold`) and the predicates
//!   are similar (>= `predicate_similarity_threshold`; embedding cosine when an embedder is
//!   set, else token Jaccard). Score: mean of subject and predicate similarity.
//! - Path B (entity / triple): the facts share an entity id (or the candidate came from the
//!   graph source and carries no entity ids), and the best subject and object fuzzy matches
//!   (direct or crossed) both reach `fuzzy_match_threshold`. Score: their mean plus
//!   `min(0.1 * shared, 0.3)`.
//! - Fallbacks when neither path validates: text overlap (`enable_text_overlap_fallback`,
//!   Jaccard >= `min_text_overlap`), else semantic similarity
//!   (`use_semantic_similarity_fallback`, cosine >= `semantic_similarity_threshold`).
//!
//! Priority: `relevance_score * (1 + 1.5 * A + 1.2 * B + 0.8 * fallback)`, times
//! `1 + 0.3 * overlap` when the text Jaccard exceeds 0.2. Pairs are sorted by priority
//! (stable, descending).
//!
//! Config keys the Python accepted but never read, implemented here:
//! - `require_entity_overlap`: when both the new fact and the candidate have entity ids and
//!   they share none, the pair is skipped.
//! - `use_assertion_boost`: a passing pair whose subject / object line up
//!   ([`FactPairPreFilter::same_assertion_entities`]) has its priority multiplied by
//!   [`ASSERTION_BOOST`].

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex};

use ce_config::services::PreFilterConfig;
use ce_config::MappingsRegistry;
use ce_core::{EntityId, ExtractedFact, FactId, RetrievedFact};
use ce_models::Embedder;
use indexmap::IndexMap;

/// Candidate fact id -> entity ids it is linked to.
pub type CandidateEntityMap = IndexMap<FactId, Vec<EntityId>>;

/// A candidate that passed the filter, with its priority (higher is checked first).
pub type ScoredCandidate<'c> = (&'c RetrievedFact, f64);

/// Priority multiplier for pairs whose assertion subject / object line up
/// (`use_assertion_boost`).
pub const ASSERTION_BOOST: f64 = 1.25;

/// Filters and prioritises `(new fact, candidate)` pairs before the judge.
pub struct FactPairPreFilter {
    config: PreFilterConfig,
    embedder: Option<Arc<dyn Embedder>>,
    mappings: &'static MappingsRegistry,
    /// Embeddings of normalised predicates and of fact texts (`None` caches a failure).
    embedding_cache: Mutex<HashMap<String, Option<Vec<f32>>>>,
}

impl fmt::Debug for FactPairPreFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FactPairPreFilter")
            .field("config", &self.config)
            .field("embedder", &self.embedder.as_ref().map(|e| e.model_name().to_owned()))
            .finish_non_exhaustive()
    }
}

impl FactPairPreFilter {
    /// A pre-filter with `config`; `embedder` enables embedding predicate similarity and the
    /// semantic fallback. Tokenisation and name normalisation use `mappings.normalization`,
    /// type comparison `mappings.taxonomy`.
    #[must_use]
    pub fn new(
        config: PreFilterConfig,
        embedder: Option<Arc<dyn Embedder>>,
        mappings: &'static MappingsRegistry,
    ) -> Self {
        Self { config, embedder, mappings, embedding_cache: Mutex::new(HashMap::new()) }
    }

    /// The pre-filter settings.
    #[must_use]
    pub fn config(&self) -> &PreFilterConfig {
        &self.config
    }

    /// The mappings registry used for tokenisation, normalisation and types.
    #[must_use]
    pub fn mappings(&self) -> &'static MappingsRegistry {
        self.mappings
    }

    /// The embedder, if any.
    #[must_use]
    pub fn embedder(&self) -> Option<&Arc<dyn Embedder>> {
        self.embedder.as_ref()
    }

    /// Candidates that pass (priority > 0), sorted by priority, highest first.
    #[must_use]
    pub fn filter_pairs<'c>(
        &self,
        new_fact: &ExtractedFact,
        candidates: &'c [RetrievedFact],
        new_fact_entity_ids: &[EntityId],
        candidate_entity_map: &CandidateEntityMap,
    ) -> Vec<ScoredCandidate<'c>> {
        todo!("worker A")
    }

    /// Priority of one pair; 0.0 means skip (module docs).
    #[must_use]
    pub fn compute_priority(
        &self,
        new_fact: &ExtractedFact,
        candidate: &RetrievedFact,
        new_fact_entity_ids: &HashSet<EntityId>,
        candidate_entity_ids: &HashSet<EntityId>,
    ) -> f64 {
        todo!("worker A")
    }

    /// Path A: `(valid, score)`.
    #[must_use]
    pub fn check_path_a(&self, new_fact: &ExtractedFact, candidate: &RetrievedFact) -> (bool, f64) {
        todo!("worker A")
    }

    /// Path B: `(valid, score)`.
    #[must_use]
    pub fn check_path_b(
        &self,
        new_fact: &ExtractedFact,
        candidate: &RetrievedFact,
        new_fact_entity_ids: &HashSet<EntityId>,
        candidate_entity_ids: &HashSet<EntityId>,
    ) -> (bool, f64) {
        todo!("worker A")
    }

    /// Token set of `text` (lowercased, stopwords removed; mappings normalisation config).
    #[must_use]
    pub fn tokenize(&self, text: &str) -> HashSet<String> {
        todo!("worker A")
    }

    /// Normalised entity name (mappings normalisation config); empty for empty input.
    #[must_use]
    pub fn normalize_entity_name(&self, name: &str) -> String {
        todo!("worker A")
    }

    /// Fuzzy similarity in `[0, 1]`: 0.0 when either is empty, 1.0 when the normalised names
    /// are equal, else the Jaccard of their tokens.
    #[must_use]
    pub fn fuzzy_string_similarity(&self, a: &str, b: &str) -> f64 {
        todo!("worker A")
    }

    /// Predicate similarity: 0.0 when either is empty, 1.0 when equal after lowercasing and
    /// `_` -> space, else embedding cosine (with an embedder) or token Jaccard.
    #[must_use]
    pub fn predicate_similarity(&self, a: &str, b: &str) -> f64 {
        todo!("worker A")
    }

    /// Cosine similarity of the embeddings of two fact texts; 0.0 without an embedder, for
    /// empty texts or when embedding fails.
    #[must_use]
    pub fn text_embedding_similarity(&self, a: &str, b: &str) -> f64 {
        todo!("worker A")
    }

    /// Whether the assertion subject / object of the new fact and the candidate match after
    /// normalisation: directly, crossed, or with at least one entity in common. False when
    /// any of the four is empty.
    #[must_use]
    pub fn same_assertion_entities(
        &self,
        new_fact: &ExtractedFact,
        candidate: &RetrievedFact,
    ) -> bool {
        todo!("worker A")
    }

    /// Entity ids shared by the new fact and the candidate, in `new_fact_entity_ids` order.
    #[must_use]
    pub fn shared_entities(
        &self,
        new_fact_entity_ids: &[EntityId],
        candidate_entity_map: &CandidateEntityMap,
        candidate_fact_id: &FactId,
    ) -> Vec<EntityId> {
        todo!("worker A")
    }
}

/// Cosine similarity; 0.0 for empty vectors, mismatched lengths or a zero norm.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    todo!("worker A")
}
