//! Cross-document pair pre-filter (Python `services/contradictions/prefilter.py`).
//!
//! A `(new fact, candidate)` pair passes when either path validates (OR logic):
//! - Path A (structural): value type and property type match (taxonomy-normalised), the
//!   subjects overlap (fuzzy similarity >= `subject_overlap_threshold`) and the predicates
//!   are similar (>= `predicate_similarity_threshold`; embedding cosine when an embedder is
//!   set, else token Jaccard). Score: mean of subject and predicate similarity. The
//!   candidate's types are read from its metadata keys `value_type` / `property_type`.
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
//!
//! Embeddings (of normalised predicates and of fact texts) are cached by text; a failed
//! embedding is cached as missing and scores 0.0.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use impeach_config::services::PreFilterConfig;
use impeach_config::MappingsRegistry;
use impeach_core::{EntityId, ExtractedFact, FactId, PropertyType, RetrievedFact, ValueType};
use impeach_models::Embedder;
use impeach_text::jaccard;
use impeach_text::normalize::py_strip;
use indexmap::IndexMap;

use crate::retrieval::SOURCE_GRAPH;

/// Candidate fact id -> entity ids it is linked to.
pub type CandidateEntityMap = IndexMap<FactId, Vec<EntityId>>;

/// A candidate that passed the filter, with its priority (higher is checked first).
pub type ScoredCandidate<'c> = (&'c RetrievedFact, f64);

/// Priority multiplier for pairs whose assertion subject / object line up
/// (`use_assertion_boost`).
pub const ASSERTION_BOOST: f64 = 1.25;

/// Path A priority weight.
const PATH_A_WEIGHT: f64 = 1.5;
/// Path B priority weight.
const PATH_B_WEIGHT: f64 = 1.2;
/// Fallback priority weight.
const FALLBACK_WEIGHT: f64 = 0.8;
/// Text Jaccard above which the overlap boost applies.
const TEXT_OVERLAP_BOOST_MIN: f64 = 0.2;
/// Overlap boost factor.
const TEXT_OVERLAP_BOOST: f64 = 0.3;
/// Path B boost per shared entity.
const ENTITY_BOOST_PER_SHARED: f64 = 0.1;
/// Cap of the Path B shared-entity boost.
const ENTITY_BOOST_CAP: f64 = 0.3;

type EmbeddingCache = HashMap<String, Option<Vec<f32>>>;

/// Filters and prioritises `(new fact, candidate)` pairs before the judge.
pub struct FactPairPreFilter {
    config: PreFilterConfig,
    embedder: Option<Arc<dyn Embedder>>,
    mappings: &'static MappingsRegistry,
    /// Embeddings of normalised predicates and of fact texts (`None` caches a failure).
    embedding_cache: Mutex<EmbeddingCache>,
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
        let new_ids: HashSet<EntityId> = new_fact_entity_ids.iter().cloned().collect();
        let mut results: Vec<ScoredCandidate<'c>> = candidates
            .iter()
            .filter_map(|candidate| {
                let candidate_ids: HashSet<EntityId> = candidate_entity_map
                    .get(&candidate.fact_id)
                    .map(|ids| ids.iter().cloned().collect())
                    .unwrap_or_default();
                let priority = self.compute_priority(new_fact, candidate, &new_ids, &candidate_ids);
                (priority > 0.0).then_some((candidate, priority))
            })
            .collect();
        results.sort_by(|a, b| b.1.total_cmp(&a.1));
        results
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
        if self.config.require_entity_overlap
            && !new_fact_entity_ids.is_empty()
            && !candidate_entity_ids.is_empty()
            && new_fact_entity_ids.is_disjoint(candidate_entity_ids)
        {
            return 0.0;
        }

        let text_overlap =
            jaccard(&self.tokenize(&new_fact.text), &self.tokenize(&candidate.fact_text));
        let (path_a_valid, path_a_score) = self.check_path_a(new_fact, candidate);
        let (path_b_valid, path_b_score) =
            self.check_path_b(new_fact, candidate, new_fact_entity_ids, candidate_entity_ids);

        let mut fallback: Option<f64> = None;
        if !path_a_valid && !path_b_valid {
            if self.config.enable_text_overlap_fallback
                && text_overlap >= self.config.min_text_overlap
            {
                fallback = Some(text_overlap);
            } else if self.config.use_semantic_similarity_fallback && self.embedder.is_some() {
                let semantic = self.text_embedding_similarity(&new_fact.text, &candidate.fact_text);
                if semantic >= self.config.semantic_similarity_threshold {
                    fallback = Some(semantic);
                }
            }
            if fallback.is_none() {
                return 0.0;
            }
        }

        let mut path_boost = 0.0;
        if path_a_valid {
            path_boost += path_a_score * PATH_A_WEIGHT;
        }
        if path_b_valid {
            path_boost += path_b_score * PATH_B_WEIGHT;
        }
        if let Some(score) = fallback {
            path_boost += score * FALLBACK_WEIGHT;
        }

        let mut priority = candidate.relevance_score * (1.0 + path_boost);
        if text_overlap > TEXT_OVERLAP_BOOST_MIN {
            priority *= 1.0 + text_overlap * TEXT_OVERLAP_BOOST;
        }
        if self.config.use_assertion_boost && self.same_assertion_entities(new_fact, candidate) {
            priority *= ASSERTION_BOOST;
        }
        priority
    }

    /// Path A: `(valid, score)`.
    #[must_use]
    pub fn check_path_a(&self, new_fact: &ExtractedFact, candidate: &RetrievedFact) -> (bool, f64) {
        let (Some(new_vt), Some(new_pt)) = (
            non_empty(new_fact.value_type.as_deref()),
            non_empty(new_fact.property_type.as_deref()),
        ) else {
            return (false, 0.0);
        };
        let metadata_str =
            |key: &str| non_empty(candidate.metadata.get(key).and_then(|v| v.as_str()));
        let (Some(cand_vt), Some(cand_pt)) =
            (metadata_str("value_type"), metadata_str("property_type"))
        else {
            return (false, 0.0);
        };
        if !self.types_match((new_vt, new_pt), (cand_vt, cand_pt)) {
            return (false, 0.0);
        }

        let triple = &new_fact.assertion_triple;
        let subject_sim = self.fuzzy_string_similarity(
            &triple.subject,
            candidate.assertion_subject.as_deref().unwrap_or(""),
        );
        if subject_sim < self.config.subject_overlap_threshold {
            return (false, 0.0);
        }
        let predicate_sim = self.predicate_similarity(
            &triple.relationship,
            candidate.assertion_relationship.as_deref().unwrap_or(""),
        );
        if predicate_sim < self.config.predicate_similarity_threshold {
            return (false, 0.0);
        }
        (true, (subject_sim + predicate_sim) / 2.0)
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
        let mut shared = new_fact_entity_ids.intersection(candidate_entity_ids).count();
        // A graph-retrieved candidate without entity ids counts as one implicit shared entity.
        if shared == 0
            && candidate_entity_ids.is_empty()
            && candidate.sources.iter().any(|s| s == SOURCE_GRAPH)
            && !new_fact_entity_ids.is_empty()
        {
            shared = 1;
        }
        if shared == 0 {
            return (false, 0.0);
        }

        let triple = &new_fact.assertion_triple;
        let candidate_triple = (
            candidate.assertion_subject.as_deref().unwrap_or(""),
            candidate.assertion_object.as_deref().unwrap_or(""),
        );
        match self.best_triple_match((&triple.subject, &triple.object), candidate_triple) {
            Some(avg) => {
                let boost = (shared as f64 * ENTITY_BOOST_PER_SHARED).min(ENTITY_BOOST_CAP);
                (true, avg + boost)
            }
            None => (false, 0.0),
        }
    }

    /// Token set of `text` (lowercased, stopwords removed; mappings normalisation config).
    #[must_use]
    pub fn tokenize(&self, text: &str) -> HashSet<String> {
        self.mappings.normalization.tokenize(text)
    }

    /// Normalised entity name (mappings normalisation config); empty for empty input.
    #[must_use]
    pub fn normalize_entity_name(&self, name: &str) -> String {
        if name.is_empty() {
            return String::new();
        }
        self.mappings.normalization.normalize_entity_name(name)
    }

    /// Fuzzy similarity in `[0, 1]`: 0.0 when either is empty, 1.0 when the normalised names
    /// are equal, else the Jaccard of their tokens.
    #[must_use]
    pub fn fuzzy_string_similarity(&self, a: &str, b: &str) -> f64 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        let norm_a = self.normalize_entity_name(a);
        let norm_b = self.normalize_entity_name(b);
        if norm_a == norm_b {
            return 1.0;
        }
        jaccard(&self.tokenize(&norm_a), &self.tokenize(&norm_b))
    }

    /// Predicate similarity: 0.0 when either is empty, 1.0 when equal after lowercasing and
    /// `_` -> space, else embedding cosine (with an embedder) or token Jaccard.
    #[must_use]
    pub fn predicate_similarity(&self, a: &str, b: &str) -> f64 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        let norm = |p: &str| py_strip(&p.to_lowercase().replace('_', " ")).to_owned();
        let (norm_a, norm_b) = (norm(a), norm(b));
        if norm_a == norm_b {
            return 1.0;
        }
        if self.embedder.is_some() {
            return self.embedding_similarity(&norm_a, &norm_b);
        }
        jaccard(&self.tokenize(&norm_a), &self.tokenize(&norm_b))
    }

    /// Cosine similarity of the embeddings of two fact texts; 0.0 without an embedder, for
    /// empty texts or when embedding fails.
    #[must_use]
    pub fn text_embedding_similarity(&self, a: &str, b: &str) -> f64 {
        if a.is_empty() || b.is_empty() || self.embedder.is_none() {
            return 0.0;
        }
        self.embedding_similarity(a, b)
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
        let new_subject = self.normalize_entity_name(&new_fact.assertion_triple.subject);
        let new_object = self.normalize_entity_name(&new_fact.assertion_triple.object);
        let cand_subject =
            self.normalize_entity_name(candidate.assertion_subject.as_deref().unwrap_or(""));
        let cand_object =
            self.normalize_entity_name(candidate.assertion_object.as_deref().unwrap_or(""));
        if new_subject.is_empty()
            || new_object.is_empty()
            || cand_subject.is_empty()
            || cand_object.is_empty()
        {
            return false;
        }
        // The direct and crossed matches are special cases of one entity in common.
        new_subject == cand_subject
            || new_subject == cand_object
            || new_object == cand_subject
            || new_object == cand_object
    }

    /// Entity ids shared by the new fact and the candidate, in `new_fact_entity_ids` order.
    #[must_use]
    pub fn shared_entities(
        &self,
        new_fact_entity_ids: &[EntityId],
        candidate_entity_map: &CandidateEntityMap,
        candidate_fact_id: &FactId,
    ) -> Vec<EntityId> {
        let Some(candidate_ids) = candidate_entity_map.get(candidate_fact_id) else {
            return Vec::new();
        };
        let mut shared: Vec<EntityId> = Vec::new();
        for id in new_fact_entity_ids {
            if candidate_ids.contains(id) && !shared.contains(id) {
                shared.push(id.clone());
            }
        }
        shared
    }

    /// Whether two raw `(value type, property type)` pairs are equal after taxonomy
    /// normalisation.
    pub(crate) fn types_match(&self, a: (&str, &str), b: (&str, &str)) -> bool {
        let vocab = &self.mappings.taxonomy;
        ValueType::from_raw(Some(a.0), vocab) == ValueType::from_raw(Some(b.0), vocab)
            && PropertyType::from_raw(Some(a.1), vocab) == PropertyType::from_raw(Some(b.1), vocab)
    }

    /// Mean of the best subject and best object fuzzy matches (direct or crossed) of two
    /// `(subject, object)` pairs when both reach `fuzzy_match_threshold`, else `None`.
    pub(crate) fn best_triple_match(&self, a: (&str, &str), b: (&str, &str)) -> Option<f64> {
        let subject_sim = self.fuzzy_string_similarity(a.0, b.0);
        let object_sim = self.fuzzy_string_similarity(a.1, b.1);
        let cross_subj_obj = self.fuzzy_string_similarity(a.0, b.1);
        let cross_obj_subj = self.fuzzy_string_similarity(a.1, b.0);
        let best_subject = subject_sim.max(cross_obj_subj);
        let best_object = object_sim.max(cross_subj_obj);
        let threshold = self.config.fuzzy_match_threshold;
        (best_subject >= threshold && best_object >= threshold)
            .then(|| (best_subject + best_object) / 2.0)
    }

    /// Cosine of the cached embeddings of two texts; 0.0 when either embedding fails.
    fn embedding_similarity(&self, a: &str, b: &str) -> f64 {
        match (self.cached_embedding(a), self.cached_embedding(b)) {
            (Some(emb_a), Some(emb_b)) => cosine_similarity(&emb_a, &emb_b),
            _ => 0.0,
        }
    }

    /// The embedding of `text`, computed once; `None` without an embedder or on failure.
    fn cached_embedding(&self, text: &str) -> Option<Vec<f32>> {
        let embedder = self.embedder.as_ref()?;
        if let Some(cached) = self.cache().get(text) {
            return cached.clone();
        }
        let embedding = match embedder.embed(&[text]) {
            Ok(vectors) => vectors.into_iter().next(),
            Err(error) => {
                tracing::debug!("Failed to compute embedding for {text:?}: {error}");
                None
            }
        };
        self.cache().insert(text.to_owned(), embedding.clone());
        embedding
    }

    fn cache(&self) -> MutexGuard<'_, EmbeddingCache> {
        self.embedding_cache.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// `Some(value)` when it is present and non-empty.
fn non_empty(value: Option<&str>) -> Option<&str> {
    value.filter(|s| !s.is_empty())
}

/// Cosine similarity; 0.0 for empty vectors, mismatched lengths or a zero norm.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_edge_cases() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 0.0]), 0.0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-12);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-12);
        assert!((cosine_similarity(&[1.0, 1.0], &[-1.0, -1.0]) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn non_empty_filters_blank_values() {
        assert_eq!(non_empty(None), None);
        assert_eq!(non_empty(Some("")), None);
        assert_eq!(non_empty(Some("TEXT")), Some("TEXT"));
    }
}
