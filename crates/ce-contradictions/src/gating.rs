//! Config-driven pair gating (Python `services/contradictions/gating.py`).
//!
//! - `dual_path`: delegates to [`FactPairPreFilter::filter_pairs`].
//! - `permissive`: hard filters (value-type mismatch, optional subject entity id mismatch),
//!   then a pair passes when any trigger fires: shared entity anchor (+2.0), semantic
//!   proximity (embedding cosine >= `semantic_proximity_threshold`, + 1.5 * cosine) or keyword
//!   overlap (token Jaccard >= `keyword_overlap_threshold`, + overlap). Priority:
//!   `relevance_score * (1 + triggers)`, sorted stable descending.
//!
//! The shared-entity trigger reads the `candidate_entity_map` built by
//! [`crate::SimilarFactRetriever::candidate_entity_map`] (Python always passed an empty map, so
//! the trigger never fired).

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex};

use ce_config::pipeline::{GatingStrategy, VerificationStrategyConfig};
use ce_core::util::lock;
use ce_core::{EntityId, ExtractedFact, RetrievedFact, ValueType};
use ce_models::Embedder;
use serde_json::Value;

use crate::nli_context::py_truthy;
use crate::prefilter::{cosine_similarity, CandidateEntityMap, FactPairPreFilter, ScoredCandidate};

/// Value types that pair with any other when `allow_text_value_type_mismatch` is set.
const WILDCARD_VALUE_TYPES: [&str; 2] = ["TEXT", "UNKNOWN"];

/// Trigger score of a shared entity anchor.
const SHARED_ENTITY_TRIGGER: f64 = 2.0;

/// Weight of the semantic proximity trigger.
const SEMANTIC_TRIGGER_WEIGHT: f64 = 1.5;

/// Gating over the configured strategy.
pub struct ContradictionGating {
    config: VerificationStrategyConfig,
    embedder: Option<Arc<dyn Embedder>>,
    prefilter: Arc<FactPairPreFilter>,
    /// Fact-text embeddings (`None` caches a failure).
    embedding_cache: Mutex<HashMap<String, Option<Vec<f32>>>>,
}

impl fmt::Debug for ContradictionGating {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContradictionGating")
            .field("config", &self.config)
            .field("embedder", &self.embedder.as_ref().map(|e| e.model_name().to_owned()))
            .field("prefilter", &self.prefilter)
            .finish_non_exhaustive()
    }
}

impl ContradictionGating {
    /// Gating with `config`; `embedder` enables the semantic proximity trigger; `prefilter`
    /// serves the `dual_path` strategy and type normalisation.
    #[must_use]
    pub fn new(
        config: VerificationStrategyConfig,
        embedder: Option<Arc<dyn Embedder>>,
        prefilter: Arc<FactPairPreFilter>,
    ) -> Self {
        Self { config, embedder, prefilter, embedding_cache: Mutex::new(HashMap::new()) }
    }

    /// The verification settings.
    #[must_use]
    pub fn config(&self) -> &VerificationStrategyConfig {
        &self.config
    }

    /// Candidates that pass the configured strategy, highest priority first.
    #[must_use]
    pub fn filter_pairs<'c>(
        &self,
        new_fact: &ExtractedFact,
        candidates: &'c [RetrievedFact],
        new_fact_entity_ids: &[EntityId],
        candidate_entity_map: &CandidateEntityMap,
    ) -> Vec<ScoredCandidate<'c>> {
        match self.config.gating_strategy {
            GatingStrategy::DualPath => self.prefilter.filter_pairs(
                new_fact,
                candidates,
                new_fact_entity_ids,
                candidate_entity_map,
            ),
            GatingStrategy::Permissive => self.permissive_gating(
                new_fact,
                candidates,
                new_fact_entity_ids,
                candidate_entity_map,
            ),
        }
    }

    /// Hard filters, then permissive triggers; stable sort by descending priority.
    fn permissive_gating<'c>(
        &self,
        new_fact: &ExtractedFact,
        candidates: &'c [RetrievedFact],
        new_fact_entity_ids: &[EntityId],
        candidate_entity_map: &CandidateEntityMap,
    ) -> Vec<ScoredCandidate<'c>> {
        let new_ids: HashSet<&EntityId> = new_fact_entity_ids.iter().collect();
        let mut results: Vec<ScoredCandidate<'c>> = Vec::new();
        for candidate in candidates {
            if !self.passes_hard_filters(new_fact, candidate) {
                continue;
            }
            let shares_entity = candidate_entity_map
                .get(&candidate.fact_id)
                .is_some_and(|ids| ids.iter().any(|id| new_ids.contains(id)));
            let priority = self.permissive_priority(new_fact, candidate, shares_entity);
            if priority > 0.0 {
                results.push((candidate, priority));
            }
        }
        results.sort_by(|a, b| b.1.total_cmp(&a.1));
        results
    }

    /// Whether the pair survives the value-type and subject-entity hard filters.
    fn passes_hard_filters(&self, new_fact: &ExtractedFact, candidate: &RetrievedFact) -> bool {
        let new_vt = new_fact.value_type.as_deref().filter(|vt| !vt.is_empty());
        let cand_vt = candidate.metadata.get("value_type").and_then(Value::as_str);
        if let (Some(new_vt), Some(cand_vt)) = (new_vt, cand_vt.filter(|vt| !vt.is_empty())) {
            let taxonomy = &self.prefilter.mappings().taxonomy;
            let new_vt = ValueType::from_raw(Some(new_vt), taxonomy);
            let cand_vt = ValueType::from_raw(Some(cand_vt), taxonomy);
            let wildcard = |vt: &ValueType| WILDCARD_VALUE_TYPES.contains(&vt.as_str());
            let exempt = self.config.allow_text_value_type_mismatch
                && (wildcard(&new_vt) || wildcard(&cand_vt));
            if !exempt && new_vt != cand_vt {
                return false;
            }
        }

        if self.config.require_entity_id_match {
            let new_id = new_fact.subject_entity_id.as_ref().filter(|id| !id.is_empty());
            let cand_id = candidate.metadata.get("subject_entity_id").filter(|v| py_truthy(v));
            if let (Some(new_id), Some(cand_id)) = (new_id, cand_id) {
                if cand_id.as_str() != Some(new_id.as_str()) {
                    return false;
                }
            }
        }
        true
    }

    /// `relevance_score * (1 + triggers)`, or 0.0 when no trigger fires.
    fn permissive_priority(
        &self,
        new_fact: &ExtractedFact,
        candidate: &RetrievedFact,
        shares_entity: bool,
    ) -> f64 {
        let mut trigger_score = 0.0;
        if shares_entity {
            trigger_score += SHARED_ENTITY_TRIGGER;
        }
        if self.embedder.is_some() {
            let similarity = self.semantic_similarity(&new_fact.text, &candidate.fact_text);
            if similarity >= self.config.semantic_proximity_threshold {
                trigger_score += similarity * SEMANTIC_TRIGGER_WEIGHT;
            }
        }
        // Lowercased alphanumeric tokens longer than one character, no stopword removal.
        let overlap = ce_text::keyword_overlap(&new_fact.text, &candidate.fact_text);
        if overlap >= self.config.keyword_overlap_threshold {
            trigger_score += overlap;
        }
        if trigger_score == 0.0 {
            return 0.0;
        }
        candidate.relevance_score * (1.0 + trigger_score)
    }

    /// Embedding cosine of two texts; 0.0 for empty texts or failed embeddings.
    fn semantic_similarity(&self, a: &str, b: &str) -> f64 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        match (self.cached_embedding(a), self.cached_embedding(b)) {
            (Some(ea), Some(eb)) => cosine_similarity(&ea, &eb),
            _ => 0.0,
        }
    }

    /// The embedding of `text`, computed once (failures are cached as `None`).
    fn cached_embedding(&self, text: &str) -> Option<Vec<f32>> {
        if let Some(cached) = lock(&self.embedding_cache).get(text) {
            return cached.clone();
        }
        let embedder = self.embedder.as_ref()?;
        let embedding = match embedder.embed(&[text]) {
            Ok(vectors) => vectors.into_iter().next(),
            Err(error) => {
                tracing::debug!("Failed to compute embedding: {error}");
                None
            }
        };
        lock(&self.embedding_cache).insert(text.to_owned(), embedding.clone());
        embedding
    }
}
