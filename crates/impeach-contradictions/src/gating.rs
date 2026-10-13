//! Config-driven pair gating (Python `services/contradictions/gating.py`).
//!
//! - `dual_path`: delegates to [`FactPairPreFilter::filter_pairs`].
//! - `permissive`: hard filters (value-type mismatch, optional subject entity id mismatch),
//!   then a pair passes when any trigger fires: shared entity anchor (+2.0), semantic
//!   proximity (embedding cosine >= `semantic_proximity_threshold`, + 1.5 * cosine) or keyword
//!   overlap (token Jaccard >= `keyword_overlap_threshold`, + overlap). Priority:
//!   `relevance_score * (1 + triggers)`, sorted stable descending. Fact-text embeddings come
//!   from the pre-filter (its embedder and cache); without an embedder the semantic trigger
//!   never fires.
//!
//! The shared-entity trigger reads the `candidate_entity_map` built by
//! [`crate::SimilarFactRetriever::candidate_entity_map`] (Python always passed an empty map, so
//! the trigger never fired).

use std::collections::HashSet;
use std::sync::Arc;

use impeach_config::pipeline::{GatingStrategy, VerificationStrategyConfig};
use impeach_core::{EntityId, ExtractedFact, RetrievedFact, ValueType};
use serde_json::Value;

use crate::nli_context::py_truthy;
use crate::prefilter::{CandidateEntityMap, FactPairPreFilter, ScoredCandidate};

/// Value types that pair with any other when `allow_text_value_type_mismatch` is set.
const WILDCARD_VALUE_TYPES: [&str; 2] = ["TEXT", "UNKNOWN"];

/// Trigger score of a shared entity anchor.
const SHARED_ENTITY_TRIGGER: f64 = 2.0;

/// Weight of the semantic proximity trigger.
const SEMANTIC_TRIGGER_WEIGHT: f64 = 1.5;

/// Gating over the configured strategy.
#[derive(Debug)]
pub struct ContradictionGating {
    config: VerificationStrategyConfig,
    prefilter: Arc<FactPairPreFilter>,
}

impl ContradictionGating {
    /// Gating with `config`; `prefilter` serves the `dual_path` strategy, type normalisation
    /// and fact-text embeddings (its embedder enables the semantic proximity trigger).
    #[must_use]
    pub fn new(config: VerificationStrategyConfig, prefilter: Arc<FactPairPreFilter>) -> Self {
        Self { config, prefilter }
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
        if self.prefilter.embedder().is_some() {
            let similarity =
                self.prefilter.text_embedding_similarity(&new_fact.text, &candidate.fact_text);
            if similarity >= self.config.semantic_proximity_threshold {
                trigger_score += similarity * SEMANTIC_TRIGGER_WEIGHT;
            }
        }
        // Lowercased alphanumeric tokens longer than one character, no stopword removal.
        let overlap = impeach_text::keyword_overlap(&new_fact.text, &candidate.fact_text);
        if overlap >= self.config.keyword_overlap_threshold {
            trigger_score += overlap;
        }
        if trigger_score == 0.0 {
            return 0.0;
        }
        candidate.relevance_score * (1.0 + trigger_score)
    }
}
