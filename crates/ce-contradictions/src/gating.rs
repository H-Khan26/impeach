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

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use ce_config::pipeline::VerificationStrategyConfig;
use ce_core::{EntityId, ExtractedFact, RetrievedFact};
use ce_models::Embedder;

use crate::prefilter::{CandidateEntityMap, FactPairPreFilter, ScoredCandidate};

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
        todo!("worker B")
    }
}
