//! The contradiction check service (Python `services/contradictions/service.py`).
//!
//! [`ContradictionCheckService::check_facts_for_contradictions`] runs three phases over the
//! facts of one document:
//! 1. Intra-document (when `intra_document_enabled` and more than one fact; skipped with a
//!    warning when `n * (n - 1) / 2 > max_intra_doc_pairs`): bucketed candidate pairs
//!    ([`IntraDocFilter`]), pre-filter, judge on enriched text of both facts. A contradiction
//!    label at or above `review_threshold` makes one pair (lower fact id as the "new" fact,
//!    `retrieval_sources = ["intra_document"]`, source `intra_document`, priority from the
//!    filter decision); both facts get a result (`contradiction` when any of their pairs
//!    reaches `contradiction_threshold`, else `needs_review`, `retrieval_stats` `{"source":
//!    "intra_document", "pairs_checked": n}`) and leave the clean set. Each pair is added to
//!    `unique_contradiction_pairs` once.
//! 2. Cross-document, per fact still clean: retrieval with the fact's own subject / object
//!    entity ids (else the document's), the candidate entity map from `FactAboutEntity` edges,
//!    gating (verification strategy) or the pre-filter, the judge on enriched premise
//!    (retrieved fact) / hypothesis (new fact) for every passing pair. Contradiction labels at
//!    or above `contradiction_threshold` are contradictions, at or above `review_threshold`
//!    needs-review; status `contradiction` (pairs: contradictions then reviews), else
//!    `needs_review`, else `clean`. Cross-document pairs carry the shared entity ids.
//! 3. Commit: the clean facts through [`ce_facts::FactIndexer::commit_facts`] (one batch, or
//!    one call per fact with `commit_as_we_go`); committed ids are the written graph ids.
//!
//! The report carries totals (candidates, pairs after pre-filter, judge calls), the
//! processing time and a debug trace entry per pair considered (Python keys: `phase`,
//! `fact_a_id`, ..., `outcome`, `reason`, `nli_label`, `nli_confidence`).

use std::fmt;
use std::sync::Arc;

use ce_config::pipeline::VerificationStrategyConfig;
use ce_config::services::ContradictionCheckConfig;
use ce_config::MappingsRegistry;
use ce_core::{ContradictionReport, DocId, EntityId, ExtractedFact, FactId};
use ce_models::Embedder;
use ce_store::GraphStore;
use indexmap::IndexMap;

use crate::error::Result;
use crate::judge::ContradictionJudge;

/// Orchestrates the contradiction check for the facts of one document.
pub struct ContradictionCheckService {
    store: Arc<dyn GraphStore>,
    embedder: Arc<dyn Embedder>,
    judge: Arc<dyn ContradictionJudge>,
    config: ContradictionCheckConfig,
    verification: Option<VerificationStrategyConfig>,
    mappings: &'static MappingsRegistry,
    entity_names: IndexMap<EntityId, String>,
}

impl fmt::Debug for ContradictionCheckService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContradictionCheckService")
            .field("store", &self.store)
            .field("embedder", &self.embedder.model_name())
            .field("judge", &self.judge)
            .field("config", &self.config)
            .field("verification", &self.verification)
            .finish_non_exhaustive()
    }
}

impl ContradictionCheckService {
    /// A service over `store`, embedding with `embedder` and judging with `judge`.
    /// `verification` selects the gating strategy (`None`: the dual-path pre-filter).
    #[must_use]
    pub fn new(
        store: Arc<dyn GraphStore>,
        embedder: Arc<dyn Embedder>,
        judge: Arc<dyn ContradictionJudge>,
        config: ContradictionCheckConfig,
        verification: Option<VerificationStrategyConfig>,
        mappings: &'static MappingsRegistry,
    ) -> Self {
        Self {
            store,
            embedder,
            judge,
            config,
            verification,
            mappings,
            entity_names: IndexMap::new(),
        }
    }

    /// Sets the entity id -> canonical name map used to enrich judge inputs.
    #[must_use]
    pub fn with_entity_names(mut self, entity_names: IndexMap<EntityId, String>) -> Self {
        self.set_entity_names(entity_names);
        self
    }

    /// Replaces the entity id -> canonical name map.
    pub fn set_entity_names(&mut self, entity_names: IndexMap<EntityId, String>) {
        self.entity_names = entity_names;
    }

    /// The service settings.
    #[must_use]
    pub fn config(&self) -> &ContradictionCheckConfig {
        &self.config
    }

    /// Checks `facts` (module docs). `entity_ids` are the document's resolved entities (the
    /// retrieval fallback for facts without linked entities).
    ///
    /// # Errors
    /// Fatal store errors and commit errors.
    pub fn check_facts_for_contradictions(
        &self,
        facts: &[ExtractedFact],
        entity_ids: &[EntityId],
        doc_id: &DocId,
    ) -> Result<ContradictionReport> {
        todo!("worker C")
    }

    /// Commits `facts` in one batch; returns the graph ids of the facts written (each fact's
    /// `metadata.helix_fact_id` is set).
    ///
    /// # Errors
    /// Fatal store errors.
    pub fn commit_clean_facts(&self, facts: &mut [ExtractedFact]) -> Result<Vec<FactId>> {
        todo!("worker C")
    }
}
