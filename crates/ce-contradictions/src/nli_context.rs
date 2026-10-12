//! Premise / hypothesis construction for the judge (Python
//! `services/contradictions/nli_context.py`).
//!
//! Enriched text: `Subject: <name> | Object: <name> | Time: <when> | Speaker: <who> | <text>`,
//! each prefix only when known. For an extracted fact the subject / object name is the
//! canonical name of its linked entity, else the trimmed assertion subject / object; time is
//! `t_valid_from [to t_valid_until]` (or `until t_valid_until`), else metadata `date` /
//! `event_date`. For a retrieved fact the subject / object come from its assertion triple and
//! time from metadata (`TemporalValidity` object, else `date` / `event_date`).

use ce_core::{EdResult, EntityId, ExtractedFact, RetrievedFact};
use indexmap::IndexMap;

/// Builds context-enriched fact text for the judge.
#[derive(Debug, Clone, Default)]
pub struct NliContextBuilder {
    entity_names: IndexMap<EntityId, String>,
}

impl NliContextBuilder {
    /// A builder resolving entity ids with `entity_names` (entity id -> canonical name).
    #[must_use]
    pub fn new(entity_names: IndexMap<EntityId, String>) -> Self {
        Self { entity_names }
    }

    /// Entity id -> canonical name.
    #[must_use]
    pub fn entity_names(&self) -> &IndexMap<EntityId, String> {
        &self.entity_names
    }

    /// Enriched text of an extracted fact (module docs).
    #[must_use]
    pub fn build_enriched_text(&self, fact: &ExtractedFact) -> String {
        todo!("worker B")
    }

    /// Enriched text of a retrieved fact (module docs).
    #[must_use]
    pub fn build_enriched_text_for_retrieved(&self, fact: &RetrievedFact) -> String {
        todo!("worker B")
    }
}

/// Entity id -> canonical name from ED results: for each result with a chosen entity, the
/// name of the candidate with that id (first match kept per result; later results overwrite).
#[must_use]
pub fn build_entity_name_map(ed_results: &[EdResult]) -> IndexMap<EntityId, String> {
    todo!("worker B")
}
