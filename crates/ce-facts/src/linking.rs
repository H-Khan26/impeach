//! Fact-to-entity linking and prompt entity context (the helpers of Python `stages/facts.py`).
//!
//! All text offsets are character offsets (mention spans, chunk `start_char`).

use ce_config::MappingsRegistry;
use ce_core::processing::FactLinkingStats;
use ce_core::{EdResult, EntityId, ExtractedFact, Mention, MentionId};
use ce_ed::NameSimilarity;
use ce_ner::definitions::RoleConfig;
use indexmap::{IndexMap, IndexSet};

use crate::extractor::PromptEntity;

/// Canonical name, type and aliases of a chosen ED entity (from its matching candidate).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalEntity {
    /// Canonical name.
    pub name: String,
    /// Candidate label; `ENTITY` when the candidate has none.
    pub entity_type: String,
    /// Candidate aliases.
    pub aliases: Vec<String>,
}

/// Normalised name -> entity id (and -> entity type) used to link fact subjects and objects.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EntityLookup {
    /// Normalised name (and raw / lowercased entity id) -> entity id.
    pub name_to_entity: IndexMap<String, EntityId>,
    /// Normalised name -> entity type.
    pub name_to_type: IndexMap<String, String>,
}

/// `[entity_id:canonical_name]` in `text` -> `(entity_id, canonical_name)`, both trimmed.
#[must_use]
pub fn parse_bracket_token(text: &str) -> Option<(&str, &str)> {
    let _ = text;
    todo!("worker B")
}

/// Entity id -> canonical entity for every ED result with a chosen entity whose candidate list
/// contains it.
#[must_use]
pub fn build_canonical_lookup(ed_results: &[EdResult]) -> IndexMap<EntityId, CanonicalEntity> {
    let _ = ed_results;
    todo!("worker B")
}

/// The chunk text with each resolved mention replaced by `[entity_id:canonical_name]`
/// (right to left; overlapping and out-of-bounds mentions skipped).
#[must_use]
pub fn build_resolved_text_view(
    chunk_text: &str,
    chunk_start_char: usize,
    chunk_mentions: &[Mention],
    resolved_mentions: &IndexMap<MentionId, EntityId>,
    canonical: &IndexMap<EntityId, CanonicalEntity>,
) -> String {
    let _ = (chunk_text, chunk_start_char, chunk_mentions, resolved_mentions, canonical);
    todo!("worker B")
}

/// The prompt entities of one chunk: one per entity resolved in it, canonical name and type
/// from `canonical` (else the first mention's), aliases = surface forms plus ED aliases minus
/// the canonical name, `alias_exclusion` pronouns and role terms; sorted by name.
#[must_use]
pub fn prompt_entities(
    chunk_mentions: &[Mention],
    resolved_mentions: &IndexMap<MentionId, EntityId>,
    canonical: &IndexMap<EntityId, CanonicalEntity>,
    alias_exclusion: &IndexSet<String>,
    roles: Option<&RoleConfig>,
) -> Vec<PromptEntity> {
    let _ = (chunk_mentions, resolved_mentions, canonical, alias_exclusion, roles);
    todo!("worker B")
}

/// Adds one fact's link outcome to the stats.
pub fn record_link(
    stats: &mut FactLinkingStats,
    subject: Option<&EntityId>,
    object: Option<&EntityId>,
) {
    let _ = (stats, subject, object);
    todo!("worker B")
}

/// Links fact subjects and objects to ED entities.
#[derive(Debug, Clone)]
pub struct FactLinker {
    similarity: NameSimilarity,
    roles: Option<RoleConfig>,
}

impl FactLinker {
    /// Linker over the mappings' name matching settings and entity types; `roles` enables
    /// role-term rewriting and `role_map` lookups.
    #[must_use]
    pub fn new(mappings: &MappingsRegistry, roles: Option<RoleConfig>) -> Self {
        Self { similarity: NameSimilarity::from_mappings(mappings), roles }
    }

    /// The role vocabulary, if any.
    #[must_use]
    pub fn roles(&self) -> Option<&RoleConfig> {
        self.roles.as_ref()
    }

    /// Name normalised for linking: person name parts for `PER`, organisation normalisation
    /// for `ORG`, [`ce_ed::normalize_for_matching`] otherwise.
    #[must_use]
    pub fn normalize_name(&self, name: &str, entity_type: Option<&str>) -> String {
        let _ = (&self.similarity, name, entity_type);
        todo!("worker B")
    }

    /// Lookup from ED canonical names and aliases first, then surface forms of resolved
    /// mentions (never overriding an earlier entry).
    #[must_use]
    pub fn build_entity_lookup(
        &self,
        ed_results: &[EdResult],
        resolved_mentions: &IndexMap<MentionId, EntityId>,
        doc_mentions: &[Mention],
    ) -> EntityLookup {
        let _ = (ed_results, resolved_mentions, doc_mentions);
        todo!("worker B")
    }

    /// Links the fact's subject and object, rewriting bracket tokens and linked role terms to
    /// canonical names; returns `(subject_entity_id, object_entity_id)`. Does not set the
    /// fact's entity id fields.
    pub fn link_fact(
        &self,
        fact: &mut ExtractedFact,
        lookup: &EntityLookup,
        canonical: &IndexMap<EntityId, CanonicalEntity>,
        role_map: Option<&IndexMap<String, EntityId>>,
    ) -> (Option<EntityId>, Option<EntityId>) {
        let _ = (fact, lookup, canonical, role_map);
        todo!("worker B")
    }
}
