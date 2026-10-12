//! Fact-to-entity linking and prompt entity context (the helpers of Python `stages/facts.py`).
//!
//! All text offsets are character offsets (mention spans, chunk `start_char`).
//!
//! Linking one side (subject or object) of a fact's triple, in order:
//! 1. a `[entity_id:canonical_name]` bracket token (from the resolved-text view) links to that
//!    id and the side is rewritten to the canonical name;
//! 2. the untyped normalised name, then the `PER` and `ORG` normalisations, looked up in the
//!    [`EntityLookup`]; a linked side that is a role term (`Plaintiff`) is rewritten to the
//!    entity's canonical name;
//! 3. a still unlinked role term is looked up in the EDStage `role_map` (keyed by the
//!    normalised role term) and rewritten to the canonical name when known. Python did not use
//!    the role map for linking.

use std::sync::LazyLock;

use ce_config::MappingsRegistry;
use ce_core::processing::FactLinkingStats;
use ce_core::{EdResult, EntityId, ExtractedFact, Mention, MentionId};
use ce_ed::{normalize_for_matching, NameSimilarity};
use ce_ner::definitions::RoleConfig;
use ce_text::normalize::py_strip;
use indexmap::{IndexMap, IndexSet};
use regex::Regex;

use crate::extractor::PromptEntity;

/// `[entity_id:canonical_name]` (Python `_BRACKET_TOKEN_PATTERN`).
static BRACKET_TOKEN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\[(?P<entity_id>[^:\]]+):(?P<canonical_name>[^\]]+)\]")
        .expect("bracket token pattern is a valid regex")
});

/// Entity type of a [`CanonicalEntity`] whose candidate has no label.
const DEFAULT_CANONICAL_TYPE: &str = "ENTITY";

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

impl EntityLookup {
    /// Adds `normalized -> entity_id` (and its type); an existing entry is replaced only when
    /// `overwrite` is set. Empty names are ignored.
    fn add(
        &mut self,
        normalized: String,
        entity_id: &EntityId,
        entity_type: &str,
        overwrite: bool,
    ) {
        if normalized.is_empty() || (!overwrite && self.name_to_entity.contains_key(&normalized)) {
            return;
        }
        if !entity_type.is_empty() {
            self.name_to_type.insert(normalized.clone(), entity_type.to_owned());
        }
        self.name_to_entity.insert(normalized, entity_id.clone());
    }
}

/// `[entity_id:canonical_name]` in `text` -> `(entity_id, canonical_name)`, both trimmed.
#[must_use]
pub fn parse_bracket_token(text: &str) -> Option<(&str, &str)> {
    if !text.contains('[') {
        return None;
    }
    let captures = BRACKET_TOKEN.captures(text)?;
    let entity_id = captures.name("entity_id")?.as_str();
    let canonical_name = captures.name("canonical_name")?.as_str();
    Some((py_strip(entity_id), py_strip(canonical_name)))
}

/// The candidate of `result` describing its chosen entity (the first with the same id).
fn chosen_candidate(result: &EdResult) -> Option<(&EntityId, &ce_core::EdCandidate)> {
    let entity_id = result.chosen_entity_id.as_ref().filter(|id| !id.as_str().is_empty())?;
    let candidate = result.candidates.iter().find(|c| &c.entity_id == entity_id)?;
    Some((entity_id, candidate))
}

/// Entity id -> canonical entity for every ED result with a chosen entity whose candidate list
/// contains it.
#[must_use]
pub fn build_canonical_lookup(ed_results: &[EdResult]) -> IndexMap<EntityId, CanonicalEntity> {
    let mut lookup = IndexMap::new();
    for result in ed_results {
        let Some((entity_id, candidate)) = chosen_candidate(result) else {
            continue;
        };
        if candidate.canonical_name.is_empty() {
            continue;
        }
        let entity_type = if candidate.label.is_empty() {
            DEFAULT_CANONICAL_TYPE.to_owned()
        } else {
            candidate.label.clone()
        };
        lookup.insert(
            entity_id.clone(),
            CanonicalEntity {
                name: candidate.canonical_name.clone(),
                entity_type,
                aliases: candidate.aliases.clone(),
            },
        );
    }
    lookup
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
    if chunk_mentions.is_empty() || resolved_mentions.is_empty() {
        return chunk_text.to_owned();
    }
    // Byte offset of every char boundary, so char offsets map to byte offsets.
    let boundaries: Vec<usize> = chunk_text
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(chunk_text.len()))
        .collect();
    let char_len = boundaries.len() - 1;

    let mut replacements: Vec<(usize, usize, String, &MentionId)> = Vec::new();
    for mention in chunk_mentions {
        let Some(entity_id) = resolved_mentions.get(&mention.id) else {
            continue;
        };
        let in_bounds = mention.span.start >= chunk_start_char
            && mention.span.end >= chunk_start_char
            && mention.span.end - chunk_start_char <= char_len;
        if !in_bounds {
            tracing::warn!(
                mention = %mention.id,
                start = mention.span.start,
                end = mention.span.end,
                chunk_start_char,
                chunk_len = char_len,
                "mention span out of bounds for chunk; skipping"
            );
            continue;
        }
        let local_start = mention.span.start - chunk_start_char;
        let local_end = mention.span.end - chunk_start_char;
        let name = canonical.get(entity_id).map_or(mention.name.as_str(), |c| c.name.as_str());
        replacements.push((local_start, local_end, format!("[{entity_id}:{name}]"), &mention.id));
    }
    if replacements.is_empty() {
        return chunk_text.to_owned();
    }

    // Right to left (stable for equal starts, like Python's sort(reverse=True)); a mention
    // overlapping the one to its right is skipped.
    replacements.sort_by_key(|r| std::cmp::Reverse(r.0));
    let mut result = chunk_text.to_owned();
    let mut prev_start = char_len + 1;
    for (local_start, local_end, replacement, mention_id) in replacements {
        if local_end > prev_start {
            tracing::debug!(mention = %mention_id, local_start, local_end, "skipping overlapping mention");
            continue;
        }
        prev_start = local_start;
        let start = boundaries[local_start.min(char_len)];
        let end = boundaries[local_end].max(start);
        result.replace_range(start..end, &replacement);
    }
    result
}

/// Per-entity data collected from a chunk's mentions.
struct ChunkEntity<'a> {
    name: &'a str,
    entity_type: &'a str,
    ed_aliases: &'a [String],
    surface_forms: IndexSet<&'a str>,
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
    let mut by_entity: IndexMap<&EntityId, ChunkEntity<'_>> = IndexMap::new();
    for mention in chunk_mentions {
        let Some(entity_id) = resolved_mentions.get(&mention.id) else {
            continue;
        };
        let entry = by_entity.entry(entity_id).or_insert_with(|| match canonical.get(entity_id) {
            Some(info) => ChunkEntity {
                name: &info.name,
                entity_type: &info.entity_type,
                ed_aliases: &info.aliases,
                surface_forms: IndexSet::new(),
            },
            None => ChunkEntity {
                name: &mention.name,
                entity_type: &mention.entity_type,
                ed_aliases: &[],
                surface_forms: IndexSet::new(),
            },
        });
        if !mention.name.is_empty() {
            entry.surface_forms.insert(&mention.name);
        }
    }

    let mut entities: Vec<PromptEntity> = by_entity
        .into_iter()
        .map(|(entity_id, info)| {
            let mut all_aliases = info.surface_forms;
            all_aliases.extend(info.ed_aliases.iter().map(String::as_str));
            let aliases = all_aliases
                .into_iter()
                .filter(|alias| *alias != info.name)
                .filter(|alias| {
                    !alias.is_empty() && !alias_exclusion.contains(py_strip(&alias.to_lowercase()))
                })
                .filter(|alias| {
                    let is_role = roles.is_some_and(|r| r.is_valid_role_term(alias));
                    if is_role {
                        tracing::debug!(alias, entity = %entity_id, "filtered role term from prompt aliases");
                    }
                    !is_role
                })
                .map(str::to_owned)
                .collect();
            PromptEntity {
                id: entity_id.clone(),
                name: info.name.to_owned(),
                entity_type: Some(info.entity_type.to_owned()).filter(|t| !t.is_empty()),
                aliases,
            }
        })
        .collect();
    entities.sort_by(|a, b| a.name.cmp(&b.name));
    entities
}

/// Adds one fact's link outcome to the stats.
pub fn record_link(
    stats: &mut FactLinkingStats,
    subject: Option<&EntityId>,
    object: Option<&EntityId>,
) {
    if subject.is_some() {
        stats.linked_subjects += 1;
    }
    if object.is_some() {
        stats.linked_objects += 1;
    }
    if subject.is_some() && object.is_some() {
        stats.linked_both += 1;
    }
    if subject.is_some() || object.is_some() {
        stats.linked_either += 1;
    }
    stats.total_facts += 1;
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
        if name.is_empty() {
            return String::new();
        }
        if let Some(entity_type) = entity_type.filter(|t| !t.is_empty()) {
            match self.similarity.entity_types().normalize_label(entity_type) {
                "PER" => {
                    let parts = self.similarity.normalize_person_name_parts(name);
                    return if parts.is_empty() {
                        normalize_for_matching(name)
                    } else {
                        parts.join(" ")
                    };
                }
                "ORG" => return self.similarity.normalize_org_name(name),
                _ => {}
            }
        }
        normalize_for_matching(name)
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
        let mut lookup = EntityLookup::default();
        for result in ed_results {
            let Some(entity_id) =
                result.chosen_entity_id.as_ref().filter(|id| !id.as_str().is_empty())
            else {
                continue;
            };
            // The id itself, in case the LLM outputs it directly.
            lookup.name_to_entity.insert(entity_id.as_str().to_owned(), entity_id.clone());
            lookup.name_to_entity.insert(entity_id.as_str().to_lowercase(), entity_id.clone());

            let Some((_, candidate)) = chosen_candidate(result) else {
                continue;
            };
            let entity_type = candidate.label.as_str();
            let type_hint = Some(entity_type).filter(|t| !t.is_empty());
            if !candidate.canonical_name.is_empty() {
                let normalized = self.normalize_name(&candidate.canonical_name, type_hint);
                lookup.add(normalized, entity_id, entity_type, true);
            }
            for alias in candidate.aliases.iter().filter(|a| !a.is_empty()) {
                let normalized = self.normalize_name(alias, type_hint);
                lookup.add(normalized, entity_id, entity_type, false);
            }
        }

        for mention in doc_mentions {
            let Some(entity_id) = resolved_mentions.get(&mention.id) else {
                continue;
            };
            if mention.name.is_empty() {
                continue;
            }
            let entity_type = mention.entity_type.as_str();
            let normalized =
                self.normalize_name(&mention.name, Some(entity_type).filter(|t| !t.is_empty()));
            lookup.add(normalized, entity_id, entity_type, false);
        }

        tracing::info!(names = lookup.name_to_entity.len(), "built fact entity lookup");
        lookup
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
        let triple = &mut fact.assertion_triple;
        let subject = self.link_side(&mut triple.subject, "subject", lookup, canonical, role_map);
        let object = self.link_side(&mut triple.object, "object", lookup, canonical, role_map);
        (subject, object)
    }

    /// Links one side of a triple (rules in the module docs), rewriting `text` in place.
    fn link_side(
        &self,
        text: &mut String,
        side: &str,
        lookup: &EntityLookup,
        canonical: &IndexMap<EntityId, CanonicalEntity>,
        role_map: Option<&IndexMap<String, EntityId>>,
    ) -> Option<EntityId> {
        if text.is_empty() {
            return None;
        }
        if let Some((entity_id, canonical_name)) = parse_bracket_token(text) {
            // An empty (whitespace-only) id leaves the side unlinked, as Python's `if subject_id:`.
            let entity_id = (!entity_id.is_empty()).then(|| EntityId::new(entity_id));
            tracing::debug!(side, text = text.as_str(), entity = ?entity_id, "bracket token");
            *text = canonical_name.to_owned();
            return entity_id;
        }

        let linked = self.lookup_name(text, lookup);
        let Some(roles) = self.roles.as_ref().filter(|r| r.is_valid_role_term(text)) else {
            return linked;
        };
        if let Some(entity_id) = linked {
            if let Some(info) = canonical.get(&entity_id) {
                tracing::info!(
                    side,
                    role = text.as_str(),
                    canonical = info.name.as_str(),
                    entity = %entity_id,
                    "rewriting role term to canonical name"
                );
                text.clone_from(&info.name);
            }
            return Some(entity_id);
        }

        let entity_id = role_map?.get(&roles.normalize_role_term(text))?.clone();
        tracing::info!(side, role = text.as_str(), entity = %entity_id, "linked role term via role map");
        if let Some(info) = canonical.get(&entity_id) {
            text.clone_from(&info.name);
        }
        Some(entity_id)
    }

    /// The untyped normalised name, then the `PER` and `ORG` normalisations, in `lookup`.
    fn lookup_name(&self, name: &str, lookup: &EntityLookup) -> Option<EntityId> {
        [None, Some("PER"), Some("ORG")]
            .into_iter()
            .find_map(|entity_type| {
                lookup.name_to_entity.get(&self.normalize_name(name, entity_type))
            })
            .cloned()
    }
}
