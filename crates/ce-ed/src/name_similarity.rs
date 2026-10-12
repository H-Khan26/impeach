//! Type-aware name similarity (Python `services/ed/name_similarity.py`): PERSON names with
//! titles, nicknames and name order; ORG names with corporate suffixes, prefixes and acronyms;
//! dates and money amounts; `difflib` sequence matching for every other type.
//!
//! Settings come from `name_matching.yaml` ([`NameMatching`]); entity types are normalised with
//! `entity_types.yaml` ([`EntityTypes::normalize_label`]).

use ce_config::mappings::{EntityTypes, NameMatching};
use ce_config::MappingsRegistry;

/// Name similarity over the `name_matching.yaml` settings and the entity type mappings.
#[derive(Debug, Clone)]
pub struct NameSimilarity {
    name_matching: NameMatching,
    entity_types: EntityTypes,
}

impl NameSimilarity {
    /// Similarity over the given settings.
    #[must_use]
    pub fn new(name_matching: NameMatching, entity_types: EntityTypes) -> Self {
        Self { name_matching, entity_types }
    }

    /// Similarity over the settings of a mappings registry.
    #[must_use]
    pub fn from_mappings(mappings: &MappingsRegistry) -> Self {
        Self::new(mappings.name_matching.clone(), mappings.entity_types.clone())
    }

    /// The `name_matching.yaml` settings.
    #[must_use]
    pub fn name_matching(&self) -> &NameMatching {
        &self.name_matching
    }

    /// The entity type mappings.
    #[must_use]
    pub fn entity_types(&self) -> &EntityTypes {
        &self.entity_types
    }

    /// Core name tokens of a person name: lowercased, punctuation removed, titles stripped,
    /// nicknames mapped to formal names (per the settings).
    #[must_use]
    pub fn normalize_person_name_parts(&self, name: &str) -> Vec<String> {
        let _ = name;
        todo!("worker A")
    }

    /// Similarity of two person names in `[0, 1]`.
    #[must_use]
    pub fn person_name_similarity(&self, s1: &str, s2: &str) -> f64 {
        let _ = (s1, s2);
        todo!("worker A")
    }

    /// Organisation name with prefixes and corporate suffixes removed.
    #[must_use]
    pub fn normalize_org_name(&self, name: &str) -> String {
        let _ = name;
        todo!("worker A")
    }

    /// Tokens of [`NameSimilarity::normalize_org_name`].
    #[must_use]
    pub fn normalize_org_name_parts(&self, name: &str) -> Vec<String> {
        let _ = name;
        todo!("worker A")
    }

    /// Similarity of two organisation names in `[0, 1]` (acronyms included).
    #[must_use]
    pub fn org_name_similarity(&self, s1: &str, s2: &str) -> f64 {
        let _ = (s1, s2);
        todo!("worker A")
    }

    /// Type-aware similarity in `[0, 1]`: the label is normalised, then `PER` -> person,
    /// `ORG` -> organisation, `DATE` -> date, `MONEY` -> money, anything else -> `difflib`
    /// ratio of the lowercased strings. Empty input scores 0.
    #[must_use]
    pub fn enhanced_string_similarity(&self, s1: &str, s2: &str, entity_type: &str) -> f64 {
        let _ = (s1, s2, entity_type);
        todo!("worker A")
    }
}

/// Similarity of two date strings in `[0, 1]`.
#[must_use]
pub fn date_similarity(s1: &str, s2: &str) -> f64 {
    let _ = (s1, s2);
    todo!("worker A")
}

/// Similarity of two money amounts in `[0, 1]`.
#[must_use]
pub fn money_similarity(s1: &str, s2: &str) -> f64 {
    let _ = (s1, s2);
    todo!("worker A")
}
