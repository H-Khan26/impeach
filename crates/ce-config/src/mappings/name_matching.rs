//! ED name matching rules (`name_matching.yaml`).

use indexmap::{IndexMap, IndexSet};
use serde::Deserialize;

/// Titles, nicknames, organisation affixes and similarity thresholds for ED.
///
/// Sets keep file order so iteration is deterministic.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NameMatching {
    /// Titles stripped from person names.
    pub person_titles: IndexSet<String>,
    /// Nickname -> formal name (a repeated key keeps its last value).
    pub nickname_mappings: IndexMap<String, String>,
    /// Corporate suffixes stripped from organisation names.
    pub org_suffixes: IndexSet<String>,
    /// Prefixes stripped from organisation names.
    pub org_prefixes: IndexSet<String>,
    /// Acronym -> expansion.
    pub org_acronyms: IndexMap<String, String>,
    /// Similarity thresholds.
    pub thresholds: NameMatchingThresholds,
    /// Feature flags.
    pub settings: NameMatchingSettings,
}

/// `thresholds:` block.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NameMatchingThresholds {
    /// Score for an exact match after normalisation; defaults to 0.98.
    pub exact_match: f64,
    /// Base score when one name is a subset of the other; defaults to 0.80.
    pub subset_match_base: f64,
    /// Bonus scaled by subset overlap; defaults to 0.15.
    pub subset_match_bonus: f64,
    /// Base score for partial overlap; defaults to 0.60.
    pub partial_overlap_base: f64,
    /// Bonus scaled by partial-overlap Jaccard; defaults to 0.30.
    pub partial_overlap_bonus: f64,
    /// Minimum length for fuzzy matching; defaults to 4.
    pub min_fuzzy_match_length: usize,
    /// Minimum name-part length; defaults to 2.
    pub min_word_length: usize,
}

impl Default for NameMatchingThresholds {
    fn default() -> Self {
        Self {
            exact_match: 0.98,
            subset_match_base: 0.80,
            subset_match_bonus: 0.15,
            partial_overlap_base: 0.60,
            partial_overlap_bonus: 0.30,
            min_fuzzy_match_length: 4,
            min_word_length: 2,
        }
    }
}

/// `settings:` block; every flag defaults to true.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NameMatchingSettings {
    /// Map nicknames to formal names.
    pub normalize_nicknames: bool,
    /// Strip person titles.
    pub strip_titles: bool,
    /// Strip corporate suffixes.
    pub strip_org_suffixes: bool,
    /// Compare case-insensitively.
    pub case_insensitive: bool,
}

impl Default for NameMatchingSettings {
    fn default() -> Self {
        Self {
            normalize_nicknames: true,
            strip_titles: true,
            strip_org_suffixes: true,
            case_insensitive: true,
        }
    }
}

impl NameMatching {
    /// Exact-match score.
    pub fn exact_match_threshold(&self) -> f64 {
        self.thresholds.exact_match
    }

    /// Subset-match base score.
    pub fn subset_match_base(&self) -> f64 {
        self.thresholds.subset_match_base
    }

    /// Subset-match bonus.
    pub fn subset_match_bonus(&self) -> f64 {
        self.thresholds.subset_match_bonus
    }

    /// Partial-overlap base score.
    pub fn partial_overlap_base(&self) -> f64 {
        self.thresholds.partial_overlap_base
    }

    /// Partial-overlap bonus.
    pub fn partial_overlap_bonus(&self) -> f64 {
        self.thresholds.partial_overlap_bonus
    }

    /// Minimum name-part length.
    pub fn min_word_length(&self) -> usize {
        self.thresholds.min_word_length
    }
}
