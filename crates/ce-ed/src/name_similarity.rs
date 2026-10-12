//! Type-aware name similarity (Python `services/ed/name_similarity.py`): PERSON names with
//! titles, nicknames and name order; ORG names with corporate suffixes, prefixes and acronyms;
//! dates and money amounts; `difflib` sequence matching for every other type.
//!
//! Settings come from `name_matching.yaml` ([`NameMatching`]); entity types are normalised with
//! `entity_types.yaml` ([`EntityTypes::normalize_label`]). Prefixes and suffixes are tried in
//! file order (Python iterated sets, whose order varies between runs).

use std::collections::BTreeSet;

use ce_config::mappings::{EntityTypes, NameMatching};
use ce_config::MappingsRegistry;
use ce_text::normalize::{py_split, py_strip};
use ce_text::{normalize_whitespace, sequence_matcher_ratio};

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
        let config = &self.name_matching;
        if name.is_empty() {
            return Vec::new();
        }
        let normalized = punctuation_to_space(&name.to_lowercase());
        py_split(&normalized)
            .filter(|part| !(config.settings.strip_titles && config.person_titles.contains(*part)))
            .map(|part| {
                let formal = if config.settings.normalize_nicknames {
                    config.nickname_mappings.get(part).map(String::as_str)
                } else {
                    None
                };
                formal.unwrap_or(part).to_owned()
            })
            .filter(|part| part.chars().count() >= config.min_word_length())
            .collect()
    }

    /// Similarity of two person names in `[0, 1]`.
    #[must_use]
    pub fn person_name_similarity(&self, s1: &str, s2: &str) -> f64 {
        let parts1: BTreeSet<String> = self.normalize_person_name_parts(s1).into_iter().collect();
        let parts2: BTreeSet<String> = self.normalize_person_name_parts(s2).into_iter().collect();
        if parts1.is_empty() || parts2.is_empty() {
            return 0.0;
        }
        self.parts_similarity(&parts1, &parts2)
            .unwrap_or_else(|| sequence_matcher_ratio(&s1.to_lowercase(), &s2.to_lowercase()))
    }

    /// Organisation name with prefixes and corporate suffixes removed.
    #[must_use]
    pub fn normalize_org_name(&self, name: &str) -> String {
        let config = &self.name_matching;
        if name.is_empty() {
            return String::new();
        }
        let lowered = name.to_lowercase();
        let mut normalized = normalize_whitespace(&punctuation_to_space(py_strip(&lowered)));

        for prefix in &config.org_prefixes {
            let with_space = format!("{prefix} ");
            if let Some(rest) = normalized.strip_prefix(with_space.as_str()) {
                normalized = rest.to_owned();
            }
        }

        if config.settings.strip_org_suffixes {
            let mut changed = true;
            while changed {
                changed = false;
                for suffix in &config.org_suffixes {
                    let with_space = format!(" {suffix}");
                    if let Some(rest) = normalized.strip_suffix(with_space.as_str()) {
                        normalized = rest.to_owned();
                        changed = true;
                    } else if normalized == *suffix {
                        // The whole name is a suffix: keep it.
                        break;
                    }
                }
            }
        }

        py_strip(&normalized).to_owned()
    }

    /// Tokens of [`NameSimilarity::normalize_org_name`].
    #[must_use]
    pub fn normalize_org_name_parts(&self, name: &str) -> Vec<String> {
        let normalized = self.normalize_org_name(name);
        if normalized.is_empty() {
            return Vec::new();
        }
        let min_len = self.name_matching.min_word_length();
        py_split(&punctuation_to_space(&normalized))
            .filter(|part| part.chars().count() >= min_len)
            .map(str::to_owned)
            .collect()
    }

    /// Similarity of two organisation names in `[0, 1]` (acronyms included).
    #[must_use]
    pub fn org_name_similarity(&self, s1: &str, s2: &str) -> f64 {
        let norm1 = self.normalize_org_name(s1);
        let norm2 = self.normalize_org_name(s2);
        if norm1.is_empty() || norm2.is_empty() {
            return 0.0;
        }
        if norm1 == norm2 {
            return self.name_matching.exact_match_threshold();
        }

        let acronym_score = self.acronym_match(s1, s2);
        if acronym_score > 0.0 {
            return acronym_score;
        }

        let parts1: BTreeSet<String> = self.normalize_org_name_parts(s1).into_iter().collect();
        let parts2: BTreeSet<String> = self.normalize_org_name_parts(s2).into_iter().collect();
        if parts1.is_empty() || parts2.is_empty() {
            return sequence_matcher_ratio(&norm1, &norm2);
        }
        self.parts_similarity(&parts1, &parts2)
            .unwrap_or_else(|| sequence_matcher_ratio(&norm1, &norm2))
    }

    /// Type-aware similarity in `[0, 1]`: the label is normalised, then `PER` -> person,
    /// `ORG` -> organisation, `DATE` -> date, `MONEY` -> money, anything else -> `difflib`
    /// ratio of the lowercased strings. Empty input scores 0.
    #[must_use]
    pub fn enhanced_string_similarity(&self, s1: &str, s2: &str, entity_type: &str) -> f64 {
        if s1.is_empty() || s2.is_empty() {
            return 0.0;
        }
        match self.entity_types.normalize_label(entity_type) {
            "PER" => self.person_name_similarity(s1, s2),
            "ORG" => self.org_name_similarity(s1, s2),
            "DATE" | "MONEY" => exact_cleaned_match(s1, s2, &self.name_matching),
            _ => sequence_matcher_ratio(&s1.to_lowercase(), &s2.to_lowercase()),
        }
    }

    /// Set-based score shared by person and organisation names: exact set match, subset,
    /// partial overlap (Jaccard), or `None` when the sets are disjoint.
    fn parts_similarity(
        &self,
        parts1: &BTreeSet<String>,
        parts2: &BTreeSet<String>,
    ) -> Option<f64> {
        let config = &self.name_matching;
        if parts1 == parts2 {
            return Some(config.exact_match_threshold());
        }
        let overlap = parts1.intersection(parts2).count();
        if parts1.is_subset(parts2) || parts2.is_subset(parts1) {
            let total = parts1.len().max(parts2.len());
            return Some(
                config.subset_match_base()
                    + config.subset_match_bonus() * overlap as f64 / total as f64,
            );
        }
        if overlap > 0 {
            let union = parts1.union(parts2).count();
            let jaccard = overlap as f64 / union as f64;
            return Some(config.partial_overlap_base() + config.partial_overlap_bonus() * jaccard);
        }
        None
    }

    /// Score when one name is a known or initials acronym of the other, else 0.
    fn acronym_match(&self, s1: &str, s2: &str) -> f64 {
        let config = &self.name_matching;
        let s1_lower = py_strip(&s1.to_lowercase()).to_owned();
        let s2_lower = py_strip(&s2.to_lowercase()).to_owned();

        for (acronym, other) in [(&s1_lower, &s2_lower), (&s2_lower, &s1_lower)] {
            if let Some(expansion) = config.org_acronyms.get(acronym.as_str()) {
                if other.contains(expansion.as_str()) || expansion.contains(other.as_str()) {
                    return config.exact_match_threshold();
                }
            }
        }

        let (short, long) = if s1_lower.chars().count() < s2_lower.chars().count() {
            (&s1_lower, &s2_lower)
        } else {
            (&s2_lower, &s1_lower)
        };
        // Python precedence: `len(short) <= 6 and s1.isupper() or s2.isupper()`.
        if (short.chars().count() <= 6 && py_isupper(s1)) || py_isupper(s2) {
            let long_parts = self.normalize_org_name_parts(long);
            if !long_parts.is_empty() {
                let initials: String =
                    long_parts.iter().filter_map(|part| part.chars().next()).collect();
                if initials == *short {
                    return config.subset_match_base();
                }
            }
        }
        0.0
    }
}

/// Similarity of two date strings in `[0, 1]`: the exact-match score of the registry's
/// `name_matching.yaml` when the cleaned strings are equal, else 0.
#[must_use]
pub fn date_similarity(s1: &str, s2: &str) -> f64 {
    exact_cleaned_match(s1, s2, registry_name_matching())
}

/// Similarity of two money amounts in `[0, 1]`: the exact-match score of the registry's
/// `name_matching.yaml` when the cleaned strings are equal, else 0.
#[must_use]
pub fn money_similarity(s1: &str, s2: &str) -> f64 {
    exact_cleaned_match(s1, s2, registry_name_matching())
}

/// The registry's name matching settings (built-in defaults if the registry fails to load).
fn registry_name_matching() -> &'static NameMatching {
    static DEFAULT: std::sync::OnceLock<NameMatching> = std::sync::OnceLock::new();
    match MappingsRegistry::load() {
        Ok(registry) => &registry.name_matching,
        Err(error) => {
            tracing::warn!(error = %error, "mappings registry failed to load; default thresholds");
            DEFAULT.get_or_init(NameMatching::default)
        }
    }
}

/// Strict date / money comparison: lowercased, punctuation removed, whitespace collapsed,
/// then equal strings score the exact-match threshold.
fn exact_cleaned_match(s1: &str, s2: &str, config: &NameMatching) -> f64 {
    let clean = |s: &str| {
        let lowered = s.to_lowercase();
        normalize_whitespace(&punctuation_to_space(py_strip(&lowered)))
    };
    let n1 = clean(s1);
    let n2 = clean(s2);
    if n1.is_empty() || n2.is_empty() {
        return 0.0;
    }
    if n1 == n2 {
        config.exact_match_threshold()
    } else {
        0.0
    }
}

/// Replaces each of `.,;:!?'"()[]{}-` with a space.
fn punctuation_to_space(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '.' | ',' | ';' | ':' | '!' | '?' | '\'' | '"' | '(' | ')' | '[' | ']' | '{' | '}'
            | '-' => ' ',
            other => other,
        })
        .collect()
}

/// Python `str.isupper()`: at least one cased character and no lowercase or titlecase one.
fn py_isupper(text: &str) -> bool {
    let mut cased = false;
    for c in text.chars() {
        if c.is_lowercase() || is_titlecase(c) {
            return false;
        }
        cased |= c.is_uppercase();
    }
    cased
}

/// Unicode general category `Lt`.
fn is_titlecase(c: char) -> bool {
    matches!(
        c,
        '\u{01C5}'
            | '\u{01C8}'
            | '\u{01CB}'
            | '\u{01F2}'
            | '\u{1F88}'..='\u{1F8F}'
            | '\u{1F98}'..='\u{1F9F}'
            | '\u{1FA8}'..='\u{1FAF}'
            | '\u{1FBC}'
            | '\u{1FCC}'
            | '\u{1FFC}'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_isupper_matches_python() {
        assert!(py_isupper("FBI"));
        assert!(py_isupper("F.B.I. 2"));
        assert!(!py_isupper("Fbi"));
        assert!(!py_isupper("123"));
        assert!(!py_isupper(""));
        assert!(!py_isupper("A\u{01C5}"));
        assert!(py_isupper("ÉCOLE"));
    }

    #[test]
    fn punctuation_to_space_replaces_each_character() {
        assert_eq!(punctuation_to_space("a.b-c[d]"), "a b c d ");
    }
}
