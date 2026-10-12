//! Structured identifiers parsed from mention text for precise entity lookup (Python
//! `services/ed/structured_keys.py`): exhibits, statutes, case citations and captions, federal
//! rules, plus the search terms each key expands to.
//!
//! Patterns are the named regexes of `detection_patterns.yaml`; [`KeyPatterns`] falls back to
//! built-in patterns for names the registry does not define (the Python fallbacks of both
//! `structured_keys.py` and `rule_resolvers.py`).

use std::collections::HashMap;

use ce_config::mappings::DetectionPatterns;
use regex::Regex;

/// Named regexes used by key extraction and the rule resolvers.
#[derive(Debug, Clone, Default)]
pub struct KeyPatterns {
    patterns: HashMap<String, Regex>,
}

impl KeyPatterns {
    /// Registry patterns, with the built-in fallback for every name the registry lacks.
    #[must_use]
    pub fn from_mappings(patterns: &DetectionPatterns) -> Self {
        let _ = patterns;
        todo!("worker A")
    }

    /// The pattern called `name` (`case_docket`, `exhibit_id`, `exhibit_id_simple`,
    /// `statute_usc`, `case_citation`, `case_caption`, `federal_rule`, ...).
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Regex> {
        self.patterns.get(name)
    }
}

/// Exhibit key: id, party attribution (`Plaintiff's`, ...) and case scope.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExhibitKey {
    /// Exhibit identifier, e.g. `12A`.
    pub exhibit_id: String,
    /// Party attribution, e.g. `Plaintiff's`.
    pub party_attribution: Option<String>,
    /// Case scope.
    pub case_id: Option<String>,
}

/// Statute key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatuteKey {
    /// Title, e.g. `42` in 42 U.S.C.
    pub title: Option<String>,
    /// Section, e.g. `1983`.
    pub section: Option<String>,
    /// Subsection, e.g. `(a)`.
    pub subsection: Option<String>,
    /// Code type: `usc`, `cfr`, `state`.
    pub code_type: String,
    /// State of a state statute.
    pub state: Option<String>,
    /// Code name, e.g. `Civ. Code`.
    pub code_name: Option<String>,
}

impl Default for StatuteKey {
    fn default() -> Self {
        Self {
            title: None,
            section: None,
            subsection: None,
            code_type: "usc".to_owned(),
            state: None,
            code_name: None,
        }
    }
}

impl StatuteKey {
    /// Normalised lookup key, e.g. `usc_42_1983_a`.
    #[must_use]
    pub fn normalized_key(&self) -> String {
        todo!("worker A")
    }
}

/// Case citation or caption key.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CaseCitationKey {
    /// Reporter volume, e.g. `123`.
    pub volume: Option<String>,
    /// Reporter, e.g. `F.3d`.
    pub reporter: Option<String>,
    /// First page, e.g. `456`.
    pub page: Option<String>,
    /// Court, e.g. `9th Cir.`.
    pub court: Option<String>,
    /// Decision year.
    pub year: Option<String>,
    /// Plaintiff from the caption.
    pub plaintiff: Option<String>,
    /// Defendant from the caption.
    pub defendant: Option<String>,
}

impl CaseCitationKey {
    /// Normalised lookup key: `{volume}_{reporter}_{page}` for citations,
    /// `{plaintiff}_v_{defendant}` for captions, empty otherwise.
    #[must_use]
    pub fn normalized_key(&self) -> String {
        todo!("worker A")
    }
}

/// Federal rule key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FederalRuleKey {
    /// `civil`, `criminal`, `appellate` or `evidence`.
    pub rule_type: String,
    /// Rule number.
    pub rule_number: String,
    /// Subdivision letter.
    pub subdivision_letter: Option<String>,
    /// Subdivision number.
    pub subdivision_number: Option<String>,
}

impl FederalRuleKey {
    /// Normalised lookup key, e.g. `frcp_12_b_6`.
    #[must_use]
    pub fn normalized_key(&self) -> String {
        todo!("worker A")
    }
}

/// Exhibit key of a mention text such as `Plaintiff's Exhibit 12A`.
#[must_use]
pub fn extract_exhibit_key(
    patterns: &KeyPatterns,
    text: &str,
    case_id: Option<&str>,
) -> Option<ExhibitKey> {
    let _ = (patterns, text, case_id);
    todo!("worker A")
}

/// Statute key of a mention text such as `42 U.S.C. § 1983(a)`.
#[must_use]
pub fn extract_statute_key(patterns: &KeyPatterns, text: &str) -> Option<StatuteKey> {
    let _ = (patterns, text);
    todo!("worker A")
}

/// Case citation or caption key of a mention text such as `123 F.3d 456` or `Smith v. Jones`.
#[must_use]
pub fn extract_case_citation_key(patterns: &KeyPatterns, text: &str) -> Option<CaseCitationKey> {
    let _ = (patterns, text);
    todo!("worker A")
}

/// Search terms (entity names to look up) for an exhibit key.
#[must_use]
pub fn exhibit_key_to_search_terms(key: &ExhibitKey) -> Vec<String> {
    let _ = key;
    todo!("worker A")
}

/// Search terms for a statute key.
#[must_use]
pub fn statute_key_to_search_terms(key: &StatuteKey) -> Vec<String> {
    let _ = key;
    todo!("worker A")
}

/// Search terms for a case citation key.
#[must_use]
pub fn case_citation_key_to_search_terms(key: &CaseCitationKey) -> Vec<String> {
    let _ = key;
    todo!("worker A")
}
