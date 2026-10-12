//! Structured identifiers parsed from mention text for precise entity lookup (Python
//! `services/ed/structured_keys.py`): exhibits, statutes, case citations and captions, federal
//! rules, plus the search terms each key expands to.
//!
//! Patterns are the named regexes of `detection_patterns.yaml`; [`KeyPatterns`] falls back to
//! built-in patterns for names the registry does not define (the Python fallbacks of both
//! `structured_keys.py` and `rule_resolvers.py`).

use std::collections::HashMap;

use ce_config::mappings::{DetectionPatterns, PatternDef, RegexFlag};
use ce_text::normalize::{is_py_whitespace, py_split, py_strip};
use regex::{Captures, Regex, RegexBuilder};

/// Built-in patterns for names missing from the registry (Python `_get_fallback_patterns` of
/// `structured_keys.py` and `_get_fallback_pattern` of `rule_resolvers.py`).
const FALLBACK_PATTERNS: &[(&str, &str)] = &[
    ("case_docket", r"(?i)(?:No\.|Case No\.|Case\s*#?)\s*(\d+[:\-][\w\-]+)"),
    (
        "exhibit_id",
        r"(?i)(?:(Plaintiff(?:'s)?|Defendant(?:'s)?|Defense|Government(?:'s)?)\s+)?(?:Exhibit|Ex\.|Exh\.)\s*([A-Za-z0-9\-]+)",
    ),
    ("exhibit_id_simple", r"(?i)(?:Exhibit|Ex\.|Exh\.)\s*([A-Za-z0-9\-]+)"),
    (
        "statute_usc",
        r"(?i)(\d+)\s+U\.?S\.?C\.?\s*(?:\x{A7}|[Ss]ec(?:tion)?\.?)?\s*(\d+)(?:\s*\(([a-z])\))?",
    ),
    (
        "case_citation",
        r"(?i)(\d+)\s+([A-Z][a-z]*\.?\s*(?:\d+[a-z]*\.?)?\s*(?:Supp\.?\s*(?:\d+[a-z]*)?)?)\s*(\d+)(?:\s*\(([^)]+)\s+(\d{4})\))?",
    ),
    (
        "case_caption",
        r"([A-Z][a-zA-Z\-']+(?:\s+[A-Z][a-zA-Z\-']+)*)\s+v\.?\s+([A-Z][a-zA-Z\-']+(?:\s+[A-Z][a-zA-Z\-']+)*)",
    ),
    (
        "federal_rule",
        r"(?i)(?:Fed(?:eral)?\.?\s*R(?:ule)?\.?\s*(Civ(?:il)?|Crim(?:inal)?|App(?:ellate)?|Evid(?:ence)?)\.?\s*P(?:roc(?:edure)?)?\.?|FRCP|FRCRP|FRAP|FRE)\s*(\d+)(?:\s*\(([a-z])\))?(?:\s*\((\d+)\))?",
    ),
];

/// Named regexes used by key extraction and the rule resolvers.
#[derive(Debug, Clone, Default)]
pub struct KeyPatterns {
    patterns: HashMap<String, Regex>,
}

impl KeyPatterns {
    /// Registry patterns, with the built-in fallback for every name the registry lacks.
    ///
    /// A name defined both as a detection pattern and as an ED pattern resolves to the ED
    /// pattern: key extraction reads the ED capture groups (Python resolved `federal_rule` to the
    /// one-group detection pattern and failed on group 2). A pattern that does not compile is
    /// logged and replaced by the fallback, like the Python registry lookup.
    #[must_use]
    pub fn from_mappings(patterns: &DetectionPatterns) -> Self {
        let mut compiled = HashMap::new();
        for (name, def) in patterns.patterns.iter().chain(&patterns.ed_patterns) {
            if let Some(regex) = compile_def(name, def) {
                compiled.insert(name.clone(), regex);
            }
        }
        for &(name, source) in FALLBACK_PATTERNS {
            compiled.entry(name.to_owned()).or_insert_with(|| {
                Regex::new(source).expect("built-in fallback patterns are valid regexes")
            });
        }
        Self { patterns: compiled }
    }

    /// The built-in fallback patterns only.
    #[must_use]
    pub fn fallback() -> Self {
        Self::from_mappings(&DetectionPatterns::default())
    }

    /// The pattern called `name` (`case_docket`, `exhibit_id`, `exhibit_id_simple`,
    /// `statute_usc`, `case_citation`, `case_caption`, `federal_rule`, ...).
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Regex> {
        self.patterns.get(name)
    }
}

/// The compiled regex of a definition (compiling it when it was not loaded through the
/// registry); `None` with a warning when it is invalid.
fn compile_def(name: &str, def: &PatternDef) -> Option<Regex> {
    if let Some(regex) = def.compiled() {
        return Some(regex.clone());
    }
    RegexBuilder::new(&def.pattern)
        .case_insensitive(def.flags == Some(RegexFlag::IgnoreCase))
        .build()
        .map_err(|error| tracing::warn!(pattern = name, error = %error, "invalid regex pattern"))
        .ok()
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
    ///
    /// State statutes: `{state}_{code name with dot/space runs as _}`; others:
    /// `{code_type}_{title}`; then the section and the subsection without parentheses. Empty
    /// fields are skipped.
    #[must_use]
    pub fn normalized_key(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        match present(&self.state) {
            Some(state) if self.code_type == "state" => {
                parts.push(state.to_lowercase());
                if let Some(code_name) = present(&self.code_name) {
                    parts.push(replace_runs(&code_name.to_lowercase(), is_dot_or_space, "_"));
                }
            }
            _ => {
                parts.push(self.code_type.clone());
                if let Some(title) = present(&self.title) {
                    parts.push(title.to_owned());
                }
            }
        }
        if let Some(section) = present(&self.section) {
            parts.push(section.to_owned());
        }
        if let Some(subsection) = present(&self.subsection) {
            parts.push(subsection.trim_matches(['(', ')']).to_owned());
        }
        parts.join("_")
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
    /// Normalised lookup key: `{volume}_{reporter}_{page}` for citations (reporter lowercased
    /// without dots and whitespace), `{plaintiff}_v_{defendant}` lowercased for captions, empty
    /// otherwise.
    #[must_use]
    pub fn normalized_key(&self) -> String {
        if let (Some(volume), Some(reporter), Some(page)) =
            (present(&self.volume), present(&self.reporter), present(&self.page))
        {
            let reporter = replace_runs(&reporter.to_lowercase(), is_dot_or_space, "");
            return format!("{volume}_{reporter}_{page}");
        }
        if let (Some(plaintiff), Some(defendant)) =
            (present(&self.plaintiff), present(&self.defendant))
        {
            return format!("{}_v_{}", plaintiff.to_lowercase(), defendant.to_lowercase());
        }
        String::new()
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
    /// Normalised lookup key, e.g. `frcp_12_b_6`: `fr{first letter of the rule type}p`, the
    /// rule number, then the subdivisions that are present.
    #[must_use]
    pub fn normalized_key(&self) -> String {
        let initial: String =
            self.rule_type.chars().next().map(char::to_lowercase).into_iter().flatten().collect();
        let mut parts = vec![format!("fr{initial}p"), self.rule_number.clone()];
        parts.extend(present(&self.subdivision_letter).map(str::to_owned));
        parts.extend(present(&self.subdivision_number).map(str::to_owned));
        parts.join("_")
    }
}

/// Exhibit key of a mention text such as `Plaintiff's Exhibit 12A`.
///
/// Tries `exhibit_id` (party attribution in group 1, id in group 2) and then
/// `exhibit_id_simple` (id in group 1). The id is uppercased; trailing `'`, `s` and spaces are
/// stripped from the party (`Plaintiff's` -> `Plaintiff`).
#[must_use]
pub fn extract_exhibit_key(
    patterns: &KeyPatterns,
    text: &str,
    case_id: Option<&str>,
) -> Option<ExhibitKey> {
    if let Some(caps) = patterns.get("exhibit_id").and_then(|re| re.captures(text)) {
        if let Some(exhibit_id) = caps.get(2) {
            let party = group(&caps, 1)
                .filter(|party| !party.is_empty())
                .map(|party| party.trim_end_matches(['\'', 's', ' ']).to_owned());
            return Some(ExhibitKey {
                exhibit_id: exhibit_id.as_str().to_uppercase(),
                party_attribution: party,
                case_id: case_id.map(str::to_owned),
            });
        }
    }
    let caps = patterns.get("exhibit_id_simple")?.captures(text)?;
    Some(ExhibitKey {
        exhibit_id: group(&caps, 1)?.to_uppercase(),
        party_attribution: None,
        case_id: case_id.map(str::to_owned),
    })
}

/// Statute key of a mention text such as `42 U.S.C. § 1983(a)`.
///
/// Tries `statute_usc` (code type `usc`), then `federal_rule`, whose code type is
/// `fr{first letter of the rule type}p`: `frcp` for civil and, as in Python, also for criminal
/// rules; `frap` and `frep` for appellate and evidence rules. Without a spelled-out type the
/// text decides (`FRCRP`, `FRAP`, `FRE`, case-sensitive), defaulting to civil.
#[must_use]
pub fn extract_statute_key(patterns: &KeyPatterns, text: &str) -> Option<StatuteKey> {
    if let Some(caps) = patterns.get("statute_usc").and_then(|re| re.captures(text)) {
        return Some(StatuteKey {
            title: owned_group(&caps, 1),
            section: owned_group(&caps, 2),
            subsection: owned_group(&caps, 3),
            code_type: "usc".to_owned(),
            ..StatuteKey::default()
        });
    }

    let caps = patterns.get("federal_rule")?.captures(text)?;
    let rule_text = group(&caps, 1).unwrap_or_default();
    let rule_type = if rule_text.is_empty() {
        if text.contains("FRCRP") {
            "criminal"
        } else if text.contains("FRAP") {
            "appellate"
        } else if text.contains("FRE") {
            "evidence"
        } else {
            "civil"
        }
    } else {
        let lowered = rule_text.to_lowercase();
        if lowered.starts_with("crim") {
            "criminal"
        } else if lowered.starts_with("app") {
            "appellate"
        } else if lowered.starts_with("evid") {
            "evidence"
        } else {
            "civil"
        }
    };
    let initial = rule_type.chars().next().unwrap_or('c');
    Some(StatuteKey {
        section: owned_group(&caps, 2),
        subsection: owned_group(&caps, 3),
        code_type: format!("fr{initial}p"),
        ..StatuteKey::default()
    })
}

/// Case citation or caption key of a mention text such as `123 F.3d 456` or `Smith v. Jones`.
///
/// Tries `case_citation` (volume, reporter, page, optional court and year) and then
/// `case_caption` (plaintiff, defendant); captured text is stripped of surrounding whitespace.
#[must_use]
pub fn extract_case_citation_key(patterns: &KeyPatterns, text: &str) -> Option<CaseCitationKey> {
    if let Some(caps) = patterns.get("case_citation").and_then(|re| re.captures(text)) {
        return Some(CaseCitationKey {
            volume: owned_group(&caps, 1),
            reporter: group(&caps, 2).map(|reporter| py_strip(reporter).to_owned()),
            page: owned_group(&caps, 3),
            court: group(&caps, 4)
                .filter(|court| !court.is_empty())
                .map(|court| py_strip(court).to_owned()),
            year: owned_group(&caps, 5),
            ..CaseCitationKey::default()
        });
    }
    let caps = patterns.get("case_caption")?.captures(text)?;
    Some(CaseCitationKey {
        plaintiff: group(&caps, 1).map(|plaintiff| py_strip(plaintiff).to_owned()),
        defendant: group(&caps, 2).map(|defendant| py_strip(defendant).to_owned()),
        ..CaseCitationKey::default()
    })
}

/// Search terms (entity names to look up) for an exhibit key: `Exhibit {id}`, the
/// party-attributed forms when a party is present, then the bare id.
#[must_use]
pub fn exhibit_key_to_search_terms(key: &ExhibitKey) -> Vec<String> {
    let id = &key.exhibit_id;
    let mut terms = vec![format!("Exhibit {id}")];
    if let Some(party) = present(&key.party_attribution) {
        terms.push(format!("{party}'s Exhibit {id}"));
        terms.push(format!("{party} Exhibit {id}"));
    }
    terms.push(id.clone());
    terms
}

/// Search terms for a statute key: the U.S.C. spellings (with and without the subsection) or
/// the federal rule abbreviations, then [`StatuteKey::normalized_key`].
#[must_use]
pub fn statute_key_to_search_terms(key: &StatuteKey) -> Vec<String> {
    let mut terms = Vec::new();
    let section = present(&key.section);
    let subsection = present(&key.subsection);
    match (present(&key.title), section) {
        (Some(title), Some(section)) if key.code_type == "usc" => {
            terms.push(format!("{title} U.S.C. {section}"));
            terms.push(format!("{title} USC {section}"));
            terms.push(format!("{title} U.S.C. \u{a7} {section}"));
            if let Some(subsection) = subsection {
                terms.push(format!("{title} U.S.C. {section}({subsection})"));
                terms.push(format!("{title} USC {section}({subsection})"));
            }
        }
        (_, Some(section)) if key.code_type.starts_with("fr") => {
            let abbrevs = match key.code_type.as_str() {
                "frcrp" => ["FRCRP", "Fed. R. Crim. P."],
                "frap" => ["FRAP", "Fed. R. App. P."],
                "frep" => ["FRE", "Fed. R. Evid."],
                _ => ["FRCP", "Fed. R. Civ. P."],
            };
            for abbrev in abbrevs {
                terms.push(format!("{abbrev} {section}"));
                if let Some(subsection) = subsection {
                    terms.push(format!("{abbrev} {section}({subsection})"));
                }
            }
        }
        _ => {}
    }
    terms.push(key.normalized_key());
    terms
}

/// Search terms for a case citation key: `{volume} {reporter} {page}` (raw and with whitespace
/// runs collapsed), the caption forms `P v. D`, `P v D` and `{last word of P} v. {last word of
/// D}`, then [`CaseCitationKey::normalized_key`] when non-empty.
#[must_use]
pub fn case_citation_key_to_search_terms(key: &CaseCitationKey) -> Vec<String> {
    let mut terms = Vec::new();
    if let (Some(volume), Some(reporter), Some(page)) =
        (present(&key.volume), present(&key.reporter), present(&key.page))
    {
        terms.push(format!("{volume} {reporter} {page}"));
        let reporter = replace_runs(reporter, is_py_whitespace, " ");
        terms.push(format!("{volume} {reporter} {page}"));
    }
    if let (Some(plaintiff), Some(defendant)) = (present(&key.plaintiff), present(&key.defendant)) {
        terms.push(format!("{plaintiff} v. {defendant}"));
        terms.push(format!("{plaintiff} v {defendant}"));
        let plaintiff_last = py_split(plaintiff).last().unwrap_or_default();
        let defendant_last = py_split(defendant).last().unwrap_or_default();
        terms.push(format!("{plaintiff_last} v. {defendant_last}"));
    }
    let normalized = key.normalized_key();
    if !normalized.is_empty() {
        terms.push(normalized);
    }
    terms
}

/// A non-empty field (Python truthiness of an optional string).
fn present(value: &Option<String>) -> Option<&str> {
    value.as_deref().filter(|value| !value.is_empty())
}

/// Text of a participating capture group.
fn group<'t>(caps: &Captures<'t>, index: usize) -> Option<&'t str> {
    caps.get(index).map(|m| m.as_str())
}

/// Owned text of a participating capture group.
fn owned_group(caps: &Captures<'_>, index: usize) -> Option<String> {
    group(caps, index).map(str::to_owned)
}

/// The `[.\s]` class of the Python key normalisation.
fn is_dot_or_space(c: char) -> bool {
    c == '.' || is_py_whitespace(c)
}

/// Replaces every maximal run of characters matching `class` with `replacement`
/// (`re.sub(r'[...]+', replacement, text)`).
fn replace_runs(text: &str, class: fn(char) -> bool, replacement: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_run = false;
    for c in text.chars() {
        if class(c) {
            if !in_run {
                out.push_str(replacement);
                in_run = true;
            }
        } else {
            out.push(c);
            in_run = false;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replace_runs_matches_re_sub() {
        assert_eq!(replace_runs("civ. code", is_dot_or_space, "_"), "civ_code");
        assert_eq!(replace_runs(".a..b. ", is_dot_or_space, "_"), "_a_b_");
        assert_eq!(replace_runs("f. 3d", is_dot_or_space, ""), "f3d");
        assert_eq!(replace_runs("F.  3d\t", is_py_whitespace, " "), "F. 3d ");
    }

    #[test]
    fn every_fallback_pattern_compiles() {
        let patterns = KeyPatterns::fallback();
        for &(name, _) in FALLBACK_PATTERNS {
            assert!(patterns.get(name).is_some(), "{name}");
        }
    }

    #[test]
    fn present_treats_empty_as_missing() {
        assert_eq!(present(&Some(String::new())), None);
        assert_eq!(present(&Some("a".to_owned())), Some("a"));
        assert_eq!(present(&None), None);
    }
}
