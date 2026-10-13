//! Regex rules from `detection_patterns.yaml` (Python `services/ner/rules.py`).
//!
//! Patterns run in `detection_order`; names without a compiled pattern or an `entity_type`
//! are skipped. Every match becomes a mention of the pattern's `entity_type` covering the
//! whole match (`capture_group` is used by the ED rule resolvers, not by detection), with
//! score 1.0 and source `regex_detector`. Flags (`IGNORECASE`) are applied when the mappings
//! registry compiles the patterns.

use impeach_config::mappings::DetectionPatterns;
use impeach_core::{Mention, Span};
use regex::Regex;

use super::{span_id, MentionDetector};
use crate::error::Result;
use crate::text::CharIndex;

/// Detects structured legal entities (dockets, exhibits, statutes, rules, money, dates).
#[derive(Debug, Clone)]
pub struct RegexDetector {
    patterns: Vec<(Regex, String)>,
}

impl RegexDetector {
    /// Source name of regex mentions.
    pub const SOURCE: &'static str = "regex_detector";

    /// The patterns of `detection_order` with their entity types.
    #[must_use]
    pub fn from_mappings(patterns: &DetectionPatterns) -> Self {
        let patterns = patterns
            .get_ordered_patterns()
            .iter()
            .filter_map(|name| {
                let regex = patterns.get_compiled_pattern(name)?;
                let entity_type = patterns.get_entity_type(name)?;
                Some((regex.clone(), entity_type.to_owned()))
            })
            .collect();
        Self { patterns }
    }

    /// Number of active patterns.
    #[must_use]
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Whether no pattern is active.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }
}

impl MentionDetector for RegexDetector {
    fn source(&self) -> &str {
        Self::SOURCE
    }

    fn detect(&self, text: &str, segment_id: &str) -> Result<Vec<Mention>> {
        let index = CharIndex::new(text);
        let mut mentions = Vec::new();
        for (regex, entity_type) in &self.patterns {
            for found in regex.find_iter(text) {
                let start = index.char_of_byte(found.start());
                let end = index.char_of_byte(found.end());
                let id = span_id(segment_id, Self::SOURCE, start, end, entity_type);
                let mut mention =
                    Mention::new(id, entity_type.as_str(), found.as_str(), Span::new(start, end));
                mention.score = Some(1.0);
                mention.source = Some(Self::SOURCE.to_owned());
                mentions.push(mention);
            }
        }
        Ok(mentions)
    }
}

#[cfg(test)]
mod tests {
    use impeach_config::MappingsRegistry;
    use pretty_assertions::assert_eq;

    use super::*;

    fn detect(text: &str) -> Vec<(String, String, usize, usize)> {
        let registry = MappingsRegistry::load().expect("mappings");
        let detector = RegexDetector::from_mappings(&registry.detection_patterns);
        detector
            .detect(text, "doc_0")
            .expect("detect")
            .into_iter()
            .map(|m| (m.entity_type, m.name, m.span.start, m.span.end))
            .collect()
    }

    #[test]
    fn loads_every_detection_pattern_in_order() {
        let registry = MappingsRegistry::load().expect("mappings");
        let detector = RegexDetector::from_mappings(&registry.detection_patterns);
        assert_eq!(detector.len(), 6);
    }

    #[test]
    fn matches_python_rules_in_detection_order() {
        let text = "Case No. 2:23-cv-01492: see Exhibit 12A, 42 U.S.C. 1983 and Fed. R. Civ. P. \
                    12(b)(6); damages of $1,234.56 million on January 15, 2023 and 01/15/2023.";
        let found = detect(text);
        let s = |needle: &str| text.find(needle).expect("needle");
        assert_eq!(
            found,
            vec![
                ("DOCKET_ENTRY".into(), "Case No. 2:23-cv-01492".into(), 0, 22),
                ("EXHIBIT_ID".into(), "Exhibit 12A".into(), s("Exhibit"), s("Exhibit") + 11),
                ("STATUTE".into(), "42 U.S.C. 1983".into(), s("42 U"), s("42 U") + 14),
                ("STATUTE".into(), "Fed. R. Civ. P. 12(b)(6)".into(), s("Fed."), s("Fed.") + 24),
                ("MONEY".into(), "$1,234.56 million".into(), s("$"), s("$") + 17),
                ("DATE".into(), "January 15, 2023".into(), s("January"), s("January") + 16),
                ("DATE".into(), "01/15/2023".into(), s("01/"), s("01/") + 10),
            ]
        );
    }

    #[test]
    fn offsets_are_characters_and_mentions_carry_python_fields() {
        let registry = MappingsRegistry::load().expect("mappings");
        let detector = RegexDetector::from_mappings(&registry.detection_patterns);
        let mentions = detector.detect("Zoë paid $300 on July 8, 2022.", "d_3").expect("detect");
        assert_eq!(mentions.len(), 2);
        let money = &mentions[0];
        assert_eq!((money.span.start, money.span.end), (9, 13));
        assert_eq!(money.name, "$300");
        assert_eq!(money.score, Some(1.0));
        assert_eq!(money.source.as_deref(), Some("regex_detector"));
        assert!(money.id.as_str().starts_with("span-"));
        assert_eq!(money.id, span_id("d_3", "regex_detector", 9, 13, "MONEY"));
        assert_eq!(mentions[1].name, "July 8, 2022");
    }

    #[test]
    fn ignorecase_flag_applies() {
        let found = detect("exh. 3 and case no. 21-55555");
        assert_eq!(found[0].1, "case no. 21-55555");
        assert_eq!(found[1].1, "exh. 3");
    }
}
