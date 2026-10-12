//! Name normalisation and loose name matching for intra-document matching and reuse checks
//! (Python `services/ed/name_matching.py`).

use std::collections::HashSet;

use ce_config::mappings::EntityTypes;
use ce_text::normalize::py_split;
use ce_text::normalize_whitespace;

/// Lowercases, removes `.,;:!?'"()[]{}` and collapses whitespace.
#[must_use]
pub fn normalize_for_matching(name: &str) -> String {
    let lowered: String = name
        .to_lowercase()
        .chars()
        .filter(|c| !matches!(c, '.' | ',' | ';' | ':' | '!' | '?' | '\'' | '"' | '(' | ')'))
        .filter(|c| !matches!(c, '[' | ']' | '{' | '}'))
        .collect();
    normalize_whitespace(&lowered)
}

/// Whether two names likely refer to the same entity: equal after
/// [`normalize_for_matching`], or the shorter (at least 4 characters) is a substring of the
/// longer and its words are a subset of the longer's words.
#[must_use]
pub fn names_match(name1: &str, name2: &str) -> bool {
    let n1 = normalize_for_matching(name1);
    let n2 = normalize_for_matching(name2);
    if n1.is_empty() || n2.is_empty() {
        return false;
    }
    if n1 == n2 {
        return true;
    }
    let (shorter, longer) =
        if n1.chars().count() <= n2.chars().count() { (&n1, &n2) } else { (&n2, &n1) };
    if shorter.chars().count() >= 4 && longer.contains(shorter.as_str()) {
        let words_longer: HashSet<&str> = py_split(longer).collect();
        return py_split(shorter).all(|word| words_longer.contains(word));
    }
    false
}

/// Whether two entity types may resolve to the same entity. The single compatibility rule of
/// the crate: delegates to [`EntityTypes::are_types_compatible`] (`incompatible_type_groups` of
/// `entity_types.yaml`).
#[must_use]
pub fn entity_types_compatible(
    entity_types: &EntityTypes,
    type1: Option<&str>,
    type2: Option<&str>,
) -> bool {
    entity_types.are_types_compatible(type1, type2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ce_config::MappingsRegistry;

    fn types() -> &'static EntityTypes {
        &MappingsRegistry::load().expect("embedded mappings load").entity_types
    }

    #[test]
    fn normalize_strips_punctuation_and_whitespace() {
        assert_eq!(normalize_for_matching("  Mr. O'Brien,  (Jr.) "), "mr obrien jr");
        assert_eq!(normalize_for_matching("MARIA\tLOPEZ"), "maria lopez");
        assert_eq!(normalize_for_matching("[A]{B}\"C\";:!?"), "abc");
        assert_eq!(normalize_for_matching("Smith-Jones"), "smith-jones");
        assert_eq!(normalize_for_matching(""), "");
        assert_eq!(normalize_for_matching(" .,; "), "");
    }

    #[test]
    fn names_match_cases() {
        assert!(names_match("Maria Lopez", "Maria Lopez"));
        assert!(names_match("MARIA LOPEZ", "maria lopez"));
        assert!(names_match("Maria Lopez", "Lopez"));
        assert!(names_match("Mr. Smith", "Smith"));
        assert!(names_match("Dr. Maria Lopez", "Maria Lopez"));
        // Shorter than four characters.
        assert!(!names_match("Lee", "Bruce Lee"));
        // Substring but not a word subset.
        assert!(!names_match("Maria", "Marianne Lopez"));
        assert!(!names_match("ria lo", "Maria Lopez"));
        assert!(!names_match("", "Maria"));
        assert!(!names_match("...", "..."));
        assert!(!names_match("Maria Lopez", "John Smith"));
    }

    #[test]
    fn names_match_counts_characters() {
        // Four characters, eight bytes.
        assert!(names_match("Élan", "Société Élan"));
        assert!(!names_match("Él", "Société Él"));
    }

    #[test]
    fn venue_types_are_compatible() {
        for (a, b) in [("ORG", "LOC"), ("ORG", "FACILITY"), ("LOC", "FACILITY"), ("GPE", "ORG")] {
            assert!(entity_types_compatible(types(), Some(a), Some(b)), "{a} vs {b}");
            assert!(entity_types_compatible(types(), Some(b), Some(a)), "{b} vs {a}");
        }
    }

    #[test]
    fn other_cross_group_types_are_incompatible() {
        for (a, b) in [
            ("PER", "ORG"),
            ("PERSON", "ORGANIZATION"),
            ("PER", "LOC"),
            ("PRODUCT", "LOC"),
            ("PRODUCT", "FACILITY"),
            ("DATE", "MONEY"),
            ("ORG", "DATE"),
        ] {
            assert!(!entity_types_compatible(types(), Some(a), Some(b)), "{a} vs {b}");
            assert!(!entity_types_compatible(types(), Some(b), Some(a)), "{b} vs {a}");
        }
    }

    #[test]
    fn missing_and_equal_types_are_compatible() {
        assert!(entity_types_compatible(types(), None, Some("PER")));
        assert!(entity_types_compatible(types(), Some("ORG"), None));
        assert!(entity_types_compatible(types(), Some(""), Some("PER")));
        assert!(entity_types_compatible(types(), Some("PERSON"), Some("PER")));
    }

    #[test]
    fn each_type_is_in_at_most_one_group() {
        let seen: Vec<&String> = types().incompatible_type_groups.iter().flatten().collect();
        let unique: HashSet<&String> = seen.iter().copied().collect();
        assert_eq!(seen.len(), unique.len());
    }
}
