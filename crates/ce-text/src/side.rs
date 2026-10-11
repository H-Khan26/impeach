//! Party-side normalization.

use crate::normalize::py_strip;

/// The canonical side values.
pub const CANONICAL_SIDES: [&str; 4] = ["DEFENSE", "PROSECUTION", "PLAINTIFF", "NEUTRAL"];

/// Maps a raw side string to the canonical vocabulary.
///
/// Missing or blank input is `NEUTRAL` (to preserve recall when side metadata is absent).
/// Matching is case-insensitive after stripping whitespace, with the aliases `DEFENDANT` and
/// `DEFENCE` -> `DEFENSE` and `PROSECUTOR` and `STATE` -> `PROSECUTION`. Unknown values return
/// `None` so callers decide whether to skip or fall back to `NEUTRAL`.
///
/// Python origin: `utils/side_utils.py:normalize_side`.
#[must_use]
pub fn normalize_side(raw: Option<&str>) -> Option<&'static str> {
    let Some(raw) = raw else {
        return Some("NEUTRAL");
    };
    let value = py_strip(raw);
    if value.is_empty() {
        return Some("NEUTRAL");
    }
    match value.to_uppercase().as_str() {
        "DEFENSE" | "DEFENDANT" | "DEFENCE" => Some("DEFENSE"),
        "PROSECUTION" | "PROSECUTOR" | "STATE" => Some("PROSECUTION"),
        "PLAINTIFF" => Some("PLAINTIFF"),
        "NEUTRAL" => Some("NEUTRAL"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Cases from tests/unit/test_side_utils.py.
    #[test]
    fn canonical_sides_unchanged() {
        for side in CANONICAL_SIDES {
            assert_eq!(normalize_side(Some(side)), Some(side));
        }
    }

    #[test]
    fn case_insensitive() {
        assert_eq!(normalize_side(Some("defense")), Some("DEFENSE"));
        assert_eq!(normalize_side(Some("Defense")), Some("DEFENSE"));
        assert_eq!(normalize_side(Some("DEFENSE")), Some("DEFENSE"));
        assert_eq!(normalize_side(Some("prosecution")), Some("PROSECUTION"));
        assert_eq!(normalize_side(Some("Plaintiff")), Some("PLAINTIFF"));
        assert_eq!(normalize_side(Some("neutral")), Some("NEUTRAL"));
    }

    #[test]
    fn whitespace_stripped() {
        assert_eq!(normalize_side(Some("  DEFENSE  ")), Some("DEFENSE"));
        assert_eq!(normalize_side(Some("\tPROSECUTION\n")), Some("PROSECUTION"));
    }

    #[test]
    fn none_defaults_to_neutral() {
        assert_eq!(normalize_side(None), Some("NEUTRAL"));
    }

    #[test]
    fn empty_string_defaults_to_neutral() {
        assert_eq!(normalize_side(Some("")), Some("NEUTRAL"));
        assert_eq!(normalize_side(Some("   ")), Some("NEUTRAL"));
    }

    #[test]
    fn unknown_value_returns_none() {
        assert_eq!(normalize_side(Some("UNKNOWN")), None);
        assert_eq!(normalize_side(Some("random_string")), None);
    }

    #[test]
    fn aliases_map_to_canonical() {
        assert_eq!(normalize_side(Some("DEFENDANT")), Some("DEFENSE"));
        assert_eq!(normalize_side(Some("defence")), Some("DEFENSE"));
        assert_eq!(normalize_side(Some("Prosecutor")), Some("PROSECUTION"));
        assert_eq!(normalize_side(Some("state")), Some("PROSECUTION"));
    }
}
