//! Whitespace and entity-name normalization with Python `str` semantics.

/// True for the characters Python's `str.isspace()` (and `str.split()` / `str.strip()` with no
/// argument) treat as whitespace: Unicode `White_Space` plus the separators U+001C..U+001F.
#[must_use]
pub fn is_py_whitespace(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}

/// Splits on runs of whitespace and drops empty pieces, like Python `str.split()`.
pub fn py_split(text: &str) -> impl Iterator<Item = &str> {
    text.split(is_py_whitespace).filter(|piece| !piece.is_empty())
}

/// Trims leading and trailing whitespace, like Python `str.strip()`.
#[must_use]
pub fn py_strip(text: &str) -> &str {
    text.trim_matches(is_py_whitespace)
}

/// Collapses every whitespace run to one space and trims the ends: `" ".join(text.split())`.
///
/// Python origin: `config/mappings/__init__.py:QualityFilters.normalize_mention_name` (which
/// also maps `\n` to a space first, a no-op under this rule), `services/rag/document_loader.py:
/// _normalize_header_footer_line` (first step) and `tests/e2e/evaluate_output.py:clean_fact_text`
/// (last step).
#[must_use]
pub fn normalize_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for piece in py_split(text) {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(piece);
    }
    out
}

/// Lowercases and collapses whitespace: `" ".join(text.lower().split())`.
///
/// Python origin: `tests/e2e/evaluate_output.py:_norm_text`.
#[must_use]
pub fn normalize_whitespace_lower(text: &str) -> String {
    normalize_whitespace(&text.to_lowercase())
}

/// Leading articles stripped by the fallback entity-name normalization.
///
/// Python origin: `services/contradictions/prefilter.py:_FALLBACK_PREFIXES` (also inlined in
/// `services/contradictions/service.py:_normalize_entity`).
pub const FALLBACK_ENTITY_PREFIXES: [&str; 3] = ["the ", "a ", "an "];

/// Corporate suffixes stripped by the fallback entity-name normalization.
///
/// Python origin: `services/contradictions/prefilter.py:_FALLBACK_SUFFIXES` (also inlined in
/// `services/contradictions/service.py:_normalize_entity`).
pub const FALLBACK_ENTITY_SUFFIXES: [&str; 6] =
    [" inc", " inc.", " llc", " corp", " corporation", " company"];

/// Normalizes an entity name for comparison: optionally lowercases, strips, then removes each
/// prefix (in order, at most once each) and each suffix (in order, at most once each), then
/// strips again. Empty input yields an empty string.
///
/// Python origin: `config/mappings/__init__.py:NormalizationConfig.normalize_entity_name`, and
/// the fallbacks in `services/contradictions/prefilter.py:_normalize_entity_name` and
/// `services/contradictions/service.py:_normalize_entity`.
///
/// An empty suffix is ignored (Python's `normalized[:-0]` would empty the whole string).
#[must_use]
pub fn normalize_entity_name<P, S>(
    name: &str,
    prefixes: &[P],
    suffixes: &[S],
    lowercase: bool,
) -> String
where
    P: AsRef<str>,
    S: AsRef<str>,
{
    if name.is_empty() {
        return String::new();
    }
    let lowered;
    let base = if lowercase {
        lowered = name.to_lowercase();
        lowered.as_str()
    } else {
        name
    };
    let mut normalized = py_strip(base);
    for prefix in prefixes {
        if let Some(rest) = normalized.strip_prefix(prefix.as_ref()) {
            normalized = rest;
        }
    }
    for suffix in suffixes {
        let suffix = suffix.as_ref();
        if suffix.is_empty() {
            continue;
        }
        if let Some(rest) = normalized.strip_suffix(suffix) {
            normalized = rest;
        }
    }
    py_strip(normalized).to_owned()
}

/// [`normalize_entity_name`] with the hardcoded fallback prefixes and suffixes, lowercasing on.
///
/// Python origin: `services/contradictions/prefilter.py:FactPairPreFilter._normalize_entity_name`
/// (fallback branch, used when the mappings registry is unavailable).
#[must_use]
pub fn normalize_entity_name_fallback(name: &str) -> String {
    normalize_entity_name(name, &FALLBACK_ENTITY_PREFIXES, &FALLBACK_ENTITY_SUFFIXES, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_whitespace_includes_file_separators() {
        assert!(is_py_whitespace('\u{1c}'));
        assert!(is_py_whitespace('\u{1f}'));
        assert!(is_py_whitespace('\u{a0}'));
        assert!(is_py_whitespace('\u{3000}'));
        assert!(!is_py_whitespace('\u{200b}'));
        assert!(!is_py_whitespace('x'));
    }

    #[test]
    fn collapses_whitespace() {
        assert_eq!(normalize_whitespace("  a \t b\n\nc  "), "a b c");
        assert_eq!(normalize_whitespace("a\u{1d}b\u{a0}c"), "a b c");
        assert_eq!(normalize_whitespace(""), "");
        assert_eq!(normalize_whitespace(" \n\t "), "");
        assert_eq!(normalize_whitespace("John\nSmith"), "John Smith");
    }

    #[test]
    fn collapses_and_lowercases() {
        assert_eq!(normalize_whitespace_lower("  The  QUICK\tFox "), "the quick fox");
        assert_eq!(normalize_whitespace_lower("ÄRZTIN  Ω"), "ärztin ω");
    }

    #[test]
    fn strips_python_whitespace() {
        assert_eq!(py_strip("\u{1c} x y \u{2003}"), "x y");
        assert_eq!(py_split(" a  b ").collect::<Vec<_>>(), vec!["a", "b"]);
    }

    #[test]
    fn fallback_entity_name_matches_python() {
        assert_eq!(normalize_entity_name_fallback("The Acme Corp"), "acme");
        assert_eq!(normalize_entity_name_fallback("  Acme Inc.  "), "acme");
        assert_eq!(normalize_entity_name_fallback("Acme Inc"), "acme");
        assert_eq!(normalize_entity_name_fallback("Acme Widget Company"), "acme widget");
        assert_eq!(normalize_entity_name_fallback("A Big Company LLC"), "big");
        assert_eq!(normalize_entity_name_fallback(""), "");
        // Prefixes apply in order, once each.
        assert_eq!(normalize_entity_name_fallback("The A Team"), "team");
        assert_eq!(normalize_entity_name_fallback("An Apple"), "apple");
        assert_eq!(normalize_entity_name_fallback("Theater"), "theater");
    }

    #[test]
    fn configurable_entity_name() {
        let prefixes = vec!["mr. ".to_owned()];
        let suffixes = vec![" jr".to_owned(), String::new()];
        assert_eq!(
            normalize_entity_name("Mr. John Smith Jr", &prefixes, &suffixes, true),
            "john smith"
        );
        assert_eq!(normalize_entity_name(" Mr. X ", &prefixes, &suffixes, false), "Mr. X");
    }
}
