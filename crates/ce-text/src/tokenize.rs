//! Alphanumeric token sets and Jaccard overlap.

use std::borrow::{Borrow, Cow};
use std::collections::HashSet;
use std::hash::Hash;
use std::sync::LazyLock;

/// Stopwords used when the mappings registry is unavailable.
///
/// Python origin: `services/contradictions/prefilter.py:_FALLBACK_STOPWORDS`.
pub const FALLBACK_STOPWORDS: [&str; 34] = [
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "it", "that", "this", "these", "those",
];

static FALLBACK_STOPWORD_SET: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| FALLBACK_STOPWORDS.into_iter().collect());

static NO_STOPWORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(HashSet::new);

/// [`FALLBACK_STOPWORDS`] as a set.
#[must_use]
pub fn fallback_stopwords() -> &'static HashSet<&'static str> {
    &FALLBACK_STOPWORD_SET
}

/// Splits `text` into maximal runs of alphanumeric characters and returns the distinct runs
/// that are not stopwords and have at least `min_token_length` characters. With `lowercase`
/// the text is lowercased before splitting (stopwords are compared after lowercasing).
///
/// Python origin: `config/mappings/__init__.py:NormalizationConfig.tokenize`; the same loop is
/// duplicated in `services/contradictions/prefilter.py:FactPairPreFilter._tokenize` (fallback
/// branch) and `services/contradictions/gating.py:ContradictionGatingService._tokenize`.
/// Python's `str.isalnum` is approximated by [`char::is_alphanumeric`].
#[must_use]
pub fn token_set<S>(
    text: &str,
    stopwords: &HashSet<S>,
    min_token_length: usize,
    lowercase: bool,
) -> HashSet<String>
where
    S: Borrow<str> + Eq + Hash,
{
    token_set_by(text, |token| stopwords.contains(token), min_token_length, lowercase)
}

/// [`token_set`] with the stopword test supplied as a predicate, for stopword collections
/// other than a `HashSet`.
#[must_use]
pub fn token_set_by(
    text: &str,
    is_stopword: impl Fn(&str) -> bool,
    min_token_length: usize,
    lowercase: bool,
) -> HashSet<String> {
    let processed: Cow<'_, str> =
        if lowercase { Cow::Owned(text.to_lowercase()) } else { Cow::Borrowed(text) };
    processed
        .split(|c: char| !c.is_alphanumeric())
        .filter(|token| {
            !token.is_empty() && !is_stopword(token) && token.chars().count() >= min_token_length
        })
        .map(str::to_owned)
        .collect()
}

/// Token set with the fallback stopwords, minimum length 2, lowercased.
///
/// Python origin: `services/contradictions/prefilter.py:FactPairPreFilter._tokenize` (fallback
/// branch, used when the mappings registry is unavailable).
#[must_use]
pub fn fallback_token_set(text: &str) -> HashSet<String> {
    token_set(text, fallback_stopwords(), 2, true)
}

/// Token set with no stopwords, minimum length 2, lowercased.
///
/// Python origin: `services/contradictions/gating.py:ContradictionGatingService._tokenize`.
#[must_use]
pub fn keyword_token_set(text: &str) -> HashSet<String> {
    token_set(text, &NO_STOPWORDS, 2, true)
}

/// Jaccard similarity `|a ∩ b| / |a ∪ b|` on a 0-1 scale; 0.0 when either set is empty.
///
/// Python origin: `services/contradictions/prefilter.py:FactPairPreFilter._jaccard_similarity`
/// and `services/contradictions/service.py:ContradictionCheckService._jaccard_similarity`
/// (identical), and the tail of `NormalizationConfig.jaccard_similarity`.
#[must_use]
pub fn jaccard<T: Eq + Hash>(a: &HashSet<T>, b: &HashSet<T>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let intersection = small.iter().filter(|item| large.contains(*item)).count();
    let union = a.len() + b.len() - intersection;
    intersection as f64 / union as f64
}

/// Keyword overlap of two texts: Jaccard of their [`keyword_token_set`]s, 0-1 scale; 0.0 when
/// either text is empty or yields no tokens.
///
/// Python origin: `services/contradictions/gating.py:ContradictionGatingService.
/// _compute_keyword_overlap`.
#[must_use]
pub fn keyword_overlap(text_a: &str, text_b: &str) -> f64 {
    if text_a.is_empty() || text_b.is_empty() {
        return 0.0;
    }
    jaccard(&keyword_token_set(text_a), &keyword_token_set(text_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(items: &[&str]) -> HashSet<String> {
        items.iter().map(|s| (*s).to_owned()).collect()
    }

    // Cases from tests/unit/test_prefilter.py::TestJaccardSimilarity.
    #[test]
    fn jaccard_cases() {
        let tokens = set(&["apple", "banana"]);
        assert!((jaccard(&tokens, &tokens) - 1.0).abs() < f64::EPSILON);
        assert!(jaccard(&set(&["apple", "banana"]), &set(&["dog", "cat"])).abs() < f64::EPSILON);
        assert!((jaccard(&set(&["a", "b", "c"]), &set(&["b", "c", "d"])) - 0.5).abs() < 1e-12);
        assert!(jaccard(&set(&[]), &set(&["a", "b"])).abs() < f64::EPSILON);
        assert!(jaccard(&set(&["a", "b"]), &set(&[])).abs() < f64::EPSILON);
        assert!(jaccard(&set(&[]), &set(&[])).abs() < f64::EPSILON);
        assert!((jaccard(&set(&["a", "b"]), &set(&["a", "b", "c", "d"])) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn fallback_tokenizer_drops_stopwords_and_short_tokens() {
        assert_eq!(
            fallback_token_set("The car was RED, and it's a 2019 model x."),
            set(&["car", "red", "2019", "model"])
        );
        assert_eq!(fallback_token_set(""), set(&[]));
        assert_eq!(fallback_token_set("café-owner's"), set(&["café", "owner"]));
    }

    #[test]
    fn keyword_tokenizer_keeps_stopwords() {
        assert_eq!(keyword_token_set("The car is red"), set(&["the", "car", "is", "red"]));
    }

    #[test]
    fn custom_stopwords_and_case() {
        let stop: HashSet<String> = set(&["foo"]);
        assert_eq!(token_set("Foo foo BAR", &stop, 1, false), set(&["Foo", "BAR"]));
        assert_eq!(token_set("Foo foo BAR", &stop, 1, true), set(&["bar"]));
        assert_eq!(token_set("ab abc", &stop, 3, true), set(&["abc"]));
    }

    #[test]
    fn keyword_overlap_matches_gating() {
        assert!(keyword_overlap("", "a b").abs() < f64::EPSILON);
        assert!(keyword_overlap("a b", "c d").abs() < f64::EPSILON);
        let overlap = keyword_overlap("The car was red", "the car was blue");
        assert!((overlap - 3.0 / 5.0).abs() < 1e-12);
    }
}
