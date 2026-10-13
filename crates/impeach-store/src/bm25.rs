//! Okapi BM25 over tantivy-style tokens, used by [`crate::MemoryStore`].
//!
//! Tokens are maximal runs of alphanumeric characters, lowercased, dropping tokens of 40 or
//! more UTF-8 bytes (tantivy's default `SimpleTokenizer` + `RemoveLongFilter` +
//! `LowerCaser`). A document scores
//! `sum over distinct query terms t of idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))`
//! with `idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))`, `k1 = 1.2`, `b = 0.75`, the formula
//! tantivy uses. HelixDB indexes text with tantivy, but its scores also depend on segment
//! statistics and quantised field norms, so the two backends rank alike without producing
//! equal scores.

/// Term-frequency saturation.
pub const K1: f64 = 1.2;
/// Length normalisation.
pub const B: f64 = 0.75;
/// Tokens of this many UTF-8 bytes or more are dropped.
const MAX_TOKEN_BYTES: usize = 40;

/// Splits `text` into lowercase alphanumeric tokens.
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|token| !token.is_empty() && token.len() < MAX_TOKEN_BYTES)
        .map(str::to_lowercase)
        .collect()
}

/// Scores every document of `corpus` (token lists) against `query`; returns
/// `(document index, score)` for documents containing at least one query term, in corpus
/// order.
#[must_use]
pub fn score_corpus(corpus: &[&[String]], query: &str) -> Vec<(usize, f64)> {
    let mut terms = tokenize(query);
    terms.sort_unstable();
    terms.dedup();
    if terms.is_empty() || corpus.is_empty() {
        return Vec::new();
    }
    let n = corpus.len() as f64;
    let total_len: usize = corpus.iter().map(|doc| doc.len()).sum();
    let avgdl = if total_len == 0 { 1.0 } else { total_len as f64 / n };
    let idf: Vec<f64> = terms
        .iter()
        .map(|term| {
            let df = corpus.iter().filter(|doc| doc.contains(term)).count() as f64;
            (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
        })
        .collect();
    corpus
        .iter()
        .enumerate()
        .filter_map(|(index, doc)| {
            let dl = doc.len() as f64;
            let norm = K1 * (1.0 - B + B * dl / avgdl);
            let mut score = 0.0;
            let mut matched = false;
            for (term, idf) in terms.iter().zip(&idf) {
                let tf = doc.iter().filter(|token| *token == term).count() as f64;
                if tf > 0.0 {
                    matched = true;
                    score += idf * tf * (K1 + 1.0) / (tf + norm);
                }
            }
            matched.then_some((index, score))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn tokenizes_like_tantivy_simple_tokenizer() {
        assert_eq!(
            tokenize(r#"["Acme Corp.", "Défendant", "O'Neil"]"#),
            vec!["acme", "corp", "défendant", "o", "neil"]
        );
        assert!(tokenize(&"x".repeat(40)).is_empty());
        assert_eq!(tokenize(&"x".repeat(39)).len(), 1);
        assert!(tokenize("  ,.; ").is_empty());
    }

    #[test]
    fn scores_rank_rarer_and_denser_matches_higher() {
        let docs: Vec<Vec<String>> = ["green valley logistics", "valley farm", "acme corp", "acme"]
            .iter()
            .map(|text| tokenize(text))
            .collect();
        let corpus: Vec<&[String]> = docs.iter().map(Vec::as_slice).collect();
        let hits = score_corpus(&corpus, "Green Valley");
        assert_eq!(hits.iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![0, 1]);
        assert!(hits[0].1 > hits[1].1);
        let acme = score_corpus(&corpus, "acme");
        assert_eq!(acme.len(), 2);
        assert!(acme[1].1 > acme[0].1, "the shorter document scores higher");
        assert!(score_corpus(&corpus, "  ").is_empty());
        assert!(score_corpus(&corpus, "missing").is_empty());
    }

    #[test]
    fn single_term_score_matches_the_formula() {
        let docs = [tokenize("alpha beta"), tokenize("gamma")];
        let corpus: Vec<&[String]> = docs.iter().map(Vec::as_slice).collect();
        let hits = score_corpus(&corpus, "alpha alpha");
        let idf = (1.0_f64 + (2.0 - 1.0 + 0.5) / (1.0 + 0.5)).ln();
        let avgdl = 1.5;
        let expected = idf * (K1 + 1.0) / (1.0 + K1 * (1.0 - B + B * 2.0 / avgdl));
        assert_eq!(hits.len(), 1);
        assert!((hits[0].1 - expected).abs() < 1e-12);
    }
}
