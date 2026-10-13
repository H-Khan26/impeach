//! Fact builders shared by worker B's integration tests.
#![allow(dead_code)]

use ce_core::{AssertionTriple, ExtractedFact, RetrievedFact, TemporalValidity};

/// An extracted fact with the given triple and no validity interval.
pub fn fact(
    id: &str,
    text: &str,
    subject: &str,
    relationship: &str,
    object: &str,
) -> ExtractedFact {
    ExtractedFact::new(
        id,
        text,
        AssertionTriple::new(subject, relationship, object),
        1.0,
        TemporalValidity::default(),
    )
}

/// A retrieved fact with the given relevance score.
pub fn retrieved(id: &str, text: &str, relevance_score: f64) -> RetrievedFact {
    let mut fact = RetrievedFact::new(id, text);
    fact.relevance_score = relevance_score;
    fact
}

/// `actual` equals `expected` within 1e-9.
pub fn assert_close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-9, "expected {expected}, got {actual}");
}
