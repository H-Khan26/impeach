//! Intra-document pairing and pre-filter (the pairing helpers of Python
//! `services/contradictions/service.py`: `_generate_intra_doc_pairs`, `_check_intra_path_a`,
//! `_check_intra_path_b`, `_has_assertion_overlap`, `_are_relationships_compatible`).
//!
//! Candidate pairs come from buckets: `slot:<slot_name>` (when set), `entity:<subject id>`
//! (when linked) and, for facts without a slot, `struct:<value_type|TEXT>:<property_type|
//! ATTRIBUTE>`. Each unordered pair appears once, lower fact id first.
//!
//! Corrected behaviour (Python skipped every pair whose `temporal_context` differed): a
//! temporal-context mismatch never removes a pair. It is a priority signal only: pairs with
//! the same context come first, and a mismatched pair's priority is multiplied by
//! [`TEMPORAL_MISMATCH_PRIORITY_FACTOR`].
//!
//! [`IntraDocFilter::evaluate_pair`] applies, in order:
//! 1. `require_assertion_overlap` (accepted but never read in Python, implemented here): a
//!    pair without assertion overlap ([`IntraDocFilter::has_assertion_overlap`]) is skipped.
//! 2. Path A (structural: types, subject overlap, predicate similarity) OR Path B (shared
//!    entity id and fuzzy subject / object match) OR the same non-empty `slot_name`; else the
//!    pair is skipped as an incompatible category.
//! 3. When only Path A holds (no Path B, no slot match), the text Jaccard must reach
//!    `min_text_overlap`.

use ce_config::services::IntraDocFilterConfig;
use ce_core::ExtractedFact;

use crate::prefilter::FactPairPreFilter;

/// Priority multiplier for a pair whose facts have different `temporal_context` values.
pub const TEMPORAL_MISMATCH_PRIORITY_FACTOR: f64 = 0.5;

/// One candidate pair: indexes into the fact slice, `first` has the lower fact id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntraDocCandidate {
    /// Index of the fact with the lower id.
    pub first: usize,
    /// Index of the fact with the higher id.
    pub second: usize,
    /// Whether both facts have the same `temporal_context`.
    pub same_temporal_context: bool,
}

/// Why a pair was skipped, or that it passed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntraDocOutcome {
    /// The pair goes to the judge.
    Pass,
    /// `require_assertion_overlap` and the facts share no subject / object.
    SkipNoAssertionOverlap,
    /// Neither Path A nor Path B holds and the slots differ.
    SkipIncompatibleCategory,
    /// Only Path A holds and the text Jaccard is below `min_text_overlap`.
    SkipLowJaccard,
}

/// The pre-filter decision for one pair, with the signals the debug trace records.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntraDocDecision {
    /// Outcome.
    pub outcome: IntraDocOutcome,
    /// Whether the facts have assertion overlap.
    pub assertion_overlap: bool,
    /// Path A result.
    pub path_a_valid: bool,
    /// Path B result.
    pub path_b_valid: bool,
    /// Same non-empty `slot_name`.
    pub slot_match: bool,
    /// Token Jaccard of the two fact texts (0.0 when not computed).
    pub text_overlap: f64,
    /// Pair priority: `text_overlap`, times [`TEMPORAL_MISMATCH_PRIORITY_FACTOR`] when the
    /// temporal contexts differ.
    pub priority: f64,
}

/// Intra-document pairing and filtering.
#[derive(Debug, Clone)]
pub struct IntraDocFilter {
    config: IntraDocFilterConfig,
}

impl IntraDocFilter {
    /// A filter with `config`.
    #[must_use]
    pub fn new(config: IntraDocFilterConfig) -> Self {
        Self { config }
    }

    /// The intra-document settings.
    #[must_use]
    pub fn config(&self) -> &IntraDocFilterConfig {
        &self.config
    }

    /// Bucketed candidate pairs (module docs), same temporal context first, then by
    /// `(first id, second id)`.
    #[must_use]
    pub fn candidate_pairs(&self, facts: &[ExtractedFact]) -> Vec<IntraDocCandidate> {
        todo!("worker A")
    }

    /// The pre-filter decision for `(a, b)` (module docs). `prefilter` supplies tokenisation,
    /// fuzzy matching, predicate similarity and the Path A / B thresholds.
    #[must_use]
    pub fn evaluate_pair(
        &self,
        a: &ExtractedFact,
        b: &ExtractedFact,
        prefilter: &FactPairPreFilter,
    ) -> IntraDocDecision {
        todo!("worker A")
    }

    /// Path A for two facts of the same document.
    #[must_use]
    pub fn check_path_a(
        &self,
        a: &ExtractedFact,
        b: &ExtractedFact,
        prefilter: &FactPairPreFilter,
    ) -> bool {
        todo!("worker A")
    }

    /// Path B for two facts of the same document.
    #[must_use]
    pub fn check_path_b(
        &self,
        a: &ExtractedFact,
        b: &ExtractedFact,
        prefilter: &FactPairPreFilter,
    ) -> bool {
        todo!("worker A")
    }

    /// Whether the facts share a subject or object: equal normalised names (subject,
    /// object, or crossed), or a common linked subject / object entity id.
    #[must_use]
    pub fn has_assertion_overlap(
        &self,
        a: &ExtractedFact,
        b: &ExtractedFact,
        prefilter: &FactPairPreFilter,
    ) -> bool {
        todo!("worker A")
    }

    /// Whether the relationships of the two facts can conflict: the mappings relationship
    /// types decide; true when either relationship is empty. The `*_relationships` keyword
    /// lists of the config are the fallback categories. (Python defined this check but never
    /// called it; it is kept as a helper and not applied by [`IntraDocFilter::evaluate_pair`].)
    #[must_use]
    pub fn relationships_compatible(
        &self,
        a: &ExtractedFact,
        b: &ExtractedFact,
        prefilter: &FactPairPreFilter,
    ) -> bool {
        todo!("worker A")
    }
}
