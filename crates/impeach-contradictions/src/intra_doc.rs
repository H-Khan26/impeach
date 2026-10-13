//! Intra-document pairing and pre-filter (the pairing helpers of Python
//! `services/contradictions/service.py`: `_generate_intra_doc_pairs`, `_check_intra_path_a`,
//! `_check_intra_path_b`, `_has_assertion_overlap`, `_are_relationships_compatible`).
//!
//! Candidate pairs come from buckets: `slot:<slot_name>` (when set), `entity:<subject id>`
//! (when linked) and, for facts without a slot, `struct:<value_type|TEXT>:<property_type|
//! ATTRIBUTE>` (raw types). Each unordered pair appears once, lower fact id first.
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
//!
//! Path A / B thresholds (`subject_overlap_threshold`, `predicate_similarity_threshold`,
//! `fuzzy_match_threshold`) come from the cross-document pre-filter config, as in Python.

use std::collections::BTreeSet;

use impeach_config::services::IntraDocFilterConfig;
use impeach_core::{EntityId, ExtractedFact};
use impeach_text::jaccard;
use indexmap::IndexMap;

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
        let mut buckets: IndexMap<String, Vec<usize>> = IndexMap::new();
        for (index, fact) in facts.iter().enumerate() {
            let slot = non_empty(fact.slot_name.as_deref());
            if let Some(slot) = slot {
                buckets.entry(format!("slot:{slot}")).or_default().push(index);
            }
            if let Some(entity) =
                fact.subject_entity_id.as_ref().filter(|id| !id.as_str().is_empty())
            {
                buckets.entry(format!("entity:{entity}")).or_default().push(index);
            }
            if slot.is_none() {
                let value_type = non_empty(fact.value_type.as_deref()).unwrap_or("TEXT");
                let property_type = non_empty(fact.property_type.as_deref()).unwrap_or("ATTRIBUTE");
                buckets
                    .entry(format!("struct:{value_type}:{property_type}"))
                    .or_default()
                    .push(index);
            }
        }

        let mut pairs: BTreeSet<(usize, usize)> = BTreeSet::new();
        for members in buckets.values() {
            for (i, &a) in members.iter().enumerate() {
                for &b in &members[i + 1..] {
                    pairs.insert(ordered_pair(facts, a, b));
                }
            }
        }

        let mut candidates: Vec<IntraDocCandidate> = pairs
            .into_iter()
            .map(|(first, second)| IntraDocCandidate {
                first,
                second,
                same_temporal_context: facts[first].temporal_context
                    == facts[second].temporal_context,
            })
            .collect();
        candidates.sort_by(|x, y| {
            y.same_temporal_context
                .cmp(&x.same_temporal_context)
                .then_with(|| facts[x.first].id.cmp(&facts[y.first].id))
                .then_with(|| facts[x.second].id.cmp(&facts[y.second].id))
                .then_with(|| (x.first, x.second).cmp(&(y.first, y.second)))
        });
        candidates
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
        let assertion_overlap = self.has_assertion_overlap(a, b, prefilter);
        let mut decision = IntraDocDecision {
            outcome: IntraDocOutcome::Pass,
            assertion_overlap,
            path_a_valid: false,
            path_b_valid: false,
            slot_match: false,
            text_overlap: 0.0,
            priority: 0.0,
        };
        if self.config.require_assertion_overlap && !assertion_overlap {
            decision.outcome = IntraDocOutcome::SkipNoAssertionOverlap;
            return decision;
        }

        decision.path_a_valid = self.check_path_a(a, b, prefilter);
        decision.path_b_valid = self.check_path_b(a, b, prefilter);
        decision.slot_match = matches!(
            (non_empty(a.slot_name.as_deref()), non_empty(b.slot_name.as_deref())),
            (Some(slot_a), Some(slot_b)) if slot_a == slot_b
        );
        if !decision.path_a_valid && !decision.path_b_valid && !decision.slot_match {
            decision.outcome = IntraDocOutcome::SkipIncompatibleCategory;
            return decision;
        }

        decision.text_overlap = jaccard(&prefilter.tokenize(&a.text), &prefilter.tokenize(&b.text));
        decision.priority = if a.temporal_context == b.temporal_context {
            decision.text_overlap
        } else {
            decision.text_overlap * TEMPORAL_MISMATCH_PRIORITY_FACTOR
        };
        if !decision.path_b_valid
            && !decision.slot_match
            && decision.text_overlap < self.config.min_text_overlap
        {
            decision.outcome = IntraDocOutcome::SkipLowJaccard;
        }
        decision
    }

    /// Path A for two facts of the same document.
    #[must_use]
    pub fn check_path_a(
        &self,
        a: &ExtractedFact,
        b: &ExtractedFact,
        prefilter: &FactPairPreFilter,
    ) -> bool {
        let (Some(types_a), Some(types_b)) = (raw_types(a), raw_types(b)) else {
            return false;
        };
        if !prefilter.types_match(types_a, types_b) {
            return false;
        }
        let (triple_a, triple_b) = (&a.assertion_triple, &b.assertion_triple);
        let config = prefilter.config();
        prefilter.fuzzy_string_similarity(&triple_a.subject, &triple_b.subject)
            >= config.subject_overlap_threshold
            && prefilter.predicate_similarity(&triple_a.relationship, &triple_b.relationship)
                >= config.predicate_similarity_threshold
    }

    /// Path B for two facts of the same document.
    #[must_use]
    pub fn check_path_b(
        &self,
        a: &ExtractedFact,
        b: &ExtractedFact,
        prefilter: &FactPairPreFilter,
    ) -> bool {
        if !share_linked_entity(a, b) {
            return false;
        }
        let (triple_a, triple_b) = (&a.assertion_triple, &b.assertion_triple);
        prefilter
            .best_triple_match(
                (&triple_a.subject, &triple_a.object),
                (&triple_b.subject, &triple_b.object),
            )
            .is_some()
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
        let norm = |name: &str| prefilter.normalize_entity_name(name);
        let (subj_a, obj_a) = (norm(&a.assertion_triple.subject), norm(&a.assertion_triple.object));
        let (subj_b, obj_b) = (norm(&b.assertion_triple.subject), norm(&b.assertion_triple.object));
        let same = |x: &str, y: &str| !x.is_empty() && x == y;
        same(&subj_a, &subj_b)
            || same(&obj_a, &obj_b)
            || same(&subj_a, &obj_b)
            || same(&obj_a, &subj_b)
            || share_linked_entity(a, b)
    }

    /// Whether the relationships of the two facts can conflict: the mappings relationship
    /// types decide (lowercased; unclassified relationships are compatible); true when either
    /// relationship is empty. The registry is always loaded, so the config's
    /// `*_relationships` keyword lists (Python's fallback without a registry) are unused.
    /// (Python defined this check but never called it; it is kept as a helper and not
    /// applied by [`IntraDocFilter::evaluate_pair`].)
    #[must_use]
    pub fn relationships_compatible(
        &self,
        a: &ExtractedFact,
        b: &ExtractedFact,
        prefilter: &FactPairPreFilter,
    ) -> bool {
        let rel_a = a.assertion_triple.relationship.to_lowercase();
        let rel_b = b.assertion_triple.relationship.to_lowercase();
        if rel_a.is_empty() || rel_b.is_empty() {
            return true;
        }
        prefilter.mappings().relationship_types.are_compatible(&rel_a, &rel_b)
    }
}

/// `(lower-id index, higher-id index)`; equal ids are ordered by index.
fn ordered_pair(facts: &[ExtractedFact], a: usize, b: usize) -> (usize, usize) {
    let key = |i: usize| (&facts[i].id, i);
    if key(a) <= key(b) {
        (a, b)
    } else {
        (b, a)
    }
}

/// Linked subject / object entity ids of a fact (non-empty ones).
fn linked_entities(fact: &ExtractedFact) -> impl Iterator<Item = &EntityId> {
    [fact.subject_entity_id.as_ref(), fact.object_entity_id.as_ref()]
        .into_iter()
        .flatten()
        .filter(|id| !id.as_str().is_empty())
}

/// Whether the facts share a linked subject / object entity id.
fn share_linked_entity(a: &ExtractedFact, b: &ExtractedFact) -> bool {
    linked_entities(a).any(|id| linked_entities(b).any(|other| other == id))
}

/// `Some(value)` when it is present and non-empty.
fn non_empty(value: Option<&str>) -> Option<&str> {
    value.filter(|s| !s.is_empty())
}

/// The raw `(value type, property type)` of a fact when both are non-empty.
fn raw_types(fact: &ExtractedFact) -> Option<(&str, &str)> {
    Some((non_empty(fact.value_type.as_deref())?, non_empty(fact.property_type.as_deref())?))
}
