//! Deterministic resolvers for rule-first entity types (Python
//! `services/ed/rule_resolvers.py`). Each returns a result only when exactly one candidate
//! matches; `None` lets the ED service fall through to its next route.
//!
//! Strategies and confidences: `rule_case_docket` 1.0, `rule_case_citation` /
//! `rule_case_caption` 0.95 (0.90 when found by the case-name fallback), `rule_case_name` 0.90;
//! `rule_statute_structured` 0.95, `rule_statute_normalized` 0.90, `rule_statute_exact` 0.95;
//! `rule_exhibit_id` 0.95, `rule_exhibit_party` 0.90; `rule_generic_exact` 0.9.

use ce_core::{EdCandidate, EdContext, EdResult, EntityId};

use crate::error::Result;
use crate::retrieval::{EdRetrieval, StructuredKeyQuery};
use crate::structured_keys::{
    extract_case_citation_key, extract_exhibit_key, extract_statute_key, KeyPatterns,
};

/// A rule result choosing `chosen` among `candidates`.
fn rule_result(
    chosen: EntityId,
    candidates: Vec<EdCandidate>,
    confidence: f64,
    strategy: &str,
) -> EdResult {
    EdResult {
        mention_id: None,
        chosen_entity_id: Some(chosen),
        candidates,
        resolved_by_llm: false,
        confidence,
        strategy_used: strategy.to_owned(),
    }
}

/// The rule result of the single candidate, `None` unless there is exactly one.
fn unique(candidates: Vec<EdCandidate>, confidence: f64, strategy: &str) -> Option<EdResult> {
    if candidates.len() != 1 {
        return None;
    }
    let chosen = candidates[0].entity_id.clone();
    Some(rule_result(chosen, candidates, confidence, strategy))
}

/// First capture group of `pattern` in `text`; an unmatched group is empty.
fn first_group<'t>(patterns: &KeyPatterns, name: &str, text: &'t str) -> Option<&'t str> {
    let captures = patterns.get(name)?.captures(text)?;
    Some(captures.get(1).map_or("", |group| group.as_str()))
}

fn is_set(value: Option<&String>) -> bool {
    value.is_some_and(|v| !v.is_empty())
}

/// CASE / MATTER mentions: docket number, then citation or caption, then exact case name.
///
/// # Errors
/// Fatal store errors.
pub fn resolve_case_rule(
    ctx: &EdContext,
    retriever: &dyn EdRetrieval,
    patterns: &KeyPatterns,
) -> Result<Option<EdResult>> {
    let text = ctx.mention.name.as_str();
    let case_id = ctx.doc_context.case_id.as_deref();
    let mut candidates = Vec::new();
    let mut strategy = "rule_case_name";

    if let Some(number) = first_group(patterns, "case_docket", text) {
        candidates = retriever
            .find_by_structured_key(&StructuredKeyQuery::DocketEntry { number }, case_id)?;
        strategy = "rule_case_docket";
        if candidates.len() == 1 {
            return Ok(unique(candidates, 1.0, strategy));
        }
    }

    if let Some(key) = extract_case_citation_key(patterns, text) {
        candidates =
            retriever.find_by_structured_key(&StructuredKeyQuery::CaseCitation(&key), case_id)?;
        strategy = if is_set(key.volume.as_ref()) && is_set(key.reporter.as_ref()) {
            "rule_case_citation"
        } else {
            "rule_case_caption"
        };
        if candidates.len() == 1 {
            return Ok(unique(candidates, 0.95, strategy));
        }
    }

    if candidates.is_empty() {
        candidates = retriever.find_by_exact_match(text, Some("CASE"))?;
        strategy = "rule_case_name";
    }
    Ok(unique(candidates, 0.90, strategy))
}

/// STATUTE / RULE mentions via structured statute keys.
///
/// # Errors
/// Fatal store errors.
pub fn resolve_statute_rule(
    ctx: &EdContext,
    retriever: &dyn EdRetrieval,
    patterns: &KeyPatterns,
) -> Result<Option<EdResult>> {
    let text = ctx.mention.name.as_str();
    let case_id = ctx.doc_context.case_id.as_deref();

    if let Some(key) = extract_statute_key(patterns, text) {
        let candidates =
            retriever.find_by_structured_key(&StructuredKeyQuery::Statute(&key), case_id)?;
        if candidates.len() == 1 {
            return Ok(unique(candidates, 0.95, "rule_statute_structured"));
        }
        if candidates.len() > 1 {
            let normalized = key.normalized_key();
            let matching = candidates.iter().find(|candidate| {
                candidate.metadata.get("normalized_key").and_then(|v| v.as_str())
                    == Some(normalized.as_str())
            });
            if let Some(chosen) = matching.map(|c| c.entity_id.clone()) {
                return Ok(Some(rule_result(chosen, candidates, 0.90, "rule_statute_normalized")));
            }
        }
    }

    let candidates = retriever.find_by_exact_match(text, Some("STATUTE"))?;
    Ok(unique(candidates, 0.95, "rule_statute_exact"))
}

/// EXHIBIT mentions via exhibit keys scoped to the document's case: the key's lookup, then a
/// party match among several candidates, then (for a key with a party) the party-less
/// `exhibit_id_simple` lookup. [`extract_exhibit_key`] already falls back to
/// `exhibit_id_simple`, so without a key nothing matches and a key without a party is that
/// lookup itself.
///
/// # Errors
/// Fatal store errors.
pub fn resolve_exhibit_rule(
    ctx: &EdContext,
    retriever: &dyn EdRetrieval,
    patterns: &KeyPatterns,
) -> Result<Option<EdResult>> {
    let text = ctx.mention.name.as_str();
    let case_id = ctx.doc_context.case_id.as_deref();
    let Some(key) = extract_exhibit_key(patterns, text, case_id) else {
        return Ok(None);
    };

    let party = key.party_attribution.as_deref();
    let query = StructuredKeyQuery::Exhibit { id: &key.exhibit_id, party_attribution: party };
    let candidates = retriever.find_by_structured_key(&query, case_id)?;
    if candidates.len() == 1 {
        return Ok(unique(candidates, 0.95, "rule_exhibit_id"));
    }
    let Some(party) = party else {
        return Ok(None);
    };
    if !party.is_empty() && candidates.len() > 1 {
        let party = party.to_lowercase();
        let mut party_matches = candidates
            .iter()
            .filter(|candidate| candidate.canonical_name.to_lowercase().contains(&party));
        if let (Some(chosen), None) = (party_matches.next(), party_matches.next()) {
            let chosen = chosen.entity_id.clone();
            return Ok(Some(rule_result(chosen, candidates, 0.90, "rule_exhibit_party")));
        }
    }

    if let Some(id) = first_group(patterns, "exhibit_id_simple", text) {
        let query = StructuredKeyQuery::Exhibit { id, party_attribution: None };
        let candidates = retriever.find_by_structured_key(&query, case_id)?;
        if candidates.len() == 1 {
            return Ok(unique(candidates, 0.95, "rule_exhibit_id"));
        }
    }
    Ok(None)
}

/// Any label: a unique exact match on the mention text.
///
/// # Errors
/// Fatal store errors.
pub fn resolve_generic_rule(
    ctx: &EdContext,
    retriever: &dyn EdRetrieval,
    patterns: &KeyPatterns,
) -> Result<Option<EdResult>> {
    let _ = patterns;
    let candidates =
        retriever.find_by_exact_match(&ctx.mention.name, Some(&ctx.mention.entity_type))?;
    Ok(unique(candidates, 0.9, "rule_generic_exact"))
}
