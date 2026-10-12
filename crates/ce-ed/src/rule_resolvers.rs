//! Deterministic resolvers for rule-first entity types (Python
//! `services/ed/rule_resolvers.py`). Each returns a result only when exactly one candidate
//! matches; `None` lets the ED service fall through to its next route.

use ce_core::{EdContext, EdResult};

use crate::error::Result;
use crate::retrieval::EdRetrieval;
use crate::structured_keys::KeyPatterns;

/// CASE / MATTER mentions: docket number, then citation or caption, then exact case name.
///
/// # Errors
/// Fatal store errors.
pub fn resolve_case_rule(
    ctx: &EdContext,
    retriever: &dyn EdRetrieval,
    patterns: &KeyPatterns,
) -> Result<Option<EdResult>> {
    let _ = (ctx, retriever, patterns);
    todo!("worker B")
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
    let _ = (ctx, retriever, patterns);
    todo!("worker B")
}

/// EXHIBIT mentions via exhibit keys scoped to the document's case.
///
/// # Errors
/// Fatal store errors.
pub fn resolve_exhibit_rule(
    ctx: &EdContext,
    retriever: &dyn EdRetrieval,
    patterns: &KeyPatterns,
) -> Result<Option<EdResult>> {
    let _ = (ctx, retriever, patterns);
    todo!("worker B")
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
    let _ = (ctx, retriever, patterns);
    todo!("worker B")
}
