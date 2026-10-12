//! Name normalisation and loose name matching for intra-document matching and reuse checks
//! (Python `services/ed/name_matching.py`).

use ce_config::mappings::EntityTypes;

/// Lowercases, removes `.,;:!?'"()[]{}` and collapses whitespace.
#[must_use]
pub fn normalize_for_matching(name: &str) -> String {
    let _ = name;
    todo!("worker A")
}

/// Whether two names likely refer to the same entity: equal after
/// [`normalize_for_matching`], or the shorter (at least 4 characters) is a substring of the
/// longer and its words are a subset of the longer's words.
#[must_use]
pub fn names_match(name1: &str, name2: &str) -> bool {
    let _ = (name1, name2);
    todo!("worker A")
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
