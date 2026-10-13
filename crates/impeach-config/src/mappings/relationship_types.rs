//! Relationship types and classifications (`relationship_types.yaml`).

use indexmap::IndexMap;
use serde::Deserialize;

/// Relationship vocabulary and keyword classifications.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RelationshipTypes {
    /// Valid relationship types.
    pub relationship_types: Vec<String>,
    /// Classification -> keywords, in file order (the first matching classification wins).
    pub relationship_classifications: IndexMap<String, Vec<String>>,
    /// Fallback relationship type.
    pub defaults: RelationshipDefaults,
}

/// `defaults:` block of `relationship_types.yaml`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RelationshipDefaults {
    /// Defaults to `OTHER`.
    pub relationship_type: String,
}

impl Default for RelationshipDefaults {
    fn default() -> Self {
        Self { relationship_type: "OTHER".to_owned() }
    }
}

impl RelationshipTypes {
    /// Classification whose keyword occurs in the lowercased relationship, if any.
    pub fn get_classification(&self, relationship: &str) -> Option<&str> {
        let lower = relationship.to_lowercase();
        self.relationship_classifications
            .iter()
            .find(|(_, keywords)| keywords.iter().any(|k| lower.contains(k.as_str())))
            .map(|(classification, _)| classification.as_str())
    }

    /// Whether two relationships share a classification; unclassified ones are compatible.
    pub fn are_compatible(&self, rel_a: &str, rel_b: &str) -> bool {
        match (self.get_classification(rel_a), self.get_classification(rel_b)) {
            (Some(a), Some(b)) => a == b,
            _ => true,
        }
    }
}
