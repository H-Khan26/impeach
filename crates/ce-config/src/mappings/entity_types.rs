//! Entity types, model label mappings, domains and ED routing (`entity_types.yaml`).

use std::collections::HashSet;

use indexmap::{IndexMap, IndexSet};
use serde::Deserialize;

/// Canonical entity types and how model labels map onto them.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EntityTypes {
    /// Canonical entity types (`PER`, `ORG`, ...).
    pub entity_types: Vec<String>,
    /// `{model}_to_entity_type` -> label -> canonical type, in file order.
    pub model_mappings: IndexMap<String, IndexMap<String, String>>,
    /// Canonical type -> Helix domain.
    pub entity_type_to_domain: IndexMap<String, String>,
    /// Fallback type and domain.
    pub defaults: EntityTypeDefaults,
    /// ED routing label sets.
    pub routing: EntityRouting,
    /// Types treated as values and skipped by ED (`DATE`, `TIME`, `MONEY`).
    pub skip_ed_types: IndexSet<String>,
    /// Groups of mutually compatible types; types in different groups never merge.
    pub incompatible_type_groups: Vec<IndexSet<String>>,
}

/// `defaults:` block of `entity_types.yaml`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EntityTypeDefaults {
    /// Defaults to `OTHER`.
    pub entity_type: String,
    /// Defaults to `general`.
    pub domain: String,
}

impl Default for EntityTypeDefaults {
    fn default() -> Self {
        Self { entity_type: "OTHER".to_owned(), domain: "general".to_owned() }
    }
}

/// `routing:` block of `entity_types.yaml`.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EntityRouting {
    /// Labels resolved rule-first.
    pub rule_first_labels: Vec<String>,
    /// Labels resolved by hybrid retrieval.
    pub hybrid_labels: Vec<String>,
    /// Label -> rule resolver route (`case`, `statute`, `exhibit`).
    pub rule_resolvers: IndexMap<String, String>,
}

impl EntityTypes {
    /// Maps a model-specific label via `{model}_to_entity_type`; unknown labels pass through.
    pub fn map_model_label<'a>(&'a self, model: &str, label: &'a str) -> &'a str {
        self.model_mappings
            .get(&format!("{model}_to_entity_type"))
            .and_then(|map| map.get(label))
            .map_or(label, String::as_str)
    }

    /// Helix domain for an entity type, or the default domain.
    pub fn get_domain(&self, entity_type: &str) -> &str {
        self.entity_type_to_domain.get(entity_type).map_or(self.default_domain(), String::as_str)
    }

    /// Normalises an arbitrary label to a canonical entity type.
    ///
    /// Empty labels become the default type; canonical types and routing-only labels are
    /// kept; otherwise the first model mapping (in file order) with a non-empty target
    /// wins; unknown labels pass through unchanged.
    pub fn normalize_label<'a>(&'a self, label: &'a str) -> &'a str {
        if label.is_empty() {
            return self.default_entity_type();
        }
        if self.entity_types.iter().any(|t| t == label)
            || self.routing.rule_first_labels.iter().any(|l| l == label)
            || self.routing.hybrid_labels.iter().any(|l| l == label)
        {
            return label;
        }
        self.model_mappings
            .values()
            .find_map(|map| map.get(label).filter(|mapped| !mapped.is_empty()))
            .map_or(label, String::as_str)
    }

    /// Labels routed through rule-first ED.
    pub fn rule_first_label_set(&self) -> HashSet<&str> {
        self.routing.rule_first_labels.iter().map(String::as_str).collect()
    }

    /// Labels routed through hybrid ED.
    pub fn hybrid_label_set(&self) -> HashSet<&str> {
        self.routing.hybrid_labels.iter().map(String::as_str).collect()
    }

    /// Default entity type.
    pub fn default_entity_type(&self) -> &str {
        &self.defaults.entity_type
    }

    /// Default Helix domain.
    pub fn default_domain(&self) -> &str {
        &self.defaults.domain
    }

    /// Rule resolver route for a label, if configured.
    pub fn get_rule_route(&self, label: &str) -> Option<&str> {
        self.routing.rule_resolvers.get(label).map(String::as_str)
    }

    /// Whether two entity types may refer to the same entity.
    ///
    /// Labels are normalised first. Missing, equal or default types are compatible; types in
    /// different `incompatible_type_groups` are not; a type in no group is compatible with all.
    pub fn are_types_compatible(&self, type_a: Option<&str>, type_b: Option<&str>) -> bool {
        let (Some(type_a), Some(type_b)) = (type_a, type_b) else {
            return true;
        };
        if type_a.is_empty() || type_b.is_empty() {
            return true;
        }
        let norm_a = self.normalize_label(type_a);
        let norm_b = self.normalize_label(type_b);
        if norm_a == norm_b {
            return true;
        }
        let default = self.default_entity_type();
        if norm_a == default || norm_b == default {
            return true;
        }
        let group_of = |t: &str| self.incompatible_type_groups.iter().position(|g| g.contains(t));
        match (group_of(norm_a), group_of(norm_b)) {
            (Some(a), Some(b)) => a == b,
            _ => true,
        }
    }
}
