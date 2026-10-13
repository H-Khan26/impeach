//! GLiNER2 label sets (`gliner2.yaml`).

use indexmap::IndexMap;
use serde::Deserialize;

use crate::error::{ConfigError, Result};

/// Named GLiNER2 entity and relation label sets and their canonical mappings.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Gliner2Labels {
    /// Entity label sets (`legal_entities`, `minimal`).
    pub entity_label_sets: IndexMap<String, Vec<String>>,
    /// Relation label sets (`legal_relations`, `minimal`).
    pub relation_label_sets: IndexMap<String, Vec<String>>,
    /// GLiNER2 relation label -> canonical relationship type.
    pub relation_to_relationship_type: IndexMap<String, String>,
    /// GLiNER2 entity label -> canonical entity type.
    pub gliner2_to_entity_type: IndexMap<String, String>,
    /// Default label set names.
    pub defaults: Gliner2Defaults,
}

/// `defaults:` block of `gliner2.yaml`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Gliner2Defaults {
    /// Defaults to `legal_entities`.
    pub entity_label_set: String,
    /// Defaults to `legal_relations`.
    pub relation_label_set: String,
}

impl Default for Gliner2Defaults {
    fn default() -> Self {
        Self {
            entity_label_set: "legal_entities".to_owned(),
            relation_label_set: "legal_relations".to_owned(),
        }
    }
}

impl Gliner2Labels {
    /// Entity labels of a named set.
    pub fn get_entity_label_set(&self, name: &str) -> Result<&[String]> {
        lookup(&self.entity_label_sets, "Entity", name)
    }

    /// Relation labels of a named set.
    pub fn get_relation_label_set(&self, name: &str) -> Result<&[String]> {
        lookup(&self.relation_label_sets, "Relation", name)
    }
}

fn lookup<'a>(
    sets: &'a IndexMap<String, Vec<String>>,
    kind: &'static str,
    name: &str,
) -> Result<&'a [String]> {
    sets.get(name).map(Vec::as_slice).ok_or_else(|| ConfigError::UnknownLabelSet {
        kind,
        name: name.to_owned(),
        available: sets.keys().cloned().collect(),
    })
}
