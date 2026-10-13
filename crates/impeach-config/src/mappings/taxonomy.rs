//! Fact taxonomy (`fact_categorization.yaml`).

use impeach_core::vocab::TypeVocabulary;
use serde::Deserialize;

/// Temporal contexts and the domain-agnostic structural fact types.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Taxonomy {
    /// `current_incident`, `prior_incident`, `general_claim`.
    pub temporal_contexts: Vec<String>,
    /// Valid value types (`NUMERIC`, `TEXT`, ...).
    pub value_types: Vec<String>,
    /// Valid property types (`TEMPORAL`, `ATTRIBUTE`, ...).
    pub property_types: Vec<String>,
    /// Defaults used when classification fails.
    pub defaults: TaxonomyDefaults,
}

/// `defaults:` block of the taxonomy.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TaxonomyDefaults {
    /// Defaults to `current_incident`.
    pub temporal_context: String,
    /// Defaults to `TEXT`.
    pub value_type: String,
    /// Defaults to `ATTRIBUTE`.
    pub property_type: String,
}

impl Default for TaxonomyDefaults {
    fn default() -> Self {
        Self {
            temporal_context: "current_incident".to_owned(),
            value_type: "TEXT".to_owned(),
            property_type: "ATTRIBUTE".to_owned(),
        }
    }
}

impl Taxonomy {
    /// Whether `value_type` is listed (exact match).
    pub fn is_valid_value_type(&self, value_type: &str) -> bool {
        self.value_types.iter().any(|v| v == value_type)
    }

    /// Whether `property_type` is listed (exact match).
    pub fn is_valid_property_type(&self, property_type: &str) -> bool {
        self.property_types.iter().any(|p| p == property_type)
    }

    /// Default value type.
    pub fn default_value_type(&self) -> &str {
        &self.defaults.value_type
    }

    /// Default property type.
    pub fn default_property_type(&self) -> &str {
        &self.defaults.property_type
    }
}

impl TypeVocabulary for Taxonomy {
    fn is_valid_value_type(&self, value_type: &str) -> bool {
        Taxonomy::is_valid_value_type(self, value_type)
    }

    fn is_valid_property_type(&self, property_type: &str) -> bool {
        Taxonomy::is_valid_property_type(self, property_type)
    }

    fn default_value_type(&self) -> &str {
        Taxonomy::default_value_type(self)
    }

    fn default_property_type(&self) -> &str {
        Taxonomy::default_property_type(self)
    }
}
