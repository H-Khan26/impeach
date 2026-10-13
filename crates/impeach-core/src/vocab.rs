//! Vocabulary contract for fact value and property types.
//!
//! `impeach-config` implements [`TypeVocabulary`] for the fact taxonomy loaded from
//! `fact_categorization.yaml`; `ValueType` / `PropertyType` normalisation in [`crate::facts`]
//! uses it when available and falls back to the hardcoded lists below otherwise.

/// Value types used when no taxonomy is loaded (matches the Python fallback list).
pub const FALLBACK_VALUE_TYPES: &[&str] = &[
    "NUMERIC",
    "TEMPORAL_POINT",
    "TEMPORAL_INTERVAL",
    "LOCATION",
    "BOOLEAN",
    "CATEGORICAL",
    "ENTITY_REF",
    "TEXT",
];

/// Property types used when no taxonomy is loaded (matches the Python fallback list).
pub const FALLBACK_PROPERTY_TYPES: &[&str] =
    &["TEMPORAL", "QUANTITATIVE", "LOCATION", "ATTRIBUTE", "RELATION", "EXISTENCE", "MODALITY"];

/// Default value type when no taxonomy is loaded.
pub const FALLBACK_DEFAULT_VALUE_TYPE: &str = "TEXT";

/// Default property type when no taxonomy is loaded.
pub const FALLBACK_DEFAULT_PROPERTY_TYPE: &str = "ATTRIBUTE";

/// Source of valid value / property types and their defaults.
pub trait TypeVocabulary {
    /// Whether `value_type` (already uppercased) is a valid value type.
    fn is_valid_value_type(&self, value_type: &str) -> bool;
    /// Whether `property_type` (already uppercased) is a valid property type.
    fn is_valid_property_type(&self, property_type: &str) -> bool;
    /// Value type used for missing or unknown input.
    fn default_value_type(&self) -> &str;
    /// Property type used for missing or unknown input.
    fn default_property_type(&self) -> &str;
}

/// The hardcoded fallback vocabulary.
#[derive(Debug, Clone, Copy, Default)]
pub struct FallbackVocabulary;

impl TypeVocabulary for FallbackVocabulary {
    fn is_valid_value_type(&self, value_type: &str) -> bool {
        FALLBACK_VALUE_TYPES.contains(&value_type)
    }

    fn is_valid_property_type(&self, property_type: &str) -> bool {
        FALLBACK_PROPERTY_TYPES.contains(&property_type)
    }

    fn default_value_type(&self) -> &str {
        FALLBACK_DEFAULT_VALUE_TYPE
    }

    fn default_property_type(&self) -> &str {
        FALLBACK_DEFAULT_PROPERTY_TYPE
    }
}
