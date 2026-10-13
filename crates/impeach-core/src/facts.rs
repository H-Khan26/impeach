//! Fact records and structural type normalisation (Python `schema/facts.py`).

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::ids::{ChunkId, DocId, EntityId, FactId};
use crate::vocab::{
    FallbackVocabulary, TypeVocabulary, FALLBACK_DEFAULT_PROPERTY_TYPE, FALLBACK_DEFAULT_VALUE_TYPE,
};
use crate::Metadata;

/// Uppercases `raw`, returning `None` for missing or empty input.
fn upper_non_empty(raw: Option<&str>) -> Option<String> {
    raw.filter(|s| !s.is_empty()).map(str::to_uppercase)
}

/// Kind of value in a fact (`NUMERIC`, `TEMPORAL_POINT`, ..., `TEXT`).
///
/// Built through [`ValueType::from_raw`], which uppercases, validates against a
/// [`TypeVocabulary`] and falls back to the vocabulary default. Deserialisation accepts any
/// string as is (the value was normalised when it was written).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ValueType(String);

impl ValueType {
    /// Normalises a raw value type: uppercase it, keep it if `vocab` accepts it, otherwise use
    /// the vocabulary default. Missing or empty input yields the default.
    #[must_use]
    pub fn from_raw(raw: Option<&str>, vocab: &dyn TypeVocabulary) -> Self {
        match upper_non_empty(raw) {
            Some(upper) if vocab.is_valid_value_type(&upper) => Self(upper),
            _ => Self(vocab.default_value_type().to_owned()),
        }
    }

    /// [`ValueType::from_raw`] against the hardcoded [`FallbackVocabulary`].
    #[must_use]
    pub fn from_raw_fallback(raw: Option<&str>) -> Self {
        Self::from_raw(raw, &FallbackVocabulary)
    }

    /// Whether `raw` (case-insensitive) is a valid value type; missing or empty input is not.
    #[must_use]
    pub fn is_valid(raw: Option<&str>, vocab: &dyn TypeVocabulary) -> bool {
        upper_non_empty(raw).is_some_and(|upper| vocab.is_valid_value_type(&upper))
    }

    /// The type name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Unwraps the type name.
    #[must_use]
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl Default for ValueType {
    fn default() -> Self {
        Self(FALLBACK_DEFAULT_VALUE_TYPE.to_owned())
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for ValueType {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl PartialEq<str> for ValueType {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for ValueType {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

/// Raw property types rewritten before validation (common LLM variants).
const PROPERTY_TYPE_ALIASES: &[(&str, &str)] = &[("EXISTS", "EXISTENCE")];

/// Kind of relationship in a fact (`TEMPORAL`, `QUANTITATIVE`, ..., `MODALITY`).
///
/// Built through [`PropertyType::from_raw`], which uppercases, maps aliases (`EXISTS` ->
/// `EXISTENCE`), validates against a [`TypeVocabulary`] and falls back to the vocabulary
/// default. Deserialisation accepts any string as is.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PropertyType(String);

impl PropertyType {
    /// Normalises a raw property type: uppercase it, apply aliases, keep it if `vocab` accepts
    /// it, otherwise use the vocabulary default. Missing or empty input yields the default.
    #[must_use]
    pub fn from_raw(raw: Option<&str>, vocab: &dyn TypeVocabulary) -> Self {
        let Some(mut upper) = upper_non_empty(raw) else {
            return Self(vocab.default_property_type().to_owned());
        };
        if let Some((_, target)) = PROPERTY_TYPE_ALIASES.iter().find(|(alias, _)| *alias == upper) {
            (*target).clone_into(&mut upper);
        }
        if vocab.is_valid_property_type(&upper) {
            Self(upper)
        } else {
            Self(vocab.default_property_type().to_owned())
        }
    }

    /// [`PropertyType::from_raw`] against the hardcoded [`FallbackVocabulary`].
    #[must_use]
    pub fn from_raw_fallback(raw: Option<&str>) -> Self {
        Self::from_raw(raw, &FallbackVocabulary)
    }

    /// Whether `raw` (case-insensitive) is a valid property type. Aliases are not applied, so
    /// `EXISTS` is not valid even though [`PropertyType::from_raw`] maps it to `EXISTENCE`.
    #[must_use]
    pub fn is_valid(raw: Option<&str>, vocab: &dyn TypeVocabulary) -> bool {
        upper_non_empty(raw).is_some_and(|upper| vocab.is_valid_property_type(&upper))
    }

    /// The type name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Unwraps the type name.
    #[must_use]
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl Default for PropertyType {
    fn default() -> Self {
        Self(FALLBACK_DEFAULT_PROPERTY_TYPE.to_owned())
    }
}

impl fmt::Display for PropertyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for PropertyType {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl PartialEq<str> for PropertyType {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for PropertyType {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

/// Validity interval of a fact, as free-form strings.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalValidity {
    /// Start of validity.
    #[serde(default)]
    pub t_valid_from: Option<String>,
    /// End of validity.
    #[serde(default)]
    pub t_valid_until: Option<String>,
}

/// Time window covered by a denial (e.g. "no accidents in the past 10 years").
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeWindow {
    /// Window start.
    #[serde(default)]
    pub start: Option<String>,
    /// Window end.
    #[serde(default)]
    pub end: Option<String>,
}

/// Subject / relationship / object triple of a fact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssertionTriple {
    /// Subject.
    pub subject: String,
    /// Relationship (predicate).
    pub relationship: String,
    /// Object.
    pub object: String,
}

impl AssertionTriple {
    /// Creates a triple.
    pub fn new(
        subject: impl Into<String>,
        relationship: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self { subject: subject.into(), relationship: relationship.into(), object: object.into() }
    }
}

/// Whether a fact concerns the current incident, a prior one, or a general claim.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalContext {
    /// `current_incident`
    #[default]
    CurrentIncident,
    /// `prior_incident`
    PriorIncident,
    /// `general_claim`
    GeneralClaim,
}

impl TemporalContext {
    /// The Python literal value.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CurrentIncident => "current_incident",
            Self::PriorIncident => "prior_incident",
            Self::GeneralClaim => "general_claim",
        }
    }
}

/// A fact extracted from text by the LLM.
///
/// `value_type` and `property_type` hold the raw strings the extractor produced; use
/// [`ExtractedFact::normalized_value_type`] / [`ExtractedFact::normalized_property_type`] for
/// the validated forms.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractedFact {
    /// Extraction id (e.g. `STATED_FACT_0`).
    pub id: FactId,
    /// Fact text.
    pub text: String,
    /// Subject / relationship / object.
    pub assertion_triple: AssertionTriple,
    /// Extractor certainty.
    pub certainty: f64,
    /// Validity interval.
    pub temporal_validity: TemporalValidity,
    /// Free-form metadata.
    #[serde(default)]
    pub metadata: Metadata,
    /// Raw value type (see [`ValueType`]).
    #[serde(default)]
    pub value_type: Option<String>,
    /// Raw property type (see [`PropertyType`]).
    #[serde(default)]
    pub property_type: Option<String>,
    /// Short snake_case property label used for grouping (e.g. `speed`).
    #[serde(default)]
    pub slot_name: Option<String>,
    /// Exact value text from the document.
    #[serde(default)]
    pub value_raw: Option<String>,
    /// Normalised value set by post-processing.
    #[serde(default)]
    pub value_normalized: Option<String>,
    /// Incident the fact relates to.
    #[serde(default)]
    pub temporal_context: TemporalContext,
    /// Period covered by a denial.
    #[serde(default)]
    pub time_window: Option<TimeWindow>,
    /// Whether the fact is a denial.
    #[serde(default)]
    pub is_denial: bool,
    /// Whether the fact is legal boilerplate.
    #[serde(default)]
    pub is_boilerplate: bool,
    /// Source chunk id.
    #[serde(default)]
    pub source_chunk_id: Option<ChunkId>,
    /// Source document id.
    #[serde(default)]
    pub source_doc_id: Option<DocId>,
    /// Resolved subject entity id.
    #[serde(default)]
    pub subject_entity_id: Option<EntityId>,
    /// Resolved object entity id.
    #[serde(default)]
    pub object_entity_id: Option<EntityId>,
    /// Raw extracted node as returned by the LLM.
    #[serde(default)]
    pub raw_node: Metadata,
}

impl ExtractedFact {
    /// Creates a fact with the Python defaults for every optional field.
    pub fn new(
        id: impl Into<FactId>,
        text: impl Into<String>,
        assertion_triple: AssertionTriple,
        certainty: f64,
        temporal_validity: TemporalValidity,
    ) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            assertion_triple,
            certainty,
            temporal_validity,
            metadata: Metadata::new(),
            value_type: None,
            property_type: None,
            slot_name: None,
            value_raw: None,
            value_normalized: None,
            temporal_context: TemporalContext::default(),
            time_window: None,
            is_denial: false,
            is_boilerplate: false,
            source_chunk_id: None,
            source_doc_id: None,
            subject_entity_id: None,
            object_entity_id: None,
            raw_node: Metadata::new(),
        }
    }

    /// The validated value type (Python `get_value_type`).
    #[must_use]
    pub fn normalized_value_type(&self, vocab: &dyn TypeVocabulary) -> ValueType {
        ValueType::from_raw(self.value_type.as_deref(), vocab)
    }

    /// The validated property type (Python `get_property_type`).
    #[must_use]
    pub fn normalized_property_type(&self, vocab: &dyn TypeVocabulary) -> PropertyType {
        PropertyType::from_raw(self.property_type.as_deref(), vocab)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::{FALLBACK_PROPERTY_TYPES, FALLBACK_VALUE_TYPES};
    use pretty_assertions::assert_eq;
    use serde_json::json;

    /// A vocabulary that differs from the fallback, standing in for a loaded taxonomy.
    struct CustomVocab;

    impl TypeVocabulary for CustomVocab {
        fn is_valid_value_type(&self, value_type: &str) -> bool {
            matches!(value_type, "MONEY" | "TEXT")
        }
        fn is_valid_property_type(&self, property_type: &str) -> bool {
            matches!(property_type, "EXISTENCE" | "OTHER")
        }
        fn default_value_type(&self) -> &str {
            "TEXT"
        }
        fn default_property_type(&self) -> &str {
            "OTHER"
        }
    }

    fn make_fact() -> ExtractedFact {
        ExtractedFact::new(
            "fact_001",
            "Text",
            AssertionTriple::new("Subject", "RELATES_TO", "Object"),
            1.0,
            TemporalValidity::default(),
        )
    }

    #[test]
    fn value_type_normalisation() {
        assert_eq!(ValueType::from_raw_fallback(Some("numeric")), "NUMERIC");
        assert_eq!(ValueType::from_raw_fallback(Some("Temporal_Point")), "TEMPORAL_POINT");
        assert_eq!(ValueType::from_raw_fallback(Some("bogus")), "TEXT");
        assert_eq!(ValueType::from_raw_fallback(Some("")), "TEXT");
        assert_eq!(ValueType::from_raw_fallback(None), "TEXT");
        assert_eq!(ValueType::from_raw_fallback(Some(" numeric")), "TEXT");
        for vt in FALLBACK_VALUE_TYPES {
            assert_eq!(ValueType::from_raw_fallback(Some(&vt.to_lowercase())), *vt);
        }
        assert!(ValueType::is_valid(Some("boolean"), &FallbackVocabulary));
        assert!(!ValueType::is_valid(Some("money"), &FallbackVocabulary));
        assert!(!ValueType::is_valid(None, &FallbackVocabulary));
        assert!(!ValueType::is_valid(Some(""), &FallbackVocabulary));
    }

    #[test]
    fn value_type_uses_supplied_vocabulary() {
        assert_eq!(ValueType::from_raw(Some("money"), &CustomVocab), "MONEY");
        assert_eq!(ValueType::from_raw(Some("numeric"), &CustomVocab), "TEXT");
        assert!(ValueType::is_valid(Some("Money"), &CustomVocab));
    }

    #[test]
    fn property_type_normalisation() {
        assert_eq!(PropertyType::from_raw_fallback(Some("quantitative")), "QUANTITATIVE");
        assert_eq!(PropertyType::from_raw_fallback(Some("bogus")), "ATTRIBUTE");
        assert_eq!(PropertyType::from_raw_fallback(Some("")), "ATTRIBUTE");
        assert_eq!(PropertyType::from_raw_fallback(None), "ATTRIBUTE");
        for pt in FALLBACK_PROPERTY_TYPES {
            assert_eq!(PropertyType::from_raw_fallback(Some(&pt.to_lowercase())), *pt);
        }
        assert!(PropertyType::is_valid(Some("relation"), &FallbackVocabulary));
        assert!(!PropertyType::is_valid(None, &FallbackVocabulary));
    }

    #[test]
    fn property_type_exists_alias() {
        assert_eq!(PropertyType::from_raw_fallback(Some("EXISTS")), "EXISTENCE");
        assert_eq!(PropertyType::from_raw_fallback(Some("exists")), "EXISTENCE");
        assert_eq!(PropertyType::from_raw(Some("Exists"), &CustomVocab), "EXISTENCE");
        assert!(!PropertyType::is_valid(Some("EXISTS"), &FallbackVocabulary));
    }

    #[test]
    fn type_newtypes_serialise_as_strings() {
        let vt = ValueType::from_raw_fallback(Some("location"));
        assert_eq!(serde_json::to_value(&vt).expect("serialises"), json!("LOCATION"));
        let pt: PropertyType = serde_json::from_value(json!("MODALITY")).expect("deserialises");
        assert_eq!(pt.as_str(), "MODALITY");
        assert_eq!(ValueType::default(), "TEXT");
        assert_eq!(PropertyType::default(), "ATTRIBUTE");
        assert_eq!(vt.to_string(), "LOCATION");
    }

    #[test]
    fn fact_defaults_match_python() {
        let value = serde_json::to_value(make_fact()).expect("fact serialises");
        assert_eq!(
            value,
            json!({
                "id": "fact_001",
                "text": "Text",
                "assertion_triple": {
                    "subject": "Subject", "relationship": "RELATES_TO", "object": "Object"
                },
                "certainty": 1.0,
                "temporal_validity": {"t_valid_from": null, "t_valid_until": null},
                "metadata": {},
                "value_type": null,
                "property_type": null,
                "slot_name": null,
                "value_raw": null,
                "value_normalized": null,
                "temporal_context": "current_incident",
                "time_window": null,
                "is_denial": false,
                "is_boilerplate": false,
                "source_chunk_id": null,
                "source_doc_id": null,
                "subject_entity_id": null,
                "object_entity_id": null,
                "raw_node": {}
            })
        );
    }

    #[test]
    fn fact_minimal_json_fills_defaults() {
        let fact: ExtractedFact = serde_json::from_value(json!({
            "id": "fact_001",
            "text": "Text",
            "assertion_triple": {"subject": "Subject", "relationship": "RELATES_TO", "object": "Object"},
            "certainty": 1,
            "temporal_validity": {}
        }))
        .expect("minimal fact deserialises");
        assert_eq!(fact, make_fact());
    }

    #[test]
    fn fact_type_getters() {
        let mut fact = make_fact();
        assert_eq!(fact.normalized_value_type(&FallbackVocabulary), "TEXT");
        assert_eq!(fact.normalized_property_type(&FallbackVocabulary), "ATTRIBUTE");
        fact.value_type = Some("numeric".into());
        fact.property_type = Some("exists".into());
        assert_eq!(fact.normalized_value_type(&FallbackVocabulary), "NUMERIC");
        assert_eq!(fact.normalized_property_type(&FallbackVocabulary), "EXISTENCE");
    }

    #[test]
    fn temporal_context_literals() {
        for (tc, s) in [
            (TemporalContext::CurrentIncident, "current_incident"),
            (TemporalContext::PriorIncident, "prior_incident"),
            (TemporalContext::GeneralClaim, "general_claim"),
        ] {
            assert_eq!(serde_json::to_value(tc).expect("serialises"), json!(s));
            assert_eq!(tc.as_str(), s);
        }
        assert!(serde_json::from_value::<TemporalContext>(json!("future")).is_err());
    }
}
