//! Mention, relationship and coreference records (Python `schema/entities.py`).

use serde::{Deserialize, Deserializer, Serialize};

use crate::error::{CoreError, Result};
use crate::ids::{ChunkId, DocId, MentionId};
use crate::Metadata;

/// Character span with start and end offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Span {
    /// Start offset (inclusive).
    pub start: usize,
    /// End offset (exclusive).
    pub end: usize,
}

impl Span {
    /// Creates a span.
    #[must_use]
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

/// One detector that produced a mention.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectionSource {
    /// Detector name, e.g. `gliner2`, `regex_detector`, `eyecite`.
    pub name: String,
    /// Label the detector assigned.
    pub label: String,
    /// Detector score, if any.
    #[serde(default)]
    pub score: Option<f64>,
}

/// A detected entity mention; the Python `ExtractedEntity` record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mention {
    /// Mention id.
    pub id: MentionId,
    /// Entity type label, e.g. `PERSON`, `ORG`.
    #[serde(rename = "type")]
    pub entity_type: String,
    /// Surface text.
    pub name: String,
    /// Character span in the document.
    pub span: Span,
    /// Detection score.
    #[serde(default)]
    pub score: Option<f64>,
    /// Primary detector name, e.g. `gliner2`, `regex_detector`, `eyecite`.
    #[serde(default)]
    pub source: Option<String>,
    /// All detectors that produced this mention.
    #[serde(default)]
    pub sources: Vec<DetectionSource>,
    /// Free-form metadata.
    #[serde(default)]
    pub metadata: Metadata,
    /// Source document id.
    #[serde(default)]
    pub doc_id: Option<DocId>,
    /// Source chunk id.
    #[serde(default)]
    pub chunk_id: Option<ChunkId>,
    /// Index of the sentence within the chunk or document.
    #[serde(default)]
    pub sentence_index: Option<usize>,
}

impl Mention {
    /// Creates a mention with the Python defaults for every optional field.
    pub fn new(
        id: impl Into<MentionId>,
        entity_type: impl Into<String>,
        name: impl Into<String>,
        span: Span,
    ) -> Self {
        Self {
            id: id.into(),
            entity_type: entity_type.into(),
            name: name.into(),
            span,
            score: None,
            source: None,
            sources: Vec::new(),
            metadata: Metadata::new(),
            doc_id: None,
            chunk_id: None,
            sentence_index: None,
        }
    }
}

/// Evidence text for a relationship.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    /// Evidence text.
    pub text: String,
    /// Span of the evidence.
    pub span: Span,
}

/// Relationship type literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RelationshipType {
    /// `FOUNDED_BY`
    FoundedBy,
    /// `EMPLOYED_BY`
    EmployedBy,
    /// `LOCATED_IN`
    LocatedIn,
    /// `PART_OF`
    PartOf,
    /// `MENTIONS`
    Mentions,
    /// `OTHER`
    Other,
}

impl RelationshipType {
    /// The Python literal value.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FoundedBy => "FOUNDED_BY",
            Self::EmployedBy => "EMPLOYED_BY",
            Self::LocatedIn => "LOCATED_IN",
            Self::PartOf => "PART_OF",
            Self::Mentions => "MENTIONS",
            Self::Other => "OTHER",
        }
    }
}

const fn default_relationship_confidence() -> f64 {
    0.5
}

fn check_unit_interval(field: &'static str, value: f64) -> Result<f64> {
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(CoreError::OutOfRange { field, value, min: 0.0, max: 1.0 })
    }
}

fn deserialize_confidence<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    let value = f64::deserialize(deserializer)?;
    check_unit_interval("confidence", value).map_err(serde::de::Error::custom)
}

/// Extracted relationship between two mentions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Relationship {
    /// Relationship type.
    #[serde(rename = "type")]
    pub relationship_type: RelationshipType,
    /// Source mention id.
    pub source_id: String,
    /// Target mention id.
    pub target_id: String,
    /// Supporting evidence.
    pub evidence: Evidence,
    /// Confidence in `[0, 1]`; deserialisation rejects values outside that range.
    #[serde(
        default = "default_relationship_confidence",
        deserialize_with = "deserialize_confidence"
    )]
    pub confidence: f64,
}

impl Relationship {
    /// Creates a relationship with the default confidence of 0.5.
    pub fn new(
        relationship_type: RelationshipType,
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        evidence: Evidence,
    ) -> Self {
        Self {
            relationship_type,
            source_id: source_id.into(),
            target_id: target_id.into(),
            evidence,
            confidence: default_relationship_confidence(),
        }
    }

    /// Sets the confidence.
    ///
    /// # Errors
    /// [`CoreError::OutOfRange`] if `confidence` is not in `[0, 1]`.
    pub fn with_confidence(mut self, confidence: f64) -> Result<Self> {
        self.confidence = check_unit_interval("confidence", confidence)?;
        Ok(self)
    }

    /// Checks the field constraints the Python model enforces.
    ///
    /// # Errors
    /// [`CoreError::OutOfRange`] if `confidence` is not in `[0, 1]`.
    pub fn validate(&self) -> Result<()> {
        check_unit_interval("confidence", self.confidence).map(|_| ())
    }
}

/// Entity and relationship extraction results for one segment (Python `SegmentER`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentEr {
    /// Document id.
    pub doc_id: DocId,
    /// Segment id.
    pub segment_id: String,
    /// Mentions found in the segment.
    #[serde(default)]
    pub entities: Vec<Mention>,
    /// Relationships found in the segment.
    #[serde(default)]
    pub relationships: Vec<Relationship>,
}

/// Relationship list returned by an LLM.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RelationshipResponse {
    /// Relationships.
    #[serde(default)]
    pub relationships: Vec<Relationship>,
}

/// Coreference cluster keyed by mention ids; pronoun spans without a mention are kept apart.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorefClusterContext {
    /// Cluster id.
    pub id: String,
    /// Ids of the mentions in the cluster.
    pub member_mention_ids: Vec<MentionId>,
    /// Pronoun spans that do not map to a mention.
    #[serde(default)]
    pub pronoun_spans: Vec<Span>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    #[test]
    fn mention_defaults_match_python() {
        let m = Mention::new("m1", "PERSON", "Maria Lopez", Span::new(0, 11));
        let value = serde_json::to_value(&m).expect("mention serialises");
        assert_eq!(
            value,
            json!({
                "id": "m1",
                "type": "PERSON",
                "name": "Maria Lopez",
                "span": {"start": 0, "end": 11},
                "score": null,
                "source": null,
                "sources": [],
                "metadata": {},
                "doc_id": null,
                "chunk_id": null,
                "sentence_index": null
            })
        );
    }

    #[test]
    fn mention_minimal_json_fills_defaults() {
        let m: Mention = serde_json::from_value(json!({
            "id": "m2",
            "type": "PERSON",
            "name": "Ms. Lopez",
            "span": {"start": 50, "end": 59}
        }))
        .expect("minimal mention deserialises");
        assert_eq!(m, Mention::new("m2", "PERSON", "Ms. Lopez", Span::new(50, 59)));
    }

    #[test]
    fn relationship_type_literals() {
        for (ty, s) in [
            (RelationshipType::FoundedBy, "FOUNDED_BY"),
            (RelationshipType::EmployedBy, "EMPLOYED_BY"),
            (RelationshipType::LocatedIn, "LOCATED_IN"),
            (RelationshipType::PartOf, "PART_OF"),
            (RelationshipType::Mentions, "MENTIONS"),
            (RelationshipType::Other, "OTHER"),
        ] {
            assert_eq!(serde_json::to_value(ty).expect("serialises"), json!(s));
            assert_eq!(ty.as_str(), s);
        }
        assert!(serde_json::from_value::<RelationshipType>(json!("founded_by")).is_err());
    }

    #[test]
    fn relationship_confidence_default_and_range() {
        let evidence = Evidence { text: "Acme was founded by Ann".into(), span: Span::new(0, 23) };
        let rel: Relationship = serde_json::from_value(json!({
            "type": "FOUNDED_BY",
            "source_id": "m1",
            "target_id": "m2",
            "evidence": {"text": "Acme was founded by Ann", "span": {"start": 0, "end": 23}}
        }))
        .expect("relationship deserialises");
        assert!((rel.confidence - 0.5).abs() < f64::EPSILON);
        assert_eq!(rel, Relationship::new(RelationshipType::FoundedBy, "m1", "m2", evidence));
        assert!(rel.validate().is_ok());

        let bad = serde_json::from_value::<Relationship>(json!({
            "type": "OTHER",
            "source_id": "a",
            "target_id": "b",
            "evidence": {"text": "", "span": {"start": 0, "end": 0}},
            "confidence": 1.5
        }));
        assert!(bad.is_err());
        assert!(rel.clone().with_confidence(-0.1).is_err());
        assert!(rel.with_confidence(1.0).is_ok());
    }

    #[test]
    fn coref_cluster_defaults() {
        let c: CorefClusterContext =
            serde_json::from_value(json!({"id": "c0", "member_mention_ids": ["m1", "m2"]}))
                .expect("cluster deserialises");
        assert!(c.pronoun_spans.is_empty());
        assert_eq!(c.member_mention_ids, vec![MentionId::new("m1"), MentionId::new("m2")]);
    }
}
