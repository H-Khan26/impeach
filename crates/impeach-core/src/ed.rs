//! Entity disambiguation records (Python `schema/ed.py`).

use serde::{Deserialize, Serialize};

use crate::entities::Mention;
use crate::ids::{DocId, EntityId, MentionId};
use crate::Metadata;

/// Local text context around a mention.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalContext {
    /// Sentence containing the mention.
    pub sentence_text: String,
    /// Previous sentence.
    #[serde(default)]
    pub prev_sentence_text: Option<String>,
    /// Next sentence.
    #[serde(default)]
    pub next_sentence_text: Option<String>,
    /// Chunk text, used as fallback context.
    pub chunk_text: String,
}

impl LocalContext {
    /// Creates a context without neighbouring sentences.
    pub fn new(sentence_text: impl Into<String>, chunk_text: impl Into<String>) -> Self {
        Self {
            sentence_text: sentence_text.into(),
            prev_sentence_text: None,
            next_sentence_text: None,
            chunk_text: chunk_text.into(),
        }
    }
}

/// Document-level context for disambiguation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocContext {
    /// Document id.
    pub doc_id: DocId,
    /// Document type.
    pub doc_type: String,
    /// Case id.
    #[serde(default)]
    pub case_id: Option<String>,
    /// Jurisdiction.
    #[serde(default)]
    pub jurisdiction: Option<String>,
    /// Party names in the case.
    #[serde(default)]
    pub party_names: Vec<String>,
}

impl DocContext {
    /// Creates a context with the Python defaults for every optional field.
    pub fn new(doc_id: impl Into<DocId>, doc_type: impl Into<String>) -> Self {
        Self {
            doc_id: doc_id.into(),
            doc_type: doc_type.into(),
            case_id: None,
            jurisdiction: None,
            party_names: Vec::new(),
        }
    }
}

fn default_candidate_source() -> String {
    "unknown".to_owned()
}

/// A candidate entity retrieved from the graph (Python `EDCandidate`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdCandidate {
    /// Entity id.
    pub entity_id: EntityId,
    /// Canonical name.
    pub canonical_name: String,
    /// Entity type label (`PERSON`, `ORG`, ...).
    pub label: String,
    /// Known aliases.
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Description.
    #[serde(default)]
    pub description: Option<String>,
    /// Retrieval score.
    #[serde(default)]
    pub score: f64,
    /// Retrieval method (`exact`, `fuzzy`, `embedding`); `unknown` by default.
    #[serde(default = "default_candidate_source")]
    pub source: String,
    /// Free-form metadata.
    #[serde(default)]
    pub metadata: Metadata,
}

impl EdCandidate {
    /// Creates a candidate with the Python defaults for every optional field.
    pub fn new(
        entity_id: impl Into<EntityId>,
        canonical_name: impl Into<String>,
        label: impl Into<String>,
    ) -> Self {
        Self {
            entity_id: entity_id.into(),
            canonical_name: canonical_name.into(),
            label: label.into(),
            aliases: Vec::new(),
            description: None,
            score: 0.0,
            source: default_candidate_source(),
            metadata: Metadata::new(),
        }
    }
}

/// Everything needed to resolve one mention (Python `EDContext`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdContext {
    /// The mention.
    pub mention: Mention,
    /// Local text context.
    pub local_context: LocalContext,
    /// Document context.
    pub doc_context: DocContext,
    /// Span label used for routing.
    pub span_label: String,
    /// Normalised lookup key.
    #[serde(default)]
    pub normalized_key: Option<String>,
}

/// Outcome of disambiguating one mention (Python `EDResult`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdResult {
    /// Id of the mention this result belongs to.
    #[serde(default)]
    pub mention_id: Option<MentionId>,
    /// Chosen entity, if any.
    pub chosen_entity_id: Option<EntityId>,
    /// Candidates considered.
    pub candidates: Vec<EdCandidate>,
    /// Whether the LLM tie-break decided.
    pub resolved_by_llm: bool,
    /// Confidence.
    #[serde(default)]
    pub confidence: f64,
    /// Strategy used (`rule`, `hybrid`, `llm`).
    pub strategy_used: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::Span;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    #[test]
    fn ed_context_from_python_shape() {
        let name = "Maria Lopez";
        let ctx: EdContext = serde_json::from_value(json!({
            "mention": {"id": "m1", "type": "PERSON", "name": name, "span": {"start": 0, "end": 11}},
            "local_context": {
                "sentence_text": "Maria Lopez did something.",
                "chunk_text": "Maria Lopez did something."
            },
            "doc_context": {"doc_id": "doc-1", "doc_type": "complaint", "case_id": "case-1"},
            "span_label": "PERSON"
        }))
        .expect("context deserialises");
        let mut doc_context = DocContext::new("doc-1", "complaint");
        doc_context.case_id = Some("case-1".into());
        let expected = EdContext {
            mention: Mention::new("m1", "PERSON", name, Span::new(0, 11)),
            local_context: LocalContext::new(
                "Maria Lopez did something.",
                "Maria Lopez did something.",
            ),
            doc_context,
            span_label: "PERSON".into(),
            normalized_key: None,
        };
        assert_eq!(ctx, expected);
    }

    #[test]
    fn candidate_defaults() {
        let c: EdCandidate = serde_json::from_value(json!({
            "entity_id": "e1", "canonical_name": "Maria Lopez", "label": "PERSON"
        }))
        .expect("candidate deserialises");
        assert_eq!(c, EdCandidate::new("e1", "Maria Lopez", "PERSON"));
        assert_eq!(c.source, "unknown");
        assert!(c.score.abs() < f64::EPSILON);
    }

    #[test]
    fn result_serialises_all_fields() {
        let result = EdResult {
            mention_id: None,
            chosen_entity_id: Some("entity-1".into()),
            candidates: vec![],
            resolved_by_llm: false,
            confidence: 0.9,
            strategy_used: "rule".into(),
        };
        assert_eq!(
            serde_json::to_value(&result).expect("serialises"),
            json!({
                "mention_id": null,
                "chosen_entity_id": "entity-1",
                "candidates": [],
                "resolved_by_llm": false,
                "confidence": 0.9,
                "strategy_used": "rule"
            })
        );
    }
}
