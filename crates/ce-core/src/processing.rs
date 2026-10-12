//! Pipeline processing records: values the stages hand to each other through the pipeline
//! context and the `run_processing` result.
//!
//! Python origin: `api/processing.py` (`ProcessingResult`), `services/definitions/extractor.py`
//! (`DefinedTerm`), the relation dicts of `stages/relation.py` and the linking statistics of
//! `stages/facts.py`. Field names and order follow the Python records.

use serde::{Deserialize, Serialize};

use crate::contradictions::ContradictionReport;
use crate::ed::EdResult;
use crate::entities::Span;
use crate::facts::ExtractedFact;
use crate::ids::{ChunkId, FactId, MentionId};

/// A role term defined in the document, e.g. `Maria Lopez ("Plaintiff")`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DefinedTerm {
    /// Id of the mention the term refers to.
    pub entity_mention_id: MentionId,
    /// Normalised role term, e.g. `PLAINTIFF`.
    pub defined_term: String,
    /// The term as written in the document.
    pub original_term: String,
    /// Where the definition occurs in the document (character offsets).
    pub span: Span,
    /// Confidence in `[0, 1]`.
    pub confidence: f64,
    /// How the definition was found, e.g. `pattern:parenthetical_quoted`.
    pub extraction_method: String,
}

/// One relation extracted from a chunk (the normalised GLiNER2 output of the relation stage).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractedRelation {
    /// Relation label from the label set, e.g. `employed_by`.
    pub relation: String,
    /// Head entity text.
    pub head: String,
    /// Tail entity text.
    pub tail: String,
    /// Graph id of the source chunk.
    pub chunk_id: ChunkId,
    /// Canonical relationship type (`OTHER` when unmapped).
    pub canonical_relation: String,
    /// Model score, when available.
    #[serde(default)]
    pub score: Option<f64>,
}

/// Entity-linking counts of the fact stage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactLinkingStats {
    /// Facts examined.
    pub total_facts: usize,
    /// Facts whose subject was linked to an entity.
    pub linked_subjects: usize,
    /// Facts whose object was linked to an entity.
    pub linked_objects: usize,
    /// Facts with both ends linked.
    pub linked_both: usize,
    /// Facts with at least one end linked.
    pub linked_either: usize,
}

/// Outputs of one `run_processing` call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessingResult {
    /// Contradiction report, when the contradiction stage ran.
    pub contradiction_report: Option<ContradictionReport>,
    /// Facts that passed every check.
    #[serde(default)]
    pub clean_facts: Vec<ExtractedFact>,
    /// Graph ids of the committed facts.
    #[serde(default)]
    pub committed_fact_ids: Vec<FactId>,
    /// Every extracted fact.
    #[serde(default)]
    pub extracted_facts: Vec<ExtractedFact>,
    /// Entity disambiguation results.
    #[serde(default)]
    pub ed_results: Vec<EdResult>,
    /// SQLite review database path, when one was requested.
    #[serde(default)]
    pub sqlite_db_path: Option<String>,
    /// Unique pairs at or above the contradiction threshold.
    #[serde(default)]
    pub contradiction_count: usize,
    /// Unique pairs between the review and contradiction thresholds.
    #[serde(default)]
    pub needs_review_count: usize,
}

impl ProcessingResult {
    /// Builds the result from the stage outputs; the counts come from the report.
    #[must_use]
    pub fn new(
        contradiction_report: Option<ContradictionReport>,
        clean_facts: Vec<ExtractedFact>,
        committed_fact_ids: Vec<FactId>,
        extracted_facts: Vec<ExtractedFact>,
        ed_results: Vec<EdResult>,
        sqlite_db_path: Option<String>,
    ) -> Self {
        let (contradiction_count, needs_review_count) = contradiction_report
            .as_ref()
            .map_or((0, 0), |report| (report.contradiction_count(), report.needs_review_count()));
        Self {
            contradiction_report,
            clean_facts,
            committed_fact_ids,
            extracted_facts,
            ed_results,
            sqlite_db_path,
            contradiction_count,
            needs_review_count,
        }
    }

    /// The result as a JSON object in Python `to_dict()` field order.
    ///
    /// # Errors
    /// Serialisation failures (not expected for these records).
    pub fn to_json(&self) -> crate::Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    #[test]
    fn empty_result_serialises_in_python_order() {
        let result = ProcessingResult::new(None, vec![], vec![], vec![], vec![], None);
        let value = result.to_json().expect("serialises");
        let keys: Vec<&str> =
            value.as_object().expect("object").keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "contradiction_report",
                "clean_facts",
                "committed_fact_ids",
                "extracted_facts",
                "ed_results",
                "sqlite_db_path",
                "contradiction_count",
                "needs_review_count"
            ]
        );
        assert_eq!(value["contradiction_report"], json!(null));
        assert_eq!(value["contradiction_count"], json!(0));
    }

    #[test]
    fn counts_come_from_the_report() {
        let report: ContradictionReport =
            serde_json::from_value(json!({"doc_id": "d1"})).expect("report");
        let result =
            ProcessingResult::new(Some(report), vec![], vec!["7".into()], vec![], vec![], None);
        assert_eq!((result.contradiction_count, result.needs_review_count), (0, 0));
        assert_eq!(result.to_json().expect("json")["committed_fact_ids"], json!(["7"]));
    }

    #[test]
    fn records_keep_python_field_order() {
        let stats = FactLinkingStats { total_facts: 2, linked_either: 1, ..Default::default() };
        assert_eq!(
            serde_json::to_string(&stats).expect("json"),
            r#"{"total_facts":2,"linked_subjects":0,"linked_objects":0,"linked_both":0,"linked_either":1}"#
        );
        let relation = ExtractedRelation {
            relation: "employed_by".into(),
            head: "Ann".into(),
            tail: "Acme".into(),
            chunk_id: "4".into(),
            canonical_relation: "EMPLOYED_BY".into(),
            score: None,
        };
        assert_eq!(
            serde_json::to_value(&relation).expect("json"),
            json!({"relation": "employed_by", "head": "Ann", "tail": "Acme", "chunk_id": "4",
                   "canonical_relation": "EMPLOYED_BY", "score": null})
        );
        let term = DefinedTerm {
            entity_mention_id: "m1".into(),
            defined_term: "PLAINTIFF".into(),
            original_term: "Plaintiff".into(),
            span: Span::new(12, 21),
            confidence: 0.95,
            extraction_method: "pattern:parenthetical_quoted".into(),
        };
        let back: DefinedTerm =
            serde_json::from_value(serde_json::to_value(&term).expect("json")).expect("parse");
        assert_eq!(back, term);
    }
}
