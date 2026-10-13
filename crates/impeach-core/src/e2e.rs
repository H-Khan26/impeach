//! End-to-end output records (the per-document JSON written by the e2e runner and read by the
//! evaluator).
//!
//! Python origin: `tests/e2e/run_e2e.py` (`run`, `process_one`, `summarize_ed`,
//! `summarize_facts`, `summarize_contradictions`, `build_aggregates`). Field names and order
//! match the Python JSON. Reading is lenient like the Python evaluator, which only uses
//! `.get(...)` lookups: every field may be missing or `null` (falling back to its default),
//! unknown fields are ignored, and the identifying header fields accept any JSON scalar.
//! A failed document carries only the header, `doc_path` and `error`; the stage blocks and the
//! processing time are omitted from the output when absent.

use indexmap::IndexMap;
use serde::{Deserialize, Deserializer, Serialize};

/// `null` or a missing value reads as the type's default (Python `x.get(k) or default`).
fn null_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

/// A string, or any other JSON scalar rendered as JSON text; `null` reads as `None`.
fn lenient_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(match serde_json::Value::deserialize(deserializer)? {
        serde_json::Value::Null => None,
        serde_json::Value::String(s) => Some(s),
        other => Some(other.to_string()),
    })
}

/// Counter-style breakdown (`dict(Counter(...))`), insertion ordered.
pub type CountMap = IndexMap<String, u64>;

/// The whole e2e run output.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct E2eOutput {
    /// `e2e_<scenario|custom>_<YYYYmmdd_HHMMSS>`.
    #[serde(deserialize_with = "null_default")]
    pub run_id: String,
    /// Local ISO timestamp of the run.
    #[serde(deserialize_with = "null_default")]
    pub timestamp: String,
    /// Preset scenario name, if one was used.
    pub scenario: Option<String>,
    /// `pipeline.name` of the config.
    #[serde(deserialize_with = "null_default")]
    pub pipeline_name: String,
    /// Helix namespace of the config.
    #[serde(deserialize_with = "null_default")]
    pub helix_namespace: String,
    /// Helix server URL (server mode only).
    pub helix_url: Option<String>,
    /// Helix mode (`embedded`, `memory`, `server`).
    #[serde(deserialize_with = "null_default")]
    pub helix_mode: String,
    /// Helix data directory.
    pub helix_data_dir: Option<String>,
    /// Pipeline config path.
    #[serde(deserialize_with = "null_default")]
    pub config_path: String,
    /// SQLite review store path.
    #[serde(deserialize_with = "null_default")]
    pub sqlite_db_path: String,
    /// Input JSONL paths in processing order.
    #[serde(deserialize_with = "null_default")]
    pub inputs: Vec<String>,
    /// `ok`, or `failed` when a document raised.
    #[serde(deserialize_with = "null_default")]
    pub status: String,
    /// Run timing.
    #[serde(deserialize_with = "null_default")]
    pub pipeline_summary: PipelineSummary,
    /// Totals over the successfully processed documents.
    #[serde(deserialize_with = "null_default")]
    pub aggregates: E2eAggregates,
    /// One entry per processed document, in processing order.
    #[serde(deserialize_with = "null_default")]
    pub documents: Vec<E2eDocument>,
}

/// Run timing summary.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PipelineSummary {
    /// Number of document entries (including a failed one).
    #[serde(deserialize_with = "null_default")]
    pub total_documents: u64,
    /// Sum of per-document processing times.
    #[serde(deserialize_with = "null_default")]
    pub total_processing_time_seconds: f64,
    /// Mean processing time per document (0.0 without documents).
    #[serde(deserialize_with = "null_default")]
    pub avg_processing_time_per_doc: f64,
}

/// Totals over the successfully processed documents (`build_aggregates`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct E2eAggregates {
    /// Mentions resolved to an entity.
    #[serde(deserialize_with = "null_default")]
    pub total_entities_resolved: u64,
    /// Entities created by disambiguation.
    #[serde(deserialize_with = "null_default")]
    pub total_entities_created: u64,
    /// Reused / resolved * 100 (0.0 when nothing resolved).
    #[serde(deserialize_with = "null_default")]
    pub entity_reuse_rate_percent: f64,
    /// Extracted facts.
    #[serde(deserialize_with = "null_default")]
    pub total_facts_extracted: u64,
    /// Facts with a linked subject or object * 100 / facts.
    #[serde(deserialize_with = "null_default")]
    pub fact_linking_rate_percent: f64,
    /// Facts committed to the store.
    #[serde(deserialize_with = "null_default")]
    pub total_facts_committed: u64,
    /// Contradiction-labelled facts.
    #[serde(deserialize_with = "null_default")]
    pub total_contradictions: u64,
    /// Facts queued for review.
    #[serde(deserialize_with = "null_default")]
    pub total_needs_review: u64,
    /// Intra-document pairs.
    #[serde(deserialize_with = "null_default")]
    pub total_intra_doc_pairs: u64,
    /// Cross-document pairs.
    #[serde(deserialize_with = "null_default")]
    pub total_cross_doc_pairs: u64,
}

/// One processed (or failed) document.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct E2eDocument {
    /// `doc_id` of the first JSONL record (or its metadata).
    #[serde(deserialize_with = "lenient_string")]
    pub doc_id: Option<String>,
    /// `case_id` of the first JSONL record (or its metadata).
    #[serde(deserialize_with = "lenient_string")]
    pub case_id: Option<String>,
    /// `db_id` of the first JSONL record (or its metadata).
    #[serde(deserialize_with = "lenient_string")]
    pub db_id: Option<String>,
    /// `title` of the first JSONL record (or its metadata).
    #[serde(deserialize_with = "lenient_string")]
    pub title: Option<String>,
    /// Input JSONL path.
    #[serde(deserialize_with = "null_default")]
    pub doc_path: String,
    /// Wall time of `run_processing` for this document.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processing_time_seconds: Option<f64>,
    /// Entity disambiguation summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_4_entity_disambiguation: Option<EdSummary>,
    /// Fact extraction summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_5_fact_extraction: Option<FactExtractionSummary>,
    /// Contradiction check summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_6_contradiction_check: Option<ContradictionCheckSummary>,
    /// `repr` of the exception for a failed document.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Entity disambiguation summary (`summarize_ed`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EdSummary {
    /// Results with a chosen entity.
    #[serde(deserialize_with = "null_default")]
    pub total_resolved: u64,
    /// Results without a chosen entity.
    #[serde(deserialize_with = "null_default")]
    pub total_unresolved: u64,
    /// Resolved count per strategy.
    #[serde(deserialize_with = "null_default")]
    pub resolution_by_strategy: CountMap,
    /// Resolutions whose strategy contains `_create`.
    #[serde(deserialize_with = "null_default")]
    pub new_entities_created_count: u64,
    /// Other resolutions.
    #[serde(deserialize_with = "null_default")]
    pub entities_reused_count: u64,
}

/// Fact extraction summary (`summarize_facts`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FactExtractionSummary {
    /// Extracted facts.
    #[serde(deserialize_with = "null_default")]
    pub total_facts: u64,
    /// Facts flagged as boilerplate.
    #[serde(deserialize_with = "null_default")]
    pub boilerplate_count: u64,
    /// Count per value type (`UNKNOWN` for none).
    #[serde(deserialize_with = "null_default")]
    pub facts_by_value_type: CountMap,
    /// Count per property type (`UNKNOWN` for none).
    #[serde(deserialize_with = "null_default")]
    pub facts_by_property_type: CountMap,
    /// Entity-linking counts.
    #[serde(deserialize_with = "null_default")]
    pub entity_linking: EntityLinkingSummary,
    /// Every extracted fact; absent in older outputs, which the evaluator distinguishes from an
    /// empty list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub all_facts: Option<Vec<FactRecord>>,
}

/// Entity-linking counts of the extracted facts.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EntityLinkingSummary {
    /// Facts with a subject entity.
    #[serde(deserialize_with = "null_default")]
    pub linked_subjects: u64,
    /// Facts with an object entity.
    #[serde(deserialize_with = "null_default")]
    pub linked_objects: u64,
    /// Facts with both.
    #[serde(deserialize_with = "null_default")]
    pub linked_both: u64,
    /// Facts with either.
    #[serde(deserialize_with = "null_default")]
    pub linked_either: u64,
    /// `linked_either / total * 100` (0.0 without facts).
    #[serde(deserialize_with = "null_default")]
    pub linking_rate_percent: f64,
}

/// One extracted fact (`fact_record`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FactRecord {
    /// Fact id.
    #[serde(deserialize_with = "null_default")]
    pub fact_id: String,
    /// Fact text.
    #[serde(deserialize_with = "null_default")]
    pub text: String,
    /// Value type name.
    pub value_type: Option<String>,
    /// Property type name.
    pub property_type: Option<String>,
    /// Linked subject entity id.
    pub subject_entity_id: Option<String>,
    /// Linked object entity id.
    pub object_entity_id: Option<String>,
    /// Boilerplate flag.
    #[serde(deserialize_with = "null_default")]
    pub is_boilerplate: bool,
}

/// One clean (non-contradicted) fact: a [`FactRecord`] without `is_boilerplate`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CleanFactRecord {
    /// Fact id.
    #[serde(deserialize_with = "null_default")]
    pub fact_id: String,
    /// Fact text.
    #[serde(deserialize_with = "null_default")]
    pub text: String,
    /// Value type name.
    pub value_type: Option<String>,
    /// Property type name.
    pub property_type: Option<String>,
    /// Linked subject entity id.
    pub subject_entity_id: Option<String>,
    /// Linked object entity id.
    pub object_entity_id: Option<String>,
}

/// One detected contradiction pair (`pair_record`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PairRecord {
    /// Text of the fact being checked.
    #[serde(deserialize_with = "null_default")]
    pub new_fact_text: String,
    /// Id of the fact being checked.
    #[serde(deserialize_with = "null_default")]
    pub new_fact_id: String,
    /// Text of the fact it contradicts.
    #[serde(deserialize_with = "null_default")]
    pub existing_fact_text: String,
    /// Id of the fact it contradicts.
    #[serde(deserialize_with = "null_default")]
    pub existing_fact_id: String,
    /// NLI / judge label.
    #[serde(deserialize_with = "null_default")]
    pub nli_label: String,
    /// NLI / judge confidence.
    #[serde(deserialize_with = "null_default")]
    pub nli_confidence: f64,
    /// Retrieval paths that surfaced the existing fact.
    #[serde(deserialize_with = "null_default")]
    pub retrieval_sources: Vec<String>,
}

/// Contradiction check summary (`summarize_contradictions`), the block the evaluator reads.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContradictionCheckSummary {
    /// Extracted facts checked.
    #[serde(deserialize_with = "null_default")]
    pub total_checked: u64,
    /// Contradiction-labelled facts.
    #[serde(deserialize_with = "null_default")]
    pub contradiction_count: u64,
    /// Facts queued for review.
    #[serde(deserialize_with = "null_default")]
    pub needs_review_count: u64,
    /// Clean facts.
    #[serde(deserialize_with = "null_default")]
    pub clean_count: u64,
    /// Facts committed to the store.
    #[serde(deserialize_with = "null_default")]
    pub committed_fact_count: u64,
    /// Every clean fact.
    #[serde(deserialize_with = "null_default")]
    pub all_clean_facts: Vec<CleanFactRecord>,
    /// Pairs whose source is `intra_document`.
    #[serde(deserialize_with = "null_default")]
    pub intra_doc_contradictions: Vec<PairRecord>,
    /// Every other pair.
    #[serde(deserialize_with = "null_default")]
    pub cross_doc_contradictions: Vec<PairRecord>,
    /// Retrieval statistics.
    #[serde(deserialize_with = "null_default")]
    pub retrieval_stats: RetrievalStats,
    /// Pre-filter statistics.
    #[serde(deserialize_with = "null_default")]
    pub filtering_stats: FilteringStats,
    /// NLI statistics.
    #[serde(deserialize_with = "null_default")]
    pub nli_stats: NliStats,
}

/// Retrieval statistics of the contradiction check.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RetrievalStats {
    /// Candidates retrieved over all facts.
    #[serde(deserialize_with = "null_default")]
    pub total_candidates_retrieved: u64,
    /// Count of each retrieval source over the detected pairs.
    #[serde(deserialize_with = "null_default")]
    pub retrieval_sources_breakdown: CountMap,
}

/// Pre-filter statistics of the contradiction check.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FilteringStats {
    /// Pairs left after the pre-filter.
    #[serde(deserialize_with = "null_default")]
    pub pairs_after_prefilter: u64,
    /// NLI / judge calls made.
    #[serde(deserialize_with = "null_default")]
    pub total_nli_calls: u64,
}

/// NLI statistics of the contradiction check.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NliStats {
    /// Count per label over the detected pairs.
    #[serde(deserialize_with = "null_default")]
    pub nli_label_breakdown: CountMap,
    /// Contradiction threshold of the report.
    pub contradiction_threshold: Option<f64>,
    /// Review threshold of the report.
    pub review_threshold: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyjson::to_python_json;

    #[test]
    fn lenient_read_of_sparse_output() {
        let json = r#"{"documents": [
            {"doc_id": 2, "stage_5_fact_extraction": null,
             "stage_6_contradiction_check": {"all_clean_facts": null,
                "cross_doc_contradictions": [{"new_fact_text": "a", "existing_fact_text": null,
                    "nli_confidence": null, "extra": 1}]}},
            {"doc_id": "003", "error": "RuntimeError('x')"}
        ], "unknown": true}"#;
        let out: E2eOutput = serde_json::from_str(json).unwrap();
        assert_eq!(out.documents.len(), 2);
        let first = &out.documents[0];
        assert_eq!(first.doc_id.as_deref(), Some("2"));
        assert!(first.stage_5_fact_extraction.is_none());
        let s6 = first.stage_6_contradiction_check.as_ref().unwrap();
        assert!(s6.all_clean_facts.is_empty());
        assert_eq!(s6.cross_doc_contradictions[0].existing_fact_text, "");
        assert!(s6.cross_doc_contradictions[0].nli_confidence.abs() < f64::EPSILON);
        assert_eq!(out.documents[1].error.as_deref(), Some("RuntimeError('x')"));
    }

    #[test]
    fn all_facts_presence_is_kept() {
        let with: FactExtractionSummary = serde_json::from_str(r#"{"all_facts": []}"#).unwrap();
        let without: FactExtractionSummary = serde_json::from_str("{}").unwrap();
        assert_eq!(with.all_facts, Some(vec![]));
        assert_eq!(without.all_facts, None);
    }

    #[test]
    fn failed_document_serialises_header_only() {
        let doc = E2eDocument {
            doc_id: Some("001".into()),
            doc_path: "a.jsonl".into(),
            error: Some("ValueError('boom')".into()),
            ..E2eDocument::default()
        };
        assert_eq!(
            to_python_json(&doc).unwrap(),
            r#"{"doc_id": "001", "case_id": null, "db_id": null, "title": null, "doc_path": "a.jsonl", "error": "ValueError('boom')"}"#
        );
    }

    #[test]
    fn document_round_trips_in_python_key_order() {
        let json = r#"{"doc_id": "001", "case_id": "c1", "db_id": "d1", "title": null, "doc_path": "x.jsonl", "processing_time_seconds": 1.5, "stage_4_entity_disambiguation": {"total_resolved": 1, "total_unresolved": 0, "resolution_by_strategy": {"exact": 1}, "new_entities_created_count": 0, "entities_reused_count": 1}, "stage_5_fact_extraction": {"total_facts": 1, "boilerplate_count": 0, "facts_by_value_type": {"DATE": 1}, "facts_by_property_type": {"UNKNOWN": 1}, "entity_linking": {"linked_subjects": 0, "linked_objects": 0, "linked_both": 0, "linked_either": 0, "linking_rate_percent": 0.0}, "all_facts": [{"fact_id": "F0", "text": "t", "value_type": "DATE", "property_type": null, "subject_entity_id": null, "object_entity_id": null, "is_boilerplate": false}]}, "stage_6_contradiction_check": {"total_checked": 1, "contradiction_count": 0, "needs_review_count": 0, "clean_count": 1, "committed_fact_count": 1, "all_clean_facts": [{"fact_id": "F0", "text": "t", "value_type": "DATE", "property_type": null, "subject_entity_id": null, "object_entity_id": null}], "intra_doc_contradictions": [], "cross_doc_contradictions": [{"new_fact_text": "a", "new_fact_id": "F1", "existing_fact_text": "b", "existing_fact_id": "H1", "nli_label": "contradiction", "nli_confidence": 0.97, "retrieval_sources": ["semantic"]}], "retrieval_stats": {"total_candidates_retrieved": 3, "retrieval_sources_breakdown": {"semantic": 1}}, "filtering_stats": {"pairs_after_prefilter": 2, "total_nli_calls": 2}, "nli_stats": {"nli_label_breakdown": {"contradiction": 1}, "contradiction_threshold": 0.7, "review_threshold": null}}}"#;
        let doc: E2eDocument = serde_json::from_str(json).unwrap();
        assert_eq!(to_python_json(&doc).unwrap(), json);
    }
}
