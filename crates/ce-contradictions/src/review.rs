//! Review-queue items from a contradiction report (Python
//! `services/contradictions/review_utils.py`).
//!
//! One item per unique pair: classification `CONTRADICTION` when the confidence reaches the
//! report's `contradiction_threshold`, else `NEEDS_REVIEW`; the new-fact payload is the
//! serialised fact of the first result whose fact id matches the pair; the existing-fact
//! payload carries `fact_id`, `fact_text`, `retrieval_sources`, `priority_score` and
//! `shared_entity_ids`; status `PENDING`; `created_at` / `updated_at` the given timestamp or
//! the current UTC time (`%Y-%m-%dT%H:%M:%S%.6f`, Python `datetime.utcnow().isoformat()`);
//! `dedupe_key` unset (the review store fills it on insert).

use ce_core::{
    ContradictionReport, ContradictionReviewItem, DocId, Metadata, ReviewClassification,
    ReviewStatus,
};
use chrono::{NaiveDateTime, Timelike};
use serde_json::{json, Value};

/// Provenance of a batch of review items.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReviewProvenance {
    /// Case id.
    pub case_id: String,
    /// Session id.
    pub session_id: String,
    /// Segment id.
    pub segment_id: String,
    /// Segment index.
    pub segment_index: i64,
    /// Document id.
    pub doc_id: String,
    /// Pipeline stage (`PROCESSING`, `EXAMINATION`).
    pub stage: String,
    /// Witness name.
    pub witness_name: Option<String>,
    /// Source stream.
    pub source_stream: Option<String>,
    /// Segment start timestamp.
    pub segment_start_ts: Option<String>,
    /// Segment end timestamp.
    pub segment_end_ts: Option<String>,
    /// Logical pipeline id, e.g. `processing_v17`.
    pub source_pipeline: Option<String>,
}

/// Review items for the unique pairs of `report` (module docs); empty without pairs. An empty
/// `created_ts` counts as missing (Python `created_ts or utcnow()`).
#[must_use]
pub fn build_review_items_from_report(
    report: &ContradictionReport,
    provenance: &ReviewProvenance,
    created_ts: Option<&str>,
) -> Vec<ContradictionReviewItem> {
    if report.unique_contradiction_pairs.is_empty() {
        return Vec::new();
    }
    let created = match created_ts.filter(|ts| !ts.is_empty()) {
        Some(ts) => ts.to_owned(),
        None => python_isoformat(&chrono::Utc::now().naive_utc()),
    };

    report
        .unique_contradiction_pairs
        .iter()
        .map(|pair| {
            let classification = if pair.nli_confidence >= report.contradiction_threshold {
                ReviewClassification::Contradiction
            } else {
                ReviewClassification::NeedsReview
            };
            let new_fact_payload = report
                .contradiction_results
                .iter()
                .find(|result| result.new_fact.id == pair.new_fact_id)
                .and_then(|result| match serde_json::to_value(&result.new_fact) {
                    Ok(Value::Object(map)) => Some(map),
                    _ => None,
                })
                .unwrap_or_default();
            let mut existing_fact_payload = Metadata::new();
            existing_fact_payload.insert("fact_id".into(), json!(pair.existing_fact_id));
            existing_fact_payload.insert("fact_text".into(), json!(pair.existing_fact_text));
            existing_fact_payload.insert("retrieval_sources".into(), json!(pair.retrieval_sources));
            existing_fact_payload.insert("priority_score".into(), json!(pair.priority_score));
            existing_fact_payload.insert("shared_entity_ids".into(), json!(pair.shared_entity_ids));

            ContradictionReviewItem {
                id: None,
                case_id: provenance.case_id.clone(),
                session_id: provenance.session_id.clone(),
                segment_id: provenance.segment_id.clone(),
                segment_index: provenance.segment_index,
                doc_id: DocId::new(provenance.doc_id.as_str()),
                stage: provenance.stage.clone(),
                witness_name: provenance.witness_name.clone(),
                source_stream: provenance.source_stream.clone(),
                source_pipeline: provenance.source_pipeline.clone(),
                segment_start_ts: provenance.segment_start_ts.clone(),
                segment_end_ts: provenance.segment_end_ts.clone(),
                new_fact_id: pair.new_fact_id.clone(),
                helix_new_fact_id: None,
                existing_fact_id: pair.existing_fact_id.clone(),
                new_fact_payload,
                existing_fact_payload,
                nli_score: pair.nli_confidence,
                nli_label: Some(pair.nli_label),
                classification,
                contradiction_source: pair.contradiction_source,
                status: ReviewStatus::Pending,
                created_at: created.clone(),
                updated_at: created.clone(),
                resolved_by: None,
                resolution_note: None,
                dedupe_key: None,
            }
        })
        .collect()
}

/// Python `datetime.isoformat()` of a naive timestamp: microseconds, omitted when zero.
fn python_isoformat(timestamp: &NaiveDateTime) -> String {
    // Leap-second nanoseconds (>= 1e9) are clamped into the last microsecond of the second.
    let micros = (timestamp.nanosecond() / 1_000).min(999_999);
    let base = timestamp.format("%Y-%m-%dT%H:%M:%S");
    if micros == 0 {
        base.to_string()
    } else {
        format!("{base}.{micros:06}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn at(h: u32, m: u32, s: u32, nanos: u32) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(2024, 3, 5)
            .and_then(|d| d.and_hms_nano_opt(h, m, s, nanos))
            .expect("valid timestamp")
    }

    #[test]
    fn isoformat_matches_python() {
        assert_eq!(python_isoformat(&at(7, 8, 9, 0)), "2024-03-05T07:08:09");
        assert_eq!(python_isoformat(&at(7, 8, 9, 1_000)), "2024-03-05T07:08:09.000001");
        assert_eq!(python_isoformat(&at(23, 59, 59, 123_456_789)), "2024-03-05T23:59:59.123456");
        // Sub-microsecond precision truncates, as Python's clock has microsecond resolution.
        assert_eq!(python_isoformat(&at(0, 0, 0, 999)), "2024-03-05T00:00:00");
    }
}
