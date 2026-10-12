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

use ce_core::{ContradictionReport, ContradictionReviewItem};

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

/// Review items for the unique pairs of `report` (module docs); empty without pairs.
#[must_use]
pub fn build_review_items_from_report(
    report: &ContradictionReport,
    provenance: &ReviewProvenance,
    created_ts: Option<&str>,
) -> Vec<ContradictionReviewItem> {
    todo!("worker B")
}
