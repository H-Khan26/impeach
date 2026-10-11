//! Contradiction detection records (Python `schema/contradictions.py`).
//!
//! The stage configuration dataclasses of that module live in `ce-config`.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::facts::ExtractedFact;
use crate::ids::{DocId, EntityId, FactId};
use crate::Metadata;

/// Label assigned by the NLI model or LLM judge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NliLabel {
    /// `contradiction`
    Contradiction,
    /// `entailment`
    Entailment,
    /// `neutral`
    Neutral,
}

impl NliLabel {
    /// The Python label string.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Contradiction => "contradiction",
            Self::Entailment => "entailment",
            Self::Neutral => "neutral",
        }
    }
}

/// Whether a pair was found within one document or against previously committed facts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContradictionSource {
    /// `intra_document`
    IntraDocument,
    /// `cross_document`
    #[default]
    CrossDocument,
}

impl ContradictionSource {
    /// The Python literal value.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::IntraDocument => "intra_document",
            Self::CrossDocument => "cross_document",
        }
    }
}

/// Outcome of the contradiction check for one fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContradictionStatus {
    /// `clean`
    Clean,
    /// `contradiction`
    Contradiction,
    /// `needs_review`
    NeedsReview,
}

impl ContradictionStatus {
    /// The Python literal value.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Clean => "clean",
            Self::Contradiction => "contradiction",
            Self::NeedsReview => "needs_review",
        }
    }
}

/// Classification of a review-queue row.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ReviewClassification {
    /// `CONTRADICTION`
    Contradiction,
    /// `NEEDS_REVIEW`
    #[default]
    NeedsReview,
}

impl ReviewClassification {
    /// The Python literal value.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Contradiction => "CONTRADICTION",
            Self::NeedsReview => "NEEDS_REVIEW",
        }
    }
}

/// Review workflow state of a review-queue row.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ReviewStatus {
    /// `PENDING`
    #[default]
    Pending,
    /// `ACCEPTED`
    Accepted,
    /// `REJECTED`
    Rejected,
}

impl ReviewStatus {
    /// The Python literal value.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "PENDING",
            Self::Accepted => "ACCEPTED",
            Self::Rejected => "REJECTED",
        }
    }
}

/// A fact retrieved from the graph for comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievedFact {
    /// Fact id.
    pub fact_id: FactId,
    /// Fact text.
    pub fact_text: String,
    /// Retrieval methods that found the fact (`graph`, `bm25`, `vector`).
    #[serde(default)]
    pub sources: Vec<String>,
    /// Ranked relevance score.
    #[serde(default)]
    pub relevance_score: f64,
    /// Free-form metadata.
    #[serde(default)]
    pub metadata: Metadata,
    /// Party side (e.g. `PLAINTIFF`).
    #[serde(default)]
    pub side: Option<String>,
    /// Assertion subject.
    #[serde(default)]
    pub assertion_subject: Option<String>,
    /// Assertion relationship.
    #[serde(default)]
    pub assertion_relationship: Option<String>,
    /// Assertion object.
    #[serde(default)]
    pub assertion_object: Option<String>,
}

impl RetrievedFact {
    /// Creates a retrieved fact with the Python defaults for every optional field.
    pub fn new(fact_id: impl Into<FactId>, fact_text: impl Into<String>) -> Self {
        Self {
            fact_id: fact_id.into(),
            fact_text: fact_text.into(),
            sources: Vec::new(),
            relevance_score: 0.0,
            metadata: Metadata::new(),
            side: None,
            assertion_subject: None,
            assertion_relationship: None,
            assertion_object: None,
        }
    }
}

/// A pair of facts that may contradict each other.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContradictionPair {
    /// Text of the new fact.
    pub new_fact_text: String,
    /// Extraction id of the new fact.
    pub new_fact_id: FactId,
    /// Id of the fact it conflicts with.
    pub existing_fact_id: FactId,
    /// Text of the fact it conflicts with.
    pub existing_fact_text: String,
    /// NLI label.
    pub nli_label: NliLabel,
    /// NLI confidence.
    pub nli_confidence: f64,
    /// Retrieval methods that surfaced the pair.
    #[serde(default)]
    pub retrieval_sources: Vec<String>,
    /// Priority score from pre-filtering (higher is more relevant).
    #[serde(default)]
    pub priority_score: f64,
    /// Entity ids shared by the two facts.
    #[serde(default)]
    pub shared_entity_ids: Vec<EntityId>,
    /// Same document or cross-document.
    #[serde(default)]
    pub contradiction_source: ContradictionSource,
}

impl ContradictionPair {
    /// Creates a pair with the Python defaults for every optional field.
    pub fn new(
        new_fact_text: impl Into<String>,
        new_fact_id: impl Into<FactId>,
        existing_fact_id: impl Into<FactId>,
        existing_fact_text: impl Into<String>,
        nli_label: NliLabel,
        nli_confidence: f64,
    ) -> Self {
        Self {
            new_fact_text: new_fact_text.into(),
            new_fact_id: new_fact_id.into(),
            existing_fact_id: existing_fact_id.into(),
            existing_fact_text: existing_fact_text.into(),
            nli_label,
            nli_confidence,
            retrieval_sources: Vec::new(),
            priority_score: 0.0,
            shared_entity_ids: Vec::new(),
            contradiction_source: ContradictionSource::default(),
        }
    }
}

/// Result of the contradiction check for one new fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactContradictionResult {
    /// The checked fact.
    pub new_fact: ExtractedFact,
    /// Check outcome.
    pub status: ContradictionStatus,
    /// Conflicting pairs (empty when clean).
    #[serde(default)]
    pub contradicting_pairs: Vec<ContradictionPair>,
    /// Retrieval statistics. Mostly integer counters, but intra-document results also carry
    /// a string `source` entry, so values are kept as JSON.
    #[serde(default)]
    pub retrieval_stats: Metadata,
    /// Graph id of the fact once committed (clean facts only).
    #[serde(default)]
    pub helix_fact_id: Option<FactId>,
}

impl FactContradictionResult {
    /// Creates a result with no pairs, no stats and no committed id.
    #[must_use]
    pub fn new(new_fact: ExtractedFact, status: ContradictionStatus) -> Self {
        Self {
            new_fact,
            status,
            contradicting_pairs: Vec::new(),
            retrieval_stats: Metadata::new(),
            helix_fact_id: None,
        }
    }

    /// Integer statistic `key`, or 0 when it is missing or not an integer.
    #[must_use]
    pub fn retrieval_stat(&self, key: &str) -> i64 {
        self.retrieval_stats.get(key).and_then(serde_json::Value::as_i64).unwrap_or(0)
    }
}

const fn default_contradiction_threshold() -> f64 {
    0.85
}

const fn default_review_threshold() -> f64 {
    0.6
}

/// Contradiction report for one document processing run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContradictionReport {
    /// Document id.
    pub doc_id: DocId,
    /// Number of facts extracted.
    #[serde(default)]
    pub total_facts_extracted: usize,
    /// Facts that passed the check and were committed.
    #[serde(default)]
    pub clean_facts: Vec<ExtractedFact>,
    /// Graph ids of the committed facts.
    #[serde(default)]
    pub committed_fact_ids: Vec<FactId>,
    /// One result per contradicting or needs-review fact; a pair may appear under both facts.
    #[serde(default)]
    pub contradiction_results: Vec<FactContradictionResult>,
    /// Pairs deduplicated across directions; counts derive from this list.
    #[serde(default)]
    pub unique_contradiction_pairs: Vec<ContradictionPair>,
    /// Confidence at or above which a pair is a definite contradiction.
    #[serde(default = "default_contradiction_threshold")]
    pub contradiction_threshold: f64,
    /// Confidence at or above which a pair needs review.
    #[serde(default = "default_review_threshold")]
    pub review_threshold: f64,
    /// Processing time in milliseconds.
    #[serde(default)]
    pub processing_time_ms: f64,
    /// Number of NLI / judge calls.
    #[serde(default)]
    pub total_nli_calls: usize,
    /// Number of cross-document candidates retrieved.
    #[serde(default)]
    pub total_candidates_retrieved: usize,
    /// Number of pairs left after the prefilter.
    #[serde(default)]
    pub total_pairs_after_prefilter: usize,
    /// Debug trace of every pair considered.
    #[serde(default)]
    pub debug_trace: Vec<Metadata>,
}

impl ContradictionReport {
    /// Creates an empty report with the default thresholds (0.85 / 0.6).
    pub fn new(doc_id: impl Into<DocId>) -> Self {
        Self {
            doc_id: doc_id.into(),
            total_facts_extracted: 0,
            clean_facts: Vec::new(),
            committed_fact_ids: Vec::new(),
            contradiction_results: Vec::new(),
            unique_contradiction_pairs: Vec::new(),
            contradiction_threshold: default_contradiction_threshold(),
            review_threshold: default_review_threshold(),
            processing_time_ms: 0.0,
            total_nli_calls: 0,
            total_candidates_retrieved: 0,
            total_pairs_after_prefilter: 0,
            debug_trace: Vec::new(),
        }
    }

    /// Whether any result has status `contradiction` or `needs_review`.
    #[must_use]
    pub fn has_contradictions(&self) -> bool {
        self.contradiction_results.iter().any(|r| {
            matches!(
                r.status,
                ContradictionStatus::Contradiction | ContradictionStatus::NeedsReview
            )
        })
    }

    /// Number of unique pairs with confidence at or above `contradiction_threshold`.
    #[must_use]
    pub fn contradiction_count(&self) -> usize {
        self.unique_contradiction_pairs
            .iter()
            .filter(|p| p.nli_confidence >= self.contradiction_threshold)
            .count()
    }

    /// Number of unique pairs with `review_threshold <= confidence < contradiction_threshold`.
    #[must_use]
    pub fn needs_review_count(&self) -> usize {
        self.unique_contradiction_pairs
            .iter()
            .filter(|p| {
                self.review_threshold <= p.nli_confidence
                    && p.nli_confidence < self.contradiction_threshold
            })
            .count()
    }
}

/// Serialises `None` as `""` and reads `""` back as `None`, as the Python review item stores an
/// unset NLI label as the empty string.
mod empty_label {
    use super::{Deserialize, Deserializer, NliLabel, Serialize, Serializer};

    pub(super) fn serialize<S: Serializer>(
        label: &Option<NliLabel>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match label {
            Some(label) => label.serialize(serializer),
            None => serializer.serialize_str(""),
        }
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<NliLabel>, D::Error> {
        let raw = String::deserialize(deserializer)?;
        match raw.as_str() {
            "" => Ok(None),
            "contradiction" => Ok(Some(NliLabel::Contradiction)),
            "entailment" => Ok(Some(NliLabel::Entailment)),
            "neutral" => Ok(Some(NliLabel::Neutral)),
            other => Err(serde::de::Error::unknown_variant(
                other,
                &["", "contradiction", "entailment", "neutral"],
            )),
        }
    }
}

/// One row of the contradiction review queue.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContradictionReviewItem {
    /// Database row id.
    pub id: Option<i64>,
    /// Case id.
    pub case_id: String,
    /// Session id.
    pub session_id: String,
    /// Segment id.
    pub segment_id: String,
    /// Segment index.
    pub segment_index: i64,
    /// Document id.
    pub doc_id: DocId,
    /// Pipeline stage, e.g. `EXAMINATION` or `PROCESSING`.
    pub stage: String,
    /// Witness name.
    pub witness_name: Option<String>,
    /// Source stream.
    pub source_stream: Option<String>,
    /// Logical pipeline identifier.
    pub source_pipeline: Option<String>,
    /// Segment start timestamp.
    pub segment_start_ts: Option<String>,
    /// Segment end timestamp.
    pub segment_end_ts: Option<String>,
    /// Extraction id of the new fact.
    pub new_fact_id: FactId,
    /// Graph id of the new fact once committed.
    pub helix_new_fact_id: Option<FactId>,
    /// Graph id of the conflicting fact.
    pub existing_fact_id: FactId,
    /// Snapshot of the new fact.
    pub new_fact_payload: Metadata,
    /// Snapshot of the existing fact.
    pub existing_fact_payload: Metadata,
    /// NLI score.
    pub nli_score: f64,
    /// NLI label; `None` serialises as the empty string.
    #[serde(with = "empty_label")]
    pub nli_label: Option<NliLabel>,
    /// Contradiction or needs-review.
    pub classification: ReviewClassification,
    /// Same document or cross-document.
    pub contradiction_source: ContradictionSource,
    /// Review workflow state.
    pub status: ReviewStatus,
    /// Creation timestamp.
    pub created_at: String,
    /// Last update timestamp.
    pub updated_at: String,
    /// Reviewer.
    pub resolved_by: Option<String>,
    /// Reviewer note.
    pub resolution_note: Option<String>,
    /// Idempotency key (unique in SQLite).
    pub dedupe_key: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facts::{AssertionTriple, TemporalValidity};
    use pretty_assertions::assert_eq;
    use serde_json::json;

    fn make_extracted_fact(text: &str) -> ExtractedFact {
        ExtractedFact::new(
            "fact_001",
            text,
            AssertionTriple::new("Subject", "RELATES_TO", "Object"),
            1.0,
            TemporalValidity::default(),
        )
    }

    fn make_result(
        text: &str,
        status: ContradictionStatus,
        pairs: Vec<ContradictionPair>,
    ) -> FactContradictionResult {
        let mut result = FactContradictionResult::new(make_extracted_fact(text), status);
        result.contradicting_pairs = pairs;
        result
    }

    fn make_pair(confidence: f64) -> ContradictionPair {
        let mut pair = ContradictionPair::new(
            "New",
            "new_001",
            "existing_001",
            "Old",
            NliLabel::Contradiction,
            confidence,
        );
        pair.retrieval_sources = vec!["vector".into()];
        pair
    }

    fn report_with_pairs(pairs: Vec<ContradictionPair>) -> ContradictionReport {
        let mut report = ContradictionReport::new("t");
        report.unique_contradiction_pairs = pairs;
        report
    }

    #[test]
    fn report_default_values() {
        let report = ContradictionReport::new("test_doc");
        assert_eq!(report.doc_id, "test_doc");
        assert_eq!(report.total_facts_extracted, 0);
        assert!(report.clean_facts.is_empty());
        assert!(report.committed_fact_ids.is_empty());
        assert!(report.contradiction_results.is_empty());
        assert!(report.unique_contradiction_pairs.is_empty());
        assert!(report.processing_time_ms.abs() < f64::EPSILON);
        assert_eq!(report.total_nli_calls, 0);
        assert_eq!(report.total_candidates_retrieved, 0);
        assert_eq!(report.total_pairs_after_prefilter, 0);
        assert!((report.contradiction_threshold - 0.85).abs() < f64::EPSILON);
        assert!((report.review_threshold - 0.6).abs() < f64::EPSILON);
        assert!(report.review_threshold < report.contradiction_threshold);
    }

    #[test]
    fn report_minimal_json_uses_default_thresholds() {
        let report: ContradictionReport =
            serde_json::from_value(json!({"doc_id": "test_doc"})).expect("report deserialises");
        assert_eq!(report, ContradictionReport::new("test_doc"));
    }

    #[test]
    fn has_contradictions_empty() {
        assert!(!ContradictionReport::new("t").has_contradictions());
    }

    #[test]
    fn has_contradictions_with_contradiction() {
        let mut report = ContradictionReport::new("t");
        report.contradiction_results =
            vec![make_result("F", ContradictionStatus::Contradiction, vec![])];
        assert!(report.has_contradictions());
    }

    #[test]
    fn has_contradictions_with_needs_review() {
        let mut report = ContradictionReport::new("t");
        report.contradiction_results =
            vec![make_result("F", ContradictionStatus::NeedsReview, vec![])];
        assert!(report.has_contradictions());
    }

    #[test]
    fn has_contradictions_clean_only() {
        let mut report = ContradictionReport::new("t");
        report.clean_facts = vec![make_extracted_fact("Clean")];
        assert!(!report.has_contradictions());
        report.contradiction_results = vec![make_result("F", ContradictionStatus::Clean, vec![])];
        assert!(!report.has_contradictions());
    }

    #[test]
    fn counts_empty() {
        let report = ContradictionReport::new("t");
        assert_eq!(report.contradiction_count(), 0);
        assert_eq!(report.needs_review_count(), 0);
    }

    #[test]
    fn counts_use_unique_pairs_and_thresholds() {
        let report = report_with_pairs(vec![
            make_pair(0.95),
            make_pair(0.85),
            make_pair(0.70),
            make_pair(0.60),
            make_pair(0.30),
        ]);
        assert_eq!(report.contradiction_count(), 2);
        assert_eq!(report.needs_review_count(), 2);
    }

    #[test]
    fn counts_respect_custom_thresholds() {
        let mut report = report_with_pairs(vec![make_pair(0.95), make_pair(0.75), make_pair(0.55)]);
        report.contradiction_threshold = 0.7;
        report.review_threshold = 0.5;
        assert_eq!(report.contradiction_count(), 2);
        assert_eq!(report.needs_review_count(), 1);
    }

    #[test]
    fn counts_ignore_contradiction_results() {
        let mut report = report_with_pairs(vec![make_pair(0.95)]);
        report.contradiction_results = vec![
            make_result("F1", ContradictionStatus::Contradiction, vec![make_pair(0.95)]),
            make_result("F2", ContradictionStatus::Contradiction, vec![make_pair(0.95)]),
        ];
        assert_eq!(report.contradiction_count(), 1);
    }

    #[test]
    fn result_clean_status() {
        let result =
            FactContradictionResult::new(make_extracted_fact("F"), ContradictionStatus::Clean);
        assert_eq!(result.status, ContradictionStatus::Clean);
        assert!(result.contradicting_pairs.is_empty());
        assert!(result.helix_fact_id.is_none());
    }

    #[test]
    fn result_contradiction_with_pairs() {
        let result = make_result(
            "F",
            ContradictionStatus::Contradiction,
            vec![make_pair(0.9), make_pair(0.95)],
        );
        assert_eq!(result.status, ContradictionStatus::Contradiction);
        assert_eq!(result.contradicting_pairs.len(), 2);
    }

    #[test]
    fn result_retrieval_stats() {
        let mut result =
            FactContradictionResult::new(make_extracted_fact("F"), ContradictionStatus::Clean);
        result.retrieval_stats = serde_json::from_value(json!({
            "total_candidates": 10, "pairs_after_prefilter": 5, "nli_calls": 5
        }))
        .expect("stats map");
        assert_eq!(result.retrieval_stat("total_candidates"), 10);
        assert_eq!(result.retrieval_stat("pairs_after_prefilter"), 5);
        assert_eq!(result.retrieval_stat("missing"), 0);
    }

    #[test]
    fn result_retrieval_stats_accept_string_source() {
        let result: FactContradictionResult = serde_json::from_value(json!({
            "new_fact": serde_json::to_value(make_extracted_fact("F")).expect("fact"),
            "status": "needs_review",
            "retrieval_stats": {"source": "intra_document", "pairs_checked": 2}
        }))
        .expect("intra-doc result deserialises");
        assert_eq!(result.retrieval_stat("pairs_checked"), 2);
        assert_eq!(result.retrieval_stat("source"), 0);
    }

    #[test]
    fn result_helix_fact_id() {
        let mut result =
            FactContradictionResult::new(make_extracted_fact("F"), ContradictionStatus::Clean);
        result.helix_fact_id = Some("helix_12345".into());
        assert_eq!(result.helix_fact_id.as_ref().map(FactId::as_str), Some("helix_12345"));
    }

    #[test]
    fn pair_all_fields() {
        let mut pair = ContradictionPair::new(
            "New fact text",
            "new_001",
            "existing_001",
            "Existing fact text",
            NliLabel::Contradiction,
            0.95,
        );
        pair.retrieval_sources = vec!["graph".into(), "bm25".into()];
        pair.priority_score = 1.5;
        pair.shared_entity_ids = vec!["ent_001".into(), "ent_002".into()];
        let value = serde_json::to_value(&pair).expect("pair serialises");
        assert_eq!(
            value,
            json!({
                "new_fact_text": "New fact text",
                "new_fact_id": "new_001",
                "existing_fact_id": "existing_001",
                "existing_fact_text": "Existing fact text",
                "nli_label": "contradiction",
                "nli_confidence": 0.95,
                "retrieval_sources": ["graph", "bm25"],
                "priority_score": 1.5,
                "shared_entity_ids": ["ent_001", "ent_002"],
                "contradiction_source": "cross_document"
            })
        );
    }

    #[test]
    fn pair_default_values() {
        let pair: ContradictionPair = serde_json::from_value(json!({
            "new_fact_text": "New",
            "new_fact_id": "new",
            "existing_fact_id": "existing",
            "existing_fact_text": "Existing",
            "nli_label": "neutral",
            "nli_confidence": 0.5
        }))
        .expect("pair deserialises");
        assert!(pair.retrieval_sources.is_empty());
        assert!(pair.priority_score.abs() < f64::EPSILON);
        assert!(pair.shared_entity_ids.is_empty());
        assert_eq!(pair.contradiction_source, ContradictionSource::CrossDocument);
        assert_eq!(pair.nli_label, NliLabel::Neutral);
    }

    #[test]
    fn pair_intra_document_source() {
        let mut pair = make_pair(0.95);
        pair.contradiction_source = ContradictionSource::IntraDocument;
        let value = serde_json::to_value(&pair).expect("pair serialises");
        assert_eq!(value["contradiction_source"], json!("intra_document"));
    }

    #[test]
    fn retrieved_fact_all_fields() {
        let fact: RetrievedFact = serde_json::from_value(json!({
            "fact_id": "fact_001",
            "fact_text": "Retrieved fact text",
            "sources": ["graph", "vector"],
            "relevance_score": 1.2,
            "metadata": {"source": "helix"},
            "side": "PLAINTIFF",
            "assertion_subject": "James Carter",
            "assertion_relationship": "BECAME",
            "assertion_object": "CEO"
        }))
        .expect("retrieved fact deserialises");
        assert_eq!(fact.fact_id, "fact_001");
        assert_eq!(fact.sources, vec!["graph", "vector"]);
        assert!((fact.relevance_score - 1.2).abs() < f64::EPSILON);
        assert_eq!(fact.metadata.get("source"), Some(&json!("helix")));
        assert_eq!(fact.side.as_deref(), Some("PLAINTIFF"));
        assert_eq!(fact.assertion_subject.as_deref(), Some("James Carter"));
        assert_eq!(fact.assertion_relationship.as_deref(), Some("BECAME"));
        assert_eq!(fact.assertion_object.as_deref(), Some("CEO"));
    }

    #[test]
    fn retrieved_fact_default_values() {
        let fact: RetrievedFact =
            serde_json::from_value(json!({"fact_id": "fact_001", "fact_text": "Fact text"}))
                .expect("retrieved fact deserialises");
        assert_eq!(fact, RetrievedFact::new("fact_001", "Fact text"));
        assert!(fact.sources.is_empty());
        assert!(fact.relevance_score.abs() < f64::EPSILON);
        assert!(fact.metadata.is_empty());
        assert!(fact.side.is_none());
    }

    #[test]
    fn literal_enums_match_python_strings() {
        let cases = [
            (serde_json::to_value(NliLabel::Contradiction), NliLabel::Contradiction.as_str()),
            (serde_json::to_value(NliLabel::Entailment), NliLabel::Entailment.as_str()),
            (serde_json::to_value(NliLabel::Neutral), NliLabel::Neutral.as_str()),
            (serde_json::to_value(ContradictionStatus::Clean), "clean"),
            (serde_json::to_value(ContradictionStatus::Contradiction), "contradiction"),
            (serde_json::to_value(ContradictionStatus::NeedsReview), "needs_review"),
            (serde_json::to_value(ContradictionSource::IntraDocument), "intra_document"),
            (serde_json::to_value(ContradictionSource::CrossDocument), "cross_document"),
            (serde_json::to_value(ReviewClassification::Contradiction), "CONTRADICTION"),
            (serde_json::to_value(ReviewClassification::NeedsReview), "NEEDS_REVIEW"),
            (serde_json::to_value(ReviewStatus::Pending), "PENDING"),
            (serde_json::to_value(ReviewStatus::Accepted), "ACCEPTED"),
            (serde_json::to_value(ReviewStatus::Rejected), "REJECTED"),
        ];
        for (value, expected) in cases {
            assert_eq!(value.expect("enum serialises"), json!(expected));
        }
        assert_eq!(NliLabel::Contradiction.as_str(), "contradiction");
        assert_eq!(ContradictionStatus::NeedsReview.as_str(), "needs_review");
        assert_eq!(ContradictionSource::IntraDocument.as_str(), "intra_document");
        assert_eq!(ReviewClassification::NeedsReview.as_str(), "NEEDS_REVIEW");
        assert_eq!(ReviewStatus::Rejected.as_str(), "REJECTED");
    }

    #[test]
    fn review_item_defaults_match_python() {
        let value =
            serde_json::to_value(ContradictionReviewItem::default()).expect("item serialises");
        assert_eq!(
            value,
            json!({
                "id": null,
                "case_id": "",
                "session_id": "",
                "segment_id": "",
                "segment_index": 0,
                "doc_id": "",
                "stage": "",
                "witness_name": null,
                "source_stream": null,
                "source_pipeline": null,
                "segment_start_ts": null,
                "segment_end_ts": null,
                "new_fact_id": "",
                "helix_new_fact_id": null,
                "existing_fact_id": "",
                "new_fact_payload": {},
                "existing_fact_payload": {},
                "nli_score": 0.0,
                "nli_label": "",
                "classification": "NEEDS_REVIEW",
                "contradiction_source": "cross_document",
                "status": "PENDING",
                "created_at": "",
                "updated_at": "",
                "resolved_by": null,
                "resolution_note": null,
                "dedupe_key": null
            })
        );
        let back: ContradictionReviewItem =
            serde_json::from_value(json!({})).expect("empty item deserialises");
        assert_eq!(back, ContradictionReviewItem::default());
    }

    #[test]
    fn review_item_nli_label() {
        let item: ContradictionReviewItem =
            serde_json::from_value(json!({"nli_label": "contradiction"})).expect("deserialises");
        assert_eq!(item.nli_label, Some(NliLabel::Contradiction));
        let value = serde_json::to_value(&item).expect("serialises");
        assert_eq!(value["nli_label"], json!("contradiction"));
        assert!(serde_json::from_value::<ContradictionReviewItem>(json!({"nli_label": "maybe"}))
            .is_err());
    }
}
