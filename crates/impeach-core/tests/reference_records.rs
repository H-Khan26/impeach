//! Round-trips the reference record JSON in `fixtures/reference/records/` (produced by the
//! Python schema classes with `json.dumps(indent=2)`) and checks the Rust output is identical.

use std::fs;
use std::path::PathBuf;

use impeach_core::{
    Chunk, ChunkEmbeddingInput, ChunkingResult, ContradictionPair, ContradictionReport,
    ContradictionReviewItem, ContradictionSource, ContradictionStatus, CorefClusterContext,
    DocumentPayload, EdContext, EdResult, ExtractedFact, FactContradictionResult,
    FallbackVocabulary, IndexingResult, Mention, NliLabel, RelationshipType, RetrievedFact,
    ReviewClassification, SegmentEr, TemporalContext,
};
use pretty_assertions::assert_eq;
use serde::de::DeserializeOwned;
use serde::Serialize;

fn fixture_text(name: &str) -> String {
    let path: PathBuf =
        [env!("CARGO_MANIFEST_DIR"), "..", "..", "fixtures", "reference", "records"]
            .iter()
            .collect::<PathBuf>()
            .join(format!("{name}.json"));
    fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
        .replace("\r\n", "\n")
}

/// Deserialises the fixture as `T`, then checks that re-serialising yields the same JSON value
/// and the same bytes as Python's `json.dumps(indent=2)`.
fn round_trip<T: DeserializeOwned + Serialize>(name: &str) -> T {
    let text = fixture_text(name);
    let record: T =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{name}: deserialise failed: {e}"));
    let expected: serde_json::Value = serde_json::from_str(&text).expect("fixture is valid JSON");
    let actual = serde_json::to_value(&record).expect("record serialises");
    assert_eq!(actual, expected, "{name}: JSON value differs");
    let pretty = serde_json::to_string_pretty(&record).expect("record serialises") + "\n";
    assert_eq!(pretty, text, "{name}: serialised bytes differ");
    record
}

#[test]
fn mention() {
    let m: Mention = round_trip("mention");
    assert_eq!(m.entity_type, "PERSON");
    assert_eq!(m.sources.len(), 2);
    assert_eq!(m.sources[1].score, None);
}

#[test]
fn segment_er() {
    let s: SegmentEr = round_trip("segment_er");
    assert_eq!(s.relationships[0].relationship_type, RelationshipType::EmployedBy);
}

#[test]
fn coref_cluster_context() {
    let c: CorefClusterContext = round_trip("coref_cluster_context");
    assert_eq!(c.member_mention_ids.len(), 2);
}

#[test]
fn extracted_fact() {
    let f: ExtractedFact = round_trip("extracted_fact");
    assert_eq!(f.temporal_context, TemporalContext::PriorIncident);
    assert_eq!(f.normalized_value_type(&FallbackVocabulary), "NUMERIC");
    assert_eq!(f.normalized_property_type(&FallbackVocabulary), "QUANTITATIVE");
    assert!(f.is_denial);
}

#[test]
fn contradiction_pair() {
    let p: ContradictionPair = round_trip("contradiction_pair");
    assert_eq!(p.nli_label, NliLabel::Contradiction);
    assert_eq!(p.contradiction_source, ContradictionSource::CrossDocument);
}

#[test]
fn retrieved_fact() {
    let f: RetrievedFact = round_trip("retrieved_fact");
    assert_eq!(f.side.as_deref(), Some("PLAINTIFF"));
}

#[test]
fn fact_contradiction_result() {
    let r: FactContradictionResult = round_trip("fact_contradiction_result");
    assert_eq!(r.status, ContradictionStatus::Contradiction);
    assert_eq!(r.retrieval_stat("total_candidates"), 10);
}

#[test]
fn contradiction_report() {
    let r: ContradictionReport = round_trip("contradiction_report");
    assert!(r.has_contradictions());
    assert_eq!(r.contradiction_count(), 1);
    assert_eq!(r.needs_review_count(), 1);
    assert_eq!(r.contradiction_results[1].retrieval_stat("pairs_checked"), 1);
}

#[test]
fn contradiction_review_item() {
    let item: ContradictionReviewItem = round_trip("contradiction_review_item");
    assert_eq!(item.classification, ReviewClassification::Contradiction);
    assert_eq!(item.nli_label, Some(NliLabel::Contradiction));
}

#[test]
fn document_payload() {
    let p: DocumentPayload = round_trip("document_payload");
    assert!(p.date_created.is_some());
}

#[test]
fn chunking_result() {
    let r: ChunkingResult = round_trip("chunking_result");
    assert_eq!(r.chunks[1].neighbors.get("prev_chunk_seq"), Some(&0));
    let _: Chunk =
        serde_json::from_value(serde_json::to_value(&r.chunks[0]).expect("chunk serialises"))
            .expect("chunk deserialises");
}

#[test]
fn chunk_embedding_input() {
    let e: ChunkEmbeddingInput = round_trip("chunk_embedding_input");
    assert_eq!(e.embedding.len(), 3);
}

#[test]
fn indexing_result() {
    let r: IndexingResult = round_trip("indexing_result");
    assert_eq!(r.timings.keys().collect::<Vec<_>>(), ["document", "chunking", "embeddings"]);
}

#[test]
fn ed_context() {
    let c: EdContext = round_trip("ed_context");
    assert_eq!(c.doc_context.case_id.as_deref(), Some("case-1"));
}

#[test]
fn ed_result() {
    let r: EdResult = round_trip("ed_result");
    assert_eq!(r.candidates[0].source, "exact");
}
