//! Shared helpers of the stage parity tests: the Python prototype's reference outputs under
//! `fixtures/reference/<stage>/<scenario>.json` (see `fixtures/reference/PROVENANCE.json`),
//! the scenario inputs, and a GLiNER2 model that replays the Python detector output.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

use impeach_config::MappingsRegistry;
use impeach_core::{
    Chunk, ChunkWriteResult, DocumentPayload, DocumentWriteResult, EmbeddingWriteResult,
    IndexingResult, Mention, Span,
};
use impeach_models::{ModelError, NerModel, NerSpan, RelationHit};
use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::Value;

/// The scenarios of the reference fixtures.
pub const SCENARIOS: [&str; 2] = ["green_valley", "mason"];

/// GLiNER2 label set of the shipped `MentionStage` config.
pub const LABEL_SET: &str = "legal_entities";

pub fn fixtures_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// `fixtures/reference/<stage>/<file>`, parsed.
pub fn reference_file<T: for<'de> Deserialize<'de>>(stage: &str, file: &str) -> T {
    let path = fixtures_root().join("reference").join(stage).join(file);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()))
}

/// The documents of `fixtures/reference/<stage>/<scenario>.json`.
pub fn reference_docs<T: for<'de> Deserialize<'de>>(stage: &str, scenario: &str) -> Vec<T> {
    #[derive(Deserialize)]
    struct Scenario<T> {
        documents: Vec<T>,
    }
    reference_file::<Scenario<T>>(stage, &format!("{scenario}.json")).documents
}

/// The scenario input `fixtures/input/<name>`.
pub fn input_document(name: &str) -> DocumentPayload {
    impeach_ingest::read_first_record(fixtures_root().join("input").join(name))
        .expect("input fixture reads")
}

/// Characters `start..end` of `text`.
pub fn char_slice(text: &str, start: usize, end: usize) -> String {
    text.chars().skip(start).take(end - start).collect()
}

/// A Python chunk of the `rag_index` reference.
#[derive(Debug, Clone, Deserialize)]
pub struct PyChunk {
    pub helix_id: String,
    pub sequence_number: usize,
    pub start_char: usize,
    pub end_char: usize,
    pub sentence_spans: Vec<Span>,
}

/// One document of the `rag_index` reference.
#[derive(Debug, Clone, Deserialize)]
pub struct PyRagDoc {
    pub input: String,
    pub doc_id: String,
    pub doc_text_len: usize,
    pub chunks: Vec<PyChunk>,
    pub span_count: usize,
    pub embedding_count: usize,
}

/// One detector call of the `mention_detection` reference.
#[derive(Debug, Clone, Deserialize)]
pub struct PyDetectorCall {
    pub detector: String,
    pub segment_id: String,
    pub entities: Vec<Mention>,
}

/// One document of the `mention_detection` reference.
#[derive(Debug, Clone, Deserialize)]
pub struct PyMentionDoc {
    pub input: String,
    pub doc_id: String,
    pub detector_raw: Vec<PyDetectorCall>,
    pub doc_mentions: Vec<Mention>,
    pub mentions_by_chunk: IndexMap<String, Vec<String>>,
}

impl PyMentionDoc {
    /// Raw output of the Python detector class `detector` for `segment_id`.
    pub fn raw(&self, detector: &str, segment_id: &str) -> &[Mention] {
        self.detector_raw
            .iter()
            .find(|call| call.detector == detector && call.segment_id == segment_id)
            .map_or(&[], |call| call.entities.as_slice())
    }
}

/// An `IndexingResult` over the Python chunks of `doc`, with the Python graph ids.
pub fn python_indexing_result(doc: &PyRagDoc, document: &DocumentPayload) -> IndexingResult {
    assert_eq!(document.text.chars().count(), doc.doc_text_len, "{}", doc.input);
    let chunks = doc
        .chunks
        .iter()
        .map(|c| {
            let text = char_slice(&document.text, c.start_char, c.end_char);
            let mut chunk =
                Chunk::new(doc.doc_id.as_str(), text, c.sequence_number, c.start_char, c.end_char);
            chunk.sentence_spans = c.sentence_spans.clone();
            ChunkWriteResult { chunk, helix_id: c.helix_id.as_str().into(), created: true }
        })
        .collect();
    IndexingResult {
        document: DocumentWriteResult {
            doc_id: doc.doc_id.as_str().into(),
            helix_id: "1".into(),
            created: true,
        },
        chunks,
        spans: Vec::new(),
        embeddings: EmbeddingWriteResult::default(),
        doc_text: Some(document.text.clone()),
        timings: IndexMap::new(),
    }
}

/// A GLiNER2 model replaying recorded spans per chunk text; an unknown text is an error.
#[derive(Debug, Default)]
pub struct ReplayNer {
    by_text: HashMap<String, Vec<NerSpan>>,
}

impl ReplayNer {
    /// Spans reproducing the Python GLiNER2 detector output `mentions` (already aligned,
    /// typed and sorted by start) for chunk `text`.
    ///
    /// Python numbers mentions (`ent_<segment>_<i>`) while walking the model output label by
    /// label in label-set order, so each mention gets the earliest label of its type that is
    /// not before the previous mention's label; the spans come back in that walk order.
    pub fn add_python_chunk(&mut self, text: &str, mentions: &[Mention]) {
        let mappings = MappingsRegistry::load().expect("mappings");
        let labels = mappings.gliner2.get_entity_label_set(LABEL_SET).expect("label set");
        let default_type = mappings.entity_types.default_entity_type();
        let type_of = |label: &str| {
            mappings
                .gliner2
                .gliner2_to_entity_type
                .get(&label.to_lowercase())
                .map_or(default_type, String::as_str)
                .to_owned()
        };
        let mut ordered: Vec<&Mention> = mentions.iter().collect();
        ordered.sort_by_key(|m| walk_index(m));
        let mut rank = 0;
        let mut spans = Vec::with_capacity(ordered.len());
        for mention in ordered {
            let found = (rank..labels.len())
                .find(|&r| type_of(&labels[r]) == mention.entity_type)
                .or_else(|| labels.iter().position(|l| type_of(l) == mention.entity_type))
                .unwrap_or_else(|| panic!("no label of type {}", mention.entity_type));
            rank = found;
            let score = mention.score.expect("GLiNER2 mentions have a score");
            spans.push(NerSpan {
                start: mention.span.start,
                end: mention.span.end,
                label: labels[found].clone(),
                score: score as f32,
                text: mention.name.clone(),
            });
        }
        self.by_text.entry(text.to_owned()).or_default().extend(spans);
    }
}

/// `i` of a Python GLiNER2 id `ent_<segment>_<i>`.
pub fn walk_index(mention: &Mention) -> usize {
    mention
        .id
        .as_str()
        .rsplit('_')
        .next()
        .and_then(|i| i.parse().ok())
        .unwrap_or_else(|| panic!("not a GLiNER2 id: {}", mention.id))
}

impl NerModel for ReplayNer {
    fn extract_entities(
        &self,
        text: &str,
        _labels: &[&str],
        threshold: f32,
    ) -> Result<Vec<NerSpan>, ModelError> {
        let spans = self
            .by_text
            .get(text)
            .ok_or_else(|| ModelError::Inference("no recorded GLiNER2 output for chunk".into()))?;
        Ok(spans.iter().filter(|s| s.score >= threshold).cloned().collect())
    }

    fn extract_relations(
        &self,
        _text: &str,
        _labels: &[&str],
        _threshold: f32,
    ) -> Result<Vec<RelationHit>, ModelError> {
        Ok(Vec::new())
    }
}

/// The comparable fields of a mention: everything but the id, `doc_id` and `chunk_id`.
#[derive(Debug, Clone, PartialEq)]
pub struct MentionKey {
    pub start: usize,
    pub end: usize,
    pub entity_type: String,
    pub source: Option<String>,
    pub name: String,
    pub score: Option<f64>,
    pub sentence_index: Option<usize>,
    pub sources: Vec<(String, String, Option<f64>)>,
    pub metadata: Value,
}

pub fn mention_key(m: &Mention) -> MentionKey {
    MentionKey {
        start: m.span.start,
        end: m.span.end,
        entity_type: m.entity_type.clone(),
        source: m.source.clone(),
        name: m.name.clone(),
        score: m.score,
        sentence_index: m.sentence_index,
        sources: m.sources.iter().map(|s| (s.name.clone(), s.label.clone(), s.score)).collect(),
        metadata: serde_json::to_value(&m.metadata).expect("metadata serialises"),
    }
}

/// `(start, end, type, source, name)` of a raw detector mention.
pub fn raw_key(m: &Mention) -> (usize, usize, String, Option<String>, String) {
    (m.span.start, m.span.end, m.entity_type.clone(), m.source.clone(), m.name.clone())
}
