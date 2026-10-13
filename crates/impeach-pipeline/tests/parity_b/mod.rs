//! Python prototype reference outputs (`fixtures/reference/<stage>/<scenario>.json`) turned into
//! Rust inputs for the coreference and ED parity tests: the Python chunks (`rag_index`), the
//! mentions the coreference stage saw (`mention_detection`), the fastcoref clusters it
//! post-processed (`coreference.raw_coref`) and its outputs.

#![allow(dead_code)]

use std::path::PathBuf;

use impeach_core::{
    Chunk, ChunkId, ChunkWriteResult, CorefClusterContext, DocId, DocumentPayload,
    DocumentWriteResult, EmbeddingWriteResult, IndexingResult, Mention, MentionId, NodeId, Span,
};
use impeach_models::CorefCluster;
use impeach_pipeline::PipelineContext;
use indexmap::IndexMap;
use serde::de::DeserializeOwned;
use serde_json::Value;

/// One document of a scenario with its Python reference data.
#[derive(Debug, Clone)]
pub struct PyDoc {
    /// Input file under `fixtures/input`.
    pub input: String,
    /// The document as read from the input file.
    pub payload: DocumentPayload,
    /// Python chunks with their graph ids (`text = doc_text[start:end]`).
    pub chunks: Vec<ChunkWriteResult>,
    /// Mentions handed to the coreference stage (`doc_mentions`).
    pub mentions: Vec<Mention>,
    /// The same mentions by chunk graph id (`mentions_by_chunk`).
    pub mentions_by_chunk: IndexMap<ChunkId, Vec<Mention>>,
    /// fastcoref clusters, in resolver order.
    pub raw_coref: Vec<CorefCluster>,
    /// fastcoref cluster ids (`<doc_id>_cluster_<idx>`) in the same order.
    pub raw_coref_ids: Vec<String>,
    /// The coreference document entry.
    pub coref: Value,
    /// The ED document entry.
    pub ed: Value,
}

impl PyDoc {
    /// Document id.
    pub fn doc_id(&self) -> &DocId {
        &self.payload.doc_id
    }

    /// Document text.
    pub fn text(&self) -> &str {
        &self.payload.text
    }

    /// Python `coref_clusters`.
    pub fn coref_clusters(&self) -> Vec<CorefClusterContext> {
        field(&self.coref, "coref_clusters")
    }

    /// Python `canonical_mention_for_cluster`.
    pub fn canonical_mention_for_cluster(&self) -> IndexMap<String, MentionId> {
        field(&self.coref, "canonical_mention_for_cluster")
    }

    /// Python `canonical_text_for_mention`.
    pub fn canonical_text_for_mention(&self) -> IndexMap<MentionId, String> {
        field(&self.coref, "canonical_text_for_mention")
    }

    /// Python `synthetic_pronoun_mentions` (empty when unset).
    pub fn synthetic_pronoun_mentions(&self) -> IndexMap<MentionId, MentionId> {
        match &self.coref["synthetic_pronoun_mentions"] {
            Value::Null => IndexMap::new(),
            value => serde_json::from_value(value.clone()).expect("synthetic_pronoun_mentions"),
        }
    }

    /// Python synthetic pronoun mentions appended to `doc_mentions`.
    pub fn added_mentions(&self) -> Vec<Mention> {
        field(&self.coref, "added_mentions")
    }

    /// Python `mentions_by_chunk` after coreference: chunk graph id -> mention ids.
    pub fn coref_mentions_by_chunk(&self) -> IndexMap<String, Vec<String>> {
        field(&self.coref, "mentions_by_chunk")
    }

    /// The indexing result of the Python chunks.
    pub fn indexing_result(&self) -> IndexingResult {
        IndexingResult {
            document: DocumentWriteResult {
                doc_id: self.payload.doc_id.clone(),
                helix_id: NodeId::new("py-doc"),
                created: true,
            },
            chunks: self.chunks.clone(),
            spans: Vec::new(),
            embeddings: EmbeddingWriteResult::default(),
            doc_text: Some(self.payload.text.clone()),
            timings: IndexMap::new(),
        }
    }

    /// Writes what RAGIndexStage and MentionStage leave in the context (Python chunks and
    /// mentions; the index counts as persisted).
    pub fn seed_context(&self, ctx: &mut PipelineContext) {
        let data = &mut ctx.data;
        data.document = Some(self.payload.clone());
        if !self.payload.metadata.is_empty() {
            data.doc_metadata = Some(self.payload.metadata.clone());
        }
        data.case_id =
            self.payload.metadata.get("case_id").and_then(Value::as_str).map(str::to_owned);
        data.doc_id = Some(self.payload.doc_id.clone());
        data.doc_text = Some(self.payload.text.clone());
        data.rag_index_result = Some(self.indexing_result());
        data.index_persisted = true;
        data.doc_mentions = Some(self.mentions.clone());
        data.mentions_by_chunk = Some(self.mentions_by_chunk.clone());
    }

    /// [`Self::seed_context`] plus the Python coreference outputs.
    pub fn seed_context_after_coref(&self, ctx: &mut PipelineContext) {
        self.seed_context(ctx);
        let added = self.added_mentions();
        let data = &mut ctx.data;
        data.coref_clusters = Some(self.coref_clusters());
        data.canonical_mention_for_cluster = Some(self.canonical_mention_for_cluster());
        data.canonical_text_for_mention = Some(self.canonical_text_for_mention());
        if !added.is_empty() {
            data.synthetic_pronoun_mentions = Some(self.synthetic_pronoun_mentions());
            let by_id: IndexMap<&str, &Mention> =
                self.mentions.iter().chain(added.iter()).map(|m| (m.id.as_str(), m)).collect();
            data.mentions_by_chunk = Some(
                self.coref_mentions_by_chunk()
                    .into_iter()
                    .map(|(chunk, ids)| {
                        let mentions = ids.iter().map(|id| (*by_id[id.as_str()]).clone()).collect();
                        (ChunkId::new(chunk), mentions)
                    })
                    .collect(),
            );
            data.doc_mentions.get_or_insert_with(Vec::new).extend(added);
        }
    }
}

fn field<T: DeserializeOwned>(value: &Value, key: &str) -> T {
    serde_json::from_value(value[key].clone()).unwrap_or_else(|e| panic!("{key}: {e}"))
}

fn workspace_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(rel)
}

/// `fixtures/reference/<stage>/<scenario>.json`.
pub fn reference(stage: &str, scenario: &str) -> Value {
    let path = workspace_path(&format!("fixtures/reference/{stage}/{scenario}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()))
}

fn documents(value: &Value) -> &Vec<Value> {
    value["documents"].as_array().expect("documents array")
}

/// Characters `start..end` of `text`.
fn char_slice(text: &str, start: usize, end: usize) -> String {
    text.chars().skip(start).take(end - start).collect()
}

fn to_usize(value: &Value) -> usize {
    usize::try_from(value.as_u64().expect("unsigned integer")).expect("fits usize")
}

fn chunks(doc: &Value, payload: &DocumentPayload) -> Vec<ChunkWriteResult> {
    doc["chunks"]
        .as_array()
        .expect("chunks")
        .iter()
        .map(|c| {
            let (start, end) = (to_usize(&c["start_char"]), to_usize(&c["end_char"]));
            let mut chunk = Chunk::new(
                payload.doc_id.clone(),
                char_slice(&payload.text, start, end),
                to_usize(&c["sequence_number"]),
                start,
                end,
            );
            let helix_id = c["helix_id"].as_str().expect("helix_id");
            chunk.helix_id = Some(NodeId::new(helix_id));
            chunk.metadata = payload.metadata.clone();
            chunk.sentence_spans = c["sentence_spans"]
                .as_array()
                .expect("sentence_spans")
                .iter()
                .map(|s| Span::new(to_usize(&s["start"]), to_usize(&s["end"])))
                .collect();
            ChunkWriteResult { chunk, helix_id: NodeId::new(helix_id), created: true }
        })
        .collect()
}

fn raw_coref(doc: &Value) -> (Vec<CorefCluster>, Vec<String>) {
    let mut clusters = Vec::new();
    let mut ids = Vec::new();
    for output in doc["raw_coref"].as_array().expect("raw_coref") {
        for cluster in output["clusters"].as_array().expect("clusters") {
            ids.push(cluster["id"].as_str().expect("cluster id").to_owned());
            let spans = cluster["mentions"]
                .as_array()
                .expect("mentions")
                .iter()
                .map(|m| (to_usize(&m["start_char"]), to_usize(&m["end_char"])))
                .collect();
            clusters.push(CorefCluster { spans });
        }
    }
    (clusters, ids)
}

/// The documents of `scenario` (`green_valley`, `mason`) in run order.
pub fn scenario(scenario: &str) -> Vec<PyDoc> {
    let rag = reference("rag_index", scenario);
    let mention = reference("mention_detection", scenario);
    let coref = reference("coreference", scenario);
    let ed = reference("ed_resolution", scenario);
    let (rag, mention, coref, ed) =
        (documents(&rag), documents(&mention), documents(&coref), documents(&ed));
    assert!(rag.len() == mention.len() && rag.len() == coref.len() && rag.len() == ed.len());
    rag.iter()
        .zip(mention)
        .zip(coref)
        .zip(ed)
        .map(|(((rag, mention), coref), ed)| {
            let input = rag["input"].as_str().expect("input").to_owned();
            let payload = impeach_ingest::read_first_record(workspace_path(&format!(
                "fixtures/input/{input}"
            )))
            .expect("input document reads");
            assert_eq!(payload.doc_id.as_str(), rag["doc_id"].as_str().expect("doc_id"));
            assert_eq!(payload.text.chars().count(), to_usize(&rag["doc_text_len"]));
            let mentions: Vec<Mention> = field(mention, "doc_mentions");
            let by_id: IndexMap<&str, &Mention> =
                mentions.iter().map(|m| (m.id.as_str(), m)).collect();
            let ids: IndexMap<String, Vec<String>> = field(mention, "mentions_by_chunk");
            let mentions_by_chunk = ids
                .into_iter()
                .map(|(chunk, ids)| {
                    let list = ids.iter().map(|id| (*by_id[id.as_str()]).clone()).collect();
                    (ChunkId::new(chunk), list)
                })
                .collect();
            let (raw_coref, raw_coref_ids) = raw_coref(coref);
            PyDoc {
                chunks: chunks(rag, &payload),
                input,
                payload,
                mentions,
                mentions_by_chunk,
                raw_coref,
                raw_coref_ids,
                coref: coref.clone(),
                ed: ed.clone(),
            }
        })
        .collect()
}
