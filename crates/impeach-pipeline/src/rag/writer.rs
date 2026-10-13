//! Graph writes of the RAG substrate (Python `services/rag/helix_writer.py` and
//! `services/rag/semantic_index.py`): documents, chunks, sentence spans and chunk embeddings
//! through the [`GraphStore`] batch methods.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use impeach_core::pyjson::to_python_json_ascii;
use impeach_core::util::saturating_i64;
use impeach_core::{
    Chunk, ChunkId, ChunkWriteResult, DocumentPayload, DocumentWriteResult, EmbeddingWriteResult,
    Metadata, NodeId, SentenceSpan, SpanWriteResult,
};
use impeach_store::{ChunkNode, DocumentNode, EmbeddingWrite, GraphStore, SpanNode, VectorLabel};

use crate::error::Result;

/// Span type and label of sentence spans.
pub const SENTENCE_SPAN_TYPE: &str = "SENTENCE";

/// Formats a timestamp as RFC 3339 UTC with whole seconds (`YYYY-MM-DDTHH:MM:SSZ`); `None`
/// becomes the current time.
#[must_use]
pub fn format_rfc3339_utc(value: Option<&DateTime<Utc>>) -> String {
    value.copied().unwrap_or_else(Utc::now).format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

/// Serialises node metadata with Python `json.dumps` (ASCII) formatting, adding
/// `external_id = doc_id` when `doc_id` is present and `external_id` is not.
#[must_use]
pub fn serialize_metadata(metadata: &Metadata) -> String {
    let mut safe = metadata.clone();
    if !safe.contains_key("external_id") {
        if let Some(doc_id) = safe.get("doc_id").cloned() {
            safe.insert("external_id".into(), doc_id);
        }
    }
    to_python_json_ascii(&safe).unwrap_or_else(|_| "{}".to_owned())
}

/// The `Document` node of a payload: empty `doc_type` -> `UNKNOWN`, empty title -> `doc_id`,
/// forward-slash `file_path`, `total_pages` 0 when unknown, metadata
/// `{doc_id, file_path, doc_type, **document.metadata}`.
#[must_use]
pub fn document_node(document: &DocumentPayload) -> DocumentNode {
    let file_path = document.file_path.as_deref().unwrap_or_default().replace('\\', "/");
    let mut metadata = Metadata::new();
    metadata.insert("doc_id".into(), document.doc_id.as_str().into());
    metadata.insert("file_path".into(), file_path.clone().into());
    metadata.insert("doc_type".into(), document.doc_type.clone().into());
    for (key, value) in &document.metadata {
        metadata.insert(key.clone(), value.clone());
    }
    DocumentNode {
        doc_type: if document.doc_type.is_empty() {
            "UNKNOWN".to_owned()
        } else {
            document.doc_type.clone()
        },
        title: if document.title.is_empty() {
            document.doc_id.as_str().to_owned()
        } else {
            document.title.clone()
        },
        date_created: format_rfc3339_utc(document.date_created.as_ref()),
        file_path,
        total_pages: i64::from(document.total_pages.unwrap_or(0)),
        metadata: serialize_metadata(&metadata),
    }
}

/// The `DocumentChunk` node of a chunk; metadata gets `chunk_index` and `doc_id` defaults.
#[must_use]
pub fn chunk_node(chunk: &Chunk) -> ChunkNode {
    let mut metadata = chunk.metadata.clone();
    metadata.entry("chunk_index").or_insert_with(|| chunk.sequence_number.into());
    metadata.entry("doc_id").or_insert_with(|| chunk.doc_id.as_str().into());
    ChunkNode {
        text: chunk.text.clone(),
        page_number: i64::from(chunk.page_number.unwrap_or(0)),
        start_char: saturating_i64(chunk.start_char),
        end_char: saturating_i64(chunk.end_char),
        section: chunk.section.clone().unwrap_or_default(),
        sequence_number: saturating_i64(chunk.sequence_number),
        metadata: serialize_metadata(&metadata),
    }
}

/// The sentence `Span` node of `span` under chunk `chunk_id` of document `doc_node_id`.
#[must_use]
pub fn sentence_span_node(
    doc_node_id: &NodeId,
    chunk_id: &ChunkId,
    span: &SentenceSpan,
) -> SpanNode {
    let mut metadata = span.metadata.clone();
    metadata.entry("chunk_sequence").or_insert_with(|| span.chunk_sequence.into());
    SpanNode {
        doc_id: doc_node_id.clone(),
        chunk_id: chunk_id.clone(),
        start_char: saturating_i64(span.start_char),
        end_char: saturating_i64(span.end_char),
        span_type: SENTENCE_SPAN_TYPE.to_owned(),
        span_label: SENTENCE_SPAN_TYPE.to_owned(),
        surface_text: span.text.clone(),
        metadata: serialize_metadata(&metadata),
    }
}

/// Creates the `Document` node, or updates it in place when `document.helix_id` is set.
/// Sets `document.helix_id` to the node id.
///
/// # Errors
/// Store errors (`NotFound` when `helix_id` names no document).
pub fn ensure_document(
    store: &dyn GraphStore,
    document: &mut DocumentPayload,
) -> Result<DocumentWriteResult> {
    let node = document_node(document);
    if let Some(helix_id) = &document.helix_id {
        store.update_document(helix_id, &node)?;
        return Ok(DocumentWriteResult {
            doc_id: document.doc_id.clone(),
            helix_id: helix_id.clone(),
            created: false,
        });
    }
    let stored = store.create_document(&node)?;
    tracing::debug!(doc_id = %document.doc_id, helix_id = %stored.id, "created Document node");
    document.helix_id = Some(stored.id.clone());
    Ok(DocumentWriteResult { doc_id: document.doc_id.clone(), helix_id: stored.id, created: true })
}

/// Writes the chunks of document `doc` in one batch and sets each chunk's `helix_id`; results
/// are in chunk order. A reused document (`doc.created == false`) first loses its existing
/// chunks, their spans and vectors ([`GraphStore::delete_document_chunks`]), so re-indexing
/// leaves exactly the new chunks and one set of spans.
///
/// # Errors
/// Store errors.
pub fn write_chunks(
    store: &dyn GraphStore,
    doc: &DocumentWriteResult,
    chunks: Vec<Chunk>,
) -> Result<Vec<ChunkWriteResult>> {
    if !doc.created {
        let removed = store.delete_document_chunks(&doc.helix_id)?;
        tracing::debug!(helix_doc_id = %doc.helix_id, removed, "deleted chunks of reused Document");
    }
    let nodes: Vec<ChunkNode> = chunks.iter().map(chunk_node).collect();
    let created = store.create_chunks_batch(&doc.helix_id, &nodes)?;
    if created.len() != chunks.len() {
        return Err(impeach_store::StoreError::Query {
            query: impeach_store::graph::query_names::CREATE_DOCUMENT_CHUNK.to_owned(),
            message: format!(
                "the store returned {} chunks for {} requested",
                created.len(),
                chunks.len()
            ),
        }
        .into());
    }
    Ok(chunks
        .into_iter()
        .zip(created)
        .map(|(mut chunk, stored)| {
            let helix_id = NodeId::from(stored.id.as_str());
            chunk.helix_id = Some(helix_id.clone());
            ChunkWriteResult { chunk, helix_id, created: true }
        })
        .collect())
}

/// Writes sentence spans under their chunks in one batch. When the batch fails (non-fatally)
/// the spans are written one by one and failures are logged and skipped, as in Python, where a
/// rejected span never aborts indexing.
///
/// # Errors
/// Fatal store errors only.
pub fn write_sentence_spans(
    store: &dyn GraphStore,
    doc_node_id: &NodeId,
    chunks: &[ChunkWriteResult],
    spans: &[SentenceSpan],
) -> Result<Vec<SpanWriteResult>> {
    let chunk_ids: HashMap<usize, ChunkId> = chunks
        .iter()
        .map(|w| (w.chunk.sequence_number, ChunkId::from(w.helix_id.as_str())))
        .collect();
    let mut nodes = Vec::with_capacity(spans.len());
    let mut sequences = Vec::with_capacity(spans.len());
    for span in spans {
        let Some(chunk_id) = chunk_ids.get(&span.chunk_sequence) else {
            tracing::warn!(
                chunk_sequence = span.chunk_sequence,
                "skipping sentence span because its chunk is missing"
            );
            continue;
        };
        nodes.push(sentence_span_node(doc_node_id, chunk_id, span));
        sequences.push(span.chunk_sequence);
    }
    let result =
        |id, chunk_sequence| SpanWriteResult { span_id: id, chunk_sequence, created: true };
    match store.create_spans_batch(&nodes) {
        Ok(stored) => {
            Ok(stored.into_iter().zip(sequences).map(|(s, seq)| result(s.id, seq)).collect())
        }
        Err(error) if error.is_fatal() => Err(error.into()),
        Err(error) => {
            tracing::error!("sentence span batch failed ({error}); writing spans one by one");
            let mut written = Vec::with_capacity(nodes.len());
            for (node, seq) in nodes.iter().zip(sequences) {
                match store.create_span_for_chunk(node) {
                    Ok(stored) => written.push(result(stored.id, seq)),
                    Err(error) if error.is_fatal() => return Err(error.into()),
                    Err(error) => tracing::error!(
                        chunk_id = %node.chunk_id,
                        chunk_sequence = seq,
                        "failed to write sentence span: {error}"
                    ),
                }
            }
            Ok(written)
        }
    }
}

/// Stores one vector per chunk (aligned with `chunks`) in one batch. When the batch fails
/// (non-fatally) the vectors are written one by one and each failure becomes an entry of
/// `errors`, as in Python.
///
/// # Errors
/// Fatal store errors only.
pub fn upsert_chunk_embeddings(
    store: &dyn GraphStore,
    chunks: &[ChunkWriteResult],
    vectors: &[Vec<f32>],
) -> Result<EmbeddingWriteResult> {
    let items: Vec<EmbeddingWrite<'_>> = chunks
        .iter()
        .zip(vectors)
        .filter(|(chunk, _)| !chunk.helix_id.as_str().is_empty())
        .map(|(chunk, vector)| EmbeddingWrite { node_id: chunk.helix_id.as_str(), vector })
        .collect();
    if items.is_empty() {
        return Ok(EmbeddingWriteResult::default());
    }
    match store.create_embeddings_batch(VectorLabel::DocumentChunk, &items) {
        Ok(count) => Ok(EmbeddingWriteResult { count, errors: Vec::new() }),
        Err(error) if error.is_fatal() => Err(error.into()),
        Err(error) => {
            tracing::error!("chunk embedding batch failed ({error}); writing one by one");
            let mut result = EmbeddingWriteResult::default();
            for item in &items {
                match store.create_chunk_embedding(&ChunkId::from(item.node_id), item.vector) {
                    Ok(()) => result.count += 1,
                    Err(error) if error.is_fatal() => return Err(error.into()),
                    Err(error) => {
                        tracing::error!(
                            chunk_id = item.node_id,
                            "failed to create chunk embedding: {error}"
                        );
                        result.errors.push(error.to_string());
                    }
                }
            }
            if !result.errors.is_empty() {
                tracing::warn!(
                    errors = result.errors.len(),
                    "chunk embedding upsert completed with errors"
                );
            }
            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use impeach_core::rag::parse_iso_datetime;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    #[test]
    fn rfc3339_formatting() {
        let parse = |s: &str| parse_iso_datetime(s).expect("valid date");
        assert_eq!(
            format_rfc3339_utc(Some(&parse("2024-03-01T10:00:00Z"))),
            "2024-03-01T10:00:00Z"
        );
        assert_eq!(
            format_rfc3339_utc(Some(&parse("2024-03-01T12:30:00.123456+02:00"))),
            "2024-03-01T10:30:00Z"
        );
        assert_eq!(format_rfc3339_utc(Some(&parse("2024-03-01T10:00:00"))), "2024-03-01T10:00:00Z");
        assert_eq!(
            format_rfc3339_utc(Some(&parse("2024-03-01T05:00:00-05:00"))),
            "2024-03-01T10:00:00Z"
        );
        assert_eq!(format_rfc3339_utc(Some(&parse("2024-03-01"))), "2024-03-01T00:00:00Z");
        let now = format_rfc3339_utc(None);
        assert!(now.ends_with('Z') && !now.contains('+'));
        assert!(chrono::NaiveDateTime::parse_from_str(&now, "%Y-%m-%dT%H:%M:%SZ").is_ok());
    }

    #[test]
    fn document_node_params() {
        let mut doc = DocumentPayload::new("001", "t");
        doc.date_created = parse_iso_datetime("2024-03-01T10:00:00Z");
        doc.file_path = Some("C:\\data\\doc.pdf".into());
        doc.metadata.insert("case_id".into(), json!("C-1"));
        doc.metadata.insert("note".into(), json!("é"));
        let node = document_node(&doc);
        assert_eq!(node.date_created, "2024-03-01T10:00:00Z");
        assert_eq!(node.title, "001");
        assert_eq!(node.doc_type, "UNKNOWN");
        assert_eq!(node.file_path, "C:/data/doc.pdf");
        assert_eq!(node.total_pages, 0);
        assert_eq!(
            node.metadata,
            r#"{"doc_id": "001", "file_path": "C:/data/doc.pdf", "doc_type": "UNKNOWN", "case_id": "C-1", "note": "\u00e9", "external_id": "001"}"#
        );
    }

    #[test]
    fn chunk_and_span_node_params() {
        let mut chunk = Chunk::new("d1", "Hello.", 2, 10, 16);
        chunk.metadata.insert("chunk_index".into(), json!(2));
        let node = chunk_node(&chunk);
        assert_eq!((node.start_char, node.end_char, node.sequence_number), (10, 16, 2));
        assert_eq!(node.metadata, r#"{"chunk_index": 2, "doc_id": "d1", "external_id": "d1"}"#);

        let span = SentenceSpan {
            chunk_sequence: 2,
            start_char: 10,
            end_char: 16,
            text: "Hello.".into(),
            metadata: Metadata::new(),
        };
        let node = sentence_span_node(&NodeId::from("1"), &ChunkId::from("2"), &span);
        assert_eq!(node.span_type, "SENTENCE");
        assert_eq!(node.span_label, "SENTENCE");
        assert_eq!(node.surface_text, "Hello.");
        assert_eq!(node.metadata, r#"{"chunk_sequence": 2}"#);
    }

    #[test]
    fn metadata_keeps_explicit_external_id() {
        let mut meta = Metadata::new();
        meta.insert("doc_id".into(), json!("a"));
        meta.insert("external_id".into(), json!("x"));
        assert_eq!(serialize_metadata(&meta), r#"{"doc_id": "a", "external_id": "x"}"#);
        assert_eq!(serialize_metadata(&Metadata::new()), "{}");
    }
}
