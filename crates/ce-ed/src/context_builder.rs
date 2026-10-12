//! ED contexts of mentions (Python `services/ed/context_builder.py`): the sentence window of
//! the mention's chunk and document-level metadata.

use std::collections::HashMap;

use ce_config::mappings::EntityTypes;
use ce_core::{
    Chunk, ChunkWriteResult, DocContext, EdContext, LocalContext, Mention, Metadata, Span,
};
use serde_json::Value;

/// Builds [`EdContext`]s from the indexed chunks and the document metadata.
#[derive(Debug)]
pub struct ContextBuilder<'a> {
    doc_metadata: &'a Metadata,
    entity_types: &'a EntityTypes,
    /// Chunks by graph id; a later chunk with the same id wins.
    chunks_by_id: HashMap<&'a str, &'a Chunk>,
}

impl<'a> ContextBuilder<'a> {
    /// A builder over the chunks written by the RAG index stage (looked up by graph id) and
    /// the document metadata (`doc_type`, `case_id`, `jurisdiction`, `party_names`).
    #[must_use]
    pub fn new(
        chunks: &'a [ChunkWriteResult],
        doc_metadata: &'a Metadata,
        entity_types: &'a EntityTypes,
    ) -> Self {
        let chunks_by_id = chunks
            .iter()
            .filter(|written| !written.helix_id.as_str().is_empty())
            .map(|written| (written.helix_id.as_str(), &written.chunk))
            .collect();
        Self { doc_metadata, entity_types, chunks_by_id }
    }

    /// Context of `mention`, or `None` when its `chunk_id` is missing or not one of the chunks.
    /// `span_label` is the mention type normalised with [`EntityTypes::normalize_label`].
    #[must_use]
    pub fn build_context(&self, mention: &Mention) -> Option<EdContext> {
        let chunk_id = mention.chunk_id.as_ref().filter(|id| !id.as_str().is_empty())?;
        let chunk = *self.chunks_by_id.get(chunk_id.as_str())?;
        Some(EdContext {
            mention: mention.clone(),
            local_context: local_context(mention, chunk),
            doc_context: self.doc_context(chunk),
            span_label: self.entity_types.normalize_label(&mention.entity_type).to_owned(),
            normalized_key: None,
        })
    }

    /// Document context from the document metadata merged with the chunk metadata (chunk
    /// keys win). `doc_type` defaults to `unknown`; a non-string `doc_type`, `case_id` or
    /// `jurisdiction` counts as missing, and `party_names` other than a list of strings as
    /// empty.
    fn doc_context(&self, chunk: &Chunk) -> DocContext {
        let lookup = |key: &str| chunk.metadata.get(key).or_else(|| self.doc_metadata.get(key));
        let string = |key: &str| lookup(key).and_then(Value::as_str).map(str::to_owned);
        let party_names = lookup("party_names")
            .and_then(Value::as_array)
            .and_then(|names| {
                names.iter().map(|name| name.as_str().map(str::to_owned)).collect::<Option<_>>()
            })
            .unwrap_or_default();
        DocContext {
            doc_id: chunk.doc_id.clone(),
            doc_type: string("doc_type").unwrap_or_else(|| "unknown".to_owned()),
            case_id: string("case_id"),
            jurisdiction: string("jurisdiction"),
            party_names,
        }
    }
}

/// Sentence window of the mention: its sentence (by `sentence_index` into the chunk's sentence
/// spans) and the neighbouring sentences of the same chunk; the whole chunk text when the
/// chunk has no spans, the index is missing or out of range.
fn local_context(mention: &Mention, chunk: &Chunk) -> LocalContext {
    let mut context = LocalContext::new(chunk.text.as_str(), chunk.text.as_str());
    let spans = &chunk.sentence_spans;
    let Some(index) = mention.sentence_index.filter(|&i| i < spans.len()) else {
        return context;
    };
    let text_len = chunk.text.chars().count();
    context.sentence_text = span_text(chunk, text_len, spans[index]);
    if index > 0 {
        context.prev_sentence_text = Some(span_text(chunk, text_len, spans[index - 1]));
    }
    if index + 1 < spans.len() {
        context.next_sentence_text = Some(span_text(chunk, text_len, spans[index + 1]));
    }
    context
}

/// Text of a document-offset span within the chunk, with the Python clamping
/// `text[max(0, start - offset):min(len, end - offset)]`.
fn span_text(chunk: &Chunk, text_len: usize, span: Span) -> String {
    let offset = to_i64(chunk.start_char);
    let start = (to_i64(span.start) - offset).max(0);
    let end = (to_i64(span.end) - offset).min(to_i64(text_len));
    py_char_slice(&chunk.text, text_len, start, end)
}

fn to_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

/// Python `text[start:end]` over characters: negative bounds count from the end, bounds are
/// clamped to the text, and `start >= end` yields an empty string.
fn py_char_slice(text: &str, text_len: usize, start: i64, end: i64) -> String {
    let len = to_i64(text_len);
    let resolve = |bound: i64| {
        let bound = if bound < 0 { (bound + len).max(0) } else { bound.min(len) };
        usize::try_from(bound).unwrap_or(0)
    };
    let (start, end) = (resolve(start), resolve(end));
    if start >= end {
        return String::new();
    }
    text.chars().skip(start).take(end - start).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ce_config::MappingsRegistry;
    use ce_core::NodeId;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    const TEXT: &str = "Alice met Bob. Bob paid Carol. Carol left.";

    /// Chunk at document offset 100 with three sentence spans.
    fn chunk() -> ChunkWriteResult {
        let mut chunk = Chunk::new("doc-1", TEXT, 0, 100, 100 + TEXT.len());
        chunk.sentence_spans = vec![Span::new(100, 114), Span::new(115, 130), Span::new(131, 142)];
        ChunkWriteResult { chunk, helix_id: NodeId::new("17"), created: true }
    }

    fn mention(sentence_index: Option<usize>) -> Mention {
        let mut mention = Mention::new("m1", "PERSON", "Bob", Span::new(110, 113));
        mention.chunk_id = Some("17".into());
        mention.sentence_index = sentence_index;
        mention
    }

    fn entity_types() -> &'static EntityTypes {
        &MappingsRegistry::load().expect("embedded mappings load").entity_types
    }

    fn local_of(chunks: &[ChunkWriteResult], mention: &Mention) -> LocalContext {
        let metadata = Metadata::new();
        let builder = ContextBuilder::new(chunks, &metadata, entity_types());
        builder.build_context(mention).expect("chunk is known").local_context
    }

    #[test]
    fn first_sentence_has_no_previous() {
        let local = local_of(&[chunk()], &mention(Some(0)));
        assert_eq!(local.sentence_text, "Alice met Bob.");
        assert_eq!(local.prev_sentence_text, None);
        assert_eq!(local.next_sentence_text.as_deref(), Some("Bob paid Carol."));
        assert_eq!(local.chunk_text, TEXT);
    }

    #[test]
    fn middle_sentence_has_both_neighbours() {
        let local = local_of(&[chunk()], &mention(Some(1)));
        assert_eq!(local.sentence_text, "Bob paid Carol.");
        assert_eq!(local.prev_sentence_text.as_deref(), Some("Alice met Bob."));
        assert_eq!(local.next_sentence_text.as_deref(), Some("Carol left."));
    }

    #[test]
    fn last_sentence_has_no_next() {
        let local = local_of(&[chunk()], &mention(Some(2)));
        assert_eq!(local.sentence_text, "Carol left.");
        assert_eq!(local.prev_sentence_text.as_deref(), Some("Bob paid Carol."));
        assert_eq!(local.next_sentence_text, None);
    }

    #[test]
    fn missing_or_out_of_range_index_uses_chunk_text() {
        for index in [None, Some(3)] {
            let local = local_of(&[chunk()], &mention(index));
            assert_eq!(local.sentence_text, TEXT);
            assert_eq!(local.prev_sentence_text, None);
            assert_eq!(local.next_sentence_text, None);
        }
        let mut no_spans = chunk();
        no_spans.chunk.sentence_spans.clear();
        let local = local_of(&[no_spans], &mention(Some(0)));
        assert_eq!(local.sentence_text, TEXT);
    }

    #[test]
    fn spans_are_clamped_to_the_chunk() {
        let mut written = chunk();
        // Starts before the chunk, ends after it.
        written.chunk.sentence_spans = vec![Span::new(90, 114), Span::new(131, 500)];
        let local = local_of(&[written.clone()], &mention(Some(0)));
        assert_eq!(local.sentence_text, "Alice met Bob.");
        assert_eq!(local.next_sentence_text.as_deref(), Some("Carol left."));
        // Inverted span yields an empty sentence.
        written.chunk.sentence_spans = vec![Span::new(120, 110)];
        assert_eq!(local_of(&[written], &mention(Some(0))).sentence_text, "");
    }

    #[test]
    fn offsets_are_characters() {
        let text = "Café José. Ünïcode here.";
        let mut chunk = Chunk::new("doc-1", text, 0, 10, 10 + text.chars().count());
        chunk.sentence_spans = vec![Span::new(10, 20), Span::new(21, 34)];
        let written = ChunkWriteResult { chunk, helix_id: NodeId::new("17"), created: true };
        let local = local_of(&[written], &mention(Some(1)));
        assert_eq!(local.sentence_text, "Ünïcode here.");
        assert_eq!(local.prev_sentence_text.as_deref(), Some("Café José."));
    }

    #[test]
    fn python_slice_semantics() {
        assert_eq!(py_char_slice("abcdef", 6, 1, 3), "bc");
        assert_eq!(py_char_slice("abcdef", 6, 4, 2), "");
        assert_eq!(py_char_slice("abcdef", 6, 0, -2), "abcd");
        assert_eq!(py_char_slice("abcdef", 6, 0, -10), "");
        assert_eq!(py_char_slice("abcdef", 6, 7, 9), "");
    }

    #[test]
    fn missing_chunk_yields_none() {
        let metadata = Metadata::new();
        let chunks = [chunk()];
        let builder = ContextBuilder::new(&chunks, &metadata, entity_types());
        let mut unknown = mention(Some(0));
        unknown.chunk_id = Some("99".into());
        assert!(builder.build_context(&unknown).is_none());
        let mut no_chunk = mention(Some(0));
        no_chunk.chunk_id = None;
        assert!(builder.build_context(&no_chunk).is_none());
    }

    #[test]
    fn doc_context_merges_metadata_with_chunk_priority() {
        let doc_metadata: Metadata = serde_json::from_value(json!({
            "doc_type": "complaint",
            "case_id": "CASE-1",
            "jurisdiction": "CA",
            "party_names": ["Acme", "Bob"],
        }))
        .expect("object");
        let mut written = chunk();
        written.chunk.metadata =
            serde_json::from_value(json!({"case_id": "CASE-2"})).expect("object");
        let chunks = [written];
        let builder = ContextBuilder::new(&chunks, &doc_metadata, entity_types());
        let ctx = builder.build_context(&mention(Some(0))).expect("context");
        let mut expected = DocContext::new("doc-1", "complaint");
        expected.case_id = Some("CASE-2".into());
        expected.jurisdiction = Some("CA".into());
        expected.party_names = vec!["Acme".into(), "Bob".into()];
        assert_eq!(ctx.doc_context, expected);
        assert_eq!(ctx.normalized_key, None);
    }

    #[test]
    fn doc_context_defaults() {
        let doc_metadata: Metadata =
            serde_json::from_value(json!({"party_names": "Acme"})).expect("object");
        let chunks = [chunk()];
        let builder = ContextBuilder::new(&chunks, &doc_metadata, entity_types());
        let ctx = builder.build_context(&mention(Some(0))).expect("context");
        assert_eq!(ctx.doc_context, DocContext::new("doc-1", "unknown"));
    }

    #[test]
    fn span_label_is_normalised() {
        let types = entity_types();
        let metadata = Metadata::new();
        let chunks = [chunk()];
        let builder = ContextBuilder::new(&chunks, &metadata, types);
        for label in ["PERSON", "person", "", "SOMETHING_UNKNOWN"] {
            let mut m = mention(Some(0));
            m.entity_type = label.to_owned();
            let ctx = builder.build_context(&m).expect("context");
            assert_eq!(ctx.span_label, types.normalize_label(label));
            assert_eq!(ctx.mention, m);
        }
        let mut empty = mention(Some(0));
        empty.entity_type = String::new();
        let ctx = builder.build_context(&empty).expect("context");
        assert_eq!(ctx.span_label, types.default_entity_type());
    }
}
