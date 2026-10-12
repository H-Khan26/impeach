//! Text chunking (Python `services/rag/chunkers/{sentence,recursive}.py`).
//!
//! Both strategies split with `text-splitter`'s [`TextSplitter`] sized in characters (Unicode
//! scalar values, Python `len`). Boundaries differ from the Chonkie chunkers of the Python
//! prototype but are deterministic. Every offset is a character offset into the document text.
//!
//! # Chunk-boundary rule
//!
//! `TextSplitter` splits the text at the highest semantic level whose pieces fit in
//! `chunk_size` characters and then merges neighbouring pieces greedily, left to right, while
//! the merged chunk still fits. Levels, highest first: runs of line breaks (longer runs rank
//! higher), Unicode sentence boundaries (UAX #29), Unicode word boundaries, grapheme clusters,
//! characters. A chunk never crosses a boundary of a higher level than the one it was merged at.
//!
//! - `sentence`: chunks are trimmed of surrounding whitespace, at most `chunk_size` characters
//!   long, and consecutive chunks overlap by up to `chunk_overlap` characters of whole pieces
//!   (`chunk_overlap < chunk_size` is required). Each chunk is also its own sentence span, as
//!   in the Python sentence chunker.
//! - `recursive`: the splitter runs untrimmed and without overlap, so the pieces partition the
//!   text exactly. Every chunk after the first is then extended backwards by `chunk_overlap`
//!   characters (clamped at 0) with its end unchanged, as in the Python recursive chunker.
//!   Sentence spans are the matches of `[^.!?]+[.!?]?` inside each chunk, stripped of
//!   whitespace, skipping empty ones.

use ce_config::pipeline::{ChunkingConfig, ChunkingStrategy};
use ce_core::{Chunk, ChunkingResult, DocumentPayload, Metadata, SentenceSpan, Span};
use indexmap::IndexMap;
use serde_json::Value;
use text_splitter::{ChunkConfig, TextSplitter};

use crate::error::{PipelineError, Result};

/// A configured chunker.
#[derive(Debug, Clone)]
pub struct Chunker {
    strategy: ChunkingStrategy,
    chunk_size: usize,
    chunk_overlap: usize,
    include_sentence_spans: bool,
}

impl Chunker {
    /// A chunker for the `rag.chunking` block.
    ///
    /// # Errors
    /// [`PipelineError::InvalidStageConfig`] when `chunk_size` is 0, or for the `sentence`
    /// strategy when `chunk_overlap >= chunk_size` (Chonkie rejects that too).
    pub fn new(config: &ChunkingConfig) -> Result<Self> {
        let invalid = |message: String| PipelineError::InvalidStageConfig {
            stage: "rag.chunking".to_owned(),
            message,
        };
        if config.chunk_size == 0 {
            return Err(invalid("chunk_size must be > 0".to_owned()));
        }
        if config.strategy == ChunkingStrategy::Sentence
            && config.chunk_overlap >= config.chunk_size
        {
            return Err(invalid(format!(
                "chunk_overlap ({}) must be smaller than chunk_size ({}) for the sentence strategy",
                config.chunk_overlap, config.chunk_size
            )));
        }
        Ok(Self {
            strategy: config.strategy,
            chunk_size: config.chunk_size,
            chunk_overlap: config.chunk_overlap,
            include_sentence_spans: config.include_sentence_spans,
        })
    }

    /// Whether sentence spans are emitted and written.
    #[must_use]
    pub fn include_sentence_spans(&self) -> bool {
        self.include_sentence_spans
    }

    /// Chunks `document.text`.
    #[must_use]
    pub fn chunk(&self, document: &DocumentPayload) -> ChunkingResult {
        let mut result = match self.strategy {
            ChunkingStrategy::Sentence => self.sentence_chunks(document),
            ChunkingStrategy::Recursive => {
                let pieces = recursive_pieces(&document.text, self.chunk_size);
                self.recursive_chunks(document, &pieces)
            }
        };
        set_neighbors(&mut result.chunks);
        result
    }

    fn sentence_chunks(&self, document: &DocumentPayload) -> ChunkingResult {
        let config = ChunkConfig::new(self.chunk_size)
            .with_overlap(self.chunk_overlap)
            .expect("Chunker::new checks chunk_overlap < chunk_size for the sentence strategy")
            .with_trim(true);
        let splitter = TextSplitter::new(config);
        let mut result = ChunkingResult::default();
        for (sequence, piece) in splitter.chunk_char_indices(&document.text).enumerate() {
            let start = piece.char_offset;
            let length = piece.chunk.chars().count();
            let end = start + length;
            let metadata = chunk_metadata(document, sequence, "sentence", None, length);
            let mut chunk = Chunk::new(document.doc_id.clone(), piece.chunk, sequence, start, end);
            chunk.sentence_spans = vec![Span::new(start, end)];
            if self.include_sentence_spans {
                result.sentence_spans.push(SentenceSpan {
                    chunk_sequence: sequence,
                    start_char: start,
                    end_char: end,
                    text: piece.chunk.to_owned(),
                    metadata: span_metadata(sequence, &metadata),
                });
            }
            chunk.metadata = metadata;
            result.chunks.push(chunk);
        }
        result
    }

    /// Builds recursive chunks from `pieces`, the `(start, end)` character ranges of the
    /// untrimmed splitter output.
    fn recursive_chunks(
        &self,
        document: &DocumentPayload,
        pieces: &[(usize, usize)],
    ) -> ChunkingResult {
        let text = &document.text;
        let byte_at = char_byte_offsets(text);
        let total_chars = byte_at.len() - 1;
        let mut result = ChunkingResult::default();
        for (sequence, &(piece_start, piece_end)) in pieces.iter().enumerate() {
            let mut start = piece_start.min(total_chars);
            if sequence > 0 && self.chunk_overlap > 0 {
                start = start.saturating_sub(self.chunk_overlap);
            }
            let end = piece_end.max(start).min(total_chars);
            let chunk_text = &text[byte_at[start]..byte_at[end]];
            let metadata = chunk_metadata(
                document,
                sequence,
                "recursive",
                Some(Value::Null),
                piece_end.saturating_sub(piece_start),
            );
            let mut chunk = Chunk::new(document.doc_id.clone(), chunk_text, sequence, start, end);
            if self.include_sentence_spans {
                let spans = sentence_spans(sequence, chunk_text, start, &metadata);
                chunk.sentence_spans =
                    spans.iter().map(|s| Span::new(s.start_char, s.end_char)).collect();
                result.sentence_spans.extend(spans);
            }
            chunk.metadata = metadata;
            result.chunks.push(chunk);
        }
        result
    }
}

/// Character ranges of the untrimmed, non-overlapping splitter output; they partition `text`.
fn recursive_pieces(text: &str, chunk_size: usize) -> Vec<(usize, usize)> {
    let splitter = TextSplitter::new(ChunkConfig::new(chunk_size).with_trim(false));
    splitter
        .chunk_char_indices(text)
        .map(|piece| (piece.char_offset, piece.char_offset + piece.chunk.chars().count()))
        .collect()
}

/// Byte offset of every character of `text`, plus `text.len()` at the end.
fn char_byte_offsets(text: &str) -> Vec<usize> {
    text.char_indices().map(|(byte, _)| byte).chain(std::iter::once(text.len())).collect()
}

/// Chunk metadata in Python key order; document metadata overrides on key clashes.
fn chunk_metadata(
    document: &DocumentPayload,
    sequence: usize,
    chunker: &str,
    chunk_id: Option<Value>,
    token_count: usize,
) -> Metadata {
    let mut metadata = Metadata::new();
    metadata.insert("chunk_index".into(), sequence.into());
    metadata.insert(
        "source_file".into(),
        document.file_path.as_ref().map_or(Value::Null, |p| Value::String(p.clone())),
    );
    metadata.insert("doc_type".into(), document.doc_type.clone().into());
    metadata.insert("chonkie_chunker".into(), chunker.into());
    if let Some(chunk_id) = chunk_id {
        metadata.insert("chonkie_chunk_id".into(), chunk_id);
    }
    metadata.insert("chonkie_token_count".into(), token_count.into());
    for (key, value) in &document.metadata {
        metadata.insert(key.clone(), value.clone());
    }
    metadata
}

/// `{"chunk_sequence": sequence, **chunk_metadata}`.
fn span_metadata(sequence: usize, chunk_metadata: &Metadata) -> Metadata {
    let mut metadata = Metadata::new();
    metadata.insert("chunk_sequence".into(), sequence.into());
    for (key, value) in chunk_metadata {
        metadata.insert(key.clone(), value.clone());
    }
    metadata
}

fn is_sentence_end(c: char) -> bool {
    matches!(c, '.' | '!' | '?')
}

/// Sentence spans of one recursive chunk: matches of `[^.!?]+[.!?]?`, stripped.
fn sentence_spans(
    sequence: usize,
    chunk_text: &str,
    chunk_start: usize,
    chunk_metadata: &Metadata,
) -> Vec<SentenceSpan> {
    let chars: Vec<char> = chunk_text.chars().collect();
    let mut spans = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if is_sentence_end(chars[i]) {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < chars.len() && !is_sentence_end(chars[j]) {
            j += 1;
        }
        if j < chars.len() {
            j += 1;
        }
        let matched = &chars[i..j];
        let leading =
            matched.iter().take_while(|c| ce_text::normalize::is_py_whitespace(**c)).count();
        let trailing =
            matched.iter().rev().take_while(|c| ce_text::normalize::is_py_whitespace(**c)).count();
        if leading < matched.len() {
            let rel_start = i + leading;
            let rel_end = j - trailing;
            spans.push(SentenceSpan {
                chunk_sequence: sequence,
                start_char: chunk_start + rel_start,
                end_char: chunk_start + rel_end,
                text: chars[rel_start..rel_end].iter().collect(),
                metadata: span_metadata(sequence, chunk_metadata),
            });
        }
        i = j;
    }
    spans
}

/// `prev_chunk_seq` / `next_chunk_seq` neighbours.
fn set_neighbors(chunks: &mut [Chunk]) {
    let count = chunks.len();
    for (index, chunk) in chunks.iter_mut().enumerate() {
        let mut neighbors = IndexMap::new();
        if index > 0 {
            neighbors.insert("prev_chunk_seq".to_owned(), as_i64(index - 1));
        }
        if index + 1 < count {
            neighbors.insert("next_chunk_seq".to_owned(), as_i64(index + 1));
        }
        chunk.neighbors = neighbors;
    }
}

fn as_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    fn config(
        strategy: ChunkingStrategy,
        size: usize,
        overlap: usize,
        spans: bool,
    ) -> ChunkingConfig {
        ChunkingConfig {
            strategy,
            chunk_size: size,
            chunk_overlap: overlap,
            include_sentence_spans: spans,
        }
    }

    fn recursive(size: usize, overlap: usize, spans: bool) -> Chunker {
        Chunker::new(&config(ChunkingStrategy::Recursive, size, overlap, spans)).expect("chunker")
    }

    fn with_neighbors(
        chunker: &Chunker,
        doc: &DocumentPayload,
        pieces: &[(usize, usize)],
    ) -> ChunkingResult {
        let mut result = chunker.recursive_chunks(doc, pieces);
        set_neighbors(&mut result.chunks);
        result
    }

    #[test]
    fn recursive_respects_base_offsets() {
        let doc = DocumentPayload::new("doc-1", "abcdefgh");
        let result = with_neighbors(&recursive(4, 0, false), &doc, &[(0, 4), (4, 8)]);
        let got: Vec<_> =
            result.chunks.iter().map(|c| (c.start_char, c.end_char, c.text.as_str())).collect();
        assert_eq!(got, [(0, 4, "abcd"), (4, 8, "efgh")]);
        assert!(result.chunks.iter().all(|c| c.doc_id.as_str() == "doc-1"));
    }

    #[test]
    fn recursive_applies_overlap() {
        let doc = DocumentPayload::new("doc-2", "abcdefghij");
        let result = with_neighbors(&recursive(6, 2, false), &doc, &[(0, 6), (6, 10)]);
        assert_eq!(result.chunks.len(), 2);
        assert_eq!((result.chunks[0].start_char, result.chunks[0].end_char), (0, 6));
        assert_eq!(result.chunks[0].text, "abcdef");
        assert_eq!((result.chunks[1].start_char, result.chunks[1].end_char), (4, 10));
        assert_eq!(result.chunks[1].text, "efghij");
    }

    #[test]
    fn recursive_emits_sentence_spans() {
        let text = "Hello world! Another day.";
        let doc = DocumentPayload::new("doc-3", text);
        let result = with_neighbors(&recursive(32, 0, true), &doc, &[(0, text.len())]);
        assert_eq!(result.chunks.len(), 1);
        let spans = &result.sentence_spans;
        let texts: Vec<_> = spans.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(texts, ["Hello world!", "Another day."]);
        assert_eq!((spans[0].start_char, spans[0].end_char), (0, "Hello world!".len()));
        assert_eq!(spans[1].start_char, text.find("Another").expect("present"));
        assert_eq!(spans[1].end_char, text.trim_end().len());
        let own: Vec<_> = spans.iter().map(|s| Span::new(s.start_char, s.end_char)).collect();
        assert_eq!(result.chunks[0].sentence_spans, own);
    }

    #[test]
    fn recursive_populates_neighbors_and_metadata() {
        let mut doc = DocumentPayload::new("doc-4", "abcdefghi");
        doc.metadata.insert("case_id".into(), json!("case-1"));
        let chunks =
            with_neighbors(&recursive(3, 0, false), &doc, &[(0, 3), (3, 6), (6, 9)]).chunks;
        let neighbors: Vec<Vec<(&str, i64)>> = chunks
            .iter()
            .map(|c| c.neighbors.iter().map(|(k, v)| (k.as_str(), *v)).collect())
            .collect();
        assert_eq!(
            neighbors,
            [
                vec![("next_chunk_seq", 1)],
                vec![("prev_chunk_seq", 0), ("next_chunk_seq", 2)],
                vec![("prev_chunk_seq", 1)]
            ]
        );
        assert!(chunks.iter().all(|c| c.metadata["case_id"] == json!("case-1")));
        let indexes: Vec<_> = chunks.iter().map(|c| c.metadata["chunk_index"].clone()).collect();
        assert_eq!(indexes, [json!(0), json!(1), json!(2)]);
        let keys: Vec<_> = chunks[0].metadata.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "chunk_index",
                "source_file",
                "doc_type",
                "chonkie_chunker",
                "chonkie_chunk_id",
                "chonkie_token_count",
                "case_id"
            ]
        );
    }

    #[test]
    fn sentence_regex_skips_bare_delimiters_and_whitespace() {
        let meta = Metadata::new();
        let spans = sentence_spans(0, "..  Hi there?!  \n Ok", 10, &meta);
        let got: Vec<_> =
            spans.iter().map(|s| (s.start_char, s.end_char, s.text.as_str())).collect();
        assert_eq!(got, [(14, 23, "Hi there?"), (28, 30, "Ok")]);
    }

    #[test]
    fn recursive_pieces_partition_the_text() {
        let text = "Première phrase. Deuxième phrase!\n\nTroisième — et dernière — phrase? Fin.";
        let pieces = recursive_pieces(text, 20);
        assert!(pieces.len() > 1);
        assert_eq!(pieces[0].0, 0);
        assert_eq!(pieces.last().map(|p| p.1), Some(text.chars().count()));
        for pair in pieces.windows(2) {
            assert_eq!(pair[0].1, pair[1].0);
        }
        assert!(pieces.iter().all(|(s, e)| e - s <= 20));
    }

    #[test]
    fn recursive_overlap_uses_char_offsets() {
        let text = "Ünïcödé text. Àccents everywhere. Ënd of tëxt here.";
        let doc = DocumentPayload::new("d", text);
        let result = recursive(20, 5, true).chunk(&doc);
        let chars: Vec<char> = text.chars().collect();
        for chunk in &result.chunks {
            let expected: String = chars[chunk.start_char..chunk.end_char].iter().collect();
            assert_eq!(chunk.text, expected);
        }
        for span in &result.sentence_spans {
            let expected: String = chars[span.start_char..span.end_char].iter().collect();
            assert_eq!(span.text, expected);
        }
        for pair in result.chunks.windows(2) {
            assert_eq!(pair[1].start_char, pair[0].end_char - 5);
        }
    }

    #[test]
    fn sentence_strategy_packs_up_to_chunk_size() {
        let text = "One sentence here. Two sentence here. Three sentence here. Four.";
        let doc = DocumentPayload::new("d", text);
        let chunker =
            Chunker::new(&config(ChunkingStrategy::Sentence, 40, 0, true)).expect("chunker");
        let result = chunker.chunk(&doc);
        let texts: Vec<_> = result.chunks.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, ["One sentence here. Two sentence here.", "Three sentence here. Four."]);
        assert_eq!(result.sentence_spans.len(), 2);
        assert_eq!(result.chunks[1].start_char, text.find("Three").expect("present"));
        assert_eq!(result.chunks[1].sentence_spans, [Span::new(38, 64)]);
        assert_eq!(result.sentence_spans[1].metadata["chunk_sequence"], json!(1));
        assert_eq!(result.chunks[0].metadata["chonkie_chunker"], json!("sentence"));
    }

    #[test]
    fn sentence_strategy_rejects_overlap_not_below_size() {
        assert!(Chunker::new(&config(ChunkingStrategy::Sentence, 10, 10, true)).is_err());
        assert!(Chunker::new(&config(ChunkingStrategy::Recursive, 10, 10, true)).is_ok());
        assert!(Chunker::new(&config(ChunkingStrategy::Recursive, 0, 0, true)).is_err());
    }

    #[test]
    fn empty_and_blank_text_give_no_chunks() {
        let chunker =
            Chunker::new(&config(ChunkingStrategy::Sentence, 10, 2, true)).expect("chunker");
        assert!(chunker.chunk(&DocumentPayload::new("d", "")).chunks.is_empty());
        assert!(chunker.chunk(&DocumentPayload::new("d", "   \n ")).chunks.is_empty());
        assert!(recursive(10, 2, true).chunk(&DocumentPayload::new("d", "")).chunks.is_empty());
    }
}
