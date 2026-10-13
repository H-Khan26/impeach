//! Chunk parity: the shipped `sentence` chunking (chunk_size 1500, overlap 150) against the
//! Python prototype's Chonkie chunks in `fixtures/reference/rag_index/<scenario>.json`.

mod common;
mod parity_a;

use impeach_pipeline::rag::Chunker;
use impeach_pipeline::PipelineInput;
use parity_a::{char_slice, input_document, reference_docs, PyRagDoc, SCENARIOS};
use pretty_assertions::assert_eq;

/// `(sequence, start, end, sentence spans)` of each chunk.
type ChunkRow = (usize, usize, usize, Vec<(usize, usize)>);

fn python_rows(doc: &PyRagDoc) -> Vec<ChunkRow> {
    doc.chunks
        .iter()
        .map(|c| {
            let spans = c.sentence_spans.iter().map(|s| (s.start, s.end)).collect();
            (c.sequence_number, c.start_char, c.end_char, spans)
        })
        .collect()
}

#[test]
fn sentence_chunks_match_chonkie() {
    let config = common::default_config();
    let chunking = &config.rag.chunking;
    assert_eq!((chunking.chunk_size, chunking.chunk_overlap), (1500, 150));
    let chunker = Chunker::new(chunking).expect("chunker");
    let mut compared = 0;
    for scenario in SCENARIOS {
        for doc in reference_docs::<PyRagDoc>("rag_index", scenario) {
            let document = input_document(&doc.input);
            let result = chunker.chunk(&document);
            let rust: Vec<ChunkRow> = result
                .chunks
                .iter()
                .map(|c| {
                    let spans = c.sentence_spans.iter().map(|s| (s.start, s.end)).collect();
                    (c.sequence_number, c.start_char, c.end_char, spans)
                })
                .collect();
            assert_eq!(rust, python_rows(&doc), "{scenario}/{}", doc.input);
            for chunk in &result.chunks {
                let expected = char_slice(&document.text, chunk.start_char, chunk.end_char);
                assert_eq!(chunk.text, expected, "{scenario}/{}", doc.input);
            }
            assert_eq!(result.sentence_spans.len(), doc.span_count, "{scenario}/{}", doc.input);
            compared += doc.chunks.len();
        }
    }
    assert_eq!(compared, 15, "7 + 6 Green Valley chunks and one per Mason document");
}

#[test]
fn rag_index_stage_writes_the_python_chunk_layout() {
    for scenario in SCENARIOS {
        for doc in reference_docs::<PyRagDoc>("rag_index", scenario) {
            let harness = common::Harness::new(common::default_config());
            let path = parity_a::fixtures_root().join("input").join(&doc.input);
            let (result, _) = harness.index(PipelineInput::Jsonl(path));
            let rust: Vec<ChunkRow> = result
                .chunks
                .iter()
                .map(|w| {
                    let c = &w.chunk;
                    let spans = c.sentence_spans.iter().map(|s| (s.start, s.end)).collect();
                    (c.sequence_number, c.start_char, c.end_char, spans)
                })
                .collect();
            assert_eq!(rust, python_rows(&doc), "{scenario}/{}", doc.input);
            assert_eq!(result.spans.len(), doc.span_count, "{scenario}/{}", doc.input);
            assert_eq!(result.embeddings.count, doc.embedding_count, "{scenario}/{}", doc.input);
        }
    }
}
