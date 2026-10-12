//! RAGIndexStage against `MemoryStore` + `MockEmbedder` on `fixtures/input/*.jsonl`.

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use ce_core::{ChunkId, DocumentPayload, EntityId, FactId, IndexingResult, NodeId, SpanId};
use ce_models::{Embedder, ModelError, ModelKind, ModelRegistry};
use ce_pipeline::{PipelineError, PipelineInput, RagIndexStage, Stage};
use ce_store::graph::{CHUNK_OF_DOCUMENT, DOCUMENT, DOCUMENT_CHUNK, SPAN, SPAN_OF_CHUNK};
use ce_store::{
    ChunkNode, DocumentNode, EmbeddingWrite, EntityHit, EntityNode, FactHit, FactNode, GraphStore,
    MemoryStore, SpanNode, StoreError, StoredChunk, StoredDocument, StoredEntity, StoredFact,
    StoredSpan, VectorLabel,
};
use common::{char_slice, config_with, default_config, read_fixture, Harness, DIM};
use pretty_assertions::assert_eq;
use serde_json::json;

fn boundaries(result: &IndexingResult) -> Vec<(usize, usize)> {
    result.chunks.iter().map(|w| (w.chunk.start_char, w.chunk.end_char)).collect()
}

fn index_payload(harness: &Harness, document: DocumentPayload) -> IndexingResult {
    harness.index(PipelineInput::from(document)).0
}

#[test]
fn indexes_every_fixture() {
    let fixtures = common::jsonl_fixtures();
    assert!(fixtures.len() >= 5, "expected the five JSONL fixtures");
    for path in fixtures {
        let name = path.display().to_string();
        let harness = Harness::new(default_config());
        let (result, data) = harness.index(PipelineInput::Jsonl(path.clone()));
        assert!(data.index_persisted, "{name}");
        let text = data.doc_text.clone().expect("doc_text");
        let document = ce_ingest::read_first_record(&path).expect("fixture");
        assert_eq!(text, document.text, "{name}");

        let chunks = result.chunks.len();
        assert!(chunks >= 1, "{name}");
        // Sentence strategy: one sentence span per chunk; one embedding per chunk.
        assert_eq!(result.spans.len(), chunks, "{name}");
        assert_eq!(result.embeddings.count, chunks, "{name}");
        assert!(result.embeddings.errors.is_empty(), "{name}");
        assert!(result.document.created, "{name}");
        assert_eq!(
            result.timings.keys().collect::<Vec<_>>(),
            ["document", "chunking", "embeddings"]
        );

        assert_eq!(harness.count_label(DOCUMENT), 1, "{name}");
        assert_eq!(harness.count_label(DOCUMENT_CHUNK), chunks, "{name}");
        assert_eq!(harness.count_label(SPAN), chunks, "{name}");
        assert_eq!(harness.count_edges(CHUNK_OF_DOCUMENT), chunks, "{name}");
        assert_eq!(harness.count_edges(SPAN_OF_CHUNK), chunks, "{name}");
        assert_eq!(harness.embedder.calls(), 1, "{name}");

        for (sequence, write) in result.chunks.iter().enumerate() {
            let chunk = &write.chunk;
            assert_eq!(chunk.sequence_number, sequence);
            assert_eq!(chunk.text, char_slice(&text, chunk.start_char, chunk.end_char), "{name}");
            assert!(chunk.text.chars().count() <= 1500, "{name}");
            assert_eq!(chunk.helix_id.as_ref(), Some(&write.helix_id));
            let stored = harness.store.embedding(write.helix_id.as_str()).expect("chunk vector");
            assert_eq!(stored, harness.embedder.vector_for(&chunk.text), "{name}");
            let (label, node) = harness.store.node_snapshot(write.helix_id.as_str()).expect("node");
            assert_eq!(label, DOCUMENT_CHUNK);
            assert_eq!(node["StartChar"], json!(chunk.start_char));
            assert_eq!(node["EndChar"], json!(chunk.end_char));
            assert_eq!(node["SequenceNumber"], json!(sequence));
            assert_eq!(node["Text"], json!(chunk.text));
        }
        // Chunks cover every non-whitespace character, in order.
        let covered = |i: usize| boundaries(&result).iter().any(|(s, e)| *s <= i && i < *e);
        for (i, c) in text.chars().enumerate() {
            assert!(c.is_whitespace() || covered(i), "{name}: char {i} not covered");
        }
        assert!(boundaries(&result).windows(2).all(|w| w[0].0 < w[1].0 && w[0].1 < w[1].1));

        assert_eq!(data.helix_doc_id.as_ref(), Some(&result.document.helix_id));
        assert_eq!(data.doc_id.as_ref(), Some(&document.doc_id));
        assert!(data.embedder.is_some());
        let written = data.document.expect("document written back");
        assert_eq!(written.helix_id.as_ref(), Some(&result.document.helix_id));
    }
}

#[test]
fn chunk_and_span_counts_per_fixture() {
    let expected = [
        ("Mason_billy.jsonl", 1),
        ("Mason_intra.jsonl", 1),
        ("Mason_watson.jsonl", 1),
        ("green_valley_incident_report.jsonl", 7),
        ("lopez_v_green_valley_complaint.jsonl", 6),
    ];
    for (name, chunks) in expected {
        let harness = Harness::new(default_config());
        let result = index_payload(&harness, read_fixture(name));
        assert_eq!(result.chunks.len(), chunks, "{name}");
        assert_eq!(result.spans.len(), chunks, "{name}");
        assert_eq!(result.embeddings.count, chunks, "{name}");
    }
}

#[test]
fn small_document_is_one_trimmed_chunk() {
    let harness = Harness::new(default_config());
    let document = read_fixture("Mason_billy.jsonl");
    let result = index_payload(&harness, document.clone());
    let trimmed = document.text.trim_end();
    assert_eq!(boundaries(&result), [(0, trimmed.chars().count())]);
    assert_eq!(result.chunks[0].chunk.text, trimmed);
    let span = harness.store.node_snapshot(result.spans[0].span_id.as_str()).expect("span").1;
    assert_eq!(span["SurfaceText"], json!(trimmed));
    assert_eq!(span["SpanType"], json!("SENTENCE"));
    assert_eq!(span["ChunkId"], json!(result.chunks[0].helix_id.as_str()));
    let doc = harness.store.node_snapshot(result.document.helix_id.as_str()).expect("doc").1;
    assert_eq!(doc["Title"], json!("001"));
    assert_eq!(doc["DocType"], json!("UNKNOWN"));
    let metadata: serde_json::Value =
        serde_json::from_str(doc["Metadata"].as_str().expect("string")).expect("json");
    assert_eq!(metadata["case_id"], json!("mason_v_billy"));
    assert_eq!(metadata["db_id"], json!("depo_1"));
    assert_eq!(metadata["external_id"], json!("001"));
}

#[test]
fn boundaries_are_deterministic() {
    for name in ["green_valley_incident_report.jsonl", "lopez_v_green_valley_complaint.jsonl"] {
        let first = index_payload(&Harness::new(default_config()), read_fixture(name));
        let second = index_payload(&Harness::new(default_config()), read_fixture(name));
        assert_eq!(boundaries(&first), boundaries(&second), "{name}");
        let texts =
            |r: &IndexingResult| r.chunks.iter().map(|w| w.chunk.text.clone()).collect::<Vec<_>>();
        assert_eq!(texts(&first), texts(&second), "{name}");
    }
}

#[test]
fn sentence_chunks_overlap_by_at_most_chunk_overlap() {
    let harness = Harness::new(default_config());
    let result = index_payload(&harness, read_fixture("green_valley_incident_report.jsonl"));
    for pair in boundaries(&result).windows(2) {
        let overlap = pair[0].1.saturating_sub(pair[1].0);
        assert!(overlap <= 150, "overlap {overlap} between {pair:?}");
    }
}

#[test]
fn char_offsets_are_correct_for_non_ascii_text() {
    let harness = Harness::new(default_config());
    let document = read_fixture("green_valley_incident_report.jsonl");
    let text = document.text.clone();
    assert!(!text.is_ascii(), "fixture has non-ASCII quotes");
    let result = index_payload(&harness, document);

    let byte_offset = |chars: usize| text.char_indices().nth(chars).map_or(text.len(), |(b, _)| b);
    let shifted = result
        .chunks
        .iter()
        .filter(|w| byte_offset(w.chunk.start_char) != w.chunk.start_char)
        .count();
    assert!(shifted > 0, "some chunk starts after a multi-byte character");
    for write in &result.chunks {
        let chunk = &write.chunk;
        assert_eq!(chunk.text, char_slice(&text, chunk.start_char, chunk.end_char));
        assert_eq!(chunk.end_char - chunk.start_char, chunk.text.chars().count());
    }
    for span in &result.spans {
        let node = harness.store.node_snapshot(span.span_id.as_str()).expect("span").1;
        let (start, end) = (
            node["StartChar"].as_u64().expect("start") as usize,
            node["EndChar"].as_u64().expect("end") as usize,
        );
        assert_eq!(node["SurfaceText"], json!(char_slice(&text, start, end)));
    }
}

#[test]
fn recursive_strategy_partitions_and_extends_back_by_overlap() {
    let harness = Harness::new(common::recursive_config(300, 30));
    let document = read_fixture("green_valley_incident_report.jsonl");
    let text = document.text.clone();
    let total = text.chars().count();
    let result = index_payload(&harness, document);
    let bounds = boundaries(&result);
    assert!(bounds.len() > 20);
    assert_eq!(bounds[0].0, 0);
    assert_eq!(bounds.last().map(|b| b.1), Some(total));
    for pair in bounds.windows(2) {
        assert_eq!(pair[1].0, pair[0].1 - 30, "{pair:?}");
    }
    for write in &result.chunks {
        let chunk = &write.chunk;
        assert_eq!(chunk.text, char_slice(&text, chunk.start_char, chunk.end_char));
        assert_eq!(chunk.metadata["chonkie_chunker"], json!("recursive"));
        assert!(chunk.metadata["chonkie_token_count"].as_u64().expect("count") <= 300);
    }
    // Sentence spans: regex sentences inside each chunk, stripped, at document offsets.
    let spans: usize = result.chunks.iter().map(|w| w.chunk.sentence_spans.len()).sum();
    assert_eq!(result.spans.len(), spans);
    assert!(spans > bounds.len());
    for span in &result.spans {
        let node = harness.store.node_snapshot(span.span_id.as_str()).expect("span").1;
        let (start, end) = (
            node["StartChar"].as_u64().expect("start") as usize,
            node["EndChar"].as_u64().expect("end") as usize,
        );
        let surface = node["SurfaceText"].as_str().expect("text");
        assert_eq!(surface, char_slice(&text, start, end));
        assert_eq!(surface, surface.trim());
    }
    // Deterministic.
    let again = index_payload(
        &Harness::new(common::recursive_config(300, 30)),
        read_fixture("green_valley_incident_report.jsonl"),
    );
    assert_eq!(boundaries(&again), bounds);
}

#[test]
fn spans_are_skipped_when_disabled() {
    let harness = Harness::new(config_with(|_| {}, |c| c.include_sentence_spans = false));
    let result = index_payload(&harness, read_fixture("lopez_v_green_valley_complaint.jsonl"));
    assert!(result.spans.is_empty());
    assert_eq!(harness.count_label(SPAN), 0);
    assert_eq!(result.embeddings.count, result.chunks.len());
}

#[test]
fn offline_mode_writes_nothing_and_fabricates_ids() {
    let harness = Harness::new(config_with(|rag| rag.offline = true, |_| {}));
    let (result, data) = harness.index(PipelineInput::from(read_fixture("Mason_watson.jsonl")));
    assert_eq!(harness.store.node_count(), 0);
    assert_eq!(harness.embedder.calls(), 0);
    assert!(data.embedder.is_none());
    assert!(!data.index_persisted);
    assert_eq!(result.document.helix_id.as_str(), "doc:002");
    assert_eq!(result.chunks[0].helix_id.as_str(), "002:chunk-00000");
    assert_eq!(result.spans[0].span_id.as_str(), "002:sent-00000");
    assert_eq!(result.embeddings.count, 0);
    assert!(result.timings.is_empty());
    assert_eq!(data.helix_doc_id.map(ce_core::NodeId::into_inner), Some("doc:002".to_owned()));
}

#[test]
fn without_cache_every_run_creates_a_fresh_node() {
    let harness = Harness::new(default_config());
    assert!(harness.stage().document_cache().is_none(), "use_document_cache defaults to false");
    let first = index_payload(&harness, read_fixture("Mason_billy.jsonl"));
    let second = index_payload(&harness, read_fixture("Mason_billy.jsonl"));
    assert_ne!(first.document.helix_id, second.document.helix_id);
    assert!(first.document.created && second.document.created);
    assert_eq!(harness.count_label(DOCUMENT), 2);
}

#[test]
fn use_document_cache_from_config_enables_the_default_cache_file() {
    let harness = Harness::new(config_with(|rag| rag.use_document_cache = true, |_| {}));
    let stage = harness.stage();
    let cache = stage.document_cache().expect("cache enabled");
    assert_eq!(cache.path(), ce_pipeline::rag::default_document_cache_path());
    assert_eq!(cache.scope(), format!("{}#processing_v17", harness.store.store_id()));
    let offline = Harness::new(config_with(
        |rag| {
            rag.use_document_cache = true;
            rag.offline = true;
        },
        |_| {},
    ));
    assert!(offline.stage().document_cache().is_none());
}

#[test]
fn cache_reuses_the_node_only_when_identity_matches() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache_path = dir.path().join("rag_documents.json");
    let harness = Harness::new(default_config());
    let cached_stage = || harness.stage().with_document_cache_at(&cache_path);

    // Mason_billy and green_valley both use doc_id "001" under different cases.
    let billy =
        harness.run_stage(cached_stage(), PipelineInput::from(read_fixture("Mason_billy.jsonl")));
    let billy = billy.rag_index_result.expect("result");
    let again =
        harness.run_stage(cached_stage(), PipelineInput::from(read_fixture("Mason_billy.jsonl")));
    let again = again.rag_index_result.expect("result");
    let green = harness.run_stage(
        cached_stage(),
        PipelineInput::from(read_fixture("green_valley_incident_report.jsonl")),
    );
    let green = green.rag_index_result.expect("result");

    assert!(billy.document.created);
    assert!(!again.document.created);
    assert_eq!(again.document.helix_id, billy.document.helix_id);
    assert!(green.document.created);
    assert_ne!(green.document.helix_id, billy.document.helix_id);
    assert_eq!(harness.count_label(DOCUMENT), 2);

    // The reused document's chunks are replaced, not duplicated.
    assert_ne!(again.chunks[0].helix_id, billy.chunks[0].helix_id);
    assert!(again.chunks[0].created);
    assert_eq!(
        harness.store.get_document_chunks(&billy.document.helix_id).expect("chunks").len(),
        billy.chunks.len()
    );
    assert!(harness.store.node_snapshot(billy.chunks[0].helix_id.as_str()).is_none());

    let written = std::fs::read_to_string(&cache_path).expect("cache file");
    let scope = format!("{}#processing_v17", harness.store.store_id());
    assert!(written.contains(&format!("\"{scope}|depo_1|mason_v_billy|001\"")), "{written}");
    assert!(written.contains(&format!("\"{scope}|depo_1|lopez_v_green_valley|001\"")), "{written}");
}

#[test]
fn stale_cached_id_is_dropped_and_recreated() {
    let dir = tempfile::tempdir().expect("tempdir");
    let harness = Harness::new(default_config());
    let stage = harness.stage().with_document_cache_at(dir.path().join("cache.json"));
    let document = read_fixture("Mason_watson.jsonl");
    let cache = stage.document_cache().expect("cache");
    cache.record("002", &ce_core::NodeId::from("424242"), Some(&document.metadata));

    let data = harness.run_stage(stage, PipelineInput::from(document.clone()));
    let result = data.rag_index_result.expect("result");
    assert!(result.document.created);
    assert_ne!(result.document.helix_id.as_str(), "424242");
    let reopened = ce_pipeline::rag::DocumentStateStore::open(
        dir.path().join("cache.json"),
        format!("{}#processing_v17", harness.store.store_id()),
    );
    assert_eq!(reopened.lookup("002", Some(&document.metadata)), Some(result.document.helix_id));
}

#[test]
fn cache_misses_after_store_reset() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache_path = dir.path().join("cache.json");
    let harness = Harness::new(default_config());
    let first = harness.run_stage(
        harness.stage().with_document_cache_at(&cache_path),
        PipelineInput::from(read_fixture("Mason_watson.jsonl")),
    );
    harness.store.reset().expect("reset");
    let second = harness.run_stage(
        harness.stage().with_document_cache_at(&cache_path),
        PipelineInput::from(read_fixture("Mason_watson.jsonl")),
    );
    let (first, second) = (first.rag_index_result.expect("r"), second.rag_index_result.expect("r"));
    assert!(second.document.created);
    assert_ne!(first.document.helix_id, second.document.helix_id);
}

#[test]
fn explicit_helix_id_updates_the_node() {
    let harness = Harness::new(default_config());
    let first = index_payload(&harness, read_fixture("Mason_watson.jsonl"));
    let mut document = read_fixture("Mason_watson.jsonl");
    document.helix_id = Some(first.document.helix_id.clone());
    document.title = "Renamed".into();
    let second = index_payload(&harness, document);
    assert!(!second.document.created);
    assert_eq!(second.document.helix_id, first.document.helix_id);
    let node = harness.store.node_snapshot(first.document.helix_id.as_str()).expect("doc").1;
    assert_eq!(node["Title"], json!("Renamed"));

    // Without a cache an unknown explicit id is an error.
    let mut missing = read_fixture("Mason_watson.jsonl");
    missing.helix_id = Some("99999".into());
    let pipeline = harness.pipeline_with(harness.stage());
    let mut ctx = pipeline.new_context();
    let error = pipeline.run(&mut ctx, PipelineInput::from(missing)).expect_err("stale id");
    assert!(matches!(
        error,
        PipelineError::Stage { ref source, .. } if matches!(**source, PipelineError::Store(_))
    ));
}

#[derive(Debug)]
struct ShortEmbedder;

impl Embedder for ShortEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, ModelError> {
        Ok(texts.iter().skip(1).map(|_| vec![1.0; DIM]).collect())
    }
    fn dimension(&self) -> usize {
        DIM
    }
    fn model_name(&self) -> &str {
        "short"
    }
}

fn harness_with_embedder(
    embedder: Arc<dyn Embedder>,
) -> (Arc<MemoryStore>, RagIndexStage, ce_pipeline::StageBuildContext) {
    let config = default_config();
    let registry = ModelRegistry::from_config(&config);
    registry.insert_embedder(registry.key(ModelKind::Bge), embedder);
    let store = Arc::new(MemoryStore::new(DIM));
    let dyn_store: Arc<dyn GraphStore> = store.clone();
    let services =
        ce_pipeline::StageBuildContext::new(Arc::new(config), Arc::new(registry), dyn_store);
    let rag = Harness::new(default_config()).rag_config();
    let stage = RagIndexStage::new(&rag, &services).expect("stage");
    (store, stage, services)
}

#[test]
fn embedder_output_must_match_chunks_and_store() {
    let (_, stage, services) = harness_with_embedder(Arc::new(ShortEmbedder));
    let mut ctx = ce_pipeline::PipelineContext::new(&services);
    ctx.data.document = Some(read_fixture("Mason_billy.jsonl"));
    let error = stage.run(&mut ctx).expect_err("count mismatch");
    assert!(matches!(error, PipelineError::EmbeddingCount { embeddings: 0, chunks: 1 }), "{error}");
    assert!(ctx.data.document.is_some(), "document is kept on failure");

    let (_, stage, services) = harness_with_embedder(Arc::new(ce_models::MockEmbedder::new(8)));
    let mut ctx = ce_pipeline::PipelineContext::new(&services);
    ctx.data.document = Some(read_fixture("Mason_billy.jsonl"));
    let error = stage.run(&mut ctx).expect_err("dimension mismatch");
    assert!(
        matches!(error, PipelineError::EmbeddingDimension { actual: 8, expected: DIM }),
        "{error}"
    );
}

#[test]
fn missing_embedder_is_a_model_error() {
    let config = default_config();
    let store: Arc<dyn GraphStore> = Arc::new(MemoryStore::new(DIM));
    let services = ce_pipeline::StageBuildContext::new(
        Arc::new(config),
        Arc::new(ModelRegistry::empty("processing_v17")),
        store,
    );
    let rag = Harness::new(default_config()).rag_config();
    let stage = RagIndexStage::new(&rag, &services).expect("stage");
    let mut ctx = ce_pipeline::PipelineContext::new(&services);
    ctx.data.document = Some(read_fixture("Mason_billy.jsonl"));
    let error = stage.run(&mut ctx).expect_err("no embedder");
    assert!(matches!(error, PipelineError::Model(ModelError::UnknownModel { .. })), "{error}");
    let document = ctx.data.document.as_ref().expect("document is restored after an error");
    assert_eq!(document.doc_id.as_str(), "001");
    assert!(ctx.data.embedder.is_none() && ctx.data.rag_index_result.is_none());
}

#[test]
fn empty_document_does_not_resolve_the_embedder() {
    let config = default_config();
    let store: Arc<dyn GraphStore> = Arc::new(MemoryStore::new(DIM));
    let services = ce_pipeline::StageBuildContext::new(
        Arc::new(config),
        Arc::new(ModelRegistry::empty("processing_v17")),
        store,
    );
    let rag = Harness::new(default_config()).rag_config();
    let stage = RagIndexStage::new(&rag, &services).expect("stage");
    let mut ctx = ce_pipeline::PipelineContext::new(&services);
    ctx.data.document = Some(DocumentPayload::new("empty", ""));
    stage.run(&mut ctx).expect("an empty document needs no embedder");
    let result = ctx.data.rag_index_result.as_ref().expect("result");
    assert!(result.chunks.is_empty() && result.spans.is_empty());
    assert_eq!(result.embeddings.count, 0);
    assert!(ctx.data.embedder.is_none());
    assert!(ctx.data.document.as_ref().is_some_and(|doc| doc.helix_id.is_some()));
}

/// `count` numbered sentences; with 40-character recursive chunks each is a chunk of its own.
fn numbered_sentences(count: usize) -> String {
    (0..count).map(|i| format!("Sentence number {i} is here.")).collect::<Vec<_>>().join(" ")
}

#[test]
fn reindexing_a_cached_document_replaces_chunks_and_spans() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache_path = dir.path().join("cache.json");
    let harness = Harness::new(common::recursive_config(40, 0));
    let cached_stage = || harness.stage().with_document_cache_at(&cache_path);
    let document = |sentences| DocumentPayload::new("doc-1", numbered_sentences(sentences));

    let first = harness.run_stage(cached_stage(), PipelineInput::from(document(10)));
    let first = first.rag_index_result.expect("result");
    assert_eq!(first.chunks.len(), 10);
    assert!(!first.spans.is_empty());

    let second = harness.run_stage(cached_stage(), PipelineInput::from(document(6)));
    let second = second.rag_index_result.expect("result");
    assert!(!second.document.created);
    assert_eq!(second.document.helix_id, first.document.helix_id);
    assert_eq!(second.chunks.len(), 6);

    let stored = harness.store.get_document_chunks(&first.document.helix_id).expect("chunks");
    assert_eq!(stored.len(), 6);
    let mut sequences: Vec<i64> = stored.iter().map(|c| c.node.sequence_number).collect();
    sequences.sort_unstable();
    assert_eq!(sequences, [0, 1, 2, 3, 4, 5]);
    assert_eq!(harness.count_label(DOCUMENT), 1);
    assert_eq!(harness.count_label(DOCUMENT_CHUNK), 6);
    assert_eq!(harness.count_edges(CHUNK_OF_DOCUMENT), 6);
    assert_eq!(harness.count_label(SPAN), second.spans.len(), "one set of sentence spans");
    assert_eq!(harness.count_edges(SPAN_OF_CHUNK), second.spans.len());
    assert!(stored.iter().all(|chunk| harness.store.embedding(chunk.id.as_str()).is_some()));
}

/// A `MemoryStore` whose `update_document` fails with a query error while `fail_updates` is
/// set.
#[derive(Debug)]
struct FailingUpdateStore {
    inner: MemoryStore,
    fail_updates: AtomicBool,
}

impl GraphStore for FailingUpdateStore {
    fn store_id(&self) -> String {
        self.inner.store_id()
    }
    fn embedding_dim(&self) -> usize {
        self.inner.embedding_dim()
    }
    fn reset(&self) -> ce_store::Result<()> {
        self.inner.reset()
    }
    fn create_document(&self, document: &DocumentNode) -> ce_store::Result<StoredDocument> {
        self.inner.create_document(document)
    }
    fn update_document(
        &self,
        doc_id: &NodeId,
        document: &DocumentNode,
    ) -> ce_store::Result<StoredDocument> {
        if self.fail_updates.load(Ordering::SeqCst) {
            return Err(StoreError::Query {
                query: "UpdateDocument".into(),
                message: "transient failure".into(),
            });
        }
        self.inner.update_document(doc_id, document)
    }
    fn get_document_chunks(&self, doc_id: &NodeId) -> ce_store::Result<Vec<StoredChunk>> {
        self.inner.get_document_chunks(doc_id)
    }
    fn create_document_chunk(
        &self,
        doc_id: &NodeId,
        chunk: &ChunkNode,
    ) -> ce_store::Result<StoredChunk> {
        self.inner.create_document_chunk(doc_id, chunk)
    }
    fn update_document_chunk(
        &self,
        chunk_id: &ChunkId,
        chunk: &ChunkNode,
    ) -> ce_store::Result<StoredChunk> {
        self.inner.update_document_chunk(chunk_id, chunk)
    }
    fn create_chunk_embedding(
        &self,
        chunk_id: &ChunkId,
        embedding: &[f32],
    ) -> ce_store::Result<()> {
        self.inner.create_chunk_embedding(chunk_id, embedding)
    }
    fn create_span_for_chunk(&self, span: &SpanNode) -> ce_store::Result<StoredSpan> {
        self.inner.create_span_for_chunk(span)
    }
    fn create_entity(&self, entity: &EntityNode) -> ce_store::Result<StoredEntity> {
        self.inner.create_entity(entity)
    }
    fn create_entity_embedding(
        &self,
        entity_id: &EntityId,
        embedding: &[f32],
    ) -> ce_store::Result<()> {
        self.inner.create_entity_embedding(entity_id, embedding)
    }
    fn find_entity_by_name(&self, name: &str) -> ce_store::Result<Vec<EntityHit>> {
        self.inner.find_entity_by_name(name)
    }
    fn find_similar_entities(
        &self,
        embedding: &[f32],
        limit: usize,
    ) -> ce_store::Result<Vec<EntityHit>> {
        self.inner.find_similar_entities(embedding, limit)
    }
    fn get_all_entities(&self, limit: usize) -> ce_store::Result<Vec<StoredEntity>> {
        self.inner.get_all_entities(limit)
    }
    fn link_span_to_entity(
        &self,
        span_id: &SpanId,
        entity_id: &EntityId,
        confidence: f64,
        method: &str,
    ) -> ce_store::Result<()> {
        self.inner.link_span_to_entity(span_id, entity_id, confidence, method)
    }
    fn create_fact(&self, fact: &FactNode) -> ce_store::Result<StoredFact> {
        self.inner.create_fact(fact)
    }
    fn create_fact_embedding(&self, fact_id: &FactId, embedding: &[f32]) -> ce_store::Result<()> {
        self.inner.create_fact_embedding(fact_id, embedding)
    }
    fn link_fact_to_document(
        &self,
        fact_id: &FactId,
        doc_id: &NodeId,
        page_number: i64,
        section: &str,
    ) -> ce_store::Result<()> {
        self.inner.link_fact_to_document(fact_id, doc_id, page_number, section)
    }
    fn link_fact_to_entity(
        &self,
        fact_id: &FactId,
        entity_id: &EntityId,
        role: &str,
    ) -> ce_store::Result<()> {
        self.inner.link_fact_to_entity(fact_id, entity_id, role)
    }
    fn link_fact_to_span(
        &self,
        fact_id: &FactId,
        span_id: &SpanId,
        role: &str,
        confidence: f64,
    ) -> ce_store::Result<()> {
        self.inner.link_fact_to_span(fact_id, span_id, role, confidence)
    }
    fn get_entity_facts(&self, entity_id: &EntityId) -> ce_store::Result<Vec<StoredFact>> {
        self.inner.get_entity_facts(entity_id)
    }
    fn get_fact_entities(&self, fact_id: &FactId) -> ce_store::Result<Vec<EntityId>> {
        self.inner.get_fact_entities(fact_id)
    }
    fn find_facts_by_text(&self, keywords: &str, limit: usize) -> ce_store::Result<Vec<FactHit>> {
        self.inner.find_facts_by_text(keywords, limit)
    }
    fn find_similar_facts(
        &self,
        embedding: &[f32],
        limit: usize,
    ) -> ce_store::Result<Vec<FactHit>> {
        self.inner.find_similar_facts(embedding, limit)
    }
    fn create_chunks_batch(
        &self,
        doc_id: &NodeId,
        chunks: &[ChunkNode],
    ) -> ce_store::Result<Vec<StoredChunk>> {
        self.inner.create_chunks_batch(doc_id, chunks)
    }
    fn delete_document_chunks(&self, doc_id: &NodeId) -> ce_store::Result<usize> {
        self.inner.delete_document_chunks(doc_id)
    }
    fn create_spans_batch(&self, spans: &[SpanNode]) -> ce_store::Result<Vec<StoredSpan>> {
        self.inner.create_spans_batch(spans)
    }
    fn create_embeddings_batch(
        &self,
        label: VectorLabel,
        items: &[EmbeddingWrite<'_>],
    ) -> ce_store::Result<usize> {
        self.inner.create_embeddings_batch(label, items)
    }
}

#[test]
fn non_not_found_update_errors_propagate_without_a_new_document() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache_path = dir.path().join("cache.json");
    let store = Arc::new(FailingUpdateStore {
        inner: MemoryStore::new(DIM),
        fail_updates: AtomicBool::new(false),
    });
    let config = default_config();
    let registry = ModelRegistry::from_config(&config);
    registry
        .insert_embedder(registry.key(ModelKind::Bge), Arc::new(ce_models::MockEmbedder::new(DIM)));
    let dyn_store: Arc<dyn GraphStore> = store.clone();
    let services =
        ce_pipeline::StageBuildContext::new(Arc::new(config), Arc::new(registry), dyn_store);
    let rag = Harness::new(default_config()).rag_config();
    let stage =
        || RagIndexStage::new(&rag, &services).expect("stage").with_document_cache_at(&cache_path);

    let mut ctx = ce_pipeline::PipelineContext::new(&services);
    ctx.data.document = Some(read_fixture("Mason_watson.jsonl"));
    stage().run(&mut ctx).expect("first run creates the document");
    let first = ctx.data.helix_doc_id.clone().expect("doc id");

    store.fail_updates.store(true, Ordering::SeqCst);
    let mut ctx = ce_pipeline::PipelineContext::new(&services);
    ctx.data.document = Some(read_fixture("Mason_watson.jsonl"));
    let error = stage().run(&mut ctx).expect_err("update fails");
    assert!(matches!(error, PipelineError::Store(StoreError::Query { .. })), "{error}");
    assert!(ctx.data.document.is_some(), "document is restored after an error");
    let documents = (0..5000_u64)
        .filter_map(|id| store.inner.node_snapshot(&id.to_string()))
        .filter(|(label, _)| *label == DOCUMENT)
        .count();
    assert_eq!(documents, 1, "no second Document node");
    let document = read_fixture("Mason_watson.jsonl");
    let reopened = stage();
    let cached = reopened.document_cache().expect("cache").lookup("002", Some(&document.metadata));
    assert_eq!(cached, Some(first), "the cached id is kept");
}

#[test]
fn stage_without_document_reports_missing_context() {
    let harness = Harness::new(default_config());
    let mut ctx = ce_pipeline::PipelineContext::new(&harness.services);
    let error = harness.stage().run(&mut ctx).expect_err("no document");
    assert!(matches!(error, PipelineError::MissingContext { key: "document", .. }), "{error}");
}
