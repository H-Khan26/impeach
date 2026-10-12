//! RAGIndexStage against `MemoryStore` + `MockEmbedder` on `fixtures/input/*.jsonl`.

mod common;

use std::sync::Arc;

use ce_core::{DocumentPayload, IndexingResult};
use ce_models::{Embedder, ModelError, ModelKind, ModelRegistry};
use ce_pipeline::{PipelineError, PipelineInput, RagIndexStage, Stage};
use ce_store::graph::{CHUNK_OF_DOCUMENT, DOCUMENT, DOCUMENT_CHUNK, SPAN, SPAN_OF_CHUNK};
use ce_store::{GraphStore, MemoryStore};
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

    // The reused document's chunk is updated in place, not duplicated.
    assert_eq!(again.chunks[0].helix_id, billy.chunks[0].helix_id);
    assert!(!again.chunks[0].created);
    assert_eq!(
        harness.store.get_document_chunks(&billy.document.helix_id).expect("chunks").len(),
        billy.chunks.len()
    );

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
}

#[test]
fn stage_without_document_reports_missing_context() {
    let harness = Harness::new(default_config());
    let mut ctx = ce_pipeline::PipelineContext::new(&harness.services);
    let error = harness.stage().run(&mut ctx).expect_err("no document");
    assert!(matches!(error, PipelineError::MissingContext { key: "document", .. }), "{error}");
}
