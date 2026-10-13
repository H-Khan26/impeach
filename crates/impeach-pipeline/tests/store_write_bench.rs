//! Store-write benchmark over the Green Valley scenario with the real local models (fastembed
//! BGE small, GLiNER2 mentions) and no LLM: FactStage, ContradictionCheckStage and
//! RelationStage (no store writes) are disabled and the ED LLM tie-break is off.
//!
//! Every store call goes through a counting wrapper, and every stage through a probe that
//! attributes store calls and store time to the stage. Prints one table per store with the
//! median of `IMPEACH_BENCH_RUNS` runs (default 3) after one warm-up run that loads the models.
//!
//! ```text
//! FASTEMBED_CACHE_DIR=<workspace>/.fastembed_cache \
//!   cargo test --release -p impeach-pipeline --test store_write_bench -- --ignored --nocapture
//! ```
//!
//! `IMPEACH_BENCH_STORES` selects the stores (`memory,helix`, the default).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use impeach_core::{ChunkId, EntityId, FactId, NodeId, SpanId};
use impeach_models::ModelRegistry;
use impeach_pipeline::{
    run_processing_with, PipelineContext, ProcessingEnv, ProcessingOptions, Stage, StageRegistry,
};
use impeach_store::graph::{
    EmbeddingWrite, EntityHit, FactHit, FactLink, SpanEntityLink, StoredChunk, StoredDocument,
    StoredEntity, StoredFact, StoredSpan, VectorLabel,
};
use impeach_store::{
    ChunkNode, DocumentNode, EntityNode, FactNode, GraphStore, MemoryStore, SpanNode,
};

const DIM: usize = 384;

/// Store calls and time spent in them, per method.
#[derive(Debug, Default, Clone)]
struct Stats {
    by_method: BTreeMap<&'static str, (u64, Duration)>,
}

impl Stats {
    fn calls(&self) -> u64 {
        self.by_method.values().map(|(calls, _)| calls).sum()
    }

    fn time(&self) -> Duration {
        self.by_method.values().map(|(_, time)| *time).sum()
    }

    fn minus(&self, before: &Stats) -> Stats {
        let mut delta = Stats::default();
        for (method, (calls, time)) in &self.by_method {
            let (calls0, time0) = before.by_method.get(method).copied().unwrap_or_default();
            if *calls > calls0 {
                delta.by_method.insert(method, (calls - calls0, *time - time0));
            }
        }
        delta
    }
}

/// A [`GraphStore`] that forwards every call and records its count and duration.
#[derive(Debug)]
struct CountingStore {
    inner: Arc<dyn GraphStore>,
    stats: Mutex<Stats>,
}

impl CountingStore {
    fn new(inner: Arc<dyn GraphStore>) -> Self {
        Self { inner, stats: Mutex::new(Stats::default()) }
    }

    fn snapshot(&self) -> Stats {
        self.stats.lock().expect("stats lock").clone()
    }

    fn timed<T>(&self, method: &'static str, call: impl FnOnce() -> T) -> T {
        let started = Instant::now();
        let result = call();
        let elapsed = started.elapsed();
        let mut stats = self.stats.lock().expect("stats lock");
        let entry = stats.by_method.entry(method).or_default();
        entry.0 += 1;
        entry.1 += elapsed;
        result
    }
}

type R<T> = impeach_store::Result<T>;

impl GraphStore for CountingStore {
    fn store_id(&self) -> String {
        self.inner.store_id()
    }
    fn embedding_dim(&self) -> usize {
        self.inner.embedding_dim()
    }
    fn reset(&self) -> R<()> {
        self.timed("reset", || self.inner.reset())
    }
    fn create_document(&self, document: &DocumentNode) -> R<StoredDocument> {
        self.timed("create_document", || self.inner.create_document(document))
    }
    fn update_document(&self, doc_id: &NodeId, document: &DocumentNode) -> R<StoredDocument> {
        self.timed("update_document", || self.inner.update_document(doc_id, document))
    }
    fn get_document_chunks(&self, doc_id: &NodeId) -> R<Vec<StoredChunk>> {
        self.timed("get_document_chunks", || self.inner.get_document_chunks(doc_id))
    }
    fn create_document_chunk(&self, doc_id: &NodeId, chunk: &ChunkNode) -> R<StoredChunk> {
        self.timed("create_document_chunk", || self.inner.create_document_chunk(doc_id, chunk))
    }
    fn update_document_chunk(&self, chunk_id: &ChunkId, chunk: &ChunkNode) -> R<StoredChunk> {
        self.timed("update_document_chunk", || self.inner.update_document_chunk(chunk_id, chunk))
    }
    fn create_chunk_embedding(&self, chunk_id: &ChunkId, embedding: &[f32]) -> R<()> {
        self.timed("create_chunk_embedding", || {
            self.inner.create_chunk_embedding(chunk_id, embedding)
        })
    }
    fn create_span_for_chunk(&self, span: &SpanNode) -> R<StoredSpan> {
        self.timed("create_span_for_chunk", || self.inner.create_span_for_chunk(span))
    }
    fn create_entity(&self, entity: &EntityNode) -> R<StoredEntity> {
        self.timed("create_entity", || self.inner.create_entity(entity))
    }
    fn create_entity_embedding(&self, entity_id: &EntityId, embedding: &[f32]) -> R<()> {
        self.timed("create_entity_embedding", || {
            self.inner.create_entity_embedding(entity_id, embedding)
        })
    }
    fn find_entity_by_name(&self, name: &str) -> R<Vec<EntityHit>> {
        self.timed("find_entity_by_name", || self.inner.find_entity_by_name(name))
    }
    fn find_similar_entities(&self, embedding: &[f32], limit: usize) -> R<Vec<EntityHit>> {
        self.timed("find_similar_entities", || self.inner.find_similar_entities(embedding, limit))
    }
    fn get_all_entities(&self, limit: usize) -> R<Vec<StoredEntity>> {
        self.timed("get_all_entities", || self.inner.get_all_entities(limit))
    }
    fn link_span_to_entity(
        &self,
        span_id: &SpanId,
        entity_id: &EntityId,
        confidence: f64,
        method: &str,
    ) -> R<()> {
        self.timed("link_span_to_entity", || {
            self.inner.link_span_to_entity(span_id, entity_id, confidence, method)
        })
    }
    fn create_fact(&self, fact: &FactNode) -> R<StoredFact> {
        self.timed("create_fact", || self.inner.create_fact(fact))
    }
    fn create_fact_embedding(&self, fact_id: &FactId, embedding: &[f32]) -> R<()> {
        self.timed("create_fact_embedding", || self.inner.create_fact_embedding(fact_id, embedding))
    }
    fn link_fact_to_document(
        &self,
        fact_id: &FactId,
        doc_id: &NodeId,
        page_number: i64,
        section: &str,
    ) -> R<()> {
        self.timed("link_fact_to_document", || {
            self.inner.link_fact_to_document(fact_id, doc_id, page_number, section)
        })
    }
    fn link_fact_to_entity(&self, fact_id: &FactId, entity_id: &EntityId, role: &str) -> R<()> {
        self.timed("link_fact_to_entity", || {
            self.inner.link_fact_to_entity(fact_id, entity_id, role)
        })
    }
    fn link_fact_to_span(
        &self,
        fact_id: &FactId,
        span_id: &SpanId,
        role: &str,
        confidence: f64,
    ) -> R<()> {
        self.timed("link_fact_to_span", || {
            self.inner.link_fact_to_span(fact_id, span_id, role, confidence)
        })
    }
    fn get_entity_facts(&self, entity_id: &EntityId) -> R<Vec<StoredFact>> {
        self.timed("get_entity_facts", || self.inner.get_entity_facts(entity_id))
    }
    fn get_fact_entities(&self, fact_id: &FactId) -> R<Vec<EntityId>> {
        self.timed("get_fact_entities", || self.inner.get_fact_entities(fact_id))
    }
    fn find_facts_by_text(&self, keywords: &str, limit: usize) -> R<Vec<FactHit>> {
        self.timed("find_facts_by_text", || self.inner.find_facts_by_text(keywords, limit))
    }
    fn find_similar_facts(&self, embedding: &[f32], limit: usize) -> R<Vec<FactHit>> {
        self.timed("find_similar_facts", || self.inner.find_similar_facts(embedding, limit))
    }
    fn create_chunks_batch(&self, doc_id: &NodeId, chunks: &[ChunkNode]) -> R<Vec<StoredChunk>> {
        self.timed("create_chunks_batch", || self.inner.create_chunks_batch(doc_id, chunks))
    }
    fn delete_document_chunks(&self, doc_id: &NodeId) -> R<usize> {
        self.timed("delete_document_chunks", || self.inner.delete_document_chunks(doc_id))
    }
    fn create_spans_batch(&self, spans: &[SpanNode]) -> R<Vec<StoredSpan>> {
        self.timed("create_spans_batch", || self.inner.create_spans_batch(spans))
    }
    fn create_embeddings_batch(
        &self,
        label: VectorLabel,
        items: &[EmbeddingWrite<'_>],
    ) -> R<usize> {
        self.timed("create_embeddings_batch", || self.inner.create_embeddings_batch(label, items))
    }
    fn create_entity_with_embedding(
        &self,
        entity: &EntityNode,
        embedding: &[f32],
    ) -> R<StoredEntity> {
        self.timed("create_entity_with_embedding", || {
            self.inner.create_entity_with_embedding(entity, embedding)
        })
    }
    fn link_spans_to_entities_batch(&self, links: &[SpanEntityLink<'_>]) -> R<usize> {
        self.timed("link_spans_to_entities_batch", || {
            self.inner.link_spans_to_entities_batch(links)
        })
    }
    fn create_facts_batch(&self, facts: &[FactNode]) -> R<Vec<StoredFact>> {
        self.timed("create_facts_batch", || self.inner.create_facts_batch(facts))
    }
    fn link_facts_batch(&self, links: &[FactLink<'_>]) -> R<usize> {
        self.timed("link_facts_batch", || self.inner.link_facts_batch(links))
    }
}

/// Per-stage store calls and time of one run, keyed `(document index, stage name)`.
type StageStats = Arc<Mutex<Vec<(String, Stats)>>>;

/// Wraps a built stage and records the store calls made while it runs.
struct Probe {
    inner: Box<dyn Stage>,
    store: Arc<CountingStore>,
    stats: StageStats,
}

impl Stage for Probe {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn run(&self, ctx: &mut PipelineContext) -> impeach_pipeline::Result<()> {
        let before = self.store.snapshot();
        let outcome = self.inner.run(ctx);
        let delta = self.store.snapshot().minus(&before);
        self.stats.lock().expect("stage stats").push((self.inner.name().to_owned(), delta));
        outcome
    }
}

fn probing_registry(store: &Arc<CountingStore>, stats: &StageStats) -> StageRegistry {
    let builtin = StageRegistry::builtin();
    let mut registry = StageRegistry::empty();
    for type_name in builtin.type_names() {
        let builtin = builtin.clone();
        let store = Arc::clone(store);
        let stats = Arc::clone(stats);
        registry.register(type_name, move |config, services| {
            let inner = builtin.build_stage(config, services)?;
            Ok(Box::new(Probe { inner, store: Arc::clone(&store), stats: Arc::clone(&stats) })
                as Box<dyn Stage>)
        });
    }
    registry
}

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// `processing_v17.yaml` without the LLM stages and the ED LLM tie-break.
fn write_offline_config(dir: &Path) -> PathBuf {
    let text =
        std::fs::read_to_string(workspace().join("assets/config/processing/processing_v17.yaml"))
            .expect("packaged config");
    let replace = |text: String, from: &str, to: &str| {
        assert!(text.contains(from), "config has `{from}`");
        text.replacen(from, to, 1)
    };
    let text = replace(text, "allow_llm: true", "allow_llm: false");
    let text = replace(
        text,
        "name: fact_extraction\n    enabled: true",
        "name: fact_extraction\n    enabled: false",
    );
    let text = replace(
        text,
        "name: contradiction_check\n    enabled: true",
        "name: contradiction_check\n    enabled: false",
    );
    let text = replace(text, "backend: \"fastcoref\"", "backend: \"rule\"");
    let text = replace(
        text,
        "name: relation_extraction\n    enabled: true",
        "name: relation_extraction\n    enabled: false",
    );
    let path = dir.join("bench_offline.yaml");
    std::fs::write(&path, text).expect("write config");
    path
}

/// One run: both Green Valley documents into one fresh store. Returns per-(doc, stage)
/// wall times and store stats.
fn run_once(
    store: Arc<dyn GraphStore>,
    registry: &Arc<ModelRegistry>,
    config: &Path,
) -> Vec<(String, Duration, Stats)> {
    let counting = Arc::new(CountingStore::new(store));
    let stage_stats: StageStats = Arc::default();
    let options = ProcessingOptions {
        config_path: Some(config.to_path_buf()),
        ..ProcessingOptions::default()
    };
    let mut rows = Vec::new();
    for (index, input) in
        ["green_valley_incident_report.jsonl", "lopez_v_green_valley_complaint.jsonl"]
            .into_iter()
            .enumerate()
    {
        let dyn_store: Arc<dyn GraphStore> = counting.clone();
        let env = ProcessingEnv {
            registry: Some(Arc::clone(registry)),
            store: Some(dyn_store),
            stages: Some(probing_registry(&counting, &stage_stats)),
            unregistered: None,
        };
        let path = workspace().join("fixtures/input").join(input);
        let run = run_processing_with(&path, &options, env).expect("document processes");
        let stats: Vec<(String, Stats)> = std::mem::take(&mut *stage_stats.lock().expect("stats"));
        for (timing, (stage, store_stats)) in run.report.stage_timings.iter().zip(stats) {
            assert_eq!(timing.stage, stage);
            rows.push((format!("doc{} {stage}", index + 1), timing.elapsed, store_stats));
        }
    }
    rows
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn report(label: &str, runs: &[Vec<(String, Duration, Stats)>]) {
    println!("\n== store: {label} (median of {} runs) ==", runs.len());
    println!("{:<34} {:>10} {:>12} {:>8}  methods", "stage", "stage ms", "store ms", "calls");
    for (row, (stage, _, stats)) in runs[0].iter().enumerate() {
        let stage_ms = median(runs.iter().map(|run| ms(run[row].1)).collect());
        let store_ms = median(runs.iter().map(|run| ms(run[row].2.time())).collect());
        let methods: Vec<String> = stats
            .by_method
            .iter()
            .map(|(method, (calls, _))| {
                let method_ms = median(
                    runs.iter()
                        .map(|run| {
                            ms(run[row].2.by_method.get(method).map_or(Duration::ZERO, |m| m.1))
                        })
                        .collect(),
                );
                format!("{method}={calls}/{method_ms:.1}ms")
            })
            .collect();
        println!(
            "{stage:<34} {stage_ms:>10.1} {store_ms:>12.1} {:>8}  {}",
            stats.calls(),
            methods.join(" ")
        );
    }
}

#[test]
#[ignore = "needs the BGE small and GLiNER2 model files; run with --ignored --nocapture"]
fn green_valley_store_writes() {
    let runs: usize =
        std::env::var("IMPEACH_BENCH_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let stores = std::env::var("IMPEACH_BENCH_STORES").unwrap_or_else(|_| "memory,helix".into());
    let dir = tempfile::tempdir().expect("tempdir");
    let config_path = write_offline_config(dir.path());
    let config = impeach_config::load_config(&config_path).expect("bench config loads");
    let registry = Arc::new(ModelRegistry::from_config(&config));

    // Warm-up: loads the models into the shared registry.
    run_once(Arc::new(MemoryStore::new(DIM)), &registry, &config_path);

    for store in stores.split(',').map(str::trim) {
        let mut all = Vec::new();
        for _ in 0..runs {
            let backend: Arc<dyn GraphStore> = match store {
                "memory" => Arc::new(MemoryStore::new(DIM)),
                #[cfg(feature = "helix")]
                "helix" => {
                    let data = dir.path().join(format!("helix_{}", all.len()));
                    impeach_store::HelixStore::open(&impeach_store::HelixOptions::embedded(
                        data, "bench", DIM,
                    ))
                    .expect("helix opens")
                }
                other => panic!("unknown store `{other}`"),
            };
            all.push(run_once(backend, &registry, &config_path));
        }
        report(store, &all);
    }
}
