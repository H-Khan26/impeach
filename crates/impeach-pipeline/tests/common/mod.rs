//! Shared helpers: the default config, a `MemoryStore`, a registry with a `MockEmbedder`.

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Arc;

use impeach_config::pipeline::{ChunkingStrategy, RagIndexStageConfig};
use impeach_config::{PipelineConfig, StageConfig};
use impeach_core::{DocumentPayload, IndexingResult};
use impeach_models::{MockEmbedder, ModelKind, ModelRegistry};
use impeach_pipeline::{
    ContextData, Pipeline, PipelineInput, RagIndexStage, Stage, StageBuildContext,
};
use impeach_store::{GraphStore, MemoryStore};

pub const DIM: usize = 384;

pub fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/input")
}

pub fn fixture(name: &str) -> PathBuf {
    fixtures_dir().join(name)
}

/// Every `fixtures/input/*.jsonl`, sorted.
pub fn jsonl_fixtures() -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(fixtures_dir())
        .expect("fixtures/input exists")
        .map(|entry| entry.expect("dir entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
        .collect();
    paths.sort();
    paths
}

pub fn default_config() -> PipelineConfig {
    impeach_config::load_default_config().expect("default config loads")
}

/// The default config with a modified `RAGIndexStage` block and chunking settings.
pub fn config_with(
    mut edit_stage: impl FnMut(&mut RagIndexStageConfig),
    edit_chunking: impl FnOnce(&mut impeach_config::pipeline::ChunkingConfig),
) -> PipelineConfig {
    let mut config = default_config();
    for stage in &mut config.stages {
        if let StageConfig::RagIndex(rag) = stage {
            edit_stage(rag);
        }
    }
    edit_chunking(&mut config.rag.chunking);
    config
}

pub fn recursive_config(size: usize, overlap: usize) -> PipelineConfig {
    config_with(
        |_| {},
        |chunking| {
            chunking.strategy = ChunkingStrategy::Recursive;
            chunking.chunk_size = size;
            chunking.chunk_overlap = overlap;
        },
    )
}

/// Services over a fresh `MemoryStore` and a registry whose `bge` key is a `MockEmbedder`.
pub struct Harness {
    pub store: Arc<MemoryStore>,
    pub embedder: Arc<MockEmbedder>,
    pub services: StageBuildContext,
}

impl Harness {
    pub fn new(config: PipelineConfig) -> Self {
        Self::with_store(config, Arc::new(MemoryStore::new(DIM)))
    }

    pub fn with_store(config: PipelineConfig, store: Arc<MemoryStore>) -> Self {
        let embedder = Arc::new(MockEmbedder::new(DIM));
        let registry = ModelRegistry::from_config(&config);
        registry.insert_embedder(registry.key(ModelKind::Bge), embedder.clone());
        let dyn_store: Arc<dyn GraphStore> = store.clone();
        let services = StageBuildContext::new(Arc::new(config), Arc::new(registry), dyn_store);
        Self { store, embedder, services }
    }

    pub fn rag_config(&self) -> RagIndexStageConfig {
        self.services
            .config
            .stages
            .iter()
            .find_map(|stage| match stage {
                StageConfig::RagIndex(rag) => Some(rag.clone()),
                _ => None,
            })
            .expect("config has a RAGIndexStage")
    }

    pub fn stage(&self) -> RagIndexStage {
        RagIndexStage::new(&self.rag_config(), &self.services).expect("stage builds")
    }

    pub fn pipeline_with(&self, stage: impl Stage + 'static) -> Pipeline {
        Pipeline::new(vec![Box::new(stage)], self.services.clone())
    }

    /// Runs a single RAGIndexStage on `input` and returns the context data.
    pub fn run_stage(&self, stage: RagIndexStage, input: PipelineInput) -> ContextData {
        let pipeline = self.pipeline_with(stage);
        let mut ctx = pipeline.new_context();
        pipeline.run(&mut ctx, input).expect("pipeline runs");
        ctx.data
    }

    pub fn index(&self, input: PipelineInput) -> (IndexingResult, ContextData) {
        let data = self.run_stage(self.stage(), input);
        let result = data.rag_index_result.clone().expect("rag_index_result is set");
        (result, data)
    }

    pub fn count_label(&self, label: &str) -> usize {
        (0..5000_u64)
            .filter_map(|id| self.store.node_snapshot(&id.to_string()))
            .filter(|(node_label, _)| *node_label == label)
            .count()
    }

    pub fn count_edges(&self, label: &str) -> usize {
        self.store.edges().iter().filter(|edge| edge.label == label).count()
    }
}

pub fn read_fixture(name: &str) -> DocumentPayload {
    impeach_ingest::read_first_record(fixture(name)).expect("fixture reads")
}

/// Characters `start..end` of `text`.
pub fn char_slice(text: &str, start: usize, end: usize) -> String {
    text.chars().skip(start).take(end - start).collect()
}
