//! Model registry keyed `<pipeline.name>_<type>` (Python `models/registry.py` and the model
//! registration in `pipeline/factory.py`).
//!
//! Models are constructed lazily on the first accessor call and shared as `Arc<dyn Trait>`.
//! Tests and callers can inject any implementation (e.g. the mocks) under a key with the
//! `insert_*` methods; an injected model takes precedence over the configured backend.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use ce_config::pipeline::ModelsConfig;
use ce_config::PipelineConfig;

use crate::error::ModelError;
use crate::traits::{Coref, Embedder, Llm, NerModel, NliJudge};

/// Model types of the `models:` config block, in Python registration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelKind {
    /// `ollama`: the LLM ([`Llm`]).
    Ollama,
    /// `coref`: coreference ([`Coref`]).
    Coref,
    /// `bge`: sentence embeddings ([`Embedder`]).
    Bge,
    /// `deberta_nli`: NLI cross-encoder ([`NliJudge`]).
    DebertaNli,
    /// `gliner2`: NER and relations ([`NerModel`]).
    Gliner2,
}

impl ModelKind {
    /// Every kind, in Python registration order.
    pub const ALL: [ModelKind; 5] = [
        ModelKind::Ollama,
        ModelKind::Coref,
        ModelKind::Bge,
        ModelKind::DebertaNli,
        ModelKind::Gliner2,
    ];

    /// Config key / registry suffix, e.g. `deberta_nli`.
    pub fn as_str(self) -> &'static str {
        match self {
            ModelKind::Ollama => "ollama",
            ModelKind::Coref => "coref",
            ModelKind::Bge => "bge",
            ModelKind::DebertaNli => "deberta_nli",
            ModelKind::Gliner2 => "gliner2",
        }
    }
}

impl fmt::Display for ModelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Registry key of a model: `<pipeline_name>_<kind>`.
pub fn model_key(pipeline_name: &str, kind: ModelKind) -> String {
    format!("{pipeline_name}_{}", kind.as_str())
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Lazily constructed, shared model instances.
pub struct ModelRegistry {
    pipeline_name: String,
    models: Option<ModelsConfig>,
    embedders: Mutex<HashMap<String, Arc<dyn Embedder>>>,
    nlis: Mutex<HashMap<String, Arc<dyn NliJudge>>>,
    llms: Mutex<HashMap<String, Arc<dyn Llm>>>,
    ners: Mutex<HashMap<String, Arc<dyn NerModel>>>,
    corefs: Mutex<HashMap<String, Arc<dyn Coref>>>,
}

impl fmt::Debug for ModelRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelRegistry")
            .field("pipeline_name", &self.pipeline_name)
            .field("configured", &self.configured_keys())
            .finish_non_exhaustive()
    }
}

impl ModelRegistry {
    /// A registry for the `models:` block of pipeline `pipeline_name`.
    pub fn new(pipeline_name: impl Into<String>, models: ModelsConfig) -> Self {
        Self::build(pipeline_name.into(), Some(models))
    }

    /// A registry for `config.models` keyed by `config.pipeline.name`.
    pub fn from_config(config: &PipelineConfig) -> Self {
        Self::new(config.pipeline.name.clone(), config.models.clone())
    }

    /// A registry with no configured models; only injected models resolve.
    pub fn empty(pipeline_name: impl Into<String>) -> Self {
        Self::build(pipeline_name.into(), None)
    }

    fn build(pipeline_name: String, models: Option<ModelsConfig>) -> Self {
        Self {
            pipeline_name,
            models,
            embedders: Mutex::default(),
            nlis: Mutex::default(),
            llms: Mutex::default(),
            ners: Mutex::default(),
            corefs: Mutex::default(),
        }
    }

    /// Pipeline name used as the key prefix.
    pub fn pipeline_name(&self) -> &str {
        &self.pipeline_name
    }

    /// Registry key of `kind` in this registry.
    pub fn key(&self, kind: ModelKind) -> String {
        model_key(&self.pipeline_name, kind)
    }

    /// The configured `models:` block, if any.
    pub fn models_config(&self) -> Option<&ModelsConfig> {
        self.models.as_ref()
    }

    /// Keys of the configured models, in registration order (`gliner2` only when configured).
    pub fn configured_keys(&self) -> Vec<String> {
        let Some(models) = &self.models else {
            return Vec::new();
        };
        ModelKind::ALL
            .into_iter()
            .filter(|k| *k != ModelKind::Gliner2 || models.gliner2.is_some())
            .map(|k| self.key(k))
            .collect()
    }

    /// Injects an embedder under `key`.
    pub fn insert_embedder(&self, key: impl Into<String>, model: Arc<dyn Embedder>) {
        lock(&self.embedders).insert(key.into(), model);
    }

    /// Injects an NLI judge under `key`.
    pub fn insert_nli(&self, key: impl Into<String>, model: Arc<dyn NliJudge>) {
        lock(&self.nlis).insert(key.into(), model);
    }

    /// Injects an LLM under `key`.
    pub fn insert_llm(&self, key: impl Into<String>, model: Arc<dyn Llm>) {
        lock(&self.llms).insert(key.into(), model);
    }

    /// Injects a NER model under `key`.
    pub fn insert_ner(&self, key: impl Into<String>, model: Arc<dyn NerModel>) {
        lock(&self.ners).insert(key.into(), model);
    }

    /// Injects a coreference resolver under `key`.
    pub fn insert_coref(&self, key: impl Into<String>, model: Arc<dyn Coref>) {
        lock(&self.corefs).insert(key.into(), model);
    }

    /// The embedder registered under `key` (the `bge` model), constructed on first use.
    pub fn embedder(&self, key: &str) -> Result<Arc<dyn Embedder>, ModelError> {
        let mut map = lock(&self.embedders);
        if let Some(m) = map.get(key) {
            return Ok(Arc::clone(m));
        }
        self.require_kind(key, ModelKind::Bge)?;
        let model = self.build_embedder()?;
        map.insert(key.to_owned(), Arc::clone(&model));
        Ok(model)
    }

    /// The NLI judge registered under `key` (the `deberta_nli` model).
    pub fn nli(&self, key: &str) -> Result<Arc<dyn NliJudge>, ModelError> {
        let mut map = lock(&self.nlis);
        if let Some(m) = map.get(key) {
            return Ok(Arc::clone(m));
        }
        self.require_kind(key, ModelKind::DebertaNli)?;
        let model = self.build_nli()?;
        map.insert(key.to_owned(), Arc::clone(&model));
        Ok(model)
    }

    /// The LLM registered under `key` (the `ollama` model).
    pub fn llm(&self, key: &str) -> Result<Arc<dyn Llm>, ModelError> {
        let mut map = lock(&self.llms);
        if let Some(m) = map.get(key) {
            return Ok(Arc::clone(m));
        }
        self.require_kind(key, ModelKind::Ollama)?;
        let model = self.build_llm()?;
        map.insert(key.to_owned(), Arc::clone(&model));
        Ok(model)
    }

    /// The NER model registered under `key` (the `gliner2` model).
    pub fn ner(&self, key: &str) -> Result<Arc<dyn NerModel>, ModelError> {
        let mut map = lock(&self.ners);
        if let Some(m) = map.get(key) {
            return Ok(Arc::clone(m));
        }
        self.require_kind(key, ModelKind::Gliner2)?;
        let model = self.build_ner()?;
        map.insert(key.to_owned(), Arc::clone(&model));
        Ok(model)
    }

    /// The coreference resolver registered under `key` (the `coref` model).
    pub fn coref(&self, key: &str) -> Result<Arc<dyn Coref>, ModelError> {
        let mut map = lock(&self.corefs);
        if let Some(m) = map.get(key) {
            return Ok(Arc::clone(m));
        }
        self.require_kind(key, ModelKind::Coref)?;
        let model = self.build_coref()?;
        map.insert(key.to_owned(), Arc::clone(&model));
        Ok(model)
    }

    /// Checks that `key` names a configured model of `kind`.
    fn require_kind(&self, key: &str, kind: ModelKind) -> Result<(), ModelError> {
        if self.configured_keys().iter().any(|k| k == key) && key == self.key(kind) {
            Ok(())
        } else {
            Err(ModelError::UnknownModel { key: key.to_owned(), known: self.configured_keys() })
        }
    }

    fn build_embedder(&self) -> Result<Arc<dyn Embedder>, ModelError> {
        Err(unavailable("fastembed"))
    }

    fn build_nli(&self) -> Result<Arc<dyn NliJudge>, ModelError> {
        Err(unavailable("ort-nli"))
    }

    fn build_llm(&self) -> Result<Arc<dyn Llm>, ModelError> {
        Err(unavailable("ollama"))
    }

    fn build_ner(&self) -> Result<Arc<dyn NerModel>, ModelError> {
        Err(unavailable("gliner2"))
    }

    fn build_coref(&self) -> Result<Arc<dyn Coref>, ModelError> {
        Err(unavailable("coref"))
    }
}

fn unavailable(backend: &str) -> ModelError {
    ModelError::Unavailable {
        backend: backend.to_owned(),
        reason: "backend not compiled in".to_owned(),
    }
}
