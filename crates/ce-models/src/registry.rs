//! Model registry keyed `<pipeline.name>_<type>` (Python `models/registry.py` and the model
//! registration in `pipeline/factory.py`).
//!
//! Models are constructed lazily on the first accessor call and shared as `Arc<dyn Trait>`.
//! Tests and callers can inject any implementation (e.g. the mocks) under a key with the
//! `insert_*` methods; an injected model takes precedence over the configured backend.
//!
//! Configured backends: `bge` -> `FastEmbedder` (feature `fastembed`), `deberta_nli` ->
//! `OrtNli` (feature `ort-nli`), `ollama` -> `OllamaLlm` (feature `ollama`), `gliner2` ->
//! `Gliner2Ner` (feature `gliner2`), `coref` by `models.coref.backend`:
//!
//! | `backend` | Resolver |
//! | --- | --- |
//! | `fastcoref` | [`SidecarCoref`] started with `models.coref.command`; without a command, [`RuleCoref`] with one warning ([`FASTCOREF_FALLBACK_WARNING`]) so the shipped config runs out of the box |
//! | `rule` | [`RuleCoref`] |
//! | `llm` | [`LlmCoref`] over the registry's `<pipeline.name>_ollama` LLM |
//! | `mock` | [`MockCoref`] returning no clusters |
//!
//! A sidecar set with [`ModelRegistry::with_coref_sidecar`] replaces the configured coref
//! backend, whatever it is. A backend whose feature is off yields [`ModelError::Unavailable`];
//! a configured sidecar that cannot start is a [`ModelError::Load`] error.
//!
//! Models are built once per key, so the fallback warning is logged once per registry.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use ce_core::util::lock;

use ce_config::pipeline::{CorefBackend, ModelsConfig};
use ce_config::PipelineConfig;

use crate::error::ModelError;
use crate::llm_coref::LlmCoref;
use crate::mock::MockCoref;
use crate::rule_coref::RuleCoref;
use crate::sidecar::{SidecarCommand, SidecarCoref};
use crate::traits::{Coref, Embedder, Llm, NerModel, NliJudge};

/// Warning logged when `models.coref.backend: fastcoref` has no `command` and the registry
/// falls back to [`RuleCoref`].
pub const FASTCOREF_FALLBACK_WARNING: &str = "fastcoref sidecar command not configured; using \
     the rule coreference backend. set models.coref.command to a fastcoref sidecar, or set \
     models.coref.backend: rule to silence this";

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

/// One model slot: empty until the model is injected or constructed.
type Slot<T> = Arc<Mutex<Option<Arc<T>>>>;

/// Models of one kind by key. The map lock is held only to find a key's slot; construction
/// runs under that slot's own lock, so a slow construction (a model download) blocks only
/// callers of the same key.
struct Models<T: ?Sized> {
    slots: Mutex<HashMap<String, Slot<T>>>,
}

impl<T: ?Sized> Default for Models<T> {
    fn default() -> Self {
        Self { slots: Mutex::default() }
    }
}

impl<T: ?Sized> Models<T> {
    fn slot(&self, key: &str) -> Slot<T> {
        let mut slots = lock(&self.slots);
        match slots.get(key) {
            Some(slot) => Arc::clone(slot),
            None => {
                let slot: Slot<T> = Arc::default();
                slots.insert(key.to_owned(), Arc::clone(&slot));
                slot
            }
        }
    }

    fn insert(&self, key: &str, model: Arc<T>) {
        *lock(&self.slot(key)) = Some(model);
    }

    /// The model under `key`, constructed with `build` when the slot is empty.
    fn get_or_build(
        &self,
        key: &str,
        build: impl FnOnce() -> Result<Arc<T>, ModelError>,
    ) -> Result<Arc<T>, ModelError> {
        let slot = self.slot(key);
        let mut model = lock(&slot);
        if let Some(existing) = model.as_ref() {
            return Ok(Arc::clone(existing));
        }
        let built = build()?;
        *model = Some(Arc::clone(&built));
        Ok(built)
    }
}

/// Lazily constructed, shared model instances.
pub struct ModelRegistry {
    pipeline_name: String,
    models: Option<ModelsConfig>,
    coref_sidecar: Option<SidecarCommand>,
    embedders: Models<dyn Embedder>,
    nlis: Models<dyn NliJudge>,
    llms: Models<dyn Llm>,
    ners: Models<dyn NerModel>,
    corefs: Models<dyn Coref>,
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
            coref_sidecar: None,
            embedders: Models::default(),
            nlis: Models::default(),
            llms: Models::default(),
            ners: Models::default(),
            corefs: Models::default(),
        }
    }

    /// Runs the `coref` model as a sidecar process started with `command` (see
    /// [`SidecarCoref`] for the protocol), replacing the configured coref backend.
    pub fn with_coref_sidecar(mut self, command: SidecarCommand) -> Self {
        self.coref_sidecar = Some(command);
        self
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
        self.embedders.insert(&key.into(), model);
    }

    /// Injects an NLI judge under `key`.
    pub fn insert_nli(&self, key: impl Into<String>, model: Arc<dyn NliJudge>) {
        self.nlis.insert(&key.into(), model);
    }

    /// Injects an LLM under `key`.
    pub fn insert_llm(&self, key: impl Into<String>, model: Arc<dyn Llm>) {
        self.llms.insert(&key.into(), model);
    }

    /// Injects a NER model under `key`.
    pub fn insert_ner(&self, key: impl Into<String>, model: Arc<dyn NerModel>) {
        self.ners.insert(&key.into(), model);
    }

    /// Injects a coreference resolver under `key`.
    pub fn insert_coref(&self, key: impl Into<String>, model: Arc<dyn Coref>) {
        self.corefs.insert(&key.into(), model);
    }

    /// The embedder registered under `key` (the `bge` model), constructed on first use.
    pub fn embedder(&self, key: &str) -> Result<Arc<dyn Embedder>, ModelError> {
        self.embedders.get_or_build(key, || {
            self.require_kind(key, ModelKind::Bge)?;
            self.build_embedder()
        })
    }

    /// The NLI judge registered under `key` (the `deberta_nli` model).
    pub fn nli(&self, key: &str) -> Result<Arc<dyn NliJudge>, ModelError> {
        self.nlis.get_or_build(key, || {
            self.require_kind(key, ModelKind::DebertaNli)?;
            self.build_nli()
        })
    }

    /// The LLM registered under `key` (the `ollama` model).
    pub fn llm(&self, key: &str) -> Result<Arc<dyn Llm>, ModelError> {
        self.llms.get_or_build(key, || {
            self.require_kind(key, ModelKind::Ollama)?;
            self.build_llm()
        })
    }

    /// The NER model registered under `key` (the `gliner2` model).
    pub fn ner(&self, key: &str) -> Result<Arc<dyn NerModel>, ModelError> {
        self.ners.get_or_build(key, || {
            self.require_kind(key, ModelKind::Gliner2)?;
            self.build_ner()
        })
    }

    /// The coreference resolver registered under `key` (the `coref` model).
    pub fn coref(&self, key: &str) -> Result<Arc<dyn Coref>, ModelError> {
        self.corefs.get_or_build(key, || {
            self.require_kind(key, ModelKind::Coref)?;
            self.build_coref()
        })
    }

    /// Checks that `key` names a configured model of `kind`.
    fn require_kind(&self, key: &str, kind: ModelKind) -> Result<(), ModelError> {
        if self.configured_keys().iter().any(|k| k == key) && key == self.key(kind) {
            Ok(())
        } else {
            Err(ModelError::UnknownModel { key: key.to_owned(), known: self.configured_keys() })
        }
    }

    /// The configured `models:` block; only called for configured keys.
    fn configured(&self) -> Result<&ModelsConfig, ModelError> {
        self.models
            .as_ref()
            .ok_or_else(|| ModelError::UnknownModel { key: String::new(), known: Vec::new() })
    }

    fn build_embedder(&self) -> Result<Arc<dyn Embedder>, ModelError> {
        let models = self.configured()?;
        warn_non_cpu("bge", &models.bge.device);
        #[cfg(feature = "fastembed")]
        {
            let options = crate::fastembed::FastEmbedOptions::from_config(&models.bge);
            Ok(Arc::new(crate::fastembed::FastEmbedder::new(&options)?))
        }
        #[cfg(not(feature = "fastembed"))]
        {
            Err(unavailable("fastembed"))
        }
    }

    fn build_nli(&self) -> Result<Arc<dyn NliJudge>, ModelError> {
        let models = self.configured()?;
        warn_non_cpu("deberta_nli", &models.deberta_nli.device);
        #[cfg(feature = "ort-nli")]
        {
            let options = crate::ort_nli::OrtNliOptions::from_config(&models.deberta_nli);
            Ok(Arc::new(crate::ort_nli::OrtNli::new(&options)?))
        }
        #[cfg(not(feature = "ort-nli"))]
        {
            Err(unavailable("ort-nli"))
        }
    }

    fn build_llm(&self) -> Result<Arc<dyn Llm>, ModelError> {
        let models = self.configured()?;
        #[cfg(feature = "ollama")]
        {
            let options = crate::ollama::OllamaOptions::from_config(&models.ollama);
            Ok(Arc::new(crate::ollama::OllamaLlm::new(&options)?))
        }
        #[cfg(not(feature = "ollama"))]
        {
            let _ = models;
            Err(unavailable("ollama"))
        }
    }

    fn build_ner(&self) -> Result<Arc<dyn NerModel>, ModelError> {
        let models = self.configured()?;
        let config = models.gliner2.as_ref().ok_or_else(|| ModelError::UnknownModel {
            key: self.key(ModelKind::Gliner2),
            known: self.configured_keys(),
        })?;
        warn_non_cpu("gliner2", &config.device);
        #[cfg(feature = "gliner2")]
        {
            let options = crate::gliner2::Gliner2Options::from_config(config);
            Ok(Arc::new(crate::gliner2::Gliner2Ner::new(&options)?))
        }
        #[cfg(not(feature = "gliner2"))]
        {
            Err(unavailable("gliner2"))
        }
    }

    fn build_coref(&self) -> Result<Arc<dyn Coref>, ModelError> {
        if let Some(command) = &self.coref_sidecar {
            return Ok(Arc::new(SidecarCoref::new(command.clone())?));
        }
        let config = &self.configured()?.coref;
        match config.backend {
            CorefBackend::Fastcoref => {
                let Some((program, args)) = config.command.as_deref().and_then(<[_]>::split_first)
                else {
                    tracing::warn!("{FASTCOREF_FALLBACK_WARNING}");
                    return Ok(Arc::new(RuleCoref::new()));
                };
                let command = SidecarCommand::new(program).args(args);
                Ok(Arc::new(SidecarCoref::new(command)?))
            }
            CorefBackend::Rule => Ok(Arc::new(RuleCoref::new())),
            CorefBackend::Llm => {
                let llm = self.llm(&self.key(ModelKind::Ollama))?;
                Ok(Arc::new(LlmCoref::new(llm)?))
            }
            CorefBackend::Mock => Ok(Arc::new(MockCoref::default())),
        }
    }
}

/// Logs that only CPU inference is supported when the config asks for another device.
fn warn_non_cpu(kind: &str, device: &str) {
    if !device.eq_ignore_ascii_case("cpu") {
        tracing::warn!(model = kind, device, "only CPU inference is supported; using the CPU");
    }
}

#[cfg_attr(
    all(feature = "fastembed", feature = "ort-nli", feature = "ollama", feature = "gliner2"),
    allow(dead_code)
)]
fn unavailable(feature: &str) -> ModelError {
    ModelError::Unavailable {
        backend: feature.to_owned(),
        reason: format!("ce-models was built without the `{feature}` feature"),
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;
    use crate::mock::{MockCoref, MockEmbedder, MockLlm, MockNer, MockNli};

    fn models(with_gliner2: bool) -> ModelsConfig {
        let mut value = json!({
            "ollama": {"model": "qwen2.5:7b-instruct-q4_K_M", "base_url": "http://localhost:11434",
                       "temperature": 0.2, "num_ctx": 16384},
            "coref": {"backend": "fastcoref", "device": "cpu"},
            "bge": {"model": "BAAI/bge-small-en-v1.5", "device": "cpu",
                    "normalize_embeddings": true, "batch_size": 64},
            "deberta_nli": {"model": "cross-encoder/nli-deberta-v3-large", "device": "cpu"}
        });
        if with_gliner2 {
            value["gliner2"] =
                json!({"model": "fastino/gliner2-base-v1", "device": "cpu", "threshold": 0.4});
        }
        serde_json::from_value(value).expect("valid models block")
    }

    #[test]
    fn keys_follow_python_naming() {
        let registry = ModelRegistry::new("v17", models(true));
        assert_eq!(registry.key(ModelKind::DebertaNli), "v17_deberta_nli");
        assert_eq!(model_key("p", ModelKind::Bge), "p_bge");
        assert_eq!(
            registry.configured_keys(),
            vec!["v17_ollama", "v17_coref", "v17_bge", "v17_deberta_nli", "v17_gliner2"]
        );
        let without = ModelRegistry::new("v17", models(false));
        assert_eq!(
            without.configured_keys(),
            vec!["v17_ollama", "v17_coref", "v17_bge", "v17_deberta_nli"]
        );
        assert!(ModelRegistry::empty("v17").configured_keys().is_empty());
    }

    #[test]
    fn injected_models_take_precedence() {
        let registry = ModelRegistry::new("p", models(true));
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(8));
        registry.insert_embedder("p_bge", Arc::clone(&embedder));
        registry.insert_nli("p_deberta_nli", Arc::new(MockNli::new()));
        registry.insert_llm("p_ollama", Arc::new(MockLlm::new()));
        registry.insert_ner("p_gliner2", Arc::new(MockNer::default()));
        registry.insert_coref("p_coref", Arc::new(MockCoref::default()));
        let got = registry.embedder("p_bge").expect("injected embedder");
        assert!(Arc::ptr_eq(&got, &embedder));
        assert_eq!(got.dimension(), 8);
        assert!(registry.nli("p_deberta_nli").is_ok());
        assert!(registry.llm("p_ollama").is_ok());
        assert!(registry.ner("p_gliner2").is_ok());
        assert!(registry.coref("p_coref").is_ok());
    }

    #[test]
    fn injection_works_under_any_key_without_config() {
        let registry = ModelRegistry::empty("p");
        registry.insert_embedder("eval_minilm", Arc::new(MockEmbedder::new(4)));
        assert_eq!(registry.embedder("eval_minilm").expect("injected").dimension(), 4);
    }

    #[test]
    fn unknown_key_lists_known_keys() {
        let registry = ModelRegistry::new("p", models(false));
        let Err(err) = registry.embedder("other_bge") else {
            panic!("unknown key resolved");
        };
        match err {
            ModelError::UnknownModel { key, known } => {
                assert_eq!(key, "other_bge");
                assert_eq!(known, vec!["p_ollama", "p_coref", "p_bge", "p_deberta_nli"]);
            }
            other => panic!("unexpected error {other:?}"),
        }
        // A configured key of another kind is not an embedder.
        assert!(matches!(registry.embedder("p_ollama"), Err(ModelError::UnknownModel { .. })));
        // gliner2 is unknown when not configured.
        assert!(matches!(registry.ner("p_gliner2"), Err(ModelError::UnknownModel { .. })));
        assert!(matches!(
            ModelRegistry::empty("p").llm("p_ollama"),
            Err(ModelError::UnknownModel { .. })
        ));
    }

    /// A models block with the given `coref` block (`device: cpu` added).
    fn coref_models(coref: &serde_json::Value) -> ModelsConfig {
        let mut models = json!({
            "ollama": {"model": "m", "base_url": "http://localhost:11434",
                       "temperature": 0.2, "num_ctx": 16384},
            "coref": coref,
            "bge": {"model": "b", "device": "cpu", "normalize_embeddings": true, "batch_size": 64},
            "deberta_nli": {"model": "n", "device": "cpu"}
        });
        models["coref"]["device"] = json!("cpu");
        serde_json::from_value(models).expect("valid models block")
    }

    #[test]
    fn fastcoref_without_command_falls_back_to_rules_with_one_warning() {
        use std::io::Write;
        use std::sync::Mutex;

        use crate::traits::CorefMention;

        /// Appends formatted log lines to a shared buffer.
        struct Capture(Arc<Mutex<Vec<u8>>>);
        impl Write for Capture {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                lock(&self.0).extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let log = Arc::new(Mutex::new(Vec::new()));
        let writer = Arc::clone(&log);
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || Capture(Arc::clone(&writer)))
            .with_ansi(false)
            .with_max_level(tracing::Level::WARN)
            .finish();
        let registry = ModelRegistry::new("p", models(false));
        let mentions = [
            CorefMention { start: 0, end: 11, text: "Maria Lopez", entity_type: "PER" },
            CorefMention { start: 17, end: 22, text: "Lopez", entity_type: "PER" },
        ];
        tracing::subscriber::with_default(subscriber, || {
            let first = registry.coref("p_coref").expect("rule fallback");
            let second = registry.coref("p_coref").expect("cached fallback");
            assert!(Arc::ptr_eq(&first, &second));
            let clusters = first
                .resolve_with_mentions("Maria Lopez said Lopez left.", &mentions)
                .expect("rule clusters");
            assert_eq!(clusters, vec![crate::CorefCluster { spans: vec![(0, 11), (17, 22)] }]);
        });
        let text = String::from_utf8(lock(&log).clone()).expect("UTF-8 log");
        assert_eq!(text.matches(FASTCOREF_FALLBACK_WARNING).count(), 1, "{text}");
        assert!(text.contains("WARN"), "{text}");
        assert!(
            FASTCOREF_FALLBACK_WARNING.contains(
                "set models.coref.command to a fastcoref sidecar, or set \
                 models.coref.backend: rule to silence this"
            ),
            "{FASTCOREF_FALLBACK_WARNING}"
        );
    }

    #[test]
    fn fastcoref_command_that_does_not_start_is_a_load_error() {
        let registry = ModelRegistry::new(
            "p",
            coref_models(&json!({"backend": "fastcoref",
                                "command": ["ce-models-no-such-sidecar-program", "--flag"]})),
        );
        let Err(err) = registry.coref("p_coref") else {
            panic!("missing sidecar program resolved");
        };
        assert!(
            matches!(err, ModelError::Load { ref message, .. }
                if message.contains("ce-models-no-such-sidecar-program")),
            "{err:?}"
        );
    }

    #[test]
    fn rule_and_mock_backends_resolve() {
        use crate::traits::CorefMention;

        let rule = ModelRegistry::new("p", coref_models(&json!({"backend": "rule"})));
        let coref = rule.coref("p_coref").expect("rule backend");
        let mentions = [
            CorefMention { start: 0, end: 11, text: "Maria Lopez", entity_type: "PER" },
            CorefMention { start: 17, end: 22, text: "Lopez", entity_type: "PER" },
        ];
        let clusters = coref
            .resolve_with_mentions("Maria Lopez said Lopez left.", &mentions)
            .expect("rule clusters");
        assert_eq!(clusters, vec![crate::CorefCluster { spans: vec![(0, 11), (17, 22)] }]);

        let mock = ModelRegistry::new("p", coref_models(&json!({"backend": "mock"})));
        let coref = mock.coref("p_coref").expect("mock backend");
        assert!(coref
            .resolve_with_mentions("Maria Lopez said Lopez left.", &mentions)
            .expect("mock clusters")
            .is_empty());
    }

    #[test]
    fn llm_backend_uses_the_registry_llm() {
        let registry = ModelRegistry::new("p", coref_models(&json!({"backend": "llm"})));
        let llm = Arc::new(MockLlm::with_responses([r#"{"clusters": [[[0, 5], [10, 13]]]}"#]));
        registry.insert_llm("p_ollama", llm.clone());
        let coref = registry.coref("p_coref").expect("llm backend");
        let clusters = coref.resolve("Maria and she.").expect("llm clusters");
        assert_eq!(clusters, vec![crate::CorefCluster { spans: vec![(0, 5), (10, 13)] }]);
        assert_eq!(llm.prompts().len(), 1);
    }

    #[test]
    fn injected_sidecar_replaces_any_backend() {
        let registry = ModelRegistry::new("p", coref_models(&json!({"backend": "rule"})))
            .with_coref_sidecar(SidecarCommand::new("ce-models-no-such-sidecar-program"));
        assert!(matches!(registry.coref("p_coref"), Err(ModelError::Load { .. })));
    }

    #[cfg(not(feature = "fastembed"))]
    #[test]
    fn missing_feature_is_named() {
        let registry = ModelRegistry::new("p", models(true));
        let Err(err) = registry.embedder("p_bge") else {
            panic!("embedder without feature resolved");
        };
        assert!(
            matches!(err, ModelError::Unavailable { ref backend, .. } if backend == "fastembed"),
            "{err:?}"
        );
        // Failures are not cached; injection still works afterwards.
        registry.insert_embedder("p_bge", Arc::new(MockEmbedder::new(3)));
        assert_eq!(registry.embedder("p_bge").expect("injected").dimension(), 3);
    }

    #[cfg(not(feature = "ort-nli"))]
    #[test]
    fn missing_nli_feature_is_named() {
        let registry = ModelRegistry::new("p", models(true));
        assert!(matches!(
            registry.nli("p_deberta_nli"),
            Err(ModelError::Unavailable { ref backend, .. }) if backend == "ort-nli"
        ));
    }

    #[cfg(not(feature = "gliner2"))]
    #[test]
    fn missing_gliner2_feature_is_named() {
        let registry = ModelRegistry::new("p", models(true));
        assert!(matches!(
            registry.ner("p_gliner2"),
            Err(ModelError::Unavailable { ref backend, .. }) if backend == "gliner2"
        ));
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn ollama_client_is_built_lazily_once() {
        let registry = ModelRegistry::new("p", models(false));
        let first = registry.llm("p_ollama").expect("ollama client");
        let second = registry.llm("p_ollama").expect("cached client");
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[cfg(not(feature = "ollama"))]
    #[test]
    fn missing_ollama_feature_is_named() {
        let registry = ModelRegistry::new("p", models(false));
        assert!(matches!(
            registry.llm("p_ollama"),
            Err(ModelError::Unavailable { ref backend, .. }) if backend == "ollama"
        ));
    }

    #[test]
    fn slow_construction_does_not_block_other_keys() {
        use std::sync::mpsc;
        use std::time::Duration;

        let registry = Arc::new(ModelRegistry::empty("p"));
        registry.insert_embedder("p_injected", Arc::new(MockEmbedder::new(4)));
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel::<()>();

        let builder = {
            let registry = Arc::clone(&registry);
            std::thread::spawn(move || {
                registry.embedders.get_or_build("p_slow", || {
                    started_tx.send(()).expect("signal start");
                    // Waits until the concurrent lookup below has finished.
                    release_rx.recv_timeout(Duration::from_secs(10)).map_err(|_| {
                        ModelError::Unavailable {
                            backend: "slow".to_owned(),
                            reason: "never released".to_owned(),
                        }
                    })?;
                    Ok(Arc::new(MockEmbedder::new(8)) as Arc<dyn Embedder>)
                })
            })
        };

        started_rx.recv_timeout(Duration::from_secs(10)).expect("construction started");
        let injected = registry.embedder("p_injected").expect("injected lookup while building");
        assert_eq!(injected.dimension(), 4);
        release_tx.send(()).expect("release the builder");

        let slow = builder.join().expect("builder thread").expect("slow model");
        assert_eq!(slow.dimension(), 8);
        let cached = registry.embedder("p_slow").expect("cached slow model");
        assert!(Arc::ptr_eq(&slow, &cached));
    }
}
