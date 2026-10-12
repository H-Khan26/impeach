//! Strict pipeline configuration (`config/processing/*.yaml`).
//!
//! Field names, defaults and allowed values match the Python prototype's Pydantic models.
//! Every struct rejects unknown fields. Deserialisation checks types, required fields and
//! choice values; [`PipelineConfig::validate`] then reproduces the Pydantic validators
//! (ranges, cross-field rules) and reports the dotted path of the offending field.
//!
//! Integer fields that Python constrains to be non-negative use unsigned types, so a
//! negative value fails at deserialisation with its path instead of in `validate`.

use serde::{Deserialize, Deserializer};

use crate::de::{deserialize_tagged, parse_yaml, NoFields, TaggedUnion};
use crate::error::{ConfigError, Result};

/// Config versions this implementation accepts.
pub const SUPPORTED_CONFIG_VERSIONS: &[&str] = &["2.0"];

fn default_true() -> bool {
    true
}

// ============================================================================
// Root
// ============================================================================

/// Complete pipeline configuration.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineConfig {
    /// Must be one of [`SUPPORTED_CONFIG_VERSIONS`].
    pub config_version: String,
    /// Pipeline identification.
    pub pipeline: PipelineMeta,
    /// Stages in execution order; must not be empty.
    pub stages: Vec<StageConfig>,
    /// Model settings.
    pub models: ModelsConfig,
    /// HelixDB connection.
    pub helix: HelixConfig,
    /// RAG indexing.
    pub rag: RagConfig,
    /// Mapping file paths.
    pub mappings: MappingsPaths,
}

impl PipelineConfig {
    /// Parses and validates a pipeline config from YAML text; `origin` names the source in errors.
    pub fn from_yaml_str(text: &str, origin: &str) -> Result<Self> {
        let config: Self = parse_yaml(text, origin)?;
        config.validate()?;
        Ok(config)
    }

    /// Pipeline name, the prefix for model registry keys.
    pub fn name(&self) -> &str {
        &self.pipeline.name
    }

    /// Helix namespace.
    pub fn helix_namespace(&self) -> &str {
        &self.helix.namespace
    }

    /// First stage whose `type` equals `stage_type` (e.g. `"CoreferenceStage"`).
    pub fn get_stage_config(&self, stage_type: &str) -> Option<&StageConfig> {
        self.stages.iter().find(|stage| stage.type_name() == stage_type)
    }

    /// Reproduces the Pydantic validators; returns the first violation.
    pub fn validate(&self) -> Result<()> {
        if !SUPPORTED_CONFIG_VERSIONS.contains(&self.config_version.as_str()) {
            return Err(ConfigError::validation(
                "config_version",
                format!(
                    "Unsupported config_version '{}'. Supported: {}.",
                    self.config_version,
                    python_list(SUPPORTED_CONFIG_VERSIONS)
                ),
            ));
        }
        if self.stages.is_empty() {
            return Err(ConfigError::validation("stages", "stages must not be empty"));
        }
        for (index, stage) in self.stages.iter().enumerate() {
            stage.validate(&format!("stages[{index}]"))?;
        }
        self.models.coref.validate("models.coref")?;
        self.helix.validate("helix")?;
        self.rag.chunking.validate("rag.chunking")
    }
}

/// Basic pipeline identification (`pipeline:` block).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineMeta {
    /// Pipeline name.
    pub name: String,
}

// ============================================================================
// Models
// ============================================================================

/// All model configurations.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelsConfig {
    /// Ollama LLM.
    pub ollama: OllamaModelConfig,
    /// Coreference model.
    pub coref: CorefModelConfig,
    /// BGE embedding model.
    pub bge: BgeModelConfig,
    /// DeBERTa NLI cross-encoder.
    pub deberta_nli: NliModelConfig,
    /// GLiNER2 NER / relation model.
    #[serde(default)]
    pub gliner2: Option<Gliner2ModelConfig>,
}

/// Ollama LLM model.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OllamaModelConfig {
    /// Model tag, e.g. `qwen2.5:7b-instruct-q4_K_M`.
    pub model: String,
    /// Server URL.
    pub base_url: String,
    /// Sampling temperature.
    pub temperature: f64,
    /// Context window.
    pub num_ctx: u32,
}

/// Coreference backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorefBackend {
    /// fastcoref (`biu-nlp/f-coref`) run as a sidecar process started with
    /// [`CorefModelConfig::command`].
    #[default]
    Fastcoref,
    /// Deterministic name / alias matching and pronoun linking.
    Rule,
    /// Clusters returned by the `ollama` LLM as JSON.
    Llm,
    /// A resolver that returns no clusters (tests).
    Mock,
}

impl CorefBackend {
    /// The config value, e.g. `fastcoref`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fastcoref => "fastcoref",
            Self::Rule => "rule",
            Self::Llm => "llm",
            Self::Mock => "mock",
        }
    }
}

/// Coreference model.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorefModelConfig {
    /// Backend; defaults to `fastcoref`.
    #[serde(default)]
    pub backend: CorefBackend,
    /// Device, e.g. `cpu`.
    pub device: String,
    /// Optional model name or path (the service default is `biu-nlp/f-coref`).
    #[serde(default)]
    pub model: Option<String>,
    /// Sidecar command line of the `fastcoref` backend: program followed by its arguments.
    /// Not part of the Python schema; absent in the shared YAML files.
    #[serde(default)]
    pub command: Option<Vec<String>>,
}

impl CorefModelConfig {
    fn validate(&self, path: &str) -> Result<()> {
        if self.command.as_ref().is_some_and(|c| c.first().is_none_or(|p| p.trim().is_empty())) {
            return Err(field_error(path, "command", "command must name a program".into()));
        }
        Ok(())
    }
}

/// BGE embedding model.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BgeModelConfig {
    /// Model id.
    pub model: String,
    /// Device.
    pub device: String,
    /// L2-normalise embeddings.
    pub normalize_embeddings: bool,
    /// Embedding batch size.
    pub batch_size: usize,
}

/// DeBERTa NLI model.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NliModelConfig {
    /// Model id.
    pub model: String,
    /// Device.
    pub device: String,
}

/// GLiNER2 model.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gliner2ModelConfig {
    /// Model id.
    pub model: String,
    /// Device.
    pub device: String,
    /// Score threshold.
    pub threshold: f64,
}

// ============================================================================
// Mention detectors
// ============================================================================

/// A MentionStage detector, discriminated by `type`.
#[derive(Debug, Clone, PartialEq)]
pub enum MentionDetector {
    /// `type: eyecite` (legal citations).
    Eyecite,
    /// `type: regex` (detection_patterns.yaml).
    Regex,
    /// `type: gliner2`.
    Gliner2(Gliner2DetectorConfig),
}

impl MentionDetector {
    /// The `type` value.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Eyecite => "eyecite",
            Self::Regex => "regex",
            Self::Gliner2(_) => "gliner2",
        }
    }
}

impl TaggedUnion for MentionDetector {
    const VARIANTS: &'static [&'static str] = &["regex", "eyecite", "gliner2"];

    fn from_tagged<'de, D: Deserializer<'de>>(tag: &str, body: D) -> Result<Self, D::Error> {
        match tag {
            "eyecite" => NoFields::deserialize(body).map(|_| Self::Eyecite),
            "regex" => NoFields::deserialize(body).map(|_| Self::Regex),
            "gliner2" => Gliner2DetectorConfig::deserialize(body).map(Self::Gliner2),
            other => Err(serde::de::Error::unknown_variant(other, Self::VARIANTS)),
        }
    }
}

impl<'de> Deserialize<'de> for MentionDetector {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserialize_tagged(deserializer)
    }
}

/// GLiNER2 detector settings.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gliner2DetectorConfig {
    /// Model registry key suffix (`{pipeline}_{suffix}`).
    pub model_name_suffix: String,
    /// Minimum mention score.
    pub min_score: f64,
    /// Entity label set name from `gliner2.yaml`.
    pub label_set: String,
}

// ============================================================================
// Stages
// ============================================================================

/// A pipeline stage, discriminated by `type`.
#[derive(Debug, Clone, PartialEq)]
pub enum StageConfig {
    /// `RAGIndexStage`.
    RagIndex(RagIndexStageConfig),
    /// `MentionStage`.
    Mention(MentionStageConfig),
    /// `DefinitionStage`.
    Definition(DefinitionStageConfig),
    /// `CoreferenceStage`.
    Coreference(CoreferenceStageConfig),
    /// `EDStage`.
    Ed(EdStageConfig),
    /// `RelationStage`.
    Relation(RelationStageConfig),
    /// `FactStage`.
    Fact(FactStageConfig),
    /// `ContradictionCheckStage`.
    ContradictionCheck(ContradictionCheckStageConfig),
}

impl StageConfig {
    /// The `type` value, e.g. `"EDStage"`.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::RagIndex(_) => "RAGIndexStage",
            Self::Mention(_) => "MentionStage",
            Self::Definition(_) => "DefinitionStage",
            Self::Coreference(_) => "CoreferenceStage",
            Self::Ed(_) => "EDStage",
            Self::Relation(_) => "RelationStage",
            Self::Fact(_) => "FactStage",
            Self::ContradictionCheck(_) => "ContradictionCheckStage",
        }
    }

    /// Stage instance name.
    pub fn name(&self) -> &str {
        match self {
            Self::RagIndex(c) => &c.name,
            Self::Mention(c) => &c.name,
            Self::Definition(c) => &c.name,
            Self::Coreference(c) => &c.name,
            Self::Ed(c) => &c.name,
            Self::Relation(c) => &c.name,
            Self::Fact(c) => &c.name,
            Self::ContradictionCheck(c) => &c.name,
        }
    }

    /// Whether the stage runs.
    pub fn enabled(&self) -> bool {
        match self {
            Self::RagIndex(c) => c.enabled,
            Self::Mention(c) => c.enabled,
            Self::Definition(c) => c.enabled,
            Self::Coreference(c) => c.enabled,
            Self::Ed(c) => c.enabled,
            Self::Relation(c) => c.enabled,
            Self::Fact(c) => c.enabled,
            Self::ContradictionCheck(c) => c.enabled,
        }
    }

    fn validate(&self, path: &str) -> Result<()> {
        match self {
            Self::Mention(c) => c.validate(path),
            Self::Ed(c) => c.validate(path),
            Self::Fact(c) => c.validate(path),
            Self::ContradictionCheck(c) => c.validate(path),
            Self::RagIndex(_) | Self::Definition(_) | Self::Coreference(_) | Self::Relation(_) => {
                Ok(())
            }
        }
    }
}

impl TaggedUnion for StageConfig {
    const VARIANTS: &'static [&'static str] = &[
        "RAGIndexStage",
        "MentionStage",
        "DefinitionStage",
        "CoreferenceStage",
        "EDStage",
        "FactStage",
        "RelationStage",
        "ContradictionCheckStage",
    ];

    fn from_tagged<'de, D: Deserializer<'de>>(tag: &str, body: D) -> Result<Self, D::Error> {
        match tag {
            "RAGIndexStage" => RagIndexStageConfig::deserialize(body).map(Self::RagIndex),
            "MentionStage" => MentionStageConfig::deserialize(body).map(Self::Mention),
            "DefinitionStage" => DefinitionStageConfig::deserialize(body).map(Self::Definition),
            "CoreferenceStage" => CoreferenceStageConfig::deserialize(body).map(Self::Coreference),
            "EDStage" => EdStageConfig::deserialize(body).map(Self::Ed),
            "RelationStage" => RelationStageConfig::deserialize(body).map(Self::Relation),
            "FactStage" => FactStageConfig::deserialize(body).map(Self::Fact),
            "ContradictionCheckStage" => {
                ContradictionCheckStageConfig::deserialize(body).map(Self::ContradictionCheck)
            }
            other => Err(serde::de::Error::unknown_variant(other, Self::VARIANTS)),
        }
    }
}

impl<'de> Deserialize<'de> for StageConfig {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserialize_tagged(deserializer)
    }
}

/// `RAGIndexStage`: chunk, embed and index documents.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RagIndexStageConfig {
    /// Stage instance name.
    pub name: String,
    /// Whether the stage runs.
    pub enabled: bool,
    /// Skip network-dependent work.
    pub offline: bool,
    /// Reuse a cached Helix Document node for the same document; off by default.
    #[serde(default)]
    pub use_document_cache: bool,
}

/// `MentionStage`: mention detection and quality filtering.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MentionStageConfig {
    /// Stage instance name.
    pub name: String,
    /// Whether the stage runs.
    pub enabled: bool,
    /// Minimum mention length in characters.
    pub min_mention_length: usize,
    /// Score threshold for short mentions.
    pub short_confidence_threshold: f64,
    /// Drop truncated words from chunk boundaries.
    pub filter_partial_words: bool,
    /// Collapse whitespace in mention names.
    pub normalize_names: bool,
    /// Detectors to run; must not be empty.
    pub detectors: Vec<MentionDetector>,
}

impl MentionStageConfig {
    fn validate(&self, path: &str) -> Result<()> {
        if self.detectors.is_empty() {
            return Err(field_error(path, "detectors", "detectors must not be empty".into()));
        }
        Ok(())
    }
}

/// `CoreferenceStage`: coreference clusters and pronoun mentions.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoreferenceStageConfig {
    /// Stage instance name.
    pub name: String,
    /// Whether the stage runs.
    pub enabled: bool,
    /// Model registry key suffix.
    pub model_name_suffix: String,
    /// Maximum document length passed to the model.
    pub max_doc_chars: usize,
    /// Mention types eligible for clustering.
    pub eligible_types: Vec<String>,
    /// Minimum span overlap to attach a mention to a cluster span.
    pub min_overlap_fraction: f64,
    /// Maximum mentions attached per cluster span.
    pub max_mentions_per_span: usize,
    /// Jaccard threshold for merging clusters.
    pub cluster_jaccard_threshold: f64,
    /// Merge clusters that overlap.
    pub merge_overlapping_clusters: bool,
    /// Mutually incompatible type groups.
    pub incompatible_type_groups: Vec<Vec<String>>,
    /// Maximum type groups allowed in one cluster.
    pub max_type_groups_per_cluster: usize,
    /// Drop clusters mixing type groups.
    pub drop_mixed_type_clusters: bool,
    /// Synthesise pronoun mentions; defaults to true.
    #[serde(default = "default_true")]
    pub synthesize_pronoun_mentions: bool,
    /// Score for synthesised pronoun mentions; defaults to 0.85.
    #[serde(default = "default_pronoun_confidence")]
    pub pronoun_confidence: f64,
}

fn default_pronoun_confidence() -> f64 {
    0.85
}

/// `DefinitionStage`: defined terms such as `("Plaintiff")`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DefinitionStageConfig {
    /// Stage instance name.
    pub name: String,
    /// Whether the stage runs.
    pub enabled: bool,
    /// Custom path to `roles.yaml`.
    #[serde(default)]
    pub role_config_path: Option<String>,
}

/// ED candidate retrieval strategy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityRetrievalStrategy {
    /// Helix BM25 over entity names and aliases.
    #[default]
    Bm25,
    /// Vector search re-ranked with fuzzy name similarity.
    VectorFuzzy,
    /// Fuzzy similarity over all entities in the case.
    AllEntitiesFuzzy,
}

impl EntityRetrievalStrategy {
    /// YAML value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bm25 => "bm25",
            Self::VectorFuzzy => "vector_fuzzy",
            Self::AllEntitiesFuzzy => "all_entities_fuzzy",
        }
    }
}

/// `EDStage`: entity disambiguation.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EdStageConfig {
    /// Stage instance name.
    pub name: String,
    /// Whether the stage runs.
    pub enabled: bool,
    /// Allow the LLM tie-break.
    pub allow_llm: bool,
    /// Resolve every mention with the LLM.
    pub llm_only: bool,
    /// Create entities for unresolved mentions.
    pub create_if_missing: bool,
    /// Maximum candidates per mention.
    pub max_candidates: usize,
    /// Match mentions within the same document.
    pub enable_intra_doc_matching: bool,
    /// Minimum embedding similarity, in [0, 1].
    pub min_embedding_similarity: f64,
    /// Hybrid resolution threshold, in [0, 1].
    pub hybrid_similarity_threshold: f64,
    /// Entity reuse threshold, in [0, 1].
    pub reuse_similarity_threshold: f64,
    /// Use similarity retrieval for candidates.
    pub enable_similarity_retrieval: bool,
    /// Boost for cross-document reuse, in [0, 0.2].
    pub cross_doc_reuse_boost: f64,
    /// ED prompt file.
    pub prompt_file: String,
    /// Candidate retrieval strategy; defaults to `bm25`.
    #[serde(default)]
    pub entity_retrieval_strategy: EntityRetrievalStrategy,
    /// Vector over-fetch multiplier, in [1, 10]; defaults to 3.
    #[serde(default = "default_overfetch")]
    pub vector_fuzzy_overfetch_multiplier: usize,
    /// Minimum combined vector/fuzzy similarity, in [0, 1]; defaults to 0.5.
    #[serde(default = "default_vector_fuzzy_min_similarity")]
    pub vector_fuzzy_min_similarity: f64,
    /// Vector weight, in [0, 1]; defaults to 0.4.
    #[serde(default = "default_vector_weight")]
    pub vector_fuzzy_vector_weight: f64,
    /// Fuzzy weight, in [0, 1]; defaults to 0.6. Weights must sum to 1 (±0.01).
    #[serde(default = "default_fuzzy_weight")]
    pub vector_fuzzy_fuzzy_weight: f64,
}

fn default_overfetch() -> usize {
    3
}
fn default_vector_fuzzy_min_similarity() -> f64 {
    0.5
}
fn default_vector_weight() -> f64 {
    0.4
}
fn default_fuzzy_weight() -> f64 {
    0.6
}

impl EdStageConfig {
    fn validate(&self, path: &str) -> Result<()> {
        let v = self.min_embedding_similarity;
        if !(0.0..=1.0).contains(&v) {
            return Err(field_error(
                path,
                "min_embedding_similarity",
                format!("min_embedding_similarity must be between 0 and 1, got {v:?}"),
            ));
        }
        for (field, v) in [
            ("hybrid_similarity_threshold", self.hybrid_similarity_threshold),
            ("reuse_similarity_threshold", self.reuse_similarity_threshold),
        ] {
            if !(0.0..=1.0).contains(&v) {
                return Err(field_error(
                    path,
                    field,
                    format!("Similarity threshold must be between 0 and 1, got {v:?}"),
                ));
            }
        }
        let v = self.cross_doc_reuse_boost;
        if !(0.0..=0.2).contains(&v) {
            return Err(field_error(
                path,
                "cross_doc_reuse_boost",
                format!("cross_doc_reuse_boost must be between 0 and 0.2, got {v:?}"),
            ));
        }
        let m = self.vector_fuzzy_overfetch_multiplier;
        if m < 1 {
            return Err(field_error(
                path,
                "vector_fuzzy_overfetch_multiplier",
                format!("vector_fuzzy_overfetch_multiplier must be >= 1, got {m}"),
            ));
        }
        if m > 10 {
            return Err(field_error(
                path,
                "vector_fuzzy_overfetch_multiplier",
                format!("vector_fuzzy_overfetch_multiplier must be <= 10, got {m}"),
            ));
        }
        for (field, v) in [
            ("vector_fuzzy_min_similarity", self.vector_fuzzy_min_similarity),
            ("vector_fuzzy_vector_weight", self.vector_fuzzy_vector_weight),
            ("vector_fuzzy_fuzzy_weight", self.vector_fuzzy_fuzzy_weight),
        ] {
            if !(0.0..=1.0).contains(&v) {
                return Err(field_error(
                    path,
                    field,
                    format!("vector_fuzzy_* values must be between 0 and 1, got {v:?}"),
                ));
            }
        }
        let total = self.vector_fuzzy_vector_weight + self.vector_fuzzy_fuzzy_weight;
        if !(0.99..=1.01).contains(&total) {
            return Err(ConfigError::validation(
                path,
                format!(
                    "vector_fuzzy_vector_weight + vector_fuzzy_fuzzy_weight must sum to 1.0 \
                     (within tolerance), got {total:?}"
                ),
            ));
        }
        Ok(())
    }
}

/// Fact extraction mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionMode {
    /// The only mode for processing pipelines.
    #[default]
    Processing,
}

/// `FactStage`: LLM fact extraction. One prompt file means single-pass extraction,
/// several mean multi-pass extraction.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactStageConfig {
    /// Stage instance name.
    pub name: String,
    /// Whether the stage runs.
    pub enabled: bool,
    /// Always `processing`.
    #[serde(default)]
    pub extraction_mode: ExtractionMode,
    /// Commit facts directly (skipping the contradiction check).
    pub commit_facts: bool,
    /// Prompt files; must not be empty.
    pub prompt_files: Vec<String>,
    /// Run multi-pass extraction in parallel; defaults to true.
    #[serde(default = "default_true")]
    pub parallel: bool,
    /// Worker count for parallel extraction; defaults to 4.
    #[serde(default = "default_max_workers")]
    pub max_workers: usize,
    /// LLM backend key; defaults to `ollama`.
    #[serde(default = "default_model_type")]
    pub model_type: String,
    /// Replace resolved mentions with `[entity_id:canonical_name]` tokens.
    #[serde(default)]
    pub use_resolved_text_view: bool,
}

fn default_max_workers() -> usize {
    4
}
fn default_model_type() -> String {
    "ollama".to_owned()
}

impl FactStageConfig {
    fn validate(&self, path: &str) -> Result<()> {
        if self.prompt_files.is_empty() {
            return Err(field_error(path, "prompt_files", "prompt_files must not be empty".into()));
        }
        Ok(())
    }
}

/// `RelationStage`: GLiNER2 relation extraction.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelationStageConfig {
    /// Stage instance name.
    pub name: String,
    /// Whether the stage runs.
    pub enabled: bool,
    /// Relation label set name from `gliner2.yaml`.
    pub relation_label_set: String,
    /// Model registry key suffix.
    pub model_name_suffix: String,
}

/// Fact retrieval strategy for the contradiction check.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactRetrievalStrategy {
    /// BM25 only.
    Bm25,
    /// Vector similarity only.
    Vector,
    /// BM25 and vector.
    #[default]
    Both,
}

impl FactRetrievalStrategy {
    /// YAML value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bm25 => "bm25",
            Self::Vector => "vector",
            Self::Both => "both",
        }
    }
}

/// `retrieval:` block of the contradiction check.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalConfig {
    /// BM25 result limit.
    pub bm25_limit: usize,
    /// Vector result limit.
    pub vector_limit: usize,
    /// Maximum candidates after ranking.
    pub max_candidates: usize,
    /// Weight for graph (shared entity) hits.
    pub graph_weight: f64,
    /// Weight for BM25 hits.
    pub bm25_weight: f64,
    /// Weight for vector hits.
    pub vector_weight: f64,
    /// Boost for candidates found by several methods.
    pub multi_source_boost: f64,
    /// Minimum ranked score, in [0, 1]; defaults to 0.
    #[serde(default)]
    pub min_score: f64,
    /// Retrieval strategy; defaults to `both`.
    #[serde(default)]
    pub fact_retrieval_strategy: FactRetrievalStrategy,
}

impl RetrievalConfig {
    fn validate(&self, path: &str) -> Result<()> {
        let v = self.min_score;
        if !(0.0..=1.0).contains(&v) {
            return Err(field_error(
                path,
                "min_score",
                format!("min_score must be between 0 and 1, got {v:?}"),
            ));
        }
        Ok(())
    }
}

/// `pre_filter:` block (dual-path gating).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreFilterConfig {
    /// Jaccard threshold for the text-overlap fallback, in [0, 1].
    pub min_text_overlap: f64,
    /// Skip pairs without shared entities.
    pub require_entity_overlap: bool,
    /// Enable the text-overlap fallback.
    pub enable_text_overlap_fallback: bool,
    /// Enable the semantic-similarity fallback.
    pub use_semantic_similarity_fallback: bool,
    /// Semantic fallback threshold, in [0, 1].
    pub semantic_similarity_threshold: f64,
    /// Path A subject overlap threshold, in [0, 1].
    pub subject_overlap_threshold: f64,
    /// Path A predicate similarity threshold, in [0, 1].
    pub predicate_similarity_threshold: f64,
    /// Path B fuzzy match threshold, in [0, 1].
    pub fuzzy_match_threshold: f64,
    /// Prioritise pairs whose subject/object line up.
    pub use_assertion_boost: bool,
}

impl PreFilterConfig {
    fn validate(&self, path: &str) -> Result<()> {
        for (field, v) in [
            ("min_text_overlap", self.min_text_overlap),
            ("semantic_similarity_threshold", self.semantic_similarity_threshold),
            ("subject_overlap_threshold", self.subject_overlap_threshold),
            ("predicate_similarity_threshold", self.predicate_similarity_threshold),
            ("fuzzy_match_threshold", self.fuzzy_match_threshold),
        ] {
            if !(0.0..=1.0).contains(&v) {
                return Err(field_error(
                    path,
                    field,
                    format!("threshold must be between 0 and 1, got {v:?}"),
                ));
            }
        }
        Ok(())
    }
}

/// `intra_doc_filter:` block.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntraDocFilterConfig {
    /// Jaccard threshold for intra-document pairs.
    pub min_text_overlap: f64,
    /// Only compare facts whose subject or object match.
    pub require_assertion_overlap: bool,
}

/// `nli:` block.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NliConfig {
    /// Model registry key suffix.
    pub model_name_suffix: String,
    /// Definite contradiction threshold, in (0, 1).
    pub contradiction_threshold: f64,
    /// Needs-review threshold, in (0, 1).
    pub review_threshold: f64,
}

impl NliConfig {
    fn validate(&self, path: &str) -> Result<()> {
        for (field, v) in [
            ("contradiction_threshold", self.contradiction_threshold),
            ("review_threshold", self.review_threshold),
        ] {
            if !(v > 0.0 && v < 1.0) {
                return Err(field_error(
                    path,
                    field,
                    format!("threshold must be between 0 and 1, got {v:?}"),
                ));
            }
        }
        Ok(())
    }
}

/// `intra_document:` block.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntraDocumentConfig {
    /// Check facts within the same document.
    pub enabled: bool,
    /// Commit clean facts while iterating instead of at the end.
    pub commit_as_we_go: bool,
}

/// `review_store:` block (SQLite review queue).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContradictionReviewStoreConfig {
    /// SQLite database path.
    pub db_path: String,
    /// Write review items; defaults to true.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Fail the stage on SQLite errors; defaults to true.
    #[serde(default = "default_true")]
    pub fail_on_sqlite_error: bool,
}

/// Contradiction judge model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeModel {
    /// DeBERTa NLI cross-encoder.
    Nli,
    /// Qwen LLM judge.
    Qwen,
}

/// Pair gating strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GatingStrategy {
    /// Path A / Path B OR logic.
    DualPath,
    /// Hard filters plus permissive triggers.
    Permissive,
}

/// `verification:` block.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationStrategyConfig {
    /// Judge model.
    pub judge_model: JudgeModel,
    /// Gating strategy.
    pub gating_strategy: GatingStrategy,
    /// Let TEXT values pair with any value type (permissive); defaults to true.
    #[serde(default = "default_true")]
    pub allow_text_value_type_mismatch: bool,
    /// Require matching subject entity ids (permissive); defaults to false.
    #[serde(default)]
    pub require_entity_id_match: bool,
    /// Embedding similarity trigger, in [0, 1]; defaults to 0.60.
    #[serde(default = "default_semantic_proximity")]
    pub semantic_proximity_threshold: f64,
    /// Token Jaccard trigger, in [0, 1]; defaults to 0.30.
    #[serde(default = "default_keyword_overlap")]
    pub keyword_overlap_threshold: f64,
    /// Qwen judge prompt file.
    #[serde(default)]
    pub qwen_prompt_file: Option<String>,
    /// Qwen batch size, in [1, 50]; defaults to 5.
    #[serde(default = "default_qwen_batch_size")]
    pub qwen_batch_size: usize,
}

fn default_semantic_proximity() -> f64 {
    0.60
}
fn default_keyword_overlap() -> f64 {
    0.30
}
fn default_qwen_batch_size() -> usize {
    5
}

impl VerificationStrategyConfig {
    fn validate(&self, path: &str) -> Result<()> {
        for (field, v) in [
            ("semantic_proximity_threshold", self.semantic_proximity_threshold),
            ("keyword_overlap_threshold", self.keyword_overlap_threshold),
        ] {
            if !(0.0..=1.0).contains(&v) {
                return Err(field_error(
                    path,
                    field,
                    format!("threshold must be between 0 and 1, got {v:?}"),
                ));
            }
        }
        let b = self.qwen_batch_size;
        if !(1..=50).contains(&b) {
            return Err(field_error(
                path,
                "qwen_batch_size",
                format!("qwen_batch_size must be between 1 and 50, got {b}"),
            ));
        }
        Ok(())
    }
}

/// `ContradictionCheckStage`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContradictionCheckStageConfig {
    /// Stage instance name.
    pub name: String,
    /// Whether the stage runs.
    pub enabled: bool,
    /// Similar-fact retrieval.
    pub retrieval: RetrievalConfig,
    /// Dual-path pre-filter.
    pub pre_filter: PreFilterConfig,
    /// Intra-document pre-filter.
    pub intra_doc_filter: IntraDocFilterConfig,
    /// NLI judge.
    pub nli: NliConfig,
    /// Intra-document checking.
    pub intra_document: IntraDocumentConfig,
    /// SQLite review store.
    #[serde(default)]
    pub review_store: Option<ContradictionReviewStoreConfig>,
    /// Verification strategy; `None` means NLI with dual-path gating.
    #[serde(default)]
    pub verification: Option<VerificationStrategyConfig>,
}

impl ContradictionCheckStageConfig {
    fn validate(&self, path: &str) -> Result<()> {
        self.retrieval.validate(&format!("{path}.retrieval"))?;
        self.pre_filter.validate(&format!("{path}.pre_filter"))?;
        self.nli.validate(&format!("{path}.nli"))?;
        if let Some(verification) = &self.verification {
            verification.validate(&format!("{path}.verification"))?;
        }
        Ok(())
    }
}

// ============================================================================
// Infrastructure
// ============================================================================

/// HelixDB mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HelixMode {
    /// On-disk database under `data_dir`.
    #[default]
    Embedded,
    /// In-process database discarded at exit.
    Memory,
    /// Remote server at `url`.
    Server,
}

impl HelixMode {
    /// YAML value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Embedded => "embedded",
            Self::Memory => "memory",
            Self::Server => "server",
        }
    }
}

/// `helix:` block.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HelixConfig {
    /// Database mode; defaults to `embedded`.
    #[serde(default)]
    pub mode: HelixMode,
    /// Data directory for `embedded` mode; defaults to `data/helix`.
    #[serde(default = "default_data_dir")]
    pub data_dir: String,
    /// Server URL; required in `server` mode, ignored otherwise.
    #[serde(default)]
    pub url: Option<String>,
    /// Database name inside `data_dir`; must not be blank.
    pub namespace: String,
    /// API key for `server` mode; defaults to empty.
    #[serde(default)]
    pub api_key: String,
    /// Write batch size.
    pub batch_size: usize,
    /// Embedding dimension; must be positive.
    pub embedding_dim: usize,
}

fn default_data_dir() -> String {
    "data/helix".to_owned()
}

impl HelixConfig {
    fn validate(&self, path: &str) -> Result<()> {
        if let Some(url) = &self.url {
            validate_helix_url(url).map_err(|message| field_error(path, "url", message))?;
        }
        if self.namespace.trim().is_empty() {
            return Err(field_error(path, "namespace", "helix.namespace must not be empty".into()));
        }
        if self.embedding_dim == 0 {
            return Err(field_error(
                path,
                "embedding_dim",
                "helix.embedding_dim must be positive".into(),
            ));
        }
        let has_url = self.url.as_deref().is_some_and(|u| !u.is_empty());
        if self.mode == HelixMode::Server && !has_url {
            return Err(ConfigError::validation(
                path,
                "helix.url is required when helix.mode is 'server'",
            ));
        }
        if self.mode == HelixMode::Embedded && self.data_dir.trim().is_empty() {
            return Err(ConfigError::validation(
                path,
                "helix.data_dir must not be empty when helix.mode is 'embedded'",
            ));
        }
        if has_url && self.mode != HelixMode::Server {
            tracing::warn!(
                "helix.url is ignored in {} mode (only used when helix.mode is 'server')",
                self.mode.as_str()
            );
        }
        Ok(())
    }
}

/// Checks a Helix URL the way the Python validator does with `urllib.parse.urlparse`.
fn validate_helix_url(url: &str) -> Result<(), String> {
    if !url.starts_with("http") {
        return Err(format!("helix.url must be a valid HTTP URL, got: {url}"));
    }
    let (scheme, rest) = match url.split_once(':') {
        Some((scheme, rest))
            if scheme.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
                && scheme.chars().all(|c| c.is_ascii_alphanumeric() || "+-.".contains(c)) =>
        {
            (scheme.to_ascii_lowercase(), rest)
        }
        _ => (String::new(), url),
    };
    if scheme != "http" && scheme != "https" {
        return Err(format!("helix.url must have http or https scheme, got: {url}"));
    }
    let netloc = rest
        .strip_prefix("//")
        .map(|after| after.split(['/', '?', '#']).next().unwrap_or(""))
        .unwrap_or("");
    let host_port = netloc.rsplit_once('@').map_or(netloc, |(_, hp)| hp);
    let (hostname, port) = if let Some(bracketed) = host_port.strip_prefix('[') {
        match bracketed.split_once(']') {
            Some((host, after)) => (host, after.strip_prefix(':')),
            None => (bracketed, None),
        }
    } else {
        match host_port.split_once(':') {
            Some((host, port)) => (host, Some(port)),
            None => (host_port, None),
        }
    };
    if hostname.is_empty() {
        return Err(format!("helix.url must include a hostname, got: {url}"));
    }
    if let Some(port) = port.filter(|p| !p.is_empty()) {
        let valid = port.chars().all(|c| c.is_ascii_digit())
            && port.parse::<u32>().is_ok_and(|p| p <= 65_535);
        if !valid {
            return Err(format!("helix.url has an invalid port, got: {url}"));
        }
    }
    Ok(())
}

/// Chunking strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChunkingStrategy {
    /// Sentence-based chunks.
    Sentence,
    /// Recursive splitting.
    Recursive,
}

/// `rag.chunking:` block.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkingConfig {
    /// Chunking strategy.
    pub strategy: ChunkingStrategy,
    /// Chunk size in characters; must be positive.
    pub chunk_size: usize,
    /// Overlap between chunks in characters.
    pub chunk_overlap: usize,
    /// Emit sentence spans with each chunk.
    pub include_sentence_spans: bool,
}

impl ChunkingConfig {
    fn validate(&self, path: &str) -> Result<()> {
        if self.chunk_size == 0 {
            return Err(field_error(
                path,
                "chunk_size",
                format!("chunk_size must be > 0, got {}", self.chunk_size),
            ));
        }
        Ok(())
    }
}

/// `rag:` block. Embedding settings come from `models.bge`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RagConfig {
    /// Chunking settings.
    pub chunking: ChunkingConfig,
}

/// `mappings:` block (paths to mapping files).
///
/// Parsed and required for YAML compatibility only; nothing reads these paths. The Python
/// prototype consults them only as a fallback when its mappings registry fails to load, and here
/// a registry load failure is an error. Mappings come from [`crate::MappingsRegistry`], which
/// always loads the ten files from `config/mappings/`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MappingsPaths {
    /// Entity type mappings (`entity_types.yaml`).
    pub entity_mappings: String,
    /// Domain mappings (`entity_types.yaml`).
    pub domain_mappings: String,
    /// Relationship types (`relationship_types.yaml`).
    pub relationship_types: String,
}

fn field_error(path: &str, field: &str, message: String) -> ConfigError {
    ConfigError::validation(format!("{path}.{field}"), message)
}

/// Formats a list of strings like Python's `repr(list)`.
fn python_list(items: &[&str]) -> String {
    let quoted: Vec<String> = items.iter().map(|s| format!("'{s}'")).collect();
    format!("[{}]", quoted.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helix_url_rules_follow_urlparse() {
        for ok in [
            "http://localhost:6969",
            "https://helix.example.com",
            "http://user:pw@host:80/path",
            "http://[::1]:8080",
            "http://host:",
        ] {
            assert_eq!(validate_helix_url(ok), Ok(()), "{ok}");
        }
        let cases = [
            ("ftp://host", "must be a valid HTTP URL"),
            ("httpx://host", "http or https scheme"),
            ("http//host", "http or https scheme"),
            ("http://", "must include a hostname"),
            ("http:///path", "must include a hostname"),
            ("http://host:abc", "invalid port"),
            ("http://host:70000", "invalid port"),
        ];
        for (url, expected) in cases {
            let err = validate_helix_url(url).expect_err(url);
            assert!(err.contains(expected), "{url}: {err}");
        }
    }

    #[test]
    fn python_list_matches_repr() {
        assert_eq!(python_list(&["2.0"]), "['2.0']");
        assert_eq!(python_list(&["a", "b"]), "['a', 'b']");
    }
}
