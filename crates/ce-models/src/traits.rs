//! Model traits and their input / output records.
//!
//! All text offsets are character offsets (Unicode scalar values, Python `str` indexing);
//! backends that work in bytes or tokens convert at their boundary.

use serde::{Deserialize, Serialize};

use crate::error::ModelError;

/// Scales `vector` to unit L2 norm in place; a zero vector is left unchanged. The norm and
/// the division are computed in `f64`.
pub fn normalize_l2(vector: &mut [f32]) {
    let norm = vector.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in vector.iter_mut() {
            *x = (f64::from(*x) / norm) as f32;
        }
    }
}

/// Sentence embedding model.
pub trait Embedder: Send + Sync {
    /// Embeds each text; the output has one vector of [`Embedder::dimension`] values per input.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, ModelError>;
    /// Output vector length.
    fn dimension(&self) -> usize;
    /// Model identifier, e.g. `BAAI/bge-small-en-v1.5`.
    fn model_name(&self) -> &str;
}

/// Softmax scores of one NLI premise / hypothesis pair.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct NliScores {
    /// Probability of contradiction.
    pub contradiction: f32,
    /// Probability of entailment.
    pub entailment: f32,
    /// Probability of neutral.
    pub neutral: f32,
}

/// Natural language inference cross-encoder.
pub trait NliJudge: Send + Sync {
    /// Scores each `(premise, hypothesis)` pair.
    fn score_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<NliScores>, ModelError>;
}

/// One LLM completion request.
#[derive(Debug, Clone, Copy, Default)]
pub struct LlmRequest<'a> {
    /// Optional system prompt.
    pub system: Option<&'a str>,
    /// User prompt.
    pub prompt: &'a str,
    /// JSON schema the output must follow (structured output), if any.
    pub json_schema: Option<&'a serde_json::Value>,
    /// Sampling temperature; `None` uses the backend's configured value.
    pub temperature: Option<f32>,
    /// Context window; `None` uses the backend's configured value.
    pub num_ctx: Option<u32>,
}

impl<'a> LlmRequest<'a> {
    /// A request with only a user prompt.
    pub fn new(prompt: &'a str) -> Self {
        Self { prompt, ..Self::default() }
    }
}

/// Large language model.
pub trait Llm: Send + Sync {
    /// Returns the raw completion text.
    fn complete(&self, req: &LlmRequest<'_>) -> Result<String, ModelError>;

    /// Returns the completion parsed as JSON. The default implementation calls
    /// [`Llm::complete`] and extracts JSON leniently with [`crate::json::extract_json`].
    fn complete_json(&self, req: &LlmRequest<'_>) -> Result<serde_json::Value, ModelError> {
        let text = self.complete(req)?;
        crate::json::extract_json(&text)
    }
}

/// An entity span found by a NER model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NerSpan {
    /// Start character offset (inclusive).
    pub start: usize,
    /// End character offset (exclusive).
    pub end: usize,
    /// Entity label as passed in `labels`.
    pub label: String,
    /// Model confidence.
    pub score: f32,
    /// Span text.
    pub text: String,
}

/// A relation between two entity spans.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelationHit {
    /// Head entity.
    pub head: NerSpan,
    /// Tail entity.
    pub tail: NerSpan,
    /// Relation label as passed in `labels`.
    pub label: String,
    /// Model confidence.
    pub score: f32,
}

/// Named-entity and relation extraction model.
pub trait NerModel: Send + Sync {
    /// Entity spans with one of `labels` scoring at least `threshold`.
    fn extract_entities(
        &self,
        text: &str,
        labels: &[&str],
        threshold: f32,
    ) -> Result<Vec<NerSpan>, ModelError>;
    /// Relations with one of `labels` scoring at least `threshold`.
    fn extract_relations(
        &self,
        text: &str,
        labels: &[&str],
        threshold: f32,
    ) -> Result<Vec<RelationHit>, ModelError>;
}

/// A coreference cluster: mention spans as `(start, end)` character offsets into the input.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorefCluster {
    /// Mention spans, end exclusive.
    pub spans: Vec<(usize, usize)>,
}

/// Pronouns flagged as such in coreference output (the Python `CorefService` list).
pub const PRONOUNS: [&str; 16] = [
    "he", "she", "they", "them", "him", "her", "his", "hers", "their", "theirs", "it", "its", "we",
    "us", "our", "ours",
];

/// Whether `text`, stripped of Python whitespace and lowercased, is one of [`PRONOUNS`]
/// (Python `CorefService`: `text.strip().lower() in pronouns`).
pub fn is_pronoun(text: &str) -> bool {
    let lowered = ce_text::normalize::py_strip(text).to_lowercase();
    PRONOUNS.contains(&lowered.as_str())
}

/// A detected mention handed to a coreference resolver: character offsets into the resolved
/// text, the surface text and the entity type label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorefMention<'a> {
    /// Start character offset (inclusive).
    pub start: usize,
    /// End character offset (exclusive).
    pub end: usize,
    /// Mention text.
    pub text: &'a str,
    /// Entity type label, e.g. `PER`.
    pub entity_type: &'a str,
}

/// Coreference resolver.
pub trait Coref: Send + Sync {
    /// Clusters of coreferent mentions in `text`.
    fn resolve(&self, text: &str) -> Result<Vec<CorefCluster>, ModelError>;

    /// Clusters of coreferent mentions in `text`, given the mentions detected in it (the
    /// CoreferenceStage passes the mentions of its eligible types, sorted by start). Resolvers
    /// that find their own mentions ignore `mentions`; the default calls [`Coref::resolve`].
    fn resolve_with_mentions(
        &self,
        text: &str,
        mentions: &[CorefMention<'_>],
    ) -> Result<Vec<CorefCluster>, ModelError> {
        let _ = mentions;
        self.resolve(text)
    }
}
