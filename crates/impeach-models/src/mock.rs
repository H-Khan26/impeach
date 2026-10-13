//! Deterministic mock models for tests. Always compiled.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Mutex;

use impeach_core::util::lock;

use crate::error::ModelError;
use crate::traits::{
    normalize_l2, Coref, CorefCluster, Embedder, Llm, LlmRequest, NerModel, NerSpan, NliJudge,
    NliScores, RelationHit,
};

/// FNV-1a 64-bit hash.
fn fnv1a(text: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// splitmix64 step.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    normalize_l2(&mut v);
    v
}

/// Embedder returning deterministic unit vectors derived from a hash of the text.
///
/// Equal texts get equal vectors; different texts get unrelated vectors. Vectors for
/// specific texts can be scripted with [`MockEmbedder::with_vector`].
#[derive(Debug)]
pub struct MockEmbedder {
    dimension: usize,
    name: String,
    scripted: HashMap<String, Vec<f32>>,
    calls: Mutex<usize>,
}

impl MockEmbedder {
    /// A mock embedder with `dimension`-length output.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            name: "mock-embedder".to_owned(),
            scripted: HashMap::new(),
            calls: Mutex::new(0),
        }
    }

    /// Returns `vector` (normalised to unit length) for exactly `text`.
    pub fn with_vector(mut self, text: impl Into<String>, vector: Vec<f32>) -> Self {
        self.scripted.insert(text.into(), normalize(vector));
        self
    }

    /// Number of `embed` calls so far.
    pub fn calls(&self) -> usize {
        *lock(&self.calls)
    }

    /// The unit vector returned for `text`.
    pub fn vector_for(&self, text: &str) -> Vec<f32> {
        if let Some(v) = self.scripted.get(text) {
            return v.clone();
        }
        let mut state = fnv1a(text);
        let raw = (0..self.dimension)
            .map(|_| {
                // Top 24 bits mapped to [-1, 1).
                let bits = (splitmix(&mut state) >> 40) as f32;
                bits / 8_388_608.0 - 1.0
            })
            .collect();
        normalize(raw)
    }
}

impl Embedder for MockEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, ModelError> {
        *lock(&self.calls) += 1;
        Ok(texts.iter().map(|t| self.vector_for(t)).collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.name
    }
}

type NliRule = Box<dyn Fn(&str, &str) -> Option<NliScores> + Send + Sync>;

/// NLI judge with scripted scores per `(premise, hypothesis)` pair, then rules (closures tried
/// in insertion order); other pairs get the default scores (neutral 1.0 unless changed).
/// Records every scored pair.
pub struct MockNli {
    scripted: HashMap<(String, String), NliScores>,
    rules: Vec<NliRule>,
    default: NliScores,
    calls: Mutex<Vec<(String, String)>>,
}

impl fmt::Debug for MockNli {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MockNli")
            .field("scripted", &self.scripted.len())
            .field("rules", &self.rules.len())
            .field("default", &self.default)
            .finish_non_exhaustive()
    }
}

impl Default for MockNli {
    fn default() -> Self {
        Self::new()
    }
}

impl MockNli {
    /// A judge scoring every pair as neutral (1.0).
    pub fn new() -> Self {
        Self {
            scripted: HashMap::new(),
            rules: Vec::new(),
            default: NliScores { contradiction: 0.0, entailment: 0.0, neutral: 1.0 },
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Scripts the scores of one pair.
    pub fn with_pair(
        mut self,
        premise: impl Into<String>,
        hypothesis: impl Into<String>,
        scores: NliScores,
    ) -> Self {
        self.scripted.insert((premise.into(), hypothesis.into()), scores);
        self
    }

    /// Adds a rule for unscripted pairs: the first rule returning `Some` decides the scores.
    pub fn with_rule<F>(mut self, rule: F) -> Self
    where
        F: Fn(&str, &str) -> Option<NliScores> + Send + Sync + 'static,
    {
        self.rules.push(Box::new(rule));
        self
    }

    /// Sets the scores of pairs no script or rule matches.
    pub fn with_default(mut self, scores: NliScores) -> Self {
        self.default = scores;
        self
    }

    /// Every `(premise, hypothesis)` pair scored so far, in call order.
    pub fn calls(&self) -> Vec<(String, String)> {
        lock(&self.calls).clone()
    }
}

impl NliJudge for MockNli {
    fn score_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<NliScores>, ModelError> {
        lock(&self.calls).extend(pairs.iter().map(|(p, h)| ((*p).to_owned(), (*h).to_owned())));
        Ok(pairs
            .iter()
            .map(|(p, h)| {
                self.scripted
                    .get(&((*p).to_owned(), (*h).to_owned()))
                    .copied()
                    .or_else(|| self.rules.iter().find_map(|rule| rule(p, h)))
                    .unwrap_or(self.default)
            })
            .collect())
    }
}

type LlmFn = Box<dyn Fn(&LlmRequest<'_>) -> Result<String, ModelError> + Send + Sync>;

/// LLM returning canned responses: a FIFO queue first, then the responder closure (if any).
/// Records every prompt it receives.
pub struct MockLlm {
    queue: Mutex<VecDeque<String>>,
    responder: Option<LlmFn>,
    prompts: Mutex<Vec<String>>,
}

impl fmt::Debug for MockLlm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MockLlm")
            .field("queued", &lock(&self.queue).len())
            .field("has_responder", &self.responder.is_some())
            .finish_non_exhaustive()
    }
}

impl Default for MockLlm {
    fn default() -> Self {
        Self::new()
    }
}

impl MockLlm {
    /// An LLM with no responses; calls fail with [`ModelError::Inference`].
    pub fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            responder: None,
            prompts: Mutex::new(Vec::new()),
        }
    }

    /// An LLM answering from `responses` in order.
    pub fn with_responses<I, S>(responses: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let llm = Self::new();
        lock(&llm.queue).extend(responses.into_iter().map(Into::into));
        llm
    }

    /// An LLM answering through `f` once the queue is empty.
    pub fn with_responder<F>(f: F) -> Self
    where
        F: Fn(&LlmRequest<'_>) -> Result<String, ModelError> + Send + Sync + 'static,
    {
        Self { responder: Some(Box::new(f)), ..Self::new() }
    }

    /// Appends a canned response.
    pub fn push_response(&self, response: impl Into<String>) {
        lock(&self.queue).push_back(response.into());
    }

    /// Prompts received so far.
    pub fn prompts(&self) -> Vec<String> {
        lock(&self.prompts).clone()
    }
}

impl Llm for MockLlm {
    fn complete(&self, req: &LlmRequest<'_>) -> Result<String, ModelError> {
        lock(&self.prompts).push(req.prompt.to_owned());
        let queued = lock(&self.queue).pop_front();
        if let Some(r) = queued {
            return Ok(r);
        }
        match &self.responder {
            Some(f) => f(req),
            None => Err(ModelError::Inference("MockLlm has no response queued".to_owned())),
        }
    }
}

/// NER model returning fixed spans and relations, filtered by label and threshold.
#[derive(Debug, Default)]
pub struct MockNer {
    entities: Vec<NerSpan>,
    relations: Vec<RelationHit>,
}

impl MockNer {
    /// A model returning `entities` and `relations` for every text.
    pub fn new(entities: Vec<NerSpan>, relations: Vec<RelationHit>) -> Self {
        Self { entities, relations }
    }
}

impl NerModel for MockNer {
    fn extract_entities(
        &self,
        _text: &str,
        labels: &[&str],
        threshold: f32,
    ) -> Result<Vec<NerSpan>, ModelError> {
        Ok(self
            .entities
            .iter()
            .filter(|s| s.score >= threshold && labels.contains(&s.label.as_str()))
            .cloned()
            .collect())
    }

    fn extract_relations(
        &self,
        _text: &str,
        labels: &[&str],
        threshold: f32,
    ) -> Result<Vec<RelationHit>, ModelError> {
        Ok(self
            .relations
            .iter()
            .filter(|r| r.score >= threshold && labels.contains(&r.label.as_str()))
            .cloned()
            .collect())
    }
}

/// Coreference resolver returning fixed clusters: scripted ones for specific texts, the
/// default clusters for every other text.
#[derive(Debug, Default)]
pub struct MockCoref {
    clusters: Vec<CorefCluster>,
    by_text: HashMap<String, Vec<CorefCluster>>,
}

impl MockCoref {
    /// A resolver returning `clusters` for every text.
    pub fn new(clusters: Vec<CorefCluster>) -> Self {
        Self { clusters, by_text: HashMap::new() }
    }

    /// Returns `clusters` for exactly `text`.
    #[must_use]
    pub fn with_text(mut self, text: impl Into<String>, clusters: Vec<CorefCluster>) -> Self {
        self.by_text.insert(text.into(), clusters);
        self
    }
}

impl Coref for MockCoref {
    fn resolve(&self, text: &str) -> Result<Vec<CorefCluster>, ModelError> {
        Ok(self.by_text.get(text).unwrap_or(&self.clusters).clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_embedder_is_deterministic_and_unit_length() {
        let e = MockEmbedder::new(16);
        let a = e.embed(&["hello", "world", "hello"]).expect("mock embed");
        assert_eq!(a[0], a[2]);
        assert_ne!(a[0], a[1]);
        let norm: f32 = a[1].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        assert_eq!(e.calls(), 1);
    }

    #[test]
    fn mock_coref_scripts_clusters_per_text() {
        let default = vec![CorefCluster { spans: vec![(0, 1)] }];
        let scripted = vec![CorefCluster { spans: vec![(0, 3), (8, 10)] }];
        let coref = MockCoref::new(default.clone()).with_text("Ann met her.", scripted.clone());
        assert_eq!(coref.resolve("Ann met her.").expect("scripted"), scripted);
        assert_eq!(coref.resolve("Other text").expect("default"), default);
    }

    #[test]
    fn mock_llm_queue_then_responder() {
        let llm = MockLlm::with_responder(|_| Ok("{\"a\": 1}".to_owned()));
        llm.push_response("first");
        assert_eq!(llm.complete(&LlmRequest::new("p1")).expect("queued"), "first");
        let v = llm.complete_json(&LlmRequest::new("p2")).expect("json");
        assert_eq!(v["a"], 1);
        assert_eq!(llm.prompts(), vec!["p1", "p2"]);
    }
}
