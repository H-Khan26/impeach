//! Contradiction judges: the NLI cross-encoder and the LLM judge (Python
//! `services/contradictions/qwen_judge.py` and the judge calls of `service.py`).
//!
//! A judge labels `(premise, hypothesis)` pairs. Failures never propagate: a failed call is
//! logged and scored [`JudgeVerdict::neutral`] (label neutral, confidence 0.0), as in Python.
//!
//! - [`NliJudgeAdapter`]: [`NliJudge::score_pairs`] on the whole batch; label = argmax of the
//!   scores (ties: contradiction, entailment, neutral), confidence = that score.
//! - [`LlmJudge`]: one LLM completion per pair with the prompt file (default
//!   `qwen_judge.json`): `System: <system>\n\nUser: <user_template formatted with fact_a /
//!   fact_b>\n\nAssistant:`; the response is parsed by [`parse_judge_response`]. Pairs are
//!   processed in groups of `batch_size` (one call per pair, as in Python).

use std::fmt;
use std::sync::Arc;

use ce_config::prompts::JudgePrompt;
use ce_core::NliLabel;
use ce_models::{Llm, NliJudge, NliScores};

use crate::error::Result;

/// Default LLM judge prompt file.
pub const DEFAULT_JUDGE_PROMPT: &str = "contradiction_engine/prompts/processing/qwen_judge.json";

/// One judgement.
#[derive(Debug, Clone, PartialEq)]
pub struct JudgeVerdict {
    /// Label.
    pub label: NliLabel,
    /// Confidence in `[0, 1]` (LLM values are clamped).
    pub confidence: f64,
    /// LLM reasoning, when the judge gives one.
    pub reasoning: Option<String>,
}

impl JudgeVerdict {
    /// Neutral with confidence 0.0 (failed or unavailable judge).
    #[must_use]
    pub fn neutral() -> Self {
        Self { label: NliLabel::Neutral, confidence: 0.0, reasoning: None }
    }
}

/// Labels premise / hypothesis pairs.
pub trait ContradictionJudge: Send + Sync + fmt::Debug {
    /// Judge name for logs (`nli`, `qwen`).
    fn name(&self) -> &str;

    /// One verdict per `(premise, hypothesis)` pair, in input order.
    fn judge_batch(&self, pairs: &[(&str, &str)]) -> Vec<JudgeVerdict>;

    /// Verdict for one pair.
    fn judge(&self, premise: &str, hypothesis: &str) -> JudgeVerdict {
        self.judge_batch(&[(premise, hypothesis)]).pop().unwrap_or_else(JudgeVerdict::neutral)
    }
}

/// The verdict of NLI scores: argmax label (ties: contradiction, entailment, neutral) and its
/// score as confidence.
#[must_use]
pub fn nli_verdict(scores: &NliScores) -> JudgeVerdict {
    todo!("worker B")
}

/// NLI cross-encoder judge.
#[derive(Clone)]
pub struct NliJudgeAdapter {
    nli: Arc<dyn NliJudge>,
}

impl fmt::Debug for NliJudgeAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NliJudgeAdapter").finish_non_exhaustive()
    }
}

impl NliJudgeAdapter {
    /// A judge over `nli`.
    #[must_use]
    pub fn new(nli: Arc<dyn NliJudge>) -> Self {
        Self { nli }
    }
}

impl ContradictionJudge for NliJudgeAdapter {
    fn name(&self) -> &str {
        "nli"
    }

    fn judge_batch(&self, pairs: &[(&str, &str)]) -> Vec<JudgeVerdict> {
        todo!("worker B")
    }
}

/// LLM judge (Python `QwenJudgeService`).
#[derive(Clone)]
pub struct LlmJudge {
    llm: Arc<dyn Llm>,
    prompt: JudgePrompt,
    batch_size: usize,
}

impl fmt::Debug for LlmJudge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlmJudge").field("batch_size", &self.batch_size).finish_non_exhaustive()
    }
}

impl LlmJudge {
    /// A judge over `llm` with a loaded prompt; `batch_size` is clamped to at least 1.
    #[must_use]
    pub fn new(llm: Arc<dyn Llm>, prompt: JudgePrompt, batch_size: usize) -> Self {
        Self { llm, prompt, batch_size: batch_size.max(1) }
    }

    /// Loads the prompt file (resolved with [`ce_config::resolve_asset`]) and builds the judge.
    ///
    /// # Errors
    /// [`crate::ContradictionError::Config`] when the file is missing or lacks `system` /
    /// `user_template`.
    pub fn load(llm: Arc<dyn Llm>, prompt_file: &str, batch_size: usize) -> Result<Self> {
        todo!("worker B")
    }

    /// The full prompt for one pair (module docs).
    #[must_use]
    pub fn build_prompt(&self, fact_a: &str, fact_b: &str) -> String {
        todo!("worker B")
    }

    /// `(label is contradiction && confidence >= threshold, confidence)`.
    #[must_use]
    pub fn is_contradiction(&self, fact_a: &str, fact_b: &str, threshold: f64) -> (bool, f64) {
        todo!("worker B")
    }
}

impl ContradictionJudge for LlmJudge {
    fn name(&self) -> &str {
        "qwen"
    }

    fn judge_batch(&self, pairs: &[(&str, &str)]) -> Vec<JudgeVerdict> {
        todo!("worker B")
    }
}

/// Parses an LLM judge response: the first flat JSON object containing `"verdict"` (keys
/// `verdict` / `label` / `classification` / `result`; `contra` -> contradiction, `entail` /
/// `support` / `agree` -> entailment, else neutral; `confidence` default 0.8; `reasoning` or
/// `explanation`), else free-text label and confidence extraction as in Python.
#[must_use]
pub fn parse_judge_response(response: &str) -> JudgeVerdict {
    todo!("worker B")
}
