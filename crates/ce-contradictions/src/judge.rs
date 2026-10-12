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
//!
//! The user template is a Python `str.format` template: `{fact_a}` / `{fact_b}` are replaced,
//! `{{` / `}}` unescaped and other fields left as written (Python raised `KeyError`).
//! LLM confidences are clamped to `[0, 1]` (Python passed e.g. `150%` through as 1.5); a NaN
//! confidence becomes 0.0.

use std::fmt;
use std::sync::{Arc, LazyLock};

use ce_config::prompts::JudgePrompt;
use ce_core::NliLabel;
use ce_models::json::find_flat_object_with_key;
use ce_models::{Llm, LlmRequest, NliJudge, NliScores};
use ce_text::normalize::py_strip;
use regex::Regex;
use serde_json::{Map, Value};

use crate::error::Result;
use crate::nli_context::py_str;

/// Default LLM judge prompt file.
pub const DEFAULT_JUDGE_PROMPT: &str = "contradiction_engine/prompts/processing/qwen_judge.json";

/// Confidence of a JSON verdict without a usable `confidence` field.
const DEFAULT_JSON_CONFIDENCE: f64 = 0.8;

/// Prefix of the reasoning of a failed LLM judgement.
const ERROR_REASONING_PREFIX: &str = "Error during LLM inference: ";

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

    /// Neutral with confidence 0.0 and the reasoning of a failed LLM call.
    fn llm_error(error: &str) -> Self {
        Self { reasoning: Some(format!("{ERROR_REASONING_PREFIX}{error}")), ..Self::neutral() }
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
    let mut label = NliLabel::Contradiction;
    let mut best = scores.contradiction;
    for (candidate, score) in
        [(NliLabel::Entailment, scores.entailment), (NliLabel::Neutral, scores.neutral)]
    {
        if score > best {
            label = candidate;
            best = score;
        }
    }
    JudgeVerdict { label, confidence: f64::from(best), reasoning: None }
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
        if pairs.is_empty() {
            return Vec::new();
        }
        match self.nli.score_pairs(pairs) {
            Ok(scores) if scores.len() == pairs.len() => scores.iter().map(nli_verdict).collect(),
            Ok(scores) => {
                tracing::warn!(
                    "NLI call failed: {} scores for {} pairs",
                    scores.len(),
                    pairs.len()
                );
                vec![JudgeVerdict::neutral(); pairs.len()]
            }
            Err(error) => {
                tracing::warn!("NLI call failed: {error}");
                vec![JudgeVerdict::neutral(); pairs.len()]
            }
        }
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
        let prompt = JudgePrompt::load(prompt_file)?;
        Ok(Self::new(llm, prompt, batch_size))
    }

    /// The full prompt for one pair (module docs).
    #[must_use]
    pub fn build_prompt(&self, fact_a: &str, fact_b: &str) -> String {
        let user = format_template(&self.prompt.user_template, fact_a, fact_b);
        format!("System: {}\n\nUser: {user}\n\nAssistant:", self.prompt.system)
    }

    /// `(label is contradiction && confidence >= threshold, confidence)`.
    #[must_use]
    pub fn is_contradiction(&self, fact_a: &str, fact_b: &str, threshold: f64) -> (bool, f64) {
        let verdict = self.judge(fact_a, fact_b);
        (
            verdict.label == NliLabel::Contradiction && verdict.confidence >= threshold,
            verdict.confidence,
        )
    }

    /// Judges one pair with one LLM call.
    fn judge_one(&self, fact_a: &str, fact_b: &str) -> JudgeVerdict {
        let prompt = self.build_prompt(fact_a, fact_b);
        match self.llm.complete(&LlmRequest::new(&prompt)) {
            Ok(response) => parse_judge_response(&response),
            Err(error) => {
                tracing::warn!("Qwen judge failed: {error}");
                JudgeVerdict::llm_error(&error.to_string())
            }
        }
    }
}

impl ContradictionJudge for LlmJudge {
    fn name(&self) -> &str {
        "qwen"
    }

    fn judge_batch(&self, pairs: &[(&str, &str)]) -> Vec<JudgeVerdict> {
        let mut verdicts = Vec::with_capacity(pairs.len());
        for batch in pairs.chunks(self.batch_size) {
            verdicts.extend(batch.iter().map(|(a, b)| self.judge_one(a, b)));
        }
        verdicts
    }
}

/// Formats a Python `str.format` template with `fact_a` / `fact_b`: `{{` / `}}` become braces,
/// `{fact_a}` / `{fact_b}` are replaced and any other field is kept as written.
fn format_template(template: &str, fact_a: &str, fact_b: &str) -> String {
    let mut out = String::with_capacity(template.len() + fact_a.len() + fact_b.len());
    let mut rest = template;
    while let Some(pos) = rest.find(['{', '}']) {
        out.push_str(&rest[..pos]);
        let tail = &rest[pos..];
        if let Some(after) = tail.strip_prefix("{{") {
            out.push('{');
            rest = after;
        } else if let Some(after) = tail.strip_prefix("}}").or_else(|| tail.strip_prefix('}')) {
            out.push('}');
            rest = after;
        } else if let Some(close) = tail.find('}') {
            match &tail[1..close] {
                "fact_a" => out.push_str(fact_a),
                "fact_b" => out.push_str(fact_b),
                _ => out.push_str(&tail[..=close]),
            }
            rest = &tail[close + 1..];
        } else {
            out.push_str(tail);
            rest = "";
        }
    }
    out.push_str(rest);
    out
}

/// Parses an LLM judge response: the first flat JSON object containing `"verdict"` (keys
/// `verdict` / `label` / `classification` / `result`; `contra` -> contradiction, `entail` /
/// `support` / `agree` -> entailment, else neutral; `confidence` default 0.8; `reasoning` or
/// `explanation`), else free-text label and confidence extraction as in Python.
///
/// A JSON object whose verdict or confidence Python could not read (no verdict key; a null or
/// list confidence) yields a neutral verdict with Python's error message as reasoning.
#[must_use]
pub fn parse_judge_response(response: &str) -> JudgeVerdict {
    let parsed = find_flat_object_with_key(response, "verdict")
        .and_then(|object| serde_json::from_str::<Map<String, Value>>(object).ok());
    if let Some(object) = parsed {
        return match verdict_from_json(&object) {
            Ok(verdict) => verdict,
            Err(error) => {
                tracing::warn!("Qwen judge failed: {error}");
                JudgeVerdict::llm_error(&error)
            }
        };
    }
    JudgeVerdict {
        label: label_from_text(response),
        confidence: clamp_confidence(confidence_from_text(response)),
        reasoning: Some(py_strip(response).to_owned()),
    }
}

/// Python `_normalize_parsed_result`; `Err` carries the Python `TypeError` message.
fn verdict_from_json(object: &Map<String, Value>) -> std::result::Result<JudgeVerdict, String> {
    let verdict = ["verdict", "label", "classification", "result"]
        .iter()
        .find_map(|key| object.get(*key))
        .map(|value| py_str(value).to_lowercase())
        .ok_or_else(|| "argument of type 'NoneType' is not iterable".to_owned())?;
    let label = if verdict.contains("contra") {
        NliLabel::Contradiction
    } else if ["entail", "support", "agree"].iter().any(|word| verdict.contains(word)) {
        NliLabel::Entailment
    } else {
        NliLabel::Neutral
    };

    let confidence = match object.get("confidence") {
        None => DEFAULT_JSON_CONFIDENCE,
        Some(Value::Number(n)) => n.as_f64().unwrap_or(DEFAULT_JSON_CONFIDENCE),
        Some(Value::Bool(b)) => f64::from(u8::from(*b)),
        Some(Value::String(s)) => py_float(s).unwrap_or(DEFAULT_JSON_CONFIDENCE),
        Some(other) => {
            let type_name = match other {
                Value::Null => "NoneType",
                Value::Array(_) => "list",
                _ => "dict",
            };
            return Err(format!(
                "float() argument must be a string or a real number, not '{type_name}'"
            ));
        }
    };

    let reasoning = object
        .get("reasoning")
        .or_else(|| object.get("explanation"))
        .map(py_str)
        .unwrap_or_default();
    Ok(JudgeVerdict { label, confidence: clamp_confidence(confidence), reasoning: Some(reasoning) })
}

/// Python `float(text)` for the forms LLMs produce (surrounding whitespace allowed).
fn py_float(text: &str) -> Option<f64> {
    py_strip(text).parse::<f64>().ok()
}

/// Clamps a confidence to `[0, 1]`; NaN becomes 0.0.
fn clamp_confidence(confidence: f64) -> f64 {
    if confidence.is_nan() {
        0.0
    } else {
        confidence.clamp(0.0, 1.0)
    }
}

/// Python `_extract_label_from_text`.
fn label_from_text(text: &str) -> NliLabel {
    let lower = text.to_lowercase();
    let has_any = |words: &[&str]| words.iter().any(|word| lower.contains(word));
    if has_any(&["contradict", "conflict", "disagree", "inconsistent"])
        && !has_any(&["not contradict", "no contradict", "don't contradict"])
    {
        return NliLabel::Contradiction;
    }
    if has_any(&["entail", "support", "agree", "consistent", "confirm"]) {
        return NliLabel::Entailment;
    }
    NliLabel::Neutral
}

static PERCENT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([0-9]+(?:\.[0-9]+)?)\s*%").expect("percent pattern is valid"));

static CONFIDENCE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"confidence[:\s]+([0-9]+(?:\.[0-9]+)?)").expect("confidence pattern is valid")
});

/// Python `_extract_confidence_from_text` (unclamped).
fn confidence_from_text(text: &str) -> f64 {
    if let Some(value) = PERCENT_RE.captures(text).and_then(|c| c[1].parse::<f64>().ok()) {
        return value / 100.0;
    }
    let lower = text.to_lowercase();
    if let Some(value) = CONFIDENCE_RE.captures(&lower).and_then(|c| c[1].parse::<f64>().ok()) {
        return if value > 1.0 { value / 100.0 } else { value };
    }
    let has_any = |words: &[&str]| words.iter().any(|word| lower.contains(word));
    if has_any(&["certain", "definite", "clear", "obvious"]) {
        0.9
    } else if has_any(&["likely", "probable", "strong"]) {
        0.8
    } else if has_any(&["possible", "maybe", "might"]) {
        0.6
    } else if has_any(&["unclear", "uncertain", "ambiguous"]) {
        0.5
    } else {
        0.7
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_formatting_follows_python_str_format() {
        assert_eq!(format_template("A: {fact_a} B: {fact_b}", "x", "y"), "A: x B: y");
        assert_eq!(format_template("{{\n \"v\": 1\n}}", "x", "y"), "{\n \"v\": 1\n}");
        assert_eq!(format_template("{{fact_a}}", "x", "y"), "{fact_a}");
        assert_eq!(format_template("{other} {fact_a}", "x", "y"), "{other} x");
        assert_eq!(format_template("open { only", "x", "y"), "open { only");
        assert_eq!(format_template("stray } brace", "x", "y"), "stray } brace");
        // Values are inserted verbatim, never re-parsed.
        assert_eq!(format_template("{fact_a}|{fact_b}", "{fact_b}", "}}"), "{fact_b}|}}");
    }

    #[test]
    fn confidence_clamping() {
        assert!((clamp_confidence(1.5) - 1.0).abs() < f64::EPSILON);
        assert!(clamp_confidence(-0.2).abs() < f64::EPSILON);
        assert!(clamp_confidence(f64::NAN).abs() < f64::EPSILON);
        assert!((clamp_confidence(0.42) - 0.42).abs() < f64::EPSILON);
    }

    #[test]
    fn free_text_confidence_cases() {
        assert!((confidence_from_text("I am 85% sure") - 0.85).abs() < 1e-12);
        assert!((confidence_from_text("Confidence: 0.65") - 0.65).abs() < 1e-12);
        assert!((confidence_from_text("confidence 90") - 0.9).abs() < 1e-12);
        assert!((confidence_from_text("It is obvious") - 0.9).abs() < 1e-12);
        // "unclear" contains "clear", which Python checks first.
        assert!((confidence_from_text("This is unclear") - 0.9).abs() < 1e-12);
        assert!((confidence_from_text("likely so") - 0.8).abs() < 1e-12);
        assert!((confidence_from_text("maybe") - 0.6).abs() < 1e-12);
        assert!((confidence_from_text("ambiguous") - 0.5).abs() < 1e-12);
        assert!((confidence_from_text("no idea") - 0.7).abs() < 1e-12);
    }
}
