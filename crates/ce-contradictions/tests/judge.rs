//! NLI and LLM judges and the LLM response parser.

mod common_b;

use std::sync::Arc;

use ce_config::prompts::JudgePrompt;
use ce_contradictions::{
    nli_verdict, parse_judge_response, ContradictionError, ContradictionJudge, JudgeVerdict,
    LlmJudge, NliJudgeAdapter, DEFAULT_JUDGE_PROMPT,
};
use ce_core::NliLabel;
use ce_models::{MockLlm, MockNli, ModelError, NliJudge, NliScores};
use common_b::assert_close;
use pretty_assertions::assert_eq;

fn scores(contradiction: f32, entailment: f32, neutral: f32) -> NliScores {
    NliScores { contradiction, entailment, neutral }
}

fn verdict(label: NliLabel, confidence: f64, reasoning: &str) -> JudgeVerdict {
    JudgeVerdict { label, confidence, reasoning: Some(reasoning.to_owned()) }
}

fn assert_verdict(actual: &JudgeVerdict, label: NliLabel, confidence: f64) {
    assert_eq!(actual.label, label, "{actual:?}");
    assert_close(actual.confidence, confidence);
}

#[test]
fn nli_verdict_is_argmax_with_ordered_ties() {
    let v = nli_verdict(&scores(0.1, 0.7, 0.2));
    assert_eq!(v.label, NliLabel::Entailment);
    assert_close(v.confidence, f64::from(0.7_f32));
    assert_eq!(v.reasoning, None);
    assert_eq!(nli_verdict(&scores(0.2, 0.2, 0.6)).label, NliLabel::Neutral);
    assert_eq!(nli_verdict(&scores(0.4, 0.4, 0.2)).label, NliLabel::Contradiction);
    assert_eq!(nli_verdict(&scores(0.2, 0.4, 0.4)).label, NliLabel::Entailment);
    assert_eq!(nli_verdict(&scores(0.0, 0.0, 0.0)).label, NliLabel::Contradiction);
}

#[test]
fn nli_adapter_scores_the_batch_in_one_call() {
    let nli = Arc::new(
        MockNli::new()
            .with_pair("p1", "h1", scores(0.9, 0.05, 0.05))
            .with_rule(|p, _| (p == "p2").then(|| scores(0.1, 0.8, 0.1))),
    );
    let judge = NliJudgeAdapter::new(nli.clone());
    assert_eq!(judge.name(), "nli");
    let verdicts = judge.judge_batch(&[("p1", "h1"), ("p2", "h2"), ("p3", "h3")]);
    assert_eq!(verdicts.len(), 3);
    assert_verdict(&verdicts[0], NliLabel::Contradiction, f64::from(0.9_f32));
    assert_verdict(&verdicts[1], NliLabel::Entailment, f64::from(0.8_f32));
    assert_verdict(&verdicts[2], NliLabel::Neutral, 1.0);
    assert_eq!(nli.calls().len(), 3);

    assert!(judge.judge_batch(&[]).is_empty());
    assert_eq!(nli.calls().len(), 3, "an empty batch makes no call");

    assert_verdict(&judge.judge("p1", "h1"), NliLabel::Contradiction, f64::from(0.9_f32));
}

struct FailingNli;

impl NliJudge for FailingNli {
    fn score_pairs(&self, _pairs: &[(&str, &str)]) -> Result<Vec<NliScores>, ModelError> {
        Err(ModelError::Inference("model crashed".to_owned()))
    }
}

struct ShortNli;

impl NliJudge for ShortNli {
    fn score_pairs(&self, _pairs: &[(&str, &str)]) -> Result<Vec<NliScores>, ModelError> {
        Ok(vec![NliScores { contradiction: 1.0, entailment: 0.0, neutral: 0.0 }])
    }
}

#[test]
fn nli_failures_score_every_pair_neutral() {
    for judge in
        [NliJudgeAdapter::new(Arc::new(FailingNli)), NliJudgeAdapter::new(Arc::new(ShortNli))]
    {
        let verdicts = judge.judge_batch(&[("a", "b"), ("c", "d")]);
        assert_eq!(verdicts, vec![JudgeVerdict::neutral(), JudgeVerdict::neutral()]);
    }
}

fn default_prompt() -> JudgePrompt {
    JudgePrompt::load(DEFAULT_JUDGE_PROMPT).expect("embedded judge prompt loads")
}

#[test]
fn prompt_follows_the_python_format() {
    let prompt = default_prompt();
    let judge = LlmJudge::new(Arc::new(MockLlm::new()), prompt.clone(), 5);
    let built = judge.build_prompt("Speed was 35 mph", "Speed was {45} mph");
    let user = "Analyze these two facts and determine if they contradict:\n\n\
                Fact A: Speed was 35 mph\n\nFact B: Speed was {45} mph\n\n\
                Respond in JSON format:\n{\n  \"verdict\": \"contradiction\" | \"entailment\" | \
                \"neutral\",\n  \"confidence\": 0.0-1.0,\n  \"reasoning\": \"Brief explanation of \
                your decision\"\n}";
    assert_eq!(built, format!("System: {}\n\nUser: {user}\n\nAssistant:", prompt.system));
    assert!(built.starts_with("System: You are a legal fact contradiction detector"));
}

#[test]
fn custom_templates_keep_unknown_fields() {
    let prompt = JudgePrompt {
        system: "S".to_owned(),
        user_template: "{fact_b} vs {fact_a} ({context}) {{x}}".to_owned(),
        output_format: None,
        notes: Vec::new(),
    };
    let judge = LlmJudge::new(Arc::new(MockLlm::new()), prompt, 0);
    assert_eq!(
        judge.build_prompt("A", "B"),
        "System: S\n\nUser: B vs A ({context}) {x}\n\nAssistant:"
    );
}

#[test]
fn load_resolves_the_default_prompt_and_rejects_bad_files() {
    let llm = Arc::new(MockLlm::new());
    let judge = LlmJudge::load(llm.clone(), DEFAULT_JUDGE_PROMPT, 5).expect("default prompt loads");
    assert_eq!(judge.name(), "qwen");

    let missing = LlmJudge::load(llm.clone(), "contradiction_engine/prompts/missing.json", 5);
    assert!(matches!(missing, Err(ContradictionError::Config(_))));

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("judge.json");
    std::fs::write(&path, r#"{"system": "only a system message"}"#).expect("write prompt");
    let incomplete = LlmJudge::load(llm, path.to_str().expect("utf-8 path"), 5);
    assert!(matches!(incomplete, Err(ContradictionError::Config(_))));
}

#[test]
fn llm_judge_makes_one_call_per_pair_in_order() {
    let llm = Arc::new(MockLlm::with_responses([
        r#"{"verdict": "contradiction", "confidence": 0.95, "reasoning": "35 vs 45"}"#,
        r#"{"verdict": "entailment", "confidence": 0.6}"#,
        "They are unrelated.",
    ]));
    let judge = LlmJudge::new(llm.clone(), default_prompt(), 2);
    let verdicts = judge.judge_batch(&[("a1", "b1"), ("a2", "b2"), ("a3", "b3")]);
    assert_eq!(
        verdicts,
        vec![
            verdict(NliLabel::Contradiction, 0.95, "35 vs 45"),
            verdict(NliLabel::Entailment, 0.6, ""),
            verdict(NliLabel::Neutral, 0.7, "They are unrelated."),
        ]
    );
    let prompts = llm.prompts();
    assert_eq!(prompts.len(), 3);
    for (i, prompt) in prompts.iter().enumerate() {
        let n = i + 1;
        assert!(prompt.contains(&format!("Fact A: a{n}\n\nFact B: b{n}")), "{prompt}");
    }
}

#[test]
fn llm_errors_score_neutral_with_the_error() {
    let judge = LlmJudge::new(Arc::new(MockLlm::new()), default_prompt(), 5);
    let v = judge.judge("a", "b");
    assert_eq!(v.label, NliLabel::Neutral);
    assert_close(v.confidence, 0.0);
    let reasoning = v.reasoning.expect("error reasoning");
    assert!(reasoning.starts_with("Error during LLM inference: "), "{reasoning}");
    assert!(reasoning.contains("no response queued"), "{reasoning}");
}

#[test]
fn is_contradiction_applies_label_and_threshold() {
    let llm = Arc::new(MockLlm::with_responses([
        r#"{"verdict": "contradiction", "confidence": 0.85}"#,
        r#"{"verdict": "contradiction", "confidence": 0.75}"#,
        r#"{"verdict": "neutral", "confidence": 0.99}"#,
    ]));
    let judge = LlmJudge::new(llm, default_prompt(), 5);
    let (hit, confidence) = judge.is_contradiction("a", "b", 0.8);
    assert!(hit);
    assert_close(confidence, 0.85);
    assert!(!judge.is_contradiction("a", "b", 0.8).0);
    let (hit, confidence) = judge.is_contradiction("a", "b", 0.8);
    assert!(!hit);
    assert_close(confidence, 0.99);
}

#[test]
fn json_verdicts_are_normalised() {
    let cases: [(&str, NliLabel, f64, &str); 12] = [
        (
            "Sure! {\"verdict\": \"CONTRADICTION\", \"confidence\": 0.9, \"reasoning\": \"r\"} done",
            NliLabel::Contradiction,
            0.9,
            "r",
        ),
        (r#"{"verdict": "Entailed", "confidence": "0.7"}"#, NliLabel::Entailment, 0.7, ""),
        (r#"{"verdict": "supports", "confidence": "high"}"#, NliLabel::Entailment, 0.8, ""),
        (r#"{"verdict": "they agree"}"#, NliLabel::Entailment, 0.8, ""),
        (r#"{"verdict": "unrelated", "confidence": true}"#, NliLabel::Neutral, 1.0, ""),
        (r#"{"verdict": "contradiction", "confidence": 1.5}"#, NliLabel::Contradiction, 1.0, ""),
        (r#"{"verdict": "contradiction", "confidence": -2}"#, NliLabel::Contradiction, 0.0, ""),
        (r#"{"verdict": 1, "explanation": "why"}"#, NliLabel::Neutral, 0.8, "why"),
        (r#"{"verdict": "neutral", "reasoning": null}"#, NliLabel::Neutral, 0.8, "None"),
        // `"verdict"` as a value still selects the object; `label` supplies the verdict.
        (r#"{"note": "verdict", "label": "contradiction"}"#, NliLabel::Contradiction, 0.8, ""),
        // Key priority: verdict, label, classification, result.
        (r#"{"result": "neutral", "verdict": "contradiction"}"#, NliLabel::Contradiction, 0.8, ""),
        // The first flat object containing "verdict" wins.
        (
            r#"{"a": 1} {"verdict": "entailment", "confidence": 0.65} {"verdict": "contradiction"}"#,
            NliLabel::Entailment,
            0.65,
            "",
        ),
    ];
    for (response, label, confidence, reasoning) in cases {
        let v = parse_judge_response(response);
        assert_eq!(v.label, label, "{response}");
        assert_close(v.confidence, confidence);
        assert_eq!(v.reasoning.as_deref(), Some(reasoning), "{response}");
    }
}

#[test]
fn json_values_python_cannot_read_become_errors() {
    let v = parse_judge_response(r#"{"note": "verdict"}"#);
    assert_eq!(v.label, NliLabel::Neutral);
    assert_close(v.confidence, 0.0);
    assert_eq!(
        v.reasoning.as_deref(),
        Some("Error during LLM inference: argument of type 'NoneType' is not iterable")
    );

    let v = parse_judge_response(r#"{"verdict": "contradiction", "confidence": null}"#);
    assert_eq!(v.label, NliLabel::Neutral);
    assert_eq!(
        v.reasoning.as_deref(),
        Some(
            "Error during LLM inference: float() argument must be a string or a real number, \
             not 'NoneType'"
        )
    );
}

#[test]
fn free_text_responses_are_classified() {
    let cases: [(&str, NliLabel, f64); 8] = [
        ("These statements conflict. Confidence: 92", NliLabel::Contradiction, 0.92),
        ("I am 150% sure they contradict.", NliLabel::Contradiction, 1.0),
        ("The facts do not contradict; they are consistent.", NliLabel::Entailment, 0.7),
        ("Fact B supports fact A, 80% likely.", NliLabel::Entailment, 0.8),
        ("No relation, maybe.", NliLabel::Neutral, 0.6),
        ("It is certain these are unrelated.", NliLabel::Neutral, 0.9),
        // Invalid JSON falls back to the text rules.
        ("{\"verdict\": contradiction}", NliLabel::Contradiction, 0.7),
        // A nested object is not a flat match, so the text rules apply.
        ("{\"verdict\": {\"label\": \"neutral\"}}", NliLabel::Neutral, 0.7),
    ];
    for (response, label, confidence) in cases {
        let v = parse_judge_response(response);
        assert_eq!(v.label, label, "{response}");
        assert_close(v.confidence, confidence);
    }
    let v = parse_judge_response("  \n It contradicts. \t");
    assert_eq!(v.reasoning.as_deref(), Some("It contradicts."));
}
