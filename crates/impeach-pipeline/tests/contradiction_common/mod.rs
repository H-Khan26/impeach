//! Mocks for runs through every stage: canned fact-extraction responses for the two Green
//! Valley documents and an NLI judge scripted for their cross-document conflicts.

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Arc;

use impeach_config::pipeline::ContradictionCheckStageConfig;
use impeach_config::{PipelineConfig, StageConfig};
use impeach_models::{MockLlm, MockNli, ModelKind, ModelRegistry, NliScores};
use serde_json::{json, Value};

/// Text of the incident report's speed fact.
pub const INCIDENT_SPEED: &str =
    "Lopez was traveling eastbound on Mission Street at 45 to 50 miles per hour.";
/// Text of the complaint's speed fact (contradicts [`INCIDENT_SPEED`]).
pub const COMPLAINT_SPEED: &str =
    "Lopez was traveling at or below the posted speed limit of 35 miles per hour.";
/// Text of the complaint's signal fact (needs review against the incident report).
pub const COMPLAINT_SIGNAL: &str =
    "Reed failed to stop for a solid red traffic signal on 14th Avenue.";

/// NLI scores whose argmax is contradiction with `confidence`.
pub fn contradiction(confidence: f32) -> NliScores {
    let rest = (1.0 - confidence) / 2.0;
    NliScores { contradiction: confidence, entailment: rest, neutral: rest }
}

fn entries(file: &str) -> Vec<Value> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/llm/facts/green_valley")
        .join(file);
    let raw = std::fs::read_to_string(path).expect("fixture reads");
    serde_json::from_str(&raw).expect("fixture is JSON")
}

/// An LLM answering fact-extraction prompts with the canned nodes of every entry whose `when`
/// phrase the prompt contains (incident report and complaint fixtures); other prompts get no
/// nodes.
pub fn green_valley_llm() -> MockLlm {
    let mut all = entries("incident_report_responses.json");
    all.extend(entries("complaint_responses.json"));
    MockLlm::with_responder(move |request| {
        let nodes: Vec<Value> = all
            .iter()
            .filter(|entry| request.prompt.contains(entry["when"].as_str().unwrap_or("\u{0}")))
            .flat_map(|entry| entry["response"]["nodes"].as_array().cloned().unwrap_or_default())
            .collect();
        Ok(format!("```json\n{}\n```", json!({ "nodes": nodes })))
    })
}

/// NLI: the complaint's speed contradicts the incident report (0.95), its red signal needs
/// review against Reed's green signal (0.7); everything else is neutral.
pub fn green_valley_nli() -> Arc<MockNli> {
    Arc::new(MockNli::new().with_rule(|premise, hypothesis| {
        if premise.contains("45 to 50 miles per hour") && hypothesis.contains("35 miles per hour") {
            Some(contradiction(0.95))
        } else if premise.contains("solid green") && hypothesis.contains("solid red") {
            Some(contradiction(0.7))
        } else {
            None
        }
    }))
}

/// Adds the Green Valley LLM and NLI mocks to `registry`.
pub fn add_models(registry: &ModelRegistry, nli: Arc<MockNli>) -> Arc<MockLlm> {
    let llm = Arc::new(green_valley_llm());
    registry.insert_llm(registry.key(ModelKind::Ollama), llm.clone());
    registry.insert_nli(registry.key(ModelKind::DebertaNli), nli);
    llm
}

/// The `ContradictionCheckStage` block of `config`.
pub fn contradiction_config(config: &PipelineConfig) -> ContradictionCheckStageConfig {
    config
        .stages
        .iter()
        .find_map(|stage| match stage {
            StageConfig::ContradictionCheck(check) => Some(check.clone()),
            _ => None,
        })
        .expect("config has a ContradictionCheckStage")
}

/// The stage names of the packaged config.
pub const ALL_STAGES: [&str; 8] = [
    "rag_index",
    "mention_detection",
    "definition_extraction",
    "coreference",
    "ed_resolution",
    "relation_extraction",
    "fact_extraction",
    "contradiction_check",
];
