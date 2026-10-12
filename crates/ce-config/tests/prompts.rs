//! Prompt file loading.

use ce_config::assets::embedded_assets;
use ce_config::prompts::{EdPrompt, ExtractionPrompt, JudgePrompt, PromptTemplate};
use ce_config::ConfigError;
use pretty_assertions::assert_eq;

const SINGLE_PASS: &str =
    "contradiction_engine/prompts/processing/singlepass/processing_single_v3.json";
const MULTI_PASS: [&str; 4] = [
    "contradiction_engine/prompts/processing/multipass/processing_mullti_v2/pass1_incident.json",
    "contradiction_engine/prompts/processing/multipass/processing_mullti_v2/pass2_party.json",
    "contradiction_engine/prompts/processing/multipass/processing_mullti_v2/pass3_history.json",
    "contradiction_engine/prompts/processing/multipass/processing_mullti_v2/pass4_damages.json",
];

#[test]
fn every_embedded_prompt_loads() {
    let prompts = embedded_assets().get_dir("prompts").expect("prompts dir");
    let mut count = 0;
    let mut stack = vec![prompts];
    while let Some(dir) = stack.pop() {
        stack.extend(dir.dirs());
        for file in dir.files() {
            let path = file.path().display().to_string();
            let text = file.contents_utf8().expect("utf-8");
            PromptTemplate::from_json_str(text, &path).unwrap_or_else(|e| panic!("{path}: {e}"));
            count += 1;
        }
    }
    assert_eq!(count, 8);
}

#[test]
fn ed_prompt() {
    let PromptTemplate::Ed(ed) =
        PromptTemplate::load("contradiction_engine/prompts/ed/disambiguation.json").expect("loads")
    else {
        panic!("ED prompt");
    };
    assert!(ed.system_template.contains("disambiguation"));
    assert_eq!(ed.user_template, "Payload:\n{payload_json}");
    assert_eq!(
        EdPrompt::load("contradiction_engine/prompts/ed/disambiguation.json").expect("ed"),
        ed
    );
}

#[test]
fn single_pass_prompt() {
    let prompt = ExtractionPrompt::load(SINGLE_PASS).expect("loads");
    assert_eq!(prompt.version.as_deref(), Some("v3"));
    assert_eq!(prompt.setting.as_deref(), Some("processing"));
    assert_eq!(prompt.system_messages.len(), 2);
    assert_eq!(prompt.examples.len(), 9);
    assert!(prompt.examples[0].output.get("nodes").is_some());
    let combined = prompt.combined_template();
    assert!(combined.starts_with(prompt.system_messages[0].content.trim()));
    assert!(combined.ends_with(&prompt.user_prompt_template));
}

#[test]
fn multi_pass_prompts() {
    for (i, path) in MULTI_PASS.iter().enumerate() {
        let prompt = ExtractionPrompt::load(path).expect("loads");
        assert_eq!(prompt.pass_number, Some(u32::try_from(i + 1).expect("small")));
        assert!(prompt.pass_name.is_some());
        assert_eq!(prompt.system_messages.len(), 3);
        assert!(matches!(PromptTemplate::load(path), Ok(PromptTemplate::Extraction(_))));
    }
}

#[test]
fn judge_prompt() {
    let path = "contradiction_engine/prompts/processing/qwen_judge.json";
    let judge = JudgePrompt::load(path).expect("loads");
    assert!(judge.user_template.contains("{fact_a}"));
    assert!(judge.output_format.is_some());
    assert!(!judge.notes.is_empty());
    assert_eq!(PromptTemplate::load(path).expect("loads"), PromptTemplate::Judge(judge));
}

#[test]
fn combined_template_rules() {
    let prompt = |system: &[&str], user: &str| ExtractionPrompt {
        description: None,
        setting: None,
        version: None,
        pass_number: None,
        pass_name: None,
        system_messages: system
            .iter()
            .map(|c| ce_config::prompts::ChatMessage {
                role: "system".into(),
                content: (*c).into(),
            })
            .collect(),
        examples: Vec::new(),
        user_prompt_template: user.into(),
    };
    assert_eq!(prompt(&[" A ", "  ", "B"], "U").combined_template(), "A\n\nB\n\nU");
    assert_eq!(prompt(&[], "U").combined_template(), "U");
    assert_eq!(prompt(&["A"], "").combined_template(), "A");
    assert_eq!(prompt(&[], "").combined_template(), "");
}

#[test]
fn ed_prompt_with_only_one_template_is_detected() {
    let user_only = r#"{"user_template": "U {payload_json}"}"#;
    let expected =
        EdPrompt { system_template: String::new(), user_template: "U {payload_json}".into() };
    assert_eq!(
        PromptTemplate::from_json_str(user_only, "x.json").expect("user_template only"),
        PromptTemplate::Ed(expected)
    );
    let system_only = r#"{"system_template": "S"}"#;
    assert!(matches!(
        PromptTemplate::from_json_str(system_only, "x.json"),
        Ok(PromptTemplate::Ed(EdPrompt { .. }))
    ));
}

#[test]
fn malformed_prompts_fail() {
    assert!(matches!(
        PromptTemplate::from_json_str(r#"{"foo": 1}"#, "x.json"),
        Err(ConfigError::PromptKind { .. })
    ));
    assert!(matches!(
        PromptTemplate::from_json_str(
            r#"{"system": "s", "user_template": "u", "extra": 1}"#,
            "x.json"
        ),
        Err(ConfigError::Json { .. })
    ));
    assert!(matches!(PromptTemplate::from_json_str("{", "x.json"), Err(ConfigError::Json { .. })));
    assert!(matches!(
        PromptTemplate::load("contradiction_engine/prompts/missing.json"),
        Err(ConfigError::AssetNotFound { .. })
    ));
}
