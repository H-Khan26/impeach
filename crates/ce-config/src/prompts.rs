//! LLM prompt files (`prompts/**/*.json`).
//!
//! Three shapes exist: the ED disambiguation prompt (`system_template` / `user_template`),
//! fact extraction prompts (single-pass and multi-pass: `system_messages`,
//! `user_prompt_template`, optional examples) and the Qwen judge prompt (`system` /
//! `user_template`). [`PromptTemplate`] detects the shape from its keys: extraction keys
//! first, then `system` (judge), then `system_template` or `user_template` (ED). Example outputs
//! stay as raw JSON until the fact extractor types them.

use std::path::Path;

use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::Value;

use crate::assets::read_asset_in;
use crate::error::{ConfigError, Result};

/// A prompt file of any known shape.
#[derive(Debug, Clone, PartialEq)]
pub enum PromptTemplate {
    /// ED disambiguation prompt.
    Ed(EdPrompt),
    /// Fact extraction prompt (single-pass or one pass of multi-pass).
    Extraction(ExtractionPrompt),
    /// Contradiction judge prompt.
    Judge(JudgePrompt),
}

impl PromptTemplate {
    /// Parses prompt JSON, choosing the shape from its keys; `origin` names the source in errors.
    pub fn from_json_str(text: &str, origin: &str) -> Result<Self> {
        let json_error = |source| ConfigError::Json { path: origin.to_owned(), source };
        let value: Value = serde_json::from_str(text).map_err(json_error)?;
        let has = |key: &str| value.get(key).is_some();
        if has("system_messages") || has("user_prompt_template") {
            serde_json::from_value(value).map(Self::Extraction).map_err(json_error)
        } else if has("system") {
            serde_json::from_value(value).map(Self::Judge).map_err(json_error)
        } else if has("system_template") || has("user_template") {
            serde_json::from_value(value).map(Self::Ed).map_err(json_error)
        } else {
            Err(ConfigError::PromptKind { path: origin.to_owned(), expected: "a known" })
        }
    }

    /// Loads a prompt file resolved with [`crate::assets::resolve_asset`].
    pub fn load(path: &str) -> Result<Self> {
        Self::load_in(None, path)
    }

    /// Loads a prompt file, resolving `path` against `base` (or the working directory) first.
    pub fn load_in(base: Option<&Path>, path: &str) -> Result<Self> {
        let (text, origin) = read_asset_in(base, path)?;
        Self::from_json_str(&text, &origin)
    }
}

fn load_typed<T: DeserializeOwned>(base: Option<&Path>, path: &str) -> Result<T> {
    let (text, origin) = read_asset_in(base, path)?;
    serde_json::from_str(&text).map_err(|source| ConfigError::Json { path: origin, source })
}

/// ED disambiguation prompt.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EdPrompt {
    /// System message template.
    #[serde(default)]
    pub system_template: String,
    /// User message template (`{payload_json}` placeholder).
    #[serde(default)]
    pub user_template: String,
}

impl EdPrompt {
    /// Loads an ED prompt file resolved with [`crate::assets::resolve_asset`].
    pub fn load(path: &str) -> Result<Self> {
        load_typed(None, path)
    }
}

/// One chat message of an extraction prompt.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    /// Message role (`system`).
    pub role: String,
    /// Message text.
    pub content: String,
}

/// A worked example inside an extraction prompt.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptExample {
    /// What the example shows.
    pub description: String,
    /// Input text.
    pub segments: String,
    /// Resolved entities passed with the input.
    #[serde(default)]
    pub resolved_entities: Vec<Value>,
    /// Expected output graph (`nodes`, `edges`).
    pub output: Value,
}

/// Fact extraction prompt.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractionPrompt {
    /// Free-text description.
    #[serde(default)]
    pub description: Option<String>,
    /// Extraction setting (`processing`).
    #[serde(default)]
    pub setting: Option<String>,
    /// Prompt version (single-pass).
    #[serde(default)]
    pub version: Option<String>,
    /// Pass number (multi-pass).
    #[serde(default)]
    pub pass_number: Option<u32>,
    /// Pass name used to prefix fact ids (multi-pass).
    #[serde(default)]
    pub pass_name: Option<String>,
    /// System messages, inlined ahead of the user template.
    #[serde(default)]
    pub system_messages: Vec<ChatMessage>,
    /// Worked examples.
    #[serde(default)]
    pub examples: Vec<PromptExample>,
    /// User prompt template.
    #[serde(default)]
    pub user_prompt_template: String,
}

impl ExtractionPrompt {
    /// Loads an extraction prompt file resolved with [`crate::assets::resolve_asset`].
    pub fn load(path: &str) -> Result<Self> {
        load_typed(None, path)
    }

    /// Single prompt text: trimmed non-empty system messages joined by blank lines, then the
    /// user template.
    pub fn combined_template(&self) -> String {
        let system_block = self
            .system_messages
            .iter()
            .map(|m| m.content.trim())
            .filter(|c| !c.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        let system_block = system_block.trim();
        match (system_block.is_empty(), self.user_prompt_template.is_empty()) {
            (false, false) => format!("{system_block}\n\n{}", self.user_prompt_template),
            (_, false) => self.user_prompt_template.clone(),
            _ => system_block.to_owned(),
        }
    }
}

/// Qwen contradiction judge prompt.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JudgePrompt {
    /// System message.
    pub system: String,
    /// User message template (`{fact_a}`, `{fact_b}`, ...).
    pub user_template: String,
    /// Description and example of the expected output.
    #[serde(default)]
    pub output_format: Option<Value>,
    /// Free-text notes.
    #[serde(default)]
    pub notes: Vec<String>,
}

impl JudgePrompt {
    /// Loads a judge prompt file resolved with [`crate::assets::resolve_asset`].
    pub fn load(path: &str) -> Result<Self> {
        load_typed(None, path)
    }
}
