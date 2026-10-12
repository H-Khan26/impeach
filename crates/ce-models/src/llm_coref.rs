//! LLM-prompted coreference: [`LlmCoref`], the `llm` backend of `models.coref`.
//!
//! The prompt ([`LlmCorefPrompt`], default [`LLM_COREF_PROMPT_PATH`]) has the same shape as the
//! ED prompt: `system_template` and `user_template` strings with Python `str.format`
//! placeholders (`{mentions_json}`, `{text}`; `{{` / `}}` are literal braces). The user prompt
//! lists the detected mentions as JSON (`index`, `text`, `type`, `start`) followed by the text.
//!
//! The LLM answers `{"clusters": [[item, ...], ...]}` (a bare top-level array is accepted too).
//! Items are mapped back to character spans and validated:
//! - `{"mention": i}` or a bare integer `i`: the span of listed mention `i`;
//! - `{"text": t, "start": s}`: `t` at `s` when the text there matches; otherwise the
//!   occurrence of `t` nearest to `s` (the earlier one on a tie); without `start`, the only
//!   occurrence of `t`;
//! - `[start, end]`: kept when `start < end <= len(text)`.
//!
//! Items that do not resolve (unknown index, text not in the document, bad offsets) are
//! dropped, as are duplicate spans; clusters left with fewer than two spans are dropped. A
//! response without a `clusters` array is a [`ModelError::Protocol`] error.

use std::sync::Arc;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::error::ModelError;
use crate::traits::{Coref, CorefCluster, CorefMention, Llm, LlmRequest};

/// Asset path of the shipped coreference prompt.
pub const LLM_COREF_PROMPT_PATH: &str = "contradiction_engine/prompts/coref/clusters.json";

/// Coreference prompt templates.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LlmCorefPrompt {
    /// System message (literal braces escaped as `{{` / `}}`).
    pub system_template: String,
    /// User message with `{mentions_json}` and `{text}` placeholders.
    pub user_template: String,
}

impl LlmCorefPrompt {
    /// Parses prompt JSON; `origin` names the source in errors. Both templates are checked for
    /// unknown placeholders and unbalanced braces.
    pub fn from_json_str(text: &str, origin: &str) -> Result<Self, ModelError> {
        let load_error = |message: String| ModelError::Load {
            model: "coref llm prompt".to_owned(),
            message: format!("{origin}: {message}"),
        };
        let prompt: Self = serde_json::from_str(text).map_err(|e| load_error(e.to_string()))?;
        fill_template(&prompt.system_template, &[]).map_err(load_error)?;
        fill_template(&prompt.user_template, &[("mentions_json", ""), ("text", "")])
            .map_err(load_error)?;
        Ok(prompt)
    }

    /// Loads a prompt file resolved with [`ce_config::resolve_asset`] (working directory first,
    /// then the embedded assets).
    pub fn load(path: &str) -> Result<Self, ModelError> {
        let load_error = |message: String| ModelError::Load {
            model: "coref llm prompt".to_owned(),
            message: format!("{path}: {message}"),
        };
        let asset = ce_config::resolve_asset(path).map_err(|e| load_error(e.to_string()))?;
        let text = asset.read_to_string().map_err(|e| load_error(e.to_string()))?;
        Self::from_json_str(&text, &asset.display_path())
    }
}

/// Fills Python `str.format` placeholders `{name}` from `values`; `{{` and `}}` are literal
/// braces. An unknown placeholder or a lone brace is an error.
fn fill_template(template: &str, values: &[(&str, &str)]) -> Result<String, String> {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(pos) = rest.find(['{', '}']) {
        out.push_str(&rest[..pos]);
        let tail = &rest[pos..];
        if tail.starts_with("{{") || tail.starts_with("}}") {
            out.push_str(&tail[..1]);
            rest = &tail[2..];
        } else if let Some(name) = tail.strip_prefix('{') {
            let end = name.find('}').ok_or_else(|| format!("unclosed `{{` in {template:?}"))?;
            let key = &name[..end];
            let value = values
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| *v)
                .ok_or_else(|| format!("unknown placeholder `{{{key}}}`"))?;
            out.push_str(value);
            rest = &name[end + 1..];
        } else {
            return Err(format!("single `}}` in {template:?}"));
        }
    }
    out.push_str(rest);
    Ok(out)
}

/// JSON schema of the expected answer (structured output).
fn response_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "clusters": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "mention": {"type": "integer"},
                            "text": {"type": "string"},
                            "start": {"type": "integer"}
                        }
                    }
                }
            }
        },
        "required": ["clusters"]
    })
}

/// Coreference by prompting an [`Llm`] (see the module docs).
pub struct LlmCoref {
    llm: Arc<dyn Llm>,
    prompt: LlmCorefPrompt,
    schema: Value,
}

impl std::fmt::Debug for LlmCoref {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmCoref").field("prompt", &self.prompt).finish_non_exhaustive()
    }
}

impl LlmCoref {
    /// A resolver over `llm` with the shipped prompt ([`LLM_COREF_PROMPT_PATH`]).
    ///
    /// # Errors
    /// [`ModelError::Load`] when the prompt cannot be read or is malformed.
    pub fn new(llm: Arc<dyn Llm>) -> Result<Self, ModelError> {
        Ok(Self::with_prompt(llm, LlmCorefPrompt::load(LLM_COREF_PROMPT_PATH)?))
    }

    /// A resolver over `llm` with `prompt`.
    pub fn with_prompt(llm: Arc<dyn Llm>, prompt: LlmCorefPrompt) -> Self {
        Self { llm, prompt, schema: response_schema() }
    }

    /// The system and user messages for `text` and `mentions`.
    fn messages(
        &self,
        text: &str,
        mentions: &[CorefMention<'_>],
    ) -> Result<(String, String), ModelError> {
        let listed: Vec<Value> = mentions
            .iter()
            .enumerate()
            .map(|(index, m)| {
                json!({"index": index, "text": m.text, "type": m.entity_type, "start": m.start})
            })
            .collect();
        let mentions_json = Value::Array(listed).to_string();
        let system = fill_template(&self.prompt.system_template, &[]);
        let user = fill_template(
            &self.prompt.user_template,
            &[("mentions_json", &mentions_json), ("text", text)],
        );
        match (system, user) {
            (Ok(system), Ok(user)) => Ok((system, user)),
            (Err(e), _) | (_, Err(e)) => {
                Err(ModelError::Load { model: "coref llm prompt".to_owned(), message: e })
            }
        }
    }
}

impl Coref for LlmCoref {
    fn resolve(&self, text: &str) -> Result<Vec<CorefCluster>, ModelError> {
        self.resolve_with_mentions(text, &[])
    }

    fn resolve_with_mentions(
        &self,
        text: &str,
        mentions: &[CorefMention<'_>],
    ) -> Result<Vec<CorefCluster>, ModelError> {
        let (system, user) = self.messages(text, mentions)?;
        let request = LlmRequest {
            system: Some(&system),
            prompt: &user,
            json_schema: Some(&self.schema),
            temperature: Some(0.0),
            num_ctx: None,
        };
        let response = self.llm.complete_json(&request)?;
        parse_clusters(&response, &DocText::new(text), mentions)
    }
}

/// Character view of the resolved text for offset validation and search.
struct DocText<'a> {
    text: &'a str,
    /// Byte offset of each character, plus the text length at the end.
    char_bytes: Vec<usize>,
}

impl<'a> DocText<'a> {
    fn new(text: &'a str) -> Self {
        let mut char_bytes: Vec<usize> = text.char_indices().map(|(b, _)| b).collect();
        char_bytes.push(text.len());
        Self { text, char_bytes }
    }

    fn char_len(&self) -> usize {
        self.char_bytes.len() - 1
    }

    fn char_at_byte(&self, byte: usize) -> usize {
        self.char_bytes.partition_point(|b| *b < byte)
    }

    /// Characters `start..end`, if in range.
    fn slice(&self, start: usize, end: usize) -> Option<&'a str> {
        (start <= end && end <= self.char_len())
            .then(|| &self.text[self.char_bytes[start]..self.char_bytes[end]])
    }

    /// The span of `needle` at `hint` or nearest to it; the only occurrence without a hint.
    fn locate(&self, needle: &str, hint: Option<usize>) -> Option<(usize, usize)> {
        let len = needle.chars().count();
        if len == 0 {
            return None;
        }
        if let Some(start) = hint {
            if self.slice(start, start + len) == Some(needle) {
                return Some((start, start + len));
            }
        }
        let mut occurrences =
            self.text.match_indices(needle).map(|(byte, _)| self.char_at_byte(byte));
        let start = match hint {
            Some(hint) => occurrences.min_by_key(|s| (s.abs_diff(hint), *s))?,
            None => {
                let first = occurrences.next()?;
                if occurrences.next().is_some() {
                    return None;
                }
                first
            }
        };
        Some((start, start + len))
    }
}

fn as_offset(value: &Value) -> Option<usize> {
    value.as_u64().and_then(|v| usize::try_from(v).ok())
}

/// One answer item as a character span.
fn resolve_item(
    item: &Value,
    doc: &DocText<'_>,
    mentions: &[CorefMention<'_>],
) -> Option<(usize, usize)> {
    let mention_span = |index: &Value| {
        as_offset(index)
            .and_then(|i| mentions.get(i))
            .filter(|m| m.start < m.end && m.end <= doc.char_len())
            .map(|m| (m.start, m.end))
    };
    match item {
        Value::Number(_) => mention_span(item),
        Value::Array(pair) => match pair.as_slice() {
            [start, end] => {
                let (start, end) = (as_offset(start)?, as_offset(end)?);
                (start < end && end <= doc.char_len()).then_some((start, end))
            }
            _ => None,
        },
        Value::Object(object) => {
            if let Some(index) = object.get("mention").filter(|v| !v.is_null()) {
                return mention_span(index);
            }
            let needle = object.get("text").and_then(Value::as_str)?;
            doc.locate(needle, object.get("start").and_then(as_offset))
        }
        _ => None,
    }
}

/// Maps an LLM answer onto character spans (see the module docs).
fn parse_clusters(
    response: &Value,
    doc: &DocText<'_>,
    mentions: &[CorefMention<'_>],
) -> Result<Vec<CorefCluster>, ModelError> {
    let clusters = match response {
        Value::Array(clusters) => clusters,
        Value::Object(object) => {
            object.get("clusters").and_then(Value::as_array).ok_or_else(|| {
                ModelError::Protocol(format!(
                    "coref LLM answer has no `clusters` array: {response}"
                ))
            })?
        }
        _ => {
            return Err(ModelError::Protocol(format!(
                "coref LLM answer is not a JSON object: {response}"
            )))
        }
    };
    let mut out = Vec::new();
    for cluster in clusters {
        let Some(items) = cluster.as_array() else {
            tracing::debug!(%cluster, "coref LLM cluster is not an array; dropped");
            continue;
        };
        let mut spans: Vec<(usize, usize)> = Vec::with_capacity(items.len());
        for item in items {
            match resolve_item(item, doc, mentions) {
                Some(span) if !spans.contains(&span) => spans.push(span),
                Some(_) => {}
                None => tracing::debug!(%item, "coref LLM item does not resolve; dropped"),
            }
        }
        if spans.len() >= 2 {
            spans.sort_unstable();
            out.push(CorefCluster { spans });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::mock::MockLlm;

    const TEXT: &str = "Zoë Brandt met Reed. She told Reed that she was late.";

    fn mentions() -> Vec<CorefMention<'static>> {
        vec![
            CorefMention { start: 0, end: 10, text: "Zoë Brandt", entity_type: "PER" },
            CorefMention { start: 15, end: 19, text: "Reed", entity_type: "PER" },
            CorefMention { start: 30, end: 34, text: "Reed", entity_type: "PER" },
        ]
    }

    fn coref(responses: &[&str]) -> (LlmCoref, Arc<MockLlm>) {
        let llm = Arc::new(MockLlm::with_responses(responses.iter().copied()));
        let coref = LlmCoref::new(Arc::clone(&llm) as Arc<dyn Llm>).expect("shipped prompt loads");
        (coref, llm)
    }

    #[test]
    fn fills_python_format_templates() {
        assert_eq!(
            fill_template("{{\"a\": {x}}} {y}", &[("x", "1"), ("y", "{z}")]).expect("fills"),
            "{\"a\": 1} {z}"
        );
        assert!(fill_template("{missing}", &[]).is_err());
        assert!(fill_template("a } b", &[]).is_err());
        assert!(fill_template("a { b", &[]).is_err());
    }

    #[test]
    fn maps_items_to_char_spans() {
        let (coref, llm) = coref(&[r#"```json
{"clusters": [
  [{"mention": 0}, {"text": "She", "start": 21}, {"text": "she", "start": 38}],
  [1, {"mention": 2}],
  [{"mention": 7}, {"text": "nobody"}]
]}
```"#]);
        let clusters = coref.resolve_with_mentions(TEXT, &mentions()).expect("resolves");
        assert_eq!(
            clusters,
            vec![
                CorefCluster { spans: vec![(0, 10), (21, 24), (40, 43)] },
                CorefCluster { spans: vec![(15, 19), (30, 34)] },
            ]
        );
        let prompt = &llm.prompts()[0];
        assert!(
            prompt.contains(r#"{"index":0,"text":"Zoë Brandt","type":"PER","start":0}"#),
            "{prompt}"
        );
        assert!(prompt.ends_with(TEXT), "{prompt}");
    }

    #[test]
    fn locates_text_by_nearest_or_unique_occurrence() {
        let doc = DocText::new(TEXT);
        assert_eq!(doc.locate("she", Some(41)), Some((40, 43)));
        assert_eq!(doc.locate("She", None), Some((21, 24)));
        assert_eq!(doc.locate("Reed", None), None, "ambiguous without a start");
        assert_eq!(doc.locate("Reed", Some(29)), Some((30, 34)));
        assert_eq!(doc.locate("Zoë", Some(3)), Some((0, 3)));
        assert_eq!(doc.locate("absent", Some(0)), None);
    }

    #[test]
    fn accepts_bare_arrays_and_offset_pairs() {
        let (coref, _) = coref(&["[[[0, 10], [21, 24], [5, 99]]]"]);
        let clusters = coref.resolve(TEXT).expect("resolves");
        assert_eq!(clusters, vec![CorefCluster { spans: vec![(0, 10), (21, 24)] }]);
    }

    #[test]
    fn malformed_answers_fail_loudly() {
        let (coref, _) = coref(&[r#"{"groups": []}"#, "\"no\""]);
        assert!(matches!(coref.resolve(TEXT), Err(ModelError::Protocol(_))));
        assert!(matches!(coref.resolve(TEXT), Err(ModelError::Protocol(_))));
        // An LLM failure propagates.
        assert!(matches!(coref.resolve(TEXT), Err(ModelError::Inference(_))));
    }

    #[test]
    fn prompt_validation() {
        let bad = r#"{"system_template": "x", "user_template": "{doc}"}"#;
        assert!(matches!(
            LlmCorefPrompt::from_json_str(bad, "test"),
            Err(ModelError::Load { ref message, .. }) if message.contains("{doc}")
        ));
        assert!(LlmCorefPrompt::from_json_str(r#"{"system_template": "x"}"#, "test").is_err());
    }
}
