//! LLM-prompted coreference: [`LlmCoref`], the `llm` backend of `models.coref`.
//!
//! The prompt (default [`LLM_COREF_PROMPT_PATH`]) is an ED-shaped prompt file loaded as a
//! [`EdPrompt`]: `system_template` and `user_template` strings with Python `str.format`
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

use ce_config::prompts::EdPrompt;
use serde_json::{json, Value};

use crate::error::ModelError;
use crate::traits::{Coref, CorefCluster, CorefMention, Llm, LlmRequest};

/// Asset path of the shipped coreference prompt.
pub const LLM_COREF_PROMPT_PATH: &str = "contradiction_engine/prompts/coref/clusters.json";

/// Placeholders the user template must use (and the only ones it may use).
const USER_PLACEHOLDERS: [&str; 2] = ["mentions_json", "text"];

/// A [`ModelError::Load`] for the coreference prompt.
fn prompt_error(message: impl std::fmt::Display) -> ModelError {
    ModelError::Load { model: "coref llm prompt".to_owned(), message: message.to_string() }
}

/// One piece of a parsed `str.format` template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Piece<'a> {
    /// Literal text (`{{` / `}}` already unescaped to one brace).
    Literal(&'a str),
    /// A `{name}` placeholder.
    Placeholder(&'a str),
}

/// Parses a Python `str.format` template: `{name}` placeholders, `{{` and `}}` literal
/// braces. An unclosed `{` or a single `}` is an error.
fn parse_template(template: &str) -> Result<Vec<Piece<'_>>, String> {
    let mut pieces = Vec::new();
    let mut rest = template;
    while let Some(pos) = rest.find(['{', '}']) {
        if pos > 0 {
            pieces.push(Piece::Literal(&rest[..pos]));
        }
        let tail = &rest[pos..];
        if tail.starts_with("{{") || tail.starts_with("}}") {
            pieces.push(Piece::Literal(&tail[..1]));
            rest = &tail[2..];
        } else if let Some(name) = tail.strip_prefix('{') {
            let end = name.find('}').ok_or_else(|| format!("unclosed `{{` in {template:?}"))?;
            pieces.push(Piece::Placeholder(&name[..end]));
            rest = &name[end + 1..];
        } else {
            return Err(format!("single `}}` in {template:?}"));
        }
    }
    if !rest.is_empty() {
        pieces.push(Piece::Literal(rest));
    }
    Ok(pieces)
}

/// The distinct placeholder names of `template`, sorted.
fn placeholders(template: &str) -> Result<Vec<&str>, String> {
    let mut names: Vec<&str> = parse_template(template)?
        .into_iter()
        .filter_map(|piece| match piece {
            Piece::Placeholder(name) => Some(name),
            Piece::Literal(_) => None,
        })
        .collect();
    names.sort_unstable();
    names.dedup();
    Ok(names)
}

/// Fills the placeholders of `template` from `values`; an unknown placeholder is an error.
fn fill_template(template: &str, values: &[(&str, &str)]) -> Result<String, String> {
    let mut out = String::with_capacity(template.len());
    for piece in parse_template(template)? {
        match piece {
            Piece::Literal(text) => out.push_str(text),
            Piece::Placeholder(name) => {
                let value = values
                    .iter()
                    .find(|(k, _)| *k == name)
                    .map(|(_, v)| *v)
                    .ok_or_else(|| format!("unknown placeholder `{{{name}}}`"))?;
                out.push_str(value);
            }
        }
    }
    Ok(out)
}

/// Checks `prompt`: the system template has no placeholders, the user template has exactly
/// `{mentions_json}` and `{text}`, and braces are balanced. `origin` (a file path), when
/// known, prefixes error messages.
fn validate_prompt(prompt: &EdPrompt, origin: Option<&str>) -> Result<(), ModelError> {
    let fail = |message: String| match origin {
        Some(origin) => prompt_error(format!("{origin}: {message}")),
        None => prompt_error(message),
    };
    let system = placeholders(&prompt.system_template).map_err(fail)?;
    if let Some(name) = system.first() {
        return Err(fail(format!(
            "system_template has placeholder `{{{name}}}`; none are allowed"
        )));
    }
    let user = placeholders(&prompt.user_template).map_err(fail)?;
    if user != USER_PLACEHOLDERS {
        return Err(fail(format!(
            "user_template must contain exactly the placeholders {{mentions_json}} and {{text}}, \
             found {user:?}"
        )));
    }
    Ok(())
}

/// Loads and validates the prompt file at `path` (load errors name the path already).
fn load_prompt(path: &str) -> Result<EdPrompt, ModelError> {
    let prompt = EdPrompt::load(path).map_err(prompt_error)?;
    validate_prompt(&prompt, Some(path))?;
    Ok(prompt)
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
    prompt: EdPrompt,
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
    /// [`ModelError::Load`] when the prompt cannot be read or fails the checks of
    /// [`LlmCoref::with_prompt`].
    pub fn new(llm: Arc<dyn Llm>) -> Result<Self, ModelError> {
        Ok(Self::build(llm, load_prompt(LLM_COREF_PROMPT_PATH)?))
    }

    /// A resolver over `llm` with `prompt`.
    ///
    /// # Errors
    /// [`ModelError::Load`] unless `user_template` contains exactly the placeholders
    /// `{mentions_json}` and `{text}` and `system_template` contains none, with balanced
    /// braces (`{{` / `}}` for literal ones) in both.
    pub fn with_prompt(llm: Arc<dyn Llm>, prompt: EdPrompt) -> Result<Self, ModelError> {
        validate_prompt(&prompt, None)?;
        Ok(Self::build(llm, prompt))
    }

    fn build(llm: Arc<dyn Llm>, prompt: EdPrompt) -> Self {
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
        let system = fill_template(&self.prompt.system_template, &[]).map_err(prompt_error)?;
        let values = [(USER_PLACEHOLDERS[0], mentions_json.as_str()), (USER_PLACEHOLDERS[1], text)];
        let user = fill_template(&self.prompt.user_template, &values).map_err(prompt_error)?;
        Ok((system, user))
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
        if let Some((start, end)) = hint.and_then(|s| Some((s, s.checked_add(len)?))) {
            if self.slice(start, end) == Some(needle) {
                return Some((start, end));
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
        assert_eq!(placeholders("{b} {a} {{c}} {a}").expect("parses"), vec!["a", "b"]);
        assert!(placeholders("{{").expect("parses").is_empty());
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
        // A huge start hint does not overflow; the nearest occurrence wins.
        assert_eq!(doc.locate("she", Some(usize::MAX)), Some((40, 43)));
    }

    #[test]
    fn huge_llm_offsets_are_dropped_or_relocated() {
        let max = u64::MAX;
        let answer = format!(
            r#"{{"clusters": [[{{"mention": 0}}, {{"text": "She", "start": {max}}}, [{max}, {max}]]]}}"#
        );
        let (coref, _) = coref(&[&answer]);
        let clusters = coref.resolve_with_mentions(TEXT, &mentions()).expect("resolves");
        assert_eq!(clusters, vec![CorefCluster { spans: vec![(0, 10), (21, 24)] }]);
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
        let prompt = |system: &str, user: &str| EdPrompt {
            system_template: system.to_owned(),
            user_template: user.to_owned(),
        };
        let llm = || Arc::new(MockLlm::new()) as Arc<dyn Llm>;
        assert!(matches!(
            LlmCoref::with_prompt(llm(), prompt("x", "{doc}")),
            Err(ModelError::Load { ref message, .. }) if message.contains("found [\"doc\"]")
        ));
        assert!(LlmCoref::with_prompt(llm(), prompt("{text}", "{mentions_json} {text}")).is_err());
        assert!(LlmCoref::with_prompt(llm(), prompt("x", "{mentions_json}")).is_err());
        assert!(LlmCoref::with_prompt(llm(), prompt("x", "{{text}} {mentions_json}")).is_err());
        assert!(LlmCoref::with_prompt(llm(), prompt("x", "{mentions_json} {text} {x}")).is_err());
        assert!(LlmCoref::with_prompt(llm(), prompt("{{x}}", "{mentions_json} {text}")).is_ok());

        // A prompt file without a user template does not load.
        let partial: EdPrompt =
            serde_json::from_str(r#"{"system_template": "x"}"#).expect("ED-shaped JSON");
        assert!(matches!(
            LlmCoref::with_prompt(llm(), partial),
            Err(ModelError::Load { ref message, .. }) if message.contains("user_template must contain")
        ));
        // Errors from files name the path.
        let missing = "contradiction_engine/prompts/coref/missing.json";
        assert!(matches!(
            load_prompt(missing),
            Err(ModelError::Load { ref message, .. }) if message.contains(missing)
        ));
    }
}
