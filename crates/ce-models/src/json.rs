//! Lenient JSON extraction from LLM output.
//!
//! The Python services parse LLM output by stripping a Markdown code fence and calling
//! `json.loads` ([`strip_code_fences`], [`parse_fenced_json`]). [`extract_json`] starts with
//! exactly that and, when it fails, falls back to recovering the JSON value from surrounding
//! prose: a fenced block anywhere in the text, then the first balanced object or array that
//! parses. Each candidate is also retried after [`repair_json`] (trailing commas, raw control
//! characters inside strings, bare `NaN` / `Infinity`).

use std::borrow::Cow;

use serde_json::Value;

use crate::error::ModelError;

/// Maximum number of `{` / `[` positions tried when searching prose for a JSON value.
const MAX_OPENERS: usize = 64;

/// Strips a Markdown code fence like the Python services: trims whitespace, removes a leading
/// `` ```json `` and then a leading `` ``` ``, then a trailing `` ``` ``. The result is not
/// trimmed again (`json.loads` ignores surrounding whitespace).
pub fn strip_code_fences(text: &str) -> &str {
    let mut clean = text.trim();
    if let Some(rest) = clean.strip_prefix("```json") {
        clean = rest;
    }
    if let Some(rest) = clean.strip_prefix("```") {
        clean = rest;
    }
    if let Some(rest) = clean.strip_suffix("```") {
        clean = rest;
    }
    clean
}

/// Parses LLM output exactly as the Python fact extractor and ED service do: strip the code
/// fence, then parse strictly.
pub fn parse_fenced_json(text: &str) -> Result<Value, ModelError> {
    serde_json::from_str(strip_code_fences(text).trim())
        .map_err(|e| ModelError::Protocol(format!("LLM output is not valid JSON: {e}")))
}

/// Parses the JSON value contained in an LLM completion, tolerating code fences, surrounding
/// prose, trailing commas, raw control characters in strings and bare `NaN` / `Infinity`
/// (read as `null`).
pub fn extract_json(text: &str) -> Result<Value, ModelError> {
    let stripped = strip_code_fences(text).trim();
    let first_error = match serde_json::from_str::<Value>(stripped) {
        Ok(value) => return Ok(value),
        Err(e) => e,
    };
    if let Some(value) = parse_repaired(stripped) {
        return Ok(value);
    }
    if let Some(block) = fenced_block(text) {
        if let Some(value) = parse_candidate(block) {
            return Ok(value);
        }
    }
    if let Some(value) = first_embedded_value(stripped) {
        return Ok(value);
    }
    Err(ModelError::Protocol(format!(
        "LLM output is not valid JSON: {first_error}; output starts with {:?}",
        excerpt(text, 200)
    )))
}

/// The first `{...}` substring that contains `"<key>"` and no `}` between its `{` and the
/// closing `}` (Python `re.search(r'\{[^}]*"<key>"[^}]*\}', text, re.DOTALL)`, used by the
/// Qwen judge to pull a flat verdict object out of free text).
pub fn find_flat_object_with_key<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let quoted = format!("\"{key}\"");
    for (open, _) in text.match_indices('{') {
        let Some(close_rel) = text[open..].find('}') else {
            // No `}` after this `{`, so none after any later `{` either.
            return None;
        };
        let candidate = &text[open..=open + close_rel];
        if candidate.contains(&quoted) {
            return Some(candidate);
        }
    }
    None
}

/// Applies text-level repairs outside of a JSON parser: drops trailing commas before `}` / `]`,
/// escapes raw control characters inside strings and replaces bare `NaN`, `Infinity` and
/// `-Infinity` outside strings with `null`. Returns the input unchanged when nothing applies.
pub fn repair_json(text: &str) -> Cow<'_, str> {
    let mut out = String::with_capacity(text.len());
    let mut changed = false;
    let mut in_string = false;
    let mut escaped = false;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if in_string {
            if escaped {
                escaped = false;
                out.push(c);
            } else if c == '\\' {
                escaped = true;
                out.push(c);
            } else if c == '"' {
                in_string = false;
                out.push(c);
            } else if (c as u32) < 0x20 {
                changed = true;
                match c {
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    other => out.push_str(&format!("\\u{:04x}", other as u32)),
                }
            } else {
                out.push(c);
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                out.push(c);
            }
            ',' => {
                let rest = &text[i + 1..];
                let next = rest.trim_start().chars().next();
                if matches!(next, Some('}' | ']')) {
                    changed = true;
                } else {
                    out.push(c);
                }
            }
            _ => {
                let rest = &text[i..];
                let replaced = ["-Infinity", "Infinity", "NaN"].into_iter().find(|word| {
                    rest.starts_with(word)
                        && !rest[word.len()..].starts_with(|ch: char| ch.is_alphanumeric())
                        && !out.ends_with(|ch: char| ch.is_alphanumeric() || ch == '_')
                });
                if let Some(word) = replaced {
                    changed = true;
                    out.push_str("null");
                    for _ in 1..word.chars().count() {
                        chars.next();
                    }
                } else {
                    out.push(c);
                }
            }
        }
    }
    if changed {
        Cow::Owned(out)
    } else {
        Cow::Borrowed(text)
    }
}

/// Parses `candidate` strictly, then after [`repair_json`].
fn parse_candidate(candidate: &str) -> Option<Value> {
    let candidate = candidate.trim();
    serde_json::from_str(candidate).ok().or_else(|| parse_repaired(candidate))
}

fn parse_repaired(candidate: &str) -> Option<Value> {
    match repair_json(candidate) {
        Cow::Owned(repaired) => serde_json::from_str(&repaired).ok(),
        Cow::Borrowed(_) => None,
    }
}

/// Content of the first fenced block (`` ```json ... ``` `` or `` ``` ... ``` ``) anywhere in
/// `text`.
fn fenced_block(text: &str) -> Option<&str> {
    let start = text.find("```")?;
    let after = &text[start + 3..];
    // Skip an info string such as `json` up to the end of the line.
    let body_start = after
        .find('\n')
        .filter(|&nl| after[..nl].trim().chars().all(|c| c.is_ascii_alphanumeric()))
        .map_or(0, |nl| nl + 1);
    let body = &after[body_start..];
    let end = body.find("```")?;
    Some(&body[..end])
}

/// The first balanced `{...}` / `[...]` substring of `text` that parses (strictly or after
/// repair), trying at most [`MAX_OPENERS`] opening positions.
fn first_embedded_value(text: &str) -> Option<Value> {
    text.char_indices().filter(|(_, c)| matches!(c, '{' | '[')).take(MAX_OPENERS).find_map(
        |(i, _)| balanced_end(&text[i..]).and_then(|end| parse_candidate(&text[i..i + end])),
    )
}

/// Byte length of the balanced JSON-like value at the start of `text` (which starts with `{` or
/// `[`), skipping brackets inside strings. `None` when it never closes or nesting mismatches.
fn balanced_end(text: &str) -> Option<usize> {
    let mut stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    for (i, c) in text.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' => {
                if stack.pop() != Some(c) {
                    return None;
                }
                if stack.is_empty() {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// The first `max_chars` characters of `text` (for log excerpts).
#[must_use]
pub fn excerpt(text: &str, max_chars: usize) -> &str {
    match text.char_indices().nth(max_chars) {
        Some((i, _)) => &text[..i],
        None => text,
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;

    #[test]
    fn excerpt_counts_characters() {
        assert_eq!(excerpt("héllo world", 5), "héllo");
        assert_eq!(excerpt("abc", 30), "abc");
    }

    #[test]
    fn strip_code_fences_matches_python() {
        assert_eq!(strip_code_fences("  ```json\n{\"a\": 1}\n```  "), "\n{\"a\": 1}\n");
        assert_eq!(strip_code_fences("```\n[1]\n```"), "\n[1]\n");
        assert_eq!(strip_code_fences("{\"a\": 1}"), "{\"a\": 1}");
        // ```json then ``` are both removed when both are present.
        assert_eq!(strip_code_fences("```json```{}```"), "{}");
        // Only a trailing fence.
        assert_eq!(strip_code_fences("{}\n```"), "{}\n");
        // ```JSON (upper case) is not the json prefix: only ``` is removed.
        assert_eq!(strip_code_fences("```JSON\n{}```"), "JSON\n{}");
    }

    #[test]
    fn parse_fenced_json_is_strict() {
        assert_eq!(
            parse_fenced_json("```json\n{\"nodes\": []}\n```").expect("fenced"),
            json!({"nodes": []})
        );
        assert!(matches!(
            parse_fenced_json("Here you go: {\"a\": 1}"),
            Err(ModelError::Protocol(_))
        ));
    }

    #[test]
    fn extract_plain_and_fenced() {
        assert_eq!(extract_json("{\"a\": 1}").expect("plain"), json!({"a": 1}));
        assert_eq!(extract_json(" [1, 2] ").expect("array"), json!([1, 2]));
        assert_eq!(
            extract_json("```json\n{\"a\": {\"b\": [true, null]}}\n```").expect("fenced"),
            json!({"a": {"b": [true, null]}})
        );
        assert_eq!(extract_json("```\n{\"a\": 1}\n```").expect("bare fence"), json!({"a": 1}));
    }

    #[test]
    fn extract_from_prose() {
        let text = "Sure! Here is the JSON you asked for:\n{\"verdict\": \"neutral\", \"confidence\": 0.7}\nLet me know if you need anything else.";
        assert_eq!(
            extract_json(text).expect("prose"),
            json!({"verdict": "neutral", "confidence": 0.7})
        );
    }

    #[test]
    fn extract_fenced_block_inside_prose() {
        let text = "The result:\n```json\n{\"nodes\": [{\"id\": \"F1\"}]}\n```\nDone.";
        assert_eq!(extract_json(text).expect("inner fence"), json!({"nodes": [{"id": "F1"}]}));
    }

    #[test]
    fn extract_skips_bracketed_prose_before_value() {
        let text = "[note] the answer follows: {\"a\": [1, 2]} [end]";
        assert_eq!(extract_json(text).expect("skip prose brackets"), json!({"a": [1, 2]}));
    }

    #[test]
    fn extract_ignores_brackets_in_strings() {
        let text = "x {\"text\": \"a } and ] inside\", \"n\": 1} y";
        assert_eq!(
            extract_json(text).expect("strings"),
            json!({"text": "a } and ] inside", "n": 1})
        );
    }

    #[test]
    fn extract_repairs_trailing_commas() {
        let text = "```json\n{\"a\": [1, 2, ], \"b\": {\"c\": 3,},}\n```";
        assert_eq!(
            extract_json(text).expect("trailing commas"),
            json!({"a": [1, 2], "b": {"c": 3}})
        );
    }

    #[test]
    fn extract_repairs_raw_newlines_in_strings() {
        let text = "{\"text\": \"line one\nline two\"}";
        assert_eq!(extract_json(text).expect("newline"), json!({"text": "line one\nline two"}));
    }

    #[test]
    fn extract_reads_non_finite_numbers_as_null() {
        let text = "{\"a\": NaN, \"b\": -Infinity, \"c\": \"NaN\", \"d\": [Infinity]}";
        assert_eq!(
            extract_json(text).expect("non-finite"),
            json!({"a": null, "b": null, "c": "NaN", "d": [null]})
        );
    }

    #[test]
    fn extract_keeps_commas_inside_strings() {
        assert_eq!(repair_json("{\"a\": \"x, }\"}"), Cow::Borrowed("{\"a\": \"x, }\"}"));
    }

    #[test]
    fn extract_fails_without_json() {
        let err = extract_json("I cannot answer that.").expect_err("no json");
        assert!(matches!(err, ModelError::Protocol(_)));
        assert!(extract_json("").is_err());
        assert!(extract_json("{\"a\": 1").is_err());
    }

    #[test]
    fn nested_json_strings_stay_strings() {
        // The extraction prompt asks for nested objects encoded as strings.
        let text = "{\"nodes\": [{\"assertion_triple\": \"{\\\"subject\\\": \\\"A\\\"}\"}]}";
        let value = extract_json(text).expect("nested");
        assert_eq!(value["nodes"][0]["assertion_triple"], json!("{\"subject\": \"A\"}"));
    }

    #[test]
    fn flat_object_with_key_matches_python_regex() {
        let text = "Reasoning... {\"verdict\": \"contradiction\", \"confidence\": 0.9} trailing";
        assert_eq!(
            find_flat_object_with_key(text, "verdict"),
            Some("{\"verdict\": \"contradiction\", \"confidence\": 0.9}")
        );
        // The earliest `{` wins even when another `{` precedes the key before any `}`.
        assert_eq!(
            find_flat_object_with_key("{ x { \"verdict\": 1 }", "verdict"),
            Some("{ x { \"verdict\": 1 }")
        );
        // A `}` before the key ends that candidate; the next `{` is tried.
        assert_eq!(
            find_flat_object_with_key("{a} {\"verdict\": 2}", "verdict"),
            Some("{\"verdict\": 2}")
        );
        assert_eq!(find_flat_object_with_key("{\"label\": 1}", "verdict"), None);
        assert_eq!(find_flat_object_with_key("{\"verdict\": 1", "verdict"), None);
    }
}
