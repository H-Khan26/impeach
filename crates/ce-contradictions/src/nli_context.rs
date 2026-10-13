//! Premise / hypothesis construction for the judge (Python
//! `services/contradictions/nli_context.py`).
//!
//! Enriched text: `Subject: <name> | Object: <name> | Time: <when> | Speaker: <who> | <text>`,
//! each prefix only when known. For an extracted fact the subject / object name is the
//! canonical name of its linked entity, else the trimmed assertion subject / object; time is
//! `t_valid_from [to t_valid_until]` (or `until t_valid_until`), else metadata `date` /
//! `event_date`. For a retrieved fact the subject / object come from its assertion triple and
//! time from metadata (`TemporalValidity` object, else `date` / `event_date`).
//!
//! Metadata values are rendered like Python `str()` (strings as is, `True` / `None`, numbers in
//! Python repr) and skipped when Python treats them as false (null, `false`, `0`, empty
//! strings, lists and objects).

use ce_core::pyjson::float_repr;
use ce_core::{EdResult, EntityId, ExtractedFact, Metadata, RetrievedFact};
use ce_text::normalize::py_strip;
use indexmap::IndexMap;
use serde_json::Value;

/// Separator between the enriched-text parts.
const SEPARATOR: &str = " | ";

/// Builds context-enriched fact text for the judge.
#[derive(Debug, Clone, Default)]
pub struct NliContextBuilder {
    entity_names: IndexMap<EntityId, String>,
}

impl NliContextBuilder {
    /// A builder resolving entity ids with `entity_names` (entity id -> canonical name).
    #[must_use]
    pub fn new(entity_names: IndexMap<EntityId, String>) -> Self {
        Self { entity_names }
    }

    /// Entity id -> canonical name.
    #[must_use]
    pub fn entity_names(&self) -> &IndexMap<EntityId, String> {
        &self.entity_names
    }

    /// Enriched text of an extracted fact (module docs).
    #[must_use]
    pub fn build_enriched_text(&self, fact: &ExtractedFact) -> String {
        let mut parts: Vec<String> = Vec::new();
        let triple = &fact.assertion_triple;
        if let Some(name) = self.entity_name(fact.subject_entity_id.as_ref(), &triple.subject) {
            parts.push(format!("Subject: {name}"));
        }
        if let Some(name) = self.entity_name(fact.object_entity_id.as_ref(), &triple.object) {
            parts.push(format!("Object: {name}"));
        }
        if let Some(time) = temporal_context(fact) {
            parts.push(format!("Time: {time}"));
        }
        if let Some(speaker) = truthy_str(fact.metadata.get("speaker")) {
            parts.push(format!("Speaker: {speaker}"));
        }
        parts.push(fact.text.clone());
        parts.join(SEPARATOR)
    }

    /// Enriched text of a retrieved fact (module docs).
    #[must_use]
    pub fn build_enriched_text_for_retrieved(&self, fact: &RetrievedFact) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(subject) = fact.assertion_subject.as_deref().filter(|s| !s.is_empty()) {
            parts.push(format!("Subject: {subject}"));
        }
        if let Some(object) = fact.assertion_object.as_deref().filter(|s| !s.is_empty()) {
            parts.push(format!("Object: {object}"));
        }
        if let Some(time) = temporal_from_metadata(&fact.metadata) {
            parts.push(format!("Time: {time}"));
        }
        if let Some(speaker) = truthy_str(fact.metadata.get("speaker")) {
            parts.push(format!("Speaker: {speaker}"));
        }
        parts.push(fact.fact_text.clone());
        parts.join(SEPARATOR)
    }

    /// The canonical name of `entity_id` when known (even if empty, as in Python), else the
    /// trimmed `fallback`; `None` when the result is empty.
    fn entity_name<'a>(
        &'a self,
        entity_id: Option<&EntityId>,
        fallback: &'a str,
    ) -> Option<&'a str> {
        let known = entity_id.filter(|id| !id.is_empty()).and_then(|id| self.entity_names.get(id));
        let name = match known {
            Some(name) => name.as_str(),
            None => py_strip(fallback),
        };
        (!name.is_empty()).then_some(name)
    }
}

/// Time of an extracted fact: its validity interval, else metadata `date` / `event_date`.
fn temporal_context(fact: &ExtractedFact) -> Option<String> {
    let validity = &fact.temporal_validity;
    let from = validity.t_valid_from.as_deref().filter(|s| !s.is_empty());
    let until = validity.t_valid_until.as_deref().filter(|s| !s.is_empty());
    match (from, until) {
        (Some(from), Some(until)) => Some(format!("{from} to {until}")),
        (Some(from), None) => Some(from.to_owned()),
        (None, Some(until)) => Some(format!("until {until}")),
        (None, None) => metadata_date(&fact.metadata),
    }
}

/// Time of a retrieved fact: the metadata `TemporalValidity` object, else `date` /
/// `event_date`.
fn temporal_from_metadata(metadata: &Metadata) -> Option<String> {
    if let Some(Value::Object(validity)) = metadata.get("TemporalValidity") {
        let mut parts = Vec::new();
        if let Some(from) = truthy_str(validity.get("t_valid_from")) {
            parts.push(from);
        }
        if let Some(until) = truthy_str(validity.get("t_valid_until")) {
            parts.push(format!("to {until}"));
        }
        if !parts.is_empty() {
            return Some(parts.join(" "));
        }
    }
    metadata_date(metadata)
}

/// Metadata `date`, else `event_date` (Python `get("date") or get("event_date")`).
fn metadata_date(metadata: &Metadata) -> Option<String> {
    truthy_str(metadata.get("date")).or_else(|| truthy_str(metadata.get("event_date")))
}

/// Python `str(value)` of a present value Python treats as true.
fn truthy_str(value: Option<&Value>) -> Option<String> {
    value.filter(|v| py_truthy(v)).map(py_str)
}

/// Python truthiness of a JSON value.
pub(crate) fn py_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|x| x != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// Python `str()` of a JSON value: strings as is, everything else in Python repr.
pub(crate) fn py_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => py_repr(other),
    }
}

/// Python `repr()` of a JSON value.
fn py_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                n.to_string()
            } else {
                n.as_f64().map_or_else(|| n.to_string(), float_repr)
            }
        }
        Value::String(s) => py_str_repr(s),
        Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(py_repr).collect();
            format!("[{}]", inner.join(", "))
        }
        Value::Object(map) => {
            let inner: Vec<String> =
                map.iter().map(|(k, v)| format!("{}: {}", py_str_repr(k), py_repr(v))).collect();
            format!("{{{}}}", inner.join(", "))
        }
    }
}

/// Python `repr()` of a string: single quotes unless it contains `'` and no `"`.
fn py_str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Entity id -> canonical name from ED results: for each result with a chosen entity, the
/// name of the candidate with that id (first match kept per result; later results overwrite).
///
/// The first candidate with that id and a non-empty name wins. Python read the attributes
/// `id` / `name`, which `EDCandidate` does not have, so its map was always empty; this reads
/// `entity_id` / `canonical_name`.
#[must_use]
pub fn build_entity_name_map(ed_results: &[EdResult]) -> IndexMap<EntityId, String> {
    let mut names = IndexMap::new();
    for result in ed_results {
        let Some(chosen) = result.chosen_entity_id.as_ref().filter(|id| !id.is_empty()) else {
            continue;
        };
        if let Some(candidate) = result
            .candidates
            .iter()
            .find(|c| &c.entity_id == chosen && !c.canonical_name.is_empty())
        {
            names.insert(chosen.clone(), candidate.canonical_name.clone());
        }
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn py_str_matches_python() {
        assert_eq!(py_str(&json!("2020-03-15")), "2020-03-15");
        assert_eq!(py_str(&json!(2020)), "2020");
        assert_eq!(py_str(&json!(1.5)), "1.5");
        assert_eq!(py_str(&json!(1e20)), "1e+20");
        assert_eq!(py_str(&json!(true)), "True");
        assert_eq!(py_str(&json!(null)), "None");
        assert_eq!(py_str(&json!(["a", 1, null])), "['a', 1, None]");
        assert_eq!(py_str(&json!({"k": "it's"})), "{'k': \"it's\"}");
    }

    #[test]
    fn truthiness_matches_python() {
        let falsy =
            [json!(null), json!(false), json!(0), json!(0.0), json!(""), json!([]), json!({})];
        for value in falsy {
            assert!(!py_truthy(&value), "{value} is false");
        }
        let truthy = [json!(true), json!(-1), json!(0.5), json!(" "), json!([0]), json!({"a": 0})];
        for value in truthy {
            assert!(py_truthy(&value), "{value} is true");
        }
    }
}
