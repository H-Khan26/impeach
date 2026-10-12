//! LLM fact extraction (Python `services/facts/extractor.py`).
//!
//! A prompt template ([`ExtractionPrompt::combined_template`]) is rendered per chunk with
//! [`render_prompt`] (plain `{segments}` / `{resolved_entities}` substitution, so the JSON
//! examples in the prompts keep their braces), sent to the [`Llm`] in one call, and the
//! response is parsed by [`parse_facts`] into [`ExtractedFact`]s: the `STATED_FACT` nodes of the
//! `{"nodes": [...]}` graph, with string-encoded `assertion_triple` / `temporal_validity` /
//! `metadata` objects decoded. [`LlmFactExtractor`] runs one prompt; [`MultiPassFactExtractor`]
//! runs several focused prompts (optionally on a scoped thread pool), prefixes ids with the
//! pass name and merges the results with [`deduplicate_facts`].
//!
//! `value_type` / `property_type` stay raw on the fact; they are normalised through the
//! taxonomy with [`ExtractedFact::normalized_value_type`] /
//! [`ExtractedFact::normalized_property_type`] (`EXISTS -> EXISTENCE` lives there).
//!
//! # Differences from the Python prototype
//! - The response is parsed with [`ce_models::json::extract_json`]: after the Python fence
//!   strip + strict parse it also repairs trailing commas, raw control characters in strings
//!   and prose around the JSON (Python lost the whole response).
//! - Truncated output (a model hitting its output limit mid-list) is salvaged: when the
//!   response does not parse into an object with `nodes` but holds a `"nodes": [` list, every
//!   complete object of that list that parses is kept (this also reads a list nested deeper,
//!   such as `{"graph": {"nodes": [...]}}`, which Python treated as no facts).
//! - Malformed nodes are skipped one at a time instead of losing the whole response: a node
//!   that is not an object, a nested field that decodes to a non-object (treated as `{}`), a
//!   non-string `time_window.start` / `end` or `t_valid_*` (read as missing), numeric
//!   `text` / triple parts / `value_raw` (stringified) no longer raise.
//! - A non-object top level or a non-list `nodes` is a [`FactError::Parse`] (logged, no facts),
//!   as before, but reported instead of swallowed.
//! - `certainty` is clamped to `[0, 1]` (the graph rejects other values) and non-finite values
//!   fall back to [`DEFAULT_CERTAINTY`].
//! - Missing ids become `STATED_FACT_<n>` (`n` = position among the `STATED_FACT` nodes) instead
//!   of `STATED_FACT_X`, and a repeated id within one response is replaced the same way, so ids
//!   are unique per response.
//! - `raw_node` is the node exactly as received (Python's `time_window` pop also edited the
//!   node's own metadata object when it was not string-encoded).
//! - Parallel passes are merged in pass order (Python: completion order), so the output is
//!   deterministic; `max_workers` 0 counts as 1.
//! - No prompts yields an extractor that returns no facts (Python built one with an empty
//!   template, which behaves the same).

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ce_config::prompts::ExtractionPrompt;
use ce_core::{
    AssertionTriple, EntityId, ExtractedFact, TemporalContext, TemporalValidity, TimeWindow,
};
use ce_models::json::{excerpt, extract_json, repair_json};
use ce_models::{Llm, LlmRequest};
use ce_text::normalize::py_strip;
use serde_json::{Map, Value};
use tracing::{debug, error, warn};

use crate::error::{FactError, Result};

/// Certainty of a fact whose `certainty` is missing or not a number.
pub const DEFAULT_CERTAINTY: f64 = 0.5;

/// Number of leading characters of the fact text that enter the [`fact_key`].
const FACT_KEY_TEXT_CHARS: usize = 50;

/// One resolved entity listed in the `{resolved_entities}` block of a prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptEntity {
    /// Entity id.
    pub id: EntityId,
    /// Canonical name.
    pub name: String,
    /// Entity type; `None` prints as `ENTITY`.
    pub entity_type: Option<String>,
    /// Aliases shown to the LLM (already filtered).
    pub aliases: Vec<String>,
}

/// Extracts facts from the text of one chunk.
pub trait FactExtractor: Send + Sync + fmt::Debug {
    /// Facts of `chunk_text`; `entities` fills the `{resolved_entities}` placeholder. LLM and
    /// parse failures are logged and yield no facts.
    fn extract_facts(&self, chunk_text: &str, entities: &[PromptEntity]) -> Vec<ExtractedFact>;
}

/// The `{resolved_entities}` block: empty without entities, else
/// `Resolved entities in this chunk:` and one `- name (ID: id, TYPE: type[, Aliases: "a", "b"])`
/// line per entity.
///
/// The `Aliases` part appears when the alias list is non-empty; empty aliases are dropped from
/// the printed list (so a list of only empty strings prints `Aliases: ` and nothing else, as in
/// Python).
#[must_use]
pub fn build_entity_context(entities: &[PromptEntity]) -> String {
    if entities.is_empty() {
        return String::new();
    }
    let lines: Vec<String> = entities
        .iter()
        .map(|entity| {
            let entity_type = entity.entity_type.as_deref().filter(|t| !t.is_empty());
            let entity_type = entity_type.unwrap_or("ENTITY");
            if entity.aliases.is_empty() {
                format!("- {} (ID: {}, TYPE: {entity_type})", entity.name, entity.id)
            } else {
                let aliases = entity
                    .aliases
                    .iter()
                    .filter(|alias| !alias.is_empty())
                    .map(|alias| format!("\"{alias}\""))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "- {} (ID: {}, TYPE: {entity_type}, Aliases: {aliases})",
                    entity.name, entity.id
                )
            }
        })
        .collect();
    format!("Resolved entities in this chunk:\n{}", lines.join("\n"))
}

/// Substitutes `{segments}` with `segments`, then `{resolved_entities}` with `entity_context`
/// (plain string replacement; other braces are left alone).
#[must_use]
pub fn render_prompt(template: &str, segments: &str, entity_context: &str) -> String {
    template.replace("{segments}", segments).replace("{resolved_entities}", entity_context)
}

/// Parses an LLM response into facts: `STATED_FACT` nodes of the `nodes` list, ids prefixed
/// with `pass_name` when given and not already starting with it.
///
/// A response without a `nodes` key yields no facts. A response that does not parse as a
/// whole but holds a `nodes` list (typically output cut off by the model's length limit)
/// yields the facts of the list's complete objects.
///
/// # Errors
/// [`crate::FactError::Parse`] when the response holds no JSON object, or `nodes` is not a list.
pub fn parse_facts(response: &str, pass_name: Option<&str>) -> Result<Vec<ExtractedFact>> {
    let nodes = response_nodes(response)?;
    Ok(facts_from_nodes(&nodes, pass_name))
}

/// Deduplication key of a fact: `subject|RELATIONSHIP|object|first 50 chars of text`.
#[must_use]
pub fn fact_key(fact: &ExtractedFact) -> String {
    let triple = &fact.assertion_triple;
    let subject = triple.subject.to_lowercase();
    let relationship = triple.relationship.to_uppercase();
    let object = triple.object.to_lowercase();
    let text = fact.text.to_lowercase();
    let text: String = py_strip(&text).chars().take(FACT_KEY_TEXT_CHARS).collect();
    format!("{}|{}|{}|{text}", py_strip(&subject), py_strip(&relationship), py_strip(&object))
}

/// Removes facts with the same [`fact_key`], keeping the one with the higher certainty.
///
/// The first occurrence keeps its position; a later duplicate with strictly higher certainty
/// replaces it by removing it and appending the newcomer at the end (Python order).
#[must_use]
pub fn deduplicate_facts(facts: Vec<ExtractedFact>) -> Vec<ExtractedFact> {
    let total = facts.len();
    let mut slots: Vec<Option<ExtractedFact>> = Vec::with_capacity(total);
    let mut seen: HashMap<String, usize> = HashMap::with_capacity(total);
    for fact in facts {
        let key = fact_key(&fact);
        match seen.get(&key).copied() {
            Some(slot) => {
                let existing = slots[slot].as_ref().expect("seen keys point at occupied slots");
                if fact.certainty > existing.certainty {
                    slots[slot] = None;
                    seen.insert(key, slots.len());
                    slots.push(Some(fact));
                }
            }
            None => {
                seen.insert(key, slots.len());
                slots.push(Some(fact));
            }
        }
    }
    let unique: Vec<ExtractedFact> = slots.into_iter().flatten().collect();
    if unique.len() != total {
        debug!("Deduplicated {total} -> {} facts", unique.len());
    }
    unique
}

/// Loads the extraction prompt files (resolved with [`ce_config::resolve_asset`]).
///
/// # Errors
/// [`crate::FactError::Config`] when a file is missing or malformed.
pub fn load_prompts(files: &[String]) -> Result<Vec<ExtractionPrompt>> {
    files.iter().map(|file| ExtractionPrompt::load(file).map_err(FactError::from)).collect()
}

/// One prompt: [`LlmFactExtractor`]; several: [`MultiPassFactExtractor`]; none: an extractor
/// that yields no facts.
#[must_use]
pub fn create_extractor(
    llm: Arc<dyn Llm>,
    prompts: &[ExtractionPrompt],
    parallel: bool,
    max_workers: usize,
) -> Box<dyn FactExtractor> {
    match prompts {
        [] => Box::new(LlmFactExtractor::with_template(llm, String::new())),
        [prompt] => Box::new(LlmFactExtractor::new(llm, prompt)),
        _ => Box::new(MultiPassFactExtractor::new(&llm, prompts, parallel, max_workers)),
    }
}

/// Single-pass extractor: one LLM call per chunk.
pub struct LlmFactExtractor {
    llm: Arc<dyn Llm>,
    template: String,
}

impl fmt::Debug for LlmFactExtractor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlmFactExtractor").field("template_len", &self.template.len()).finish()
    }
}

impl LlmFactExtractor {
    /// Extractor over `prompt` ([`ExtractionPrompt::combined_template`]).
    #[must_use]
    pub fn new(llm: Arc<dyn Llm>, prompt: &ExtractionPrompt) -> Self {
        Self::with_template(llm, prompt.combined_template())
    }

    fn with_template(llm: Arc<dyn Llm>, template: String) -> Self {
        Self { llm, template }
    }

    /// The combined prompt template.
    #[must_use]
    pub fn template(&self) -> &str {
        &self.template
    }

    /// Facts of `chunk_text`, ids prefixed with `pass_name` as in [`parse_facts`].
    ///
    /// Empty text or an empty template yields no facts without calling the LLM; LLM and parse
    /// failures are logged and yield no facts.
    #[must_use]
    pub fn extract_with_pass(
        &self,
        chunk_text: &str,
        entities: &[PromptEntity],
        pass_name: Option<&str>,
    ) -> Vec<ExtractedFact> {
        if chunk_text.is_empty() || self.template.is_empty() {
            return Vec::new();
        }
        let prompt = render_prompt(&self.template, chunk_text, &build_entity_context(entities));
        let response = match self.llm.complete(&LlmRequest::new(&prompt)) {
            Ok(response) => response,
            Err(e) => {
                error!("Failed to extract facts via LLM: {e}");
                return Vec::new();
            }
        };
        match parse_facts(&response, pass_name) {
            Ok(facts) => facts,
            Err(e) => {
                error!("Failed to parse LLM fact extraction response: {e}");
                debug!("Response text: {}...", excerpt(&response, 200));
                Vec::new()
            }
        }
    }
}

impl FactExtractor for LlmFactExtractor {
    fn extract_facts(&self, chunk_text: &str, entities: &[PromptEntity]) -> Vec<ExtractedFact> {
        self.extract_with_pass(chunk_text, entities, None)
    }
}

/// Multi-pass extractor: one focused prompt per pass, merged and deduplicated.
#[derive(Debug)]
pub struct MultiPassFactExtractor {
    passes: Vec<(String, LlmFactExtractor)>,
    parallel: bool,
    max_workers: usize,
}

impl MultiPassFactExtractor {
    /// One pass per prompt, named by its `pass_name` (else `pass_<index>`).
    #[must_use]
    pub fn new(
        llm: &Arc<dyn Llm>,
        prompts: &[ExtractionPrompt],
        parallel: bool,
        max_workers: usize,
    ) -> Self {
        let passes: Vec<(String, LlmFactExtractor)> = prompts
            .iter()
            .enumerate()
            .map(|(index, prompt)| {
                let name = prompt.pass_name.clone().unwrap_or_else(|| format!("pass_{index}"));
                (name, LlmFactExtractor::new(Arc::clone(llm), prompt))
            })
            .collect();
        debug!(
            "MultiPassFactExtractor initialized with {} passes: {:?}",
            passes.len(),
            passes.iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>()
        );
        Self { passes, parallel, max_workers }
    }

    /// Pass names in run order.
    #[must_use]
    pub fn pass_names(&self) -> Vec<&str> {
        self.passes.iter().map(|(name, _)| name.as_str()).collect()
    }

    /// Facts of one pass: ids prefixed with the pass name, `metadata.pass_name` set.
    fn run_pass(
        &self,
        index: usize,
        chunk_text: &str,
        entities: &[PromptEntity],
    ) -> Vec<ExtractedFact> {
        let (pass_name, extractor) = &self.passes[index];
        let mut facts = extractor.extract_with_pass(chunk_text, entities, Some(pass_name));
        for fact in &mut facts {
            fact.metadata.insert("pass_name".to_owned(), Value::String(pass_name.clone()));
        }
        debug!("Pass '{pass_name}' extracted {} facts", facts.len());
        facts
    }

    /// Runs every pass on `workers` scoped threads; results are returned in pass order.
    fn run_parallel(
        &self,
        workers: usize,
        chunk_text: &str,
        entities: &[PromptEntity],
    ) -> Vec<Vec<ExtractedFact>> {
        let next = AtomicUsize::new(0);
        let mut results: Vec<Vec<ExtractedFact>> = vec![Vec::new(); self.passes.len()];
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|_| {
                    scope.spawn(|| {
                        let mut done = Vec::new();
                        loop {
                            let index = next.fetch_add(1, Ordering::Relaxed);
                            if index >= self.passes.len() {
                                break done;
                            }
                            done.push((index, self.run_pass(index, chunk_text, entities)));
                        }
                    })
                })
                .collect();
            for handle in handles {
                let done = handle.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic));
                for (index, facts) in done {
                    results[index] = facts;
                }
            }
        });
        results
    }
}

impl FactExtractor for MultiPassFactExtractor {
    fn extract_facts(&self, chunk_text: &str, entities: &[PromptEntity]) -> Vec<ExtractedFact> {
        if chunk_text.is_empty() {
            return Vec::new();
        }
        let workers = self.max_workers.max(1).min(self.passes.len());
        let per_pass = if self.parallel && workers > 1 {
            self.run_parallel(workers, chunk_text, entities)
        } else {
            (0..self.passes.len()).map(|index| self.run_pass(index, chunk_text, entities)).collect()
        };
        deduplicate_facts(per_pass.into_iter().flatten().collect())
    }
}

/// The `nodes` list of a response, salvaging complete nodes from output that does not parse
/// as a whole.
fn response_nodes(response: &str) -> Result<Vec<Value>> {
    let parsed = match extract_json(response) {
        Ok(Value::Object(mut map)) => match map.remove("nodes") {
            Some(Value::Array(nodes)) => return Ok(nodes),
            Some(other) => {
                return Err(FactError::Parse(format!(
                    "`nodes` is {}, not a list",
                    json_kind(&other)
                )))
            }
            None => Ok(Value::Object(map)),
        },
        other => other,
    };
    if let Some(salvage) = salvage_nodes(response) {
        warn!(
            "Fact extraction output does not parse as a whole (list truncated: {}); salvaged {} complete nodes, {} unparseable",
            salvage.truncated,
            salvage.nodes.len(),
            salvage.unparseable
        );
        return Ok(salvage.nodes);
    }
    match parsed {
        Ok(Value::Object(_)) => Ok(Vec::new()),
        Ok(other) => {
            Err(FactError::Parse(format!("expected a JSON object, got {}", json_kind(&other))))
        }
        Err(e) => Err(FactError::Parse(e.to_string())),
    }
}

/// Complete objects recovered from a `"nodes": [` list.
struct Salvage {
    nodes: Vec<Value>,
    unparseable: usize,
    truncated: bool,
}

/// Recovers every complete top-level object of the first `"nodes": [...]` list in `text`
/// (string-aware brace matching), whether or not the list is closed.
fn salvage_nodes(text: &str) -> Option<Salvage> {
    let list = nodes_list_start(text)?;
    let mut nodes = Vec::new();
    let mut unparseable = 0;
    let mut depth = 0usize;
    let mut object_start = None;
    let mut in_string = false;
    let mut escaped = false;
    let mut truncated = true;
    for (offset, c) in list.char_indices() {
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
            '{' | '[' => {
                if depth == 0 && c == '{' {
                    object_start = Some(offset);
                }
                depth += 1;
            }
            '}' | ']' => {
                if depth == 0 {
                    // `]` closes the list; anything else is a stray closer.
                    truncated = false;
                    break;
                }
                depth -= 1;
                if depth == 0 {
                    if let Some(start) = object_start.take() {
                        match parse_object(&list[start..=offset]) {
                            Some(node) => nodes.push(node),
                            None => unparseable += 1,
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Some(Salvage { nodes, unparseable, truncated })
}

/// The text right after the `[` of the first `"nodes": [` in `text`.
fn nodes_list_start(text: &str) -> Option<&str> {
    const KEY: &str = "\"nodes\"";
    text.match_indices(KEY).find_map(|(at, _)| {
        if text[..at].ends_with('\\') {
            return None;
        }
        let rest = text[at + KEY.len()..].trim_start();
        let rest = rest.strip_prefix(':')?.trim_start();
        rest.strip_prefix('[')
    })
}

/// Parses one salvaged object, strictly and then after [`repair_json`].
fn parse_object(candidate: &str) -> Option<Value> {
    serde_json::from_str::<Value>(candidate)
        .ok()
        .or_else(|| serde_json::from_str(&repair_json(candidate)).ok())
        .filter(Value::is_object)
}

/// Converts the `STATED_FACT` nodes of a response into facts.
fn facts_from_nodes(nodes: &[Value], pass_name: Option<&str>) -> Vec<ExtractedFact> {
    let pass_name = pass_name.filter(|name| !name.is_empty());
    let mut facts = Vec::new();
    let mut used_ids: HashSet<String> = HashSet::new();
    let mut stated_index = 0usize;
    for node in nodes {
        let Value::Object(node) = node else {
            warn!("Skipping fact node that is not an object: {}", json_kind(node));
            continue;
        };
        if node.get("node_type").and_then(Value::as_str) != Some("STATED_FACT") {
            continue;
        }
        let index = stated_index;
        stated_index += 1;
        let Some(mut fact) = fact_from_node(node) else {
            continue;
        };
        let id = unique_id(node.get("id"), index, &mut used_ids);
        fact.id = match pass_name {
            Some(pass) if !id.starts_with(pass) => format!("{pass}_{id}").into(),
            _ => id.into(),
        };
        facts.push(fact);
    }
    facts
}

/// The node's own id when usable and unused, else `STATED_FACT_<index>` (suffixed `_<k>` while
/// taken). Records the chosen id in `used`.
fn unique_id(raw: Option<&Value>, index: usize, used: &mut HashSet<String>) -> String {
    let own = match raw {
        Some(Value::String(id)) if !id.is_empty() => Some(id.clone()),
        Some(Value::Number(id)) => Some(id.to_string()),
        _ => None,
    };
    let id = match own {
        Some(id) if !used.contains(&id) => id,
        own => {
            if let Some(own) = own {
                warn!("Duplicate fact id {own:?} in one response; renamed");
            }
            let base = format!("STATED_FACT_{index}");
            let mut candidate = base.clone();
            let mut k = 1usize;
            while used.contains(&candidate) {
                candidate = format!("{base}_{k}");
                k += 1;
            }
            candidate
        }
    };
    used.insert(id.clone());
    id
}

/// Builds the fact of one `STATED_FACT` node (id filled in by the caller); `None` when a
/// string-encoded nested field is not valid JSON.
fn fact_from_node(node: &Map<String, Value>) -> Option<ExtractedFact> {
    let nested = |key: &str| -> std::result::Result<Map<String, Value>, serde_json::Error> {
        match node.get(key) {
            Some(Value::String(raw)) => {
                serde_json::from_str::<Value>(raw).map(|value| match value {
                    Value::Object(map) => map,
                    _ => Map::new(),
                })
            }
            Some(Value::Object(map)) => Ok(map.clone()),
            _ => Ok(Map::new()),
        }
    };
    let decoded = nested("assertion_triple").and_then(|triple| {
        let temporal = nested("temporal_validity")?;
        Ok((triple, temporal, nested("metadata")?))
    });
    let (triple, temporal, mut metadata) = match decoded {
        Ok(decoded) => decoded,
        Err(e) => {
            warn!("Failed to parse nested JSON in fact node {:?}: {e}", node.get("id"));
            return None;
        }
    };

    let time_window = match metadata.remove("time_window") {
        Some(Value::Object(window)) => Some(TimeWindow {
            start: string_field(&window, "start"),
            end: string_field(&window, "end"),
        }),
        Some(Value::String(raw)) => {
            metadata.insert("time_window_raw".to_owned(), Value::String(raw));
            Some(TimeWindow::default())
        }
        _ => None,
    };

    let relationship = scalar_text(triple.get("relationship"));
    let assertion_triple = AssertionTriple::new(
        scalar_text(triple.get("subject")),
        if relationship.is_empty() { "RELATED_TO".to_owned() } else { relationship },
        scalar_text(triple.get("object")),
    );
    let temporal_validity = TemporalValidity {
        t_valid_from: string_field(&temporal, "t_valid_from"),
        t_valid_until: string_field(&temporal, "t_valid_until"),
    };
    let mut fact = ExtractedFact::new(
        String::new(),
        scalar_text(node.get("text")),
        assertion_triple,
        parse_certainty(node.get("certainty")),
        temporal_validity,
    );
    fact.metadata = metadata;
    fact.value_type = string_field(node, "value_type");
    fact.property_type = string_field(node, "property_type");
    fact.slot_name = string_field(node, "slot_name");
    fact.value_raw = match node.get("value_raw") {
        Some(Value::String(raw)) => Some(raw.clone()),
        Some(Value::Number(raw)) => Some(raw.to_string()),
        _ => None,
    };
    fact.temporal_context = parse_temporal_context(node.get("temporal_context"));
    fact.time_window = time_window;
    fact.is_denial = node.get("is_denial").is_some_and(parse_bool);
    fact.is_boilerplate = node.get("is_boilerplate").is_some_and(parse_bool);
    fact.raw_node = node.clone();
    Some(fact)
}

/// A string field, `None` when missing or not a string.
fn string_field(map: &Map<String, Value>, key: &str) -> Option<String> {
    map.get(key).and_then(Value::as_str).map(str::to_owned)
}

/// A string, or a stringified number; anything else (and missing) is `""`.
fn scalar_text(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Number(number)) => number.to_string(),
        _ => String::new(),
    }
}

/// Python `float(str(value))` with a [`DEFAULT_CERTAINTY`] fallback, clamped to `[0, 1]`.
fn parse_certainty(value: Option<&Value>) -> f64 {
    let parsed = match value {
        None => Some(DEFAULT_CERTAINTY),
        Some(Value::Number(number)) => number.as_f64(),
        Some(Value::String(text)) => parse_py_float(text),
        Some(_) => None,
    };
    match parsed {
        Some(certainty) if certainty.is_finite() => certainty.clamp(0.0, 1.0),
        _ => DEFAULT_CERTAINTY,
    }
}

/// Python `float(text)`: surrounding whitespace allowed, single `_` between digits allowed.
fn parse_py_float(text: &str) -> Option<f64> {
    let text = py_strip(text);
    if !text.contains('_') {
        return text.parse().ok();
    }
    let chars: Vec<char> = text.chars().collect();
    let separators_ok = chars.iter().enumerate().all(|(i, &c)| {
        c != '_'
            || (i > 0
                && chars[i - 1].is_ascii_digit()
                && chars.get(i + 1).is_some_and(char::is_ascii_digit))
    });
    if separators_ok {
        text.replace('_', "").parse().ok()
    } else {
        None
    }
}

/// Python `_parse_bool`: booleans as is, strings `true` / `1` / `yes` (any case), otherwise
/// truthiness.
fn parse_bool(value: &Value) -> bool {
    match value {
        Value::Bool(flag) => *flag,
        Value::String(text) => matches!(text.to_lowercase().as_str(), "true" | "1" | "yes"),
        Value::Null => false,
        Value::Number(number) => number.as_f64().is_some_and(|n| n != 0.0),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// One of the three temporal contexts, `current_incident` otherwise.
fn parse_temporal_context(value: Option<&Value>) -> TemporalContext {
    match value.and_then(Value::as_str) {
        Some("prior_incident") => TemporalContext::PriorIncident,
        Some("general_claim") => TemporalContext::GeneralClaim,
        _ => TemporalContext::CurrentIncident,
    }
}

fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "a list",
        Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;

    fn fact(id: &str, subject: &str, text: &str, certainty: f64) -> ExtractedFact {
        ExtractedFact::new(
            id,
            text,
            AssertionTriple::new(subject, "HAS", "x"),
            certainty,
            TemporalValidity::default(),
        )
    }

    fn entity(id: &str, name: &str, entity_type: Option<&str>, aliases: &[&str]) -> PromptEntity {
        PromptEntity {
            id: EntityId::new(id),
            name: name.to_owned(),
            entity_type: entity_type.map(str::to_owned),
            aliases: aliases.iter().map(|a| (*a).to_owned()).collect(),
        }
    }

    #[test]
    fn entity_context_formats_like_python() {
        assert_eq!(build_entity_context(&[]), "");
        let context = build_entity_context(&[
            entity("E1", "Maria Lopez", Some("PER"), &["Ms. Lopez", "", "Plaintiff"]),
            entity("E2", "Green Valley", None, &[]),
            entity("E3", "Truck", Some(""), &[""]),
        ]);
        assert_eq!(
            context,
            "Resolved entities in this chunk:\n\
             - Maria Lopez (ID: E1, TYPE: PER, Aliases: \"Ms. Lopez\", \"Plaintiff\")\n\
             - Green Valley (ID: E2, TYPE: ENTITY)\n\
             - Truck (ID: E3, TYPE: ENTITY, Aliases: )"
        );
    }

    #[test]
    fn render_substitutes_segments_then_entities() {
        let template = "A {resolved_entities} B {segments} {\"k\": 1}";
        assert_eq!(render_prompt(template, "TEXT", "ENT"), "A ENT B TEXT {\"k\": 1}");
        // Segments are substituted first, so a placeholder inside the text is filled too.
        assert_eq!(render_prompt("{segments}", "x {resolved_entities}", "E"), "x E");
        assert_eq!(render_prompt("no placeholders", "T", "E"), "no placeholders");
    }

    #[test]
    fn certainty_follows_python_float_of_str() {
        let cases = [
            (Some(json!(0.9)), 0.9),
            (Some(json!(1)), 1.0),
            (Some(json!("0.95")), 0.95),
            (Some(json!(" 0.7\n")), 0.7),
            (Some(json!("1_0e-1")), 1.0),
            (Some(json!("0._5")), DEFAULT_CERTAINTY),
            (Some(json!("high")), DEFAULT_CERTAINTY),
            (Some(json!("nan")), DEFAULT_CERTAINTY),
            (Some(json!("inf")), DEFAULT_CERTAINTY),
            (Some(json!(true)), DEFAULT_CERTAINTY),
            (Some(json!(null)), DEFAULT_CERTAINTY),
            (Some(json!([0.9])), DEFAULT_CERTAINTY),
            (Some(json!(95)), 1.0),
            (Some(json!("-0.2")), 0.0),
            (None, DEFAULT_CERTAINTY),
        ];
        for (value, expected) in cases {
            assert_eq!(parse_certainty(value.as_ref()), expected, "{value:?}");
        }
    }

    #[test]
    fn bool_parsing_matches_python() {
        let cases = [
            (json!(true), true),
            (json!(false), false),
            (json!("True"), true),
            (json!("YES"), true),
            (json!("1"), true),
            (json!("false"), false),
            (json!("no"), false),
            (json!(" true"), false),
            (json!(""), false),
            (json!(null), false),
            (json!(0), false),
            (json!(0.0), false),
            (json!(2), true),
            (json!(-1.5), true),
            (json!([]), false),
            (json!([0]), true),
            (json!({}), false),
            (json!({"a": 1}), true),
        ];
        for (value, expected) in cases {
            assert_eq!(parse_bool(&value), expected, "{value}");
        }
    }

    #[test]
    fn temporal_context_falls_back_to_current() {
        assert_eq!(
            parse_temporal_context(Some(&json!("prior_incident"))),
            TemporalContext::PriorIncident
        );
        assert_eq!(
            parse_temporal_context(Some(&json!("general_claim"))),
            TemporalContext::GeneralClaim
        );
        for value in [json!("current_incident"), json!("PRIOR_INCIDENT"), json!(3), json!(null)] {
            assert_eq!(
                parse_temporal_context(Some(&value)),
                TemporalContext::CurrentIncident,
                "{value}"
            );
        }
        assert_eq!(parse_temporal_context(None), TemporalContext::CurrentIncident);
    }

    #[test]
    fn unique_ids_fill_missing_and_duplicates() {
        let mut used = HashSet::new();
        assert_eq!(unique_id(Some(&json!("A")), 0, &mut used), "A");
        assert_eq!(unique_id(None, 1, &mut used), "STATED_FACT_1");
        assert_eq!(unique_id(Some(&json!("")), 2, &mut used), "STATED_FACT_2");
        assert_eq!(unique_id(Some(&json!(7)), 3, &mut used), "7");
        assert_eq!(unique_id(Some(&json!("A")), 4, &mut used), "STATED_FACT_4");
        assert_eq!(unique_id(Some(&json!("STATED_FACT_5")), 5, &mut used), "STATED_FACT_5");
        // The explicit id took STATED_FACT_5; the fallback for a later duplicate gets a suffix.
        used.insert("STATED_FACT_6".to_owned());
        assert_eq!(unique_id(Some(&json!("A")), 6, &mut used), "STATED_FACT_6_1");
        assert_eq!(unique_id(Some(&json!(true)), 6, &mut used), "STATED_FACT_6_2");
    }

    #[test]
    fn fact_key_normalises_like_python() {
        let mut f = fact("1", "  Maria Lopez ", "  The Truck WAS red\t", 0.5);
        f.assertion_triple.relationship = " had_color ".to_owned();
        f.assertion_triple.object = "RED\u{1f}".to_owned();
        assert_eq!(fact_key(&f), "maria lopez|HAD_COLOR|red|the truck was red");
        // The text part is the first 50 characters (not bytes) after stripping.
        let long = format!(" {}", "é".repeat(60));
        let f = fact("2", "s", &long, 0.5);
        assert_eq!(fact_key(&f), format!("s|HAS|x|{}", "é".repeat(50)));
    }

    #[test]
    fn dedup_keeps_python_order() {
        let facts = vec![
            fact("a1", "A", "t", 0.5),
            fact("b", "B", "t", 0.9),
            fact("a2", "A", "t", 0.8),
            fact("c", "C", "t", 0.4),
            fact("a3", "A", "t", 0.8),
            fact("b2", "b ", "T", 0.9),
        ];
        let ids: Vec<String> =
            deduplicate_facts(facts).into_iter().map(|f| f.id.into_inner()).collect();
        // a2 (higher certainty) removes a1 and is appended; a3 (equal) and b2 (equal) drop.
        assert_eq!(ids, ["b", "a2", "c"]);
        assert!(deduplicate_facts(Vec::new()).is_empty());
    }

    #[test]
    fn salvage_finds_complete_objects_only() {
        let text = r#"{"nodes": [{"a": "}"}, {"b": [1, {"c": 2}]}, {"d": "unterminated"#;
        let salvage = salvage_nodes(text).expect("nodes list");
        assert!(salvage.truncated);
        assert_eq!(salvage.nodes, vec![json!({"a": "}"}), json!({"b": [1, {"c": 2}]})]);
        let closed = salvage_nodes(r#"x "nodes" : [ {"a": 1,}, {bad}, 3 ] }"#).expect("list");
        assert!(!closed.truncated);
        assert_eq!(closed.nodes, vec![json!({"a": 1})]);
        assert_eq!(closed.unparseable, 1);
        assert!(salvage_nodes(r#"{"text": "say \"nodes\": [ here"}"#).is_none());
        assert!(salvage_nodes(r#"{"nodes": {}}"#).is_none());
    }

    #[test]
    fn py_float_underscores() {
        assert_eq!(parse_py_float("1_000"), Some(1000.0));
        assert_eq!(parse_py_float("_1"), None);
        assert_eq!(parse_py_float("1__0"), None);
        assert_eq!(parse_py_float("1_"), None);
    }
}
