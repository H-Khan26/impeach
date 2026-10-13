//! The ED service (Python `services/ed/service.py`): routes a mention context to the rule-first,
//! hybrid or LLM-first strategy, retrieves candidates (exact, similarity, embedding) and breaks
//! ties with an [`impeach_models::Llm`] using `assets/prompts/ed/disambiguation.json`.
//!
//! Routing (from `entity_types.yaml`): `llm_only` with an LLM -> LLM-first; rule-first labels
//! try their rule resolver and the generic exact rule and fall through on failure; hybrid
//! labels use retrieval heuristics and then the LLM; every other label goes LLM-first.
//!
//! LLM failures (model errors, unparsable answers) are logged and leave the mention
//! unresolved; a missing prompt or a template with an unknown placeholder is an
//! [`EdError::Prompt`] (Python raised there too).

use std::collections::HashSet;
use std::sync::Arc;

use impeach_config::mappings::EntityTypes;
use impeach_config::prompts::EdPrompt;
use impeach_core::{EdCandidate, EdContext, EdResult};
use impeach_models::{Llm, LlmRequest};
use impeach_text::normalize::py_strip;
use serde_json::{json, Map, Value};

use crate::error::{EdError, Result};
use crate::name_matching::{entity_types_compatible, normalize_for_matching};
use crate::name_similarity::NameSimilarity;
use crate::retrieval::EdRetrieval;
use crate::rule_resolvers::{
    resolve_case_rule, resolve_exhibit_rule, resolve_generic_rule, resolve_statute_rule,
};
use crate::structured_keys::KeyPatterns;

/// Entity-type guidance inserted into the system prompt (`{type_specific_guidance}`), keyed by
/// the normalised span label (`entity_types.yaml` types, so persons are `PER`); other labels
/// get an empty string.
pub const TYPE_SPECIFIC_GUIDANCE: [(&str, &str); 4] = [
    (
        "PER",
        "PERSON Entity Guidelines:\n\
         - Consider roles (plaintiff, defendant, witness, attorney, judge).\n\
         - Names may appear with titles (Dr., Atty., Hon.) or without.\n\
         - The same person may be referred to by full name, last name only, or title + last name.\n\
         - Attorneys are often identified by their role (defense counsel, plaintiff's attorney).\n",
    ),
    (
        "ORG",
        "ORGANIZATION Entity Guidelines:\n\
         - Company names may include or omit suffixes (Inc., LLC, Corp., LLP).\n\
         - Government entities may use acronyms or full names (FBI, Federal Bureau of Investigation).\n\
         - Consider parent/subsidiary relationships and law firm naming patterns.\n",
    ),
    (
        "CASE",
        "CASE Entity Guidelines:\n\
         - Cases may be referenced by caption (Smith v. Jones) or docket number (2:23-cv-01492).\n\
         - Short-form references (\"the Smith case\", \"this matter\") rely on context.\n\
         - Related cases are distinct; match the specific case indicated by citation or docket.\n",
    ),
    (
        "EXHIBIT",
        "EXHIBIT Entity Guidelines:\n\
         - Exhibits are typically scoped to a specific proceeding or case context.\n\
         - The same exhibit ID string may exist in different settings; rely on the surrounding textual context and party attribution.\n\
         - Consider party attribution (Plaintiff's Exhibit 1 vs Defense Exhibit 1).\n",
    ),
];

/// Guidance for the normalised label `entity_type` from [`TYPE_SPECIFIC_GUIDANCE`], or `""`.
#[must_use]
pub fn type_specific_guidance(entity_type: &str) -> &'static str {
    TYPE_SPECIFIC_GUIDANCE
        .iter()
        .find(|(label, _)| *label == entity_type)
        .map_or("", |(_, guidance)| guidance)
}

/// Tuning of [`EdService`] (the service fields of `EDStageConfig`).
#[derive(Debug, Clone, PartialEq)]
pub struct EdServiceSettings {
    /// Resolve every mention LLM-first (when the LLM is allowed and present).
    pub llm_only: bool,
    /// Minimum cosine score of embedding candidates.
    pub min_embedding_similarity: f64,
    /// Similarity a hybrid candidate must exceed to be chosen without the LLM.
    pub hybrid_similarity_threshold: f64,
    /// Add similarity candidates between exact and embedding candidates.
    pub enable_similarity_retrieval: bool,
    /// Boost for candidates created from another document.
    pub cross_doc_reuse_boost: f64,
}

impl Default for EdServiceSettings {
    fn default() -> Self {
        Self {
            llm_only: false,
            min_embedding_similarity: 0.55,
            hybrid_similarity_threshold: 0.72,
            enable_similarity_retrieval: true,
            cross_doc_reuse_boost: 0.05,
        }
    }
}

impl EdServiceSettings {
    /// Settings of an `EDStage` config block.
    #[must_use]
    pub fn from_config(config: &impeach_config::pipeline::EdStageConfig) -> Self {
        Self {
            llm_only: config.llm_only,
            min_embedding_similarity: config.min_embedding_similarity,
            hybrid_similarity_threshold: config.hybrid_similarity_threshold,
            enable_similarity_retrieval: config.enable_similarity_retrieval,
            cross_doc_reuse_boost: config.cross_doc_reuse_boost,
        }
    }
}

/// Entity disambiguation of one mention context (Python `EDService`).
pub struct EdService<'a> {
    retriever: &'a dyn EdRetrieval,
    llm: Option<Arc<dyn Llm>>,
    settings: EdServiceSettings,
    rule_first_labels: HashSet<String>,
    hybrid_labels: HashSet<String>,
    entity_types: &'a EntityTypes,
    similarity: &'a NameSimilarity,
    patterns: &'a KeyPatterns,
    prompt: Option<&'a EdPrompt>,
}

impl std::fmt::Debug for EdService<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EdService")
            .field("retriever", &self.retriever)
            .field("has_llm", &self.llm.is_some())
            .field("settings", &self.settings)
            .field("rule_first_labels", &self.rule_first_labels)
            .field("hybrid_labels", &self.hybrid_labels)
            .finish_non_exhaustive()
    }
}

impl<'a> EdService<'a> {
    /// A service over `retriever` with the routing labels of `similarity`'s entity types.
    #[must_use]
    pub fn new(
        retriever: &'a dyn EdRetrieval,
        settings: EdServiceSettings,
        similarity: &'a NameSimilarity,
        patterns: &'a KeyPatterns,
    ) -> Self {
        let entity_types = similarity.entity_types();
        let owned = |set: HashSet<&str>| set.into_iter().map(str::to_owned).collect();
        Self {
            retriever,
            llm: None,
            settings,
            rule_first_labels: owned(entity_types.rule_first_label_set()),
            hybrid_labels: owned(entity_types.hybrid_label_set()),
            entity_types,
            similarity,
            patterns,
            prompt: None,
        }
    }

    /// Uses `llm` for the tie-break.
    #[must_use]
    pub fn with_llm(mut self, llm: Option<Arc<dyn Llm>>) -> Self {
        self.llm = llm;
        self
    }

    /// Uses `prompt` to build LLM prompts.
    #[must_use]
    pub fn with_prompt(mut self, prompt: Option<&'a EdPrompt>) -> Self {
        self.prompt = prompt;
        self
    }

    /// Replaces the routing labels.
    #[must_use]
    pub fn with_labels(
        mut self,
        rule_first_labels: HashSet<String>,
        hybrid_labels: HashSet<String>,
    ) -> Self {
        self.rule_first_labels = rule_first_labels;
        self.hybrid_labels = hybrid_labels;
        self
    }

    /// The settings.
    #[must_use]
    pub fn settings(&self) -> &EdServiceSettings {
        &self.settings
    }

    /// Resolves one mention context to an entity id (or none) with the candidates considered.
    ///
    /// # Errors
    /// Fatal store errors; [`EdError::Prompt`] when the LLM is called without a usable prompt.
    pub fn resolve_span(
        &self,
        ctx: &EdContext,
        allow_llm: bool,
        max_candidates: usize,
    ) -> Result<EdResult> {
        let label = label_of(ctx);
        if self.settings.llm_only && allow_llm && self.llm.is_some() {
            return self.resolve_llm_first(ctx, true, max_candidates);
        }
        if self.rule_first_labels.contains(label) {
            if let Some(result) = self.resolve_rule_first(ctx)? {
                if result.chosen_entity_id.as_ref().is_some_and(|id| !id.as_str().is_empty()) {
                    return Ok(result);
                }
            }
        }
        if self.hybrid_labels.contains(label) {
            return self.resolve_hybrid(ctx, allow_llm, max_candidates);
        }
        self.resolve_llm_first(ctx, allow_llm, max_candidates)
    }

    /// The configured rule resolver of the label, then the generic exact-match rule.
    fn resolve_rule_first(&self, ctx: &EdContext) -> Result<Option<EdResult>> {
        let routed = match self.entity_types.get_rule_route(label_of(ctx)) {
            Some("case") => resolve_case_rule(ctx, self.retriever, self.patterns)?,
            Some("statute") => resolve_statute_rule(ctx, self.retriever, self.patterns)?,
            Some("exhibit") => resolve_exhibit_rule(ctx, self.retriever, self.patterns)?,
            _ => None,
        };
        if routed.is_some() {
            return Ok(routed);
        }
        resolve_generic_rule(ctx, self.retriever, self.patterns)
    }

    /// Whether the normalised surface equals the candidate's normalised name or an alias
    /// (BM25 "exact" hits can be fuzzy).
    fn is_true_exact_match(surface: &str, candidate: &EdCandidate) -> bool {
        let surface_norm = normalize_for_matching(surface);
        normalize_for_matching(&candidate.canonical_name) == surface_norm
            || candidate.aliases.iter().any(|alias| normalize_for_matching(alias) == surface_norm)
    }

    /// Hybrid route: a unique verified exact match, else the best boosted similarity above
    /// the threshold, else the LLM, else unresolved.
    fn resolve_hybrid(
        &self,
        ctx: &EdContext,
        allow_llm: bool,
        max_candidates: usize,
    ) -> Result<EdResult> {
        let candidates = self.retrieve_candidates(ctx, max_candidates)?;

        let exact: Vec<&EdCandidate> = candidates
            .iter()
            .filter(|c| c.source == "exact" && Self::is_true_exact_match(&ctx.mention.name, c))
            .collect();
        if let [only] = exact.as_slice() {
            let chosen = only.entity_id.clone();
            return Ok(result(Some(chosen), candidates, false, 0.85, "hybrid_heuristic"));
        }

        let surface = ctx.mention.name.as_str();
        let entity_type = label_of(ctx);
        let normalized_surface = normalize_for_matching(surface);
        let mut best_score = 0.0_f64;
        let mut best: Option<usize> = None;
        for (index, candidate) in candidates.iter().enumerate() {
            let mut score = self.string_similarity(surface, &candidate.canonical_name, entity_type);
            for alias in &candidate.aliases {
                let alias_score = self.string_similarity(surface, alias, entity_type);
                if alias_score > score {
                    score = alias_score;
                }
            }
            let boost = self.metadata_boost(ctx, candidate, &normalized_surface);
            let effective = (score + boost).min(1.0);
            if effective > best_score {
                best_score = effective;
                best = Some(index);
            }
        }
        if let Some(index) = best {
            if best_score > self.settings.hybrid_similarity_threshold {
                let candidate = &candidates[index];
                let strategy = if candidate.source == "embedding" {
                    "vector_fuzzy"
                } else {
                    "hybrid_similarity"
                };
                let chosen = candidate.entity_id.clone();
                return Ok(result(Some(chosen), candidates, false, best_score * 0.9, strategy));
            }
        }

        if allow_llm && self.llm.is_some() {
            if let Some(llm_result) = self.resolve_with_llm(ctx, &candidates, "hybrid_llm")? {
                return Ok(llm_result);
            }
        }
        Ok(result(None, candidates, false, 0.0, "hybrid_unresolved"))
    }

    /// Boosts from candidate metadata: `normalized_name` equal to the surface (+0.05), or
    /// without one an equal normalised canonical name (+0.02); `cross_doc_reuse_boost` for an
    /// entity from another document.
    fn metadata_boost(
        &self,
        ctx: &EdContext,
        candidate: &EdCandidate,
        normalized_surface: &str,
    ) -> f64 {
        let meta = &candidate.metadata;
        let mut boost = 0.0;
        if meta.get("normalized_name").and_then(Value::as_str) == Some(normalized_surface) {
            boost += 0.05;
        }
        if !meta.get("normalized_name").is_some_and(is_truthy)
            && normalize_for_matching(&candidate.canonical_name) == normalized_surface
        {
            boost += 0.02;
        }
        let current_doc_id = ctx.doc_context.doc_id.as_str();
        if let Some(source_doc) = meta.get("source_doc_id").filter(|v| is_truthy(v)) {
            let same_doc = source_doc.as_str() == Some(current_doc_id);
            if !current_doc_id.is_empty() && !same_doc {
                boost += self.settings.cross_doc_reuse_boost;
                tracing::debug!(
                    candidate = %candidate.entity_id,
                    source_doc = %source_doc,
                    current_doc = current_doc_id,
                    boost,
                    "cross-doc boost"
                );
            }
        }
        boost
    }

    /// LLM-first route: candidates, then the LLM when allowed; `llm_skipped` otherwise.
    fn resolve_llm_first(
        &self,
        ctx: &EdContext,
        allow_llm: bool,
        max_candidates: usize,
    ) -> Result<EdResult> {
        let candidates = self.retrieve_candidates(ctx, max_candidates)?;
        if allow_llm && self.llm.is_some() {
            if let Some(llm_result) = self.resolve_with_llm(ctx, &candidates, "llm_first")? {
                return Ok(llm_result);
            }
        }
        Ok(result(None, candidates, false, 0.0, "llm_skipped"))
    }

    /// Asks the LLM to choose among `candidates`. `None` without candidates, without an LLM,
    /// on a model error or an unparsable answer.
    ///
    /// # Errors
    /// [`EdError::Prompt`] when no prompt is set or a template is malformed.
    pub fn resolve_with_llm(
        &self,
        ctx: &EdContext,
        candidates: &[EdCandidate],
        strategy: &str,
    ) -> Result<Option<EdResult>> {
        let Some(llm) = &self.llm else {
            return Ok(None);
        };
        if candidates.is_empty() {
            return Ok(None);
        }
        let payload_json = build_llm_payload_json(ctx, candidates)?;
        let prompt = self.build_llm_prompt(label_of(ctx), &payload_json)?;
        let raw = match llm.complete(&LlmRequest::new(&prompt)) {
            Ok(raw) => raw,
            Err(error) => {
                tracing::error!(mention = %ctx.mention.id, error = %error, "ED LLM call failed");
                return Ok(None);
            }
        };
        let Some(data) = parse_llm_response(&raw) else {
            return Ok(None);
        };

        let chosen = match data.get("chosen_entity_id") {
            Some(Value::String(id))
                if !id.is_empty() && !candidates.iter().any(|c| c.entity_id.as_str() == id) =>
            {
                tracing::warn!(
                    mention = %ctx.mention.id,
                    chosen = id.as_str(),
                    "ED LLM chose an id that is not a candidate; ignored"
                );
                None
            }
            Some(Value::String(id)) if !id.is_empty() => Some(id.clone()),
            Some(Value::Null | Value::String(_)) | None => None,
            Some(other) => {
                tracing::warn!(
                    mention = %ctx.mention.id,
                    value = %other,
                    "ED LLM chose a non-string entity id; ignored"
                );
                None
            }
        };
        if let Some(id) = &chosen {
            let rationale = data.get("rationale").map(ToString::to_string).unwrap_or_default();
            tracing::debug!(
                chosen = id.as_str(),
                mention = %ctx.mention.id,
                name = ctx.mention.name.as_str(),
                rationale = rationale.as_str(),
                "ED LLM selected a candidate"
            );
        }
        let confidence = data.get("confidence").map_or(0.0, python_float);
        let resolved_by_llm = chosen.is_some();
        Ok(Some(result(
            chosen.map(Into::into),
            candidates.to_vec(),
            resolved_by_llm,
            confidence,
            strategy,
        )))
    }

    /// System and user templates rendered with the type guidance and the payload, joined by a
    /// blank line.
    ///
    /// # Errors
    /// [`EdError::Prompt`] when no prompt is set or a template is malformed.
    pub fn build_llm_prompt(&self, entity_type: &str, payload_json: &str) -> Result<String> {
        let prompt = self
            .prompt
            .ok_or_else(|| EdError::Prompt("EDService: prompt_config is not set".to_owned()))?;
        let system = py_format(
            &prompt.system_template,
            &[("type_specific_guidance", type_specific_guidance(entity_type))],
        )?;
        let user = py_format(&prompt.user_template, &[("payload_json", payload_json)])?;
        Ok(format!("{system}\n\n{user}"))
    }

    /// Candidates in the order exact (score 1, source `exact`), similarity, embedding (types
    /// incompatible with the mention dropped), each entity once, up to `limit` per step.
    /// Retrieval is global; `case_id` does not scope it.
    ///
    /// # Errors
    /// Fatal store errors.
    pub fn retrieve_candidates(&self, ctx: &EdContext, limit: usize) -> Result<Vec<EdCandidate>> {
        let mut candidates: Vec<EdCandidate> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let mention_type = label_of(ctx);
        let surface = ctx.mention.name.as_str();

        for mut candidate in self.retriever.find_by_exact_match(surface, Some(mention_type))? {
            candidate.score = 1.0;
            "exact".clone_into(&mut candidate.source);
            if seen.insert(candidate.entity_id.as_str().to_owned()) {
                candidates.push(candidate);
            }
        }

        let remaining = limit.saturating_sub(candidates.len());
        if remaining > 0 && self.settings.enable_similarity_retrieval {
            let min_similarity = (self.settings.hybrid_similarity_threshold - 0.1).max(0.65);
            for candidate in self.retriever.find_by_similarity(
                surface,
                Some(mention_type),
                remaining,
                min_similarity,
            )? {
                if seen.insert(candidate.entity_id.as_str().to_owned()) {
                    tracing::debug!(
                        surface,
                        candidate = candidate.canonical_name.as_str(),
                        score = candidate.score,
                        "similarity retrieval"
                    );
                    candidates.push(candidate);
                }
            }
        }

        let remaining = limit.saturating_sub(candidates.len());
        if remaining > 0 {
            let query = format!("{surface} {}", ctx.local_context.sentence_text);
            let query = py_strip(&query);
            for mut candidate in self.retriever.find_by_embedding(
                query,
                remaining,
                self.settings.min_embedding_similarity,
            )? {
                if !entity_types_compatible(
                    self.entity_types,
                    Some(mention_type),
                    Some(&candidate.label),
                ) {
                    continue;
                }
                if candidate.source.is_empty() {
                    "embedding".clone_into(&mut candidate.source);
                }
                if seen.insert(candidate.entity_id.as_str().to_owned()) {
                    candidates.push(candidate);
                }
            }
        }

        let count = |source: &str| candidates.iter().filter(|c| c.source == source).count();
        tracing::debug!(
            surface,
            total = candidates.len(),
            exact = count("exact"),
            similarity = count("similarity"),
            embedding = count("embedding"),
            "retrieved ED candidates"
        );
        Ok(candidates)
    }

    /// Type-aware similarity when a type is known, `difflib` ratio otherwise.
    fn string_similarity(&self, s1: &str, s2: &str, entity_type: &str) -> f64 {
        if entity_type.is_empty() {
            impeach_text::sequence_matcher_ratio(s1, s2)
        } else {
            self.similarity.enhanced_string_similarity(s1, s2, entity_type)
        }
    }
}

/// The mention's span label, or its type when the label is empty.
fn label_of(ctx: &EdContext) -> &str {
    if ctx.span_label.is_empty() {
        &ctx.mention.entity_type
    } else {
        &ctx.span_label
    }
}

fn result(
    chosen: Option<impeach_core::EntityId>,
    candidates: Vec<EdCandidate>,
    resolved_by_llm: bool,
    confidence: f64,
    strategy: &str,
) -> EdResult {
    EdResult {
        mention_id: None,
        chosen_entity_id: chosen,
        candidates,
        resolved_by_llm,
        confidence,
        strategy_used: strategy.to_owned(),
    }
}

/// Python truthiness of a JSON value.
pub(crate) fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Python `float(value or 0.0)` for a JSON value; values `float` rejects give 0.
fn python_float(value: &Value) -> f64 {
    match value {
        Value::Number(n) => n.as_f64().unwrap_or(0.0),
        Value::Bool(true) => 1.0,
        Value::String(s) if !s.is_empty() => py_strip(s).parse().unwrap_or(0.0),
        _ => 0.0,
    }
}

/// The LLM payload (mention, local and document context, candidates) as Python
/// `json.dumps(payload, ensure_ascii=False)`.
///
/// # Errors
/// [`EdError::Prompt`] when serialisation fails.
pub fn build_llm_payload_json(ctx: &EdContext, candidates: &[EdCandidate]) -> Result<String> {
    let candidates: Vec<Value> = candidates
        .iter()
        .map(|c| {
            json!({
                "entity_id": c.entity_id.as_str(),
                "canonical_name": c.canonical_name,
                "label": c.label,
                "aliases": c.aliases,
                "description": c.description,
            })
        })
        .collect();
    let payload = json!({
        "mention": {
            "id": ctx.mention.id.as_str(),
            "text": ctx.mention.name,
            "label": label_of(ctx),
        },
        "local_context": {
            "sentence": ctx.local_context.sentence_text,
            "prev_sentence": ctx.local_context.prev_sentence_text,
            "next_sentence": ctx.local_context.next_sentence_text,
        },
        "doc_context": {
            "doc_id": ctx.doc_context.doc_id.as_str(),
            "doc_type": ctx.doc_context.doc_type,
            "case_id": ctx.doc_context.case_id,
            "jurisdiction": ctx.doc_context.jurisdiction,
        },
        "candidates": candidates,
    });
    impeach_core::pyjson::to_python_json(&payload)
        .map_err(|e| EdError::Prompt(format!("ED payload serialisation: {e}")))
}

/// Parses an LLM answer: trims, removes a leading ```` ```json ```` / ```` ``` ```` fence and a
/// trailing ```` ``` ````, and requires a JSON object.
#[must_use]
pub fn parse_llm_response(text: &str) -> Option<Map<String, Value>> {
    if text.is_empty() {
        return None;
    }
    let mut clean = py_strip(text);
    if let Some(rest) = clean.strip_prefix("```json") {
        clean = rest;
    }
    if let Some(rest) = clean.strip_prefix("```") {
        clean = rest;
    }
    if let Some(rest) = clean.strip_suffix("```") {
        clean = rest;
    }
    match serde_json::from_str::<Value>(clean) {
        Ok(Value::Object(map)) => Some(map),
        Ok(_) => None,
        Err(_) => {
            let preview: String = clean.chars().take(200).collect();
            tracing::warn!(preview = preview.as_str(), "failed to parse ED LLM JSON");
            None
        }
    }
}

/// Python `str.format` with keyword arguments only: `{name}` is replaced by its value, `{{`
/// and `}}` become literal braces.
///
/// # Errors
/// [`EdError::Prompt`] for an unknown or malformed field, or a single `}`.
pub fn py_format(template: &str, args: &[(&str, &str)]) -> Result<String> {
    let mut out = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' if chars.peek() == Some(&'{') => {
                chars.next();
                out.push('{');
            }
            '{' => {
                let mut field = String::new();
                loop {
                    match chars.next() {
                        Some('}') => break,
                        Some('{') | None => {
                            return Err(EdError::Prompt(format!(
                                "malformed placeholder in prompt template near `{{{field}`"
                            )))
                        }
                        Some(other) => field.push(other),
                    }
                }
                let value = args
                    .iter()
                    .find(|(name, _)| *name == field)
                    .map(|(_, value)| *value)
                    .ok_or_else(|| {
                    EdError::Prompt(format!("unknown placeholder `{{{field}}}` in prompt"))
                })?;
                out.push_str(value);
            }
            '}' if chars.peek() == Some(&'}') => {
                chars.next();
                out.push('}');
            }
            '}' => {
                return Err(EdError::Prompt("single '}' encountered in prompt template".to_owned()))
            }
            other => out.push(other),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use impeach_core::{DocContext, LocalContext, Mention, Span};
    use pretty_assertions::assert_eq;

    #[test]
    fn py_format_replaces_fields_and_unescapes_braces() {
        let out = py_format("a {x} {{\"k\": 1}} b", &[("x", "{X}")]).expect("formats");
        assert_eq!(out, "a {X} {\"k\": 1} b");
        assert!(py_format("{missing}", &[]).is_err());
        assert!(py_format("oops }", &[]).is_err());
        assert!(py_format("oops {", &[]).is_err());
    }

    #[test]
    fn parse_llm_response_strips_fences() {
        let parsed =
            parse_llm_response("```json\n{\"chosen_entity_id\": \"e1\"}\n```").expect("object");
        assert_eq!(parsed["chosen_entity_id"], "e1");
        assert!(parse_llm_response("```\n{\"a\": 1}```").is_some());
        assert!(parse_llm_response("[1, 2]").is_none());
        assert!(parse_llm_response("not json").is_none());
        assert!(parse_llm_response("").is_none());
    }

    #[test]
    fn python_float_follows_float_of_or() {
        assert!((python_float(&json!(0.9)) - 0.9).abs() < f64::EPSILON);
        assert!((python_float(&json!("0.5")) - 0.5).abs() < f64::EPSILON);
        assert!(python_float(&Value::Null).abs() < f64::EPSILON);
        assert!(python_float(&json!("")).abs() < f64::EPSILON);
    }

    #[test]
    fn payload_matches_python_json_dumps() {
        let mut doc = DocContext::new("doc-1", "complaint");
        doc.case_id = Some("CASE-1".into());
        let ctx = EdContext {
            mention: Mention::new("m1", "PER", "José Ruiz", Span::new(0, 9)),
            local_context: LocalContext::new("José Ruiz drove.", "José Ruiz drove."),
            doc_context: doc,
            span_label: "PER".into(),
            normalized_key: None,
        };
        let mut candidate = EdCandidate::new("7", "Jose Ruiz", "PER");
        candidate.aliases = vec!["Mr. Ruiz".into()];
        let payload = build_llm_payload_json(&ctx, &[candidate]).expect("payload");
        assert_eq!(
            payload,
            "{\"mention\": {\"id\": \"m1\", \"text\": \"José Ruiz\", \"label\": \"PER\"}, \
             \"local_context\": {\"sentence\": \"José Ruiz drove.\", \"prev_sentence\": null, \
             \"next_sentence\": null}, \"doc_context\": {\"doc_id\": \"doc-1\", \"doc_type\": \
             \"complaint\", \"case_id\": \"CASE-1\", \"jurisdiction\": null}, \"candidates\": \
             [{\"entity_id\": \"7\", \"canonical_name\": \"Jose Ruiz\", \"label\": \"PER\", \
             \"aliases\": [\"Mr. Ruiz\"], \"description\": null}]}"
        );
    }

    #[test]
    fn guidance_is_looked_up_by_normalised_label() {
        assert!(type_specific_guidance("PER").starts_with("PERSON Entity Guidelines:\n"));
        assert!(type_specific_guidance("ORG").ends_with("law firm naming patterns.\n"));
        assert!(type_specific_guidance("CASE").starts_with("CASE Entity Guidelines:\n"));
        assert!(type_specific_guidance("EXHIBIT").starts_with("EXHIBIT Entity Guidelines:\n"));
        assert_eq!(type_specific_guidance("PERSON"), "");
        assert_eq!(type_specific_guidance("LOC"), "");
    }

    #[test]
    fn shipped_prompt_renders_like_python() {
        let prompt = EdPrompt::load("impeach/prompts/ed/disambiguation.json").expect("prompt");
        let system = py_format(
            &prompt.system_template,
            &[("type_specific_guidance", type_specific_guidance("ORG"))],
        )
        .expect("system");
        assert!(system.contains("\n\nORGANIZATION Entity Guidelines:\n"));
        assert!(system.contains(
            "Format:\n{\"chosen_entity_id\": \"entity-id-or-null\", \"confidence\": 0.95, \
             \"rationale\": \"short reason\"}"
        ));
        let user =
            py_format(&prompt.user_template, &[("payload_json", "{\"a\": 1}")]).expect("user");
        assert_eq!(user, "Payload:\n{\"a\": 1}");
    }

    #[test]
    fn truthiness() {
        assert!(!is_truthy(&Value::Null));
        assert!(!is_truthy(&json!("")));
        assert!(!is_truthy(&json!(0)));
        assert!(is_truthy(&json!("doc")));
        assert!(is_truthy(&json!(3)));
    }
}
