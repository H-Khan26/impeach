//! LLM fact extraction (Python `services/facts/extractor.py`).

use std::fmt;
use std::sync::Arc;

use ce_config::prompts::ExtractionPrompt;
use ce_core::{EntityId, ExtractedFact};
use ce_models::Llm;

use crate::error::Result;

/// Certainty of a fact whose `certainty` is missing or not a number.
pub const DEFAULT_CERTAINTY: f64 = 0.5;

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
#[must_use]
pub fn build_entity_context(entities: &[PromptEntity]) -> String {
    let _ = entities;
    todo!("worker A")
}

/// Substitutes `{segments}` with `segments`, then `{resolved_entities}` with `entity_context`
/// (plain string replacement; other braces are left alone).
#[must_use]
pub fn render_prompt(template: &str, segments: &str, entity_context: &str) -> String {
    let _ = (template, segments, entity_context);
    todo!("worker A")
}

/// Parses an LLM response into facts: `STATED_FACT` nodes of the `nodes` list, ids prefixed
/// with `pass_name` when given.
///
/// # Errors
/// [`crate::FactError::Parse`] when the response holds no JSON object.
pub fn parse_facts(response: &str, pass_name: Option<&str>) -> Result<Vec<ExtractedFact>> {
    let _ = (response, pass_name);
    todo!("worker A")
}

/// Deduplication key of a fact: `subject|RELATIONSHIP|object|first 50 chars of text`.
#[must_use]
pub fn fact_key(fact: &ExtractedFact) -> String {
    let _ = fact;
    todo!("worker A")
}

/// Removes facts with the same [`fact_key`], keeping the one with the higher certainty.
#[must_use]
pub fn deduplicate_facts(facts: Vec<ExtractedFact>) -> Vec<ExtractedFact> {
    let _ = facts;
    todo!("worker A")
}

/// Loads the extraction prompt files (resolved with [`ce_config::resolve_asset`]).
///
/// # Errors
/// [`crate::FactError::Config`] when a file is missing or malformed.
pub fn load_prompts(files: &[String]) -> Result<Vec<ExtractionPrompt>> {
    let _ = files;
    todo!("worker A")
}

/// One prompt: [`LlmFactExtractor`]; several: [`MultiPassFactExtractor`].
#[must_use]
pub fn create_extractor(
    llm: Arc<dyn Llm>,
    prompts: &[ExtractionPrompt],
    parallel: bool,
    max_workers: usize,
) -> Box<dyn FactExtractor> {
    let _ = (llm, prompts, parallel, max_workers);
    todo!("worker A")
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
        Self { llm, template: prompt.combined_template() }
    }

    /// The combined prompt template.
    #[must_use]
    pub fn template(&self) -> &str {
        &self.template
    }

    /// Facts of `chunk_text`, ids prefixed with `pass_name` when given.
    #[must_use]
    pub fn extract_with_pass(
        &self,
        chunk_text: &str,
        entities: &[PromptEntity],
        pass_name: Option<&str>,
    ) -> Vec<ExtractedFact> {
        let _ = (&self.llm, chunk_text, entities, pass_name);
        todo!("worker A")
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
        let _ = (llm, prompts);
        Self { passes: Vec::new(), parallel, max_workers }
    }

    /// Pass names in run order.
    #[must_use]
    pub fn pass_names(&self) -> Vec<&str> {
        self.passes.iter().map(|(name, _)| name.as_str()).collect()
    }
}

impl FactExtractor for MultiPassFactExtractor {
    fn extract_facts(&self, chunk_text: &str, entities: &[PromptEntity]) -> Vec<ExtractedFact> {
        let _ = (chunk_text, entities, self.parallel, self.max_workers);
        todo!("worker A")
    }
}
