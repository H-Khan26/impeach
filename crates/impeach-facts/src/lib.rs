//! Fact extraction: LLM facts per chunk, entity linking and fact indexing (Python
//! `services/facts/` and the helpers of `stages/facts.py`).
//!
//! - [`extractor`]: prompt rendering, LLM output parsing into [`impeach_core::ExtractedFact`]s,
//!   [`LlmFactExtractor`] (single pass) and [`MultiPassFactExtractor`] (several focused passes,
//!   merged and deduplicated).
//! - [`linking`]: subject / object linking to ED entities (bracket tokens, type-aware
//!   normalised names, role terms), the resolved-text view and the per-chunk prompt entities.
//! - [`indexer`]: [`FactIndexer`] writes `Fact` nodes, fact embeddings and the
//!   `FactInDocument` / `FactAboutEntity` / `FactDerivedFromSpan` edges.

pub mod error;
pub mod extractor;
pub mod indexer;
pub mod linking;

pub use error::{FactError, Result};
pub use extractor::{
    build_entity_context, create_extractor, deduplicate_facts, fact_key, load_prompts, parse_facts,
    render_prompt, FactExtractor, LlmFactExtractor, MultiPassFactExtractor, PromptEntity,
};
pub use indexer::{fact_node, FactContextLinks, FactIndexer, IndexResult, FACT_EMBED_BATCH_SIZE};
pub use linking::{
    build_canonical_lookup, build_resolved_text_view, parse_bracket_token, prompt_entities,
    record_link, CanonicalEntity, EntityLookup, FactLinker,
};
