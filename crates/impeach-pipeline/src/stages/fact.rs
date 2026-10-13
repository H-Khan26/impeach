//! `FactStage`: LLM fact extraction per chunk, entity linking and optional commit (Python
//! `stages/facts.py`).
//!
//! Flow:
//! 1. Nothing to do (context unchanged) without `rag_index_result`. An offline run
//!    (`index_persisted == false` or no context `embedder`; Python: no `helix_client` /
//!    `embedding_service`) and a missing LLM (`<pipeline.name>_<model_type>` not in the
//!    registry or not buildable) log a warning and leave the context unchanged.
//!    The registry-backed extractor is built on first use and cached for the stage's lifetime.
//! 2. Lookups from `ed_results`, `resolved_mentions`, `doc_mentions` and `role_map`
//!    ([`FactLinker::build_entity_lookup`], [`build_canonical_lookup`]).
//! 3. Per chunk of `rag_index_result` in order (keyed by its graph id), skipping chunks
//!    shorter than [`MIN_CHUNK_CHARS`] characters: the prompt entities of the chunk's resolved
//!    mentions, the sentence spans of the chunk, the text (the resolved-text view when
//!    `use_resolved_text_view`, also stored in `resolved_text_by_chunk`), then extraction.
//! 4. Per fact: boilerplate facts are dropped; the source chunk / document ids and the
//!    metadata keys `source_chunk_id`, `source_span_ids` (when the chunk has sentence spans),
//!    `slot_name` (when set), `temporal_context`, `is_denial`, `extraction_mode`, `value_raw`
//!    (when non-empty) and `time_window` (when set) are set; subject and object are linked
//!    ([`FactLinker::link_fact`]) and counted in the linking stats.
//! 5. Writes `extracted_facts` and `fact_linking_stats`. With facts: `commit_facts: true`
//!    commits them through [`FactIndexer::commit_facts`] (nodes, embeddings, context edges,
//!    `metadata.helix_fact_id`) and writes `indexed_facts_count` and `fact_helix_ids`;
//!    `commit_facts: false` writes `indexed_facts_count = 0`, `fact_helix_ids = []` and the
//!    unwritten nodes in `fact_nodes_pending`. Finally `extraction_mode`.
//!
//! Store errors while committing are logged per fact or edge, except fatal ones, which fail the
//! stage. LLM and parse failures yield no facts for the affected chunk or pass.

use std::borrow::Cow;
use std::sync::{Arc, OnceLock};

use impeach_config::mappings::Taxonomy;
use impeach_config::pipeline::FactStageConfig;
use impeach_config::prompts::ExtractionPrompt;
use impeach_config::{MappingsRegistry, StageConfig};
use impeach_core::processing::FactLinkingStats;
use impeach_core::{
    ChunkId, DocId, EntityId, ExtractedFact, IndexingResult, Mention, MentionId, NodeId, SpanId,
};
use impeach_facts::{
    build_canonical_lookup, build_resolved_text_view, create_extractor, fact_node, load_prompts,
    prompt_entities, record_link, CanonicalEntity, EntityLookup, FactError, FactExtractor,
    FactIndexer, FactLinker,
};
use impeach_models::{Embedder, ModelError, ModelRegistry};
use impeach_ner::definitions::RoleConfig;
use impeach_store::FactNode;
use indexmap::{IndexMap, IndexSet};
use serde_json::Value;

use crate::context::{ContextData, PipelineContext, StageBuildContext};
use crate::error::{PipelineError, Result};
use crate::stage::Stage;

/// Chunks with fewer characters are not sent to the extractor.
pub const MIN_CHUNK_CHARS: usize = 50;

/// The fact extraction stage.
#[derive(Debug)]
pub struct FactStage {
    name: String,
    config: FactStageConfig,
    pipeline_name: String,
    prompts: Vec<ExtractionPrompt>,
    taxonomy: &'static Taxonomy,
    alias_exclusion: &'static IndexSet<String>,
    linker: FactLinker,
    extractor: OnceLock<Arc<dyn FactExtractor>>,
}

/// Lookups shared by every chunk of one run.
struct Lookups<'a> {
    entity_lookup: EntityLookup,
    canonical: IndexMap<EntityId, CanonicalEntity>,
    resolved: Cow<'a, IndexMap<MentionId, EntityId>>,
    mentions_by_chunk: Option<&'a IndexMap<ChunkId, Vec<Mention>>>,
    role_map: Option<&'a IndexMap<String, EntityId>>,
    source_doc_id: Option<DocId>,
}

/// Facts of one run and what they produced.
#[derive(Default)]
struct Extraction {
    facts: Vec<ExtractedFact>,
    nodes: Vec<FactNode>,
    stats: FactLinkingStats,
    resolved_text_by_chunk: Option<IndexMap<ChunkId, String>>,
}

impl FactStage {
    /// Config `type` of this stage.
    pub const TYPE_NAME: &'static str = "FactStage";

    /// Builds the stage from its config block and the shared services: loads the mappings,
    /// the prompt files and the shipped `roles.yaml`.
    ///
    /// # Errors
    /// [`PipelineError::Config`] when a prompt file, the mappings or `roles.yaml` cannot be
    /// loaded.
    pub fn new(config: &FactStageConfig, services: &StageBuildContext) -> Result<Self> {
        let mappings = MappingsRegistry::load()?;
        let prompts = load_prompts(&config.prompt_files).map_err(|error| match error {
            FactError::Config(error) => PipelineError::Config(error),
            other => PipelineError::Fact(other),
        })?;
        let roles = RoleConfig::load(None)?;
        Ok(Self {
            name: config.name.clone(),
            config: config.clone(),
            pipeline_name: services.config.name().to_owned(),
            prompts,
            taxonomy: &mappings.taxonomy,
            alias_exclusion: &mappings.normalization.alias_exclusion,
            linker: FactLinker::new(mappings, Some(roles)),
            extractor: OnceLock::new(),
        })
    }

    /// [`crate::StageConstructor`] of this stage.
    ///
    /// # Errors
    /// [`PipelineError::InvalidStageConfig`] for another stage type; errors of
    /// [`FactStage::new`].
    pub fn construct(stage: &StageConfig, services: &StageBuildContext) -> Result<Box<dyn Stage>> {
        match stage {
            StageConfig::Fact(config) => Ok(Box::new(Self::new(config, services)?)),
            other => Err(PipelineError::InvalidStageConfig {
                stage: other.name().to_owned(),
                message: format!(
                    "expected a {} config, got {}",
                    Self::TYPE_NAME,
                    other.type_name()
                ),
            }),
        }
    }

    /// Replaces the extractor built from the registry's LLM and the prompt files.
    #[must_use]
    pub fn with_extractor(mut self, extractor: Arc<dyn FactExtractor>) -> Self {
        self.extractor = OnceLock::from(extractor);
        self
    }

    /// Extraction strategy name for logs.
    fn strategy(&self) -> &'static str {
        if self.config.prompt_files.len() == 1 {
            "single-pass"
        } else {
            "multi-pass"
        }
    }

    /// The injected or cached extractor, else one over the registry's
    /// `<pipeline>_<model_type>` LLM, built once per stage; `None` (logged) when that model is
    /// unavailable.
    fn extractor(&self, registry: &ModelRegistry) -> Option<Arc<dyn FactExtractor>> {
        if let Some(extractor) = self.extractor.get() {
            return Some(Arc::clone(extractor));
        }
        let key = format!("{}_{}", self.pipeline_name, self.config.model_type);
        match registry.llm(&key) {
            Ok(llm) => {
                tracing::info!(
                    key = key.as_str(),
                    strategy = self.strategy(),
                    prompts = self.prompts.len(),
                    "FactStage: extractor ready"
                );
                let extractor = self.extractor.get_or_init(|| {
                    Arc::from(create_extractor(
                        llm,
                        &self.prompts,
                        self.config.parallel,
                        self.config.max_workers,
                    ))
                });
                Some(Arc::clone(extractor))
            }
            Err(ModelError::UnknownModel { .. }) => {
                tracing::warn!(
                    key = key.as_str(),
                    "FactStage: model not found in registry; skipping"
                );
                None
            }
            Err(error) => {
                tracing::warn!(key = key.as_str(), error = %error, "FactStage: model unavailable; skipping");
                None
            }
        }
    }

    /// Lookups over the ED outputs in the context.
    fn lookups<'a>(&self, data: &'a ContextData) -> Lookups<'a> {
        let ed_results = data.ed_results.as_deref().unwrap_or_default();
        let resolved = data
            .resolved_mentions
            .as_ref()
            .map_or_else(|| Cow::Owned(IndexMap::new()), Cow::Borrowed);
        let doc_mentions = data.doc_mentions.as_deref().unwrap_or_default();
        let source_doc_id = data
            .helix_doc_id
            .as_ref()
            .map(NodeId::as_str)
            .filter(|id| !id.is_empty())
            .or_else(|| data.doc_id.as_ref().map(DocId::as_str).filter(|id| !id.is_empty()))
            .map(DocId::new);
        Lookups {
            entity_lookup: self.linker.build_entity_lookup(ed_results, &resolved, doc_mentions),
            canonical: build_canonical_lookup(ed_results),
            resolved,
            mentions_by_chunk: data.mentions_by_chunk.as_ref(),
            role_map: data.role_map.as_ref(),
            source_doc_id,
        }
    }

    /// Extracts, annotates and links the facts of every eligible chunk.
    fn extract(
        &self,
        index: &IndexingResult,
        lookups: &Lookups<'_>,
        extractor: &dyn FactExtractor,
    ) -> Extraction {
        let mut extraction = Extraction::default();
        if self.config.use_resolved_text_view {
            extraction.resolved_text_by_chunk = Some(IndexMap::new());
        }
        let spans_by_chunk = spans_by_chunk(index);
        for chunk_result in &index.chunks {
            let chunk = &chunk_result.chunk;
            if chunk.text.chars().count() < MIN_CHUNK_CHARS {
                continue;
            }
            let chunk_id = ChunkId::new(chunk_result.helix_id.as_str());
            let chunk_mentions = lookups
                .mentions_by_chunk
                .and_then(|by_chunk| by_chunk.get(&chunk_id))
                .map_or(&[][..], Vec::as_slice);
            let entities = prompt_entities(
                chunk_mentions,
                &lookups.resolved,
                &lookups.canonical,
                self.alias_exclusion,
                self.linker.roles(),
            );
            let span_ids =
                spans_by_chunk.get(&chunk.sequence_number).map_or(&[][..], Vec::as_slice);

            let text: Cow<'_, str> = match extraction.resolved_text_by_chunk.as_mut() {
                Some(resolved_texts) => {
                    let view = build_resolved_text_view(
                        &chunk.text,
                        chunk.start_char,
                        chunk_mentions,
                        &lookups.resolved,
                        &lookups.canonical,
                    );
                    tracing::debug!(
                        chunk = %chunk_id,
                        original_chars = chunk.text.chars().count(),
                        resolved_chars = view.chars().count(),
                        "FactStage: using resolved text view"
                    );
                    resolved_texts.insert(chunk_id.clone(), view.clone());
                    Cow::Owned(view)
                }
                None => Cow::Borrowed(chunk.text.as_str()),
            };

            for mut fact in extractor.extract_facts(&text, &entities) {
                if fact.is_boilerplate {
                    tracing::debug!(
                        fact = fact.id.as_str(),
                        "FactStage: skipping boilerplate fact"
                    );
                    continue;
                }
                self.annotate(&mut fact, &chunk_id, lookups.source_doc_id.as_ref(), span_ids);
                let (subject, object) = self.linker.link_fact(
                    &mut fact,
                    &lookups.entity_lookup,
                    &lookups.canonical,
                    lookups.role_map,
                );
                record_link(&mut extraction.stats, subject.as_ref(), object.as_ref());
                fact.subject_entity_id = subject;
                fact.object_entity_id = object;
                if !self.config.commit_facts {
                    extraction.nodes.push(fact_node(&fact, self.taxonomy));
                }
                extraction.facts.push(fact);
            }
        }
        extraction
    }

    /// Sets the source ids and the stage's metadata keys of one fact.
    fn annotate(
        &self,
        fact: &mut ExtractedFact,
        chunk_id: &ChunkId,
        source_doc_id: Option<&DocId>,
        span_ids: &[&SpanId],
    ) {
        fact.source_chunk_id = Some(chunk_id.clone());
        fact.source_doc_id = source_doc_id.cloned();
        let metadata = &mut fact.metadata;
        metadata.insert("source_chunk_id".to_owned(), Value::String(chunk_id.to_string()));
        if !span_ids.is_empty() {
            let ids = span_ids.iter().map(|id| Value::String(id.to_string())).collect();
            metadata.insert("source_span_ids".to_owned(), Value::Array(ids));
        }
        if let Some(slot_name) = fact.slot_name.as_ref().filter(|s| !s.is_empty()) {
            metadata.insert("slot_name".to_owned(), Value::String(slot_name.clone()));
        }
        metadata.insert(
            "temporal_context".to_owned(),
            Value::String(fact.temporal_context.as_str().to_owned()),
        );
        metadata.insert("is_denial".to_owned(), Value::Bool(fact.is_denial));
        metadata.insert(
            "extraction_mode".to_owned(),
            Value::String(extraction_mode_name(self.config.extraction_mode).to_owned()),
        );
        if let Some(value_raw) = fact.value_raw.as_ref().filter(|v| !v.is_empty()) {
            metadata.insert("value_raw".to_owned(), Value::String(value_raw.clone()));
        }
        if let Some(window) = &fact.time_window {
            let window = serde_json::to_value(window).expect("a time window serialises to JSON");
            metadata.insert("time_window".to_owned(), window);
        }
    }

    /// Logs the link rate and span provenance of the run.
    fn log_summary(extraction: &Extraction) {
        let stats = &extraction.stats;
        if stats.total_facts > 0 {
            #[allow(clippy::cast_precision_loss)]
            let link_rate = stats.linked_either as f64 / stats.total_facts as f64 * 100.0;
            tracing::info!(
                linked = stats.linked_either,
                total = stats.total_facts,
                link_rate = format!("{link_rate:.1}%"),
                subjects = stats.linked_subjects,
                objects = stats.linked_objects,
                both = stats.linked_both,
                "FactStage: linked facts to entities"
            );
        }
        let with_spans =
            extraction.facts.iter().filter(|f| f.metadata.contains_key("source_span_ids")).count();
        if with_spans > 0 {
            tracing::info!(
                with_spans,
                facts = extraction.facts.len(),
                "FactStage: facts with source span provenance for FactDerivedFromSpan edges"
            );
        }
    }
}

/// The Python literal of an extraction mode.
fn extraction_mode_name(mode: impeach_config::pipeline::ExtractionMode) -> &'static str {
    match mode {
        impeach_config::pipeline::ExtractionMode::Processing => "processing",
    }
}

/// Graph ids of the written sentence spans, grouped by chunk sequence number.
fn spans_by_chunk(index: &IndexingResult) -> IndexMap<usize, Vec<&SpanId>> {
    let mut by_chunk: IndexMap<usize, Vec<&SpanId>> = IndexMap::new();
    for span in index.spans.iter().filter(|span| !span.span_id.as_str().is_empty()) {
        by_chunk.entry(span.chunk_sequence).or_default().push(&span.span_id);
    }
    by_chunk
}

impl Stage for FactStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, ctx: &mut PipelineContext) -> Result<()> {
        let Some(index) = ctx.data.rag_index_result.as_ref() else {
            return Ok(());
        };
        let embedder: Arc<dyn Embedder> = match &ctx.data.embedder {
            Some(embedder) if ctx.data.index_persisted => Arc::clone(&embedder.0),
            _ => {
                tracing::warn!(
                    "FactStage: index not persisted or no embedder (offline run); skipping fact extraction"
                );
                return Ok(());
            }
        };
        let Some(extractor) = self.extractor(&ctx.registry) else {
            return Ok(());
        };

        let lookups = self.lookups(&ctx.data);
        let mut extraction = self.extract(index, &lookups, extractor.as_ref());
        drop(lookups);
        Self::log_summary(&extraction);

        let data = &mut ctx.data;
        if !extraction.facts.is_empty() {
            if self.config.commit_facts {
                let indexer = FactIndexer::new(Arc::clone(&ctx.store), embedder);
                let result = indexer.commit_facts(&mut extraction.facts, self.taxonomy)?;
                tracing::info!(count = result.count, "FactStage: indexed facts");
                data.indexed_facts_count = Some(result.count);
                data.fact_helix_ids = Some(result.fact_ids);
            } else {
                let denials = extraction.facts.iter().filter(|f| f.is_denial).count();
                tracing::info!(
                    strategy = self.strategy(),
                    facts = extraction.facts.len(),
                    denials,
                    "FactStage: extracted facts (commit_facts=false, skipping graph write)"
                );
                data.indexed_facts_count = Some(0);
                data.fact_helix_ids = Some(Vec::new());
                data.fact_nodes_pending = Some(extraction.nodes);
            }
        }
        if let Some(resolved_texts) = extraction.resolved_text_by_chunk.filter(|t| !t.is_empty()) {
            data.resolved_text_by_chunk = Some(resolved_texts);
        }
        data.extracted_facts = Some(extraction.facts);
        data.fact_linking_stats = Some(extraction.stats);
        data.extraction_mode = Some(self.config.extraction_mode);
        Ok(())
    }
}
