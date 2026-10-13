//! Processing pipeline: stage trait, typed pipeline context, runner, factory and stages.
//!
//! Python origin: `pipeline/{factory,runner}.py`, `stages/{base,__init__,rag_index}.py`,
//! `services/rag/*` and `api/processing.py`.
//!
//! - [`Stage`]: one processing step; `run(&self, &mut PipelineContext)`.
//! - [`PipelineContext`]: shared services (config, model registry, graph store) plus
//!   [`ContextData`], one typed `Option` slot per Python context key.
//! - [`Pipeline`]: ordered stages; [`Pipeline::run`] takes a [`PipelineInput`] (a
//!   `DocumentPayload`, a JSONL path or a JSON file path) and logs per-stage timings.
//! - [`StageRegistry`]: stage `type` name -> [`StageConstructor`]; [`build_pipeline`] builds
//!   the enabled stages in config order (see [`factory`] for disabled and unregistered stage
//!   types). [`StageRegistry::builtin`] registers all eight stages.
//! - [`RagIndexStage`]: chunking ([`rag::chunker`] documents the boundary rule), Document /
//!   Chunk / Span nodes, chunk embeddings, optional document-identity cache.
//! - [`MentionStage`], [`DefinitionStage`], [`CoreferenceStage`]: mentions, defined role terms
//!   and coreference clusters; [`EdStage`]: mentions -> entities (`ce_ed`), role map and
//!   mention span links; [`RelationStage`]: GLiNER2 relations kept in the context.
//! - [`FactStage`]: LLM facts per chunk (`ce_facts`), linked to ED entities; committed to the
//!   store (`commit_facts: true`) or left in `fact_nodes_pending` for the contradiction stage.
//! - [`ContradictionStage`]: intra- and cross-document contradiction check
//!   (`ce_contradictions`), commit of the clean facts and the SQLite review queue.
//! - [`run_processing`]: the public entry point over a pre-ingested JSONL file;
//!   [`run_processing_with`] injects the model registry, graph store and stage registry.
//! - [`e2e`]: the end-to-end runner (sequential documents into one store, the stage-based
//!   output JSON `ce-eval` reads, evaluation against a key).
//!
//! ```no_run
//! use ce_pipeline::{run_processing, ProcessingOptions};
//!
//! let result = run_processing("output/complaint.jsonl", &ProcessingOptions::default())?;
//! println!("{} contradictions", result.contradiction_count);
//! # Ok::<(), ce_pipeline::PipelineError>(())
//! ```
//!
//! Differences from the Python prototype:
//! - Chunk boundaries come from `text-splitter` instead of Chonkie (deterministic; rule in
//!   [`rag::chunker`]). The `chonkie_*` chunk metadata keys are kept; `chonkie_chunk_id` is null
//!   and `chonkie_token_count` is the character count.
//! - Chunks, spans and embeddings are written with the store's batch methods. A failed span or
//!   embedding batch falls back to one-by-one writes so a single bad item is skipped (spans) or
//!   reported in `errors` (embeddings) as in Python.
//! - When a document node is reused (document cache or explicit `helix_id`), its existing
//!   chunks, their sentence spans and chunk vectors are deleted in one store call and the new
//!   chunks and spans are created, so re-indexing leaves exactly one set of each. Python updated
//!   chunks by sequence number within one process, created duplicate chunks under a node reused
//!   from the on-disk cache, and re-created sentence spans on every reuse.
//! - `ProcessingResult.sqlite_db_path` is set only when a `ContradictionCheckStage` with the
//!   review store was built and ran; Python echoed `db_path` whenever it was given.
//! - Embedder output whose count or dimension does not match the chunks / store fails the stage
//!   instead of producing one store error per vector.
//! - Without the `helix` feature, [`run_processing`] uses an in-memory store.
//! - [`EdStage`] loads its prompt file and `roles.yaml` when it is built and fails the build
//!   when either is missing (Python logged a missing prompt and failed at the first LLM
//!   tie-break); an offline run (`index_persisted == false`) resolves nothing, like Python
//!   without a Helix client.
//! - [`RelationStage`] uses [`stages::relation::DEFAULT_RELATION_THRESHOLD`] when the config has
//!   no `models.gliner2` block (an injected model); Python required the block.
//! - [`FactStage`] loads its prompt files and `roles.yaml` when it is built and fails the build
//!   when one is missing (Python logged it and skipped the stage at run time). Its commit path
//!   (`commit_facts: true`) also writes the `FactInDocument` / `FactAboutEntity` /
//!   `FactDerivedFromSpan` edges and `metadata.helix_fact_id`, like the contradiction stage's
//!   commit (Python wrote only nodes and embeddings there). Fact embeddings are computed in
//!   batches of [`ce_facts::FACT_EMBED_BATCH_SIZE`] with vectors kept aligned with their facts.
//!   Fact node value / property types are always normalised through the taxonomy. A role term
//!   that links to no name is looked up in the EDStage `role_map`. Prompt aliases keep
//!   first-seen order (Python used a set).
//! - [`ContradictionStage`] loads the LLM judge prompt at build time (`judge_model: qwen`) and
//!   fails the build without `qwen_prompt_file`. Facts the fact stage already committed
//!   (`commit_facts: true`) are excluded from their own document's cross-document retrieval
//!   and not written twice. The judge is resolved once per stage. A judge-less run reports
//!   every fact clean with the fact stage's committed ids (Python: none). Judge calls are
//!   batched per fact (cross-document) and per document (intra-document).
//! - Every stage type has a constructor, so [`run_processing`] and [`build_pipeline`] fail on
//!   an unregistered stage type ([`UnregisteredStages::Error`]); a custom registry with only
//!   some stages passes [`UnregisteredStages::Skip`].
//! - The e2e runner ([`e2e`]) records a failed document's error message (Python stored the
//!   exception `repr`) and renders header scalars as text.

pub mod api;
pub mod context;
pub mod e2e;
pub mod error;
pub mod factory;
pub mod rag;
pub mod runner;
pub mod stage;
pub mod stages;

pub use api::{
    load_processing_config, open_store, run_processing, run_processing_with, ProcessingEnv,
    ProcessingOptions, ProcessingRun,
};
pub use ce_core::processing::ProcessingResult;
pub use context::{ContextData, PipelineContext, SharedEmbedder, StageBuildContext};
pub use error::{PipelineError, Result};
pub use factory::{build_pipeline, StageConstructor, StageRegistry, UnregisteredStages};
pub use runner::{Pipeline, PipelineInput, RunReport, StageTiming};
pub use stage::Stage;
pub use stages::{
    ContradictionStage, CoreferenceStage, DefinitionStage, EdStage, FactStage, MentionStage,
    RagIndexStage, RelationStage,
};
