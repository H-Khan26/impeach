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
//!   the enabled stages in config order (see [`factory`] for disabled and not-yet-ported
//!   stage types).
//! - [`RagIndexStage`]: chunking ([`rag::chunker`] documents the boundary rule), Document /
//!   Chunk / Span nodes, chunk embeddings, optional document-identity cache.
//! - [`run_processing`]: the public entry point over a pre-ingested JSONL file;
//!   [`run_processing_with`] injects the model registry, graph store and stage registry.
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

pub mod api;
pub mod context;
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
pub use stages::RagIndexStage;
