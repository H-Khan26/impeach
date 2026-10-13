//! Contradiction check: similar-fact retrieval, pair pre-filter, gating strategies, NLI
//! context, judges, the check service and review-queue items.
//!
//! Python origin: `services/contradictions/` (`retrieval.py`, `prefilter.py`, `gating.py`,
//! `nli_context.py`, `qwen_judge.py`, `review_utils.py`, `service.py`).
//!
//! - [`retrieval`]: [`SimilarFactRetriever`] (graph / BM25 / vector sources, weights,
//!   multi-source boost, `min_score`, strategy) and the candidate entity map.
//! - [`prefilter`]: [`FactPairPreFilter`], the cross-document dual-path filter.
//! - [`intra_doc`]: [`IntraDocFilter`], intra-document pairing and filtering.
//! - [`gating`]: [`ContradictionGating`], `dual_path` or `permissive` gating.
//! - [`nli_context`]: [`NliContextBuilder`], enriched premise / hypothesis text.
//! - [`judge`]: [`ContradictionJudge`] with [`NliJudgeAdapter`] and [`LlmJudge`].
//! - [`review`]: review-queue items from a report.
//! - [`service`]: [`ContradictionCheckService`], the three-phase check.
//!
//! Differences from the Python prototype (handoff bugs fixed):
//! - Intra-document pairs whose `temporal_context` differs are no longer skipped; the
//!   mismatch only lowers their priority ([`intra_doc`]).
//! - Cross-document gating receives the candidates' entity ids from their `FactAboutEntity`
//!   edges, so the shared-entity signals fire (Python passed an empty map).
//! - `pre_filter.require_entity_overlap`, `pre_filter.use_assertion_boost` and
//!   `intra_doc_filter.require_assertion_overlap` take effect ([`prefilter`], [`intra_doc`]).

pub mod error;
pub mod gating;
pub mod intra_doc;
pub mod judge;
pub mod nli_context;
pub mod prefilter;
pub mod retrieval;
pub mod review;
pub mod service;

pub use error::{ContradictionError, Result};
pub use gating::ContradictionGating;
pub use intra_doc::{
    IntraDocCandidate, IntraDocDecision, IntraDocFilter, IntraDocOutcome,
    TEMPORAL_MISMATCH_PRIORITY_FACTOR,
};
pub use judge::{
    nli_verdict, parse_judge_response, ContradictionJudge, JudgeVerdict, LlmJudge, NliJudgeAdapter,
    DEFAULT_JUDGE_PROMPT,
};
pub use nli_context::{build_entity_name_map, NliContextBuilder};
pub use prefilter::{
    cosine_similarity, CandidateEntityMap, FactPairPreFilter, ScoredCandidate, ASSERTION_BOOST,
};
pub use retrieval::{
    retrieved_fact_from_node, GraphLookupCache, SimilarFactRetriever, SOURCE_BM25, SOURCE_GRAPH,
    SOURCE_VECTOR,
};
pub use review::{build_review_items_from_report, ReviewProvenance};
pub use service::{ContradictionCheckService, IntraDocCheck, IntraDocStats};
