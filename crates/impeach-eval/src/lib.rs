//! Evaluator: parses a markdown contradiction key, matches detected pairs to key pairs and
//! reports recall, precision and F1.
//!
//! Python origin: `tests/e2e/evaluate_output.py`.
//!
//! 1. [`parse_key`] reads the key's "Interdocument Contradictions" table (`C-*` rows, fact A
//!    from Doc 1, fact B from Doc 2) and "Intradocument Contradictions" tables (`D*` rows).
//! 2. [`PipelineOutput::from_e2e`] takes the detected pairs (`documents[*]
//!    .stage_6_contradiction_check`) and the facts used for coverage from a
//!    [`impeach_core::e2e::E2eOutput`].
//! 3. Key facts and pipeline facts are embedded with an [`impeach_models::Embedder`]
//!    (`all-MiniLM-L6-v2` in production); similarity is cosine similarity.
//! 4. A key row (A, B) against a detected pair (new, existing) scores
//!    `min(sim(A, x), sim(B, y))` in both orientations and keeps the better one. Inter-document
//!    rows are compared only with cross-document detections, intra-document rows only with
//!    intra-document detections.
//! 5. Value guard ([`impeach_text::values`]): when both the key fact and the aligned pipeline fact
//!    carry numbers, amounts, dates, years or clock times and none agree, the alignment scores 0.
//! 6. Matching is one-to-one: [`assignment::linear_sum_assignment`] (a port of SciPy's solver)
//!    maximises the total admissible score; only pairs with score >= threshold count.
//! 7. [`EvalReport`] holds recall / precision / F1 overall and per bucket, key-fact coverage
//!    and the audit lists, serialises to the Python report JSON ([`EvalReport::to_json`]) and
//!    renders the Python console summary ([`EvalReport::summary`]).

pub mod assignment;
pub mod error;
pub mod evaluate;
pub mod key;
pub mod pipeline_output;
pub mod report;
pub mod similarity;

pub use assignment::{assign_one_to_one, AssignmentMethod};
pub use error::{EvalError, Result};
pub use evaluate::{
    default_report_path, evaluate, evaluate_body, run_evaluation, run_evaluation_with_key,
    score_pairs, EvalBody, EvalOptions, PairScore, DEFAULT_MODEL, DEFAULT_THRESHOLD,
};
pub use key::{parse_key, parse_key_file, GroundTruthContradiction, GroundTruthFact, Key};
pub use pipeline_output::{
    load_e2e_output, DetectedContradiction, FactOrigin, PipelineFact, PipelineOutput, SourceType,
};
pub use report::EvalReport;
pub use similarity::SimilarityCache;
