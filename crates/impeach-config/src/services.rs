//! Service configs for the contradiction check, built from validated stage YAML.
//!
//! These mirror the Python dataclasses in `schema/contradictions.py`. They share names
//! with some YAML blocks in [`crate::pipeline`] but carry service defaults and extra knobs
//! that the YAML does not expose; [`ContradictionCheckConfig::from_stage`] builds them.

use crate::pipeline::{ContradictionCheckStageConfig, FactRetrievalStrategy};

/// Similar-fact retrieval settings.
#[derive(Debug, Clone, PartialEq)]
pub struct SimilarFactRetrieverConfig {
    /// BM25 result limit.
    pub bm25_limit: usize,
    /// Vector result limit.
    pub vector_limit: usize,
    /// Maximum candidates after ranking.
    pub max_candidates: usize,
    /// Weight for graph (shared entity) hits.
    pub graph_weight: f64,
    /// Weight for BM25 hits.
    pub bm25_weight: f64,
    /// Weight for vector hits.
    pub vector_weight: f64,
    /// Boost for candidates found by several methods.
    pub multi_source_boost: f64,
    /// Minimum ranked score.
    pub min_score: f64,
    /// Retrieval strategy.
    pub fact_retrieval_strategy: FactRetrievalStrategy,
}

impl Default for SimilarFactRetrieverConfig {
    fn default() -> Self {
        Self {
            bm25_limit: 8,
            vector_limit: 8,
            max_candidates: 25,
            graph_weight: 1.0,
            bm25_weight: 0.9,
            vector_weight: 0.8,
            multi_source_boost: 1.2,
            min_score: 0.0,
            fact_retrieval_strategy: FactRetrievalStrategy::Both,
        }
    }
}

/// Cross-document pair pre-filter settings.
#[derive(Debug, Clone, PartialEq)]
pub struct PreFilterConfig {
    /// Jaccard threshold for the text-overlap fallback.
    pub min_text_overlap: f64,
    /// Skip pairs without shared entities (when entity ids are available).
    pub require_entity_overlap: bool,
    /// Enable the text-overlap fallback.
    pub enable_text_overlap_fallback: bool,
    /// Enable the semantic-similarity fallback.
    pub use_semantic_similarity_fallback: bool,
    /// Semantic fallback threshold.
    pub semantic_similarity_threshold: f64,
    /// Path A subject overlap threshold.
    pub subject_overlap_threshold: f64,
    /// Path A predicate similarity threshold.
    pub predicate_similarity_threshold: f64,
    /// Path B fuzzy match threshold.
    pub fuzzy_match_threshold: f64,
    /// Prioritise pairs whose subject/object line up.
    pub use_assertion_boost: bool,
}

impl Default for PreFilterConfig {
    fn default() -> Self {
        Self {
            min_text_overlap: 0.05,
            require_entity_overlap: false,
            enable_text_overlap_fallback: false,
            use_semantic_similarity_fallback: false,
            semantic_similarity_threshold: 0.45,
            subject_overlap_threshold: 0.5,
            predicate_similarity_threshold: 0.7,
            fuzzy_match_threshold: 0.6,
            use_assertion_boost: true,
        }
    }
}

/// Intra-document pre-filter settings.
#[derive(Debug, Clone, PartialEq)]
pub struct IntraDocFilterConfig {
    /// Jaccard threshold for intra-document pairs.
    pub min_text_overlap: f64,
    /// Only compare facts whose subject or object match.
    pub require_assertion_overlap: bool,
    /// Temporal relationship keywords.
    pub temporal_relationships: Vec<String>,
    /// Quantitative relationship keywords.
    pub quantitative_relationships: Vec<String>,
    /// Role relationship keywords.
    pub role_relationships: Vec<String>,
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

impl Default for IntraDocFilterConfig {
    fn default() -> Self {
        Self {
            min_text_overlap: 0.15,
            require_assertion_overlap: true,
            temporal_relationships: strings(&[
                "hired",
                "started",
                "joined",
                "left",
                "terminated",
                "resigned",
                "founded",
                "established",
                "created",
                "dissolved",
                "closed",
                "occurred",
                "happened",
                "began",
                "ended",
                "completed",
                "born",
                "died",
                "married",
                "divorced",
            ]),
            quantitative_relationships: strings(&[
                "earned", "paid", "cost", "valued", "worth", "employed", "counted", "numbered",
                "measured", "weighed", "sized",
            ]),
            role_relationships: strings(&[
                "ceo",
                "president",
                "director",
                "manager",
                "supervisor",
                "employee",
                "contractor",
                "owner",
                "founder",
                "partner",
                "reports_to",
                "supervised_by",
                "managed_by",
            ]),
        }
    }
}

/// Full contradiction check service configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct ContradictionCheckConfig {
    /// Retrieval settings.
    pub retrieval: SimilarFactRetrieverConfig,
    /// Cross-document pre-filter.
    pub pre_filter: PreFilterConfig,
    /// Intra-document pre-filter.
    pub intra_doc_filter: IntraDocFilterConfig,
    /// NLI score at or above which a pair is a contradiction.
    pub contradiction_threshold: f64,
    /// NLI score at or above which (and below `contradiction_threshold`) a pair needs review.
    pub review_threshold: f64,
    /// Check facts within the same document.
    pub intra_document_enabled: bool,
    /// Commit clean facts while iterating instead of at the end.
    pub commit_as_we_go: bool,
    /// Cap on intra-document pairs.
    pub max_intra_doc_pairs: usize,
}

impl Default for ContradictionCheckConfig {
    fn default() -> Self {
        Self {
            retrieval: SimilarFactRetrieverConfig::default(),
            pre_filter: PreFilterConfig::default(),
            intra_doc_filter: IntraDocFilterConfig::default(),
            contradiction_threshold: 0.85,
            review_threshold: 0.6,
            intra_document_enabled: true,
            commit_as_we_go: false,
            max_intra_doc_pairs: 2000,
        }
    }
}

impl ContradictionCheckConfig {
    /// Builds the service config from a validated `ContradictionCheckStage` block.
    pub fn from_stage(stage: &ContradictionCheckStageConfig) -> Self {
        let r = &stage.retrieval;
        let p = &stage.pre_filter;
        Self {
            retrieval: SimilarFactRetrieverConfig {
                bm25_limit: r.bm25_limit,
                vector_limit: r.vector_limit,
                max_candidates: r.max_candidates,
                graph_weight: r.graph_weight,
                bm25_weight: r.bm25_weight,
                vector_weight: r.vector_weight,
                multi_source_boost: r.multi_source_boost,
                min_score: r.min_score,
                fact_retrieval_strategy: r.fact_retrieval_strategy,
            },
            pre_filter: PreFilterConfig {
                min_text_overlap: p.min_text_overlap,
                require_entity_overlap: p.require_entity_overlap,
                enable_text_overlap_fallback: p.enable_text_overlap_fallback,
                use_semantic_similarity_fallback: p.use_semantic_similarity_fallback,
                semantic_similarity_threshold: p.semantic_similarity_threshold,
                subject_overlap_threshold: p.subject_overlap_threshold,
                predicate_similarity_threshold: p.predicate_similarity_threshold,
                fuzzy_match_threshold: p.fuzzy_match_threshold,
                use_assertion_boost: p.use_assertion_boost,
            },
            intra_doc_filter: IntraDocFilterConfig {
                min_text_overlap: stage.intra_doc_filter.min_text_overlap,
                require_assertion_overlap: stage.intra_doc_filter.require_assertion_overlap,
                ..IntraDocFilterConfig::default()
            },
            contradiction_threshold: stage.nli.contradiction_threshold,
            review_threshold: stage.nli.review_threshold,
            intra_document_enabled: stage.intra_document.enabled,
            commit_as_we_go: stage.intra_document.commit_as_we_go,
            ..Self::default()
        }
    }
}

impl From<&ContradictionCheckStageConfig> for ContradictionCheckConfig {
    fn from(stage: &ContradictionCheckStageConfig) -> Self {
        Self::from_stage(stage)
    }
}
