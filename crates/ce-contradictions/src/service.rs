//! The contradiction check service (Python `services/contradictions/service.py`).
//!
//! [`ContradictionCheckService::check_facts_for_contradictions`] runs three phases over the
//! facts of one document:
//! 1. Intra-document (when `intra_document_enabled` and more than one fact; skipped with a
//!    warning when `n * (n - 1) / 2 > max_intra_doc_pairs`): bucketed candidate pairs
//!    ([`IntraDocFilter`]), pre-filter, judge on enriched text of both facts. A contradiction
//!    label at or above `review_threshold` makes one pair (lower fact id as the "new" fact,
//!    `retrieval_sources = ["intra_document"]`, source `intra_document`, priority from the
//!    filter decision); both facts get a result (`contradiction` when any of their pairs
//!    reaches `contradiction_threshold`, else `needs_review`, `retrieval_stats` `{"source":
//!    "intra_document", "pairs_checked": n}`) and leave the clean set. Each pair is added to
//!    `unique_contradiction_pairs` once.
//! 2. Cross-document, per fact still clean: retrieval with the fact's own subject / object
//!    entity ids (else the document's), the candidate entity map from `FactAboutEntity` edges,
//!    gating (verification strategy) or the pre-filter, the judge on enriched premise
//!    (retrieved fact) / hypothesis (new fact) for every passing pair. Contradiction labels at
//!    or above `contradiction_threshold` are contradictions, at or above `review_threshold`
//!    needs-review; status `contradiction` (pairs: contradictions then reviews), else
//!    `needs_review`, else `clean`. Cross-document pairs carry the shared entity ids.
//! 3. Commit: the clean facts through [`ce_facts::FactIndexer::commit_facts`] (one batch, or
//!    one call per fact with `commit_as_we_go`); committed ids are the written graph ids.
//!
//! The report carries totals (candidates, pairs after pre-filter, judge calls), the
//! processing time and a debug trace entry per pair considered (Python keys: `phase`,
//! `fact_a_id`, ..., `outcome`, `reason`, `nli_label`, `nli_confidence`).
//!
//! Judge calls are batched: all passing pairs of one fact (cross-document) and all passing
//! pairs of the document (intra-document) go to the judge in one
//! [`ContradictionJudge::judge_batch`] call each; results keep pair order. Facts are keyed by
//! their position in the input, so facts sharing an extraction id stay distinct. Graph
//! lookups of the cross-document phase (facts of an entity, entities of a candidate) are
//! memoised for the check in a [`GraphLookupCache`].
//!
//! [`ContradictionCheckService::check_facts_excluding`] additionally excludes graph facts of
//! this document that are already in the store (written by a fact stage that commits): they
//! are never retrieved as candidates, and a clean fact carrying one of them in
//! `metadata.helix_fact_id` is reported as committed without being written again.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use ce_config::pipeline::VerificationStrategyConfig;
use ce_config::services::ContradictionCheckConfig;
use ce_config::MappingsRegistry;
use ce_core::{
    ContradictionPair, ContradictionReport, ContradictionSource, ContradictionStatus, DocId,
    EntityId, ExtractedFact, FactContradictionResult, FactId, Metadata, NliLabel,
};
use ce_facts::FactIndexer;
use ce_models::Embedder;
use ce_store::GraphStore;
use indexmap::IndexMap;
use serde_json::{json, Value};

use crate::error::Result;
use crate::gating::ContradictionGating;
use crate::intra_doc::{IntraDocFilter, IntraDocOutcome};
use crate::judge::{ContradictionJudge, JudgeVerdict};
use crate::nli_context::NliContextBuilder;
use crate::prefilter::{FactPairPreFilter, ScoredCandidate};
use crate::retrieval::{GraphLookupCache, SimilarFactRetriever};

/// Retrieval source and contradiction source name of intra-document pairs.
const INTRA_DOCUMENT: &str = "intra_document";
/// Debug-trace phase of cross-document entries.
const CROSS_DOCUMENT: &str = "cross_document";

/// Counters of the intra-document phase (the Python `intra_doc_*` stats).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IntraDocStats {
    /// `n * (n - 1) / 2` for `n` facts.
    pub total_pairs: usize,
    /// Bucketed candidate pairs.
    pub candidate_pairs: usize,
    /// Pairs that passed the pre-filter.
    pub pairs_after_prefilter: usize,
    /// Judge calls.
    pub nli_calls: usize,
    /// Pairs skipped by `require_assertion_overlap`.
    pub skipped_no_assertion_overlap: usize,
    /// Pairs skipped for incompatible relationships (never incremented; kept for the Python
    /// stat key).
    pub skipped_incompatible_relationship: usize,
    /// Pairs skipped by the Jaccard gate.
    pub skipped_low_jaccard: usize,
    /// Pairs skipped because neither path held and the slots differ.
    pub skipped_incompatible_category: usize,
    /// Unique contradiction / needs-review pairs.
    pub unique_contradiction_pairs: usize,
}

impl IntraDocStats {
    /// The stats under their Python keys, in Python order.
    #[must_use]
    pub fn to_metadata(&self) -> Metadata {
        let mut map = Metadata::new();
        for (key, value) in [
            ("intra_doc_total_pairs", self.total_pairs),
            ("intra_doc_pairs_after_prefilter", self.pairs_after_prefilter),
            ("intra_doc_nli_calls", self.nli_calls),
            ("intra_doc_skipped_no_assertion_overlap", self.skipped_no_assertion_overlap),
            ("intra_doc_skipped_incompatible_relationship", self.skipped_incompatible_relationship),
            ("intra_doc_skipped_low_jaccard", self.skipped_low_jaccard),
            ("intra_doc_skipped_incompatible_category", self.skipped_incompatible_category),
            ("intra_doc_candidate_pairs", self.candidate_pairs),
            ("intra_doc_unique_contradiction_pairs", self.unique_contradiction_pairs),
        ] {
            map.insert(key.to_owned(), Value::from(value));
        }
        map
    }
}

/// Outcome of the intra-document phase.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IntraDocCheck {
    /// Indexes (into the checked facts) of the facts without an intra-document conflict, in
    /// input order.
    pub clean: Vec<usize>,
    /// One result per conflicting fact, in input order.
    pub results: Vec<FactContradictionResult>,
    /// Each conflicting pair once.
    pub unique_pairs: Vec<ContradictionPair>,
    /// Counters.
    pub stats: IntraDocStats,
}

/// Orchestrates the contradiction check for the facts of one document.
pub struct ContradictionCheckService {
    store: Arc<dyn GraphStore>,
    embedder: Arc<dyn Embedder>,
    judge: Arc<dyn ContradictionJudge>,
    config: ContradictionCheckConfig,
    verification: Option<VerificationStrategyConfig>,
    mappings: &'static MappingsRegistry,
    retriever: SimilarFactRetriever,
    prefilter: Arc<FactPairPreFilter>,
    gating: Option<ContradictionGating>,
    intra_filter: IntraDocFilter,
    context_builder: NliContextBuilder,
    indexer: FactIndexer,
}

impl fmt::Debug for ContradictionCheckService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContradictionCheckService")
            .field("store", &self.store)
            .field("embedder", &self.embedder.model_name())
            .field("judge", &self.judge)
            .field("config", &self.config)
            .field("verification", &self.verification)
            .finish_non_exhaustive()
    }
}

impl ContradictionCheckService {
    /// A service over `store`, embedding with `embedder` and judging with `judge`.
    /// `verification` selects the gating strategy (`None`: the dual-path pre-filter).
    #[must_use]
    pub fn new(
        store: Arc<dyn GraphStore>,
        embedder: Arc<dyn Embedder>,
        judge: Arc<dyn ContradictionJudge>,
        config: ContradictionCheckConfig,
        verification: Option<VerificationStrategyConfig>,
        mappings: &'static MappingsRegistry,
    ) -> Self {
        let retriever = SimilarFactRetriever::new(
            Arc::clone(&store),
            Arc::clone(&embedder),
            config.retrieval.clone(),
        );
        let prefilter = Arc::new(FactPairPreFilter::new(
            config.pre_filter.clone(),
            Some(Arc::clone(&embedder)),
            mappings,
        ));
        let gating = verification.as_ref().map(|verification| {
            ContradictionGating::new(verification.clone(), Arc::clone(&prefilter))
        });
        let intra_filter = IntraDocFilter::new(config.intra_doc_filter.clone());
        let indexer = FactIndexer::new(Arc::clone(&store), Arc::clone(&embedder));
        Self {
            store,
            embedder,
            judge,
            config,
            verification,
            mappings,
            retriever,
            prefilter,
            gating,
            intra_filter,
            context_builder: NliContextBuilder::default(),
            indexer,
        }
    }

    /// Sets the entity id -> canonical name map used to enrich judge inputs.
    #[must_use]
    pub fn with_entity_names(mut self, entity_names: IndexMap<EntityId, String>) -> Self {
        self.set_entity_names(entity_names);
        self
    }

    /// Replaces the entity id -> canonical name map.
    pub fn set_entity_names(&mut self, entity_names: IndexMap<EntityId, String>) {
        self.context_builder = NliContextBuilder::new(entity_names);
    }

    /// The service settings.
    #[must_use]
    pub fn config(&self) -> &ContradictionCheckConfig {
        &self.config
    }

    /// Checks `facts` (module docs). `entity_ids` are the document's resolved entities (the
    /// retrieval fallback for facts without linked entities).
    ///
    /// # Errors
    /// Fatal store errors and commit errors.
    pub fn check_facts_for_contradictions(
        &self,
        facts: &[ExtractedFact],
        entity_ids: &[EntityId],
        doc_id: &DocId,
    ) -> Result<ContradictionReport> {
        self.check_facts_excluding(facts, entity_ids, doc_id, &[])
    }

    /// [`Self::check_facts_for_contradictions`] that never retrieves the graph facts
    /// `exclude_fact_ids` (this document's facts already in the store) and does not write a
    /// clean fact again when its `metadata.helix_fact_id` is one of them (its id is reported
    /// in `committed_fact_ids`).
    ///
    /// # Errors
    /// Fatal store errors and commit errors.
    pub fn check_facts_excluding(
        &self,
        facts: &[ExtractedFact],
        entity_ids: &[EntityId],
        doc_id: &DocId,
        exclude_fact_ids: &[FactId],
    ) -> Result<ContradictionReport> {
        let start = Instant::now();
        let mut report = ContradictionReport::new(doc_id.clone());
        report.total_facts_extracted = facts.len();
        report.contradiction_threshold = self.config.contradiction_threshold;
        report.review_threshold = self.config.review_threshold;
        if facts.is_empty() {
            return Ok(report);
        }

        // Phase 1: intra-document.
        let clean_after_intra: Vec<usize> = if self.config.intra_document_enabled && facts.len() > 1
        {
            let intra = self.check_intra_document(facts, &mut report.debug_trace);
            report.total_nli_calls += intra.stats.nli_calls;
            tracing::info!(
                "Phase 1 (intra-doc): {} facts -> {} clean, {} contradicting, {} unique pairs",
                facts.len(),
                intra.clean.len(),
                intra.results.len(),
                intra.unique_pairs.len()
            );
            report.contradiction_results.extend(intra.results);
            report.unique_contradiction_pairs.extend(intra.unique_pairs);
            intra.clean
        } else {
            (0..facts.len()).collect()
        };

        // Phase 2: cross-document.
        // The graph is not written before phase 3, so its lookups are memoised across facts.
        let mut lookups = GraphLookupCache::default();
        let mut clean_after_cross: Vec<usize> = Vec::with_capacity(clean_after_intra.len());
        for &index in &clean_after_intra {
            let result = self.check_single_fact_cross_doc(
                &facts[index],
                entity_ids,
                exclude_fact_ids,
                &mut lookups,
                &mut report.debug_trace,
            )?;
            report.total_candidates_retrieved += stat(&result, "total_candidates");
            report.total_pairs_after_prefilter += stat(&result, "pairs_after_prefilter");
            report.total_nli_calls += stat(&result, "nli_calls");
            if result.status == ContradictionStatus::Clean {
                clean_after_cross.push(index);
            } else {
                report
                    .unique_contradiction_pairs
                    .extend(result.contradicting_pairs.iter().cloned());
                report.contradiction_results.push(result);
            }
        }
        tracing::info!(
            "Phase 2 (cross-doc): {} facts -> {} clean, {} contradicting",
            clean_after_intra.len(),
            clean_after_cross.len(),
            clean_after_intra.len() - clean_after_cross.len()
        );

        // Phase 3: commit.
        let mut clean: Vec<ExtractedFact> =
            clean_after_cross.iter().map(|&index| facts[index].clone()).collect();
        if !clean.is_empty() {
            report.committed_fact_ids = self.commit_clean(&mut clean, exclude_fact_ids)?;
            tracing::info!(
                "Phase 3 (commit): {} facts committed ({} IDs returned)",
                clean.len(),
                report.committed_fact_ids.len()
            );
        }
        report.clean_facts = clean;
        report.processing_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        tracing::info!(
            "ContradictionCheck complete: doc={}, total={}, clean={}, contradictions={}, \
             needs_review={}, time={:.1}ms",
            doc_id,
            report.total_facts_extracted,
            report.clean_facts.len(),
            report.contradiction_count(),
            report.needs_review_count(),
            report.processing_time_ms
        );
        Ok(report)
    }

    /// Commits `facts` in one batch; returns the graph ids of the facts written (each fact's
    /// `metadata.helix_fact_id` is set).
    ///
    /// # Errors
    /// Fatal store errors.
    pub fn commit_clean_facts(&self, facts: &mut [ExtractedFact]) -> Result<Vec<FactId>> {
        let result = self.indexer.commit_facts(facts, &self.mappings.taxonomy)?;
        Ok(result.fact_ids.into_iter().flatten().collect())
    }

    /// Commits the clean facts (batch, or one call per fact with `commit_as_we_go`), keeping
    /// the ids of facts already in the store; ids are returned in fact order.
    fn commit_clean(
        &self,
        facts: &mut [ExtractedFact],
        existing_fact_ids: &[FactId],
    ) -> Result<Vec<FactId>> {
        let existing: HashSet<&str> = existing_fact_ids.iter().map(FactId::as_str).collect();
        let stored_id = |fact: &ExtractedFact| -> Option<FactId> {
            fact.metadata
                .get("helix_fact_id")
                .and_then(Value::as_str)
                .filter(|id| existing.contains(id))
                .map(FactId::new)
        };
        let known: Vec<Option<FactId>> = facts.iter().map(stored_id).collect();
        let pending: Vec<usize> = (0..facts.len()).filter(|&i| known[i].is_none()).collect();

        let mut written: Vec<Option<FactId>> = vec![None; facts.len()];
        if self.config.commit_as_we_go {
            for &index in &pending {
                let ids = self.commit_clean_facts(std::slice::from_mut(&mut facts[index]))?;
                written[index] = ids.into_iter().next();
            }
        } else if !pending.is_empty() {
            let mut batch: Vec<ExtractedFact> =
                pending.iter().map(|&index| facts[index].clone()).collect();
            let result = self.indexer.commit_facts(&mut batch, &self.mappings.taxonomy)?;
            for ((&index, fact), id) in pending.iter().zip(batch).zip(result.fact_ids) {
                facts[index] = fact;
                written[index] = id;
            }
        }
        if pending.len() < facts.len() {
            tracing::info!(
                already_committed = facts.len() - pending.len(),
                "ContradictionCheckService: clean facts already in the store are not written again"
            );
        }
        Ok(known
            .into_iter()
            .zip(written)
            .filter_map(|(known, written)| known.or(written))
            .collect())
    }

    /// The intra-document phase (module docs); appends one debug-trace entry per candidate
    /// pair, in candidate order.
    #[must_use]
    pub fn check_intra_document(
        &self,
        facts: &[ExtractedFact],
        debug_trace: &mut Vec<Metadata>,
    ) -> IntraDocCheck {
        let n = facts.len();
        let mut check = IntraDocCheck::default();
        if n < 2 {
            check.clean = (0..n).collect();
            return check;
        }
        check.stats.total_pairs = n * (n - 1) / 2;
        if check.stats.total_pairs > self.config.max_intra_doc_pairs {
            tracing::warn!(
                "Intra-doc check: {} pairs exceeds limit of {}. Skipping intra-doc check.",
                check.stats.total_pairs,
                self.config.max_intra_doc_pairs
            );
            check.clean = (0..n).collect();
            return check;
        }

        let candidates = self.intra_filter.candidate_pairs(facts);
        check.stats.candidate_pairs = candidates.len();
        let min_text_overlap = self.intra_filter.config().min_text_overlap;

        // Trace entries are built in candidate order; judged entries are completed after the
        // batched judge call.
        let mut entries: Vec<Metadata> = Vec::with_capacity(candidates.len());
        let mut passing: Vec<(usize, usize, usize, f64)> = Vec::new();
        for candidate in &candidates {
            let (a, b) = (&facts[candidate.first], &facts[candidate.second]);
            let decision = self.intra_filter.evaluate_pair(a, b, &self.prefilter);
            let mut entry = Metadata::new();
            entry.insert("phase".to_owned(), json!(INTRA_DOCUMENT));
            entry.insert("fact_a_id".to_owned(), json!(a.id));
            entry.insert("fact_a_text".to_owned(), json!(a.text));
            entry.insert("fact_b_id".to_owned(), json!(b.id));
            entry.insert("fact_b_text".to_owned(), json!(b.text));
            entry.insert("path_a_valid".to_owned(), json!(decision.path_a_valid));
            entry.insert("path_b_valid".to_owned(), json!(decision.path_b_valid));
            entry.insert("slot_match".to_owned(), json!(decision.slot_match));
            match decision.outcome {
                IntraDocOutcome::SkipNoAssertionOverlap => {
                    check.stats.skipped_no_assertion_overlap += 1;
                    entry.insert("outcome".to_owned(), json!("FAIL_ASSERTION_OVERLAP"));
                    entry.insert(
                        "reason".to_owned(),
                        json!("No shared subject or object (require_assertion_overlap)"),
                    );
                }
                IntraDocOutcome::SkipIncompatibleCategory => {
                    check.stats.skipped_incompatible_category += 1;
                    entry.insert("outcome".to_owned(), json!("FAIL_PREFILTER"));
                    entry.insert(
                        "reason".to_owned(),
                        json!("Neither Path A nor Path B valid (and no slot match)"),
                    );
                }
                IntraDocOutcome::SkipLowJaccard => {
                    check.stats.skipped_low_jaccard += 1;
                    entry.insert("text_overlap".to_owned(), json!(decision.text_overlap));
                    entry.insert("outcome".to_owned(), json!("FAIL_JACCARD"));
                    entry.insert(
                        "reason".to_owned(),
                        json!(format!(
                            "Jaccard {:.3} < {}",
                            decision.text_overlap,
                            Value::from(min_text_overlap)
                        )),
                    );
                }
                IntraDocOutcome::Pass => {
                    check.stats.pairs_after_prefilter += 1;
                    entry.insert("text_overlap".to_owned(), json!(decision.text_overlap));
                    passing.push((
                        entries.len(),
                        candidate.first,
                        candidate.second,
                        decision.priority,
                    ));
                }
            }
            entries.push(entry);
        }

        let texts: Vec<(String, String)> = passing
            .iter()
            .map(|&(_, first, second, _)| {
                (
                    self.context_builder.build_enriched_text(&facts[first]),
                    self.context_builder.build_enriched_text(&facts[second]),
                )
            })
            .collect();
        let verdicts = self.judge_pairs(&texts);
        check.stats.nli_calls = passing.len();

        // Fact index -> indexes into `pairs`, in first-involvement order.
        let mut involvement: IndexMap<usize, Vec<usize>> = IndexMap::new();
        let mut pairs: Vec<ContradictionPair> = Vec::new();
        for (&(entry_index, first, second, priority), verdict) in passing.iter().zip(&verdicts) {
            let entry = &mut entries[entry_index];
            entry.insert("outcome".to_owned(), json!("NLI_RUN"));
            entry.insert("nli_label".to_owned(), json!(verdict.label.as_str()));
            entry.insert("nli_confidence".to_owned(), json!(verdict.confidence));

            if verdict.label != NliLabel::Contradiction
                || verdict.confidence < self.config.review_threshold
            {
                continue;
            }
            // Candidate pairs already put the lower fact id first.
            let mut pair = ContradictionPair::new(
                facts[first].text.clone(),
                facts[first].id.clone(),
                facts[second].id.clone(),
                facts[second].text.clone(),
                verdict.label,
                verdict.confidence,
            );
            pair.retrieval_sources = vec![INTRA_DOCUMENT.to_owned()];
            pair.priority_score = priority;
            pair.contradiction_source = ContradictionSource::IntraDocument;
            let pair_index = pairs.len();
            pairs.push(pair);
            involvement.entry(first).or_default().push(pair_index);
            involvement.entry(second).or_default().push(pair_index);
        }
        debug_trace.extend(entries);

        let mut seen = vec![false; pairs.len()];
        for indexes in involvement.values() {
            for &pair_index in indexes {
                if !std::mem::replace(&mut seen[pair_index], true) {
                    check.unique_pairs.push(pairs[pair_index].clone());
                }
            }
        }
        for (index, fact) in facts.iter().enumerate() {
            let Some(indexes) = involvement.get(&index) else {
                check.clean.push(index);
                continue;
            };
            let mut fact_pairs: Vec<ContradictionPair> = Vec::with_capacity(indexes.len());
            let mut seen_for_fact: HashSet<usize> = HashSet::new();
            for &pair_index in indexes {
                if seen_for_fact.insert(pair_index) {
                    fact_pairs.push(pairs[pair_index].clone());
                }
            }
            let status = if fact_pairs
                .iter()
                .any(|pair| pair.nli_confidence >= self.config.contradiction_threshold)
            {
                ContradictionStatus::Contradiction
            } else {
                ContradictionStatus::NeedsReview
            };
            let mut result = FactContradictionResult::new(fact.clone(), status);
            result.retrieval_stats.insert("source".to_owned(), json!(INTRA_DOCUMENT));
            result.retrieval_stats.insert("pairs_checked".to_owned(), json!(fact_pairs.len()));
            result.contradicting_pairs = fact_pairs;
            check.results.push(result);
        }
        check.stats.unique_contradiction_pairs = check.unique_pairs.len();

        let stats = &check.stats;
        tracing::info!(
            "Intra-doc check: {} facts, {} pairs, {} after prefilter (skipped: {} no assertion \
             overlap, {} incompatible rel, {} low jaccard, {} incompatible category), {} NLI \
             calls, {} contradicting facts, {} unique contradiction pairs",
            n,
            stats.total_pairs,
            stats.pairs_after_prefilter,
            stats.skipped_no_assertion_overlap,
            stats.skipped_incompatible_relationship,
            stats.skipped_low_jaccard,
            stats.skipped_incompatible_category,
            stats.nli_calls,
            involvement.len(),
            stats.unique_contradiction_pairs
        );
        check
    }

    /// The cross-document check of one fact (module docs); appends one debug-trace entry.
    fn check_single_fact_cross_doc(
        &self,
        fact: &ExtractedFact,
        document_entity_ids: &[EntityId],
        exclude_fact_ids: &[FactId],
        lookups: &mut GraphLookupCache,
        debug_trace: &mut Vec<Metadata>,
    ) -> Result<FactContradictionResult> {
        let mut per_fact: Vec<EntityId> = Vec::with_capacity(2);
        if let Some(subject) = fact.subject_entity_id.as_ref().filter(|id| !id.is_empty()) {
            per_fact.push(subject.clone());
        }
        if let Some(object) = fact.object_entity_id.as_ref().filter(|id| !id.is_empty()) {
            if !per_fact.contains(object) {
                per_fact.push(object.clone());
            }
        }
        let entity_ids: &[EntityId] =
            if per_fact.is_empty() { document_entity_ids } else { &per_fact };

        let candidates =
            self.retriever.retrieve_similar_facts(fact, entity_ids, exclude_fact_ids, lookups)?;
        let mut stats =
            CrossDocStats { total_candidates: candidates.len(), ..CrossDocStats::default() };
        if candidates.is_empty() {
            debug_trace.push(cross_doc_entry(fact, 0, Vec::new()));
            return Ok(stats.result(fact.clone(), ContradictionStatus::Clean, Vec::new()));
        }

        let candidate_entity_map = self.retriever.candidate_entity_map(&candidates, lookups)?;
        let filtered: Vec<ScoredCandidate<'_>> = match &self.gating {
            Some(gating) => {
                gating.filter_pairs(fact, &candidates, entity_ids, &candidate_entity_map)
            }
            None => {
                self.prefilter.filter_pairs(fact, &candidates, entity_ids, &candidate_entity_map)
            }
        };
        stats.pairs_after_prefilter = filtered.len();

        let hypothesis = self.context_builder.build_enriched_text(fact);
        let texts: Vec<(String, String)> = filtered
            .iter()
            .map(|(candidate, _)| {
                (
                    self.context_builder.build_enriched_text_for_retrieved(candidate),
                    hypothesis.clone(),
                )
            })
            .collect();
        let verdicts = self.judge_pairs(&texts);
        stats.nli_calls = filtered.len();

        let mut contradicting: Vec<ContradictionPair> = Vec::new();
        let mut needs_review: Vec<ContradictionPair> = Vec::new();
        let mut trace_pairs: Vec<Value> = Vec::with_capacity(candidates.len());
        for (&(candidate, priority), verdict) in filtered.iter().zip(&verdicts) {
            trace_pairs.push(json!({
                "candidate_id": candidate.fact_id,
                "candidate_text": candidate.fact_text,
                "pre_filter": "PASS",
                "nli_label": verdict.label.as_str(),
                "nli_confidence": verdict.confidence,
                "priority_score": priority,
            }));
            if verdict.label != NliLabel::Contradiction {
                continue;
            }
            let mut pair = ContradictionPair::new(
                fact.text.clone(),
                fact.id.clone(),
                candidate.fact_id.clone(),
                candidate.fact_text.clone(),
                verdict.label,
                verdict.confidence,
            );
            pair.retrieval_sources = candidate.sources.clone();
            pair.priority_score = priority;
            pair.shared_entity_ids = self.prefilter.shared_entities(
                entity_ids,
                &candidate_entity_map,
                &candidate.fact_id,
            );
            pair.contradiction_source = ContradictionSource::CrossDocument;
            if verdict.confidence >= self.config.contradiction_threshold {
                contradicting.push(pair);
            } else if verdict.confidence >= self.config.review_threshold {
                needs_review.push(pair);
            }
        }
        let passed: HashSet<&FactId> = filtered.iter().map(|(c, _)| &c.fact_id).collect();
        for candidate in candidates.iter().filter(|c| !passed.contains(&c.fact_id)) {
            trace_pairs.push(json!({
                "candidate_id": candidate.fact_id,
                "candidate_text": candidate.fact_text,
                "pre_filter": "FAIL",
                "reason": "Filtered by pre-filter (structure/overlap)",
                "nli_label": "N/A",
            }));
        }
        debug_trace.push(cross_doc_entry(fact, candidates.len(), trace_pairs));

        Ok(if !contradicting.is_empty() {
            contradicting.extend(needs_review);
            stats.result(fact.clone(), ContradictionStatus::Contradiction, contradicting)
        } else if !needs_review.is_empty() {
            stats.result(fact.clone(), ContradictionStatus::NeedsReview, needs_review)
        } else {
            stats.result(fact.clone(), ContradictionStatus::Clean, Vec::new())
        })
    }

    /// One batched judge call; a judge returning fewer verdicts than pairs pads with neutral.
    fn judge_pairs(&self, texts: &[(String, String)]) -> Vec<JudgeVerdict> {
        if texts.is_empty() {
            return Vec::new();
        }
        let pairs: Vec<(&str, &str)> = texts
            .iter()
            .map(|(premise, hypothesis)| (premise.as_str(), hypothesis.as_str()))
            .collect();
        let mut verdicts = self.judge.judge_batch(&pairs);
        if verdicts.len() != pairs.len() {
            tracing::error!(
                judge = self.judge.name(),
                pairs = pairs.len(),
                verdicts = verdicts.len(),
                "judge returned a different number of verdicts than pairs; missing ones are neutral"
            );
            verdicts.resize_with(pairs.len(), JudgeVerdict::neutral);
        }
        verdicts
    }
}

/// Retrieval statistics of one cross-document check.
#[derive(Debug, Default)]
struct CrossDocStats {
    total_candidates: usize,
    pairs_after_prefilter: usize,
    nli_calls: usize,
}

impl CrossDocStats {
    fn result(
        &self,
        fact: ExtractedFact,
        status: ContradictionStatus,
        pairs: Vec<ContradictionPair>,
    ) -> FactContradictionResult {
        let mut result = FactContradictionResult::new(fact, status);
        result.contradicting_pairs = pairs;
        for (key, value) in [
            ("total_candidates", self.total_candidates),
            ("pairs_after_prefilter", self.pairs_after_prefilter),
            ("nli_calls", self.nli_calls),
        ] {
            result.retrieval_stats.insert(key.to_owned(), Value::from(value));
        }
        result
    }
}

/// The cross-document debug-trace entry of one fact.
fn cross_doc_entry(fact: &ExtractedFact, candidates_found: usize, pairs: Vec<Value>) -> Metadata {
    let mut entry = Metadata::new();
    entry.insert("phase".to_owned(), json!(CROSS_DOCUMENT));
    entry.insert("new_fact_id".to_owned(), json!(fact.id));
    entry.insert("new_fact_text".to_owned(), json!(fact.text));
    entry.insert("candidates_found".to_owned(), json!(candidates_found));
    entry.insert("pairs".to_owned(), Value::Array(pairs));
    entry
}

/// Non-negative integer statistic `key` of a result.
fn stat(result: &FactContradictionResult, key: &str) -> usize {
    usize::try_from(result.retrieval_stat(key)).unwrap_or(0)
}
