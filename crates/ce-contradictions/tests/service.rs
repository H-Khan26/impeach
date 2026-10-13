//! `ContradictionCheckService`: intra-document statuses and thresholds, the pair cap,
//! cross-document checks against committed facts, pair ordering and dedupe, commits, the debug
//! trace, the three corrected Python bugs, and the service config defaults (the remaining
//! cases of `test_contradiction_schema.py`).

use std::sync::Arc;

use ce_config::pipeline::{
    FactRetrievalStrategy, GatingStrategy, JudgeModel, VerificationStrategyConfig,
};
use ce_config::services::{
    ContradictionCheckConfig, IntraDocFilterConfig, PreFilterConfig, SimilarFactRetrieverConfig,
};
use ce_config::MappingsRegistry;
use ce_contradictions::{ContradictionCheckService, NliJudgeAdapter};
use ce_core::{
    AssertionTriple, ContradictionReport, ContradictionSource, ContradictionStatus, DocId,
    EntityId, ExtractedFact, FactId, Metadata, NliLabel, TemporalContext, TemporalValidity,
};
use ce_facts::FactIndexer;
use ce_models::{MockEmbedder, MockNli, NliScores};
use ce_store::{EntityNode, GraphStore, MemoryStore};
use pretty_assertions::assert_eq;
use serde_json::{json, Value};

const DIM: usize = 16;

fn mappings() -> &'static MappingsRegistry {
    MappingsRegistry::load().expect("mappings load")
}

/// NLI scores whose argmax is contradiction with `confidence` (for `confidence > 1/3`).
fn contradiction(confidence: f32) -> NliScores {
    let rest = (1.0 - confidence) / 2.0;
    NliScores { contradiction: confidence, entailment: rest, neutral: rest }
}

fn fact(id: &str, text: &str, triple: (&str, &str, &str)) -> ExtractedFact {
    ExtractedFact::new(
        id,
        text,
        AssertionTriple::new(triple.0, triple.1, triple.2),
        0.9,
        TemporalValidity::default(),
    )
}

/// A numeric speed fact about Daniel Reed in the `speed` slot.
fn speed_fact(id: &str, mph: u32) -> ExtractedFact {
    let mut fact = fact(
        id,
        &format!("Daniel Reed drove at {mph} miles per hour."),
        ("Daniel Reed", "DROVE_AT_SPEED", &format!("{mph} mph")),
    );
    fact.slot_name = Some("speed".to_owned());
    fact.value_type = Some("NUMERIC".to_owned());
    fact.property_type = Some("QUANTITATIVE".to_owned());
    fact
}

/// NLI scoring `confidence` for the pair of texts (either direction), neutral otherwise.
fn nli_for(pairs: &[(&str, &str, f32)]) -> Arc<MockNli> {
    let pairs: Vec<(String, String, f32)> =
        pairs.iter().map(|(a, b, c)| ((*a).to_owned(), (*b).to_owned(), *c)).collect();
    Arc::new(MockNli::new().with_rule(move |premise, hypothesis| {
        pairs.iter().find_map(|(a, b, confidence)| {
            let matches = (premise.contains(a.as_str()) && hypothesis.contains(b.as_str()))
                || (premise.contains(b.as_str()) && hypothesis.contains(a.as_str()));
            matches.then(|| contradiction(*confidence))
        })
    }))
}

struct Env {
    store: Arc<MemoryStore>,
    embedder: Arc<MockEmbedder>,
}

impl Env {
    fn new() -> Self {
        Self { store: Arc::new(MemoryStore::new(DIM)), embedder: Arc::new(MockEmbedder::new(DIM)) }
    }

    fn service(
        &self,
        nli: &Arc<MockNli>,
        config: ContradictionCheckConfig,
        verification: Option<VerificationStrategyConfig>,
    ) -> ContradictionCheckService {
        let store: Arc<dyn GraphStore> = self.store.clone();
        ContradictionCheckService::new(
            store,
            self.embedder.clone(),
            Arc::new(NliJudgeAdapter::new(nli.clone())),
            config,
            verification,
            mappings(),
        )
    }

    fn entity(&self, name: &str) -> EntityId {
        self.store.create_entity(&EntityNode::new(name, "PERSON")).expect("entity").id
    }

    /// Commits `facts` as an earlier document would; returns their graph ids.
    fn commit(&self, facts: &mut [ExtractedFact]) -> Vec<FactId> {
        let store: Arc<dyn GraphStore> = self.store.clone();
        let indexer = FactIndexer::new(store, self.embedder.clone());
        let result = indexer.commit_facts(facts, &mappings().taxonomy).expect("commit");
        result.fact_ids.into_iter().flatten().collect()
    }

    fn fact_count(&self) -> usize {
        (0..1000_u64)
            .filter_map(|id| self.store.node_snapshot(&id.to_string()))
            .filter(|(label, _)| *label == ce_store::graph::FACT)
            .count()
    }
}

fn check(service: &ContradictionCheckService, facts: &[ExtractedFact]) -> ContradictionReport {
    service
        .check_facts_for_contradictions(facts, &[], &DocId::new("doc-2"))
        .expect("check succeeds")
}

fn intra_entries(report: &ContradictionReport) -> Vec<&Metadata> {
    report.debug_trace.iter().filter(|e| e["phase"] == "intra_document").collect()
}

fn cross_entries(report: &ContradictionReport) -> Vec<&Metadata> {
    report.debug_trace.iter().filter(|e| e["phase"] == "cross_document").collect()
}

fn no_intra() -> ContradictionCheckConfig {
    ContradictionCheckConfig {
        intra_document_enabled: false,
        ..ContradictionCheckConfig::default()
    }
}

// --- intra-document ---

#[test]
fn empty_input_gives_an_empty_report() {
    let env = Env::new();
    let service = env.service(&Arc::new(MockNli::new()), ContradictionCheckConfig::default(), None);
    let report = check(&service, &[]);
    assert_eq!(report.doc_id, "doc-2");
    assert_eq!(report.total_facts_extracted, 0);
    assert!(report.debug_trace.is_empty() && report.committed_fact_ids.is_empty());
    assert!((report.contradiction_threshold - 0.85).abs() < f64::EPSILON);
}

#[test]
fn intra_document_contradiction_flags_both_facts() {
    let env = Env::new();
    let facts = [speed_fact("STATED_FACT_1", 45), speed_fact("STATED_FACT_0", 30)];
    let nli = nli_for(&[("45 miles", "30 miles", 0.95)]);
    let report = check(&env.service(&nli, ContradictionCheckConfig::default(), None), &facts);

    assert_eq!(report.unique_contradiction_pairs.len(), 1);
    let pair = &report.unique_contradiction_pairs[0];
    // The lower id is the "new" fact.
    assert_eq!(pair.new_fact_id, "STATED_FACT_0");
    assert_eq!(pair.existing_fact_id, "STATED_FACT_1");
    assert_eq!(pair.nli_label, NliLabel::Contradiction);
    assert!((pair.nli_confidence - 0.95).abs() < 1e-6);
    assert_eq!(pair.retrieval_sources, ["intra_document"]);
    assert_eq!(pair.contradiction_source, ContradictionSource::IntraDocument);
    assert!(pair.shared_entity_ids.is_empty());

    assert_eq!(report.contradiction_results.len(), 2);
    for (result, id) in report.contradiction_results.iter().zip(["STATED_FACT_1", "STATED_FACT_0"])
    {
        assert_eq!(result.new_fact.id, id);
        assert_eq!(result.status, ContradictionStatus::Contradiction);
        assert_eq!(result.contradicting_pairs, std::slice::from_ref(pair));
        assert_eq!(
            Value::Object(result.retrieval_stats.clone()),
            json!({"source": "intra_document", "pairs_checked": 1})
        );
    }
    assert!(report.clean_facts.is_empty() && report.committed_fact_ids.is_empty());
    assert_eq!(report.total_nli_calls, 1);
    assert_eq!(report.contradiction_count(), 1);
    assert_eq!(env.fact_count(), 0, "contradicting facts are not committed");
    assert!(cross_entries(&report).is_empty(), "no clean fact left for the cross-doc phase");

    let entries = intra_entries(&report);
    assert_eq!(entries.len(), 1);
    let keys: Vec<&str> = entries[0].keys().map(String::as_str).collect();
    assert_eq!(
        keys,
        [
            "phase",
            "fact_a_id",
            "fact_a_text",
            "fact_b_id",
            "fact_b_text",
            "path_a_valid",
            "path_b_valid",
            "slot_match",
            "text_overlap",
            "outcome",
            "nli_label",
            "nli_confidence"
        ]
    );
    assert_eq!(entries[0]["outcome"], "NLI_RUN");
    assert_eq!(entries[0]["nli_label"], "contradiction");
    assert_eq!(entries[0]["slot_match"], true);
}

#[test]
fn intra_document_needs_review_between_thresholds() {
    let env = Env::new();
    let facts = [speed_fact("STATED_FACT_0", 45), speed_fact("STATED_FACT_1", 30)];
    let nli = nli_for(&[("45 miles", "30 miles", 0.7)]);
    let report = check(&env.service(&nli, ContradictionCheckConfig::default(), None), &facts);
    assert_eq!(report.contradiction_results.len(), 2);
    assert!(report
        .contradiction_results
        .iter()
        .all(|r| r.status == ContradictionStatus::NeedsReview));
    assert_eq!((report.contradiction_count(), report.needs_review_count()), (0, 1));
    assert!(report.clean_facts.is_empty());
}

#[test]
fn intra_document_below_review_threshold_is_clean_and_committed() {
    let env = Env::new();
    let facts = [speed_fact("STATED_FACT_0", 45), speed_fact("STATED_FACT_1", 30)];
    let nli = nli_for(&[("45 miles", "30 miles", 0.5)]);
    let report = check(&env.service(&nli, ContradictionCheckConfig::default(), None), &facts);
    assert!(
        report.contradiction_results.is_empty() && report.unique_contradiction_pairs.is_empty()
    );
    assert_eq!(report.clean_facts.len(), 2);
    assert_eq!(report.committed_fact_ids.len(), 2);
    assert_eq!(env.fact_count(), 2);
    // Committed facts carry their graph id; ids follow fact order.
    for (fact, id) in report.clean_facts.iter().zip(&report.committed_fact_ids) {
        assert_eq!(fact.metadata["helix_fact_id"], json!(id.as_str()));
    }
    assert_eq!(intra_entries(&report)[0]["outcome"], "NLI_RUN");
}

#[test]
fn non_contradiction_labels_are_ignored() {
    let env = Env::new();
    let facts = [speed_fact("STATED_FACT_0", 45), speed_fact("STATED_FACT_1", 30)];
    let nli = Arc::new(MockNli::new().with_default(NliScores {
        contradiction: 0.3,
        entailment: 0.6,
        neutral: 0.1,
    }));
    let report = check(&env.service(&nli, ContradictionCheckConfig::default(), None), &facts);
    assert!(report.contradiction_results.is_empty());
    assert_eq!(intra_entries(&report)[0]["nli_label"], "entailment");
}

#[test]
fn max_intra_doc_pairs_skips_the_intra_document_phase() {
    let env = Env::new();
    let facts = [speed_fact("STATED_FACT_0", 45), speed_fact("STATED_FACT_1", 30)];
    let nli = nli_for(&[("45 miles", "30 miles", 0.95)]);
    let config =
        ContradictionCheckConfig { max_intra_doc_pairs: 0, ..ContradictionCheckConfig::default() };
    let service = env.service(&nli, config, None);

    let mut trace = Vec::new();
    let intra = service.check_intra_document(&facts, &mut trace);
    assert_eq!(intra.clean, [0, 1]);
    assert_eq!(intra.stats.total_pairs, 1);
    assert_eq!(intra.stats.candidate_pairs, 0);
    assert!(trace.is_empty());

    let report = check(&service, &facts);
    assert!(intra_entries(&report).is_empty());
    assert!(report.contradiction_results.is_empty());
    assert!(nli.calls().is_empty());
    assert_eq!(report.clean_facts.len(), 2);
}

#[test]
fn disabled_intra_document_check_runs_cross_document_only() {
    let env = Env::new();
    let facts = [speed_fact("STATED_FACT_0", 45), speed_fact("STATED_FACT_1", 30)];
    let nli = nli_for(&[("45 miles", "30 miles", 0.95)]);
    let report = check(&env.service(&nli, no_intra(), None), &facts);
    assert!(intra_entries(&report).is_empty());
    assert_eq!(cross_entries(&report).len(), 2);
    assert_eq!(cross_entries(&report)[0]["candidates_found"], 0);
    assert_eq!(cross_entries(&report)[0]["pairs"], json!([]));
    assert_eq!(report.clean_facts.len(), 2);
}

#[test]
fn intra_document_pairs_are_deduplicated() {
    let env = Env::new();
    let facts = [
        speed_fact("STATED_FACT_0", 45),
        speed_fact("STATED_FACT_1", 30),
        speed_fact("STATED_FACT_2", 60),
    ];
    let nli = nli_for(&[
        ("45 miles", "30 miles", 0.95),
        ("45 miles", "60 miles", 0.9),
        ("30 miles", "60 miles", 0.7),
    ]);
    let service = env.service(&nli, ContradictionCheckConfig::default(), None);
    let mut trace = Vec::new();
    let intra = service.check_intra_document(&facts, &mut trace);
    assert_eq!(intra.stats.total_pairs, 3);
    assert_eq!(intra.stats.candidate_pairs, 3);
    assert_eq!(intra.stats.pairs_after_prefilter, 3);
    assert_eq!(intra.stats.nli_calls, 3);
    assert_eq!(intra.stats.unique_contradiction_pairs, 3);
    let stats = intra.stats.to_metadata();
    assert_eq!(stats["intra_doc_nli_calls"], 3);
    assert_eq!(stats["intra_doc_skipped_low_jaccard"], 0);
    assert!(intra.clean.is_empty());
    let ids: Vec<(&str, &str)> = intra
        .unique_pairs
        .iter()
        .map(|p| (p.new_fact_id.as_str(), p.existing_fact_id.as_str()))
        .collect();
    assert_eq!(
        ids,
        [
            ("STATED_FACT_0", "STATED_FACT_1"),
            ("STATED_FACT_0", "STATED_FACT_2"),
            ("STATED_FACT_1", "STATED_FACT_2")
        ]
    );
    let statuses: Vec<ContradictionStatus> = intra.results.iter().map(|r| r.status).collect();
    assert_eq!(
        statuses,
        [
            ContradictionStatus::Contradiction,
            ContradictionStatus::Contradiction,
            ContradictionStatus::Contradiction
        ]
    );
    for result in &intra.results {
        assert_eq!(result.contradicting_pairs.len(), 2);
        assert_eq!(result.retrieval_stats["pairs_checked"], 2);
    }

    let report = check(&service, &facts);
    assert_eq!(report.unique_contradiction_pairs.len(), 3);
    assert_eq!((report.contradiction_count(), report.needs_review_count()), (2, 1));
    // One batched judge call per phase; every pair judged once.
    assert_eq!(nli.calls().len(), 6);
}

#[test]
fn facts_sharing_an_extraction_id_stay_distinct() {
    let env = Env::new();
    // Per-chunk extraction ids repeat across chunks.
    let facts = [speed_fact("STATED_FACT_0", 45), speed_fact("STATED_FACT_0", 30)];
    let nli = nli_for(&[("45 miles", "30 miles", 0.95)]);
    let report = check(&env.service(&nli, ContradictionCheckConfig::default(), None), &facts);
    assert_eq!(report.contradiction_results.len(), 2);
    assert_eq!(report.unique_contradiction_pairs.len(), 1);
    assert!(report.clean_facts.is_empty());
}

// --- corrected bugs: intra-document ---

#[test]
fn temporal_context_mismatch_is_judged_with_lower_priority() {
    let env = Env::new();
    let mut prior = speed_fact("STATED_FACT_1", 30);
    prior.temporal_context = TemporalContext::PriorIncident;
    let facts = [speed_fact("STATED_FACT_0", 45), prior];
    let nli = nli_for(&[("45 miles", "30 miles", 0.95)]);
    let report = check(&env.service(&nli, ContradictionCheckConfig::default(), None), &facts);

    // Python skipped every pair whose temporal_context differed.
    assert_eq!(nli.calls().len(), 1);
    assert_eq!(report.unique_contradiction_pairs.len(), 1);
    let entry = intra_entries(&report)[0];
    let overlap = entry["text_overlap"].as_f64().expect("overlap");
    let priority = report.unique_contradiction_pairs[0].priority_score;
    assert!(overlap > 0.0);
    assert!(
        (priority - overlap * ce_contradictions::TEMPORAL_MISMATCH_PRIORITY_FACTOR).abs() < 1e-9
    );
}

#[test]
fn require_assertion_overlap_skips_pairs_about_different_subjects() {
    let env = Env::new();
    let reed = speed_fact("STATED_FACT_0", 45);
    let mut lopez = speed_fact("STATED_FACT_1", 30);
    lopez.text = "Maria Lopez drove at 30 miles per hour.".to_owned();
    lopez.assertion_triple = AssertionTriple::new("Maria Lopez", "DROVE_AT_SPEED", "30 mph");
    let facts = [reed, lopez];
    let nli = nli_for(&[("45 miles", "30 miles", 0.95)]);

    let report = check(&env.service(&nli, ContradictionCheckConfig::default(), None), &facts);
    let entry = intra_entries(&report)[0];
    assert_eq!(entry["outcome"], "FAIL_ASSERTION_OVERLAP");
    assert!(entry.contains_key("reason"));
    assert!(report.contradiction_results.is_empty());
    let service = env.service(&nli, ContradictionCheckConfig::default(), None);
    let intra = service.check_intra_document(&facts, &mut Vec::new());
    assert_eq!(intra.stats.skipped_no_assertion_overlap, 1);

    // With the key off the same-slot pair reaches the judge (Python ignored the key).
    let env = Env::new();
    let config = ContradictionCheckConfig {
        intra_doc_filter: IntraDocFilterConfig {
            require_assertion_overlap: false,
            ..IntraDocFilterConfig::default()
        },
        ..ContradictionCheckConfig::default()
    };
    let report = check(&env.service(&nli, config, None), &facts);
    assert_eq!(intra_entries(&report)[0]["outcome"], "NLI_RUN");
    assert_eq!(report.unique_contradiction_pairs.len(), 1);
}

#[test]
fn unrelated_pairs_fail_the_prefilter_or_jaccard_gate() {
    let env = Env::new();
    let config = ContradictionCheckConfig {
        intra_doc_filter: IntraDocFilterConfig {
            require_assertion_overlap: false,
            ..IntraDocFilterConfig::default()
        },
        ..ContradictionCheckConfig::default()
    };
    // Same structural bucket, no slots, different subjects and predicates.
    let mut a = fact("STATED_FACT_0", "The truck was red.", ("truck", "HAS_COLOR", "red"));
    let mut b = fact("STATED_FACT_1", "The road was wet.", ("road", "HAS_STATE", "wet"));
    for f in [&mut a, &mut b] {
        f.value_type = Some("TEXT".to_owned());
        f.property_type = Some("ATTRIBUTE".to_owned());
    }
    let nli = Arc::new(MockNli::new());
    let report = check(&env.service(&nli, config.clone(), None), &[a.clone(), b]);
    let entry = intra_entries(&report)[0];
    assert_eq!(entry["outcome"], "FAIL_PREFILTER");
    assert_eq!(entry["reason"], "Neither Path A nor Path B valid (and no slot match)");
    assert!(!entry.contains_key("text_overlap"));

    // Path A only (same subject and predicate) with little text overlap: the Jaccard gate.
    let mut c = fact(
        "STATED_FACT_2",
        "Its paint looked crimson under streetlights yesterday.",
        ("truck", "HAS_COLOR", "crimson"),
    );
    c.value_type = Some("TEXT".to_owned());
    c.property_type = Some("ATTRIBUTE".to_owned());
    let report = check(&env.service(&nli, config, None), &[a, c]);
    let entry = intra_entries(&report)[0];
    assert_eq!(entry["outcome"], "FAIL_JACCARD");
    assert_eq!(entry["path_a_valid"], true);
    assert!(entry["reason"].as_str().is_some_and(|r| r.ends_with("< 0.15")), "{entry:?}");
}

// --- cross-document ---

/// Commits `Daniel Reed drove the Honda Civic.` linked to Reed; returns (Reed, fact id).
fn earlier_document(env: &Env) -> (EntityId, FactId) {
    let reed = env.entity("Daniel Reed");
    let mut existing = fact(
        "STATED_FACT_0",
        "Daniel Reed drove the Honda Civic.",
        ("Daniel Reed", "DROVE", "Honda Civic"),
    );
    existing.subject_entity_id = Some(reed.clone());
    let ids = env.commit(std::slice::from_mut(&mut existing));
    (reed, ids[0].clone())
}

fn new_civic_fact(reed: &EntityId) -> ExtractedFact {
    let mut new = fact(
        "STATED_FACT_0",
        "Daniel Reed never drove the Honda Civic.",
        ("Daniel Reed", "DROVE", "Honda Civic"),
    );
    new.subject_entity_id = Some(reed.clone());
    new
}

#[test]
fn cross_document_contradiction_against_committed_facts() {
    let env = Env::new();
    let (reed, existing_id) = earlier_document(&env);
    let new = new_civic_fact(&reed);
    let nli = nli_for(&[("drove the Honda", "never drove", 0.92)]);
    let report = check(&env.service(&nli, no_intra(), None), std::slice::from_ref(&new));

    assert_eq!(report.contradiction_results.len(), 1);
    let result = &report.contradiction_results[0];
    assert_eq!(result.status, ContradictionStatus::Contradiction);
    let pair = &result.contradicting_pairs[0];
    assert_eq!(pair.new_fact_id, "STATED_FACT_0");
    assert_eq!(pair.existing_fact_id, existing_id);
    assert_eq!(pair.existing_fact_text, "Daniel Reed drove the Honda Civic.");
    assert_eq!(pair.contradiction_source, ContradictionSource::CrossDocument);
    assert!(pair.retrieval_sources.contains(&"graph".to_owned()), "{:?}", pair.retrieval_sources);
    assert_eq!(pair.shared_entity_ids, [reed]);
    assert_eq!(report.unique_contradiction_pairs, result.contradicting_pairs);
    assert_eq!(result.retrieval_stat("nli_calls"), 1);
    assert_eq!(result.retrieval_stat("pairs_after_prefilter"), 1);
    assert!(result.retrieval_stat("total_candidates") >= 1);
    assert_eq!(report.total_nli_calls, 1);
    assert_eq!(report.total_pairs_after_prefilter, 1);
    assert!(report.committed_fact_ids.is_empty());
    assert_eq!(env.fact_count(), 1, "the contradicting fact is not committed");

    // The judge saw the retrieved fact as premise and the new fact as hypothesis.
    let calls = nli.calls();
    assert!(calls[0].0.contains("drove the Honda Civic") && calls[0].1.contains("never drove"));

    let entry = cross_entries(&report)[0];
    assert_eq!(entry["new_fact_id"], "STATED_FACT_0");
    let pairs = entry["pairs"].as_array().expect("pairs");
    assert_eq!(pairs[0]["pre_filter"], "PASS");
    assert_eq!(pairs[0]["candidate_id"], json!(existing_id.as_str()));
    assert_eq!(pairs[0]["nli_label"], "contradiction");
}

#[test]
fn cross_document_statuses_and_pair_order() {
    let env = Env::new();
    let reed = env.entity("Daniel Reed");
    let texts = [
        "Daniel Reed drove the Honda Civic in March.",
        "Daniel Reed drove the Honda Civic in April.",
        "Daniel Reed drove the Honda Civic in May.",
    ];
    let mut existing: Vec<ExtractedFact> = texts
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let mut f =
                fact(&format!("STATED_FACT_{i}"), text, ("Daniel Reed", "DROVE", "Honda Civic"));
            f.subject_entity_id = Some(reed.clone());
            f
        })
        .collect();
    env.commit(&mut existing);
    let new = new_civic_fact(&reed);
    // March: review, April: contradiction, May: below review.
    let nli = nli_for(&[
        ("in March", "never", 0.7),
        ("in April", "never", 0.95),
        ("in May", "never", 0.4),
    ]);
    let report = check(&env.service(&nli, no_intra(), None), std::slice::from_ref(&new));
    let result = &report.contradiction_results[0];
    assert_eq!(result.status, ContradictionStatus::Contradiction);
    let order: Vec<&str> =
        result.contradicting_pairs.iter().map(|p| p.existing_fact_text.as_str()).collect();
    // Contradictions first, then needs-review pairs.
    assert_eq!(order, [texts[1], texts[0]]);
    assert_eq!(result.retrieval_stat("nli_calls"), 3);
    assert_eq!((report.contradiction_count(), report.needs_review_count()), (1, 1));

    // Only a needs-review pair: status needs_review.
    let nli = nli_for(&[("in March", "never", 0.7)]);
    let report = check(&env.service(&nli, no_intra(), None), std::slice::from_ref(&new));
    assert_eq!(report.contradiction_results[0].status, ContradictionStatus::NeedsReview);
    assert_eq!(report.contradiction_results[0].contradicting_pairs.len(), 1);

    // Nothing above the review threshold: clean and committed.
    let nli = nli_for(&[("in May", "never", 0.4)]);
    let report = check(&env.service(&nli, no_intra(), None), std::slice::from_ref(&new));
    assert!(report.contradiction_results.is_empty());
    assert_eq!(report.committed_fact_ids.len(), 1);
}

#[test]
fn prefilter_failures_are_traced() {
    let env = Env::new();
    let (reed, _) = earlier_document(&env);
    // Linked to Reed but about a different object: Path B fails, no fallback is enabled.
    let mut new =
        fact("STATED_FACT_0", "Daniel Reed owns a bicycle.", ("Daniel Reed", "OWNS", "bicycle"));
    new.subject_entity_id = Some(reed);
    let nli = Arc::new(MockNli::new());
    let report = check(&env.service(&nli, no_intra(), None), &[new]);
    let entry = cross_entries(&report)[0];
    let pairs = entry["pairs"].as_array().expect("pairs");
    assert!(!pairs.is_empty());
    assert!(pairs.iter().all(|p| p["pre_filter"] == "FAIL" && p["nli_label"] == "N/A"));
    assert_eq!(pairs[0]["reason"], "Filtered by pre-filter (structure/overlap)");
    assert!(nli.calls().is_empty());
    assert_eq!(report.committed_fact_ids.len(), 1);
}

#[test]
fn commit_as_we_go_commits_each_clean_fact() {
    for commit_as_we_go in [false, true] {
        let env = Env::new();
        let facts = [speed_fact("STATED_FACT_0", 45), speed_fact("STATED_FACT_1", 30)];
        let config = ContradictionCheckConfig { commit_as_we_go, ..no_intra() };
        let report = check(&env.service(&Arc::new(MockNli::new()), config, None), &facts);
        assert_eq!(report.committed_fact_ids.len(), 2, "commit_as_we_go={commit_as_we_go}");
        assert_eq!(env.fact_count(), 2);
        let ids: Vec<&str> = report
            .clean_facts
            .iter()
            .map(|f| f.metadata["helix_fact_id"].as_str().expect("helix id"))
            .collect();
        let committed: Vec<&str> = report.committed_fact_ids.iter().map(FactId::as_str).collect();
        assert_eq!(ids, committed);
    }
}

#[test]
fn commit_clean_facts_returns_written_ids() {
    let env = Env::new();
    let service = env.service(&Arc::new(MockNli::new()), ContradictionCheckConfig::default(), None);
    let mut facts = vec![speed_fact("STATED_FACT_0", 45)];
    let ids = service.commit_clean_facts(&mut facts).expect("commit");
    assert_eq!(ids.len(), 1);
    assert_eq!(facts[0].metadata["helix_fact_id"], json!(ids[0].as_str()));
    assert!(service.commit_clean_facts(&mut []).expect("empty commit").is_empty());
}

#[test]
fn excluded_facts_are_not_retrieved_or_committed_twice() {
    let env = Env::new();
    let reed = env.entity("Daniel Reed");
    // The fact stage already committed this document's fact.
    let mut own = new_civic_fact(&reed);
    let ids = env.commit(std::slice::from_mut(&mut own));
    // A judge that calls every pair a contradiction (identical texts included).
    let nli = Arc::new(MockNli::new().with_default(contradiction(0.99)));
    let service = env.service(&nli, no_intra(), None);

    let report = service
        .check_facts_excluding(std::slice::from_ref(&own), &[], &DocId::new("doc-2"), &ids)
        .expect("check");
    assert!(report.contradiction_results.is_empty(), "the fact is not compared with itself");
    assert_eq!(cross_entries(&report)[0]["candidates_found"], 0);
    assert_eq!(report.committed_fact_ids, ids);
    assert_eq!(env.fact_count(), 1, "not written twice");

    // Without the exclusion the fact meets its own stored copy (the Python behaviour).
    let report = check(&service, std::slice::from_ref(&own));
    assert_eq!(report.contradiction_results.len(), 1);
}

// --- corrected bugs: gating ---

fn permissive() -> VerificationStrategyConfig {
    VerificationStrategyConfig {
        judge_model: JudgeModel::Nli,
        gating_strategy: GatingStrategy::Permissive,
        allow_text_value_type_mismatch: true,
        require_entity_id_match: false,
        semantic_proximity_threshold: 0.6,
        keyword_overlap_threshold: 0.3,
        qwen_prompt_file: None,
        qwen_batch_size: 5,
    }
}

#[test]
fn permissive_shared_entity_trigger_fires_through_fact_entity_edges() {
    let nli = nli_for(&[("ran the red light", "signal was green", 0.9)]);
    let new_fact = |reed: &EntityId| {
        let mut new = fact(
            "STATED_FACT_0",
            "The signal was green for the truck driver near Reed.",
            ("signal", "WAS", "green"),
        );
        new.subject_entity_id = Some(reed.clone());
        new
    };

    // The earlier fact is linked to Reed: the shared-entity trigger lets the pair through even
    // though keywords and (mock) embeddings do not overlap.
    let env = Env::new();
    let reed = env.entity("Daniel Reed");
    let mut existing =
        fact("STATED_FACT_0", "Reed ran the red light.", ("Reed", "RAN", "red light"));
    existing.subject_entity_id = Some(reed.clone());
    env.commit(std::slice::from_mut(&mut existing));
    let report = check(&env.service(&nli, no_intra(), Some(permissive())), &[new_fact(&reed)]);
    assert_eq!(report.contradiction_results.len(), 1);
    let pair = &report.contradiction_results[0].contradicting_pairs[0];
    assert_eq!(pair.shared_entity_ids, [reed]);
    assert_eq!(pair.existing_fact_text, "Reed ran the red light.");

    // The same fact without the FactAboutEntity edge (found by BM25 / vector only) is gated
    // out: no trigger fires, as for every pair when Python passed an empty map.
    let env = Env::new();
    let reed = env.entity("Daniel Reed");
    let mut unlinked =
        fact("STATED_FACT_0", "Reed ran the red light.", ("Reed", "RAN", "red light"));
    env.commit(std::slice::from_mut(&mut unlinked));
    let calls_before = nli.calls().len();
    let report = check(&env.service(&nli, no_intra(), Some(permissive())), &[new_fact(&reed)]);
    assert!(report.total_candidates_retrieved >= 1);
    assert_eq!(report.total_pairs_after_prefilter, 0);
    assert!(report.contradiction_results.is_empty());
    assert_eq!(nli.calls().len(), calls_before);
}

#[test]
fn dual_path_verification_delegates_to_the_prefilter() {
    let env = Env::new();
    let (reed, _) = earlier_document(&env);
    let mut dual = permissive();
    dual.gating_strategy = GatingStrategy::DualPath;
    let nli = nli_for(&[("drove the Honda", "never drove", 0.92)]);
    let report = check(&env.service(&nli, no_intra(), Some(dual)), &[new_civic_fact(&reed)]);
    assert_eq!(report.contradiction_results.len(), 1);
}

// --- config defaults (test_contradiction_schema.py) ---

#[test]
fn retriever_config_defaults() {
    let config = SimilarFactRetrieverConfig::default();
    assert_eq!((config.bm25_limit, config.vector_limit, config.max_candidates), (8, 8, 25));
    assert!((config.graph_weight - 1.0).abs() < f64::EPSILON);
    assert!((config.bm25_weight - 0.9).abs() < f64::EPSILON);
    assert!((config.vector_weight - 0.8).abs() < f64::EPSILON);
    assert!((config.multi_source_boost - 1.2).abs() < f64::EPSILON);
    assert!(config.min_score.abs() < f64::EPSILON);
    assert_eq!(config.fact_retrieval_strategy, FactRetrievalStrategy::Both);

    let custom = SimilarFactRetrieverConfig {
        bm25_limit: 10,
        vector_limit: 15,
        max_candidates: 50,
        graph_weight: 1.5,
        bm25_weight: 1.0,
        vector_weight: 0.5,
        multi_source_boost: 2.0,
        fact_retrieval_strategy: FactRetrievalStrategy::Vector,
        ..SimilarFactRetrieverConfig::default()
    };
    assert_eq!(custom.max_candidates, 50);
    assert_eq!(custom.fact_retrieval_strategy, FactRetrievalStrategy::Vector);
}

#[test]
fn prefilter_config_defaults() {
    let config = PreFilterConfig::default();
    assert!((config.min_text_overlap - 0.05).abs() < f64::EPSILON);
    assert!(!config.require_entity_overlap);
    assert!(!config.enable_text_overlap_fallback);
    assert!(!config.use_semantic_similarity_fallback);
    assert!((config.subject_overlap_threshold - 0.5).abs() < f64::EPSILON);
    assert!((config.predicate_similarity_threshold - 0.7).abs() < f64::EPSILON);
    assert!((config.fuzzy_match_threshold - 0.6).abs() < f64::EPSILON);
    assert!(config.use_assertion_boost);
}

#[test]
fn intra_doc_filter_config_defaults() {
    let config = IntraDocFilterConfig::default();
    assert!((config.min_text_overlap - 0.15).abs() < f64::EPSILON);
    assert!(config.require_assertion_overlap);
    assert!(!config.temporal_relationships.is_empty());
    assert!(!config.quantitative_relationships.is_empty());
    assert!(!config.role_relationships.is_empty());
    for term in ["hired", "founded", "started"] {
        assert!(config.temporal_relationships.iter().any(|t| t == term), "{term}");
    }
}

#[test]
fn contradiction_check_config_defaults() {
    let config = ContradictionCheckConfig::default();
    assert!((config.contradiction_threshold - 0.85).abs() < f64::EPSILON);
    assert!((config.review_threshold - 0.6).abs() < f64::EPSILON);
    assert!(config.review_threshold < config.contradiction_threshold);
    assert!(config.intra_document_enabled);
    assert!(!config.commit_as_we_go);
    assert_eq!(config.max_intra_doc_pairs, 2000);
    assert_eq!(config.retrieval.bm25_limit, 8);
    assert!((config.retrieval.graph_weight - 1.0).abs() < f64::EPSILON);
    assert!((config.pre_filter.min_text_overlap - 0.05).abs() < f64::EPSILON);
    assert!((config.intra_doc_filter.min_text_overlap - 0.15).abs() < f64::EPSILON);
    assert!(config.intra_doc_filter.require_assertion_overlap);
}

#[test]
fn service_reports_the_configured_thresholds() {
    let env = Env::new();
    let config = ContradictionCheckConfig {
        contradiction_threshold: 0.9,
        review_threshold: 0.7,
        ..ContradictionCheckConfig::default()
    };
    let service = env.service(&Arc::new(MockNli::new()), config, None);
    assert!((service.config().contradiction_threshold - 0.9).abs() < f64::EPSILON);
    let report = check(&service, &[speed_fact("STATED_FACT_0", 45)]);
    assert!((report.contradiction_threshold - 0.9).abs() < f64::EPSILON);
    assert!((report.review_threshold - 0.7).abs() < f64::EPSILON);
    assert_eq!(report.total_facts_extracted, 1);
}
