//! Rule resolvers against a scripted retriever: unique, ambiguous and missing candidates for
//! each route, with their confidences and strategy names.

use std::sync::Mutex;

use ce_config::MappingsRegistry;
use ce_core::{DocContext, EdCandidate, EdContext, EdResult, LocalContext, Mention, Span};
use ce_ed::retrieval::{EdRetrieval, StructuredKeyQuery};
use ce_ed::structured_keys::extract_statute_key;
use ce_ed::{
    resolve_case_rule, resolve_exhibit_rule, resolve_generic_rule, resolve_statute_rule,
    KeyPatterns, Result,
};
use pretty_assertions::assert_eq;
use serde_json::json;

type StructuredFn = dyn Fn(&StructuredKeyQuery<'_>) -> Vec<EdCandidate> + Send + Sync;
type ExactFn = dyn Fn(&str, Option<&str>) -> Vec<EdCandidate> + Send + Sync;

/// Retriever answering from closures and recording every call.
struct FakeRetriever {
    structured: Box<StructuredFn>,
    exact: Box<ExactFn>,
    calls: Mutex<Vec<String>>,
}

impl std::fmt::Debug for FakeRetriever {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeRetriever").finish_non_exhaustive()
    }
}

impl FakeRetriever {
    fn new(
        structured: impl Fn(&StructuredKeyQuery<'_>) -> Vec<EdCandidate> + Send + Sync + 'static,
        exact: impl Fn(&str, Option<&str>) -> Vec<EdCandidate> + Send + Sync + 'static,
    ) -> Self {
        Self { structured: Box::new(structured), exact: Box::new(exact), calls: Mutex::new(vec![]) }
    }

    fn empty() -> Self {
        Self::new(|_| vec![], |_, _| vec![])
    }

    fn calls(&self) -> Vec<String> {
        self.calls.lock().expect("calls lock").clone()
    }

    fn record(&self, call: String) {
        self.calls.lock().expect("calls lock").push(call);
    }
}

impl EdRetrieval for FakeRetriever {
    fn find_by_exact_match(
        &self,
        surface_form: &str,
        entity_type: Option<&str>,
    ) -> Result<Vec<EdCandidate>> {
        self.record(format!("exact {surface_form} {entity_type:?}"));
        Ok((self.exact)(surface_form, entity_type))
    }

    fn find_by_embedding(&self, _: &str, _: usize, _: f64) -> Result<Vec<EdCandidate>> {
        unreachable!("rule resolvers do not use vector search")
    }

    fn find_by_similarity(
        &self,
        _: &str,
        _: Option<&str>,
        _: usize,
        _: f64,
    ) -> Result<Vec<EdCandidate>> {
        unreachable!("rule resolvers do not use similarity search")
    }

    fn find_by_structured_key(
        &self,
        key: &StructuredKeyQuery<'_>,
        case_id: Option<&str>,
    ) -> Result<Vec<EdCandidate>> {
        let call = match key {
            StructuredKeyQuery::DocketEntry { number } => format!("docket {number}"),
            StructuredKeyQuery::Exhibit { id, party_attribution } => {
                format!("exhibit {id} {party_attribution:?} {case_id:?}")
            }
            StructuredKeyQuery::Statute(_) => "statute".to_owned(),
            StructuredKeyQuery::CaseCitation(_) => "case_citation".to_owned(),
        };
        self.record(call);
        Ok((self.structured)(key))
    }
}

fn patterns() -> KeyPatterns {
    KeyPatterns::from_mappings(
        &MappingsRegistry::load().expect("embedded mappings load").detection_patterns,
    )
}

fn ctx(name: &str, label: &str) -> EdContext {
    let mention = Mention::new("m1", label, name, Span::new(0, name.chars().count()));
    let mut doc_context = DocContext::new("doc-1", "complaint");
    doc_context.case_id = Some("CASE-1".into());
    EdContext {
        mention,
        local_context: LocalContext::new(name, name),
        doc_context,
        span_label: label.to_owned(),
        normalized_key: None,
    }
}

fn cand(id: &str, name: &str) -> EdCandidate {
    EdCandidate::new(id, name, "CASE")
}

fn expect_result(
    result: Option<EdResult>,
    chosen: &str,
    candidates: &[EdCandidate],
    confidence: f64,
    strategy: &str,
) {
    let result = result.expect("rule resolves the mention");
    assert_eq!(
        result,
        EdResult {
            mention_id: None,
            chosen_entity_id: Some(chosen.into()),
            candidates: candidates.to_vec(),
            resolved_by_llm: false,
            confidence,
            strategy_used: strategy.to_owned(),
        }
    );
}

// --- CASE ---------------------------------------------------------------------------------

#[test]
fn case_unique_docket_match() {
    let found = vec![cand("e1", "Case No. 2:23-cv-01492")];
    let returned = found.clone();
    let retriever = FakeRetriever::new(move |_| returned.clone(), |_, _| vec![]);
    let result = resolve_case_rule(&ctx("Case No. 2:23-cv-01492", "CASE"), &retriever, &patterns())
        .expect("resolves");
    expect_result(result, "e1", &found, 1.0, "rule_case_docket");
    assert_eq!(retriever.calls(), vec!["docket 2:23-cv-01492"]);
}

#[test]
fn case_ambiguous_docket_match_is_unresolved() {
    let retriever = FakeRetriever::new(
        |key| match key {
            StructuredKeyQuery::DocketEntry { .. } => vec![cand("e1", "A"), cand("e2", "B")],
            _ => vec![],
        },
        |_, _| vec![cand("e3", "C")],
    );
    let result = resolve_case_rule(&ctx("Case No. 2:23-cv-01492", "CASE"), &retriever, &patterns())
        .expect("resolves");
    assert_eq!(result, None);
    // Docket candidates exist, so the case-name fallback is skipped.
    assert!(retriever.calls().iter().all(|c| !c.starts_with("exact")));
}

#[test]
fn case_unique_citation_match() {
    let found = vec![cand("e1", "123 F.3d 456")];
    let returned = found.clone();
    let retriever = FakeRetriever::new(
        move |key| match key {
            StructuredKeyQuery::CaseCitation(key) => {
                assert_eq!(key.volume.as_deref(), Some("123"));
                assert_eq!(key.page.as_deref(), Some("456"));
                returned.clone()
            }
            _ => vec![],
        },
        |_, _| vec![],
    );
    let result =
        resolve_case_rule(&ctx("123 F.3d 456", "CASE"), &retriever, &patterns()).expect("resolves");
    expect_result(result, "e1", &found, 0.95, "rule_case_citation");
}

#[test]
fn case_unique_caption_match() {
    let found = vec![cand("e1", "Smith v. Jones")];
    let returned = found.clone();
    let retriever = FakeRetriever::new(
        move |key| match key {
            StructuredKeyQuery::CaseCitation(key) => {
                assert_eq!(key.plaintiff.as_deref(), Some("Smith"));
                assert_eq!(key.defendant.as_deref(), Some("Jones"));
                returned.clone()
            }
            _ => vec![],
        },
        |_, _| vec![],
    );
    let result = resolve_case_rule(&ctx("Smith v. Jones", "CASE"), &retriever, &patterns())
        .expect("resolves");
    expect_result(result, "e1", &found, 0.95, "rule_case_caption");
}

#[test]
fn case_ambiguous_caption_is_unresolved() {
    let retriever = FakeRetriever::new(
        |_| vec![cand("e1", "A"), cand("e2", "B")],
        |_, _| vec![cand("e3", "C")],
    );
    let result = resolve_case_rule(&ctx("Smith v. Jones", "CASE"), &retriever, &patterns())
        .expect("resolves");
    assert_eq!(result, None);
}

#[test]
fn case_falls_back_to_exact_case_name() {
    let found = vec![cand("e9", "Smith v. Jones")];
    let returned = found.clone();
    let retriever = FakeRetriever::new(
        |_| vec![],
        move |surface, entity_type| {
            assert_eq!((surface, entity_type), ("Smith v. Jones", Some("CASE")));
            returned.clone()
        },
    );
    let result = resolve_case_rule(&ctx("Smith v. Jones", "CASE"), &retriever, &patterns())
        .expect("resolves");
    expect_result(result, "e9", &found, 0.90, "rule_case_name");
    assert_eq!(retriever.calls(), vec!["case_citation", "exact Smith v. Jones Some(\"CASE\")"]);
}

#[test]
fn case_without_any_match_is_unresolved() {
    let retriever = FakeRetriever::empty();
    let result =
        resolve_case_rule(&ctx("the matter", "CASE"), &retriever, &patterns()).expect("resolves");
    assert_eq!(result, None);
    assert_eq!(retriever.calls(), vec!["exact the matter Some(\"CASE\")"]);

    let two = FakeRetriever::new(|_| vec![], |_, _| vec![cand("e1", "A"), cand("e2", "B")]);
    assert_eq!(resolve_case_rule(&ctx("the matter", "CASE"), &two, &patterns()).expect("ok"), None);
}

// --- STATUTE ------------------------------------------------------------------------------

const USC: &str = "42 U.S.C. § 1983";

#[test]
fn statute_unique_structured_match() {
    let found = vec![cand("s1", "42 U.S.C. 1983")];
    let returned = found.clone();
    let retriever = FakeRetriever::new(
        move |key| match key {
            StructuredKeyQuery::Statute(key) => {
                assert_eq!(key.title.as_deref(), Some("42"));
                assert_eq!(key.section.as_deref(), Some("1983"));
                returned.clone()
            }
            _ => vec![],
        },
        |_, _| vec![],
    );
    let result = resolve_statute_rule(&ctx(USC, "STATUTE"), &retriever, &patterns()).expect("ok");
    expect_result(result, "s1", &found, 0.95, "rule_statute_structured");
}

#[test]
fn statute_ambiguous_prefers_normalized_key_metadata() {
    let normalized = extract_statute_key(&patterns(), USC).expect("statute key").normalized_key();
    let mut keyed = cand("s2", "42 USC 1983");
    keyed.metadata.insert("normalized_key".into(), json!(normalized));
    let mut other = cand("s1", "42 U.S.C. 1983");
    other.metadata.insert("normalized_key".into(), json!("something_else"));
    let found = vec![other, keyed];
    let returned = found.clone();
    let retriever = FakeRetriever::new(move |_| returned.clone(), |_, _| vec![]);
    let result = resolve_statute_rule(&ctx(USC, "STATUTE"), &retriever, &patterns()).expect("ok");
    expect_result(result, "s2", &found, 0.90, "rule_statute_normalized");
}

#[test]
fn statute_ambiguous_without_key_falls_back_to_exact() {
    let exact = vec![cand("s3", USC)];
    let returned = exact.clone();
    let retriever = FakeRetriever::new(
        |_| vec![cand("s1", "A"), cand("s2", "B")],
        move |surface, entity_type| {
            assert_eq!((surface, entity_type), (USC, Some("STATUTE")));
            returned.clone()
        },
    );
    let result = resolve_statute_rule(&ctx(USC, "STATUTE"), &retriever, &patterns()).expect("ok");
    expect_result(result, "s3", &exact, 0.95, "rule_statute_exact");
}

#[test]
fn statute_without_key_uses_exact_match() {
    let exact = vec![cand("s3", "the Act")];
    let returned = exact.clone();
    let retriever = FakeRetriever::new(|_| vec![], move |_, _| returned.clone());
    let result =
        resolve_statute_rule(&ctx("the Act", "STATUTE"), &retriever, &patterns()).expect("ok");
    expect_result(result, "s3", &exact, 0.95, "rule_statute_exact");
    assert_eq!(retriever.calls(), vec!["exact the Act Some(\"STATUTE\")"]);

    let none = FakeRetriever::empty();
    assert_eq!(resolve_statute_rule(&ctx(USC, "STATUTE"), &none, &patterns()).expect("ok"), None);
}

// --- EXHIBIT ------------------------------------------------------------------------------

#[test]
fn exhibit_unique_id_match() {
    let found = vec![cand("x1", "Exhibit 12A")];
    let returned = found.clone();
    let retriever = FakeRetriever::new(move |_| returned.clone(), |_, _| vec![]);
    let result =
        resolve_exhibit_rule(&ctx("Exhibit 12a", "EXHIBIT"), &retriever, &patterns()).expect("ok");
    expect_result(result, "x1", &found, 0.95, "rule_exhibit_id");
    assert_eq!(retriever.calls(), vec!["exhibit 12A None Some(\"CASE-1\")"]);
}

#[test]
fn exhibit_ambiguous_resolved_by_party() {
    let found = vec![cand("x1", "Defendant's Exhibit 12A"), cand("x2", "Plaintiff's Exhibit 12A")];
    let returned = found.clone();
    let retriever = FakeRetriever::new(move |_| returned.clone(), |_, _| vec![]);
    let text = "Plaintiff's Exhibit 12A";
    let result = resolve_exhibit_rule(&ctx(text, "EXHIBIT"), &retriever, &patterns()).expect("ok");
    expect_result(result, "x2", &found, 0.90, "rule_exhibit_party");
    let calls = retriever.calls();
    assert_eq!(calls.len(), 1);
    assert!(calls[0].starts_with("exhibit 12A Some("), "{calls:?}");
}

#[test]
fn exhibit_ambiguous_party_falls_back_to_simple_id() {
    let unique = vec![cand("x3", "Exhibit 12A")];
    let returned = unique.clone();
    let retriever = FakeRetriever::new(
        move |key| match key {
            StructuredKeyQuery::Exhibit { party_attribution: None, .. } => returned.clone(),
            _ => vec![cand("x1", "Plaintiff Exhibit 12A"), cand("x2", "Plaintiff's Exhibit 12A")],
        },
        |_, _| vec![],
    );
    let text = "Plaintiff's Exhibit 12A";
    let result = resolve_exhibit_rule(&ctx(text, "EXHIBIT"), &retriever, &patterns()).expect("ok");
    expect_result(result, "x3", &unique, 0.95, "rule_exhibit_id");
    let calls = retriever.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[1], "exhibit 12A None Some(\"CASE-1\")");
}

#[test]
fn exhibit_without_match_is_unresolved() {
    let retriever = FakeRetriever::empty();
    let result =
        resolve_exhibit_rule(&ctx("Exhibit 12A", "EXHIBIT"), &retriever, &patterns()).expect("ok");
    assert_eq!(result, None);
    let none = FakeRetriever::empty();
    assert_eq!(
        resolve_exhibit_rule(&ctx("the photo", "EXHIBIT"), &none, &patterns()).expect("ok"),
        None
    );
    assert!(none.calls().is_empty());
}

// --- GENERIC ------------------------------------------------------------------------------

#[test]
fn generic_unique_exact_match() {
    let found = vec![EdCandidate::new("p1", "Maria Lopez", "PERSON")];
    let returned = found.clone();
    let retriever = FakeRetriever::new(
        |_| vec![],
        move |surface, entity_type| {
            assert_eq!((surface, entity_type), ("Maria Lopez", Some("PERSON")));
            returned.clone()
        },
    );
    let result =
        resolve_generic_rule(&ctx("Maria Lopez", "PERSON"), &retriever, &patterns()).expect("ok");
    expect_result(result, "p1", &found, 0.9, "rule_generic_exact");
}

#[test]
fn generic_ambiguous_or_missing_is_unresolved() {
    let two = FakeRetriever::new(
        |_| vec![],
        |_, _| vec![EdCandidate::new("p1", "A", "PERSON"), EdCandidate::new("p2", "B", "PERSON")],
    );
    assert_eq!(resolve_generic_rule(&ctx("Lopez", "PERSON"), &two, &patterns()).expect("ok"), None);
    let none = FakeRetriever::empty();
    assert_eq!(
        resolve_generic_rule(&ctx("Lopez", "PERSON"), &none, &patterns()).expect("ok"),
        None
    );
}
