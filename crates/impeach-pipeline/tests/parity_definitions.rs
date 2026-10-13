//! Definition parity: `DefinitionStage` / `DefinitionExtractor` against the Python prototype's
//! `fixtures/reference/definition_extraction/`: the scenario documents (`<scenario>.json`,
//! with the Python mentions of `mention_detection/<scenario>.json` as input), extra documents
//! run through the Python RAGIndex, Mention and Definition stages (`documents.json`) and
//! hand-written texts and mentions (`cases.json`).

mod common;
mod parity_a;

use impeach_config::StageConfig;
use impeach_core::processing::DefinedTerm;
use impeach_core::Mention;
use impeach_ner::definitions::{DefinitionExtractor, RoleConfig};
use impeach_pipeline::{DefinitionStage, PipelineContext, Stage};
use parity_a::{input_document, reference_docs, reference_file, PyMentionDoc, SCENARIOS};
use pretty_assertions::assert_eq;
use serde::Deserialize;

/// Runs the shipped `DefinitionStage` over `mentions` and `text`; `None` when it finds nothing.
fn run_stage(doc_id: &str, text: &str, mentions: Vec<Mention>) -> Option<Vec<DefinedTerm>> {
    let harness = common::Harness::new(common::default_config());
    let config = harness
        .services
        .config
        .stages
        .iter()
        .find_map(|stage| match stage {
            StageConfig::Definition(config) => Some(config.clone()),
            _ => None,
        })
        .expect("DefinitionStage config");
    let stage = DefinitionStage::new(&config, &harness.services).expect("stage");
    let mut ctx = PipelineContext::new(&harness.services);
    ctx.data.doc_id = Some(doc_id.into());
    ctx.data.doc_text = Some(text.to_owned());
    ctx.data.doc_mentions = Some(mentions);
    stage.run(&mut ctx).expect("stage runs");
    ctx.data.defined_terms
}

#[derive(Debug, Deserialize)]
struct PyDefinitionDoc {
    input: String,
    defined_terms: Vec<DefinedTerm>,
}

#[test]
fn scenario_documents_define_no_terms() {
    for scenario in SCENARIOS {
        let definitions = reference_docs::<PyDefinitionDoc>("definition_extraction", scenario);
        let mentions = reference_docs::<PyMentionDoc>("mention_detection", scenario);
        for (python, mentions) in definitions.iter().zip(mentions) {
            assert_eq!(python.input, mentions.input);
            assert!(python.defined_terms.is_empty());
            let text = input_document(&python.input).text;
            let rust = run_stage(&mentions.doc_id, &text, mentions.doc_mentions);
            assert_eq!(rust, None, "{scenario}/{}", python.input);
        }
    }
}

/// A document of `documents.json`.
#[derive(Debug, Deserialize)]
struct ExtraDocument {
    input: String,
    doc_id: String,
    doc_text: String,
    doc_mentions: Vec<Mention>,
    defined_terms: Vec<DefinedTerm>,
}

#[derive(Debug, Deserialize)]
struct ExtraDocuments {
    documents: Vec<ExtraDocument>,
}

#[test]
fn extra_documents_match_python() {
    let docs: ExtraDocuments = reference_file("definition_extraction", "documents.json");
    let mut found = 0;
    for doc in docs.documents {
        let rust = run_stage(&doc.doc_id, &doc.doc_text, doc.doc_mentions).unwrap_or_default();
        assert_eq!(rust, doc.defined_terms, "{}", doc.input);
        found += rust.len();
    }
    assert!(found > 0, "the synthetic document defines terms");
}

/// A case of `cases.json`.
#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    text: String,
    mentions: Vec<Mention>,
    defined_terms: Vec<DefinedTerm>,
}

#[derive(Debug, Deserialize)]
struct Cases {
    cases: Vec<Case>,
}

#[test]
fn hand_written_cases_match_python() {
    let cases: Cases = reference_file("definition_extraction", "cases.json");
    let extractor = DefinitionExtractor::new(RoleConfig::load(None).expect("roles.yaml"));
    let mut found = 0;
    for case in &cases.cases {
        let mut rust = extractor.extract_from_text(&case.text, &case.mentions, &case.name);
        rust.extend(extractor.extract_from_metadata(&case.mentions));
        assert_eq!(rust, case.defined_terms, "{}", case.name);
        found += rust.len();
    }
    assert_eq!((cases.cases.len(), found), (19, 23));
}
