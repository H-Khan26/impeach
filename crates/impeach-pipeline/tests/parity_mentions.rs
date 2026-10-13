//! Mention parity against `fixtures/reference/mention_detection/<scenario>.json`.
//!
//! The Rust detectors and `MentionStage` run over the Python chunks (`rag_index` reference),
//! with the Python GLiNER2 output replayed through [`parity_a::ReplayNer`], so the comparison
//! covers the rule detectors, score filtering, name normalisation, split healing, quality
//! filtering, sentence indexes and merging on equal input. Regex and citation mention ids are
//! deterministic hashes in Rust (random in Python) and are not compared.

mod common;
mod parity_a;

use std::sync::Arc;

use impeach_config::MappingsRegistry;
use impeach_core::{IndexingResult, Mention};
use impeach_models::ModelKind;
use impeach_ner::{CitationDetector, GlinerDetector, MentionDetector, RegexDetector};
use impeach_pipeline::{MentionStage, PipelineContext, Stage};
use parity_a::{
    input_document, mention_key, python_indexing_result, raw_key, reference_docs, reference_file,
    MentionKey, PyMentionDoc, PyRagDoc, ReplayNer, LABEL_SET, SCENARIOS,
};
use pretty_assertions::assert_eq;
use serde::Deserialize;
use serde_json::Value;

const EYECITE: &str = "EyeciteMentionDetector";
const REGEX: &str = "RegexMentionDetector";
const GLINER: &str = "Gliner2MentionDetector";

/// A scenario document: Python chunks, Python mentions and the input text.
struct Doc {
    rag: PyRagDoc,
    mentions: PyMentionDoc,
    result: IndexingResult,
}

fn docs(scenario: &str) -> Vec<Doc> {
    let rag = reference_docs::<PyRagDoc>("rag_index", scenario);
    let mentions = reference_docs::<PyMentionDoc>("mention_detection", scenario);
    assert_eq!(rag.len(), mentions.len());
    rag.into_iter()
        .zip(mentions)
        .map(|(rag, mentions)| {
            assert_eq!(rag.input, mentions.input);
            let result = python_indexing_result(&rag, &input_document(&rag.input));
            Doc { rag, mentions, result }
        })
        .collect()
}

fn segment_id(doc: &Doc, sequence: usize) -> String {
    format!("{}_{sequence}", doc.rag.doc_id)
}

/// A replay model holding the Python GLiNER2 output of every chunk of `docs`.
fn replay(docs: &[Doc]) -> ReplayNer {
    let mut ner = ReplayNer::default();
    for doc in docs {
        for write in &doc.result.chunks {
            let raw = doc.mentions.raw(GLINER, &segment_id(doc, write.chunk.sequence_number));
            ner.add_python_chunk(&write.chunk.text, raw);
        }
    }
    ner
}

#[test]
fn rule_detectors_match_python_per_chunk() {
    let mappings = MappingsRegistry::load().expect("mappings");
    let regex = RegexDetector::from_mappings(&mappings.detection_patterns);
    let mut counts = [0_usize; 2];
    for scenario in SCENARIOS {
        for doc in docs(scenario) {
            for write in &doc.result.chunks {
                let segment = segment_id(&doc, write.chunk.sequence_number);
                let text = &write.chunk.text;
                for (index, (detector, class)) in
                    [(&regex as &dyn MentionDetector, REGEX), (&CitationDetector, EYECITE)]
                        .into_iter()
                        .enumerate()
                {
                    let rust = detector.detect(text, &segment).expect("rule detectors never fail");
                    let python = doc.mentions.raw(class, &segment);
                    let rust_keys: Vec<_> = rust.iter().map(raw_key).collect();
                    let python_keys: Vec<_> = python.iter().map(raw_key).collect();
                    assert_eq!(rust_keys, python_keys, "{class} {scenario} {segment}");
                    for (r, p) in rust.iter().zip(python) {
                        assert_eq!((r.score, &r.metadata), (p.score, &p.metadata), "{segment}");
                    }
                    counts[index] += rust.len();
                }
            }
        }
    }
    // Regex: 12 + 7 Green Valley and 2 + 2 Mason mentions; no citation in either scenario.
    assert_eq!(counts, [23, 0]);
}

#[test]
fn replayed_gliner_output_aligns_like_python() {
    let mappings = MappingsRegistry::load().expect("mappings");
    let mut total = 0;
    for scenario in SCENARIOS {
        let docs = docs(scenario);
        let model = Arc::new(replay(&docs));
        let detector = GlinerDetector::new(model, mappings, LABEL_SET, 0.4).expect("detector");
        for doc in &docs {
            for write in &doc.result.chunks {
                let segment = segment_id(doc, write.chunk.sequence_number);
                let rust = detector.detect(&write.chunk.text, &segment).expect("replay");
                let python = doc.mentions.raw(GLINER, &segment);
                let rows = |ms: &[Mention]| -> Vec<(String, MentionKey)> {
                    ms.iter().map(|m| (m.id.to_string(), mention_key(m))).collect()
                };
                assert_eq!(rows(&rust), rows(python), "{scenario} {segment}");
                total += rust.len();
            }
        }
    }
    assert_eq!(total, 120 + 111 + 10 + 10);
}

/// Runs `MentionStage` (shipped config) over the Python chunks of `doc` with the replay model.
fn run_stage(docs: &[Doc], doc: &Doc) -> PipelineContext {
    let harness = common::Harness::new(common::default_config());
    let registry = &harness.services.registry;
    registry.insert_ner(registry.key(ModelKind::Gliner2), Arc::new(replay(docs)));
    let config = harness
        .services
        .config
        .stages
        .iter()
        .find_map(|stage| match stage {
            impeach_config::StageConfig::Mention(m) => Some(m.clone()),
            _ => None,
        })
        .expect("MentionStage config");
    let stage = MentionStage::new(&config, &harness.services).expect("stage");
    let mut ctx = PipelineContext::new(&harness.services);
    ctx.data.rag_index_result = Some(doc.result.clone());
    stage.run(&mut ctx).expect("stage runs");
    ctx
}

#[test]
fn final_mentions_match_python_with_replayed_gliner() {
    let mut totals = Vec::new();
    for scenario in SCENARIOS {
        let docs = docs(scenario);
        for doc in &docs {
            let ctx = run_stage(&docs, doc);
            let rust = ctx.data.doc_mentions.as_ref().expect("doc_mentions");
            let python = &doc.mentions.doc_mentions;
            let keys = |ms: &[Mention]| -> Vec<MentionKey> { ms.iter().map(mention_key).collect() };
            assert_eq!(keys(rust), keys(python), "{scenario}/{}", doc.rag.input);
            for (r, p) in rust.iter().zip(python) {
                assert_eq!(r.doc_id, p.doc_id, "{}", p.name);
                assert_eq!(r.chunk_id, p.chunk_id, "{}", p.name);
                // GLiNER2 ids follow the Python scheme; rule ids are hashes (random in Python).
                if p.id.as_str().starts_with("ent_") {
                    assert_eq!(r.id, p.id, "{}", p.name);
                } else {
                    assert!(r.id.as_str().starts_with("span-"), "{}", r.id);
                }
            }
            let by_chunk = ctx.data.mentions_by_chunk.as_ref().expect("mentions_by_chunk");
            let rust_chunks: Vec<(&str, usize)> =
                by_chunk.iter().map(|(chunk, ms)| (chunk.as_str(), ms.len())).collect();
            let python_chunks: Vec<(&str, usize)> = doc
                .mentions
                .mentions_by_chunk
                .iter()
                .map(|(chunk, ids)| (chunk.as_str(), ids.len()))
                .collect();
            assert_eq!(rust_chunks, python_chunks, "{scenario}/{}", doc.rag.input);
            totals.push(rust.len());
        }
    }
    assert_eq!(totals, [112, 109, 10, 10]);
}

/// One text of `fixtures/reference/mention_detection/detector_cases.json`.
#[derive(Debug, Deserialize)]
struct DetectorCase {
    segment_id: String,
    text: String,
    eyecite: Vec<Mention>,
    regex: Vec<Mention>,
}

#[derive(Debug, Deserialize)]
struct DetectorCases {
    cases: Vec<DetectorCase>,
}

/// `(start, end, type, name, metadata)` of a detector mention.
fn citation_row(m: &Mention) -> (usize, usize, String, String, Value) {
    let metadata = serde_json::to_value(&m.metadata).expect("metadata");
    (m.span.start, m.span.end, m.entity_type.clone(), m.name.clone(), metadata)
}

#[test]
fn detectors_match_python_on_citation_texts() {
    let cases: DetectorCases = reference_file("mention_detection", "detector_cases.json");
    let mappings = MappingsRegistry::load().expect("mappings");
    let regex = RegexDetector::from_mappings(&mappings.detection_patterns);
    let mut citations = 0;
    for case in &cases.cases {
        let rust = regex.detect(&case.text, &case.segment_id).expect("regex");
        let keys = |ms: &[Mention]| ms.iter().map(raw_key).collect::<Vec<_>>();
        assert_eq!(keys(&rust), keys(&case.regex), "regex {}", case.segment_id);

        let rust = CitationDetector.detect(&case.text, &case.segment_id).expect("citations");
        let mut rust_rows: Vec<_> = rust.iter().map(citation_row).collect();
        let mut python_rows: Vec<_> = case.eyecite.iter().map(citation_row).collect();
        if case.segment_id == "case_1" {
            // In a string of citations eyecite gives the first one the year of a later
            // parenthetical ("(Cal. 2004)") and the others none; Rust reads the parenthetical
            // right after each citation.
            let years = |rows: &mut Vec<(usize, usize, String, String, Value)>| -> Vec<Value> {
                rows.iter_mut()
                    .map(|row| {
                        row.4.as_object_mut().and_then(|m| m.remove("year")).unwrap_or_default()
                    })
                    .collect()
            };
            let none = Value::Null;
            let rust_years = [2016.into(), none.clone(), 2004.into(), 2005.into(), none.clone()];
            let rust_years = [&rust_years[..], &[1993.into(), 1996.into(), none.clone()]].concat();
            assert_eq!(years(&mut rust_rows), rust_years);
            let python_years = [vec![2004.into()], vec![none; 7]].concat();
            assert_eq!(years(&mut python_rows), python_years);
        }
        assert_eq!(rust_rows, python_rows, "eyecite {}", case.segment_id);
        citations += rust.len();
    }
    assert_eq!(citations, 22);
}

#[cfg(feature = "gliner2")]
#[test]
#[ignore = "needs fastino/gliner2-base-v1 in the Hugging Face cache"]
fn real_gliner2_over_python_chunks() {
    use std::collections::HashSet;

    for scenario in SCENARIOS {
        let docs = docs(scenario);
        for doc in &docs {
            let harness = common::Harness::new(common::default_config());
            let config = harness
                .services
                .config
                .stages
                .iter()
                .find_map(|stage| match stage {
                    impeach_config::StageConfig::Mention(m) => Some(m.clone()),
                    _ => None,
                })
                .expect("MentionStage config");
            let stage = MentionStage::new(&config, &harness.services).expect("stage");
            let mut ctx = PipelineContext::new(&harness.services);
            ctx.data.rag_index_result = Some(doc.result.clone());
            let started = std::time::Instant::now();
            stage.run(&mut ctx).expect("stage runs with the real model");
            let rust = ctx.data.doc_mentions.clone().unwrap_or_default();
            let python = &doc.mentions.doc_mentions;
            let span_type =
                |m: &Mention| (m.span.start, m.span.end, m.entity_type.clone(), m.name.clone());
            let rust_set: HashSet<_> = rust.iter().map(span_type).collect();
            let python_set: HashSet<_> = python.iter().map(span_type).collect();
            let shared = rust_set.intersection(&python_set).count();
            let raw_python: usize = doc
                .mentions
                .detector_raw
                .iter()
                .filter(|call| call.detector == GLINER)
                .map(|call| call.entities.len())
                .sum();
            eprintln!(
                "{scenario}/{}: python {} final ({raw_python} GLiNER2 raw), rust {} final, \
                 {shared} identical (start, end, type, name), {:?}",
                doc.rag.input,
                python.len(),
                rust.len(),
                started.elapsed()
            );
            // Candle and PyTorch scores differ slightly near the threshold, so most (not all)
            // spans agree; names per chunk are compared as well.
            assert!(shared * 10 >= python.len() * 9, "{scenario}/{}", doc.rag.input);
            let chunk_name =
                |m: &Mention| (m.chunk_id.clone(), m.entity_type.clone(), m.name.clone());
            let rust_names: HashSet<_> = rust.iter().map(chunk_name).collect();
            let python_names: HashSet<_> = python.iter().map(chunk_name).collect();
            let shared_names = rust_names.intersection(&python_names).count();
            eprintln!(
                "  (chunk, type, name): python {}, rust {}, shared {shared_names}",
                python_names.len(),
                rust_names.len()
            );
            for m in python_names.difference(&rust_names) {
                eprintln!("  python only: {m:?}");
            }
            for m in rust_names.difference(&python_names) {
                eprintln!("  rust only: {m:?}");
            }
            assert!(shared_names * 10 >= python_names.len() * 8, "{scenario}/{}", doc.rag.input);
        }
    }
}
