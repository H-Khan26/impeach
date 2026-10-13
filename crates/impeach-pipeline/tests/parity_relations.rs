//! Relation parity against `fixtures/reference/relation_extraction/<scenario>.json`.
//!
//! The Python prototype extracts no relation on either scenario: its relation stage looks for
//! a `relations` key holding `{label: [{"head", "tail"}]}` dicts, but GLiNER2 returns
//! `{"relation_extraction": {label: [[head, tail]]}}`, so every chunk normalises to nothing.
//! The Rust stage reads `relation_extraction` with the model's head and tail spans, so with the
//! real model it does find relations; that run is the ignored test below.

mod common;
mod parity_a;

use parity_a::{reference_docs, SCENARIOS};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
struct PyRelationDoc {
    input: String,
    extracted_relations: Vec<Value>,
    relations_by_type: serde_json::Map<String, Value>,
}

#[test]
fn python_extracts_no_relations() {
    let mut docs = 0;
    for scenario in SCENARIOS {
        for doc in reference_docs::<PyRelationDoc>("relation_extraction", scenario) {
            assert!(doc.extracted_relations.is_empty(), "{}", doc.input);
            assert!(doc.relations_by_type.is_empty(), "{}", doc.input);
            docs += 1;
        }
    }
    assert_eq!(docs, 4);
}

#[cfg(feature = "gliner2")]
#[test]
#[ignore = "needs fastino/gliner2-base-v1 in the Hugging Face cache"]
fn real_gliner2_relations_over_python_chunks() {
    use impeach_config::StageConfig;
    use impeach_pipeline::{PipelineContext, RelationStage, Stage};
    use parity_a::{input_document, python_indexing_result, PyRagDoc};

    let mut total = 0;
    for scenario in SCENARIOS {
        for doc in reference_docs::<PyRagDoc>("rag_index", scenario) {
            let harness = common::Harness::new(common::default_config());
            let config = harness
                .services
                .config
                .stages
                .iter()
                .find_map(|stage| match stage {
                    StageConfig::Relation(config) => Some(config.clone()),
                    _ => None,
                })
                .expect("RelationStage config");
            let stage = RelationStage::new(&config, &harness.services).expect("stage");
            let mut ctx = PipelineContext::new(&harness.services);
            ctx.data.rag_index_result =
                Some(python_indexing_result(&doc, &input_document(&doc.input)));
            let started = std::time::Instant::now();
            stage.run(&mut ctx).expect("stage runs");
            let relations = ctx.data.extracted_relations.clone().unwrap_or_default();
            eprintln!(
                "{scenario}/{}: {} relations over {} chunks in {:?}; by type {:?}",
                doc.input,
                relations.len(),
                doc.chunks.len(),
                started.elapsed(),
                ctx.data.relations_by_type
            );
            for r in &relations {
                eprintln!(
                    "  {} ({}) {} -> {} [{:?}] chunk {}",
                    r.relation, r.canonical_relation, r.head, r.tail, r.score, r.chunk_id
                );
            }
            total += relations.len();
        }
    }
    assert!(total > 0, "the real model finds relations where the Python prototype finds none");
}
