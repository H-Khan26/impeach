//! Coreference parity with the Python prototype: the fastcoref clusters the Python stage
//! post-processed (`fixtures/reference/coreference/<scenario>.json`, `raw_coref`) are scripted
//! through `MockCoref`, the Python mentions and chunks are seeded into the context, and the
//! Rust CoreferenceStage (processing_v17 settings) must produce the Python outputs.
//!
//! The ignored `fastcoref_sidecar_matches_python_clusters` checks the sidecar itself: with
//! `IMPEACH_FASTCOREF_PYTHON` (a Python with fastcoref) and `IMPEACH_FASTCOREF_SCRIPT` (a
//! sidecar script, see the `impeach_models::sidecar` docs) set, `SidecarCoref` must return the
//! Python clusters for every document.

mod common;
mod parity_b;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use common::{default_config, DIM};
use impeach_config::StageConfig;
use impeach_core::MentionId;
use impeach_models::{Coref, MockCoref, ModelKind, ModelRegistry, SidecarCommand, SidecarCoref};
use impeach_pipeline::{ContextData, CoreferenceStage, PipelineContext, Stage, StageBuildContext};
use impeach_store::{GraphStore, MemoryStore};
use parity_b::{scenario, PyDoc};
use pretty_assertions::assert_eq;
use serde_json::Value;

/// Services over processing_v17 with `MockCoref` scripted with every document's clusters.
fn services(docs: &[PyDoc]) -> StageBuildContext {
    let config = default_config();
    let registry = ModelRegistry::empty(config.name());
    let mut coref = MockCoref::default();
    for doc in docs {
        coref = coref.with_text(doc.text(), doc.raw_coref.clone());
    }
    registry.insert_coref(registry.key(ModelKind::Coref), Arc::new(coref));
    let store: Arc<dyn GraphStore> = Arc::new(MemoryStore::new(DIM));
    StageBuildContext::new(Arc::new(config), Arc::new(registry), store)
}

fn stage(services: &StageBuildContext) -> CoreferenceStage {
    let block = services
        .config
        .stages
        .iter()
        .find_map(|s| match s {
            StageConfig::Coreference(c) => Some(c.clone()),
            _ => None,
        })
        .expect("processing_v17 has a CoreferenceStage");
    CoreferenceStage::new(&block, services).expect("stage builds")
}

fn run(services: &StageBuildContext, doc: &PyDoc) -> ContextData {
    let mut ctx = PipelineContext::new(services);
    doc.seed_context(&mut ctx);
    stage(services).run(&mut ctx).expect("coreference runs");
    ctx.data
}

fn ids(list: &[MentionId]) -> Vec<String> {
    list.iter().map(|id| id.as_str().to_owned()).collect()
}

/// Counts of one document, for the parity summary.
#[derive(Debug, PartialEq, Eq)]
struct Counts {
    raw_clusters: usize,
    clusters: usize,
    synthetic: usize,
    doc_mentions: usize,
}

/// Compares every coreference output of `doc`; returns the counts.
fn assert_document_parity(services: &StageBuildContext, doc: &PyDoc) -> Counts {
    let label = format!("{} ({})", doc.input, doc.doc_id());

    // Cluster ids and spans of the resolver output.
    let labelled = impeach_coref::label_clusters(doc.doc_id().as_str(), doc.text(), &doc.raw_coref);
    let labelled_ids: Vec<&str> = labelled.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(labelled_ids, doc.raw_coref_ids, "{label}: raw cluster ids");
    let py_raw: Vec<(usize, usize, String, bool)> = doc.coref["raw_coref"]
        .as_array()
        .expect("raw_coref")
        .iter()
        .flat_map(|o| o["clusters"].as_array().expect("clusters").iter())
        .flat_map(|c| c["mentions"].as_array().expect("mentions").iter())
        .map(|m| {
            (
                m["start_char"].as_u64().expect("start") as usize,
                m["end_char"].as_u64().expect("end") as usize,
                m["text"].as_str().expect("text").to_owned(),
                m["is_pronoun"].as_bool().expect("is_pronoun"),
            )
        })
        .collect();
    let rust_raw: Vec<(usize, usize, String, bool)> = labelled
        .iter()
        .flat_map(|c| c.spans.iter())
        .map(|s| (s.start, s.end, s.text.clone(), s.is_pronoun))
        .collect();
    assert_eq!(rust_raw, py_raw, "{label}: raw cluster spans");

    let data = run(services, doc);
    let py_clusters = doc.coref_clusters();
    let clusters = data.coref_clusters.clone().unwrap_or_default();

    // Cluster ids and pronoun spans in order. Members compare as sets: the Python stage
    // rebuilds member lists from sets when it deduplicates, Rust keeps first-seen order.
    let summary = |c: &impeach_core::CorefClusterContext| {
        (
            c.id.clone(),
            ids(&c.member_mention_ids).into_iter().collect::<BTreeSet<_>>(),
            c.pronoun_spans.clone(),
        )
    };
    assert_eq!(
        clusters.iter().map(summary).collect::<Vec<_>>(),
        py_clusters.iter().map(summary).collect::<Vec<_>>(),
        "{label}: coref_clusters"
    );

    assert_eq!(
        data.canonical_mention_for_cluster.clone().unwrap_or_default(),
        doc.canonical_mention_for_cluster(),
        "{label}: canonical_mention_for_cluster"
    );
    // Keys follow member order, so the map compares unordered.
    let canonical_text: BTreeMap<MentionId, String> =
        data.canonical_text_for_mention.clone().unwrap_or_default().into_iter().collect();
    let py_canonical_text: BTreeMap<MentionId, String> =
        doc.canonical_text_for_mention().into_iter().collect();
    assert_eq!(canonical_text, py_canonical_text, "{label}: canonical_text_for_mention");

    assert_eq!(
        data.synthetic_pronoun_mentions.clone().unwrap_or_default(),
        doc.synthetic_pronoun_mentions(),
        "{label}: synthetic_pronoun_mentions"
    );

    // Synthetic pronoun mentions: every field, appended after the detected mentions.
    let doc_mentions = data.doc_mentions.clone().expect("doc_mentions");
    let added: Vec<Value> = doc_mentions[doc.mentions.len()..]
        .iter()
        .map(|m| serde_json::to_value(m).expect("mention serialises"))
        .collect();
    let py_added: Vec<Value> = doc.coref["added_mentions"].as_array().expect("added").clone();
    assert_eq!(added, py_added, "{label}: synthetic pronoun mentions");
    assert_eq!(
        doc_mentions.len(),
        doc.coref["doc_mention_count"].as_u64().expect("count") as usize,
        "{label}: doc_mentions after coreference"
    );

    let by_chunk: BTreeMap<String, Vec<String>> = data
        .mentions_by_chunk
        .clone()
        .unwrap_or_default()
        .into_iter()
        .map(|(chunk, mentions)| {
            (chunk.as_str().to_owned(), mentions.iter().map(|m| m.id.as_str().to_owned()).collect())
        })
        .collect();
    let py_by_chunk: BTreeMap<String, Vec<String>> =
        doc.coref_mentions_by_chunk().into_iter().collect();
    assert_eq!(by_chunk, py_by_chunk, "{label}: mentions_by_chunk");

    Counts {
        raw_clusters: labelled.len(),
        clusters: clusters.len(),
        synthetic: added.len(),
        doc_mentions: doc_mentions.len(),
    }
}

fn assert_scenario_parity(name: &str) -> Vec<Counts> {
    let docs = scenario(name);
    let services = services(&docs);
    docs.iter().map(|doc| assert_document_parity(&services, doc)).collect()
}

#[test]
fn green_valley_coreference_matches_python() {
    let counts = assert_scenario_parity("green_valley");
    assert_eq!(
        counts,
        [
            Counts { raw_clusters: 25, clusters: 8, synthetic: 39, doc_mentions: 151 },
            Counts { raw_clusters: 20, clusters: 7, synthetic: 15, doc_mentions: 124 },
        ]
    );
}

#[test]
fn mason_coreference_matches_python() {
    let counts = assert_scenario_parity("mason");
    assert_eq!(
        counts,
        [
            Counts { raw_clusters: 2, clusters: 1, synthetic: 3, doc_mentions: 13 },
            Counts { raw_clusters: 2, clusters: 1, synthetic: 3, doc_mentions: 13 },
        ]
    );
}

/// Resolver clusters as `(id, [(start, end, text, is_pronoun)])`.
type LabelledClusters = Vec<(String, Vec<(usize, usize, String, bool)>)>;

fn labelled(doc: &PyDoc, clusters: &[impeach_models::CorefCluster]) -> LabelledClusters {
    impeach_coref::label_clusters(doc.doc_id().as_str(), doc.text(), clusters)
        .into_iter()
        .map(|c| {
            let spans =
                c.spans.into_iter().map(|s| (s.start, s.end, s.text, s.is_pronoun)).collect();
            (c.id, spans)
        })
        .collect()
}

#[test]
#[ignore = "needs a fastcoref sidecar: IMPEACH_FASTCOREF_PYTHON and IMPEACH_FASTCOREF_SCRIPT"]
fn fastcoref_sidecar_matches_python_clusters() {
    let python = std::env::var_os("IMPEACH_FASTCOREF_PYTHON").expect("IMPEACH_FASTCOREF_PYTHON");
    let script = std::env::var_os("IMPEACH_FASTCOREF_SCRIPT").expect("IMPEACH_FASTCOREF_SCRIPT");
    let sidecar = SidecarCoref::new(SidecarCommand::new(python).arg(script)).expect("starts");
    let mut clusters = 0;
    for name in ["green_valley", "mason"] {
        for doc in scenario(name) {
            let raw = sidecar.resolve(doc.text()).expect("sidecar resolves");
            let rust = labelled(&doc, &raw);
            assert_eq!(rust, labelled(&doc, &doc.raw_coref), "{}", doc.input);
            let ids: Vec<&str> = rust.iter().map(|(id, _)| id.as_str()).collect();
            assert_eq!(ids, doc.raw_coref_ids, "{}", doc.input);
            clusters += rust.len();
        }
    }
    assert_eq!(clusters, 25 + 20 + 2 + 2);
}
