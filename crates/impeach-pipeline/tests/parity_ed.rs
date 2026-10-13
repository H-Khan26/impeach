//! ED parity with the Python prototype: the Rust EDStage (processing_v17, `allow_llm: false`)
//! runs on the Python post-coreference context (mentions, synthetic pronoun mentions, clusters,
//! canonical maps, chunks) of each document, in scenario order against one store, and its
//! decisions must equal `fixtures/reference/ed_resolution/<scenario>.json`.
//!
//! Decisions, not scores, are compared: which entity each result chose (or none), the
//! strategy, created vs reused, `resolved_mentions`, and the entities in the store (name, type,
//! aliases) after each document. Entity ids differ between the stores, so entities are
//! compared by name and type.

mod common;
mod parity_b;

use std::collections::BTreeMap;
use std::sync::Arc;

use common::{default_config, DIM};
use impeach_config::StageConfig;
use impeach_models::{Embedder, MockEmbedder, ModelKind, ModelRegistry};
use impeach_pipeline::{EdStage, PipelineContext, SharedEmbedder, Stage, StageBuildContext};
use impeach_store::{GraphStore, MemoryStore};
use parity_b::{scenario, PyDoc};
use pretty_assertions::assert_eq;

/// One ED result: mention id, strategy, chosen entity (`name (TYPE)`), resolved by the LLM.
type Decision = (String, String, Option<String>, bool);

/// An entity as `(name, type, sorted aliases)`.
type EntitySummary = (String, String, Vec<String>);

/// Decisions of one document.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DocDecisions {
    results: Vec<Decision>,
    resolved: BTreeMap<String, String>,
    entities_after: Vec<EntitySummary>,
}

#[cfg_attr(not(feature = "fastembed"), allow(dead_code))]
impl DocDecisions {
    fn count(&self, predicate: impl Fn(&str) -> bool) -> usize {
        self.results.iter().filter(|(_, strategy, _, _)| predicate(strategy)).count()
    }

    /// `(results, created, resolved mentions, entities after)`.
    fn counts(&self) -> (usize, usize, usize, usize) {
        let created = self.count(|s| s.ends_with("_create"));
        (self.results.len(), created, self.resolved.len(), self.entities_after.len())
    }
}

fn label(name: &str, entity_type: &str) -> String {
    format!("{name} ({entity_type})")
}

/// Entity id -> label; panics when two entities share a label (ambiguous comparison).
fn labels(entities: &[(String, EntitySummary)]) -> BTreeMap<String, String> {
    let mut seen = std::collections::BTreeSet::new();
    entities
        .iter()
        .map(|(id, (name, entity_type, _))| {
            let label = label(name, entity_type);
            assert!(seen.insert(label.clone()), "two entities named {label}");
            (id.clone(), label)
        })
        .collect()
}

fn summaries(entities: &[(String, EntitySummary)]) -> Vec<EntitySummary> {
    entities.iter().map(|(_, summary)| summary.clone()).collect()
}

/// The Python decisions of `doc` (entities in creation order, i.e. by numeric id).
fn python_decisions(doc: &PyDoc) -> DocDecisions {
    let mut entities: Vec<(String, EntitySummary)> = doc.ed["entities_after"]
        .as_array()
        .expect("entities_after")
        .iter()
        .map(|e| {
            let mut aliases: Vec<String> = e["aliases"]
                .as_array()
                .expect("aliases")
                .iter()
                .map(|a| a.as_str().expect("alias").to_owned())
                .collect();
            aliases.sort();
            let text = |key: &str| e[key].as_str().expect("entity field").to_owned();
            (text("id"), (text("name"), text("entity_type"), aliases))
        })
        .collect();
    entities.sort_by_key(|(id, _)| id.parse::<u64>().expect("numeric Python entity id"));
    let names = labels(&entities);
    let name_of = |id: &str| names.get(id).cloned().unwrap_or_else(|| format!("<unknown {id}>"));
    let results = doc.ed["ed_results"]
        .as_array()
        .expect("ed_results")
        .iter()
        .map(|r| {
            (
                r["mention_id"].as_str().expect("mention_id").to_owned(),
                r["strategy_used"].as_str().expect("strategy").to_owned(),
                r["chosen_entity_id"].as_str().map(name_of),
                r["resolved_by_llm"].as_bool().expect("resolved_by_llm"),
            )
        })
        .collect();
    let resolved = doc.ed["resolved_mentions"]
        .as_object()
        .expect("resolved_mentions")
        .iter()
        .map(|(mention, entity)| (mention.clone(), name_of(entity.as_str().expect("entity id"))))
        .collect();
    DocDecisions { results, resolved, entities_after: summaries(&entities) }
}

/// Every entity of `store` in creation order.
fn store_entities(store: &dyn GraphStore) -> Vec<(String, EntitySummary)> {
    let mut entities: Vec<(String, EntitySummary)> = store
        .get_all_entities(10_000)
        .expect("entities")
        .into_iter()
        .map(|e| {
            let mut aliases = e.node.aliases();
            aliases.sort();
            (e.id.as_str().to_owned(), (e.node.name.clone(), e.node.entity_type.clone(), aliases))
        })
        .collect();
    if entities.iter().all(|(id, _)| id.parse::<u64>().is_ok()) {
        entities.sort_by_key(|(id, _)| id.parse::<u64>().unwrap_or_default());
    }
    entities
}

/// Services over processing_v17 with `allow_llm: false`, `embedder` as BGE and `store`.
fn services(embedder: Arc<dyn Embedder>, store: Arc<dyn GraphStore>) -> StageBuildContext {
    let mut config = default_config();
    for stage in &mut config.stages {
        if let StageConfig::Ed(ed) = stage {
            ed.allow_llm = false;
        }
    }
    let registry = ModelRegistry::empty(config.name());
    registry.insert_embedder(registry.key(ModelKind::Bge), embedder);
    StageBuildContext::new(Arc::new(config), Arc::new(registry), store)
}

fn ed_stage(services: &StageBuildContext) -> EdStage {
    let block = services
        .config
        .stages
        .iter()
        .find_map(|s| match s {
            StageConfig::Ed(c) => Some(c.clone()),
            _ => None,
        })
        .expect("processing_v17 has an EDStage");
    EdStage::new(&block, services).expect("stage builds")
}

/// Runs the Rust EDStage over the scenario's documents in order against one store.
fn rust_decisions(
    docs: &[PyDoc],
    embedder: &Arc<dyn Embedder>,
    store: &Arc<dyn GraphStore>,
) -> Vec<DocDecisions> {
    let services = services(Arc::clone(embedder), Arc::clone(store));
    let stage = ed_stage(&services);
    docs.iter()
        .map(|doc| {
            let mut ctx = PipelineContext::new(&services);
            doc.seed_context_after_coref(&mut ctx);
            ctx.data.embedder = Some(SharedEmbedder(Arc::clone(embedder)));
            stage.run(&mut ctx).expect("ED runs");
            let entities = store_entities(store.as_ref());
            let names = labels(&entities);
            let name_of = |id: &str| names.get(id).cloned().unwrap_or_else(|| format!("<{id}>"));
            let results = ctx
                .data
                .ed_results
                .clone()
                .unwrap_or_default()
                .into_iter()
                .map(|r| {
                    (
                        r.mention_id.map(|id| id.as_str().to_owned()).unwrap_or_default(),
                        r.strategy_used,
                        r.chosen_entity_id.map(|id| name_of(id.as_str())),
                        r.resolved_by_llm,
                    )
                })
                .collect();
            let resolved = ctx
                .data
                .resolved_mentions
                .clone()
                .unwrap_or_default()
                .into_iter()
                .map(|(mention, entity)| (mention.as_str().to_owned(), name_of(entity.as_str())))
                .collect();
            DocDecisions { results, resolved, entities_after: summaries(&entities) }
        })
        .collect()
}

fn assert_same(scenario_name: &str, docs: &[PyDoc], rust: &[DocDecisions]) {
    for (doc, rust) in docs.iter().zip(rust) {
        let python = python_decisions(doc);
        let at = format!("{scenario_name} {}", doc.input);
        assert_eq!(rust.results, python.results, "{at}: ed_results");
        assert_eq!(rust.resolved, python.resolved, "{at}: resolved_mentions");
        assert_eq!(rust.entities_after, python.entities_after, "{at}: entities after the document");
    }
}

/// The real BGE model through fastembed (`FASTEMBED_CACHE_DIR`, else `.fastembed_cache` at
/// the workspace root).
#[cfg(feature = "fastembed")]
fn bge() -> Arc<dyn Embedder> {
    let cache_dir = std::env::var_os("FASTEMBED_CACHE_DIR").map_or_else(
        || std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.fastembed_cache"),
        std::path::PathBuf::from,
    );
    let options =
        impeach_models::FastEmbedOptions { cache_dir: Some(cache_dir), ..Default::default() };
    Arc::new(impeach_models::FastEmbedder::new(&options).expect("fastembed BGE loads"))
}

#[cfg(feature = "fastembed")]
fn assert_bge_parity(name: &str) -> Vec<(usize, usize, usize, usize)> {
    let docs = scenario(name);
    let rust = {
        let store: Arc<dyn GraphStore> = Arc::new(MemoryStore::new(DIM));
        rust_decisions(&docs, &bge(), &store)
    };
    assert_same(name, &docs, &rust);
    rust.iter().map(DocDecisions::counts).collect()
}

#[cfg(feature = "fastembed")]
#[test]
#[ignore = "needs the fastembed BGE model (BAAI/bge-small-en-v1.5) in FASTEMBED_CACHE_DIR or .fastembed_cache"]
fn green_valley_ed_decisions_match_python_with_bge() {
    assert_eq!(assert_bge_parity("green_valley"), [(37, 22, 130, 22), (36, 14, 104, 36)]);
}

#[cfg(feature = "fastembed")]
#[test]
#[ignore = "needs the fastembed BGE model (BAAI/bge-small-en-v1.5) in FASTEMBED_CACHE_DIR or .fastembed_cache"]
fn mason_ed_decisions_match_python_with_bge() {
    assert_eq!(assert_bge_parity("mason"), [(6, 6, 10, 6), (6, 2, 10, 8)]);
}

/// Runs the scenario with a `MockEmbedder` (no embedding candidate reaches the similarity
/// floors), so only the rule, exact, BM25 and intra-document decisions take effect.
fn mock_decisions(name: &str) -> (Vec<PyDoc>, Vec<DocDecisions>) {
    let docs = scenario(name);
    let rust = {
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(DIM));
        let store: Arc<dyn GraphStore> = Arc::new(MemoryStore::new(DIM));
        rust_decisions(&docs, &embedder, &store)
    };
    (docs, rust)
}

#[test]
fn green_valley_ed_decisions_match_python_without_embeddings() {
    let (docs, rust) = mock_decisions("green_valley");
    // Every Green Valley decision is rule, exact, BM25 or intra-document based.
    assert_same("green_valley", &docs, &rust);
}

#[test]
fn mason_ed_decisions_match_python_without_embeddings() {
    let (docs, rust) = mock_decisions("mason");
    let mut expected: Vec<DocDecisions> = docs.iter().map(python_decisions).collect();
    // Python reuses Mason for "Watson" through an embedding candidate (`vector_fuzzy`); without
    // embeddings the hybrid route finds nothing and a Watson entity is created.
    let watson = expected[1]
        .results
        .iter_mut()
        .find(|(mention, _, _, _)| mention == "ent_002_0_2")
        .expect("Watson result");
    assert_eq!((watson.1.as_str(), watson.2.as_deref()), ("vector_fuzzy", Some("Mason (PER)")));
    watson.1 = "hybrid_unresolved_create".to_owned();
    watson.2 = Some("Watson (PER)".to_owned());
    expected[1].resolved.insert("ent_002_0_2".to_owned(), "Watson (PER)".to_owned());
    expected[1].entities_after.push(("Watson".to_owned(), "PER".to_owned(), Vec::new()));
    assert_eq!(rust, expected);
}
