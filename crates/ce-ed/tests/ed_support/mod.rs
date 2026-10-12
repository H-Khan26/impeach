//! Shared fixtures of the resolver, entity creator, service and cross-document tests.

#![allow(dead_code)]

use std::sync::Arc;

use ce_config::pipeline::EdStageConfig;
use ce_config::{MappingsRegistry, StageConfig};
use ce_core::{DocContext, EdContext, EntityId, LocalContext, Mention, Span};
use ce_ed::{EntityCreator, NameSimilarity};
use ce_models::Embedder;
use ce_ner::definitions::RoleConfig;
use ce_store::{EntityNode, GraphStore, MemoryStore};

pub const DIM: usize = 8;

pub fn mappings() -> &'static MappingsRegistry {
    MappingsRegistry::load().expect("mappings load")
}

pub fn similarity() -> NameSimilarity {
    NameSimilarity::from_mappings(mappings())
}

pub fn roles() -> RoleConfig {
    RoleConfig::load(None).expect("roles.yaml loads")
}

/// The `EDStage` block of the shipped default config.
pub fn ed_config() -> EdStageConfig {
    ce_config::load_default_config()
        .expect("default config")
        .stages
        .into_iter()
        .find_map(|stage| match stage {
            StageConfig::Ed(ed) => Some(ed),
            _ => None,
        })
        .expect("default config has an EDStage")
}

/// A fresh memory store, also as a trait object.
pub fn store() -> (Arc<MemoryStore>, Arc<dyn GraphStore>) {
    let store = Arc::new(MemoryStore::new(DIM));
    let dyn_store: Arc<dyn GraphStore> = store.clone();
    (store, dyn_store)
}

/// An entity creator with role blocking over `store`.
pub fn creator(store: &Arc<dyn GraphStore>, embedder: Option<Arc<dyn Embedder>>) -> EntityCreator {
    EntityCreator::new(Arc::clone(store), embedder, similarity(), Some(roles()))
}

/// Python `_make_ed_context`: the mention spans its whole name; the sentence is
/// `"{name} did something."`.
pub fn ed_context(
    name: &str,
    label: &str,
    case_id: &str,
    doc_id: &str,
    mention_id: &str,
) -> EdContext {
    let mut doc = DocContext::new(doc_id, "complaint");
    doc.case_id = Some(case_id.to_owned());
    let sentence = format!("{name} did something.");
    EdContext {
        mention: Mention::new(mention_id, label, name, Span::new(0, name.chars().count())),
        local_context: LocalContext::new(sentence.as_str(), sentence.as_str()),
        doc_context: doc,
        span_label: label.to_owned(),
        normalized_key: None,
    }
}

/// Adds an entity with raw `AliasesJson` and `Metadata`.
pub fn seed_entity(
    store: &MemoryStore,
    name: &str,
    entity_type: &str,
    aliases_json: &str,
    metadata: &str,
) -> EntityId {
    let node = EntityNode {
        name: name.to_owned(),
        entity_type: entity_type.to_owned(),
        aliases_json: aliases_json.to_owned(),
        description: String::new(),
        metadata: metadata.to_owned(),
    };
    store.create_entity(&node).expect("entity created").id
}

/// Every entity of the store.
pub fn entities(store: &MemoryStore) -> Vec<ce_store::StoredEntity> {
    store.get_all_entities(1000).expect("entities")
}
