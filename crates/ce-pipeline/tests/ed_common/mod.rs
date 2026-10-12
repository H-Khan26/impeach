//! Models for the ED and relation stage tests: a gazetteer NER model that finds fixed names in
//! any text, and a processing environment over one shared store.

#![allow(dead_code)]

use std::sync::Arc;

use ce_models::{
    MockEmbedder, ModelError, ModelKind, ModelRegistry, NerModel, NerSpan, RelationHit, RuleCoref,
};
use ce_pipeline::ProcessingEnv;
use ce_store::{GraphStore, MemoryStore};

pub const DIM: usize = 384;

/// Finds every occurrence of its `(surface, label)` entries (character offsets, score 0.9)
/// and returns relations whose head and tail texts both occur.
#[derive(Debug, Default)]
pub struct GazetteerNer {
    entries: Vec<(String, String)>,
    relations: Vec<(String, String, String, f32)>,
}

impl GazetteerNer {
    pub fn new(entries: &[(&str, &str)]) -> Self {
        Self {
            entries: entries.iter().map(|(s, l)| ((*s).to_owned(), (*l).to_owned())).collect(),
            relations: Vec::new(),
        }
    }

    /// Adds a relation `label(head, tail)` reported for texts containing both names.
    pub fn with_relation(mut self, head: &str, label: &str, tail: &str, score: f32) -> Self {
        self.relations.push((head.to_owned(), label.to_owned(), tail.to_owned(), score));
        self
    }

    fn occurrences(text: &str, surface: &str, label: &str) -> Vec<NerSpan> {
        text.match_indices(surface)
            .map(|(byte, _)| {
                let start = text[..byte].chars().count();
                NerSpan {
                    start,
                    end: start + surface.chars().count(),
                    label: label.to_owned(),
                    score: 0.9,
                    text: surface.to_owned(),
                }
            })
            .collect()
    }
}

impl NerModel for GazetteerNer {
    fn extract_entities(
        &self,
        text: &str,
        labels: &[&str],
        threshold: f32,
    ) -> Result<Vec<NerSpan>, ModelError> {
        if threshold > 0.9 {
            return Ok(Vec::new());
        }
        let mut spans: Vec<NerSpan> = self
            .entries
            .iter()
            .filter(|(_, label)| labels.contains(&label.as_str()))
            .flat_map(|(surface, label)| Self::occurrences(text, surface, label))
            .collect();
        spans.sort_by_key(|s| (s.start, std::cmp::Reverse(s.end)));
        Ok(spans)
    }

    fn extract_relations(
        &self,
        text: &str,
        labels: &[&str],
        threshold: f32,
    ) -> Result<Vec<RelationHit>, ModelError> {
        Ok(self
            .relations
            .iter()
            .filter(|(_, label, _, score)| *score >= threshold && labels.contains(&label.as_str()))
            .filter_map(|(head, label, tail, score)| {
                let head = Self::occurrences(text, head, "head").into_iter().next()?;
                let tail = Self::occurrences(text, tail, "tail").into_iter().next()?;
                Some(RelationHit { head, tail, label: label.clone(), score: *score })
            })
            .collect())
    }
}

/// Names of the two Green Valley documents with GLiNER2 labels.
pub const GREEN_VALLEY_NAMES: &[(&str, &str)] = &[
    ("Maria Lopez", "person"),
    ("Daniel Reed", "person"),
    ("Lopez", "person"),
    ("Reed", "person"),
    ("Green Valley Logistics, Inc.", "company"),
    ("Green Valley", "organization"),
    ("San Francisco Police Department", "organization"),
    ("San Francisco", "location"),
    ("Mission Street", "location"),
    ("14th Avenue", "location"),
    ("Honda Civic", "vehicle"),
    ("Freightliner box truck", "vehicle"),
];

/// Registry with a `MockEmbedder`, `ner` as the GLiNER2 model and rule coreference; no LLM.
pub fn registry(ner: Arc<dyn NerModel>) -> Arc<ModelRegistry> {
    let registry = ModelRegistry::empty("processing_v17");
    registry.insert_embedder(registry.key(ModelKind::Bge), Arc::new(MockEmbedder::new(DIM)));
    registry.insert_ner(registry.key(ModelKind::Gliner2), ner);
    registry.insert_coref(registry.key(ModelKind::Coref), Arc::new(RuleCoref::new()));
    Arc::new(registry)
}

/// A processing environment over `store` with the built-in stages.
pub fn env(store: &Arc<MemoryStore>, ner: Arc<dyn NerModel>) -> ProcessingEnv {
    let dyn_store: Arc<dyn GraphStore> = store.clone();
    ProcessingEnv {
        registry: Some(registry(ner)),
        store: Some(dyn_store),
        stages: None,
        unregistered: None,
    }
}
