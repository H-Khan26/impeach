//! Similar-fact retrieval for the cross-document check (Python
//! `services/contradictions/retrieval.py`).
//!
//! Sources, in order: graph (facts linked to the given entities by `FactAboutEntity`), BM25
//! over fact text and vector search over fact embeddings; BM25 / vector run according to
//! `fact_retrieval_strategy`. Candidates are deduplicated by fact id (first source wins the
//! insertion slot, later sources are appended to `sources`), scored by the highest weight of
//! their sources times `multi_source_boost` when found by several, filtered by `min_score`
//! (when positive), sorted by score (stable, descending) and capped at `max_candidates`.
//!
//! Each source fails on its own: a non-fatal store error or an embedding failure is logged
//! and that source (or that entity) contributes nothing; fatal store errors propagate.
//!
//! Graph lookups (facts of an entity, entities of a fact) go through a [`GraphLookupCache`]
//! owned by the caller for one contradiction check, during which the graph is not written.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use ce_config::pipeline::FactRetrievalStrategy;
use ce_config::services::SimilarFactRetrieverConfig;
use ce_core::{EntityId, ExtractedFact, FactId, Metadata, RetrievedFact};
use ce_models::Embedder;
use ce_store::{GraphStore, StoreError, StoredFact};
use indexmap::IndexMap;
use serde_json::Value;

use crate::error::Result;
use crate::prefilter::CandidateEntityMap;

/// Retrieval source name of graph (shared entity) hits.
pub const SOURCE_GRAPH: &str = "graph";
/// Retrieval source name of BM25 hits.
pub const SOURCE_BM25: &str = "bm25";
/// Retrieval source name of vector hits.
pub const SOURCE_VECTOR: &str = "vector";

/// Fact node properties copied into a retrieved fact's metadata, in this order.
const NODE_METADATA_KEYS: [&str; 4] = ["FactType", "FactOrigin", "Status", "Certainty"];

/// Retrieves facts already in the graph that might contradict a new fact.
#[derive(Clone)]
pub struct SimilarFactRetriever {
    store: Arc<dyn GraphStore>,
    embedder: Arc<dyn Embedder>,
    config: SimilarFactRetrieverConfig,
}

impl fmt::Debug for SimilarFactRetriever {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimilarFactRetriever")
            .field("store", &self.store)
            .field("embedder", &self.embedder.model_name())
            .field("config", &self.config)
            .finish()
    }
}

/// Candidates by fact id in insertion order.
type Candidates = IndexMap<FactId, RetrievedFact>;

/// Graph lookups memoised for the duration of one contradiction check. Only successful
/// lookups are kept; a recovered store failure is retried on the next lookup.
#[derive(Debug, Default)]
pub struct GraphLookupCache {
    /// Facts linked to an entity, as retrieved facts without sources.
    entity_facts: HashMap<EntityId, Vec<RetrievedFact>>,
    /// Entities a fact is linked to.
    fact_entities: HashMap<FactId, Vec<EntityId>>,
}

impl SimilarFactRetriever {
    /// A retriever over `store`, embedding queries with `embedder`.
    #[must_use]
    pub fn new(
        store: Arc<dyn GraphStore>,
        embedder: Arc<dyn Embedder>,
        config: SimilarFactRetrieverConfig,
    ) -> Self {
        Self { store, embedder, config }
    }

    /// The retrieval settings.
    #[must_use]
    pub fn config(&self) -> &SimilarFactRetrieverConfig {
        &self.config
    }

    /// Candidates for `new_fact`: graph hits for each of `entity_ids` (memoised in `cache`),
    /// then BM25 / vector hits for its text, minus `exclude_fact_ids`, ranked and capped
    /// (module docs).
    ///
    /// # Errors
    /// [`crate::ContradictionError::Store`] for a fatal store error; other store and embedding
    /// failures are logged and that source contributes nothing.
    pub fn retrieve_similar_facts(
        &self,
        new_fact: &ExtractedFact,
        entity_ids: &[EntityId],
        exclude_fact_ids: &[FactId],
        cache: &mut GraphLookupCache,
    ) -> Result<Vec<RetrievedFact>> {
        let mut candidates = Candidates::new();
        let mut add = |facts: &[RetrievedFact], source: &str| {
            for fact in facts {
                if !exclude_fact_ids.contains(&fact.fact_id) {
                    add_candidate(&mut candidates, fact, source);
                }
            }
        };

        for entity_id in entity_ids {
            add(self.entity_facts(entity_id, cache)?, SOURCE_GRAPH);
        }

        let strategy = self.config.fact_retrieval_strategy;
        if matches!(strategy, FactRetrievalStrategy::Bm25 | FactRetrievalStrategy::Both) {
            let hits = recover(
                self.store.find_facts_by_text(&new_fact.text, self.config.bm25_limit),
                "BM25 search failed",
            )?;
            add(&nodes_to_facts(hits.iter().map(|hit| &hit.node)), SOURCE_BM25);
        }

        if matches!(strategy, FactRetrievalStrategy::Vector | FactRetrievalStrategy::Both) {
            if let Some(vector) = self.embed_query(&new_fact.text) {
                let hits = recover(
                    self.store.find_similar_facts(&vector, self.config.vector_limit),
                    "Vector search failed",
                )?;
                add(&nodes_to_facts(hits.iter().map(|hit| &hit.node)), SOURCE_VECTOR);
            }
        }

        Ok(self.rank_and_cap(candidates.into_values().collect()))
    }

    /// Entity ids each candidate is linked to by `FactAboutEntity` edges
    /// ([`GraphStore::get_fact_entities`], memoised in `cache`), keyed by candidate fact id in
    /// candidate order. Candidates without links are omitted. This is the
    /// `candidate_entity_map` the gating receives (Python always passed an empty map).
    ///
    /// # Errors
    /// [`crate::ContradictionError::Store`] for a fatal store error; other failures are
    /// logged and leave that candidate out.
    pub fn candidate_entity_map(
        &self,
        candidates: &[RetrievedFact],
        cache: &mut GraphLookupCache,
    ) -> Result<CandidateEntityMap> {
        let mut map = CandidateEntityMap::new();
        for candidate in candidates {
            let entities = match cache.fact_entities.entry(candidate.fact_id.clone()) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => match self.store.get_fact_entities(&candidate.fact_id) {
                    Ok(entities) => entry.insert(entities),
                    Err(error) => {
                        let context =
                            format!("Failed to get entities for fact {}", candidate.fact_id);
                        skip_error(error, &context)?;
                        continue;
                    }
                },
            };
            if !entities.is_empty() {
                map.insert(candidate.fact_id.clone(), entities.clone());
            }
        }
        Ok(map)
    }

    /// The facts linked to `entity_id`, fetched once per `cache`; empty (logged, not cached)
    /// for a non-fatal store error.
    fn entity_facts<'c>(
        &self,
        entity_id: &EntityId,
        cache: &'c mut GraphLookupCache,
    ) -> Result<&'c [RetrievedFact]> {
        match cache.entity_facts.entry(entity_id.clone()) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => match self.store.get_entity_facts(entity_id) {
                Ok(stored) => Ok(entry.insert(nodes_to_facts(stored.iter()))),
                Err(error) => {
                    skip_error(error, &format!("Failed to get facts for entity {entity_id}"))?;
                    Ok(&[])
                }
            },
        }
    }

    /// The query embedding of `text`; `None` (logged) when embedding fails or returns nothing.
    fn embed_query(&self, text: &str) -> Option<Vec<f32>> {
        match self.embedder.embed(&[text]) {
            Ok(vectors) => vectors.into_iter().next(),
            Err(error) => {
                tracing::warn!("Failed to embed query for vector search: {error}");
                None
            }
        }
    }

    /// Scores by best source weight (boosted for several sources), applies `min_score` when
    /// positive, sorts (stable, descending) and caps at `max_candidates`.
    fn rank_and_cap(&self, mut candidates: Vec<RetrievedFact>) -> Vec<RetrievedFact> {
        let config = &self.config;
        for candidate in &mut candidates {
            let has = |source: &str| candidate.sources.iter().any(|s| s == source);
            let weight = |present: bool, weight: f64| if present { weight } else { 0.0 };
            let mut score = weight(has(SOURCE_GRAPH), config.graph_weight)
                .max(weight(has(SOURCE_BM25), config.bm25_weight))
                .max(weight(has(SOURCE_VECTOR), config.vector_weight));
            if candidate.sources.len() > 1 {
                score *= config.multi_source_boost;
            }
            candidate.relevance_score = score;
        }
        if config.min_score > 0.0 {
            candidates.retain(|c| c.relevance_score >= config.min_score);
        }
        candidates.sort_by(|a, b| b.relevance_score.total_cmp(&a.relevance_score));
        candidates.truncate(config.max_candidates);
        candidates
    }
}

/// Inserts `fact` found by `source`, or records `source` on the existing candidate.
fn add_candidate(candidates: &mut Candidates, fact: &RetrievedFact, source: &str) {
    if let Some(existing) = candidates.get_mut(&fact.fact_id) {
        if !existing.sources.iter().any(|s| s == source) {
            existing.sources.push(source.to_owned());
        }
        return;
    }
    let mut fact = fact.clone();
    fact.sources = vec![source.to_owned()];
    candidates.insert(fact.fact_id.clone(), fact);
}

/// Stored fact nodes as retrieved facts, skipping nodes without text.
fn nodes_to_facts<'a>(nodes: impl Iterator<Item = &'a StoredFact>) -> Vec<RetrievedFact> {
    nodes.filter_map(retrieved_fact_from_node).collect()
}

/// The store result, or an empty list (logged) for a non-fatal store error.
fn recover<T>(result: std::result::Result<Vec<T>, StoreError>, context: &str) -> Result<Vec<T>> {
    result.or_else(|error| skip_error(error, context).map(|()| Vec::new()))
}

/// Propagates a fatal store error; logs any other.
fn skip_error(error: StoreError, context: &str) -> Result<()> {
    if error.is_fatal() {
        return Err(error.into());
    }
    tracing::warn!("{context}: {error}");
    Ok(())
}

/// A stored fact node as a [`RetrievedFact`] with no sources and score 0.0: the parsed
/// `AssertionTriple` (subject / relationship / object), the parsed `Metadata` JSON object
/// plus the node's `FactType`, `FactOrigin`, `Status` and `Certainty` under those keys, and
/// the side from metadata `side` or `er.side` ([`ce_text::normalize_side`]). `None` for a
/// node without text.
///
/// The node's `ValueType` / `PropertyType` are not copied (as in Python), so a candidate
/// carries `value_type` / `property_type` only when its stored `Metadata` JSON has them.
/// Unparseable JSON fields are treated as empty.
#[must_use]
pub fn retrieved_fact_from_node(stored: &StoredFact) -> Option<RetrievedFact> {
    let node = &stored.node;
    if node.fact_text.is_empty() {
        return None;
    }
    let mut fact = RetrievedFact::new(stored.id.clone(), node.fact_text.clone());

    if !node.assertion_triple.is_empty() && node.assertion_triple != "{}" {
        if let Ok(Value::Object(triple)) = serde_json::from_str::<Value>(&node.assertion_triple) {
            let field = |key: &str| triple.get(key).and_then(Value::as_str).map(str::to_owned);
            fact.assertion_subject = field("subject");
            fact.assertion_relationship = field("relationship");
            fact.assertion_object = field("object");
        }
    }

    let mut metadata = match serde_json::from_str::<Value>(&node.metadata) {
        Ok(Value::Object(map)) => map,
        _ => Metadata::new(),
    };
    let node_values = [
        Value::from(node.fact_type.as_str()),
        Value::from(node.fact_origin.as_str()),
        Value::from(node.status.as_str()),
        Value::from(node.certainty),
    ];
    for (key, value) in NODE_METADATA_KEYS.into_iter().zip(node_values) {
        metadata.insert(key.to_owned(), value);
    }

    fact.side = side_from_metadata(&metadata).map(str::to_owned);
    fact.metadata = metadata;
    Some(fact)
}

/// The normalised side from metadata `side`, else `er.side`, else the default (`NEUTRAL`).
fn side_from_metadata(metadata: &Metadata) -> Option<&'static str> {
    if let Some(side) = metadata.get("side") {
        return normalize_side_value(side);
    }
    if let Some(side) = metadata.get("er").and_then(Value::as_object).and_then(|er| er.get("side"))
    {
        return normalize_side_value(side);
    }
    ce_text::normalize_side(None)
}

/// [`ce_text::normalize_side`] of a JSON value: null is missing, non-string values are
/// unknown sides.
fn normalize_side_value(value: &Value) -> Option<&'static str> {
    match value {
        Value::Null => ce_text::normalize_side(None),
        Value::String(side) => ce_text::normalize_side(Some(side)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ce_store::{FactNode, Stored};
    use serde_json::json;

    fn stored(node: FactNode) -> StoredFact {
        Stored { id: FactId::new("7"), node }
    }

    #[test]
    fn node_without_text_is_skipped() {
        assert!(retrieved_fact_from_node(&stored(FactNode::default())).is_none());
    }

    #[test]
    fn node_fields_are_parsed() {
        let mut node = FactNode::new(
            "Driver ran the red light",
            r#"{"subject": "Driver", "relationship": "RAN", "object": "red light"}"#,
            "{}",
            0.9,
        );
        node.metadata =
            r#"{"value_type": "TEXT", "er": {"side": "defendant"}, "Status": "x"}"#.into();
        let fact = retrieved_fact_from_node(&stored(node)).expect("fact has text");
        assert_eq!(fact.fact_id.as_str(), "7");
        assert_eq!(fact.assertion_subject.as_deref(), Some("Driver"));
        assert_eq!(fact.assertion_relationship.as_deref(), Some("RAN"));
        assert_eq!(fact.assertion_object.as_deref(), Some("red light"));
        assert_eq!(fact.side.as_deref(), Some("DEFENSE"));
        assert!(fact.sources.is_empty());
        assert_eq!(fact.relevance_score, 0.0);
        // Existing keys keep their position; node values overwrite them.
        let keys: Vec<&str> = fact.metadata.keys().map(String::as_str).collect();
        assert_eq!(keys, ["value_type", "er", "Status", "FactType", "FactOrigin", "Certainty"]);
        assert_eq!(fact.metadata["Status"], json!("RAW"));
        assert_eq!(fact.metadata["FactType"], json!("STATED"));
        assert_eq!(fact.metadata["Certainty"], json!(0.9));
    }

    #[test]
    fn bad_json_fields_are_empty() {
        let mut node = FactNode::new("Some fact", "not json", "{}", 0.5);
        node.metadata = "[1, 2]".into();
        let fact = retrieved_fact_from_node(&stored(node)).expect("fact has text");
        assert_eq!(fact.assertion_subject, None);
        assert_eq!(fact.metadata.len(), 4);
        assert_eq!(fact.side.as_deref(), Some("NEUTRAL"));
    }

    #[test]
    fn side_sources_and_defaults() {
        let side = |value: Value| {
            let Value::Object(map) = value else { unreachable!("object literal") };
            side_from_metadata(&map)
        };
        assert_eq!(side(json!({"side": "plaintiff"})), Some("PLAINTIFF"));
        assert_eq!(side(json!({"side": null})), Some("NEUTRAL"));
        assert_eq!(side(json!({"side": "judge"})), None);
        assert_eq!(side(json!({"side": 3})), None);
        assert_eq!(side(json!({"side": "state", "er": {"side": "defense"}})), Some("PROSECUTION"));
        assert_eq!(side(json!({"er": {"side": "defence"}})), Some("DEFENSE"));
        assert_eq!(side(json!({"er": "x"})), Some("NEUTRAL"));
        assert_eq!(side(json!({})), Some("NEUTRAL"));
    }

    #[test]
    fn recover_propagates_only_fatal_errors() {
        let lost: std::result::Result<Vec<u8>, StoreError> =
            Err(StoreError::DatabaseLost("closed".into()));
        assert!(recover(lost, "ctx").is_err());
        let failed: std::result::Result<Vec<u8>, StoreError> =
            Err(StoreError::Connection("busy".into()));
        assert_eq!(recover(failed, "ctx").expect("non-fatal is recovered"), Vec::<u8>::new());
        assert_eq!(recover(Ok(vec![1_u8]), "ctx").expect("ok passes"), vec![1]);
    }
}
