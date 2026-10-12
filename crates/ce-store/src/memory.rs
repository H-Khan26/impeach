//! [`MemoryStore`]: a complete in-process [`GraphStore`] for tests and model-free runs.
//!
//! Nodes and edges live in ordered maps behind one `RwLock`; every method holds the lock for
//! its whole duration, so each call is atomic. Vector search is brute-force cosine over the
//! nodes that have a vector; text search is the Okapi BM25 of [`crate::bm25`] computed over
//! the current nodes of the label, one corpus per indexed property as in HelixDB. Node ids
//! are sequential integers shared by all labels and never reused, also across
//! [`GraphStore::reset`].

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use ce_core::{ChunkId, EntityId, FactId, NodeId, SpanId};
use serde_json::{json, Map, Value};

use crate::bm25;
use crate::error::{Result, StoreError};
use crate::graph::{
    clamp_limit, merge_hits, or_default, parse_node_id, validate_confidence, validate_vector,
    ChunkNode, DocumentNode, EmbeddingWrite, EntityHit, EntityNode, FactHit, FactNode, SearchHit,
    SpanNode, Stored, StoredChunk, StoredDocument, StoredEntity, StoredFact, StoredSpan,
    VectorLabel, CHUNK_OF_DOCUMENT, DEFAULT_FACT_ENTITY_ROLE, DEFAULT_FACT_SPAN_ROLE,
    DEFAULT_SPAN_ENTITY_METHOD, DOCUMENT, DOCUMENT_CHUNK, ENTITY, ENTITY_NAME_SEARCH_K, FACT,
    FACT_ABOUT_ENTITY, FACT_DERIVED_FROM_SPAN, FACT_IN_DOCUMENT, SPAN, SPAN_OF_CHUNK,
    SPAN_REFERS_TO_ENTITY,
};
use crate::store::GraphStore;

/// One edge as stored by [`MemoryStore`], for inspection in tests.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeRecord {
    /// Edge label.
    pub label: &'static str,
    /// Source node id.
    pub from: String,
    /// Target node id.
    pub to: String,
    /// Edge properties.
    pub properties: Map<String, Value>,
}

#[derive(Debug, Clone)]
enum NodeData {
    Document(DocumentNode),
    Chunk(ChunkNode),
    Span(SpanNode),
    Entity(EntityNode),
    Fact(FactNode),
}

impl NodeData {
    fn label(&self) -> &'static str {
        match self {
            Self::Document(_) => DOCUMENT,
            Self::Chunk(_) => DOCUMENT_CHUNK,
            Self::Span(_) => SPAN,
            Self::Entity(_) => ENTITY,
            Self::Fact(_) => FACT,
        }
    }
}

#[derive(Debug, Clone)]
struct Node {
    data: NodeData,
    embedding: Option<Vec<f32>>,
}

#[derive(Debug, Default)]
struct Graph {
    next_id: u64,
    store_id: String,
    nodes: BTreeMap<u64, Node>,
    edges: Vec<EdgeRecord>,
    /// Token lists of text-indexed properties, keyed by node id, one map per index.
    entity_names: BTreeMap<u64, Vec<String>>,
    entity_aliases: BTreeMap<u64, Vec<String>>,
    fact_texts: BTreeMap<u64, Vec<String>>,
}

impl Graph {
    fn insert(&mut self, data: NodeData) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        match &data {
            NodeData::Entity(entity) => {
                self.entity_names.insert(id, bm25::tokenize(&entity.name));
                self.entity_aliases.insert(id, bm25::tokenize(&entity.aliases_json));
            }
            NodeData::Fact(fact) => {
                self.fact_texts.insert(id, bm25::tokenize(&fact.fact_text));
            }
            NodeData::Document(_) | NodeData::Chunk(_) | NodeData::Span(_) => {}
        }
        self.nodes.insert(id, Node { data, embedding: None });
        id
    }

    /// The node `id` if it carries `label`.
    fn get(&self, label: &str, id: u64) -> Option<&Node> {
        self.nodes.get(&id).filter(|node| node.data.label() == label)
    }

    fn get_mut(&mut self, label: &str, id: u64) -> Option<&mut Node> {
        self.nodes.get_mut(&id).filter(|node| node.data.label() == label)
    }

    /// Parses `value` and checks that it names a `label` node.
    fn require(&self, label: &str, field: &str, value: &str) -> Result<u64> {
        let id = parse_node_id(field, value)?;
        if self.get(label, id).is_none() {
            return Err(StoreError::not_found(label, value));
        }
        Ok(id)
    }

    fn add_edge(&mut self, label: &'static str, from: u64, to: u64, properties: Value) {
        let properties = match properties {
            Value::Object(map) => map,
            _ => Map::new(),
        };
        self.edges.push(EdgeRecord {
            label,
            from: from.to_string(),
            to: to.to_string(),
            properties,
        });
    }

    /// Distinct sources of `label` edges pointing at `to`, in first-edge order (a HelixDB
    /// `in_` step yields each neighbour once).
    fn sources(&self, label: &str, to: u64) -> Vec<u64> {
        let to = to.to_string();
        let mut sources: Vec<u64> = Vec::new();
        for edge in self.edges.iter().filter(|edge| edge.label == label && edge.to == to) {
            if let Ok(from) = edge.from.parse() {
                if !sources.contains(&from) {
                    sources.push(from);
                }
            }
        }
        sources
    }

    /// Distinct targets of `label` edges leaving `from`, in first-edge order (a HelixDB `out`
    /// step yields each neighbour once).
    fn targets(&self, label: &str, from: u64) -> Vec<u64> {
        let from = from.to_string();
        let mut targets: Vec<u64> = Vec::new();
        for edge in self.edges.iter().filter(|edge| edge.label == label && edge.from == from) {
            if let Ok(to) = edge.to.parse() {
                if !targets.contains(&to) {
                    targets.push(to);
                }
            }
        }
        targets
    }

    fn vector_search(&self, label: &str, query: &[f32], limit: usize) -> Vec<(u64, f64)> {
        let query_norm = norm(query);
        let mut scored: Vec<(u64, f64)> = self
            .nodes
            .iter()
            .filter(|(_, node)| node.data.label() == label)
            .filter_map(|(id, node)| {
                let vector = node.embedding.as_deref()?;
                Some((*id, cosine(query, query_norm, vector)))
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        scored.truncate(limit);
        scored.into_iter().map(|(id, cos)| (id, cos.clamp(0.0, 1.0))).collect()
    }
}

fn norm(vector: &[f32]) -> f64 {
    vector.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt()
}

fn cosine(query: &[f32], query_norm: f64, vector: &[f32]) -> f64 {
    let dot: f64 = query.iter().zip(vector).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
    let denominator = query_norm * norm(vector);
    if denominator == 0.0 {
        0.0
    } else {
        dot / denominator
    }
}

/// BM25 search over one token index; `(node id, score)` best first, ties by id.
fn text_search(index: &BTreeMap<u64, Vec<String>>, query: &str) -> Vec<(u64, f64)> {
    let ids: Vec<u64> = index.keys().copied().collect();
    let corpus: Vec<&[String]> = index.values().map(Vec::as_slice).collect();
    let mut hits: Vec<(u64, f64)> = bm25::score_corpus(&corpus, query)
        .into_iter()
        .map(|(position, score)| (ids[position], score))
        .collect();
    hits.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    hits
}

/// In-memory [`GraphStore`].
#[derive(Debug)]
pub struct MemoryStore {
    embedding_dim: usize,
    graph: RwLock<Graph>,
}

impl MemoryStore {
    /// Empty store whose vector indexes have `embedding_dim` dimensions.
    #[must_use]
    pub fn new(embedding_dim: usize) -> Self {
        let graph = Graph { store_id: new_store_id(), ..Graph::default() };
        Self { embedding_dim, graph: RwLock::new(graph) }
    }

    /// Every edge, in creation order.
    #[must_use]
    pub fn edges(&self) -> Vec<EdgeRecord> {
        self.read().edges.clone()
    }

    /// Number of nodes (excluding the implicit store-info record).
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.read().nodes.len()
    }

    /// Label and properties of node `id` (PascalCase JSON object), for inspection in tests.
    #[must_use]
    pub fn node_snapshot(&self, id: &str) -> Option<(&'static str, Value)> {
        let id = id.trim().parse::<u64>().ok()?;
        let graph = self.read();
        let node = graph.nodes.get(&id)?;
        let properties = match &node.data {
            NodeData::Document(data) => serde_json::to_value(data),
            NodeData::Chunk(data) => serde_json::to_value(data),
            NodeData::Span(data) => serde_json::to_value(data),
            NodeData::Entity(data) => serde_json::to_value(data),
            NodeData::Fact(data) => serde_json::to_value(data),
        }
        .ok()?;
        Some((node.data.label(), properties))
    }

    /// Vector stored on node `id`, if any.
    #[must_use]
    pub fn embedding(&self, id: &str) -> Option<Vec<f32>> {
        let id = id.trim().parse::<u64>().ok()?;
        self.read().nodes.get(&id)?.embedding.clone()
    }

    fn read(&self) -> RwLockReadGuard<'_, Graph> {
        self.graph.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write(&self) -> RwLockWriteGuard<'_, Graph> {
        self.graph.write().unwrap_or_else(PoisonError::into_inner)
    }

    fn set_embedding(&self, label: VectorLabel, id: &str, vector: &[f32]) -> Result<()> {
        validate_vector(vector, self.embedding_dim)?;
        let mut graph = self.write();
        let node_id = graph.require(label.as_str(), label.id_field(), id)?;
        if let Some(node) = graph.get_mut(label.as_str(), node_id) {
            node.embedding = Some(vector.to_vec());
        }
        Ok(())
    }

    fn update(&self, label: &str, field: &str, id: &str, data: NodeData) -> Result<()> {
        let mut graph = self.write();
        let node_id = graph.require(label, field, id)?;
        if let Some(node) = graph.get_mut(label, node_id) {
            node.data = data;
        }
        Ok(())
    }
}

impl Default for MemoryStore {
    /// Store with [`crate::DEFAULT_EMBEDDING_DIM`] dimensions.
    fn default() -> Self {
        Self::new(crate::DEFAULT_EMBEDDING_DIM)
    }
}

fn new_store_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

fn entity(graph: &Graph, id: u64) -> Option<StoredEntity> {
    match &graph.nodes.get(&id)?.data {
        NodeData::Entity(node) => {
            Some(Stored { id: EntityId::new(id.to_string()), node: node.clone() })
        }
        _ => None,
    }
}

fn fact(graph: &Graph, id: u64) -> Option<StoredFact> {
    match &graph.nodes.get(&id)?.data {
        NodeData::Fact(node) => {
            Some(Stored { id: FactId::new(id.to_string()), node: node.clone() })
        }
        _ => None,
    }
}

fn chunk(graph: &Graph, id: u64) -> Option<StoredChunk> {
    match &graph.nodes.get(&id)?.data {
        NodeData::Chunk(node) => {
            Some(Stored { id: ChunkId::new(id.to_string()), node: node.clone() })
        }
        _ => None,
    }
}

fn hits<T>(found: Vec<(u64, f64)>, read: impl Fn(u64) -> Option<T>) -> Vec<SearchHit<T>> {
    found
        .into_iter()
        .filter_map(|(id, score)| read(id).map(|node| SearchHit { node, score }))
        .collect()
}

/// Inserts a span under its chunk; the chunk must exist.
fn insert_span(graph: &mut Graph, span: &SpanNode, chunk_id: u64) -> StoredSpan {
    let id = graph.insert(NodeData::Span(span.clone()));
    graph.add_edge(SPAN_OF_CHUNK, id, chunk_id, json!({"SequenceNumber": span.start_char}));
    Stored { id: SpanId::new(id.to_string()), node: span.clone() }
}

fn insert_chunk(graph: &mut Graph, chunk: &ChunkNode, doc_id: u64) -> StoredChunk {
    let id = graph.insert(NodeData::Chunk(chunk.clone()));
    graph.add_edge(CHUNK_OF_DOCUMENT, id, doc_id, json!({"SequenceNumber": chunk.sequence_number}));
    Stored { id: ChunkId::new(id.to_string()), node: chunk.clone() }
}

impl GraphStore for MemoryStore {
    fn store_id(&self) -> String {
        self.read().store_id.clone()
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn reset(&self) -> Result<()> {
        let mut graph = self.write();
        let next_id = graph.next_id;
        *graph = Graph { next_id, store_id: new_store_id(), ..Graph::default() };
        Ok(())
    }

    fn create_document(&self, document: &DocumentNode) -> Result<StoredDocument> {
        let id = self.write().insert(NodeData::Document(document.clone()));
        Ok(Stored { id: NodeId::new(id.to_string()), node: document.clone() })
    }

    fn update_document(&self, doc_id: &NodeId, document: &DocumentNode) -> Result<StoredDocument> {
        self.update(DOCUMENT, "doc_id", doc_id.as_str(), NodeData::Document(document.clone()))?;
        Ok(Stored { id: doc_id.clone(), node: document.clone() })
    }

    fn get_document_chunks(&self, doc_id: &NodeId) -> Result<Vec<StoredChunk>> {
        let id = parse_node_id("doc_id", doc_id.as_str())?;
        let graph = self.read();
        if graph.get(DOCUMENT, id).is_none() {
            return Ok(Vec::new());
        }
        Ok(graph
            .sources(CHUNK_OF_DOCUMENT, id)
            .into_iter()
            .filter_map(|chunk_id| chunk(&graph, chunk_id))
            .collect())
    }

    fn create_document_chunk(&self, doc_id: &NodeId, chunk: &ChunkNode) -> Result<StoredChunk> {
        let mut graph = self.write();
        let parent = graph.require(DOCUMENT, "doc_id", doc_id.as_str())?;
        Ok(insert_chunk(&mut graph, chunk, parent))
    }

    fn update_document_chunk(&self, chunk_id: &ChunkId, chunk: &ChunkNode) -> Result<StoredChunk> {
        self.update(DOCUMENT_CHUNK, "chunk_id", chunk_id.as_str(), NodeData::Chunk(chunk.clone()))?;
        Ok(Stored { id: chunk_id.clone(), node: chunk.clone() })
    }

    fn create_chunk_embedding(&self, chunk_id: &ChunkId, embedding: &[f32]) -> Result<()> {
        self.set_embedding(VectorLabel::DocumentChunk, chunk_id.as_str(), embedding)
    }

    fn create_span_for_chunk(&self, span: &SpanNode) -> Result<StoredSpan> {
        let mut graph = self.write();
        let parent = graph.require(DOCUMENT_CHUNK, "chunk_id", span.chunk_id.as_str())?;
        Ok(insert_span(&mut graph, span, parent))
    }

    fn create_entity(&self, entity: &EntityNode) -> Result<StoredEntity> {
        let node = entity.validated()?;
        let id = self.write().insert(NodeData::Entity(node.clone()));
        Ok(Stored { id: EntityId::new(id.to_string()), node })
    }

    fn create_entity_embedding(&self, entity_id: &EntityId, embedding: &[f32]) -> Result<()> {
        self.set_embedding(VectorLabel::Entity, entity_id.as_str(), embedding)
    }

    fn find_entity_by_name(&self, name: &str) -> Result<Vec<EntityHit>> {
        let graph = self.read();
        let found = merge_hits(
            vec![text_search(&graph.entity_names, name), text_search(&graph.entity_aliases, name)],
            ENTITY_NAME_SEARCH_K,
        );
        Ok(hits(found, |id| entity(&graph, id)))
    }

    fn find_similar_entities(&self, embedding: &[f32], limit: usize) -> Result<Vec<EntityHit>> {
        validate_vector(embedding, self.embedding_dim)?;
        let limit = clamp_limit(limit);
        let graph = self.read();
        Ok(hits(graph.vector_search(ENTITY, embedding, limit), |id| entity(&graph, id)))
    }

    fn get_all_entities(&self, limit: usize) -> Result<Vec<StoredEntity>> {
        let graph = self.read();
        Ok(graph.nodes.keys().filter_map(|id| entity(&graph, *id)).take(limit).collect())
    }

    fn link_span_to_entity(
        &self,
        span_id: &SpanId,
        entity_id: &EntityId,
        confidence: f64,
        method: &str,
    ) -> Result<()> {
        let confidence = validate_confidence(confidence)?;
        let mut graph = self.write();
        let from = graph.require(SPAN, "span_id", span_id.as_str())?;
        let to = graph.require(ENTITY, "entity_id", entity_id.as_str())?;
        let method = or_default(method, DEFAULT_SPAN_ENTITY_METHOD);
        graph.add_edge(
            SPAN_REFERS_TO_ENTITY,
            from,
            to,
            json!({"Confidence": confidence, "Method": method}),
        );
        Ok(())
    }

    fn create_fact(&self, fact: &FactNode) -> Result<StoredFact> {
        fact.validate()?;
        let id = self.write().insert(NodeData::Fact(fact.clone()));
        Ok(Stored { id: FactId::new(id.to_string()), node: fact.clone() })
    }

    fn create_fact_embedding(&self, fact_id: &FactId, embedding: &[f32]) -> Result<()> {
        self.set_embedding(VectorLabel::Fact, fact_id.as_str(), embedding)
    }

    fn link_fact_to_document(
        &self,
        fact_id: &FactId,
        doc_id: &NodeId,
        page_number: i64,
        section: &str,
    ) -> Result<()> {
        let mut graph = self.write();
        let from = graph.require(FACT, "fact_id", fact_id.as_str())?;
        let to = graph.require(DOCUMENT, "doc_id", doc_id.as_str())?;
        graph.add_edge(
            FACT_IN_DOCUMENT,
            from,
            to,
            json!({"PageNumber": page_number, "Section": section}),
        );
        Ok(())
    }

    fn link_fact_to_entity(
        &self,
        fact_id: &FactId,
        entity_id: &EntityId,
        role: &str,
    ) -> Result<()> {
        let mut graph = self.write();
        let from = graph.require(FACT, "fact_id", fact_id.as_str())?;
        let to = graph.require(ENTITY, "entity_id", entity_id.as_str())?;
        let role = or_default(role, DEFAULT_FACT_ENTITY_ROLE);
        graph.add_edge(FACT_ABOUT_ENTITY, from, to, json!({"Role": role}));
        Ok(())
    }

    fn link_fact_to_span(
        &self,
        fact_id: &FactId,
        span_id: &SpanId,
        role: &str,
        confidence: f64,
    ) -> Result<()> {
        let confidence = validate_confidence(confidence)?;
        let mut graph = self.write();
        let from = graph.require(FACT, "fact_id", fact_id.as_str())?;
        let to = graph.require(SPAN, "span_id", span_id.as_str())?;
        let role = or_default(role, DEFAULT_FACT_SPAN_ROLE);
        graph.add_edge(
            FACT_DERIVED_FROM_SPAN,
            from,
            to,
            json!({"Role": role, "Confidence": confidence}),
        );
        Ok(())
    }

    fn get_entity_facts(&self, entity_id: &EntityId) -> Result<Vec<StoredFact>> {
        let id = parse_node_id("entity_id", entity_id.as_str())?;
        let graph = self.read();
        if graph.get(ENTITY, id).is_none() {
            return Ok(Vec::new());
        }
        Ok(graph
            .sources(FACT_ABOUT_ENTITY, id)
            .into_iter()
            .filter_map(|fact_id| fact(&graph, fact_id))
            .collect())
    }

    fn get_fact_entities(&self, fact_id: &FactId) -> Result<Vec<EntityId>> {
        let id = parse_node_id("fact_id", fact_id.as_str())?;
        let graph = self.read();
        if graph.get(FACT, id).is_none() {
            return Ok(Vec::new());
        }
        Ok(graph
            .targets(FACT_ABOUT_ENTITY, id)
            .into_iter()
            .filter(|entity_id| graph.get(ENTITY, *entity_id).is_some())
            .map(|entity_id| EntityId::new(entity_id.to_string()))
            .collect())
    }

    fn find_facts_by_text(&self, keywords: &str, limit: usize) -> Result<Vec<FactHit>> {
        let limit = clamp_limit(limit);
        let graph = self.read();
        let found = merge_hits(vec![text_search(&graph.fact_texts, keywords)], limit);
        Ok(hits(found, |id| fact(&graph, id)))
    }

    fn find_similar_facts(&self, embedding: &[f32], limit: usize) -> Result<Vec<FactHit>> {
        validate_vector(embedding, self.embedding_dim)?;
        let limit = clamp_limit(limit);
        let graph = self.read();
        Ok(hits(graph.vector_search(FACT, embedding, limit), |id| fact(&graph, id)))
    }

    fn create_chunks_batch(
        &self,
        doc_id: &NodeId,
        chunks: &[ChunkNode],
    ) -> Result<Vec<StoredChunk>> {
        if chunks.is_empty() {
            parse_node_id("doc_id", doc_id.as_str())?;
            return Ok(Vec::new());
        }
        let mut graph = self.write();
        let parent = graph.require(DOCUMENT, "doc_id", doc_id.as_str())?;
        Ok(chunks.iter().map(|chunk| insert_chunk(&mut graph, chunk, parent)).collect())
    }

    fn delete_document_chunks(&self, doc_id: &NodeId) -> Result<usize> {
        let id = parse_node_id("doc_id", doc_id.as_str())?;
        let mut graph = self.write();
        if graph.get(DOCUMENT, id).is_none() {
            return Ok(0);
        }
        let chunks: Vec<u64> = graph
            .sources(CHUNK_OF_DOCUMENT, id)
            .into_iter()
            .filter(|chunk_id| graph.get(DOCUMENT_CHUNK, *chunk_id).is_some())
            .collect();
        let spans: Vec<u64> = chunks
            .iter()
            .flat_map(|chunk_id| graph.sources(SPAN_OF_CHUNK, *chunk_id))
            .filter(|span_id| graph.get(SPAN, *span_id).is_some())
            .collect();
        let removed: BTreeSet<String> =
            chunks.iter().chain(&spans).map(ToString::to_string).collect();
        for node_id in chunks.iter().chain(&spans) {
            graph.nodes.remove(node_id);
        }
        graph.edges.retain(|edge| !removed.contains(&edge.from) && !removed.contains(&edge.to));
        Ok(chunks.len())
    }

    fn create_spans_batch(&self, spans: &[SpanNode]) -> Result<Vec<StoredSpan>> {
        let mut graph = self.write();
        let parents = spans
            .iter()
            .map(|span| graph.require(DOCUMENT_CHUNK, "chunk_id", span.chunk_id.as_str()))
            .collect::<Result<Vec<u64>>>()?;
        Ok(spans
            .iter()
            .zip(parents)
            .map(|(span, parent)| insert_span(&mut graph, span, parent))
            .collect())
    }

    fn create_embeddings_batch(
        &self,
        label: VectorLabel,
        items: &[EmbeddingWrite<'_>],
    ) -> Result<usize> {
        for item in items {
            validate_vector(item.vector, self.embedding_dim)?;
        }
        let mut graph = self.write();
        let ids = items
            .iter()
            .map(|item| graph.require(label.as_str(), label.id_field(), item.node_id))
            .collect::<Result<Vec<u64>>>()?;
        for (item, id) in items.iter().zip(ids) {
            if let Some(node) = graph.get_mut(label.as_str(), id) {
                node.embedding = Some(item.vector.to_vec());
            }
        }
        Ok(items.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn merge_keeps_the_best_score_per_node() {
        let merged = merge_hits(vec![vec![(1, 0.5), (2, 0.4)], vec![(2, 0.9), (3, 0.1)]], 20);
        assert_eq!(merged, vec![(2, 0.9), (1, 0.5), (3, 0.1)]);
        assert_eq!(merge_hits(vec![vec![(1, 0.5), (2, 0.4)]], 1), vec![(1, 0.5)]);
    }

    #[test]
    fn cosine_of_parallel_and_orthogonal_vectors() {
        let a = [1.0_f32, 0.0];
        assert!((cosine(&a, norm(&a), &[2.0, 0.0]) - 1.0).abs() < 1e-12);
        assert!(cosine(&a, norm(&a), &[0.0, 3.0]).abs() < 1e-12);
    }

    #[test]
    fn reset_keeps_counting_ids_and_changes_store_id() {
        let store = MemoryStore::new(2);
        let before = store.store_id();
        let first = store.create_document(&DocumentNode::default()).expect("created");
        store.reset().expect("reset");
        assert_ne!(store.store_id(), before);
        assert_eq!(store.node_count(), 0);
        let second = store.create_document(&DocumentNode::default()).expect("created");
        assert_ne!(first.id, second.id);
    }

    #[test]
    fn edges_carry_python_properties_and_default_roles() {
        let store = MemoryStore::new(2);
        let doc = store.create_document(&DocumentNode::default()).expect("doc");
        let chunk_node = ChunkNode { sequence_number: 4, ..ChunkNode::default() };
        let chunk = store.create_document_chunk(&doc.id, &chunk_node).expect("chunk");
        let span_node =
            SpanNode { chunk_id: chunk.id.clone(), start_char: 12, ..SpanNode::default() };
        let span = store.create_span_for_chunk(&span_node).expect("span");
        let entity = store.create_entity(&EntityNode::new("Acme", "ORG")).expect("entity");
        let fact = store.create_fact(&FactNode::new("x", "{}", "{}", 0.5)).expect("fact");
        store.link_span_to_entity(&span.id, &entity.id, 0.8, "").expect("link");
        store.link_fact_to_document(&fact.id, &doc.id, 3, "Facts").expect("link");
        store.link_fact_to_entity(&fact.id, &entity.id, "").expect("link");
        store.link_fact_to_span(&fact.id, &span.id, "", 1.0).expect("link");

        let edges: Vec<(&str, Value)> = store
            .edges()
            .into_iter()
            .map(|edge| (edge.label, Value::Object(edge.properties)))
            .collect();
        assert_eq!(
            edges,
            vec![
                (CHUNK_OF_DOCUMENT, json!({"SequenceNumber": 4})),
                (SPAN_OF_CHUNK, json!({"SequenceNumber": 12})),
                (SPAN_REFERS_TO_ENTITY, json!({"Confidence": 0.8, "Method": "NER+ED"})),
                (FACT_IN_DOCUMENT, json!({"PageNumber": 3, "Section": "Facts"})),
                (FACT_ABOUT_ENTITY, json!({"Role": "RELATED"})),
                (FACT_DERIVED_FROM_SPAN, json!({"Role": "SOURCE_SPAN", "Confidence": 1.0})),
            ]
        );
        let edge = &store.edges()[1];
        assert_eq!((edge.from.as_str(), edge.to.as_str()), (span.id.as_str(), chunk.id.as_str()));
    }

    #[test]
    fn snapshots_expose_stored_properties_and_vectors() {
        let store = MemoryStore::new(2);
        let doc =
            store.create_document(&DocumentNode { title: "T".into(), ..DocumentNode::default() });
        let doc = doc.expect("doc");
        let (label, properties) = store.node_snapshot(doc.id.as_str()).expect("snapshot");
        assert_eq!(label, DOCUMENT);
        assert_eq!(properties["Title"], json!("T"));
        assert_eq!(store.node_snapshot("999"), None);
        let entity = store.create_entity(&EntityNode::new("Acme", "ORG")).expect("entity");
        assert_eq!(store.embedding(entity.id.as_str()), None);
        store.create_entity_embedding(&entity.id, &[0.5, 0.5]).expect("vector");
        assert_eq!(store.embedding(entity.id.as_str()), Some(vec![0.5, 0.5]));
    }
}
