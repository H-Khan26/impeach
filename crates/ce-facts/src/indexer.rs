//! Fact indexing (Python `services/facts/indexer.py`): `Fact` nodes, fact embeddings and the
//! edges linking a fact to its document, entities and source spans.
//!
//! [`FactIndexer::index_facts`] writes the nodes one by one (a node the store rejects is
//! logged and skipped), then embeds the written facts in batches of
//! [`FACT_EMBED_BATCH_SIZE`] texts and writes the vectors with one store batch per embedder
//! batch. Vectors stay aligned with their facts across batch boundaries; a batch whose
//! embedder call fails or returns the wrong number of vectors leaves its facts without a vector.
//!
//! [`FactIndexer::commit_facts`] is the commit path of both `FactStage` (`commit_facts: true`)
//! and the contradiction stage (clean facts only): nodes, vectors, then the context edges and
//! `metadata.helix_fact_id` of each written fact.

use std::fmt;
use std::sync::Arc;

use ce_core::pyjson::to_python_json_ascii;
use ce_core::{EntityId, ExtractedFact, FactId, Metadata, NodeId, SpanId, TypeVocabulary};
use ce_models::Embedder;
use ce_store::{EmbeddingWrite, FactNode, GraphStore, StoreError, VectorLabel};
use serde::Serialize;
use serde_json::Value;

use crate::error::Result;

/// Number of fact texts embedded per embedder call.
pub const FACT_EMBED_BATCH_SIZE: usize = 64;

/// Role of the `FactAboutEntity` edge to the subject entity.
pub const SUBJECT_ROLE: &str = "SUBJECT";
/// Role of the `FactAboutEntity` edge to the object entity.
pub const OBJECT_ROLE: &str = "OBJECT";
/// Role of a `FactDerivedFromSpan` edge.
pub const SOURCE_SPAN_ROLE: &str = "SOURCE_SPAN";
/// Confidence of a `FactDerivedFromSpan` edge.
pub const SOURCE_SPAN_CONFIDENCE: f64 = 1.0;

/// Outcome of [`FactIndexer::index_facts`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndexResult {
    /// Facts written with their embedding.
    pub count: usize,
    /// Graph id per input fact, in input order (`None` when the node was not written).
    pub fact_ids: Vec<Option<FactId>>,
}

/// Edges to create for one written fact ([`FactIndexer::link_fact_context`]).
#[derive(Debug, Clone, Copy, Default)]
pub struct FactContextLinks<'a> {
    /// Graph id of the source `Document` (`FactInDocument`).
    pub doc_id: Option<&'a NodeId>,
    /// Subject entity (`FactAboutEntity`, role `SUBJECT`).
    pub subject_entity_id: Option<&'a EntityId>,
    /// Object entity (`FactAboutEntity`, role `OBJECT`).
    pub object_entity_id: Option<&'a EntityId>,
    /// Source spans (`FactDerivedFromSpan`, role `SOURCE_SPAN`, confidence 1.0).
    pub source_span_ids: &'a [SpanId],
    /// Page number of the `FactInDocument` edge.
    pub page_number: i64,
    /// Section of the `FactInDocument` edge.
    pub section: &'a str,
}

/// Python `json.dumps` of a record (ASCII-escaped, Python separators).
fn python_json<T: Serialize + ?Sized>(value: &T) -> String {
    to_python_json_ascii(value).expect("fact records and JSON metadata serialise to JSON")
}

/// The `Fact` node of an extracted fact: text, JSON triple / validity / metadata, certainty
/// and the taxonomy-normalised value and property types.
#[must_use]
pub fn fact_node(fact: &ExtractedFact, vocab: &dyn TypeVocabulary) -> FactNode {
    FactNode {
        fact_text: fact.text.clone(),
        fact_origin: "STATED".to_owned(),
        fact_type: "STATED".to_owned(),
        status: "RAW".to_owned(),
        assertion_triple: python_json(&fact.assertion_triple),
        temporal_validity: python_json(&fact.temporal_validity),
        certainty: fact.certainty,
        metadata: python_json(&fact.metadata),
        value_type: fact.normalized_value_type(vocab).into_inner(),
        property_type: fact.normalized_property_type(vocab).into_inner(),
    }
}

/// The `source_span_ids` metadata entry: an array of strings, or a string holding a JSON
/// array of strings (non-string items are ignored).
fn source_span_ids(metadata: &Metadata) -> Vec<SpanId> {
    let ids = |items: &[Value]| -> Vec<SpanId> {
        items.iter().filter_map(Value::as_str).map(SpanId::new).collect()
    };
    match metadata.get("source_span_ids") {
        Some(Value::Array(items)) => ids(items),
        Some(Value::String(encoded)) => match serde_json::from_str::<Value>(encoded) {
            Ok(Value::Array(items)) => ids(&items),
            _ => Vec::new(),
        },
        _ => Vec::new(),
    }
}

/// Writes facts, their embeddings and context edges.
pub struct FactIndexer {
    store: Arc<dyn GraphStore>,
    embedder: Arc<dyn Embedder>,
    batch_size: usize,
}

impl fmt::Debug for FactIndexer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FactIndexer")
            .field("embedder", &self.embedder.model_name())
            .field("batch_size", &self.batch_size)
            .finish_non_exhaustive()
    }
}

impl FactIndexer {
    /// Indexer over a store and an embedder, embedding [`FACT_EMBED_BATCH_SIZE`] texts per call.
    #[must_use]
    pub fn new(store: Arc<dyn GraphStore>, embedder: Arc<dyn Embedder>) -> Self {
        Self { store, embedder, batch_size: FACT_EMBED_BATCH_SIZE }
    }

    /// Sets the embedding batch size (at least 1).
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    /// Writes the nodes, then embeds the written ones and stores their vectors.
    ///
    /// # Errors
    /// [`crate::FactError::Store`] for a fatal store error; other failures are logged.
    pub fn index_facts(&self, nodes: &[FactNode]) -> Result<IndexResult> {
        let mut fact_ids = Vec::with_capacity(nodes.len());
        let mut written: Vec<(FactId, &FactNode)> = Vec::with_capacity(nodes.len());
        for node in nodes {
            match self.store.create_fact(node) {
                Ok(stored) => {
                    fact_ids.push(Some(stored.id.clone()));
                    written.push((stored.id, node));
                }
                Err(error) if error.is_fatal() => return Err(error.into()),
                Err(error) => {
                    tracing::error!(
                        fact = truncate(&node.fact_text, 30),
                        error = %error,
                        "failed to index fact"
                    );
                    fact_ids.push(None);
                }
            }
        }

        let mut count = 0;
        for (index, batch) in written.chunks(self.batch_size).enumerate() {
            let texts: Vec<&str> = batch.iter().map(|(_, node)| node.fact_text.as_str()).collect();
            let vectors = match self.embedder.embed(&texts) {
                Ok(vectors) => vectors,
                Err(error) => {
                    tracing::error!(batch = index, facts = texts.len(), error = %error, "failed to embed facts");
                    continue;
                }
            };
            if vectors.len() != texts.len() {
                tracing::error!(
                    batch = index,
                    facts = texts.len(),
                    vectors = vectors.len(),
                    "fact embedder returned a different number of vectors; batch left unembedded"
                );
                continue;
            }
            count += self.write_vectors(batch, &vectors)?;
        }
        Ok(IndexResult { count, fact_ids })
    }

    /// Writes one batch of fact vectors; falls back to one-by-one writes when the batch write
    /// fails. Returns the number of vectors written.
    fn write_vectors(&self, batch: &[(FactId, &FactNode)], vectors: &[Vec<f32>]) -> Result<usize> {
        let items: Vec<EmbeddingWrite<'_>> = batch
            .iter()
            .zip(vectors)
            .map(|((id, _), vector)| EmbeddingWrite { node_id: id.as_str(), vector })
            .collect();
        match self.store.create_embeddings_batch(VectorLabel::Fact, &items) {
            Ok(written) => return Ok(written),
            Err(error) if error.is_fatal() => return Err(error.into()),
            Err(error) => {
                tracing::warn!(error = %error, "fact embedding batch failed; writing one by one");
            }
        }
        let mut written = 0;
        for ((fact_id, _), vector) in batch.iter().zip(vectors) {
            match self.store.create_fact_embedding(fact_id, vector) {
                Ok(()) => written += 1,
                Err(error) if error.is_fatal() => return Err(error.into()),
                Err(error) => {
                    tracing::error!(fact = %fact_id, error = %error, "failed to write fact embedding");
                }
            }
        }
        Ok(written)
    }

    /// Creates the context edges of one written fact.
    ///
    /// # Errors
    /// [`crate::FactError::Store`] for a fatal store error; other failures are logged.
    pub fn link_fact_context(&self, fact_id: &FactId, links: &FactContextLinks<'_>) -> Result<()> {
        if let Some(doc_id) = links.doc_id {
            let outcome =
                self.store.link_fact_to_document(fact_id, doc_id, links.page_number, links.section);
            log_link_error(outcome, fact_id, "document", doc_id.as_str())?;
        }
        if let Some(entity_id) = links.subject_entity_id {
            let outcome = self.store.link_fact_to_entity(fact_id, entity_id, SUBJECT_ROLE);
            log_link_error(outcome, fact_id, "subject", entity_id.as_str())?;
        }
        if let Some(entity_id) = links.object_entity_id {
            let outcome = self.store.link_fact_to_entity(fact_id, entity_id, OBJECT_ROLE);
            log_link_error(outcome, fact_id, "object", entity_id.as_str())?;
        }
        for span_id in links.source_span_ids {
            let outcome = self.store.link_fact_to_span(
                fact_id,
                span_id,
                SOURCE_SPAN_ROLE,
                SOURCE_SPAN_CONFIDENCE,
            );
            log_link_error(outcome, fact_id, "span", span_id.as_str())?;
        }
        Ok(())
    }

    /// Commits extracted facts: [`fact_node`] + [`FactIndexer::index_facts`], then the context
    /// edges of each written fact (`source_doc_id`, subject / object entity ids, the
    /// `source_span_ids` metadata list) and `metadata.helix_fact_id`.
    ///
    /// # Errors
    /// [`crate::FactError::Store`] for a fatal store error; other failures are logged.
    pub fn commit_facts(
        &self,
        facts: &mut [ExtractedFact],
        vocab: &dyn TypeVocabulary,
    ) -> Result<IndexResult> {
        if facts.is_empty() {
            return Ok(IndexResult::default());
        }
        let nodes: Vec<FactNode> = facts.iter().map(|fact| fact_node(fact, vocab)).collect();
        let result = self.index_facts(&nodes)?;

        let mut with_spans = 0;
        for (fact, fact_id) in facts.iter_mut().zip(&result.fact_ids) {
            let Some(fact_id) = fact_id else {
                continue;
            };
            fact.metadata.insert("helix_fact_id".to_owned(), Value::String(fact_id.to_string()));
            let span_ids = source_span_ids(&fact.metadata);
            if !span_ids.is_empty() {
                with_spans += 1;
            }
            let doc_id = fact
                .source_doc_id
                .as_ref()
                .filter(|id| !id.as_str().is_empty())
                .map(|id| NodeId::new(id.as_str()));
            let links = FactContextLinks {
                doc_id: doc_id.as_ref(),
                subject_entity_id: fact.subject_entity_id.as_ref(),
                object_entity_id: fact.object_entity_id.as_ref(),
                source_span_ids: &span_ids,
                page_number: 0,
                section: "",
            };
            self.link_fact_context(fact_id, &links)?;
        }
        tracing::info!(
            facts = facts.len(),
            written = result.fact_ids.iter().flatten().count(),
            embedded = result.count,
            with_span_provenance = with_spans,
            "committed facts"
        );
        Ok(result)
    }
}

/// Logs a failed edge write; a fatal store error is returned.
fn log_link_error(
    outcome: std::result::Result<(), StoreError>,
    fact_id: &FactId,
    target_kind: &str,
    target_id: &str,
) -> Result<()> {
    match outcome {
        Ok(()) => Ok(()),
        Err(error) if error.is_fatal() => Err(error.into()),
        Err(error) => {
            tracing::error!(
                fact = %fact_id,
                target_kind,
                target = target_id,
                error = %error,
                "failed to link fact"
            );
            Ok(())
        }
    }
}

/// The first `max_chars` characters of `text`.
fn truncate(text: &str, max_chars: usize) -> &str {
    text.char_indices().nth(max_chars).map_or(text, |(end, _)| &text[..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn metadata(value: Value) -> Metadata {
        match value {
            Value::Object(map) => map,
            _ => Metadata::new(),
        }
    }

    #[test]
    fn source_span_ids_accepts_lists_and_encoded_lists() {
        let expected = vec![SpanId::new("1"), SpanId::new("2")];
        assert_eq!(source_span_ids(&metadata(json!({"source_span_ids": ["1", "2"]}))), expected);
        assert_eq!(
            source_span_ids(&metadata(json!({"source_span_ids": "[\"1\", \"2\"]"}))),
            expected
        );
        assert!(source_span_ids(&metadata(json!({"source_span_ids": "not json"}))).is_empty());
        assert!(source_span_ids(&metadata(json!({"source_span_ids": 3}))).is_empty());
        assert!(source_span_ids(&Metadata::new()).is_empty());
    }

    #[test]
    fn truncate_counts_characters() {
        assert_eq!(truncate("héllo world", 5), "héllo");
        assert_eq!(truncate("abc", 30), "abc");
    }
}
