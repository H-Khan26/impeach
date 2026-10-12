//! The graph model: labels, property names, edges, indexes, typed nodes and their validation.
//!
//! Python origin: the docstring and parameter builders of `storage/helix/queries.py` and the
//! models in `storage/helix/models.py`.
//!
//! ```text
//! Nodes (property names are PascalCase; callers read them back by these names)
//!   Document       DocType, Title, DateCreated (RFC3339 string), FilePath, TotalPages, Metadata
//!   DocumentChunk  Text, PageNumber, StartChar, EndChar, Section, SequenceNumber, Metadata, embedding
//!   Span           DocId, ChunkId, StartChar, EndChar, SpanType, SpanLabel, SurfaceText, Metadata
//!   Entity         Name, EntityType, AliasesJson, Description, Metadata, embedding
//!   Fact           FactText, FactOrigin, FactType, Status, AssertionTriple, TemporalValidity,
//!                  Certainty, Metadata, ValueType, PropertyType, embedding
//!   StoreInfo      store_id (random id of this database, used to scope client-side caches)
//!   Metadata / AliasesJson / AssertionTriple / TemporalValidity hold JSON strings.
//!
//! Edges
//!   ChunkOfDocument      DocumentChunk -> Document       SequenceNumber
//!   SpanOfChunk          Span -> DocumentChunk           SequenceNumber (= span start_char)
//!   SpanRefersToEntity   Span -> Entity                  Confidence, Method
//!   FactInDocument       Fact -> Document                PageNumber, Section
//!   FactAboutEntity      Fact -> Entity                  Role
//!   FactDerivedFromSpan  Fact -> Span                    Role, Confidence
//!
//! Indexes
//!   vector (cosine, embedding_dim): DocumentChunk.embedding, Entity.embedding, Fact.embedding
//!   text (BM25): Entity.Name, Entity.AliasesJson, Fact.FactText
//! ```
//!
//! Node ids are non-negative integers assigned by the store and carried as strings in the
//! ce-core id newtypes: [`NodeId`] for documents, [`ChunkId`], [`SpanId`], [`EntityId`] and
//! [`FactId`] for the other labels.

use ce_core::{ChunkId, EntityId, FactId, NodeId, SpanId};
use serde::{Deserialize, Serialize};

use crate::error::{Result, StoreError};

/// Query names of the Python client (`HelixClient._execute_query`); used in logs and errors.
pub mod query_names {
    /// `CreateDocument`
    pub const CREATE_DOCUMENT: &str = "CreateDocument";
    /// `UpdateDocument`
    pub const UPDATE_DOCUMENT: &str = "UpdateDocument";
    /// `GetDocumentChunks`
    pub const GET_DOCUMENT_CHUNKS: &str = "GetDocumentChunks";
    /// `CreateDocumentChunk`
    pub const CREATE_DOCUMENT_CHUNK: &str = "CreateDocumentChunk";
    /// `UpdateDocumentChunk`
    pub const UPDATE_DOCUMENT_CHUNK: &str = "UpdateDocumentChunk";
    /// `CreateChunkEmbedding`
    pub const CREATE_CHUNK_EMBEDDING: &str = "CreateChunkEmbedding";
    /// `CreateSpanForChunk`
    pub const CREATE_SPAN_FOR_CHUNK: &str = "CreateSpanForChunk";
    /// `CreateEntity`
    pub const CREATE_ENTITY: &str = "CreateEntity";
    /// `CreateEntityEmbedding`
    pub const CREATE_ENTITY_EMBEDDING: &str = "CreateEntityEmbedding";
    /// `FindEntityByName`
    pub const FIND_ENTITY_BY_NAME: &str = "FindEntityByName";
    /// `FindSimilarEntities`
    pub const FIND_SIMILAR_ENTITIES: &str = "FindSimilarEntities";
    /// `GetAllEntities`
    pub const GET_ALL_ENTITIES: &str = "GetAllEntities";
    /// `LinkSpanToEntity`
    pub const LINK_SPAN_TO_ENTITY: &str = "LinkSpanToEntity";
    /// `CreateFact`
    pub const CREATE_FACT: &str = "CreateFact";
    /// `CreateFactEmbedding`
    pub const CREATE_FACT_EMBEDDING: &str = "CreateFactEmbedding";
    /// `LinkFactToDocument`
    pub const LINK_FACT_TO_DOCUMENT: &str = "LinkFactToDocument";
    /// `LinkFactToEntity`
    pub const LINK_FACT_TO_ENTITY: &str = "LinkFactToEntity";
    /// `LinkFactToSpan`
    pub const LINK_FACT_TO_SPAN: &str = "LinkFactToSpan";
    /// `GetEntityFacts`
    pub const GET_ENTITY_FACTS: &str = "GetEntityFacts";
    /// `GetFactEntities`
    pub const GET_FACT_ENTITIES: &str = "GetFactEntities";
    /// `FindFactsByText`
    pub const FIND_FACTS_BY_TEXT: &str = "FindFactsByText";
    /// `FindSimilarFacts`
    pub const FIND_SIMILAR_FACTS: &str = "FindSimilarFacts";
}

/// `Document` node label.
pub const DOCUMENT: &str = "Document";
/// `DocumentChunk` node label.
pub const DOCUMENT_CHUNK: &str = "DocumentChunk";
/// `Span` node label.
pub const SPAN: &str = "Span";
/// `Entity` node label.
pub const ENTITY: &str = "Entity";
/// `Fact` node label.
pub const FACT: &str = "Fact";
/// `StoreInfo` node label (one node holding `store_id`).
pub const STORE_INFO: &str = "StoreInfo";
/// Property of the `StoreInfo` node holding the store id.
pub const STORE_ID_PROPERTY: &str = "store_id";

/// `DocumentChunk -> Document` edge label.
pub const CHUNK_OF_DOCUMENT: &str = "ChunkOfDocument";
/// `Span -> DocumentChunk` edge label.
pub const SPAN_OF_CHUNK: &str = "SpanOfChunk";
/// `Span -> Entity` edge label.
pub const SPAN_REFERS_TO_ENTITY: &str = "SpanRefersToEntity";
/// `Fact -> Document` edge label.
pub const FACT_IN_DOCUMENT: &str = "FactInDocument";
/// `Fact -> Entity` edge label.
pub const FACT_ABOUT_ENTITY: &str = "FactAboutEntity";
/// `Fact -> Span` edge label.
pub const FACT_DERIVED_FROM_SPAN: &str = "FactDerivedFromSpan";

/// Node property holding the embedding vector.
pub const EMBEDDING_PROPERTY: &str = "embedding";

/// Properties of each node label, in the order the Python model lists them.
pub const NODE_PROPERTIES: [(&str, &[&str]); 5] = [
    (DOCUMENT, &["DocType", "Title", "DateCreated", "FilePath", "TotalPages", "Metadata"]),
    (
        DOCUMENT_CHUNK,
        &["Text", "PageNumber", "StartChar", "EndChar", "Section", "SequenceNumber", "Metadata"],
    ),
    (
        SPAN,
        &[
            "DocId",
            "ChunkId",
            "StartChar",
            "EndChar",
            "SpanType",
            "SpanLabel",
            "SurfaceText",
            "Metadata",
        ],
    ),
    (ENTITY, &["Name", "EntityType", "AliasesJson", "Description", "Metadata"]),
    (
        FACT,
        &[
            "FactText",
            "FactOrigin",
            "FactType",
            "Status",
            "AssertionTriple",
            "TemporalValidity",
            "Certainty",
            "Metadata",
            "ValueType",
            "PropertyType",
        ],
    ),
];

/// Edge labels with their properties, in the order the Python model lists them.
pub const EDGE_PROPERTIES: [(&str, &[&str]); 6] = [
    (CHUNK_OF_DOCUMENT, &["SequenceNumber"]),
    (SPAN_OF_CHUNK, &["SequenceNumber"]),
    (SPAN_REFERS_TO_ENTITY, &["Confidence", "Method"]),
    (FACT_IN_DOCUMENT, &["PageNumber", "Section"]),
    (FACT_ABOUT_ENTITY, &["Role"]),
    (FACT_DERIVED_FROM_SPAN, &["Role", "Confidence"]),
];

/// Labels with a cosine vector index on [`EMBEDDING_PROPERTY`].
pub const VECTOR_INDEXED_LABELS: [&str; 3] = [DOCUMENT_CHUNK, ENTITY, FACT];
/// `(label, property)` pairs with a BM25 text index.
pub const TEXT_INDEXES: [(&str, &str); 3] =
    [(ENTITY, "Name"), (ENTITY, "AliasesJson"), (FACT, "FactText")];

/// Hits returned by [`crate::GraphStore::find_entity_by_name`].
pub const ENTITY_NAME_SEARCH_K: usize = 20;
/// Upper bound applied to every search `limit` (server-side cap for whole-index searches).
pub const MAX_SEARCH_K: usize = 800;

/// Default `Role` of `FactAboutEntity` edges.
pub const DEFAULT_FACT_ENTITY_ROLE: &str = "RELATED";
/// Default `Method` of `SpanRefersToEntity` edges.
pub const DEFAULT_SPAN_ENTITY_METHOD: &str = "NER+ED";
/// Default `Role` of `FactDerivedFromSpan` edges.
pub const DEFAULT_FACT_SPAN_ROLE: &str = "SOURCE_SPAN";

/// Properties of `label`, or `None` for an unknown label.
#[must_use]
pub fn node_properties(label: &str) -> Option<&'static [&'static str]> {
    NODE_PROPERTIES.iter().find(|(name, _)| *name == label).map(|(_, props)| *props)
}

/// A node label with a vector index; selects the target of
/// [`crate::GraphStore::create_embeddings_batch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorLabel {
    /// `DocumentChunk`
    DocumentChunk,
    /// `Entity`
    Entity,
    /// `Fact`
    Fact,
}

impl VectorLabel {
    /// The node label.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DocumentChunk => DOCUMENT_CHUNK,
            Self::Entity => ENTITY,
            Self::Fact => FACT,
        }
    }

    /// Name of the id parameter in errors (`chunk_id`, `entity_id`, `fact_id`).
    #[must_use]
    pub fn id_field(self) -> &'static str {
        match self {
            Self::DocumentChunk => "chunk_id",
            Self::Entity => "entity_id",
            Self::Fact => "fact_id",
        }
    }
}

/// `Document` node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase", default)]
pub struct DocumentNode {
    /// Document type, e.g. `complaint` (`UNKNOWN` when the payload has none).
    pub doc_type: String,
    /// Title (the external doc id when the payload has none).
    pub title: String,
    /// Creation time as an RFC3339 UTC string.
    pub date_created: String,
    /// Source path with forward slashes.
    pub file_path: String,
    /// Page count (0 when unknown).
    pub total_pages: i64,
    /// JSON object string.
    pub metadata: String,
}

impl Default for DocumentNode {
    fn default() -> Self {
        Self {
            doc_type: String::new(),
            title: String::new(),
            date_created: String::new(),
            file_path: String::new(),
            total_pages: 0,
            metadata: "{}".to_owned(),
        }
    }
}

/// `DocumentChunk` node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase", default)]
pub struct ChunkNode {
    /// Chunk text.
    pub text: String,
    /// Page number (0 when unknown).
    pub page_number: i64,
    /// Start offset in the document text.
    pub start_char: i64,
    /// End offset in the document text.
    pub end_char: i64,
    /// Section name (empty when unknown).
    pub section: String,
    /// Position of the chunk in its document.
    pub sequence_number: i64,
    /// JSON object string.
    pub metadata: String,
}

impl Default for ChunkNode {
    fn default() -> Self {
        Self {
            text: String::new(),
            page_number: 0,
            start_char: 0,
            end_char: 0,
            section: String::new(),
            sequence_number: 0,
            metadata: "{}".to_owned(),
        }
    }
}

/// `Span` node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase", default)]
pub struct SpanNode {
    /// Graph id of the document the span belongs to.
    pub doc_id: NodeId,
    /// Graph id of the chunk the span belongs to (the target of its `SpanOfChunk` edge).
    pub chunk_id: ChunkId,
    /// Start offset.
    pub start_char: i64,
    /// End offset.
    pub end_char: i64,
    /// Span kind, e.g. `SENTENCE` or `MENTION`.
    pub span_type: String,
    /// Span label, e.g. `SENTENCE` or an entity type.
    pub span_label: String,
    /// Covered text.
    pub surface_text: String,
    /// JSON object string.
    pub metadata: String,
}

impl Default for SpanNode {
    fn default() -> Self {
        Self {
            doc_id: NodeId::default(),
            chunk_id: ChunkId::default(),
            start_char: 0,
            end_char: 0,
            span_type: String::new(),
            span_label: String::new(),
            surface_text: String::new(),
            metadata: "{}".to_owned(),
        }
    }
}

/// `Entity` node (concrete actors and objects).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase", default)]
pub struct EntityNode {
    /// Canonical name; required.
    pub name: String,
    /// Entity type, e.g. `PERSON`, `ORG`; required.
    pub entity_type: String,
    /// JSON array of aliases, e.g. `["Acme Corp.", "Acme"]`; empty means `[]`.
    pub aliases_json: String,
    /// Short summary.
    pub description: String,
    /// JSON object string.
    pub metadata: String,
}

impl Default for EntityNode {
    fn default() -> Self {
        Self {
            name: String::new(),
            entity_type: String::new(),
            aliases_json: "[]".to_owned(),
            description: String::new(),
            metadata: "{}".to_owned(),
        }
    }
}

impl EntityNode {
    /// Entity with the given name and type and default aliases, description and metadata.
    pub fn new(name: impl Into<String>, entity_type: impl Into<String>) -> Self {
        Self { name: name.into(), entity_type: entity_type.into(), ..Self::default() }
    }

    /// Checks the node as Python `build_create_entity_params` does and returns the node to
    /// store (an empty `AliasesJson` becomes `[]`).
    ///
    /// # Errors
    /// [`StoreError::Validation`] when `Name` or `EntityType` is empty or `AliasesJson` is not
    /// a JSON array.
    pub fn validated(&self) -> Result<Self> {
        if self.name.is_empty() {
            return Err(StoreError::validation("Name", "", "Entity Name is required"));
        }
        if self.entity_type.is_empty() {
            return Err(StoreError::validation("EntityType", "", "EntityType is required"));
        }
        let aliases_json = or_default(&self.aliases_json, "[]");
        match serde_json::from_str::<serde_json::Value>(aliases_json) {
            Ok(serde_json::Value::Array(_)) => {}
            Ok(_) => {
                return Err(StoreError::validation(
                    "AliasesJson",
                    aliases_json,
                    "AliasesJson must be a JSON array of strings",
                ))
            }
            Err(error) => {
                return Err(StoreError::validation(
                    "AliasesJson",
                    aliases_json,
                    format!("AliasesJson must be valid JSON: {error}"),
                ))
            }
        }
        let mut node = self.clone();
        aliases_json.clone_into(&mut node.aliases_json);
        Ok(node)
    }

    /// The aliases parsed from `AliasesJson`; string elements only, invalid JSON gives none.
    #[must_use]
    pub fn aliases(&self) -> Vec<String> {
        serde_json::from_str::<Vec<serde_json::Value>>(&self.aliases_json)
            .map(|values| {
                values.into_iter().filter_map(|v| v.as_str().map(str::to_owned)).collect()
            })
            .unwrap_or_default()
    }
}

/// `Fact` node (assertions, events, obligations).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase", default)]
pub struct FactNode {
    /// Human-readable statement; required.
    pub fact_text: String,
    /// `STATED` or `DERIVED`.
    pub fact_origin: String,
    /// Deprecated fact category.
    pub fact_type: String,
    /// `RAW`, `NORMALIZED` or `REVIEWED`.
    pub status: String,
    /// JSON `{subject, relationship, object}`; must be valid JSON.
    pub assertion_triple: String,
    /// JSON `{t_valid_from, t_valid_until}`; must be valid JSON.
    pub temporal_validity: String,
    /// Confidence in `[0, 1]`.
    pub certainty: f64,
    /// JSON object string.
    pub metadata: String,
    /// Structural value type (validated upstream against the fact taxonomy).
    pub value_type: String,
    /// Structural property type (validated upstream against the fact taxonomy).
    pub property_type: String,
}

impl Default for FactNode {
    fn default() -> Self {
        Self {
            fact_text: String::new(),
            fact_origin: "STATED".to_owned(),
            fact_type: "STATED".to_owned(),
            status: "RAW".to_owned(),
            assertion_triple: "{}".to_owned(),
            temporal_validity: "{}".to_owned(),
            certainty: 0.0,
            metadata: "{}".to_owned(),
            value_type: "TEXT".to_owned(),
            property_type: "ATTRIBUTE".to_owned(),
        }
    }
}

impl FactNode {
    /// Fact with the required fields and Python defaults for the rest.
    pub fn new(
        fact_text: impl Into<String>,
        assertion_triple: impl Into<String>,
        temporal_validity: impl Into<String>,
        certainty: f64,
    ) -> Self {
        Self {
            fact_text: fact_text.into(),
            assertion_triple: assertion_triple.into(),
            temporal_validity: temporal_validity.into(),
            certainty,
            ..Self::default()
        }
    }

    /// Checks the node as Python `build_create_fact_params` does.
    ///
    /// # Errors
    /// [`StoreError::Validation`] when `FactText` is empty, `Certainty` is not a finite number
    /// in `[0, 1]`, or `AssertionTriple` / `TemporalValidity` is not valid JSON.
    pub fn validate(&self) -> Result<()> {
        if self.fact_text.is_empty() {
            return Err(StoreError::validation("FactText", "", "FactText is required"));
        }
        if !self.certainty.is_finite() {
            return Err(StoreError::validation(
                "Certainty",
                self.certainty.to_string(),
                format!("Certainty must be a valid float, got {}", self.certainty),
            ));
        }
        if !(0.0..=1.0).contains(&self.certainty) {
            return Err(StoreError::validation(
                "Certainty",
                self.certainty.to_string(),
                format!("Certainty must be between 0.0 and 1.0, got {}", self.certainty),
            ));
        }
        for (field, text) in [
            ("AssertionTriple", &self.assertion_triple),
            ("TemporalValidity", &self.temporal_validity),
        ] {
            if serde_json::from_str::<serde_json::Value>(text).is_err() {
                return Err(StoreError::validation(
                    field,
                    text.as_str(),
                    format!("{field} must be valid JSON"),
                ));
            }
        }
        Ok(())
    }
}

/// A node read back from the store: its id plus its properties.
///
/// Serialises flat, like the Python node dicts: `{"id": "7", "Name": ..., ...}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stored<I, T> {
    /// Store-assigned node id.
    pub id: I,
    /// Node properties.
    #[serde(flatten)]
    pub node: T,
}

/// A search hit: a stored node and its score.
///
/// Vector search scores are cosine similarity clamped to `[0, 1]`; text search scores are raw
/// BM25 scores (non-negative, unbounded).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHit<T> {
    /// The matching node.
    #[serde(flatten)]
    pub node: T,
    /// Similarity score.
    pub score: f64,
}

/// Stored `Document`.
pub type StoredDocument = Stored<NodeId, DocumentNode>;
/// Stored `DocumentChunk`.
pub type StoredChunk = Stored<ChunkId, ChunkNode>;
/// Stored `Span`.
pub type StoredSpan = Stored<SpanId, SpanNode>;
/// Stored `Entity`.
pub type StoredEntity = Stored<EntityId, EntityNode>;
/// Stored `Fact`.
pub type StoredFact = Stored<FactId, FactNode>;
/// Entity search hit.
pub type EntityHit = SearchHit<StoredEntity>;
/// Fact search hit.
pub type FactHit = SearchHit<StoredFact>;

/// One vector to write with [`crate::GraphStore::create_embeddings_batch`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmbeddingWrite<'a> {
    /// Graph id of the node that receives the vector.
    pub node_id: &'a str,
    /// The vector (`embedding_dim` finite values, not all zero).
    pub vector: &'a [f32],
}

/// Parses a graph node id (`field` names the parameter in errors).
///
/// # Errors
/// [`StoreError::Validation`] when the id is empty or not a non-negative integer.
pub fn parse_node_id(field: &str, value: &str) -> Result<u64> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(StoreError::validation(field, value, format!("{field} is required")));
    }
    trimmed.parse::<u64>().map_err(|_| {
        StoreError::validation(
            field,
            value,
            format!("{field} must be a graph node id, got '{value}'"),
        )
    })
}

/// Checks an edge confidence like Python `_validate_confidence`.
///
/// # Errors
/// [`StoreError::Validation`] when the value is not a finite number in `[0, 1]`.
pub fn validate_confidence(confidence: f64) -> Result<f64> {
    if !confidence.is_finite() {
        return Err(StoreError::validation(
            "confidence",
            confidence.to_string(),
            format!("Confidence must be a valid float, got {confidence}"),
        ));
    }
    if !(0.0..=1.0).contains(&confidence) {
        return Err(StoreError::validation(
            "confidence",
            confidence.to_string(),
            format!("Confidence must be between 0.0 and 1.0, got {confidence}"),
        ));
    }
    Ok(confidence)
}

/// Checks a vector for an index of `dim` dimensions: right length, finite values and a
/// non-zero norm (cosine indexes reject zero vectors).
///
/// # Errors
/// [`StoreError::Validation`] on field `embedding_vector`.
pub fn validate_vector(vector: &[f32], dim: usize) -> Result<()> {
    if vector.len() != dim {
        return Err(StoreError::validation(
            "embedding_vector",
            "",
            format!("embedding_vector has {} dimensions, expected {dim}", vector.len()),
        ));
    }
    if !vector.iter().all(|x| x.is_finite()) {
        return Err(StoreError::validation(
            "embedding_vector",
            "",
            "embedding_vector contains non-finite values",
        ));
    }
    if vector.iter().all(|x| *x == 0.0) {
        return Err(StoreError::validation(
            "embedding_vector",
            "",
            "embedding_vector must not be the zero vector",
        ));
    }
    Ok(())
}

/// Clamps a search limit to [`MAX_SEARCH_K`].
#[must_use]
pub fn clamp_limit(limit: usize) -> usize {
    limit.min(MAX_SEARCH_K)
}

/// `value` unless it is empty, then `default` (Python `value or default`).
#[must_use]
pub fn or_default<'a>(value: &'a str, default: &'a str) -> &'a str {
    if value.is_empty() {
        default
    } else {
        value
    }
}

/// Merges per-property hit lists by node (best score wins) and keeps the best `limit`;
/// Python `_text_search`.
pub(crate) fn merge_hits(lists: Vec<Vec<(u64, f64)>>, limit: usize) -> Vec<(u64, f64)> {
    let mut merged: Vec<(u64, f64)> = Vec::new();
    for list in lists {
        for (id, score) in list.into_iter().take(limit) {
            match merged.iter_mut().find(|(seen, _)| *seen == id) {
                Some(existing) if score > existing.1 => existing.1 = score,
                Some(_) => {}
                None => merged.push((id, score)),
            }
        }
    }
    merged.sort_by(|a, b| b.1.total_cmp(&a.1));
    merged.truncate(limit);
    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn field_of(error: &StoreError) -> &str {
        match error {
            StoreError::Validation { field, .. } => field,
            other => panic!("expected a validation error, got {other:?}"),
        }
    }

    #[test]
    fn entity_requires_name_and_type() {
        let error = EntityNode::new("", "PERSON").validated().expect_err("empty name");
        assert_eq!(field_of(&error), "Name");
        assert_eq!(error.to_string(), "Entity Name is required");
        let error = EntityNode::new("Maria", "").validated().expect_err("empty type");
        assert_eq!(field_of(&error), "EntityType");
    }

    #[test]
    fn entity_aliases_must_be_a_json_array() {
        let mut entity = EntityNode::new("Acme", "ORG");
        entity.aliases_json = String::new();
        assert_eq!(entity.validated().expect("empty aliases default").aliases_json, "[]");
        entity.aliases_json = r#"{"a": 1}"#.into();
        let error = entity.validated().expect_err("object is not an array");
        assert_eq!(field_of(&error), "AliasesJson");
        assert_eq!(error.to_string(), "AliasesJson must be a JSON array of strings");
        entity.aliases_json = "[not json".into();
        let error = entity.validated().expect_err("invalid JSON");
        assert!(error.to_string().starts_with("AliasesJson must be valid JSON: "));
        entity.aliases_json = r#"["Acme Corp.", "Acme", 3]"#.into();
        let ok = entity.validated().expect("an array of any values passes, as in Python");
        assert_eq!(ok.aliases(), vec!["Acme Corp.".to_owned(), "Acme".to_owned()]);
    }

    #[test]
    fn fact_validation_matches_python_builder() {
        let fact = FactNode::new("X paid Y", "{}", "{}", 0.9);
        fact.validate().expect("valid fact");
        let mut bad = fact.clone();
        bad.fact_text.clear();
        assert_eq!(field_of(&bad.validate().expect_err("empty text")), "FactText");
        for certainty in [-0.1, 1.01, f64::NAN, f64::INFINITY] {
            let mut bad = fact.clone();
            bad.certainty = certainty;
            assert_eq!(field_of(&bad.validate().expect_err("certainty")), "Certainty");
        }
        for certainty in [0.0, 1.0] {
            let mut ok = fact.clone();
            ok.certainty = certainty;
            ok.validate().expect("boundaries are inclusive");
        }
        let mut bad = fact.clone();
        bad.assertion_triple = "{".into();
        assert_eq!(field_of(&bad.validate().expect_err("triple")), "AssertionTriple");
        let mut bad = fact;
        bad.temporal_validity = String::new();
        assert_eq!(field_of(&bad.validate().expect_err("validity")), "TemporalValidity");
    }

    #[test]
    fn default_metadata_is_a_json_object() {
        let metadata = [
            DocumentNode::default().metadata,
            ChunkNode::default().metadata,
            SpanNode::default().metadata,
            EntityNode::default().metadata,
            FactNode::default().metadata,
        ];
        for text in metadata {
            let value: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
            assert!(value.is_object(), "{text}");
        }
        let chunk: ChunkNode = serde_json::from_str(r#"{"Text": "t"}"#).expect("partial chunk");
        assert_eq!(chunk.metadata, "{}");
    }

    #[test]
    fn fact_defaults_match_python_model() {
        let fact = FactNode::default();
        assert_eq!(fact.fact_origin, "STATED");
        assert_eq!(fact.fact_type, "STATED");
        assert_eq!(fact.status, "RAW");
        assert_eq!(fact.metadata, "{}");
        assert_eq!(fact.value_type, "TEXT");
        assert_eq!(fact.property_type, "ATTRIBUTE");
    }

    #[test]
    fn node_ids_parse_like_python() {
        assert_eq!(parse_node_id("doc_id", " 42 ").expect("trimmed"), 42);
        assert_eq!(field_of(&parse_node_id("doc_id", "").expect_err("empty")), "doc_id");
        assert!(parse_node_id("doc_id", "-1").is_err());
        assert!(parse_node_id("doc_id", "abc").is_err());
    }

    #[test]
    fn confidence_and_vectors_are_checked() {
        assert!(validate_confidence(0.5).is_ok());
        assert!(validate_confidence(1.5).is_err());
        assert!(validate_confidence(f64::NAN).is_err());
        assert!(validate_vector(&[1.0, 0.0], 2).is_ok());
        assert!(validate_vector(&[1.0], 2).is_err());
        assert!(validate_vector(&[f32::NAN, 1.0], 2).is_err());
        assert!(validate_vector(&[0.0, 0.0], 2).is_err());
        assert_eq!(clamp_limit(5000), MAX_SEARCH_K);
    }

    #[test]
    fn nodes_serialise_with_the_model_property_names() {
        let stored = StoredEntity { id: EntityId::new("3"), node: EntityNode::new("Acme", "ORG") };
        let value = serde_json::to_value(&stored).expect("serialises");
        assert_eq!(
            value,
            serde_json::json!({
                "id": "3", "Name": "Acme", "EntityType": "ORG", "AliasesJson": "[]",
                "Description": "", "Metadata": "{}"
            })
        );
        for (label, props) in NODE_PROPERTIES {
            let value = match label {
                DOCUMENT => serde_json::to_value(DocumentNode::default()),
                DOCUMENT_CHUNK => serde_json::to_value(ChunkNode::default()),
                SPAN => serde_json::to_value(SpanNode::default()),
                ENTITY => serde_json::to_value(EntityNode::default()),
                _ => serde_json::to_value(FactNode::default()),
            }
            .expect("serialises");
            let keys: Vec<&str> =
                value.as_object().expect("object").keys().map(String::as_str).collect();
            assert_eq!(keys, props.to_vec(), "{label}");
        }
        assert_eq!(node_properties(ENTITY).map(<[&str]>::len), Some(5));
        assert_eq!(node_properties("Nope"), None);
    }
}
