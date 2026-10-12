//! ED contexts of mentions (Python `services/ed/context_builder.py`): the sentence window of
//! the mention's chunk and document-level metadata.

use ce_config::mappings::EntityTypes;
use ce_core::{ChunkWriteResult, EdContext, Mention, Metadata};

/// Builds [`EdContext`]s from the indexed chunks and the document metadata.
#[derive(Debug)]
pub struct ContextBuilder<'a> {
    chunks: &'a [ChunkWriteResult],
    doc_metadata: &'a Metadata,
    entity_types: &'a EntityTypes,
}

impl<'a> ContextBuilder<'a> {
    /// A builder over the chunks written by the RAG index stage (looked up by graph id) and
    /// the document metadata (`doc_type`, `case_id`, `jurisdiction`, `party_names`).
    #[must_use]
    pub fn new(
        chunks: &'a [ChunkWriteResult],
        doc_metadata: &'a Metadata,
        entity_types: &'a EntityTypes,
    ) -> Self {
        Self { chunks, doc_metadata, entity_types }
    }

    /// Context of `mention`, or `None` when its `chunk_id` is missing or not one of the chunks.
    /// `span_label` is the mention type normalised with [`EntityTypes::normalize_label`].
    #[must_use]
    pub fn build_context(&self, mention: &Mention) -> Option<EdContext> {
        let _ = (mention, self.chunks, self.doc_metadata, self.entity_types);
        todo!("worker B")
    }
}
