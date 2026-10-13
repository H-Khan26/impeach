//! Synthetic pronoun mentions (Python `CoreferenceStage._synthesize_pronoun_mentions`).

use impeach_core::{ChunkId, ChunkWriteResult, DocId, Mention, MentionId, Metadata, Span};
use indexmap::IndexMap;
use serde_json::Value;

use crate::clusters::{ClusterContext, MentionsById};
use crate::text::CharText;

/// `source` of synthetic pronoun mentions.
pub const PRONOUN_SOURCE: &str = "coref_pronoun_synthesis";

/// Synthetic pronoun mentions of a document.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PronounSynthesis {
    /// One mention per pronoun span of each cluster with a canonical mention, in cluster order.
    pub(crate) mentions: Vec<Mention>,
    /// The same mentions grouped by the graph id of the chunk containing them (mentions
    /// outside every chunk are left out).
    pub(crate) by_chunk: IndexMap<ChunkId, Vec<Mention>>,
    /// Synthetic mention id -> canonical mention id.
    pub(crate) synthetic_to_canonical: IndexMap<MentionId, MentionId>,
}

/// A chunk's character range, logical id (`<doc_id>_<sequence_number>`) and graph id.
struct ChunkRange<'a> {
    start: usize,
    end: usize,
    chunk_id: String,
    helix_id: &'a str,
}

/// Chunk ranges sorted by start.
fn chunk_index(chunks: &[ChunkWriteResult]) -> Vec<ChunkRange<'_>> {
    let mut index: Vec<ChunkRange<'_>> = chunks
        .iter()
        .map(|w| ChunkRange {
            start: w.chunk.start_char,
            end: w.chunk.end_char,
            chunk_id: format!("{}_{}", w.chunk.doc_id, w.chunk.sequence_number),
            helix_id: w.helix_id.as_str(),
        })
        .collect();
    index.sort_by_key(|c| c.start);
    index
}

/// The first chunk (by start) containing or overlapping `span`.
fn chunk_for<'i, 'a>(span: Span, index: &'i [ChunkRange<'a>]) -> Option<&'i ChunkRange<'a>> {
    for chunk in index {
        if chunk.start > span.end {
            break;
        }
        let contained = chunk.start <= span.start && span.end <= chunk.end;
        let overlaps = (chunk.start <= span.start && span.start < chunk.end)
            || (chunk.start < span.end && span.end <= chunk.end);
        if contained || overlaps {
            return Some(chunk);
        }
    }
    None
}

/// Turns every pronoun span of `context`'s clusters into a mention carrying the canonical
/// mention's name and type: id `synth_pronoun_<cluster id>_<start>_<end>`, score
/// `confidence`, source [`PRONOUN_SOURCE`], chunk id `<doc_id>_<sequence_number>` of the
/// containing chunk, metadata `synthetic_pronoun`, `original_pronoun_text`, `cluster_id`,
/// `canonical_mention_id`, `canonical_name`. Clusters without a canonical mention are skipped.
pub(crate) fn synthesize(
    context: &ClusterContext,
    by_id: &MentionsById<'_>,
    chunks: &[ChunkWriteResult],
    text: &CharText<'_>,
    doc_id: &DocId,
    confidence: f64,
) -> PronounSynthesis {
    let index = chunk_index(chunks);
    let mut out = PronounSynthesis::default();
    for cluster in &context.clusters {
        if cluster.pronoun_spans.is_empty() {
            continue;
        }
        let Some(canonical_id) = context.canonical_mention_for_cluster.get(&cluster.id) else {
            tracing::debug!(cluster = %cluster.id, "pronoun synthesis: no canonical mention");
            continue;
        };
        let Some(canonical) = by_id.get(canonical_id.as_str()) else {
            tracing::debug!(cluster = %cluster.id, "pronoun synthesis: canonical mention missing");
            continue;
        };
        for &span in &cluster.pronoun_spans {
            let pronoun_text = if span.start < span.end {
                text.slice(span.start, span.end).unwrap_or_default()
            } else {
                ""
            };
            let chunk = chunk_for(span, &index);
            let mut metadata = Metadata::new();
            metadata.insert("synthetic_pronoun".to_owned(), Value::Bool(true));
            metadata.insert("original_pronoun_text".to_owned(), pronoun_text.into());
            metadata.insert("cluster_id".to_owned(), cluster.id.clone().into());
            metadata.insert("canonical_mention_id".to_owned(), canonical_id.as_str().into());
            metadata.insert("canonical_name".to_owned(), canonical.name.clone().into());
            let mut mention = Mention::new(
                MentionId::new(format!("synth_pronoun_{}_{}_{}", cluster.id, span.start, span.end)),
                canonical.entity_type.clone(),
                canonical.name.clone(),
                span,
            );
            mention.score = Some(confidence);
            mention.source = Some(PRONOUN_SOURCE.to_owned());
            mention.doc_id = Some(doc_id.clone());
            mention.chunk_id = chunk.map(|c| ChunkId::new(c.chunk_id.clone()));
            mention.metadata = metadata;
            if let Some(chunk) = chunk.filter(|c| !c.helix_id.is_empty()) {
                out.by_chunk.entry(ChunkId::new(chunk.helix_id)).or_default().push(mention.clone());
            }
            out.synthetic_to_canonical.insert(mention.id.clone(), canonical_id.clone());
            out.mentions.push(mention);
        }
    }
    if !out.mentions.is_empty() {
        let clusters = context.clusters.iter().filter(|c| !c.pronoun_spans.is_empty()).count();
        tracing::info!(
            mentions = out.mentions.len(),
            clusters,
            "coreference: synthesized pronoun mentions"
        );
    }
    out
}
