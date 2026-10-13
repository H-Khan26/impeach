//! Mention detection and defined-term extraction (Python `services/ner/`,
//! `models/gliner2.py` span logic and `stages/mention.py` per-chunk processing).
//!
//! - [`detectors`]: the three detector types of a `MentionStage` config: regex rules from
//!   `detection_patterns.yaml` ([`RegexDetector`], source `regex_detector`), legal citations
//!   ([`CitationDetector`], a regex port of the eyecite forms the pipeline uses, source
//!   `eyecite`) and the GLiNER2 adapter over [`impeach_models::NerModel`] ([`GlinerDetector`],
//!   source `gliner2`).
//! - [`quality`]: `quality_filters.yaml` rules, name normalisation and split-mention healing.
//! - [`merging`]: overlap resolution by `detector_config.yaml` priority with `sources`
//!   accumulation.
//! - [`chunk`]: the per-chunk flow of `MentionStage` (detect, score filter, normalise, heal,
//!   quality filter, document offsets, sentence index, merge).
//! - [`definitions`]: defined role terms such as `("Plaintiff")` from `roles.yaml` patterns.
//!
//! # Mention conventions
//!
//! - Offsets are character offsets (Python `str` indices). Detectors report chunk-local spans;
//!   [`chunk::MentionDetection::detect_chunk`] shifts them to document offsets.
//! - Ids: GLiNER2 mentions use the Python scheme `ent_{segment_id}_{i}` where `segment_id` is
//!   `{doc_id}_{chunk.sequence_number}` and `i` counts the mentions the adapter has kept for
//!   that chunk. Python gives regex and eyecite mentions a random `span-{uuid4 hex[:8]}`; here
//!   they are deterministic with the same shape: `span-` followed by the 8 lowercase hex digits
//!   of the 32-bit FNV-1a hash of `{segment_id}|{source}|{start}|{end}|{label}` (chunk-local
//!   offsets, `label` = the emitted mention type), see [`detectors::span_id`].
//! - Source names are the Python ones: `regex_detector`, `eyecite`, `gliner2`. The merge
//!   priorities in `detector_config.yaml` match sources by substring, so `gliner2` matches the
//!   `gliner` entries.

pub mod chunk;
pub mod definitions;
pub mod detectors;
pub mod error;
pub mod merging;
pub mod quality;
mod text;

pub use chunk::{ChunkView, ConfiguredDetector, MentionDetection};
pub use detectors::{CitationDetector, GlinerDetector, MentionDetector, RegexDetector};
pub use error::{NerError, Result};
pub use merging::merge_mentions;
pub use quality::MentionQuality;
