//! Mention detectors run per chunk (Python `services/ner/` detector classes).
//!
//! Every detector returns mentions with chunk-local character spans, `score`, `source` and
//! type set; the chunk flow ([`crate::chunk`]) handles filtering, offsets and merging.

pub mod citation;
pub mod gliner;
pub mod regex;

use ce_core::{Mention, MentionId};

pub use citation::CitationDetector;
pub use gliner::GlinerDetector;
pub use regex::RegexDetector;

use crate::error::Result;

/// A mention detector (Python `BaseMentionDetector` protocol).
pub trait MentionDetector: Send + Sync + std::fmt::Debug {
    /// Source name written to [`Mention::source`] (`regex_detector`, `eyecite`, `gliner2`).
    fn source(&self) -> &str;

    /// Mentions in `text` with chunk-local character spans; `segment_id` is
    /// `{doc_id}_{chunk.sequence_number}` and seeds the mention ids.
    ///
    /// # Errors
    /// Model failures (GLiNER2); the rule detectors never fail.
    fn detect(&self, text: &str, segment_id: &str) -> Result<Vec<Mention>>;
}

/// 32-bit FNV-1a hash of `bytes`.
#[must_use]
pub fn fnv1a32(bytes: &[u8]) -> u32 {
    const OFFSET_BASIS: u32 = 0x811c_9dc5;
    const PRIME: u32 = 0x0100_0193;
    bytes.iter().fold(OFFSET_BASIS, |hash, &b| (hash ^ u32::from(b)).wrapping_mul(PRIME))
}

/// Deterministic id of a rule-detected mention: `span-` and the 8 lowercase hex digits of
/// [`fnv1a32`] over `{segment_id}|{source}|{start}|{end}|{label}` (chunk-local offsets).
#[must_use]
pub fn span_id(segment_id: &str, source: &str, start: usize, end: usize, label: &str) -> MentionId {
    let key = format!("{segment_id}|{source}|{start}|{end}|{label}");
    MentionId::from(format!("span-{:08x}", fnv1a32(key.as_bytes())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_reference_values() {
        assert_eq!(fnv1a32(b""), 0x811c_9dc5);
        assert_eq!(fnv1a32(b"a"), 0xe40c_292c);
        assert_eq!(fnv1a32(b"foobar"), 0xbf9c_f968);
    }

    #[test]
    fn span_ids_are_stable_and_shaped_like_python() {
        let id = span_id("doc_0", "regex_detector", 3, 10, "DATE");
        assert_eq!(id, span_id("doc_0", "regex_detector", 3, 10, "DATE"));
        assert_ne!(id, span_id("doc_1", "regex_detector", 3, 10, "DATE"));
        let hex = id.as_str().strip_prefix("span-").expect("prefix");
        assert_eq!(hex.len(), 8);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }
}
