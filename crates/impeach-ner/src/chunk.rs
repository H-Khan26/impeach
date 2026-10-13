//! Per-chunk mention detection: the body of Python `MentionStage.run` for one chunk.
//!
//! 1. Every detector runs on the chunk text with segment id `{doc_id}_{sequence_number}`.
//! 2. Mentions scoring below the detector's `min_score` are dropped (mentions without a score
//!    are kept); names are normalised; types go through `EntityTypes::normalize_label`.
//! 3. Split shards are healed ([`MentionQuality::heal_split_mentions`]).
//! 4. Low-quality mentions are dropped.
//! 5. Spans shift to document offsets; `doc_id`, `chunk_id` and the index of the chunk
//!    sentence span containing the mention start are set.
//! 6. Overlaps are merged ([`merge_mentions`]).

use std::sync::Arc;

use impeach_config::mappings::{DetectorConfig, EntityTypes};
use impeach_config::MappingsRegistry;
use impeach_core::{ChunkId, DocId, Mention, Span};

use crate::detectors::MentionDetector;
use crate::error::Result;
use crate::merging::merge_mentions;
use crate::quality::MentionQuality;

/// A detector with its stage-level `min_score`.
#[derive(Debug, Clone)]
pub struct ConfiguredDetector {
    /// The detector.
    pub detector: Arc<dyn MentionDetector>,
    /// Mentions with a lower score are dropped (0.0 keeps everything).
    pub min_score: f64,
}

impl ConfiguredDetector {
    /// A detector with `min_score`.
    pub fn new(detector: Arc<dyn MentionDetector>, min_score: f64) -> Self {
        Self { detector, min_score }
    }
}

/// The chunk fields mention detection reads.
#[derive(Debug, Clone, Copy)]
pub struct ChunkView<'a> {
    /// Chunk text.
    pub text: &'a str,
    /// Document id.
    pub doc_id: &'a DocId,
    /// Graph id of the chunk, written to [`Mention::chunk_id`].
    pub chunk_id: &'a ChunkId,
    /// Position of the chunk in the document.
    pub sequence_number: usize,
    /// Document offset of the chunk start.
    pub start_char: usize,
    /// Sentence spans of the chunk (document offsets).
    pub sentence_spans: &'a [Span],
}

impl ChunkView<'_> {
    /// `{doc_id}_{sequence_number}`, the segment id passed to detectors.
    #[must_use]
    pub fn segment_id(&self) -> String {
        format!("{}_{}", self.doc_id, self.sequence_number)
    }

    /// Index of the first sentence span containing document offset `start`.
    #[must_use]
    pub fn sentence_index(&self, start: usize) -> Option<usize> {
        self.sentence_spans.iter().position(|s| s.start <= start && start < s.end)
    }
}

/// Mention detection settings shared by every chunk of a run.
#[derive(Debug, Clone)]
pub struct MentionDetection {
    quality: MentionQuality,
    entity_types: EntityTypes,
    detector_config: DetectorConfig,
}

impl MentionDetection {
    /// Settings from the mappings registry; `normalize_names` is the stage switch.
    #[must_use]
    pub fn new(mappings: &MappingsRegistry, normalize_names: bool) -> Self {
        Self {
            quality: MentionQuality::new(mappings.quality_filters.clone(), normalize_names),
            entity_types: mappings.entity_types.clone(),
            detector_config: mappings.detector_config.clone(),
        }
    }

    /// The quality rules.
    #[must_use]
    pub fn quality(&self) -> &MentionQuality {
        &self.quality
    }

    /// Detected, healed, filtered and merged mentions of one chunk, with document offsets
    /// (steps in the module docs).
    ///
    /// # Errors
    /// The first detector failure.
    pub fn detect_chunk(
        &self,
        detectors: &[ConfiguredDetector],
        chunk: &ChunkView<'_>,
    ) -> Result<Vec<Mention>> {
        let segment_id = chunk.segment_id();
        let mut candidates = Vec::new();
        for configured in detectors {
            for mut mention in configured.detector.detect(chunk.text, &segment_id)? {
                if mention.score.is_some_and(|score| score < configured.min_score) {
                    continue;
                }
                if self.quality.normalize_names() {
                    mention.name = self.quality.normalize_name(&mention.name);
                }
                if !mention.entity_type.is_empty() {
                    let label = self.entity_types.normalize_label(&mention.entity_type);
                    if label != mention.entity_type {
                        mention.entity_type = label.to_owned();
                    }
                }
                candidates.push(mention);
            }
        }
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let healed = self.quality.heal_split_mentions(candidates, chunk.text);
        let mut kept = Vec::with_capacity(healed.len());
        for mut mention in healed {
            if self.quality.is_low_quality(&mention) {
                continue;
            }
            mention.span.start += chunk.start_char;
            mention.span.end += chunk.start_char;
            mention.doc_id = Some(chunk.doc_id.clone());
            mention.chunk_id = Some(chunk.chunk_id.clone());
            if mention.sentence_index.is_none() {
                mention.sentence_index = chunk.sentence_index(mention.span.start);
            }
            kept.push(mention);
        }
        if kept.is_empty() {
            return Ok(kept);
        }
        Ok(merge_mentions(kept, &self.detector_config))
    }
}

#[cfg(test)]
mod tests {
    use impeach_core::DetectionSource;
    use pretty_assertions::assert_eq;

    use super::*;

    /// Returns fixed mentions (Python `_FakeDetector`).
    #[derive(Debug)]
    struct Fixed(Vec<Mention>);

    impl MentionDetector for Fixed {
        fn source(&self) -> &str {
            "fixed"
        }

        fn detect(&self, _text: &str, _segment_id: &str) -> Result<Vec<Mention>> {
            Ok(self.0.clone())
        }
    }

    fn mention(ty: &str, name: &str, start: usize, end: usize, score: Option<f64>) -> Mention {
        let mut m = Mention::new(format!("m{start}"), ty, name, Span::new(start, end));
        m.score = score;
        m.source = Some("gliner2".into());
        m
    }

    fn detection() -> MentionDetection {
        MentionDetection::new(MappingsRegistry::load().expect("mappings"), true)
    }

    // Port of test_mention_stage_run_applies_split_healing_before_merging (offsets and ids).
    #[test]
    fn heals_before_filtering_and_merging() {
        let text = "Plainti ff sued the Defendant.";
        let (doc_id, chunk_id) = (DocId::from("doc-1"), ChunkId::from("chunk-1"));
        let chunk = ChunkView {
            text,
            doc_id: &doc_id,
            chunk_id: &chunk_id,
            sequence_number: 0,
            start_char: 0,
            sentence_spans: &[],
        };
        let detector = Fixed(vec![
            mention("PER", "Plainti", 0, 7, Some(0.95)),
            mention("PER", "ff", 8, 10, Some(0.95)),
        ]);
        let found = detection()
            .detect_chunk(&[ConfiguredDetector::new(Arc::new(detector), 0.0)], &chunk)
            .expect("detect");
        assert_eq!(found.len(), 1);
        let m = &found[0];
        assert_eq!((m.span.start, m.span.end), (0, 10));
        assert_eq!(m.name, "Plainti ff");
        assert_eq!(m.doc_id.as_ref(), Some(&doc_id));
        assert_eq!(m.chunk_id.as_ref(), Some(&chunk_id));
    }

    #[test]
    fn min_score_normalisation_offsets_and_sentences() {
        let text = "Thomas\nReynolds met ann at Acme Corp.";
        let (doc_id, chunk_id) = (DocId::from("d"), ChunkId::from("7"));
        let sentences = [Span::new(90, 105), Span::new(105, 200)];
        let chunk = ChunkView {
            text,
            doc_id: &doc_id,
            chunk_id: &chunk_id,
            sequence_number: 3,
            start_char: 100,
            sentence_spans: &sentences,
        };
        assert_eq!(chunk.segment_id(), "d_3");
        let detector = Fixed(vec![
            mention("person", "Thomas\nReynolds", 0, 15, Some(0.9)),
            mention("PER", "ann", 20, 23, Some(0.3)),
            mention("ORG", "Acme Corp", 27, 36, None),
        ]);
        let found = detection()
            .detect_chunk(&[ConfiguredDetector::new(Arc::new(detector), 0.5)], &chunk)
            .expect("detect");
        let rows: Vec<(String, String, usize, usize, Option<usize>)> = found
            .iter()
            .map(|m| {
                (m.entity_type.clone(), m.name.clone(), m.span.start, m.span.end, m.sentence_index)
            })
            .collect();
        assert_eq!(
            rows,
            vec![
                ("PER".into(), "Thomas Reynolds".into(), 100, 115, Some(0)),
                ("ORG".into(), "Acme Corp".into(), 127, 136, Some(1)),
            ]
        );
        assert_eq!(
            found[0].sources,
            vec![DetectionSource { name: "gliner2".into(), label: "PER".into(), score: Some(0.9) }]
        );
    }

    #[test]
    fn low_quality_mentions_are_dropped_and_overlaps_merged() {
        let text = "On March 4, 2023 Ms. Lo paid $300.";
        let (doc_id, chunk_id) = (DocId::from("d"), ChunkId::from("c"));
        let chunk = ChunkView {
            text,
            doc_id: &doc_id,
            chunk_id: &chunk_id,
            sequence_number: 0,
            start_char: 0,
            sentence_spans: &[],
        };
        let gliner = Fixed(vec![
            mention("date", "March 4, 2023", 3, 16, Some(0.8)),
            mention("PER", "Lo", 21, 23, Some(0.99)),
        ]);
        let mappings = MappingsRegistry::load().expect("mappings");
        let regex = crate::RegexDetector::from_mappings(&mappings.detection_patterns);
        let found = detection()
            .detect_chunk(
                &[
                    ConfiguredDetector::new(Arc::new(gliner), 0.4),
                    ConfiguredDetector::new(Arc::new(regex), 0.0),
                ],
                &chunk,
            )
            .expect("detect");
        let names: Vec<(&str, &str)> =
            found.iter().map(|m| (m.name.as_str(), m.source.as_deref().unwrap_or(""))).collect();
        assert_eq!(names, [("March 4, 2023", "regex_detector"), ("$300", "regex_detector")]);
        assert_eq!(found[0].sources.len(), 2);
    }
}
