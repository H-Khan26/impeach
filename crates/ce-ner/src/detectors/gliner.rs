//! GLiNER2 mention detector (Python `services/ner/gliner2.py` and the span logic of
//! `models/gliner2.py:GLiNER2Service._align_entity_dict`).
//!
//! The model call goes through [`NerModel`] with the label set from `gliner2.yaml` and the
//! `models.gliner2.threshold`; the stage's `min_score` is applied later by the chunk flow.
//! Model labels map to entity types through `gliner2.yaml` `gliner2_to_entity_type`
//! (lowercased label), falling back to the `entity_types.yaml` default type (`OTHER`).
//!
//! Span handling follows `_align_entity_dict`: spans with invalid offsets (start >= end or
//! past the text) are skipped with a warning; the mention name is the text under the span;
//! a span claimed twice for the same entity type is kept once; ids are
//! `ent_{segment_id}_{i}` with `i` the number of mentions kept before it, counted while
//! walking the spans grouped by label in label-set order (the order of the model's
//! `{label: [items]}` output); the result is stably sorted by start offset. [`NerModel`]
//! backends always report offsets, so Python's whole-word string alignment for offset-less
//! items has no counterpart here.

use std::collections::HashSet;
use std::sync::Arc;

use ce_config::MappingsRegistry;
use ce_core::{Mention, Span};
use ce_models::{NerModel, NerSpan};
use indexmap::IndexMap;

use super::MentionDetector;
use crate::error::{NerError, Result};
use crate::text::CharIndex;

/// Runs a GLiNER2 [`NerModel`] over chunk text.
pub struct GlinerDetector {
    model: Arc<dyn NerModel>,
    labels: Vec<String>,
    threshold: f32,
    label_map: IndexMap<String, String>,
    default_type: String,
}

impl std::fmt::Debug for GlinerDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlinerDetector")
            .field("labels", &self.labels)
            .field("threshold", &self.threshold)
            .finish_non_exhaustive()
    }
}

impl GlinerDetector {
    /// Source name of GLiNER2 mentions.
    pub const SOURCE: &'static str = "gliner2";

    /// A detector over `model` with the entity label set `label_set` of `gliner2.yaml`;
    /// `threshold` is the model's score threshold (`models.gliner2.threshold`).
    ///
    /// # Errors
    /// [`NerError::Config`] for an unknown label set.
    pub fn new(
        model: Arc<dyn NerModel>,
        mappings: &MappingsRegistry,
        label_set: &str,
        threshold: f32,
    ) -> Result<Self> {
        let labels = mappings.gliner2.get_entity_label_set(label_set)?.to_vec();
        if labels.is_empty() {
            return Err(NerError::InvalidConfig(format!(
                "GLiNER2 label set '{label_set}' has no labels"
            )));
        }
        tracing::info!(label_set, labels = labels.len(), "GLiNER2 mention detector ready");
        Ok(Self {
            model,
            labels,
            threshold,
            label_map: mappings.gliner2.gliner2_to_entity_type.clone(),
            default_type: mappings.entity_types.default_entity_type().to_owned(),
        })
    }

    /// Labels passed to the model.
    #[must_use]
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Entity type of a model label: `gliner2_to_entity_type[label.lower()]`, else the
    /// default type.
    #[must_use]
    pub fn entity_type(&self, label: &str) -> &str {
        self.label_map.get(&label.to_lowercase()).map_or(&self.default_type, String::as_str)
    }

    /// Mentions from model spans over `text` (the alignment rules in the module docs).
    #[must_use]
    pub fn mentions_from_spans(
        &self,
        spans: Vec<NerSpan>,
        text: &str,
        segment_id: &str,
    ) -> Vec<Mention> {
        let index = CharIndex::new(text);
        let len = index.char_len();
        let rank = |label: &str| self.labels.iter().position(|l| l == label).unwrap_or(usize::MAX);
        let mut spans = spans;
        spans.sort_by_key(|s| rank(&s.label));

        let mut seen: HashSet<(usize, usize, String)> = HashSet::new();
        let mut mentions: Vec<Mention> = Vec::new();
        for span in spans {
            if span.text.is_empty() {
                continue;
            }
            if span.start >= span.end || span.end > len {
                tracing::warn!(
                    start = span.start,
                    end = span.end,
                    len,
                    "GLiNER2 span out of range for the text, skipping"
                );
                continue;
            }
            let entity_type = self.entity_type(&span.label).to_owned();
            if !seen.insert((span.start, span.end, entity_type.clone())) {
                continue;
            }
            let id = format!("ent_{segment_id}_{}", mentions.len());
            let name = index.slice(span.start, span.end);
            let mut mention = Mention::new(id, entity_type, name, Span::new(span.start, span.end));
            mention.score = Some(f64::from(span.score));
            mention.source = Some(Self::SOURCE.to_owned());
            mentions.push(mention);
        }
        mentions.sort_by_key(|m| m.span.start);
        mentions
    }
}

impl MentionDetector for GlinerDetector {
    fn source(&self) -> &str {
        Self::SOURCE
    }

    fn detect(&self, text: &str, segment_id: &str) -> Result<Vec<Mention>> {
        let labels: Vec<&str> = self.labels.iter().map(String::as_str).collect();
        let spans = self.model.extract_entities(text, &labels, self.threshold)?;
        Ok(self.mentions_from_spans(spans, text, segment_id))
    }
}

#[cfg(test)]
mod tests {
    use ce_models::{MockNer, ModelError, RelationHit};
    use pretty_assertions::assert_eq;

    use super::*;

    fn span(start: usize, end: usize, label: &str, score: f32, text: &str) -> NerSpan {
        NerSpan { start, end, label: label.into(), score, text: text.into() }
    }

    fn detector(spans: Vec<NerSpan>) -> GlinerDetector {
        let mappings = MappingsRegistry::load().expect("mappings");
        GlinerDetector::new(
            Arc::new(MockNer::new(spans, Vec::new())),
            mappings,
            "legal_entities",
            0.4,
        )
        .expect("detector")
    }

    /// `(id, type, name, start, end)` of each mention.
    fn rows(mentions: &[Mention]) -> Vec<(String, String, String, usize, usize)> {
        mentions
            .iter()
            .map(|m| {
                (m.id.to_string(), m.entity_type.clone(), m.name.clone(), m.span.start, m.span.end)
            })
            .collect()
    }

    #[test]
    fn labels_map_through_gliner2_yaml() {
        let d = detector(Vec::new());
        assert_eq!(d.labels().len(), 18);
        assert_eq!(d.entity_type("person"), "PER");
        assert_eq!(d.entity_type("Judge"), "PER");
        assert_eq!(d.entity_type("monetary amount"), "MONEY");
        assert_eq!(d.entity_type("legal case"), "CASE");
        assert_eq!(d.entity_type("spaceship"), "OTHER");
    }

    #[test]
    fn unknown_label_set_is_a_config_error() {
        let mappings = MappingsRegistry::load().expect("mappings");
        let err = GlinerDetector::new(Arc::new(MockNer::default()), mappings, "nope", 0.4)
            .expect_err("unknown set");
        assert!(matches!(err, NerError::Config(_)), "{err}");
    }

    // Port of test_gliner2_spans.py: every mention's name is text[start:end], spans are in
    // range, the source is gliner2.
    #[test]
    fn mention_names_are_the_text_under_their_spans() {
        let text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday.";
        let d = detector(vec![
            span(0, 5, "company", 0.9, "Apple"),
            span(10, 18, "person", 0.95, "Tim Cook"),
            span(29, 38, "product", 0.7, "iPhone 15"),
            span(42, 51, "location", 0.8, "Cupertino"),
        ]);
        let mentions = d.detect(text, "test_segment").expect("detect");
        assert_eq!(mentions.len(), 4);
        let chars: Vec<char> = text.chars().collect();
        for m in &mentions {
            assert!(m.span.start < m.span.end && m.span.end <= chars.len());
            let under: String = chars[m.span.start..m.span.end].iter().collect();
            assert_eq!(under, m.name);
            assert_eq!(m.source.as_deref(), Some("gliner2"));
        }
        assert_eq!(
            rows(&mentions),
            vec![
                ("ent_test_segment_1".into(), "ORG".into(), "Apple".into(), 0, 5),
                ("ent_test_segment_0".into(), "PER".into(), "Tim Cook".into(), 10, 18),
                ("ent_test_segment_3".into(), "PRODUCT".into(), "iPhone 15".into(), 29, 38),
                ("ent_test_segment_2".into(), "LOC".into(), "Cupertino".into(), 42, 51),
            ]
        );
        assert_eq!(mentions[1].score, Some(f64::from(0.95_f32)));
    }

    #[test]
    fn invalid_spans_are_skipped_and_duplicates_dropped() {
        let text = "John Smith works for Acme Corporation in New York.";
        let d = detector(vec![
            span(0, 10, "person", 0.9, "John Smith"),
            span(0, 10, "judge", 0.8, "John Smith"),
            span(21, 37, "organization", 0.9, "Acme Corporation"),
            span(21, 37, "company", 0.85, "Acme Corporation"),
            span(41, 41, "location", 0.9, "x"),
            span(41, 99, "location", 0.9, "New York."),
            span(41, 49, "location", 0.9, ""),
        ]);
        let mentions = d.detect(text, "test_multi").expect("detect");
        assert_eq!(
            rows(&mentions),
            vec![
                ("ent_test_multi_0".into(), "PER".into(), "John Smith".into(), 0, 10),
                ("ent_test_multi_1".into(), "ORG".into(), "Acme Corporation".into(), 21, 37),
            ]
        );
    }

    #[test]
    fn names_use_character_offsets() {
        let d = detector(Vec::new());
        let mentions = d.mentions_from_spans(
            vec![span(0, 9, "person", 0.9, "Zoë Smith"), span(15, 19, "company", 0.8, "Acme")],
            "Zoë Smith sued Acme Corp.",
            "s",
        );
        assert_eq!(mentions[0].name, "Zoë Smith");
        assert_eq!(mentions[1].name, "Acme");
    }

    #[derive(Debug)]
    struct FailingNer;

    impl NerModel for FailingNer {
        fn extract_entities(
            &self,
            _: &str,
            _: &[&str],
            _: f32,
        ) -> Result<Vec<NerSpan>, ModelError> {
            Err(ModelError::Inference("boom".into()))
        }

        fn extract_relations(
            &self,
            _: &str,
            _: &[&str],
            _: f32,
        ) -> Result<Vec<RelationHit>, ModelError> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn model_failures_propagate() {
        let mappings = MappingsRegistry::load().expect("mappings");
        let d =
            GlinerDetector::new(Arc::new(FailingNer), mappings, "minimal", 0.4).expect("detector");
        assert!(matches!(d.detect("text", "s"), Err(NerError::Model(_))));
    }
}
