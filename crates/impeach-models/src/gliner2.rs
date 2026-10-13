//! GLiNER2 entities and relations through the `gliner2` crate (candle, CPU). Replaces the
//! Python `GLiNER2Service`; mapping labels to entity types stays with the caller.
//!
//! The crate reports byte offsets; [`Gliner2Ner`] converts them to character offsets. A
//! relation's score is the lower of its head and tail confidences (GLiNER2 scores the two
//! argument spans, not the relation).

use std::collections::HashMap;
use std::path::PathBuf;

use ::gliner2::config::{download_model, ModelFiles};
use ::gliner2::schema::RelationTypesInput;
use ::gliner2::{CandleExtractor, ExtractOptions, ExtractorConfig, SchemaTransformer};
use impeach_config::pipeline::Gliner2ModelConfig;
use serde_json::Value;

use crate::error::{missing_model, ModelError};
use crate::traits::{NerModel, NerSpan, RelationHit};

/// Settings of a [`Gliner2Ner`] model.
#[derive(Debug, Clone, PartialEq)]
pub struct Gliner2Options {
    /// Hugging Face repo id, e.g. `fastino/gliner2-base-v1`.
    pub model: String,
    /// Directory with `config.json`, `encoder_config/config.json`, `tokenizer.json` and
    /// `model.safetensors`; `None` downloads `model` into the Hugging Face cache.
    pub model_dir: Option<PathBuf>,
}

impl Gliner2Options {
    /// Options for repo `model`, downloaded on first use.
    pub fn new(model: impl Into<String>) -> Self {
        Self { model: model.into(), model_dir: None }
    }

    /// Options from the `models.gliner2` config block (the threshold is passed per call).
    pub fn from_config(config: &Gliner2ModelConfig) -> Self {
        Self::new(config.model.clone())
    }
}

/// A GLiNER2 model on the CPU.
pub struct Gliner2Ner {
    name: String,
    extractor: CandleExtractor,
    transformer: SchemaTransformer,
}

impl std::fmt::Debug for Gliner2Ner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gliner2Ner").field("name", &self.name).finish_non_exhaustive()
    }
}

impl Gliner2Ner {
    /// Loads the model (downloading it unless `options.model_dir` is set).
    pub fn new(options: &Gliner2Options) -> Result<Self, ModelError> {
        let load_error =
            |message: String| ModelError::Load { model: options.model.clone(), message };
        let files = match &options.model_dir {
            Some(dir) => ModelFiles {
                config: dir.join("config.json"),
                encoder_config: dir.join("encoder_config").join("config.json"),
                tokenizer: dir.join("tokenizer.json"),
                weights: dir.join("model.safetensors"),
            },
            None => download_model(&options.model).map_err(|e| {
                missing_model(
                    &options.model,
                    &crate::error::hf_hub_cache_dir(false),
                    "copy the repo's files into that Hugging Face cache (the GLiNER2 backend \
                     does not read HF_HOME) or set models.gliner2.model to a cached repo",
                    &format!("{e:#}"),
                )
            })?,
        };
        if let Some(dir) = &options.model_dir {
            let missing: Vec<String> =
                [&files.config, &files.encoder_config, &files.tokenizer, &files.weights]
                    .into_iter()
                    .filter(|path| !path.is_file())
                    .filter_map(|path| path.strip_prefix(dir).ok())
                    .map(|path| path.display().to_string())
                    .collect();
            if !missing.is_empty() {
                return Err(load_error(format!(
                    "model directory {} lacks {}; put the files of `{}` there (config.json, \
                     encoder_config/config.json, tokenizer.json, model.safetensors)",
                    dir.display(),
                    missing.join(", "),
                    options.model
                )));
            }
        }
        let tokenizer = files
            .tokenizer
            .to_str()
            .ok_or_else(|| load_error("tokenizer path is not UTF-8".to_owned()))?;
        let transformer =
            SchemaTransformer::new(tokenizer).map_err(|e| load_error(format!("{e:#}")))?;
        let config_text = std::fs::read_to_string(&files.config)
            .map_err(|e| load_error(format!("cannot read {}: {e}", files.config.display())))?;
        let config: ExtractorConfig = serde_json::from_str(&config_text)
            .map_err(|e| load_error(format!("invalid {}: {e}", files.config.display())))?;
        let vocab = transformer.tokenizer.get_vocab_size(true);
        tracing::info!(model = %options.model, "loading GLiNER2 model");
        let extractor = CandleExtractor::load_cpu(&files, config, vocab)
            .map_err(|e| load_error(format!("{e:#}")))?;
        Ok(Self { name: options.model.clone(), extractor, transformer })
    }

    /// Model id.
    pub fn model_name(&self) -> &str {
        &self.name
    }
}

fn options(threshold: f32) -> ExtractOptions {
    ExtractOptions {
        threshold,
        include_confidence: true,
        include_spans: true,
        ..ExtractOptions::default()
    }
}

impl NerModel for Gliner2Ner {
    fn extract_entities(
        &self,
        text: &str,
        labels: &[&str],
        threshold: f32,
    ) -> Result<Vec<NerSpan>, ModelError> {
        if labels.is_empty() || text.is_empty() {
            return Ok(Vec::new());
        }
        let labels: Vec<String> = labels.iter().map(|l| (*l).to_owned()).collect();
        let output = self
            .extractor
            .extract_entities(&self.transformer, text, &labels, &options(threshold))
            .map_err(|e| ModelError::Inference(format!("GLiNER2 entity extraction: {e:#}")))?;
        let value = serde_json::to_value(&output)
            .map_err(|e| ModelError::Protocol(format!("GLiNER2 output: {e}")))?;
        entities_from_output(&value, text)
    }

    fn extract_relations(
        &self,
        text: &str,
        labels: &[&str],
        threshold: f32,
    ) -> Result<Vec<RelationHit>, ModelError> {
        if labels.is_empty() || text.is_empty() {
            return Ok(Vec::new());
        }
        let relations = RelationTypesInput::Many(labels.iter().map(|l| (*l).to_owned()).collect());
        let output = self
            .extractor
            .extract_relations(&self.transformer, text, relations, &options(threshold))
            .map_err(|e| ModelError::Inference(format!("GLiNER2 relation extraction: {e:#}")))?;
        let value = serde_json::to_value(&output)
            .map_err(|e| ModelError::Protocol(format!("GLiNER2 output: {e}")))?;
        relations_from_output(&value, text)
    }
}

/// Spans from formatted output `{"entities": {label: [{text, confidence, start, end}]}}`,
/// sorted by start offset (stable within equal starts).
///
/// The crate reports each name once per label; every other whole-word occurrence of a reported
/// name is added with the same label and score, as the Python GLiNER2 reports every occurrence.
fn entities_from_output(output: &Value, text: &str) -> Result<Vec<NerSpan>, ModelError> {
    let Some(entities) = output.get("entities").and_then(Value::as_object) else {
        return Ok(Vec::new());
    };
    let mut spans = Vec::new();
    for (label, items) in entities {
        for item in items.as_array().into_iter().flatten() {
            if let Some(span) = span_from_item(item, label, text)? {
                spans.push(span);
            }
        }
    }
    add_other_occurrences(&mut spans, text);
    spans.sort_by_key(|s| s.start);
    Ok(spans)
}

/// Adds every whole-word occurrence of a reported span's text that no span of the same label
/// overlaps, with the reported label and score.
fn add_other_occurrences(spans: &mut Vec<NerSpan>, text: &str) {
    // Character offset of every byte offset that starts a character (and of the end).
    let mut char_at = vec![0; text.len() + 1];
    for (chars, (byte, _)) in text.char_indices().enumerate() {
        char_at[byte] = chars;
    }
    let char_len = text.chars().count();
    char_at[text.len()] = char_len;
    // Characters covered by a span, per label.
    let mut covered: HashMap<String, Vec<bool>> = HashMap::new();
    for span in spans.iter() {
        let taken = covered.entry(span.label.clone()).or_insert_with(|| vec![false; char_len]);
        taken[span.start..span.end].fill(true);
    }
    let is_word = |c: Option<char>| c.is_some_and(char::is_alphanumeric);
    for index in 0..spans.len() {
        let reported = spans[index].clone();
        let Some(taken) = covered.get_mut(&reported.label) else {
            continue;
        };
        for (byte, found) in text.match_indices(reported.text.as_str()) {
            let byte_end = byte + found.len();
            if is_word(text[..byte].chars().next_back()) || is_word(text[byte_end..].chars().next())
            {
                continue;
            }
            let (start, end) = (char_at[byte], char_at[byte_end]);
            if taken[start..end].iter().any(|&t| t) {
                continue;
            }
            taken[start..end].fill(true);
            spans.push(NerSpan { start, end, ..reported.clone() });
        }
    }
}

/// Relations from formatted output
/// `{"relation_extraction": {label: [{"head": {...}, "tail": {...}}]}}`.
fn relations_from_output(output: &Value, text: &str) -> Result<Vec<RelationHit>, ModelError> {
    let Some(relations) = output.get("relation_extraction").and_then(Value::as_object) else {
        return Ok(Vec::new());
    };
    let mut hits = Vec::new();
    for (label, items) in relations {
        for item in items.as_array().into_iter().flatten() {
            let (Some(head), Some(tail)) = (item.get("head"), item.get("tail")) else {
                continue;
            };
            let (Some(head), Some(tail)) =
                (span_from_item(head, label, text)?, span_from_item(tail, label, text)?)
            else {
                continue;
            };
            hits.push(RelationHit {
                score: head.score.min(tail.score),
                label: label.clone(),
                head,
                tail,
            });
        }
    }
    Ok(hits)
}

/// One `{text, confidence, start, end}` item (byte offsets) as a character-offset span.
fn span_from_item(item: &Value, label: &str, text: &str) -> Result<Option<NerSpan>, ModelError> {
    let (Some(start), Some(end)) =
        (item.get("start").and_then(Value::as_u64), item.get("end").and_then(Value::as_u64))
    else {
        return Ok(None);
    };
    let (start, end) = (start as usize, end as usize);
    if start >= end || end > text.len() {
        tracing::warn!(start, end, len = text.len(), "GLiNER2 span out of range, skipping");
        return Ok(None);
    }
    let (Some(char_start), Some(char_end)) = (char_offset(text, start), char_offset(text, end))
    else {
        return Err(ModelError::Protocol(format!(
            "GLiNER2 span {start}..{end} does not fall on character boundaries"
        )));
    };
    let score = item.get("confidence").and_then(Value::as_f64).unwrap_or(0.0) as f32;
    Ok(Some(NerSpan {
        start: char_start,
        end: char_end,
        label: label.to_owned(),
        score,
        text: text[start..end].to_owned(),
    }))
}

/// Character offset of byte offset `byte`, or `None` when it is not a character boundary.
fn char_offset(text: &str, byte: usize) -> Option<usize> {
    text.is_char_boundary(byte).then(|| text[..byte].chars().count())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;

    #[test]
    fn entity_offsets_become_char_offsets() {
        let text = "Zoë Smith sued Acme Corp.";
        // "Zoë Smith" is bytes 0..10 (ë is two bytes), "Acme Corp" bytes 16..25.
        let output = json!({"entities": {
            "person": [{"text": "Zoë Smith", "confidence": 0.9, "start": 0, "end": 10}],
            "company": [{"text": "Acme Corp", "confidence": 0.8, "start": 16, "end": 25}]
        }});
        let spans = entities_from_output(&output, text).expect("spans");
        assert_eq!(
            spans,
            vec![
                NerSpan {
                    start: 0,
                    end: 9,
                    label: "person".into(),
                    score: 0.9,
                    text: "Zoë Smith".into()
                },
                NerSpan {
                    start: 15,
                    end: 24,
                    label: "company".into(),
                    score: 0.8,
                    text: "Acme Corp".into()
                },
            ]
        );
    }

    #[test]
    fn reported_names_expand_to_every_whole_word_occurrence() {
        let text = "Lopez met Lopezville's Lopez and Lopez.";
        let output = json!({"entities": {
            "person": [{"text": "Lopez", "confidence": 0.7, "start": 23, "end": 28}]
        }});
        let spans = entities_from_output(&output, text).expect("spans");
        let starts: Vec<(usize, usize)> = spans.iter().map(|s| (s.start, s.end)).collect();
        assert_eq!(starts, vec![(0, 5), (23, 28), (33, 38)]);
        assert!(spans.iter().all(|s| s.label == "person" && (s.score - 0.7).abs() < 1e-6));
    }

    #[test]
    fn relations_use_min_argument_score() {
        let text = "Zoë works for Acme.";
        let output = json!({"relation_extraction": {"works_for": [{
            "head": {"text": "Zoë", "confidence": 0.9, "start": 0, "end": 4},
            "tail": {"text": "Acme", "confidence": 0.7, "start": 15, "end": 19}
        }]}});
        let hits = relations_from_output(&output, text).expect("relations");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].label, "works_for");
        assert_eq!((hits[0].head.start, hits[0].head.end), (0, 3));
        assert_eq!((hits[0].tail.start, hits[0].tail.end), (14, 18));
        assert!((hits[0].score - 0.7).abs() < 1e-6);
    }

    #[test]
    fn invalid_spans_are_skipped_or_rejected() {
        let output = json!({"entities": {"person": [
            {"text": "x", "confidence": 0.5, "start": 3, "end": 3},
            {"text": "x", "confidence": 0.5, "start": 0, "end": 99},
            "bare string"
        ]}});
        assert!(entities_from_output(&output, "abc").expect("skipped").is_empty());
        let split_char = json!({"entities": {"person": [{"text": "x", "start": 0, "end": 1}]}});
        assert!(entities_from_output(&split_char, "ëa").is_err());
    }

    #[test]
    #[ignore = "loads fastino/gliner2-base-v1 from the Hugging Face cache (downloads ~800 MB if absent)"]
    fn gliner2_base_extracts_people() {
        let ner = Gliner2Ner::new(&Gliner2Options::new("fastino/gliner2-base-v1")).expect("load");
        let text = "Zoë Smith sued Acme Corporation in Boston.";
        let spans = ner.extract_entities(text, &["person", "company"], 0.4).expect("extract");
        let chars: Vec<char> = text.chars().collect();
        for span in &spans {
            assert_eq!(chars[span.start..span.end].iter().collect::<String>(), span.text);
        }
        assert!(spans.iter().any(|s| s.label == "person" && s.text.contains("Smith")), "{spans:?}");
    }
}
