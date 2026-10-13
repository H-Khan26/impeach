//! NLI cross-encoder on ONNX Runtime. Replaces the Python transformers `DeBERTaNLIService`.
//!
//! Model files come from the Hugging Face hub repo (default
//! `cross-encoder/nli-deberta-v3-large`, which publishes an ONNX export at `onnx/model.onnx`)
//! and are cached in the standard Hugging Face cache. At load time the model's `config.json`
//! `id2label` must equal [`LABEL_MAP`]; any other label order is an error.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use hf_hub::api::sync::ApiBuilder;
use impeach_config::pipeline::NliModelConfig;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::{
    EncodeInput, Encoding, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams,
    TruncationStrategy,
};

use crate::error::{missing_model, ModelError};
use crate::traits::{NliJudge, NliScores};

/// Logit index of each label (Python `DeBERTaNLIService.LABEL_MAP`).
pub const LABEL_MAP: [&str; 3] = ["contradiction", "entailment", "neutral"];

/// Settings of an [`OrtNli`] judge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrtNliOptions {
    /// Hugging Face repo id.
    pub model: String,
    /// Path of the ONNX graph inside the repo.
    pub onnx_file: String,
    /// Hugging Face cache directory; `None` uses `HF_HOME` / the default cache.
    pub cache_dir: Option<PathBuf>,
    /// Token limit per pair (longest-first truncation), as the Python `max_length=512`.
    pub max_length: usize,
    /// Pairs per inference batch.
    pub batch_size: usize,
    /// ONNX Runtime intra-op threads; `None` uses every core.
    pub intra_threads: Option<usize>,
}

impl Default for OrtNliOptions {
    fn default() -> Self {
        Self::new("cross-encoder/nli-deberta-v3-large")
    }
}

impl OrtNliOptions {
    /// Options for repo `model` with `onnx/model.onnx`, max length 512 and batch size 8.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            onnx_file: "onnx/model.onnx".to_owned(),
            cache_dir: None,
            max_length: 512,
            batch_size: 8,
            intra_threads: None,
        }
    }

    /// Options from the `models.deberta_nli` config block.
    pub fn from_config(config: &NliModelConfig) -> Self {
        Self::new(config.model.clone())
    }
}

/// Paths of the downloaded model files.
#[derive(Debug, Clone)]
struct ModelFiles {
    config: PathBuf,
    tokenizer: PathBuf,
    onnx: PathBuf,
}

/// An NLI cross-encoder running on ONNX Runtime.
pub struct OrtNli {
    name: String,
    tokenizer: Tokenizer,
    session: Mutex<Session>,
    input_names: Vec<String>,
    batch_size: usize,
}

impl std::fmt::Debug for OrtNli {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrtNli")
            .field("name", &self.name)
            .field("input_names", &self.input_names)
            .field("batch_size", &self.batch_size)
            .finish_non_exhaustive()
    }
}

impl OrtNli {
    /// Downloads (on first use) and loads the model.
    pub fn new(options: &OrtNliOptions) -> Result<Self, ModelError> {
        let files = download(options)?;
        Self::from_files(&options.model, &files, options)
    }

    /// Loads a judge from files already on disk: `config.json` and `tokenizer.json` in `dir`
    /// and the ONNX graph at `dir/<options.onnx_file>`.
    pub fn from_dir(dir: &Path, options: &OrtNliOptions) -> Result<Self, ModelError> {
        let files = ModelFiles {
            config: dir.join("config.json"),
            tokenizer: dir.join("tokenizer.json"),
            onnx: dir.join(&options.onnx_file),
        };
        Self::from_files(&options.model, &files, options)
    }

    fn from_files(
        name: &str,
        files: &ModelFiles,
        options: &OrtNliOptions,
    ) -> Result<Self, ModelError> {
        let load_error = |message: String| ModelError::Load { model: name.to_owned(), message };
        if options.batch_size == 0 {
            return Err(load_error("batch_size must be greater than 0".to_owned()));
        }
        let config = std::fs::read_to_string(&files.config)
            .map_err(|e| load_error(format!("cannot read {}: {e}", files.config.display())))?;
        verify_labels(&config).map_err(load_error)?;

        let mut tokenizer = Tokenizer::from_file(&files.tokenizer)
            .map_err(|e| load_error(format!("cannot load {}: {e}", files.tokenizer.display())))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: options.max_length,
                strategy: TruncationStrategy::LongestFirst,
                ..TruncationParams::default()
            }))
            .map_err(|e| load_error(format!("invalid truncation settings: {e}")))?;
        let pad_token = tokenizer
            .get_padding()
            .map(|p| p.pad_token.clone())
            .unwrap_or_else(|| "[PAD]".to_owned());
        let pad_id = tokenizer.token_to_id(&pad_token).unwrap_or(0);
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_id,
            pad_token,
            ..PaddingParams::default()
        }));

        let threads = match options.intra_threads {
            Some(n) => n,
            None => std::thread::available_parallelism().map_or(1, std::num::NonZero::get),
        };
        let ort_error = |e: ort::Error<_>| load_error(format!("ONNX Runtime: {e}"));
        let session = Session::builder()
            .map_err(|e| load_error(format!("ONNX Runtime: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_error)?
            .with_intra_threads(threads)
            .map_err(ort_error)?
            .commit_from_file(&files.onnx)
            .map_err(|e| load_error(format!("cannot load {}: {e}", files.onnx.display())))?;
        let input_names: Vec<String> =
            session.inputs().iter().map(|i| i.name().to_owned()).collect();
        for name in &input_names {
            if !matches!(name.as_str(), "input_ids" | "attention_mask" | "token_type_ids") {
                return Err(load_error(format!("unsupported ONNX input `{name}`")));
            }
        }
        Ok(Self {
            name: name.to_owned(),
            tokenizer,
            session: Mutex::new(session),
            input_names,
            batch_size: options.batch_size,
        })
    }

    /// Model id.
    pub fn model_name(&self) -> &str {
        &self.name
    }

    fn score_batch(&self, pairs: &[(&str, &str)]) -> Result<Vec<NliScores>, ModelError> {
        let inputs: Vec<EncodeInput<'_>> = pairs.iter().map(|&(p, h)| (p, h).into()).collect();
        let encodings = self
            .tokenizer
            .encode_batch(inputs, true)
            .map_err(|e| ModelError::Inference(format!("tokenization failed: {e}")))?;
        let batch = encodings.len();
        let seq = encodings.first().map_or(0, Encoding::len);
        let column = |get: fn(&Encoding) -> &[u32]| -> Vec<i64> {
            encodings.iter().flat_map(|e| get(e).iter().map(|&x| i64::from(x))).collect()
        };
        let mut named = Vec::with_capacity(self.input_names.len());
        for name in &self.input_names {
            let data = match name.as_str() {
                "input_ids" => column(Encoding::get_ids),
                "attention_mask" => column(Encoding::get_attention_mask),
                _ => column(Encoding::get_type_ids),
            };
            let tensor = Tensor::from_array((vec![batch as i64, seq as i64], data))
                .map_err(|e| ModelError::Inference(format!("cannot build `{name}` tensor: {e}")))?;
            named.push((name.clone(), tensor));
        }

        let mut session = self.session.lock().unwrap_or_else(PoisonError::into_inner);
        let outputs =
            session.run(named).map_err(|e| ModelError::Inference(format!("ONNX Runtime: {e}")))?;
        let (shape, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| ModelError::Inference(format!("unexpected NLI output: {e}")))?;
        if shape.len() != 2 || shape[0] != batch as i64 || shape[1] != LABEL_MAP.len() as i64 {
            return Err(ModelError::Inference(format!(
                "NLI logits have shape {shape:?}, expected [{batch}, {}]",
                LABEL_MAP.len()
            )));
        }
        Ok(logits.as_chunks::<3>().0.iter().map(scores_from_logits).collect())
    }
}

impl NliJudge for OrtNli {
    fn score_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<NliScores>, ModelError> {
        let mut scores = Vec::with_capacity(pairs.len());
        for batch in pairs.chunks(self.batch_size) {
            scores.extend(self.score_batch(batch)?);
        }
        Ok(scores)
    }
}

/// Downloads `config.json`, `tokenizer.json` and the ONNX graph of `options.model`.
fn download(options: &OrtNliOptions) -> Result<ModelFiles, ModelError> {
    let mut builder = ApiBuilder::from_env().with_progress(false);
    if let Some(dir) = &options.cache_dir {
        builder = builder.with_cache_dir(dir.clone());
    }
    let cache_dir =
        options.cache_dir.clone().unwrap_or_else(|| crate::error::hf_hub_cache_dir(true));
    let missing = |cause: String| {
        missing_model(
            &options.model,
            &cache_dir,
            "set HF_HOME to a Hugging Face cache that holds it (models.deberta_nli.model names \
             the repo)",
            &cause,
        )
    };
    let api = builder
        .build()
        .map_err(|e| missing(format!("cannot initialise the Hugging Face hub client: {e}")))?;
    let repo = api.model(options.model.clone());
    let get = |file: &str| {
        tracing::debug!(model = %options.model, file, "fetching NLI model file");
        repo.get(file).map_err(|e| missing(format!("{file}: {e}")))
    };
    Ok(ModelFiles {
        config: get("config.json")?,
        tokenizer: get("tokenizer.json")?,
        onnx: get(&options.onnx_file)?,
    })
}

/// Checks that `config.json` maps logit indexes to labels exactly as [`LABEL_MAP`].
fn verify_labels(config_json: &str) -> Result<(), String> {
    let config: serde_json::Value =
        serde_json::from_str(config_json).map_err(|e| format!("config.json is not JSON: {e}"))?;
    let id2label = config
        .get("id2label")
        .and_then(serde_json::Value::as_object)
        .ok_or("config.json has no id2label map")?;
    let mut labels: Vec<(usize, String)> = Vec::with_capacity(id2label.len());
    for (id, label) in id2label {
        let id: usize = id.parse().map_err(|_| format!("id2label key `{id}` is not an index"))?;
        let label = label.as_str().ok_or_else(|| format!("id2label[{id}] is not a string"))?;
        labels.push((id, label.to_lowercase()));
    }
    labels.sort();
    let expected: Vec<(usize, String)> =
        LABEL_MAP.iter().enumerate().map(|(i, l)| (i, (*l).to_owned())).collect();
    if labels == expected {
        Ok(())
    } else {
        Err(format!(
            "model id2label {labels:?} does not match the expected label order {expected:?}"
        ))
    }
}

/// Softmax over `[contradiction, entailment, neutral]` logits.
fn scores_from_logits(logits: &[f32; 3]) -> NliScores {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp = logits.map(|x| (x - max).exp());
    let sum: f32 = exp.iter().sum();
    NliScores { contradiction: exp[0] / sum, entailment: exp[1] / sum, neutral: exp[2] / sum }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_order_is_verified() {
        let good = r#"{"id2label": {"0": "contradiction", "1": "entailment", "2": "neutral"}}"#;
        assert_eq!(verify_labels(good), Ok(()));
        let upper = r#"{"id2label": {"2": "NEUTRAL", "0": "CONTRADICTION", "1": "ENTAILMENT"}}"#;
        assert_eq!(verify_labels(upper), Ok(()));
        let swapped = r#"{"id2label": {"0": "entailment", "1": "neutral", "2": "contradiction"}}"#;
        assert!(verify_labels(swapped).is_err());
        let binary = r#"{"id2label": {"0": "entailment", "1": "contradiction"}}"#;
        assert!(verify_labels(binary).is_err());
        assert!(verify_labels(r#"{"architectures": []}"#).is_err());
    }

    #[test]
    fn softmax_maps_label_indexes() {
        let s = scores_from_logits(&[2.0, 0.0, 0.0]);
        let total = s.contradiction + s.entailment + s.neutral;
        assert!((total - 1.0).abs() < 1e-6);
        assert!(s.contradiction > s.entailment);
        assert!((s.entailment - s.neutral).abs() < 1e-7);
        // Large logits do not overflow.
        let s = scores_from_logits(&[1000.0, 0.0, 999.0]);
        assert!((s.contradiction - 0.7311).abs() < 1e-3);
    }

    #[test]
    #[ignore = "downloads cross-encoder/nli-deberta-v3-large ONNX (~1.7 GB) into the Hugging Face cache"]
    fn deberta_large_scores_pairs() {
        let nli = OrtNli::new(&OrtNliOptions::default()).expect("load NLI model");
        let scores = nli
            .score_pairs(&[
                ("The car was red.", "The car was blue."),
                ("The car was red.", "The car had a color."),
                ("The car was red.", "The driver was tired."),
            ])
            .expect("score");
        assert_eq!(scores.len(), 3);
        assert!(scores[0].contradiction > 0.5, "{:?}", scores[0]);
        assert!(scores[1].entailment > 0.5, "{:?}", scores[1]);
        assert!(scores[2].neutral > 0.5, "{:?}", scores[2]);
    }
}
