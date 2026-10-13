//! Sentence embeddings through fastembed (ONNX Runtime). Replaces the Python
//! sentence-transformers `BGEService` and the evaluator's MiniLM model.

use std::path::PathBuf;
use std::sync::{Mutex, PoisonError};

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use impeach_config::pipeline::BgeModelConfig;

use crate::error::{error_chain, ModelError};
use crate::traits::{normalize_l2, Embedder};

/// Model ids accepted by [`FastEmbedder`] (case-insensitive) and their fastembed models.
const SUPPORTED_MODELS: &[(&str, EmbeddingModel)] = &[
    ("BAAI/bge-small-en-v1.5", EmbeddingModel::BGESmallENV15),
    ("bge-small-en-v1.5", EmbeddingModel::BGESmallENV15),
    ("sentence-transformers/all-MiniLM-L6-v2", EmbeddingModel::AllMiniLML6V2),
    ("all-MiniLM-L6-v2", EmbeddingModel::AllMiniLML6V2),
];

/// Settings of a [`FastEmbedder`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FastEmbedOptions {
    /// Model id, e.g. `BAAI/bge-small-en-v1.5` or `all-MiniLM-L6-v2`.
    pub model: String,
    /// Texts per inference batch.
    pub batch_size: usize,
    /// L2-normalise every output vector.
    pub normalize: bool,
    /// Model cache directory; `None` uses `FASTEMBED_CACHE_DIR` or `.fastembed_cache/`.
    pub cache_dir: Option<PathBuf>,
    /// Show a progress bar while downloading model files.
    pub show_download_progress: bool,
}

impl Default for FastEmbedOptions {
    fn default() -> Self {
        Self::new("BAAI/bge-small-en-v1.5")
    }
}

impl FastEmbedOptions {
    /// Options for `model` with batch size 64, normalisation on, the default cache and no
    /// download progress bar.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            batch_size: 64,
            normalize: true,
            cache_dir: None,
            show_download_progress: false,
        }
    }

    /// Options from the `models.bge` config block.
    pub fn from_config(config: &BgeModelConfig) -> Self {
        Self {
            batch_size: config.batch_size,
            normalize: config.normalize_embeddings,
            ..Self::new(config.model.clone())
        }
    }
}

/// A fastembed text embedding model.
pub struct FastEmbedder {
    model: Mutex<TextEmbedding>,
    name: String,
    dimension: usize,
    batch_size: usize,
    normalize: bool,
}

impl std::fmt::Debug for FastEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastEmbedder")
            .field("name", &self.name)
            .field("dimension", &self.dimension)
            .field("batch_size", &self.batch_size)
            .field("normalize", &self.normalize)
            .finish_non_exhaustive()
    }
}

impl FastEmbedder {
    /// Loads the model, downloading it into the cache directory on first use.
    pub fn new(options: &FastEmbedOptions) -> Result<Self, ModelError> {
        let model = Self::resolve_model(&options.model).ok_or_else(|| ModelError::Load {
            model: options.model.clone(),
            message: format!(
                "unsupported fastembed model; supported: {}",
                SUPPORTED_MODELS.iter().map(|(id, _)| *id).collect::<Vec<_>>().join(", ")
            ),
        })?;
        if options.batch_size == 0 {
            return Err(ModelError::Load {
                model: options.model.clone(),
                message: "batch_size must be greater than 0".to_owned(),
            });
        }
        let load_error = |e: fastembed::Error| match e {
            fastembed::Error::ModelRetrieval { .. } => {
                ModelError::Download { resource: options.model.clone(), message: error_chain(&e) }
            }
            other => {
                ModelError::Load { model: options.model.clone(), message: error_chain(&other) }
            }
        };
        let dimension = TextEmbedding::get_model_info(&model).map_err(load_error)?.dim;
        let mut init =
            TextInitOptions::new(model).with_show_download_progress(options.show_download_progress);
        if let Some(dir) = &options.cache_dir {
            init = init.with_cache_dir(dir.clone());
        }
        tracing::info!(model = %options.model, "loading fastembed model");
        let embedding = TextEmbedding::try_new(init).map_err(load_error)?;
        Ok(Self {
            model: Mutex::new(embedding),
            name: options.model.clone(),
            dimension,
            batch_size: options.batch_size,
            normalize: options.normalize,
        })
    }

    /// The fastembed model for a supported model id (case-insensitive).
    pub fn resolve_model(model: &str) -> Option<EmbeddingModel> {
        SUPPORTED_MODELS
            .iter()
            .find(|(id, _)| id.eq_ignore_ascii_case(model))
            .map(|(_, m)| m.clone())
    }
}

impl Embedder for FastEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, ModelError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut model = self.model.lock().unwrap_or_else(PoisonError::into_inner);
        let mut vectors = model
            .embed(texts, Some(self.batch_size))
            .map_err(|e| ModelError::Inference(error_chain(&e)))?;
        drop(model);
        if self.normalize {
            vectors.iter_mut().for_each(|v| normalize_l2(v));
        }
        Ok(vectors)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_supported_model_ids() {
        assert_eq!(
            FastEmbedder::resolve_model("BAAI/bge-small-en-v1.5"),
            Some(EmbeddingModel::BGESmallENV15)
        );
        assert_eq!(
            FastEmbedder::resolve_model("all-MiniLM-L6-v2"),
            Some(EmbeddingModel::AllMiniLML6V2)
        );
        assert_eq!(
            FastEmbedder::resolve_model("sentence-transformers/all-minilm-l6-v2"),
            Some(EmbeddingModel::AllMiniLML6V2)
        );
        assert_eq!(FastEmbedder::resolve_model("BAAI/bge-large-en-v1.5"), None);
    }

    #[test]
    fn unsupported_model_fails_before_download() {
        let err =
            FastEmbedder::new(&FastEmbedOptions::new("no/such-model")).expect_err("unsupported");
        assert!(matches!(err, ModelError::Load { .. }), "{err:?}");
    }

    #[test]
    fn normalize_scales_to_unit_length() {
        let mut v = vec![3.0, 4.0];
        normalize_l2(&mut v);
        assert_eq!(v, vec![0.6, 0.8]);
        let mut zero = vec![0.0, 0.0];
        normalize_l2(&mut zero);
        assert_eq!(zero, vec![0.0, 0.0]);
    }

    #[test]
    #[ignore = "downloads BAAI/bge-small-en-v1.5 (~130 MB) into <workspace>/.fastembed_cache/"]
    fn bge_small_embeds_normalised_384_dim_vectors() {
        let cache = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.fastembed_cache");
        let options = FastEmbedOptions { cache_dir: Some(cache), ..FastEmbedOptions::default() };
        let embedder = FastEmbedder::new(&options).expect("load bge-small");
        assert_eq!(embedder.dimension(), 384);
        let texts = ["The car was red.", "The vehicle was red.", "Interest rates rose in May."];
        let vectors = embedder.embed(&texts).expect("embed");
        assert_eq!(vectors.len(), 3);
        for v in &vectors {
            assert_eq!(v.len(), 384);
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "norm {norm}");
        }
        let cos = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        assert!(cos(&vectors[0], &vectors[1]) > cos(&vectors[0], &vectors[2]));
        // Batching does not change the result.
        let single = embedder.embed(&texts[..1]).expect("embed one");
        let diff: f32 = single[0].iter().zip(&vectors[0]).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff < 1e-3, "batch drift {diff}");
    }
}
