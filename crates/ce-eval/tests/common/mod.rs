//! Shared test helpers.
#![allow(dead_code)]

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::Mutex;

use ce_models::{Embedder, ModelError};
use ce_text::normalize::py_split;

/// Workspace root.
pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// A fixture file's text with line endings normalised to `\n`.
pub fn read_fixture(relative: &str) -> String {
    let path = workspace_root().join(relative);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .replace("\r\n", "\n")
}

/// Lowercase tokens with surrounding punctuation stripped (the Python test `TokenEmbedder`).
fn tokens(text: &str) -> BTreeSet<String> {
    py_split(text)
        .map(|t| t.trim_matches(|c| ".,;:\"'()$".contains(c)).to_lowercase())
        .filter(|t| !t.is_empty())
        .collect()
}

/// Binary bag-of-tokens embedder: cosine similarity is `|A & B| / sqrt(|A| |B|)`.
/// Token indices are assigned on first sight.
pub struct TokenEmbedder {
    vocab: Mutex<HashMap<String, usize>>,
}

impl TokenEmbedder {
    const DIMENSION: usize = 4096;

    pub fn new() -> Self {
        Self { vocab: Mutex::new(HashMap::new()) }
    }
}

impl Embedder for TokenEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, ModelError> {
        let mut vocab = self.vocab.lock().expect("vocabulary lock");
        Ok(texts
            .iter()
            .map(|text| {
                let mut vector = vec![0.0; Self::DIMENSION];
                for token in tokens(text) {
                    let next = vocab.len();
                    let index = *vocab.entry(token).or_insert(next);
                    assert!(index < Self::DIMENSION, "token vocabulary exhausted");
                    vector[index] = 1.0;
                }
                vector
            })
            .collect())
    }

    fn dimension(&self) -> usize {
        Self::DIMENSION
    }

    fn model_name(&self) -> &str {
        "token-cosine"
    }
}
