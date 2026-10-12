//! Cached cosine-similarity matrices over an [`Embedder`].

use std::collections::{BTreeSet, HashMap};

use ce_models::Embedder;

use crate::error::{EvalError, Result};

/// Embeds texts once and serves cosine-similarity matrices.
///
/// Mirrors the Python evaluator's `Embedder.matrix`: each call embeds the distinct non-empty
/// texts not seen before in one batch (sorted, as Python does), vectors are unit-normalised,
/// an empty text scores 0 against everything, and similarities are clipped to [-1, 1].
/// Similarities are `f32` like the Python float32 matrices; each dot product is accumulated in
/// `f64` and rounded once.
pub struct SimilarityCache<'a> {
    embedder: &'a dyn Embedder,
    cache: HashMap<String, Vec<f32>>,
}

impl std::fmt::Debug for SimilarityCache<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimilarityCache")
            .field("model", &self.embedder.model_name())
            .field("cached", &self.cache.len())
            .finish()
    }
}

fn normalize(vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
    if norm > 0.0 {
        vector.into_iter().map(|x| (f64::from(x) / norm) as f32).collect()
    } else {
        vector
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum::<f64>() as f32
}

impl<'a> SimilarityCache<'a> {
    /// An empty cache over `embedder`.
    #[must_use]
    pub fn new(embedder: &'a dyn Embedder) -> Self {
        Self { embedder, cache: HashMap::new() }
    }

    /// Embeds every distinct non-empty text not cached yet, in one sorted batch.
    ///
    /// # Errors
    /// The embedder's error, or [`EvalError::EmbeddingCount`] on a short batch.
    pub fn encode<'t>(&mut self, texts: impl IntoIterator<Item = &'t str>) -> Result<()> {
        let missing: BTreeSet<&str> =
            texts.into_iter().filter(|t| !t.is_empty() && !self.cache.contains_key(*t)).collect();
        if missing.is_empty() {
            return Ok(());
        }
        let batch: Vec<&str> = missing.into_iter().collect();
        let vectors = self.embedder.embed(&batch)?;
        if vectors.len() != batch.len() {
            return Err(EvalError::EmbeddingCount { expected: batch.len(), got: vectors.len() });
        }
        for (text, vector) in batch.into_iter().zip(vectors) {
            self.cache.insert(text.to_owned(), normalize(vector));
        }
        Ok(())
    }

    /// Cosine-similarity matrix (`rows` x `cols`); empty strings score 0.
    ///
    /// # Errors
    /// See [`SimilarityCache::encode`].
    pub fn matrix(&mut self, rows: &[&str], cols: &[&str]) -> Result<Vec<Vec<f32>>> {
        if rows.is_empty() || cols.is_empty() {
            return Ok(vec![vec![0.0; cols.len()]; rows.len()]);
        }
        self.encode(rows.iter().chain(cols).copied())?;
        let lookup = |text: &str| self.cache.get(text);
        let col_vectors: Vec<Option<&Vec<f32>>> = cols.iter().map(|c| lookup(c)).collect();
        Ok(rows
            .iter()
            .map(|r| {
                let row_vector = lookup(r);
                col_vectors
                    .iter()
                    .map(|col_vector| match (row_vector, col_vector) {
                        (Some(a), Some(b)) => dot(a, b).clamp(-1.0, 1.0),
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use ce_models::MockEmbedder;

    use super::*;

    #[test]
    fn scripted_vectors_give_cosines() {
        let embedder = MockEmbedder::new(2)
            .with_vector("a", vec![1.0, 0.0])
            .with_vector("b", vec![0.0, 2.0])
            .with_vector("c", vec![1.0, 1.0]);
        let mut cache = SimilarityCache::new(&embedder);
        let m = cache.matrix(&["a", "b", ""], &["c", "a", ""]).unwrap();
        assert!((m[0][0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert!((m[0][1] - 1.0).abs() < 1e-6);
        assert!((m[1][0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert!(m[1][1].abs() < 1e-6);
        assert_eq!(m[2], vec![0.0, 0.0, 0.0]);
        assert_eq!(m[0][2], 0.0);
        assert_eq!(embedder.calls(), 1);
        cache.matrix(&["a"], &["b"]).unwrap();
        assert_eq!(embedder.calls(), 1, "cached texts are not re-embedded");
    }

    #[test]
    fn empty_sides_give_empty_shapes() {
        let embedder = MockEmbedder::new(4);
        let mut cache = SimilarityCache::new(&embedder);
        assert_eq!(cache.matrix(&["x", "y"], &[]).unwrap(), vec![Vec::<f32>::new(); 2]);
        assert!(cache.matrix(&[], &["x"]).unwrap().is_empty());
        assert_eq!(embedder.calls(), 0);
    }
}
