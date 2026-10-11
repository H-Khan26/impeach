//! Mention merge priorities (`detector_config.yaml`).

use indexmap::{IndexMap, IndexSet};
use serde::Deserialize;

/// Source priority used when merging overlapping mentions from several detectors.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DetectorConfig {
    /// Label -> detector sources, most trusted first.
    pub label_priority: IndexMap<String, Vec<String>>,
    /// Source order for labels without an entry in `label_priority`.
    pub default_source_priority: Vec<String>,
    /// Labels boosted by +1000.
    pub high_precision_labels: IndexSet<String>,
}

impl DetectorConfig {
    /// Priority score for a `(label, source)` pair; higher is more trusted.
    ///
    /// High-precision labels add 1000. A label-specific source list adds
    /// `(len - idx) * 10` for the first entry contained in `source`; otherwise the default
    /// list adds `len - idx`.
    pub fn get_source_priority(&self, label: &str, source: &str) -> i64 {
        let mut score = 0;
        if self.high_precision_labels.contains(label) {
            score += 1000;
        }
        if let Some(sources) = self.label_priority.get(label) {
            if let Some(idx) = sources.iter().position(|s| source.contains(s.as_str())) {
                return score + rank(sources.len(), idx) * 10;
            }
        }
        let defaults = &self.default_source_priority;
        if let Some(idx) = defaults.iter().position(|s| source.contains(s.as_str())) {
            return score + rank(defaults.len(), idx);
        }
        score
    }

    /// Whether a label is high-precision.
    pub fn is_high_precision_label(&self, label: &str) -> bool {
        self.high_precision_labels.contains(label)
    }
}

fn rank(len: usize, idx: usize) -> i64 {
    i64::try_from(len - idx).unwrap_or(i64::MAX)
}
