//! Post-processing settings (the `CoreferenceStage` config block).

use impeach_config::pipeline::CoreferenceStageConfig;

/// Eligible types used when the config lists none.
pub const DEFAULT_ELIGIBLE_TYPES: [&str; 4] = ["PER", "ORG", "LOC", "DATE"];

/// Settings of the cluster mapping, safety rules and pronoun synthesis. Defaults are the
/// Python stage defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct CorefSettings {
    /// Documents longer than this many characters are skipped; `None` never skips.
    pub max_doc_chars: Option<usize>,
    /// Mention types that may join a cluster (never empty).
    pub eligible_types: Vec<String>,
    /// Minimum overlap (of either span) for a coref span to map onto a mention.
    pub min_overlap_fraction: f64,
    /// Mentions mapped per coref span.
    pub max_mentions_per_span: usize,
    /// Member-set Jaccard similarity at which two clusters merge (or the smaller is dropped).
    pub cluster_jaccard_threshold: f64,
    /// Merge high-Jaccard clusters instead of dropping the smaller one.
    pub merge_overlapping_clusters: bool,
    /// Mutually incompatible type groups (never empty).
    pub incompatible_type_groups: Vec<Vec<String>>,
    /// Type groups one cluster may span.
    pub max_type_groups_per_cluster: usize,
    /// Drop mixed-type clusters (`true`) or split them into type-coherent sub-clusters.
    pub drop_mixed_type_clusters: bool,
    /// Turn unmapped pronoun spans into synthetic mentions.
    pub synthesize_pronoun_mentions: bool,
    /// Score of synthetic pronoun mentions.
    pub pronoun_confidence: f64,
}

impl Default for CorefSettings {
    fn default() -> Self {
        Self {
            max_doc_chars: None,
            eligible_types: DEFAULT_ELIGIBLE_TYPES.iter().map(|t| (*t).to_owned()).collect(),
            min_overlap_fraction: 0.5,
            max_mentions_per_span: 1,
            cluster_jaccard_threshold: 0.8,
            merge_overlapping_clusters: true,
            incompatible_type_groups: default_incompatible_type_groups(),
            max_type_groups_per_cluster: 1,
            drop_mixed_type_clusters: true,
            synthesize_pronoun_mentions: true,
            pronoun_confidence: 0.85,
        }
    }
}

/// `[["PER"], ["PRODUCT", "FACILITY"], ["DATE", "TIME"]]`, used when the config lists none.
pub fn default_incompatible_type_groups() -> Vec<Vec<String>> {
    [&["PER"][..], &["PRODUCT", "FACILITY"], &["DATE", "TIME"]]
        .iter()
        .map(|group| group.iter().map(|t| (*t).to_owned()).collect())
        .collect()
}

impl CorefSettings {
    /// Settings from a `CoreferenceStage` block; empty `eligible_types` or
    /// `incompatible_type_groups` fall back to the defaults like the Python stage.
    pub fn from_config(config: &CoreferenceStageConfig) -> Self {
        let defaults = Self::default();
        Self {
            max_doc_chars: Some(config.max_doc_chars),
            eligible_types: if config.eligible_types.is_empty() {
                defaults.eligible_types
            } else {
                config.eligible_types.clone()
            },
            min_overlap_fraction: config.min_overlap_fraction,
            max_mentions_per_span: config.max_mentions_per_span,
            cluster_jaccard_threshold: config.cluster_jaccard_threshold,
            merge_overlapping_clusters: config.merge_overlapping_clusters,
            incompatible_type_groups: if config.incompatible_type_groups.is_empty() {
                defaults.incompatible_type_groups
            } else {
                config.incompatible_type_groups.clone()
            },
            max_type_groups_per_cluster: config.max_type_groups_per_cluster,
            drop_mixed_type_clusters: config.drop_mixed_type_clusters,
            synthesize_pronoun_mentions: config.synthesize_pronoun_mentions,
            pronoun_confidence: config.pronoun_confidence,
        }
    }

    /// Whether mentions of `entity_type` may join a cluster.
    pub fn is_eligible(&self, entity_type: &str) -> bool {
        self.eligible_types.is_empty() || self.eligible_types.iter().any(|t| t == entity_type)
    }

    /// Index of the first incompatible type group containing `entity_type`.
    pub(crate) fn type_group(&self, entity_type: &str) -> Option<usize> {
        self.incompatible_type_groups.iter().position(|g| g.iter().any(|t| t == entity_type))
    }
}
