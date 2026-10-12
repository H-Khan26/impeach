//! Coreference post-processing: maps resolver clusters onto detected mentions, applies the
//! cluster safety rules, selects canonical mentions and synthesises pronoun mentions
//! (Python `stages/coreference.py` without the model call and the context plumbing, which live
//! in the `CoreferenceStage` of ce-pipeline).
//!
//! - [`label_clusters`]: validates [`ce_models::CorefCluster`] spans against the text, flags
//!   pronouns and assigns the ids `<doc_id>_cluster_<idx>`.
//! - [`ClusterBuilder`]: span-to-mention mapping (`min_overlap_fraction`,
//!   `max_mentions_per_span`, `eligible_types`), exclusive membership, deduplication
//!   (`cluster_jaccard_threshold`, `merge_overlapping_clusters`), type-safety guards
//!   (`incompatible_type_groups`, `max_type_groups_per_cluster`, `drop_mixed_type_clusters`)
//!   and canonical mention selection with [`RoleConfig`] (shared with the definition stage).
//! - [`CorefPostProcessor::process`]: all of the above plus pronoun mention synthesis
//!   (`synthesize_pronoun_mentions`, `pronoun_confidence`), producing a [`CorefOutput`].
//!
//! All offsets are character offsets into the document text.

mod clusters;
mod pronouns;
mod settings;
mod text;

use ce_core::{ChunkId, ChunkWriteResult, CorefClusterContext, DocId, Mention, MentionId};
use indexmap::IndexMap;

pub use ce_ner::definitions::RoleConfig;
pub use clusters::{label_clusters, ClusterBuilder, ClusterContext, ClusterSpan, ResolvedCluster};
pub use pronouns::PRONOUN_SOURCE;
pub use settings::{default_incompatible_type_groups, CorefSettings, DEFAULT_ELIGIBLE_TYPES};
pub use text::reconstruct_doc_text;

/// Everything coreference adds to a document's context.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CorefOutput {
    /// Clusters mapped onto mentions (context `coref_clusters`).
    pub clusters: Vec<CorefClusterContext>,
    /// Cluster id -> canonical mention id (context `canonical_mention_for_cluster`).
    pub canonical_mention_for_cluster: IndexMap<String, MentionId>,
    /// Mention id -> canonical name, for cluster members and synthetic pronoun mentions
    /// (context `canonical_text_for_mention`).
    pub canonical_text_for_mention: IndexMap<MentionId, String>,
    /// Synthetic pronoun mentions, to append to `doc_mentions`.
    pub synthetic_mentions: Vec<Mention>,
    /// Synthetic pronoun mentions by chunk graph id, to append to `mentions_by_chunk`.
    pub synthetic_by_chunk: IndexMap<ChunkId, Vec<Mention>>,
    /// Synthetic mention id -> canonical mention id (context `synthetic_pronoun_mentions`);
    /// `None` when no pronoun mention was synthesised (the context key stays unset).
    pub synthetic_pronoun_mentions: Option<IndexMap<MentionId, MentionId>>,
}

/// Coreference post-processing with fixed settings and role terms.
#[derive(Debug, Clone)]
pub struct CorefPostProcessor {
    settings: CorefSettings,
    roles: RoleConfig,
}

impl CorefPostProcessor {
    /// A post-processor applying `settings`, ranking canonical mentions with `roles`.
    pub fn new(settings: CorefSettings, roles: RoleConfig) -> Self {
        Self { settings, roles }
    }

    /// The settings.
    pub fn settings(&self) -> &CorefSettings {
        &self.settings
    }

    /// The cluster builder over these settings and role terms.
    pub fn builder(&self) -> ClusterBuilder<'_> {
        ClusterBuilder::new(&self.settings, &self.roles)
    }

    /// Whether a document of `text` is too long for coreference (`max_doc_chars`).
    pub fn exceeds_max_doc_chars(&self, text: &str) -> bool {
        self.settings.max_doc_chars.is_some_and(|max| text.chars().count() > max)
    }

    /// Post-processes the resolver output `raw` for document `doc_id` with text `text`,
    /// detected `mentions` and the document's `chunks`. `None` when no cluster survives (the
    /// Python stage then writes nothing).
    pub fn process(
        &self,
        doc_id: &DocId,
        text: &str,
        raw: &[ce_models::CorefCluster],
        mentions: &[Mention],
        chunks: &[ChunkWriteResult],
    ) -> Option<CorefOutput> {
        let text = text::CharText::new(text);
        let resolved = clusters::label_clusters_in(doc_id.as_str(), &text, raw);
        if resolved.is_empty() {
            return None;
        }
        let context = self.builder().build(&resolved, mentions);
        if context.clusters.is_empty() {
            return None;
        }
        let mut output = CorefOutput::default();
        if self.settings.synthesize_pronoun_mentions {
            let by_id = clusters::mentions_by_id(mentions);
            let synthesis = pronouns::synthesize(
                &context,
                &by_id,
                chunks,
                &text,
                doc_id,
                self.settings.pronoun_confidence,
            );
            if !synthesis.mentions.is_empty() {
                output.synthetic_pronoun_mentions = Some(synthesis.synthetic_to_canonical);
                output.synthetic_mentions = synthesis.mentions;
                output.synthetic_by_chunk = synthesis.by_chunk;
            }
        }
        output.clusters = context.clusters;
        output.canonical_mention_for_cluster = context.canonical_mention_for_cluster;
        output.canonical_text_for_mention = context.canonical_text_for_mention;
        for mention in &output.synthetic_mentions {
            output.canonical_text_for_mention.insert(mention.id.clone(), mention.name.clone());
        }
        Some(output)
    }
}
