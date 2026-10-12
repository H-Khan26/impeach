//! Resolver clusters -> mention clusters: span validation, span-to-mention mapping, exclusive
//! membership, deduplication, type-safety guards and canonical mention selection.

use std::cmp::Ordering;
use std::collections::HashMap;

use ce_core::{CorefClusterContext, Mention, MentionId, Span};
use ce_models::{is_pronoun, CorefCluster};
use indexmap::{IndexMap, IndexSet};

use crate::settings::CorefSettings;
use crate::text::CharText;
use ce_ner::definitions::RoleConfig;

/// One span of a resolver cluster (Python `CorefMention`), doc-global character offsets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterSpan {
    /// Start offset (inclusive).
    pub start: usize,
    /// End offset (exclusive).
    pub end: usize,
    /// Span text.
    pub text: String,
    /// Whether the text is one of [`ce_models::PRONOUNS`].
    pub is_pronoun: bool,
}

/// A resolver cluster with its id (Python `CorefCluster`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedCluster {
    /// `<doc_id>_cluster_<index in the resolver output>`.
    pub id: String,
    /// Valid, distinct spans in resolver order.
    pub spans: Vec<ClusterSpan>,
}

/// Validates resolver output like the Python `CorefService`: spans with `end <= start` or past
/// the text and repeated spans are dropped, the span text and pronoun flag are filled in, and
/// clusters left empty are dropped. Ids keep the resolver index: `<doc_id>_cluster_<idx>`.
pub fn label_clusters(doc_id: &str, text: &str, raw: &[CorefCluster]) -> Vec<ResolvedCluster> {
    label_clusters_in(doc_id, &CharText::new(text), raw)
}

pub(crate) fn label_clusters_in(
    doc_id: &str,
    text: &CharText<'_>,
    raw: &[CorefCluster],
) -> Vec<ResolvedCluster> {
    raw.iter()
        .enumerate()
        .filter_map(|(idx, cluster)| {
            let mut spans: Vec<ClusterSpan> = Vec::with_capacity(cluster.spans.len());
            for &(start, end) in &cluster.spans {
                if end <= start || spans.iter().any(|s| s.start == start && s.end == end) {
                    continue;
                }
                let Some(span_text) = text.slice(start, end) else { continue };
                spans.push(ClusterSpan {
                    start,
                    end,
                    is_pronoun: is_pronoun(span_text),
                    text: span_text.to_owned(),
                });
            }
            (!spans.is_empty())
                .then(|| ResolvedCluster { id: format!("{doc_id}_cluster_{idx}"), spans })
        })
        .collect()
}

/// Clusters mapped onto mentions, with their canonical mentions.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ClusterContext {
    /// Surviving clusters (at least two member mentions each).
    pub clusters: Vec<CorefClusterContext>,
    /// Cluster id -> canonical mention id.
    pub canonical_mention_for_cluster: IndexMap<String, MentionId>,
    /// Member mention id -> canonical mention name.
    pub canonical_text_for_mention: IndexMap<MentionId, String>,
}

/// A cluster during post-processing.
#[derive(Debug, Clone)]
struct WorkCluster {
    id: String,
    members: IndexSet<MentionId>,
    pronoun_spans: Vec<Span>,
}

/// Mention lookup by id (the last mention wins for repeated ids, like a Python dict).
pub(crate) type MentionsById<'a> = HashMap<&'a str, &'a Mention>;

pub(crate) fn mentions_by_id(mentions: &[Mention]) -> MentionsById<'_> {
    mentions.iter().map(|m| (m.id.as_str(), m)).collect()
}

/// Cluster mapping and canonical selection (Python `CoreferenceStage._build_cluster_context`
/// and its helpers).
#[derive(Debug, Clone)]
pub struct ClusterBuilder<'a> {
    settings: &'a CorefSettings,
    roles: &'a RoleConfig,
}

impl<'a> ClusterBuilder<'a> {
    /// A builder applying `settings`, ranking canonical mentions with `roles`.
    pub fn new(settings: &'a CorefSettings, roles: &'a RoleConfig) -> Self {
        Self { settings, roles }
    }

    /// Maps `clusters` onto `mentions` and applies the safety rules:
    /// 1. each span maps to the best overlapping mentions ([`Self::overlapping_mentions`]),
    ///    then mentions of ineligible types are removed; pronoun spans that map to nothing are
    ///    kept as `pronoun_spans`; clusters with no mapped mention are dropped;
    /// 2. a mention in several clusters stays only in the best one (most members of its type,
    ///    then most members, then the earliest cluster);
    /// 3. duplicate and subset clusters are dropped, clusters with Jaccard similarity at or
    ///    above the threshold are merged (or the smaller dropped);
    /// 4. clusters spanning more than `max_type_groups_per_cluster` incompatible type groups
    ///    are dropped or split;
    /// 5. clusters with fewer than two members are dropped; each survivor gets a canonical
    ///    mention ([`Self::select_canonical_mention`]) whose name every member maps to.
    ///
    /// Member order is first-seen order (merged clusters append the other cluster's new
    /// members), where the Python implementation iterates unordered sets.
    pub fn build(&self, clusters: &[ResolvedCluster], mentions: &[Mention]) -> ClusterContext {
        if clusters.is_empty() {
            return ClusterContext::default();
        }
        let by_id = mentions_by_id(mentions);
        let mut index: Vec<&Mention> = mentions.iter().collect();
        index.sort_by_key(|m| m.span.start);

        let mut initial: Vec<WorkCluster> = Vec::new();
        for cluster in clusters {
            let mut members: IndexSet<MentionId> = IndexSet::new();
            let mut pronoun_spans = Vec::new();
            for span in &cluster.spans {
                let overlapping: Vec<&Mention> = self
                    .overlapping_mentions(span.start, span.end, &index)
                    .into_iter()
                    .filter(|m| self.settings.is_eligible(&m.entity_type))
                    .collect();
                if !overlapping.is_empty() {
                    members.extend(overlapping.into_iter().map(|m| m.id.clone()));
                } else if span.is_pronoun {
                    pronoun_spans.push(Span::new(span.start, span.end));
                }
            }
            if !members.is_empty() {
                initial.push(WorkCluster { id: cluster.id.clone(), members, pronoun_spans });
            }
        }

        let exclusive = self.enforce_exclusive_membership(initial, &by_id);
        let deduplicated = self.deduplicate(exclusive);
        let safe = self.apply_type_safety_guards(deduplicated, &by_id);

        let mut out = ClusterContext::default();
        for cluster in safe {
            if cluster.members.len() < 2 {
                continue;
            }
            let member_ids: Vec<MentionId> = cluster.members.into_iter().collect();
            let canonical = self.select_canonical_mention(&member_ids, &by_id);
            out.clusters.push(CorefClusterContext {
                id: cluster.id.clone(),
                member_mention_ids: member_ids.clone(),
                pronoun_spans: cluster.pronoun_spans,
            });
            let Some(canonical) = canonical else { continue };
            let canonical_name = by_id.get(canonical.as_str()).map(|m| m.name.clone());
            out.canonical_mention_for_cluster.insert(cluster.id, canonical);
            let Some(canonical_name) = canonical_name else { continue };
            for member in member_ids {
                out.canonical_text_for_mention.insert(member, canonical_name.clone());
            }
        }
        out
    }

    /// The best `max_mentions_per_span` mentions overlapping `start..end` (Python
    /// `_find_overlapping_mentions`). A mention qualifies when the overlap covers at least
    /// `min_overlap_fraction` of either span; candidates rank by that fraction, then overlap
    /// length (both descending), then start and id. `index` is sorted by start.
    pub fn overlapping_mentions<'m>(
        &self,
        start: usize,
        end: usize,
        index: &[&'m Mention],
    ) -> Vec<&'m Mention> {
        if end <= start {
            return Vec::new();
        }
        let c_len = (end - start) as f64;
        let mut candidates: Vec<(f64, usize, usize, &'m Mention)> = Vec::new();
        for &mention in index {
            let (m_start, m_end) = (mention.span.start, mention.span.end);
            if m_start >= end {
                break;
            }
            if m_end <= start || m_end <= m_start {
                continue;
            }
            let overlap = m_end.min(end).saturating_sub(m_start.max(start));
            if overlap == 0 {
                continue;
            }
            let overlap_f = overlap as f64;
            let quality = (overlap_f / c_len).max(overlap_f / (m_end - m_start) as f64);
            if quality < self.settings.min_overlap_fraction {
                continue;
            }
            candidates.push((quality, overlap, m_start, mention));
        }
        candidates.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then(b.1.cmp(&a.1))
                .then(a.2.cmp(&b.2))
                .then_with(|| a.3.id.cmp(&b.3.id))
        });
        candidates
            .into_iter()
            .take(self.settings.max_mentions_per_span)
            .map(|(_, _, _, m)| m)
            .collect()
    }

    /// Keeps each mention only in its best cluster (Python `_enforce_exclusive_membership`).
    fn enforce_exclusive_membership(
        &self,
        mut clusters: Vec<WorkCluster>,
        by_id: &MentionsById<'_>,
    ) -> Vec<WorkCluster> {
        let mut membership: IndexMap<&MentionId, Vec<usize>> = IndexMap::new();
        for (idx, cluster) in clusters.iter().enumerate() {
            for member in &cluster.members {
                membership.entry(member).or_default().push(idx);
            }
        }
        let mut removals: Vec<(usize, MentionId)> = Vec::new();
        let mut conflicts = 0_usize;
        for (member, indexes) in &membership {
            if indexes.len() < 2 {
                continue;
            }
            conflicts += 1;
            let best = match by_id.get(member.as_str()) {
                Some(mention) => self.best_cluster_for(mention, indexes, &clusters, by_id),
                None => indexes[0],
            };
            removals
                .extend(indexes.iter().filter(|i| **i != best).map(|i| (*i, (*member).clone())));
        }
        if conflicts > 0 {
            tracing::warn!(
                conflicts,
                "coreference: mentions in several clusters; keeping each in its best cluster"
            );
        }
        for (idx, member) in removals {
            clusters[idx].members.shift_remove(&member);
        }
        clusters
    }

    /// Python `_choose_best_cluster_for_mention`: most members of the mention's type, then
    /// most members, then the earliest cluster.
    fn best_cluster_for(
        &self,
        mention: &Mention,
        indexes: &[usize],
        clusters: &[WorkCluster],
        by_id: &MentionsById<'_>,
    ) -> usize {
        let score = |idx: usize| {
            let members = &clusters[idx].members;
            let same_type = members
                .iter()
                .filter(|id| {
                    by_id.get(id.as_str()).is_some_and(|m| m.entity_type == mention.entity_type)
                })
                .count();
            (same_type, members.len(), std::cmp::Reverse(idx))
        };
        indexes.iter().copied().max_by_key(|idx| score(*idx)).unwrap_or(indexes[0])
    }

    /// Drops duplicate and subset clusters and merges high-Jaccard ones (Python
    /// `_deduplicate_clusters`).
    fn deduplicate(&self, mut clusters: Vec<WorkCluster>) -> Vec<WorkCluster> {
        if clusters.len() < 2 {
            return clusters;
        }
        let total = clusters.len();
        let mut keep = vec![true; total];
        for i in 0..total {
            if !keep[i] {
                continue;
            }
            for j in i + 1..total {
                if !keep[j] {
                    continue;
                }
                let (set_i, set_j) = (&clusters[i].members, &clusters[j].members);
                let i_in_j = set_i.is_subset(set_j);
                let j_in_i = set_j.is_subset(set_i);
                if i_in_j && j_in_i {
                    tracing::debug!(dropped = %clusters[j].id, kept = %clusters[i].id, "coreference: duplicate cluster");
                    keep[j] = false;
                    continue;
                }
                if i_in_j {
                    tracing::debug!(dropped = %clusters[i].id, kept = %clusters[j].id, "coreference: subset cluster");
                    keep[i] = false;
                    break;
                }
                if j_in_i {
                    tracing::debug!(dropped = %clusters[j].id, kept = %clusters[i].id, "coreference: subset cluster");
                    keep[j] = false;
                    continue;
                }
                let intersection = set_i.intersection(set_j).count();
                let union = set_i.len() + set_j.len() - intersection;
                let jaccard = if union > 0 { intersection as f64 / union as f64 } else { 0.0 };
                if jaccard < self.settings.cluster_jaccard_threshold {
                    continue;
                }
                if self.settings.merge_overlapping_clusters {
                    let (merged_members, merged_pronouns) = {
                        let other = &clusters[j];
                        (other.members.clone(), other.pronoun_spans.clone())
                    };
                    let target = &mut clusters[i];
                    target.members.extend(merged_members);
                    target.pronoun_spans.extend(merged_pronouns);
                    keep[j] = false;
                } else if set_j.len() > set_i.len() {
                    keep[i] = false;
                    break;
                } else {
                    keep[j] = false;
                }
            }
        }
        let before = clusters.len();
        let kept: Vec<WorkCluster> =
            clusters.into_iter().zip(keep).filter_map(|(c, k)| k.then_some(c)).collect();
        if kept.len() < before {
            tracing::info!(before, after = kept.len(), "coreference: deduplicated clusters");
        }
        kept
    }

    /// Drops or splits clusters that mix incompatible type groups (Python
    /// `_apply_type_safety_guards`).
    fn apply_type_safety_guards(
        &self,
        clusters: Vec<WorkCluster>,
        by_id: &MentionsById<'_>,
    ) -> Vec<WorkCluster> {
        let mut safe = Vec::with_capacity(clusters.len());
        for cluster in clusters {
            let mut types: IndexSet<&str> = IndexSet::new();
            let mut member_types: Vec<(&MentionId, &str)> = Vec::new();
            for member in &cluster.members {
                if let Some(mention) = by_id.get(member.as_str()) {
                    types.insert(&mention.entity_type);
                    member_types.push((member, &mention.entity_type));
                }
            }
            if types.is_empty() {
                continue;
            }
            let groups_present = self
                .settings
                .incompatible_type_groups
                .iter()
                .filter(|group| group.iter().any(|t| types.contains(t.as_str())))
                .count();
            if groups_present <= self.settings.max_type_groups_per_cluster {
                safe.push(cluster);
                continue;
            }
            tracing::warn!(cluster = %cluster.id, types = ?types, "coreference: cluster mixes incompatible types");
            if self.settings.drop_mixed_type_clusters {
                continue;
            }
            let split = self.split_by_type_groups(&cluster, &member_types);
            tracing::info!(cluster = %cluster.id, parts = split.len(), "coreference: split mixed-type cluster");
            safe.extend(split);
        }
        safe
    }

    /// Python `_split_cluster_by_type_groups`: one sub-cluster per type group with at least
    /// two members (`<id>_typegroup_<group>`), plus `<id>_typegroup_other` for members of no
    /// group; the pronoun spans go to the largest sub-cluster (the first on a tie).
    fn split_by_type_groups(
        &self,
        cluster: &WorkCluster,
        member_types: &[(&MentionId, &str)],
    ) -> Vec<WorkCluster> {
        let mut grouped: IndexMap<usize, IndexSet<MentionId>> = IndexMap::new();
        let mut ungrouped: IndexSet<MentionId> = IndexSet::new();
        for (member, entity_type) in member_types {
            match self.settings.type_group(entity_type) {
                Some(group) => {
                    grouped.entry(group).or_default().insert((*member).clone());
                }
                None => {
                    ungrouped.insert((*member).clone());
                }
            }
        }
        let mut parts: Vec<WorkCluster> = grouped
            .into_iter()
            .filter(|(_, members)| members.len() >= 2)
            .map(|(group, members)| WorkCluster {
                id: format!("{}_typegroup_{group}", cluster.id),
                members,
                pronoun_spans: Vec::new(),
            })
            .collect();
        if ungrouped.len() >= 2 {
            parts.push(WorkCluster {
                id: format!("{}_typegroup_other", cluster.id),
                members: ungrouped,
                pronoun_spans: Vec::new(),
            });
        }
        if !cluster.pronoun_spans.is_empty() {
            let mut largest: Option<usize> = None;
            for (idx, part) in parts.iter().enumerate() {
                if largest.is_none_or(|l| part.members.len() > parts[l].members.len()) {
                    largest = Some(idx);
                }
            }
            if let Some(idx) = largest {
                parts[idx].pronoun_spans.clone_from(&cluster.pronoun_spans);
            }
        }
        parts
    }

    /// The canonical mention of a cluster (Python `_select_canonical_mention`): from the best
    /// non-empty tier (real names, then role + name such as `Defendant Reed`, then bare roles
    /// such as `Plaintiff`), the longest span, then the earliest, then the highest score
    /// (missing = 0), then member order.
    pub fn select_canonical_mention(
        &self,
        member_ids: &[MentionId],
        by_id: &MentionsById<'_>,
    ) -> Option<MentionId> {
        let candidates: Vec<&Mention> =
            member_ids.iter().filter_map(|id| by_id.get(id.as_str()).copied()).collect();
        let (mut real, mut hybrid, mut role) = (Vec::new(), Vec::new(), Vec::new());
        for candidate in candidates {
            if self.roles.is_hybrid_role_name(&candidate.name) {
                hybrid.push(candidate);
            } else if self.roles.is_valid_role_term(&candidate.name) {
                role.push(candidate);
            } else {
                real.push(candidate);
            }
        }
        let pool = [real, hybrid, role].into_iter().find(|tier| !tier.is_empty())?;
        let rank = |a: &&Mention, b: &&Mention| -> Ordering {
            let len = |m: &Mention| m.span.end.saturating_sub(m.span.start);
            len(b)
                .cmp(&len(a))
                .then(a.span.start.cmp(&b.span.start))
                .then_with(|| b.score.unwrap_or(0.0).total_cmp(&a.score.unwrap_or(0.0)))
        };
        pool.into_iter().min_by(rank).map(|m| m.id.clone())
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    fn work(id: &str, members: &[&str], pronouns: &[(usize, usize)]) -> WorkCluster {
        WorkCluster {
            id: id.to_owned(),
            members: members.iter().map(|m| MentionId::new(*m)).collect(),
            pronoun_spans: pronouns.iter().map(|(s, e)| Span::new(*s, *e)).collect(),
        }
    }

    fn summary(clusters: &[WorkCluster]) -> Vec<(String, Vec<String>, usize)> {
        clusters
            .iter()
            .map(|c| {
                let members = c.members.iter().map(|m| m.as_str().to_owned()).collect();
                (c.id.clone(), members, c.pronoun_spans.len())
            })
            .collect()
    }

    fn dedupe(settings: &CorefSettings, clusters: Vec<WorkCluster>) -> Vec<WorkCluster> {
        ClusterBuilder::new(settings, &RoleConfig::default()).deduplicate(clusters)
    }

    #[test]
    fn high_jaccard_clusters_merge_with_their_pronouns() {
        let settings = CorefSettings { cluster_jaccard_threshold: 0.6, ..CorefSettings::default() };
        let out = dedupe(
            &settings,
            vec![
                work("x", &["a", "b", "c", "d"], &[(1, 2)]),
                work("y", &["a", "b", "c", "e"], &[(3, 4)]),
                work("z", &["q", "r"], &[]),
            ],
        );
        assert_eq!(
            summary(&out),
            vec![
                ("x".to_owned(), ["a", "b", "c", "d", "e"].map(str::to_owned).to_vec(), 2),
                ("z".to_owned(), vec!["q".to_owned(), "r".to_owned()], 0),
            ]
        );
    }

    #[test]
    fn without_merging_the_larger_cluster_survives() {
        let settings = CorefSettings {
            cluster_jaccard_threshold: 0.5,
            merge_overlapping_clusters: false,
            ..CorefSettings::default()
        };
        let out = dedupe(
            &settings,
            vec![work("x", &["a", "b", "c", "d"], &[]), work("y", &["a", "b", "c", "e", "f"], &[])],
        );
        assert_eq!(summary(&out)[0].0, "y");
        assert_eq!(out.len(), 1);
        // Below the threshold both stay.
        let out =
            dedupe(&settings, vec![work("x", &["a", "b"], &[]), work("y", &["b", "c", "d"], &[])]);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn empty_and_subset_clusters_are_dropped() {
        let out = dedupe(
            &CorefSettings::default(),
            vec![work("x", &[], &[]), work("y", &["a"], &[]), work("z", &["a", "b"], &[])],
        );
        assert_eq!(summary(&out), vec![("z".to_owned(), vec!["a".to_owned(), "b".to_owned()], 0)]);
    }

    #[test]
    fn labels_and_validates_resolver_spans() {
        let text = "Zoë said she left.";
        let raw = vec![
            ce_models::CorefCluster { spans: vec![(0, 3), (9, 12), (9, 12), (5, 5), (10, 40)] },
            ce_models::CorefCluster { spans: vec![] },
            ce_models::CorefCluster { spans: vec![(4, 8)] },
        ];
        let labelled = label_clusters("doc", text, &raw);
        assert_eq!(
            labelled,
            vec![
                ResolvedCluster {
                    id: "doc_cluster_0".to_owned(),
                    spans: vec![
                        ClusterSpan {
                            start: 0, end: 3, text: "Zoë".to_owned(), is_pronoun: false
                        },
                        ClusterSpan { start: 9, end: 12, text: "she".to_owned(), is_pronoun: true },
                    ],
                },
                ResolvedCluster {
                    id: "doc_cluster_2".to_owned(),
                    spans: vec![ClusterSpan {
                        start: 4,
                        end: 8,
                        text: "said".to_owned(),
                        is_pronoun: false
                    }],
                },
            ]
        );
    }
}
