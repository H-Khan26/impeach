//! Merging overlapping mentions from several detectors (Python `services/ner/merging.py`).
//!
//! Mentions are stably sorted by start; a cluster grows while the next mention starts before
//! the cluster's furthest end. Each cluster keeps one winner, compared pairwise in cluster
//! order: higher [`DetectorConfig::get_source_priority`] of `(type, source)` (high-precision
//! labels +1000, then the per-label source order, sources matched by substring), then the
//! longer span, then the higher score (missing = 0); ties keep the earlier mention. The
//! winner's `sources` aggregate every member's `sources` and own `(source, type, score)`,
//! deduplicated by `(name, label)`; its metadata keeps its own keys and fills missing or
//! falsy values from the other members. A single-mention cluster only gets its own source
//! added to an empty `sources`.

use ce_config::mappings::DetectorConfig;
use ce_core::{DetectionSource, Mention};

use crate::quality::fill_metadata;

/// Resolves overlapping mentions into one mention per overlap cluster, in start order.
#[must_use]
pub fn merge_mentions(mentions: Vec<Mention>, config: &DetectorConfig) -> Vec<Mention> {
    let mut sorted = mentions;
    sorted.sort_by_key(|m| m.span.start);
    let mut merged = Vec::new();
    let mut cluster: Vec<Mention> = Vec::new();
    let mut cluster_end = 0;
    for mention in sorted {
        if !cluster.is_empty() && mention.span.start >= cluster_end {
            merged.push(resolve_cluster(std::mem::take(&mut cluster), config));
        }
        cluster_end =
            if cluster.is_empty() { mention.span.end } else { cluster_end.max(mention.span.end) };
        cluster.push(mention);
    }
    if !cluster.is_empty() {
        merged.push(resolve_cluster(cluster, config));
    }
    merged
}

fn priority(mention: &Mention, config: &DetectorConfig) -> i64 {
    config.get_source_priority(&mention.entity_type, mention.source.as_deref().unwrap_or(""))
}

fn span_len(mention: &Mention) -> usize {
    mention.span.end.saturating_sub(mention.span.start)
}

/// Index of the winning mention of a cluster.
fn best_index(cluster: &[Mention], config: &DetectorConfig) -> usize {
    let mut best = 0;
    for (i, candidate) in cluster.iter().enumerate().skip(1) {
        let current = &cluster[best];
        let (p_best, p_cand) = (priority(current, config), priority(candidate, config));
        let better = if p_cand != p_best {
            p_cand > p_best
        } else if span_len(candidate) != span_len(current) {
            span_len(candidate) > span_len(current)
        } else {
            candidate.score.unwrap_or(0.0) > current.score.unwrap_or(0.0)
        };
        if better {
            best = i;
        }
    }
    best
}

fn resolve_cluster(mut cluster: Vec<Mention>, config: &DetectorConfig) -> Mention {
    if cluster.len() == 1 {
        let mut mention = cluster.remove(0);
        if mention.sources.is_empty() {
            if let Some(source) = mention.source.clone().filter(|s| !s.is_empty()) {
                mention.sources = vec![DetectionSource {
                    name: source,
                    label: mention.entity_type.clone(),
                    score: mention.score,
                }];
            }
        }
        return mention;
    }
    let best = best_index(&cluster, config);
    let mut sources: Vec<DetectionSource> = Vec::new();
    let mut push = |source: DetectionSource| {
        if !sources.iter().any(|s| s.name == source.name && s.label == source.label) {
            sources.push(source);
        }
    };
    let mut metadata = cluster[best].metadata.clone();
    for member in &cluster {
        for source in &member.sources {
            push(source.clone());
        }
        if let Some(name) = member.source.as_deref().filter(|s| !s.is_empty()) {
            push(DetectionSource {
                name: name.to_owned(),
                label: member.entity_type.clone(),
                score: member.score,
            });
        }
        fill_metadata(&mut metadata, &member.metadata);
    }
    let mut winner = cluster.swap_remove(best);
    winner.sources = sources;
    winner.metadata = metadata;
    winner
}

#[cfg(test)]
mod tests {
    use ce_config::MappingsRegistry;
    use ce_core::Span;
    use pretty_assertions::assert_eq;
    use serde_json::{json, Value};

    use super::*;

    fn config() -> &'static DetectorConfig {
        &MappingsRegistry::load().expect("mappings").detector_config
    }

    fn mention(
        ty: &str,
        name: &str,
        start: usize,
        end: usize,
        score: f64,
        source: &str,
    ) -> Mention {
        let mut m = Mention::new(format!("{source}-{start}"), ty, name, Span::new(start, end));
        m.score = Some(score);
        m.source = Some(source.into());
        m
    }

    fn summary(mentions: &[Mention]) -> Vec<(String, String, String)> {
        mentions
            .iter()
            .map(|m| (m.entity_type.clone(), m.name.clone(), m.source.clone().unwrap_or_default()))
            .collect()
    }

    #[test]
    fn priority_scores_follow_detector_config() {
        let c = config();
        assert_eq!(c.get_source_priority("CASE_CITATION", "eyecite"), 1040);
        assert_eq!(c.get_source_priority("STATUTE", "regex_detector"), 1020);
        assert_eq!(c.get_source_priority("PER", "gliner2"), 20);
        assert_eq!(c.get_source_priority("MONEY", "gliner2"), 1002);
        assert_eq!(c.get_source_priority("PRODUCT", "gliner2"), 2);
    }

    #[test]
    fn single_mentions_get_their_own_source() {
        let merged =
            merge_mentions(vec![mention("PER", "Ann Lee", 0, 7, 0.9, "gliner2")], config());
        assert_eq!(
            merged[0].sources,
            vec![DetectionSource { name: "gliner2".into(), label: "PER".into(), score: Some(0.9) }]
        );
    }

    #[test]
    fn label_source_order_wins_overlaps() {
        let merged = merge_mentions(
            vec![
                mention("DATE", "March 4, 2023", 10, 23, 0.8, "gliner2"),
                mention("DATE", "March 4, 2023", 10, 23, 1.0, "regex_detector"),
                mention("MONEY", "$300", 30, 34, 1.0, "regex_detector"),
                mention("MONEY", "$300 fee", 30, 38, 0.95, "gliner2"),
            ],
            config(),
        );
        assert_eq!(
            summary(&merged),
            vec![
                ("DATE".into(), "March 4, 2023".into(), "regex_detector".into()),
                ("MONEY".into(), "$300".into(), "regex_detector".into()),
            ]
        );
        // MONEY is high precision for every source; regex_detector leads its source list (1020
        // against 1002 for gliner2 through the default order), so it beats the longer span.
        let sources: Vec<(&str, &str)> =
            merged[1].sources.iter().map(|s| (s.name.as_str(), s.label.as_str())).collect();
        assert_eq!(sources, [("regex_detector", "MONEY"), ("gliner2", "MONEY")]);
    }

    #[test]
    fn eyecite_beats_regex_on_statutes() {
        let merged = merge_mentions(
            vec![
                mention("STATUTE", "42 U.S.C. § 1983", 0, 16, 1.0, "regex_detector"),
                mention("CASE_CITATION", "42 U.S.C. § 1983", 0, 16, 1.0, "eyecite"),
                mention("ORG", "U.S.C.", 3, 9, 0.99, "gliner2"),
            ],
            config(),
        );
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].entity_type, "CASE_CITATION");
        assert_eq!(merged[0].sources.len(), 3);
        // High-precision labels beat longer generic spans.
        let merged = merge_mentions(
            vec![
                mention("ORG", "Exhibit 12 Holdings", 0, 19, 0.99, "gliner2"),
                mention("EXHIBIT_ID", "Exhibit 12", 0, 10, 1.0, "regex_detector"),
            ],
            config(),
        );
        assert_eq!(merged[0].entity_type, "EXHIBIT_ID");
    }

    #[test]
    fn ties_fall_back_to_length_then_score_then_order() {
        let merged = merge_mentions(
            vec![
                mention("PER", "Ann", 0, 3, 0.9, "gliner2"),
                mention("PER", "Ann Lee", 0, 7, 0.5, "gliner2"),
            ],
            config(),
        );
        assert_eq!(merged[0].name, "Ann Lee");
        let merged = merge_mentions(
            vec![
                mention("PER", "Ann Lee", 0, 7, 0.5, "gliner2"),
                mention("ORG", "Ann Lee", 0, 7, 0.6, "gliner2"),
            ],
            config(),
        );
        assert_eq!(merged[0].entity_type, "ORG");
        let merged = merge_mentions(
            vec![
                mention("PER", "Ann Lee", 0, 7, 0.6, "gliner2"),
                mention("ORG", "Ann Lee", 0, 7, 0.6, "gliner2"),
            ],
            config(),
        );
        assert_eq!(merged[0].entity_type, "PER");
    }

    #[test]
    fn clusters_chain_through_overlaps_and_stop_at_touching_spans() {
        let merged = merge_mentions(
            vec![
                mention("PER", "c", 8, 12, 0.9, "gliner2"),
                mention("PER", "a", 0, 5, 0.9, "gliner2"),
                mention("PER", "b", 4, 9, 0.9, "gliner2"),
                mention("PER", "d", 12, 15, 0.9, "gliner2"),
            ],
            config(),
        );
        let names: Vec<&str> = merged.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["a", "d"]);
    }

    #[test]
    fn metadata_prefers_the_winner_and_fills_gaps() {
        let mut regex = mention("CASE_CITATION", "347 U.S. 483", 0, 12, 1.0, "regex_detector");
        regex.metadata.insert("volume".into(), json!("999"));
        regex.metadata.insert("court".into(), json!("scotus"));
        let mut cite = mention("CASE_CITATION", "347 U.S. 483", 0, 12, 1.0, "eyecite");
        cite.metadata.insert("volume".into(), json!("347"));
        cite.metadata.insert("year".into(), json!(null));
        let mut other = mention("CASE_CITATION", "347 U.S.", 0, 8, 1.0, "regex_detector");
        other.metadata.insert("year".into(), json!(1954));
        let merged = merge_mentions(vec![regex, cite, other], config());
        assert_eq!(merged[0].source.as_deref(), Some("eyecite"));
        assert_eq!(
            Value::Object(merged[0].metadata.clone()),
            json!({"volume": "347", "year": 1954, "court": "scotus"})
        );
    }
}
