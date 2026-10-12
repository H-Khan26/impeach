//! Mention quality filters and split-mention healing (Python `stages/mention.py`:
//! `_normalize_mention_name`, `_is_low_quality_mention`, `_heal_split_mentions`).
//!
//! The rules come from `quality_filters.yaml` through the mappings registry, as in Python
//! whenever the registry loads: names shorter than `min_mention_length`, truncated legal
//! words (`partial_word_patterns`, when `filter_partial_words`), short names
//! (`< short_mention_length`) scoring below `short_mention_confidence_threshold`, and the
//! `additional_filter_patterns` are low quality. The stage-level `min_mention_length`,
//! `short_confidence_threshold` and `filter_partial_words` are Python's fallbacks for a
//! registry that fails to load; the Rust registry always loads (or the stage fails to
//! build), so only the stage's `normalize_names` switch applies here.

use ce_config::mappings::QualityFilters;
use ce_core::{DetectionSource, Mention, Metadata};
use serde_json::Value;

use crate::text::CharIndex;

/// Quality rules of one `MentionStage`.
#[derive(Debug, Clone)]
pub struct MentionQuality {
    filters: QualityFilters,
    normalize_names: bool,
}

impl MentionQuality {
    /// Rules from `quality_filters.yaml`; `normalize_names` is the stage switch (names are
    /// normalised only when it and the filters' `normalize_names` are both set).
    #[must_use]
    pub fn new(filters: QualityFilters, normalize_names: bool) -> Self {
        Self { filters, normalize_names }
    }

    /// Whether the stage normalises names.
    #[must_use]
    pub fn normalize_names(&self) -> bool {
        self.normalize_names
    }

    /// Collapses whitespace in a name (`"Thomas\nReynolds"` -> `"Thomas Reynolds"`) when
    /// normalisation is on; otherwise returns it unchanged.
    #[must_use]
    pub fn normalize_name(&self, name: &str) -> String {
        if self.normalize_names {
            self.filters.normalize_mention_name(name)
        } else {
            name.to_owned()
        }
    }

    /// Whether a mention is a likely NER artifact (too short, truncated word, short with a low
    /// score, or matching an additional filter pattern).
    #[must_use]
    pub fn is_low_quality(&self, mention: &Mention) -> bool {
        self.filters.is_low_quality(&mention.name, mention.score)
    }

    /// Merges runs of adjacent same-type shards inside one chunk (chunk-local spans over
    /// `text`).
    ///
    /// Mentions are sorted by `(start, end)`. A run grows while the next mention has the same
    /// type, starts after the previous shard ends, and the gap holds no alphanumeric
    /// character; it is merged only when every shard is low quality, the merged span is not,
    /// and the merged text has at most [`max_merge_tokens`] tokens. The merged mention keeps
    /// the first shard's id, type, score and source, covers the whole run, takes its name
    /// from the text (normalised), and aggregates the shards' sources and metadata.
    #[must_use]
    pub fn heal_split_mentions(&self, mentions: Vec<Mention>, text: &str) -> Vec<Mention> {
        if mentions.is_empty() {
            return mentions;
        }
        let index = CharIndex::new(text);
        let mut sorted = mentions;
        sorted.sort_by_key(|m| (m.span.start, m.span.end));
        let n = sorted.len();
        let mut healed: Vec<Mention> = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            let mut run_end = i;
            let mut best: Option<Mention> = None;
            let mut j = i + 1;
            while j < n {
                let (last, next) = (&sorted[run_end], &sorted[j]);
                if !can_merge(last, next, &index) {
                    break;
                }
                let first = &sorted[i];
                let merged_text = index.slice(first.span.start, next.span.end);
                let mut candidate = first.clone();
                candidate.span.end = next.span.end;
                candidate.name = self.normalize_name(merged_text);

                let low_shards = sorted[i..=j].iter().all(|shard| self.is_low_quality(shard));
                let low_merged = self.is_low_quality(&candidate);
                let tokens = ce_text::normalize::py_split(merged_text).count();
                if low_shards && !low_merged && tokens <= max_merge_tokens(&candidate.entity_type) {
                    run_end = j;
                    best = Some(candidate);
                    j += 1;
                } else {
                    break;
                }
            }
            match best {
                Some(template) if run_end > i => {
                    healed.push(combine_shards(&sorted[i..=run_end], template));
                    i = run_end + 1;
                }
                _ => {
                    healed.push(sorted[i].clone());
                    i += 1;
                }
            }
        }
        healed
    }
}

/// Whether `right` may extend a run ending with `left`: same type, strictly after it, and
/// only whitespace or punctuation in between.
fn can_merge(left: &Mention, right: &Mention, index: &CharIndex<'_>) -> bool {
    if left.entity_type != right.entity_type || right.span.start <= left.span.end {
        return false;
    }
    !index.slice(left.span.end, right.span.start).chars().any(char::is_alphanumeric)
}

/// Token cap of a healed mention by type: `PER` 4, `ORG` / `LOC` / `FACILITY` 6, else 5.
#[must_use]
pub fn max_merge_tokens(entity_type: &str) -> usize {
    match entity_type.to_uppercase().as_str() {
        "PER" => 4,
        "ORG" | "LOC" | "FACILITY" => 6,
        _ => 5,
    }
}

/// The healed mention: `template` with the shards' sources (deduplicated by
/// `(name, label)`, each shard's `sources` then its own source) and metadata (template keys
/// first; missing or falsy values filled from the shards).
fn combine_shards(shards: &[Mention], template: Mention) -> Mention {
    let mut winner = template;
    let mut sources: Vec<DetectionSource> = Vec::new();
    let mut add = |name: &str, label: &str, score: Option<f64>| {
        if name.is_empty() || label.is_empty() {
            return;
        }
        if sources.iter().any(|s| s.name == name && s.label == label) {
            return;
        }
        sources.push(DetectionSource { name: name.to_owned(), label: label.to_owned(), score });
    };
    for shard in shards {
        for source in &shard.sources {
            add(&source.name, &source.label, source.score);
        }
        if let Some(name) = &shard.source {
            add(name, &shard.entity_type, shard.score);
        }
    }
    winner.sources = sources;
    let mut metadata = winner.metadata.clone();
    for shard in shards {
        fill_metadata(&mut metadata, &shard.metadata);
    }
    winner.metadata = metadata;
    winner
}

/// Adds keys of `from` missing in `into`, and replaces falsy values of `into` with truthy
/// ones (Python `if k not in merged or (not merged[k] and v)`).
pub(crate) fn fill_metadata(into: &mut Metadata, from: &Metadata) {
    for (key, value) in from {
        match into.get_mut(key) {
            None => {
                into.insert(key.clone(), value.clone());
            }
            Some(existing) if !is_truthy(existing) && is_truthy(value) => {
                *existing = value.clone();
            }
            Some(_) => {}
        }
    }
}

/// Python truthiness of a JSON value.
pub(crate) fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use ce_config::MappingsRegistry;
    use ce_core::Span;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;

    fn quality() -> MentionQuality {
        let mappings = MappingsRegistry::load().expect("mappings");
        MentionQuality::new(mappings.quality_filters.clone(), true)
    }

    fn mention(id: &str, ty: &str, name: &str, start: usize, end: usize, score: f64) -> Mention {
        let mut m = Mention::new(id, ty, name, Span::new(start, end));
        m.score = Some(score);
        m.source = Some("gliner2".into());
        m
    }

    #[test]
    fn low_quality_rules_from_quality_filters_yaml() {
        let q = quality();
        let low = |name: &str, score: Option<f64>| {
            let mut m = Mention::new("m", "PER", name, Span::new(0, 1));
            m.score = score;
            q.is_low_quality(&m)
        };
        assert!(low("ff", Some(0.99)), "shorter than min_mention_length");
        assert!(low("  ab  ", None), "length is measured after strip");
        assert!(low("Plainti", Some(0.99)), "partial word");
        assert!(low("defenda", Some(0.99)), "partial words ignore case");
        assert!(low("Ann", Some(0.5)), "short and below 0.75");
        assert!(!low("Ann", Some(0.75)));
        assert!(!low("Ann", None), "no score, no confidence rule");
        assert!(!low("Annabel", Some(0.1)), "long names skip the confidence rule");
        assert!(low("e.g.", Some(1.0)), "additional filter pattern");
        assert!(low("pp. 12", Some(1.0)), "page reference");
        assert!(low("---", Some(1.0)), "punctuation only");
        assert!(!low("Plaintiff", Some(0.9)));
    }

    #[test]
    fn names_normalise_only_when_enabled() {
        let q = quality();
        assert_eq!(q.normalize_name(" Thomas\nReynolds  Jr "), "Thomas Reynolds Jr");
        let mappings = MappingsRegistry::load().expect("mappings");
        let off = MentionQuality::new(mappings.quality_filters.clone(), false);
        assert_eq!(off.normalize_name("Thomas\nReynolds"), "Thomas\nReynolds");
    }

    // Port of test_mention_split_healing.py::test_heal_split_mentions_merges_low_quality_shards.
    #[test]
    fn heals_low_quality_shards() {
        let q = quality();
        let text = "Plainti ff";
        let healed = q.heal_split_mentions(
            vec![
                mention("e1", "PER", "Plainti", 0, 7, 0.95),
                mention("e2", "PER", "ff", 8, 10, 0.95),
            ],
            text,
        );
        assert_eq!(healed.len(), 1);
        let merged = &healed[0];
        assert_eq!((merged.span.start, merged.span.end), (0, 10));
        assert_eq!(merged.name, "Plainti ff");
        assert_eq!(merged.id.as_str(), "e1");
        assert_eq!(
            merged.sources,
            vec![DetectionSource {
                name: "gliner2".into(),
                label: "PER".into(),
                score: Some(0.95)
            }]
        );
    }

    // Port of test_heal_split_mentions_keeps_high_quality_enumeration_separate.
    #[test]
    fn keeps_high_quality_neighbours_separate() {
        let q = quality();
        let text = "Wilshire Blvd San Francisco";
        let healed = q.heal_split_mentions(
            vec![
                mention("e1", "LOC", "Wilshire Blvd", 0, 13, 0.95),
                mention("e2", "LOC", "San Francisco", 14, 27, 0.95),
            ],
            text,
        );
        let names: Vec<&str> = healed.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["Wilshire Blvd", "San Francisco"]);
    }

    #[test]
    fn healing_requires_same_type_a_gap_and_no_words_between() {
        let q = quality();
        // Different types.
        let healed = q.heal_split_mentions(
            vec![mention("a", "PER", "Plainti", 0, 7, 0.9), mention("b", "ORG", "ff", 8, 10, 0.9)],
            "Plainti ff",
        );
        assert_eq!(healed.len(), 2);
        // Touching spans (no gap) are not merged.
        let healed = q.heal_split_mentions(
            vec![mention("a", "PER", "Plainti", 0, 7, 0.9), mention("b", "PER", "ff", 7, 9, 0.9)],
            "Plaintiff",
        );
        assert_eq!(healed.len(), 2);
        // A word in the gap blocks the merge.
        let healed = q.heal_split_mentions(
            vec![mention("a", "PER", "Plainti", 0, 7, 0.9), mention("b", "PER", "ff", 11, 13, 0.9)],
            "Plainti and ff",
        );
        assert_eq!(healed.len(), 2);
        // Punctuation in the gap is fine.
        let healed = q.heal_split_mentions(
            vec![mention("a", "PER", "Plainti", 0, 7, 0.9), mention("b", "PER", "ff", 9, 11, 0.9)],
            "Plainti- ff",
        );
        assert_eq!(healed.len(), 1);
        assert_eq!(healed[0].name, "Plainti- ff");
    }

    #[test]
    fn healing_respects_the_token_cap_and_grows_runs() {
        let q = quality();
        // Three shards heal into one five-token ORG name.
        let text = "Ab Cd, Ef Gh Ij";
        let healed = q.heal_split_mentions(
            vec![
                mention("a", "ORG", "Ab", 0, 2, 0.9),
                mention("b", "ORG", "Cd", 3, 5, 0.9),
                mention("c", "ORG", "Ef", 7, 9, 0.9),
            ],
            text,
        );
        assert_eq!(healed.len(), 1);
        assert_eq!(healed[0].name, "Ab Cd, Ef");
        // PER caps at four tokens; the fifth shard stays on its own.
        let text = "Xy Q R S T";
        let shards = vec![
            mention("a", "PER", "Xy", 0, 2, 0.9),
            mention("b", "PER", "Q", 3, 4, 0.9),
            mention("c", "PER", "R", 5, 6, 0.9),
            mention("d", "PER", "S", 7, 8, 0.9),
            mention("e", "PER", "T", 9, 10, 0.9),
        ];
        let healed = q.heal_split_mentions(shards, text);
        let names: Vec<&str> = healed.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["Xy Q R S", "T"]);
        assert_eq!(max_merge_tokens("facility"), 6);
        assert_eq!(max_merge_tokens("DATE"), 5);
    }

    #[test]
    fn healed_mentions_aggregate_sources_and_metadata() {
        let q = quality();
        let mut a = mention("a", "PER", "Plainti", 0, 7, 0.9);
        a.metadata.insert("k".into(), json!(""));
        let mut b = mention("b", "PER", "ff", 8, 10, 0.8);
        b.source = Some("regex_detector".into());
        b.sources = vec![DetectionSource { name: "x".into(), label: "PER".into(), score: None }];
        b.metadata.insert("k".into(), json!("v"));
        b.metadata.insert("z".into(), json!(1));
        let healed = q.heal_split_mentions(vec![b, a], "Plainti ff");
        assert_eq!(healed.len(), 1);
        let names: Vec<&str> = healed[0].sources.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["gliner2", "x", "regex_detector"]);
        assert_eq!(Value::Object(healed[0].metadata.clone()), json!({"k": "v", "z": 1}));
    }

    #[test]
    fn truthiness_matches_python() {
        for falsy in
            [json!(null), json!(false), json!(0), json!(0.0), json!(""), json!([]), json!({})]
        {
            assert!(!is_truthy(&falsy), "{falsy}");
        }
        for truthy in [json!(true), json!(1), json!(-0.5), json!("x"), json!([0]), json!({"a": 0})]
        {
            assert!(is_truthy(&truthy), "{truthy}");
        }
    }
}
