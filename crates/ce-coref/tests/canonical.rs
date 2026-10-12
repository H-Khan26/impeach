//! Role-aware canonical mention selection (coreference cases of the Python
//! `tests/unit/test_coref_canonical_role_handling.py`).

use std::collections::HashMap;

use ce_core::{Mention, MentionId, Span};
use ce_coref::{ClusterBuilder, CorefSettings, RoleConfig};

fn mention(id: &str, name: &str, start: usize, end: usize, score: Option<f64>) -> Mention {
    let mut m = Mention::new(id, "PER", name, Span::new(start, end));
    m.score = score;
    m.source = Some("test".to_owned());
    m.doc_id = Some("doc1".into());
    m
}

/// The canonical id among `mentions` (all members) with the shipped `roles.yaml`.
fn canonical(mentions: &[Mention]) -> Option<String> {
    let roles = RoleConfig::load(None).expect("roles.yaml loads");
    let settings = CorefSettings::default();
    let builder = ClusterBuilder::new(&settings, &roles);
    let by_id: HashMap<&str, &Mention> = mentions.iter().map(|m| (m.id.as_str(), m)).collect();
    let ids: Vec<MentionId> = mentions.iter().map(|m| m.id.clone()).collect();
    builder.select_canonical_mention(&ids, &by_id).map(MentionId::into_inner)
}

#[test]
fn pure_real_name_beats_role_term() {
    let got = canonical(&[
        mention("m1", "Plaintiff", 0, 9, None),
        mention("m2", "Maria Lopez", 100, 111, None),
    ]);
    assert_eq!(got.as_deref(), Some("m2"));
}

#[test]
fn pure_real_name_beats_hybrid_role() {
    let got = canonical(&[
        mention("m1", "Defendant Reed", 0, 14, None),
        mention("m2", "John Reed", 100, 109, None),
    ]);
    assert_eq!(got.as_deref(), Some("m2"));
}

#[test]
fn hybrid_role_beats_pure_role() {
    let got = canonical(&[
        mention("m1", "Defendant", 0, 9, None),
        mention("m2", "Defendant Reed", 100, 114, None),
    ]);
    assert_eq!(got.as_deref(), Some("m2"));
}

#[test]
fn role_only_cluster_selects_the_earliest_role() {
    let got = canonical(&[
        mention("m1", "Plaintiff", 0, 9, None),
        mention("m2", "Defendant", 100, 109, None),
    ]);
    assert_eq!(got.as_deref(), Some("m1"));
}

#[test]
fn within_tier_prefers_earlier_position() {
    let got = canonical(&[
        mention("m1", "John Doe", 100, 110, None),
        mention("m2", "Jane Roe", 50, 60, None),
    ]);
    assert_eq!(got.as_deref(), Some("m2"));
}

#[test]
fn within_tier_prefers_longer_span() {
    let got = canonical(&[
        mention("m1", "Lopez", 50, 55, None),
        mention("m2", "Maria Lopez", 50, 61, None),
    ]);
    assert_eq!(got.as_deref(), Some("m2"));
}

#[test]
fn within_tier_longer_span_beats_earlier_position() {
    let got = canonical(&[
        mention("m1", "Alice", 10, 15, None),
        mention("m2", "Alice Johnson", 100, 113, None),
    ]);
    assert_eq!(got.as_deref(), Some("m2"));
}

#[test]
fn within_tier_higher_score_breaks_ties() {
    let got = canonical(&[
        mention("m1", "Maria Lopez", 50, 61, Some(0.60)),
        mention("m2", "Maria Lopez", 50, 61, Some(0.95)),
    ]);
    assert_eq!(got.as_deref(), Some("m2"));
    // Full ties keep member order.
    let got = canonical(&[
        mention("m1", "Maria Lopez", 50, 61, None),
        mention("m2", "Maria Lopez", 50, 61, Some(0.0)),
    ]);
    assert_eq!(got.as_deref(), Some("m1"));
}

#[test]
fn role_detection() {
    let roles = RoleConfig::load(None).expect("roles.yaml loads");
    for role in ["Plaintiff", "Defendant", "Manager"] {
        assert!(roles.is_valid_role_term(role), "{role}");
    }
    for name in ["Maria Lopez", "John Smith"] {
        assert!(!roles.is_valid_role_term(name), "{name}");
        assert!(!roles.is_hybrid_role_name(name), "{name}");
    }
    for hybrid in ["Defendant Reed", "Plaintiff Smith", "Manager Johnson"] {
        assert!(roles.is_hybrid_role_name(hybrid), "{hybrid}");
    }
    assert!(!roles.is_hybrid_role_name("Defendant"));
}
