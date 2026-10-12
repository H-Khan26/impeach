//! Cluster safety rules (cases of the Python `tests/unit/test_coref_cluster_safety.py`) and
//! span mapping.

use std::collections::HashSet;

use ce_core::{Mention, MentionId, Span};
use ce_coref::{
    ClusterBuilder, ClusterContext, ClusterSpan, CorefSettings, ResolvedCluster, RoleConfig,
};
use pretty_assertions::assert_eq;

fn mention(id: &str, entity_type: &str, name: &str, start: usize, end: usize) -> Mention {
    let mut m = Mention::new(id, entity_type, name, Span::new(start, end));
    m.doc_id = Some("doc1".into());
    m
}

fn span(start: usize, end: usize, text: &str) -> ClusterSpan {
    ClusterSpan { start, end, text: text.to_owned(), is_pronoun: ce_coref::is_pronoun(text) }
}

fn cluster(id: &str, spans: Vec<ClusterSpan>) -> ResolvedCluster {
    ResolvedCluster { id: id.to_owned(), spans }
}

fn settings(drop_mixed: bool) -> CorefSettings {
    CorefSettings {
        eligible_types: ["PER", "ORG", "LOC", "DATE", "PRODUCT", "FACILITY"]
            .map(str::to_owned)
            .to_vec(),
        drop_mixed_type_clusters: drop_mixed,
        ..CorefSettings::default()
    }
}

fn build(
    settings: &CorefSettings,
    clusters: &[ResolvedCluster],
    mentions: &[Mention],
) -> ClusterContext {
    ClusterBuilder::new(settings, &RoleConfig::default()).build(clusters, mentions)
}

fn person_vehicle() -> (Vec<Mention>, Vec<ResolvedCluster>) {
    let mentions = vec![
        mention("ent_001", "PER", "Maria Lopez", 0, 11),
        mention("ent_002", "PER", "Lopez", 50, 55),
        mention("ent_003", "PRODUCT", "Honda", 100, 105),
        mention("ent_004", "PRODUCT", "Civic", 106, 111),
    ];
    let clusters = vec![cluster(
        "cluster_mixed",
        vec![
            span(0, 11, "Maria Lopez"),
            span(50, 55, "Lopez"),
            span(100, 105, "Honda"),
            span(106, 111, "Civic"),
        ],
    )];
    (mentions, clusters)
}

#[test]
fn mixed_person_vehicle_cluster_is_dropped() {
    let (mentions, clusters) = person_vehicle();
    let out = build(&settings(true), &clusters, &mentions);
    assert!(out.clusters.is_empty());
    assert!(out.canonical_mention_for_cluster.is_empty());
    assert!(out.canonical_text_for_mention.is_empty());
}

#[test]
fn mixed_cluster_is_split_into_type_coherent_parts() {
    let (mentions, clusters) = person_vehicle();
    let out = build(&settings(false), &clusters, &mentions);
    let ids: Vec<&str> = out.clusters.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(ids, vec!["cluster_mixed_typegroup_0", "cluster_mixed_typegroup_1"]);
    for c in &out.clusters {
        let types: HashSet<&str> = c
            .member_mention_ids
            .iter()
            .map(|id| {
                mentions.iter().find(|m| &m.id == id).expect("member exists").entity_type.as_str()
            })
            .collect();
        assert_eq!(types.len(), 1, "{c:?}");
    }
    assert_eq!(out.canonical_mention_for_cluster["cluster_mixed_typegroup_0"].as_str(), "ent_001");
}

#[test]
fn split_gives_pronouns_to_the_largest_part_and_keeps_ungrouped_members() {
    let mentions = vec![
        mention("p1", "PER", "Maria Lopez", 0, 11),
        mention("p2", "PER", "Lopez", 20, 25),
        mention("p3", "PER", "Maria", 30, 35),
        mention("o1", "ORG", "Acme", 40, 44),
        mention("o2", "ORG", "Acme Corp", 50, 59),
        mention("v1", "PRODUCT", "Civic", 60, 65),
        mention("v2", "PRODUCT", "Honda", 70, 75),
    ];
    let clusters = vec![cluster(
        "c",
        vec![
            span(0, 11, "Maria Lopez"),
            span(20, 25, "Lopez"),
            span(30, 35, "Maria"),
            span(40, 44, "Acme"),
            span(50, 59, "Acme Corp"),
            span(60, 65, "Civic"),
            span(70, 75, "Honda"),
            span(80, 83, "she"),
        ],
    )];
    let out = build(&settings(false), &clusters, &mentions);
    let summary: Vec<(&str, usize, usize)> = out
        .clusters
        .iter()
        .map(|c| (c.id.as_str(), c.member_mention_ids.len(), c.pronoun_spans.len()))
        .collect();
    assert_eq!(
        summary,
        vec![("c_typegroup_0", 3, 1), ("c_typegroup_1", 2, 0), ("c_typegroup_other", 2, 0)]
    );
}

#[test]
fn exact_duplicate_clusters_are_deduplicated() {
    let mentions = vec![
        mention("ent_001", "ORG", "Green Valley", 0, 12),
        mention("ent_002", "ORG", "Green Valley", 50, 62),
    ];
    let spans = vec![span(0, 12, "Green Valley"), span(50, 62, "Green Valley")];
    let clusters = vec![cluster("cluster_1", spans.clone()), cluster("cluster_2", spans)];
    let out = build(&settings(true), &clusters, &mentions);
    assert_eq!(out.clusters.len(), 1);
    assert_eq!(out.clusters[0].id, "cluster_1");
    assert_eq!(out.clusters[0].member_mention_ids.len(), 2);
}

#[test]
fn subset_cluster_gives_way_to_its_superset() {
    let mentions = vec![
        mention("ent_001", "ORG", "Green Valley Logistics, Inc", 0, 28),
        mention("ent_002", "ORG", "Green Valley", 50, 62),
        mention("ent_003", "ORG", "Valley", 100, 106),
        mention("ent_004", "ORG", "Green", 150, 155),
    ];
    let clusters = vec![
        cluster(
            "cluster_subset",
            vec![span(0, 28, "Green Valley Logistics, Inc"), span(50, 62, "Green Valley")],
        ),
        cluster(
            "cluster_superset",
            vec![
                span(0, 28, "Green Valley Logistics, Inc"),
                span(50, 62, "Green Valley"),
                span(100, 106, "Valley"),
                span(150, 155, "Green"),
            ],
        ),
    ];
    let out = build(&settings(true), &clusters, &mentions);
    assert_eq!(out.clusters.len(), 1);
    assert_eq!(out.clusters[0].id, "cluster_superset");
    assert_eq!(out.clusters[0].member_mention_ids.len(), 4);
    assert_eq!(out.canonical_mention_for_cluster["cluster_superset"].as_str(), "ent_001");
}

#[test]
fn a_mention_belongs_to_one_cluster_only() {
    let mentions = vec![
        mention("ent_lopez_1", "PER", "Maria Lopez", 0, 11),
        mention("ent_lopez_2", "PER", "Lopez", 50, 55),
        mention("ent_reed", "PER", "Reed", 100, 104),
    ];
    let clusters = vec![
        cluster("cluster_person_1", vec![span(0, 11, "Maria Lopez"), span(50, 55, "Lopez")]),
        cluster(
            "cluster_person_2",
            vec![span(0, 11, "Maria Lopez"), span(50, 55, "Lopez"), span(100, 104, "Reed")],
        ),
    ];
    let out = build(&settings(true), &clusters, &mentions);
    let all: Vec<_> = out.clusters.iter().flat_map(|c| c.member_mention_ids.iter()).collect();
    let unique: HashSet<_> = all.iter().collect();
    assert_eq!(all.len(), unique.len());
    // The larger cluster wins both Lopez mentions; the emptied one is dropped.
    assert_eq!(out.clusters.len(), 1);
    assert_eq!(out.clusters[0].id, "cluster_person_2");
}

#[test]
fn exclusive_membership_ties_go_to_the_earliest_cluster() {
    let mentions = vec![
        mention("ent_001", "PER", "John", 0, 4),
        mention("ent_002", "PER", "Smith", 50, 55),
        mention("ent_003", "PER", "Alice", 100, 105),
    ];
    let clusters = vec![
        cluster("cluster_A", vec![span(0, 4, "John"), span(50, 55, "Smith")]),
        cluster("cluster_B", vec![span(0, 4, "John"), span(100, 105, "Alice")]),
    ];
    let first = build(&settings(true), &clusters, &mentions);
    let second = build(&settings(true), &clusters, &mentions);
    assert_eq!(first, second);
    let holding: Vec<&str> = first
        .clusters
        .iter()
        .filter(|c| c.member_mention_ids.iter().any(|id| id.as_str() == "ent_001"))
        .map(|c| c.id.as_str())
        .collect();
    assert_eq!(holding, vec!["cluster_A"]);
}

#[test]
fn superset_wins_over_a_low_jaccard_subset() {
    let mentions = vec![
        mention("ent_001", "ORG", "Acme Corp", 0, 9),
        mention("ent_002", "ORG", "Acme", 50, 54),
        mention("ent_003", "ORG", "Corp", 100, 104),
    ];
    let clusters = vec![
        cluster("cluster_1", vec![span(0, 9, "Acme Corp"), span(50, 54, "Acme")]),
        cluster(
            "cluster_2",
            vec![span(0, 9, "Acme Corp"), span(50, 54, "Acme"), span(100, 104, "Corp")],
        ),
    ];
    let out = build(&settings(true), &clusters, &mentions);
    assert_eq!(out.clusters.len(), 1);
    assert_eq!(out.clusters[0].member_mention_ids.len(), 3);
}

/// Exclusive membership runs before deduplication, so clusters sharing mentions end up
/// disjoint: the shared mentions move to the bigger cluster and the other keeps its own.
#[test]
fn shared_members_move_to_the_bigger_cluster() {
    let mentions: Vec<Mention> = (0..6)
        .map(|i| mention(&format!("m{i}"), "PER", &format!("Name{i}"), i * 10, i * 10 + 5))
        .collect();
    let spans = |ids: &[usize]| -> Vec<ClusterSpan> {
        ids.iter().map(|i| span(i * 10, i * 10 + 5, &format!("Name{i}"))).collect()
    };
    let clusters = vec![cluster("a", spans(&[0, 1, 2, 3])), cluster("b", spans(&[2, 3, 4, 5, 1]))];
    let out = build(&settings(true), &clusters, &mentions);
    let members: Vec<Vec<&str>> = out
        .clusters
        .iter()
        .map(|c| c.member_mention_ids.iter().map(MentionId::as_str).collect())
        .collect();
    assert_eq!(members, vec![vec!["m2", "m3", "m4", "m5", "m1"]]);
}

#[test]
fn span_mapping_respects_overlap_and_eligibility() {
    let settings =
        CorefSettings { eligible_types: vec!["PER".to_owned()], ..CorefSettings::default() };
    let mentions = vec![
        mention("lopez", "PER", "Maria Lopez", 0, 11),
        mention("car", "PRODUCT", "car", 20, 23),
        mention("reed", "PER", "Reed", 30, 34),
        mention("reed2", "PER", "Reed", 40, 44),
    ];
    let roles = RoleConfig::default();
    let builder = ClusterBuilder::new(&settings, &roles);
    let index: Vec<&Mention> = mentions.iter().collect();
    // "Lopez" covers 5/5 of its span but 5/11 of the mention: quality 1.0.
    assert_eq!(builder.overlapping_mentions(6, 11, &index)[0].id.as_str(), "lopez");
    // "Mar" covers 3/11 of the mention and 3/3 of the span.
    assert_eq!(builder.overlapping_mentions(0, 3, &index).len(), 1);
    // A span half outside "Reed": 2/4 of the mention, 2/6 of the span -> 0.5 passes.
    assert_eq!(builder.overlapping_mentions(32, 38, &index).len(), 1);
    // 1/4 and 1/5: below 0.5.
    assert!(builder.overlapping_mentions(33, 38, &index).is_empty());

    // The ineligible "car" is mapped and then removed; the unmapped pronoun is kept.
    let clusters = vec![cluster(
        "c",
        vec![
            span(0, 11, "Maria Lopez"),
            span(20, 23, "car"),
            span(30, 34, "Reed"),
            span(50, 53, "her"),
        ],
    )];
    let out = build(&settings, &clusters, &mentions);
    assert_eq!(out.clusters.len(), 1);
    let ids: Vec<&str> = out.clusters[0].member_mention_ids.iter().map(MentionId::as_str).collect();
    assert_eq!(ids, vec!["lopez", "reed"]);
    assert_eq!(out.clusters[0].pronoun_spans, vec![Span::new(50, 53)]);
    assert_eq!(out.canonical_text_for_mention["reed"], "Maria Lopez");
}

#[test]
fn a_duplicate_cluster_loses_its_pronoun_spans() {
    let mentions = vec![mention("a", "PER", "Ann", 0, 3), mention("b", "PER", "Ann Lee", 10, 17)];
    let clusters = vec![
        cluster("x", vec![span(0, 3, "Ann"), span(10, 17, "Ann Lee")]),
        cluster("y", vec![span(10, 17, "Ann Lee"), span(0, 3, "Ann"), span(30, 33, "she")]),
    ];
    let out = build(&settings(true), &clusters, &mentions);
    assert_eq!(out.clusters.len(), 1);
    assert_eq!(out.clusters[0].id, "x");
    assert!(out.clusters[0].pronoun_spans.is_empty());
}
