//! End-to-end post-processing: labelling, mapping, canonical selection and pronoun mention
//! synthesis.

use ce_core::{Chunk, ChunkWriteResult, DocId, Mention, NodeId, Span};
use ce_coref::{CorefPostProcessor, CorefSettings, RoleConfig, PRONOUN_SOURCE};
use ce_models::CorefCluster;
use pretty_assertions::assert_eq;
use serde_json::json;

const TEXT: &str = "Maria Lopez drove. She stopped. Lopez said her car broke.";

fn mentions() -> Vec<Mention> {
    let mut lopez = Mention::new("m1", "PER", "Maria Lopez", Span::new(0, 11));
    lopez.score = Some(0.9);
    let short = Mention::new("m2", "PER", "Lopez", Span::new(32, 37));
    vec![lopez, short]
}

fn chunks() -> Vec<ChunkWriteResult> {
    let chunk = |seq: usize, start: usize, end: usize| ChunkWriteResult {
        chunk: Chunk::new("doc1", &TEXT[start..end], seq, start, end),
        helix_id: NodeId::new(format!("h{seq}")),
        created: true,
    };
    vec![chunk(1, 32, 57), chunk(0, 0, 31)]
}

fn raw() -> Vec<CorefCluster> {
    vec![
        CorefCluster { spans: vec![(0, 11), (19, 22), (32, 37), (43, 46)] },
        CorefCluster { spans: vec![(47, 50)] },
    ]
}

fn processor(settings: CorefSettings) -> CorefPostProcessor {
    CorefPostProcessor::new(settings, RoleConfig::default())
}

#[test]
fn clusters_canonical_mentions_and_pronoun_mentions() {
    let out = processor(CorefSettings::default())
        .process(&DocId::new("doc1"), TEXT, &raw(), &mentions(), &chunks())
        .expect("a cluster survives");
    assert_eq!(out.clusters.len(), 1);
    let cluster = &out.clusters[0];
    assert_eq!(cluster.id, "doc1_cluster_0");
    assert_eq!(
        cluster.member_mention_ids,
        vec![ce_core::MentionId::new("m1"), ce_core::MentionId::new("m2")]
    );
    assert_eq!(cluster.pronoun_spans, vec![Span::new(19, 22), Span::new(43, 46)]);
    assert_eq!(out.canonical_mention_for_cluster["doc1_cluster_0"].as_str(), "m1");

    let ids: Vec<&str> = out.synthetic_mentions.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(
        ids,
        vec!["synth_pronoun_doc1_cluster_0_19_22", "synth_pronoun_doc1_cluster_0_43_46"]
    );
    let first = serde_json::to_value(&out.synthetic_mentions[0]).expect("serialises");
    assert_eq!(
        first,
        json!({
            "id": "synth_pronoun_doc1_cluster_0_19_22",
            "type": "PER",
            "name": "Maria Lopez",
            "span": {"start": 19, "end": 22},
            "score": 0.85,
            "source": PRONOUN_SOURCE,
            "sources": [],
            "metadata": {
                "synthetic_pronoun": true,
                "original_pronoun_text": "She",
                "cluster_id": "doc1_cluster_0",
                "canonical_mention_id": "m1",
                "canonical_name": "Maria Lopez"
            },
            "doc_id": "doc1",
            "chunk_id": "doc1_0",
            "sentence_index": null
        })
    );
    assert_eq!(
        out.synthetic_mentions[1].chunk_id.as_ref().map(ce_core::ChunkId::as_str),
        Some("doc1_1")
    );

    let by_chunk: Vec<(&str, usize)> =
        out.synthetic_by_chunk.iter().map(|(k, v)| (k.as_str(), v.len())).collect();
    assert_eq!(by_chunk, vec![("h0", 1), ("h1", 1)]);

    let synthetic = out.synthetic_pronoun_mentions.as_ref().expect("synthetic map");
    assert_eq!(synthetic["synth_pronoun_doc1_cluster_0_43_46"].as_str(), "m1");
    let canonical_text: Vec<(&str, &str)> =
        out.canonical_text_for_mention.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    assert_eq!(
        canonical_text,
        vec![
            ("m1", "Maria Lopez"),
            ("m2", "Maria Lopez"),
            ("synth_pronoun_doc1_cluster_0_19_22", "Maria Lopez"),
            ("synth_pronoun_doc1_cluster_0_43_46", "Maria Lopez"),
        ]
    );
}

#[test]
fn synthesis_can_be_disabled() {
    let settings = CorefSettings { synthesize_pronoun_mentions: false, ..CorefSettings::default() };
    let out = processor(settings)
        .process(&DocId::new("doc1"), TEXT, &raw(), &mentions(), &chunks())
        .expect("a cluster survives");
    assert!(out.synthetic_mentions.is_empty());
    assert!(out.synthetic_pronoun_mentions.is_none());
    assert_eq!(out.canonical_text_for_mention.len(), 2);
    // Pronoun spans stay on the cluster.
    assert_eq!(out.clusters[0].pronoun_spans.len(), 2);
}

#[test]
fn nothing_survives_without_two_mapped_mentions() {
    let doc = DocId::new("doc1");
    let p = processor(CorefSettings::default());
    assert!(p.process(&doc, TEXT, &[], &mentions(), &chunks()).is_none());
    let pronoun_only = [CorefCluster { spans: vec![(19, 22), (43, 46)] }];
    assert!(p.process(&doc, TEXT, &pronoun_only, &mentions(), &chunks()).is_none());
    let one_member = [CorefCluster { spans: vec![(0, 11), (19, 22)] }];
    assert!(p.process(&doc, TEXT, &one_member, &mentions(), &chunks()).is_none());
}

#[test]
fn pronouns_outside_every_chunk_have_no_chunk() {
    let out = processor(CorefSettings::default())
        .process(&DocId::new("doc1"), TEXT, &raw(), &mentions(), &chunks()[..1])
        .expect("a cluster survives");
    assert_eq!(out.synthetic_mentions[0].chunk_id, None);
    assert_eq!(
        out.synthetic_by_chunk.keys().map(ce_core::ChunkId::as_str).collect::<Vec<_>>(),
        vec!["h1"]
    );
}

#[test]
fn max_doc_chars_counts_characters() {
    let settings = CorefSettings { max_doc_chars: Some(3), ..CorefSettings::default() };
    let p = processor(settings);
    assert!(!p.exceeds_max_doc_chars("Zoë"));
    assert!(p.exceeds_max_doc_chars("Zoë!"));
    assert!(!processor(CorefSettings::default()).exceeds_max_doc_chars(TEXT));
}
