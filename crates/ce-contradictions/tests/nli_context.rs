//! NliContextBuilder and build_entity_name_map.

mod common_b;

use ce_contradictions::{build_entity_name_map, NliContextBuilder};
use ce_core::{EdCandidate, EdResult, EntityId, RetrievedFact, TemporalValidity};
use common_b::{fact, retrieved};
use indexmap::IndexMap;
use pretty_assertions::assert_eq;
use serde_json::json;

fn names(pairs: &[(&str, &str)]) -> IndexMap<EntityId, String> {
    pairs.iter().map(|(id, name)| (EntityId::new(*id), (*name).to_owned())).collect()
}

fn validity(from: Option<&str>, until: Option<&str>) -> TemporalValidity {
    TemporalValidity {
        t_valid_from: from.map(str::to_owned),
        t_valid_until: until.map(str::to_owned),
    }
}

#[test]
fn extracted_fact_with_all_context() {
    let builder = NliContextBuilder::new(names(&[("e1", "Maria Lopez"), ("e2", "Green Valley")]));
    let mut f = fact("f1", "She was home by 8pm", "she", "WAS_AT", "home");
    f.subject_entity_id = Some("e1".into());
    f.object_entity_id = Some("e2".into());
    f.temporal_validity = validity(Some("2020-03-15"), Some("2020-03-16"));
    f.metadata.insert("speaker".into(), json!("Maria Lopez"));
    assert_eq!(
        builder.build_enriched_text(&f),
        "Subject: Maria Lopez | Object: Green Valley | Time: 2020-03-15 to 2020-03-16 | \
         Speaker: Maria Lopez | She was home by 8pm"
    );
}

#[test]
fn extracted_fact_falls_back_to_trimmed_triple_names() {
    let builder = NliContextBuilder::new(names(&[("e1", "Maria Lopez")]));
    let mut f = fact("f1", "Text.", "  the driver ", "DROVE", " \t ");
    // An id missing from the map falls back to the triple subject.
    f.subject_entity_id = Some("unknown".into());
    assert_eq!(builder.build_enriched_text(&f), "Subject: the driver | Text.");
}

#[test]
fn known_entity_with_empty_name_is_skipped_without_fallback() {
    let builder = NliContextBuilder::new(names(&[("e1", "")]));
    let mut f = fact("f1", "Text.", "Maria", "R", "Bob");
    f.subject_entity_id = Some("e1".into());
    assert_eq!(builder.build_enriched_text(&f), "Object: Bob | Text.");
}

#[test]
fn extracted_fact_time_variants() {
    let builder = NliContextBuilder::default();
    let mut f = fact("f1", "Text.", "", "R", "");
    f.temporal_validity = validity(None, Some("2021"));
    assert_eq!(builder.build_enriched_text(&f), "Time: until 2021 | Text.");

    f.temporal_validity = validity(Some("2019"), Some(""));
    assert_eq!(builder.build_enriched_text(&f), "Time: 2019 | Text.");

    // Without a validity interval: metadata `date`, else `event_date`, rendered like str().
    f.temporal_validity = validity(Some(""), None);
    f.metadata.insert("date".into(), json!(""));
    f.metadata.insert("event_date".into(), json!(2020));
    assert_eq!(builder.build_enriched_text(&f), "Time: 2020 | Text.");

    f.metadata.insert("date".into(), json!("March 5"));
    assert_eq!(builder.build_enriched_text(&f), "Time: March 5 | Text.");

    // A validity interval wins over metadata.
    f.temporal_validity = validity(Some("2018"), None);
    assert_eq!(builder.build_enriched_text(&f), "Time: 2018 | Text.");
}

#[test]
fn falsy_metadata_values_are_skipped() {
    let builder = NliContextBuilder::default();
    let mut f = fact("f1", "Text.", "", "R", "");
    for value in [json!(null), json!(false), json!(""), json!(0), json!([])] {
        f.metadata.insert("speaker".into(), value.clone());
        f.metadata.insert("date".into(), value);
        assert_eq!(builder.build_enriched_text(&f), "Text.");
    }
    f.metadata.insert("speaker".into(), json!(true));
    assert_eq!(builder.build_enriched_text(&f), "Speaker: True | Text.");
}

#[test]
fn bare_fact_is_its_text() {
    let builder = NliContextBuilder::default();
    assert_eq!(builder.build_enriched_text(&fact("f1", "Only text", "", "R", "")), "Only text");
    assert_eq!(
        builder.build_enriched_text_for_retrieved(&retrieved("r1", "Only text", 1.0)),
        "Only text"
    );
}

fn retrieved_with(subject: Option<&str>, object: Option<&str>) -> RetrievedFact {
    let mut r = retrieved("r1", "The light was red.", 1.0);
    r.assertion_subject = subject.map(str::to_owned);
    r.assertion_object = object.map(str::to_owned);
    r
}

#[test]
fn retrieved_fact_with_all_context() {
    let builder = NliContextBuilder::new(names(&[("e1", "Ignored")]));
    let mut r = retrieved_with(Some("Traffic light"), Some("red"));
    r.metadata.insert(
        "TemporalValidity".into(),
        json!({"t_valid_from": "2020-03-15", "t_valid_until": "2020-03-16"}),
    );
    r.metadata.insert("speaker".into(), json!("Officer Smith"));
    assert_eq!(
        builder.build_enriched_text_for_retrieved(&r),
        "Subject: Traffic light | Object: red | Time: 2020-03-15 to 2020-03-16 | \
         Speaker: Officer Smith | The light was red."
    );
}

#[test]
fn retrieved_fact_names_are_not_trimmed_and_empty_ones_skipped() {
    let builder = NliContextBuilder::default();
    let r = retrieved_with(Some(" Bob "), Some(""));
    assert_eq!(
        builder.build_enriched_text_for_retrieved(&r),
        "Subject:  Bob  | The light was red."
    );
}

#[test]
fn retrieved_fact_time_variants() {
    let builder = NliContextBuilder::default();
    let mut r = retrieved_with(None, None);

    // Python prefixes the end with "to" even without a start.
    r.metadata.insert("TemporalValidity".into(), json!({"t_valid_until": "2021"}));
    assert_eq!(builder.build_enriched_text_for_retrieved(&r), "Time: to 2021 | The light was red.");

    // An empty validity object falls back to `date` / `event_date`.
    r.metadata.insert("TemporalValidity".into(), json!({"t_valid_from": null}));
    r.metadata.insert("event_date".into(), json!(2019.5));
    assert_eq!(builder.build_enriched_text_for_retrieved(&r), "Time: 2019.5 | The light was red.");

    // A validity stored as a string is not read.
    r.metadata.insert("TemporalValidity".into(), json!("2020"));
    r.metadata.insert("date".into(), json!("2018-01-01"));
    assert_eq!(
        builder.build_enriched_text_for_retrieved(&r),
        "Time: 2018-01-01 | The light was red."
    );
}

fn candidate(id: &str, name: &str) -> EdCandidate {
    serde_json::from_value(json!({"entity_id": id, "canonical_name": name, "label": "PERSON"}))
        .expect("candidate deserialises")
}

fn ed_result(chosen: Option<&str>, candidates: Vec<EdCandidate>) -> EdResult {
    EdResult {
        mention_id: None,
        chosen_entity_id: chosen.map(EntityId::new),
        candidates,
        resolved_by_llm: false,
        confidence: 1.0,
        strategy_used: "rule".to_owned(),
    }
}

#[test]
fn entity_name_map_from_ed_results() {
    let results = vec![
        ed_result(Some("e1"), vec![candidate("e0", "Other"), candidate("e1", "Maria Lopez")]),
        ed_result(None, vec![candidate("e9", "Unchosen")]),
        ed_result(Some("e2"), vec![candidate("e3", "No match")]),
        ed_result(
            Some("e4"),
            vec![candidate("e4", ""), candidate("e4", "Bob"), candidate("e4", "Rob")],
        ),
        ed_result(Some(""), vec![candidate("", "Empty id")]),
        ed_result(Some("e1"), vec![candidate("e1", "M. Lopez")]),
    ];
    let map = build_entity_name_map(&results);
    let entries: Vec<(&str, &str)> = map.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    // Later results overwrite earlier names but keep the first insertion position.
    assert_eq!(entries, vec![("e1", "M. Lopez"), ("e4", "Bob")]);
    assert!(build_entity_name_map(&[]).is_empty());
}

#[test]
fn entity_name_map_feeds_the_builder() {
    let map = build_entity_name_map(&[ed_result(Some("e1"), vec![candidate("e1", "Maria Lopez")])]);
    let builder = NliContextBuilder::new(map);
    let mut f = fact("f1", "She testified.", "She", "TESTIFIED", "");
    f.subject_entity_id = Some("e1".into());
    assert_eq!(builder.build_enriched_text(&f), "Subject: Maria Lopez | She testified.");
    assert_eq!(builder.entity_names().len(), 1);
}
