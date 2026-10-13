//! Parsing the checked-in contradiction keys, compared with the Python parser's output
//! (`fixtures/input/eval/keys_parsed_expected.json`).

mod common;

use impeach_eval::{parse_key_file, Key};
use serde_json::{json, Value};

fn key(name: &str) -> Key {
    parse_key_file(&common::workspace_root().join("fixtures").join("keys").join(name))
        .expect("key fixture parses")
}

fn as_json(key: &Key) -> Value {
    Value::Array(
        key.contradictions
            .iter()
            .map(|g| {
                json!({
                    "id": g.id,
                    "contradiction_type": g.contradiction_type,
                    "fact_a": {"text": g.fact_a.text, "source": g.fact_a.source, "paragraph_ref": g.fact_a.paragraph_ref},
                    "fact_b": {"text": g.fact_b.text, "source": g.fact_b.source, "paragraph_ref": g.fact_b.paragraph_ref},
                    "notes": g.notes,
                    "is_inter_doc": g.is_inter_doc,
                })
            })
            .collect(),
    )
}

fn expected(name: &str) -> Value {
    let all: Value = serde_json::from_str(&common::read_fixture(
        "fixtures/input/eval/keys_parsed_expected.json",
    ))
    .expect("parsed-key fixture is JSON");
    all[name].clone()
}

#[test]
fn green_valley_key() {
    let gts = key("GREEN_VALLEY_CONTRA_KEY.md");
    assert_eq!(gts.len(), 19);
    assert_eq!(gts.inter_doc().count(), 15);
    assert_eq!(gts.intra_doc().count(), 4);
    let c01 = &gts.contradictions[0];
    assert_eq!(c01.id, "C-01");
    assert_eq!(c01.fact_a.text, "Collision date: March 5, 2023");
    assert_eq!(c01.fact_b.text, "Collision date: March 4, 2023");
    let sources: Vec<&str> = gts.intra_doc().map(|g| g.fact_a.source.as_str()).collect();
    assert_eq!(sources, ["Doc 1", "Doc 1", "Doc 2", "Doc 2"]);
    pretty_assertions::assert_eq!(as_json(&gts), expected("GREEN_VALLEY_CONTRA_KEY.md"));
}

#[test]
fn mason_key_strips_entity_refs() {
    let gts = key("MASON_KEY.md");
    let ids: Vec<&str> = gts.contradictions.iter().map(|g| g.id.as_str()).collect();
    assert_eq!(ids, ["C-01", "C-02", "C-03", "C-04", "C-05", "C-06"]);
    assert_eq!(gts.intra_doc().count(), 0);
    assert_eq!(gts.contradictions[0].fact_a.text, "Eastvale, California");
    assert_eq!(gts.contradictions[5].fact_b.text, "Total hospital expenses $300");
    pretty_assertions::assert_eq!(as_json(&gts), expected("MASON_KEY.md"));
}
