//! Compares impeach-text against reference values produced by the Python libraries.
//!
//! `tests/data/fuzzy_reference.json` holds `rapidfuzz` 3.14 (`fuzz.ratio`, `partial_ratio`,
//! `token_sort_ratio`, `token_set_ratio`) and `difflib.SequenceMatcher(None, a, b).ratio()`
//! values (with and without autojunk) for hand-picked and random string pairs.
//! `tests/data/values_reference.json` holds `extract_values` output of the Python evaluator
//! (`tests/e2e/evaluate_output.py`) for hand-picked and random texts.

use std::collections::BTreeSet;

use impeach_text::fuzzy::{
    partial_ratio, ratio, sequence_matcher_ratio, sequence_matcher_ratio_with, token_set_ratio,
    token_sort_ratio,
};
use impeach_text::values::{extract_values, Number, Value};
use serde_json::Value as Json;

fn load(json: &str) -> Vec<Json> {
    match serde_json::from_str(json).expect("reference file is valid JSON") {
        Json::Array(items) => items,
        other => panic!("expected an array, got {other}"),
    }
}

fn field<'a>(case: &'a Json, key: &str) -> &'a Json {
    case.get(key).unwrap_or_else(|| panic!("missing {key} in {case}"))
}

fn text<'a>(case: &'a Json, key: &str) -> &'a str {
    field(case, key).as_str().unwrap_or_else(|| panic!("{key} is not a string in {case}"))
}

fn float(case: &Json, key: &str) -> f64 {
    field(case, key).as_f64().unwrap_or_else(|| panic!("{key} is not a number in {case}"))
}

#[test]
fn fuzzy_scores_match_rapidfuzz_and_difflib() {
    let cases = load(include_str!("data/fuzzy_reference.json"));
    assert!(cases.len() > 300);
    type Scorer = fn(&str, &str) -> f64;
    let scorers: [(&str, Scorer); 6] = [
        ("ratio", ratio),
        ("partial_ratio", partial_ratio),
        ("token_sort_ratio", token_sort_ratio),
        ("token_set_ratio", token_set_ratio),
        ("sequence_matcher", sequence_matcher_ratio),
        ("sequence_matcher_noautojunk", |a, b| sequence_matcher_ratio_with(a, b, false)),
    ];
    let mut failures = Vec::new();
    for case in &cases {
        let (a, b) = (text(case, "a"), text(case, "b"));
        for (name, scorer) in scorers {
            let expected = float(case, name);
            let actual = scorer(a, b);
            if (actual - expected).abs() > 1e-9 {
                failures.push(format!("{name}({a:?}, {b:?}) = {actual}, expected {expected}"));
            }
        }
    }
    assert!(failures.is_empty(), "{} mismatches:\n{}", failures.len(), failures.join("\n"));
}

fn opt_u32(item: &Json) -> Option<u32> {
    item.as_u64().map(|v| u32::try_from(v).expect("reference integers fit u32"))
}

fn req_u32(item: &Json) -> u32 {
    opt_u32(item).unwrap_or_else(|| panic!("expected an integer, got {item}"))
}

fn parse_value(item: &Json) -> Value {
    let parts = item.as_array().expect("value tuple is an array");
    match parts[0].as_str().expect("value kind is a string") {
        "date" => Value::Date {
            year: opt_u32(&parts[1]),
            month: req_u32(&parts[2]),
            day: opt_u32(&parts[3]),
        },
        "year" => Value::Year(req_u32(&parts[1])),
        "time" => Value::Time {
            hour: req_u32(&parts[1]),
            minute: req_u32(&parts[2]),
            meridiem: parts[3].as_str().and_then(|s| s.chars().next()),
        },
        "num" => Value::Num(Number::from(parts[1].as_f64().expect("num value is a number"))),
        other => panic!("unknown value kind {other}"),
    }
}

#[test]
fn extracted_values_match_python_evaluator() {
    let cases = load(include_str!("data/values_reference.json"));
    assert!(cases.len() > 300);
    let mut failures = Vec::new();
    for case in &cases {
        let input = text(case, "text");
        let expected: BTreeSet<Value> = field(case, "values")
            .as_array()
            .expect("values is an array")
            .iter()
            .map(parse_value)
            .collect();
        let actual = extract_values(input);
        if actual != expected {
            failures.push(format!("{input:?}: got {actual:?}, expected {expected:?}"));
        }
    }
    assert!(failures.is_empty(), "{} mismatches:\n{}", failures.len(), failures.join("\n"));
}
