//! Fact extractor tests: prompt files, rendering, LLM output parsing and repair, single-pass and
//! multi-pass extraction, and the 64+ facts regression.

use std::path::PathBuf;
use std::sync::Arc;

use ce_config::prompts::{ChatMessage, ExtractionPrompt};
use ce_config::MappingsRegistry;
use ce_core::{EntityId, ExtractedFact, TemporalContext, TimeWindow};
use ce_facts::{
    build_entity_context, create_extractor, deduplicate_facts, load_prompts, parse_facts,
    render_prompt, FactError, FactExtractor, LlmFactExtractor, MultiPassFactExtractor,
    PromptEntity,
};
use ce_models::{Llm, MockLlm, ModelError};
use pretty_assertions::assert_eq;
use serde_json::json;

const SINGLE_PASS: &str =
    "contradiction_engine/prompts/processing/singlepass/processing_single_v3.json";
const MULTI_PASS: [&str; 4] = [
    "contradiction_engine/prompts/processing/multipass/processing_mullti_v2/pass1_incident.json",
    "contradiction_engine/prompts/processing/multipass/processing_mullti_v2/pass2_party.json",
    "contradiction_engine/prompts/processing/multipass/processing_mullti_v2/pass3_history.json",
    "contradiction_engine/prompts/processing/multipass/processing_mullti_v2/pass4_damages.json",
];
/// A distinctive phrase of each multi-pass user template, in pass order.
const PASS_MARKERS: [&str; 4] =
    ["incident core facts", "party state facts", "history and employment", "damages and financial"];
const PASS_NAMES: [&str; 4] =
    ["incident_core", "party_state", "history_employment", "damages_expenses"];

const CHUNK: &str = "On March 4, 2023, at approximately 9:20 p.m., a collision occurred at the \
                     intersection of Mission Street and 14th Avenue involving Maria Lopez.";

fn fixture(name: &str) -> String {
    let path: PathBuf =
        [env!("CARGO_MANIFEST_DIR"), "..", "..", "fixtures", "llm", "facts", name].iter().collect();
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn strings(paths: &[&str]) -> Vec<String> {
    paths.iter().map(|p| (*p).to_owned()).collect()
}

fn single_prompt() -> ExtractionPrompt {
    load_prompts(&strings(&[SINGLE_PASS])).expect("single-pass prompt").remove(0)
}

fn multi_prompts() -> Vec<ExtractionPrompt> {
    load_prompts(&strings(&MULTI_PASS)).expect("multi-pass prompts")
}

fn entities() -> Vec<PromptEntity> {
    vec![
        PromptEntity {
            id: EntityId::new("ent-lopez"),
            name: "Maria Lopez".to_owned(),
            entity_type: Some("PER".to_owned()),
            aliases: vec!["Lopez".to_owned(), "Plaintiff".to_owned()],
        },
        PromptEntity {
            id: EntityId::new("ent-gv"),
            name: "Green Valley Logistics, Inc.".to_owned(),
            entity_type: Some("ORG".to_owned()),
            aliases: Vec::new(),
        },
    ]
}

fn ids(facts: &[ExtractedFact]) -> Vec<String> {
    facts.iter().map(|f| f.id.as_str().to_owned()).collect()
}

fn parse(response: &str) -> Vec<ExtractedFact> {
    parse_facts(response, None).expect("response parses")
}

/// One single-pass (v3) node with nested objects.
fn v3_node(index: usize, tag: &str) -> String {
    json!({
        "node_type": "STATED_FACT",
        "id": format!("STATED_FACT_{index}"),
        "text": format!("{tag} fact number {index} about the collision."),
        "value_type": "NUMERIC",
        "property_type": "QUANTITATIVE",
        "slot_name": "count",
        "value_raw": format!("{index}"),
        "assertion_triple": {
            "subject": format!("{tag} subject {index}"),
            "relationship": "had_count",
            "object": format!("{index}")
        },
        "certainty": "0.9",
        "temporal_context": "current_incident",
        "is_denial": false,
        "is_boilerplate": false,
        "temporal_validity": {"t_valid_from": null, "t_valid_until": null},
        "metadata": {}
    })
    .to_string()
}

fn v3_response(count: usize, tag: &str) -> String {
    let nodes: Vec<String> = (0..count).map(|i| v3_node(i, tag)).collect();
    format!("{{\"nodes\": [\n{}\n], \"edges\": []}}", nodes.join(",\n"))
}

fn pass_index(prompt: &str) -> usize {
    PASS_MARKERS.iter().position(|m| prompt.contains(m)).expect("prompt of a known pass")
}

// ---------------------------------------------------------------------------------------------
// Prompt files and rendering
// ---------------------------------------------------------------------------------------------

#[test]
fn every_prompt_file_loads_and_renders() {
    let mut all = vec![SINGLE_PASS];
    all.extend(MULTI_PASS);
    let prompts = load_prompts(&strings(&all)).expect("prompts load");
    let context = build_entity_context(&entities());
    for (path, prompt) in all.iter().zip(&prompts) {
        let template = prompt.combined_template();
        assert_eq!(template.matches("{segments}").count(), 1, "{path}");
        assert_eq!(template.matches("{resolved_entities}").count(), 1, "{path}");
        let rendered = render_prompt(&template, CHUNK, &context);
        assert!(!rendered.contains("{segments}"), "{path}");
        assert!(!rendered.contains("{resolved_entities}"), "{path}");
        assert!(rendered.contains(CHUNK), "{path}");
        assert!(rendered.contains(&context), "{path}");
        // Only the two placeholders lose their braces; the JSON examples are intact.
        assert_eq!(rendered.matches('{').count(), template.matches('{').count() - 2, "{path}");
        assert_eq!(rendered.matches('}').count(), template.matches('}').count() - 2, "{path}");
        assert!(rendered.contains("\"node_type\": \"STATED_FACT\""), "{path}");
        assert!(rendered.ends_with("Return JSON:"), "{path}");
    }
    let names: Vec<Option<&str>> = prompts.iter().map(|p| p.pass_name.as_deref()).collect();
    assert_eq!(
        names,
        [
            None,
            Some("incident_core"),
            Some("party_state"),
            Some("history_employment"),
            Some("damages_expenses")
        ]
    );
}

#[test]
fn missing_prompt_file_is_a_config_error() {
    let err = load_prompts(&strings(&[SINGLE_PASS, "contradiction_engine/prompts/nope.json"]))
        .expect_err("missing file");
    assert!(matches!(err, FactError::Config(_)), "{err:?}");
}

#[test]
fn entity_context_block() {
    assert_eq!(
        build_entity_context(&entities()),
        "Resolved entities in this chunk:\n\
         - Maria Lopez (ID: ent-lopez, TYPE: PER, Aliases: \"Lopez\", \"Plaintiff\")\n\
         - Green Valley Logistics, Inc. (ID: ent-gv, TYPE: ORG)"
    );
    assert_eq!(build_entity_context(&[]), "");
}

#[test]
fn no_entities_leave_an_empty_block() {
    let template = single_prompt().combined_template();
    let rendered = render_prompt(&template, CHUNK, "");
    assert!(rendered.contains(&format!("and denials.\n\n\n\n{CHUNK}\n\nReturn JSON:")));
}

// ---------------------------------------------------------------------------------------------
// Single-pass extraction with canned responses
// ---------------------------------------------------------------------------------------------

#[test]
fn single_pass_parses_canned_v3_response() {
    let llm = Arc::new(MockLlm::with_responses([fixture("single_v3_green_valley.txt")]));
    let extractor = LlmFactExtractor::new(llm.clone(), &single_prompt());
    let facts = extractor.extract_facts(CHUNK, &entities());

    let prompts = llm.prompts();
    assert_eq!(prompts.len(), 1);
    assert_eq!(
        prompts[0],
        render_prompt(extractor.template(), CHUNK, &build_entity_context(&entities()))
    );
    assert!(prompts[0].contains(CHUNK));
    assert!(prompts[0]
        .contains("- Maria Lopez (ID: ent-lopez, TYPE: PER, Aliases: \"Lopez\", \"Plaintiff\")"));

    // The ENTITY node is skipped; boilerplate facts are kept (the stage drops them).
    assert_eq!(
        ids(&facts),
        ["STATED_FACT_0", "STATED_FACT_1", "STATED_FACT_2", "STATED_FACT_3", "STATED_FACT_4"]
    );
    let date = &facts[0];
    assert_eq!(date.text, "The collision occurred on March 4, 2023.");
    assert_eq!(date.assertion_triple.subject, "collision");
    assert_eq!(date.assertion_triple.relationship, "occurred_on");
    assert_eq!(date.assertion_triple.object, "March 4, 2023");
    assert!((date.certainty - 0.95).abs() < 1e-12);
    assert_eq!(date.temporal_validity.t_valid_from.as_deref(), Some("2023-03-04"));
    assert_eq!(date.slot_name.as_deref(), Some("incident_date"));
    assert_eq!(date.value_raw.as_deref(), Some("March 4, 2023"));
    assert_eq!(date.temporal_context, TemporalContext::CurrentIncident);
    assert!(!date.is_denial && !date.is_boilerplate);
    assert!(date.metadata.is_empty());
    assert_eq!(date.time_window, None);
    assert_eq!(date.raw_node["slot_name"], json!("incident_date"));

    let speed = &facts[1];
    assert_eq!(speed.value_type.as_deref(), Some("numeric"));
    assert_eq!(speed.property_type.as_deref(), Some("Quantitative"));
    assert!((speed.certainty - 0.9).abs() < 1e-12);
    assert_eq!(speed.metadata.get("source_paragraph"), Some(&json!(5)));

    let denial = &facts[2];
    assert!(denial.is_denial);
    assert_eq!(denial.temporal_context, TemporalContext::PriorIncident);
    assert_eq!(
        denial.time_window,
        Some(TimeWindow {
            start: Some("2013-03-04".to_owned()),
            end: Some("2023-03-04".to_owned())
        })
    );
    assert!(!denial.metadata.contains_key("time_window"));
    // raw_node is the node as received.
    assert!(denial.raw_node["metadata"].get("time_window").is_some());

    let boilerplate = &facts[4];
    assert!(boilerplate.is_boilerplate);
    assert_eq!(boilerplate.temporal_context, TemporalContext::GeneralClaim);
    assert!(facts.iter().all(|f| f.source_chunk_id.is_none() && f.subject_entity_id.is_none()));
}

#[test]
fn parsed_types_normalise_through_the_taxonomy() {
    let taxonomy = &MappingsRegistry::load().expect("mappings").taxonomy;
    let facts = parse(&fixture("single_v3_green_valley.txt"));
    let normalized: Vec<(String, String)> = facts
        .iter()
        .map(|f| {
            (
                f.normalized_value_type(taxonomy).into_inner(),
                f.normalized_property_type(taxonomy).into_inner(),
            )
        })
        .collect();
    let expected = [
        ("TEMPORAL_POINT", "TEMPORAL"),
        ("NUMERIC", "QUANTITATIVE"),
        ("BOOLEAN", "EXISTENCE"),
        ("ENTITY_REF", "RELATION"),
        ("TEXT", "MODALITY"),
    ];
    let expected: Vec<(String, String)> =
        expected.iter().map(|(v, p)| ((*v).to_owned(), (*p).to_owned())).collect();
    assert_eq!(normalized, expected);

    let response = json!({"nodes": [
        {"node_type": "STATED_FACT", "value_type": "temporal_interval", "property_type": "exists"},
        {"node_type": "STATED_FACT", "value_type": "SPEED", "property_type": "INCIDENT_DATE"},
        {"node_type": "STATED_FACT", "value_type": 3, "property_type": ""},
        {"node_type": "STATED_FACT"}
    ]})
    .to_string();
    let facts = parse(&response);
    let normalized: Vec<(String, String)> = facts
        .iter()
        .map(|f| {
            (
                f.normalized_value_type(taxonomy).into_inner(),
                f.normalized_property_type(taxonomy).into_inner(),
            )
        })
        .collect();
    let expected: Vec<(String, String)> = [
        ("TEMPORAL_INTERVAL", "EXISTENCE"),
        ("TEXT", "ATTRIBUTE"),
        ("TEXT", "ATTRIBUTE"),
        ("TEXT", "ATTRIBUTE"),
    ]
    .iter()
    .map(|(v, p)| ((*v).to_owned(), (*p).to_owned()))
    .collect();
    assert_eq!(normalized, expected);
    // The raw strings stay on the fact.
    assert_eq!(facts[0].property_type.as_deref(), Some("exists"));
    assert_eq!(facts[2].value_type, None);
}

#[test]
fn multipass_format_with_string_encoded_fields() {
    let facts =
        parse_facts(&fixture("multipass_incident_core.json"), Some("incident_core")).expect("ok");
    assert_eq!(
        ids(&facts),
        [
            "incident_core_STATED_FACT_0",
            "incident_core_STATED_FACT_1",
            "incident_core_STATED_FACT_2"
        ]
    );
    assert_eq!(facts[0].assertion_triple.relationship, "OCCURRED_ON");
    assert_eq!(facts[0].temporal_validity.t_valid_until.as_deref(), Some("2023-03-04"));
    assert_eq!(facts[1].temporal_validity.t_valid_from, None);
    assert!((facts[1].certainty - 0.9).abs() < 1e-12);
    assert_eq!(facts[2].time_window, Some(TimeWindow::default()));
    assert_eq!(
        facts[2].metadata.get("time_window_raw"),
        Some(&json!("at the time of the incident"))
    );
    assert_eq!(facts[2].raw_node["fact_category"], json!("WEATHER"));
}

#[test]
fn llm_error_yields_no_facts() {
    let llm = Arc::new(MockLlm::with_responder(|_| Err(ModelError::Inference("down".to_owned()))));
    let extractor = LlmFactExtractor::new(llm.clone(), &single_prompt());
    assert!(extractor.extract_facts(CHUNK, &[]).is_empty());
    assert_eq!(llm.prompts().len(), 1);
}

#[test]
fn unparseable_response_yields_no_facts() {
    let llm = Arc::new(MockLlm::with_responses(["I could not find any facts."]));
    let extractor = LlmFactExtractor::new(llm, &single_prompt());
    assert!(extractor.extract_facts(CHUNK, &[]).is_empty());
}

#[test]
fn empty_chunk_or_template_skips_the_llm() {
    let llm = Arc::new(MockLlm::new());
    let extractor = LlmFactExtractor::new(llm.clone(), &single_prompt());
    assert!(extractor.extract_facts("", &entities()).is_empty());

    let empty = ExtractionPrompt {
        description: None,
        setting: None,
        version: None,
        pass_number: None,
        pass_name: None,
        system_messages: vec![ChatMessage { role: "system".to_owned(), content: "  ".to_owned() }],
        examples: Vec::new(),
        user_prompt_template: String::new(),
    };
    let extractor = LlmFactExtractor::new(llm.clone(), &empty);
    assert_eq!(extractor.template(), "");
    assert!(extractor.extract_facts(CHUNK, &[]).is_empty());

    let none = create_extractor(llm.clone(), &[], true, 4);
    assert!(none.extract_facts(CHUNK, &entities()).is_empty());

    let multi =
        MultiPassFactExtractor::new(&(llm.clone() as Arc<dyn Llm>), &multi_prompts(), true, 4);
    assert!(multi.extract_facts("", &[]).is_empty());
    assert!(llm.prompts().is_empty());
}

#[test]
fn create_extractor_picks_by_prompt_count() {
    let llm: Arc<dyn Llm> = Arc::new(MockLlm::new());
    let single = create_extractor(Arc::clone(&llm), &[single_prompt()], true, 4);
    assert!(format!("{single:?}").starts_with("LlmFactExtractor"));
    let multi = create_extractor(llm, &multi_prompts(), true, 4);
    assert!(format!("{multi:?}").starts_with("MultiPassFactExtractor"));
}

// ---------------------------------------------------------------------------------------------
// Malformed output
// ---------------------------------------------------------------------------------------------

const ONE_NODE: &str = r#"{"nodes": [{"node_type": "STATED_FACT", "id": "F1", "text": "A was red", "assertion_triple": {"subject": "A", "relationship": "had_color", "object": "red"}}]}"#;

#[test]
fn code_fences_and_prose_are_tolerated() {
    let wrapped = [
        format!("```json\n{ONE_NODE}\n```"),
        format!("```\n{ONE_NODE}\n```"),
        format!("Here are the facts:\n{ONE_NODE}\nLet me know if you need more."),
        format!("Sure.\n```json\n{ONE_NODE}\n```\nDone."),
        format!("  \n{ONE_NODE}\n\n"),
    ];
    for response in &wrapped {
        let facts = parse(response);
        assert_eq!(ids(&facts), ["F1"], "{response}");
        assert_eq!(facts[0].assertion_triple.object, "red");
    }
}

#[test]
fn trailing_commas_and_raw_newlines_are_repaired() {
    let response = "{\"nodes\": [{\"node_type\": \"STATED_FACT\", \"id\": \"F1\", \"text\": \"line one\nline two\", \"assertion_triple\": {\"subject\": \"A\", \"relationship\": \"r\", \"object\": \"b\",},},], \"edges\": [],}";
    let facts = parse(response);
    assert_eq!(ids(&facts), ["F1"]);
    assert_eq!(facts[0].text, "line one\nline two");
    assert_eq!(facts[0].assertion_triple.object, "b");
}

#[test]
fn response_shape_errors() {
    assert!(matches!(parse_facts("[1, 2]", None), Err(FactError::Parse(_))));
    assert!(matches!(
        parse_facts(r#"[{"node_type": "STATED_FACT"}]"#, None),
        Err(FactError::Parse(_))
    ));
    assert!(matches!(parse_facts("no json here", None), Err(FactError::Parse(_))));
    assert!(matches!(parse_facts(r#"{"nodes": {"a": 1}}"#, None), Err(FactError::Parse(_))));
    assert!(matches!(parse_facts(r#"{"nodes": null}"#, None), Err(FactError::Parse(_))));
    assert!(parse(r#"{"edges": []}"#).is_empty());
    assert!(parse(r#"{"nodes": []}"#).is_empty());
}

#[test]
fn non_fact_nodes_are_skipped() {
    let response = json!({"nodes": [
        "a string",
        42,
        null,
        {"node_type": "ENTITY", "id": "E1"},
        {"node_type": "stated_fact", "id": "lower"},
        {"id": "no type"},
        {"node_type": "STATED_FACT", "id": "kept"}
    ]})
    .to_string();
    assert_eq!(ids(&parse(&response)), ["kept"]);
}

#[test]
fn invalid_nested_json_skips_only_that_node() {
    let response = json!({"nodes": [
        {"node_type": "STATED_FACT", "id": "a", "assertion_triple": "{\"subject\": \"A\""},
        {"node_type": "STATED_FACT", "id": "b", "temporal_validity": "not json"},
        {"node_type": "STATED_FACT", "id": "c", "metadata": ""},
        {"node_type": "STATED_FACT", "id": "d",
         "assertion_triple": "{\"subject\": \"D\", \"relationship\": \"\", \"object\": null}",
         "temporal_validity": "{\"t_valid_from\": \"2020\", \"t_valid_until\": 5}",
         "metadata": "{\"note\": \"x\"}"}
    ]})
    .to_string();
    let facts = parse(&response);
    assert_eq!(ids(&facts), ["d"]);
    let d = &facts[0];
    assert_eq!(d.assertion_triple.subject, "D");
    assert_eq!(d.assertion_triple.relationship, "RELATED_TO");
    assert_eq!(d.assertion_triple.object, "");
    assert_eq!(d.temporal_validity.t_valid_from.as_deref(), Some("2020"));
    assert_eq!(d.temporal_validity.t_valid_until, None);
    assert_eq!(d.metadata.get("note"), Some(&json!("x")));
}

#[test]
fn nested_values_that_are_not_objects_read_as_empty() {
    let response = json!({"nodes": [
        {"node_type": "STATED_FACT", "id": "a", "assertion_triple": "[1, 2]",
         "temporal_validity": null, "metadata": 7},
        {"node_type": "STATED_FACT", "id": "b", "assertion_triple": null,
         "temporal_validity": "\"2020\"", "metadata": "null"},
        {"node_type": "STATED_FACT", "id": "c", "assertion_triple": ["x"],
         "metadata": {"k": [1]}}
    ]})
    .to_string();
    let facts = parse(&response);
    assert_eq!(ids(&facts), ["a", "b", "c"]);
    for fact in &facts {
        assert_eq!(fact.assertion_triple.subject, "");
        assert_eq!(fact.assertion_triple.relationship, "RELATED_TO");
        assert_eq!(fact.temporal_validity.t_valid_from, None);
        assert!((fact.certainty - 0.5).abs() < 1e-12);
    }
    assert!(facts[0].metadata.is_empty() && facts[1].metadata.is_empty());
    assert_eq!(facts[2].metadata.get("k"), Some(&json!([1])));
}

#[test]
fn scalar_fields_are_read_leniently() {
    let response = json!({"nodes": [
        {"node_type": "STATED_FACT", "id": "a", "text": 35, "value_raw": 35,
         "slot_name": ["speed"], "value_type": null, "property_type": "QUANTITATIVE",
         "assertion_triple": {"subject": "truck", "relationship": "traveled_at_speed", "object": 35},
         "certainty": 2, "is_denial": "yes", "is_boilerplate": 0, "temporal_context": "PRIOR"},
        {"node_type": "STATED_FACT", "id": "b", "text": null, "value_raw": true,
         "assertion_triple": {"subject": false, "relationship": null, "object": ""},
         "certainty": "0.25", "is_denial": [], "is_boilerplate": "1"}
    ]})
    .to_string();
    let facts = parse(&response);
    let a = &facts[0];
    assert_eq!(a.text, "35");
    assert_eq!(a.value_raw.as_deref(), Some("35"));
    assert_eq!(a.slot_name, None);
    assert_eq!(a.value_type, None);
    assert_eq!(a.property_type.as_deref(), Some("QUANTITATIVE"));
    assert_eq!(a.assertion_triple.object, "35");
    assert!((a.certainty - 1.0).abs() < 1e-12);
    assert!(a.is_denial && !a.is_boilerplate);
    assert_eq!(a.temporal_context, TemporalContext::CurrentIncident);
    let b = &facts[1];
    assert_eq!(b.text, "");
    assert_eq!(b.value_raw, None);
    assert_eq!(b.assertion_triple.subject, "");
    assert_eq!(b.assertion_triple.relationship, "RELATED_TO");
    assert!((b.certainty - 0.25).abs() < 1e-12);
    assert!(!b.is_denial && b.is_boilerplate);
}

#[test]
fn time_window_forms() {
    let response = json!({"nodes": [
        {"node_type": "STATED_FACT", "id": "obj",
         "metadata": {"time_window": {"start": "2010", "end": 2020}, "keep": 1}},
        {"node_type": "STATED_FACT", "id": "str", "metadata": "{\"time_window\": \"past 10 years\"}"},
        {"node_type": "STATED_FACT", "id": "num", "metadata": {"time_window": 10}},
        {"node_type": "STATED_FACT", "id": "nul", "metadata": {"time_window": null}},
        {"node_type": "STATED_FACT", "id": "none", "metadata": {}}
    ]})
    .to_string();
    let facts = parse(&response);
    assert_eq!(
        facts[0].time_window,
        Some(TimeWindow { start: Some("2010".to_owned()), end: None })
    );
    assert_eq!(facts[0].metadata, json!({"keep": 1}).as_object().cloned().expect("object"));
    assert_eq!(facts[1].time_window, Some(TimeWindow::default()));
    assert_eq!(
        facts[1].metadata,
        json!({"time_window_raw": "past 10 years"}).as_object().cloned().expect("object")
    );
    for fact in &facts[2..] {
        assert_eq!(fact.time_window, None, "{}", fact.id);
        assert!(fact.metadata.is_empty(), "{}", fact.id);
    }
}

#[test]
fn missing_and_duplicate_ids() {
    let response = json!({"nodes": [
        {"node_type": "STATED_FACT", "text": "zero"},
        {"node_type": "ENTITY", "id": "E"},
        {"node_type": "STATED_FACT", "id": "STATED_FACT_0", "text": "one"},
        {"node_type": "STATED_FACT", "id": "", "text": "two"},
        {"node_type": "STATED_FACT", "id": 9, "text": "three"},
        {"node_type": "STATED_FACT", "id": "STATED_FACT_0", "text": "four"}
    ]})
    .to_string();
    assert_eq!(
        ids(&parse(&response)),
        ["STATED_FACT_0", "STATED_FACT_1", "STATED_FACT_2", "9", "STATED_FACT_4"]
    );
    let prefixed = parse_facts(&response, Some("pass_x")).expect("ok");
    assert_eq!(ids(&prefixed)[..2], ["pass_x_STATED_FACT_0", "pass_x_STATED_FACT_1"]);
    // An empty pass name adds no prefix.
    assert_eq!(ids(&parse_facts(&response, Some("")).expect("ok"))[0], "STATED_FACT_0");
}

#[test]
fn pass_prefix_is_not_repeated() {
    let response = json!({"nodes": [
        {"node_type": "STATED_FACT", "id": "pass_x_STATED_FACT_0", "text": "a"},
        {"node_type": "STATED_FACT", "id": "STATED_FACT_1", "text": "b"}
    ]})
    .to_string();
    let facts = parse_facts(&response, Some("pass_x")).expect("ok");
    assert_eq!(ids(&facts), ["pass_x_STATED_FACT_0", "pass_x_STATED_FACT_1"]);
}

// ---------------------------------------------------------------------------------------------
// Multi-pass extraction
// ---------------------------------------------------------------------------------------------

/// Pass responses: each pass reports its own fact, passes 0 and 1 share one fact (pass 1 with
/// higher certainty) and pass 3 repeats pass 2's fact with equal certainty.
fn multi_pass_llm() -> Arc<MockLlm> {
    Arc::new(MockLlm::with_responder(|req| {
        let shared = |certainty: &str| {
            json!({"node_type": "STATED_FACT", "id": "STATED_FACT_9", "text": "The collision occurred on March 4, 2023",
                   "assertion_triple": {"subject": "collision", "relationship": "occurred_on", "object": "March 4, 2023"},
                   "certainty": certainty})
        };
        let own = |pass: usize| {
            json!({"node_type": "STATED_FACT", "id": "STATED_FACT_0", "text": format!("Own fact of pass {pass}"),
                   "assertion_triple": {"subject": format!("s{pass}"), "relationship": "r", "object": "o"},
                   "certainty": "0.8"})
        };
        let nodes = match pass_index(req.prompt) {
            0 => vec![shared("0.6"), own(0)],
            1 => vec![own(1), shared("0.9")],
            2 => vec![
                own(2),
                json!({"node_type": "STATED_FACT", "id": "STATED_FACT_1", "text": "Common", "certainty": "0.7"}),
            ],
            _ => vec![
                json!({"node_type": "STATED_FACT", "id": "STATED_FACT_1", "text": " common ", "certainty": "0.7"}),
                own(3),
            ],
        };
        Ok(json!({"nodes": nodes, "edges": []}).to_string())
    }))
}

#[test]
fn multi_pass_prefixes_merges_and_deduplicates() {
    let expected = [
        "incident_core_STATED_FACT_0",
        "party_state_STATED_FACT_0",
        "party_state_STATED_FACT_9",
        "history_employment_STATED_FACT_0",
        "history_employment_STATED_FACT_1",
        "damages_expenses_STATED_FACT_0",
    ];
    for (parallel, workers) in [(false, 4), (true, 4), (true, 2), (true, 1), (true, 0), (true, 16)]
    {
        let llm = multi_pass_llm();
        let extractor = MultiPassFactExtractor::new(
            &(llm.clone() as Arc<dyn Llm>),
            &multi_prompts(),
            parallel,
            workers,
        );
        assert_eq!(extractor.pass_names(), PASS_NAMES);
        let facts = extractor.extract_facts(CHUNK, &entities());
        assert_eq!(ids(&facts), expected, "parallel={parallel} workers={workers}");
        for fact in &facts {
            let pass = fact.metadata.get("pass_name").and_then(|v| v.as_str()).expect("pass_name");
            assert!(fact.id.as_str().starts_with(&format!("{pass}_")), "{}", fact.id);
        }
        let shared = &facts[2];
        assert!((shared.certainty - 0.9).abs() < 1e-12);
        let prompts = llm.prompts();
        assert_eq!(prompts.len(), 4);
        let mut seen: Vec<usize> = prompts.iter().map(|p| pass_index(p)).collect();
        seen.sort_unstable();
        assert_eq!(seen, [0, 1, 2, 3]);
        assert!(prompts.iter().all(|p| p.contains(CHUNK) && p.contains("ent-lopez")));
    }
}

#[test]
fn multi_pass_keeps_ids_that_already_start_with_the_pass_name() {
    let llm: Arc<dyn Llm> = Arc::new(MockLlm::with_responder(|_| {
        Ok(json!({"nodes": [{"node_type": "STATED_FACT", "id": "pass_0_custom", "text": "t"}]})
            .to_string())
    }));
    let mut prompt = single_prompt();
    prompt.pass_name = None;
    let mut second = prompt.clone();
    second.user_prompt_template.push_str(" second");
    let extractor = MultiPassFactExtractor::new(&llm, &[prompt, second], false, 1);
    assert_eq!(extractor.pass_names(), ["pass_0", "pass_1"]);
    let facts = extractor.extract_facts(CHUNK, &[]);
    // Same text in both passes: the first is kept, the second is a duplicate.
    assert_eq!(ids(&facts), ["pass_0_custom"]);
}

#[test]
fn a_failing_pass_does_not_stop_the_others() {
    let llm: Arc<dyn Llm> = Arc::new(MockLlm::with_responder(|req| {
        if pass_index(req.prompt) == 1 {
            Err(ModelError::Inference("timeout".to_owned()))
        } else if pass_index(req.prompt) == 2 {
            Ok("garbage".to_owned())
        } else {
            Ok(v3_response(2, PASS_NAMES[pass_index(req.prompt)]))
        }
    }));
    let extractor = MultiPassFactExtractor::new(&llm, &multi_prompts(), true, 4);
    assert_eq!(
        ids(&extractor.extract_facts(CHUNK, &[])),
        [
            "incident_core_STATED_FACT_0",
            "incident_core_STATED_FACT_1",
            "damages_expenses_STATED_FACT_0",
            "damages_expenses_STATED_FACT_1"
        ]
    );
}

#[test]
fn dedup_across_passes_matches_python_order() {
    let facts = parse(
        &json!({"nodes": [
            {"node_type": "STATED_FACT", "id": "a", "text": "Same", "certainty": 0.4},
            {"node_type": "STATED_FACT", "id": "b", "text": "Other", "certainty": 0.4},
            {"node_type": "STATED_FACT", "id": "c", "text": "same", "certainty": 0.5},
            {"node_type": "STATED_FACT", "id": "d", "text": "SAME ", "certainty": 0.45}
        ]})
        .to_string(),
    );
    assert_eq!(ids(&deduplicate_facts(facts)), ["b", "c"]);
}

// ---------------------------------------------------------------------------------------------
// Regression: documents with 64+ facts
// ---------------------------------------------------------------------------------------------

fn expected_ids(count: usize, prefix: &str) -> Vec<String> {
    (0..count).map(|i| format!("{prefix}STATED_FACT_{i}")).collect()
}

#[test]
fn hundred_facts_parse_in_order() {
    let facts = parse(&v3_response(100, "doc"));
    assert_eq!(ids(&facts), expected_ids(100, ""));
    assert_eq!(facts[99].assertion_triple.subject, "doc subject 99");

    // Wrapped in prose and with trailing commas, still all 100.
    let messy = format!(
        "Here are the facts:\n{}\nThat is all.",
        v3_response(100, "doc").replace("}\n]", "},\n]")
    );
    assert_eq!(ids(&parse(&messy)), expected_ids(100, ""));

    // Ids missing on every node.
    let unnamed = v3_response(100, "doc").replace("\"id\":", "\"ignored\":");
    assert_eq!(ids(&parse(&unnamed)), expected_ids(100, ""));
}

#[test]
fn hundred_facts_through_the_single_pass_extractor() {
    let llm =
        Arc::new(MockLlm::with_responses([format!("```json\n{}\n```", v3_response(100, "doc"))]));
    let extractor = create_extractor(llm, &[single_prompt()], true, 4);
    assert_eq!(ids(&extractor.extract_facts(CHUNK, &entities())), expected_ids(100, ""));
}

#[test]
fn hundred_facts_per_pass_through_the_multi_pass_extractor() {
    let llm: Arc<dyn Llm> = Arc::new(MockLlm::with_responder(|req| {
        Ok(v3_response(100, PASS_NAMES[pass_index(req.prompt)]))
    }));
    let parallel = create_extractor(Arc::clone(&llm), &multi_prompts(), true, 4);
    let facts = parallel.extract_facts(CHUNK, &entities());
    assert_eq!(facts.len(), 400);
    let expected: Vec<String> =
        PASS_NAMES.iter().flat_map(|pass| expected_ids(100, &format!("{pass}_"))).collect();
    assert_eq!(ids(&facts), expected);
    let sequential = create_extractor(llm, &multi_prompts(), false, 4);
    assert_eq!(sequential.extract_facts(CHUNK, &entities()), facts);
}

#[test]
fn truncated_output_salvages_complete_facts() {
    let full = v3_response(100, "doc");
    let fact_80 = full.find("\"STATED_FACT_80\"").expect("fact 80");
    let truncated = &full[..fact_80 + 40];
    let facts = parse(truncated);
    assert_eq!(ids(&facts), expected_ids(80, ""));
    assert_eq!(facts[79].assertion_triple.subject, "doc subject 79");

    // Cut inside a string of a fenced response.
    let fenced =
        format!("```json\n{}", &full[..full.find("doc fact number 80").expect("text 80") + 5]);
    assert_eq!(ids(&parse(&fenced)), expected_ids(80, ""));

    // Through the extractor.
    let llm = Arc::new(MockLlm::with_responses([truncated.to_owned()]));
    let extractor = LlmFactExtractor::new(llm, &single_prompt());
    assert_eq!(ids(&extractor.extract_facts(CHUNK, &[])), expected_ids(80, ""));
}

#[test]
fn one_corrupt_node_among_hundred_loses_only_that_node() {
    let response =
        v3_response(100, "doc").replacen("\"certainty\":\"0.9\"", "\"certainty\":0.9.1", 51);
    let response = response.replacen("\"certainty\":0.9.1", "\"certainty\":\"0.9\"", 50);
    let facts = parse(&response);
    assert_eq!(facts.len(), 99);
    assert!(!ids(&facts).contains(&"STATED_FACT_50".to_owned()));
    assert_eq!(facts[50].id.as_str(), "STATED_FACT_51");
}
