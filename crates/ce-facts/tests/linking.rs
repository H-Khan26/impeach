//! Fact-entity linking: name normalisation, entity lookup, subject / object linking with
//! bracket tokens, type-aware fallbacks and role terms, the resolved-text view and the per-chunk
//! prompt entities (cases of Python `tests/unit/test_fact_entity_linking.py` plus the role-map
//! extension).

use ce_config::MappingsRegistry;
use ce_core::processing::FactLinkingStats;
use ce_core::{
    AssertionTriple, EdCandidate, EdResult, EntityId, ExtractedFact, Mention, MentionId, Span,
    TemporalValidity,
};
use ce_facts::{
    build_canonical_lookup, build_resolved_text_view, parse_bracket_token, prompt_entities,
    record_link, CanonicalEntity, EntityLookup, FactLinker,
};
use ce_ner::definitions::RoleConfig;
use indexmap::{IndexMap, IndexSet};
use pretty_assertions::assert_eq;

fn mappings() -> &'static MappingsRegistry {
    MappingsRegistry::load().expect("mappings load")
}

fn roles() -> RoleConfig {
    RoleConfig::load(None).expect("roles.yaml loads")
}

/// Linker without a role vocabulary (Python `role_config=None`).
fn linker() -> FactLinker {
    FactLinker::new(mappings(), None)
}

fn linker_with_roles() -> FactLinker {
    FactLinker::new(mappings(), Some(roles()))
}

fn norm(name: &str, entity_type: Option<&str>) -> String {
    linker().normalize_name(name, entity_type)
}

fn fact(subject: &str, relationship: &str, object: &str) -> ExtractedFact {
    ExtractedFact::new(
        "f1",
        format!("{subject} {relationship} {object}"),
        AssertionTriple::new(subject, relationship, object),
        0.95,
        TemporalValidity::default(),
    )
}

fn candidate(id: &str, name: &str, label: &str, aliases: &[&str]) -> EdCandidate {
    let mut candidate = EdCandidate::new(id, name, label);
    candidate.aliases = aliases.iter().map(|a| (*a).to_owned()).collect();
    candidate.score = 0.95;
    candidate.source = "exact".to_owned();
    candidate
}

fn ed_result(mention_id: &str, chosen: &str, candidates: Vec<EdCandidate>) -> EdResult {
    EdResult {
        mention_id: Some(mention_id.into()),
        chosen_entity_id: Some(chosen.into()),
        candidates,
        resolved_by_llm: false,
        confidence: 0.95,
        strategy_used: "rule".to_owned(),
    }
}

fn mention(id: &str, entity_type: &str, name: &str, start: usize, end: usize) -> Mention {
    let mut mention = Mention::new(id, entity_type, name, Span::new(start, end));
    mention.score = Some(0.9);
    mention
}

fn resolved(pairs: &[(&str, &str)]) -> IndexMap<MentionId, EntityId> {
    pairs.iter().map(|(m, e)| (MentionId::new(*m), EntityId::new(*e))).collect()
}

/// A lookup with the given `normalised name -> (entity id, type)` entries.
fn lookup(entries: &[(&str, &str, &str)]) -> EntityLookup {
    let mut lookup = EntityLookup::default();
    for (name, id, entity_type) in entries {
        lookup.name_to_entity.insert((*name).to_owned(), EntityId::new(*id));
        lookup.name_to_type.insert((*name).to_owned(), (*entity_type).to_owned());
    }
    lookup
}

fn canonical(entries: &[(&str, &str, &str, &[&str])]) -> IndexMap<EntityId, CanonicalEntity> {
    entries
        .iter()
        .map(|(id, name, entity_type, aliases)| {
            (
                EntityId::new(*id),
                CanonicalEntity {
                    name: (*name).to_owned(),
                    entity_type: (*entity_type).to_owned(),
                    aliases: aliases.iter().map(|a| (*a).to_owned()).collect(),
                },
            )
        })
        .collect()
}

fn ids(linked: &(Option<EntityId>, Option<EntityId>)) -> (Option<&str>, Option<&str>) {
    (linked.0.as_ref().map(EntityId::as_str), linked.1.as_ref().map(EntityId::as_str))
}

fn no_canonical() -> IndexMap<EntityId, CanonicalEntity> {
    IndexMap::new()
}

// --- normalisation ---

#[test]
fn basic_normalization_without_type() {
    assert_eq!(norm("Green Valley Trucking, Inc.", None), "green valley trucking inc");
}

#[test]
fn person_normalization_strips_titles() {
    let result = norm("Dr. Maria Lopez, Esq.", Some("PERSON"));
    let words: Vec<&str> = result.split(' ').collect();
    assert!(words.contains(&"maria"), "{result}");
    assert!(words.contains(&"lopez"), "{result}");
    assert!(!words.contains(&"dr"), "{result}");
    assert!(!words.contains(&"esq"), "{result}");
}

#[test]
fn org_normalization_strips_prefixes_and_suffixes() {
    let result = norm("The Acme Corporation, LLC", Some("ORG"));
    assert!(result.contains("acme"), "{result}");
    assert!(!result.contains("the") || result.starts_with("acme"), "{result}");
    assert!(!result.contains("corporation"), "{result}");
    assert!(!result.contains("llc"), "{result}");
}

#[test]
fn empty_and_whitespace_names_normalise_to_empty() {
    assert_eq!(norm("", None), "");
    assert_eq!(norm("", Some("PERSON")), "");
    assert_eq!(norm("   ", None), "");
}

#[test]
fn unknown_type_uses_basic_normalization() {
    assert_eq!(norm("The Honda Civic.", Some("VEHICLE")), "the honda civic");
    assert_eq!(norm("Maria Lopez", Some("")), "maria lopez");
}

#[test]
fn person_name_without_parts_falls_back_to_basic_normalization() {
    // Only a title: no person name parts remain.
    assert_eq!(norm("Dr.", Some("PER")), "dr");
}

// --- entity lookup ---

#[test]
fn lookup_holds_ed_canonical_names() {
    let results = vec![ed_result(
        "m1",
        "e1",
        vec![candidate("e1", "Green Valley Trucking", "ORG", &["GVT", "Green Valley"])],
    )];
    let lookup = linker().build_entity_lookup(&results, &IndexMap::new(), &[]);
    let normalized = norm("Green Valley Trucking", Some("ORG"));
    assert_eq!(lookup.name_to_entity[&normalized].as_str(), "e1");
    assert_eq!(lookup.name_to_type[&normalized], "ORG");
}

#[test]
fn lookup_holds_ed_aliases() {
    let results = vec![ed_result(
        "m1",
        "e1",
        vec![candidate(
            "e1",
            "Maria Elena Lopez",
            "PERSON",
            &["Maria Lopez", "M. Lopez", "Ms. Lopez"],
        )],
    )];
    let lookup = linker().build_entity_lookup(&results, &IndexMap::new(), &[]);
    for alias in ["Maria Elena Lopez", "Maria Lopez", "M. Lopez", "Ms. Lopez"] {
        let normalized = norm(alias, Some("PERSON"));
        assert_eq!(
            lookup.name_to_entity.get(&normalized).map(EntityId::as_str),
            Some("e1"),
            "alias {alias} ({normalized})"
        );
    }
}

#[test]
fn lookup_holds_raw_and_lowercased_entity_ids() {
    let results = vec![ed_result("m1", "Ent_42", vec![candidate("Ent_42", "Acme", "ORG", &[])])];
    let lookup = linker().build_entity_lookup(&results, &IndexMap::new(), &[]);
    assert_eq!(lookup.name_to_entity["Ent_42"].as_str(), "Ent_42");
    assert_eq!(lookup.name_to_entity["ent_42"].as_str(), "Ent_42");
}

#[test]
fn lookup_uses_mention_surface_forms_as_fallback() {
    let mentions = vec![
        mention("m1", "PERSON", "Lopez", 0, 5),
        mention("m2", "PERSON", "Maria Lopez", 10, 21),
    ];
    let lookup =
        linker().build_entity_lookup(&[], &resolved(&[("m1", "e1"), ("m2", "e1")]), &mentions);
    assert_eq!(lookup.name_to_entity[&norm("Lopez", Some("PERSON"))].as_str(), "e1");
    assert_eq!(lookup.name_to_entity[&norm("Maria Lopez", Some("PERSON"))].as_str(), "e1");
    assert_eq!(lookup.name_to_type[&norm("Lopez", Some("PERSON"))], "PERSON");
}

#[test]
fn lookup_skips_unresolved_mentions() {
    let mentions = vec![mention("m1", "PERSON", "Lopez", 0, 5)];
    let lookup = linker().build_entity_lookup(&[], &resolved(&[("m9", "e1")]), &mentions);
    assert!(lookup.name_to_entity.is_empty());
}

#[test]
fn ed_canonical_takes_precedence_over_mentions() {
    let results = vec![ed_result(
        "m1",
        "e_canonical",
        vec![candidate("e_canonical", "Green Valley Trucking Inc", "ORG", &[])],
    )];
    let mentions = vec![mention("m2", "ORG", "Green Valley Trucking Inc", 0, 26)];
    let lookup =
        linker().build_entity_lookup(&results, &resolved(&[("m2", "e_mention")]), &mentions);
    let normalized = norm("Green Valley Trucking Inc", Some("ORG"));
    assert_eq!(lookup.name_to_entity[&normalized].as_str(), "e_canonical");
}

#[test]
fn only_the_chosen_candidate_is_used() {
    let results = vec![ed_result(
        "m1",
        "e1",
        vec![
            candidate("e1", "Maria Lopez", "PERSON", &["M. Lopez"]),
            candidate("e2", "Maria Rodriguez", "PERSON", &[]),
        ],
    )];
    let lookup = linker().build_entity_lookup(&results, &IndexMap::new(), &[]);
    assert_eq!(lookup.name_to_entity[&norm("Maria Lopez", Some("PERSON"))].as_str(), "e1");
    assert!(!lookup.name_to_entity.contains_key(&norm("Maria Rodriguez", Some("PERSON"))));
}

#[test]
fn canonical_name_overwrites_and_aliases_do_not() {
    // The alias of the first entity equals the canonical name of the second: the canonical
    // name wins; the second entity's alias equal to the first's canonical name does not.
    let results = vec![
        ed_result("m1", "e1", vec![candidate("e1", "Acme", "ORG", &["Beta Works"])]),
        ed_result("m2", "e2", vec![candidate("e2", "Beta Works", "ORG", &["Acme"])]),
    ];
    let lookup = linker().build_entity_lookup(&results, &IndexMap::new(), &[]);
    assert_eq!(lookup.name_to_entity[&norm("Acme", Some("ORG"))].as_str(), "e1");
    assert_eq!(lookup.name_to_entity[&norm("Beta Works", Some("ORG"))].as_str(), "e2");
}

#[test]
fn results_without_a_chosen_entity_are_ignored() {
    let mut result = ed_result("m1", "e1", vec![candidate("e1", "Acme", "ORG", &[])]);
    result.chosen_entity_id = None;
    let mut empty = ed_result("m2", "", vec![candidate("", "Beta", "ORG", &[])]);
    empty.chosen_entity_id = Some(EntityId::new(""));
    let lookup = linker().build_entity_lookup(&[result, empty], &IndexMap::new(), &[]);
    assert!(lookup.name_to_entity.is_empty());
    assert!(build_canonical_lookup(&[]).is_empty());
}

// --- canonical lookup ---

#[test]
fn canonical_lookup_uses_the_chosen_candidate() {
    let mut unlabeled = candidate("e2", "Main Street", "", &[]);
    unlabeled.label = String::new();
    let results = vec![
        ed_result(
            "m1",
            "e1",
            vec![
                candidate("e0", "Other", "PER", &[]),
                candidate("e1", "Maria Lopez", "PER", &["Lopez"]),
            ],
        ),
        ed_result("m2", "e2", vec![unlabeled]),
        ed_result("m3", "e3", vec![candidate("e3", "", "ORG", &[])]),
        ed_result("m4", "e4", vec![candidate("e9", "Not chosen", "ORG", &[])]),
    ];
    let lookup = build_canonical_lookup(&results);
    assert_eq!(
        lookup,
        canonical(&[
            ("e1", "Maria Lopez", "PER", &["Lopez"]),
            ("e2", "Main Street", "ENTITY", &[])
        ])
    );
}

// --- linking ---

#[test]
fn exact_match_subject() {
    let lookup = lookup(&[(&norm("Maria Lopez", Some("PERSON")), "e1", "PERSON")]);
    let mut fact = fact("Maria Lopez", "SUSTAINED_INJURY", "cervical strain");
    let linked = linker().link_fact(&mut fact, &lookup, &no_canonical(), None);
    assert_eq!(ids(&linked), (Some("e1"), None));
}

#[test]
fn exact_match_object() {
    let lookup = lookup(&[(&norm("General Hospital", Some("ORG")), "e_hosp", "ORG")]);
    let mut fact = fact("patient", "RECEIVED_TRANSPORT", "General Hospital");
    let linked = linker().link_fact(&mut fact, &lookup, &no_canonical(), None);
    assert_eq!(ids(&linked), (None, Some("e_hosp")));
}

#[test]
fn both_subject_and_object_linked() {
    let lookup = lookup(&[
        (&norm("Maria Lopez", Some("PERSON")), "e1", "PERSON"),
        (&norm("James Reed", Some("PERSON")), "e2", "PERSON"),
    ]);
    let mut fact = fact("Maria Lopez", "FILED_SUIT_AGAINST", "James Reed");
    let linked = linker().link_fact(&mut fact, &lookup, &no_canonical(), None);
    assert_eq!(ids(&linked), (Some("e1"), Some("e2")));
}

#[test]
fn matching_is_case_insensitive() {
    let lookup = lookup(&[(&norm("maria lopez", Some("PERSON")), "e1", "PERSON")]);
    let mut fact = fact("MARIA LOPEZ", "SUSTAINED_INJURY", "injury");
    let linked = linker().link_fact(&mut fact, &lookup, &no_canonical(), None);
    assert_eq!(linked.0.as_ref().map(EntityId::as_str), Some("e1"));
}

#[test]
fn org_suffix_variations_link() {
    let lookup = lookup(&[(&norm("Green Valley Trucking", Some("ORG")), "e_gvt", "ORG")]);
    let mut fact = fact("Green Valley Trucking, Inc.", "OPERATED", "vehicle");
    let linked = linker().link_fact(&mut fact, &lookup, &no_canonical(), None);
    assert_eq!(linked.0.as_ref().map(EntityId::as_str), Some("e_gvt"));
}

#[test]
fn empty_subject_and_object_do_not_link() {
    let lookup = lookup(&[("test", "e1", "PERSON")]);
    let mut fact = fact("", "RELATED_TO", "");
    let linked = linker().link_fact(&mut fact, &lookup, &no_canonical(), None);
    assert_eq!(ids(&linked), (None, None));
}

#[test]
fn no_match_returns_none() {
    let lookup = lookup(&[(&norm("Someone Else", Some("PERSON")), "e_other", "PERSON")]);
    let mut fact = fact("unknown person", "DID", "something");
    let linked = linker().link_fact(&mut fact, &lookup, &no_canonical(), None);
    assert_eq!(ids(&linked), (None, None));
}

#[test]
fn type_aware_fallback_links_person_names() {
    let lookup = lookup(&[(&norm("John Smith", Some("PERSON")), "e1", "PERSON")]);
    let mut fact = fact("John Smith", "WAS_PRESENT", "scene");
    let linked = linker().link_fact(&mut fact, &lookup, &no_canonical(), None);
    assert_eq!(linked.0.as_ref().map(EntityId::as_str), Some("e1"));
}

#[test]
fn typed_fallback_finds_titled_person_and_org_suffix() {
    // Only the PER normalisation strips the title; only the ORG normalisation the suffix.
    let lookup = lookup(&[
        (&norm("Maria Lopez", Some("PER")), "e1", "PER"),
        (&norm("Acme Holdings", Some("ORG")), "e2", "ORG"),
    ]);
    assert_ne!(norm("Dr. Maria Lopez", None), norm("Maria Lopez", Some("PER")));
    let mut fact = fact("Dr. Maria Lopez", "SUED", "Acme Holdings LLC");
    let linked = linker().link_fact(&mut fact, &lookup, &no_canonical(), None);
    assert_eq!(ids(&linked), (Some("e1"), Some("e2")));
    assert_eq!(fact.assertion_triple.subject, "Dr. Maria Lopez");
}

#[test]
fn linking_does_not_set_fact_entity_fields() {
    let lookup = lookup(&[("maria lopez", "e1", "PER")]);
    let mut fact = fact("Maria Lopez", "MET", "someone");
    let _ = linker().link_fact(&mut fact, &lookup, &no_canonical(), None);
    assert_eq!(fact.subject_entity_id, None);
    assert_eq!(fact.object_entity_id, None);
}

// --- end to end ---

#[test]
fn full_pipeline_with_ed_and_mentions() {
    let results = vec![
        ed_result(
            "m1",
            "e_maria",
            vec![candidate("e_maria", "Maria Elena Lopez", "PERSON", &["Maria Lopez", "M. Lopez"])],
        ),
        ed_result(
            "m2",
            "e_gvt",
            vec![candidate("e_gvt", "Green Valley Trucking Inc", "ORG", &["GVT"])],
        ),
    ];
    let mentions = vec![
        mention("m1", "PERSON", "Maria Lopez", 0, 11),
        mention("m2", "ORG", "Green Valley Trucking", 50, 71),
    ];
    let linker = linker();
    let lookup = linker.build_entity_lookup(
        &results,
        &resolved(&[("m1", "e_maria"), ("m2", "e_gvt")]),
        &mentions,
    );
    let mut fact = fact("Maria Lopez", "EMPLOYED_BY", "Green Valley Trucking, Inc.");
    let linked = linker.link_fact(&mut fact, &lookup, &build_canonical_lookup(&results), None);
    assert_eq!(ids(&linked), (Some("e_maria"), Some("e_gvt")));
}

#[test]
fn alias_matching_end_to_end() {
    let results = vec![ed_result(
        "m1",
        "e_company",
        vec![candidate("e_company", "Acme Corporation", "ORG", &["Acme Corp", "Acme Co.", "Acme"])],
    )];
    let linker = linker();
    let lookup = linker.build_entity_lookup(&results, &IndexMap::new(), &[]);
    let mut fact = fact("Acme Corp", "MANUFACTURED", "product");
    let linked = linker.link_fact(&mut fact, &lookup, &no_canonical(), None);
    assert_eq!(linked.0.as_ref().map(EntityId::as_str), Some("e_company"));
}

#[test]
fn llm_output_of_an_entity_id_links() {
    let results = vec![ed_result("m1", "E17", vec![candidate("E17", "Acme", "ORG", &[])])];
    let linker = linker();
    let lookup = linker.build_entity_lookup(&results, &IndexMap::new(), &[]);
    let mut fact = fact("E17", "OWNS", "e17");
    let linked = linker.link_fact(&mut fact, &lookup, &no_canonical(), None);
    assert_eq!(ids(&linked), (Some("E17"), Some("E17")));
}

// --- bracket tokens ---

#[test]
fn parse_simple_bracket_token() {
    assert_eq!(parse_bracket_token("[e1:Maria Lopez]"), Some(("e1", "Maria Lopez")));
}

#[test]
fn parse_bracket_token_with_complex_id() {
    assert_eq!(
        parse_bracket_token("[entity_123abc:Green Valley Trucking Inc]"),
        Some(("entity_123abc", "Green Valley Trucking Inc"))
    );
}

#[test]
fn parse_bracket_token_in_sentence() {
    assert_eq!(
        parse_bracket_token("The company [e_gvt:Green Valley Trucking] operates in California"),
        Some(("e_gvt", "Green Valley Trucking"))
    );
}

#[test]
fn parse_bracket_token_with_punctuation() {
    assert_eq!(parse_bracket_token("[e1:Maria Lopez],"), Some(("e1", "Maria Lopez")));
}

#[test]
fn parse_without_brackets_returns_none() {
    assert_eq!(parse_bracket_token("Maria Lopez"), None);
}

#[test]
fn parse_malformed_bracket_returns_none() {
    assert_eq!(parse_bracket_token("[e1 Maria Lopez]"), None);
    assert_eq!(parse_bracket_token("[e1:Maria Lopez"), None);
}

#[test]
fn parse_empty_string_returns_none() {
    assert_eq!(parse_bracket_token(""), None);
}

#[test]
fn parse_takes_the_first_token_and_trims_both_parts() {
    assert_eq!(
        parse_bracket_token("[ e1 : Maria Lopez ] and [e2:James Reed]"),
        Some(("e1", "Maria Lopez"))
    );
    // The canonical name stops at the first closing bracket; ids cannot contain a colon.
    assert_eq!(parse_bracket_token("[ns:e1:Maria]"), Some(("ns", "e1:Maria")));
}

#[test]
fn link_subject_via_bracket_token() {
    let mut fact = fact("[e1:Maria Lopez]", "SUSTAINED_INJURY", "injury");
    let linked = linker().link_fact(&mut fact, &EntityLookup::default(), &no_canonical(), None);
    assert_eq!(ids(&linked), (Some("e1"), None));
    assert_eq!(fact.assertion_triple.subject, "Maria Lopez");
}

#[test]
fn link_object_via_bracket_token() {
    let mut fact = fact("patient", "TRANSPORTED_TO", "[e_hosp:General Hospital]");
    let linked = linker().link_fact(&mut fact, &EntityLookup::default(), &no_canonical(), None);
    assert_eq!(ids(&linked), (None, Some("e_hosp")));
    assert_eq!(fact.assertion_triple.object, "General Hospital");
}

#[test]
fn link_both_via_bracket_tokens() {
    let mut fact = fact("[e1:Maria Lopez]", "FILED_SUIT_AGAINST", "[e2:James Reed]");
    let linked = linker().link_fact(&mut fact, &EntityLookup::default(), &no_canonical(), None);
    assert_eq!(ids(&linked), (Some("e1"), Some("e2")));
    assert_eq!(fact.assertion_triple.subject, "Maria Lopez");
    assert_eq!(fact.assertion_triple.object, "James Reed");
}

#[test]
fn bracket_token_takes_precedence_over_normalization() {
    let lookup = lookup(&[(&norm("Maria Lopez", Some("PERSON")), "e_wrong", "PERSON")]);
    let mut fact = fact("[e_correct:Maria Lopez]", "SUSTAINED_INJURY", "injury");
    let linked = linker().link_fact(&mut fact, &lookup, &no_canonical(), None);
    assert_eq!(linked.0.as_ref().map(EntityId::as_str), Some("e_correct"));
}

#[test]
fn fallback_to_normalization_without_bracket() {
    let lookup = lookup(&[(&norm("Maria Lopez", Some("PERSON")), "e1", "PERSON")]);
    let mut fact = fact("Maria Lopez", "SUSTAINED_INJURY", "injury");
    let linked = linker().link_fact(&mut fact, &lookup, &no_canonical(), None);
    assert_eq!(linked.0.as_ref().map(EntityId::as_str), Some("e1"));
}

#[test]
fn bracket_token_inside_a_phrase_rewrites_the_whole_side() {
    let mut fact = fact("the driver [e2:Daniel Reed]", "DROVE", "truck");
    let linked = linker().link_fact(&mut fact, &EntityLookup::default(), &no_canonical(), None);
    assert_eq!(ids(&linked), (Some("e2"), None));
    assert_eq!(fact.assertion_triple.subject, "Daniel Reed");
}

// --- resolved text view ---

#[test]
fn resolved_view_simple_replacement() {
    let mentions = vec![mention("m1", "PERSON", "Maria Lopez", 0, 11)];
    let result = build_resolved_text_view(
        "Maria Lopez was injured in the incident.",
        0,
        &mentions,
        &resolved(&[("m1", "e1")]),
        &canonical(&[("e1", "Maria Lopez", "PERSON", &[])]),
    );
    assert_eq!(result, "[e1:Maria Lopez] was injured in the incident.");
}

#[test]
fn resolved_view_multiple_replacements() {
    let mentions = vec![
        mention("m1", "PERSON", "Maria Lopez", 100, 111),
        mention("m2", "PERSON", "James Reed", 131, 141),
    ];
    let result = build_resolved_text_view(
        "Maria Lopez filed suit against James Reed yesterday.",
        100,
        &mentions,
        &resolved(&[("m1", "e1"), ("m2", "e2")]),
        &canonical(&[("e1", "Maria Lopez", "PERSON", &[]), ("e2", "James Reed", "PERSON", &[])]),
    );
    assert_eq!(result, "[e1:Maria Lopez] filed suit against [e2:James Reed] yesterday.");
}

#[test]
fn resolved_view_python_case_with_shifted_second_span() {
    // The Python case gives the second mention [132, 142) (local 32-42): the replacement
    // covers "ames Reed " and the text still ends with "yesterday.".
    let mentions = vec![
        mention("m1", "PERSON", "Maria Lopez", 100, 111),
        mention("m2", "PERSON", "James Reed", 132, 142),
    ];
    let result = build_resolved_text_view(
        "Maria Lopez filed suit against James Reed yesterday.",
        100,
        &mentions,
        &resolved(&[("m1", "e1"), ("m2", "e2")]),
        &canonical(&[("e1", "Maria Lopez", "PERSON", &[]), ("e2", "James Reed", "PERSON", &[])]),
    );
    assert!(result.starts_with("[e1:Maria Lopez]"), "{result}");
    assert!(result.contains("[e2:James Reed]"), "{result}");
    assert!(result.ends_with("yesterday."), "{result}");
}

#[test]
fn resolved_view_converts_chunk_offsets() {
    let mentions = vec![mention("m1", "EVENT", "incident", 504, 512)];
    let result = build_resolved_text_view(
        "The incident occurred at noon.",
        500,
        &mentions,
        &resolved(&[("m1", "e_incident")]),
        &canonical(&[("e_incident", "incident", "EVENT", &[])]),
    );
    assert_eq!(result, "The [e_incident:incident] occurred at noon.");
}

#[test]
fn resolved_view_skips_unresolved_mentions() {
    let mentions = vec![
        mention("m1", "PERSON", "Maria Lopez", 0, 11),
        mention("m2", "PERSON", "John Doe", 16, 24),
    ];
    let result = build_resolved_text_view(
        "Maria Lopez and John Doe were present.",
        0,
        &mentions,
        &resolved(&[("m1", "e1")]),
        &canonical(&[("e1", "Maria Lopez", "PERSON", &[])]),
    );
    assert_eq!(result, "[e1:Maria Lopez] and John Doe were present.");
}

#[test]
fn resolved_view_without_mentions_returns_original() {
    let text = "Some text here.";
    assert_eq!(build_resolved_text_view(text, 0, &[], &IndexMap::new(), &no_canonical()), text);
    let mentions = vec![mention("m1", "PERSON", "Some", 0, 4)];
    assert_eq!(
        build_resolved_text_view(text, 0, &mentions, &IndexMap::new(), &no_canonical()),
        text
    );
}

#[test]
fn resolved_view_falls_back_to_mention_name() {
    let mentions = vec![mention("m1", "PERSON", "Maria Lopez", 0, 11)];
    let result = build_resolved_text_view(
        "Maria Lopez was present.",
        0,
        &mentions,
        &resolved(&[("m1", "e1")]),
        &no_canonical(),
    );
    assert_eq!(result, "[e1:Maria Lopez] was present.");
}

#[test]
fn resolved_view_uses_canonical_name_over_surface_form() {
    let mentions = vec![mention("m1", "PERSON", "Lopez", 0, 5)];
    let result = build_resolved_text_view(
        "Lopez was present.",
        0,
        &mentions,
        &resolved(&[("m1", "e1")]),
        &canonical(&[("e1", "Maria Lopez", "PER", &[])]),
    );
    assert_eq!(result, "[e1:Maria Lopez] was present.");
}

#[test]
fn resolved_view_keeps_the_rightmost_of_overlapping_mentions() {
    let mentions = vec![
        mention("m1", "PERSON", "Maria Lopez", 0, 11),
        mention("m2", "PERSON", "Lopez-Reed", 6, 16),
    ];
    let result = build_resolved_text_view(
        "Maria Lopez-Reed was present.",
        0,
        &mentions,
        &resolved(&[("m1", "e1"), ("m2", "e2")]),
        &canonical(&[("e1", "Maria Lopez", "PERSON", &[]), ("e2", "Lopez-Reed", "PERSON", &[])]),
    );
    assert_eq!(result, "Maria [e2:Lopez-Reed] was present.");
}

#[test]
fn resolved_view_skips_out_of_bounds_mentions() {
    let mentions = vec![
        mention("before", "PERSON", "Ann", 5, 8),
        mention("after", "PERSON", "Bob", 20, 40),
        mention("inside", "PERSON", "Cy", 10, 12),
    ];
    let result = build_resolved_text_view(
        "Cy and friends.",
        10,
        &mentions,
        &resolved(&[("before", "e1"), ("after", "e2"), ("inside", "e3")]),
        &no_canonical(),
    );
    assert_eq!(result, "[e3:Cy] and friends.");
}

#[test]
fn resolved_view_uses_character_offsets() {
    // "José" and the em dash are multi-byte; offsets count characters.
    let text = "José Núñez — the driver — met Ana.";
    let mentions =
        vec![mention("m1", "PERSON", "José Núñez", 0, 10), mention("m2", "PERSON", "Ana", 30, 33)];
    let result = build_resolved_text_view(
        text,
        0,
        &mentions,
        &resolved(&[("m1", "e1"), ("m2", "e2")]),
        &no_canonical(),
    );
    assert_eq!(result, "[e1:José Núñez] — the driver — met [e2:Ana].");
}

#[test]
fn resolved_view_mention_ending_at_chunk_end() {
    let mentions = vec![mention("m1", "ORG", "Acme", 6, 10)];
    let result = build_resolved_text_view(
        "Hired Acme",
        0,
        &mentions,
        &resolved(&[("m1", "e1")]),
        &no_canonical(),
    );
    assert_eq!(result, "Hired [e1:Acme]");
}

// --- role terms ---

fn plaintiff_lookup() -> EntityLookup {
    lookup(&[("plaintiff", "e1", "PER"), ("maria lopez", "e1", "PER")])
}

#[test]
fn rewrite_subject_role_to_canonical() {
    let mut fact = fact("Plaintiff", "filed", "the complaint");
    let canonical = canonical(&[("e1", "Maria Lopez", "PER", &["Plaintiff"])]);
    let linked = linker_with_roles().link_fact(&mut fact, &plaintiff_lookup(), &canonical, None);
    assert_eq!(ids(&linked), (Some("e1"), None));
    assert_eq!(fact.assertion_triple.subject, "Maria Lopez");
}

#[test]
fn rewrite_object_role_to_canonical() {
    let lookup = lookup(&[("defendant", "e2", "PER"), ("john smith", "e2", "PER")]);
    let canonical = canonical(&[("e2", "John Smith", "PER", &["Defendant"])]);
    let mut fact = fact("the notice", "was served to", "Defendant");
    let linked = linker_with_roles().link_fact(&mut fact, &lookup, &canonical, None);
    assert_eq!(linked.1.as_ref().map(EntityId::as_str), Some("e2"));
    assert_eq!(fact.assertion_triple.object, "John Smith");
}

#[test]
fn rewrite_both_subject_and_object_roles() {
    let lookup = lookup(&[
        ("plaintiff", "e1", "PER"),
        ("maria lopez", "e1", "PER"),
        ("defendant", "e2", "PER"),
        ("john smith", "e2", "PER"),
    ]);
    let canonical = canonical(&[
        ("e1", "Maria Lopez", "PER", &["Plaintiff"]),
        ("e2", "John Smith", "PER", &["Defendant"]),
    ]);
    let mut fact = fact("Plaintiff", "sued", "Defendant");
    let linked = linker_with_roles().link_fact(&mut fact, &lookup, &canonical, None);
    assert_eq!(ids(&linked), (Some("e1"), Some("e2")));
    assert_eq!(fact.assertion_triple.subject, "Maria Lopez");
    assert_eq!(fact.assertion_triple.object, "John Smith");
}

#[test]
fn no_rewrite_for_non_role_terms() {
    let lookup = lookup(&[("maria lopez", "e1", "PER"), ("john smith", "e2", "PER")]);
    let canonical =
        canonical(&[("e1", "Maria Lopez", "PER", &[]), ("e2", "John Smith", "PER", &[])]);
    let mut fact = fact("Maria Lopez", "met", "John Smith");
    let linked = linker_with_roles().link_fact(&mut fact, &lookup, &canonical, None);
    assert_eq!(ids(&linked), (Some("e1"), Some("e2")));
    assert_eq!(fact.assertion_triple.subject, "Maria Lopez");
    assert_eq!(fact.assertion_triple.object, "John Smith");
}

#[test]
fn no_rewrite_when_role_not_linked() {
    let mut fact = fact("Plaintiff", "filed", "the complaint");
    let linked =
        linker_with_roles().link_fact(&mut fact, &EntityLookup::default(), &no_canonical(), None);
    assert_eq!(ids(&linked), (None, None));
    assert_eq!(fact.assertion_triple.subject, "Plaintiff");
}

#[test]
fn no_rewrite_without_role_config() {
    let canonical = canonical(&[("e1", "Maria Lopez", "PER", &["Plaintiff"])]);
    let mut fact = fact("Plaintiff", "filed", "the complaint");
    let linked = linker().link_fact(&mut fact, &plaintiff_lookup(), &canonical, None);
    assert_eq!(linked.0.as_ref().map(EntityId::as_str), Some("e1"));
    assert_eq!(fact.assertion_triple.subject, "Plaintiff");
}

#[test]
fn linked_role_without_canonical_entry_keeps_its_text() {
    let mut fact = fact("Plaintiff", "filed", "the complaint");
    let linked =
        linker_with_roles().link_fact(&mut fact, &plaintiff_lookup(), &no_canonical(), None);
    assert_eq!(linked.0.as_ref().map(EntityId::as_str), Some("e1"));
    assert_eq!(fact.assertion_triple.subject, "Plaintiff");
}

// --- role map (extension) ---

fn role_map(pairs: &[(&str, &str)]) -> IndexMap<String, EntityId> {
    pairs.iter().map(|(role, id)| ((*role).to_owned(), EntityId::new(*id))).collect()
}

#[test]
fn role_map_links_unlinked_role_terms() {
    let roles = role_map(&[("PLAINTIFF", "e1"), ("DEFENDANT", "e2")]);
    let canonical =
        canonical(&[("e1", "Maria Lopez", "PER", &[]), ("e2", "Daniel Reed", "PER", &[])]);
    let mut fact = fact("Plaintiff", "sued", "defendant");
    let linked = linker_with_roles().link_fact(
        &mut fact,
        &EntityLookup::default(),
        &canonical,
        Some(&roles),
    );
    assert_eq!(ids(&linked), (Some("e1"), Some("e2")));
    assert_eq!(fact.assertion_triple.subject, "Maria Lopez");
    assert_eq!(fact.assertion_triple.object, "Daniel Reed");
}

#[test]
fn role_map_normalises_plural_role_terms() {
    let plural = "Plaintiffs";
    assert_eq!(roles().normalize_role_term(plural), "PLAINTIFF");
    let roles = role_map(&[("PLAINTIFF", "e1")]);
    let mut fact = fact(plural, "filed", "complaint");
    let linked = linker_with_roles().link_fact(
        &mut fact,
        &EntityLookup::default(),
        &canonical(&[("e1", "Maria Lopez", "PER", &[])]),
        Some(&roles),
    );
    assert_eq!(linked.0.as_ref().map(EntityId::as_str), Some("e1"));
    assert_eq!(fact.assertion_triple.subject, "Maria Lopez");
}

#[test]
fn role_map_link_without_canonical_entry_keeps_text() {
    let roles = role_map(&[("PLAINTIFF", "e1")]);
    let mut fact = fact("Plaintiff", "filed", "complaint");
    let linked = linker_with_roles().link_fact(
        &mut fact,
        &EntityLookup::default(),
        &no_canonical(),
        Some(&roles),
    );
    assert_eq!(linked.0.as_ref().map(EntityId::as_str), Some("e1"));
    assert_eq!(fact.assertion_triple.subject, "Plaintiff");
}

#[test]
fn name_lookup_takes_precedence_over_role_map() {
    let roles = role_map(&[("PLAINTIFF", "e_role")]);
    let canonical =
        canonical(&[("e1", "Maria Lopez", "PER", &[]), ("e_role", "Other", "PER", &[])]);
    let mut fact = fact("Plaintiff", "filed", "complaint");
    let linked =
        linker_with_roles().link_fact(&mut fact, &plaintiff_lookup(), &canonical, Some(&roles));
    assert_eq!(linked.0.as_ref().map(EntityId::as_str), Some("e1"));
    assert_eq!(fact.assertion_triple.subject, "Maria Lopez");
}

#[test]
fn role_map_needs_a_role_vocabulary_and_a_role_term() {
    let roles = role_map(&[("PLAINTIFF", "e1"), ("MARIA LOPEZ", "e1")]);
    let canonical = canonical(&[("e1", "Maria Lopez", "PER", &[])]);
    let mut without_roles = fact("Plaintiff", "filed", "complaint");
    let linked =
        linker().link_fact(&mut without_roles, &EntityLookup::default(), &canonical, Some(&roles));
    assert_eq!(ids(&linked), (None, None));
    // Not a role term: the role map is not consulted even when it has the key.
    let mut not_a_role = fact("Maria Lopez", "filed", "complaint");
    let linked = linker_with_roles().link_fact(
        &mut not_a_role,
        &EntityLookup::default(),
        &canonical,
        Some(&roles),
    );
    assert_eq!(ids(&linked), (None, None));
    // A role term missing from the map stays unlinked.
    let mut unbound = fact("Defendant", "denied", "liability");
    let linked = linker_with_roles().link_fact(
        &mut unbound,
        &EntityLookup::default(),
        &canonical,
        Some(&roles),
    );
    assert_eq!(ids(&linked), (None, None));
    assert_eq!(unbound.assertion_triple.subject, "Defendant");
}

// --- linking stats ---

#[test]
fn record_link_counts_like_python() {
    let mut stats = FactLinkingStats::default();
    let e1 = EntityId::new("e1");
    let e2 = EntityId::new("e2");
    record_link(&mut stats, Some(&e1), Some(&e2));
    record_link(&mut stats, Some(&e1), None);
    record_link(&mut stats, None, Some(&e2));
    record_link(&mut stats, None, None);
    assert_eq!(
        stats,
        FactLinkingStats {
            total_facts: 4,
            linked_subjects: 2,
            linked_objects: 2,
            linked_both: 1,
            linked_either: 3,
        }
    );
}

// --- prompt entities ---

fn exclusion() -> IndexSet<String> {
    mappings().normalization.alias_exclusion.clone()
}

#[test]
fn prompt_entities_use_canonical_info_and_surface_forms() {
    let mentions = vec![
        mention("m1", "PERSON", "Lopez", 0, 5),
        mention("m2", "PERSON", "Ms. Lopez", 10, 19),
        mention("m3", "PERSON", "Lopez", 30, 35),
        mention("m4", "ORG", "Acme", 40, 44),
    ];
    let canonical = canonical(&[("e1", "Maria Lopez", "PER", &["M. Lopez", "Lopez"])]);
    let entities = prompt_entities(
        &mentions,
        &resolved(&[("m1", "e1"), ("m2", "e1"), ("m3", "e1"), ("m4", "e2")]),
        &canonical,
        &exclusion(),
        Some(&roles()),
    );
    assert_eq!(entities.len(), 2);
    // Sorted by name: "Acme" (mention fallback) before "Maria Lopez".
    assert_eq!(entities[0].id.as_str(), "e2");
    assert_eq!(entities[0].name, "Acme");
    assert_eq!(entities[0].entity_type.as_deref(), Some("ORG"));
    assert!(entities[0].aliases.is_empty(), "{:?}", entities[0].aliases);
    assert_eq!(entities[1].id.as_str(), "e1");
    assert_eq!(entities[1].name, "Maria Lopez");
    assert_eq!(entities[1].entity_type.as_deref(), Some("PER"));
    // Surface forms in first-seen order, then ED aliases not already present.
    assert_eq!(entities[1].aliases, vec!["Lopez", "Ms. Lopez", "M. Lopez"]);
}

#[test]
fn prompt_entities_filter_pronouns_role_terms_and_canonical_name() {
    let exclusion = exclusion();
    assert!(exclusion.contains("she"), "alias_exclusion lists pronouns");
    let mentions = vec![
        mention("m1", "PER", "Maria Lopez", 0, 11),
        mention("m2", "PER", "She", 20, 23),
        mention("m3", "PER", "Plaintiff", 30, 39),
        mention("m4", "PER", "Ms. Lopez", 40, 49),
    ];
    let canonical = canonical(&[("e1", "Maria Lopez", "PER", &["", "Plaintiff", " HER "])]);
    let resolved = resolved(&[("m1", "e1"), ("m2", "e1"), ("m3", "e1"), ("m4", "e1")]);
    let entities = prompt_entities(&mentions, &resolved, &canonical, &exclusion, Some(&roles()));
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].aliases, vec!["Ms. Lopez"]);

    // Without a role vocabulary the role term stays.
    let entities = prompt_entities(&mentions, &resolved, &canonical, &exclusion, None);
    assert_eq!(entities[0].aliases, vec!["Plaintiff", "Ms. Lopez"]);
}

#[test]
fn prompt_entities_without_canonical_use_first_mention() {
    let mentions = vec![
        mention("m1", "", "the truck", 0, 9),
        mention("m2", "VEHICLE", "Freightliner", 20, 32),
        mention("m3", "PERSON", "Unresolved", 40, 50),
    ];
    let entities = prompt_entities(
        &mentions,
        &resolved(&[("m1", "e9"), ("m2", "e9")]),
        &no_canonical(),
        &exclusion(),
        None,
    );
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].name, "the truck");
    assert_eq!(entities[0].entity_type, None);
    assert_eq!(entities[0].aliases, vec!["Freightliner"]);
}

#[test]
fn prompt_entities_sort_is_stable_for_equal_names() {
    let mentions = vec![
        mention("m1", "PER", "Alex", 0, 4),
        mention("m2", "ORG", "Alex", 10, 14),
        mention("m3", "PER", "Zed", 20, 23),
        mention("m4", "PER", "Bea", 30, 33),
    ];
    let entities = prompt_entities(
        &mentions,
        &resolved(&[("m1", "e1"), ("m2", "e2"), ("m3", "e3"), ("m4", "e4")]),
        &no_canonical(),
        &exclusion(),
        None,
    );
    let order: Vec<&str> = entities.iter().map(|e| e.id.as_str()).collect();
    assert_eq!(order, vec!["e1", "e2", "e4", "e3"]);
}
