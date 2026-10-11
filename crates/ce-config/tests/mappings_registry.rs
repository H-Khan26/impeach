//! Mappings registry loading and helper behaviour.

use std::path::PathBuf;

use ce_config::mappings::{MappingsRegistry, Normalization, Roles, MAPPING_FILES, ROLES_PATH};
use ce_config::ConfigError;
use ce_core::vocab::TypeVocabulary;
use pretty_assertions::assert_eq;

fn assets_mappings_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/config/mappings")
}

fn registry() -> &'static MappingsRegistry {
    MappingsRegistry::load().expect("mappings load")
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

#[test]
fn load_is_cached() {
    assert!(std::ptr::eq(registry(), registry()));
}

#[test]
fn all_ten_files_load_from_disk_and_embedded() {
    assert_eq!(MAPPING_FILES.len(), 10);
    let from_dir = MappingsRegistry::from_dir(&assets_mappings_dir()).expect("from_dir");
    let embedded = MappingsRegistry::embedded().expect("embedded");
    assert_eq!(from_dir, embedded);
    assert_eq!(&embedded, registry());
    assert!(!embedded.taxonomy.value_types.is_empty());
    assert!(!embedded.entity_types.entity_types.is_empty());
    assert!(!embedded.relationship_types.relationship_types.is_empty());
    assert!(!embedded.detection_patterns.patterns.is_empty());
    assert!(!embedded.detector_config.high_precision_labels.is_empty());
    assert!(!embedded.normalization.stopwords.is_empty());
    assert_eq!(embedded.quality_filters.min_mention_length, 3);
    assert!(!embedded.name_matching.nickname_mappings.is_empty());
    assert!(!embedded.gliner2.entity_label_sets.is_empty());
    assert!(!embedded.roles.definition_patterns.is_empty());
}

#[test]
fn missing_file_or_unknown_key_fails() {
    let dir = tempfile::tempdir().expect("tempdir");
    for file in MAPPING_FILES {
        std::fs::copy(assets_mappings_dir().join(file), dir.path().join(file)).expect("copy");
    }
    MappingsRegistry::from_dir(dir.path()).expect("copied files load");

    let gliner = dir.path().join("gliner2.yaml");
    let text = std::fs::read_to_string(&gliner).expect("read");
    std::fs::write(&gliner, text.replace("defaults:\n", "defaults:\n  bogus: x\n")).expect("write");
    let err = MappingsRegistry::from_dir(dir.path()).expect_err("unknown key");
    assert!(err.to_string().contains("defaults: unknown field `bogus`"), "{err}");

    std::fs::remove_file(&gliner).expect("remove");
    assert!(matches!(MappingsRegistry::from_dir(dir.path()), Err(ConfigError::Io { .. })));
}

#[test]
fn roles_load_standalone() {
    let roles = Roles::load(ROLES_PATH).expect("roles.yaml");
    assert_eq!(&roles, &registry().roles);
    assert_eq!(roles.role_seed_list.len(), 4);
    assert!(roles.role_seed_list["litigation"].contains(&"Plaintiff".to_owned()));
    assert_eq!(roles.definition_patterns.len(), 6);
    assert_eq!(roles.definition_patterns[0].name, "parenthetical_quoted");
    assert_eq!(roles.definition_patterns[0].confidence, 0.95);
    assert_eq!(roles.role_normalization["PLAINTIFFS"], "PLAINTIFF");
    assert!(roles.exclusions.contains(&"Board of Directors".to_owned()));
    assert_eq!(roles.min_confidence, 0.75);
    for pattern in &roles.definition_patterns {
        let re = regex::Regex::new(&pattern.pattern)
            .unwrap_or_else(|e| panic!("{} compiles: {e}", pattern.name));
        assert!(re.capture_names().flatten().any(|n| n == "defined_term"), "{}", pattern.name);
    }
    let caps = regex::Regex::new(&roles.definition_patterns[0].pattern)
        .expect("compiles")
        .captures("Maria Lopez (\"Plaintiff\")")
        .expect("matches");
    assert_eq!(&caps["entity_name"], "Maria Lopez");
    assert_eq!(&caps["defined_term"], "Plaintiff");
}

// ---------------------------------------------------------------------------
// YAML 1.1 boolean words
// ---------------------------------------------------------------------------

#[test]
fn yaml_boolean_words_load_as_strings() {
    let yaml = "stopwords:\n  - on\n  - off\n  - yes\n  - no\n  - y\n  - n\n  - \"on\"\n  - true\n  - 12\nalias_exclusion: [On, NO, Yes]\n";
    let n: Normalization = yaml_serde::from_str(yaml).expect("parses");
    let stopwords: Vec<&str> = n.stopwords.iter().map(String::as_str).collect();
    assert_eq!(stopwords, ["on", "off", "yes", "no", "y", "n", "true", "12"]);
    let aliases: Vec<&str> = n.alias_exclusion.iter().map(String::as_str).collect();
    assert_eq!(aliases, ["On", "NO", "Yes"]);

    let shipped = &registry().normalization;
    for word in ["on", "no", "yes", "the", "a"] {
        assert!(shipped.stopwords.contains(word), "{word}");
    }
    let tokens = shipped.tokenize("yes on the motion no objection");
    assert_eq!(tokens.len(), 2, "{tokens:?}");
}

#[test]
fn duplicate_yaml_keys_keep_the_last_value() {
    assert_eq!(registry().name_matching.nickname_mappings["teddy"], "theodore");
}

// ---------------------------------------------------------------------------
// Taxonomy
// ---------------------------------------------------------------------------

#[test]
fn taxonomy_types_and_defaults() {
    let t = &registry().taxonomy;
    for v in [
        "NUMERIC",
        "TEMPORAL_POINT",
        "TEMPORAL_INTERVAL",
        "LOCATION",
        "BOOLEAN",
        "CATEGORICAL",
        "ENTITY_REF",
        "TEXT",
    ] {
        assert!(t.is_valid_value_type(v), "{v}");
    }
    assert!(!t.is_valid_value_type("INVALID"));
    for p in
        ["TEMPORAL", "QUANTITATIVE", "LOCATION", "ATTRIBUTE", "RELATION", "EXISTENCE", "MODALITY"]
    {
        assert!(t.is_valid_property_type(p), "{p}");
    }
    assert!(!t.is_valid_property_type("INVALID"));
    assert_eq!(t.defaults.temporal_context, "current_incident");
    assert_eq!(t.default_value_type(), "TEXT");
    assert_eq!(t.default_property_type(), "ATTRIBUTE");
    assert_eq!(t.temporal_contexts, ["current_incident", "prior_incident", "general_claim"]);

    let vocab: &dyn TypeVocabulary = t;
    assert!(vocab.is_valid_value_type("ENTITY_REF"));
    assert!(!vocab.is_valid_property_type("EXISTS"));
    assert_eq!(vocab.default_value_type(), "TEXT");
    assert_eq!(vocab.default_property_type(), "ATTRIBUTE");
}

// ---------------------------------------------------------------------------
// Entity types
// ---------------------------------------------------------------------------

#[test]
fn entity_types_loaded() {
    let e = &registry().entity_types;
    for t in ["PER", "ORG", "CASE"] {
        assert!(e.entity_types.iter().any(|x| x == t), "{t}");
    }
    assert_eq!(e.default_entity_type(), "OTHER");
    assert_eq!(e.default_domain(), "general");
    assert_eq!(e.model_mappings["gliner_to_entity_type"]["person"], "PER");
    assert_eq!(e.entity_type_to_domain["CASE"], "legal");
    let skip: Vec<&str> = e.skip_ed_types.iter().map(String::as_str).collect();
    assert_eq!(skip, ["DATE", "TIME", "MONEY"]);
}

#[test]
fn map_model_label() {
    let e = &registry().entity_types;
    assert_eq!(e.map_model_label("gliner", "person"), "PER");
    assert_eq!(e.map_model_label("gliner", "organization"), "ORG");
    assert_eq!(e.map_model_label("gliner", "company"), "ORG");
    assert_eq!(e.map_model_label("gliner", "location"), "LOC");
    assert_eq!(e.map_model_label("gliner", "unknown_type"), "unknown_type");
    assert_eq!(e.map_model_label("unknown_model", "person"), "person");
}

#[test]
fn get_domain() {
    let e = &registry().entity_types;
    assert_eq!(e.get_domain("PER"), "person");
    assert_eq!(e.get_domain("ORG"), "organization");
    assert_eq!(e.get_domain("CASE"), "legal");
    assert_eq!(e.get_domain("UNKNOWN"), "general");
}

#[test]
fn normalize_label() {
    let e = &registry().entity_types;
    assert_eq!(e.normalize_label(""), "OTHER");
    assert_eq!(e.normalize_label("PER"), "PER");
    assert_eq!(e.normalize_label("MATTER"), "MATTER");
    assert_eq!(e.normalize_label("PERSON"), "PER");
    assert_eq!(e.normalize_label("GPE"), "LOC");
    assert_eq!(e.normalize_label("FAC"), "FACILITY");
    assert_eq!(e.normalize_label("person"), "PER");
    assert_eq!(e.normalize_label("legal case"), "CASE");
    assert_eq!(e.normalize_label("SOMETHING"), "SOMETHING");
}

#[test]
fn routing_helpers() {
    let e = &registry().entity_types;
    assert!(e.rule_first_label_set().contains("CASE"));
    assert!(e.rule_first_label_set().contains("JURISDICTION"));
    assert!(e.hybrid_label_set().contains("PER"));
    assert!(!e.hybrid_label_set().contains("CASE"));
    assert_eq!(e.get_rule_route("MATTER"), Some("case"));
    assert_eq!(e.get_rule_route("REGULATION"), Some("statute"));
    assert_eq!(e.get_rule_route("EXHIBIT"), Some("exhibit"));
    assert_eq!(e.get_rule_route("PER"), None);
}

#[test]
fn type_compatibility_groups() {
    let e = &registry().entity_types;
    for (a, b) in [("ORG", "LOC"), ("ORG", "FACILITY"), ("LOC", "FACILITY"), ("GPE", "ORG")] {
        assert!(e.are_types_compatible(Some(a), Some(b)), "{a} {b}");
        assert!(e.are_types_compatible(Some(b), Some(a)), "{b} {a}");
    }
    for (a, b) in [
        ("PER", "ORG"),
        ("PERSON", "ORGANIZATION"),
        ("PER", "LOC"),
        ("PRODUCT", "LOC"),
        ("DATE", "MONEY"),
        ("ORG", "DATE"),
    ] {
        assert!(!e.are_types_compatible(Some(a), Some(b)), "{a} {b}");
    }
    assert!(e.are_types_compatible(None, Some("PER")));
    assert!(e.are_types_compatible(Some(""), Some("PER")));
    assert!(e.are_types_compatible(Some("OTHER"), Some("PER")));
    assert!(e.are_types_compatible(Some("CASE"), Some("PER")));

    let groups = &e.incompatible_type_groups;
    let all: Vec<&String> = groups.iter().flatten().collect();
    let unique: std::collections::HashSet<&&String> = all.iter().collect();
    assert_eq!(all.len(), unique.len(), "each type is in at most one group");
}

// ---------------------------------------------------------------------------
// Relationship types
// ---------------------------------------------------------------------------

#[test]
fn relationship_types_loaded() {
    let r = &registry().relationship_types;
    for t in ["FOUNDED_BY", "EMPLOYED_BY", "LOCATED_IN"] {
        assert!(r.relationship_types.iter().any(|x| x == t), "{t}");
    }
}

#[test]
fn relationship_classification() {
    let r = &registry().relationship_types;
    assert_eq!(r.get_classification("hired_in"), Some("temporal"));
    assert_eq!(r.get_classification("started_on"), Some("temporal"));
    assert_eq!(r.get_classification("cost_amount"), Some("quantitative"));
    assert_eq!(r.get_classification("ceo_of"), Some("role"));
    assert_eq!(r.get_classification("unknown_rel"), None);
}

#[test]
fn relationship_compatibility() {
    let r = &registry().relationship_types;
    assert!(r.are_compatible("hired_in", "started_on"));
    assert!(r.are_compatible("cost", "paid"));
    assert!(!r.are_compatible("hired_in", "cost"));
    assert!(!r.are_compatible("ceo_of", "started_on"));
    assert!(r.are_compatible("unknown1", "unknown2"));
    assert!(r.are_compatible("hired_in", "unknown"));
}

// ---------------------------------------------------------------------------
// Detection patterns
// ---------------------------------------------------------------------------

#[test]
fn every_shipped_pattern_compiles() {
    let d = &registry().detection_patterns;
    assert!(d.patterns.contains_key("docket_entry") && d.patterns.contains_key("exhibit"));
    for name in d.patterns.keys().chain(d.ed_patterns.keys()) {
        assert!(d.get_compiled_pattern(name).is_some(), "{name} compiles");
    }
    for name in d.get_ordered_patterns() {
        assert!(d.patterns.contains_key(name), "{name} in detection_order is defined");
    }
}

#[test]
fn compiled_patterns_are_cached() {
    let d = &registry().detection_patterns;
    let a = d.get_compiled_pattern("docket_entry").expect("docket_entry");
    let b = d.get_compiled_pattern("docket_entry").expect("docket_entry");
    assert!(std::ptr::eq(a, b));
    assert!(d.get_compiled_pattern("unknown_pattern").is_none());
}

#[test]
fn pattern_entity_types() {
    let d = &registry().detection_patterns;
    assert_eq!(d.get_entity_type("docket_entry"), Some("DOCKET_ENTRY"));
    assert_eq!(d.get_entity_type("exhibit"), Some("EXHIBIT_ID"));
    assert_eq!(d.get_entity_type("unknown"), None);
    assert_eq!(d.get_entity_type("case_docket"), None);
    assert_eq!(d.ed_patterns["statute_usc"].capture_groups[&2], "section");
}

#[test]
fn docket_and_exhibit_patterns_match() {
    let d = &registry().detection_patterns;
    let docket = d.get_compiled_pattern("docket_entry").expect("docket");
    for (text, expected) in [
        ("Case No. 2:23-cv-01492", true),
        ("No. 21-55555", true),
        ("Case 1:20-cr-123", true),
        ("Random text without docket", false),
    ] {
        assert_eq!(docket.is_match(text), expected, "{text}");
    }
    let exhibit = d.get_compiled_pattern("exhibit").expect("exhibit");
    for (text, expected) in [
        ("Exhibit 12", true),
        ("Ex. 12A", true),
        ("Exh. 3", true),
        ("Deposition Exhibit 5", true),
        ("Random text without references", false),
    ] {
        assert_eq!(exhibit.is_match(text), expected, "{text}");
    }
    let caps = d
        .get_compiled_pattern("statute_usc")
        .expect("statute_usc")
        .captures("see 42 U.S.C. § 1983(a)")
        .expect("matches");
    assert_eq!((&caps[1], &caps[2], &caps[3]), ("42", "1983", "a"));
}

// ---------------------------------------------------------------------------
// Detector config
// ---------------------------------------------------------------------------

#[test]
fn source_priority() {
    let c = &registry().detector_config;
    assert!(c.high_precision_labels.contains("CASE_CITATION"));
    assert!(c.high_precision_labels.contains("EXHIBIT"));
    assert!(c.get_source_priority("CASE_CITATION", "regex_detector") >= 1000);
    assert!(
        c.get_source_priority("EXHIBIT", "regex_detector")
            > c.get_source_priority("EXHIBIT", "spacy_ner")
    );
    assert_eq!(c.get_source_priority("EXHIBIT", "regex_detector"), 1000 + 20);
    assert_eq!(c.get_source_priority("PERSON", "gliner2_detector"), 20);
    assert_eq!(c.get_source_priority("UNLISTED", "gliner2"), 2);
    assert_eq!(c.get_source_priority("UNLISTED", "unknown"), 0);
    assert_eq!(c.get_source_priority("UNLISTED", ""), 0);
    assert!(c.is_high_precision_label("CASE_CITATION"));
    assert!(c.is_high_precision_label("EXHIBIT"));
    assert!(!c.is_high_precision_label("PERSON"));
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

#[test]
fn normalize_entity_name() {
    let n = &registry().normalization;
    for (input, expected) in [
        ("The Acme Corp", "acme"),
        ("A Company", "company"),
        ("An Organization", "organization"),
        ("Acme Inc", "acme"),
        ("Acme Inc.", "acme"),
        ("Acme Corporation", "acme"),
        ("Acme LLC", "acme"),
        ("The Acme Corporation", "acme"),
        ("", ""),
        ("   ", ""),
    ] {
        assert_eq!(n.normalize_entity_name(input), expected, "{input:?}");
    }
}

#[test]
fn tokenize() {
    let n = &registry().normalization;
    let tokens = n.tokenize("The quick brown fox");
    assert!(tokens.contains("quick") && tokens.contains("brown") && tokens.contains("fox"));
    assert!(!tokens.contains("the"));
    let tokens = n.tokenize("I am a big dog");
    assert!(tokens.contains("big") && tokens.contains("dog"));
    assert!(!tokens.contains("i") && !tokens.contains("am"));
    assert!(n.tokenize("").is_empty());
    let tokens = n.tokenize("Case 21-CV-0042, x");
    let mut sorted: Vec<&str> = tokens.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    assert_eq!(sorted, ["0042", "21", "case", "cv"]);
}

#[test]
fn alias_exclusion_holds_pronouns() {
    let aliases = &registry().normalization.alias_exclusion;
    for word in ["he", "she", "they", "yours"] {
        assert!(aliases.contains(word), "{word}");
    }
}

// ---------------------------------------------------------------------------
// Quality filters
// ---------------------------------------------------------------------------

#[test]
fn low_quality_mentions() {
    let q = &registry().quality_filters;
    assert!(q.is_low_quality("AB", None));
    assert!(!q.is_low_quality("ABC", None));
    assert!(q.is_low_quality("Plainti", None));
    assert!(q.is_low_quality("defenda", None));
    assert!(!q.is_low_quality("Plaintiff", None));
    assert!(q.is_low_quality("John", Some(0.5)));
    assert!(!q.is_low_quality("John", Some(0.9)));
    assert!(!q.is_low_quality("John Smith", Some(0.5)));
    assert!(q.is_low_quality("...", None));
    assert!(q.is_low_quality("e.g.", None));
    assert!(q.is_low_quality("pp. 12", None));
    assert!(!q.is_low_quality("Exhibit pp. 12", None));
    assert!(q.is_low_quality("  AB  ", None));
}

#[test]
fn normalize_mention_name() {
    let q = &registry().quality_filters;
    assert_eq!(q.normalize_mention_name("John\nSmith"), "John Smith");
    assert_eq!(q.normalize_mention_name("John   Smith"), "John Smith");
    assert_eq!(q.normalize_mention_name("  John Smith  "), "John Smith");
}

// ---------------------------------------------------------------------------
// Name matching and GLiNER2
// ---------------------------------------------------------------------------

#[test]
fn name_matching_values() {
    let m = &registry().name_matching;
    assert!(m.person_titles.contains("dr") && m.person_titles.contains("esq"));
    assert_eq!(m.nickname_mappings["bob"], "robert");
    assert!(m.org_suffixes.contains("llc") && m.org_prefixes.contains("the"));
    assert_eq!(m.org_acronyms["fbi"], "federal bureau of investigation");
    assert_eq!(m.exact_match_threshold(), 0.98);
    assert_eq!(m.subset_match_base(), 0.80);
    assert_eq!(m.subset_match_bonus(), 0.15);
    assert_eq!(m.partial_overlap_base(), 0.60);
    assert_eq!(m.partial_overlap_bonus(), 0.30);
    assert_eq!(m.min_word_length(), 2);
    assert_eq!(m.thresholds.min_fuzzy_match_length, 4);
    assert!(m.settings.strip_titles && m.settings.normalize_nicknames);
}

#[test]
fn gliner2_label_sets() {
    let g = &registry().gliner2;
    let entities = g.get_entity_label_set("legal_entities").expect("legal_entities");
    assert_eq!(entities.len(), 18);
    assert_eq!(entities[0], "person");
    assert_eq!(g.get_entity_label_set("minimal").expect("minimal").len(), 4);
    assert_eq!(g.get_relation_label_set("legal_relations").expect("relations").len(), 10);
    match g.get_entity_label_set("nope") {
        Err(ConfigError::UnknownLabelSet { kind, name, available }) => {
            assert_eq!((kind, name.as_str()), ("Entity", "nope"));
            assert_eq!(available, ["legal_entities", "minimal"]);
        }
        other => panic!("expected UnknownLabelSet, got {other:?}"),
    }
    assert!(g.get_relation_label_set("nope").is_err());
    assert_eq!(g.relation_to_relationship_type["works_for"], "EMPLOYED_BY");
    assert_eq!(g.gliner2_to_entity_type["lawyer"], "PER");
    assert_eq!(g.defaults.entity_label_set, "legal_entities");
}
