//! Defined-term extraction and role-term helpers. Expected values come from the Python
//! `DefinitionExtractor` / `RoleConfig` run on the same inputs.

use ce_config::mappings::{DefinitionPattern as PatternConfig, Roles};
use ce_core::processing::DefinedTerm;
use ce_core::{EntityId, Mention, MentionId, Metadata, Span};
use ce_ner::definitions::{DefinitionExtractor, RoleConfig};
use indexmap::IndexMap;
use pretty_assertions::assert_eq;
use serde_json::{json, Value};

fn shipped() -> RoleConfig {
    RoleConfig::load(None).expect("the shipped roles.yaml loads")
}

fn extractor() -> DefinitionExtractor {
    DefinitionExtractor::new(shipped())
}

fn mention(id: &str, name: &str, start: usize, end: usize) -> Mention {
    Mention::new(id, "PERSON", name, Span::new(start, end))
}

fn mention_with(id: &str, start: usize, metadata: &Value) -> Mention {
    let mut mention = mention(id, id, start, start + 3);
    mention.metadata = metadata.as_object().cloned().unwrap_or_default();
    mention
}

fn term(
    id: &str,
    role: &str,
    original: &str,
    span: (usize, usize),
    conf: f64,
    method: &str,
) -> DefinedTerm {
    DefinedTerm {
        entity_mention_id: MentionId::new(id),
        defined_term: role.to_owned(),
        original_term: original.to_owned(),
        span: Span::new(span.0, span.1),
        confidence: conf,
        extraction_method: method.to_owned(),
    }
}

#[test]
fn shipped_config_loads_every_category_and_pattern() {
    let config = shipped();
    assert_eq!(config.role_seed_terms().len(), 41);
    assert!(config.role_seed_terms().contains("THIRD-PARTY DEFENDANT"));
    let names: Vec<&str> = config.definition_patterns().iter().map(|p| p.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "parenthetical_quoted",
            "parenthetical_with_article",
            "hereinafter_quoted",
            "hereinafter_unquoted",
            "hereafter_quoted",
            "defined_as"
        ]
    );
    assert!((config.min_confidence() - 0.75).abs() < f64::EPSILON);
    assert_eq!(config.exclusions().len(), 6);
}

#[test]
fn custom_role_config_path_is_resolved() {
    let dir = std::env::temp_dir().join(format!("ce-ner-roles-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    std::fs::write(
        dir.join("roles.yaml"),
        "role_seed_list:\n  x: [\"Hero\"]\ndefinition_patterns:\n  - name: \"p\"\n    pattern: '(?P<entity_name>[A-Z][a-z]+) is the (?P<defined_term>[A-Z][a-z]+)'\n",
    )
    .expect("write roles.yaml");
    let config = RoleConfig::load_in(Some(&dir), Some("roles.yaml")).expect("custom roles load");
    assert_eq!(config.role_seed_terms().iter().collect::<Vec<_>>(), ["HERO"]);
    assert!((config.definition_patterns()[0].confidence - 0.8).abs() < f64::EPSILON);
    assert!((config.min_confidence() - 0.75).abs() < f64::EPSILON);

    let extractor = DefinitionExtractor::new(config);
    let terms = extractor.extract_from_text("Ann is the Hero.", &[mention("m1", "Ann", 0, 3)], "d");
    assert_eq!(terms, [term("m1", "HERO", "Hero", (0, 15), 0.8, "pattern:p")]);

    assert!(RoleConfig::load_in(Some(&dir), Some("missing.yaml")).is_err());
    std::fs::write(dir.join("bad.yaml"), "unknown_key: 1\n").expect("write bad.yaml");
    assert!(RoleConfig::load_in(Some(&dir), Some("bad.yaml")).is_err());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn invalid_pattern_is_skipped() {
    let roles = Roles {
        definition_patterns: vec![
            PatternConfig {
                name: "broken".into(),
                pattern: "(?P<entity_name>[A-Z".into(),
                confidence: 0.9,
                description: String::new(),
            },
            PatternConfig {
                name: "ok".into(),
                pattern: "(?P<entity_name>[A-Z]\\w+) as (?P<defined_term>[A-Z]\\w+)".into(),
                confidence: 0.9,
                description: String::new(),
            },
        ],
        ..Roles::default()
    };
    let config = RoleConfig::from_roles(&roles);
    assert_eq!(config.definition_patterns().len(), 1);
    assert_eq!(config.definition_patterns()[0].name, "ok");
}

#[test]
fn pattern_without_named_groups_matches_nothing() {
    let roles = Roles {
        role_seed_list: IndexMap::from([("x".to_owned(), vec!["Buyer".to_owned()])]),
        definition_patterns: vec![PatternConfig {
            name: "nogroups".into(),
            pattern: "Buyer".into(),
            confidence: 0.9,
            description: String::new(),
        }],
        ..Roles::default()
    };
    let extractor = DefinitionExtractor::new(RoleConfig::from_roles(&roles));
    assert!(extractor.extract_from_text("Buyer", &[mention("m1", "Buyer", 0, 5)], "d").is_empty());
}

// Python `TestRoleDetection` (definition helpers of test_coref_canonical_role_handling.py).
#[test]
fn role_term_detection() {
    let config = shipped();
    for name in
        ["Plaintiff", "Defendant", "Manager", "plaintiffs", " Tenants ", "Third-Party Defendant"]
    {
        assert!(config.is_valid_role_term(name), "{name}");
    }
    for name in ["Maria Lopez", "John Smith", "Court"] {
        assert!(!config.is_valid_role_term(name), "{name}");
    }
}

#[test]
fn hybrid_role_name_detection() {
    let config = shipped();
    for name in [
        "Defendant Reed",
        "Plaintiff Smith",
        "Manager Johnson",
        "  defendant reed  ",
        "Third-Party Defendant Smith",
    ] {
        assert!(config.is_hybrid_role_name(name), "{name}");
    }
    for name in ["Defendant", "Maria Lopez", "John Smith", "Defendant Al", "Defendant  Al", ""] {
        assert!(!config.is_hybrid_role_name(name), "{name}");
    }
}

#[test]
fn strip_role_prefix() {
    let config = shipped();
    let cases = [
        ("Defendant Reed", Some("Reed")),
        ("Plaintiff Smith", Some("Smith")),
        ("Manager Johnson", Some("Johnson")),
        ("Maria Lopez", None),
        ("Defendant", None),
        ("defendant  reed ", Some("reed")),
        ("Defendant Al", None),
        ("Third-Party Defendant Smith", Some("Smith")),
        ("Officer Jo", None),
        ("", None),
    ];
    for (name, expected) in cases {
        assert_eq!(config.strip_role_prefix(name), expected, "{name}");
    }
}

#[test]
fn normalization_and_exclusions() {
    let config = shipped();
    for (term, expected) in [
        ("Plaintiffs", "PLAINTIFF"),
        (" lessees ", "LESSEE"),
        ("Court", "COURT"),
        ("agent", "AGENT"),
    ] {
        assert_eq!(config.normalize_role_term(term), expected);
    }
    for (term, expected) in
        [("court", true), ("Board of Directors", true), (" Judge", false), ("Plaintiff", false)]
    {
        assert_eq!(config.is_excluded(term), expected, "{term}");
    }
}

#[test]
fn parenthetical_quoted() {
    let text = r#"Maria Lopez ("Plaintiff") filed suit against Acme Corp."#;
    let mentions = [mention("m1", "Maria Lopez", 0, 11), mention("m2", "Acme Corp", 45, 54)];
    assert_eq!(
        extractor().extract_from_text(text, &mentions, "doc1"),
        [term("m1", "PLAINTIFF", "Plaintiff", (0, 25), 0.95, "pattern:parenthetical_quoted")]
    );
}

#[test]
fn article_and_hereinafter_quoted() {
    let text =
        r#"Acme Corporation (the "Company") hired John Smith, hereinafter "Employee", on May 1."#;
    let mentions = [mention("m1", "Acme Corporation", 0, 16), mention("m2", "John Smith", 39, 49)];
    assert_eq!(
        extractor().extract_from_text(text, &mentions, "doc1"),
        [
            term("m1", "COMPANY", "Company", (0, 32), 0.95, "pattern:parenthetical_with_article"),
            term("m2", "EMPLOYEE", "Employee", (39, 73), 0.9, "pattern:hereinafter_quoted"),
        ]
    );
}

#[test]
fn hereinafter_forms() {
    let mentions = [mention("m1", "Jane Doe", 0, 8)];
    assert_eq!(
        extractor().extract_from_text(
            "Jane Doe, hereinafter referred to as the Tenant, signed the lease.",
            &mentions,
            "doc1"
        ),
        [term("m1", "TENANT", "Tenant", (0, 47), 0.85, "pattern:hereinafter_unquoted")]
    );
    assert_eq!(
        extractor().extract_from_text(
            r#"Jane Doe, hereinafter referred to as "the Tenant", signed."#,
            &mentions,
            "doc1"
        ),
        [term("m1", "TENANT", "Tenant", (0, 49), 0.9, "pattern:hereinafter_quoted")]
    );
}

#[test]
fn hereafter_quoted() {
    let text = r#"Robert Reed (hereafter "Defendant") denied it."#;
    assert_eq!(
        extractor().extract_from_text(text, &[mention("m1", "Robert Reed", 0, 11)], "doc1"),
        [term("m1", "DEFENDANT", "Defendant", (0, 35), 0.9, "pattern:hereafter_quoted")]
    );
}

#[test]
fn defined_as_matches_by_substring() {
    let text = r#""Buyer" defined as Global Traders LLC. The rest follows."#;
    assert_eq!(
        extractor().extract_from_text(text, &[mention("m1", "Global Traders LLC", 19, 37)], "doc1"),
        [term("m1", "BUYER", "Buyer", (0, 56), 0.85, "pattern:defined_as")]
    );
}

#[test]
fn leftmost_match_binds_the_closest_partial_mention() {
    let text = r#"On Monday, Maria Lopez ("Plaintiff") testified. Later Maria Lopez left."#;
    let mentions = [mention("m1", "Maria Lopez", 54, 65), mention("m2", "Maria Lopez", 11, 22)];
    assert_eq!(
        extractor().extract_from_text(text, &mentions, "doc1"),
        [term("m2", "PLAINTIFF", "Plaintiff", (0, 36), 0.95, "pattern:parenthetical_quoted")]
    );
}

#[test]
fn excluded_and_non_role_terms_are_skipped_and_plurals_normalised() {
    let text = r#"Judge Brown ("Court") ruled. Acme Ltd ("Widget") sold. Smith and Jones ("Plaintiffs") sued."#;
    let mentions = [
        mention("m1", "Judge Brown", 0, 11),
        mention("m2", "Acme Ltd", 29, 37),
        mention("m3", "Smith and Jones", 55, 70),
    ];
    assert_eq!(
        extractor().extract_from_text(text, &mentions, "doc1"),
        [term("m3", "PLAINTIFF", "Plaintiffs", (55, 85), 0.95, "pattern:parenthetical_quoted")]
    );
}

#[test]
fn definition_without_matching_mention_is_dropped() {
    let text = r#"Unknown Person ("Plaintiff") appeared."#;
    assert!(extractor()
        .extract_from_text(text, &[mention("m1", "Somebody", 0, 8)], "doc1")
        .is_empty());
    assert!(extractor().extract_from_text(text, &[], "doc1").is_empty());
}

#[test]
fn spans_are_character_offsets() {
    let text = "R\u{e9}sum\u{e9}: Ana Ruiz (\"Tenant\") paid \u{fc}ber rent.";
    assert_eq!(
        extractor().extract_from_text(text, &[mention("m1", "Ana Ruiz", 8, 16)], "doc1"),
        [term("m1", "TENANT", "Tenant", (8, 27), 0.95, "pattern:parenthetical_quoted")]
    );
}

#[test]
fn closest_mention_wins_and_ties_keep_index_order() {
    let ex = extractor();
    let text = r#"Lopez met Maria Lopez ("Plaintiff") and Lopez."#;
    let mentions = [
        mention("m1", "Lopez", 40, 45),
        mention("m2", "Lopez", 0, 5),
        mention("m3", "Maria", 10, 15),
    ];
    assert_eq!(ex.extract_from_text(text, &mentions, "d")[0].entity_mention_id.as_str(), "m2");

    let text = r#"Maria Lopez ("Plaintiff") sued."#;
    let lopez = mention("m1", "Lopez", 20, 25);
    let maria = mention("m2", "Maria", 20, 25);
    let exact_far = mention("m3", "maria lopez ", 40, 51);
    let exact_near = mention("m4", "Maria Lopez", 30, 41);
    let all = [lopez.clone(), maria.clone(), exact_far, exact_near];
    let first = |mentions: &[Mention]| {
        ex.extract_from_text(text, mentions, "d")[0].entity_mention_id.clone()
    };
    assert_eq!(first(&all).as_str(), "m4");
    assert_eq!(first(&[lopez.clone(), maria.clone()]).as_str(), "m1");
    assert_eq!(first(&[maria, lopez]).as_str(), "m2");
}

#[test]
fn metadata_roles() {
    let mentions = [
        mention_with("a", 0, &json!({"role": "Plaintiffs"})),
        mention_with("b", 10, &json!({"role": "", "role_in_doc": "Tenant"})),
        mention_with("c", 20, &json!({"role": "Court"})),
        mention_with("d", 30, &json!({"role_in_doc": "witness"})),
        mention_with("e", 40, &json!({"role": "Victim"})),
        mention_with("f", 50, &json!({})),
        mention_with("g", 60, &json!({"role": " manager "})),
        mention_with("h", 70, &json!({"role": 7, "role_in_doc": "Buyer"})),
        mention_with("i", 80, &json!({"role": null, "role_in_doc": "Buyer"})),
    ];
    assert_eq!(
        extractor().extract_from_metadata(&mentions),
        [
            term("a", "PLAINTIFF", "Plaintiffs", (0, 3), 0.9, "metadata:role"),
            term("b", "TENANT", "Tenant", (10, 13), 0.9, "metadata:role"),
            term("g", "MANAGER", " manager ", (60, 63), 0.9, "metadata:role"),
            term("i", "BUYER", "Buyer", (80, 83), 0.9, "metadata:role"),
        ]
    );
    assert!(extractor()
        .extract_from_metadata(&[mention_with("x", 0, &Value::Object(Metadata::new()))])
        .is_empty());
}

#[test]
fn role_map_skips_low_confidence_and_unresolved_and_keeps_first_binding() {
    let terms = [
        term("m1", "PLAINTIFF", "Plaintiff", (0, 5), 0.95, "pattern:x"),
        term("m2", "DEFENDANT", "Defendant", (0, 5), 0.70, "pattern:x"),
        term("m3", "TENANT", "Tenant", (0, 5), 0.90, "metadata:role"),
        term("m4", "PLAINTIFF", "Plaintiff", (0, 5), 0.95, "pattern:x"),
        term("m5", "PLAINTIFF", "Plaintiff", (0, 5), 0.95, "pattern:x"),
        term("m6", "BUYER", "Buyer", (0, 5), 0.95, "pattern:x"),
        term("m7", "SELLER", "Seller", (0, 5), 0.75, "pattern:x"),
        term("m8", "LESSOR", "Lessor", (0, 5), 0.95, "pattern:x"),
    ];
    let ed: IndexMap<MentionId, EntityId> = [
        ("m1", "e1"),
        ("m2", "e2"),
        ("m3", "e3"),
        ("m4", "e9"),
        ("m5", "e1"),
        ("m6", ""),
        ("m7", "e7"),
    ]
    .into_iter()
    .map(|(m, e)| (MentionId::new(m), EntityId::new(e)))
    .collect();
    let role_map = extractor().build_role_map(&terms, &ed);
    let expected: IndexMap<String, EntityId> =
        [("PLAINTIFF", "e1"), ("TENANT", "e3"), ("SELLER", "e7")]
            .into_iter()
            .map(|(r, e)| (r.to_owned(), EntityId::new(e)))
            .collect();
    assert_eq!(role_map, expected);
}
