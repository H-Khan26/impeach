//! Structured keys and their search terms. Expected values come from the Python
//! `services/ed/structured_keys.py` run on the same inputs with the shipped
//! `detection_patterns.yaml`, its `federal_rule` lookup pointed at the ED pattern (the Python
//! registry returns the one-group detection pattern there and raises `IndexError`).

use impeach_config::mappings::DetectionPatterns;
use impeach_config::MappingsRegistry;
use impeach_ed::structured_keys::{
    case_citation_key_to_search_terms, exhibit_key_to_search_terms, extract_case_citation_key,
    extract_exhibit_key, extract_statute_key, statute_key_to_search_terms,
};
use impeach_ed::{CaseCitationKey, ExhibitKey, FederalRuleKey, KeyPatterns, StatuteKey};
use pretty_assertions::assert_eq;
use serde_json::json;

fn registry_patterns() -> KeyPatterns {
    KeyPatterns::from_mappings(
        &MappingsRegistry::load().expect("embedded mappings load").detection_patterns,
    )
}

/// Input text and the expected key with its search terms.
type ExhibitCase = (&'static str, Option<(ExhibitKey, Vec<&'static str>)>);

/// Input text and the expected key, search terms and normalised key.
type KeyCase<K> = (&'static str, Option<(K, Vec<&'static str>, &'static str)>);

fn exhibit_cases() -> Vec<ExhibitCase> {
    vec![
        (
            "Plaintiff's Exhibit 12A",
            Some((
                ExhibitKey {
                    exhibit_id: "12A".to_owned(),
                    party_attribution: Some("Plaintiff".to_owned()),
                    case_id: Some("CASE-1".to_owned()),
                },
                vec!["Exhibit 12A", "Plaintiff's Exhibit 12A", "Plaintiff Exhibit 12A", "12A"],
            )),
        ),
        (
            "Defendant's Ex. 3",
            Some((
                ExhibitKey {
                    exhibit_id: "3".to_owned(),
                    party_attribution: Some("Defendant".to_owned()),
                    case_id: Some("CASE-1".to_owned()),
                },
                vec!["Exhibit 3", "Defendant's Exhibit 3", "Defendant Exhibit 3", "3"],
            )),
        ),
        (
            "Exh. b-2",
            Some((
                ExhibitKey {
                    exhibit_id: "B-2".to_owned(),
                    party_attribution: None,
                    case_id: Some("CASE-1".to_owned()),
                },
                vec!["Exhibit B-2", "B-2"],
            )),
        ),
        (
            "Deposition Exhibit 5",
            Some((
                ExhibitKey {
                    exhibit_id: "5".to_owned(),
                    party_attribution: None,
                    case_id: Some("CASE-1".to_owned()),
                },
                vec!["Exhibit 5", "5"],
            )),
        ),
        (
            "GOVERNMENT'S EXHIBIT 7",
            Some((
                ExhibitKey {
                    exhibit_id: "7".to_owned(),
                    party_attribution: Some("GOVERNMENT'S".to_owned()),
                    case_id: Some("CASE-1".to_owned()),
                },
                vec!["Exhibit 7", "GOVERNMENT'S's Exhibit 7", "GOVERNMENT'S Exhibit 7", "7"],
            )),
        ),
        (
            "Defense Exhibit A",
            Some((
                ExhibitKey {
                    exhibit_id: "A".to_owned(),
                    party_attribution: Some("Defense".to_owned()),
                    case_id: Some("CASE-1".to_owned()),
                },
                vec!["Exhibit A", "Defense's Exhibit A", "Defense Exhibit A", "A"],
            )),
        ),
        (
            "Plaintiffs Exhibit 9",
            Some((
                ExhibitKey {
                    exhibit_id: "9".to_owned(),
                    party_attribution: None,
                    case_id: Some("CASE-1".to_owned()),
                },
                vec!["Exhibit 9", "9"],
            )),
        ),
        (
            "see exhibit here",
            Some((
                ExhibitKey {
                    exhibit_id: "HERE".to_owned(),
                    party_attribution: None,
                    case_id: Some("CASE-1".to_owned()),
                },
                vec!["Exhibit HERE", "HERE"],
            )),
        ),
        ("nothing", None),
    ]
}

fn statute_cases() -> Vec<KeyCase<StatuteKey>> {
    vec![
        (
            "42 U.S.C. § 1983(a)",
            Some((
                StatuteKey {
                    title: Some("42".to_owned()),
                    section: Some("1983".to_owned()),
                    subsection: Some("a".to_owned()),
                    code_type: "usc".to_owned(),
                    state: None,
                    code_name: None,
                },
                vec![
                    "42 U.S.C. 1983",
                    "42 USC 1983",
                    "42 U.S.C. § 1983",
                    "42 U.S.C. 1983(a)",
                    "42 USC 1983(a)",
                    "usc_42_1983_a",
                ],
                "usc_42_1983_a",
            )),
        ),
        (
            "18 USC 1001",
            Some((
                StatuteKey {
                    title: Some("18".to_owned()),
                    section: Some("1001".to_owned()),
                    subsection: None,
                    code_type: "usc".to_owned(),
                    state: None,
                    code_name: None,
                },
                vec!["18 U.S.C. 1001", "18 USC 1001", "18 U.S.C. § 1001", "usc_18_1001"],
                "usc_18_1001",
            )),
        ),
        (
            "28 U.S.C. Sec. 1332",
            Some((
                StatuteKey {
                    title: Some("28".to_owned()),
                    section: Some("1332".to_owned()),
                    subsection: None,
                    code_type: "usc".to_owned(),
                    state: None,
                    code_name: None,
                },
                vec!["28 U.S.C. 1332", "28 USC 1332", "28 U.S.C. § 1332", "usc_28_1332"],
                "usc_28_1332",
            )),
        ),
        (
            "42 usc section 1983 (b)",
            Some((
                StatuteKey {
                    title: Some("42".to_owned()),
                    section: Some("1983".to_owned()),
                    subsection: Some("b".to_owned()),
                    code_type: "usc".to_owned(),
                    state: None,
                    code_name: None,
                },
                vec![
                    "42 U.S.C. 1983",
                    "42 USC 1983",
                    "42 U.S.C. § 1983",
                    "42 U.S.C. 1983(b)",
                    "42 USC 1983(b)",
                    "usc_42_1983_b",
                ],
                "usc_42_1983_b",
            )),
        ),
        (
            "Fed. R. Civ. P. 12(b)(6)",
            Some((
                StatuteKey {
                    title: None,
                    section: Some("12".to_owned()),
                    subsection: Some("b".to_owned()),
                    code_type: "frcp".to_owned(),
                    state: None,
                    code_name: None,
                },
                vec![
                    "FRCP 12",
                    "FRCP 12(b)",
                    "Fed. R. Civ. P. 12",
                    "Fed. R. Civ. P. 12(b)",
                    "frcp_12_b",
                ],
                "frcp_12_b",
            )),
        ),
        (
            "Fed. R. Crim. P. 11",
            Some((
                StatuteKey {
                    title: None,
                    section: Some("11".to_owned()),
                    subsection: None,
                    code_type: "frcp".to_owned(),
                    state: None,
                    code_name: None,
                },
                vec!["FRCP 11", "Fed. R. Civ. P. 11", "frcp_11"],
                "frcp_11",
            )),
        ),
        (
            "Federal Rule Appellate Procedure 4(a)",
            Some((
                StatuteKey {
                    title: None,
                    section: Some("4".to_owned()),
                    subsection: Some("a".to_owned()),
                    code_type: "frap".to_owned(),
                    state: None,
                    code_name: None,
                },
                vec![
                    "FRAP 4",
                    "FRAP 4(a)",
                    "Fed. R. App. P. 4",
                    "Fed. R. App. P. 4(a)",
                    "frap_4_a",
                ],
                "frap_4_a",
            )),
        ),
        ("Fed. R. Evid. 403", None),
        (
            "FRE 403",
            Some((
                StatuteKey {
                    title: None,
                    section: Some("403".to_owned()),
                    subsection: None,
                    code_type: "frep".to_owned(),
                    state: None,
                    code_name: None,
                },
                vec!["FRE 403", "Fed. R. Evid. 403", "frep_403"],
                "frep_403",
            )),
        ),
        (
            "FRCRP 5",
            Some((
                StatuteKey {
                    title: None,
                    section: Some("5".to_owned()),
                    subsection: None,
                    code_type: "frcp".to_owned(),
                    state: None,
                    code_name: None,
                },
                vec!["FRCP 5", "Fed. R. Civ. P. 5", "frcp_5"],
                "frcp_5",
            )),
        ),
        (
            "FRAP 4(a)",
            Some((
                StatuteKey {
                    title: None,
                    section: Some("4".to_owned()),
                    subsection: Some("a".to_owned()),
                    code_type: "frap".to_owned(),
                    state: None,
                    code_name: None,
                },
                vec![
                    "FRAP 4",
                    "FRAP 4(a)",
                    "Fed. R. App. P. 4",
                    "Fed. R. App. P. 4(a)",
                    "frap_4_a",
                ],
                "frap_4_a",
            )),
        ),
        (
            "frcp 56",
            Some((
                StatuteKey {
                    title: None,
                    section: Some("56".to_owned()),
                    subsection: None,
                    code_type: "frcp".to_owned(),
                    state: None,
                    code_name: None,
                },
                vec!["FRCP 56", "Fed. R. Civ. P. 56", "frcp_56"],
                "frcp_56",
            )),
        ),
        ("nothing", None),
    ]
}

fn citation_cases() -> Vec<KeyCase<CaseCitationKey>> {
    vec![
        (
            "123 F.3d 456 (9th Cir. 2020)",
            Some((
                CaseCitationKey {
                    volume: Some("123".to_owned()),
                    reporter: Some("F.3d".to_owned()),
                    page: Some("456".to_owned()),
                    court: Some("9th Cir.".to_owned()),
                    year: Some("2020".to_owned()),
                    plaintiff: None,
                    defendant: None,
                },
                vec!["123 F.3d 456", "123 F.3d 456", "123_f3d_456"],
                "123_f3d_456",
            )),
        ),
        ("550 U.S. 544", None),
        (
            "410 F. Supp. 2d 100 (S.D.N.Y. 2005)",
            Some((
                CaseCitationKey {
                    volume: Some("410".to_owned()),
                    reporter: Some("F. Supp. 2d".to_owned()),
                    page: Some("100".to_owned()),
                    court: Some("S.D.N.Y.".to_owned()),
                    year: Some("2005".to_owned()),
                    plaintiff: None,
                    defendant: None,
                },
                vec!["410 F. Supp. 2d 100", "410 F. Supp. 2d 100", "410_fsupp2d_100"],
                "410_fsupp2d_100",
            )),
        ),
        (
            "Smith v. Jones",
            Some((
                CaseCitationKey {
                    volume: None,
                    reporter: None,
                    page: None,
                    court: None,
                    year: None,
                    plaintiff: Some("Smith".to_owned()),
                    defendant: Some("Jones".to_owned()),
                },
                vec!["Smith v. Jones", "Smith v Jones", "Smith v. Jones", "smith_v_jones"],
                "smith_v_jones",
            )),
        ),
        (
            "Brown v Board of Education",
            Some((
                CaseCitationKey {
                    volume: None,
                    reporter: None,
                    page: None,
                    court: None,
                    year: None,
                    plaintiff: Some("Brown".to_owned()),
                    defendant: Some("Board of Education".to_owned()),
                },
                vec![
                    "Brown v. Board of Education",
                    "Brown v Board of Education",
                    "Brown v. Education",
                    "brown_v_board of education",
                ],
                "brown_v_board of education",
            )),
        ),
        (
            "Maria Lopez v. Green Valley Logistics",
            Some((
                CaseCitationKey {
                    volume: None,
                    reporter: None,
                    page: None,
                    court: None,
                    year: None,
                    plaintiff: Some("Maria Lopez".to_owned()),
                    defendant: Some("Green Valley Logistics".to_owned()),
                },
                vec![
                    "Maria Lopez v. Green Valley Logistics",
                    "Maria Lopez v Green Valley Logistics",
                    "Lopez v. Logistics",
                    "maria lopez_v_green valley logistics",
                ],
                "maria lopez_v_green valley logistics",
            )),
        ),
        ("nothing", None),
    ]
}

fn statute_direct_cases() -> Vec<(StatuteKey, &'static str, Vec<&'static str>)> {
    vec![
        (
            StatuteKey {
                title: None,
                section: Some("1714".to_owned()),
                subsection: Some("(a)".to_owned()),
                code_type: "state".to_owned(),
                state: Some("Cal".to_owned()),
                code_name: Some("Civ.  Code".to_owned()),
            },
            "cal_civ_code_1714_a",
            vec!["cal_civ_code_1714_a"],
        ),
        (
            StatuteKey {
                title: Some("".to_owned()),
                section: Some("1983".to_owned()),
                subsection: Some("a".to_owned()),
                code_type: "usc".to_owned(),
                state: None,
                code_name: None,
            },
            "usc_1983_a",
            vec!["usc_1983_a"],
        ),
        (
            StatuteKey {
                title: None,
                section: Some("12".to_owned()),
                subsection: Some("b".to_owned()),
                code_type: "frcrp".to_owned(),
                state: None,
                code_name: None,
            },
            "frcrp_12_b",
            vec![
                "FRCRP 12",
                "FRCRP 12(b)",
                "Fed. R. Crim. P. 12",
                "Fed. R. Crim. P. 12(b)",
                "frcrp_12_b",
            ],
        ),
        (
            StatuteKey {
                title: None,
                section: Some("5".to_owned()),
                subsection: None,
                code_type: "state".to_owned(),
                state: None,
                code_name: None,
            },
            "state_5",
            vec!["state_5"],
        ),
    ]
}

fn citation_direct_cases() -> Vec<(CaseCitationKey, &'static str, Vec<&'static str>)> {
    vec![
        (
            CaseCitationKey {
                volume: Some("410".to_owned()),
                reporter: Some("F. Supp.  2d".to_owned()),
                page: Some("100".to_owned()),
                court: None,
                year: None,
                plaintiff: Some("John Smith".to_owned()),
                defendant: Some("Acme Corp".to_owned()),
            },
            "410_fsupp2d_100",
            vec![
                "410 F. Supp.  2d 100",
                "410 F. Supp. 2d 100",
                "John Smith v. Acme Corp",
                "John Smith v Acme Corp",
                "Smith v. Corp",
                "410_fsupp2d_100",
            ],
        ),
        (
            CaseCitationKey {
                volume: Some("1".to_owned()),
                reporter: Some("U.S.".to_owned()),
                page: None,
                court: None,
                year: None,
                plaintiff: None,
                defendant: None,
            },
            "",
            vec![],
        ),
    ]
}

fn federal_rule_cases() -> Vec<(FederalRuleKey, &'static str)> {
    vec![
        (
            FederalRuleKey {
                rule_type: "Civil".to_owned(),
                rule_number: "12".to_owned(),
                subdivision_letter: Some("b".to_owned()),
                subdivision_number: Some("6".to_owned()),
            },
            "frcp_12_b_6",
        ),
        (
            FederalRuleKey {
                rule_type: "evidence".to_owned(),
                rule_number: "403".to_owned(),
                subdivision_letter: None,
                subdivision_number: None,
            },
            "frep_403",
        ),
    ]
}

#[test]
fn exhibit_keys_match_python() {
    let patterns = registry_patterns();
    for (text, expected) in exhibit_cases() {
        let key = extract_exhibit_key(&patterns, text, Some("CASE-1"));
        match expected {
            None => assert_eq!(key, None, "{text:?}"),
            Some((expected_key, terms)) => {
                let key = key.unwrap_or_else(|| panic!("{text:?}: no key"));
                assert_eq!(key, expected_key, "{text:?}");
                assert_eq!(exhibit_key_to_search_terms(&key), terms, "{text:?}");
            }
        }
    }
}

#[test]
fn statute_keys_match_python() {
    let patterns = registry_patterns();
    for (text, expected) in statute_cases() {
        let key = extract_statute_key(&patterns, text);
        match expected {
            None => assert_eq!(key, None, "{text:?}"),
            Some((expected_key, terms, normalized)) => {
                let key = key.unwrap_or_else(|| panic!("{text:?}: no key"));
                assert_eq!(key, expected_key, "{text:?}");
                assert_eq!(statute_key_to_search_terms(&key), terms, "{text:?}");
                assert_eq!(key.normalized_key(), normalized, "{text:?}");
            }
        }
    }
}

#[test]
fn case_citation_keys_match_python() {
    let patterns = registry_patterns();
    for (text, expected) in citation_cases() {
        let key = extract_case_citation_key(&patterns, text);
        match expected {
            None => assert_eq!(key, None, "{text:?}"),
            Some((expected_key, terms, normalized)) => {
                let key = key.unwrap_or_else(|| panic!("{text:?}: no key"));
                assert_eq!(key, expected_key, "{text:?}");
                assert_eq!(case_citation_key_to_search_terms(&key), terms, "{text:?}");
                assert_eq!(key.normalized_key(), normalized, "{text:?}");
            }
        }
    }
}

#[test]
fn constructed_keys_match_python() {
    for (key, normalized, terms) in statute_direct_cases() {
        assert_eq!(key.normalized_key(), normalized, "{key:?}");
        assert_eq!(statute_key_to_search_terms(&key), terms, "{key:?}");
    }
    for (key, normalized, terms) in citation_direct_cases() {
        assert_eq!(key.normalized_key(), normalized, "{key:?}");
        assert_eq!(case_citation_key_to_search_terms(&key), terms, "{key:?}");
    }
    for (key, normalized) in federal_rule_cases() {
        assert_eq!(key.normalized_key(), normalized, "{key:?}");
    }
}

#[test]
fn federal_rule_resolves_to_the_ed_pattern() {
    let registry = MappingsRegistry::load().expect("embedded mappings load");
    // The detection pattern of the same name has a single group.
    let detection =
        registry.detection_patterns.patterns["federal_rule"].compiled().expect("compiled");
    assert_eq!(detection.captures_len(), 2);
    let patterns = registry_patterns();
    assert_eq!(patterns.get("federal_rule").expect("federal_rule").captures_len(), 5);
}

#[test]
fn registry_defines_every_key_pattern() {
    let patterns = registry_patterns();
    let ed =
        &MappingsRegistry::load().expect("embedded mappings load").detection_patterns.ed_patterns;
    for name in ["case_docket", "exhibit_id", "exhibit_id_simple", "statute_usc", "case_citation"] {
        assert_eq!(
            patterns.get(name).map(regex::Regex::as_str),
            ed[name].compiled().map(regex::Regex::as_str),
            "{name}"
        );
    }
    assert!(patterns.get("state_statute").is_some());
    assert!(patterns.get("exhibit").is_some());
    assert!(patterns.get("unknown").is_none());
}

#[test]
fn fallback_patterns_extract_the_same_keys() {
    let fallback = KeyPatterns::fallback();
    let registry = registry_patterns();
    for text in [
        "Plaintiff's Exhibit 12A",
        "Exh. b-2",
        "42 U.S.C. § 1983(a)",
        "28 U.S.C. Sec. 1332",
        "Fed. R. Civ. P. 12(b)(6)",
        "FRAP 4(a)",
        "123 F.3d 456 (9th Cir. 2020)",
        "Smith v. Jones",
    ] {
        assert_eq!(
            extract_exhibit_key(&fallback, text, None),
            extract_exhibit_key(&registry, text, None),
            "{text:?}"
        );
        assert_eq!(extract_statute_key(&fallback, text), extract_statute_key(&registry, text));
        assert_eq!(
            extract_case_citation_key(&fallback, text),
            extract_case_citation_key(&registry, text),
            "{text:?}"
        );
    }
    let docket = fallback.get("case_docket").expect("case_docket");
    let caps = docket.captures("Case No. 2:23-cv-01492").expect("docket matches");
    assert_eq!(&caps[1], "2:23-cv-01492");
    // The fallback caption pattern has no space in its name class.
    let key = extract_case_citation_key(&fallback, "Brown v Board of Education").expect("caption");
    assert_eq!(key.defendant.as_deref(), Some("Board"));
}

#[test]
fn missing_and_invalid_registry_patterns_fall_back() {
    let patterns: DetectionPatterns = serde_json::from_value(json!({
        "ed_patterns": {
            "exhibit_id": {"pattern": "(unclosed", "flags": "IGNORECASE"},
            "custom": {"pattern": "abc", "flags": "IGNORECASE"}
        }
    }))
    .expect("valid detection patterns");
    let keys = KeyPatterns::from_mappings(&patterns);
    assert!(keys.get("custom").expect("custom").is_match("xABCx"));
    let key = extract_exhibit_key(&keys, "Defendant's Ex. 3", None).expect("exhibit");
    assert_eq!(key.party_attribution.as_deref(), Some("Defendant"));
    assert_eq!(key.exhibit_id, "3");
}
