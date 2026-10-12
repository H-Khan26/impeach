//! Name matching and type-aware name similarity. Expected values come from the Python
//! `services/ed/name_similarity.py` / `name_matching.py` run on the same inputs with the shipped
//! `name_matching.yaml`.

use ce_config::mappings::{NameMatching, NameMatchingSettings};
use ce_config::MappingsRegistry;
use ce_ed::{
    date_similarity, money_similarity, names_match, normalize_for_matching, NameSimilarity,
};
use pretty_assertions::assert_eq;

const ENHANCED: &[(&str, &str, &str, f64)] = &[
    ("Maria Lopez", "Ms. Lopez", "PERSON", 0.875),
    ("Maria Lopez", "Lopez, Maria", "PER", 0.98),
    ("Bob Smith", "Robert Smith", "PERSON", 0.98),
    ("Dr. John Smith", "John Smith, MD", "PER", 0.98),
    ("Smith", "John Smith", "PER", 0.875),
    ("John Smith", "Jane Smith", "PER", 0.7),
    ("John A. Smith", "John Smith Jr.", "PER", 0.9),
    ("Jon Smyth", "John Smith", "PER", 0.8421052631578947),
    ("Mr. X", "Dr. Y", "PER", 0.0),
    ("Teddy Roosevelt", "Theodore Roosevelt", "PER", 0.98),
    ("Acme Corp.", "Acme Corporation", "ORG", 0.98),
    ("The Acme Company", "Acme", "ORG", 0.98),
    ("Acme Inc.", "Acme Holdings Group LLC", "ORGANIZATION", 0.98),
    ("FBI", "Federal Bureau of Investigation", "ORG", 0.98),
    ("Federal Bureau of Investigation", "FBI", "ORG", 0.98),
    ("fbi", "the FBI", "ORG", 0.98),
    ("IBM", "International Business Machines", "ORG", 0.8),
    ("GVL", "Green Valley Logistics, Inc.", "ORG", 0.8),
    ("Green Valley Logistics", "Green Valley Trucking", "ORG", 0.75),
    ("Green Valley Logistics, Inc.", "Green Valley Logistics", "ORG", 0.98),
    ("Riverside Arena", "Riverside Medical Center", "ORG", 0.6749999999999999),
    ("Acme", "Zenith", "ORG", 0.2),
    ("Co", "Company", "ORG", 0.4444444444444444),
    ("A", "B", "ORG", 0.0),
    ("Department of Justice", "DOJ", "ORG", 0.98),
    ("March 4, 2023", "March 4 2023", "DATE", 0.98),
    ("March 4", "March 5", "DATE", 0.0),
    ("$1,000.00", "$1 000 00", "MONEY", 0.98),
    ("$1,000", "$1,001", "MONEY", 0.0),
    ("Main Street", "Main St.", "LOC", 0.7368421052631579),
    ("Main Street", "main street", "FACILITY", 1.0),
    ("Exhibit 12", "Exhibit 12A", "EXHIBIT", 0.9523809523809523),
    ("Acme", "Acme", "UNKNOWN_TYPE", 1.0),
    ("", "Acme", "ORG", 0.0),
    ("Acme Corp", "Acme Corp", "ORG", 0.98),
    ("Maria Lopez", "maria lopez", "person", 0.98),
    ("Bob", "Robert", "PERSON", 0.98),
    ("J. Smith", "John Smith", "PER", 0.875),
];

const PERSON: &[(&str, &str, f64)] = &[
    ("Maria Lopez", "Ms. Lopez", 0.875),
    ("Maria Lopez", "Lopez, Maria", 0.98),
    ("Bob Smith", "Robert Smith", 0.98),
    ("Dr. John Smith", "John Smith, MD", 0.98),
    ("Smith", "John Smith", 0.875),
    ("John Smith", "Jane Smith", 0.7),
    ("John A. Smith", "John Smith Jr.", 0.9),
    ("Jon Smyth", "John Smith", 0.8421052631578947),
    ("Mr. X", "Dr. Y", 0.0),
    ("Teddy Roosevelt", "Theodore Roosevelt", 0.98),
];

const ORG: &[(&str, &str, f64)] = &[
    ("Acme Corp.", "Acme Corporation", 0.98),
    ("The Acme Company", "Acme", 0.98),
    ("Acme Inc.", "Acme Holdings Group LLC", 0.98),
    ("FBI", "Federal Bureau of Investigation", 0.98),
    ("Federal Bureau of Investigation", "FBI", 0.98),
    ("fbi", "the FBI", 0.98),
    ("IBM", "International Business Machines", 0.8),
    ("GVL", "Green Valley Logistics, Inc.", 0.8),
    ("Green Valley Logistics", "Green Valley Trucking", 0.75),
    ("Green Valley Logistics, Inc.", "Green Valley Logistics", 0.98),
    ("Riverside Arena", "Riverside Medical Center", 0.6749999999999999),
    ("Acme", "Zenith", 0.2),
    ("Co", "Company", 0.4444444444444444),
    ("A", "B", 0.0),
    ("Department of Justice", "DOJ", 0.98),
];

const PERSON_PARTS: &[(&str, &[&str])] = &[
    ("Dr. Maria Lopez", &["maria", "lopez"]),
    ("Mr. Bob O'Brien-Smith", &["robert", "brien", "smith"]),
    ("Judge Hon. William Jones, Esq.", &["william", "jones"]),
    ("Al Gore", &["alexander", "gore"]),
    ("Liz", &["elizabeth"]),
    ("Jas. Q. Public", &["james", "public"]),
    ("", &[]),
    ("Teddy", &["theodore"]),
];

const ORG_NORMALIZED: &[(&str, &str, &[&str])] = &[
    ("The Acme Company", "acme", &["acme"]),
    ("Acme Corporation Inc.", "acme", &["acme"]),
    ("Green Valley Logistics, Inc.", "green valley logistics", &["green", "valley", "logistics"]),
    ("Company", "company", &["company"]),
    ("The Company", "company", &["company"]),
    ("A Group Holdings", "group", &["group"]),
    ("Smith & Sons, L.L.C.", "smith & sons l l c", &["smith", "sons"]),
    ("  The   Acme, Co.  ", "acme", &["acme"]),
    ("International", "international", &["international"]),
    ("Acme Intl. Ltd", "acme", &["acme"]),
];

const DATES: &[(&str, &str, f64)] = &[
    ("March 4, 2023", "march 4 2023", 0.98),
    ("March 4", "March 5", 0.0),
    ("", "", 0.0),
    ("...", "...", 0.0),
];

const MONEY: &[(&str, &str, f64)] =
    &[("$1,000", "$1 000", 0.98), ("$1,000", "$1,001", 0.0), ("-", "-", 0.0)];

const NAMES_MATCH: &[(&str, &str, bool)] = &[
    ("Maria Lopez", "Maria Lopez", true),
    ("MARIA LOPEZ", "maria lopez", true),
    ("Maria Lopez", "Lopez", true),
    ("Mr. Smith", "Smith", true),
    ("Lee", "Bruce Lee", false),
    ("Maria", "Marianne Lopez", false),
    ("ria lo", "Maria Lopez", false),
    ("", "Maria", false),
    ("...", "...", false),
    ("Élan", "Société Élan", true),
    ("Él", "Société Él", false),
    ("Dr. Maria Lopez", "Maria Lopez", true),
];

const NORMALIZED_FOR_MATCHING: &[(&str, &str)] = &[
    ("  Mr. O'Brien,  (Jr.) ", "mr obrien jr"),
    ("MARIA\tLOPEZ", "maria lopez"),
    ("[A]{B}\"C\";:!?", "abc"),
    ("Smith-Jones", "smith-jones"),
    ("", ""),
    (" .,; ", ""),
];

fn similarity() -> NameSimilarity {
    NameSimilarity::from_mappings(MappingsRegistry::load().expect("embedded mappings load"))
}

fn assert_close(actual: f64, expected: f64, what: &str) {
    assert!((actual - expected).abs() < 1e-9, "{what}: expected {expected}, got {actual}");
}

#[test]
fn enhanced_string_similarity_matches_python() {
    let sim = similarity();
    for &(a, b, entity_type, expected) in ENHANCED {
        let actual = sim.enhanced_string_similarity(a, b, entity_type);
        assert_close(actual, expected, &format!("{a:?} vs {b:?} as {entity_type}"));
    }
}

#[test]
fn person_name_similarity_matches_python() {
    let sim = similarity();
    for &(a, b, expected) in PERSON {
        assert_close(sim.person_name_similarity(a, b), expected, &format!("{a:?} vs {b:?}"));
    }
}

#[test]
fn org_name_similarity_matches_python() {
    let sim = similarity();
    for &(a, b, expected) in ORG {
        assert_close(sim.org_name_similarity(a, b), expected, &format!("{a:?} vs {b:?}"));
    }
}

#[test]
fn person_name_parts_match_python() {
    let sim = similarity();
    for &(name, expected) in PERSON_PARTS {
        assert_eq!(sim.normalize_person_name_parts(name), expected, "{name:?}");
    }
}

#[test]
fn org_name_normalization_matches_python() {
    let sim = similarity();
    for &(name, normalized, parts) in ORG_NORMALIZED {
        assert_eq!(sim.normalize_org_name(name), normalized, "{name:?}");
        assert_eq!(sim.normalize_org_name_parts(name), parts, "{name:?}");
    }
}

#[test]
fn date_and_money_similarity_match_python() {
    for &(a, b, expected) in DATES {
        assert_close(date_similarity(a, b), expected, &format!("date {a:?} vs {b:?}"));
    }
    for &(a, b, expected) in MONEY {
        assert_close(money_similarity(a, b), expected, &format!("money {a:?} vs {b:?}"));
    }
}

#[test]
fn names_match_matches_python() {
    for &(a, b, expected) in NAMES_MATCH {
        assert_eq!(names_match(a, b), expected, "{a:?} vs {b:?}");
        assert_eq!(names_match(b, a), expected, "{b:?} vs {a:?}");
    }
    for &(name, expected) in NORMALIZED_FOR_MATCHING {
        assert_eq!(normalize_for_matching(name), expected, "{name:?}");
    }
}

#[test]
fn settings_switch_off_titles_nicknames_and_suffixes() {
    let registry = MappingsRegistry::load().expect("embedded mappings load");
    let mut settings = registry.name_matching.clone();
    settings.settings = NameMatchingSettings {
        normalize_nicknames: false,
        strip_titles: false,
        strip_org_suffixes: false,
        case_insensitive: true,
    };
    let sim = NameSimilarity::new(settings, registry.entity_types.clone());
    assert_eq!(sim.normalize_person_name_parts("Dr. Bob Smith"), ["dr", "bob", "smith"]);
    assert_eq!(sim.normalize_org_name("The Acme Corp."), "acme corp");
    // Subset {smith} of {bob, smith}: 0.80 + 0.15 * 1 / 2.
    assert_close(sim.person_name_similarity("Bob Smith", "Smith"), 0.875, "subset");
}

#[test]
fn thresholds_come_from_the_settings() {
    let registry = MappingsRegistry::load().expect("embedded mappings load");
    let mut settings: NameMatching = registry.name_matching.clone();
    settings.thresholds.exact_match = 0.5;
    settings.thresholds.min_word_length = 1;
    let sim = NameSimilarity::new(settings, registry.entity_types.clone());
    assert_close(sim.person_name_similarity("Bob Smith", "Robert Smith"), 0.5, "exact");
    assert_close(sim.enhanced_string_similarity("May 1", "may 1", "DATE"), 0.5, "date");
    assert_eq!(sim.normalize_person_name_parts("J. Smith"), ["j", "smith"]);
}

#[test]
fn defaults_match_the_python_fallbacks() {
    let defaults = NameMatching::default();
    assert_close(defaults.exact_match_threshold(), 0.98, "exact_match");
    assert_close(defaults.subset_match_base(), 0.80, "subset_match_base");
    assert_close(defaults.subset_match_bonus(), 0.15, "subset_match_bonus");
    assert_close(defaults.partial_overlap_base(), 0.60, "partial_overlap_base");
    assert_close(defaults.partial_overlap_bonus(), 0.30, "partial_overlap_bonus");
    assert_eq!(defaults.min_word_length(), 2);
    assert!(defaults.settings.strip_titles && defaults.settings.normalize_nicknames);
}
