//! Legal citation detector; a regex port of the eyecite forms the Python pipeline consumes
//! (Python `services/ner/eyecite_detector.py`).
//!
//! Source name `eyecite`, so the `detector_config.yaml` priorities for eyecite apply. Every
//! mention has score 1.0 and an id from [`super::span_id`].
//!
//! | Form | Example | Label | Metadata |
//! | --- | --- | --- | --- |
//! | Full case citation | `347 U.S. 483` | `CASE_CITATION` | `volume`, `reporter`, `page`, `year` (from a following `(... 1954)` parenthetical) |
//! | Short case citation | `347 U.S. at 490` | `CASE_CITATION` | `volume`, `reporter`, `page` |
//! | `Id.` / `Ibid.` | `Id.` | `CASE_CITATION` | `is_short_form: true` |
//! | `supra` | `supra,` | `CASE_CITATION` | none |
//! | U.S. Code | `42 U.S.C. § 1983` | `CASE_CITATION` | `title`, `reporter`, `section` |
//! | C.F.R. | `28 C.F.R. § 35.130` | `CASE_CITATION` | `chapter`, `reporter`, `section` |
//! | California / Texas codes | `Cal. Civ. Code § 1714`, `Tex. Penal Code Ann. § 22.01` | `STATUTE_CITATION` | `reporter`, `subject`, `section` |
//! | New York laws | `N.Y. Penal Law § 120.00` | `CASE_CITATION` | `reporter`, `subject`, `section` |
//! | Florida statutes | `Fla. Stat. § 768.81` | `CASE_CITATION` | `reporter`, `section` |
//!
//! Labels follow the Python classification of eyecite objects: a statute is
//! `STATUTE_CITATION` only when its reporter contains `Code` (Python tests `"Code" in
//! str(cit)`); every other citation, U.S. Code included, is `CASE_CITATION`. Python's
//! `RULE_CITATION` branch never fires with eyecite 2.x, so it is never emitted. Statutes need
//! a `§`, as in eyecite; `42 U.S.C. 1983` and federal rules are left to the regex detector.
//! Overlapping matches keep the earliest (then longest) one.
//!
//! Gaps against eyecite: the reporter list is a fixed set of common federal, regional and
//! state reporters (eyecite knows thousands); no case names, pin cites, courts or
//! parentheticals are extracted; statute coverage is limited to the forms above; docket
//! numbers are not citations in eyecite either (the regex detector's `docket_entry` covers
//! them). Spans are the match positions; Python re-finds the first occurrence of the matched
//! text when eyecite gives no offsets, so a citation repeated in a chunk gets the position of
//! its first occurrence there. `year` comes from the parenthetical directly after the
//! citation (eyecite sometimes takes a later one).

use ce_core::{Mention, Metadata, Span};
use once_cell::sync::Lazy;
use regex::{Captures, Regex};
use serde_json::Value;

use super::{span_id, MentionDetector};
use crate::error::Result;
use crate::text::CharIndex;

/// Case-law reporters recognised in case citations, written with Bluebook spacing.
const REPORTERS: &[&str] = &[
    // Federal
    "U.S.",
    "U. S.",
    "US",
    "S. Ct.",
    "L. Ed.",
    "L. Ed. 2d",
    "F.",
    "F.2d",
    "F.3d",
    "F.4th",
    "F. Supp.",
    "F. Supp. 2d",
    "F. Supp. 3d",
    "F. App'x",
    "Fed. Appx.",
    "F.R.D.",
    "B.R.",
    "Fed. Cl.",
    // Regional
    "A.",
    "A.2d",
    "A.3d",
    "P.",
    "P.2d",
    "P.3d",
    "N.E.",
    "N.E.2d",
    "N.E.3d",
    "N.W.",
    "N.W.2d",
    "S.E.",
    "S.E.2d",
    "S.W.",
    "S.W.2d",
    "S.W.3d",
    "So.",
    "So. 2d",
    "So. 3d",
    // State
    "Cal.",
    "Cal. 2d",
    "Cal. 3d",
    "Cal. 4th",
    "Cal. 5th",
    "Cal. App.",
    "Cal. App. 2d",
    "Cal. App. 3d",
    "Cal. App. 4th",
    "Cal. App. 5th",
    "Cal. Rptr.",
    "Cal. Rptr. 2d",
    "Cal. Rptr. 3d",
    "N.Y.",
    "N.Y.2d",
    "N.Y.3d",
    "N.Y.S.",
    "N.Y.S.2d",
    "N.Y.S.3d",
    "A.D.2d",
    "A.D.3d",
    "Ill. 2d",
    "Ill. Dec.",
    "Mass.",
    "N.J.",
    "Pa.",
    "Tex.",
    "Wash. 2d",
    "Ohio St. 3d",
];

/// Section number: digits with `.`/`-` separated parts (`1983`, `340.8`, `1983-1985`).
const SECTION: &str = r"(?P<section>\d+(?:[.\-–]\d+)*)\b";

/// Subject words of a state code (`Civ.`, `Bus. & Prof.`, `Penal`).
const SUBJECT: &str = r"(?P<subject>[A-Z][a-z]*\.?(?:\s+(?:&\s+)?[A-Z][a-z]*\.?)*?)";

/// Regex source of one reporter: optional whitespace after an interior period, required
/// whitespace between words without a period.
fn reporter_pattern(reporter: &str) -> String {
    let mut out = String::new();
    for (i, word) in reporter.split(' ').enumerate() {
        if i > 0 {
            out.push_str(if out.ends_with(r"\.") { r"\s?" } else { r"\s+" });
        }
        let parts: Vec<&str> = word.split_inclusive('.').collect();
        for (j, part) in parts.iter().enumerate() {
            out.push_str(&regex::escape(part).replace('\'', "['’]"));
            if j + 1 < parts.len() {
                out.push_str(r"\s?");
            }
        }
    }
    out
}

fn compile(source: &str) -> Regex {
    Regex::new(source).expect("citation patterns are valid regexes")
}

static CASE: Lazy<Regex> = Lazy::new(|| {
    let mut reporters: Vec<&str> = REPORTERS.to_vec();
    reporters.sort_by_key(|r| std::cmp::Reverse(r.len()));
    let alternation: Vec<String> = reporters.iter().map(|r| reporter_pattern(r)).collect();
    compile(&format!(
        r"\b(?P<volume>\d{{1,4}})\s+(?P<reporter>{})\s+(?P<at>at\s+)?(?P<page>\d{{1,5}})\b",
        alternation.join("|")
    ))
});

/// A parenthetical with a year after a full citation and optional pin cites.
static YEAR: Lazy<Regex> = Lazy::new(|| {
    compile(
        r"\A(?:,\s*\d+(?:\s*[-–]\s*\d+)?(?:\s*n\.\s*\d+)?)*\s*\((?:[^()]*?\s)?(?P<year>\d{4})\)",
    )
});

static USC: Lazy<Regex> = Lazy::new(|| {
    compile(&format!(
        r"\b(?P<title>\d+)\s+(?P<reporter>U\.\s?S\.\s?C\.(?:\s?[AS]\.)?|USCA|USC)\s*§§?\s*{SECTION}"
    ))
});

static CFR: Lazy<Regex> = Lazy::new(|| {
    compile(&format!(r"\b(?P<chapter>\d+)\s+(?P<reporter>C\.\s?F\.\s?R\.|CFR)\s*§§?\s*{SECTION}"))
});

static CAL_CODE: Lazy<Regex> = Lazy::new(|| {
    compile(&format!(r"\b(?P<reporter>Cal\.\s+{SUBJECT}\s+Code(?:\s+Ann\.)?)\s*§§?\s*{SECTION}"))
});

static TEX_CODE: Lazy<Regex> = Lazy::new(|| {
    compile(&format!(r"\b(?P<reporter>Tex\.\s+{SUBJECT}\s+Code\s+Ann\.)\s*§§?\s*{SECTION}"))
});

static NY_LAW: Lazy<Regex> = Lazy::new(|| {
    compile(&format!(r"\b(?P<reporter>N\.\s?Y\.\s+{SUBJECT}\s+Law)\s*§§?\s*{SECTION}"))
});

static FLA_STAT: Lazy<Regex> = Lazy::new(|| {
    compile(&format!(r"\b(?P<reporter>Fla\.\s?Stat\.(?:\s+Ann\.)?)\s*§§?\s*{SECTION}"))
});

static ID: Lazy<Regex> = Lazy::new(|| compile(r"\b(?:Id|id|Ibid|ibid)\."));

static SUPRA: Lazy<Regex> = Lazy::new(|| compile(r"\b[Ss]upra\b[.,]?"));

/// Statute patterns and the metadata keys they report, in eyecite group order.
static STATUTES: [(&Lazy<Regex>, &[&str]); 6] = [
    (&USC, &["title", "reporter", "section"]),
    (&CFR, &["chapter", "reporter", "section"]),
    (&CAL_CODE, &["reporter", "subject", "section"]),
    (&TEX_CODE, &["reporter", "subject", "section"]),
    (&NY_LAW, &["reporter", "subject", "section"]),
    (&FLA_STAT, &["reporter", "section"]),
];

/// One citation found in the text (byte offsets).
#[derive(Debug)]
struct Found {
    start: usize,
    end: usize,
    label: &'static str,
    metadata: Metadata,
}

/// Detects legal citations (the eyecite replacement).
#[derive(Debug, Clone, Copy, Default)]
pub struct CitationDetector;

impl CitationDetector {
    /// Source name of citation mentions (the Python eyecite detector's).
    pub const SOURCE: &'static str = "eyecite";

    /// Case citation label.
    pub const CASE_CITATION: &'static str = "CASE_CITATION";

    /// Statute citation label (codes).
    pub const STATUTE_CITATION: &'static str = "STATUTE_CITATION";

    /// A detector.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Every citation in `text` (byte offsets), overlaps resolved, in text order.
    fn find_all(text: &str) -> Vec<Found> {
        let mut found = Vec::new();
        for caps in CASE.captures_iter(text) {
            let whole = caps.get(0).expect("group 0 always matches");
            let mut metadata = group_metadata(&caps, &["volume", "reporter", "page"]);
            if caps.name("at").is_none() {
                let year = YEAR
                    .captures(&text[whole.end()..])
                    .and_then(|c| c.name("year").and_then(|y| y.as_str().parse::<i64>().ok()));
                if let Some(year) = year {
                    metadata.insert("year".into(), Value::from(year));
                }
            }
            found.push(Found {
                start: whole.start(),
                end: whole.end(),
                label: Self::CASE_CITATION,
                metadata,
            });
        }
        for (regex, keys) in &STATUTES {
            for caps in regex.captures_iter(text) {
                let whole = caps.get(0).expect("group 0 always matches");
                let reporter = caps.name("reporter").map_or("", |m| m.as_str());
                let label = if reporter.contains("Code") {
                    Self::STATUTE_CITATION
                } else {
                    Self::CASE_CITATION
                };
                found.push(Found {
                    start: whole.start(),
                    end: whole.end(),
                    label,
                    metadata: group_metadata(&caps, keys),
                });
            }
        }
        for m in ID.find_iter(text) {
            let mut metadata = Metadata::new();
            metadata.insert("is_short_form".into(), Value::Bool(true));
            found.push(Found {
                start: m.start(),
                end: m.end(),
                label: Self::CASE_CITATION,
                metadata,
            });
        }
        for m in SUPRA.find_iter(text) {
            found.push(Found {
                start: m.start(),
                end: m.end(),
                label: Self::CASE_CITATION,
                metadata: Metadata::new(),
            });
        }
        found.sort_by(|a, b| a.start.cmp(&b.start).then(b.end.cmp(&a.end)));
        let mut kept: Vec<Found> = Vec::with_capacity(found.len());
        for item in found {
            if kept.last().is_some_and(|last| item.start < last.end) {
                continue;
            }
            kept.push(item);
        }
        kept
    }
}

/// Named groups as string metadata, in `keys` order; missing groups are left out.
fn group_metadata(caps: &Captures<'_>, keys: &[&str]) -> Metadata {
    let mut metadata = Metadata::new();
    for key in keys {
        if let Some(m) = caps.name(key) {
            let value = m.as_str().trim();
            if !value.is_empty() {
                metadata.insert((*key).to_owned(), Value::from(value));
            }
        }
    }
    metadata
}

impl MentionDetector for CitationDetector {
    fn source(&self) -> &str {
        Self::SOURCE
    }

    fn detect(&self, text: &str, segment_id: &str) -> Result<Vec<Mention>> {
        let found = Self::find_all(text);
        if found.is_empty() {
            return Ok(Vec::new());
        }
        let index = CharIndex::new(text);
        Ok(found
            .into_iter()
            .map(|item| {
                let start = index.char_of_byte(item.start);
                let end = index.char_of_byte(item.end);
                let id = span_id(segment_id, Self::SOURCE, start, end, item.label);
                let name = &text[item.start..item.end];
                let mut mention = Mention::new(id, item.label, name, Span::new(start, end));
                mention.score = Some(1.0);
                mention.source = Some(Self::SOURCE.to_owned());
                mention.metadata = item.metadata;
                mention
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;

    type Row = (String, String, usize, usize, Value);

    /// `(label, name, start, end, metadata)` of each citation.
    fn detect(text: &str) -> Vec<Row> {
        CitationDetector
            .detect(text, "seg")
            .expect("detect")
            .into_iter()
            .map(|m| (m.entity_type, m.name, m.span.start, m.span.end, Value::Object(m.metadata)))
            .collect()
    }

    fn row(label: &str, name: &str, start: usize, end: usize, metadata: Value) -> Row {
        (label.into(), name.into(), start, end, metadata)
    }

    #[test]
    fn reporter_patterns_allow_bluebook_spacing_variants() {
        assert_eq!(reporter_pattern("F. Supp. 2d"), r"F\.\s?Supp\.\s?2d");
        assert_eq!(reporter_pattern("N.Y.S.2d"), r"N\.\s?Y\.\s?S\.\s?2d");
        assert_eq!(reporter_pattern("Ohio St. 3d"), r"Ohio\s+St\.\s?3d");
        assert_eq!(reporter_pattern("F. App'x"), r"F\.\s?App['’]x");
    }

    // Expected values are the Python eyecite detector's output for the same text, except
    // where the module docs list a deliberate difference.
    #[test]
    fn case_citations_match_eyecite() {
        let text = "See Brown v. Board of Education, 347 U.S. 483, 495 (1954). Id. at 496. \
                    Smith, 347 U.S. at 490.";
        assert_eq!(
            detect(text),
            vec![
                row(
                    "CASE_CITATION",
                    "347 U.S. 483",
                    33,
                    45,
                    json!({"volume": "347", "reporter": "U.S.", "page": "483", "year": 1954})
                ),
                row("CASE_CITATION", "Id.", 59, 62, json!({"is_short_form": true})),
                row(
                    "CASE_CITATION",
                    "347 U.S. at 490",
                    78,
                    93,
                    json!({"volume": "347", "reporter": "U.S.", "page": "490"})
                ),
            ]
        );
    }

    #[test]
    fn reporters_and_years() {
        let text = "Lopez v. Green Valley, 5 Cal. App. 5th 100 (2016); 12 Cal. Rptr. 3d 45; \
                    99 P.3d 1 (Cal. 2004); 410 F. Supp. 2d 77 (S.D.N.Y. 2005); 19 F. App'x 20; \
                    5 Cal.4th 1, 10 (1993); 100 F.3d 200, 205 n.3 (2d Cir. 1996); 347 US 483";
        let found = detect(text);
        let names: Vec<&str> = found.iter().map(|r| r.1.as_str()).collect();
        assert_eq!(
            names,
            [
                "5 Cal. App. 5th 100",
                "12 Cal. Rptr. 3d 45",
                "99 P.3d 1",
                "410 F. Supp. 2d 77",
                "19 F. App'x 20",
                "5 Cal.4th 1",
                "100 F.3d 200",
                "347 US 483"
            ]
        );
        let years: Vec<Option<i64>> =
            found.iter().map(|r| r.4.get("year").and_then(Value::as_i64)).collect();
        assert_eq!(
            years,
            [Some(2016), None, Some(2004), Some(2005), None, Some(1993), Some(1996), None]
        );
        assert_eq!(found[3].4["reporter"], json!("F. Supp. 2d"));
        assert_eq!(found[5].4["reporter"], json!("Cal.4th"));
    }

    #[test]
    fn statutes_match_eyecite_labels_and_groups() {
        let cases = [
            (
                "A: 42 U.S.C. § 1983 B",
                row(
                    "CASE_CITATION",
                    "42 U.S.C. § 1983",
                    3,
                    19,
                    json!({"title": "42", "reporter": "U.S.C.", "section": "1983"}),
                ),
            ),
            (
                "A: 42 U.S.C. §§ 1983-1985 B",
                row(
                    "CASE_CITATION",
                    "42 U.S.C. §§ 1983-1985",
                    3,
                    25,
                    json!({"title": "42", "reporter": "U.S.C.", "section": "1983-1985"}),
                ),
            ),
            (
                "x 18 USC § 1001 y",
                row(
                    "CASE_CITATION",
                    "18 USC § 1001",
                    2,
                    15,
                    json!({"title": "18", "reporter": "USC", "section": "1001"}),
                ),
            ),
            (
                "A: 28 C.F.R. § 35.130 B",
                row(
                    "CASE_CITATION",
                    "28 C.F.R. § 35.130",
                    3,
                    21,
                    json!({"chapter": "28", "reporter": "C.F.R.", "section": "35.130"}),
                ),
            ),
            (
                "x Cal. Bus. & Prof. Code § 17200 y",
                row(
                    "STATUTE_CITATION",
                    "Cal. Bus. & Prof. Code § 17200",
                    2,
                    32,
                    json!({
                        "reporter": "Cal. Bus. & Prof. Code",
                        "subject": "Bus. & Prof.",
                        "section": "17200"
                    }),
                ),
            ),
            (
                "x Cal. Civ. Code §1714 y",
                row(
                    "STATUTE_CITATION",
                    "Cal. Civ. Code §1714",
                    2,
                    22,
                    json!({"reporter": "Cal. Civ. Code", "subject": "Civ.", "section": "1714"}),
                ),
            ),
            (
                "x Tex. Penal Code Ann. § 22.01 y",
                row(
                    "STATUTE_CITATION",
                    "Tex. Penal Code Ann. § 22.01",
                    2,
                    30,
                    json!({
                        "reporter": "Tex. Penal Code Ann.",
                        "subject": "Penal",
                        "section": "22.01"
                    }),
                ),
            ),
            (
                "x N.Y. Penal Law § 120.00 y",
                row(
                    "CASE_CITATION",
                    "N.Y. Penal Law § 120.00",
                    2,
                    25,
                    json!({"reporter": "N.Y. Penal Law", "subject": "Penal", "section": "120.00"}),
                ),
            ),
            (
                "x Fla. Stat. § 768.81 y",
                row(
                    "CASE_CITATION",
                    "Fla. Stat. § 768.81",
                    2,
                    21,
                    json!({"reporter": "Fla. Stat.", "section": "768.81"}),
                ),
            ),
        ];
        for (text, expected) in cases {
            assert_eq!(detect(text), vec![expected], "{text}");
        }
        // Subsections stay outside the span, as in eyecite.
        assert_eq!(detect("Cal. Gov. Code § 12940(a)")[0].1, "Cal. Gov. Code § 12940");
    }

    #[test]
    fn forms_eyecite_does_not_report_are_skipped() {
        for text in [
            "A: 42 U.S.C. 1983 B",
            "Fed. R. Evid. 801(d)(2) and Fed. R. Civ. P. 56",
            "Docket No. 12, ECF No. 45",
            "15 U.S.C. § 78j(b)",
            "x Tex. Penal Code § 22.01 y",
            "Cal. Code Civ. Proc. § 335.1",
            "He said. Ibidem was absent; Idaho.",
        ] {
            assert_eq!(detect(text), vec![], "{text}");
        }
    }

    #[test]
    fn short_forms() {
        let found = detect("Smith, supra, at 5. supra note 3. Ibid. Supra.");
        let names: Vec<(&str, usize)> = found.iter().map(|r| (r.1.as_str(), r.2)).collect();
        assert_eq!(names, [("supra,", 7), ("supra", 20), ("Ibid.", 34), ("Supra.", 40)]);
        assert_eq!(found[2].4, json!({"is_short_form": true}));
        assert_eq!(found[0].4, json!({}));
    }

    #[test]
    fn mentions_carry_ids_scores_and_char_offsets() {
        let mentions = CitationDetector.detect("Zoë: 42 U.S.C. § 1983.", "d_0").expect("detect");
        assert_eq!(mentions.len(), 1);
        let m = &mentions[0];
        assert_eq!((m.span.start, m.span.end), (5, 21));
        assert_eq!(m.name, "42 U.S.C. § 1983");
        assert_eq!(m.score, Some(1.0));
        assert_eq!(m.source.as_deref(), Some("eyecite"));
        assert_eq!(m.id, span_id("d_0", "eyecite", 5, 21, "CASE_CITATION"));
    }
}
