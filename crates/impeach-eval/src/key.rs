//! Markdown contradiction key parsing.
//!
//! A key has an "Interdocument Contradictions" section whose table rows with an id starting
//! `C-` give (Doc 1 fact, Doc 2 fact) pairs, and optional "Intradocument Contradictions"
//! sections (`D*` rows) whose facts both come from the document named in the heading
//! (`(Doc N)`, else the section's ordinal). Only `## ` headings switch sections; the first
//! table row after a non-table line is the header row and is skipped.
//!
//! Regex semantics follow Python `re`; `\s` includes U+001C..U+001F as in Python.

use std::path::Path;
use std::sync::LazyLock;

use impeach_text::normalize::{normalize_whitespace, py_strip};
use regex::Regex;

use crate::error::{EvalError, Result};

/// One fact of a key contradiction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroundTruthFact {
    /// Cleaned fact text (paragraph refs, entity ids and markdown removed).
    pub text: String,
    /// `Doc 1`, `Doc 2`, ...
    pub source: String,
    /// Paragraph reference (`¶6`, `¶23 vs ¶24`), if any.
    pub paragraph_ref: Option<String>,
}

/// One key row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroundTruthContradiction {
    /// Row id, e.g. `C-01`, `D1-INTRA-01`.
    pub id: String,
    /// Type column.
    pub contradiction_type: String,
    /// First fact.
    pub fact_a: GroundTruthFact,
    /// Second fact.
    pub fact_b: GroundTruthFact,
    /// Notes column (empty when absent).
    pub notes: String,
    /// True for `C-*` rows, false for `D*` rows.
    pub is_inter_doc: bool,
}

/// A parsed key: every contradiction row in file order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Key {
    /// Rows in file order.
    pub contradictions: Vec<GroundTruthContradiction>,
}

impl Key {
    /// Number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.contradictions.len()
    }

    /// True when no row was parsed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.contradictions.is_empty()
    }

    /// Inter-document (`C-*`) rows.
    pub fn inter_doc(&self) -> impl Iterator<Item = &GroundTruthContradiction> {
        self.contradictions.iter().filter(|g| g.is_inter_doc)
    }

    /// Intra-document (`D*`) rows.
    pub fn intra_doc(&self) -> impl Iterator<Item = &GroundTruthContradiction> {
        self.contradictions.iter().filter(|g| !g.is_inter_doc)
    }
}

macro_rules! ws {
    () => {
        r"[\s\x1C-\x1F]"
    };
}

fn compile(pattern: &str) -> Regex {
    Regex::new(pattern).expect("key-parser patterns are valid")
}

static TABLE_SEPARATOR: LazyLock<Regex> =
    LazyLock::new(|| compile(concat!(r"^\|?", ws!(), r"*:?-{2,}")));
static PARA_REF: LazyLock<Regex> = LazyLock::new(|| {
    compile(concat!(r"[¶P]\d+(?:", ws!(), r"*(?:vs|,|–|-)", ws!(), r"*[¶P]?\d+)*:", ws!(), "*"))
});
static PARA_REF_PREFIX: LazyLock<Regex> = LazyLock::new(|| {
    compile(concat!(r"^([P¶]\d+(?:", ws!(), r"*(?:vs|,|–|-)", ws!(), r"*[P¶]?\d+)*):"))
});
static ENTITY_REF: LazyLock<Regex> =
    LazyLock::new(|| compile(concat!(r"\(", ws!(), r"*`[^`]*`", ws!(), r"*\)")));
static BOLD: LazyLock<Regex> = LazyLock::new(|| compile(r"\*\*([^*]+)\*\*"));
static QUOTED: LazyLock<Regex> = LazyLock::new(|| compile(r#"^["“]([^"“”]+)["”]\.?$"#));
static DOC_NUMBER: LazyLock<Regex> =
    LazyLock::new(|| compile(concat!(r"\(Doc", ws!(), r"*(\d+)\)")));

/// Splits a markdown table row into stripped cells, dropping one leading and one trailing `|`.
#[must_use]
pub fn parse_table_row(row: &str) -> Vec<String> {
    let mut row = py_strip(row);
    if let Some(rest) = row.strip_prefix('|') {
        row = rest;
    }
    if let Some(rest) = row.strip_suffix('|') {
        row = rest;
    }
    row.split('|').map(|cell| py_strip(cell).to_owned()).collect()
}

/// True for a table separator row (`|---|`, `| :--- |`).
#[must_use]
pub fn is_table_separator(row: &str) -> bool {
    TABLE_SEPARATOR.is_match(py_strip(row))
}

/// Strips paragraph refs (`¶6:`), entity-id refs (`` (`LOC-001`) ``), backticks, bold
/// markers and one pair of enclosing quotes, and collapses whitespace.
#[must_use]
pub fn clean_fact_text(text: &str) -> String {
    let text = PARA_REF.replace_all(text, "");
    let text = ENTITY_REF.replace_all(&text, "");
    let text = text.replace('`', "");
    let text = BOLD.replace_all(&text, "${1}");
    let text = py_strip(&text);
    let text = QUOTED.replace(text, "${1}");
    normalize_whitespace(&text)
}

/// The leading paragraph reference of a key cell (`¶23 vs ¶24` in `¶23 vs ¶24: ...`).
#[must_use]
pub fn extract_paragraph_ref(text: &str) -> Option<String> {
    PARA_REF_PREFIX.captures(py_strip(text)).and_then(|c| c.get(1)).map(|m| m.as_str().to_owned())
}

fn intra_doc_source(header: &str, section_index: usize) -> String {
    match DOC_NUMBER.captures(header).and_then(|c| c.get(1)) {
        Some(number) => format!("Doc {}", number.as_str()),
        None => format!("Doc {section_index}"),
    }
}

/// Python `str.splitlines()`.
fn py_splitlines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((index, c)) = chars.next() {
        let is_break = matches!(
            c,
            '\n' | '\r'
                | '\u{0b}'
                | '\u{0c}'
                | '\u{1c}'
                | '\u{1d}'
                | '\u{1e}'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        );
        if !is_break {
            continue;
        }
        lines.push(&text[start..index]);
        let mut end = index + c.len_utf8();
        if c == '\r' {
            if let Some(&(next, '\n')) = chars.peek() {
                chars.next();
                end = next + 1;
            }
        }
        start = end;
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Section {
    Inter,
    Intra,
}

/// Parses the inter-document (`C-*`) and intra-document (`D*`) contradiction tables.
#[must_use]
pub fn parse_key(markdown: &str) -> Key {
    let mut contradictions = Vec::new();
    let mut section = None;
    let mut intra_source = "Doc 1".to_owned();
    let mut intra_sections_seen = 0;
    let mut in_table = false;

    for line in py_splitlines(markdown) {
        let stripped = py_strip(line);
        if stripped.starts_with("## ") {
            in_table = false;
            if stripped.contains("Interdocument Contradictions") {
                section = Some(Section::Inter);
            } else if stripped.contains("Intradocument Contradictions") {
                section = Some(Section::Intra);
                intra_sections_seen += 1;
                intra_source = intra_doc_source(stripped, intra_sections_seen);
            } else {
                section = None;
            }
            continue;
        }
        let Some(current) = section else {
            continue;
        };
        if !stripped.contains('|') {
            in_table = false;
            continue;
        }
        if is_table_separator(stripped) {
            continue;
        }
        if !in_table {
            in_table = true;
            continue;
        }

        let cells = parse_table_row(stripped);
        let notes = |index: usize| cells.get(index).cloned().unwrap_or_default();
        match current {
            Section::Inter if cells.len() >= 4 && cells[0].starts_with("C-") => {
                contradictions.push(GroundTruthContradiction {
                    id: cells[0].clone(),
                    contradiction_type: cells[1].clone(),
                    fact_a: GroundTruthFact {
                        text: clean_fact_text(&cells[2]),
                        source: "Doc 1".to_owned(),
                        paragraph_ref: extract_paragraph_ref(&cells[2]),
                    },
                    fact_b: GroundTruthFact {
                        text: clean_fact_text(&cells[3]),
                        source: "Doc 2".to_owned(),
                        paragraph_ref: extract_paragraph_ref(&cells[3]),
                    },
                    notes: notes(4),
                    is_inter_doc: true,
                });
            }
            Section::Intra if cells.len() >= 5 && cells[0].starts_with('D') => {
                contradictions.push(GroundTruthContradiction {
                    id: cells[0].clone(),
                    contradiction_type: cells[1].clone(),
                    fact_a: GroundTruthFact {
                        text: clean_fact_text(&cells[3]),
                        source: intra_source.clone(),
                        paragraph_ref: Some(cells[2].clone()),
                    },
                    fact_b: GroundTruthFact {
                        text: clean_fact_text(&cells[4]),
                        source: intra_source.clone(),
                        paragraph_ref: Some(cells[2].clone()),
                    },
                    notes: notes(5),
                    is_inter_doc: false,
                });
            }
            _ => {}
        }
    }
    Key { contradictions }
}

/// Reads and parses a key file (UTF-8).
///
/// # Errors
/// [`EvalError::Io`] when the file cannot be read.
pub fn parse_key_file(path: &Path) -> Result<Key> {
    let text = std::fs::read_to_string(path)
        .map_err(|source| EvalError::Io { path: path.to_path_buf(), source })?;
    Ok(parse_key(&text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_rows_and_separators() {
        assert_eq!(parse_table_row("| a | b |c|"), vec!["a", "b", "c"]);
        assert_eq!(parse_table_row("a | b"), vec!["a", "b"]);
        assert_eq!(parse_table_row("| a || b |"), vec!["a", "", "b"]);
        assert!(is_table_separator("| ---- | --- |"));
        assert!(is_table_separator("  |:--|"));
        assert!(is_table_separator("--|--"));
        assert!(!is_table_separator("| C-01 | x |"));
        assert!(!is_table_separator("|-|"));
    }

    #[test]
    fn cleans_fact_text() {
        assert_eq!(
            clean_fact_text("¶6: Collision date: **March 5, 2023**"),
            "Collision date: March 5, 2023"
        );
        assert_eq!(clean_fact_text("¶6: “Approximately **5:30 p.m.**”"), "Approximately 5:30 p.m.");
        assert_eq!(clean_fact_text("**Eastvale, California** (`LOC-001`)"), "Eastvale, California");
        assert_eq!(clean_fact_text("¶20 vs ¶31: \"quoted\"."), "quoted");
        assert_eq!(clean_fact_text("P3, P4: two  spaces `x`"), "two spaces x");
        assert_eq!(clean_fact_text("“a” and “b”"), "“a” and “b”");
    }

    #[test]
    fn paragraph_refs() {
        assert_eq!(extract_paragraph_ref(" ¶23 vs ¶24: text").as_deref(), Some("¶23 vs ¶24"));
        assert_eq!(extract_paragraph_ref("¶28 vs ¶29–30: x").as_deref(), Some("¶28 vs ¶29–30"));
        assert_eq!(extract_paragraph_ref("P6: x").as_deref(), Some("P6"));
        assert_eq!(extract_paragraph_ref("text ¶6: x"), None);
    }

    #[test]
    fn splitlines_matches_python() {
        assert_eq!(py_splitlines("a\r\nb\rc\n\nd\u{2028}e\n"), vec!["a", "b", "c", "", "d", "e"]);
        assert!(py_splitlines("").is_empty());
    }

    #[test]
    fn parses_sections_and_sources() {
        let md = "\
# Key
## 1. Interdocument Contradictions
| ID | Type | Doc 1 | Doc 2 | Notes |
| -- | ---- | ----- | ----- | ----- |
| C-01 | Date | ¶6: **March 5, 2023** | ¶2: March 4, 2023 | differs |
| X-01 | Date | a | b | ignored id |
| C-02 | Short | a | b |
text breaks the table
| C-03 | Header | treated | as header |
| C-04 | Kept | a | b |
## 2. Intradocument Contradictions – Report (Doc 2)
| ID | Type | ¶ | A | B |
|---|---|---|---|---|
| D2-INTRA-01 | T | ¶1 vs ¶2 | first | second | n |
## 3. Intradocument Contradictions – Other
| ID | Type | ¶ | A | B |
|---|---|---|---|---|
| D3-INTRA-01 | T | ¶1 | x | y |
| D3-INTRA-02 | too short | ¶1 | x |
## Entities
| C-09 | ignored | a | b |
";
        let key = parse_key(md);
        let ids: Vec<&str> = key.contradictions.iter().map(|g| g.id.as_str()).collect();
        assert_eq!(ids, ["C-01", "C-02", "C-04", "D2-INTRA-01", "D3-INTRA-01"]);
        let c01 = &key.contradictions[0];
        assert_eq!(c01.fact_a.text, "March 5, 2023");
        assert_eq!(c01.fact_a.paragraph_ref.as_deref(), Some("¶6"));
        assert_eq!(c01.fact_b.source, "Doc 2");
        assert_eq!(c01.notes, "differs");
        assert_eq!(key.contradictions[1].notes, "");
        let d2 = &key.contradictions[3];
        assert_eq!((d2.fact_a.source.as_str(), d2.notes.as_str()), ("Doc 2", "n"));
        assert_eq!(d2.fact_b.paragraph_ref.as_deref(), Some("¶1 vs ¶2"));
        assert_eq!(key.contradictions[4].fact_a.source, "Doc 2");
        assert_eq!(key.inter_doc().count(), 3);
        assert_eq!(key.intra_doc().count(), 2);
    }
}
