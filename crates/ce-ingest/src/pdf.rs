//! PDF text cleaning: repeated header / footer removal across pages and line-break
//! normalisation into paragraphs, plus page extraction through `pdf-extract`.
//!
//! The cleaning is a byte-exact port of the Python loader. Extraction itself is not: Python uses
//! pdfplumber, whose per-page text differs from `pdf-extract`, so PDF output is not
//! byte-comparable across implementations.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use regex::Regex;

use crate::py;

/// Header / footer cleaning settings (Python `DocumentLoader` `pdf_*` arguments).
#[derive(Debug, Clone, PartialEq)]
pub struct PdfCleaning {
    /// Top / bottom lines per page considered as header / footer candidates.
    pub header_footer_max_lines: usize,
    /// Fraction of pages a normalised line must appear on to be removed.
    pub header_footer_min_repetition: f64,
    /// Remove repeated header lines.
    pub clean_headers: bool,
    /// Remove repeated footer lines.
    pub clean_footers: bool,
    /// Drop short, low-information lines.
    pub drop_short_lines: bool,
    /// Maximum length (characters) of a line considered for short-line dropping.
    pub short_line_max_chars: usize,
    /// Maximum average word length of a dropped short line.
    pub short_line_max_avg_token_len: f64,
}

impl Default for PdfCleaning {
    fn default() -> Self {
        Self {
            header_footer_max_lines: 10,
            header_footer_min_repetition: 0.8,
            clean_headers: true,
            clean_footers: true,
            drop_short_lines: false,
            short_line_max_chars: 40,
            short_line_max_avg_token_len: 2.5,
        }
    }
}

static DIGIT_RUN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\d+").expect("digit-run pattern is valid"));
static WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\w+").expect("word pattern is valid"));
static SPACE_RUN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(" {2,}").expect("space-run pattern is valid"));
static NEWLINE_RUN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new("\n{2,}").expect("newline-run pattern is valid"));

/// Paragraph placeholder used while unwrapping single line breaks.
const PARA: &str = "\x00PARA\x00";

/// Comparison key of a header / footer candidate: whitespace collapsed, digit runs (page
/// numbers, dates) replaced by `#`.
fn header_footer_key(line: &str) -> String {
    let collapsed = ce_text::normalize_whitespace(line);
    if collapsed.is_empty() {
        return collapsed;
    }
    DIGIT_RUN.replace_all(&collapsed, "#").into_owned()
}

impl PdfCleaning {
    /// True for a short line whose words average at most `short_line_max_avg_token_len`
    /// characters (decorative rails, line-number rows).
    #[must_use]
    pub fn is_short_noisy_line(&self, line: &str) -> bool {
        let s = ce_text::normalize::py_strip(line);
        if s.is_empty() {
            return true;
        }
        if s.chars().count() > self.short_line_max_chars {
            return false;
        }
        let lengths: Vec<usize> = WORD.find_iter(s).map(|m| m.as_str().chars().count()).collect();
        if lengths.is_empty() {
            return true;
        }
        let avg = lengths.iter().sum::<usize>() as f64 / lengths.len() as f64;
        avg <= self.short_line_max_avg_token_len
    }

    /// Removes header / footer lines that repeat (after normalisation) on at least
    /// `max(2, int(min_repetition * pages))` occasions, optionally drops short noisy lines,
    /// right-strips lines and joins the non-empty pages with a blank line (`"\n\n"`).
    #[must_use]
    pub fn clean_pages<S: AsRef<str>>(&self, pages: &[S]) -> String {
        if pages.is_empty() {
            return String::new();
        }
        let pages_lines: Vec<Vec<&str>> =
            pages.iter().map(|p| py::splitlines(p.as_ref())).collect();
        let max_lines = self.header_footer_max_lines;

        let mut header_remove = HashSet::new();
        let mut footer_remove = HashSet::new();
        if self.clean_headers || self.clean_footers {
            let mut header_counts: HashMap<String, usize> = HashMap::new();
            let mut footer_counts: HashMap<String, usize> = HashMap::new();
            for lines in pages_lines.iter().filter(|lines| !lines.is_empty()) {
                let n = lines.len();
                if self.clean_headers {
                    for line in &lines[..max_lines.min(n)] {
                        let key = header_footer_key(line);
                        if !key.is_empty() {
                            *header_counts.entry(key).or_default() += 1;
                        }
                    }
                }
                if self.clean_footers {
                    for line in &lines[n.saturating_sub(max_lines)..] {
                        let key = header_footer_key(line);
                        if !key.is_empty() {
                            *footer_counts.entry(key).or_default() += 1;
                        }
                    }
                }
            }
            // Python: max(2, int(min_repetition * num_pages)); the cast truncates like int().
            let scaled = (self.header_footer_min_repetition * pages_lines.len() as f64) as usize;
            let min_rep = scaled.max(2);
            if self.clean_headers {
                header_remove = keys_at_least(header_counts, min_rep);
            }
            if self.clean_footers {
                footer_remove = keys_at_least(footer_counts, min_rep);
            }
        }

        let mut cleaned_pages: Vec<String> = Vec::new();
        for lines in pages_lines.iter().filter(|lines| !lines.is_empty()) {
            let n = lines.len();
            let header_end = max_lines.min(n);
            let footer_start = n.saturating_sub(max_lines);
            let mut kept: Vec<&str> = Vec::with_capacity(n);
            for (idx, line) in lines.iter().enumerate() {
                let key = header_footer_key(line);
                let mut drop = (self.clean_headers
                    && idx < header_end
                    && header_remove.contains(&key))
                    || (self.clean_footers && idx >= footer_start && footer_remove.contains(&key));
                if !drop && self.drop_short_lines {
                    drop = self.is_short_noisy_line(line);
                }
                if !drop {
                    kept.push(py::rstrip(line));
                }
            }
            let joined = kept.join("\n");
            let page_text = ce_text::normalize::py_strip(&joined);
            if !page_text.is_empty() {
                cleaned_pages.push(page_text.to_owned());
            }
        }
        cleaned_pages.join("\n\n")
    }
}

fn keys_at_least(counts: HashMap<String, usize>, min_rep: usize) -> HashSet<String> {
    counts.into_iter().filter_map(|(key, count)| (count >= min_rep).then_some(key)).collect()
}

/// Unwraps layout line breaks into paragraphs (Python `_normalize_pdf_linebreaks`): runs of two
/// or more newlines become a paragraph break (`"\n\n"`), `-\n` before a lowercase ASCII letter
/// is removed (de-hyphenation), every other single newline becomes a space (`\n` before a NUL
/// is kept), and runs of spaces collapse to one.
#[must_use]
pub fn normalize_pdf_linebreaks(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let text = NEWLINE_RUN.replace_all(text, PARA);

    // `-\n(?=[a-z])` -> ""
    let mut dehyphenated = String::with_capacity(text.len());
    let mut rest: &str = &text;
    while let Some(pos) = rest.find("-\n") {
        let after = &rest[pos + 2..];
        if after.starts_with(|c: char| c.is_ascii_lowercase()) {
            dehyphenated.push_str(&rest[..pos]);
        } else {
            dehyphenated.push_str(&rest[..pos + 2]);
        }
        rest = after;
    }
    dehyphenated.push_str(rest);

    // `(?<![.!?:;])\n(?=[A-Za-z])` -> " ", then `\n(?!\x00)` -> " ". Both passes read the
    // original neighbours; together they turn every newline not followed by NUL into a space.
    let mut unwrapped = String::with_capacity(dehyphenated.len());
    let mut chars = dehyphenated.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\n' && chars.peek() != Some(&'\x00') {
            unwrapped.push(' ');
        } else {
            unwrapped.push(c);
        }
    }

    let restored = unwrapped.replace(PARA, "\n\n");
    SPACE_RUN.replace_all(&restored, " ").into_owned()
}

/// Extracts the text of every page with `pdf-extract`. A panic inside the extractor is reported
/// as an error.
#[cfg(feature = "pdf")]
pub(crate) fn extract_pages(path: &std::path::Path) -> Result<Vec<String>, String> {
    match std::panic::catch_unwind(|| pdf_extract::extract_text_by_pages(path)) {
        Ok(Ok(pages)) => Ok(pages),
        Ok(Err(err)) => Err(err.to_string()),
        Err(panic) => Err(panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
            .unwrap_or_else(|| "PDF extractor panicked".to_owned())),
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    fn clean_and_normalize(pages: &[&str]) -> (String, String) {
        let cleaned = PdfCleaning::default().clean_pages(pages);
        let normalized = normalize_pdf_linebreaks(&cleaned);
        (cleaned, normalized)
    }

    // Expected values produced by the Python DocumentLoader with run_ingestion's settings.
    #[test]
    fn removes_repeated_headers_and_footers() {
        let pages = [
            "ACME CORP CONFIDENTIAL\nPage 1 of 3\nThomas\nReynolds met the\nemploy-\nment board.\nFooter line 1",
            "ACME CORP CONFIDENTIAL\nPage 2 of 3\nSecond page text ends here.\nNew sentence: starts\n42 items\nFooter line 2",
            "ACME CORP CONFIDENTIAL\nPage 3 of 3\nThird page.   Spaced   out.\nFooter line 3   ",
        ];
        let (cleaned, normalized) = clean_and_normalize(&pages);
        assert_eq!(
            cleaned,
            "Thomas\nReynolds met the\nemploy-\nment board.\n\nSecond page text ends here.\nNew sentence: starts\n42 items\n\nThird page.   Spaced   out."
        );
        assert_eq!(
            normalized,
            "Thomas Reynolds met the employment board.\n\nSecond page text ends here. New sentence: starts 42 items\n\nThird page. Spaced out."
        );
    }

    #[test]
    fn single_page_keeps_every_line() {
        let (cleaned, normalized) = clean_and_normalize(&["Only page\nwith two lines"]);
        assert_eq!(cleaned, "Only page\nwith two lines");
        assert_eq!(normalized, "Only page with two lines");
    }

    #[test]
    fn splits_pages_on_python_line_boundaries() {
        let (cleaned, normalized) =
            clean_and_normalize(&["Head\nBody one\r\nmore\u{0b}vt", "Head\nBody two\u{2028}sep"]);
        assert_eq!(cleaned, "Body one\nmore\nvt\n\nBody two\nsep");
        assert_eq!(normalized, "Body one more vt\n\nBody two sep");
    }

    #[test]
    fn empty_pages_give_empty_text() {
        let none: [&str; 0] = [];
        assert_eq!(PdfCleaning::default().clean_pages(&none), "");
        assert_eq!(PdfCleaning::default().clean_pages(&["  \n \n"]), "");
    }

    #[test]
    fn normalizes_line_breaks_like_python() {
        let cases = [
            ("a\nb", "a b"),
            ("end.\nNext", "end. Next"),
            ("x-\ny", "xy"),
            ("x-\nY", "x- Y"),
            ("1\n2", "1 2"),
            ("a\n\n\nb", "a\n\nb"),
            ("a\n\x00b", "a\n\x00b"),
            ("\nStart", " Start"),
            ("a  b   c", "a b c"),
            ("Q:\nfoo", "Q: foo"),
            ("café\nnext", "café next"),
            ("a\n1", "a 1"),
            ("", ""),
        ];
        for (input, expected) in cases {
            assert_eq!(normalize_pdf_linebreaks(input), expected, "{input:?}");
        }
    }

    #[test]
    fn short_noisy_lines() {
        let cleaning = PdfCleaning { drop_short_lines: true, ..PdfCleaning::default() };
        assert!(cleaning.is_short_noisy_line("   "));
        assert!(cleaning.is_short_noisy_line("1 2 3 4"));
        assert!(cleaning.is_short_noisy_line("| | |"));
        assert!(!cleaning.is_short_noisy_line("Plaintiff alleges"));
        assert!(!cleaning.is_short_noisy_line(&"a ".repeat(30)));
        assert_eq!(
            cleaning.clean_pages(&["Real sentence here\n1 2 3\nAnother real line"]),
            "Real sentence here\nAnother real line"
        );
    }
}
