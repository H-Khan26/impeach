//! Python `str` and `json` semantics the loaders rely on: text-mode file reading, `splitlines`,
//! `isdigit` / `isupper`, truthiness, `str()` / `repr()` of JSON values.

use std::fmt::Write as _;
use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

use crate::error::{IngestError, Result};

/// Reads a UTF-8 file the way Python's text mode does (`open(path, encoding="utf-8")`), except
/// that a leading byte-order mark is dropped: `\r\n` and lone `\r` become `\n`.
pub(crate) fn read_text(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)
        .map_err(|source| IngestError::Io { path: path.to_path_buf(), source })?;
    let body = strip_bom(&bytes);
    let text = std::str::from_utf8(body)
        .map_err(|source| IngestError::Encoding { path: path.to_path_buf(), source })?;
    Ok(translate_newlines(text))
}

/// `bytes` without a leading UTF-8 byte-order mark.
pub(crate) fn strip_bom(bytes: &[u8]) -> &[u8] {
    bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes)
}

/// Python universal-newline translation: `\r\n` and lone `\r` become `\n`.
pub(crate) fn translate_newlines(text: &str) -> String {
    if !text.contains('\r') {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\r' {
            if chars.peek() == Some(&'\n') {
                chars.next();
            }
            out.push('\n');
        } else {
            out.push(c);
        }
    }
    out
}

/// Line boundaries of Python `str.splitlines()`.
fn is_line_boundary(c: char) -> bool {
    matches!(
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
    )
}

/// Python `str.splitlines()`: splits on every line boundary (`\r\n` counts once) without a
/// trailing empty piece.
pub(crate) fn splitlines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if is_line_boundary(c) {
            lines.push(&text[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' && chars.peek().is_some_and(|&(_, next)| next == '\n') {
                chars.next();
                end += 1;
            }
            start = end;
        }
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

/// Python `str.rstrip()`.
pub(crate) fn rstrip(text: &str) -> &str {
    text.trim_end_matches(ce_text::normalize::is_py_whitespace)
}

/// Digits for Python `str.isdigit()`: Unicode decimal digits plus the superscript, subscript and
/// circled digits (Numeric_Type=Digit).
static DIGITS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^[\d\x{B2}\x{B3}\x{B9}\x{2070}\x{2074}-\x{2079}\x{2080}-\x{2089}\x{2460}-\x{2468}\x{24F5}-\x{24FD}\x{2776}-\x{277E}\x{2780}-\x{2788}\x{278A}-\x{2792}]+$",
    )
    .expect("digit pattern is valid")
});

/// Python `str.isdigit()` (false for the empty string).
pub(crate) fn isdigit(text: &str) -> bool {
    DIGITS.is_match(text)
}

/// Unicode titlecase letters (general category Lt).
fn is_titlecase(c: char) -> bool {
    matches!(
        c,
        '\u{01C5}'
            | '\u{01C8}'
            | '\u{01CB}'
            | '\u{01F2}'
            | '\u{1F88}'..='\u{1F8F}'
            | '\u{1F98}'..='\u{1F9F}'
            | '\u{1FA8}'..='\u{1FAF}'
            | '\u{1FBC}'
            | '\u{1FCC}'
            | '\u{1FFC}'
    )
}

/// Python `str.isupper()`: at least one uppercase character and no lowercase or titlecase one.
pub(crate) fn isupper(text: &str) -> bool {
    let mut cased = false;
    for c in text.chars() {
        if c.is_lowercase() || is_titlecase(c) {
            return false;
        }
        cased |= c.is_uppercase();
    }
    cased
}

/// Python truthiness of a JSON value.
pub(crate) fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// Python `str()` of a value decoded by `json.loads`.
pub(crate) fn py_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => {
            let mut out = String::new();
            write_repr(&mut out, other);
            out
        }
    }
}

/// Appends Python `repr()` of a value decoded by `json.loads`.
fn write_repr(out: &mut String, value: &Value) {
    match value {
        Value::Null => out.push_str("None"),
        Value::Bool(true) => out.push_str("True"),
        Value::Bool(false) => out.push_str("False"),
        Value::Number(n) => match n.as_f64().filter(|_| n.is_f64()) {
            Some(f) => out.push_str(&ce_core::pyjson::float_repr(f)),
            None => out.push_str(&n.to_string()),
        },
        Value::String(s) => write_str_repr(out, s),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_repr(out, item);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (key, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_str_repr(out, key);
                out.push_str(": ");
                write_repr(out, item);
            }
            out.push('}');
        }
    }
}

/// Appends Python `repr()` of a string: single quotes unless the text contains `'` and no `"`;
/// every non-printable character (see `is_py_printable`) is escaped.
fn write_str_repr(out: &mut String, text: &str) {
    let quote = if text.contains('\'') && !text.contains('"') { '"' } else { '\'' };
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if !is_py_printable(c) => {
                let code = u32::from(c);
                // Writing to a String cannot fail.
                let _ = if code < 0x100 {
                    write!(out, "\\x{code:02x}")
                } else if code < 0x1_0000 {
                    write!(out, "\\u{code:04x}")
                } else {
                    write!(out, "\\U{code:08x}")
                };
            }
            c => out.push(c),
        }
    }
    out.push(quote);
}

/// Python `str.isprintable()` for one character: false for the general categories Cc, Cf, Cs,
/// Co, Cn, Zl, Zp and Zs, except the ASCII space. Category data comes from
/// `unicode-general-category`, which may know characters newer than the running Python's
/// Unicode database.
fn is_py_printable(c: char) -> bool {
    use unicode_general_category::{get_general_category, GeneralCategory};
    c == ' '
        || !matches!(
            get_general_category(c),
            GeneralCategory::Control
                | GeneralCategory::Format
                | GeneralCategory::Surrogate
                | GeneralCategory::PrivateUse
                | GeneralCategory::Unassigned
                | GeneralCategory::LineSeparator
                | GeneralCategory::ParagraphSeparator
                | GeneralCategory::SpaceSeparator
        )
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;

    #[test]
    fn translates_universal_newlines() {
        assert_eq!(translate_newlines("a\r\nb\rc\n"), "a\nb\nc\n");
        assert_eq!(translate_newlines("\r\r\n"), "\n\n");
    }

    #[test]
    fn splitlines_matches_python() {
        assert_eq!(splitlines(""), Vec::<&str>::new());
        assert_eq!(splitlines("\n"), vec![""]);
        assert_eq!(splitlines("a\n"), vec!["a"]);
        assert_eq!(
            splitlines("a\r\nb\rc\u{0b}d\u{2028}e\n\nf"),
            vec!["a", "b", "c", "d", "e", "", "f"]
        );
    }

    #[test]
    fn isdigit_and_isupper_match_python() {
        assert!(isdigit("12"));
        assert!(isdigit("\u{0663}\u{00B2}"));
        assert!(!isdigit(""));
        assert!(!isdigit("1 2"));
        assert!(!isdigit("\u{00BD}"));
        assert!(isupper("1   RANDOM HEADER"));
        assert!(isupper("ÉTÉ"));
        assert!(!isupper("123"));
        assert!(!isupper("Random"));
        assert!(!isupper("A\u{01C5}"));
    }

    #[test]
    fn truthiness_matches_python() {
        let falsy =
            [json!(null), json!(false), json!(0), json!(0.0), json!(""), json!([]), json!({})];
        for value in falsy {
            assert!(!truthy(&value), "{value}");
        }
        let truthy_values =
            [json!(true), json!(-1), json!(0.5), json!(" "), json!([0]), json!({"a": null})];
        for value in truthy_values {
            assert!(truthy(&value), "{value}");
        }
    }

    #[test]
    fn py_str_matches_python() {
        assert_eq!(py_str(&json!("x")), "x");
        assert_eq!(py_str(&json!(null)), "None");
        assert_eq!(py_str(&json!(true)), "True");
        assert_eq!(py_str(&json!(2)), "2");
        assert_eq!(py_str(&json!(1.5)), "1.5");
        assert_eq!(py_str(&json!([1, "a"])), "[1, 'a']");
        assert_eq!(py_str(&json!({"k": null})), "{'k': None}");
        assert_eq!(
            py_str(&json!(["it's", "say \"hi\"\n", "a\u{a0}b"])),
            r#"["it's", 'say "hi"\n', 'a\xa0b']"#
        );
        assert_eq!(py_str(&json!(["both ' and \""])), r#"['both \' and "']"#);
        // Format, private-use, unassigned and separator characters are escaped as in Python.
        assert_eq!(py_str(&json!(["a\u{200b}b\u{feff}c\u{ad}d\u{2028}e\u{f0000}f\u{378}g\u{85} h\u{3000}\u{e9}\u{1f600}"])), r#"['a\u200bb\ufeffc\xadd\u2028e\U000f0000f\u0378g\x85 h\u3000é😀']"#);
    }
}
