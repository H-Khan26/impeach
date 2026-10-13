//! Text and value helpers with Python semantics: character offsets over UTF-8 text and JSON
//! truthiness.

use serde_json::Value;

/// Maps between character offsets (Python `str` indices) and byte offsets of one text.
#[derive(Debug)]
pub(crate) struct CharIndex<'a> {
    text: &'a str,
    /// Byte offset of every character, plus `text.len()` at the end.
    bytes: Vec<usize>,
}

impl<'a> CharIndex<'a> {
    pub(crate) fn new(text: &'a str) -> Self {
        let mut bytes: Vec<usize> = text.char_indices().map(|(b, _)| b).collect();
        bytes.push(text.len());
        Self { text, bytes }
    }

    /// Number of characters.
    pub(crate) fn char_len(&self) -> usize {
        self.bytes.len() - 1
    }

    /// Character offset of a byte offset that falls on a character boundary.
    pub(crate) fn char_of_byte(&self, byte: usize) -> usize {
        self.bytes.partition_point(|&b| b < byte)
    }

    /// Characters `start..end`, clamped to the text.
    pub(crate) fn slice(&self, start: usize, end: usize) -> &'a str {
        let len = self.char_len();
        let (start, end) = (start.min(len), end.min(len));
        if start >= end {
            return "";
        }
        &self.text[self.bytes[start]..self.bytes[end]]
    }
}

/// Python truthiness of a JSON value.
pub(crate) fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn maps_multibyte_offsets() {
        let index = CharIndex::new("Zoë § 12");
        assert_eq!(index.char_len(), 8);
        assert_eq!(index.char_of_byte(4), 3);
        assert_eq!(index.char_of_byte(index.text.len()), 8);
        assert_eq!(index.slice(0, 3), "Zoë");
        assert_eq!(index.slice(4, 5), "§");
        assert_eq!(index.slice(6, 99), "12");
        assert_eq!(index.slice(5, 2), "");
    }

    #[test]
    fn truthiness_matches_python() {
        for falsy in
            [json!(null), json!(false), json!(0), json!(0.0), json!(""), json!([]), json!({})]
        {
            assert!(!is_truthy(&falsy), "{falsy}");
        }
        for truthy in [json!(true), json!(1), json!(-0.5), json!("x"), json!([0]), json!({"a": 0})]
        {
            assert!(is_truthy(&truthy), "{truthy}");
        }
    }
}
