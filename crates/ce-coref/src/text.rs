//! Character-offset access to a document text.

use ce_core::IndexingResult;

/// A text with a character -> byte offset table, for slicing by character offsets.
#[derive(Debug, Clone)]
pub(crate) struct CharText<'a> {
    text: &'a str,
    /// Byte offset of each character, plus the text length at the end.
    char_bytes: Vec<usize>,
}

impl<'a> CharText<'a> {
    pub(crate) fn new(text: &'a str) -> Self {
        let mut char_bytes: Vec<usize> = text.char_indices().map(|(b, _)| b).collect();
        char_bytes.push(text.len());
        Self { text, char_bytes }
    }

    /// Number of characters.
    pub(crate) fn char_len(&self) -> usize {
        self.char_bytes.len() - 1
    }

    /// Characters `start..end`; `None` unless `start <= end <= len`.
    pub(crate) fn slice(&self, start: usize, end: usize) -> Option<&'a str> {
        (start <= end && end <= self.char_len())
            .then(|| &self.text[self.char_bytes[start]..self.char_bytes[end]])
    }
}

/// The document text rebuilt from chunk texts in `sequence_number` order (Python
/// `_reconstruct_doc_text`; offsets only line up when chunks partition the text).
pub fn reconstruct_doc_text(result: &IndexingResult) -> String {
    let mut chunks: Vec<_> = result.chunks.iter().map(|w| &w.chunk).collect();
    chunks.sort_by_key(|c| c.sequence_number);
    chunks.into_iter().map(|c| c.text.as_str()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slices_by_characters() {
        let text = CharText::new("Zoë said");
        assert_eq!(text.char_len(), 8);
        assert_eq!(text.slice(0, 3), Some("Zoë"));
        assert_eq!(text.slice(4, 8), Some("said"));
        assert_eq!(text.slice(4, 9), None);
        assert_eq!(text.slice(5, 4), None);
    }
}
