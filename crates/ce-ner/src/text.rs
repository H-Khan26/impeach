//! Character-offset helpers over UTF-8 text.

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

#[cfg(test)]
mod tests {
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
}
