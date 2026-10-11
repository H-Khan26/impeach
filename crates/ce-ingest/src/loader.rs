//! Standard document loaders: `.pdf`, `.txt` and `.json` (Python `DocumentLoader`).

use std::path::Path;

use serde_json::Value;

use crate::error::{IngestError, Result};
use crate::pdf::PdfCleaning;
use crate::py;

/// A document read from disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedDocument {
    /// The file stem, or the `doc_id` of a JSON document.
    pub doc_id: String,
    /// Extracted text.
    pub text: String,
    /// Page count of a PDF (`None` for zero pages), or the `total_pages` of a JSON document.
    pub total_pages: Option<u32>,
}

/// Loads standard documents.
#[derive(Debug, Clone, Default)]
pub struct DocumentLoader {
    /// PDF header / footer cleaning settings.
    pub pdf_cleaning: PdfCleaning,
}

/// Python `Path.suffix`: the last extension with its dot, or empty.
pub(crate) fn suffix(path: &Path) -> String {
    path.extension().map(|ext| format!(".{}", ext.to_string_lossy())).unwrap_or_default()
}

/// Python `Path.stem`.
pub(crate) fn stem(path: &Path) -> String {
    path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

impl DocumentLoader {
    /// Creates a loader with the given PDF cleaning settings.
    #[must_use]
    pub fn new(pdf_cleaning: PdfCleaning) -> Self {
        Self { pdf_cleaning }
    }

    /// Loads one document by extension (case-insensitive):
    /// - `.txt`: the whole file as UTF-8 (a leading BOM is dropped, line endings become `\n`);
    /// - `.json`: an object with string `doc_id` and `text` (other fields are ignored apart from
    ///   an optional non-negative integer `total_pages`);
    /// - `.pdf`: page text from `pdf-extract`, cleaned with [`PdfCleaning::clean_pages`] and
    ///   [`crate::pdf::normalize_pdf_linebreaks`].
    ///
    /// # Errors
    /// [`IngestError::UnsupportedExtension`] for other extensions, and the read, decode and
    /// shape errors of each format.
    pub fn load_document(&self, path: &Path) -> Result<LoadedDocument> {
        match suffix(path).to_lowercase().as_str() {
            ".txt" => Ok(LoadedDocument {
                doc_id: stem(path),
                text: py::read_text(path)?,
                total_pages: None,
            }),
            ".json" => load_json(path),
            ".pdf" => self.load_pdf(path),
            _ => Err(IngestError::UnsupportedExtension(suffix(path))),
        }
    }

    #[cfg(feature = "pdf")]
    fn load_pdf(&self, path: &Path) -> Result<LoadedDocument> {
        let pages = crate::pdf::extract_pages(path)
            .map_err(|message| IngestError::Pdf { path: path.to_path_buf(), message })?;
        let total_pages = u32::try_from(pages.len()).ok().filter(|n| *n > 0);
        let non_empty: Vec<&String> = pages.iter().filter(|page| !page.is_empty()).collect();
        let cleaned = self.pdf_cleaning.clean_pages(&non_empty);
        Ok(LoadedDocument {
            doc_id: stem(path),
            text: crate::pdf::normalize_pdf_linebreaks(&cleaned),
            total_pages,
        })
    }

    #[cfg(not(feature = "pdf"))]
    fn load_pdf(&self, _path: &Path) -> Result<LoadedDocument> {
        Err(IngestError::Unsupported("pdf"))
    }
}

fn load_json(path: &Path) -> Result<LoadedDocument> {
    let invalid = |reason: &str| IngestError::InvalidDocument {
        path: path.to_path_buf(),
        reason: reason.to_owned(),
    };
    let text = py::read_text(path)?;
    let value: Value = serde_json::from_str(&text)
        .map_err(|source| IngestError::Json { path: path.to_path_buf(), source })?;
    let Value::Object(mut obj) = value else {
        return Err(invalid("JSON file must be an object with 'doc_id' and 'text' fields"));
    };
    if !obj.contains_key("doc_id") || !obj.contains_key("text") {
        return Err(invalid("JSON file must have 'doc_id' and 'text' fields"));
    }
    let Some(Value::String(doc_id)) = obj.remove("doc_id") else {
        return Err(invalid("'doc_id' must be a string"));
    };
    let Some(Value::String(text)) = obj.remove("text") else {
        return Err(invalid("'text' must be a string"));
    };
    let total_pages = match obj.get("total_pages") {
        None | Some(Value::Null) => None,
        Some(Value::Number(n)) => Some(
            n.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| invalid("'total_pages' must be a non-negative integer"))?,
        ),
        Some(_) => return Err(invalid("'total_pages' must be a non-negative integer")),
    };
    Ok(LoadedDocument { doc_id, text, total_pages })
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    fn write(dir: &Path, name: &str, content: &[u8]) -> std::path::PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, content).expect("write test file");
        path
    }

    fn load(path: &Path) -> Result<LoadedDocument> {
        DocumentLoader::default().load_document(path)
    }

    #[test]
    fn loads_txt_with_bom_and_crlf() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(dir.path(), "Note.TXT", b"\xEF\xBB\xBFline one\r\nline two\rthree\n");
        let doc = load(&path).expect("load txt");
        assert_eq!(doc.doc_id, "Note");
        assert_eq!(doc.text, "line one\nline two\nthree\n");
        assert_eq!(doc.total_pages, None);
    }

    #[test]
    fn rejects_invalid_utf8() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(dir.path(), "bad.txt", b"caf\xE9");
        assert!(matches!(load(&path), Err(IngestError::Encoding { .. })));
    }

    #[test]
    fn loads_json_document() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(
            dir.path(),
            "memo.json",
            br#"{"doc_id": "m-1", "text": "Body", "total_pages": 3, "author": "x"}"#,
        );
        let doc = load(&path).expect("load json");
        assert_eq!(
            doc,
            LoadedDocument { doc_id: "m-1".into(), text: "Body".into(), total_pages: Some(3) }
        );
    }

    #[test]
    fn json_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = write(dir.path(), "a.json", br#"{"doc_id": "a"}"#);
        assert!(matches!(load(&missing), Err(IngestError::InvalidDocument { .. })));
        let malformed = write(dir.path(), "b.json", b"{not json");
        assert!(matches!(load(&malformed), Err(IngestError::Json { .. })));
        let array = write(dir.path(), "c.json", b"[1, 2]");
        assert!(matches!(load(&array), Err(IngestError::InvalidDocument { .. })));
        let numeric = write(dir.path(), "d.json", br#"{"doc_id": 7, "text": "t"}"#);
        assert!(matches!(load(&numeric), Err(IngestError::InvalidDocument { .. })));
        let pages =
            write(dir.path(), "e.json", br#"{"doc_id": "e", "text": "t", "total_pages": -1}"#);
        assert!(matches!(load(&pages), Err(IngestError::InvalidDocument { .. })));
    }

    #[test]
    fn rejects_other_extensions() {
        let dir = tempfile::tempdir().expect("tempdir");
        let docx = write(dir.path(), "brief.docx", b"x");
        match load(&docx) {
            Err(IngestError::UnsupportedExtension(ext)) => assert_eq!(ext, ".docx"),
            other => panic!("unexpected {other:?}"),
        }
        let bare = write(dir.path(), "README", b"x");
        match load(&bare) {
            Err(IngestError::UnsupportedExtension(ext)) => assert_eq!(ext, ""),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[cfg(feature = "pdf")]
    #[test]
    fn reports_unreadable_pdf() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(dir.path(), "broken.pdf", b"this is not a pdf");
        assert!(matches!(load(&path), Err(IngestError::Pdf { .. })));
    }

    /// Builds a minimal PDF with one Helvetica text line per entry of each page.
    #[cfg(feature = "pdf")]
    fn minimal_pdf(pages: &[&[&str]]) -> Vec<u8> {
        let font_id = 3 + 2 * pages.len();
        let kids: Vec<String> = (0..pages.len()).map(|i| format!("{} 0 R", 3 + 2 * i)).collect();
        let mut objects = vec![
            "<< /Type /Catalog /Pages 2 0 R >>".to_owned(),
            format!("<< /Type /Pages /Kids [{}] /Count {} >>", kids.join(" "), pages.len()),
        ];
        for (i, lines) in pages.iter().enumerate() {
            objects.push(format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {} 0 R >>",
                4 + 2 * i
            ));
            let mut content = String::from("BT /F1 12 Tf 72 720 Td");
            for line in *lines {
                content.push_str(&format!(" ({line}) Tj 0 -14 Td"));
            }
            content.push_str(" ET");
            objects.push(format!("<< /Length {} >>\nstream\n{content}\nendstream", content.len()));
        }
        objects.push(
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
                .to_owned(),
        );
        let mut pdf = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for (i, object) in objects.iter().enumerate() {
            offsets.push(pdf.len());
            pdf.extend_from_slice(format!("{} 0 obj\n{object}\nendobj\n", i + 1).as_bytes());
        }
        let xref_at = pdf.len();
        let mut xref = format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1);
        for offset in offsets {
            xref.push_str(&format!("{offset:010} 00000 n \n"));
        }
        xref.push_str(&format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n",
            objects.len() + 1
        ));
        pdf.extend_from_slice(xref.as_bytes());
        pdf
    }

    #[cfg(feature = "pdf")]
    #[test]
    fn loads_pdf_pages_with_cleaning() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pdf = minimal_pdf(&[
            &["ACME CONFIDENTIAL", "Thomas", "Reynolds met the board."],
            &["ACME CONFIDENTIAL", "Second page."],
        ]);
        let path = write(dir.path(), "Filing.PDF", &pdf);
        let doc = load(&path).expect("load pdf");
        assert_eq!(doc.doc_id, "Filing");
        assert_eq!(doc.total_pages, Some(2));
        assert_eq!(doc.text, "Thomas Reynolds met the board.\n\nSecond page.");
    }

    #[cfg(not(feature = "pdf"))]
    #[test]
    fn pdf_needs_the_feature() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(dir.path(), "doc.pdf", b"%PDF-1.4");
        assert!(matches!(load(&path), Err(IngestError::Unsupported("pdf"))));
    }
}
