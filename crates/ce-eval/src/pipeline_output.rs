//! Facts and detected pairs taken from an e2e output.

use std::collections::HashSet;
use std::path::Path;

use ce_core::e2e::{E2eOutput, PairRecord};
use ce_text::normalize_whitespace_lower;
use serde::{Deserialize, Serialize};

use crate::error::{EvalError, Result};

/// Where a pipeline fact used for coverage came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactOrigin {
    /// `stage_5_fact_extraction.all_facts`.
    Extracted,
    /// `stage_6_contradiction_check.all_clean_facts`.
    Clean,
    /// A fact of a detected pair (older outputs without `all_facts`).
    DetectedPair,
}

impl FactOrigin {
    /// The Python origin string.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Extracted => "extracted",
            Self::Clean => "clean",
            Self::DetectedPair => "detected_pair",
        }
    }
}

/// Which bucket a detected pair belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    /// `intra_doc_contradictions`.
    IntraDoc,
    /// `cross_doc_contradictions`.
    CrossDoc,
}

impl SourceType {
    /// The Python source-type string.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::IntraDoc => "intra_doc",
            Self::CrossDoc => "cross_doc",
        }
    }
}

/// One detected contradiction pair.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectedContradiction {
    /// Text of the fact being checked.
    pub new_fact_text: String,
    /// Id of the fact being checked.
    pub new_fact_id: String,
    /// Text of the fact it contradicts.
    pub existing_fact_text: String,
    /// Id of the fact it contradicts.
    pub existing_fact_id: String,
    /// NLI / judge label.
    pub nli_label: String,
    /// NLI / judge confidence.
    pub nli_confidence: f64,
    /// Retrieval paths that surfaced the existing fact.
    pub retrieval_sources: Vec<String>,
    /// Intra- or cross-document.
    pub source_type: SourceType,
    /// `doc_id` of the document whose check produced the pair.
    pub doc_id: Option<String>,
}

impl DetectedContradiction {
    fn from_record(raw: &PairRecord, source_type: SourceType, doc_id: Option<&String>) -> Self {
        Self {
            new_fact_text: raw.new_fact_text.clone(),
            new_fact_id: raw.new_fact_id.clone(),
            existing_fact_text: raw.existing_fact_text.clone(),
            existing_fact_id: raw.existing_fact_id.clone(),
            nli_label: raw.nli_label.clone(),
            nli_confidence: raw.nli_confidence,
            retrieval_sources: raw.retrieval_sources.clone(),
            source_type,
            doc_id: doc_id.cloned(),
        }
    }
}

/// One pipeline fact considered for key-fact coverage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineFact {
    /// Fact text.
    pub text: String,
    /// Fact id.
    pub fact_id: String,
    /// Where the fact came from.
    pub origin: FactOrigin,
    /// `doc_id` of its document, when known.
    pub doc_id: Option<String>,
}

/// What the evaluator uses from an e2e output.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineOutput {
    /// Coverage facts, de-duplicated by normalised text (first occurrence kept).
    pub facts: Vec<PipelineFact>,
    /// Number of clean facts over all documents.
    pub clean_fact_count: usize,
    /// Intra-document pairs in document order.
    pub intra_doc: Vec<DetectedContradiction>,
    /// Cross-document pairs in document order.
    pub cross_doc: Vec<DetectedContradiction>,
    /// True when any document carried `stage_5_fact_extraction.all_facts`.
    pub has_all_facts: bool,
}

/// Lowercase, whitespace-collapsed text used to de-duplicate facts.
#[must_use]
pub fn norm_text(text: &str) -> String {
    normalize_whitespace_lower(text)
}

impl PipelineOutput {
    /// Collects facts and pairs from `output`.
    ///
    /// Coverage facts come from `stage_5_fact_extraction.all_facts` when present. A document
    /// without it contributes its clean facts, and when no document has it the facts of every
    /// detected pair are added too (the existing side without a `doc_id`), so older outputs
    /// are not under-counted.
    #[must_use]
    pub fn from_e2e(output: &E2eOutput) -> Self {
        let mut facts = Vec::new();
        let mut intra = Vec::new();
        let mut cross = Vec::new();
        let mut clean_count = 0;
        let mut has_all_facts = false;

        for doc in &output.documents {
            let doc_id = doc.doc_id.as_ref();
            let all_facts =
                doc.stage_5_fact_extraction.as_ref().and_then(|s5| s5.all_facts.as_ref());
            if let Some(all_facts) = all_facts {
                has_all_facts = true;
                facts.extend(all_facts.iter().map(|f| PipelineFact {
                    text: f.text.clone(),
                    fact_id: f.fact_id.clone(),
                    origin: FactOrigin::Extracted,
                    doc_id: doc_id.cloned(),
                }));
            }
            let Some(s6) = doc.stage_6_contradiction_check.as_ref() else {
                continue;
            };
            clean_count += s6.all_clean_facts.len();
            if all_facts.is_none() {
                facts.extend(s6.all_clean_facts.iter().map(|f| PipelineFact {
                    text: f.text.clone(),
                    fact_id: f.fact_id.clone(),
                    origin: FactOrigin::Clean,
                    doc_id: doc_id.cloned(),
                }));
            }
            intra.extend(
                s6.intra_doc_contradictions
                    .iter()
                    .map(|c| DetectedContradiction::from_record(c, SourceType::IntraDoc, doc_id)),
            );
            cross.extend(
                s6.cross_doc_contradictions
                    .iter()
                    .map(|c| DetectedContradiction::from_record(c, SourceType::CrossDoc, doc_id)),
            );
        }

        if !has_all_facts {
            for d in intra.iter().chain(&cross) {
                facts.push(PipelineFact {
                    text: d.new_fact_text.clone(),
                    fact_id: d.new_fact_id.clone(),
                    origin: FactOrigin::DetectedPair,
                    doc_id: d.doc_id.clone(),
                });
                facts.push(PipelineFact {
                    text: d.existing_fact_text.clone(),
                    fact_id: d.existing_fact_id.clone(),
                    origin: FactOrigin::DetectedPair,
                    doc_id: None,
                });
            }
        }

        let mut seen = HashSet::new();
        facts.retain(|f| {
            let key = norm_text(&f.text);
            !key.is_empty() && seen.insert(key)
        });

        Self {
            facts,
            clean_fact_count: clean_count,
            intra_doc: intra,
            cross_doc: cross,
            has_all_facts,
        }
    }
}

/// Reads an e2e output JSON file.
///
/// # Errors
/// [`EvalError::Io`] or [`EvalError::Json`].
pub fn load_e2e_output(path: &Path) -> Result<E2eOutput> {
    let text = std::fs::read_to_string(path)
        .map_err(|source| EvalError::Io { path: path.to_path_buf(), source })?;
    serde_json::from_str(&text)
        .map_err(|source| EvalError::Json { path: path.to_path_buf(), source })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(json: &str) -> PipelineOutput {
        PipelineOutput::from_e2e(&serde_json::from_str(json).unwrap())
    }

    #[test]
    fn clean_and_pair_facts_without_all_facts() {
        let out = load(
            r#"{"documents": [{"doc_id": "002", "stage_6_contradiction_check": {
                "all_clean_facts": [{"fact_id": "F1", "text": "Lopez  was alone"}, {"text": ""}],
                "cross_doc_contradictions": [
                    {"new_fact_text": "A b", "existing_fact_text": "lopez was ALONE"},
                    {"new_fact_text": "c", "existing_fact_text": "d", "nli_confidence": 0.5}],
                "intra_doc_contradictions": [{"new_fact_text": "e", "existing_fact_text": "a  B"}]}}]}"#,
        );
        assert!(!out.has_all_facts);
        assert_eq!(out.clean_fact_count, 2);
        assert_eq!((out.intra_doc.len(), out.cross_doc.len()), (1, 2));
        let texts: Vec<(&str, FactOrigin)> =
            out.facts.iter().map(|f| (f.text.as_str(), f.origin)).collect();
        // Intra pairs are added before cross pairs, as in Python `intra + cross`.
        assert_eq!(
            texts,
            [
                ("Lopez  was alone", FactOrigin::Clean),
                ("e", FactOrigin::DetectedPair),
                ("a  B", FactOrigin::DetectedPair),
                ("c", FactOrigin::DetectedPair),
                ("d", FactOrigin::DetectedPair),
            ]
        );
        assert_eq!(out.facts[1].doc_id.as_deref(), Some("002"));
        assert_eq!(out.facts[2].doc_id, None);
        assert_eq!(out.cross_doc[1].source_type, SourceType::CrossDoc);
    }

    #[test]
    fn all_facts_replace_clean_and_pair_facts() {
        let out = load(
            r#"{"documents": [
                {"doc_id": "001", "stage_5_fact_extraction": {"all_facts": [{"fact_id": "F0", "text": "x"}]},
                 "stage_6_contradiction_check": {"all_clean_facts": [{"text": "clean one"}],
                    "cross_doc_contradictions": [{"new_fact_text": "p", "existing_fact_text": "q"}]}},
                {"doc_id": "002", "stage_6_contradiction_check": {"all_clean_facts": [{"text": "clean two"}]}}
            ]}"#,
        );
        assert!(out.has_all_facts);
        let texts: Vec<&str> = out.facts.iter().map(|f| f.text.as_str()).collect();
        assert_eq!(texts, ["x", "clean two"]);
    }
}
