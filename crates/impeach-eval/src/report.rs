//! The evaluation report: records in the Python JSON shape, the JSON writer and the console
//! summary.

use std::fmt::Write as _;
use std::io;

use impeach_core::pyjson::{float_repr, PythonFormatter};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::ser::{CharEscape, Formatter, PrettyFormatter};

use crate::error::{EvalError, Result};
use crate::pipeline_output::FactOrigin;

/// Recall / precision / F1 with raw counts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Prf {
    /// Matched key rows.
    pub true_positives: usize,
    /// Key rows.
    pub ground_truth: usize,
    /// Detected pairs.
    pub detected: usize,
    /// Unmatched key rows.
    pub false_negatives: usize,
    /// Unmatched detected pairs.
    pub false_positives: usize,
    /// `tp / ground_truth` (0.0 without key rows).
    pub recall: f64,
    /// `tp / detected` (0.0 without detections).
    pub precision: f64,
    /// Harmonic mean (0.0 when both are 0).
    pub f1: f64,
}

impl Prf {
    /// Metrics for `tp` matches out of `gt_total` key rows and `detected_total` detections.
    #[must_use]
    pub fn new(tp: usize, gt_total: usize, detected_total: usize) -> Self {
        let recall = if gt_total > 0 { tp as f64 / gt_total as f64 } else { 0.0 };
        let precision = if detected_total > 0 { tp as f64 / detected_total as f64 } else { 0.0 };
        let f1 = if precision + recall != 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        Self {
            true_positives: tp,
            ground_truth: gt_total,
            detected: detected_total,
            false_negatives: gt_total - tp,
            false_positives: detected_total - tp,
            recall,
            precision,
            f1,
        }
    }
}

/// Key-fact coverage summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactCoverageSummary {
    /// Distinct key-fact refs covered by some pipeline fact.
    pub covered: usize,
    /// Key facts (two per key row).
    pub total: usize,
    /// `covered / total` (0.0 without key facts).
    pub ratio: f64,
    /// Pipeline facts compared against.
    pub pipeline_facts_considered: usize,
    /// `all_extracted_facts` or `clean_facts+detected_pair_facts`.
    pub fact_source: String,
}

/// Every metric of the report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    /// Both buckets together.
    pub overall: Prf,
    /// `C-*` rows against cross-document detections.
    pub inter_doc: Prf,
    /// `D*` rows against intra-document detections.
    pub intra_doc: Prf,
    /// Key-fact coverage.
    pub fact_coverage: FactCoverageSummary,
}

/// One key fact as written in the report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyFactRecord {
    /// Cleaned fact text.
    pub text: String,
    /// `Doc 1`, `Doc 2`, ...
    pub source: String,
}

/// The key-row part of a matched or missed entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroundTruthRecord {
    /// Key row id.
    pub ground_truth_id: String,
    /// Key row type column.
    pub ground_truth_type: String,
    /// First fact.
    pub fact_a: KeyFactRecord,
    /// Second fact.
    pub fact_b: KeyFactRecord,
}

/// A detected pair as written in the report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectedPairRecord {
    /// Text of the fact being checked.
    pub new_fact_text: String,
    /// Text of the fact it contradicts.
    pub existing_fact_text: String,
    /// Id of the fact being checked.
    pub new_fact_id: String,
    /// Id of the fact it contradicts.
    pub existing_fact_id: String,
    /// NLI / judge label.
    pub nli_label: String,
    /// NLI / judge confidence.
    pub nli_confidence: f64,
    /// `intra_doc` or `cross_doc`.
    pub source_type: String,
    /// `doc_id` of the document whose check produced the pair.
    pub doc_id: Option<String>,
}

/// Similarities of a matched pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchScores {
    /// `min(fact_a, fact_b)`.
    pub combined: f64,
    /// Similarity of key fact A to its aligned detected fact.
    pub fact_a: f64,
    /// Similarity of key fact B to its aligned detected fact.
    pub fact_b: f64,
    /// `a=new,b=existing` or `a=existing,b=new`.
    pub orientation: String,
}

/// A key row matched to a detected pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchedContradiction {
    /// The key row.
    #[serde(flatten)]
    pub ground_truth: GroundTruthRecord,
    /// The detected pair it was assigned.
    pub detected_pair: DetectedPairRecord,
    /// The match similarities.
    pub scores: MatchScores,
}

/// The best-scoring detected pair of a missed key row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClosestDetectedPair {
    /// The detected pair.
    #[serde(flatten)]
    pub pair: DetectedPairRecord,
    /// Its best-orientation score ignoring the value guard.
    pub combined: f64,
    /// Whether the value guard rejected the preferred orientation.
    pub value_guard_rejected: bool,
    /// Id of the key row the pair was assigned to instead, if any.
    pub taken_by: Option<String>,
}

/// Coverage of one key fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactCoverage {
    /// `similarity >= threshold`.
    pub covered: bool,
    /// Best similarity over value-compatible pipeline facts (0.0 when none is positive).
    pub similarity: f64,
    /// Text of that pipeline fact.
    pub best_fact: Option<String>,
    /// Origin of that pipeline fact.
    pub best_fact_origin: Option<FactOrigin>,
}

/// A key row without an assigned detection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MissedContradiction {
    /// The key row.
    #[serde(flatten)]
    pub ground_truth: GroundTruthRecord,
    /// The detection scoring highest against it, if the bucket has detections.
    pub closest_detected_pair: Option<ClosestDetectedPair>,
    /// Coverage of key fact A.
    pub fact_a_extracted: FactCoverage,
    /// Coverage of key fact B.
    pub fact_b_extracted: FactCoverage,
}

/// A detection without an assigned key row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FalsePositive {
    /// The detected pair.
    pub detected_pair: DetectedPairRecord,
    /// Id of the key row scoring highest against it, if the bucket has key rows.
    pub closest_ground_truth: Option<String>,
    /// That row's best-orientation score ignoring the value guard.
    pub closest_score: f64,
}

/// A covered key fact whose best pipeline fact is in no detected pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnpairedFact {
    /// `<key id>.fact_a` or `<key id>.fact_b`.
    pub key_fact: String,
    /// The covering pipeline fact.
    pub fact_text: Option<String>,
    /// Its similarity.
    pub similarity: f64,
}

/// The full evaluation report, in the key order of the Python JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    /// `eval_<YYYYmmdd_HHMMSS>`.
    pub evaluation_id: String,
    /// Local ISO timestamp (Python `datetime.now().isoformat()`).
    pub timestamp: String,
    /// Pipeline output path as given.
    pub input_file: String,
    /// Key path as given.
    pub ground_truth_file: String,
    /// Similarity threshold for a match.
    pub similarity_threshold: f64,
    /// Embedding model name.
    pub embedding_model: String,
    /// Whether the value guard was on.
    pub value_guard: bool,
    /// Assignment method description.
    pub assignment: String,
    /// Metrics.
    pub metrics: Metrics,
    /// Matched key rows, sorted by id.
    pub matched_contradictions: Vec<MatchedContradiction>,
    /// Missed key rows, sorted by id.
    pub missed_contradictions: Vec<MissedContradiction>,
    /// Unmatched detections, highest closest score first.
    pub false_positives: Vec<FalsePositive>,
    /// Covered key facts whose covering fact is in no detected pair.
    pub unpaired_correct_facts: Vec<UnpairedFact>,
    /// Coverage per key-fact ref.
    pub fact_coverage_detail: IndexMap<String, FactCoverage>,
}

/// `json.dumps(indent=2)` layout with Python float repr and `ensure_ascii` escaping.
struct IndentedPythonFormatter {
    layout: PrettyFormatter<'static>,
    text: PythonFormatter,
}

impl Formatter for IndentedPythonFormatter {
    fn begin_array<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        self.layout.begin_array(w)
    }
    fn end_array<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        self.layout.end_array(w)
    }
    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.layout.begin_array_value(w, first)
    }
    fn end_array_value<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        self.layout.end_array_value(w)
    }
    fn begin_object<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        self.layout.begin_object(w)
    }
    fn end_object<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        self.layout.end_object(w)
    }
    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.layout.begin_object_key(w, first)
    }
    fn begin_object_value<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        self.layout.begin_object_value(w)
    }
    fn end_object_value<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        self.layout.end_object_value(w)
    }
    fn write_f64<W: ?Sized + io::Write>(&mut self, w: &mut W, value: f64) -> io::Result<()> {
        self.text.write_f64(w, value)
    }
    fn write_f32<W: ?Sized + io::Write>(&mut self, w: &mut W, value: f32) -> io::Result<()> {
        self.text.write_f32(w, value)
    }
    fn write_string_fragment<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        fragment: &str,
    ) -> io::Result<()> {
        self.text.write_string_fragment(w, fragment)
    }
    fn write_char_escape<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        char_escape: CharEscape,
    ) -> io::Result<()> {
        self.text.write_char_escape(w, char_escape)
    }
}

/// Python `json.dumps(value, indent=2)` (ASCII-escaped, `,` + newline item separators).
///
/// # Errors
/// The serialisation error of `value`.
pub fn to_python_json_indent2<T: Serialize + ?Sized>(value: &T) -> serde_json::Result<String> {
    let formatter = IndentedPythonFormatter {
        layout: PrettyFormatter::with_indent(b"  "),
        text: PythonFormatter { ensure_ascii: true },
    };
    let mut out = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut out, formatter);
    value.serialize(&mut serializer)?;
    String::from_utf8(out).map_err(serde::ser::Error::custom)
}

/// Python `f"{x:.{digits}%}"`.
fn percent(value: f64, digits: usize) -> String {
    format!("{:.digits$}%", value * 100.0)
}

fn prf_line(m: &Prf) -> String {
    format!(
        "recall {}/{} = {} | precision {}/{} = {} | F1 {:.3}",
        m.true_positives,
        m.ground_truth,
        percent(m.recall, 1),
        m.true_positives,
        m.detected,
        percent(m.precision, 1),
        m.f1
    )
}

/// The first `n` characters (Python `text[:n]`).
fn head(text: &str, n: usize) -> &str {
    text.char_indices().nth(n).map_or(text, |(index, _)| &text[..index])
}

impl EvalReport {
    /// The report as Python writes it (`json.dumps(report, indent=2)`).
    ///
    /// # Errors
    /// [`EvalError::Serialize`] (not expected for these records).
    pub fn to_json(&self) -> Result<String> {
        to_python_json_indent2(self).map_err(EvalError::Serialize)
    }

    /// The console summary the Python evaluator prints, one `print` per line; `report_path`
    /// adds the closing "Report written to" line.
    #[must_use]
    pub fn summary(&self, report_path: Option<&str>) -> String {
        let m = &self.metrics;
        let cov = &m.fact_coverage;
        let mut out = String::new();
        // Writing to a String cannot fail.
        let _ = writeln!(out, "\n=== Contradiction evaluation ===");
        let _ = writeln!(out, "input:     {}", self.input_file);
        let _ = writeln!(out, "key:       {}", self.ground_truth_file);
        let _ = writeln!(
            out,
            "threshold: {}  model: {}  value guard: {}  assignment: {}",
            float_repr(self.similarity_threshold),
            self.embedding_model,
            if self.value_guard { "on" } else { "off" },
            self.assignment
        );
        let _ = writeln!(out, "overall    {}", prf_line(&m.overall));
        let _ = writeln!(out, "inter-doc  {}", prf_line(&m.inter_doc));
        let _ = writeln!(out, "intra-doc  {}", prf_line(&m.intra_doc));
        let _ = writeln!(
            out,
            "fact coverage {}/{} = {} ({})",
            cov.covered,
            cov.total,
            percent(cov.ratio, 1),
            cov.fact_source
        );

        if !self.matched_contradictions.is_empty() {
            let _ = writeln!(out, "\nMatched pairs (audit):");
            for item in &self.matched_contradictions {
                let s = &item.scores;
                let d = &item.detected_pair;
                let g = &item.ground_truth;
                let _ = writeln!(
                    out,
                    "  {}  combined={:.3} (a={:.3}, b={:.3})",
                    g.ground_truth_id, s.combined, s.fact_a, s.fact_b
                );
                let _ = writeln!(out, "     key A: {}", head(&g.fact_a.text, 110));
                let _ = writeln!(out, "     key B: {}", head(&g.fact_b.text, 110));
                let _ = writeln!(out, "     det  : {}", head(&d.new_fact_text, 110));
                let _ = writeln!(
                    out,
                    "     vs   : {}  [{} {:.2}]",
                    head(&d.existing_fact_text, 110),
                    d.nli_label,
                    d.nli_confidence
                );
            }
        }
        if !self.missed_contradictions.is_empty() {
            let ids: Vec<&str> = self
                .missed_contradictions
                .iter()
                .map(|m| m.ground_truth.ground_truth_id.as_str())
                .collect();
            let _ = writeln!(out, "\nMissed: {}", ids.join(", "));
        }
        if let Some(path) = report_path {
            let _ = writeln!(out, "\nReport written to {path}");
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indent2_layout_matches_python() {
        let value =
            serde_json::json!({"a": [], "b": {}, "c": [1, 2.0], "d": {"e": "é\n", "f": null}});
        assert_eq!(
            to_python_json_indent2(&value).unwrap(),
            "{\n  \"a\": [],\n  \"b\": {},\n  \"c\": [\n    1,\n    2.0\n  ],\n  \"d\": {\n    \"e\": \"\\u00e9\\n\",\n    \"f\": null\n  }\n}"
        );
    }

    #[test]
    fn prf_edge_cases() {
        let zero = Prf::new(0, 0, 0);
        assert_eq!((zero.recall, zero.precision, zero.f1), (0.0, 0.0, 0.0));
        let m = Prf::new(1, 3, 3);
        assert!((m.recall - 1.0 / 3.0).abs() < 1e-12);
        assert!((m.f1 - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!((m.false_negatives, m.false_positives), (2, 2));
    }

    #[test]
    fn formatting_helpers() {
        assert_eq!(percent(1.0 / 3.0, 1), "33.3%");
        assert_eq!(percent(0.0, 1), "0.0%");
        assert_eq!(
            prf_line(&Prf::new(1, 3, 2)),
            "recall 1/3 = 33.3% | precision 1/2 = 50.0% | F1 0.400"
        );
        assert_eq!(head("¶abc", 2), "¶a");
        assert_eq!(head("ab", 5), "ab");
    }
}
