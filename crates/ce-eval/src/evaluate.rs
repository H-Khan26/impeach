//! Pair scoring and the evaluation itself.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

use ce_core::e2e::E2eOutput;
use ce_models::Embedder;
use ce_text::values::{extract_values, values_compatible, Value};
use chrono::{Local, NaiveDateTime, Timelike};
use indexmap::IndexMap;

use crate::assignment::{assign_one_to_one, AssignmentMethod};
use crate::error::{EvalError, Result};
use crate::key::{parse_key_file, GroundTruthContradiction, Key};
use crate::pipeline_output::{load_e2e_output, norm_text, DetectedContradiction, PipelineOutput};
use crate::report::{
    ClosestDetectedPair, DetectedPairRecord, EvalReport, FactCoverage, FactCoverageSummary,
    FalsePositive, GroundTruthRecord, KeyFactRecord, MatchScores, MatchedContradiction, Metrics,
    MissedContradiction, Prf, UnpairedFact,
};
use crate::similarity::SimilarityCache;

/// Default embedding model of the Python evaluator.
pub const DEFAULT_MODEL: &str = "all-MiniLM-L6-v2";
/// Default similarity threshold.
pub const DEFAULT_THRESHOLD: f64 = 0.75;

/// Orientation label: key fact A aligned with the new fact, B with the existing one.
pub const ORIENTATION_A_NEW: &str = "a=new,b=existing";
/// Orientation label: key fact A aligned with the existing fact, B with the new one.
pub const ORIENTATION_A_EXISTING: &str = "a=existing,b=new";

/// Evaluation settings and the report header values.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalOptions {
    /// Similarity a match (and a covered key fact) needs.
    pub threshold: f64,
    /// Reject alignments whose numbers / dates / amounts disagree.
    pub use_value_guard: bool,
    /// Assignment method.
    pub assignment: AssignmentMethod,
    /// `embedding_model` written to the report; `None` uses [`Embedder::model_name`].
    pub embedding_model: Option<String>,
    /// `input_file` written to the report.
    pub input_file: String,
    /// `ground_truth_file` written to the report.
    pub ground_truth_file: String,
    /// Report time; `None` uses the current local time.
    pub timestamp: Option<NaiveDateTime>,
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_THRESHOLD,
            use_value_guard: true,
            assignment: AssignmentMethod::Optimal,
            embedding_model: None,
            input_file: String::new(),
            ground_truth_file: String::new(),
            timestamp: None,
        }
    }
}

/// Score of one key row against one detected pair.
#[derive(Debug, Clone, PartialEq)]
pub struct PairScore {
    /// `min(sim_a, sim_b)` of the chosen orientation; 0 if the value guard rejected it.
    pub score: f64,
    /// Similarity of key fact A to its aligned detected fact.
    pub sim_a: f64,
    /// Similarity of key fact B to its aligned detected fact.
    pub sim_b: f64,
    /// [`ORIENTATION_A_NEW`] or [`ORIENTATION_A_EXISTING`].
    pub orientation: &'static str,
    /// Best orientation score ignoring the value guard.
    pub raw_score: f64,
    /// Whether the value guard rejected the chosen orientation.
    pub guard_rejected: bool,
}

fn values_of(texts: &[&str]) -> Vec<BTreeSet<Value>> {
    texts.iter().map(|t| extract_values(t)).collect()
}

/// Scores every key row against every detected pair in both orientations and keeps the
/// better one: higher (guarded) score first, then higher raw score, then the
/// [`ORIENTATION_A_NEW`] orientation.
///
/// # Errors
/// The embedder's error.
pub fn score_pairs(
    gts: &[&GroundTruthContradiction],
    detected: &[&DetectedContradiction],
    sims: &mut SimilarityCache<'_>,
    use_value_guard: bool,
) -> Result<Vec<Vec<PairScore>>> {
    let a_texts: Vec<&str> = gts.iter().map(|g| g.fact_a.text.as_str()).collect();
    let b_texts: Vec<&str> = gts.iter().map(|g| g.fact_b.text.as_str()).collect();
    let new_texts: Vec<&str> = detected.iter().map(|d| d.new_fact_text.as_str()).collect();
    let old_texts: Vec<&str> = detected.iter().map(|d| d.existing_fact_text.as_str()).collect();

    let sim_a_new = sims.matrix(&a_texts, &new_texts)?;
    let sim_b_old = sims.matrix(&b_texts, &old_texts)?;
    let sim_a_old = sims.matrix(&a_texts, &old_texts)?;
    let sim_b_new = sims.matrix(&b_texts, &new_texts)?;

    let (va, vb) = (values_of(&a_texts), values_of(&b_texts));
    let (vnew, vold) = (values_of(&new_texts), values_of(&old_texts));

    let mut table = Vec::with_capacity(gts.len());
    for i in 0..gts.len() {
        let mut row = Vec::with_capacity(detected.len());
        for j in 0..detected.len() {
            let options = [
                (
                    ORIENTATION_A_NEW,
                    sim_a_new[i][j],
                    sim_b_old[i][j],
                    values_compatible(&va[i], &vnew[j]) && values_compatible(&vb[i], &vold[j]),
                ),
                (
                    ORIENTATION_A_EXISTING,
                    sim_a_old[i][j],
                    sim_b_new[i][j],
                    values_compatible(&va[i], &vold[j]) && values_compatible(&vb[i], &vnew[j]),
                ),
            ];
            let mut best: Option<PairScore> = None;
            let mut raw_max = f64::NEG_INFINITY;
            for (orientation, sa, sb, ok) in options {
                let raw = f64::from(sa.min(sb));
                let guarded = use_value_guard && !ok;
                let score = if guarded { 0.0 } else { raw };
                raw_max = if raw > raw_max { raw } else { raw_max };
                let better = best
                    .as_ref()
                    .is_none_or(|b| score > b.score || (score == b.score && raw > b.raw_score));
                if better {
                    best = Some(PairScore {
                        score,
                        sim_a: f64::from(sa),
                        sim_b: f64::from(sb),
                        orientation,
                        raw_score: raw,
                        guard_rejected: guarded,
                    });
                }
            }
            let mut best = best.expect("two orientations are always scored");
            best.raw_score = raw_max;
            row.push(best);
        }
        table.push(row);
    }
    Ok(table)
}

/// Index of the first maximum (NumPy `argmax`).
fn argmax(values: impl Iterator<Item = f64>) -> usize {
    let mut best: Option<(usize, f64)> = None;
    for (index, value) in values.enumerate() {
        if best.is_none_or(|(_, b)| value > b) {
            best = Some((index, value));
        }
    }
    best.map_or(0, |(index, _)| index)
}

fn detected_record(d: &DetectedContradiction) -> DetectedPairRecord {
    DetectedPairRecord {
        new_fact_text: d.new_fact_text.clone(),
        existing_fact_text: d.existing_fact_text.clone(),
        new_fact_id: d.new_fact_id.clone(),
        existing_fact_id: d.existing_fact_id.clone(),
        nli_label: d.nli_label.clone(),
        nli_confidence: d.nli_confidence,
        source_type: d.source_type.as_str().to_owned(),
        doc_id: d.doc_id.clone(),
    }
}

fn gt_record(g: &GroundTruthContradiction) -> GroundTruthRecord {
    GroundTruthRecord {
        ground_truth_id: g.id.clone(),
        ground_truth_type: g.contradiction_type.clone(),
        fact_a: KeyFactRecord { text: g.fact_a.text.clone(), source: g.fact_a.source.clone() },
        fact_b: KeyFactRecord { text: g.fact_b.text.clone(), source: g.fact_b.source.clone() },
    }
}

fn coverage_of(coverage: &IndexMap<String, FactCoverage>, reference: &str) -> FactCoverage {
    coverage.get(reference).cloned().expect("every key fact has a coverage entry")
}

/// The report sections computed from the inputs (everything but the header).
#[derive(Debug, Clone, PartialEq)]
pub struct EvalBody {
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

/// Matches key rows to detected pairs and computes the report sections.
///
/// # Errors
/// The embedder's error, or [`EvalError::InvalidScores`].
pub fn evaluate_body(
    key: &Key,
    output: &PipelineOutput,
    embedder: &dyn Embedder,
    threshold: f64,
    use_value_guard: bool,
    method: AssignmentMethod,
) -> Result<EvalBody> {
    let ground_truth = &key.contradictions;
    let mut sims = SimilarityCache::new(embedder);
    let mut matched = Vec::new();
    let mut missed = Vec::new();
    let mut false_positives = Vec::new();

    // Fact coverage first so missed contradictions can say whether the facts were extracted.
    let fact_texts: Vec<&str> = output.facts.iter().map(|f| f.text.as_str()).collect();
    let fact_values = values_of(&fact_texts);
    let mut key_facts: Vec<(String, &str)> = Vec::with_capacity(ground_truth.len() * 2);
    for g in ground_truth {
        key_facts.push((format!("{}.fact_a", g.id), g.fact_a.text.as_str()));
        key_facts.push((format!("{}.fact_b", g.id), g.fact_b.text.as_str()));
    }
    let key_texts: Vec<&str> = key_facts.iter().map(|(_, text)| *text).collect();
    let cov_sim = sims.matrix(&key_texts, &fact_texts)?;
    let mut coverage: IndexMap<String, FactCoverage> = IndexMap::new();
    for (i, (reference, text)) in key_facts.iter().enumerate() {
        let kv = extract_values(text);
        let mut best_j = None;
        let mut best = 0.0;
        for (j, values) in fact_values.iter().enumerate() {
            let s = f64::from(cov_sim[i][j]);
            if use_value_guard && !values_compatible(&kv, values) {
                continue;
            }
            if s > best {
                best_j = Some(j);
                best = s;
            }
        }
        coverage.insert(
            reference.clone(),
            FactCoverage {
                covered: best >= threshold,
                similarity: best,
                best_fact: best_j.map(|j| output.facts[j].text.clone()),
                best_fact_origin: best_j.map(|j| output.facts[j].origin),
            },
        );
    }
    let covered_refs = coverage.values().filter(|c| c.covered).count();

    let detected_texts: HashSet<String> = output
        .intra_doc
        .iter()
        .chain(&output.cross_doc)
        .flat_map(|d| [norm_text(&d.new_fact_text), norm_text(&d.existing_fact_text)])
        .collect();
    let unpaired: Vec<UnpairedFact> = coverage
        .iter()
        .filter(|(_, c)| {
            c.covered && !detected_texts.contains(&norm_text(c.best_fact.as_deref().unwrap_or("")))
        })
        .map(|(reference, c)| UnpairedFact {
            key_fact: reference.clone(),
            fact_text: c.best_fact.clone(),
            similarity: c.similarity,
        })
        .collect();

    let inter_gts: Vec<&GroundTruthContradiction> = key.inter_doc().collect();
    let intra_gts: Vec<&GroundTruthContradiction> = key.intra_doc().collect();
    let mut split = Vec::with_capacity(2);
    for (gts, dets) in [(&inter_gts, &output.cross_doc), (&intra_gts, &output.intra_doc)] {
        let dets: Vec<&DetectedContradiction> = dets.iter().collect();
        let table = if gts.is_empty() || dets.is_empty() {
            Vec::new()
        } else {
            score_pairs(gts, &dets, &mut sims, use_value_guard)?
        };
        let scores: Vec<Vec<f64>> =
            table.iter().map(|row| row.iter().map(|c| c.score).collect()).collect();
        let pairs = assign_one_to_one(&scores, threshold, method)?;
        let gt_hit: HashMap<usize, usize> = pairs.iter().copied().collect();
        let det_hit: HashMap<usize, usize> = pairs.iter().map(|&(r, c)| (c, r)).collect();

        for (i, g) in gts.iter().enumerate() {
            if let Some(&j) = gt_hit.get(&i) {
                let ps = &table[i][j];
                matched.push(MatchedContradiction {
                    ground_truth: gt_record(g),
                    detected_pair: detected_record(dets[j]),
                    scores: MatchScores {
                        combined: ps.score,
                        fact_a: ps.sim_a,
                        fact_b: ps.sim_b,
                        orientation: ps.orientation.to_owned(),
                    },
                });
            } else {
                let closest = (!table.is_empty()).then(|| {
                    let j = argmax(table[i].iter().map(|c| c.raw_score));
                    let ps = &table[i][j];
                    ClosestDetectedPair {
                        pair: detected_record(dets[j]),
                        combined: ps.raw_score,
                        value_guard_rejected: ps.guard_rejected,
                        taken_by: det_hit.get(&j).map(|&r| gts[r].id.clone()),
                    }
                });
                missed.push(MissedContradiction {
                    ground_truth: gt_record(g),
                    closest_detected_pair: closest,
                    fact_a_extracted: coverage_of(&coverage, &format!("{}.fact_a", g.id)),
                    fact_b_extracted: coverage_of(&coverage, &format!("{}.fact_b", g.id)),
                });
            }
        }

        for (j, d) in dets.iter().enumerate() {
            if det_hit.contains_key(&j) {
                continue;
            }
            let (closest_gt, closest_score) = if table.is_empty() {
                (None, 0.0)
            } else {
                let i = argmax(table.iter().map(|row| row[j].raw_score));
                (Some(gts[i].id.clone()), table[i][j].raw_score)
            };
            false_positives.push(FalsePositive {
                detected_pair: detected_record(d),
                closest_ground_truth: closest_gt,
                closest_score,
            });
        }

        split.push(Prf::new(pairs.len(), gts.len(), dets.len()));
    }
    let intra_doc = split.pop().expect("two buckets");
    let inter_doc = split.pop().expect("two buckets");

    let total_tp = inter_doc.true_positives + intra_doc.true_positives;
    let metrics = Metrics {
        overall: Prf::new(
            total_tp,
            ground_truth.len(),
            output.intra_doc.len() + output.cross_doc.len(),
        ),
        inter_doc,
        intra_doc,
        fact_coverage: FactCoverageSummary {
            covered: covered_refs,
            total: key_facts.len(),
            ratio: if key_facts.is_empty() {
                0.0
            } else {
                covered_refs as f64 / key_facts.len() as f64
            },
            pipeline_facts_considered: output.facts.len(),
            fact_source: if output.has_all_facts {
                "all_extracted_facts"
            } else {
                "clean_facts+detected_pair_facts"
            }
            .to_owned(),
        },
    };
    matched.sort_by(|a, b| a.ground_truth.ground_truth_id.cmp(&b.ground_truth.ground_truth_id));
    missed.sort_by(|a, b| a.ground_truth.ground_truth_id.cmp(&b.ground_truth.ground_truth_id));
    false_positives.sort_by(|a, b| {
        b.closest_score.partial_cmp(&a.closest_score).unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(EvalBody {
        metrics,
        matched_contradictions: matched,
        missed_contradictions: missed,
        false_positives,
        unpaired_correct_facts: unpaired,
        fact_coverage_detail: coverage,
    })
}

/// Python `datetime.isoformat()` of a naive local time (microseconds only when non-zero).
fn python_isoformat(time: &NaiveDateTime) -> String {
    let micros = time.nanosecond() / 1000;
    if micros == 0 {
        time.format("%Y-%m-%dT%H:%M:%S").to_string()
    } else {
        format!("{}.{micros:06}", time.format("%Y-%m-%dT%H:%M:%S"))
    }
}

/// Evaluates an e2e output against a key and builds the full report.
///
/// # Errors
/// The embedder's error, or [`EvalError::InvalidScores`].
pub fn evaluate(
    output: &E2eOutput,
    key: &Key,
    embedder: &dyn Embedder,
    options: &EvalOptions,
) -> Result<EvalReport> {
    let pipeline = PipelineOutput::from_e2e(output);
    let body = evaluate_body(
        key,
        &pipeline,
        embedder,
        options.threshold,
        options.use_value_guard,
        options.assignment,
    )?;
    let now = options.timestamp.unwrap_or_else(|| Local::now().naive_local());
    Ok(EvalReport {
        evaluation_id: format!("eval_{}", now.format("%Y%m%d_%H%M%S")),
        timestamp: python_isoformat(&now),
        input_file: options.input_file.clone(),
        ground_truth_file: options.ground_truth_file.clone(),
        similarity_threshold: options.threshold,
        embedding_model: options
            .embedding_model
            .clone()
            .unwrap_or_else(|| embedder.model_name().to_owned()),
        value_guard: options.use_value_guard,
        assignment: options.assignment.as_str().to_owned(),
        metrics: body.metrics,
        matched_contradictions: body.matched_contradictions,
        missed_contradictions: body.missed_contradictions,
        false_positives: body.false_positives,
        unpaired_correct_facts: body.unpaired_correct_facts,
        fact_coverage_detail: body.fact_coverage_detail,
    })
}

/// `<input stem>_eval.json` next to the input.
#[must_use]
pub fn default_report_path(input_path: &Path) -> PathBuf {
    let stem = input_path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    input_path.with_file_name(format!("{stem}_eval.json"))
}

/// Parses the key, loads the e2e output, evaluates, and writes the report JSON to
/// `report_path` (creating its directory) when given. `options.input_file` and
/// `options.ground_truth_file` are filled from the paths when empty. The caller prints
/// [`EvalReport::summary`].
///
/// # Errors
/// [`EvalError::EmptyKey`] when the key has no rows, plus the load, embedding and write errors.
pub fn run_evaluation(
    input_path: &Path,
    key_path: &Path,
    report_path: Option<&Path>,
    embedder: &dyn Embedder,
    options: &EvalOptions,
) -> Result<EvalReport> {
    let key = parse_key_file(key_path)?;
    run_evaluation_with_key(input_path, &key, key_path, report_path, embedder, options)
}

/// [`run_evaluation`] with a key the caller already parsed from `key_path` (the path is used
/// for the report and the empty-key error).
///
/// # Errors
/// [`EvalError::EmptyKey`] when the key has no rows, plus the load, embedding and write errors.
pub fn run_evaluation_with_key(
    input_path: &Path,
    key: &Key,
    key_path: &Path,
    report_path: Option<&Path>,
    embedder: &dyn Embedder,
    options: &EvalOptions,
) -> Result<EvalReport> {
    if key.is_empty() {
        return Err(EvalError::EmptyKey { path: key_path.to_path_buf() });
    }
    let output = load_e2e_output(input_path)?;
    let mut options = options.clone();
    if options.input_file.is_empty() {
        options.input_file = input_path.display().to_string();
    }
    if options.ground_truth_file.is_empty() {
        options.ground_truth_file = key_path.display().to_string();
    }
    let report = evaluate(&output, key, embedder, &options)?;
    if let Some(path) = report_path {
        let io_error = |source| EvalError::Io { path: path.to_path_buf(), source };
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(io_error)?;
        }
        std::fs::write(path, report.to_json()?).map_err(io_error)?;
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_takes_first_maximum() {
        assert_eq!(argmax([0.1, 0.5, 0.5, 0.2].into_iter()), 1);
        assert_eq!(argmax(std::iter::empty()), 0);
    }

    #[test]
    fn isoformat_like_python() {
        let t = NaiveDateTime::parse_from_str("2026-01-02 03:04:05", "%Y-%m-%d %H:%M:%S").unwrap();
        assert_eq!(python_isoformat(&t), "2026-01-02T03:04:05");
        let t = t + chrono::Duration::microseconds(1234);
        assert_eq!(python_isoformat(&t), "2026-01-02T03:04:05.001234");
    }

    #[test]
    fn report_path_next_to_input() {
        assert_eq!(
            default_report_path(Path::new("data/output/green_valley.json")),
            Path::new("data/output/green_valley_eval.json")
        );
    }
}
