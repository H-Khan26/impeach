//! `--timings`: per-stage wall-time summaries printed to stderr.

use std::fmt::Write as _;
use std::time::Duration;

use impeach_pipeline::{RunReport, StageTiming};

/// Width of the stage-name column.
const NAME_WIDTH: usize = 28;

/// One table: a row per stage (milliseconds and share of the summed stage time), the stage
/// total and, when given, the wall time of the whole run.
fn table(out: &mut String, title: &str, timings: &[StageTiming], total: Option<Duration>) {
    let stages: Duration = timings.iter().map(|timing| timing.elapsed).sum();
    let stages_ms = stages.as_secs_f64() * 1000.0;
    // Writing to a String cannot fail.
    let _ = writeln!(out, "{title}");
    let _ = writeln!(out, "  {:<NAME_WIDTH$} {:>12} {:>7}", "stage", "ms", "share");
    if timings.is_empty() {
        let _ = writeln!(out, "  (no stages ran)");
    }
    for timing in timings {
        let ms = timing.elapsed.as_secs_f64() * 1000.0;
        let share = if stages_ms > 0.0 { ms / stages_ms * 100.0 } else { 0.0 };
        let _ = writeln!(out, "  {:<NAME_WIDTH$} {ms:>12.1} {share:>6.1}%", timing.stage);
    }
    let _ = writeln!(out, "  {:<NAME_WIDTH$} {stages_ms:>12.1} {:>6.1}%", "all stages", 100.0);
    if let Some(total) = total {
        let total_ms = total.as_secs_f64() * 1000.0;
        let _ = writeln!(out, "  {:<NAME_WIDTH$} {total_ms:>12.1}", "run (incl. input loading)");
    }
}

/// The timing summary of one processed document.
pub fn render_run(document: &str, report: &RunReport) -> String {
    let mut out = String::new();
    table(
        &mut out,
        &format!("Stage timings: {document}"),
        &report.stage_timings,
        Some(report.total),
    );
    out
}

/// The timing summary of an e2e run: one table per document, then the per-stage totals over
/// every document (stages in first-run order).
pub fn render_e2e<'a>(documents: impl IntoIterator<Item = (String, &'a RunReport)>) -> String {
    let mut out = String::new();
    let mut totals: Vec<StageTiming> = Vec::new();
    let mut total_run = Duration::ZERO;
    let mut count = 0_usize;
    for (document, report) in documents {
        table(
            &mut out,
            &format!("Stage timings: {document}"),
            &report.stage_timings,
            Some(report.total),
        );
        out.push('\n');
        for timing in &report.stage_timings {
            match totals.iter_mut().find(|total| total.stage == timing.stage) {
                Some(total) => total.elapsed += timing.elapsed,
                None => totals.push(timing.clone()),
            }
        }
        total_run += report.total;
        count += 1;
    }
    let title = format!("Stage timings: total over {count} document(s)");
    table(&mut out, &title, &totals, Some(total_run));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(stages: &[(&str, u64)], total_ms: u64) -> RunReport {
        RunReport {
            stage_timings: stages
                .iter()
                .map(|(stage, ms)| StageTiming {
                    stage: (*stage).to_owned(),
                    elapsed: Duration::from_millis(*ms),
                })
                .collect(),
            total: Duration::from_millis(total_ms),
        }
    }

    #[test]
    fn run_table_lists_ms_and_share() {
        let text = render_run("doc.jsonl", &report(&[("rag_index", 300), ("ed", 100)], 450));
        assert!(text.starts_with("Stage timings: doc.jsonl\n"), "{text}");
        let rag =
            text.lines().find(|line| line.trim_start().starts_with("rag_index")).expect("row");
        assert!(rag.contains("300.0") && rag.ends_with("75.0%"), "{rag}");
        let ed = text.lines().find(|line| line.trim_start().starts_with("ed ")).expect("row");
        assert!(ed.ends_with("25.0%"), "{ed}");
        assert!(text.contains("all stages") && text.contains("400.0"), "{text}");
        assert!(text.contains("run (incl. input loading)") && text.contains("450.0"), "{text}");
    }

    #[test]
    fn e2e_totals_sum_stages_by_name() {
        let first = report(&[("rag_index", 100), ("ed", 100)], 210);
        let second = report(&[("rag_index", 300), ("ed", 100)], 420);
        let text = render_e2e([("a.jsonl".to_owned(), &first), ("b.jsonl".to_owned(), &second)]);
        assert!(text.contains("Stage timings: a.jsonl") && text.contains("Stage timings: b.jsonl"));
        let totals = text.split("total over 2 document(s)").nth(1).expect("totals table");
        let rag =
            totals.lines().find(|line| line.trim_start().starts_with("rag_index")).expect("row");
        assert!(rag.contains("400.0") && rag.ends_with("66.7%"), "{rag}");
        assert!(totals.contains("630.0"), "{totals}");
    }

    #[test]
    fn empty_runs_render_without_division_by_zero() {
        let text = render_run("doc.jsonl", &RunReport::default());
        assert!(text.contains("(no stages ran)"), "{text}");
        assert!(!text.contains("NaN"), "{text}");
    }
}
