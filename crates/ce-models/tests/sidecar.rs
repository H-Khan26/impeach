//! Sidecar client tests. This test binary doubles as the sidecar: when started with
//! `CE_MODELS_SIDECAR_HELPER` set it serves the JSON-lines protocol on stdio instead of running
//! the tests (`harness = false`).

use std::io::{BufRead, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use ce_models::{
    Coref, CorefCluster, ModelError, ModelRegistry, SidecarCommand, SidecarCoref, SidecarProcess,
};
use serde_json::{json, Value};

const HELPER_ENV: &str = "CE_MODELS_SIDECAR_HELPER";
const SPAWN_LOG_ENV: &str = "CE_MODELS_SIDECAR_SPAWN_LOG";

fn main() {
    if std::env::var_os(HELPER_ENV).is_some() {
        serve();
        return;
    }
    let tests: &[(&str, fn())] = &[
        ("coref_returns_char_offsets", coref_returns_char_offsets),
        ("error_member_is_inference_error", error_member_is_inference_error),
        (
            "non_json_line_is_protocol_error_then_restart",
            non_json_line_is_protocol_error_then_restart,
        ),
        ("crash_is_reported_and_restarted", crash_is_reported_and_restarted),
        ("timeout_is_reported_and_restarted", timeout_is_reported_and_restarted),
        ("registry_builds_sidecar_coref_once", registry_builds_sidecar_coref_once),
    ];
    let mut failed = Vec::new();
    for (name, test) in tests {
        let outcome = catch_unwind(AssertUnwindSafe(test));
        println!("test {name} ... {}", if outcome.is_ok() { "ok" } else { "FAILED" });
        if outcome.is_err() {
            failed.push(*name);
        }
    }
    println!("\n{} passed; {} failed", tests.len() - failed.len(), failed.len());
    if !failed.is_empty() {
        std::process::exit(1);
    }
}

/// Sidecar mode: answers one JSON line per request.
fn serve() {
    if let Some(log) = std::env::var_os(SPAWN_LOG_ENV) {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log)
            .expect("spawn log is writable");
        writeln!(file, "spawn").expect("write spawn log");
    }
    eprintln!("helper sidecar ready");
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line.expect("stdin is UTF-8");
        let request: Value = serde_json::from_str(&line).expect("request is JSON");
        let response = match request["op"].as_str() {
            Some("coref") => coref(request["text"].as_str().unwrap_or_default()),
            Some("error") => json!({"error": "boom"}),
            Some("garbage") => {
                writeln!(stdout, "this is not json").expect("write");
                stdout.flush().expect("flush");
                continue;
            }
            Some("crash") => std::process::exit(3),
            Some("sleep") => {
                std::thread::sleep(Duration::from_secs(3));
                json!({"slept": true})
            }
            _ => json!({"echo": request}),
        };
        // A blank line first: the client skips it.
        writeln!(stdout).expect("write");
        writeln!(stdout, "{response}").expect("write");
        stdout.flush().expect("flush");
    }
}

/// Clusters every occurrence of `Zoë` / `she` and of `Alice` / `her`, in character offsets,
/// plus one out-of-range span the client drops.
fn coref(text: &str) -> Value {
    let chars: Vec<char> = text.chars().collect();
    let find = |word: &str| -> Vec<Value> {
        let w: Vec<char> = word.chars().collect();
        (0..chars.len().saturating_sub(w.len() - 1))
            .filter(|&i| chars[i..i + w.len()] == w[..])
            .map(|i| json!([i, i + w.len()]))
            .collect()
    };
    let mut first = find("Zoë");
    first.extend(find("she"));
    let mut second = find("Alice");
    second.extend(find("her"));
    second.push(json!([0, chars.len() + 10]));
    json!({"clusters": [first, second]})
}

fn helper_command(spawn_log: Option<&Path>) -> SidecarCommand {
    let exe = std::env::current_exe().expect("test executable path");
    let mut command = SidecarCommand::new(exe).env(HELPER_ENV, "1");
    if let Some(log) = spawn_log {
        command = command.env(SPAWN_LOG_ENV, log.as_os_str());
    }
    command
}

fn spawn_count(log: &Path) -> usize {
    std::fs::read_to_string(log).map_or(0, |s| s.lines().count())
}

fn temp_log() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("temp dir");
    let log = dir.path().join("spawns.log");
    (dir, log)
}

fn coref_returns_char_offsets() {
    let coref = SidecarCoref::new(helper_command(None)).expect("start sidecar");
    let text = "Zoë met Alice. Then she thanked her.";
    let clusters = coref.resolve(text).expect("resolve");
    assert_eq!(
        clusters,
        vec![
            CorefCluster { spans: vec![(0, 3), (20, 23)] },
            CorefCluster { spans: vec![(8, 13), (32, 35)] },
        ]
    );
    let chars: Vec<char> = text.chars().collect();
    let span_text = |(s, e): (usize, usize)| chars[s..e].iter().collect::<String>();
    assert_eq!(span_text(clusters[0].spans[1]), "she");
    assert_eq!(span_text(clusters[1].spans[1]), "her");
    // The same process serves further requests.
    assert_eq!(
        coref.resolve("nothing to see").expect("resolve"),
        vec![CorefCluster { spans: vec![] }, CorefCluster { spans: vec![] },]
    );
}

fn error_member_is_inference_error() {
    let process = SidecarProcess::spawn(helper_command(None)).expect("start sidecar");
    let err = process.request(&json!({"op": "error"})).expect_err("error response");
    assert!(matches!(err, ModelError::Inference(ref m) if m.contains("boom")), "{err:?}");
    let echo = process.request(&json!({"op": "echo", "x": "é"})).expect("echo");
    assert_eq!(echo["echo"]["x"], "é");
}

fn non_json_line_is_protocol_error_then_restart() {
    let (_dir, log) = temp_log();
    let process = SidecarProcess::spawn(helper_command(Some(&log))).expect("start sidecar");
    let err = process.request(&json!({"op": "garbage"})).expect_err("garbage");
    assert!(matches!(err, ModelError::Protocol(_)), "{err:?}");
    let echo = process.request(&json!({"op": "echo"})).expect("echo after restart");
    assert_eq!(echo["echo"]["op"], "echo");
    assert_eq!(spawn_count(&log), 2);
}

fn crash_is_reported_and_restarted() {
    let (_dir, log) = temp_log();
    let process = SidecarProcess::spawn(helper_command(Some(&log))).expect("start sidecar");
    let err = process.request(&json!({"op": "crash"})).expect_err("crash");
    assert!(matches!(err, ModelError::Protocol(ref m) if m.contains("stopped")), "{err:?}");
    process.request(&json!({"op": "echo"})).expect("echo after restart");
    assert_eq!(spawn_count(&log), 2);
}

fn timeout_is_reported_and_restarted() {
    let (_dir, log) = temp_log();
    let command = helper_command(Some(&log)).timeout(Duration::from_millis(500));
    let process = SidecarProcess::spawn(command).expect("start sidecar");
    let err = process.request(&json!({"op": "sleep"})).expect_err("timeout");
    assert!(matches!(err, ModelError::Inference(ref m) if m.contains("did not answer")), "{err:?}");
    process.request(&json!({"op": "echo"})).expect("echo after restart");
    assert_eq!(spawn_count(&log), 2);
}

fn registry_builds_sidecar_coref_once() {
    let (_dir, log) = temp_log();
    let config = ce_config::load_default_config().expect("default config");
    let registry =
        ModelRegistry::from_config(&config).with_coref_sidecar(helper_command(Some(&log)));
    let key = registry.key(ce_models::ModelKind::Coref);
    let first = registry.coref(&key).expect("coref sidecar");
    let second = registry.coref(&key).expect("cached coref");
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(first.resolve("Zoë").expect("resolve")[0].spans, vec![(0, 3)]);
    assert_eq!(spawn_count(&log), 1);
}
