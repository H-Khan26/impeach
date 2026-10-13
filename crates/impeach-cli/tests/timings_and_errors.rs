//! `--timings` summaries and the locked-store message of the `impeach` binary; no models or
//! network needed.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture(relative: &str) -> PathBuf {
    workspace_root().join("fixtures").join(relative)
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_impeach"))
        .args(args)
        .env_remove("RUST_LOG")
        .output()
        .expect("spawn impeach")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn path_arg(path: &Path) -> &str {
    path.to_str().expect("UTF-8 path")
}

/// The packaged config without models (offline RAG index, no GLiNER2 detector, rule
/// coreference, no relation stage), with the `extra` replacements applied.
fn offline_config(dir: &Path, extra: &[(&str, &str)]) -> PathBuf {
    let source = workspace_root().join("assets/config/processing/processing_v17.yaml");
    let mut text = std::fs::read_to_string(source).expect("read packaged config");
    let replacements = [
        ("    offline: false", "    offline: true"),
        (
            "      - type: \"gliner2\"\n        model_name_suffix: \"gliner2\"\n        min_score: 0.4\n        label_set: \"legal_entities\"\n",
            "",
        ),
        ("backend: \"fastcoref\"", "backend: \"rule\""),
        (
            "    name: relation_extraction\n    enabled: true",
            "    name: relation_extraction\n    enabled: false",
        ),
    ];
    for (from, to) in replacements.iter().chain(extra) {
        assert!(text.contains(from), "config contains {from:?}");
        text = text.replacen(from, to, 1);
    }
    let path = dir.join("config.yaml");
    std::fs::write(&path, text).expect("write config");
    path
}

#[test]
fn process_timings_go_to_stderr() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = offline_config(dir.path(), &[("mode: \"embedded\"", "mode: \"memory\"")]);
    let jsonl = fixture("input/Mason_intra.jsonl");
    let output = run(&["process", path_arg(&jsonl), "-c", path_arg(&config), "--timings"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stderr(&output);
    assert!(text.contains("Stage timings: "), "{text}");
    for stage in ["rag_index", "mention_detection", "ed_resolution", "all stages"] {
        assert!(text.contains(stage), "`{stage}` missing from: {text}");
    }
    assert!(text.contains("100.0%"), "{text}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.contains("Stage timings:"), "timings stay off stdout: {stdout}");

    let without = run(&["process", path_arg(&jsonl), "-c", path_arg(&config)]);
    assert!(without.status.success(), "{}", stderr(&without));
    assert!(!stderr(&without).contains("Stage timings:"));
}

#[test]
fn run_e2e_timings_list_documents_and_totals() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = offline_config(dir.path(), &[("mode: \"embedded\"", "mode: \"memory\"")]);
    let out = dir.path().join("e2e.json");
    let output = run(&[
        "run-e2e",
        "--inputs",
        path_arg(&fixture("input/Mason_billy.jsonl")),
        path_arg(&fixture("input/Mason_watson.jsonl")),
        "-c",
        path_arg(&config),
        "-o",
        path_arg(&out),
        "--no-eval",
        "--timings",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stderr(&output);
    assert!(text.contains("Mason_billy.jsonl"), "{text}");
    assert!(text.contains("Mason_watson.jsonl"), "{text}");
    assert!(text.contains("Stage timings: total over 2 document(s)"), "{text}");
}

#[test]
#[cfg(feature = "helix")]
fn a_locked_store_names_the_data_dir_and_the_other_process() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data_dir = dir.path().join("helix");
    std::fs::create_dir_all(&data_dir).expect("data dir");
    let data_dir_yaml = data_dir.to_string_lossy().replace('\\', "/");
    let data_dir_line = format!("data_dir: \"{data_dir_yaml}\"");
    let config = offline_config(dir.path(), &[("data_dir: \"data/helix\"", &data_dir_line)]);
    // Another process's writer lock on the configured namespace.
    let holder = std::fs::File::create(data_dir.join("processing_v17.lock")).expect("lock file");
    fs4::FileExt::try_lock(&holder).expect("take the writer lock");

    let jsonl = fixture("input/Mason_intra.jsonl");
    let output = run(&["process", path_arg(&jsonl), "-c", path_arg(&config)]);
    assert_eq!(output.status.code(), Some(1));
    let text = stderr(&output);
    for part in ["is locked", "another impeach process holds it", "processing_v17", "helix"] {
        assert!(text.contains(part), "`{part}` missing from: {text}");
    }
    fs4::FileExt::unlock(&holder).expect("unlock");
}
