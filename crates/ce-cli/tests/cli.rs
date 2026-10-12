//! End-to-end tests of the `contradiction-engine` binary that need no models or network.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture(relative: &str) -> PathBuf {
    workspace_root().join("fixtures").join(relative)
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_contradiction-engine"))
        .args(args)
        .env_remove("RUST_LOG")
        .output()
        .expect("spawn contradiction-engine")
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("stdout is UTF-8")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn path_arg(path: &Path) -> &str {
    path.to_str().expect("UTF-8 path")
}

/// The packaged config with the given replacements applied, written to `dir/config.yaml`.
fn write_config(dir: &Path, replacements: &[(&str, &str)]) -> PathBuf {
    let source = workspace_root().join("assets/config/processing/processing_v17.yaml");
    let mut text = std::fs::read_to_string(source).expect("read packaged config");
    for (from, to) in replacements {
        assert!(text.contains(from), "config contains {from:?}");
        text = text.replacen(from, to, 1);
    }
    let path = dir.join("config.yaml");
    std::fs::write(&path, text).expect("write config");
    path
}

#[test]
fn help_lists_every_subcommand() {
    let output = run(&["--help"]);
    assert!(output.status.success());
    let text = stdout(&output);
    for command in ["ingest", "process", "run", "eval", "store"] {
        assert!(text.contains(command), "help lists {command}: {text}");
    }
}

#[test]
fn ingest_writes_one_jsonl_record() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("note.jsonl");
    let input = fixture("input/ingest/sample_note.txt");
    let output = run(&["ingest", path_arg(&input), "-o", path_arg(&out), "--case-id", "CASE-7"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("Ingested document: sample_note"), "{text}");

    let jsonl = std::fs::read_to_string(&out).expect("read jsonl");
    assert_eq!(jsonl.lines().count(), 1);
    let record: serde_json::Value = serde_json::from_str(jsonl.trim_end()).expect("JSON line");
    assert_eq!(record["doc_id"], "sample_note");
    assert_eq!(record["metadata"]["case_id"], "CASE-7");
    assert!(record["text"].as_str().is_some_and(|t| !t.is_empty()));
}

#[test]
fn missing_input_is_a_general_error() {
    let output = run(&["process", "does-not-exist.jsonl"]);
    assert_eq!(output.status.code(), Some(1));
    let text = stderr(&output);
    assert!(text.starts_with("Error: "), "{text}");
}

#[test]
fn invalid_config_is_a_configuration_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = write_config(dir.path(), &[("helix:", "unknown_block: 1\nhelix:")]);
    let jsonl = fixture("input/Mason_intra.jsonl");
    let output = run(&["process", path_arg(&jsonl), "-c", path_arg(&config)]);
    assert_eq!(output.status.code(), Some(1));
    let text = stderr(&output);
    assert!(text.starts_with("Configuration Error: "), "{text}");
}

#[test]
fn process_without_enabled_stages_writes_the_summary() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = write_config(
        dir.path(),
        &[("    enabled: true", "    enabled: false"), ("mode: \"embedded\"", "mode: \"memory\"")],
    );
    let summary = dir.path().join("summary.json");
    let db_path = dir.path().join("review.sqlite3");
    let jsonl = fixture("input/Mason_intra.jsonl");
    let output = run(&[
        "process",
        path_arg(&jsonl),
        "-c",
        path_arg(&config),
        "-o",
        path_arg(&summary),
        "--db-path",
        path_arg(&db_path),
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("Contradiction Detection Report"), "{text}");
    assert!(text.contains("Results written to:"), "{text}");
    // No ContradictionCheckStage ran, so nothing was written to SQLite.
    assert!(!text.contains("SQLite output:"), "{text}");

    let value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary).expect("read summary"))
            .expect("summary JSON");
    let keys: Vec<&str> = value.as_object().expect("object").keys().map(String::as_str).collect();
    assert_eq!(
        keys,
        [
            "file_path",
            "config_path",
            "total_facts_extracted",
            "clean_facts_count",
            "contradiction_count",
            "needs_review_count",
            "committed_fact_ids",
            "sqlite_db_path",
        ]
    );
    assert_eq!(value["total_facts_extracted"], 0);
    assert_eq!(value["sqlite_db_path"], serde_json::Value::Null);
}

#[test]
fn store_reset_deletes_the_embedded_namespace() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data_dir = dir.path().join("helix");
    let data_dir_yaml = format!("data_dir: '{}'", data_dir.display());
    let config = write_config(dir.path(), &[("data_dir: \"data/helix\"", &data_dir_yaml)]);
    let db = data_dir.join("processing_v17");
    std::fs::create_dir_all(&db).expect("create db dir");
    std::fs::write(db.join("data.mdb"), b"x").expect("write db file");

    let output = run(&["store", "reset", "-c", path_arg(&config)]);
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(!db.exists());

    let output = run(&["store", "reset", "-c", path_arg(&config)]);
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(stdout(&output).contains("Nothing to reset"));
}

#[test]
fn eval_rejects_a_missing_key() {
    let output = run(&["eval", "-i", "missing.json", "-k", "missing.md"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("Input file not found"), "{}", stderr(&output));
}

#[test]
fn process_runs_mention_definition_and_rule_coreference_stages() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = write_config(
        dir.path(),
        &[
            ("    offline: false", "    offline: true"),
            (
                "      - type: \"gliner2\"\n        model_name_suffix: \"gliner2\"\n        min_score: 0.4\n        label_set: \"legal_entities\"\n",
                "",
            ),
            ("backend: \"fastcoref\"", "backend: \"rule\""),
            ("mode: \"embedded\"", "mode: \"memory\""),
        ],
    );
    let jsonl = fixture("input/green_valley_incident_report.jsonl");
    let output = run(&["-v", "process", path_arg(&jsonl), "-c", path_arg(&config)]);
    assert!(output.status.success(), "{}", stderr(&output));
    let log = stderr(&output);
    for stage in ["rag_index", "mention_detection", "definition_extraction", "coreference"] {
        assert!(log.contains(&format!("stage finished stage=\"{stage}\"")), "{stage}: {log}");
    }
}

#[test]
fn default_fastcoref_backend_falls_back_to_rules_with_a_warning() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = write_config(
        dir.path(),
        &[
            ("    offline: false", "    offline: true"),
            (
                "      - type: \"gliner2\"\n        model_name_suffix: \"gliner2\"\n        min_score: 0.4\n        label_set: \"legal_entities\"\n",
                "",
            ),
            ("mode: \"embedded\"", "mode: \"memory\""),
        ],
    );
    let jsonl = fixture("input/green_valley_incident_report.jsonl");
    let output = run(&["process", path_arg(&jsonl), "-c", path_arg(&config)]);
    let log = stderr(&output);
    assert!(output.status.success(), "{log}");
    assert!(log.contains("stage finished stage=\"coreference\""), "{log}");
    let warning = "set models.coref.command to a fastcoref sidecar, or set \
                   models.coref.backend: rule to silence this";
    assert_eq!(log.matches(warning).count(), 1, "{log}");
    assert!(log.contains("WARN"), "{log}");
}
