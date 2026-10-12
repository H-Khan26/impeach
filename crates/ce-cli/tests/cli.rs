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

/// Disables `RelationStage`, which would load the GLiNER2 model.
const RELATION_STAGE_OFF: (&str, &str) = (
    "    name: relation_extraction\n    enabled: true",
    "    name: relation_extraction\n    enabled: false",
);

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
    for command in ["ingest", "process", "run", "eval", "run-e2e", "store"] {
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
    // The ContradictionCheckStage ran with the review store (no facts, so no rows).
    assert!(text.contains("SQLite output:"), "{text}");
    assert!(db_path.exists());

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
    assert_eq!(value["sqlite_db_path"], path_arg(&db_path));
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
fn process_runs_mention_definition_coreference_and_ed_stages() {
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
            RELATION_STAGE_OFF,
        ],
    );
    let jsonl = fixture("input/green_valley_incident_report.jsonl");
    let output = run(&["-v", "process", path_arg(&jsonl), "-c", path_arg(&config)]);
    assert!(output.status.success(), "{}", stderr(&output));
    let log = stderr(&output);
    for stage in
        ["rag_index", "mention_detection", "definition_extraction", "coreference", "ed_resolution"]
    {
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
            RELATION_STAGE_OFF,
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

/// The packaged config without models: offline RAG index, no GLiNER2 detector, rule
/// coreference, in-memory Helix and no relation stage.
fn offline_config(dir: &Path) -> PathBuf {
    write_config(
        dir,
        &[
            ("    offline: false", "    offline: true"),
            (
                "      - type: \"gliner2\"\n        model_name_suffix: \"gliner2\"\n        min_score: 0.4\n        label_set: \"legal_entities\"\n",
                "",
            ),
            ("backend: \"fastcoref\"", "backend: \"rule\""),
            ("mode: \"embedded\"", "mode: \"memory\""),
            RELATION_STAGE_OFF,
        ],
    )
}

#[test]
fn run_e2e_requires_a_scenario_or_inputs() {
    let output = run(&["run-e2e"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("provide --scenario or --inputs"), "{}", stderr(&output));
}

#[test]
fn run_e2e_lists_missing_files() {
    let output =
        run(&["run-e2e", "--inputs", "missing_a.jsonl", "missing_b.jsonl", "-c", "nope.yaml"]);
    assert_eq!(output.status.code(), Some(1));
    let text = stderr(&output);
    assert!(
        text.contains("file(s) not found: missing_a.jsonl, missing_b.jsonl, nope.yaml"),
        "{text}"
    );
}

#[test]
fn run_e2e_rejects_a_missing_key() {
    let output = run(&[
        "run-e2e",
        "--inputs",
        path_arg(&fixture("input/Mason_intra.jsonl")),
        "--key",
        "missing_key.md",
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("key not found: missing_key.md"), "{}", stderr(&output));
}

#[test]
fn run_e2e_offline_writes_the_output_json() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = offline_config(dir.path());
    let out = dir.path().join("out").join("e2e_custom.json");
    let first = fixture("input/Mason_billy.jsonl");
    let second = fixture("input/Mason_watson.jsonl");
    let output = run(&[
        "run-e2e",
        "--inputs",
        path_arg(&first),
        path_arg(&second),
        "-c",
        path_arg(&config),
        "-o",
        path_arg(&out),
        "--no-eval",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("Document 1: Mason_billy.jsonl"), "{text}");
    assert!(text.contains("Document 2: Mason_watson.jsonl"), "{text}");
    assert!(text.contains("Aggregates"), "{text}");
    assert!(text.contains("Output written to"), "{text}");

    let value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&out).expect("read output"))
            .expect("output JSON");
    let keys: Vec<&str> = value.as_object().expect("object").keys().map(String::as_str).collect();
    assert_eq!(
        keys,
        [
            "run_id",
            "timestamp",
            "scenario",
            "pipeline_name",
            "helix_namespace",
            "helix_url",
            "helix_mode",
            "helix_data_dir",
            "config_path",
            "sqlite_db_path",
            "inputs",
            "status",
            "pipeline_summary",
            "aggregates",
            "documents",
        ]
    );
    assert_eq!(value["status"], "ok");
    assert!(value["run_id"].as_str().is_some_and(|id| id.starts_with("e2e_custom_")));
    assert_eq!(value["scenario"], serde_json::Value::Null);
    assert_eq!(value["helix_mode"], "memory");
    assert_eq!(value["pipeline_summary"]["total_documents"], 2);
    let sqlite = dir.path().join("out").join("e2e_custom.sqlite3");
    assert_eq!(value["sqlite_db_path"], path_arg(&sqlite));
    assert!(sqlite.exists());
    let documents = value["documents"].as_array().expect("documents");
    assert_eq!(documents.len(), 2);
    assert_eq!(documents[0]["case_id"], "mason_v_billy");
    let check = &documents[1]["stage_6_contradiction_check"];
    assert_eq!(check["total_checked"], 0);
    assert_eq!(check["cross_doc_contradictions"], serde_json::json!([]));
    // The text is Python `json.dumps(indent=2)`.
    let raw = std::fs::read_to_string(&out).expect("read output");
    assert!(raw.starts_with("{\n  \"run_id\": "), "{raw}");
}

#[test]
fn run_e2e_scenario_resolves_fixture_inputs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = offline_config(dir.path());
    let out = dir.path().join("gv.json");
    let fixtures = workspace_root().join("fixtures");
    let output = run(&[
        "run-e2e",
        "--scenario",
        "green_valley",
        "--fixtures-dir",
        path_arg(&fixtures),
        "-c",
        path_arg(&config),
        "-o",
        path_arg(&out),
        "--no-eval",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&out).expect("read output"))
            .expect("output JSON");
    assert_eq!(value["scenario"], "green_valley");
    assert!(value["run_id"].as_str().is_some_and(|id| id.starts_with("e2e_green_valley_")));
    let inputs: Vec<&str> =
        value["inputs"].as_array().expect("inputs").iter().filter_map(|v| v.as_str()).collect();
    assert_eq!(inputs.len(), 2);
    assert!(inputs[0].ends_with("green_valley_incident_report.jsonl"), "{inputs:?}");
    assert!(inputs[1].ends_with("lopez_v_green_valley_complaint.jsonl"), "{inputs:?}");
    assert!(!dir.path().join("gv_eval.json").exists());
}

#[test]
fn run_e2e_failing_document_fails_the_run() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = offline_config(dir.path());
    let bad = dir.path().join("bad.jsonl");
    std::fs::write(&bad, "{\"doc_id\": \"bad-1\", \"case_id\": 7}\n").expect("write");
    let out = dir.path().join("failed.json");
    let good = fixture("input/Mason_billy.jsonl");
    let output = run(&[
        "run-e2e",
        "--inputs",
        path_arg(&bad),
        path_arg(&good),
        "-c",
        path_arg(&config),
        "-o",
        path_arg(&out),
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("e2e run failed"), "{}", stderr(&output));
    let value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&out).expect("read output"))
            .expect("output JSON");
    assert_eq!(value["status"], "failed");
    let documents = value["documents"].as_array().expect("documents");
    assert_eq!(documents.len(), 1, "later documents are not run");
    assert_eq!(documents[0]["doc_id"], "bad-1");
    assert_eq!(documents[0]["case_id"], "7");
    assert!(documents[0]["error"].as_str().is_some_and(|e| e.contains("text")), "{value}");
    assert!(documents[0].get("stage_6_contradiction_check").is_none());
}
