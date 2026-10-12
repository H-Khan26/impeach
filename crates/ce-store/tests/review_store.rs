//! `SqliteReviewStore` against temporary SQLite files.

use ce_core::{
    ContradictionReviewItem, ContradictionSource, DocId, FactId, NliLabel, ReviewClassification,
    ReviewStatus,
};
use ce_store::review_store::{build_dedupe_key, TABLE_NAME};
use ce_store::{ReviewStoreConfig, SqliteReviewStore, StoreError};
use pretty_assertions::assert_eq;
use rusqlite::Connection;
use serde_json::json;

fn item(new_fact: &str, existing_fact: &str) -> ContradictionReviewItem {
    ContradictionReviewItem {
        case_id: "CASE-1".into(),
        session_id: "doc-1".into(),
        segment_id: "doc-1".into(),
        segment_index: 0,
        doc_id: DocId::new("doc-1"),
        stage: "PROCESSING".into(),
        source_stream: Some("processing_pipeline".into()),
        source_pipeline: Some("processing_v17".into()),
        new_fact_id: FactId::new(new_fact),
        existing_fact_id: FactId::new(existing_fact),
        new_fact_payload: json!({"fact_text": "Rent was paid.", "certainty": 0.9})
            .as_object()
            .cloned()
            .unwrap_or_default(),
        nli_score: 0.91,
        nli_label: Some(NliLabel::Contradiction),
        classification: ReviewClassification::Contradiction,
        contradiction_source: ContradictionSource::IntraDocument,
        ..ContradictionReviewItem::default()
    }
}

fn open(dir: &tempfile::TempDir) -> SqliteReviewStore {
    SqliteReviewStore::open(&ReviewStoreConfig::new(dir.path().join("review.db")))
        .expect("schema initialises")
        .expect("store enabled")
}

#[test]
fn schema_matches_python_ddl_and_is_idempotent() {
    let dir = tempfile::tempdir().expect("temp dir");
    let store = open(&dir);
    store.initialize_schema().expect("second initialisation is a no-op");
    let conn = Connection::open(store.db_path()).expect("open");
    let mut statement = conn.prepare(&format!("PRAGMA table_info({TABLE_NAME})")).expect("pragma");
    let columns: Vec<(String, String, bool)> = statement
        .query_map([], |row| Ok((row.get(1)?, row.get(2)?, row.get::<_, i64>(3)? == 1)))
        .expect("rows")
        .collect::<rusqlite::Result<_>>()
        .expect("columns");
    let names: Vec<&str> = columns.iter().map(|(name, _, _)| name.as_str()).collect();
    assert_eq!(names.len(), 27);
    assert_eq!(names[..4], ["id", "case_id", "session_id", "segment_id"]);
    assert_eq!(names[26], "dedupe_key");
    assert!(columns.contains(&("nli_score".into(), "REAL".into(), true)));
    assert!(columns.contains(&("witness_name".into(), "TEXT".into(), false)));
    let unique: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_index_list(?1) WHERE \"unique\" = 1",
            [TABLE_NAME],
            |row| row.get(0),
        )
        .expect("index list");
    assert_eq!(unique, 1, "dedupe_key is UNIQUE");
}

#[test]
fn migration_adds_source_pipeline_to_old_tables() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("old.db");
    let conn = Connection::open(&path).expect("open");
    conn.execute(
        &format!("CREATE TABLE {TABLE_NAME} (id INTEGER PRIMARY KEY AUTOINCREMENT, dedupe_key TEXT NOT NULL UNIQUE)"),
        [],
    )
    .expect("old table");
    let store = SqliteReviewStore::new(&ReviewStoreConfig::new(&path));
    store.initialize_schema().expect("migrates");
    let has_column: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info(?1) WHERE name = 'source_pipeline'",
            [TABLE_NAME],
            |row| row.get(0),
        )
        .expect("table info");
    assert_eq!(has_column, 1);
}

#[test]
fn insert_fills_keys_and_ignores_duplicates() {
    let dir = tempfile::tempdir().expect("temp dir");
    let store = open(&dir);
    let mut items = vec![item("FACT_1", "77"), item("FACT_2", "78")];
    assert_eq!(store.insert_review_items(&mut items).expect("insert"), 2);
    assert_eq!(
        items[0].dedupe_key.as_deref(),
        Some("CASE-1|doc-1|doc-1|doc-1|FACT_1|77|intra_document")
    );
    assert_eq!(items[0].dedupe_key.clone(), Some(build_dedupe_key(&items[0])));
    assert!(!items[0].created_at.is_empty());
    assert_eq!(items[0].created_at, items[0].updated_at);
    assert!(!items[0].created_at.ends_with('Z'), "naive UTC like datetime.utcnow()");

    let mut again = vec![item("FACT_1", "77"), item("FACT_3", "79")];
    assert_eq!(store.insert_review_items(&mut again).expect("insert"), 1, "duplicate ignored");
    let mut custom = vec![item("FACT_1", "77")];
    custom[0].dedupe_key = Some("custom-key".into());
    assert_eq!(store.insert_review_items(&mut custom).expect("insert"), 1, "explicit key kept");
    assert_eq!(store.insert_review_items(&mut []).expect("empty"), 0);

    let pending = store.get_by_status(ReviewStatus::Pending, None).expect("pending");
    assert_eq!(pending.len(), 4);
    let first = &pending[0];
    assert_eq!(first.id, Some(1));
    let mut expected = items[0].clone();
    expected.id = Some(1);
    assert_eq!(first, &expected, "round trip");
    assert_eq!(store.get_by_status(ReviewStatus::Pending, Some(2)).expect("limit").len(), 2);
}

#[test]
fn payloads_are_written_in_python_json_format() {
    let dir = tempfile::tempdir().expect("temp dir");
    let store = open(&dir);
    let mut items = vec![item("FACT_1", "77")];
    items[0].existing_fact_payload.clear();
    items[0].nli_label = None;
    store.insert_review_items(&mut items).expect("insert");
    let conn = Connection::open(store.db_path()).expect("open");
    let (new_payload, existing, label): (String, Option<String>, String) = conn
        .query_row(
            &format!("SELECT new_fact_payload, existing_fact_payload, nli_label FROM {TABLE_NAME}"),
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .expect("row");
    assert_eq!(new_payload, r#"{"fact_text": "Rent was paid.", "certainty": 0.9}"#);
    assert_eq!(existing, None, "an empty payload is stored as NULL");
    assert_eq!(label, "");
    let read =
        store.get_by_dedupe_key(items[0].dedupe_key.as_deref().unwrap_or_default()).expect("read");
    let read = read.expect("row exists");
    assert!(read.existing_fact_payload.is_empty());
    assert_eq!(read.nli_label, None);
    assert_eq!(store.get_by_dedupe_key("missing").expect("read"), None);
}

#[test]
fn status_updates_and_commit_tracking() {
    let dir = tempfile::tempdir().expect("temp dir");
    let store = open(&dir);
    let mut items = vec![item("FACT_1", "77"), item("FACT_2", "78"), item("FACT_3", "79")];
    store.insert_review_items(&mut items).expect("insert");
    assert!(store
        .update_status(1, ReviewStatus::Accepted, Some("reviewer"), Some("ok"))
        .expect("update"));
    assert!(store.update_status(2, ReviewStatus::Accepted, None, None).expect("update"));
    assert!(store
        .update_status(3, ReviewStatus::Rejected, Some("reviewer"), None)
        .expect("update"));
    assert!(!store.update_status(99, ReviewStatus::Accepted, None, None).expect("missing row"));

    let accepted = store.get_by_status(ReviewStatus::Accepted, None).expect("accepted");
    assert_eq!(accepted.iter().map(|row| row.id).collect::<Vec<_>>(), vec![Some(1), Some(2)]);
    assert_eq!(accepted[0].resolved_by.as_deref(), Some("reviewer"));
    assert_eq!(accepted[0].resolution_note.as_deref(), Some("ok"));

    assert!(store.mark_committed(1, &FactId::new("501")).expect("commit"));
    let uncommitted = store.get_accepted_uncommitted(None).expect("uncommitted");
    assert_eq!(uncommitted.iter().map(|row| row.id).collect::<Vec<_>>(), vec![Some(2)]);
    let committed = store.get_by_status(ReviewStatus::Accepted, None).expect("accepted");
    assert_eq!(committed[0].helix_new_fact_id, Some(FactId::new("501")));
    assert!(store.get_accepted_uncommitted(Some(0)).expect("limit 0").is_empty());
}

#[test]
fn upsert_refreshes_detection_and_keeps_review_state() {
    let dir = tempfile::tempdir().expect("temp dir");
    let store = open(&dir);
    let mut items = vec![item("FACT_1", "77")];
    store.insert_review_items(&mut items).expect("insert");
    store.update_status(1, ReviewStatus::Accepted, Some("reviewer"), None).expect("update");

    let mut rerun = vec![item("FACT_1", "77"), item("FACT_2", "78")];
    rerun[0].nli_score = 0.42;
    rerun[0].classification = ReviewClassification::NeedsReview;
    assert_eq!(store.upsert_review_items(&mut rerun).expect("upsert"), 2);
    let row = store
        .get_by_dedupe_key(items[0].dedupe_key.as_deref().unwrap_or_default())
        .expect("read")
        .expect("row exists");
    assert_eq!(row.id, Some(1), "same row");
    assert!((row.nli_score - 0.42).abs() < f64::EPSILON);
    assert_eq!(row.classification, ReviewClassification::NeedsReview);
    assert_eq!(row.status, ReviewStatus::Accepted, "review state kept");
    assert_eq!(row.resolved_by.as_deref(), Some("reviewer"));
    assert_eq!(row.created_at, items[0].created_at);
    assert_eq!(store.get_by_status(ReviewStatus::Pending, None).expect("pending").len(), 1);
}

#[test]
fn fail_on_sqlite_error_controls_propagation() {
    let dir = tempfile::tempdir().expect("temp dir");
    let unreachable = dir.path().join("missing-dir").join("review.db");

    let strict = ReviewStoreConfig::new(&unreachable);
    assert!(matches!(SqliteReviewStore::open(&strict), Err(StoreError::Sqlite(_))));
    let lenient = ReviewStoreConfig { fail_on_sqlite_error: false, ..strict.clone() };
    assert!(SqliteReviewStore::open(&lenient).expect("swallowed").is_none(), "writes disabled");

    // A database without the table makes inserts fail.
    let no_table = dir.path().join("empty.db");
    let strict = SqliteReviewStore::new(&ReviewStoreConfig::new(&no_table));
    let mut items = vec![item("FACT_1", "77")];
    assert!(matches!(strict.try_insert_review_items(&mut items), Err(StoreError::Sqlite(_))));
    let lenient = SqliteReviewStore::new(&ReviewStoreConfig {
        db_path: no_table,
        fail_on_sqlite_error: false,
    });
    assert!(!lenient.fail_on_sqlite_error());
    assert_eq!(lenient.try_insert_review_items(&mut items).expect("swallowed"), 0);
}
