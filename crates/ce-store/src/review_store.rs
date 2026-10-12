//! [`SqliteReviewStore`]: the `contradiction_review_queue` table in SQLite.
//!
//! Python origin: `storage/sqlite/review_store.py` (`ContradictionReviewStore`) and the
//! `fail_on_sqlite_error` handling of `stages/contradiction_check.py`. Like the Python store,
//! every operation opens its own connection, so the store is cheap to share across threads.
//! Timestamps are naive UTC ISO strings (Python `datetime.utcnow().isoformat()`), and JSON
//! payloads are written in Python `json.dumps` format so rows match the Python output.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use ce_core::{
    ContradictionReviewItem, ContradictionSource, DocId, FactId, Metadata, NliLabel,
    ReviewClassification, ReviewStatus,
};
use chrono::{NaiveDateTime, Timelike, Utc};
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::error::{Result, StoreError};

/// Name of the review table.
pub const TABLE_NAME: &str = "contradiction_review_queue";

/// DDL of the review table (byte-for-byte the Python statement's columns and constraints).
pub const CREATE_TABLE_SQL: &str = "CREATE TABLE IF NOT EXISTS contradiction_review_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    segment_id TEXT NOT NULL,
    segment_index INTEGER NOT NULL,
    doc_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    witness_name TEXT,
    source_stream TEXT,
    source_pipeline TEXT,
    segment_start_ts TEXT,
    segment_end_ts TEXT,
    new_fact_id TEXT NOT NULL,
    helix_new_fact_id TEXT,
    existing_fact_id TEXT NOT NULL,
    new_fact_payload TEXT NOT NULL,
    existing_fact_payload TEXT,
    nli_score REAL NOT NULL,
    nli_label TEXT NOT NULL,
    classification TEXT NOT NULL,
    contradiction_source TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    resolved_by TEXT,
    resolution_note TEXT,
    dedupe_key TEXT NOT NULL UNIQUE
)";

/// Every column except `id`, in table order (the insert column list).
const INSERT_COLUMNS: [&str; 26] = [
    "case_id",
    "session_id",
    "segment_id",
    "segment_index",
    "doc_id",
    "stage",
    "witness_name",
    "source_stream",
    "source_pipeline",
    "segment_start_ts",
    "segment_end_ts",
    "new_fact_id",
    "helix_new_fact_id",
    "existing_fact_id",
    "new_fact_payload",
    "existing_fact_payload",
    "nli_score",
    "nli_label",
    "classification",
    "contradiction_source",
    "status",
    "created_at",
    "updated_at",
    "resolved_by",
    "resolution_note",
    "dedupe_key",
];

/// Columns an upsert refreshes on a `dedupe_key` conflict: the detection result. Workflow
/// state (`status`, `resolved_by`, `resolution_note`, `helix_new_fact_id`) and `created_at`
/// keep their stored values so a re-run never undoes a review.
const UPSERT_REFRESHED_COLUMNS: [&str; 14] = [
    "segment_index",
    "stage",
    "witness_name",
    "source_stream",
    "source_pipeline",
    "segment_start_ts",
    "segment_end_ts",
    "new_fact_payload",
    "existing_fact_payload",
    "nli_score",
    "nli_label",
    "classification",
    "contradiction_source",
    "updated_at",
];

/// Review-store settings (the `review_store:` block of the contradiction stage).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewStoreConfig {
    /// SQLite database file; its directory must exist.
    pub db_path: PathBuf,
    /// When false, SQLite failures in [`SqliteReviewStore::open`] and
    /// [`SqliteReviewStore::try_insert_review_items`] are logged and swallowed.
    pub fail_on_sqlite_error: bool,
}

impl ReviewStoreConfig {
    /// Config for `db_path` with `fail_on_sqlite_error = true` (the Python default).
    pub fn new(db_path: impl Into<PathBuf>) -> Self {
        Self { db_path: db_path.into(), fail_on_sqlite_error: true }
    }
}

/// Data access layer for the contradiction review queue.
#[derive(Debug, Clone)]
pub struct SqliteReviewStore {
    db_path: PathBuf,
    fail_on_sqlite_error: bool,
}

impl SqliteReviewStore {
    /// Store for `config` without touching the database.
    #[must_use]
    pub fn new(config: &ReviewStoreConfig) -> Self {
        Self { db_path: config.db_path.clone(), fail_on_sqlite_error: config.fail_on_sqlite_error }
    }

    /// Creates the store and its schema, as the contradiction stage does on start-up.
    ///
    /// Returns `Ok(None)` (review writes disabled) when initialisation fails and
    /// `fail_on_sqlite_error` is false.
    ///
    /// # Errors
    /// The initialisation error when `fail_on_sqlite_error` is true.
    pub fn open(config: &ReviewStoreConfig) -> Result<Option<Self>> {
        let store = Self::new(config);
        match store.initialize_schema() {
            Ok(()) => Ok(Some(store)),
            Err(error) => {
                tracing::error!(
                    "failed to initialize SQLite review store at {}: {error}",
                    store.db_path.display()
                );
                if store.fail_on_sqlite_error {
                    Err(error)
                } else {
                    Ok(None)
                }
            }
        }
    }

    /// Database file path.
    #[must_use]
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    /// Whether SQLite failures propagate from the guarded operations.
    #[must_use]
    pub fn fail_on_sqlite_error(&self) -> bool {
        self.fail_on_sqlite_error
    }

    fn connect(&self) -> Result<Connection> {
        Ok(Connection::open(&self.db_path)?)
    }

    /// Creates the table if missing and adds the `source_pipeline` column to tables created
    /// before it existed. Idempotent.
    ///
    /// # Errors
    /// SQLite failures.
    pub fn initialize_schema(&self) -> Result<()> {
        let conn = self.connect()?;
        conn.execute(CREATE_TABLE_SQL, [])?;
        let mut statement = conn.prepare(&format!("PRAGMA table_info({TABLE_NAME})"))?;
        let columns = statement
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<rusqlite::Result<Vec<String>>>()?;
        if !columns.iter().any(|name| name == "source_pipeline") {
            conn.execute(&format!("ALTER TABLE {TABLE_NAME} ADD COLUMN source_pipeline TEXT"), [])?;
        }
        Ok(())
    }

    /// Inserts items with `INSERT OR IGNORE`; rows whose `dedupe_key` already exists are
    /// skipped. Fills a missing `dedupe_key`, `created_at` and `updated_at` on the items.
    /// Returns the number of rows inserted.
    ///
    /// # Errors
    /// SQLite failures (nothing is committed then).
    pub fn insert_review_items(&self, items: &mut [ContradictionReviewItem]) -> Result<usize> {
        self.write_items(items, "INSERT OR IGNORE INTO", "")
    }

    /// [`Self::insert_review_items`] under the `fail_on_sqlite_error` policy: with the flag
    /// off, a SQLite failure is logged and reported as 0 rows inserted.
    ///
    /// # Errors
    /// Non-SQLite failures, and SQLite failures when `fail_on_sqlite_error` is true.
    pub fn try_insert_review_items(&self, items: &mut [ContradictionReviewItem]) -> Result<usize> {
        match self.insert_review_items(items) {
            Err(StoreError::Sqlite(error)) if !self.fail_on_sqlite_error => {
                tracing::error!("SQLite error while inserting review rows: {error}");
                Ok(0)
            }
            other => other,
        }
    }

    /// Inserts items, or on a `dedupe_key` conflict refreshes the detection columns of the
    /// existing row (scores, label, classification, payloads, provenance, `updated_at`) while
    /// keeping its review state. Returns the number of rows inserted or refreshed.
    ///
    /// # Errors
    /// SQLite failures (nothing is committed then).
    pub fn upsert_review_items(&self, items: &mut [ContradictionReviewItem]) -> Result<usize> {
        let assignments = UPSERT_REFRESHED_COLUMNS
            .iter()
            .map(|column| format!("{column} = excluded.{column}"))
            .collect::<Vec<_>>()
            .join(", ");
        self.write_items(
            items,
            "INSERT INTO",
            &format!(" ON CONFLICT(dedupe_key) DO UPDATE SET {assignments}"),
        )
    }

    fn write_items(
        &self,
        items: &mut [ContradictionReviewItem],
        verb: &str,
        conflict_clause: &str,
    ) -> Result<usize> {
        if items.is_empty() {
            return Ok(0);
        }
        let now = python_isoformat(Utc::now().naive_utc());
        let sql = format!(
            "{verb} {TABLE_NAME} ({}) VALUES ({}){conflict_clause}",
            INSERT_COLUMNS.join(", "),
            vec!["?"; INSERT_COLUMNS.len()].join(", ")
        );
        let mut conn = self.connect()?;
        let tx = conn.transaction()?;
        let mut written = 0;
        {
            let mut statement = tx.prepare(&sql)?;
            for item in items.iter_mut() {
                if item.dedupe_key.as_deref().unwrap_or_default().is_empty() {
                    item.dedupe_key = Some(build_dedupe_key(item));
                }
                if item.created_at.is_empty() {
                    item.created_at.clone_from(&now);
                }
                if item.updated_at.is_empty() {
                    item.updated_at.clone_from(&now);
                }
                let new_payload = python_json_dumps(&item.new_fact_payload);
                let existing_payload = (!item.existing_fact_payload.is_empty())
                    .then(|| python_json_dumps(&item.existing_fact_payload));
                written += statement.execute(params![
                    item.case_id,
                    item.session_id,
                    item.segment_id,
                    item.segment_index,
                    item.doc_id.as_str(),
                    item.stage,
                    item.witness_name,
                    item.source_stream,
                    item.source_pipeline,
                    item.segment_start_ts,
                    item.segment_end_ts,
                    item.new_fact_id.as_str(),
                    item.helix_new_fact_id.as_ref().map(FactId::as_str),
                    item.existing_fact_id.as_str(),
                    new_payload,
                    existing_payload,
                    item.nli_score,
                    item.nli_label.map_or("", NliLabel::as_str),
                    item.classification.as_str(),
                    item.contradiction_source.as_str(),
                    item.status.as_str(),
                    item.created_at,
                    item.updated_at,
                    item.resolved_by,
                    item.resolution_note,
                    item.dedupe_key,
                ])?;
            }
        }
        tx.commit()?;
        Ok(written)
    }

    /// Rows with `status`, in id order, at most `limit` when given.
    ///
    /// # Errors
    /// SQLite failures and rows holding values outside the record enums.
    pub fn get_by_status(
        &self,
        status: ReviewStatus,
        limit: Option<usize>,
    ) -> Result<Vec<ContradictionReviewItem>> {
        self.select("WHERE status = ?1", status.as_str(), limit)
    }

    /// `ACCEPTED` rows without a graph fact id yet (for jobs that promote reviewer-approved
    /// facts into the graph), in id order.
    ///
    /// # Errors
    /// As [`Self::get_by_status`].
    pub fn get_accepted_uncommitted(
        &self,
        limit: Option<usize>,
    ) -> Result<Vec<ContradictionReviewItem>> {
        self.select(
            "WHERE status = ?1 AND (helix_new_fact_id IS NULL OR helix_new_fact_id = '')",
            ReviewStatus::Accepted.as_str(),
            limit,
        )
    }

    /// The row with `dedupe_key`, if any.
    ///
    /// # Errors
    /// As [`Self::get_by_status`].
    pub fn get_by_dedupe_key(&self, dedupe_key: &str) -> Result<Option<ContradictionReviewItem>> {
        let conn = self.connect()?;
        let sql = format!(
            "SELECT id, {} FROM {TABLE_NAME} WHERE dedupe_key = ?1",
            INSERT_COLUMNS.join(", ")
        );
        let row = conn.query_row(&sql, [dedupe_key], RawRow::read).optional()?;
        row.map(RawRow::into_item).transpose()
    }

    fn select(
        &self,
        filter: &str,
        status: &str,
        limit: Option<usize>,
    ) -> Result<Vec<ContradictionReviewItem>> {
        let conn = self.connect()?;
        let mut sql = format!(
            "SELECT id, {} FROM {TABLE_NAME} {filter} ORDER BY id",
            INSERT_COLUMNS.join(", ")
        );
        if let Some(limit) = limit {
            let _ = write!(sql, " LIMIT {limit}");
        }
        let mut statement = conn.prepare(&sql)?;
        let rows = statement
            .query_map([status], RawRow::read)?
            .collect::<rusqlite::Result<Vec<RawRow>>>()?;
        rows.into_iter().map(RawRow::into_item).collect()
    }

    /// Sets the workflow status of one row; returns whether the row exists.
    ///
    /// # Errors
    /// SQLite failures.
    pub fn update_status(
        &self,
        review_id: i64,
        status: ReviewStatus,
        resolved_by: Option<&str>,
        resolution_note: Option<&str>,
    ) -> Result<bool> {
        let now = python_isoformat(Utc::now().naive_utc());
        let changed = self.connect()?.execute(
            &format!(
                "UPDATE {TABLE_NAME} SET status = ?1, resolved_by = ?2, resolution_note = ?3, \
                 updated_at = ?4 WHERE id = ?5"
            ),
            params![status.as_str(), resolved_by, resolution_note, now, review_id],
        )?;
        Ok(changed > 0)
    }

    /// Records the graph fact id of a promoted row; returns whether the row exists.
    ///
    /// # Errors
    /// SQLite failures.
    pub fn mark_committed(&self, review_id: i64, helix_fact_id: &FactId) -> Result<bool> {
        let now = python_isoformat(Utc::now().naive_utc());
        let changed = self.connect()?.execute(
            &format!(
                "UPDATE {TABLE_NAME} SET helix_new_fact_id = ?1, updated_at = ?2 WHERE id = ?3"
            ),
            params![helix_fact_id.as_str(), now, review_id],
        )?;
        Ok(changed > 0)
    }
}

/// Deterministic idempotency key: `case|session|segment|doc|new_fact|existing_fact|source`.
#[must_use]
pub fn build_dedupe_key(item: &ContradictionReviewItem) -> String {
    [
        item.case_id.as_str(),
        item.session_id.as_str(),
        item.segment_id.as_str(),
        item.doc_id.as_str(),
        item.new_fact_id.as_str(),
        item.existing_fact_id.as_str(),
        item.contradiction_source.as_str(),
    ]
    .join("|")
}

/// One selected row before enum and JSON decoding.
struct RawRow {
    id: i64,
    text: Vec<Option<String>>,
    segment_index: i64,
    nli_score: f64,
}

impl RawRow {
    /// Reads `id` followed by [`INSERT_COLUMNS`].
    fn read(row: &Row<'_>) -> rusqlite::Result<Self> {
        let mut text = Vec::with_capacity(INSERT_COLUMNS.len());
        let mut segment_index = 0;
        let mut nli_score = 0.0;
        for (offset, column) in INSERT_COLUMNS.iter().enumerate() {
            let index = offset + 1;
            match *column {
                "segment_index" => {
                    segment_index = row.get(index)?;
                    text.push(None);
                }
                "nli_score" => {
                    nli_score = row.get(index)?;
                    text.push(None);
                }
                _ => text.push(row.get(index)?),
            }
        }
        Ok(Self { id: row.get(0)?, text, segment_index, nli_score })
    }

    fn take(&mut self, column: &str) -> Option<String> {
        INSERT_COLUMNS
            .iter()
            .position(|name| *name == column)
            .and_then(|index| self.text[index].take())
    }

    fn into_item(mut self) -> Result<ContradictionReviewItem> {
        let new_payload = self.take("new_fact_payload");
        let existing_payload = self.take("existing_fact_payload");
        let nli_label = self.take("nli_label").unwrap_or_default();
        Ok(ContradictionReviewItem {
            id: Some(self.id),
            case_id: self.take("case_id").unwrap_or_default(),
            session_id: self.take("session_id").unwrap_or_default(),
            segment_id: self.take("segment_id").unwrap_or_default(),
            segment_index: self.segment_index,
            doc_id: DocId::new(self.take("doc_id").unwrap_or_default()),
            stage: self.take("stage").unwrap_or_default(),
            witness_name: self.take("witness_name"),
            source_stream: self.take("source_stream"),
            source_pipeline: self.take("source_pipeline"),
            segment_start_ts: self.take("segment_start_ts"),
            segment_end_ts: self.take("segment_end_ts"),
            new_fact_id: FactId::new(self.take("new_fact_id").unwrap_or_default()),
            helix_new_fact_id: self.take("helix_new_fact_id").map(FactId::new),
            existing_fact_id: FactId::new(self.take("existing_fact_id").unwrap_or_default()),
            new_fact_payload: parse_payload("new_fact_payload", new_payload.as_deref())?,
            existing_fact_payload: parse_payload(
                "existing_fact_payload",
                existing_payload.as_deref(),
            )?,
            nli_score: self.nli_score,
            nli_label: if nli_label.is_empty() {
                None
            } else {
                Some(parse_enum::<NliLabel>("nli_label", &nli_label)?)
            },
            classification: parse_enum::<ReviewClassification>(
                "classification",
                &self.take("classification").unwrap_or_default(),
            )?,
            contradiction_source: parse_enum::<ContradictionSource>(
                "contradiction_source",
                &self.take("contradiction_source").unwrap_or_default(),
            )?,
            status: parse_enum::<ReviewStatus>("status", &self.take("status").unwrap_or_default())?,
            created_at: self.take("created_at").unwrap_or_default(),
            updated_at: self.take("updated_at").unwrap_or_default(),
            resolved_by: self.take("resolved_by"),
            resolution_note: self.take("resolution_note"),
            dedupe_key: self.take("dedupe_key"),
        })
    }
}

fn parse_enum<T: DeserializeOwned>(field: &str, value: &str) -> Result<T> {
    serde_json::from_value(Value::String(value.to_owned())).map_err(|_| {
        StoreError::validation(
            field,
            value,
            format!("unknown {field} value '{value}' in review row"),
        )
    })
}

/// Empty or missing payloads read as `{}`, like Python.
fn parse_payload(field: &str, text: Option<&str>) -> Result<Metadata> {
    match text {
        None | Some("") => Ok(Metadata::new()),
        Some(text) => match serde_json::from_str::<Value>(text) {
            Ok(Value::Object(map)) => Ok(map),
            _ => Err(StoreError::validation(field, text, format!("{field} is not a JSON object"))),
        },
    }
}

/// Python `datetime.isoformat()` of a naive datetime: microseconds only when non-zero.
fn python_isoformat(value: NaiveDateTime) -> String {
    if value.nanosecond() / 1000 == 0 {
        value.format("%Y-%m-%dT%H:%M:%S").to_string()
    } else {
        value.format("%Y-%m-%dT%H:%M:%S%.6f").to_string()
    }
}

/// Python `json.dumps(value)` of a payload object.
fn python_json_dumps(value: &Metadata) -> String {
    ce_core::pyjson::to_python_json_ascii(value)
        .expect("a map of serde_json values always serialises")
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn isoformat_omits_zero_microseconds() {
        let base = chrono::NaiveDate::from_ymd_opt(2026, 1, 2)
            .and_then(|d| d.and_hms_micro_opt(3, 4, 5, 0))
            .expect("valid date");
        assert_eq!(python_isoformat(base), "2026-01-02T03:04:05");
        let micro = base.with_nanosecond(123_000).expect("valid nanos");
        assert_eq!(python_isoformat(micro), "2026-01-02T03:04:05.000123");
    }
}
