//! Errors of the graph and review stores.
//!
//! Python origin: `storage/helix/errors.py`. `HelixQueryError` (a failed query) maps to the
//! variant describing why it failed; `HelixDatabaseLostError` is [`StoreError::DatabaseLost`].

use thiserror::Error;

/// Result alias for store operations.
pub type Result<T, E = StoreError> = std::result::Result<T, E>;

/// Failure of a graph-store or review-store operation.
#[derive(Debug, Error)]
pub enum StoreError {
    /// A node referenced by id does not exist or has another label.
    #[error("{label} {id} not found")]
    NotFound {
        /// Node label, e.g. `Entity`.
        label: String,
        /// The id that was looked up.
        id: String,
    },
    /// An input failed validation before reaching the database.
    #[error("{message}")]
    Validation {
        /// Name of the offending field (Python property or parameter name).
        field: String,
        /// The offending value, rendered as text (empty when not applicable).
        value: String,
        /// Human-readable description.
        message: String,
    },
    /// The database could not be opened, prepared or reached, or a query failed.
    #[error("{0}")]
    Connection(String),
    /// The database handle is unusable (closed, or another process took over the embedded
    /// database). Fatal: callers must not swallow it.
    #[error("{0}")]
    DatabaseLost(String),
    /// A database query failed for a reason other than a missing node, bad input or a lost
    /// connection (Python `HelixQueryError`).
    #[error("Query '{query}' failed: {message}")]
    Query {
        /// Query name, e.g. `FindFactsByText`.
        query: String,
        /// Error text from the database.
        message: String,
    },
    /// A transaction conflict persisted after every retry.
    #[error("transaction conflict persisted after {attempts} attempt(s): {message}")]
    Conflict {
        /// Attempts made.
        attempts: u32,
        /// Last error text.
        message: String,
    },
    /// SQLite failure in the review store.
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// File-system failure (lock file, data directory).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl StoreError {
    /// Builds a [`StoreError::Validation`].
    pub fn validation(
        field: impl Into<String>,
        value: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::Validation { field: field.into(), value: value.into(), message: message.into() }
    }

    /// Builds a [`StoreError::NotFound`].
    pub fn not_found(label: impl Into<String>, id: impl Into<String>) -> Self {
        Self::NotFound { label: label.into(), id: id.into() }
    }

    /// Whether the error means the database handle is gone for good.
    #[must_use]
    pub fn is_fatal(&self) -> bool {
        matches!(self, Self::DatabaseLost(_))
    }
}
