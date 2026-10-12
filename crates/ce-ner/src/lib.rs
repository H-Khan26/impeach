//! Mention detection and defined-term extraction.
//!
//! - [`detectors`]: regex rules (`detection_patterns.yaml`), legal citations, GLiNER2 adapter.
//! - [`merging`]: overlap resolution and source accumulation by detector priority.
//! - [`quality`]: `quality_filters.yaml` rules and split-mention healing.
//! - [`definitions`]: defined role terms such as `("Plaintiff")` from `roles.yaml` patterns.

pub mod definitions;
pub mod detectors;
pub mod error;
pub mod merging;
pub mod quality;

pub use error::{NerError, Result};
