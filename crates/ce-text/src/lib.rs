//! Pure text utilities shared by the pipeline crates.
//!
//! Every helper names its Python origin (`file:function`) in its doc comment so later phases
//! can find the Rust equivalent of a Python call site.
//!
//! - [`normalize`]: Python-compatible whitespace splitting/stripping, whitespace collapsing and
//!   entity-name prefix/suffix stripping.
//! - [`tokenize`]: alphanumeric token sets with stopword and length filters, and Jaccard overlap.
//! - [`fuzzy`]: `rapidfuzz.fuzz` scorers (`ratio`, `partial_ratio`, `token_sort_ratio`,
//!   `token_set_ratio`) on a 0-100 scale, and `difflib.SequenceMatcher.ratio` on a 0-1 scale.
//! - [`values`]: number / money / date / year / clock-time extraction and the value-agreement
//!   guard used by the evaluator.
//! - [`side`]: party-side normalization.
//! - [`retry`]: synchronous retry with exponential backoff, injectable sleep and jitter.
//!
//! Character semantics: all lengths and positions are counted in Unicode scalar values, which is
//! what Python `str` indexing uses. Python's `str.isalnum` / `\w` are approximated by
//! [`char::is_alphanumeric`] (Rust additionally treats combining vowel signs of some scripts as
//! alphabetic); whitespace follows Python `str.isspace` exactly (see
//! [`normalize::is_py_whitespace`]).

pub mod fuzzy;
pub mod normalize;
pub mod retry;
pub mod side;
pub mod tokenize;
pub mod values;

pub use fuzzy::{partial_ratio, ratio, sequence_matcher_ratio, token_set_ratio, token_sort_ratio};
pub use normalize::{normalize_entity_name, normalize_whitespace, normalize_whitespace_lower};
pub use retry::{RetryPolicy, Sleeper};
pub use side::normalize_side;
pub use tokenize::{jaccard, keyword_overlap, token_set};
pub use values::{extract_values, values_compatible, Value};
