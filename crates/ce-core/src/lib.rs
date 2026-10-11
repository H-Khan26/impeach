//! Data records shared by every contradiction-engine crate.
//!
//! Field names serialise exactly as the Python prototype's records so that JSONL payloads,
//! facts and contradiction reports diff cleanly across implementations.

pub mod vocab;
