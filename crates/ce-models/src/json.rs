//! Lenient JSON extraction from LLM output.

use crate::error::ModelError;

/// Parses the JSON value contained in an LLM completion.
pub fn extract_json(text: &str) -> Result<serde_json::Value, ModelError> {
    serde_json::from_str(text.trim())
        .map_err(|e| ModelError::Protocol(format!("LLM output is not valid JSON: {e}")))
}
