//! Name normalisation and tokenisation rules (`normalization.yaml`).

use std::collections::HashSet;

use indexmap::IndexSet;
use serde::Deserialize;

/// Prefix/suffix stripping, stopwords and alias exclusions.
///
/// Stopwords and aliases are plain strings: YAML 1.1 words such as `on`, `no` or `yes`
/// load as strings whether quoted or not, because `yaml_serde` follows YAML 1.2.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Normalization {
    /// Prefixes stripped in order (`"the "`, ...).
    pub entity_name_prefixes: Vec<String>,
    /// Suffixes stripped in order (`" inc"`, ...).
    pub entity_name_suffixes: Vec<String>,
    /// Tokens dropped by [`Normalization::tokenize`].
    pub stopwords: IndexSet<String>,
    /// Aliases (mostly pronouns) kept out of LLM entity prompts.
    pub alias_exclusion: IndexSet<String>,
    /// Minimum token length in characters; defaults to 2.
    pub min_token_length: usize,
    /// Lowercase before normalising; defaults to true.
    pub lowercase: bool,
}

impl Default for Normalization {
    fn default() -> Self {
        Self {
            entity_name_prefixes: Vec::new(),
            entity_name_suffixes: Vec::new(),
            stopwords: IndexSet::new(),
            alias_exclusion: IndexSet::new(),
            min_token_length: 2,
            lowercase: true,
        }
    }
}

impl Normalization {
    /// Lowercases and trims `name`, then strips each configured prefix and suffix in order.
    pub fn normalize_entity_name(&self, name: &str) -> String {
        if name.is_empty() {
            return String::new();
        }
        let mut normalized = if self.lowercase {
            name.to_lowercase().trim().to_owned()
        } else {
            name.trim().to_owned()
        };
        for prefix in &self.entity_name_prefixes {
            if let Some(rest) = normalized.strip_prefix(prefix.as_str()) {
                normalized = rest.to_owned();
            }
        }
        for suffix in &self.entity_name_suffixes {
            if let Some(rest) = normalized.strip_suffix(suffix.as_str()) {
                normalized = rest.to_owned();
            }
        }
        normalized.trim().to_owned()
    }

    /// Splits on non-alphanumeric characters, dropping stopwords and short tokens.
    pub fn tokenize(&self, text: &str) -> HashSet<String> {
        let processed = if self.lowercase { text.to_lowercase() } else { text.to_owned() };
        processed
            .split(|c: char| !c.is_alphanumeric())
            .filter(|token| {
                !token.is_empty()
                    && !self.stopwords.contains(*token)
                    && token.chars().count() >= self.min_token_length
            })
            .map(str::to_owned)
            .collect()
    }
}
