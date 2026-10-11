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
    /// Lowercases (if configured) and strips `name`, then removes each configured prefix and
    /// suffix in order (see [`ce_text::normalize_entity_name`]).
    pub fn normalize_entity_name(&self, name: &str) -> String {
        ce_text::normalize_entity_name(
            name,
            &self.entity_name_prefixes,
            &self.entity_name_suffixes,
            self.lowercase,
        )
    }

    /// Splits on non-alphanumeric characters, dropping stopwords and short tokens (see
    /// [`ce_text::token_set_by`]).
    pub fn tokenize(&self, text: &str) -> HashSet<String> {
        ce_text::token_set_by(
            text,
            |token| self.stopwords.contains(token),
            self.min_token_length,
            self.lowercase,
        )
    }
}
