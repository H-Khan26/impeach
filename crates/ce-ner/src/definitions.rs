//! Defined role terms (`roles.yaml` patterns) and role-term helpers.
//!
//! Python origin: `services/definitions/extractor.py` (`RoleConfig`, `DefinitionExtractor`).
//! [`DefinitionExtractor::extract_from_text`] finds explicit definitions such as
//! `Maria Lopez ("Plaintiff")` with the configured patterns and binds each to the nearest mention
//! of the defined name; [`DefinitionExtractor::extract_from_metadata`] turns `role` /
//! `role_in_doc` mention metadata into definitions. Spans are character offsets, like mention
//! spans.

use std::path::Path;

use ce_config::mappings::{Roles, ROLES_PATH};
use ce_config::ConfigError;
use ce_core::processing::DefinedTerm;
use ce_core::{EntityId, Mention, MentionId, Metadata, Span};
use ce_text::normalize::{py_split, py_strip};
use indexmap::{IndexMap, IndexSet};
use regex::Regex;
use serde_json::Value;

/// Confidence of a definition taken from mention metadata.
pub const METADATA_ROLE_CONFIDENCE: f64 = 0.90;

/// Extraction method of a definition taken from mention metadata.
pub const METADATA_ROLE_METHOD: &str = "metadata:role";

/// A compiled definition pattern with named groups `entity_name` and `defined_term`.
#[derive(Debug, Clone)]
pub struct DefinitionPattern {
    /// Pattern name; the extraction method is `pattern:<name>`.
    pub name: String,
    /// Compiled regex.
    pub pattern: Regex,
    /// Confidence of a match.
    pub confidence: f64,
    /// Free-text description.
    pub description: String,
}

/// Role vocabulary and definition patterns from `roles.yaml` (Python `RoleConfig`).
#[derive(Debug, Clone)]
pub struct RoleConfig {
    role_seed_terms: IndexSet<String>,
    /// Seed terms ordered longest first (stable), the order prefix checks try them in.
    prefix_terms: Vec<String>,
    definition_patterns: Vec<DefinitionPattern>,
    role_normalization: IndexMap<String, String>,
    exclusions: IndexSet<String>,
    min_confidence: f64,
}

impl RoleConfig {
    /// Builds the config from parsed `roles.yaml` contents. Seed terms and exclusions are
    /// uppercased; a pattern that does not compile is skipped with a warning.
    #[must_use]
    pub fn from_roles(roles: &Roles) -> Self {
        let role_seed_terms: IndexSet<String> =
            roles.role_seed_list.values().flatten().map(|term| term.to_uppercase()).collect();
        let mut prefix_terms: Vec<String> = role_seed_terms.iter().cloned().collect();
        prefix_terms.sort_by_key(|term| std::cmp::Reverse(term.chars().count()));

        let definition_patterns = roles
            .definition_patterns
            .iter()
            .filter_map(|config| match Regex::new(&config.pattern) {
                Ok(pattern) => Some(DefinitionPattern {
                    name: config.name.clone(),
                    pattern,
                    confidence: config.confidence,
                    description: config.description.clone(),
                }),
                Err(error) => {
                    tracing::warn!("Failed to compile pattern '{}': {error}", config.name);
                    None
                }
            })
            .collect::<Vec<_>>();

        let config = Self {
            role_seed_terms,
            prefix_terms,
            definition_patterns,
            role_normalization: roles.role_normalization.clone(),
            exclusions: roles.exclusions.iter().map(|term| term.to_uppercase()).collect(),
            min_confidence: roles.min_confidence,
        };
        tracing::info!(
            "Loaded role config: {} seed terms, {} patterns",
            config.role_seed_terms.len(),
            config.definition_patterns.len()
        );
        config
    }

    /// Loads `roles.yaml` from `path` (a stage's `role_config_path`), or the shipped
    /// [`ROLES_PATH`] when `None`; paths resolve against the working directory, then the
    /// embedded assets.
    ///
    /// # Errors
    /// The file is missing, is not valid YAML or has unknown fields.
    pub fn load(path: Option<&str>) -> Result<Self, ConfigError> {
        Self::load_in(None, path)
    }

    /// Like [`RoleConfig::load`], resolving `path` against `base` (or the working directory)
    /// first.
    ///
    /// # Errors
    /// The file is missing, is not valid YAML or has unknown fields.
    pub fn load_in(base: Option<&Path>, path: Option<&str>) -> Result<Self, ConfigError> {
        let roles = Roles::load_in(base, path.unwrap_or(ROLES_PATH))?;
        Ok(Self::from_roles(&roles))
    }

    /// Uppercased role terms of every seed-list category.
    #[must_use]
    pub fn role_seed_terms(&self) -> &IndexSet<String> {
        &self.role_seed_terms
    }

    /// Compiled definition patterns, in config order.
    #[must_use]
    pub fn definition_patterns(&self) -> &[DefinitionPattern] {
        &self.definition_patterns
    }

    /// Uppercase variant -> canonical role term.
    #[must_use]
    pub fn role_normalization(&self) -> &IndexMap<String, String> {
        &self.role_normalization
    }

    /// Uppercased excluded terms.
    #[must_use]
    pub fn exclusions(&self) -> &IndexSet<String> {
        &self.exclusions
    }

    /// Minimum confidence of a definition bound into a role map.
    #[must_use]
    pub fn min_confidence(&self) -> f64 {
        self.min_confidence
    }

    /// Canonical form of a role term: uppercased, stripped, then mapped through
    /// `role_normalization` (`"Plaintiffs"` -> `"PLAINTIFF"`).
    #[must_use]
    pub fn normalize_role_term(&self, term: &str) -> String {
        let upper = term.to_uppercase();
        let stripped = py_strip(&upper);
        self.role_normalization.get(stripped).map_or_else(|| stripped.to_owned(), Clone::clone)
    }

    /// Whether the normalised term is in the seed list.
    #[must_use]
    pub fn is_valid_role_term(&self, term: &str) -> bool {
        self.role_seed_terms.contains(&self.normalize_role_term(term))
    }

    /// Whether the uppercased term is in the exclusion list.
    #[must_use]
    pub fn is_excluded(&self, term: &str) -> bool {
        self.exclusions.contains(&term.to_uppercase())
    }

    /// Whether `name` is a role prefix followed by a name (`"Defendant Reed"`): after a seed term
    /// and a space, at least one word and more than two characters remain.
    #[must_use]
    pub fn is_hybrid_role_name(&self, name: &str) -> bool {
        if name.is_empty() {
            return false;
        }
        let normalized = py_strip(name).to_uppercase();
        self.prefix_terms.iter().any(|role| {
            strip_upper_prefix(&normalized, role).is_some_and(|rest| {
                let remainder = py_strip(rest);
                py_split(remainder).next().is_some() && remainder.chars().count() > 2
            })
        })
    }

    /// The name part of a hybrid role name (`"Defendant Reed"` -> `"Reed"`), or `None` when no
    /// seed term followed by a space prefixes `name` with more than two characters after it.
    /// Longer seed terms are tried first.
    #[must_use]
    pub fn strip_role_prefix<'a>(&self, name: &'a str) -> Option<&'a str> {
        if name.is_empty() {
            return None;
        }
        let upper = name.to_uppercase();
        self.prefix_terms.iter().find_map(|role| {
            strip_upper_prefix(&upper, role)?;
            // The prefix length is counted in characters of the uppercased name and cut from
            // the original name.
            let skip = role.chars().count() + 1;
            let start = name.char_indices().nth(skip).map_or(name.len(), |(index, _)| index);
            let remainder = py_strip(&name[start..]);
            (remainder.chars().count() > 2).then_some(remainder)
        })
    }
}

/// The rest of `upper` after `role` and a space, if `upper` starts with them.
fn strip_upper_prefix<'a>(upper: &'a str, role: &str) -> Option<&'a str> {
    upper.strip_prefix(role)?.strip_prefix(' ')
}

/// Extracts defined role terms from document text and mention metadata (Python
/// `DefinitionExtractor`).
#[derive(Debug, Clone)]
pub struct DefinitionExtractor {
    config: RoleConfig,
}

impl DefinitionExtractor {
    /// An extractor over a role config.
    #[must_use]
    pub fn new(config: RoleConfig) -> Self {
        Self { config }
    }

    /// The role config.
    #[must_use]
    pub fn config(&self) -> &RoleConfig {
        &self.config
    }

    /// Definitions found by the patterns, in pattern order then match order. A match is kept
    /// when its `defined_term` is not excluded, is a seed role term, and its `entity_name`
    /// matches a mention name (exactly, else by substring either way, case-insensitive); the
    /// mention starting closest to the match start wins, the earliest indexed on ties.
    #[must_use]
    pub fn extract_from_text(
        &self,
        text: &str,
        mentions: &[Mention],
        doc_id: &str,
    ) -> Vec<DefinedTerm> {
        let mention_by_name = build_mention_name_index(mentions);
        let mut defined_terms = Vec::new();

        for pattern_def in &self.config.definition_patterns {
            let mut cursor = CharCursor::new(text);
            for captures in pattern_def.pattern.captures_iter(text) {
                let whole = captures.get(0).expect("group 0 is always present in a match");
                let start = cursor.char_offset(whole.start());
                let end = cursor.char_offset(whole.end());

                let (Some(entity_group), Some(term_group)) =
                    (captures.name("entity_name"), captures.name("defined_term"))
                else {
                    tracing::warn!(
                        "Error processing match at offset {start}: pattern '{}' did not capture \
                         both entity_name and defined_term",
                        pattern_def.name
                    );
                    continue;
                };
                let entity_name = py_strip(entity_group.as_str());
                let defined_term = py_strip(term_group.as_str());

                let normalized_term = self.config.normalize_role_term(defined_term);

                if self.config.is_excluded(defined_term) {
                    tracing::debug!("Skipping excluded term '{defined_term}' at {start}");
                    continue;
                }
                if !self.config.is_valid_role_term(defined_term) {
                    tracing::debug!("Skipping non-role term '{defined_term}' at {start}");
                    continue;
                }

                if let Some(mention_id) =
                    find_mention_for_name(entity_name, &mention_by_name, start)
                {
                    tracing::debug!(
                        "Extracted definition: {entity_name} -> {normalized_term} (pattern: {})",
                        pattern_def.name
                    );
                    defined_terms.push(DefinedTerm {
                        entity_mention_id: mention_id.clone(),
                        defined_term: normalized_term,
                        original_term: defined_term.to_owned(),
                        span: Span::new(start, end),
                        confidence: pattern_def.confidence,
                        extraction_method: format!("pattern:{}", pattern_def.name),
                    });
                } else {
                    tracing::debug!(
                        "Could not match entity name '{entity_name}' to any mention at offset \
                         {start}"
                    );
                }
            }
        }

        tracing::trace!(doc_id, count = defined_terms.len(), "pattern definitions");
        defined_terms
    }

    /// Definitions from mention metadata: the first truthy of `role` / `role_in_doc` (Python
    /// truthiness), when it is a non-excluded seed role term. Uses the mention's span,
    /// confidence [`METADATA_ROLE_CONFIDENCE`] and method [`METADATA_ROLE_METHOD`]. A truthy
    /// non-string role is skipped with a warning.
    #[must_use]
    pub fn extract_from_metadata(&self, mentions: &[Mention]) -> Vec<DefinedTerm> {
        let mut defined_terms = Vec::new();
        for mention in mentions {
            let Some(role) = metadata_role(&mention.metadata) else {
                continue;
            };
            let Some(role) = role.as_str() else {
                tracing::warn!(
                    "Skipping non-string role {role} from mention {} metadata",
                    mention.id
                );
                continue;
            };

            let normalized_term = self.config.normalize_role_term(role);
            if self.config.is_excluded(role) {
                continue;
            }
            if !self.config.is_valid_role_term(role) {
                tracing::debug!(
                    "Skipping non-seed role '{role}' from mention {} metadata",
                    mention.id
                );
                continue;
            }

            tracing::debug!("Extracted role from metadata: {} -> {normalized_term}", mention.name);
            defined_terms.push(DefinedTerm {
                entity_mention_id: mention.id.clone(),
                defined_term: normalized_term,
                original_term: role.to_owned(),
                span: mention.span,
                confidence: METADATA_ROLE_CONFIDENCE,
                extraction_method: METADATA_ROLE_METHOD.to_owned(),
            });
        }
        defined_terms
    }

    /// Role term -> entity id from definitions and resolved mentions. Definitions below
    /// `min_confidence` or whose mention has no (non-empty) entity id are skipped; the first
    /// binding of a role wins and later bindings to another entity are logged as conflicts.
    #[must_use]
    pub fn build_role_map(
        &self,
        defined_terms: &[DefinedTerm],
        ed_results: &IndexMap<MentionId, EntityId>,
    ) -> IndexMap<String, EntityId> {
        let mut role_map: IndexMap<String, EntityId> = IndexMap::new();
        let mut role_conflicts: IndexMap<&str, Vec<EntityId>> = IndexMap::new();

        for defined_term in defined_terms {
            if defined_term.confidence < self.config.min_confidence {
                continue;
            }
            let Some(entity_id) = ed_results
                .get(&defined_term.entity_mention_id)
                .filter(|id| !id.as_str().is_empty())
            else {
                tracing::debug!(
                    "No entity ID for mention {}, skipping role binding for {}",
                    defined_term.entity_mention_id,
                    defined_term.defined_term
                );
                continue;
            };

            if let Some(existing_id) = role_map.get(&defined_term.defined_term) {
                if existing_id != entity_id {
                    tracing::warn!(
                        "Role conflict for '{}': multiple entities ({existing_id}, {entity_id})",
                        defined_term.defined_term
                    );
                    role_conflicts
                        .entry(defined_term.defined_term.as_str())
                        .or_insert_with(|| vec![existing_id.clone()])
                        .push(entity_id.clone());
                    continue;
                }
            }

            tracing::info!(
                "Bound role '{}' -> entity {entity_id} (confidence: {:.2}, method: {})",
                defined_term.defined_term,
                defined_term.confidence,
                defined_term.extraction_method
            );
            role_map.insert(defined_term.defined_term.clone(), entity_id.clone());
        }

        if !role_conflicts.is_empty() {
            tracing::warn!(
                "Found {} role conflicts in document: {:?}",
                role_conflicts.len(),
                role_conflicts.keys().collect::<Vec<_>>()
            );
        }
        role_map
    }
}

/// Mentions by stripped lowercase name, in first-seen order.
fn build_mention_name_index(mentions: &[Mention]) -> IndexMap<String, Vec<&Mention>> {
    let mut index: IndexMap<String, Vec<&Mention>> = IndexMap::new();
    for mention in mentions {
        index.entry(py_strip(&mention.name).to_lowercase()).or_default().push(mention);
    }
    index
}

/// Id of the mention matching `entity_name` (exact key, else every key contained in the name or
/// containing it, in index order) whose start is closest to `definition_start`; the first such
/// candidate on ties.
fn find_mention_for_name<'a>(
    entity_name: &str,
    mention_by_name: &IndexMap<String, Vec<&'a Mention>>,
    definition_start: usize,
) -> Option<&'a MentionId> {
    let normalized_name = py_strip(entity_name).to_lowercase();
    let distance = |mention: &&'a Mention| mention.span.start.abs_diff(definition_start);
    let chosen = match mention_by_name.get(&normalized_name) {
        Some(exact) => exact.iter().copied().min_by_key(distance),
        None => mention_by_name
            .iter()
            .filter(|(key, _)| {
                normalized_name.contains(key.as_str()) || key.contains(&normalized_name)
            })
            .flat_map(|(_, mentions)| mentions.iter().copied())
            .min_by_key(distance),
    };
    chosen.map(|mention| &mention.id)
}

/// The metadata role: `role` when truthy, else `role_in_doc` when truthy.
fn metadata_role(metadata: &Metadata) -> Option<&Value> {
    metadata
        .get("role")
        .filter(|value| is_truthy(value))
        .or_else(|| metadata.get("role_in_doc").filter(|value| is_truthy(value)))
}

/// Python truthiness of a JSON value.
fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|n| n != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// Converts non-decreasing byte offsets of one text into character offsets incrementally.
struct CharCursor<'a> {
    text: &'a str,
    byte: usize,
    chars: usize,
}

impl<'a> CharCursor<'a> {
    fn new(text: &'a str) -> Self {
        Self { text, byte: 0, chars: 0 }
    }

    fn char_offset(&mut self, byte: usize) -> usize {
        if byte < self.byte {
            self.byte = 0;
            self.chars = 0;
        }
        self.chars += self.text[self.byte..byte].chars().count();
        self.byte = byte;
        self.chars
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_cursor_counts_characters() {
        let text = "\u{e9} (\"Plaintiff\") \u{fc}";
        let mut cursor = CharCursor::new(text);
        assert_eq!(cursor.char_offset(0), 0);
        assert_eq!(cursor.char_offset(2), 1);
        assert_eq!(cursor.char_offset(text.len()), text.chars().count());
        assert_eq!(cursor.char_offset(2), 1);
    }

    #[test]
    fn truthiness_follows_python() {
        for falsy in [Value::Null, Value::Bool(false), 0.into(), 0.0.into(), "".into()] {
            assert!(!is_truthy(&falsy), "{falsy}");
        }
        assert!(!is_truthy(&serde_json::json!([])));
        assert!(!is_truthy(&serde_json::json!({})));
        for truthy in [Value::Bool(true), 1.into(), (-0.5).into(), "x".into()] {
            assert!(is_truthy(&truthy), "{truthy}");
        }
    }

    #[test]
    fn metadata_role_falls_back_to_role_in_doc() {
        let mut metadata = Metadata::new();
        metadata.insert("role".into(), "".into());
        metadata.insert("role_in_doc".into(), "Tenant".into());
        assert_eq!(metadata_role(&metadata), Some(&Value::from("Tenant")));
        metadata.insert("role".into(), "Landlord".into());
        assert_eq!(metadata_role(&metadata), Some(&Value::from("Landlord")));
        assert_eq!(metadata_role(&Metadata::new()), None);
    }
}
