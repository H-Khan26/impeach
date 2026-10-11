//! Regex detector and ED structured-key patterns (`detection_patterns.yaml`).

use std::sync::OnceLock;

use indexmap::IndexMap;
use regex::Regex;
use serde::Deserialize;

use super::{build_regex, RegexFlag};

/// Named regex patterns, compiled lazily on first use and cached.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DetectionPatterns {
    /// Mention detection patterns.
    pub patterns: IndexMap<String, PatternDef>,
    /// Order in which detection patterns run.
    pub detection_order: Vec<String>,
    /// Patterns used by ED rule resolvers and structured keys.
    pub ed_patterns: IndexMap<String, PatternDef>,
}

/// One named pattern.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PatternDef {
    /// Regex source.
    pub pattern: String,
    /// `IGNORECASE` or `NONE`.
    #[serde(default)]
    pub flags: Option<RegexFlag>,
    /// Entity type assigned to matches (detection patterns).
    #[serde(default)]
    pub entity_type: Option<String>,
    /// Capture group holding the mention (0 = whole match).
    #[serde(default)]
    pub capture_group: Option<usize>,
    /// Capture group index -> field name (ED patterns).
    #[serde(default)]
    pub capture_groups: IndexMap<usize, String>,
    /// Free-text description.
    #[serde(default)]
    pub description: Option<String>,
    #[serde(skip)]
    compiled: OnceLock<Option<Regex>>,
}

impl PartialEq for PatternDef {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern
            && self.flags == other.flags
            && self.entity_type == other.entity_type
            && self.capture_group == other.capture_group
            && self.capture_groups == other.capture_groups
            && self.description == other.description
    }
}

impl PatternDef {
    /// The compiled regex, or `None` (logged once) if the pattern does not compile.
    pub fn compiled(&self, name: &str) -> Option<&Regex> {
        self.compiled
            .get_or_init(|| match build_regex(&self.pattern, self.flags, false) {
                Ok(regex) => Some(regex),
                Err(err) => {
                    tracing::error!("Invalid regex pattern '{name}': {err}");
                    None
                }
            })
            .as_ref()
    }
}

impl DetectionPatterns {
    /// Pattern definition by name; detection patterns shadow ED patterns.
    pub fn get_pattern_def(&self, name: &str) -> Option<&PatternDef> {
        self.patterns.get(name).or_else(|| self.ed_patterns.get(name))
    }

    /// Compiled regex by name (detection patterns first, then ED patterns).
    pub fn get_compiled_pattern(&self, name: &str) -> Option<&Regex> {
        self.get_pattern_def(name).and_then(|def| def.compiled(name))
    }

    /// Entity type of a pattern.
    pub fn get_entity_type(&self, name: &str) -> Option<&str> {
        self.get_pattern_def(name).and_then(|def| def.entity_type.as_deref())
    }

    /// Pattern names in detection order.
    pub fn get_ordered_patterns(&self) -> &[String] {
        &self.detection_order
    }
}
