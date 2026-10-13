//! Role terms and defined-term patterns (`roles.yaml`), used by the DefinitionStage.

use std::path::Path;

use indexmap::IndexMap;
use serde::Deserialize;

use crate::assets::read_asset_in;
use crate::de::parse_yaml;
use crate::error::Result;

/// Path of the shipped `roles.yaml`.
pub const ROLES_PATH: &str = "impeach/config/mappings/roles.yaml";

/// Contents of `roles.yaml`. Matching logic lives in the definition extractor.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Roles {
    /// Category -> known role terms.
    pub role_seed_list: IndexMap<String, Vec<String>>,
    /// Definition extraction patterns (named groups `entity_name`, `defined_term`).
    pub definition_patterns: Vec<DefinitionPattern>,
    /// Uppercase variant -> canonical role term.
    pub role_normalization: IndexMap<String, String>,
    /// Terms that look like roles but are not parties.
    pub exclusions: Vec<String>,
    /// Minimum confidence for accepting a definition; defaults to 0.75.
    pub min_confidence: f64,
}

impl Default for Roles {
    fn default() -> Self {
        Self {
            role_seed_list: IndexMap::new(),
            definition_patterns: Vec::new(),
            role_normalization: IndexMap::new(),
            exclusions: Vec::new(),
            min_confidence: 0.75,
        }
    }
}

/// One entry of `definition_patterns:`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DefinitionPattern {
    /// Pattern name.
    pub name: String,
    /// Regex source.
    pub pattern: String,
    /// Confidence of a match; defaults to 0.8.
    #[serde(default = "default_pattern_confidence")]
    pub confidence: f64,
    /// Free-text description; defaults to empty.
    #[serde(default)]
    pub description: String,
}

fn default_pattern_confidence() -> f64 {
    0.8
}

impl Roles {
    /// Loads `roles.yaml` from a path resolved with [`crate::assets::resolve_asset`]
    /// (a custom `role_config_path` or [`ROLES_PATH`]).
    pub fn load(path: &str) -> Result<Self> {
        Self::load_in(None, path)
    }

    /// Loads `roles.yaml`, resolving `path` against `base` (or the working directory) first.
    pub fn load_in(base: Option<&Path>, path: &str) -> Result<Self> {
        let (text, origin) = read_asset_in(base, path)?;
        parse_yaml(&text, &origin)
    }
}
