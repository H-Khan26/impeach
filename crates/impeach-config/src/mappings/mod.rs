//! Mappings registry: typed access to the YAML vocabularies and rules in
//! `config/mappings/`.
//!
//! [`MappingsRegistry::load`] loads all ten files once per process and caches them. File
//! paths resolve with [`crate::assets::resolve_asset`] (working directory first, then the
//! embedded assets). Unlike the Python registry, a missing or malformed file is an error;
//! keys missing inside a file fall back to the Python defaults.

mod detection_patterns;
mod detector_config;
mod entity_types;
mod gliner2;
mod name_matching;
mod normalization;
mod quality_filters;
mod relationship_types;
mod roles;
mod taxonomy;

use std::path::Path;
use std::sync::OnceLock;

use regex::{Regex, RegexBuilder};
use serde::de::DeserializeOwned;
use serde::Deserialize;

pub use detection_patterns::{DetectionPatterns, PatternDef};
pub use detector_config::DetectorConfig;
pub use entity_types::{
    types_compatible_in_groups, EntityRouting, EntityTypeDefaults, EntityTypes,
};
pub use gliner2::{Gliner2Defaults, Gliner2Labels};
pub use name_matching::{NameMatching, NameMatchingSettings, NameMatchingThresholds};
pub use normalization::Normalization;
pub use quality_filters::{FilterPattern, PartialWordPattern, QualityFilters};
pub use relationship_types::{RelationshipDefaults, RelationshipTypes};
pub use roles::{DefinitionPattern, Roles, ROLES_PATH};
pub use taxonomy::{Taxonomy, TaxonomyDefaults};

use crate::assets::{embedded_file, read_asset_in, MAPPINGS_DIR};
use crate::de::parse_yaml;
use crate::error::{ConfigError, Result};

/// File names of the ten mapping files, in load order.
pub const MAPPING_FILES: [&str; 10] = [
    "fact_categorization.yaml",
    "entity_types.yaml",
    "relationship_types.yaml",
    "detection_patterns.yaml",
    "detector_config.yaml",
    "normalization.yaml",
    "quality_filters.yaml",
    "name_matching.yaml",
    "gliner2.yaml",
    "roles.yaml",
];

/// Regex flags accepted in mapping files.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
pub enum RegexFlag {
    /// Case-insensitive matching.
    #[serde(rename = "IGNORECASE")]
    IgnoreCase,
    /// No flags.
    #[default]
    #[serde(rename = "NONE")]
    None,
}

/// Compiles a mapping regex; `anchored` gives Python `re.match` (start-anchored) semantics.
pub(crate) fn build_regex(
    pattern: &str,
    flags: Option<RegexFlag>,
    anchored: bool,
) -> Result<Regex, regex::Error> {
    let source = if anchored { format!(r"\A(?:{pattern})") } else { pattern.to_owned() };
    RegexBuilder::new(&source).case_insensitive(flags == Some(RegexFlag::IgnoreCase)).build()
}

/// All mapping configurations.
#[derive(Debug, Clone, PartialEq)]
pub struct MappingsRegistry {
    /// `fact_categorization.yaml`.
    pub taxonomy: Taxonomy,
    /// `entity_types.yaml`.
    pub entity_types: EntityTypes,
    /// `relationship_types.yaml`.
    pub relationship_types: RelationshipTypes,
    /// `detection_patterns.yaml`.
    pub detection_patterns: DetectionPatterns,
    /// `detector_config.yaml`.
    pub detector_config: DetectorConfig,
    /// `normalization.yaml`.
    pub normalization: Normalization,
    /// `quality_filters.yaml`.
    pub quality_filters: QualityFilters,
    /// `name_matching.yaml`.
    pub name_matching: NameMatching,
    /// `gliner2.yaml`.
    pub gliner2: Gliner2Labels,
    /// `roles.yaml`.
    pub roles: Roles,
}

static GLOBAL: OnceLock<MappingsRegistry> = OnceLock::new();

impl MappingsRegistry {
    /// The process-wide registry, loaded on first call with [`MappingsRegistry::load_uncached`].
    pub fn load() -> Result<&'static Self> {
        if let Some(registry) = GLOBAL.get() {
            return Ok(registry);
        }
        let registry = Self::load_uncached()?;
        Ok(GLOBAL.get_or_init(|| registry))
    }

    /// Loads a fresh registry from `impeach/config/mappings/`, resolving each file
    /// against the working directory first, then the embedded assets.
    pub fn load_uncached() -> Result<Self> {
        Self::build(|file| read_asset_in(None, &format!("{MAPPINGS_DIR}/{file}")))
    }

    /// Loads a fresh registry from the embedded assets only.
    pub fn embedded() -> Result<Self> {
        Self::build(|file| {
            let path = format!("{MAPPINGS_DIR}/{file}");
            let asset = embedded_file(&path).ok_or(ConfigError::AssetNotFound { path })?;
            let text = asset
                .contents_utf8()
                .ok_or_else(|| ConfigError::NotUtf8 { path: asset.path().display().to_string() })?;
            Ok((text.to_owned(), format!("embedded:{}", asset.path().display())))
        })
    }

    /// Loads a fresh registry from the YAML files in `dir`.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        Self::build(|file| {
            let path = dir.join(file);
            let text = std::fs::read_to_string(&path)
                .map_err(|source| ConfigError::Io { path: path.display().to_string(), source })?;
            Ok((text, path.display().to_string()))
        })
    }

    fn build(read: impl Fn(&str) -> Result<(String, String)>) -> Result<Self> {
        fn load<T: DeserializeOwned>(
            read: &impl Fn(&str) -> Result<(String, String)>,
            file: &str,
        ) -> Result<(T, String)> {
            let (text, origin) = read(file)?;
            Ok((parse_yaml(&text, &origin)?, origin))
        }

        let (taxonomy, _) = load(&read, "fact_categorization.yaml")?;
        let (entity_types, _) = load(&read, "entity_types.yaml")?;
        let (relationship_types, _) = load(&read, "relationship_types.yaml")?;
        let (mut detection_patterns, patterns_origin): (DetectionPatterns, _) =
            load(&read, "detection_patterns.yaml")?;
        detection_patterns.compile(&patterns_origin)?;
        let (detector_config, _) = load(&read, "detector_config.yaml")?;
        let (normalization, _) = load(&read, "normalization.yaml")?;
        let (mut quality_filters, quality_origin): (QualityFilters, _) =
            load(&read, "quality_filters.yaml")?;
        quality_filters.compile(&quality_origin)?;
        let (name_matching, _) = load(&read, "name_matching.yaml")?;
        let (gliner2, _) = load(&read, "gliner2.yaml")?;
        let (roles, _) = load(&read, "roles.yaml")?;

        let registry = Self {
            taxonomy,
            entity_types,
            relationship_types,
            detection_patterns,
            detector_config,
            normalization,
            quality_filters,
            name_matching,
            gliner2,
            roles,
        };
        tracing::info!(
            "Loaded mappings: {} value types, {} property types, {} entity types, \
             {} relationship types, {} GLiNER2 entity label sets, {} GLiNER2 relation label sets",
            registry.taxonomy.value_types.len(),
            registry.taxonomy.property_types.len(),
            registry.entity_types.entity_types.len(),
            registry.relationship_types.relationship_types.len(),
            registry.gliner2.entity_label_sets.len(),
            registry.gliner2.relation_label_sets.len(),
        );
        Ok(registry)
    }
}
