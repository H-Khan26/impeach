//! Embedded assets and asset path resolution.
//!
//! Asset paths in config files (`impeach/prompts/...`,
//! `impeach/config/mappings/...`) resolve against a base directory (the working
//! directory by default) first, then against the assets compiled into the binary, with the
//! `impeach/` prefix mapped to the asset root.

use std::path::{Path, PathBuf};

use include_dir::{include_dir, Dir, File};

use crate::error::{ConfigError, Result};
use crate::pipeline::PipelineConfig;

static ASSETS: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/../../assets");

/// Prefix of Python package paths that maps to the embedded asset root.
pub const PACKAGE_PREFIX: &str = "impeach/";

/// Path of the default pipeline config (`processing_v17.yaml`).
pub const DEFAULT_CONFIG_PATH: &str = "impeach/config/processing/processing_v17.yaml";

/// Directory holding the mapping YAML files.
pub const MAPPINGS_DIR: &str = "impeach/config/mappings";

/// Where an asset was found.
#[derive(Debug, Clone)]
pub enum ResolvedAsset {
    /// A file on disk.
    Disk(PathBuf),
    /// A file compiled into the binary.
    Embedded(&'static File<'static>),
}

impl ResolvedAsset {
    /// Human-readable location for error messages (`embedded:` prefix for embedded files).
    pub fn display_path(&self) -> String {
        match self {
            Self::Disk(path) => path.display().to_string(),
            Self::Embedded(file) => format!("embedded:{}", file.path().display()),
        }
    }

    /// Whether the asset comes from the embedded assets.
    pub fn is_embedded(&self) -> bool {
        matches!(self, Self::Embedded(_))
    }

    /// Reads the asset as UTF-8 text.
    pub fn read_to_string(&self) -> Result<String> {
        match self {
            Self::Disk(path) => std::fs::read_to_string(path)
                .map_err(|source| ConfigError::Io { path: path.display().to_string(), source }),
            Self::Embedded(file) => file
                .contents_utf8()
                .map(str::to_owned)
                .ok_or_else(|| ConfigError::NotUtf8 { path: self.display_path() }),
        }
    }
}

/// The embedded asset tree (`config/`, `prompts/`).
pub fn embedded_assets() -> &'static Dir<'static> {
    &ASSETS
}

/// Looks up an embedded file, mapping the `impeach/` prefix to the asset root.
pub fn embedded_file(path: impl AsRef<Path>) -> Option<&'static File<'static>> {
    let normalized = path.as_ref().to_string_lossy().replace('\\', "/");
    let mut rel = normalized.as_str();
    while let Some(stripped) = rel.strip_prefix("./") {
        rel = stripped;
    }
    let rel = rel.strip_prefix(PACKAGE_PREFIX).unwrap_or(rel);
    if rel.is_empty() {
        return None;
    }
    ASSETS.get_file(rel)
}

/// Resolves an asset path: the working directory first, then the embedded assets.
pub fn resolve_asset(path: impl AsRef<Path>) -> Result<ResolvedAsset> {
    match std::env::current_dir() {
        Ok(cwd) => resolve_asset_in(&cwd, path),
        Err(_) => resolve_embedded(path.as_ref()),
    }
}

/// Resolves an asset path against `base` first, then the embedded assets.
///
/// Absolute paths are only looked up on disk.
pub fn resolve_asset_in(base: &Path, path: impl AsRef<Path>) -> Result<ResolvedAsset> {
    let path = path.as_ref();
    if path.is_absolute() {
        return if path.is_file() {
            Ok(ResolvedAsset::Disk(path.to_path_buf()))
        } else {
            Err(ConfigError::AssetNotFound { path: path.display().to_string() })
        };
    }
    let on_disk = base.join(path);
    if on_disk.is_file() {
        return Ok(ResolvedAsset::Disk(on_disk));
    }
    resolve_embedded(path)
}

fn resolve_embedded(path: &Path) -> Result<ResolvedAsset> {
    embedded_file(path)
        .map(ResolvedAsset::Embedded)
        .ok_or_else(|| ConfigError::AssetNotFound { path: path.display().to_string() })
}

/// Reads an asset resolved with [`resolve_asset_in`], returning its text and display path.
pub(crate) fn read_asset_in(base: Option<&Path>, path: &str) -> Result<(String, String)> {
    let asset = match base {
        Some(base) => resolve_asset_in(base, path)?,
        None => resolve_asset(path)?,
    };
    Ok((asset.read_to_string()?, asset.display_path()))
}

/// Loads and validates a pipeline config, resolving `path` with [`resolve_asset`].
pub fn load_config(path: impl AsRef<Path>) -> Result<PipelineConfig> {
    let asset = resolve_asset(path)?;
    PipelineConfig::from_yaml_str(&asset.read_to_string()?, &asset.display_path())
}

/// Loads and validates a pipeline config, resolving `path` against `base` first.
pub fn load_config_in(base: &Path, path: impl AsRef<Path>) -> Result<PipelineConfig> {
    let asset = resolve_asset_in(base, path)?;
    PipelineConfig::from_yaml_str(&asset.read_to_string()?, &asset.display_path())
}

/// Loads the default pipeline config (`processing_v17.yaml`).
pub fn load_default_config() -> Result<PipelineConfig> {
    load_config(DEFAULT_CONFIG_PATH)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_maps_to_asset_root() {
        let file = embedded_file("impeach/config/mappings/entity_types.yaml")
            .expect("entity_types.yaml is embedded");
        assert!(file.contents_utf8().is_some_and(|s| s.contains("entity_types:")));
        assert!(embedded_file("impeach/prompts/ed/disambiguation.json").is_some());
        assert!(embedded_file("config/processing/processing_v17.yaml").is_some());
        assert!(embedded_file("./impeach/prompts/processing/qwen_judge.json").is_some());
        assert!(embedded_file("impeach\\prompts\\processing\\qwen_judge.json").is_some());
    }

    #[test]
    fn unknown_assets_are_not_found() {
        assert!(embedded_file("impeach/nope.yaml").is_none());
        assert!(embedded_file("impeach/").is_none());
        assert!(embedded_file("").is_none());
        let dir = tempfile::tempdir().expect("tempdir");
        let err = resolve_asset_in(dir.path(), "impeach/missing.json").expect_err("missing asset");
        assert!(matches!(err, ConfigError::AssetNotFound { .. }));
    }

    #[test]
    fn base_directory_wins_over_embedded() {
        let dir = tempfile::tempdir().expect("tempdir");
        let rel = "impeach/prompts/ed/disambiguation.json";
        let on_disk = dir.path().join(rel);
        std::fs::create_dir_all(on_disk.parent().expect("parent")).expect("mkdir");
        std::fs::write(&on_disk, r#"{"system_template": "local", "user_template": ""}"#)
            .expect("write");

        let resolved = resolve_asset_in(dir.path(), rel).expect("resolves");
        assert!(!resolved.is_embedded());
        assert!(resolved.read_to_string().expect("read").contains("local"));

        let empty = tempfile::tempdir().expect("tempdir");
        let resolved = resolve_asset_in(empty.path(), rel).expect("resolves");
        assert!(resolved.is_embedded());
        assert!(resolved.display_path().starts_with("embedded:"));
    }

    #[test]
    fn absolute_paths_only_use_disk() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("x.yaml");
        std::fs::write(&file, "a: 1").expect("write");
        assert!(!resolve_asset_in(dir.path(), &file).expect("resolves").is_embedded());
        let missing = dir.path().join("impeach/config/mappings/roles.yaml");
        assert!(resolve_asset_in(dir.path(), missing).is_err());
    }
}
