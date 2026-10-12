//! `store reset`: delete the embedded Helix database of the configured namespace.

use std::fs::{File, OpenOptions, TryLockError};
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use ce_config::pipeline::{HelixConfig, HelixMode};

use crate::cli::StoreResetArgs;
use crate::output::println;

/// Data directory used when `helix.data_dir` is empty (as the store does).
const DEFAULT_DATA_DIR: &str = "data/helix";

/// Deletes `<data_dir>/<namespace>` and `<data_dir>/<namespace>.lock` for an embedded store.
///
/// # Errors
/// Config errors, a server-mode config, a lock held by another process, delete errors.
pub fn reset(args: &StoreResetArgs) -> anyhow::Result<()> {
    let config = match &args.config {
        Some(path) => ce_config::load_config(path)?,
        None => ce_config::load_default_config()?,
    };
    let helix = &config.helix;
    match helix.mode {
        HelixMode::Embedded => reset_embedded(helix),
        HelixMode::Memory => {
            println(&format!(
                "helix.mode is memory (namespace {}): nothing to reset; the in-memory store is \
                 discarded when the process exits",
                helix.namespace
            ));
            Ok(())
        }
        HelixMode::Server => anyhow::bail!(
            "store reset supports helix.mode embedded only; this config uses a Helix server ({}), \
             whose data the server manages",
            helix.url.as_deref().unwrap_or("no url")
        ),
    }
}

/// Paths of an embedded store: the database directory and its lock file.
fn embedded_paths(helix: &HelixConfig) -> (PathBuf, PathBuf) {
    let data_dir = if helix.data_dir.is_empty() {
        PathBuf::from(DEFAULT_DATA_DIR)
    } else {
        PathBuf::from(&helix.data_dir)
    };
    let db_dir = data_dir.join(&helix.namespace);
    let lock = data_dir.join(format!("{}.lock", helix.namespace));
    (db_dir, lock)
}

fn reset_embedded(helix: &HelixConfig) -> anyhow::Result<()> {
    let (db_dir, lock_path) = embedded_paths(helix);
    if !db_dir.exists() && !lock_path.exists() {
        println(&format!("Nothing to reset: {} does not exist", db_dir.display()));
        return Ok(());
    }
    // Holding the store's lock while deleting guarantees no pipeline has the database open.
    let lock = lock_path.exists().then(|| acquire_lock(&lock_path)).transpose()?;
    if db_dir.exists() {
        std::fs::remove_dir_all(&db_dir)
            .with_context(|| format!("failed to delete {}", db_dir.display()))?;
        println(&format!("Deleted {}", db_dir.display()));
    }
    if let Some(lock) = lock {
        drop(lock);
        std::fs::remove_file(&lock_path)
            .with_context(|| format!("failed to delete {}", lock_path.display()))?;
        println(&format!("Deleted {}", lock_path.display()));
    }
    Ok(())
}

fn acquire_lock(path: &Path) -> anyhow::Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(TryLockError::WouldBlock) => anyhow::bail!(
            "Helix database lock {} is held by another process; stop it before resetting",
            path.display()
        ),
        Err(TryLockError::Error(error)) => {
            Err(error).with_context(|| format!("failed to lock {}", path.display()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(data_dir: &str) -> HelixConfig {
        HelixConfig {
            mode: HelixMode::Embedded,
            data_dir: data_dir.to_owned(),
            url: None,
            namespace: "ns".to_owned(),
            api_key: String::new(),
            batch_size: 50,
            embedding_dim: 384,
        }
    }

    #[test]
    fn paths_follow_the_store_layout() {
        let (db, lock) = embedded_paths(&config("data/helix"));
        assert_eq!(db, Path::new("data/helix").join("ns"));
        assert_eq!(lock, Path::new("data/helix").join("ns.lock"));
        let (db, _) = embedded_paths(&config(""));
        assert_eq!(db, Path::new(DEFAULT_DATA_DIR).join("ns"));
    }

    #[test]
    fn reset_deletes_database_and_lock() {
        let dir = tempfile::tempdir().expect("tempdir");
        let data_dir = dir.path().to_string_lossy().into_owned();
        let helix = config(&data_dir);
        let (db, lock) = embedded_paths(&helix);
        std::fs::create_dir_all(db.join("sub")).expect("create db dir");
        std::fs::write(db.join("sub").join("data.mdb"), b"x").expect("write db file");
        std::fs::write(&lock, b"").expect("write lock");
        let other = dir.path().join("other");
        std::fs::create_dir_all(&other).expect("create other namespace");

        reset_embedded(&helix).expect("reset");
        assert!(!db.exists());
        assert!(!lock.exists());
        assert!(other.exists(), "other namespaces are kept");
    }

    #[test]
    fn reset_refuses_a_locked_store() {
        let dir = tempfile::tempdir().expect("tempdir");
        let helix = config(&dir.path().to_string_lossy());
        let (db, lock_path) = embedded_paths(&helix);
        std::fs::create_dir_all(&db).expect("create db dir");
        let held = File::create(&lock_path).expect("create lock");
        held.lock().expect("hold lock");

        let error = reset_embedded(&helix).expect_err("lock is held");
        assert!(error.to_string().contains("held by another process"), "{error}");
        assert!(db.exists());
    }
}
