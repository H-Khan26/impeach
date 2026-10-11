//! Rebuilds the crate when any embedded asset changes (`include_dir!` does not track files).

use std::path::Path;

fn main() {
    let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets");
    emit(&assets);
}

/// Emits `rerun-if-changed` for `dir` and everything below it, so added, removed and edited
/// files all trigger a rebuild.
fn emit(dir: &Path) {
    println!("cargo:rerun-if-changed={}", dir.display());
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            emit(&path);
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}
