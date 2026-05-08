//! Filesystem helpers that transparently use embedded data when the `bundle`
//! feature is enabled, and fall back to disk reads otherwise.
//!
//! Bevy's `AssetServer` is handled separately by `bevy_embedded_assets`. This
//! module covers the handful of direct `std::fs::read_to_string("assets/...")`
//! callsites and the YAML directory walker in [`crate::item_universe`], which
//! don't go through the asset pipeline.
//!
//! Paths passed in are relative to the project root (e.g. `"assets/foo.ron"`).
//! With `bundle` on, only the `assets/` subtree is embedded — paths outside it
//! still hit the filesystem.

use serde_yaml::{Mapping, Value};

#[cfg(feature = "bundle")]
use include_dir::{Dir, include_dir};

#[cfg(feature = "bundle")]
static EMBEDDED_ASSETS: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/assets");

/// Read a UTF-8 text asset. `path` is relative to the project root, e.g.
/// `"assets/sprites/people/civilians.ron"`.
pub fn read_to_string(path: &str) -> std::io::Result<String> {
    #[cfg(feature = "bundle")]
    {
        if let Some(rel) = path.strip_prefix("assets/") {
            return EMBEDDED_ASSETS
                .get_file(rel)
                .and_then(|f| f.contents_utf8().map(|s| s.to_string()))
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("embedded asset not found or not utf-8: {path}"),
                    )
                });
        }
    }
    std::fs::read_to_string(path)
}

/// Recursively read every `*.yaml`/`*.yml` file under `dir`, returning a
/// nested `serde_yaml::Mapping` keyed by file/directory *stems*. Mirrors the
/// behavior of the disk walker in [`crate::item_universe::dir_to_yaml`] but
/// transparently sources from the embedded archive when `bundle` is enabled.
///
/// Returns `None` if the directory is empty / unreadable / not embedded.
pub fn dir_to_yaml(dir: &std::path::Path) -> Option<Value> {
    #[cfg(feature = "bundle")]
    {
        // Only paths rooted at `assets/` are embedded.
        if let Ok(rel) = dir.strip_prefix("assets") {
            let sub = if rel.as_os_str().is_empty() {
                Some(&EMBEDDED_ASSETS)
            } else {
                EMBEDDED_ASSETS.get_dir(rel)
            };
            return sub.and_then(embedded_dir_to_yaml);
        }
    }
    disk_dir_to_yaml(dir)
}

fn disk_dir_to_yaml(dir: &std::path::Path) -> Option<Value> {
    let mut map = Mapping::new();
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        let Some(stem_os) = path.file_stem() else {
            continue;
        };
        let stem = stem_os.to_string_lossy().into_owned();

        if path.is_dir() {
            if let Some(v) = disk_dir_to_yaml(&path) {
                map.insert(stem.into(), v);
            }
        } else if path.extension().is_some_and(|e| e == "yaml" || e == "yml") {
            let Ok(text) = std::fs::read_to_string(&path) else {
                eprintln!("[embedded_assets] WARNING: could not read {}", path.display());
                continue;
            };
            match serde_yaml::from_str::<Value>(&text) {
                Ok(val) => {
                    map.insert(stem.into(), val);
                }
                Err(e) => {
                    eprintln!("[embedded_assets] WARNING: could not parse {}: {e}", path.display());
                }
            }
        }
    }
    if map.is_empty() {
        None
    } else {
        Some(Value::Mapping(map))
    }
}

#[cfg(feature = "bundle")]
fn embedded_dir_to_yaml(dir: &Dir<'_>) -> Option<Value> {
    let mut map = Mapping::new();
    for entry in dir.entries() {
        match entry {
            include_dir::DirEntry::Dir(sub) => {
                let stem = sub
                    .path()
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned());
                if let (Some(stem), Some(v)) = (stem, embedded_dir_to_yaml(sub)) {
                    map.insert(stem.into(), v);
                }
            }
            include_dir::DirEntry::File(file) => {
                let path = file.path();
                if !path.extension().is_some_and(|e| e == "yaml" || e == "yml") {
                    continue;
                }
                let Some(stem) = path.file_stem().map(|s| s.to_string_lossy().into_owned())
                else {
                    continue;
                };
                let Some(text) = file.contents_utf8() else {
                    continue;
                };
                match serde_yaml::from_str::<Value>(text) {
                    Ok(val) => {
                        map.insert(stem.into(), val);
                    }
                    Err(e) => {
                        eprintln!(
                            "[embedded_assets] WARNING: could not parse {}: {e}",
                            path.display()
                        );
                    }
                }
            }
        }
    }
    if map.is_empty() {
        None
    } else {
        Some(Value::Mapping(map))
    }
}
