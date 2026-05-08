//! Bevy asset for the trained inference policy.
//!
//! Wraps the raw bytes of a burn `BinFileRecorder`-serialised checkpoint so
//! the file ships under `assets/` and is bundled into the binary via the
//! standard asset pipeline. The bytes are deserialised into the live
//! `InferenceNet` by the inference plugin.

use bevy::asset::{AssetLoader, LoadContext, io::Reader};
use bevy::prelude::*;

/// Default asset path for the runtime policy checkpoint.
pub const DEFAULT_POLICY_ASSET_PATH: &str = "ai/policy.bin";

/// Raw bytes of a serialised inference policy checkpoint.
#[derive(Asset, TypePath, Debug)]
pub struct PolicyAsset {
    pub bytes: Vec<u8>,
}

#[derive(Default, TypePath)]
struct PolicyAssetLoader;

#[derive(Debug)]
pub enum PolicyAssetError {
    Io(std::io::Error),
}

impl std::fmt::Display for PolicyAssetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error reading policy: {e}"),
        }
    }
}

impl std::error::Error for PolicyAssetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
        }
    }
}

impl From<std::io::Error> for PolicyAssetError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl AssetLoader for PolicyAssetLoader {
    type Asset = PolicyAsset;
    type Settings = ();
    type Error = PolicyAssetError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        Ok(PolicyAsset { bytes })
    }

    fn extensions(&self) -> &[&str] {
        &["policy.bin"]
    }
}

pub struct PolicyAssetPlugin;

impl Plugin for PolicyAssetPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<PolicyAsset>()
            .init_asset_loader::<PolicyAssetLoader>();
    }
}
