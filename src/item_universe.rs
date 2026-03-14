// This file loads configs for the weapons, ships, planets, etc.
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::asteroids::AsteroidFieldData;
use crate::planets::PlanetData;
use crate::weapons::Weapon;

use serde::de::DeserializeOwned;
use serde_yaml::{Mapping, Value};
use std::path::Path;

#[derive(Resource, Deserialize, Serialize)]
pub struct ItemUniverse {
    pub weapons: HashMap<String, Weapon>,
    pub star_systems: HashMap<String, StarSystem>,
}

#[derive(Deserialize, Serialize)]
pub struct StarSystem {
    #[serde(default)]
    pub connections: Vec<String>,
    pub planets: HashMap<String, PlanetData>,
    pub astroid_fields: Vec<AsteroidFieldData>,
    pub ships: u8,
}

pub fn item_universe_plugin(app: &mut App) {
    // Load the ItemUniverse from disk
    let item_universe: ItemUniverse =
        parse_dir::<ItemUniverse>(Path::new("assets")).expect("failed to parse asset config");

    app.insert_resource::<ItemUniverse>(item_universe);
}

// Parsing code:

fn dir_to_yaml(dir: &Path) -> Option<Value> {
    let mut map = Mapping::new();

    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        let stem = path.file_stem()?.to_string_lossy().into_owned();

        if path.is_dir() {
            if let Some(v) = dir_to_yaml(&path) {
                map.insert(stem.into(), v);
            }
        } else if path
            .extension()
            .map_or(false, |e| e == "yaml" || e == "yml")
        {
            let text = std::fs::read_to_string(&path).ok()?;
            let val: Value = serde_yaml::from_str(&text).ok()?;
            map.insert(stem.into(), val);
        }
    }

    if map.is_empty() {
        None
    } else {
        Some(Value::Mapping(map))
    }
}

pub fn parse_dir<T: DeserializeOwned>(dir: &Path) -> Result<T, serde_yaml::Error> {
    let value = dir_to_yaml(dir).unwrap_or(Value::Mapping(Mapping::new()));
    serde_yaml::from_value(value)
}
