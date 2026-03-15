// This file loads configs for the weapons, ships, planets, etc.
use bevy::math::Vec2;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::planets::PlanetData;
use crate::weapons::Weapon;
use crate::{asteroids::AsteroidFieldData, ship::ShipData};

use serde::de::DeserializeOwned;
use serde_yaml::{Mapping, Value};
use std::path::Path;

#[derive(Deserialize, Serialize)]
pub struct CommodityData {
    pub color: [f32; 3],
}

#[derive(Resource, Deserialize, Serialize)]
pub struct ItemUniverse {
    pub weapons: HashMap<String, Weapon>,
    pub ships: HashMap<String, ShipData>,
    pub star_systems: HashMap<String, StarSystem>,
    pub outfitter_items: HashMap<String, OutfitterItem>,
    #[serde(default)]
    pub commodities: HashMap<String, CommodityData>,
    /// Average price of each commodity across all planets in all star systems.
    #[serde(skip)]
    pub global_average_price: HashMap<String, f64>,
    /// system → commodity → planet name with the highest price (best place to sell).
    #[serde(skip)]
    pub system_commodity_best_planet_to_sell: HashMap<String, HashMap<String, String>>,
    /// system → planet name → commodity with the biggest discount vs. system average (best to buy).
    #[serde(skip)]
    pub system_planet_best_commodity_to_buy: HashMap<String, HashMap<String, String>>,
}

impl ItemUniverse {
    fn compute_global_averages(&mut self) {
        let mut sums: HashMap<String, (f64, u32)> = HashMap::new();
        for system in self.star_systems.values() {
            for planet in system.planets.values() {
                for (commodity, &price) in &planet.commodities {
                    let entry = sums.entry(commodity.clone()).or_insert((0.0, 0));
                    entry.0 += price as f64;
                    entry.1 += 1;
                }
            }
        }
        self.global_average_price = sums
            .into_iter()
            .map(|(k, (sum, count))| (k, sum / count as f64))
            .collect();
    }

    fn compute_trade_maps(&mut self) {
        for (system_name, system) in &self.star_systems {
            // ── system_commodity_best_planet_to_sell ─────────────────────────
            // For each commodity, which planet in this system pays the most?
            let mut best_sell: HashMap<String, (String, i128)> = HashMap::new();
            for (planet_name, planet) in &system.planets {
                for (commodity, &price) in &planet.commodities {
                    let entry = best_sell
                        .entry(commodity.clone())
                        .or_insert_with(|| (planet_name.clone(), i128::MIN));
                    if price > entry.1 {
                        *entry = (planet_name.clone(), price);
                    }
                }
            }
            self.system_commodity_best_planet_to_sell.insert(
                system_name.clone(),
                best_sell.into_iter().map(|(k, (p, _))| (k, p)).collect(),
            );

            // ── system_planet_best_commodity_to_buy ──────────────────────────
            // For each planet, which commodity has the biggest discount vs. the
            // average price of that commodity across this system?
            let mut sys_sums: HashMap<String, (i128, u32)> = HashMap::new();
            for planet in system.planets.values() {
                for (commodity, &price) in &planet.commodities {
                    let e = sys_sums.entry(commodity.clone()).or_insert((0, 0));
                    e.0 += price;
                    e.1 += 1;
                }
            }
            let sys_avg: HashMap<String, f64> = sys_sums
                .into_iter()
                .map(|(k, (sum, n))| (k, sum as f64 / n as f64))
                .collect();

            let mut best_buy: HashMap<String, String> = HashMap::new();
            for (planet_name, planet) in &system.planets {
                let best = planet
                    .commodities
                    .iter()
                    .filter_map(|(commodity, &price)| {
                        sys_avg
                            .get(commodity)
                            .map(|&avg| (commodity.clone(), avg - price as f64))
                    })
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                if let Some((commodity, _)) = best {
                    best_buy.insert(planet_name.clone(), commodity);
                }
            }
            self.system_planet_best_commodity_to_buy
                .insert(system_name.clone(), best_buy);
        }
    }
}

#[derive(Deserialize, Serialize)]
pub enum OutfitterItem {
    Weapon {
        price: i128,
        space: u16,
        weapon_type: String,
    },
}

impl OutfitterItem {
    pub fn price(&self) -> i128 {
        match self {
            OutfitterItem::Weapon { price, .. } => *price,
        }
    }
    pub fn space(&self) -> u16 {
        match self {
            OutfitterItem::Weapon { space, .. } => *space,
        }
    }
}

#[derive(Deserialize, Serialize)]
pub struct StarSystem {
    #[serde(default)]
    pub map_position: Vec2,
    #[serde(default)]
    pub connections: Vec<String>,
    pub planets: HashMap<String, PlanetData>,
    pub astroid_fields: Vec<AsteroidFieldData>,
    pub ships: HashMap<String, u16>,
}

pub fn item_universe_plugin(app: &mut App) {
    // Load the ItemUniverse from disk
    let mut item_universe: ItemUniverse =
        parse_dir::<ItemUniverse>(Path::new("assets")).expect("failed to parse asset config");

    item_universe.compute_global_averages();
    item_universe.compute_trade_maps();
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
