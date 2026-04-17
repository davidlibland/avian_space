// This file loads configs for the weapons, ships, planets, etc.
use bevy::math::Vec2;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::missions::{MissionDef, MissionTemplate};
use crate::planets::PlanetData;
use crate::weapons::{Weapon, WeaponSystems};
use crate::{asteroids::AsteroidFieldData, ship::ShipData};

use serde::de::DeserializeOwned;
use serde_yaml::{Mapping, Value};
use std::path::Path;

#[derive(Deserialize, Serialize)]
pub struct CommodityData {
    pub color: [f32; 3],
    #[serde(default)]
    pub display_name: String,
}

#[derive(Resource, Deserialize, Serialize)]
pub struct ItemUniverse {
    pub weapons: HashMap<String, Weapon>,
    pub ships: HashMap<String, ShipData>,
    pub star_systems: HashMap<String, StarSystem>,
    pub outfitter_items: HashMap<String, OutfitterItem>,
    // A map from my faction, to which factions engage me
    pub enemies: HashMap<String, Vec<String>>,
    // The starting ship for the player:
    pub starting_ship: String,
    // The starting system for the player:
    pub starting_system: String,
    #[serde(default)]
    pub commodities: HashMap<String, CommodityData>,
    #[serde(default)]
    pub missions: HashMap<String, MissionDef>,
    #[serde(default)]
    pub mission_templates: HashMap<String, MissionTemplate>,
    /// Average price of each commodity across all planets in all star systems.
    #[serde(skip)]
    pub global_average_price: HashMap<String, f64>,
    /// system → commodity → planet name with the highest price (best place to sell).
    #[serde(skip)]
    pub system_commodity_best_planet_to_sell: HashMap<String, HashMap<String, String>>,
    /// system → planet name → commodity with the biggest discount vs. system average (best to buy).
    #[serde(skip)]
    pub system_planet_best_commodity_to_buy: HashMap<String, HashMap<String, String>>,
    /// system → planet name → best commodity margin (min over commodities of
    /// `price_here - max_price_elsewhere`). Negative = profitable to buy here and sell elsewhere.
    #[serde(skip)]
    pub planet_best_margin: HashMap<String, HashMap<String, f64>>,
    /// planet name → set of ship types that can replenish ammo at this planet.
    #[serde(skip)]
    pub planet_has_ammo_for: HashMap<String, std::collections::HashSet<String>>,
    /// Expected value of shattering an asteroid in each field, keyed by
    /// `AsteroidField` entity index within the system (system_name → vec index → value).
    /// Computed as `sum(weight_i * avg_price_i) / total_weight * E[qty]`.
    #[serde(skip)]
    pub asteroid_field_expected_value: HashMap<String, Vec<f64>>,
    /// Per-ship-type credit scale: cargo_space * avg_commodity_value + max ammo refill cost.
    /// Used to normalise credit observations and cargo-sale rewards.
    #[serde(skip)]
    pub ship_credit_scale: HashMap<String, f32>,
}

impl ItemUniverse {
    /// Fill in `display_name` fields with a title-cased fallback derived from
    /// the HashMap key whenever the YAML didn't specify one. Runs once at load
    /// so all downstream UI code can read `.display_name` directly.
    fn fill_display_names(&mut self) {
        use crate::utils::title_case;
        fn fill(field: &mut String, key: &str) {
            if field.is_empty() {
                *field = title_case(key);
            }
        }
        for (k, v) in &mut self.ships {
            fill(&mut v.display_name, k);
        }
        for (k, v) in &mut self.weapons {
            fill(&mut v.display_name, k);
        }
        for (k, v) in &mut self.commodities {
            fill(&mut v.display_name, k);
        }
        for (k, v) in &mut self.outfitter_items {
            fill(v.display_name_mut(), k);
        }
        for (sys_key, sys) in &mut self.star_systems {
            fill(&mut sys.display_name, sys_key);
            for (planet_key, planet) in &mut sys.planets {
                fill(&mut planet.display_name, planet_key);
            }
        }
    }

    pub fn validate(&self) {
        for (ship_name, ship_data) in &self.ships {
            let consumed: i32 = WeaponSystems::build(&ship_data.base_weapons, self)
                .iter_all()
                .map(|(_, s)| s.space_consumed())
                .sum();
            if consumed > ship_data.item_space as i32 {
                warn!(
                    "Ship \"{ship_name}\" base weapons consume {consumed} item space \
                     but ship only has {} — check assets/ships.yaml",
                    ship_data.item_space
                );
            }
        }
        for (sys_name, system) in &self.star_systems {
            for (planet_name, planet) in &system.planets {
                if planet.uncolonized && !planet.commodities.is_empty() {
                    warn!(
                        "Planet \"{planet_name}\" in \"{sys_name}\" is uncolonized \
                         but has commodities — these will be inaccessible"
                    );
                }
                if planet.uncolonized && !planet.outfitter.is_empty() {
                    warn!(
                        "Planet \"{planet_name}\" in \"{sys_name}\" is uncolonized \
                         but has an outfitter — this will be inaccessible"
                    );
                }
                if planet.uncolonized && !planet.shipyard.is_empty() {
                    warn!(
                        "Planet \"{planet_name}\" in \"{sys_name}\" is uncolonized \
                         but has a shipyard — this will be inaccessible"
                    );
                }
            }
        }
    }

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

            // ── planet_best_margin ──────────────────────────────────────────
            // For each planet, compute: min over commodities of
            // (price_here - max_price_on_any_other_planet_in_system).
            // A large negative value means there's a commodity that's cheap
            // here relative to the best sell price elsewhere → good trade.
            let mut planet_margins: HashMap<String, f64> = HashMap::new();
            for (planet_name, planet) in &system.planets {
                let mut best_margin = f64::INFINITY;
                for (commodity, &price) in &planet.commodities {
                    // Max price of this commodity on any OTHER planet in system.
                    let max_elsewhere = system
                        .planets
                        .iter()
                        .filter(|(pn, _)| *pn != planet_name)
                        .filter_map(|(_, p)| p.commodities.get(commodity))
                        .copied()
                        .max()
                        .unwrap_or(price);
                    let margin = price as f64 - max_elsewhere as f64;
                    if margin < best_margin {
                        best_margin = margin;
                    }
                }
                if best_margin.is_finite() {
                    planet_margins.insert(planet_name.clone(), best_margin);
                }
            }
            self.planet_best_margin
                .insert(system_name.clone(), planet_margins);
        }
    }

    /// For each planet, determine which ship types can replenish ammo there.
    /// A ship type can replenish if the planet's outfitter sells any secondary
    /// weapon type that appears in the ship's base_weapons.
    fn compute_planet_ammo(&mut self) {
        use std::collections::HashSet;

        for system in self.star_systems.values() {
            for (planet_name, planet) in &system.planets {
                let mut ship_set: HashSet<String> = HashSet::new();
                // Which secondary weapon types does this planet sell?
                let planet_secondary: HashSet<&str> = planet
                    .outfitter
                    .iter()
                    .filter(|item_name| {
                        self.outfitter_items
                            .get(*item_name)
                            .map(|item| matches!(item, OutfitterItem::SecondaryWeapon { .. }))
                            .unwrap_or(false)
                    })
                    .filter_map(|item_name| match self.outfitter_items.get(item_name) {
                        Some(OutfitterItem::SecondaryWeapon { weapon_type, .. }) => {
                            Some(weapon_type.as_str())
                        }
                        _ => None,
                    })
                    .collect();
                // For each ship type, check if any of its secondary weapons match.
                for (ship_type, ship_data) in &self.ships {
                    let has_ammo = ship_data.base_weapons.iter().any(|(wname, _)| {
                        // Check if this weapon is a secondary type sold here.
                        planet_secondary.contains(wname.as_str())
                    });
                    if has_ammo {
                        ship_set.insert(ship_type.clone());
                    }
                }
                self.planet_has_ammo_for
                    .insert(planet_name.clone(), ship_set);
            }
        }
    }

    /// Compute expected value of shattering asteroids in each field.
    fn compute_asteroid_values(&mut self) {
        for (system_name, system) in &self.star_systems {
            let mut field_values = Vec::new();
            for field_data in &system.astroid_fields {
                let total_weight: f32 = field_data.commodities.values().sum();
                if total_weight <= 0.0 {
                    field_values.push(0.0);
                    continue;
                }
                // Expected value = sum(weight_i/total * avg_price_i)
                // We don't multiply by expected quantity here — that depends on
                // asteroid size which varies. The caller can multiply by size.
                let ev: f64 = field_data
                    .commodities
                    .iter()
                    .map(|(commodity, &weight)| {
                        let prob = weight as f64 / total_weight as f64;
                        let avg_price = self
                            .global_average_price
                            .get(commodity)
                            .copied()
                            .unwrap_or(0.0);
                        prob * avg_price
                    })
                    .sum();
                field_values.push(ev);
            }
            self.asteroid_field_expected_value
                .insert(system_name.clone(), field_values);
        }
    }

    /// Compute per-ship-type credit scale.
    ///
    /// `credit_scale = cargo_space * avg_commodity_value
    ///                + max_ammo_fill_cost`
    ///
    /// where `max_ammo_fill_cost` is the cost of filling the ship's remaining
    /// item space with the most expensive ammo among its base weapons.
    fn compute_ship_credit_scales(&mut self) {
        let avg_commodity_value: f64 = if self.global_average_price.is_empty() {
            1.0
        } else {
            let sum: f64 = self.global_average_price.values().sum();
            sum / self.global_average_price.len() as f64
        };

        for (ship_type, ship_data) in &self.ships {
            let cargo_value = ship_data.cargo_space as f64 * avg_commodity_value;

            // Find the most expensive ammo per item-space unit among this ship's weapons.
            let max_ammo_cost_per_space: f64 = ship_data
                .base_weapons
                .keys()
                .filter_map(|wt| self.outfitter_items.get(wt))
                .filter_map(|item| match item {
                    OutfitterItem::SecondaryWeapon {
                        ammo_price,
                        ammo_space,
                        ..
                    } => Some(*ammo_price as f64 / (*ammo_space).max(1) as f64),
                    _ => None,
                })
                .fold(0.0_f64, f64::max);

            // Remaining item space after base weapons are mounted.
            let consumed: i32 = WeaponSystems::build(&ship_data.base_weapons, self)
                .iter_all()
                .map(|(_, s)| s.space_consumed())
                .sum();
            let remaining_item_space =
                (ship_data.item_space as i32 - consumed).max(0) as f64;

            let ammo_fill_cost = remaining_item_space * max_ammo_cost_per_space;

            let scale = (cargo_value + ammo_fill_cost).max(1.0) as f32;
            self.ship_credit_scale.insert(ship_type.clone(), scale);
        }
    }
}

#[derive(Deserialize, Serialize)]
pub enum OutfitterItem {
    PrimaryWeapon {
        price: i128,
        space: u16,
        weapon_type: String,
        #[serde(default)]
        display_name: String,
        #[serde(default)]
        required_unlocks: Vec<String>,
    },
    SecondaryWeapon {
        price: i128,
        space: u16,
        weapon_type: String,
        ammo_price: i128,
        ammo_space: u16,
        #[serde(default)]
        display_name: String,
        #[serde(default)]
        required_unlocks: Vec<String>,
    },
}

impl OutfitterItem {
    pub fn price(&self) -> i128 {
        match self {
            OutfitterItem::PrimaryWeapon { price, .. } => *price,
            OutfitterItem::SecondaryWeapon { price, .. } => *price,
        }
    }
    pub fn space(&self) -> u16 {
        match self {
            OutfitterItem::PrimaryWeapon { space, .. } => *space,
            OutfitterItem::SecondaryWeapon { space, .. } => *space,
        }
    }
    pub fn display_name(&self) -> &str {
        match self {
            OutfitterItem::PrimaryWeapon { display_name, .. } => display_name,
            OutfitterItem::SecondaryWeapon { display_name, .. } => display_name,
        }
    }
    pub fn required_unlocks(&self) -> &[String] {
        match self {
            OutfitterItem::PrimaryWeapon { required_unlocks, .. } => required_unlocks,
            OutfitterItem::SecondaryWeapon { required_unlocks, .. } => required_unlocks,
        }
    }
    fn display_name_mut(&mut self) -> &mut String {
        match self {
            OutfitterItem::PrimaryWeapon { display_name, .. } => display_name,
            OutfitterItem::SecondaryWeapon { display_name, .. } => display_name,
        }
    }
}

/// Probability distribution + population limits for ships in a star system.
#[derive(Deserialize, Serialize)]
pub struct ShipDistribution {
    /// Minimum number of AI ships to maintain in the system.
    pub min: usize,
    /// Maximum number of AI ships allowed in the system.
    pub max: usize,
    /// Relative spawn weights per ship type.  Higher values = more likely.
    pub types: HashMap<String, f32>,
}

impl Default for ShipDistribution {
    fn default() -> Self {
        ShipDistribution { min: 0, max: 0, types: HashMap::new() }
    }
}

impl ShipDistribution {
    /// Sample `count` ship types from the distribution.
    /// Returns an empty vec if the distribution has no types or all weights ≤ 0.
    pub fn sample(&self, count: usize, rng: &mut impl rand::Rng) -> Vec<String> {
        let total: f32 = self.types.values().sum();
        if total <= 0.0 || self.types.is_empty() {
            return vec![];
        }
        (0..count)
            .filter_map(|_| {
                let mut roll = rng.gen_range(0.0..total);
                for (ship_type, &weight) in &self.types {
                    roll -= weight;
                    if roll <= 0.0 {
                        return Some(ship_type.clone());
                    }
                }
                self.types.keys().next().cloned()
            })
            .collect()
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
    #[serde(default)]
    pub ships: ShipDistribution,
    #[serde(default)]
    pub display_name: String,
}

pub fn item_universe_plugin(app: &mut App) {
    // Load the ItemUniverse from disk
    let mut item_universe: ItemUniverse =
        parse_dir::<ItemUniverse>(Path::new("assets")).expect("failed to parse asset config");

    item_universe.fill_display_names();
    item_universe.compute_global_averages();
    item_universe.compute_trade_maps();
    item_universe.compute_planet_ammo();
    item_universe.compute_asteroid_values();
    item_universe.compute_ship_credit_scales();
    item_universe.validate();
    app.insert_resource::<ItemUniverse>(item_universe);
    app.add_systems(Startup, preload_sprites);
}

fn preload_sprites(
    asset_server: Res<AssetServer>,
    mut item_universe: ResMut<ItemUniverse>,
) {
    for data in item_universe.ships.values_mut() {
        data.sprite_handle = asset_server.load(data.sprite_path.clone());
    }
    for system in item_universe.star_systems.values_mut() {
        for (name, planet) in system.planets.iter_mut() {
            let path = format!("sprites/planets/{}.png", name);
            planet.sprite_handle = asset_server.load(path);
        }
    }
    for weapon in item_universe.weapons.values_mut() {
        weapon.sprite_handle = weapon
            .sprite_path
            .as_ref()
            .map(|path| asset_server.load(path.clone()));
        weapon.sound_handle = weapon
            .sound_path
            .as_ref()
            .map(|path| asset_server.load(path.clone()));
    }
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

#[cfg(test)]
#[path = "tests/item_universe_tests.rs"]
mod tests;
