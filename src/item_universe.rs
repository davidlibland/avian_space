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

/// A recurring NPC character (assets/npcs.yaml): a consistent name and
/// appearance for storyline mission givers / objective NPCs. Referenced by
/// id from missions.yaml (`npc:` on npc_offer / meet_npc / catch_npc).
#[derive(Deserialize, Serialize, Clone, Default)]
pub struct NpcDef {
    /// Display name shown as the conversation title (e.g. "Foreman Okafor").
    pub name: String,
    /// Authored look. When absent, a deterministic appearance is derived
    /// from the npc id, so the character still looks the same everywhere.
    #[serde(default)]
    pub avatar: Option<crate::character_compositor::AvatarSpec>,
    /// Outfit bias for the derived appearance ("civilian", "guard", "miner",
    /// "merchant", "alien", ...). Ignored when `avatar` is authored.
    #[serde(default)]
    pub role: Option<String>,
}

/// A faction's identity and traits (assets/factions.yaml) — the single
/// source of truth game logic keys on, instead of hardcoded faction names.
#[derive(Deserialize, Serialize, Clone, Default)]
pub struct FactionData {
    #[serde(default)]
    pub display_name: String,
    /// Star-map colour (0–255 RGB).
    #[serde(default)]
    pub color: [u8; 3],
    /// Fighters follow wealth, not territory (pirates).
    #[serde(default)]
    pub lawless: bool,
    /// A guild, not a government: universal logistics, holds no systems.
    #[serde(default)]
    pub stateless: bool,
    /// Unaligned: never takes a side, restricts markets, or tracks standing.
    #[serde(default)]
    pub neutral: bool,
}

#[derive(Deserialize, Serialize)]
pub struct CommodityData {
    pub color: [f32; 3],
    #[serde(default)]
    pub display_name: String,
}

#[derive(Resource, Deserialize, Serialize)]
pub struct ItemUniverse {
    #[serde(default)]
    pub weapons: HashMap<String, Weapon>,
    #[serde(default)]
    pub ships: HashMap<String, ShipData>,
    #[serde(default)]
    pub star_systems: HashMap<String, StarSystem>,
    /// Definition of the isolated "simulator" star system used by RL training.
    /// Loaded from `assets/simulator_system.yaml` and folded into
    /// `star_systems` under the key "simulator" by `materialize_simulator_system`.
    #[serde(default)]
    pub simulator_system: Option<StarSystem>,
    /// Isolated "escort" training system (assets/escort_system.yaml): traders +
    /// pirates + Fed/Rebel defenders. Pirates hunt traders, defenders hunt
    /// pirates → escort/protection dynamic. Folded into `star_systems` as "escort".
    #[serde(default)]
    pub escort_system: Option<StarSystem>,
    /// Isolated "mining" training system (assets/mining_system.yaml): dense
    /// asteroid fields + miners (+ traders/pirates/defender for context).
    /// Folded into `star_systems` as "mining".
    #[serde(default)]
    pub mining_system: Option<StarSystem>,
    #[serde(default)]
    pub outfitter_items: HashMap<String, OutfitterItem>,
    // A map from my faction, to which factions engage me
    #[serde(default)]
    pub enemies: HashMap<String, Vec<String>>,
    /// Cross-faction ally declarations for reward sharing.
    /// Key = faction, Value = list of factions whose rewards are shared.
    /// Same-faction sharing is always implicit; this adds cross-faction allies.
    #[serde(default)]
    pub allies: HashMap<String, Vec<String>>,
    // The starting ship for the player:
    pub starting_ship: String,
    // The starting system for the player:
    pub starting_system: String,
    #[serde(default)]
    pub commodities: HashMap<String, CommodityData>,
    /// Faction registry (assets/factions.yaml).
    #[serde(default)]
    pub factions: HashMap<String, FactionData>,
    #[serde(default)]
    pub missions: HashMap<String, MissionDef>,
    #[serde(default)]
    pub mission_templates: HashMap<String, MissionTemplate>,
    /// Recurring NPC characters (assets/npcs.yaml): consistent names +
    /// appearances for storyline mission givers and objective NPCs.
    #[serde(default)]
    pub npcs: HashMap<String, NpcDef>,
    /// Average price of each commodity across all planets in all star systems.
    #[serde(skip)]
    pub global_average_price: HashMap<String, f64>,
    /// Minimum price of each commodity across all planets. Used as the
    /// sell-anywhere price when a planet doesn't normally trade a commodity.
    #[serde(skip)]
    pub global_minimum_price: HashMap<String, i128>,
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
        for (k, v) in &mut self.factions {
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
        crate::validate_assets::validate(self);
    }

    /// Fold the isolated RL-training systems (simulator / escort / mining),
    /// each loaded from its own `assets/<name>_system.yaml`, into
    /// `star_systems` under a fixed key. They are intentionally disconnected
    /// from the gameplay galaxy (no jump connections) so they can never be
    /// reached or seen by the player — used only to give training dedicated
    /// scenario worlds (combat arena, escort run, mining belt).
    pub const TRAINING_SYSTEM_KEYS: [&'static str; 3] = ["simulator", "escort", "mining"];

    fn materialize_training_systems(&mut self) {
        let entries = [
            (self.simulator_system.take(), "simulator", "Simulator"),
            (self.escort_system.take(), "escort", "Escort Run"),
            (self.mining_system.take(), "mining", "Mining Belt"),
        ];
        for (sys, key, name) in entries {
            if self.star_systems.contains_key(key) {
                continue; // already defined explicitly in star_systems.yaml
            }
            let Some(mut s) = sys else {
                continue;
            };
            // Connections empty so it stays unreachable from the gameplay galaxy.
            s.connections.clear();
            if s.display_name.is_empty() {
                s.display_name = name.to_string();
            }
            self.star_systems.insert(key.to_string(), s);
        }
    }

    /// Whether a faction name is "real" for territorial purposes: known,
    /// non-empty, and not flagged neutral. Unknown names count as real so a
    /// typo shows up as odd behavior + a validator warning, not silence.
    pub fn faction_takes_sides(&self, name: &str) -> bool {
        if name.is_empty() {
            return false;
        }
        self.factions.get(name).map(|f| !f.neutral).unwrap_or(true)
    }
    pub fn faction_is_lawless(&self, name: &str) -> bool {
        self.factions.get(name).map(|f| f.lawless).unwrap_or(false)
    }
    pub fn faction_is_stateless(&self, name: &str) -> bool {
        self.factions.get(name).map(|f| f.stateless).unwrap_or(false)
    }

    /// Find a planet by name across GAMEPLAY systems only. The RL-training
    /// systems clone real planets (the simulator is a copy of Sol), so any
    /// name-based scan that included them would nondeterministically resolve
    /// to a world outside the live galaxy.
    pub fn find_gameplay_planet(&self, name: &str) -> Option<(&str, &PlanetData)> {
        self.star_systems
            .iter()
            .filter(|(sys, _)| !Self::TRAINING_SYSTEM_KEYS.contains(&sys.as_str()))
            .find_map(|(sys, s)| s.planets.get(name).map(|p| (sys.as_str(), p)))
    }

    /// The static (asset-authored) faction of a system: its explicit
    /// `faction:` or the majority faction among its planets. Used to seed
    /// GalaxyControl and for the initial market/traffic derivation.
    pub fn static_system_faction(iu: &ItemUniverse, system: &str) -> Option<String> {
        let sys = iu.star_systems.get(system)?;
        if iu.faction_takes_sides(&sys.faction) {
            return Some(sys.faction.clone());
        }
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for planet in sys.planets.values() {
            if iu.faction_takes_sides(&planet.faction) {
                *counts.entry(planet.faction.as_str()).or_insert(0) += 1;
            }
        }
        counts
            .into_iter()
            .max_by_key(|(_, n)| *n)
            .map(|(f, _)| f.to_string())
    }

    /// Factions that own at least one planet: only these can restrict a
    /// ship's default sale territory. Landless factions (Merchant, Pirate,
    /// Precursor) can't — their unannotated hulls sell universally.
    fn planet_owning_factions(&self) -> std::collections::HashSet<String> {
        self.star_systems
            .values()
            .flat_map(|s| s.planets.values())
            .map(|p| p.faction.clone())
            .filter(|f| self.faction_takes_sides(f))
            .collect()
    }

    /// Fill one planet's catalogs from its stashed explicit entries plus
    /// everything its tech level + `seller` faction admits. The single rule
    /// both the initial derivation and runtime re-derivation go through.
    fn derive_planet_market(
        planet: &mut PlanetData,
        seller: Option<&str>,
        items: &HashMap<String, OutfitterItem>,
        ships: &HashMap<String, ShipData>,
        planet_factions: &std::collections::HashSet<String>,
    ) {
        planet.outfitter = planet.explicit_outfitter.clone();
        planet.shipyard = planet.explicit_shipyard.clone();
        if planet.tech_level == 0 || planet.uncolonized {
            return;
        }
        let faction_ok = |factions: &[String]| {
            factions.is_empty() || seller.is_some_and(|f| factions.iter().any(|x| x == f))
        };
        for (name, item) in items {
            if item.tech_level() <= planet.tech_level
                && faction_ok(item.factions())
                && !planet.outfitter.contains(name)
            {
                planet.outfitter.push(name.clone());
            }
        }
        for (name, ship) in ships {
            let sellers: &[String] = if ship.sold_by.is_empty() {
                match &ship.faction {
                    Some(f) if planet_factions.contains(f) => std::slice::from_ref(f),
                    _ => &[],
                }
            } else {
                &ship.sold_by
            };
            if ship.tech_level <= planet.tech_level
                && faction_ok(sellers)
                && !planet.shipyard.contains(name)
            {
                planet.shipyard.push(name.clone());
            }
        }
        planet.outfitter.sort();
        planet.shipyard.sort();
    }

    /// Initial market derivation (at load): stash the YAML lists as the
    /// explicit extras, then derive with each planet's own faction as the
    /// seller (planets and their systems agree in the shipped assets).
    /// tech_level 0 (the default) = no trade buildings at all.
    fn derive_market_catalogs(&mut self) {
        let planet_factions = self.planet_owning_factions();
        let neutral: std::collections::HashSet<String> = self
            .factions
            .iter()
            .filter(|(_, f)| f.neutral)
            .map(|(n, _)| n.clone())
            .collect();
        let takes_sides = |f: &str| !f.is_empty() && !neutral.contains(f);
        for sys in self.star_systems.values_mut() {
            let system_faction = sys.faction.clone();
            for planet in sys.planets.values_mut() {
                planet.explicit_outfitter = planet.outfitter.clone();
                planet.explicit_shipyard = planet.shipyard.clone();
                let seller: Option<String> = [planet.faction.as_str(), system_faction.as_str()]
                    .iter()
                    .find(|f| !f.is_empty())
                    .filter(|f| takes_sides(f))
                    .map(|f| f.to_string());
                Self::derive_planet_market(
                    planet,
                    seller.as_deref(),
                    &self.outfitter_items,
                    &self.ships,
                    &planet_factions,
                );
            }
        }
    }

    /// Runtime re-derivation when a system changes hands: every planet in it
    /// now sells the new `controller`'s gear (None = contested → universal
    /// stock only). Independent enclaves stay independent.
    pub fn rederive_system_market(&mut self, system: &str, controller: Option<&str>) {
        let planet_factions = self.planet_owning_factions();
        let neutral: std::collections::HashSet<String> = self
            .factions
            .iter()
            .filter(|(_, f)| f.neutral)
            .map(|(n, _)| n.clone())
            .collect();
        let Some(sys) = self.star_systems.get_mut(system) else {
            return;
        };
        for planet in sys.planets.values_mut() {
            let seller = if neutral.contains(&planet.faction) {
                None
            } else {
                controller
            };
            Self::derive_planet_market(
                planet,
                seller,
                &self.outfitter_items,
                &self.ships,
                &planet_factions,
            );
        }
    }

    // ── Derived ship traffic ─────────────────────────────────────────────
    // Replaces hand-authored per-system `ships:` maps (kept verbatim where
    // declared — training worlds, the precursor rift). Constants tuned for
    // lively skies: a typical core system carries ~6-8 ships with combat
    // hulls the biggest bucket, so border systems see real skirmishes.

    /// Population: how many AI ships a system sustains.
    const TRAFFIC_PER_TECH: f32 = 0.35;
    const TRAFFIC_PER_FIELD: f32 = 0.9;
    const TRAFFIC_PER_CONNECTION: f32 = 0.45;
    /// Role mix.
    const MERCHANTS_BASE: f32 = 1.0;
    const MERCHANTS_PER_LANDABLE: f32 = 0.6;
    const MERCHANTS_PER_CONNECTION: f32 = 0.25;
    const MINERS_PER_FIELD: f32 = 1.2;
    /// Pirates follow wealth, scaled by how unaligned the system is.
    const PIRATE_SHARE_OF_MERCHANTS: f32 = 0.5;
    /// Combat presence multiplier per faction influence share.
    const COMBAT_PER_PRESENCE: f32 = 3.5;
    /// Unfactioned fighters (mercenaries) drift everywhere lightly.
    const MERCENARY_SHARE: f32 = 0.3;

    /// Re-derive every non-authored system's ship distribution from the live
    /// influence simplex. Faction presence propagates λ-per-jump from
    /// neighbors, so a war next door bleeds patrols across the border.
    pub fn rederive_ship_presence(&mut self, galaxy: &crate::galaxy::GalaxyControl) {
        use crate::ship::Personality;
        let lambda = crate::galaxy::PRESENCE_LAMBDA;

        // neighbor lists (1 and 2 jumps), from the static jump graph
        let neighbors: HashMap<String, Vec<String>> = self
            .star_systems
            .iter()
            .map(|(k, s)| (k.clone(), s.connections.clone()))
            .collect();

        let system_names: Vec<String> = self.star_systems.keys().cloned().collect();
        for name in &system_names {
            if self.star_systems[name].authored_traffic
                || Self::TRAINING_SYSTEM_KEYS.contains(&name.as_str())
            {
                continue;
            }
            // presence(F) = local + λ·1-jump + λ²·2-jump influence
            let mut presence: HashMap<String, f32> = HashMap::new();
            let mut add = |map: &mut HashMap<String, f32>, sys: &str, w: f32| {
                if let Some(inf) = galaxy.influence.get(sys) {
                    for (f, v) in inf {
                        *map.entry(f.clone()).or_insert(0.0) += v * w;
                    }
                }
            };
            add(&mut presence, name, 1.0);
            let hop1: &[String] = neighbors.get(name).map(Vec::as_slice).unwrap_or(&[]);
            let mut seen: std::collections::HashSet<&str> =
                std::collections::HashSet::from([name.as_str()]);
            for n1 in hop1 {
                if seen.insert(n1) {
                    add(&mut presence, n1, lambda);
                }
            }
            for n1 in hop1 {
                for n2 in neighbors.get(n1).map(Vec::as_slice).unwrap_or(&[]) {
                    if seen.insert(n2) {
                        add(&mut presence, n2, lambda * lambda);
                    }
                }
            }
            let total_presence: f32 = presence.values().sum();
            if total_presence > 0.0 {
                for v in presence.values_mut() {
                    *v /= total_presence.max(1.0);
                }
            }
            let local_unaligned =
                1.0 - galaxy.influence.get(name).map(|m| m.values().sum()).unwrap_or(0.0);

            let sys = &self.star_systems[name];
            let tech: f32 = sys.planets.values().map(|p| p.tech_level as f32).sum();
            let landable = sys
                .planets
                .values()
                .filter(|p| !p.commodities.is_empty())
                .count() as f32;
            let fields = sys.astroid_fields.len() as f32;
            let conn = sys.connections.len() as f32;

            let population = (1.2
                + Self::TRAFFIC_PER_TECH * tech
                + Self::TRAFFIC_PER_FIELD * fields
                + Self::TRAFFIC_PER_CONNECTION * conn)
                .round()
                .clamp(4.0, 10.0) as usize;

            // Traders only loiter where a within-system route exists (≥2
            // colonised planets) — matches the trade-route validator.
            let merchants_w = if landable >= 2.0 {
                Self::MERCHANTS_BASE
                    + Self::MERCHANTS_PER_LANDABLE * landable
                    + Self::MERCHANTS_PER_CONNECTION * conn
            } else {
                0.0
            };
            let miners_w = Self::MINERS_PER_FIELD * fields;
            // Pirates follow wealth but never vanish: lawless transit systems
            // are exactly where they lurk.
            let pirates_w = Self::PIRATE_SHARE_OF_MERCHANTS
                * merchants_w.max(1.5)
                * (0.4 + local_unaligned);

            // tech-weighted share within a role bucket: small ships common.
            let techw = |t: u8| 1.0 / f32::powi(2.0, t.saturating_sub(1) as i32);

            let mut types: HashMap<String, f32> = HashMap::new();
            for (ship_name, ship) in &self.ships {
                // Faction TRAITS (assets/factions.yaml) drive the buckets —
                // no faction names are special-cased in code.
                let f = ship
                    .faction
                    .as_deref()
                    .filter(|f| !self.faction_is_stateless(f));
                let lawless = f.is_some_and(|f| self.faction_is_lawless(f));
                let w = match (&ship.personality, f) {
                    // Traders/miners: stateless/unfactioned hulls at the
                    // universal base rate; a faction's own logistics scale
                    // with its presence.
                    (Personality::Trader, None) => merchants_w,
                    (Personality::Trader, Some(f)) => {
                        2.0 * merchants_w * presence.get(f).copied().unwrap_or(0.0)
                    }
                    (Personality::Miner, None) => miners_w,
                    (Personality::Miner, Some(f)) => {
                        2.0 * miners_w * presence.get(f).copied().unwrap_or(0.0)
                    }
                    // Combat: LAWLESS fighters follow wealth; faction patrols
                    // follow propagated presence (they roam borders); CAPITAL
                    // ships (tech 4+) deploy only where their faction holds
                    // actual ground — a titan doesn't wander far from home.
                    (Personality::Fighter, Some(_)) if lawless => pirates_w,
                    (Personality::Fighter, Some(f)) if ship.tech_level >= 4 => {
                        Self::COMBAT_PER_PRESENCE
                            * galaxy
                                .influence
                                .get(name.as_str())
                                .and_then(|m| m.get(f))
                                .copied()
                                .unwrap_or(0.0)
                    }
                    (Personality::Fighter, Some(f)) => {
                        Self::COMBAT_PER_PRESENCE * presence.get(f).copied().unwrap_or(0.0)
                    }
                    (Personality::Fighter, None) => Self::MERCENARY_SHARE,
                };
                let w = w * techw(ship.tech_level);
                // Cutoff prunes trace-level presence (a distant faction's
                // capital ships shouldn't appear 2 jumps from home on 5%
                // spillover).
                if w > 0.02 {
                    types.insert(ship_name.clone(), w);
                }
            }

            let sys = self.star_systems.get_mut(name).unwrap();
            sys.ships.types = types;
            sys.ships.max = population;
            sys.ships.min = (population / 2).max(1);
        }
    }

    /// All post-parse processing, in dependency order. Call after `parse_dir`
    /// (the plugin does; tests that need derived catalogs must too).
    pub fn finalize(&mut self) {
        self.fill_display_names();
        self.materialize_training_systems();
        for sys in self.star_systems.values_mut() {
            sys.authored_traffic = !sys.ships.types.is_empty();
        }
        self.derive_market_catalogs();
        let seed = crate::galaxy::GalaxyControl::seeded_from(self);
        self.rederive_ship_presence(&seed);
        self.compute_global_averages();
        self.compute_trade_maps();
        self.compute_planet_ammo();
        self.compute_asteroid_values();
        self.compute_ship_credit_scales();
    }

    fn compute_global_averages(&mut self) {
        let mut sums: HashMap<String, (f64, u32)> = HashMap::new();
        let mut mins: HashMap<String, i128> = HashMap::new();
        for system in self.star_systems.values() {
            for planet in system.planets.values() {
                for (commodity, &price) in &planet.commodities {
                    let entry = sums.entry(commodity.clone()).or_insert((0.0, 0));
                    entry.0 += price as f64;
                    entry.1 += 1;
                    let min_entry = mins.entry(commodity.clone()).or_insert(price);
                    if price < *min_entry {
                        *min_entry = price;
                    }
                }
            }
        }
        self.global_average_price = sums
            .into_iter()
            .map(|(k, (sum, count))| (k, sum / count as f64))
            .collect();
        self.global_minimum_price = mins;
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
        /// Outfitter tech level required to stock this item (1 = anywhere).
        #[serde(default = "default_item_tech")]
        tech_level: u8,
        /// Factions whose outfitters stock it. Empty = sold universally.
        #[serde(default)]
        factions: Vec<String>,
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
        /// Outfitter tech level required to stock this item (1 = anywhere).
        #[serde(default = "default_item_tech")]
        tech_level: u8,
        /// Factions whose outfitters stock it. Empty = sold universally.
        #[serde(default)]
        factions: Vec<String>,
    },
    /// A passive ship modification (engine tune, armor plating, repair bot).
    /// Occupies item space like a weapon; its `effect` is folded into the
    /// ship's cached `ModStats` on buy/sell/load.
    ShipMod {
        price: i128,
        space: u16,
        effect: ModEffect,
        #[serde(default)]
        display_name: String,
        #[serde(default)]
        required_unlocks: Vec<String>,
        /// Outfitter tech level required to stock this item (1 = anywhere).
        #[serde(default = "default_item_tech")]
        tech_level: u8,
        /// Factions whose outfitters stock it. Empty = sold universally.
        #[serde(default)]
        factions: Vec<String>,
    },
}

impl OutfitterItem {
    pub fn price(&self) -> i128 {
        match self {
            OutfitterItem::PrimaryWeapon { price, .. } => *price,
            OutfitterItem::SecondaryWeapon { price, .. } => *price,
            OutfitterItem::ShipMod { price, .. } => *price,
        }
    }
    pub fn space(&self) -> u16 {
        match self {
            OutfitterItem::PrimaryWeapon { space, .. } => *space,
            OutfitterItem::SecondaryWeapon { space, .. } => *space,
            OutfitterItem::ShipMod { space, .. } => *space,
        }
    }
    pub fn display_name(&self) -> &str {
        match self {
            OutfitterItem::PrimaryWeapon { display_name, .. } => display_name,
            OutfitterItem::SecondaryWeapon { display_name, .. } => display_name,
            OutfitterItem::ShipMod { display_name, .. } => display_name,
        }
    }
    pub fn required_unlocks(&self) -> &[String] {
        match self {
            OutfitterItem::PrimaryWeapon { required_unlocks, .. } => required_unlocks,
            OutfitterItem::SecondaryWeapon { required_unlocks, .. } => required_unlocks,
            OutfitterItem::ShipMod { required_unlocks, .. } => required_unlocks,
        }
    }
    pub fn tech_level(&self) -> u8 {
        match self {
            OutfitterItem::PrimaryWeapon { tech_level, .. } => *tech_level,
            OutfitterItem::SecondaryWeapon { tech_level, .. } => *tech_level,
            OutfitterItem::ShipMod { tech_level, .. } => *tech_level,
        }
    }
    pub fn factions(&self) -> &[String] {
        match self {
            OutfitterItem::PrimaryWeapon { factions, .. } => factions,
            OutfitterItem::SecondaryWeapon { factions, .. } => factions,
            OutfitterItem::ShipMod { factions, .. } => factions,
        }
    }
    /// The mod effect, when this item is a ship mod.
    pub fn mod_effect(&self) -> Option<&ModEffect> {
        match self {
            OutfitterItem::ShipMod { effect, .. } => Some(effect),
            _ => None,
        }
    }
    fn display_name_mut(&mut self) -> &mut String {
        match self {
            OutfitterItem::PrimaryWeapon { display_name, .. } => display_name,
            OutfitterItem::SecondaryWeapon { display_name, .. } => display_name,
            OutfitterItem::ShipMod { display_name, .. } => display_name,
        }
    }
}

/// What a ship mod does. One variant per mod family keeps every effect's
/// parameters together and makes `Ship::recompute_mod_stats` a single match.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub enum ModEffect {
    /// Multipliers on the drive train: top speed, acceleration (thrust) and
    /// turn rate. Stacks multiplicatively across copies.
    Engine {
        #[serde(default = "one")]
        speed: f32,
        #[serde(default = "one")]
        thrust: f32,
        #[serde(default = "one")]
        torque: f32,
    },
    /// Flat bonus hull points on top of the hull's max_health. Stacks.
    Armor { bonus_hp: i32 },
    /// Passive in-flight repair, hull points per second. Stacks additively.
    RepairBot { hp_per_sec: f32 },
}

fn one() -> f32 {
    1.0
}

fn default_item_tech() -> u8 {
    1
}

/// Probability distribution + population limits for ships in a star system.
#[derive(Clone, Deserialize, Serialize)]
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

#[derive(Clone, Deserialize, Serialize)]
pub struct StarSystem {
    /// Controlling faction of the whole system — the seller faction for
    /// planets that don't declare their own, and the standing system's
    /// controller. Empty = derived from the majority of planet factions.
    #[serde(default)]
    pub faction: String,
    /// Whether galactic-war influence shifts may contest this system.
    /// Story-critical systems (Sol, faction capitals) stay false so their
    /// planets can never change hands. See docs/galactic_war_design.md.
    #[serde(default)]
    pub contestable: bool,
    /// True when the YAML declares an explicit `ships:` distribution — such
    /// systems (RL training worlds, the precursor rift) keep it verbatim and
    /// are skipped by the traffic derivation.
    #[serde(skip)]
    pub authored_traffic: bool,
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
        parse_dir::<ItemUniverse>(Path::new("assets")).expect(
            "failed to parse asset config — check that all .yaml files in assets/ are valid",
        );

    item_universe.finalize();
    item_universe.validate();
    app.insert_resource::<ItemUniverse>(item_universe);
    app.add_systems(Startup, preload_sprites);
}

fn preload_sprites(
    asset_server: Res<AssetServer>,
    mut item_universe: ResMut<ItemUniverse>,
    // Optional so headless training apps without the sprite asset type still run.
    atlas_layouts: Option<ResMut<Assets<TextureAtlasLayout>>>,
) {
    use crate::ship::{SHIP_ATLAS_COLS, SHIP_ATLAS_ROWS, SHIP_ATLAS_TILE};
    // One shared layout for every ship atlas (all are N×N grids of equal tiles).
    let ship_atlas_layout = atlas_layouts.map(|mut layouts| {
        layouts.add(TextureAtlasLayout::from_grid(
            UVec2::splat(SHIP_ATLAS_TILE),
            SHIP_ATLAS_COLS,
            SHIP_ATLAS_ROWS,
            None,
            None,
        ))
    });
    for data in item_universe.ships.values_mut() {
        data.sprite_handle = asset_server.load(data.sprite_path.clone());
        data.atlas_layout = ship_atlas_layout.clone();
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

pub fn parse_dir<T: DeserializeOwned>(dir: &Path) -> Result<T, serde_yaml::Error> {
    let value = crate::embedded_assets::dir_to_yaml(dir).unwrap_or(Value::Mapping(Mapping::new()));
    serde_yaml::from_value(value)
}

#[cfg(test)]
#[path = "tests/item_universe_tests.rs"]
mod tests;
