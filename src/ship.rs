use crate::ai_ships::{JumpingIn, JumpingOut};
use crate::carrier::DockingEscort;
use crate::item_universe::{ItemUniverse, OutfitterItem};
use crate::rl_collection::{RLAgent, RLReward, RLShipDied, build_rl_ship_died};
use crate::weapons::{WeaponSystem, WeaponSystems};
use crate::{GameLayer, PlayState, Player, TravelContext, TravelPhase};
use avian2d::{math::*, prelude::*};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::PI;

/// Timer inserted as a resource when the player ship is destroyed.
/// Once it finishes the game transitions back to the main menu.
#[derive(Resource)]
struct PlayerDeathTimer(Timer);

/// Half-life in seconds for the distressed flag's geometric decay.
const DISTRESSED_HALF_LIFE_SECS: f32 = 3.0;

/// Marks a ship that was recently damaged.  The `level` starts at 1.0 on each
/// hit and decays geometrically toward 0 with half-life `DISTRESSED_HALF_LIFE_SECS`.
#[derive(Component, Clone, Default)]
pub struct Distressed {
    pub level: f32,
}

/// Tracks the ratio of good (non-neutral) to neutral combat hits across all
/// RL agents, used to compute the adaptive neutral-hit penalty.
#[derive(Resource, Default)]
struct CombatHitStats {
    good_hits: u64,
    neutral_hits: u64,
}

impl CombatHitStats {
    /// Fraction of good hits: `good / (good + neutral)`.  Returns 0 when no hits recorded.
    fn good_fraction(&self) -> f32 {
        let total = self.good_hits + self.neutral_hits;
        if total == 0 {
            0.0
        } else {
            self.good_hits as f32 / total as f32
        }
    }

    /// Serialise to a file as two little-endian u64 values.
    fn save(&self, path: &str) {
        use std::io::Write;
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&self.good_hits.to_le_bytes());
        buf.extend_from_slice(&self.neutral_hits.to_le_bytes());
        if let Ok(mut f) = std::fs::File::create(path) {
            let _ = f.write_all(&buf);
        }
    }

    /// Deserialise from a file written by [`save`].
    fn load(path: &str) -> Option<Self> {
        let data = std::fs::read(path).ok()?;
        if data.len() < 16 {
            return None;
        }
        Some(Self {
            good_hits: u64::from_le_bytes(data[0..8].try_into().ok()?),
            neutral_hits: u64::from_le_bytes(data[8..16].try_into().ok()?),
        })
    }
}

/// Interval (in seconds) between periodic saves of `CombatHitStats`.
const COMBAT_STATS_SAVE_INTERVAL: f32 = 60.0;

/// Timer resource that drives periodic saving of `CombatHitStats`.
#[derive(Resource)]
struct CombatStatsSaveTimer(Timer);

/// Load `CombatHitStats` from the experiment directory at startup.
fn load_combat_stats(
    exp_dir: Option<Res<crate::experiments::ExperimentDir>>,
    mut stats: ResMut<CombatHitStats>,
) {
    let Some(exp) = exp_dir else { return };
    if exp.is_fresh {
        return;
    }
    let path = exp.combat_stats_path();
    if let Some(loaded) = CombatHitStats::load(&path) {
        println!(
            "[combat_stats] Loaded: {} good, {} neutral hits from {path}",
            loaded.good_hits, loaded.neutral_hits
        );
        *stats = loaded;
    }
}

/// Periodically save `CombatHitStats` to the experiment directory.
fn save_combat_stats_periodic(
    time: Res<Time>,
    mut timer: ResMut<CombatStatsSaveTimer>,
    stats: Res<CombatHitStats>,
    exp_dir: Option<Res<crate::experiments::ExperimentDir>>,
) {
    timer.0.tick(time.delta());
    if !timer.0.just_finished() {
        return;
    }
    let Some(exp) = exp_dir else { return };
    stats.save(&exp.combat_stats_path());
}

/// Geometrically decay the distressed level toward 0 each frame.
fn tick_distressed(time: Res<Time>, mut query: Query<&mut Distressed>) {
    let dt = time.delta_secs();
    // decay = 0.5^(dt / half_life)
    let decay = (0.5_f32).powf(dt / DISTRESSED_HALF_LIFE_SECS);
    for mut d in &mut query {
        if d.level > 0.0 {
            d.level *= decay;
            // Snap to zero to avoid denormals.
            if d.level < 1e-4 {
                d.level = 0.0;
            }
        }
    }
}

pub fn ship_plugin(app: &mut App) {
    app.init_resource::<CombatHitStats>()
        .insert_resource(CombatStatsSaveTimer(Timer::from_seconds(
            COMBAT_STATS_SAVE_INTERVAL,
            TimerMode::Repeating,
        )))
        .add_message::<ShipCommand>()
        .add_message::<DamageShip>()
        .add_message::<ScoreHit>()
        .add_message::<BuyShip>()
        .add_systems(Startup, load_combat_stats)
        .add_systems(
            FixedUpdate,
            (apply_damage_handicap, ship_movement)
                .chain()
                .run_if(in_state(PlayState::Flying)),
        )
        .add_systems(
            Update,
            (apply_damage, sync_hostilites, score_hits, tick_distressed, repair_bot_tick)
                .run_if(in_state(PlayState::Flying)),
        )
        .add_systems(
            Update,
            update_ship_sprite_frame.run_if(in_state(PlayState::Flying)),
        )
        .add_systems(Update, save_combat_stats_periodic)
        .add_systems(Update, tick_player_death_timer)
        .add_systems(Update, handle_buy_ship);
}

#[derive(Event, Message)]
pub struct ShipCommand {
    pub entity: Entity,
    pub thrust: Scalar,
    pub turn: Scalar,
    pub reverse: Scalar,
}

#[derive(Event, Message)]
pub struct DamageShip {
    pub entity: Entity,
    pub damage: Scalar,
}

#[derive(Event, Message)]
pub struct BuyShip {
    pub ship_type: String,
}

#[derive(Event, Message)]
pub enum ScoreHit {
    OnShip { source: Entity, target: Entity },
    OnAsteroid { source: Entity },
}

/// Stores the ship's hostility map as a standalone component so other systems
/// can read it without conflicting with mutable `Ship` queries.
/// Keys are faction names that are hostile toward this ship; values are weights.
#[derive(Component, Clone, Default)]
pub struct ShipHostility(pub HashMap<String, f32>);

#[derive(Clone, Default, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum Personality {
    #[default]
    Trader, // Likes to trade, traveling from planet to planet
    Fighter, // Will attack other ships
    Miner,   // Will mine asteroids
}

fn default_angular_drag() -> Scalar {
    3.0
}
fn default_thrust_kp() -> Scalar {
    5.0
}
fn default_thrust_kd() -> Scalar {
    1.0
}
fn default_reverse_kp() -> Scalar {
    20.0
}
fn default_reverse_kd() -> Scalar {
    1.5
}
fn default_fuel_capacity() -> u16 {
    4
}
fn default_gun_mounts() -> u8 {
    2
}

fn default_tech_level() -> u8 {
    1
}

/// All-zero derived default — used only as a sentinel (e.g. uninitialised resource).
/// Real ship data always comes from the YAML item universe.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct ShipData {
    pub thrust: Scalar,    // N — forward force
    pub max_speed: Scalar, // m/s — speed cap
    pub torque: Scalar,    // N·m — maximum turning torque
    pub max_health: i32,
    pub cargo_space: u16,
    pub item_space: u16,
    pub base_weapons: HashMap<String, (u8, Option<u32>)>, // A list of the basic weapons this ship starts with, along with counts
    pub sprite_path: String,
    #[serde(skip)]
    pub sprite_handle: Handle<Image>,
    /// Layout for the baked rotation/thrust sprite atlas (None = plain image).
    #[serde(skip)]
    pub atlas_layout: Option<Handle<TextureAtlasLayout>>,
    pub radius: f32,
    pub price: i128,
    pub personality: Personality,
    pub faction: Option<String>,
    /// Shipyard tech level required to stock this hull (1 = any shipyard …
    /// 4 = faction capitals). Consumed by `derive_market_catalogs`.
    #[serde(default = "default_tech_level")]
    pub tech_level: u8,
    /// Factions whose shipyards stock this hull. Empty = derived: the ship's
    /// own `faction` if set, otherwise sold universally.
    #[serde(default)]
    pub sold_by: Vec<String>,
    #[serde(default)]
    pub display_name: String,
    /// Optional continuous particle aura around the hull (e.g. the violet
    /// precursor shimmer that makes tiny alien hulls readable against the
    /// void). Rides the same emitter as weapon particle trails.
    #[serde(default)]
    pub aura: Option<crate::weapons::ParticleFx>,
    /// Named unlock flags the player must have (all of) before this ship
    /// appears in shipyard listings. Empty = always available.
    #[serde(default)]
    pub required_unlocks: Vec<String>,
    /// Fixed/tube weapon hardpoints (lasers, cannons, missile and mine
    /// tubes). Carrier bays and decoy pods don't use mounts — item space
    /// governs those.
    #[serde(default = "default_gun_mounts")]
    pub gun_mounts: u8,
    /// Rotating turret rings (full-arc ballistic weapons). Small hulls have
    /// none — a turret ring needs a hull big enough to carry it.
    #[serde(default)]
    pub turret_mounts: u8,

    // Tuning fields — rarely set in YAML, sensible defaults are fine:
    #[serde(default = "default_angular_drag")]
    pub angular_drag: Scalar, // s⁻¹ — exponential decay rate for angular velocity
    // PD gains for thrust: F = kp*(v_target - v) - kd*dv
    #[serde(default = "default_thrust_kp")]
    pub thrust_kp: Scalar,
    #[serde(default = "default_thrust_kd")]
    pub thrust_kd: Scalar,
    // PD gains for reverse heading correction
    #[serde(default = "default_reverse_kp")]
    pub reverse_kp: Scalar,
    #[serde(default = "default_reverse_kd")]
    pub reverse_kd: Scalar,
    /// Max system-to-system jumps on a full tank. Higher = longer range
    /// (e.g. courier > shuttle). Each jump consumes 1.
    #[serde(default = "default_fuel_capacity")]
    pub fuel_capacity: u16,
}

/// Number of pre-baked heading frames per ship sprite atlas. MUST match
/// `scripts/ship3d/bake_atlases.py` (frames 0..N idle, N..2N thrust, packed
/// row-major in an 8×8 grid).
pub const SHIP_HEADINGS: usize = 32;
/// Atlas grid columns/rows (8×8 = 64 tiles = 2 × SHIP_HEADINGS).
pub const SHIP_ATLAS_COLS: u32 = 8;
pub const SHIP_ATLAS_ROWS: u32 = 8;
/// Atlas tile size in pixels (must match the bake's TILE).
pub const SHIP_ATLAS_TILE: u32 = 128;
/// On-screen sprite size = radius × this. Bumped from the prior 2.2 so ships
/// read more clearly (atlas tiles scale via custom_size — no re-bake needed).
const SHIP_SPRITE_SCALE: f32 = 2.6;

/// Full on-screen size of a ship sprite, derived from its collision radius.
/// Use this (not the texture size) for scale animations: ship atlases are a
/// fixed tile resolution decoupled from the displayed size.
pub fn ship_display_size(radius: f32) -> Vec2 {
    Vec2::splat(radius * SHIP_SPRITE_SCALE)
}

/// Drive-firing flag: set by `ship_movement` from the thrust command, read by
/// `update_ship_sprite_frame` to select the exhaust-on atlas frames.
#[derive(Component, Default)]
pub struct DriveActive(pub bool);

/// Build a ship's sprite. When a baked atlas layout is present, use an atlas
/// sprite (per-heading + thrust frames); otherwise fall back to a plain image.
/// Always sized to the ship's radius so tile resolution is independent of the
/// on-screen size.
pub fn ship_sprite(data: &ShipData) -> Sprite {
    let mut sprite = match &data.atlas_layout {
        Some(layout) => Sprite::from_atlas_image(
            data.sprite_handle.clone(),
            TextureAtlas {
                layout: layout.clone(),
                index: 0,
            },
        ),
        None => Sprite::from_image(data.sprite_handle.clone()),
    };
    sprite.custom_size = Some(Vec2::splat(data.radius * SHIP_SPRITE_SCALE));
    sprite
}

/// Hard cap on ship angular speed (rad/s). Above each ship's natural
/// steady-state turn rate, so the cap only bites on transient overshoot
/// (collisions, abrupt input reversals).
pub const MAX_ANG_SPEED: Scalar = 2.5 * PI;

/// Rotation accumulated in `dt` seconds when full turn input is applied from
/// rest, under the dynamics `ω̇ = -torque - angular_drag · ω`. Returns radians.
pub fn first_tick_rotation_rad(torque: Scalar, angular_drag: Scalar, dt: Scalar) -> Scalar {
    if angular_drag <= 0.0 {
        return 0.5 * torque * dt * dt;
    }
    let u = angular_drag * dt;
    torque / (angular_drag * angular_drag) * (u - 1.0 + (-u).exp())
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Target {
    Ship(Entity),
    Planet(Entity),
    Asteroid(Entity),
    Pickup(Entity),
}

impl Target {
    pub fn get_entity(&self) -> Entity {
        match self {
            Target::Ship(e) => e.clone(),
            Target::Planet(e) => e.clone(),
            Target::Asteroid(e) => e.clone(),
            Target::Pickup(e) => e.clone(),
        }
    }
}

/// All-zero derived default — used only as a sentinel (e.g. uninitialised resource).
/// Real ships are always built from item-universe data via `ship_bundle` / `ship_bundle_from_pilot`.
#[derive(Component, Clone, Default, Serialize, Deserialize)]
pub struct Ship {
    pub ship_type: String, // The type of the ship
    pub data: ShipData, // The ship data, can be looked up in the item universe, but stored here for convenience
    pub health: i32,
    pub cargo: HashMap<String, u16>, // Map from commodities to quantity
    /// Cumulative purchase cost (credits) of the cargo currently held, per
    /// commodity. Pickups and other "free" acquisitions contribute zero.
    /// Used to compute trade *profit* rather than gross sale value.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub cargo_cost: HashMap<String, i128>,
    /// Quantity of each commodity that is locked (e.g. mission cargo) and
    /// can't be sold or dropped by the player. Always `<= cargo[commodity]`.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub reserved_cargo: HashMap<String, u16>,
    /// Recently-visited planets, mapping planet_name → game-time seconds at
    /// which the cooldown expires. Used to discourage re-landing immediately
    /// after a sale; runtime-only, not persisted.
    #[serde(skip)]
    pub recent_landings: HashMap<String, f32>,
    pub credits: i128,
    /// Current fuel = jumps remaining (max = `data.fuel_capacity`). Each system
    /// jump consumes 1; refuel at the fuel station.
    #[serde(default)]
    pub fuel: u16,
    /// Extra cargo capacity contributed by the player's wing: companions and
    /// hired escorts lend their hulls' holds while enrolled. Derived from the
    /// EscortRoster every frame (see carrier::sync_escort_cargo_bonus) — NOT
    /// persisted; the roster is the source of truth.
    #[serde(skip)]
    pub escort_cargo_bonus: u16,
    // A map indicating inclusion in factions
    pub enemies: HashMap<String, f32>,
    /// Factions whose rewards are shared into this ship's reward signal.
    /// Includes own faction (always) plus cross-faction allies from `allies.yaml`.
    #[serde(skip)]
    pub allies: Vec<String>,
    /// Navigation target — where the ship is heading (planet, asteroid, pickup).
    #[serde(skip)]
    pub nav_target: Option<Target>,
    /// Weapons target — what the ship is aiming at (ship, asteroid).
    #[serde(skip)]
    pub weapons_target: Option<Target>,
    #[serde(skip)]
    pub weapon_systems: WeaponSystems,
    /// Installed ship mods: outfitter item name → count. Persisted with the
    /// ship; `mod_stats` is the derived cache.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub mods: HashMap<String, u8>,
    /// Aggregate effect of `mods`, recomputed on buy/sell/load (never per
    /// frame) by [`Ship::recompute_mod_stats`].
    #[serde(skip)]
    pub mod_stats: ModStats,
    /// Item space consumed by installed mods (cached alongside `mod_stats`).
    #[serde(skip)]
    pub mod_space: i32,
}

/// Cached aggregate of all installed [`ModEffect`]s. Defaults are the
/// identity (multipliers 1, bonuses 0), so ships without mods — every AI
/// ship — behave exactly as their raw `ShipData`.
#[derive(Clone, Debug, PartialEq)]
pub struct ModStats {
    pub speed_mult: f32,
    pub thrust_mult: f32,
    pub torque_mult: f32,
    pub armor_hp: i32,
    pub repair_per_sec: f32,
}

impl Default for ModStats {
    fn default() -> Self {
        Self {
            speed_mult: 1.0,
            thrust_mult: 1.0,
            torque_mult: 1.0,
            armor_hp: 0,
            repair_per_sec: 0.0,
        }
    }
}

impl Ship {
    pub fn consumed_item_space(&self) -> i32 {
        let weapon_space: i32 = self
            .weapon_systems
            .iter_all()
            .map(|(_, s)| s.space_consumed())
            .sum();
        weapon_space + self.mod_space
    }
    pub fn from_ship_data(data: &ShipData, ship_type: &str) -> Self {
        Self {
            ship_type: ship_type.to_string(),
            data: data.clone(),
            health: data.max_health,
            cargo: HashMap::new(),
            cargo_cost: HashMap::new(),
            reserved_cargo: HashMap::new(),
            recent_landings: HashMap::new(),
            credits: 10000,
            escort_cargo_bonus: 0,
            fuel: data.fuel_capacity,
            allies: Vec::new(),
            nav_target: None,
            weapons_target: None,
            weapon_systems: WeaponSystems::default(),
            enemies: HashMap::new(),
            mods: HashMap::new(),
            mod_stats: ModStats::default(),
            mod_space: 0,
        }
    }
    pub fn remaining_item_space(&self) -> i32 {
        return (self.data.item_space as i32 - self.consumed_item_space()).max(0);
    }

    // ── Effective ship stats ────────────────────────────────────────────────
    // The single choke points combining hull data, installed mods, and battle
    // damage. Physics and AI for the PLAYER go through these; raw `data.*`
    // stays the source of truth for unmodded AI ships and the RL observation
    // code (whose ships never carry mods, so the values coincide).

    /// Effective top speed: hull × engine mods × damage handling.
    pub fn max_speed(&self) -> f32 {
        self.data.max_speed * self.mod_stats.speed_mult * self.handling_factor()
    }
    /// Effective forward thrust (acceleration).
    pub fn thrust(&self) -> f32 {
        self.data.thrust * self.mod_stats.thrust_mult * self.handling_factor()
    }
    /// Effective turning torque.
    pub fn torque(&self) -> f32 {
        self.data.torque * self.mod_stats.torque_mult * self.handling_factor()
    }
    /// Effective maximum hull points (hull + armor mods).
    pub fn max_health(&self) -> i32 {
        self.data.max_health + self.mod_stats.armor_hp
    }

    /// Rebuild the cached [`ModStats`] from `self.mods`. Call after any mod
    /// purchase/sale and when a saved ship is rehydrated — never per frame.
    pub fn recompute_mod_stats(&mut self, item_universe: &ItemUniverse) {
        let mut stats = ModStats::default();
        let mut space = 0i32;
        for (name, &count) in &self.mods {
            let Some(item) = item_universe.outfitter_items.get(name) else {
                continue;
            };
            space += item.space() as i32 * count as i32;
            let Some(effect) = item.mod_effect() else {
                continue;
            };
            for _ in 0..count {
                match effect {
                    crate::item_universe::ModEffect::Engine {
                        speed,
                        thrust,
                        torque,
                    } => {
                        stats.speed_mult *= speed;
                        stats.thrust_mult *= thrust;
                        stats.torque_mult *= torque;
                    }
                    crate::item_universe::ModEffect::Armor { bonus_hp } => {
                        stats.armor_hp += bonus_hp;
                    }
                    crate::item_universe::ModEffect::RepairBot { hp_per_sec } => {
                        stats.repair_per_sec += hp_per_sec;
                    }
                }
            }
        }
        self.mod_stats = stats;
        self.mod_space = space;
    }

    /// Buy one copy of a ship mod. Refuses on unknown item / wrong item kind /
    /// insufficient credits or item space.
    pub fn buy_mod(&mut self, name: &str, item_universe: &ItemUniverse, markup: f32) {
        let Some(item) = item_universe.outfitter_items.get(name) else {
            return;
        };
        if item.mod_effect().is_none() {
            return;
        }
        let price = crate::standing::markup_price(item.price(), markup);
        if price > self.credits || (item.space() as i32) > self.remaining_item_space() {
            return;
        }
        self.credits -= price;
        *self.mods.entry(name.to_string()).or_insert(0) += 1;
        self.recompute_mod_stats(item_universe);
        // Armor changes effective max health; clamp (selling handled in sell_mod).
        self.health = self.health.min(self.max_health());
    }

    /// Sell one copy of a ship mod at full price (consistent with weapons).
    pub fn sell_mod(&mut self, name: &str, item_universe: &ItemUniverse) {
        let Some(item) = item_universe.outfitter_items.get(name) else {
            return;
        };
        let Some(count) = self.mods.get_mut(name) else {
            return;
        };
        if *count == 0 {
            return;
        }
        *count -= 1;
        if *count == 0 {
            self.mods.remove(name);
        }
        self.credits += item.price();
        self.recompute_mod_stats(item_universe);
        // Selling armor can drop max health below current health.
        self.health = self.health.min(self.max_health());
    }
    /// Compute the trade-in value of this ship: 80% of the ship price plus
    /// 80% of all equipped weapons/outfitter items.
    pub fn trade_in_value(&self, item_universe: &ItemUniverse) -> i128 {
        let ship_value = self.data.price * 80 / 100;
        let weapon_value: i128 = self
            .weapon_systems
            .iter_all()
            .map(|(wtype, ws)| {
                let unit_price = item_universe
                    .outfitter_items
                    .get(wtype)
                    .map(|item| item.price())
                    .unwrap_or(0);
                unit_price * ws.number as i128
            })
            .sum::<i128>()
            * 80
            / 100;
        let mod_value: i128 = self
            .mods
            .iter()
            .map(|(name, count)| {
                item_universe
                    .outfitter_items
                    .get(name)
                    .map(|item| item.price())
                    .unwrap_or(0)
                    * *count as i128
            })
            .sum::<i128>()
            * 80
            / 100;
        ship_value + weapon_value + mod_value
    }
    pub fn current_cargo(&self) -> u16 {
        self.cargo.values().sum()
    }
    /// Total hold: own hull + the wing's contributed holds.
    pub fn cargo_capacity(&self) -> u16 {
        self.data.cargo_space.saturating_add(self.escort_cargo_bonus)
    }

    /// Free space. Losing a contributing escort while loaded can leave the
    /// fleet OVER capacity — nothing is confiscated, this just saturates to
    /// zero so no new cargo fits until the player sells down.
    pub fn remaining_cargo_space(&self) -> u16 {
        self.cargo_capacity().saturating_sub(self.current_cargo())
    }
    fn add_cargo(&mut self, commodity: &str, quantity_desired: u16) -> u16 {
        let quantity_added = std::cmp::min(quantity_desired, self.remaining_cargo_space());
        *self.cargo.entry(commodity.to_string()).or_insert(0) += quantity_added;
        return quantity_added;
    }
    pub fn sell_cargo(&mut self, commodity: &str, quantity: u16, price: i128) {
        let held = *self.cargo.get(commodity).unwrap_or(&0u16);
        let reserved = *self.reserved_cargo.get(commodity).unwrap_or(&0u16);
        let sellable = held.saturating_sub(reserved);
        let quantity = std::cmp::min(sellable, quantity);
        if quantity == 0 {
            return;
        }
        // Deduct cost basis proportionally to the units sold, then update qty.
        let cost_basis = self.cargo_cost.get(commodity).copied().unwrap_or(0);
        let cost_removed = if held > 0 {
            cost_basis * quantity as i128 / held as i128
        } else {
            0
        };
        *self.cargo.entry(commodity.to_string()).or_insert(0) -= quantity;
        self.credits += (quantity as i128) * price;
        let remaining_qty = self.cargo.get(commodity).copied().unwrap_or(0);
        if remaining_qty == 0 {
            self.cargo.remove(commodity);
            self.cargo_cost.remove(commodity);
        } else {
            self.cargo_cost
                .insert(commodity.to_string(), (cost_basis - cost_removed).max(0));
        }
    }
    /// Drop a single ton of `commodity` overboard. Reserved (mission) cargo
    /// can't be jettisoned. Returns true iff a unit was actually removed.
    /// Cost basis is decremented proportionally so trade profit accounting
    /// stays consistent with `sell_cargo`.
    pub fn jettison_cargo(&mut self, commodity: &str) -> bool {
        let held = *self.cargo.get(commodity).unwrap_or(&0u16);
        let reserved = *self.reserved_cargo.get(commodity).unwrap_or(&0u16);
        if held.saturating_sub(reserved) == 0 {
            return false;
        }
        let cost_basis = self.cargo_cost.get(commodity).copied().unwrap_or(0);
        let cost_removed = if held > 0 {
            cost_basis / held as i128
        } else {
            0
        };
        *self.cargo.entry(commodity.to_string()).or_insert(0) -= 1;
        let remaining = self.cargo.get(commodity).copied().unwrap_or(0);
        if remaining == 0 {
            self.cargo.remove(commodity);
            self.cargo_cost.remove(commodity);
        } else {
            self.cargo_cost
                .insert(commodity.to_string(), (cost_basis - cost_removed).max(0));
        }
        true
    }
    pub fn buy_cargo(&mut self, commodity: &str, quantity_desired: u16, price: i128) {
        if price <= 0 {
            return; // malformed price data — refuse rather than divide by zero
        }
        // Clamp before the u16 cast: credits/price can exceed 65 535 (a rich
        // trader), and a raw `as u16` would wrap and buy almost nothing.
        let affordable = (self.credits / price).clamp(0, u16::MAX as i128) as u16;
        let quantity_desired = std::cmp::min(quantity_desired, affordable);
        let quantity_added = self.add_cargo(commodity, quantity_desired);
        self.credits -= (quantity_added as i128) * price;
        if quantity_added > 0 {
            *self.cargo_cost.entry(commodity.to_string()).or_insert(0) +=
                (quantity_added as i128) * price;
        }
    }
    /// (guns, turrets) currently occupying this hull's weapon mounts.
    pub fn mounts_used(&self) -> (u8, u8) {
        let mut guns = 0u8;
        let mut turrets = 0u8;
        for (_, ws) in self.weapon_systems.iter_all() {
            if !ws.weapon.uses_mount() {
                continue;
            }
            if ws.weapon.is_turret() {
                turrets = turrets.saturating_add(ws.number);
            } else {
                guns = guns.saturating_add(ws.number);
            }
        }
        (guns, turrets)
    }

    /// Whether one more of `weapon` fits this hull's gun/turret mounts.
    pub fn mount_free_for(&self, weapon: &crate::weapons::Weapon) -> bool {
        if !weapon.uses_mount() {
            return true;
        }
        let (guns, turrets) = self.mounts_used();
        if weapon.is_turret() {
            turrets < self.data.turret_mounts
        } else {
            guns < self.data.gun_mounts
        }
    }

    pub fn buy_weapon(&mut self, weapon_type: &str, item_universe: &ItemUniverse, markup: f32) {
        let Some(outfitter_item) = item_universe.outfitter_items.get(weapon_type) else {
            return;
        };
        let price = crate::standing::markup_price(outfitter_item.price(), markup);
        if price > self.credits {
            return;
        }
        if outfitter_item.space() as i32 > self.remaining_item_space() {
            return;
        }
        if let Some(weapon) = item_universe.weapons.get(weapon_type) {
            if !self.mount_free_for(weapon) {
                return;
            }
        }
        // Increment count if the weapon is already present in either map.
        if let Some(ws) = self.weapon_systems.primary.get_mut(weapon_type) {
            ws.number += 1;
            self.credits -= price;
            return;
        }
        if let Some(ws) = self.weapon_systems.secondary.get_mut(weapon_type) {
            ws.number += 1;
            self.credits -= price;
            return;
        }
        // New weapon: insert into the correct map based on whether it uses ammo.
        if let Some(new_weapon) = WeaponSystem::from_type(weapon_type, 1, None, item_universe) {
            self.credits -= outfitter_item.price();
            if new_weapon.ammo_quantity.is_some() {
                self.weapon_systems
                    .secondary
                    .insert(weapon_type.to_string(), new_weapon);
            } else {
                self.weapon_systems
                    .primary
                    .insert(weapon_type.to_string(), new_weapon);
            }
        }
    }
    pub fn sell_weapon(&mut self, weapon_type: &str, item_universe: &ItemUniverse) {
        let Some(outfitter_item) = item_universe.outfitter_items.get(weapon_type) else {
            return;
        };
        match self.weapon_systems.find_weapon_entry(weapon_type) {
            std::collections::hash_map::Entry::Occupied(mut view) => {
                self.credits += outfitter_item.price();
                let val = view.get_mut();
                if val.number > 0 {
                    val.number -= 1;
                } else {
                    view.remove_entry();
                }
            }
            _ => (),
        }
    }
    pub fn buy_max_ammo(&mut self, weapon_type: &str, item_universe: &ItemUniverse, markup: f32) {
        let Some(outfitter_item) = item_universe.outfitter_items.get(weapon_type) else {
            return;
        };
        match outfitter_item {
            OutfitterItem::SecondaryWeapon {
                ammo_price,
                ammo_space,
                ..
            } => {
                let ammo_price = crate::standing::markup_price(*ammo_price, markup);
                let max_price_qty = if ammo_price > 0 {
                    self.credits / ammo_price
                } else {
                    100
                };
                let max_space_qty = if *ammo_space > 0 {
                    self.remaining_item_space() / *ammo_space as i32
                } else {
                    100
                };
                let qty = std::cmp::min(max_price_qty, max_space_qty as i128).max(0);
                match self.weapon_systems.find_weapon_entry(weapon_type) {
                    std::collections::hash_map::Entry::Occupied(view) => {
                        let val = view.into_mut();
                        val.ammo_quantity = val.ammo_quantity.map(|n| n + qty as u32);
                        self.credits -= ammo_price * qty;
                    }
                    _ => (),
                }
            }
            _ => (),
        }
    }
    pub fn buy_ammo(&mut self, weapon_type: &str, item_universe: &ItemUniverse, markup: f32) {
        let Some(outfitter_item) = item_universe.outfitter_items.get(weapon_type) else {
            return;
        };
        match outfitter_item {
            OutfitterItem::SecondaryWeapon {
                ammo_price,
                ammo_space,
                ..
            } => {
                let ammo_price = crate::standing::markup_price(*ammo_price, markup);
                if ammo_price > self.credits {
                    return;
                }
                if *ammo_space as i32 > self.remaining_item_space() {
                    return;
                }
                match self.weapon_systems.find_weapon_entry(weapon_type) {
                    std::collections::hash_map::Entry::Occupied(view) => {
                        let val = view.into_mut();
                        val.ammo_quantity = val.ammo_quantity.map(|n| n + 1);
                        self.credits -= ammo_price;
                    }
                    _ => (),
                }
            }
            _ => (),
        }
    }
    pub fn sell_ammo(&mut self, weapon_type: &str, item_universe: &ItemUniverse) {
        let Some(outfitter_item) = item_universe.outfitter_items.get(weapon_type) else {
            return;
        };
        match outfitter_item {
            OutfitterItem::SecondaryWeapon { ammo_price, .. } => {
                match self.weapon_systems.find_weapon_entry(weapon_type) {
                    std::collections::hash_map::Entry::Occupied(mut view) => {
                        // self.credits += outfitter_item.price();
                        let val = view.get_mut();
                        val.ammo_quantity = match val.ammo_quantity {
                            Some(qty) => Some(if qty > 0 {
                                self.credits += ammo_price;
                                qty - 1
                            } else {
                                0
                            }),
                            _ => None,
                        }
                    }
                    _ => (),
                }
            }
            _ => (),
        }
    }
    /// Sell every round of ammo for `weapon_type` in one transaction
    /// (shift-click in the outfitter UI).
    pub fn sell_all_ammo(&mut self, weapon_type: &str, item_universe: &ItemUniverse) {
        let Some(OutfitterItem::SecondaryWeapon { ammo_price, .. }) =
            item_universe.outfitter_items.get(weapon_type)
        else {
            return;
        };
        if let std::collections::hash_map::Entry::Occupied(mut view) =
            self.weapon_systems.find_weapon_entry(weapon_type)
        {
            let val = view.get_mut();
            if let Some(qty) = val.ammo_quantity {
                self.credits += ammo_price * qty as i128;
                val.ammo_quantity = Some(0);
            }
        }
    }

    /// Returns true when this ship should fight `other`.
    /// Checks whether this ship's own faction appears in `other`'s hostility map,
    /// i.e. other is flagged as hostile toward my faction.
    pub fn should_engage(&self, other: &ShipHostility) -> bool {
        let Some(my_faction) = self.data.faction.as_ref() else {
            return false;
        };
        other.0.get(my_faction).map(|v| *v > 0.0).unwrap_or(false)
    }

    /// Battle damage degrades handling: a linear roll-off from 1.0 at full health
    /// to 0.5 when fully destroyed. The same factor scales acceleration, torque,
    /// top speed, and max turn rate, so a damaged ship handles like itself — just
    /// more sluggishly.
    pub fn handling_factor(&self) -> f32 {
        let hp_frac =
            (self.health.max(0) as f32 / self.max_health().max(1) as f32).clamp(0.0, 1.0);
        // Sub-linear roll-off: handling barely sags from light damage and falls off
        // faster as the hull nears destruction. 1.0 at full health -> 0.5 at 0.
        0.5 + 0.5 * hp_frac.sqrt()
    }
}

#[derive(Bundle)]
pub struct ShipBundle {
    ship: Ship,
    faction: ShipHostility,
    distressed: Distressed,
    repair_buffer: RepairBuffer,
    tracer_slots: crate::weapons::TracerSlots,
    pub sprite: Sprite,
    drive: DriveActive,
    transform: Transform,
    body: RigidBody,
    angular_damping: AngularDamping,
    max_speed: MaxLinearSpeed,
    max_angular_speed: MaxAngularSpeed,
    collider: Collider,
    colider_density: ColliderDensity,
    collision_events: CollisionEventsEnabled,
    layer: CollisionLayers,
}

impl ShipBundle {
    /// Current hull of the bundled ship (escort roster bookkeeping).
    pub fn ship_health(&self) -> i32 {
        self.ship.health
    }

    /// Override the bundled ship's hull (persistent escorts respawn with
    /// the damage they carried through the jump).
    pub fn set_ship_health(&mut self, health: i32) {
        self.ship.health = health.max(1);
    }

    /// Current secondary ammo of the bundled ship (weapon type → rounds).
    pub fn ship_ammo(&self) -> std::collections::HashMap<String, u32> {
        self.ship
            .weapon_systems
            .iter_all()
            .filter_map(|(k, ws)| ws.ammo_quantity.map(|n| (k.clone(), n)))
            .collect()
    }

    /// Override the bundled ship's secondary ammo (persistent escorts
    /// respawn with the rounds they actually had left).
    pub fn set_ship_ammo(&mut self, ammo: &std::collections::HashMap<String, u32>) {
        for (k, n) in ammo {
            if let Some(ws) = self.ship.weapon_systems.find_weapon(k) {
                ws.ammo_quantity = ws.ammo_quantity.map(|_| *n);
            }
        }
    }

    pub fn get_personality(&self) -> Personality {
        self.ship.data.personality.clone()
    }

    /// Collision radius of the bundled ship (for sizing scale animations).
    pub fn radius(&self) -> f32 {
        self.ship.data.radius
    }

    /// Replace the random starting cargo with a fixed commodity and quantity.
    /// Used when a ship is being spawned for a mission whose objective has a
    /// `collect` requirement, so the guaranteed drop on death matches the
    /// player's objective. Warns if the cargo hold is too small.
    pub fn set_mission_cargo(&mut self, commodity: String, quantity: u16) {
        let hold = self.ship.data.cargo_space;
        if quantity > hold {
            warn!(
                "Mission target '{}' cargo hold ({}) is smaller than required quantity ({}) of '{}'; loading only what fits",
                self.ship.ship_type, hold, quantity, commodity,
            );
        }
        self.ship.cargo.clear();
        self.ship.cargo_cost.clear();
        self.ship.reserved_cargo.clear();
        self.ship.cargo.insert(commodity, quantity.min(hold));
    }
}

/// Fill a ship's cargo hold to a uniformly random fraction with a single
/// commodity drawn uniformly from `commodity_pool`. No-op if the pool is
/// empty or the ship has no cargo space.
fn fill_cargo_randomly(ship: &mut Ship, commodity_pool: &[String], rng: &mut impl rand::Rng) {
    if commodity_pool.is_empty() || ship.data.cargo_space == 0 {
        return;
    }
    let fill_frac: f32 = rng.gen_range(0.0..=1.0);
    let qty = (ship.data.cargo_space as f32 * fill_frac).floor() as u16;
    if qty == 0 {
        return;
    }
    let commodity = commodity_pool[rng.gen_range(0..commodity_pool.len())].clone();
    ship.cargo.insert(commodity, qty);
}

pub fn ship_bundle(
    ship_type: &str,
    item_universe: &Res<ItemUniverse>,
    system_name: &str,
    pos: Vec2,
) -> ShipBundle {
    use rand::Rng;
    let ship_data = item_universe
        .ships
        .get(ship_type)
        .expect(&format!("Un recognized ship type {}", ship_type));
    let mut ship = Ship::from_ship_data(ship_data, ship_type);
    ship.weapon_systems = WeaponSystems::build(&ship_data.base_weapons, item_universe);
    ship.enemies = ship_data
        .faction
        .clone()
        .and_then(|faction| {
            item_universe
                .enemies
                .get(&faction)
                .map(|v| v.iter().map(|s| (s.clone(), 1.0)).collect())
        })
        .unwrap_or_default();

    // Build allies list: own faction (always) + cross-faction allies from allies.yaml.
    let mut allies: Vec<String> = Vec::new();
    if let Some(ref faction) = ship_data.faction {
        allies.push(faction.clone());
        if let Some(cross) = item_universe.allies.get(faction) {
            allies.extend(cross.iter().cloned());
        }
    }
    ship.allies = allies;

    // Randomize AI starting credits uniformly in [0, ship.price].
    let mut rng = rand::thread_rng();
    ship.credits = if ship_data.price > 0 {
        rng.gen_range(0..=ship_data.price)
    } else {
        0
    };

    // Decide which commodity pool (if any) this ship can spawn carrying.
    // - Systems with planets: only Traders carry cargo, drawn from commodities
    //   sold at any planet in the system.
    // - Planetless systems: any ship may carry cargo, drawn from the full
    //   commodity catalog (there's no local economy to anchor to).
    let commodity_pool: Vec<String> = match item_universe.star_systems.get(system_name) {
        Some(sys) if !sys.planets.is_empty() => {
            if matches!(ship_data.personality, Personality::Trader) {
                use std::collections::HashSet;
                sys.planets
                    .values()
                    .flat_map(|p| p.commodities.keys().cloned())
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect()
            } else {
                Vec::new()
            }
        }
        _ => item_universe.commodities.keys().cloned().collect(),
    };
    fill_cargo_randomly(&mut ship, &commodity_pool, &mut rng);

    ShipBundle {
        faction: ShipHostility(ship.enemies.clone()),
        ship,
        distressed: Distressed::default(),
        repair_buffer: RepairBuffer::default(),
        tracer_slots: crate::weapons::TracerSlots::new(crate::rl_obs::K_OWN_PROJECTILES),
        sprite: ship_sprite(&ship_data),
        drive: DriveActive::default(),
        transform: Transform::from_xyz(pos.x, pos.y, 0.0),
        body: RigidBody::Dynamic,
        angular_damping: AngularDamping(ship_data.angular_drag), // equivalent to angular_drag = 3.0
        max_speed: MaxLinearSpeed(ship_data.max_speed),          // Restitution::new(1.5),
        max_angular_speed: MaxAngularSpeed(MAX_ANG_SPEED),
        collider: Collider::circle(ship_data.radius),
        colider_density: ColliderDensity(2.0),
        collision_events: CollisionEventsEnabled,
        layer: CollisionLayers::new(
            GameLayer::Ship,
            [
                GameLayer::Weapon,
                GameLayer::Asteroid,
                GameLayer::Planet,
                GameLayer::Radar,
                GameLayer::Pickup,
            ],
        ),
    }
}

/// Like `ship_bundle`, but uses a pre-built `Ship` (e.g. loaded from a save file)
/// and an explicit weapon loadout instead of deriving both from item_universe defaults.
pub fn ship_bundle_from_pilot(
    mut ship: Ship,
    weapon_loadout: &HashMap<String, (u8, Option<u32>)>,
    item_universe: &Res<ItemUniverse>,
    pos: Vec2,
) -> ShipBundle {
    ship.weapon_systems = WeaponSystems::build(weapon_loadout, item_universe);
    ship.recompute_mod_stats(item_universe);
    // Player ships have no faction — prevents faction-level hostility contagion
    // (e.g. hitting one Merchant making ALL Merchants hostile).  The player's
    // per-entity enemies map still drives engagement via ShipHostility.
    ship.data.faction = None;
    ShipBundle {
        faction: ShipHostility(ship.enemies.clone()),
        distressed: Distressed::default(),
        repair_buffer: RepairBuffer::default(),
        tracer_slots: crate::weapons::TracerSlots::new(crate::rl_obs::K_OWN_PROJECTILES),
        angular_damping: AngularDamping(ship.data.angular_drag),
        max_speed: MaxLinearSpeed(ship.data.max_speed),
        max_angular_speed: MaxAngularSpeed(MAX_ANG_SPEED),
        sprite: ship_sprite(&ship.data),
        drive: DriveActive::default(),
        transform: Transform::from_xyz(pos.x, pos.y, 0.0),
        body: RigidBody::Dynamic,
        collider: Collider::circle(ship.data.radius),
        colider_density: ColliderDensity(2.0),
        collision_events: CollisionEventsEnabled,
        layer: CollisionLayers::new(
            GameLayer::Ship,
            [
                GameLayer::Weapon,
                GameLayer::Asteroid,
                GameLayer::Planet,
                GameLayer::Radar,
                GameLayer::Pickup,
            ],
        ),
        ship,
    }
}

// Physics system reads ShipCommand messages - runs in FixedUpdate before physics
fn ship_movement(
    mut reader: MessageReader<ShipCommand>,
    time: Res<Time>,
    mut query: Query<
        (
            &mut LinearVelocity,
            &mut AngularVelocity,
            &Transform,
            &Ship,
            &mut DriveActive,
        ),
        (With<RigidBody>, Without<Sensor>),
    >,
) {
    for cmd in reader.read() {
        let Ok((mut velocity, mut ang_vel, transform, ship, mut drive)) =
            query.get_mut(cmd.entity)
        else {
            continue;
        };
        let dt = time.delta_secs();
        // Battle damage degrades handling (see Ship::handling_factor): the same
        // factor scales acceleration and torque here, and the physics speed/turn
        // caps in apply_damage_handicap. Effective stats (max_speed()/thrust()/
        // torque()) fold hull data × engine mods × damage in one place.
        let (max_speed, thrust, torque) = (ship.max_speed(), ship.thrust(), ship.torque());
        // Drive flame is shown whenever forward thrust is commanded.
        drive.0 = cmd.thrust.abs() > f32::EPSILON;

        let forward = (transform.rotation * Vec3::Y).xy();
        let speed = velocity.length();

        if cmd.thrust.abs() > f32::EPSILON {
            let forward_speed = velocity.dot(forward);
            let speed_deficit = max_speed - forward_speed;
            let pd_force = (ship.data.thrust_kp * speed_deficit
                - ship.data.thrust_kd * forward_speed)
                .clamp(0.0, thrust);
            (*velocity).0 += forward * pd_force * cmd.thrust * dt;
        }

        if cmd.turn.abs() > f32::EPSILON {
            (*ang_vel).0 += -torque * cmd.turn * dt;
        }

        let new_ang_vel = ang_vel.0 * (-ship.data.angular_drag * dt).exp();
        (*ang_vel).0 = new_ang_vel;

        if cmd.reverse.abs() > f32::EPSILON && speed > f32::EPSILON {
            let retrograde = -velocity.normalize();
            let angle_err = forward.angle_to(retrograde);
            let pd_torque = (ship.data.reverse_kp * angle_err - ship.data.reverse_kd * new_ang_vel)
                .clamp(-torque, torque);
            (*ang_vel).0 += pd_torque * cmd.reverse * dt;
        }
    }
}

/// Scale the physics speed/turn caps by battle damage, so a damaged ship's top
/// speed and max turn rate fall in step with its (already-scaled) acceleration
/// and torque — see [`Ship::handling_factor`]. Ships mid-jump or mid-dock have
/// their caps driven by those animations, so they are skipped.
fn apply_damage_handicap(
    travel: Res<TravelContext>,
    mut query: Query<
        (&Ship, &mut MaxLinearSpeed, &mut MaxAngularSpeed, Has<Player>),
        (Without<JumpingIn>, Without<JumpingOut>, Without<DockingEscort>),
    >,
) {
    // The player's jump (accelerate_for_jump) drives the player's speed cap up to
    // JUMP_SPEED; don't reset it back to the normal cap while a jump is underway,
    // or the player can never reach jump speed and gets stuck mid-jump.
    let player_jumping = travel.phase != TravelPhase::Idle;
    for (ship, mut max_lin, mut max_ang, is_player) in &mut query {
        if is_player && player_jumping {
            continue;
        }
        max_lin.0 = ship.max_speed();
        max_ang.0 = MAX_ANG_SPEED * ship.handling_factor();
    }
}

/// Passive in-flight repair from installed repair-bot mods. Runs on virtual
/// time, so it (correctly) pauses while a pausing UI is open. Fractional
/// progress is accumulated in the timer-free `repair_buffer` so slow rates
/// still add up across frames.
fn repair_bot_tick(time: Res<Time>, mut ships: Query<(&mut Ship, &mut RepairBuffer)>) {
    let dt = time.delta_secs();
    for (mut ship, mut buffer) in &mut ships {
        let rate = ship.mod_stats.repair_per_sec;
        if rate <= 0.0 || ship.health <= 0 || ship.health >= ship.max_health() {
            continue;
        }
        buffer.0 += rate * dt;
        let whole = buffer.0.floor() as i32;
        if whole > 0 {
            buffer.0 -= whole as f32;
            ship.health = (ship.health + whole).min(ship.max_health());
        }
    }
}

/// Fractional hull-repair accumulator for [`repair_bot_tick`].
#[derive(Component, Default)]
pub struct RepairBuffer(pub f32);

/// Pick the atlas frame from each ship's heading and drive state.
///
/// Frames are pre-baked "nose-up, light rotated by -i·step" so that rotating
/// the sprite by the physics heading lands the baked light world-fixed → the
/// hull keeps rotating smoothly while highlights glide across it. Frames
/// `0..N` are idle, `N..2N` add the drive flame.
fn update_ship_sprite_frame(mut ships: Query<(&Transform, &DriveActive, &mut Sprite), With<Ship>>) {
    let step = std::f32::consts::TAU / SHIP_HEADINGS as f32;
    for (transform, drive, mut sprite) in &mut ships {
        let Some(atlas) = sprite.texture_atlas.as_mut() else {
            continue; // plain-image ship (no baked atlas)
        };
        let (heading, _, _) = transform.rotation.to_euler(bevy::math::EulerRot::ZYX);
        let i = (heading / step).round().rem_euclid(SHIP_HEADINGS as f32) as usize;
        atlas.index = i + if drive.0 { SHIP_HEADINGS } else { 0 };
    }
}

pub(crate) fn apply_damage(
    mut commands: Commands,
    mut reader: MessageReader<DamageShip>,
    mut ships: Query<(&mut Ship, &Transform, &mut Distressed)>,
    ai_ships: Query<(), With<crate::ai_ships::AIShip>>,
    rl_agents: Query<&RLAgent>,
    player_ships: Query<(), With<Player>>,
    mut explosion_writer: MessageWriter<crate::explosions::TriggerExplosion>,
    mut pickup_writer: MessageWriter<crate::pickups::PickupDrop>,
    mut rl_died_writer: MessageWriter<RLShipDied>,
    mut rl_reward_writer: MessageWriter<RLReward>,
    mut ship_destroyed_writer: MessageWriter<crate::missions::ShipDestroyed>,
    mission_targets: Query<&crate::missions::MissionTarget>,
    persistent_escorts: Query<&crate::carrier::PersistentEscort>,
    mut escort_roster: Option<ResMut<crate::carrier::EscortRoster>>,
    model_mode: Res<crate::ModelMode>,
    reward_cfg: Res<crate::config::RewardConfig>,
) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for event in reader.read() {
        let Ok((mut ship, transform, mut distressed)) = ships.get_mut(event.entity) else {
            continue;
        };
        // Health fraction BEFORE damage — used to scale the damage penalty.
        let health_before = ship.health;
        let h_frac_before = health_before as f32 / ship.data.max_health.max(1) as f32;
        ship.health = (ship.health - event.damage as i32).max(0);
        distressed.level = 1.0;

        // RL damage penalty: -k · damage · (1 - h/h_max)_before.
        // At full health the penalty is 0 (combat is free); at low health it
        // approaches the full damage magnitude (strong pressure to disengage).
        if rl_agents.contains(event.entity) {
            let dmg_frac = event.damage as f32 / ship.data.max_health.max(1) as f32;
            let penalty = -reward_cfg.health_damage_penalty * dmg_frac * (1.0 - h_frac_before);
            rl_reward_writer.write(RLReward {
                entity: event.entity,
                reward: penalty,
                reward_type: crate::consts::REWARD_DAMAGE,
            });
        }
        // Only trigger death on the first hit that kills — ignore subsequent
        // damage events for an entity already at 0 HP in the same frame.
        if ship.health == 0 && health_before > 0 {
            let location = transform.translation.xy();
            if player_ships.contains(event.entity) {
                // During training mode, we don't math the player die
                if matches!(*model_mode, crate::ModelMode::Eval) {
                    // Bigger explosion for the player, then return to main menu.
                    explosion_writer.write(crate::explosions::TriggerExplosion {
                        location,
                        size: 50.0,
                    });
                    crate::utils::safe_despawn(&mut commands, event.entity);
                    commands.insert_resource(PlayerDeathTimer(Timer::from_seconds(
                        1.5,
                        TimerMode::Once,
                    )));
                }
            } else if ai_ships.contains(event.entity) {
                // A persistent escort that dies is gone for good — the
                // roster entry retires with the hull.
                if let (Ok(pe), Some(roster)) =
                    (persistent_escorts.get(event.entity), &mut escort_roster)
                {
                    roster.record_death(pe.0);
                }
                ship_destroyed_writer.write(crate::missions::ShipDestroyed {
                    entity: event.entity,
                    mission_target: mission_targets.get(event.entity).ok().cloned(),
                });
                // Flush RL segment with terminal flag before despawning.
                if let Ok(agent) = rl_agents.get(event.entity) {
                    rl_died_writer.write(build_rl_ship_died(event.entity, agent));
                }
                explosion_writer.write(crate::explosions::TriggerExplosion {
                    location,
                    size: 20.0,
                });
                // Mission targets drop their full cargo deterministically so
                // the player can always meet a `collect` objective. Regular
                // AI ships drop a random subset (1..=qty) for loot variance.
                let is_mission_target = mission_targets.contains(event.entity);
                for (commodity, &qty) in &ship.cargo {
                    if qty > 0 {
                        let drop_qty = if is_mission_target {
                            qty
                        } else {
                            rng.gen_range(1..=qty)
                        };
                        pickup_writer.write(crate::pickups::PickupDrop {
                            location,
                            commodity: commodity.clone(),
                            quantity: drop_qty,
                        });
                    }
                }
                crate::utils::safe_despawn(&mut commands, event.entity);
            }
        }
    }
}

fn tick_player_death_timer(
    mut commands: Commands,
    time: Res<Time>,
    timer: Option<ResMut<PlayerDeathTimer>>,
    mut next_state: ResMut<NextState<PlayState>>,
) {
    let Some(mut timer) = timer else { return };
    if timer.0.tick(time.delta()).just_finished() {
        commands.remove_resource::<PlayerDeathTimer>();
        next_state.set(PlayState::MainMenu);
    }
}

fn handle_buy_ship(
    mut commands: Commands,
    mut reader: MessageReader<BuyShip>,
    mut player_query: Query<(Entity, &mut Ship), With<crate::Player>>,
    item_universe: Res<ItemUniverse>,
    unlocks: Res<crate::missions::PlayerUnlocks>,
    standings: Res<crate::standing::FactionStandings>,
    galaxy: Res<crate::galaxy::GalaxyControl>,
    landed: Res<crate::planet_ui::LandedContext>,
) {
    // Poor standing with the shipyard's EFFECTIVE faction (its system's live
    // controller) inflates the sticker price.
    let markup = landed
        .planet_name
        .as_deref()
        .and_then(|p| crate::galaxy::effective_planet_faction(&galaxy, &item_universe, p))
        .map(|f| crate::standing::price_markup(standings.get(&f)))
        .unwrap_or(1.0);
    for event in reader.read() {
        let Ok((entity, mut ship)) = player_query.single_mut() else {
            continue;
        };
        let Some(new_data) = item_universe.ships.get(&event.ship_type) else {
            continue;
        };
        // Defence in depth: the shipyard UI filters locked ships, but the
        // licence gate must hold for any programmatic BuyShip sender too.
        if new_data.required_unlocks.iter().any(|u| !unlocks.has(u)) {
            continue;
        }
        let trade_in = ship.trade_in_value(&item_universe);
        let net_cost = crate::standing::markup_price(new_data.price, markup) - trade_in;
        if ship.credits < net_cost {
            continue;
        }
        ship.credits -= net_cost;

        // Replace the ship component, preserving credits and cargo (capped to new space).
        let mut new_ship = Ship::from_ship_data(new_data, &event.ship_type);
        new_ship.weapon_systems = WeaponSystems::build(&new_data.base_weapons, &item_universe);
        new_ship.credits = ship.credits;
        // Transfer reserved (mission) cargo first so it always fits.
        for (commodity, &reserved_qty) in &ship.reserved_cargo {
            let space_left = new_ship
                .data
                .cargo_space
                .saturating_sub(new_ship.cargo.values().sum::<u16>());
            let transfer = reserved_qty.min(space_left);
            if transfer > 0 {
                *new_ship.cargo.entry(commodity.clone()).or_insert(0) += transfer;
                new_ship.reserved_cargo.insert(commodity.clone(), transfer);
                let src_qty = ship.cargo.get(commodity).copied().unwrap_or(0);
                let src_cost = ship.cargo_cost.get(commodity).copied().unwrap_or(0);
                if src_qty > 0 {
                    let cost_transfer = src_cost * transfer as i128 / src_qty as i128;
                    *new_ship.cargo_cost.entry(commodity.clone()).or_insert(0) += cost_transfer;
                }
            }
        }
        // Then transfer remaining (unreserved) cargo, capped to new space.
        for (commodity, &qty) in &ship.cargo {
            let reserved = ship.reserved_cargo.get(commodity).copied().unwrap_or(0);
            let unreserved = qty.saturating_sub(reserved);
            if unreserved == 0 {
                continue;
            }
            let space_left = new_ship
                .data
                .cargo_space
                .saturating_sub(new_ship.cargo.values().sum::<u16>());
            let transfer = unreserved.min(space_left);
            if transfer > 0 {
                *new_ship.cargo.entry(commodity.clone()).or_insert(0) += transfer;
                let src_cost = ship.cargo_cost.get(commodity).copied().unwrap_or(0);
                if qty > 0 {
                    let cost_transfer = src_cost * transfer as i128 / qty as i128;
                    *new_ship.cargo_cost.entry(commodity.clone()).or_insert(0) += cost_transfer;
                }
            }
        }
        *ship = new_ship;

        // Replace physics/render components.
        commands.entity(entity).insert((
            ship_sprite(new_data),
            Collider::circle(new_data.radius),
            MaxLinearSpeed(new_data.max_speed),
            AngularDamping(new_data.angular_drag),
        ));
    }
}

fn score_hits(
    mut reader: MessageReader<ScoreHit>,
    mut ship_hostilities: Query<&mut ShipHostility>,
    players: Query<(), With<Player>>,
    ships: Query<&Ship>,
    positions: Query<&Position>,
    rl_agents: Query<&RLAgent>,
    // Read-only scan of allies (entity + ship + position + distress) used to
    // detect the distress-gated focus-fire assist. Overlaps `ships`/`positions`
    // read-only, which Bevy permits.
    ally_scan: Query<(Entity, &Ship, &Position, &Distressed)>,
    mut rl_reward_writer: MessageWriter<RLReward>,
    mut combat_stats: ResMut<CombatHitStats>,
    reward_cfg: Res<crate::config::RewardConfig>,
) {
    for event in reader.read() {
        match event {
            ScoreHit::OnShip { source, target } => {
                let source_ship = ships.get(*source).ok();
                let target_ship = ships.get(*target).ok();
                let on_target = source_ship
                    .and_then(|s| s.weapons_target.as_ref())
                    .map(|t| t.get_entity() == *target)
                    .unwrap_or(false);

                // Faction hostility: only escalate for INTENTIONAL hits (the
                // target was the source's weapons_target).  Accidental/stray
                // hits don't trigger faction-level contagion. The PLAYER's
                // hostility map is derived from signed faction standing
                // instead (see standing::derive_player_hostility).
                if on_target && players.get(*source).is_err() {
                    if let Ok(mut source_hostility) = ship_hostilities.get_mut(*source) {
                        if let Some(ts) = &target_ship {
                            if let Some(target_faction) = ts.data.faction.as_ref() {
                                *(source_hostility
                                    .0
                                    .entry(target_faction.clone())
                                    .or_default()) += 1.0;
                            }
                        }
                    }
                }
                // RL reward for hitting a ship.
                if let Ok(agent) = rl_agents.get(*source) {
                    // Check if the target is hostile or should-engage.
                    let is_engaged = match (&source_ship, &target_ship) {
                        (Some(ss), Some(ts)) => {
                            let should_engage = ss.should_engage(
                                &ship_hostilities
                                    .get(*target)
                                    .map(|h| h.clone())
                                    .unwrap_or_default(),
                            );
                            let hostile = ts.should_engage(
                                &ship_hostilities
                                    .get(*source)
                                    .map(|h| h.clone())
                                    .unwrap_or_default(),
                            );
                            should_engage || hostile
                        }
                        _ => false,
                    };

                    let r = if is_engaged && on_target {
                        combat_stats.good_hits += 1;
                        reward_cfg.combat_hit_engaged_targeted
                    } else if is_engaged {
                        combat_stats.good_hits += 1;
                        reward_cfg.combat_hit_engaged_untargeted
                    } else {
                        // Adaptive neutral penalty: c = -p * r / (EPS + (1-p))
                        // bounded in [-r / EPS, 0].
                        combat_stats.neutral_hits += 1;
                        let p = combat_stats.good_fraction();
                        let c = -p * reward_cfg.combat_hit_engaged_untargeted
                            / (reward_cfg.combat_hit_eps + (1.0 - p));
                        c.clamp(
                            -reward_cfg.combat_hit_engaged_untargeted / reward_cfg.combat_hit_eps,
                            0.0,
                        )
                    };

                    let personality_scale = match agent.personality {
                        Personality::Fighter => reward_cfg.combat_personality_fighter,
                        Personality::Miner | Personality::Trader => {
                            reward_cfg.combat_personality_other
                        }
                    };

                    // Cooperative assist (escort / threat-interception): a Fighter
                    // that hits a ship earns a bonus in two cases —
                    //   (a) the ship it hit is itself attacking a NEARBY ALLY
                    //       ("kill what's shooting my teammate"), or
                    //   (b) the ship it hit is the weapons_target of a nearby
                    //       DISTRESSED ally — converging on the threat a teammate
                    //       in trouble has identified (distress-gated focus-fire /
                    //       target-sharing).
                    // Off by default (cooperative_assist_bonus = 0).
                    const ASSIST_RADIUS: f32 = 800.0;
                    const DISTRESS_MIN: f32 = 0.3;
                    let assist_bonus = if is_engaged
                        && matches!(agent.personality, Personality::Fighter)
                        && reward_cfg.cooperative_assist_bonus > 0.0
                    {
                        // (a) the ship I hit is attacking a nearby ally.
                        let victim = target_ship
                            .and_then(|ts| ts.weapons_target.as_ref())
                            .map(|t| t.get_entity());
                        let helped_aggressor = match (victim, source_ship) {
                            (Some(victim_e), Some(ss)) if victim_e != *source => {
                                match (
                                    ships.get(victim_e).ok(),
                                    positions.get(*source).ok(),
                                    positions.get(victim_e).ok(),
                                ) {
                                    (Some(vs), Some(sp), Some(vp)) => {
                                        let is_ally = vs
                                            .data
                                            .faction
                                            .as_ref()
                                            .map(|f| ss.allies.contains(f))
                                            .unwrap_or(false);
                                        is_ally && (sp.0 - vp.0).length() <= ASSIST_RADIUS
                                    }
                                    _ => false,
                                }
                            }
                            _ => false,
                        };
                        // (b) the ship I hit is a nearby distressed ally's target.
                        let helped_distressed = match (source_ship, positions.get(*source).ok()) {
                            (Some(ss), Some(sp)) => ally_scan.iter().any(
                                |(ally_e, ally_ship, ally_pos, ally_distress)| {
                                    ally_e != *source
                                        && ally_distress.level >= DISTRESS_MIN
                                        && ally_ship
                                            .data
                                            .faction
                                            .as_ref()
                                            .map(|f| ss.allies.contains(f))
                                            .unwrap_or(false)
                                        && ally_ship
                                            .weapons_target
                                            .as_ref()
                                            .map(|t| t.get_entity() == *target)
                                            .unwrap_or(false)
                                        && (sp.0 - ally_pos.0).length() <= ASSIST_RADIUS
                                },
                            ),
                            _ => false,
                        };
                        if helped_aggressor || helped_distressed {
                            reward_cfg.cooperative_assist_bonus
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };

                    rl_reward_writer.write(RLReward {
                        entity: *source,
                        reward: r * personality_scale + assist_bonus,
                        reward_type: crate::consts::REWARD_SHIP_HIT,
                    });
                    if let Some(ss) = ships.get(*source).ok() {
                        let h_frac = ss.health as f32 / ss.data.max_health.max(1) as f32;
                        rl_reward_writer.write(RLReward {
                            entity: *source,
                            reward: reward_cfg.health_bonus_per_event * h_frac,
                            reward_type: crate::consts::REWARD_HEALTH_GATED,
                        });
                    }
                }
            }
            ScoreHit::OnAsteroid { source, .. } => {
                if let Ok(agent) = rl_agents.get(*source) {
                    let reward = match agent.personality {
                        Personality::Miner => reward_cfg.asteroid_hit_miner,
                        Personality::Fighter | Personality::Trader => reward_cfg.asteroid_hit_other,
                    };
                    rl_reward_writer.write(RLReward {
                        entity: *source,
                        reward,
                        reward_type: crate::consts::REWARD_ASTEROID_HIT,
                    });
                    if let Some(ss) = ships.get(*source).ok() {
                        let h_frac = ss.health as f32 / ss.data.max_health.max(1) as f32;
                        rl_reward_writer.write(RLReward {
                            entity: *source,
                            reward: reward_cfg.health_bonus_per_event * h_frac,
                            reward_type: crate::consts::REWARD_HEALTH_GATED,
                        });
                    }
                }
            }
        }
    }
}

fn sync_hostilites(mut ships: Query<(&mut Ship, &mut ShipHostility)>) {
    for (mut ship, mut hostility) in ships.iter_mut() {
        // Runs every frame over every ship, so avoid deep clones and — just as
        // important — avoid dirtying change detection when nothing changed
        // (`&mut` derefs mark the component changed even on no-op writes).
        if let Some(faction) = ship.data.faction.as_deref() {
            if hostility.0.contains_key(faction) {
                let faction = faction.to_string();
                hostility.0.remove(&faction);
            }
        }
        // Copy the hostility onto the ship only when it actually differs.
        if ship.enemies != hostility.0 {
            ship.enemies = hostility.0.clone();
        }
    }
}

#[cfg(test)]
#[path = "tests/ship_tests.rs"]
mod tests;
