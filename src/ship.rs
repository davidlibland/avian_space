use crate::item_universe::{ItemUniverse, OutfitterItem};
use crate::rl_collection::{RLAgent, RLReward, RLShipDied, build_rl_ship_died};
use crate::weapons::{WeaponSystem, WeaponSystems};
use crate::{GameLayer, PlayState, Player};
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
            ship_movement.run_if(in_state(PlayState::Flying)),
        )
        .add_systems(
            Update,
            (apply_damage, sync_hostilites, score_hits, tick_distressed)
                .run_if(in_state(PlayState::Flying)),
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

#[derive(Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
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
    pub radius: f32,
    pub price: i128,
    pub personality: Personality,
    pub faction: Option<String>,
    #[serde(default)]
    pub display_name: String,
    /// Named unlock flags the player must have (all of) before this ship
    /// appears in shipyard listings. Empty = always available.
    #[serde(default)]
    pub required_unlocks: Vec<String>,

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
}

#[derive(Clone, PartialEq, Eq)]
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
}

impl Ship {
    pub fn consumed_item_space(&self) -> i32 {
        self.weapon_systems
            .iter_all()
            .map(|(_, s)| s.space_consumed())
            .sum()
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
            allies: Vec::new(),
            nav_target: None,
            weapons_target: None,
            weapon_systems: WeaponSystems::default(),
            enemies: HashMap::new(),
        }
    }
    pub fn remaining_item_space(&self) -> i32 {
        return (self.data.item_space as i32 - self.consumed_item_space()).max(0);
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
        ship_value + weapon_value
    }
    fn current_cargo(&self) -> u16 {
        self.cargo.values().sum()
    }
    pub fn remaining_cargo_space(&self) -> u16 {
        return self.data.cargo_space.saturating_sub(self.current_cargo());
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
    pub fn buy_cargo(&mut self, commodity: &str, quantity_desired: u16, price: i128) {
        let quantity_desired = std::cmp::min(quantity_desired, (self.credits / price) as u16);
        let quantity_added = self.add_cargo(commodity, quantity_desired);
        self.credits -= (quantity_added as i128) * price;
        if quantity_added > 0 {
            *self.cargo_cost.entry(commodity.to_string()).or_insert(0) +=
                (quantity_added as i128) * price;
        }
    }
    pub fn buy_weapon(&mut self, weapon_type: &str, item_universe: &ItemUniverse) {
        let Some(outfitter_item) = item_universe.outfitter_items.get(weapon_type) else {
            return;
        };
        if outfitter_item.price() > self.credits {
            return;
        }
        if outfitter_item.space() as i32 > self.remaining_item_space() {
            return;
        }
        // Increment count if the weapon is already present in either map.
        if let Some(ws) = self.weapon_systems.primary.get_mut(weapon_type) {
            ws.number += 1;
            self.credits -= outfitter_item.price();
            return;
        }
        if let Some(ws) = self.weapon_systems.secondary.get_mut(weapon_type) {
            ws.number += 1;
            self.credits -= outfitter_item.price();
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
    pub fn buy_max_ammo(&mut self, weapon_type: &str, item_universe: &ItemUniverse) {
        let Some(outfitter_item) = item_universe.outfitter_items.get(weapon_type) else {
            return;
        };
        match outfitter_item {
            OutfitterItem::SecondaryWeapon {
                ammo_price,
                ammo_space,
                ..
            } => {
                let max_price_qty = if *ammo_price > 0 {
                    self.credits / *ammo_price
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
    pub fn buy_ammo(&mut self, weapon_type: &str, item_universe: &ItemUniverse) {
        let Some(outfitter_item) = item_universe.outfitter_items.get(weapon_type) else {
            return;
        };
        match outfitter_item {
            OutfitterItem::SecondaryWeapon {
                ammo_price,
                ammo_space,
                ..
            } => {
                if *ammo_price > self.credits {
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
    /// Returns true when this ship should fight `other`.
    /// Checks whether this ship's own faction appears in `other`'s hostility map,
    /// i.e. other is flagged as hostile toward my faction.
    pub fn should_engage(&self, other: &ShipHostility) -> bool {
        let Some(my_faction) = self.data.faction.as_ref() else {
            return false;
        };
        other.0.get(my_faction).map(|v| *v > 0.0).unwrap_or(false)
    }
}

#[derive(Bundle)]
pub struct ShipBundle {
    ship: Ship,
    faction: ShipHostility,
    distressed: Distressed,
    tracer_slots: crate::weapons::TracerSlots,
    pub sprite: Sprite,
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
    pub fn get_personality(&self) -> Personality {
        self.ship.data.personality.clone()
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
        tracer_slots: crate::weapons::TracerSlots::new(crate::rl_obs::K_OWN_PROJECTILES),
        sprite: Sprite::from_image(ship_data.sprite_handle.clone()),
        transform: Transform::from_xyz(pos.x, pos.y, 0.0),
        body: RigidBody::Dynamic,
        angular_damping: AngularDamping(ship_data.angular_drag), // equivalent to angular_drag = 3.0
        max_speed: MaxLinearSpeed(ship_data.max_speed),          // Restitution::new(1.5),
        max_angular_speed: MaxAngularSpeed(4.0 * PI),
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
    // Player ships have no faction — prevents faction-level hostility contagion
    // (e.g. hitting one Merchant making ALL Merchants hostile).  The player's
    // per-entity enemies map still drives engagement via ShipHostility.
    ship.data.faction = None;
    ShipBundle {
        faction: ShipHostility(ship.enemies.clone()),
        distressed: Distressed::default(),
        tracer_slots: crate::weapons::TracerSlots::new(crate::rl_obs::K_OWN_PROJECTILES),
        angular_damping: AngularDamping(ship.data.angular_drag),
        max_speed: MaxLinearSpeed(ship.data.max_speed),
        max_angular_speed: MaxAngularSpeed(4.0 * PI),
        sprite: Sprite::from_image(ship.data.sprite_handle.clone()),
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
        (&mut LinearVelocity, &mut AngularVelocity, &Transform, &Ship),
        (With<RigidBody>, Without<Sensor>),
    >,
) {
    for cmd in reader.read() {
        let Ok((mut velocity, mut ang_vel, transform, ship)) = query.get_mut(cmd.entity) else {
            continue;
        };
        let dt = time.delta_secs();

        let forward = (transform.rotation * Vec3::Y).xy();
        let speed = velocity.length();

        if cmd.thrust.abs() > f32::EPSILON {
            let forward_speed = velocity.dot(forward);
            let speed_deficit = ship.data.max_speed - forward_speed;
            let pd_force = (ship.data.thrust_kp * speed_deficit
                - ship.data.thrust_kd * forward_speed)
                .clamp(0.0, ship.data.thrust);
            (*velocity).0 += forward * pd_force * cmd.thrust * dt;
        }

        if cmd.turn.abs() > f32::EPSILON {
            (*ang_vel).0 += -ship.data.torque * cmd.turn * dt;
        }

        let new_ang_vel = ang_vel.0 * (-ship.data.angular_drag * dt).exp();
        (*ang_vel).0 = new_ang_vel;

        if cmd.reverse.abs() > f32::EPSILON && speed > f32::EPSILON {
            let retrograde = -velocity.normalize();
            let angle_err = forward.angle_to(retrograde);
            let pd_torque = (ship.data.reverse_kp * angle_err - ship.data.reverse_kd * new_ang_vel)
                .clamp(-ship.data.torque, ship.data.torque);
            (*ang_vel).0 += pd_torque * cmd.reverse * dt;
        }
    }
}

fn apply_damage(
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
    model_mode: Res<crate::ModelMode>,
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
            let penalty = -crate::consts::HEALTH_DAMAGE_PENALTY * dmg_frac * (1.0 - h_frac_before);
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
) {
    for event in reader.read() {
        let Ok((entity, mut ship)) = player_query.single_mut() else {
            continue;
        };
        let Some(new_data) = item_universe.ships.get(&event.ship_type) else {
            continue;
        };
        let trade_in = ship.trade_in_value(&item_universe);
        let net_cost = new_data.price - trade_in;
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
                new_ship
                    .reserved_cargo
                    .insert(commodity.clone(), transfer);
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
            Sprite::from_image(new_data.sprite_handle.clone()),
            Collider::circle(new_data.radius),
            MaxLinearSpeed(new_data.max_speed),
            AngularDamping(new_data.angular_drag),
        ));
    }
}

fn score_hits(
    mut reader: MessageReader<ScoreHit>,
    mut ship_hostilities: Query<&mut ShipHostility>,
    ships: Query<&Ship>,
    rl_agents: Query<&RLAgent>,
    mut rl_reward_writer: MessageWriter<RLReward>,
    mut combat_stats: ResMut<CombatHitStats>,
) {
    use crate::consts::*;
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
                // hits don't trigger faction-level contagion.
                if on_target {
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
                        COMBAT_HIT_ENGAGED_TARGETED
                    } else if is_engaged {
                        combat_stats.good_hits += 1;
                        COMBAT_HIT_ENGAGED_UNTARGETED
                    } else {
                        // Adaptive neutral penalty: c = -p * r / (EPS + (1-p))
                        // bounded in [-r / EPS, 0].
                        combat_stats.neutral_hits += 1;
                        // let p = combat_stats.good_fraction();
                        // let c = -p * COMBAT_HIT_ENGAGED_UNTARGETED / (COMBAT_HIT_EPS + (1.0 - p));
                        // c.clamp(-COMBAT_HIT_ENGAGED_UNTARGETED / COMBAT_HIT_EPS, 0.0)
                        -COMBAT_HIT_ENGAGED_UNTARGETED
                    };

                    let personality_scale = match agent.personality {
                        Personality::Fighter => COMBAT_PERSONALITY_FIGHTER,
                        Personality::Miner | Personality::Trader => COMBAT_PERSONALITY_OTHER,
                    };

                    rl_reward_writer.write(RLReward {
                        entity: *source,
                        reward: r * personality_scale,
                        reward_type: crate::consts::REWARD_SHIP_HIT,
                    });
                    if let Some(ss) = ships.get(*source).ok() {
                        let h_frac = ss.health as f32 / ss.data.max_health.max(1) as f32;
                        rl_reward_writer.write(RLReward {
                            entity: *source,
                            reward: HEALTH_BONUS_PER_EVENT * h_frac,
                            reward_type: crate::consts::REWARD_HEALTH_GATED,
                        });
                    }
                }
            }
            ScoreHit::OnAsteroid { source, .. } => {
                if let Ok(agent) = rl_agents.get(*source) {
                    let reward = match agent.personality {
                        Personality::Miner => ASTEROID_HIT_MINER,
                        Personality::Fighter | Personality::Trader => ASTEROID_HIT_OTHER,
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
                            reward: HEALTH_BONUS_PER_EVENT * h_frac,
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
        // Clear hostility against ship's own faction
        if let Some(faction) = ship.data.clone().faction {
            hostility.0.remove(&faction);
        }
        // Copy the hostility onto the ship
        ship.enemies = hostility.0.clone();
    }
}
