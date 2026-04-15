/// Observation encoding for RL trajectory collection.
///
/// All observations are ego-centric: positions and velocities are expressed
/// relative to the ship and rotated into the ship's local frame, where +x is
/// the ship's forward direction.
use std::f32::consts::PI;

use crate::ship::{Personality, Ship};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Total length of the observation vector (self + entity slots only).
/// Projectile data is stored in a separate tensor.
pub const OBS_DIM: usize = SELF_SIZE
    + (N_ENTITY_SLOTS) * SLOT_SIZE;

/// Number of entity slots (one per bucket position).
pub const N_ENTITY_SLOTS: usize =
    K_PLANETS + K_ASTEROIDS + K_HOSTILE_SHIPS + K_FRIENDLY_SHIPS + K_PICKUPS;

/// Detection radius (must match ai_ships.rs DETECTION_RADIUS).
pub const DETECTION_RADIUS: f32 = 2000.0;

/// Length scale for the proximity feature: K / (distance + K).
/// Gives proximity = 0.5 at distance K, ~0.2 at the detection edge.
const PROXIMITY_SCALE: f32 = DETECTION_RADIUS / 4.0;

/// Reference ammo count for ammo normalisation.
const MAX_AMMO_REF: f32 = 50.0;

/// Maximum angular speed (Avian2D cap set in ship_bundle: 4π rad/s).
const MAX_ANG_SPEED: f32 = 4.0 * PI;

// Entity bucket sizes
pub const K_PLANETS: usize = 2;
pub const K_ASTEROIDS: usize = 3;
pub const K_HOSTILE_SHIPS: usize = 3;
pub const K_FRIENDLY_SHIPS: usize = 2;
pub const K_PICKUPS: usize = 2;

// Projectile bucket sizes (stored in a separate tensor, not in entity slots)
/// Nearest projectiles fired by other ships.
pub const K_OTHER_PROJECTILES: usize = 5;
/// Tracer projectiles fired by this ship (tracked across their full lifetime).
pub const K_OWN_PROJECTILES: usize = 5;
/// Total projectile slots.
pub const K_PROJECTILES: usize = K_OTHER_PROJECTILES + K_OWN_PROJECTILES;

// ── Entity slot layout (offsets within each SLOT_SIZE block) ─────────────
//
// These constants are the single source of truth for the slot encoding.
// Both `encode_slot` (which writes) and any code that reads back from the
// flat observation (e.g. `choose_target_slot`) must use these offsets.
//
// The slot is divided into 4 contiguous blocks for the forward pass:
//
//   Block 1: type_onehot  (N_ENTITY_TYPES floats)   — selects type-specific embedding
//   Block 2: is_present   (1 float)                  — mask for attention / target head
//   Block 3: core features (CORE_FEAT_SIZE floats)   — shared embedding (same meaning for all types)
//   Block 4: type-specific (TYPE_BLOCK_SIZE floats)  — type-conditioned embedding
//

/// Number of entity types.
pub const N_ENTITY_TYPES: usize = 4;

// ── Block 1: type one-hot ────────────────────────────────────────────────
/// Offset and size of the type one-hot block.
pub const SLOT_TYPE_ONEHOT: usize = 0;
pub const TYPE_ONEHOT_SIZE: usize = N_ENTITY_TYPES; // 4

// ── Block 2: is_present ──────────────────────────────────────────────────
/// Offset of is_present (1 float).
pub const SLOT_IS_PRESENT: usize = TYPE_ONEHOT_SIZE; // 4

// ── Block 3: core features (type-independent) ────────────────────────────
/// Offset of the core feature block.
pub const CORE_BLOCK_START: usize = SLOT_IS_PRESENT + 1; // 5
/// Offsets within the core block (absolute, for reading from flat obs).
pub const SLOT_REL_POS: usize = CORE_BLOCK_START;         // 5  (2 floats)
pub const SLOT_REL_VEL: usize = CORE_BLOCK_START + 2;     // 7  (2 floats)
pub const SLOT_IS_NAV_TARGET: usize = CORE_BLOCK_START + 4;     // 9
pub const SLOT_IS_WEAPONS_TARGET: usize = CORE_BLOCK_START + 5; // 10
pub const SLOT_PROXIMITY: usize = CORE_BLOCK_START + 6;   // 11
pub const SLOT_PURSUIT_ANGLE: usize = CORE_BLOCK_START + 7;     // 12
pub const SLOT_PURSUIT_INDICATOR: usize = CORE_BLOCK_START + 8; // 13
pub const SLOT_FIRE_ANGLE: usize = CORE_BLOCK_START + 9;        // 14
pub const SLOT_FIRE_INDICATOR: usize = CORE_BLOCK_START + 10;   // 15
pub const SLOT_IN_RANGE: usize = CORE_BLOCK_START + 11;         // 16
// PD-based pursuit features (from `optimal_control::pursuit_features_ego`).
pub const SLOT_PD_TARGET_ANGLE: usize = CORE_BLOCK_START + 12;  // 17
pub const SLOT_PD_ALIGNMENT: usize = CORE_BLOCK_START + 13;     // 18
pub const SLOT_PD_THRUST_PROB: usize = CORE_BLOCK_START + 14;   // 19
/// Number of core features.
pub const CORE_FEAT_SIZE: usize = 15; // 12 prior + 3 PD features (target_angle, alignment, thrust_prob)

// ── Block 4: type-specific features ──────────────────────────────────────
/// Offset of the type-specific block (value + entity-kind features).
pub const TYPE_BLOCK_START: usize = CORE_BLOCK_START + CORE_FEAT_SIZE; // 16
/// Offset of `value` (first float of type-specific block).
pub const SLOT_VALUE: usize = TYPE_BLOCK_START; // 16
/// Offset of entity-kind features (after value).
pub const SLOT_TYPE_SPECIFIC: usize = TYPE_BLOCK_START + 1; // 17
/// Number of entity-kind feature floats (padded to max across all types).
pub const TYPE_SPECIFIC_SIZE: usize = 15;
/// Total size of block 4 (value + type_specific).
pub const TYPE_BLOCK_SIZE: usize = 1 + TYPE_SPECIFIC_SIZE; // 16

// ── Ship type-specific feature indices (local to SLOT_TYPE_SPECIFIC) ─────
pub const SHIP_IS_HOSTILE: usize = 4;
pub const SHIP_SHOULD_ENGAGE: usize = 5;
/// Ship's current `health / max_health`.
pub const SHIP_HEALTH_FRAC: usize = 10;
/// Ship's primary-weapon range (world units) — helps assess threat reach.
pub const SHIP_WEAPON_RANGE: usize = 11;
/// 1.0 if this ship's weapons_target is the observing ship, else 0.0.
pub const SHIP_IS_TARGETING_ME: usize = 12;
/// Ship's thrust (forward force, raw) — affects agility along with torque.
pub const SHIP_THRUST: usize = 13;
/// Ship's primary shots-per-second (1 / effective_cooldown).
pub const SHIP_PRIMARY_FIRE_RATE: usize = 14;

// ── Planet type-specific feature indices (local to SLOT_TYPE_SPECIFIC) ───
pub const PLANET_CARGO_PROFIT_VALUE: usize = 0;
pub const PLANET_HAS_AMMO: usize = 1;
pub const PLANET_COMMODITY_MARGIN: usize = 2;
/// 1.0 if the ship landed at this planet within the recent-visited cooldown.
pub const PLANET_IS_RECENTLY_VISITED: usize = 3;

// ── Type one-hot indices ─────────────────────────────────────────────────
pub const TYPE_IDX_SHIP: usize = 0;
pub const TYPE_IDX_ASTEROID: usize = 1;
pub const TYPE_IDX_PLANET: usize = 2;
pub const TYPE_IDX_PICKUP: usize = 3;

/// Floats per entity slot (sum of all blocks).
pub const SLOT_SIZE: usize = TYPE_ONEHOT_SIZE + 1 + CORE_FEAT_SIZE + TYPE_BLOCK_SIZE; // = 26

// ── Self-state block offsets ─────────────────────────────────────────────
// Offsets within the self-state block. Must be kept contiguous; `SELF_SIZE`
// below is derived from the final offset + its size.
pub const SELF_HEALTH_FRAC: usize = 0;
pub const SELF_SPEED_FRAC: usize = 1;
pub const SELF_VEL_COS: usize = 2;
pub const SELF_VEL_SIN: usize = 3;
pub const SELF_ANG_VEL: usize = 4;
pub const SELF_CARGO_FRAC: usize = 5;
pub const SELF_AMMO_FRAC: usize = 6;
pub const SELF_CREDITS_NORM: usize = 7;
pub const SELF_DISTRESSED: usize = 8;
pub const SELF_PERSONALITY: usize = 9;
pub const SELF_PERSONALITY_SIZE: usize = 3;
/// Normalised `stop_angle` from `optimal_control::control_features`: how far
/// the ship would rotate under max braking before stopping, divided by π.
pub const SELF_STOP_ANGLE: usize = SELF_PERSONALITY + SELF_PERSONALITY_SIZE; // 12

// ── Own ship class stats (absolute, so the agent knows its own "class") ──
// These are the raw counterparts of the HEALTH_FRAC / SPEED_FRAC features —
// the agent can infer relative matchup strength against other ships whose
// equivalent raw stats are exposed in their entity slot.
pub const SELF_MAX_HEALTH: usize = SELF_STOP_ANGLE + 1; // 13
pub const SELF_MAX_SPEED: usize = SELF_MAX_HEALTH + 1; // 14
pub const SELF_THRUST: usize = SELF_MAX_SPEED + 1; // 15
pub const SELF_TORQUE: usize = SELF_THRUST + 1; // 16
pub const SELF_PRIMARY_RANGE: usize = SELF_TORQUE + 1; // 17
pub const SELF_PRIMARY_SPEED: usize = SELF_PRIMARY_RANGE + 1; // 18

// ── Weapon stats (self) ──────────────────────────────────────────────────
/// Primary weapon cooldown fraction remaining (1.0 = just fired, 0.0 = ready).
pub const SELF_PRIMARY_COOLDOWN: usize = SELF_PRIMARY_SPEED + 1; // 19
/// Secondary weapon cooldown fraction remaining.
pub const SELF_SECONDARY_COOLDOWN: usize = SELF_PRIMARY_COOLDOWN + 1; // 20
/// Primary weapon damage normalised by `WEAPON_DAMAGE_REF`.
pub const SELF_PRIMARY_DAMAGE: usize = SELF_SECONDARY_COOLDOWN + 1; // 21
/// Secondary weapon range normalised by `WEAPON_RANGE_REF`.
pub const SELF_SECONDARY_RANGE: usize = SELF_PRIMARY_DAMAGE + 1; // 22
/// Secondary weapon damage normalised by `WEAPON_DAMAGE_REF`.
pub const SELF_SECONDARY_DAMAGE: usize = SELF_SECONDARY_RANGE + 1; // 23
/// Secondary weapon speed normalised by `WEAPON_SPEED_REF`.
pub const SELF_SECONDARY_SPEED: usize = SELF_SECONDARY_DAMAGE + 1; // 24
/// Primary shots-per-second (= 1 / effective_cooldown_seconds). Already
/// accounts for multi-barrel ships (effective cooldown = base / number).
pub const SELF_PRIMARY_FIRE_RATE: usize = SELF_SECONDARY_SPEED + 1; // 25
/// Secondary shots-per-second.
pub const SELF_SECONDARY_FIRE_RATE: usize = SELF_PRIMARY_FIRE_RATE + 1; // 26

// ── Current target type one-hots (self) ──────────────────────────────────
// 5-wide one-hot: ship, asteroid, planet, pickup, none.
pub const TARGET_TYPE_SIZE: usize = 5;
pub const TARGET_TYPE_IDX_SHIP: usize = 0;
pub const TARGET_TYPE_IDX_ASTEROID: usize = 1;
pub const TARGET_TYPE_IDX_PLANET: usize = 2;
pub const TARGET_TYPE_IDX_PICKUP: usize = 3;
pub const TARGET_TYPE_IDX_NONE: usize = 4;

pub const SELF_NAV_TARGET_TYPE: usize = SELF_SECONDARY_SPEED + 1; // 19
pub const SELF_WEP_TARGET_TYPE: usize = SELF_NAV_TARGET_TYPE + TARGET_TYPE_SIZE; // 24

/// Floats for the self-state block.
pub const SELF_SIZE: usize = SELF_WEP_TARGET_TYPE + TARGET_TYPE_SIZE; // 29

/// Reference damage for weapon-damage normalisation (≈ max single-hit damage).
pub const WEAPON_DAMAGE_REF: f32 = 50.0;
/// Reference range for weapon-range normalisation (world units).
pub const WEAPON_RANGE_REF: f32 = 800.0;
/// Reference speed for weapon-speed normalisation (world units / s).
pub const WEAPON_SPEED_REF: f32 = 500.0;

// ── Projectile slot layout ─────────────────────────────────────────────
//
// Projectile slots are separate from entity slots. They cannot be targeted.
// Layout: is_present(1) + core(5) + features(5) = 11 floats.
//
//   is_present    (1 float)
//   rel_pos       (2 floats) — relative position in ego frame
//   rel_vel       (2 floats) — relative velocity in ego frame
//   proximity     (1 float)  — K / (distance + K)
//   is_ours       (1 float)  — 1.0 if fired by this ship, 0.0 otherwise
//   is_guided     (1 float)  — 1.0 if guided missile
//   speed_frac    (1 float)  — projectile speed / max_speed (ship's)
//   damage_norm   (1 float)  — damage / 50 (reference scale)
//   lifetime_frac (1 float)  — remaining lifetime / initial lifetime
//   collision_ind (1 float)  — 0→1 indicator of collision likelihood

/// Floats per projectile slot.
/// Layout: is_present + rel_pos(2) + rel_vel(2) + proximity + is_ours +
/// is_guided + speed_frac + damage_norm + lifetime_frac + collision_ind +
/// is_tracking_me = 13.
pub const PROJ_SLOT_SIZE: usize = 13;

/// Reference damage for normalisation.
const PROJ_DAMAGE_REF: f32 = 50.0;

// ---------------------------------------------------------------------------
// Data structures passed from the Bevy system to the encoder
// ---------------------------------------------------------------------------

/// Core spatial data shared by all entity types.
#[derive(Clone, Default)]
pub struct CoreSlotData {
    /// Relative position in ego frame (raw world units).
    pub rel_pos: [f32; 2],
    /// Relative velocity in ego frame (raw world units/s).
    pub rel_vel: [f32; 2],
    /// Entity type: 0=Ship, 1=Asteroid, 2=Planet, 3=Pickup.
    pub entity_type: u8,
}

/// Ship-specific features.
#[derive(Clone, Default)]
pub struct ShipSlotData {
    pub max_health: f32,
    pub health: f32,
    pub max_speed: f32,
    pub torque: f32,
    /// 1.0 = hostile, 0.0 = unknown / neutral, -1.0 = friendly.
    pub is_hostile: f32,
    /// 1.0 = should_engage, 0.0 = unknown / neutral, -1.0 = friendly.
    pub should_engage: f32,
    pub personality: Personality,
    /// Distressed level (1.0 = just hit, decays toward 0).
    pub distressed: f32,
    /// Primary-weapon range (world units), 0.0 if unknown/unarmed.
    pub primary_weapon_range: f32,
    /// 1.0 if this ship's weapons_target is the observing ship, else 0.0.
    pub is_targeting_me: f32,
    /// Forward thrust (raw); affects agility.
    pub thrust: f32,
    /// Primary-weapon shots-per-second (1 / effective_cooldown_seconds), 0 if
    /// unarmed or cooldown unknown.
    pub primary_fire_rate: f32,
}

/// Planet-specific features.
#[derive(Clone, Default)]
pub struct PlanetSlotData {
    /// Profit (sale value - acquisition cost) the ship would realise by
    /// selling its current cargo at this planet. May be negative.
    pub cargo_profit_value: f32,
    /// Whether the ship can replenish ammo here.
    pub has_ammo: f32,
    /// Best commodity margin (most negative = best trade opportunity).
    pub commodity_margin: f32,
    /// 1.0 if the ship landed at this planet within the cooldown window;
    /// 0.0 otherwise.
    pub is_recently_visited: f32,
}

/// Asteroid-specific features.
#[derive(Clone, Default)]
pub struct AsteroidSlotData {
    /// Size of the asteroid (proxy for durability).
    pub size: f32,
    /// Expected value of shattering (avg_price * expected_quantity).
    pub value: f32,
    /// 0→1 indicator: likelihood of a collision with the ship given current trajectories.
    pub collision_indicator: f32,
}

/// Pickup-specific features.
#[derive(Clone, Default)]
pub struct PickupSlotData {
    /// Value of the pickup (avg_price * quantity).
    pub value: f32,
}

/// Type-specific features for an entity slot.
#[derive(Clone)]
pub enum EntityKind {
    Ship(ShipSlotData),
    Asteroid(AsteroidSlotData),
    Planet(PlanetSlotData),
    Pickup(PickupSlotData),
}

impl Default for EntityKind {
    fn default() -> Self {
        EntityKind::Ship(ShipSlotData::default())
    }
}

/// Unified entity slot data, used for both nearby entities and the target.
///
/// All vectors are already in the ship's ego frame (raw, un-normalised).
/// All entries in the `entity_slots` vec of `ObsInput` are real (present)
/// entities. Empty/absent slots are represented by the vec being shorter
/// than `N_ENTITY_SLOTS`; the encoder pads the remainder with zeros.
#[derive(Clone, Default)]
pub struct EntitySlotData {
    pub core: CoreSlotData,
    pub kind: EntityKind,
    /// Value of this entity (e.g. cargo sale value for planets, pickup value, etc.)
    pub value: f32,
    /// Whether this entity is the ship's navigation target.
    pub is_nav_target: bool,
    /// Whether this entity is the ship's weapons target.
    pub is_weapons_target: bool,
}

/// Projectile slot data (separate from entity slots).
#[derive(Clone, Default)]
pub struct ProjectileSlotData {
    /// Relative position in ego frame (raw world units).
    pub rel_pos: [f32; 2],
    /// Relative velocity in ego frame (raw world units/s).
    pub rel_vel: [f32; 2],
    /// 1.0 if fired by this ship, 0.0 otherwise.
    pub is_ours: f32,
    /// 1.0 if guided missile, 0.0 otherwise.
    pub is_guided: f32,
    /// Projectile speed (world units/s).
    pub speed: f32,
    /// Damage dealt on hit.
    pub damage: f32,
    /// Remaining lifetime (seconds).
    pub lifetime_remaining: f32,
    /// Initial lifetime of this weapon type (seconds), for normalisation.
    pub lifetime_max: f32,
    /// Radius of the target (ship or asteroid) for collision prediction.
    pub target_radius: f32,
    /// 1.0 if this is a guided missile whose target is the observing ship,
    /// else 0.0.  Huge signal for evasion.
    pub is_tracking_me: f32,
}

/// All pre-processed inputs needed by `encode_observation`.
///
/// The Bevy system is responsible for collecting entity data, performing the
/// spatial query, rotating vectors into the ego frame, and normalising values
/// before populating this struct.
pub struct ObsInput<'a> {
    pub personality: &'a Personality,
    pub ship: &'a Ship,
    /// Ship velocity in world space (used to derive speed and heading angle).
    pub velocity: [f32; 2],
    /// Angular velocity in rad/s.
    pub angular_velocity: f32,
    /// Unit vector of the ship's forward direction in world space.
    pub ship_heading: [f32; 2],
    /// Nearby entities, ordered by bucket (planets, asteroids, hostile ships,
    /// friendly ships, pickups). Length may be less than `N_ENTITY_SLOTS`;
    /// the encoder pads the remainder with zero slots.
    pub entity_slots: Vec<EntitySlotData>,
    /// Speed of the ship's primary weapon (m/s), used to compute the fire lead angle.
    /// 0.0 if the ship has no primary weapon.
    pub primary_weapon_speed: f32,
    /// Range of the ship's primary weapon (m), used to compute the in-range flag.
    /// 0.0 if the ship has no primary weapon.
    pub primary_weapon_range: f32,
    /// Primary weapon damage (raw); 0.0 if unarmed.
    pub primary_weapon_damage: f32,
    /// Primary weapon cooldown fraction remaining in [0, 1] (1.0 = just fired).
    pub primary_cooldown_frac: f32,
    /// Secondary weapon range / damage / speed / cooldown.
    /// 0.0 if the ship has no secondary weapon.
    pub secondary_weapon_range: f32,
    pub secondary_weapon_damage: f32,
    pub secondary_weapon_speed: f32,
    pub secondary_cooldown_frac: f32,
    /// Primary shots-per-second = 1 / effective_cooldown_seconds.
    pub primary_fire_rate: f32,
    /// Secondary shots-per-second.
    pub secondary_fire_rate: f32,
    /// Current nav-target entity type (0..=3 for ship/asteroid/planet/pickup,
    /// or 4 for none).  Duplicates information from entity_slots but provides
    /// a direct shortcut to the policy.
    pub nav_target_type: u8,
    /// Current weapons-target entity type (same encoding as `nav_target_type`).
    pub weapons_target_type: u8,
    /// Per-ship-type credit scale for normalising the credit observation.
    pub credit_scale: f32,
    /// Current distressed level (1.0 = just hit, decays toward 0).
    pub distressed: f32,
    /// Nearest projectiles from other ships (up to `K_OTHER_PROJECTILES`).
    pub other_projectile_slots: Vec<ProjectileSlotData>,
    /// Tracer projectiles fired by this ship (up to `K_OWN_PROJECTILES`).
    pub own_projectile_slots: Vec<ProjectileSlotData>,
}

// ---------------------------------------------------------------------------
// Ego-frame rotation helpers (public for use by the Bevy system)
// ---------------------------------------------------------------------------

/// Returns `(sin_a, cos_a)` for the ego-frame rotation matrix.
///
/// Applying `rotate_to_ego(v, sin_a, cos_a)` maps a world-space 2-D vector
/// `v` into the ship's local frame where +x is the ship's forward direction.
/// This matches the rotation used in `classic_ai_control`.
pub fn ego_frame_sincos(ship_heading: [f32; 2]) -> (f32, f32) {
    let frame_angle = -ship_heading[1].atan2(ship_heading[0]);
    frame_angle.sin_cos()
}

/// Rotate a world-space 2-D offset into the ship's ego frame.
pub fn rotate_to_ego(v: [f32; 2], sin_a: f32, cos_a: f32) -> [f32; 2] {
    [v[0] * cos_a - v[1] * sin_a, v[0] * sin_a + v[1] * cos_a]
}

// ---------------------------------------------------------------------------
// Secondary-weapon helpers
// ---------------------------------------------------------------------------

/// Choose which secondary weapon to use for the observation and deterministic
/// firing in RL mode.
///
/// Preference order:
/// 1. Guided weapons when the target is a ship.
/// 2. The weapon with the most remaining ammo.
///
/// Returns `None` if no secondary weapon has ammo.
pub fn select_secondary_weapon<'a>(ship: &'a Ship, target_is_ship: bool) -> Option<(&'a str, u32)> {
    ship.weapon_systems
        .secondary
        .iter()
        .filter(|(_, ws)| ws.ammo_quantity.map(|a| a > 0).unwrap_or(false))
        .max_by_key(|(_, ws)| {
            let guided_bonus = if ws.weapon.guided && target_is_ship {
                1_000_000u32
            } else {
                0
            };
            guided_bonus + ws.ammo_quantity.unwrap_or(0)
        })
        .map(|(name, ws)| (name.as_str(), ws.ammo_quantity.unwrap_or(0)))
}

// ---------------------------------------------------------------------------
// Action-space helpers
// ---------------------------------------------------------------------------

/// Action space: (turn, thrust, fire_primary, fire_secondary, nav_target_idx, weapons_target_idx)
/// - turn_idx:           0 = left, 1 = straight, 2 = right
/// - thrust_idx:         0 = no thrust, 1 = thrust
/// - fire_primary:       0 = no, 1 = yes
/// - fire_secondary:     0 = no, 1 = yes  (which weapon fires is chosen deterministically)
/// - nav_target_idx:     index into entity slots (0..N_OBJECTS-1), or N_OBJECTS = "no target"
/// - weapons_target_idx: index into entity slots (0..N_OBJECTS-1), or N_OBJECTS = "no target"
pub type DiscreteAction = (u8, u8, u8, u8, u8, u8);

/// Convert discrete indices to continuous `(thrust, turn)` floats suitable for
/// `ShipCommand`.  Note: the parameter order here is `(thrust_idx, turn_idx)`
/// which is the **opposite** of the `DiscreteAction` tuple order `(turn, thrust, ...)`.
pub fn discrete_to_controls(thrust_idx: u8, turn_idx: u8) -> (f32, f32) {
    let thrust = if thrust_idx == 1 { 1.0_f32 } else { 0.0_f32 };
    let turn = match turn_idx {
        0 => -1.0_f32,
        2 => 1.0_f32,
        _ => 0.0_f32,
    };
    (thrust, turn)
}

/// Map continuous controls (from the rule-based system) to the nearest discrete
/// action index. Useful for BC data collection.
pub fn controls_to_discrete(thrust: f32, turn: f32) -> (u8, u8) {
    let thrust_idx = if thrust > 0.5 { 1u8 } else { 0u8 };
    let turn_idx = if turn < -0.25 {
        0u8
    } else if turn > 0.25 {
        2u8
    } else {
        1u8
    };
    (thrust_idx, turn_idx)
}

// ---------------------------------------------------------------------------
// Private angle-computation helpers
// ---------------------------------------------------------------------------

/// Compute the bearing angle (radians, in ego frame) to fire/navigate toward
/// an intercept point.  Returns `None` when no positive-time intercept exists.
///
/// * `proj_vel`  – speed of the projectile or ship.
/// * `obj_pos`   – target position in ego frame.
/// * `obj_vel`   – target velocity relative to the ship (`target_vel − ship_vel`)
///                 in ego frame, for both navigation and firing (projectiles
///                 inherit the ship's velocity).
///
/// The angle convention matches `angle_to_hit` in `utils.rs`:
/// 0 = directly ahead, positive = left, negative = right, range `(−π, π]`.
fn intercept_angle(proj_vel: f32, obj_pos: [f32; 2], obj_vel: [f32; 2]) -> Option<f32> {
    let [px, py] = obj_pos;
    let [vx, vy] = obj_vel;
    // Quadratic in t: |obj_pos + t·obj_vel|² = (proj_vel·t)²
    let a = vx * vx + vy * vy - proj_vel * proj_vel;
    if a == 0.0 {
        return None;
    }
    let b = 2.0 * (px * vx + py * vy);
    let c = px * px + py * py;
    let disc_sq = b * b - 4.0 * a * c;
    if disc_sq < 0.0 {
        return None;
    }
    let disc = disc_sq.sqrt();
    let t1 = (-b + disc) / (2.0 * a);
    let t2 = (-b - disc) / (2.0 * a);
    if t1 < 0.0 && t2 < 0.0 {
        return None;
    }
    let t = match (t1 < 0.0, t2 < 0.0) {
        (true, _) => t2,
        (_, true) => t1,
        _ => f32::min(t1, t2),
    };
    let cx = px + t * vx;
    let cy = py + t * vy;
    Some(cy.atan2(cx))
}

/// Smooth 0 → 1 indicator of how well the ship is aimed at an intercept angle.
///
/// * Returns **1.0** when `angle_error = 0` (perfectly aimed).
/// * Returns **0.0** when `angle_error ≥ π/2` or when `angle` is `None`.
/// * Uses a quadratic ramp in between.
fn aim_indicator(angle: Option<f32>) -> f32 {
    let error = match angle {
        None => return 0.0,
        Some(a) => ((a + PI).rem_euclid(2.0 * PI) - PI).abs(),
    };
    if error < PI / 2.0 {
        4.0 * (error - PI / 2.0).powi(2) / (PI * PI)
    } else {
        0.0
    }
}

/// Collision-course indicator for an approaching entity.
///
/// Given the entity's relative position and velocity in the observer's ego
/// frame, computes the closest future approach distance `d`, then returns
/// `2 r / (d + r)` where `r = combined_radius` (sum of both entities' radii).
///
/// Value interpretation:
/// - `= 2.0` direct hit (closest approach distance = 0)
/// - `= 1.0` grazing (closest approach = combined_radius)
/// - `→ 0.0` as entities miss by many radii
/// - `= 0.0` when the entity is receding with a closest-approach-in-the-past,
///   unless it's still close (in which case `d = current distance`)
pub fn collision_indicator(rel_pos: [f32; 2], rel_vel: [f32; 2], combined_radius: f32) -> f32 {
    let [px, py] = rel_pos;
    let [vx, vy] = rel_vel;
    let r = combined_radius.max(1.0);
    let current_dist_sq = px * px + py * py;
    let v_sq = vx * vx + vy * vy;

    if v_sq < f32::EPSILON {
        // Stationary — use current distance.
        let d = current_dist_sq.sqrt();
        return 2.0 * r / (d + r);
    }

    let t_closest = -(px * vx + py * vy) / v_sq;
    let d = if t_closest < 0.0 {
        // Closest point in the past — entity is receding; future closest is now.
        current_dist_sq.sqrt()
    } else {
        let cx = px + t_closest * vx;
        let cy = py + t_closest * vy;
        (cx * cx + cy * cy).sqrt()
    };
    2.0 * r / (d + r)
}

// ---------------------------------------------------------------------------
// Core encoder
// ---------------------------------------------------------------------------

/// Build a fixed-length observation vector from pre-processed `ObsInput`.
///
/// Layout (total = OBS_DIM):
/// ```text
/// [self: SELF_SIZE]
/// [target: SLOT_SIZE] [planets: K_PLANETS×SLOT_SIZE] [asteroids: K_ASTEROIDS×SLOT_SIZE]
/// [hostile_ships: K_HOSTILE_SHIPS×SLOT_SIZE]
/// [friendly_ships: K_FRIENDLY_SHIPS×SLOT_SIZE]
/// [pickups: K_PICKUPS×SLOT_SIZE]
/// ```
pub fn encode_observation(input: &ObsInput<'_>) -> Vec<f32> {
    let mut obs = Vec::with_capacity(OBS_DIM);

    // -- Self state -----------------------------------------------------------
    let health_frac = input.ship.health as f32 / input.ship.data.max_health.max(1) as f32;

    let vel = input.velocity;
    let speed = (vel[0] * vel[0] + vel[1] * vel[1]).sqrt();
    let speed_frac = speed / input.ship.data.max_speed.max(f32::EPSILON);

    // Velocity direction in ego frame (cos, sin of angle vs forward).
    let (sin_a, cos_a) = ego_frame_sincos(input.ship_heading);
    let ego_vel = rotate_to_ego(vel, sin_a, cos_a);
    let (vel_cos, vel_sin) = if speed > f32::EPSILON {
        (ego_vel[0] / speed, ego_vel[1] / speed)
    } else {
        (1.0, 0.0) // stationary → pretend aligned with heading
    };

    let cargo_used: u16 = input.ship.cargo.values().sum();
    let cargo_frac = cargo_used as f32 / input.ship.data.cargo_space.max(1) as f32;

    let target_is_ship = input
        .entity_slots
        .iter()
        .any(|s| s.is_weapons_target && s.core.entity_type == 0);
    let ammo_frac = select_secondary_weapon(input.ship, target_is_ship)
        .map(|(_, ammo)| (ammo as f32 / MAX_AMMO_REF).min(1.0))
        .unwrap_or(0.0);

    let personality = personality_onehot(input.personality);

    let ctrl = crate::optimal_control::control_features(
        input.angular_velocity,
        input.ship.data.torque,
        input.ship.data.angular_drag,
    );

    let mut self_buf = [0.0_f32; SELF_SIZE];
    self_buf[SELF_HEALTH_FRAC] = health_frac.clamp(0.0, 1.0);
    self_buf[SELF_SPEED_FRAC] = speed_frac.clamp(0.0, 2.0); // allow brief over-speed
    self_buf[SELF_VEL_COS] = vel_cos;
    self_buf[SELF_VEL_SIN] = vel_sin;
    self_buf[SELF_ANG_VEL] = (input.angular_velocity / MAX_ANG_SPEED).clamp(-1.0, 1.0);
    self_buf[SELF_CARGO_FRAC] = cargo_frac.clamp(0.0, 1.0);
    self_buf[SELF_AMMO_FRAC] = ammo_frac;
    // Credits normalised per ship type: scale / (credits + scale).
    let c = input.ship.credits as f32;
    let s = input.credit_scale;
    self_buf[SELF_CREDITS_NORM] = s / (c + s);
    self_buf[SELF_DISTRESSED] = input.distressed;
    self_buf[SELF_PERSONALITY..SELF_PERSONALITY + SELF_PERSONALITY_SIZE]
        .copy_from_slice(&personality);
    // stop_angle is unbounded in principle; normalise by π and clamp.
    self_buf[SELF_STOP_ANGLE] = (ctrl.stop_angle / PI).clamp(-2.0, 2.0);

    // Own ship class stats (raw values — matches what we expose about other
    // ships' slots so the net can compare matchups).
    self_buf[SELF_MAX_HEALTH] = input.ship.data.max_health as f32;
    self_buf[SELF_MAX_SPEED] = input.ship.data.max_speed;
    self_buf[SELF_THRUST] = input.ship.data.thrust;
    self_buf[SELF_TORQUE] = input.ship.data.torque;
    self_buf[SELF_PRIMARY_RANGE] =
        (input.primary_weapon_range / WEAPON_RANGE_REF).clamp(0.0, 4.0);
    self_buf[SELF_PRIMARY_SPEED] =
        (input.primary_weapon_speed / WEAPON_SPEED_REF).clamp(0.0, 4.0);

    // Weapon state + stats.
    self_buf[SELF_PRIMARY_COOLDOWN] = input.primary_cooldown_frac.clamp(0.0, 1.0);
    self_buf[SELF_SECONDARY_COOLDOWN] = input.secondary_cooldown_frac.clamp(0.0, 1.0);
    self_buf[SELF_PRIMARY_DAMAGE] =
        (input.primary_weapon_damage / WEAPON_DAMAGE_REF).clamp(0.0, 4.0);
    self_buf[SELF_SECONDARY_RANGE] =
        (input.secondary_weapon_range / WEAPON_RANGE_REF).clamp(0.0, 4.0);
    self_buf[SELF_SECONDARY_DAMAGE] =
        (input.secondary_weapon_damage / WEAPON_DAMAGE_REF).clamp(0.0, 4.0);
    self_buf[SELF_SECONDARY_SPEED] =
        (input.secondary_weapon_speed / WEAPON_SPEED_REF).clamp(0.0, 4.0);
    self_buf[SELF_PRIMARY_FIRE_RATE] = input.primary_fire_rate;
    self_buf[SELF_SECONDARY_FIRE_RATE] = input.secondary_fire_rate;

    // Current target-type one-hots.
    let nav_idx = (input.nav_target_type as usize).min(TARGET_TYPE_IDX_NONE);
    let wep_idx = (input.weapons_target_type as usize).min(TARGET_TYPE_IDX_NONE);
    self_buf[SELF_NAV_TARGET_TYPE + nav_idx] = 1.0;
    self_buf[SELF_WEP_TARGET_TYPE + wep_idx] = 1.0;
    obs.extend_from_slice(&self_buf);
    debug_assert_eq!(obs.len(), SELF_SIZE);

    // Context for angle computations.
    let max_speed = input.ship.data.max_speed.max(f32::EPSILON);

    // -- Entity slots ---------------------------------------------------------
    debug_assert!(
        input.entity_slots.len() <= N_ENTITY_SLOTS,
        "Too many entity slots: {} > {N_ENTITY_SLOTS}",
        input.entity_slots.len()
    );
    for slot in &input.entity_slots {
        encode_slot(
            Some(slot),
            input,
            max_speed,
            input.primary_weapon_speed,
            input.primary_weapon_range,
            &mut obs,
        );
    }
    // Pad remaining slots with zeros.
    for _ in input.entity_slots.len()..N_ENTITY_SLOTS {
        encode_slot(None, input, max_speed, 0.0, 0.0, &mut obs);
    }

    debug_assert_eq!(
        obs.len(),
        OBS_DIM,
        "Observation size mismatch: got {}",
        obs.len()
    );

    obs
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Encode a single entity slot (SLOT_SIZE floats). `None` → all zeros.
fn encode_slot(
    slot: Option<&EntitySlotData>,
    input: &ObsInput<'_>,
    max_speed: f32,
    weapon_speed: f32,
    weapon_range: f32,
    obs: &mut Vec<f32>,
) {
    let Some(s) = slot else {
        obs.extend(std::iter::repeat(0.0_f32).take(SLOT_SIZE));
        return;
    };
    let rel_pos = s.core.rel_pos;
    let rel_vel = s.core.rel_vel;

    // Write into a fixed-size buffer using the named offsets so that any
    // reordering is caught at compile time via the final length assert.
    let mut buf = [0.0_f32; SLOT_SIZE];

    buf[SLOT_IS_PRESENT] = 1.0;
    buf[SLOT_TYPE_ONEHOT..SLOT_TYPE_ONEHOT + 4]
        .copy_from_slice(&entity_type_onehot(s.core.entity_type));
    buf[SLOT_REL_POS..SLOT_REL_POS + 2].copy_from_slice(&rel_pos);
    buf[SLOT_REL_VEL..SLOT_REL_VEL + 2].copy_from_slice(&rel_vel);
    buf[SLOT_IS_NAV_TARGET] = if s.is_nav_target { 1.0 } else { 0.0 };
    buf[SLOT_IS_WEAPONS_TARGET] = if s.is_weapons_target { 1.0 } else { 0.0 };

    let dist = (rel_pos[0].powi(2) + rel_pos[1].powi(2)).sqrt();
    buf[SLOT_PROXIMITY] = PROXIMITY_SCALE / (dist + PROXIMITY_SCALE);

    let pursuit = intercept_angle(max_speed, rel_pos, rel_vel);
    buf[SLOT_PURSUIT_ANGLE] = pursuit.unwrap_or(0.0);
    buf[SLOT_PURSUIT_INDICATOR] = aim_indicator(pursuit);

    let fire = intercept_angle(weapon_speed, rel_pos, rel_vel);
    buf[SLOT_FIRE_ANGLE] = fire.unwrap_or(0.0);
    buf[SLOT_FIRE_INDICATOR] = aim_indicator(fire);
    buf[SLOT_IN_RANGE] = if weapon_range > 0.0 && dist <= weapon_range {
        1.0
    } else {
        0.0
    };

    // PD-based pursuit features. Damping factor is chosen per entity type to
    // match the rule-based controller's tuning in `ai_ships::compute_ai_action`.
    let damping = match s.core.entity_type as usize {
        TYPE_IDX_PLANET => 1.0,  // landing: critical damping
        TYPE_IDX_PICKUP => 0.2,  // fly-through
        _ => 0.4,                // ship/asteroid pursuit
    };
    let feat = crate::optimal_control::pursuit_features_ego(
        bevy::math::Vec2::new(rel_pos[0], rel_pos[1]),
        bevy::math::Vec2::new(rel_vel[0], rel_vel[1]),
        input.angular_velocity,
        input.ship.data.torque,
        input.ship.data.angular_drag,
        input.ship.data.thrust,
        input.ship.data.max_speed,
        damping,
    );
    buf[SLOT_PD_TARGET_ANGLE] = feat.target_angle;
    buf[SLOT_PD_ALIGNMENT] = feat.alignment;
    buf[SLOT_PD_THRUST_PROB] = feat.thrust_prob;

    buf[SLOT_VALUE] = s.value;

    // Type-specific (zero-padded to TYPE_SPECIFIC_SIZE).
    let ts = &mut buf[SLOT_TYPE_SPECIFIC..];
    match &s.kind {
        EntityKind::Ship(ship) => {
            ts[0] = ship.max_health;
            ts[1] = ship.health;
            ts[2] = ship.max_speed;
            ts[3] = ship.torque;
            ts[SHIP_IS_HOSTILE] = ship.is_hostile;
            ts[SHIP_SHOULD_ENGAGE] = ship.should_engage;
            let p = personality_onehot(&ship.personality);
            ts[6..9].copy_from_slice(&p);
            ts[9] = ship.distressed;
            ts[SHIP_HEALTH_FRAC] = (ship.health / ship.max_health.max(1.0)).clamp(0.0, 1.0);
            ts[SHIP_WEAPON_RANGE] =
                (ship.primary_weapon_range / WEAPON_RANGE_REF).clamp(0.0, 2.0);
            ts[SHIP_IS_TARGETING_ME] = ship.is_targeting_me;
            ts[SHIP_THRUST] = ship.thrust;
            ts[SHIP_PRIMARY_FIRE_RATE] = ship.primary_fire_rate;
        }
        EntityKind::Planet(planet) => {
            ts[0] = planet.cargo_profit_value;
            ts[1] = planet.has_ammo;
            ts[2] = planet.commodity_margin;
            ts[3] = planet.is_recently_visited;
        }
        EntityKind::Asteroid(asteroid) => {
            ts[0] = asteroid.size;
            ts[1] = asteroid.value;
            ts[2] = asteroid.collision_indicator;
        }
        EntityKind::Pickup(pickup) => {
            ts[0] = pickup.value;
        }
    }

    obs.extend_from_slice(&buf);
}

/// Encode all projectile slots into a flat vector (separate from the main obs).
///
/// Layout: `[other_projectiles: K_OTHER_PROJECTILES × PROJ_SLOT_SIZE]
///          [own_projectiles:   K_OWN_PROJECTILES   × PROJ_SLOT_SIZE]`
///
/// Total length = `K_PROJECTILES × PROJ_SLOT_SIZE`.
pub fn encode_projectiles(
    other_projectiles: &[ProjectileSlotData],
    own_projectiles: &[ProjectileSlotData],
    max_speed: f32,
) -> Vec<f32> {
    let total = K_PROJECTILES * PROJ_SLOT_SIZE;
    let mut out = Vec::with_capacity(total);

    for i in 0..K_OTHER_PROJECTILES {
        encode_projectile_slot(other_projectiles.get(i), max_speed, &mut out);
    }
    for i in 0..K_OWN_PROJECTILES {
        encode_projectile_slot(own_projectiles.get(i), max_speed, &mut out);
    }

    debug_assert_eq!(out.len(), total);
    out
}

/// Encode a single projectile slot (PROJ_SLOT_SIZE floats). `None` → all zeros.
fn encode_projectile_slot(
    slot: Option<&ProjectileSlotData>,
    max_speed: f32,
    obs: &mut Vec<f32>,
) {
    let Some(p) = slot else {
        obs.extend(std::iter::repeat(0.0_f32).take(PROJ_SLOT_SIZE));
        return;
    };

    let dist = (p.rel_pos[0].powi(2) + p.rel_pos[1].powi(2)).sqrt();
    let lifetime_frac = if p.lifetime_max > f32::EPSILON {
        p.lifetime_remaining / p.lifetime_max
    } else {
        0.0
    };

    let col_ind = collision_indicator(p.rel_pos, p.rel_vel, p.target_radius);

    obs.push(1.0); // is_present
    obs.push(p.rel_pos[0]);
    obs.push(p.rel_pos[1]);
    obs.push(p.rel_vel[0]);
    obs.push(p.rel_vel[1]);
    obs.push(PROXIMITY_SCALE / (dist + PROXIMITY_SCALE));
    obs.push(p.is_ours);
    obs.push(p.is_guided);
    obs.push(p.speed / max_speed.max(f32::EPSILON));
    obs.push(p.damage / PROJ_DAMAGE_REF);
    obs.push(lifetime_frac.clamp(0.0, 1.0));
    obs.push(col_ind);
    obs.push(p.is_tracking_me);
}

fn entity_type_onehot(entity_type: u8) -> [f32; 4] {
    match entity_type {
        0 => [1.0, 0.0, 0.0, 0.0], // Ship
        1 => [0.0, 1.0, 0.0, 0.0], // Asteroid
        2 => [0.0, 0.0, 1.0, 0.0], // Planet
        3 => [0.0, 0.0, 0.0, 1.0], // Pickup
        _ => [0.0; 4],
    }
}

fn personality_onehot(personality: &Personality) -> [f32; 3] {
    match personality {
        Personality::Miner => [1.0, 0.0, 0.0],
        Personality::Fighter => [0.0, 1.0, 0.0],
        Personality::Trader => [0.0, 0.0, 1.0],
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "tests/rl_obs_tests.rs"]
mod tests;

