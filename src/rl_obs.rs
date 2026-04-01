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

/// Total length of the observation vector (computed from slot sizes below).
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
pub const SLOT_IS_CURRENT_TARGET: usize = CORE_BLOCK_START + 4; // 9
pub const SLOT_PROXIMITY: usize = CORE_BLOCK_START + 5;   // 10
pub const SLOT_PURSUIT_ANGLE: usize = CORE_BLOCK_START + 6;     // 11
pub const SLOT_PURSUIT_INDICATOR: usize = CORE_BLOCK_START + 7; // 12
pub const SLOT_FIRE_ANGLE: usize = CORE_BLOCK_START + 8;        // 13
pub const SLOT_FIRE_INDICATOR: usize = CORE_BLOCK_START + 9;    // 14
pub const SLOT_IN_RANGE: usize = CORE_BLOCK_START + 10;         // 15
/// Number of core features.
pub const CORE_FEAT_SIZE: usize = 11; // rel_pos(2)+rel_vel(2)+is_current_target+proximity+pursuit_angle+pursuit_indicator+fire_angle+fire_indicator+in_range

// ── Block 4: type-specific features ──────────────────────────────────────
/// Offset of the type-specific block (value + entity-kind features).
pub const TYPE_BLOCK_START: usize = CORE_BLOCK_START + CORE_FEAT_SIZE; // 16
/// Offset of `value` (first float of type-specific block).
pub const SLOT_VALUE: usize = TYPE_BLOCK_START; // 16
/// Offset of entity-kind features (after value).
pub const SLOT_TYPE_SPECIFIC: usize = TYPE_BLOCK_START + 1; // 17
/// Number of entity-kind feature floats (padded to max across all types).
pub const TYPE_SPECIFIC_SIZE: usize = 9;
/// Total size of block 4 (value + type_specific).
pub const TYPE_BLOCK_SIZE: usize = 1 + TYPE_SPECIFIC_SIZE; // 10

/// Floats per entity slot (sum of all blocks).
pub const SLOT_SIZE: usize = TYPE_ONEHOT_SIZE + 1 + CORE_FEAT_SIZE + TYPE_BLOCK_SIZE; // = 26

/// Floats for the self-state block.
pub const SELF_SIZE: usize = 10; // health, speed, vel_cos, vel_sin, ang_vel, cargo, ammo, personality(3)

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
}

/// Planet-specific features.
#[derive(Clone, Default)]
pub struct PlanetSlotData {
    /// Total value of selling the ship's current cargo at this planet.
    pub cargo_sale_value: f32,
    /// Whether the ship can replenish ammo here.
    pub has_ammo: f32,
    /// Best commodity margin (most negative = best trade opportunity).
    pub commodity_margin: f32,
}

/// Asteroid-specific features.
#[derive(Clone, Default)]
pub struct AsteroidSlotData {
    /// Size of the asteroid (proxy for durability).
    pub size: f32,
    /// Expected value of shattering (avg_price * expected_quantity).
    pub value: f32,
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
    /// Whether this entity is the ship's current target.
    pub is_current_target: bool,
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

/// Action space: (turn_idx, thrust_idx, fire_primary, fire_secondary, target_idx)
/// - turn_idx:      0 = left, 1 = straight, 2 = right
/// - thrust_idx:    0 = no thrust, 1 = thrust
/// - fire_primary:  0 = no, 1 = yes
/// - fire_secondary: 0 = no, 1 = yes  (which weapon fires is chosen deterministically)
/// - target_idx:    index into entity slots (0..N_OBJECTS-1), or N_OBJECTS = "no target"
pub type DiscreteAction = (u8, u8, u8, u8, u8);

/// Map a bearing angle (convention from `ai_ships::angle_to_controls`) to a
/// discrete `(turn_idx, thrust_idx)` pair.
///
/// Positive angle = target is to the left; negative = to the right.
pub fn angle_to_discrete(angle: f32) -> (u8, u8) {
    const PI_THIRD: f32 = PI / 3.0;
    if angle > PI_THIRD {
        (0, 0) // turn left, no thrust
    } else if angle > 0.0 {
        (0, 1) // turn left, thrust
    } else if angle > -PI_THIRD {
        (2, 1) // turn right, thrust
    } else {
        (2, 0) // turn right, no thrust
    }
}

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
/// * `obj_vel`   – for **navigation** pass `(target_vel − ship_vel)` (relative);
///                 for **firing** pass target's absolute velocity in ego frame.
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

    // -- Self state (10 floats) -----------------------------------------------
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
        .any(|s| s.is_current_target && s.core.entity_type == 0);
    let ammo_frac = select_secondary_weapon(input.ship, target_is_ship)
        .map(|(_, ammo)| (ammo as f32 / MAX_AMMO_REF).min(1.0))
        .unwrap_or(0.0);

    let personality = personality_onehot(input.personality);

    obs.push(health_frac.clamp(0.0, 1.0));
    obs.push(speed_frac.clamp(0.0, 2.0)); // allow brief over-speed
    obs.push(vel_cos);
    obs.push(vel_sin);
    obs.push((input.angular_velocity / MAX_ANG_SPEED).clamp(-1.0, 1.0));
    obs.push(cargo_frac.clamp(0.0, 1.0));
    obs.push(ammo_frac);
    obs.extend_from_slice(&personality);
    debug_assert_eq!(obs.len(), SELF_SIZE);

    // Context for angle computations.
    let max_speed = input.ship.data.max_speed.max(f32::EPSILON);
    let ego_vel = rotate_to_ego(input.velocity, sin_a, cos_a);

    // -- Entity slots ---------------------------------------------------------
    debug_assert!(
        input.entity_slots.len() <= N_ENTITY_SLOTS,
        "Too many entity slots: {} > {N_ENTITY_SLOTS}",
        input.entity_slots.len()
    );
    for slot in &input.entity_slots {
        encode_slot(
            Some(slot),
            max_speed,
            input.primary_weapon_speed,
            input.primary_weapon_range,
            ego_vel,
            &mut obs,
        );
    }
    // Pad remaining slots with zeros.
    for _ in input.entity_slots.len()..N_ENTITY_SLOTS {
        encode_slot(None, max_speed, 0.0, 0.0, ego_vel, &mut obs);
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
    max_speed: f32,
    weapon_speed: f32,
    weapon_range: f32,
    ego_vel: [f32; 2],
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
    buf[SLOT_IS_CURRENT_TARGET] = if s.is_current_target { 1.0 } else { 0.0 };

    let dist = (rel_pos[0].powi(2) + rel_pos[1].powi(2)).sqrt();
    buf[SLOT_PROXIMITY] = PROXIMITY_SCALE / (dist + PROXIMITY_SCALE);

    let pursuit = intercept_angle(max_speed, rel_pos, rel_vel);
    buf[SLOT_PURSUIT_ANGLE] = pursuit.unwrap_or(0.0);
    buf[SLOT_PURSUIT_INDICATOR] = aim_indicator(pursuit);

    let abs_vel = [rel_vel[0] + ego_vel[0], rel_vel[1] + ego_vel[1]];
    let fire = intercept_angle(weapon_speed, rel_pos, abs_vel);
    buf[SLOT_FIRE_ANGLE] = fire.unwrap_or(0.0);
    buf[SLOT_FIRE_INDICATOR] = aim_indicator(fire);
    buf[SLOT_IN_RANGE] = if weapon_range > 0.0 && dist <= weapon_range {
        1.0
    } else {
        0.0
    };

    buf[SLOT_VALUE] = s.value;

    // Type-specific (zero-padded to TYPE_SPECIFIC_SIZE).
    let ts = &mut buf[SLOT_TYPE_SPECIFIC..];
    match &s.kind {
        EntityKind::Ship(ship) => {
            ts[0] = ship.max_health;
            ts[1] = ship.health;
            ts[2] = ship.max_speed;
            ts[3] = ship.torque;
            ts[4] = ship.is_hostile;
            ts[5] = ship.should_engage;
            let p = personality_onehot(&ship.personality);
            ts[6..9].copy_from_slice(&p);
        }
        EntityKind::Planet(planet) => {
            ts[0] = planet.cargo_sale_value;
            ts[1] = planet.has_ammo;
            ts[2] = planet.commodity_margin;
        }
        EntityKind::Asteroid(asteroid) => {
            ts[0] = asteroid.size;
            ts[1] = asteroid.value;
        }
        EntityKind::Pickup(pickup) => {
            ts[0] = pickup.value;
        }
    }

    obs.extend_from_slice(&buf);
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

