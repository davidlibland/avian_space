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
    + TARGET_SLOT_SIZE
    + (K_PLANETS + K_ASTEROIDS + K_HOSTILE_SHIPS + K_FRIENDLY_SHIPS + K_PICKUPS) * ENTITY_SLOT_SIZE;

/// Detection radius (must match ai_ships.rs DETECTION_RADIUS).
pub const DETECTION_RADIUS: f32 = 2000.0;

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

/// Floats per entity slot in a nearby-entity bucket.
pub const ENTITY_SLOT_SIZE: usize = 7; // rel_pos(2) + rel_vel(2) + extra(1) + pursuit_angle(1) + pursuit_indicator(1)

/// Floats for the target slot.
/// present(1) + type_onehot(4) + rel_pos(2) + rel_vel(2) + should_engage(1) + value(1)
/// + pursuit_angle(1) + pursuit_indicator(1) + fire_angle(1) + fire_indicator(1) + in_range(1)
pub const TARGET_SLOT_SIZE: usize = 16;

/// Floats for the self-state block.
pub const SELF_SIZE: usize = 10; // health, speed, vel_cos, vel_sin, ang_vel, cargo, ammo, personality(3)

// ---------------------------------------------------------------------------
// Data structures passed from the Bevy system to the encoder
// ---------------------------------------------------------------------------

/// Pre-processed data for a single nearby entity slot.
///
/// All vectors are already in the ship's ego frame (raw, un-normalised).
#[derive(Clone, Default)]
pub struct EntitySlotData {
    /// Relative position in ego frame (raw world units).
    pub rel_pos: [f32; 2],
    /// Relative velocity in ego frame (raw world units/s).
    pub rel_vel: [f32; 2],
    /// Entity-specific feature:
    /// - Ships: health fraction (health / max_health)
    /// - Planets: 1.0
    /// - Asteroids: 1.0
    /// - Pickups: 1.0
    pub extra: f32,
    /// Ship related data (zero for )
    pub max_health: f32,
    pub health: f32,
    pub max_speed: f32,
    pub torque: f32,
    /// 1.0 = hostile, 0.0 = unknown / neutral, -1.0 = friendly.
    pub is_hostile: f32,
    /// 1.0 = should_engage, 0.0 = unknown / neutral, -1.0 = friendly.
    pub should_engage: f32,
}

/// Pre-processed data for the dedicated target slot.
///
/// All vectors are already in the ship's ego frame (raw, un-normalised).
#[derive(Clone)]
pub struct TargetSlotData {
    /// Entity type: 0=Ship, 1=Asteroid, 2=Planet, 3=Pickup.
    pub entity_type: u8,
    /// Relative position in ego frame (raw world units).
    pub rel_pos: [f32; 2],
    /// Relative velocity in ego frame (raw world units/s).
    pub rel_vel: [f32; 2],
    /// 1.0 = hostile, 0.0 = unknown / neutral, -1.0 = friendly.
    pub is_hostile: f32,
    /// 1.0 = should_engage, 0.0 = unknown / neutral, -1.0 = friendly.
    pub should_engage: f32,
    /// Normalised value hint (e.g. 1.0 for planets, 1.0 for asteroids).
    pub value_hint: f32,
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
    /// Current target, if any (pre-processed into ego frame).
    pub target: Option<TargetSlotData>,
    /// Up to K_PLANETS nearest planets, sorted by distance, in ego frame.
    pub nearby_planets: Vec<EntitySlotData>,
    /// Up to K_ASTEROIDS nearest asteroids, sorted by distance, in ego frame.
    pub nearby_asteroids: Vec<EntitySlotData>,
    /// Up to K_HOSTILE_SHIPS nearest hostile ships, sorted by distance, in ego frame.
    pub nearby_hostile_ships: Vec<EntitySlotData>,
    /// Up to K_FRIENDLY_SHIPS nearest friendly ships, sorted by distance, in ego frame.
    pub nearby_friendly_ships: Vec<EntitySlotData>,
    /// Up to K_PICKUPS nearest pickups, sorted by distance, in ego frame.
    pub nearby_pickups: Vec<EntitySlotData>,
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
/// This matches the rotation used in `simple_ai_control`.
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

/// Action space: (turn_idx, thrust_idx, fire_primary, fire_secondary)
/// - turn_idx:      0 = left, 1 = straight, 2 = right
/// - thrust_idx:    0 = no thrust, 1 = thrust
/// - fire_primary:  0 = no, 1 = yes
/// - fire_secondary: 0 = no, 1 = yes  (which weapon fires is chosen deterministically)
pub type DiscreteAction = (u8, u8, u8, u8);

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
/// [self: SELF_SIZE] [target: TARGET_SLOT_SIZE]
/// [planets: K_PLANETS×ENTITY_SLOT_SIZE] [asteroids: K_ASTEROIDS×ENTITY_SLOT_SIZE]
/// [hostile_ships: K_HOSTILE_SHIPS×ENTITY_SLOT_SIZE]
/// [friendly_ships: K_FRIENDLY_SHIPS×ENTITY_SLOT_SIZE]
/// [pickups: K_PICKUPS×ENTITY_SLOT_SIZE]
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
        .target
        .as_ref()
        .map(|t| t.entity_type == 0)
        .unwrap_or(false);
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

    // Context for angle computations in sub-encoders.
    let max_speed = input.ship.data.max_speed.max(f32::EPSILON);
    // Ship's own velocity in ego frame (needed to recover target absolute velocity
    // from the stored relative velocity for the fire-lead angle).
    let ego_vel = rotate_to_ego(input.velocity, sin_a, cos_a);

    // -- Target slot ----------------------------------------------------------
    encode_target_slot(
        input.target.as_ref(),
        max_speed,
        input.primary_weapon_speed,
        input.primary_weapon_range,
        ego_vel,
        &mut obs,
    );
    debug_assert_eq!(obs.len(), SELF_SIZE + TARGET_SLOT_SIZE);

    // -- Nearby entity buckets ------------------------------------------------
    encode_entity_bucket(&input.nearby_planets, K_PLANETS, max_speed, &mut obs);
    encode_entity_bucket(&input.nearby_asteroids, K_ASTEROIDS, max_speed, &mut obs);
    encode_entity_bucket(
        &input.nearby_hostile_ships,
        K_HOSTILE_SHIPS,
        max_speed,
        &mut obs,
    );
    encode_entity_bucket(
        &input.nearby_friendly_ships,
        K_FRIENDLY_SHIPS,
        max_speed,
        &mut obs,
    );
    encode_entity_bucket(&input.nearby_pickups, K_PICKUPS, max_speed, &mut obs);

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

fn encode_target_slot(
    target: Option<&TargetSlotData>,
    max_speed: f32,
    weapon_speed: f32,
    weapon_range: f32,
    ego_vel: [f32; 2],
    obs: &mut Vec<f32>,
) {
    if let Some(t) = target {
        obs.push(1.0); // is_present
        obs.extend_from_slice(&entity_type_onehot(t.entity_type));
        obs.extend_from_slice(&t.rel_pos);
        obs.extend_from_slice(&t.rel_vel);
        obs.push(t.should_engage);
        obs.push(t.value_hint);

        // Pursuit angle — intercept using ship max speed; obj_vel = relative velocity.
        let pursuit = intercept_angle(max_speed, t.rel_pos, t.rel_vel);
        obs.push(pursuit.unwrap_or(0.0));
        obs.push(aim_indicator(pursuit));

        // Fire-lead angle — obj_vel = absolute target velocity (rel_vel + ship_vel_ego).
        let tgt_vel_abs = [t.rel_vel[0] + ego_vel[0], t.rel_vel[1] + ego_vel[1]];
        let fire = intercept_angle(weapon_speed, t.rel_pos, tgt_vel_abs);
        obs.push(fire.unwrap_or(0.0));
        obs.push(aim_indicator(fire));

        // In-range flag.
        let dist = (t.rel_pos[0].powi(2) + t.rel_pos[1].powi(2)).sqrt();
        obs.push(if weapon_range > 0.0 && dist <= weapon_range {
            1.0
        } else {
            0.0
        });
    } else {
        obs.extend(std::iter::repeat(0.0_f32).take(TARGET_SLOT_SIZE));
    }
}

fn encode_entity_bucket(
    slots: &[EntitySlotData],
    max_count: usize,
    max_speed: f32,
    obs: &mut Vec<f32>,
) {
    for i in 0..max_count {
        if let Some(slot) = slots.get(i) {
            // Pursuit angle — rel_pos and rel_vel are raw (un-normalised) ego-frame values.
            let pursuit = intercept_angle(max_speed, slot.rel_pos, slot.rel_vel);
            obs.extend_from_slice(&slot.rel_pos);
            obs.extend_from_slice(&slot.rel_vel);
            obs.push(slot.extra);
            obs.push(pursuit.unwrap_or(0.0));
            obs.push(aim_indicator(pursuit));
        } else {
            obs.extend(std::iter::repeat(0.0_f32).take(ENTITY_SLOT_SIZE));
        }
    }
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
mod tests {
    use super::*;
    use crate::ship::{Ship, ShipData};
    use std::collections::HashMap;

    fn dummy_ship() -> Ship {
        Ship {
            ship_type: "test".to_string(),
            data: ShipData {
                max_health: 100,
                max_speed: 200.0,
                cargo_space: 50,
                thrust: 100.0,
                torque: 10.0,
                ..Default::default()
            },
            health: 50,
            cargo: {
                let mut m = HashMap::new();
                m.insert("ore".to_string(), 10u16);
                m
            },
            ..Default::default()
        }
    }

    fn minimal_obs_input(ship: &Ship) -> ObsInput<'_> {
        ObsInput {
            personality: &ship.data.personality,
            ship,
            velocity: [0.0, 0.0],
            angular_velocity: 0.0,
            ship_heading: [0.0, 1.0], // facing up
            target: None,
            nearby_planets: vec![],
            nearby_asteroids: vec![],
            nearby_hostile_ships: vec![],
            nearby_friendly_ships: vec![],
            nearby_pickups: vec![],
            primary_weapon_speed: 400.0,
            primary_weapon_range: 800.0,
        }
    }

    #[test]
    fn test_observation_shape_constant() {
        let ship = dummy_ship();

        // No nearby entities.
        let obs0 = encode_observation(&minimal_obs_input(&ship));
        assert_eq!(obs0.len(), OBS_DIM, "zero entities");

        // One of each type.
        let slot = EntitySlotData {
            rel_pos: [0.1, 0.2],
            rel_vel: [0.0, 0.0],
            extra: 1.0,
            ..Default::default()
        };
        let obs1 = encode_observation(&ObsInput {
            personality: &ship.data.personality,
            ship: &ship,
            velocity: [10.0, 0.0],
            angular_velocity: 0.5,
            ship_heading: [1.0, 0.0],
            target: Some(TargetSlotData {
                entity_type: 1,
                rel_pos: [0.3, 0.4],
                rel_vel: [0.0, 0.0],
                is_hostile: 0.0,
                should_engage: 0.0,
                value_hint: 1.0,
            }),
            nearby_planets: vec![slot.clone()],
            nearby_asteroids: vec![slot.clone(), slot.clone()],
            nearby_hostile_ships: vec![slot.clone()],
            nearby_friendly_ships: vec![],
            nearby_pickups: vec![slot.clone()],
            primary_weapon_speed: 400.0,
            primary_weapon_range: 800.0,
        });
        assert_eq!(obs1.len(), OBS_DIM, "some entities");

        // Fully-populated buckets.
        let obs2 = encode_observation(&ObsInput {
            personality: &ship.data.personality,
            ship: &ship,
            velocity: [0.0, 0.0],
            angular_velocity: 0.0,
            ship_heading: [0.0, 1.0],
            target: Some(TargetSlotData {
                entity_type: 0,
                rel_pos: [0.1, 0.0],
                rel_vel: [0.0, 0.0],
                is_hostile: 1.0,
                should_engage: 1.0,
                value_hint: 0.5,
            }),
            nearby_planets: vec![slot.clone(); K_PLANETS],
            nearby_asteroids: vec![slot.clone(); K_ASTEROIDS],
            nearby_hostile_ships: vec![slot.clone(); K_HOSTILE_SHIPS],
            nearby_friendly_ships: vec![slot.clone(); K_FRIENDLY_SHIPS],
            nearby_pickups: vec![slot.clone(); K_PICKUPS],
            primary_weapon_speed: 400.0,
            primary_weapon_range: 800.0,
        });
        assert_eq!(obs2.len(), OBS_DIM, "full buckets");
    }

    #[test]
    fn test_ego_centric_encoding() {
        // Ship facing +x (right). A target directly in front should have
        // rel_pos = [positive, ~0] in ego frame.
        let (sin_a, cos_a) = ego_frame_sincos([1.0_f32, 0.0]);
        let world_offset = [100.0_f32, 0.0]; // directly to the right in world = forward in ego
        let ego = rotate_to_ego(world_offset, sin_a, cos_a);
        assert!(
            (ego[0] - 100.0).abs() < 1e-4,
            "ego x should be ~100, got {}",
            ego[0]
        );
        assert!(ego[1].abs() < 1e-4, "ego y should be ~0, got {}", ego[1]);

        // Ship facing +y (up). A target directly above should map to forward.
        let (sin_a, cos_a) = ego_frame_sincos([0.0_f32, 1.0]);
        let world_offset = [0.0_f32, 100.0];
        let ego = rotate_to_ego(world_offset, sin_a, cos_a);
        assert!(
            (ego[0] - 100.0).abs() < 1e-4,
            "ego x should be ~100, got {}",
            ego[0]
        );
        assert!(ego[1].abs() < 1e-4, "ego y should be ~0, got {}", ego[1]);
    }

    #[test]
    fn test_action_to_ship_command() {
        // thrust=1, turn=left(0) → thrust=1.0, turn=-1.0
        let (thrust, turn) = discrete_to_controls(1, 0);
        assert_eq!(thrust, 1.0);
        assert_eq!(turn, -1.0);

        // thrust=0, turn=right(2) → thrust=0.0, turn=1.0
        let (thrust, turn) = discrete_to_controls(0, 2);
        assert_eq!(thrust, 0.0);
        assert_eq!(turn, 1.0);

        // thrust=1, turn=straight(1) → thrust=1.0, turn=0.0
        let (thrust, turn) = discrete_to_controls(1, 1);
        assert_eq!(thrust, 1.0);
        assert_eq!(turn, 0.0);
    }

    #[test]
    fn test_angle_to_discrete() {
        use std::f32::consts::PI;

        // Large left angle → turn left, no thrust
        let (turn, thrust) = angle_to_discrete(PI / 2.0);
        assert_eq!(turn, 0);
        assert_eq!(thrust, 0);

        // Small left angle → turn left, thrust
        let (turn, thrust) = angle_to_discrete(0.1);
        assert_eq!(turn, 0);
        assert_eq!(thrust, 1);

        // Small right angle → turn right, thrust
        let (turn, thrust) = angle_to_discrete(-0.1);
        assert_eq!(turn, 2);
        assert_eq!(thrust, 1);

        // Large right angle → turn right, no thrust
        let (turn, thrust) = angle_to_discrete(-PI / 2.0);
        assert_eq!(turn, 2);
        assert_eq!(thrust, 0);
    }

    #[test]
    fn test_controls_to_discrete_roundtrip() {
        // Verify that discrete_to_controls and controls_to_discrete are consistent.
        for thrust_idx in 0u8..=1 {
            for turn_idx in [0u8, 1, 2] {
                let (t, r) = discrete_to_controls(thrust_idx, turn_idx);
                let (t2, r2) = controls_to_discrete(t, r);
                assert_eq!(
                    thrust_idx, t2,
                    "thrust mismatch for ({}, {})",
                    thrust_idx, turn_idx
                );
                assert_eq!(
                    turn_idx, r2,
                    "turn mismatch for ({}, {})",
                    thrust_idx, turn_idx
                );
            }
        }
    }

    #[test]
    fn test_personality_onehot() {
        assert_eq!(personality_onehot(&Personality::Miner), [1.0, 0.0, 0.0]);
        assert_eq!(personality_onehot(&Personality::Fighter), [0.0, 1.0, 0.0]);
        assert_eq!(personality_onehot(&Personality::Trader), [0.0, 0.0, 1.0]);
    }

    // ── Action pipeline consistency ────────────────────────────────────────

    /// Verify that the BC collection path and the inference → execution path
    /// use the same turn/thrust ordering.
    ///
    /// BC collection: `controls_to_discrete(thrust, turn)` → `(thrust_idx, turn_idx)`
    ///   → stored as `DiscreteAction = (turn_idx, thrust_idx, ...)`.
    ///
    /// Inference:  `logits_to_discrete_action` → `(turn_idx, thrust_idx, ...)`
    ///   → `repeat_actions` destructures as `(turn_idx, thrust_idx, ...)`
    ///   → calls `discrete_to_controls(thrust_idx, turn_idx)`.
    ///
    /// This test ensures the full roundtrip is correct.
    #[test]
    fn test_action_bc_roundtrip() {
        // Simulate what BC collection does: rule-based AI produces (thrust, turn),
        // then we go controls_to_discrete → DiscreteAction → discrete_to_controls.
        let cases: &[(f32, f32)] = &[
            (1.0, -1.0), // thrust forward, turn left
            (1.0, 0.0),  // thrust forward, straight
            (1.0, 1.0),  // thrust forward, turn right
            (0.0, -1.0), // no thrust, turn left
            (0.0, 0.0),  // no thrust, straight
            (0.0, 1.0),  // no thrust, turn right
        ];
        for &(thrust, turn) in cases {
            let (thrust_idx, turn_idx) = controls_to_discrete(thrust, turn);
            // BC stores as (turn_idx, thrust_idx, fp, fs):
            let action: DiscreteAction = (turn_idx, thrust_idx, 0, 0);
            // Inference path destructures the same way:
            let (turn_out, thrust_out, _, _) = action;
            let (t_back, r_back) = discrete_to_controls(thrust_out, turn_out);
            assert_eq!(
                t_back, thrust,
                "thrust mismatch for input ({thrust}, {turn}): got {t_back}"
            );
            assert_eq!(
                r_back, turn,
                "turn mismatch for input ({thrust}, {turn}): got {r_back}"
            );
        }
    }

    // ── intercept_angle / aim_indicator ─────────────────────────────────────

    /// Object directly ahead (ego +X), stationary → intercept angle = 0.
    #[test]
    fn test_intercept_angle_directly_ahead() {
        let angle = intercept_angle(200.0, [500.0, 0.0], [0.0, 0.0]).unwrap();
        assert!(angle.abs() < 1e-4, "expected ~0, got {}", angle);
    }

    /// Object directly to the left (ego +Y) → angle should be +π/2.
    #[test]
    fn test_intercept_angle_left() {
        let angle = intercept_angle(200.0, [0.0, 500.0], [0.0, 0.0]).unwrap();
        assert!(
            (angle - PI / 2.0).abs() < 1e-4,
            "expected ~π/2, got {}",
            angle
        );
    }

    /// Object directly to the right (ego −Y) → angle should be −π/2.
    #[test]
    fn test_intercept_angle_right() {
        let angle = intercept_angle(200.0, [0.0, -500.0], [0.0, 0.0]).unwrap();
        assert!(
            (angle + PI / 2.0).abs() < 1e-4,
            "expected ~−π/2, got {}",
            angle
        );
    }

    /// All intercept angles must lie in [−π, π].
    #[test]
    fn test_intercept_angle_always_in_range() {
        let cases: &[([f32; 2], [f32; 2])] = &[
            ([100.0, 0.0], [0.0, 0.0]),
            ([0.0, 100.0], [0.0, 0.0]),
            ([-100.0, 0.0], [0.0, 0.0]),
            ([100.0, 100.0], [50.0, -50.0]),
            ([-200.0, 150.0], [-30.0, 20.0]),
            ([50.0, -300.0], [10.0, -5.0]),
        ];
        for &(pos, vel) in cases {
            if let Some(a) = intercept_angle(200.0, pos, vel) {
                assert!(
                    a >= -PI && a <= PI,
                    "angle {} out of [-π, π] for pos={:?} vel={:?}",
                    a,
                    pos,
                    vel
                );
            }
        }
    }

    /// aim_indicator is 1.0 when angle is exactly 0 (perfectly aimed forward).
    #[test]
    fn test_aim_indicator_perfect() {
        let ind = aim_indicator(Some(0.0));
        assert!((ind - 1.0).abs() < 1e-5, "expected 1.0, got {}", ind);
    }

    /// aim_indicator is ≤ epsilon when angle is ±π/2 (boundary, floating point).
    /// aim_indicator is exactly 0 when |angle| > π/2.
    #[test]
    fn test_aim_indicator_perpendicular() {
        // At the exact boundary the value should be essentially zero (float rounding).
        assert!(aim_indicator(Some(PI / 2.0)) < 1e-10);
        assert!(aim_indicator(Some(-PI / 2.0)) < 1e-10);
        // Clearly beyond the boundary: must be exactly 0.
        assert_eq!(aim_indicator(Some(PI)), 0.0);
        assert_eq!(aim_indicator(Some(-PI)), 0.0);
        assert_eq!(aim_indicator(Some(PI * 0.75)), 0.0);
    }

    /// aim_indicator is 0.0 when there is no intercept.
    #[test]
    fn test_aim_indicator_none() {
        assert_eq!(aim_indicator(None), 0.0);
    }

    /// aim_indicator is strictly between 0 and 1 for a small off-axis angle.
    #[test]
    fn test_aim_indicator_partial() {
        let ind = aim_indicator(Some(PI / 4.0));
        assert!(ind > 0.0 && ind < 1.0, "expected (0,1), got {}", ind);
    }

    /// Angles near ±π are treated as "facing backward" → indicator = 0.
    #[test]
    fn test_aim_indicator_backwards_wrapping() {
        // An angle of just under π should give 0.
        let ind = aim_indicator(Some(PI - 0.01));
        assert_eq!(ind, 0.0);
        // An angle of just over −π should give 0 (wrapping handles it).
        let ind2 = aim_indicator(Some(-PI + 0.01));
        assert_eq!(ind2, 0.0);
    }

    /// Pursuit angle for a moving target: object ahead moving away should still
    /// give a small angle (near 0) if ship is faster than the target.
    #[test]
    fn test_intercept_angle_moving_target_ahead() {
        // Object 300 m ahead, moving away at 50 m/s; ship max speed = 200 m/s.
        let angle = intercept_angle(200.0, [300.0, 0.0], [50.0, 0.0]).unwrap();
        assert!(angle.abs() < 0.1, "expected near-zero angle, got {}", angle);
    }

    /// encode_observation produces the correct OBS_DIM even with the new slot sizes.
    #[test]
    fn test_obs_dim_matches_constant() {
        let expected = SELF_SIZE
            + TARGET_SLOT_SIZE
            + (K_PLANETS + K_ASTEROIDS + K_HOSTILE_SHIPS + K_FRIENDLY_SHIPS + K_PICKUPS)
                * ENTITY_SLOT_SIZE;
        assert_eq!(
            OBS_DIM, expected,
            "OBS_DIM constant does not match slot layout"
        );

        let ship = dummy_ship();
        let obs = encode_observation(&minimal_obs_input(&ship));
        assert_eq!(obs.len(), OBS_DIM);
    }

    /// Target pursuit angle is near 0 when target is directly ahead and stationary.
    #[test]
    fn test_target_pursuit_angle_ahead() {
        let ship = dummy_ship();
        // Ship facing +x, target directly ahead at (500, 0) in world space.
        // In ego frame (+x = forward) this maps to rel_pos ≈ (0.25, 0) after /DETECTION_RADIUS.
        let obs = encode_observation(&ObsInput {
            personality: &ship.data.personality,
            ship: &ship,
            velocity: [0.0, 0.0],
            angular_velocity: 0.0,
            ship_heading: [1.0, 0.0], // facing right
            target: Some(TargetSlotData {
                entity_type: 1,
                rel_pos: [500.0, 0.0], // ahead in ego frame (raw, un-normalised)
                rel_vel: [0.0, 0.0],
                is_hostile: 0.0,
                should_engage: 0.0,
                value_hint: 1.0,
            }),
            nearby_planets: vec![],
            nearby_asteroids: vec![],
            nearby_hostile_ships: vec![],
            nearby_friendly_ships: vec![],
            nearby_pickups: vec![],
            primary_weapon_speed: 400.0,
            primary_weapon_range: 1000.0,
        });
        // Target slot starts at SELF_SIZE; pursuit_angle is at offset 11 (after present+4+2+2+1+1).
        let pursuit_angle = obs[SELF_SIZE + 11];
        assert!(
            pursuit_angle.abs() < 1e-4,
            "expected ~0, got {}",
            pursuit_angle
        );
        // Aim indicator should be ~1.
        let aim_ind = obs[SELF_SIZE + 12];
        assert!((aim_ind - 1.0).abs() < 1e-4, "expected ~1, got {}", aim_ind);
        // In range (weapon_range=1000 > dist=500).
        let in_range = obs[SELF_SIZE + 15];
        assert_eq!(in_range, 1.0);
    }
}
