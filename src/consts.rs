//! RL reward constants.
//!
//! All personality-weighted reward values live here so they can be tuned in one
//! place.  Each constant documents the event that triggers it and the
//! personality it applies to.

// ---------------------------------------------------------------------------
// Reward types (indices into per-step reward arrays)
// ---------------------------------------------------------------------------

/// Number of distinct reward channels tracked separately.
pub const N_REWARD_TYPES: usize = 8;

/// Index for planet-landing rewards.
pub const REWARD_LANDING: usize = 0;
/// Index for cargo-sold rewards.
pub const REWARD_CARGO_SOLD: usize = 1;
/// Index for weapon-hit rewards (ship + asteroid).
pub const REWARD_WEAPON_HIT: usize = 2;
/// Index for pickup-collection rewards.
pub const REWARD_PICKUP: usize = 3;
/// Index for per-step health rewards.
pub const REWARD_HEALTH: usize = 4;
/// Index for per-step nav-target rewards (correct navigation target for personality).
pub const REWARD_NAV_TARGET: usize = 5;
/// Index for per-step weapons-target rewards (targeting hostile/engaged entities).
pub const REWARD_WEAPONS_TARGET: usize = 6;
/// Index for per-step movement/position rewards (proximity + approach + velocity matching).
pub const REWARD_MOVEMENT: usize = 7;

/// Human-readable names for TensorBoard logging, indexed by reward type.
pub const REWARD_TYPE_NAMES: [&str; N_REWARD_TYPES] = [
    "landing",
    "cargo_sold",
    "weapon_hit",
    "pickup",
    "health",
    "nav_target",
    "weapons_target",
    "movement",
];

/// Per-type weights applied when summing head rewards into the total reward
/// used for the policy advantage.  Tune these to control the relative
/// importance of each reward channel.
/// Indexed by: [LANDING, CARGO_SOLD, WEAPON_HIT, PICKUP, HEALTH, NAV_TARGET, WEAPONS_TARGET, MOVEMENT].
pub const REWARD_TYPE_WEIGHTS: [f32; N_REWARD_TYPES] = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1];

// ---------------------------------------------------------------------------
// Combat: hit on a ship
// ---------------------------------------------------------------------------

/// Reward multiplier when the hit ship is hostile/should-engage AND targeted.
pub const COMBAT_HIT_ENGAGED_TARGETED: f32 = 1.0;
/// Reward multiplier when the hit ship is hostile/should-engage but NOT targeted.
pub const COMBAT_HIT_ENGAGED_UNTARGETED: f32 = 0.5;
/// Smoothing constant for the adaptive neutral-hit penalty.
/// The penalty is `c = -p * r / (EPS + (1-p))` where p = good_frac, r = ENGAGED_UNTARGETED.
/// Bounded in `[-r / EPS, 0]`.
pub const COMBAT_HIT_EPS: f32 = 0.5;

/// Personality scale applied to the combat hit reward for Fighters.
pub const COMBAT_PERSONALITY_FIGHTER: f32 = 1.0;
/// Personality scale applied to the combat hit reward for non-Fighters.
pub const COMBAT_PERSONALITY_OTHER: f32 = 0.3;

// ---------------------------------------------------------------------------
// Combat: hit on an asteroid
// ---------------------------------------------------------------------------

/// Reward for hitting an asteroid when the firing ship is a Miner.
pub const ASTEROID_HIT_MINER: f32 = 1.0;
/// Reward for hitting an asteroid when the firing ship is NOT a Miner.
pub const ASTEROID_HIT_OTHER: f32 = 0.3;

// ---------------------------------------------------------------------------
// Landing on a planet
// ---------------------------------------------------------------------------
// The landing reward is the sum of applicable condition bonuses, multiplied by
// a targeting bonus when the planet is the ship's current target.

/// Trader: bonus for landing when able to sell cargo (has cargo to sell).
/// Scaled by `cargo_held / cargo_space`.
pub const LANDING_TRADER_CAN_SELL: f32 = 0.5;
/// Trader: bonus for landing when able to buy (has cargo space and credits).
pub const LANDING_TRADER_CAN_BUY: f32 = 0.3;
/// Fighter: bonus for landing when able to replenish ammo.
pub const LANDING_FIGHTER_CAN_REARM: f32 = 0.5;
/// Fighter: bonus for landing when cargo hold is full.
pub const LANDING_FIGHTER_CARGO_FULL: f32 = 0.3;
/// Miner: bonus for landing when able to sell cargo.
/// Scaled by `cargo_held / cargo_space`.
pub const LANDING_MINER_CAN_SELL: f32 = 0.8;
/// Any personality: bonus for landing with low health.
/// Scaled by `(1 - health / max_health)`.
pub const LANDING_LOW_HEALTH: f32 = 0.5;
/// Multiplier applied to the total landing reward when the planet is targeted.
pub const LANDING_ON_TARGET_MULTIPLIER: f32 = 2.0;
/// Multiplier when the planet is NOT the ship's current target.
pub const LANDING_OFF_TARGET_MULTIPLIER: f32 = 1.0;

// ---------------------------------------------------------------------------
// Cargo sold at a planet
// ---------------------------------------------------------------------------

/// Cargo-sale reward weight (Fighter). Multiplied by `sold_frac`.
pub const CARGO_SOLD_FIGHTER: f32 = 0.1;
/// Cargo-sale reward weight (Miner). Multiplied by `sold_frac`.
pub const CARGO_SOLD_MINER: f32 = 0.8;
/// Cargo-sale reward weight (Trader). Multiplied by `sold_frac`.
pub const CARGO_SOLD_TRADER: f32 = 1.0;

// ---------------------------------------------------------------------------
// Pickup collection
// ---------------------------------------------------------------------------

/// Reward for collecting a pickup (Fighter).
pub const PICKUP_REWARD_FIGHTER: f32 = 0.1;
/// Reward for collecting a pickup (Miner).
pub const PICKUP_REWARD_MINER: f32 = 0.8;
/// Reward for collecting a pickup (Trader).
pub const PICKUP_REWARD_TRADER: f32 = 0.1;

// ---------------------------------------------------------------------------
// Per-step health reward
// ---------------------------------------------------------------------------

/// Per-step health fraction weight (Fighter). Multiplied by `health / max_health`.
pub const HEALTH_STEP_FIGHTER: f32 = 0.03;
/// Per-step health fraction weight (Miner / Trader). Multiplied by `health / max_health`.
pub const HEALTH_STEP_MINER_TRADER: f32 = 0.05;

// ---------------------------------------------------------------------------
// Per-step nav-target reward
// ---------------------------------------------------------------------------

/// Trader: nav-target is planet → scaled by cargo_sale_value or commodity_margin.
pub const NAV_TARGET_TRADER_PLANET: f32 = 0.05;
/// Miner: nav-target is asteroid/pickup (when cargo space available).
pub const NAV_TARGET_MINER_RESOURCE: f32 = 0.05;
/// Miner: nav-target is planet (when cargo hold is full), scaled by cargo_frac.
pub const NAV_TARGET_MINER_PLANET: f32 = 0.05;
/// Fighter: same gating as old goal_target.
pub const NAV_TARGET_FIGHTER: f32 = 0.05;

// ---------------------------------------------------------------------------
// Per-step weapons-target reward
// ---------------------------------------------------------------------------

/// Fighter: weapons-target is hostile or should_engage.
pub const WEAPONS_TARGET_FIGHTER: f32 = 0.05;
/// Miner/Trader: weapons-target is hostile (defensive only).
pub const WEAPONS_TARGET_DEFENSIVE: f32 = 0.05;

// ---------------------------------------------------------------------------
// Per-step movement/position reward (dense shaping toward nav target)
// ---------------------------------------------------------------------------

/// Proximity reward: `LENGTH_SCALE / (distance + LENGTH_SCALE)`.
pub const MOVEMENT_LENGTH_SCALE: f32 = 200.0;
/// Radius within which the approach penalty fades to zero (allows turning/braking).
pub const MOVEMENT_THRESHOLD_DIST: f32 = 300.0;
/// Velocity-matching reward scale: `VEL_SCALE / (rel_vel + VEL_SCALE)`.
pub const MOVEMENT_VEL_SCALE: f32 = 50.0;
/// Overall per-step weight for the movement reward.
pub const MOVEMENT_STEP_WEIGHT: f32 = 0.05;

// ---------------------------------------------------------------------------
// Braking reward (approaching a planet nav-target)
// ---------------------------------------------------------------------------

/// Distance threshold within which braking rewards activate.
pub const BRAKING_THRESHOLD_DIST: f32 = 500.0;
/// Reward for being below LANDING_SPEED when approaching a planet.
pub const BRAKING_SLOW_REWARD: f32 = 0.05;
/// Max reward for facing retrograde (scales linearly from 0 at forward to max at fully retrograde).
pub const BRAKING_RETROGRADE_REWARD: f32 = 0.03;
/// Bonus for thrusting while nearly retrograde (within ~30° of fully retrograde).
pub const BRAKING_THRUST_REWARD: f32 = 0.02;
/// Cosine threshold for "nearly retrograde" (cos(150°) ≈ -0.87).
pub const BRAKING_RETROGRADE_COS_THRESH: f32 = -0.87;
/// Overall per-step weight for the braking reward.
pub const BRAKING_STEP_WEIGHT: f32 = 0.05;

// ---------------------------------------------------------------------------
// Weapons engagement shaping (incentivise aiming/pursuit toward nav target)
// ---------------------------------------------------------------------------

/// Reward when weapons_target matches nav_target (or nav_target is not a
/// combatable entity, so any weapons_target is valid).
pub const WEAPONS_FOCUS_REWARD: f32 = 0.02;
/// Pursuit indicator reward when navigating toward a pickup.
pub const PICKUP_PURSUIT_WEIGHT: f32 = 0.03;
/// Pursuit indicator reward when navigating toward a ship/asteroid (out of range).
pub const COMBAT_PURSUIT_WEIGHT: f32 = 0.03;
/// Fire indicator reward when navigating toward a ship/asteroid (in range).
pub const COMBAT_FIRE_WEIGHT: f32 = 0.05;
