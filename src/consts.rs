//! RL reward constants.
//!
//! All personality-weighted reward values live here so they can be tuned in one
//! place.  Each constant documents the event that triggers it and the
//! personality it applies to.

// ---------------------------------------------------------------------------
// Reward types (indices into per-step reward arrays)
// ---------------------------------------------------------------------------

/// Number of distinct reward channels tracked separately.
pub const N_REWARD_TYPES: usize = 6;

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
/// Index for per-step goal-target rewards (targeting the right entity).
pub const REWARD_GOAL_TARGET: usize = 5;

/// Human-readable names for TensorBoard logging, indexed by reward type.
pub const REWARD_TYPE_NAMES: [&str; N_REWARD_TYPES] =
    ["landing", "cargo_sold", "weapon_hit", "pickup", "health", "goal_target"];

/// Per-type weights applied when summing head rewards into the total reward
/// used for the policy advantage.  Tune these to control the relative
/// importance of each reward channel.
/// Indexed by: [LANDING, CARGO_SOLD, WEAPON_HIT, PICKUP, HEALTH, GOAL_TARGET].
pub const REWARD_TYPE_WEIGHTS: [f32; N_REWARD_TYPES] = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1];

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
// Per-step goal-target reward
// ---------------------------------------------------------------------------

/// Per-step weight for the goal-target reward.
/// Gated by `health_frac >= 0.3` and `distressed <= 0.5`, scaled by both.
pub const GOAL_TARGET_STEP_WEIGHT: f32 = 0.05;
