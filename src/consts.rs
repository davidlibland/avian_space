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
/// Index for ship-hit (combat) rewards.
pub const REWARD_SHIP_HIT: usize = 2;
/// Index for asteroid-hit (mining) rewards.
pub const REWARD_ASTEROID_HIT: usize = 3;
/// Index for pickup-collection rewards.
pub const REWARD_PICKUP: usize = 4;
/// Index for gated health-at-event rewards.  Fired at the same moments as the
/// event rewards above (landing, cargo_sold, ship_hit, asteroid_hit, pickup),
/// scaled by the firing ship's current `health / max_health`.
pub const REWARD_HEALTH_GATED: usize = 5;
/// Diagnostic channel: per-step `h_t / h_max` written every decision step.
/// Weight = 0.0 so it does NOT influence the policy advantage.  The value
/// head trained on this channel predicts expected discounted future health —
/// used for monitoring survival behaviour, not policy updates.
pub const REWARD_HEALTH_RAW: usize = 6;
/// Index for damage-taken penalties.  Fired on each `DamageShip` event with
/// reward = `-HEALTH_DAMAGE_PENALTY · damage_frac · (1 - h/h_max)`.
/// At full health the penalty is 0 (combat is free); at low health it
/// approaches the full magnitude (strong pressure to disengage / retreat).
pub const REWARD_DAMAGE: usize = 7;

/// Human-readable names for TensorBoard logging, indexed by reward type.
pub const REWARD_TYPE_NAMES: [&str; N_REWARD_TYPES] = [
    "landing",
    "cargo_sold",
    "ship_hit",
    "asteroid_hit",
    "pickup",
    "health_gated",
    "health_raw",
    "damage",
];

/// Per-type weights applied when summing head rewards into the total reward
/// used for the policy advantage.
pub const REWARD_TYPE_WEIGHTS: [f32; N_REWARD_TYPES] = [
    1.0, // landing
    1.0, // cargo_sold
    1.0, // ship_hit
    1.0, // asteroid_hit
    1.0, // pickup
    1.0, // health_gated (event-gated)
    0.0, // health_raw (diagnostic only — zero advantage contribution)
    1.0, // damage
];

/// Scalar applied to the health bonus written alongside each event reward.
/// Interpretation: at full health, each event emits a `HEALTH_BONUS_PER_EVENT`
/// addition to the health channel; at zero health, zero.
pub const HEALTH_BONUS_PER_EVENT: f32 = 0.3;

/// Scalar applied to the damage-taken penalty.
/// Penalty = `-HEALTH_DAMAGE_PENALTY · (damage / max_health) · (1 - h/h_max)`.
/// Tuned so a full-health-to-zero run of damage (impossible in one hit) would
/// peak at roughly the same magnitude as a single event bonus.
pub const HEALTH_DAMAGE_PENALTY: f32 = 0.3;

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

