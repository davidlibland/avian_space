//! Runtime-tunable training configuration.
//!
//! [`TrainingConfig`] bundles the reward weights consumed by the game thread
//! ([`RewardConfig`]) with the PPO hyperparameters consumed by the trainer
//! thread ([`PpoConfig`]).  Defaults come from [`crate::consts`] and the
//! constants at the top of this module.
//!
//! # File format
//!
//! Configs are loaded from YAML.  The struct-level `#[serde(default)]`
//! attribute means any missing field falls back to the constant default, so
//! partial files are allowed: a user who only wants to override
//! `ppo.policy_lr` can supply a one-line YAML file listing exactly that.

use std::path::Path;

use bevy::prelude::Resource;
use serde::{Deserialize, Serialize};

use crate::consts::{
    self, ASTEROID_HIT_MINER, ASTEROID_HIT_OTHER, CARGO_SOLD_FIGHTER, CARGO_SOLD_MINER,
    CARGO_SOLD_TRADER, COMBAT_HIT_ENGAGED_TARGETED, COMBAT_HIT_ENGAGED_UNTARGETED, COMBAT_HIT_EPS,
    COMBAT_PERSONALITY_FIGHTER, COMBAT_PERSONALITY_OTHER, HEALTH_BONUS_PER_EVENT,
    HEALTH_DAMAGE_PENALTY, LANDING_COOLDOWN_SECS, LANDING_FIGHTER_CAN_REARM,
    LANDING_FIGHTER_CARGO_FULL, LANDING_LOW_HEALTH, LANDING_MINER_CAN_SELL,
    LANDING_OFF_TARGET_MULTIPLIER, LANDING_ON_TARGET_MULTIPLIER, LANDING_TRADER_CAN_BUY,
    LANDING_TRADER_CAN_SELL, N_REWARD_TYPES, PICKUP_REWARD_FIGHTER, PICKUP_REWARD_MINER,
    PICKUP_REWARD_TRADER, REWARD_SHARING_FIGHTER, REWARD_SHARING_MINER, REWARD_SHARING_TRADER,
};

// ---------------------------------------------------------------------------
// PPO defaults (mirrors the original constants in src/ppo/train.rs).
// ---------------------------------------------------------------------------

pub const PPO_GAMMA: f32 = 0.99;
pub const PPO_LAMBDA: f32 = 0.95;
pub const PPO_CLIP_EPS: f32 = 0.1;
pub const PPO_ENTROPY_COEFF: f32 = 0.01;
pub const PPO_BC_COEFF: f32 = 0.01;
pub const PPO_POLICY_LR: f64 = 3e-4;
pub const PPO_VALUE_LR: f64 = 1e-3;
pub const PPO_POLICY_EPOCHS: usize = 2;
pub const PPO_VALUE_EPOCHS: usize = 4;
pub const PPO_MINI_BATCH_SIZE: usize = 512;
pub const PPO_MIN_SEGMENTS: usize = 16;
pub const PPO_MAX_SEGMENTS: usize = 64;
pub const PPO_WEIGHT_SYNC_INTERVAL: usize = 1;
pub const PPO_SAVE_INTERVAL: usize = 30;
pub const PPO_HUBER_DELTA: f32 = 1.0;
pub const PPO_VALUE_BURNIN_EV_THRESHOLD: f32 = 0.3;
pub const VALUE_REPLAY_CAPACITY: usize = 8192;
pub const VALUE_REPLAY_FRACTION: f32 = 0.25;
pub const VALUE_REPLAY_EXTRA_BATCHES: usize = 4;

/// Number of completed segments between training-environment system swaps.
/// `0` disables swapping entirely (training stays in the starting system).
pub const PPO_SYSTEM_SWAP_SEGMENTS: usize = 0;
/// Probability that a swap targets the isolated `simulator` system instead of
/// a randomly chosen real system.
pub const PPO_SIMULATOR_FRACTION: f32 = 0.25;

// ---------------------------------------------------------------------------
// Reward weights
// ---------------------------------------------------------------------------

/// Tunable reward weights consumed by game-thread systems.  Each field
/// corresponds 1:1 to a former `pub const` in [`crate::consts`].
#[derive(Resource, Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RewardConfig {
    // Per-channel total-advantage weights (REWARD_TYPE_WEIGHTS).
    pub reward_weight_landing: f32,
    pub reward_weight_cargo_sold: f32,
    pub reward_weight_ship_hit: f32,
    pub reward_weight_asteroid_hit: f32,
    pub reward_weight_pickup: f32,
    pub reward_weight_health_gated: f32,
    pub reward_weight_health_raw: f32,
    pub reward_weight_damage: f32,

    // Health-channel scalars.
    pub health_bonus_per_event: f32,
    pub health_damage_penalty: f32,

    // Combat (ship-hit) rewards.
    pub combat_hit_engaged_targeted: f32,
    pub combat_hit_engaged_untargeted: f32,
    pub combat_hit_eps: f32,
    pub combat_personality_fighter: f32,
    pub combat_personality_other: f32,

    // Asteroid-hit rewards.
    pub asteroid_hit_miner: f32,
    pub asteroid_hit_other: f32,

    // Landing rewards.
    pub landing_trader_can_sell: f32,
    pub landing_trader_can_buy: f32,
    pub landing_fighter_can_rearm: f32,
    pub landing_fighter_cargo_full: f32,
    pub landing_miner_can_sell: f32,
    pub landing_low_health: f32,
    pub landing_on_target_multiplier: f32,
    pub landing_off_target_multiplier: f32,
    pub landing_cooldown_secs: f32,

    // Cargo-sold rewards.
    pub cargo_sold_fighter: f32,
    pub cargo_sold_miner: f32,
    pub cargo_sold_trader: f32,

    // Pickup rewards.
    pub pickup_reward_fighter: f32,
    pub pickup_reward_miner: f32,
    pub pickup_reward_trader: f32,

    // Reward sharing (ally-mixed rewards).
    pub reward_sharing_fighter: f32,
    pub reward_sharing_miner: f32,
    pub reward_sharing_trader: f32,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            reward_weight_landing: consts::REWARD_TYPE_WEIGHTS[consts::REWARD_LANDING],
            reward_weight_cargo_sold: consts::REWARD_TYPE_WEIGHTS[consts::REWARD_CARGO_SOLD],
            reward_weight_ship_hit: consts::REWARD_TYPE_WEIGHTS[consts::REWARD_SHIP_HIT],
            reward_weight_asteroid_hit: consts::REWARD_TYPE_WEIGHTS[consts::REWARD_ASTEROID_HIT],
            reward_weight_pickup: consts::REWARD_TYPE_WEIGHTS[consts::REWARD_PICKUP],
            reward_weight_health_gated: consts::REWARD_TYPE_WEIGHTS[consts::REWARD_HEALTH_GATED],
            reward_weight_health_raw: consts::REWARD_TYPE_WEIGHTS[consts::REWARD_HEALTH_RAW],
            reward_weight_damage: consts::REWARD_TYPE_WEIGHTS[consts::REWARD_DAMAGE],
            health_bonus_per_event: HEALTH_BONUS_PER_EVENT,
            health_damage_penalty: HEALTH_DAMAGE_PENALTY,
            combat_hit_engaged_targeted: COMBAT_HIT_ENGAGED_TARGETED,
            combat_hit_engaged_untargeted: COMBAT_HIT_ENGAGED_UNTARGETED,
            combat_hit_eps: COMBAT_HIT_EPS,
            combat_personality_fighter: COMBAT_PERSONALITY_FIGHTER,
            combat_personality_other: COMBAT_PERSONALITY_OTHER,
            asteroid_hit_miner: ASTEROID_HIT_MINER,
            asteroid_hit_other: ASTEROID_HIT_OTHER,
            landing_trader_can_sell: LANDING_TRADER_CAN_SELL,
            landing_trader_can_buy: LANDING_TRADER_CAN_BUY,
            landing_fighter_can_rearm: LANDING_FIGHTER_CAN_REARM,
            landing_fighter_cargo_full: LANDING_FIGHTER_CARGO_FULL,
            landing_miner_can_sell: LANDING_MINER_CAN_SELL,
            landing_low_health: LANDING_LOW_HEALTH,
            landing_on_target_multiplier: LANDING_ON_TARGET_MULTIPLIER,
            landing_off_target_multiplier: LANDING_OFF_TARGET_MULTIPLIER,
            landing_cooldown_secs: LANDING_COOLDOWN_SECS,
            cargo_sold_fighter: CARGO_SOLD_FIGHTER,
            cargo_sold_miner: CARGO_SOLD_MINER,
            cargo_sold_trader: CARGO_SOLD_TRADER,
            pickup_reward_fighter: PICKUP_REWARD_FIGHTER,
            pickup_reward_miner: PICKUP_REWARD_MINER,
            pickup_reward_trader: PICKUP_REWARD_TRADER,
            reward_sharing_fighter: REWARD_SHARING_FIGHTER,
            reward_sharing_miner: REWARD_SHARING_MINER,
            reward_sharing_trader: REWARD_SHARING_TRADER,
        }
    }
}

impl RewardConfig {
    /// Per-channel weights as a fixed-size array, indexed by `consts::REWARD_*`.
    pub fn reward_type_weights(&self) -> [f32; N_REWARD_TYPES] {
        let mut w = [0.0_f32; N_REWARD_TYPES];
        w[consts::REWARD_LANDING] = self.reward_weight_landing;
        w[consts::REWARD_CARGO_SOLD] = self.reward_weight_cargo_sold;
        w[consts::REWARD_SHIP_HIT] = self.reward_weight_ship_hit;
        w[consts::REWARD_ASTEROID_HIT] = self.reward_weight_asteroid_hit;
        w[consts::REWARD_PICKUP] = self.reward_weight_pickup;
        w[consts::REWARD_HEALTH_GATED] = self.reward_weight_health_gated;
        w[consts::REWARD_HEALTH_RAW] = self.reward_weight_health_raw;
        w[consts::REWARD_DAMAGE] = self.reward_weight_damage;
        w
    }
}

// ---------------------------------------------------------------------------
// PPO hyperparameters
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PpoConfig {
    pub gamma: f32,
    pub lambda: f32,
    pub clip_eps: f32,
    pub entropy_coeff: f32,
    pub bc_coeff: f32,
    pub policy_lr: f64,
    pub value_lr: f64,
    pub policy_epochs: usize,
    pub value_epochs: usize,
    pub mini_batch_size: usize,
    pub min_segments: usize,
    pub max_segments: usize,
    pub weight_sync_interval: usize,
    pub save_interval: usize,
    pub huber_delta: f32,
    pub value_burnin_ev_threshold: f32,
    pub value_replay_capacity: usize,
    pub value_replay_fraction: f32,
    pub value_replay_extra_batches: usize,
    /// Swap to a randomly chosen training system every `system_swap_segments`
    /// completed segments.  `0` disables swapping.
    pub system_swap_segments: usize,
    /// Probability a swap picks the isolated `simulator` system.
    pub simulator_fraction: f32,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            gamma: PPO_GAMMA,
            lambda: PPO_LAMBDA,
            clip_eps: PPO_CLIP_EPS,
            entropy_coeff: PPO_ENTROPY_COEFF,
            bc_coeff: PPO_BC_COEFF,
            policy_lr: PPO_POLICY_LR,
            value_lr: PPO_VALUE_LR,
            policy_epochs: PPO_POLICY_EPOCHS,
            value_epochs: PPO_VALUE_EPOCHS,
            mini_batch_size: PPO_MINI_BATCH_SIZE,
            min_segments: PPO_MIN_SEGMENTS,
            max_segments: PPO_MAX_SEGMENTS,
            weight_sync_interval: PPO_WEIGHT_SYNC_INTERVAL,
            save_interval: PPO_SAVE_INTERVAL,
            huber_delta: PPO_HUBER_DELTA,
            value_burnin_ev_threshold: PPO_VALUE_BURNIN_EV_THRESHOLD,
            value_replay_capacity: VALUE_REPLAY_CAPACITY,
            value_replay_fraction: VALUE_REPLAY_FRACTION,
            value_replay_extra_batches: VALUE_REPLAY_EXTRA_BATCHES,
            system_swap_segments: PPO_SYSTEM_SWAP_SEGMENTS,
            simulator_fraction: PPO_SIMULATOR_FRACTION,
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level training config
// ---------------------------------------------------------------------------

/// Combined runtime training configuration loaded from a YAML file.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrainingConfig {
    pub rewards: RewardConfig,
    pub ppo: PpoConfig,
}

impl TrainingConfig {
    /// Read and deserialize a YAML file at `path`.  Missing fields fall back
    /// to the constant defaults (via `#[serde(default)]`).
    pub fn load_from_path(path: &Path) -> Result<Self, ConfigError> {
        let s = std::fs::read_to_string(path).map_err(|e| ConfigError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        serde_yaml::from_str(&s).map_err(|e| ConfigError::Parse {
            path: path.to_path_buf(),
            source: e,
        })
    }

    /// Serialize as YAML to `path`, creating any missing parent directories.
    pub fn write_to_path(&self, path: &Path) -> Result<(), ConfigError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| ConfigError::Io {
                    path: parent.to_path_buf(),
                    source: e,
                })?;
            }
        }
        let s = serde_yaml::to_string(self).map_err(|e| ConfigError::Serialize {
            path: path.to_path_buf(),
            source: e,
        })?;
        std::fs::write(path, s).map_err(|e| ConfigError::Io {
            path: path.to_path_buf(),
            source: e,
        })
    }
}

#[derive(Debug)]
pub enum ConfigError {
    Io {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    Parse {
        path: std::path::PathBuf,
        source: serde_yaml::Error,
    },
    Serialize {
        path: std::path::PathBuf,
        source: serde_yaml::Error,
    },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "{}: {}", path.display(), source),
            Self::Parse { path, source } => {
                write!(f, "{}: parse error: {}", path.display(), source)
            }
            Self::Serialize { path, source } => {
                write!(f, "{}: serialize error: {}", path.display(), source)
            }
        }
    }
}

impl std::error::Error for ConfigError {}
