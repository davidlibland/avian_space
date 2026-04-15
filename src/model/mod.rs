//! Neural-network architecture for the RL agent.
//!
//! Two inputs:
//! - Self features: `[B, SELF_INPUT_DIM]`
//! - Object features: `[B, N_OBJECTS, OBJECT_INPUT_DIM]`
//!
//! Observations are split into these two tensors via [`split_obs`].
//! Policy logits → [`DiscreteAction`] conversion is handled by
//! [`sample_discrete_action`] during rollout (stochastic, with coupled
//! Gumbel sampling across the nav/weapon target heads).

use std::sync::{Arc, Mutex};

use bevy::prelude::Resource;
use burn::backend::{Autodiff, ndarray::NdArray, wgpu::Wgpu};

use crate::rl_obs::{K_PROJECTILES, OBS_DIM, PROJ_SLOT_SIZE, SELF_SIZE, SLOT_SIZE};

mod inference;
mod net;
mod sampling;
mod training;

pub use inference::{
    InferenceNet, load_inference_net, load_training_net, load_training_net_with_dim,
    save_training_net, training_net_to_bytes,
};
pub use net::RLNet;
#[cfg(test)]
pub use net::split_obj_feat;
#[cfg(test)]
pub use sampling::logits_to_discrete_action;
pub use sampling::sample_discrete_action;
pub use training::{RLInner, load_optimizer, save_optimizer};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Self-state feature count — must equal `SELF_SIZE` in `rl_obs.rs`.
pub const SELF_INPUT_DIM: usize = SELF_SIZE;

/// Per-object feature count — must equal `SLOT_SIZE` in `rl_obs.rs`.
pub const OBJECT_INPUT_DIM: usize = SLOT_SIZE;

/// Total entity floats per observation (`N_OBJECTS × OBJECT_INPUT_DIM`).
pub const ENTITIES_FLAT_DIM: usize = N_OBJECTS * OBJECT_INPUT_DIM;

/// Number of projectile slots per observation.
pub const N_PROJECTILE_SLOTS: usize = K_PROJECTILES;

/// Per-projectile feature count.
pub const PROJ_INPUT_DIM: usize = PROJ_SLOT_SIZE;

/// Total projectile floats per observation.
pub const PROJECTILES_FLAT_DIM: usize = N_PROJECTILE_SLOTS * PROJ_INPUT_DIM;

/// Number of object slots per observation.
pub const N_OBJECTS: usize = crate::rl_obs::N_ENTITY_SLOTS;

/// Policy output: factored logits `turn(3) | thrust(2) | fire_primary(2) | fire_secondary(2)`.
pub const POLICY_OUTPUT_DIM: usize = 9;

/// Target-selection output: one logit per object slot plus one "no target" logit.
pub const TARGET_OUTPUT_DIM: usize = N_OBJECTS + 1;

/// Value output: one head per reward type.
pub const VALUE_OUTPUT_DIM: usize = crate::consts::N_REWARD_TYPES;

/// Default hidden dimension for all network layers.
pub const HIDDEN_DIM: usize = 64;

/// Whether to use skip connections in `NetBlock`.
pub const USE_SKIP: bool = false;

// ---------------------------------------------------------------------------
// Backend aliases
// ---------------------------------------------------------------------------

/// CPU backend used for game-thread inference.
pub type InferBackend = NdArray;

/// GPU backend with autodiff, used by the background training thread.
pub type TrainBackend = Autodiff<Wgpu>;

// ---------------------------------------------------------------------------
// Bevy resource
// ---------------------------------------------------------------------------

/// Game-thread Bevy resource.
#[derive(Resource)]
pub struct RLResource {
    pub inference_net: Arc<Mutex<InferenceNet>>,
}

impl RLResource {
    pub fn new() -> Self {
        Self {
            inference_net: Arc::new(Mutex::new(InferenceNet::new())),
        }
    }
}

// ---------------------------------------------------------------------------
// Observation conversion
// ---------------------------------------------------------------------------

/// Split a flat `OBS_DIM` observation into self-features and entity-features.
pub fn split_obs(obs: &[f32]) -> (&[f32], &[f32]) {
    debug_assert_eq!(obs.len(), OBS_DIM, "obs length mismatch");
    (&obs[0..SELF_SIZE], &obs[SELF_SIZE..])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../tests/model_tests.rs"]
mod tests;
