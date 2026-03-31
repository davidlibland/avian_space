//! Neural-network architecture for the RL agent.
//!
//! Two inputs:
//! - Self features: `[B, SELF_INPUT_DIM]`
//! - Object features: `[B, N_OBJECTS, OBJECT_INPUT_DIM]`
//!
//! Observations are split into these two tensors via [`split_obs`].
//! Policy logits → [`DiscreteAction`] conversion is handled by
//! [`logits_to_discrete_action`] (greedy argmax; stochastic sampling is
//! added at training time).

use std::sync::{Arc, Mutex};

use bevy::prelude::Resource;
use burn::{
    backend::{ndarray::NdArray, wgpu::Wgpu, Autodiff},
    module::Initializer,
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    optim::{Adam, AdamConfig, adaptor::OptimizerAdaptor},
    prelude::*,
    tensor::{
        Tensor, TensorData,
        activation::{silu, softmax},
        backend::AutodiffBackend,
    },
};

use crate::rl_obs::{
    DiscreteAction, K_ASTEROIDS, K_FRIENDLY_SHIPS, K_HOSTILE_SHIPS, K_PICKUPS,
    K_PLANETS, N_ENTITY_SLOTS, OBS_DIM, SELF_SIZE, SLOT_SIZE,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Self-state feature count — must equal `SELF_SIZE` in `rl_obs.rs`.
pub const SELF_INPUT_DIM: usize = SELF_SIZE;

/// Per-object feature count — must equal `SLOT_SIZE` in `rl_obs.rs`.
/// Layout: `is_present(1) + type_onehot(4) + rel_pos(2) + rel_vel(2)
///          + pursuit_angle(1) + pursuit_indicator(1)
///          + fire_angle(1) + fire_indicator(1) + in_range(1) + value(1)
///          + type_specific(9)`
pub const OBJECT_INPUT_DIM: usize = SLOT_SIZE;

/// Total entity floats per observation (`N_OBJECTS × OBJECT_INPUT_DIM`).
pub const ENTITIES_FLAT_DIM: usize = N_OBJECTS * OBJECT_INPUT_DIM;

/// Number of object slots per observation: target + all buckets.
pub const N_OBJECTS: usize = N_ENTITY_SLOTS;

/// Policy output: factored logits `turn(3) | thrust(2) | fire_primary(2) | fire_secondary(2)`.
pub const POLICY_OUTPUT_DIM: usize = 9;

/// Target-selection output: one logit per object slot plus one "no target" logit.
pub const TARGET_OUTPUT_DIM: usize = N_OBJECTS + 1;

/// Value output: scalar state estimate.
pub const VALUE_OUTPUT_DIM: usize = 1;

/// Default hidden dimension for all network layers.
pub const HIDDEN_DIM: usize = 64;

// ---------------------------------------------------------------------------
// Backend aliases
// ---------------------------------------------------------------------------

/// CPU backend used for game-thread inference (stays on CPU — inference runs
/// at 4 Hz per ship and shares the process with Bevy's renderer).
pub type InferBackend = NdArray;

/// GPU backend with autodiff, used by the background training thread.
/// `Wgpu` selects Metal on macOS automatically via `WgpuDevice::BestAvailable`.
pub type TrainBackend = Autodiff<Wgpu>;

// ---------------------------------------------------------------------------
// NetBlock — pre-norm feedforward block
// ---------------------------------------------------------------------------

/// `LayerNorm → Linear(dim→dim, Xavier) → SiLU`
#[derive(Module, Debug)]
pub struct NetBlock<B: Backend> {
    norm: LayerNorm<B>,
    fc: Linear<B>,
}

impl<B: Backend> NetBlock<B> {
    pub fn new(device: &B::Device, dim: usize) -> Self {
        Self {
            norm: LayerNormConfig::new(dim).init(device),
            fc: LinearConfig::new(dim, dim)
                .with_initializer(Initializer::XavierNormal { gain: 1.52 })
                .init(device),
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.norm.forward(x);
        let x = self.fc.forward(x);
        silu(x)
    }
}

type Block<B> = NetBlock<B>;

// ---------------------------------------------------------------------------
// RLNet — policy / value network
// ---------------------------------------------------------------------------

/// Shared policy and value network.
///
/// ## Architecture
///
/// 1. **Object encoder**: `obj_features [B,N,OBJECT_DIM] → [B,N,H]`
///    via `Linear + 2×NetBlock`.
/// 2. **Self encoder**: `self_features [B,SELF_DIM] → [B,H]`
///    via 2-layer MLP.
/// 3. **Attention**: self as query, object-enc as keys/values → `[B,H]`.
/// 4. **Merge**: `concat(attn, self_enc) [B,2H] → Linear → [B,H]`.
/// 5. **Decoder**: `2×NetBlock + LayerNorm + SiLU → output [B,output_dim]`.
/// 6. **Target head** (pointer network): separate Q/K projections from
///    merged rep and obj_enc → one logit per entity slot + "no target".
///
/// The output projection is zero-initialised so the network starts near a
/// uniform policy / zero value.
#[derive(Module, Debug)]
pub struct RLNet<B: Backend> {
    // Stored for the attention scale; excluded from saved weights because
    // `new()` is always called before `load_record()`.
    #[module(skip)]
    hidden_dim: usize,

    // Object encoder
    obj_embed: Linear<B>,
    eblock1: Block<B>,
    eblock2: Block<B>,

    // Self encoder
    self_embed: Linear<B>,
    self_fc2: Linear<B>,

    // Single-head attention
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,

    // Post-attention merge
    merge_proj: Linear<B>,

    // Decoder
    dblock1: Block<B>,
    dblock2: Block<B>,
    norm: LayerNorm<B>,
    output: Linear<B>,

    // Target-selection pointer head
    target_q_proj: Linear<B>,
    target_k_proj: Linear<B>,
    /// Bias logit for the "no target" option (learned scalar).
    target_no_tgt: Linear<B>,
}

impl<B: Backend> RLNet<B> {
    pub fn new(device: &B::Device, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            hidden_dim,
            obj_embed: LinearConfig::new(OBJECT_INPUT_DIM, hidden_dim).init(device),
            eblock1: Block::new(device, hidden_dim),
            eblock2: Block::new(device, hidden_dim),
            self_embed: LinearConfig::new(SELF_INPUT_DIM, hidden_dim).init(device),
            self_fc2: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            q_proj: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            k_proj: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            v_proj: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            merge_proj: LinearConfig::new(hidden_dim * 2, hidden_dim).init(device),
            dblock1: Block::new(device, hidden_dim),
            dblock2: Block::new(device, hidden_dim),
            norm: LayerNormConfig::new(hidden_dim).init(device),
            output: LinearConfig::new(hidden_dim, output_dim)
                .with_initializer(Initializer::Zeros)
                .init(device),
            target_q_proj: LinearConfig::new(hidden_dim, hidden_dim)
                .with_initializer(Initializer::Zeros)
                .init(device),
            target_k_proj: LinearConfig::new(hidden_dim, hidden_dim)
                .with_initializer(Initializer::Zeros)
                .init(device),
            target_no_tgt: LinearConfig::new(hidden_dim, 1)
                .with_initializer(Initializer::Zeros)
                .init(device),
        }
    }

    /// Forward pass.
    ///
    /// * `self_feat`: `[B, SELF_INPUT_DIM]` — ship self-state features.
    /// * `obj_feat`: `[B, N_OBJECTS, OBJECT_INPUT_DIM]` — per-entity features.
    ///
    /// Returns `(action_logits [B, output_dim], target_logits [B, N_OBJECTS+1])`.
    /// The last target logit (index N_OBJECTS) is the "no target" option.
    /// Empty entity slots (is_present=0) are masked to `-inf`.
    pub fn forward(
        &self,
        self_feat: Tensor<B, 2>,
        obj_feat: Tensor<B, 3>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // is_present flag: first float of object features in each row.
        let is_present = obj_feat.clone().narrow(2, 0, 1).squeeze_dim(2); // [B, N]

        // Object encoder: [B, N, OBJECT_INPUT_DIM] → [B, N, H]
        let obj_enc = self.obj_embed.forward(obj_feat);
        let obj_enc = self.eblock1.forward(obj_enc);
        let obj_enc = self.eblock2.forward(obj_enc);

        // Self encoder: [B, SELF_INPUT_DIM] → [B, H]
        let self_enc = silu(self.self_embed.forward(self_feat));
        let self_enc = silu(self.self_fc2.forward(self_enc));

        // Scaled dot-product attention (self as Q, objects as K/V)
        let scale = (self.hidden_dim as f64).sqrt() as f32;
        let q = self.q_proj.forward(self_enc.clone()).unsqueeze_dim::<3>(1); // [B, 1, H]
        let k = self.k_proj.forward(obj_enc.clone()).swap_dims(1, 2);        // [B, H, N]
        let v = self.v_proj.forward(obj_enc.clone());                        // [B, N, H]
        let scores = q.matmul(k).div_scalar(scale);                          // [B, 1, N]
        let weights = softmax(scores, 2);                                    // [B, 1, N]
        let attn_out = weights.matmul(v).squeeze_dim(1);                     // [B, H]

        // Merge: concat(attn_out, self_enc) → H
        let merged = Tensor::cat(vec![attn_out, self_enc], 1); // [B, 2H]
        let x = silu(self.merge_proj.forward(merged.clone()));  // [B, H]

        // Decoder → action logits
        let x = self.dblock1.forward(x);
        let x = self.dblock2.forward(x);
        let x = self.norm.forward(x);
        let x = silu(x);
        let action_logits = self.output.forward(x.clone()); // [B, output_dim]

        // Target-selection pointer head
        let tq = self.target_q_proj.forward(x.clone()).unsqueeze_dim::<3>(1);  // [B, 1, H]
        let tk = self.target_k_proj.forward(obj_enc).swap_dims(1, 2);   // [B, H, N]
        let entity_logits = tq.matmul(tk).div_scalar(scale).squeeze_dim(1); // [B, N]

        // "No target" logit: learned bias from the merged representation.
        let no_tgt_logit = self.target_no_tgt.forward(x); // [B, 1]
        let target_logits = Tensor::cat(vec![entity_logits, no_tgt_logit], 1); // [B, N+1]

        // Mask empty slots to -inf so they can't be selected.
        // is_present [B, N] → append a 1.0 column for "no target" (always available).
        let batch_size = is_present.dims()[0];
        let device = is_present.device();
        let ones = Tensor::<B, 2>::ones([batch_size, 1], &device);
        let mask = Tensor::cat(vec![is_present, ones], 1); // [B, N+1]
        // Where mask == 0, set logits to -1e9 (large negative, acts as -inf).
        let neg_inf = Tensor::<B, 2>::full([batch_size, N_OBJECTS + 1], -1e9, &device);
        let target_logits = target_logits.clone() * mask.clone() + neg_inf * (mask.ones_like() - mask);

        (action_logits, target_logits)
    }
}

// ---------------------------------------------------------------------------
// InferenceNet — game-thread wrapper
// ---------------------------------------------------------------------------

/// Wraps a policy `RLNet` for synchronous game-thread inference.
///
/// Stored inside `RLResource` behind an `Arc<Mutex<_>>` so the training
/// thread can push updated weights while the game thread reads them.
pub struct InferenceNet {
    net: RLNet<InferBackend>,
    device: <InferBackend as Backend>::Device,
}

impl InferenceNet {
    /// Create a fresh (randomly initialised) inference network.
    pub fn new() -> Self {
        let device = Default::default();
        Self {
            net: RLNet::new(&device, HIDDEN_DIM, POLICY_OUTPUT_DIM),
            device,
        }
    }

    /// Run a batched forward pass.
    ///
    /// * `self_flat`: flattened `[batch_size × SELF_INPUT_DIM]`
    /// * `obj_flat`: flattened `[batch_size × N_OBJECTS × OBJECT_INPUT_DIM]`
    ///
    /// Returns `(action_logits, target_logits)` where:
    /// - `action_logits`: flattened `[batch_size × POLICY_OUTPUT_DIM]`
    /// - `target_logits`: flattened `[batch_size × TARGET_OUTPUT_DIM]`
    pub fn run_inference(
        &self,
        self_flat: Vec<f32>,
        obj_flat: Vec<f32>,
        batch_size: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let self_input = Tensor::<InferBackend, 2>::from_data(
            TensorData::new(self_flat, [batch_size, SELF_INPUT_DIM]),
            &self.device,
        );
        let obj_input = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(obj_flat, [batch_size, N_OBJECTS, OBJECT_INPUT_DIM]),
            &self.device,
        );
        let (action, target) = self.net.forward(self_input, obj_input);
        let action_logits = action
            .into_data()
            .into_vec::<f32>()
            .expect("action logit extraction failed");
        let target_logits = target
            .into_data()
            .into_vec::<f32>()
            .expect("target logit extraction failed");
        (action_logits, target_logits)
    }

    /// Serialize the current weights to bytes (compatible with [`Self::load_bytes`]).
    pub fn to_bytes(&self) -> Vec<u8> {
        net_to_bytes(self.net.clone())
    }

    /// Replace weights from a byte buffer produced by [`net_to_bytes`].
    ///
    /// Called by the training thread to push updated weights.
    pub fn load_bytes(&mut self, bytes: Vec<u8>) {
        use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let record = Recorder::<InferBackend>::load(&recorder, bytes, &self.device)
            .expect("failed to deserialize weights");
        self.net =
            RLNet::new(&self.device, HIDDEN_DIM, POLICY_OUTPUT_DIM).load_record(record);
    }
}

// ---------------------------------------------------------------------------
// RLInner — training-thread state (defined now; used when training is added)
// ---------------------------------------------------------------------------

/// All mutable training state owned by the background thread.
///
/// Generic over `B: AutodiffBackend` so the same code runs on CPU or GPU.
/// Fields are `Option<_>` so they can be temporarily moved out during
/// optimizer step (burn's `Optimizer::step` consumes the module).
#[allow(dead_code)]
pub struct RLInner<B: AutodiffBackend> {
    pub policy_net: Option<RLNet<B>>,
    pub value_net: Option<RLNet<B>>,
    pub policy_optim: OptimizerAdaptor<Adam, RLNet<B>, B>,
    pub value_optim: OptimizerAdaptor<Adam, RLNet<B>, B>,
}

#[allow(dead_code)]
impl<B: AutodiffBackend> RLInner<B> {
    pub fn new(device: &B::Device) -> Self {
        let adam = AdamConfig::new();
        Self {
            policy_net: Some(RLNet::new(device, HIDDEN_DIM, POLICY_OUTPUT_DIM)),
            value_net: Some(RLNet::new(device, HIDDEN_DIM, VALUE_OUTPUT_DIM)),
            policy_optim: adam.init::<B, RLNet<B>>(),
            value_optim: adam.init::<B, RLNet<B>>(),
        }
    }
}

// ---------------------------------------------------------------------------
// Bevy resource
// ---------------------------------------------------------------------------

/// Game-thread Bevy resource.
///
/// Holds the inference policy network behind a mutex so the (future) training
/// thread can write updated weights while the game thread reads them.
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
// Observation → model-input conversion
// ---------------------------------------------------------------------------

/// Split a flat `OBS_DIM` observation into self-features and entity-features.
///
/// Returns `(self_feat [SELF_INPUT_DIM], obj_feat [N_OBJECTS × OBJECT_INPUT_DIM])`.
pub fn split_obs(obs: &[f32]) -> (&[f32], &[f32]) {
    debug_assert_eq!(obs.len(), OBS_DIM, "obs length mismatch");
    (&obs[0..SELF_SIZE], &obs[SELF_SIZE..])
}

// ---------------------------------------------------------------------------
// Action conversion
// ---------------------------------------------------------------------------

/// Convert policy logits and target logits to a `DiscreteAction` via greedy
/// argmax over each factored head.
///
/// Action logit layout: `turn(3) | thrust(2) | fire_primary(2) | fire_secondary(2)`
/// Target logits: `[N_OBJECTS + 1]` (one per entity slot + "no target")
pub fn logits_to_discrete_action(
    action_logits: &[f32],
    target_logits: &[f32],
) -> DiscreteAction {
    debug_assert_eq!(action_logits.len(), POLICY_OUTPUT_DIM);
    debug_assert_eq!(target_logits.len(), TARGET_OUTPUT_DIM);
    let turn_idx = argmax(&action_logits[0..3]) as u8;
    let thrust_idx = argmax(&action_logits[3..5]) as u8;
    let fire_primary = argmax(&action_logits[5..7]) as u8;
    let fire_secondary = argmax(&action_logits[7..9]) as u8;
    let target_idx = argmax(target_logits) as u8;
    (turn_idx, thrust_idx, fire_primary, fire_secondary, target_idx)
}

fn argmax(vals: &[f32]) -> usize {
    vals.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Save / load
// ---------------------------------------------------------------------------

/// Save the inference network weights to `path` (`.bin` extension appended by burn).
pub fn save_inference_net(net: &InferenceNet, path: &str) {
    use burn::record::{BinFileRecorder, FullPrecisionSettings};
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    if let Err(e) = net.net.clone().save_file(path, &recorder) {
        eprintln!("[model] Failed to save policy net to {path}: {e}");
    }
}

/// Load inference network weights from `path`, returning `None` on failure.
pub fn load_inference_net(path: &str) -> Option<InferenceNet> {
    use burn::record::{BinFileRecorder, FullPrecisionSettings};
    let device: <InferBackend as Backend>::Device = Default::default();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    match RLNet::<InferBackend>::new(&device, HIDDEN_DIM, POLICY_OUTPUT_DIM)
        .load_file(path, &recorder, &device)
    {
        Ok(net) => {
            println!("[model] Loaded policy net from {path}");
            Some(InferenceNet { net, device })
        }
        Err(e) => {
            eprintln!("[model] Failed to load policy net from {path}: {e}");
            None
        }
    }
}

/// Save a `TrainBackend` net to disk as an inference-compatible checkpoint.
///
/// Strips the autodiff wrapper via `.valid()` so the file can be loaded by
/// [`load_inference_net`] on any backend.
pub fn save_training_net(net: &RLNet<TrainBackend>, path: &str) {
    // Serialise through bytes (backend-agnostic) then write to disk so the
    // file is loadable by `load_inference_net` with any backend.
    let bytes = net_to_bytes(net.clone());
    let file = format!("{path}.bin");
    if let Err(e) = std::fs::write(&file, bytes) {
        eprintln!("[model] Failed to save checkpoint to {file}: {e}");
    } else {
        println!("[model] Checkpoint saved to {file}");
    }
}

/// Load a checkpoint from `path` into a `TrainBackend` net, returning `None`
/// on failure.  Uses [`load_inference_net`] as an intermediary so the on-disk
/// format is always `InferBackend` (written by [`save_training_net`]).
pub fn load_training_net(
    path: &str,
    device: &<TrainBackend as Backend>::Device,
) -> Option<RLNet<TrainBackend>> {
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    // Load via inference path first, then transfer bytes to the training backend.
    let infer = load_inference_net(path)?;
    let bytes = infer.to_bytes();
    let rec = BinBytesRecorder::<FullPrecisionSettings>::default();
    match Recorder::<TrainBackend>::load(&rec, bytes, device) {
        Ok(record) => {
            let net =
                RLNet::<TrainBackend>::new(device, HIDDEN_DIM, POLICY_OUTPUT_DIM).load_record(record);
            println!("[model] Loaded training net from {path}");
            Some(net)
        }
        Err(e) => {
            eprintln!("[model] Failed to deserialise training net from {path}: {e}");
            None
        }
    }
}

/// Serialize a `TrainBackend` net to bytes via `.valid()` + `net_to_bytes`.
///
/// Convenience wrapper so callers in other modules don't need burn's
/// `AutodiffModule` trait in scope.
pub fn training_net_to_bytes(net: &RLNet<TrainBackend>) -> Vec<u8> {
    net_to_bytes(net.clone())
}

/// Serialize a training-backend net to bytes for cross-thread weight transfer.
#[allow(dead_code)]
pub fn net_to_bytes<B: Backend>(net: RLNet<B>) -> Vec<u8> {
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    Recorder::<B>::record(&recorder, net.into_record(), ())
        .expect("failed to serialize net to bytes")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "tests/model_tests.rs"]
mod tests;
