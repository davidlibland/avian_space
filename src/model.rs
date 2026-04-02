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
    backend::{Autodiff, ndarray::NdArray, wgpu::Wgpu},
    module::{Initializer, Param},
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
    self, CORE_BLOCK_START, CORE_FEAT_SIZE, DiscreteAction, K_ASTEROIDS, K_FRIENDLY_SHIPS,
    K_HOSTILE_SHIPS, K_PICKUPS, K_PLANETS, N_ENTITY_SLOTS, N_ENTITY_TYPES, OBS_DIM, SELF_SIZE,
    SLOT_IS_PRESENT, SLOT_SIZE, SLOT_TYPE_ONEHOT, TYPE_BLOCK_SIZE, TYPE_BLOCK_START,
    TYPE_ONEHOT_SIZE,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Self-state feature count — must equal `SELF_SIZE` in `rl_obs.rs`.
pub const SELF_INPUT_DIM: usize = SELF_SIZE;

/// Per-object feature count — must equal `SLOT_SIZE` in `rl_obs.rs`.
pub const OBJECT_INPUT_DIM: usize = SLOT_SIZE;

/// Total entity floats per observation (`N_OBJECTS × OBJECT_INPUT_DIM`).
pub const ENTITIES_FLAT_DIM: usize = N_OBJECTS * OBJECT_INPUT_DIM;

/// Number of object slots per observation.
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

/// Split a `[B, N, SLOT_SIZE]` object-feature tensor into the 4 blocks
/// defined by the slot layout.
///
/// Returns `(type_onehot, is_present, core_feat, type_feat)`:
/// - `type_onehot`: `[B, N, N_ENTITY_TYPES]`
/// - `is_present`:  `[B, N]`
/// - `core_feat`:   `[B, N, CORE_FEAT_SIZE]`
/// - `type_feat`:   `[B, N, TYPE_BLOCK_SIZE]`
pub fn split_obj_feat<B: Backend>(
    obj_feat: Tensor<B, 3>,
) -> (Tensor<B, 3>, Tensor<B, 2>, Tensor<B, 3>, Tensor<B, 3>) {
    let type_onehot = obj_feat
        .clone()
        .narrow(2, SLOT_TYPE_ONEHOT, TYPE_ONEHOT_SIZE);
    let is_present = obj_feat
        .clone()
        .narrow(2, SLOT_IS_PRESENT, 1)
        .squeeze_dim(2);
    let core_feat = obj_feat.clone().narrow(2, CORE_BLOCK_START, CORE_FEAT_SIZE);
    let type_feat = obj_feat.narrow(2, TYPE_BLOCK_START, TYPE_BLOCK_SIZE);
    (type_onehot, is_present, core_feat, type_feat)
}

/// Shared policy and value network.
///
/// ## Architecture
///
/// 1. **Object encoder** (type-conditioned):
///    Each entity type (Ship, Asteroid, Planet, Pickup) gets its own linear
///    projection via a shared weight tensor `obj_embed_w [T, H, F]` and bias
///    `obj_embed_b [T, H]`, where T=4 entity types, H=hidden_dim, F=feature_dim
///    (slot features excluding the type one-hot).
///
///    Given `one_hot [B,N,T]` and `feat [B,N,F]`:
///    ```text
///    W_blended = einsum("thf, bnt -> bnhf", obj_embed_w, one_hot)  // [B,N,H,F]
///    b_blended = einsum("th, bnt -> bnh", obj_embed_b, one_hot)    // [B,N,H]
///    obj_enc   = einsum("bnhf, bnf -> bnh", W_blended, feat) + b_blended
///    ```
///    Followed by `2×NetBlock`.
///
/// 2. **Self encoder**: `self_features [B,SELF_DIM] → [B,H]` via 2-layer MLP.
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

    // Object encoder — dual embedding:
    //   core_embed:  shared Linear for core features (same meaning for all types)
    //   type_embed_w / type_embed_b:  type-conditioned for type-specific features
    /// Shared linear for core features: `[CORE_FEAT_SIZE] → [H]`.
    core_embed: Linear<B>,
    /// Type-conditioned weight: `[N_ENTITY_TYPES, hidden_dim, TYPE_BLOCK_SIZE]`.
    type_embed_w: Param<Tensor<B, 3>>,
    /// Type-conditioned bias: `[N_ENTITY_TYPES, hidden_dim]`.
    type_embed_b: Param<Tensor<B, 2>>,
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
        // Type-conditioned embedding for type-specific features.
        let type_std = (2.0 / (TYPE_BLOCK_SIZE + hidden_dim) as f64).sqrt();
        let type_w = Tensor::<B, 3>::random(
            [N_ENTITY_TYPES, hidden_dim, TYPE_BLOCK_SIZE],
            burn::tensor::Distribution::Normal(0.0, type_std),
            device,
        );
        let type_b = Tensor::<B, 2>::zeros([N_ENTITY_TYPES, hidden_dim], device);

        Self {
            hidden_dim,
            core_embed: LinearConfig::new(CORE_FEAT_SIZE, hidden_dim).init(device),
            type_embed_w: Param::from_tensor(type_w),
            type_embed_b: Param::from_tensor(type_b),
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
            target_q_proj: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            target_k_proj: LinearConfig::new(hidden_dim, hidden_dim).init(device),
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
        let (type_onehot, is_present, core_feat, type_feat) = split_obj_feat(obj_feat);

        // ── Dual object embedding ──────────────────────────────────────────
        //
        // 1. core_embed (shared Linear): core_feat [B,N,C] → [B,N,H]
        //    All entity types use the same projection since these features
        //    (rel_pos, pursuit_angle, etc.) have the same meaning for all types.
        //
        // 2. type_embed (type-conditioned):
        //    type_embed_w [T, H, S] blended via type_onehot [B,N,T]:
        //      W_blended = einsum("ths, bnt -> bnhs", type_embed_w, type_onehot)
        //      b_blended = einsum("th, bnt -> bnh", type_embed_b, type_onehot)
        //      type_enc  = einsum("bnhs, bns -> bnh", W_blended, type_feat) + b_blended
        //
        // 3. obj_enc = core_enc + type_enc    (summed, then through NetBlocks)

        let h = self.hidden_dim;

        // Core embedding: [B, N, CORE_FEAT_SIZE] → [B, N, H]
        let core_enc = self.core_embed.forward(core_feat);

        // Type-conditioned embedding via matmul with one-hot blending:
        // W: [T, H*S] → one_hot [B,N,T] @ W [T,H*S] → [B,N,H*S] → [B,N,H,S]
        let ts = TYPE_BLOCK_SIZE;
        let w_flat = self.type_embed_w.val().reshape([N_ENTITY_TYPES, h * ts]);
        let w_blended = type_onehot.clone().matmul(w_flat.unsqueeze()); // [B, N, H*S]
        let [batch_size, n_ents, _] = w_blended.dims();
        let w_blended = w_blended.reshape([batch_size, n_ents, h, ts]); // [B, N, H, S]

        // b: one_hot [B,N,T] @ b [T,H] → [B,N,H]
        let b_blended = type_onehot.matmul(self.type_embed_b.val().unsqueeze()); // [B, N, H]

        // Batched matmul: [B,N,H,S] × [B,N,S,1] → [B,N,H,1] → [B,N,H]
        let type_feat_col = type_feat.unsqueeze_dim::<4>(3); // [B, N, S, 1]
        let type_enc = w_blended.matmul(type_feat_col).squeeze_dim(3); // [B, N, H]
        let type_enc = type_enc + b_blended;

        // Sum and refine through NetBlocks.
        let obj_enc = core_enc + type_enc;
        let obj_enc = self.eblock1.forward(obj_enc);
        let obj_enc = self.eblock2.forward(obj_enc);

        // Self encoder: [B, SELF_INPUT_DIM] → [B, H]
        let self_enc = silu(self.self_embed.forward(self_feat));
        let self_enc = silu(self.self_fc2.forward(self_enc));

        // Scaled dot-product attention (self as Q, objects as K/V)
        let scale = (self.hidden_dim as f64).sqrt() as f32;
        let q = self.q_proj.forward(self_enc.clone()).unsqueeze_dim::<3>(1); // [B, 1, H]
        let k = self.k_proj.forward(obj_enc.clone()).swap_dims(1, 2); // [B, H, N]
        let v = self.v_proj.forward(obj_enc.clone()); // [B, N, H]
        let scores = q.matmul(k).div_scalar(scale); // [B, 1, N]
        let weights = softmax(scores, 2); // [B, 1, N]
        let attn_out = weights.matmul(v).squeeze_dim(1); // [B, H]

        // Merge: concat(attn_out, self_enc) → H
        let merged = Tensor::cat(vec![attn_out, self_enc], 1); // [B, 2H]
        let x = silu(self.merge_proj.forward(merged.clone())); // [B, H]

        // Decoder → action logits
        let x = self.dblock1.forward(x);
        let x = self.dblock2.forward(x);
        let x = self.norm.forward(x);
        let x = silu(x);
        let action_logits = self.output.forward(x.clone()); // [B, output_dim]

        // Target-selection pointer head
        let tq = self.target_q_proj.forward(x.clone()).unsqueeze_dim::<3>(1); // [B, 1, H]
        let tk = self.target_k_proj.forward(obj_enc).swap_dims(1, 2); // [B, H, N]
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
        let target_logits =
            target_logits.clone() * mask.clone() + neg_inf * (mask.ones_like() - mask);

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
        self.net = RLNet::new(&self.device, HIDDEN_DIM, POLICY_OUTPUT_DIM).load_record(record);
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
pub fn logits_to_discrete_action(action_logits: &[f32], target_logits: &[f32]) -> DiscreteAction {
    debug_assert_eq!(action_logits.len(), POLICY_OUTPUT_DIM);
    debug_assert_eq!(target_logits.len(), TARGET_OUTPUT_DIM);
    let turn_idx = argmax(&action_logits[0..3]) as u8;
    let thrust_idx = argmax(&action_logits[3..5]) as u8;
    let fire_primary = argmax(&action_logits[5..7]) as u8;
    let fire_secondary = argmax(&action_logits[7..9]) as u8;
    let target_idx = argmax(target_logits) as u8;
    (
        turn_idx,
        thrust_idx,
        fire_primary,
        fire_secondary,
        target_idx,
    )
}

/// Sample an action stochastically from the policy logits.
///
/// Returns `(action, log_prob)` where `log_prob` is the sum of per-head
/// log-probabilities for the sampled action.
pub fn sample_discrete_action(
    action_logits: &[f32],
    target_logits: &[f32],
    rng: &mut impl rand::Rng,
) -> (DiscreteAction, f32) {
    debug_assert_eq!(action_logits.len(), POLICY_OUTPUT_DIM);
    debug_assert_eq!(target_logits.len(), TARGET_OUTPUT_DIM);

    let mut total_log_prob: f32 = 0.0;

    let turn_idx = sample_categorical(&action_logits[0..3], rng, &mut total_log_prob);
    let thrust_idx = sample_categorical(&action_logits[3..5], rng, &mut total_log_prob);
    let fire_primary = sample_categorical(&action_logits[5..7], rng, &mut total_log_prob);
    let fire_secondary = sample_categorical(&action_logits[7..9], rng, &mut total_log_prob);
    let target_idx = sample_categorical(target_logits, rng, &mut total_log_prob);

    let action = (
        turn_idx as u8,
        thrust_idx as u8,
        fire_primary as u8,
        fire_secondary as u8,
        target_idx as u8,
    );
    (action, total_log_prob)
}

/// Sample from a categorical distribution defined by logits.
///
/// Applies the log-sum-exp trick for numerical stability, samples an index
/// from the resulting probabilities, and accumulates the log-prob of the
/// sampled index into `log_prob_acc`.
fn sample_categorical(logits: &[f32], rng: &mut impl rand::Rng, log_prob_acc: &mut f32) -> usize {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|l| (l - max_logit).exp()).sum();
    let log_sum_exp = max_logit + exp_sum.ln();

    // Sample from the distribution.
    let u: f32 = rng.r#gen();
    let mut cumulative = 0.0_f32;
    let mut sampled = logits.len() - 1; // fallback to last
    for (i, &l) in logits.iter().enumerate() {
        cumulative += (l - log_sum_exp).exp();
        if u < cumulative {
            sampled = i;
            break;
        }
    }

    *log_prob_acc += logits[sampled] - log_sum_exp;
    sampled
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
            let net = RLNet::<TrainBackend>::new(device, HIDDEN_DIM, POLICY_OUTPUT_DIM)
                .load_record(record);
            println!("[model] Loaded training net from {path}");
            Some(net)
        }
        Err(e) => {
            eprintln!("[model] Failed to deserialise training net from {path}: {e}");
            None
        }
    }
}

/// Like [`load_training_net`] but with an explicit `output_dim`, so it can
/// load networks with a non-policy output size (e.g. the value network).
pub fn load_training_net_with_dim(
    path: &str,
    device: &<TrainBackend as Backend>::Device,
    output_dim: usize,
) -> Option<RLNet<TrainBackend>> {
    use burn::record::{BinBytesRecorder, BinFileRecorder, FullPrecisionSettings, Recorder};
    // Load directly from file into the inference backend with the given output_dim.
    let infer_device: <InferBackend as Backend>::Device = Default::default();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    let infer_net = match RLNet::<InferBackend>::new(&infer_device, HIDDEN_DIM, output_dim)
        .load_file(path, &recorder, &infer_device)
    {
        Ok(net) => net,
        Err(e) => {
            eprintln!("[model] Failed to load net (dim={output_dim}) from {path}: {e}");
            return None;
        }
    };
    // Transfer to training backend via bytes.
    let bytes = net_to_bytes(infer_net);
    let rec = BinBytesRecorder::<FullPrecisionSettings>::default();
    match Recorder::<TrainBackend>::load(&rec, bytes, device) {
        Ok(record) => {
            let net =
                RLNet::<TrainBackend>::new(device, HIDDEN_DIM, output_dim).load_record(record);
            println!("[model] Loaded training net (dim={output_dim}) from {path}");
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
// Optimizer save / load
// ---------------------------------------------------------------------------

/// Save an optimizer's state to `path` (`.bin` extension appended).
pub fn save_optimizer(
    optim: &OptimizerAdaptor<Adam, RLNet<TrainBackend>, TrainBackend>,
    path: &str,
) {
    use burn::optim::Optimizer;
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    let record = optim.to_record();
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    match Recorder::<TrainBackend>::record(&recorder, record, ()) {
        Ok(bytes) => {
            let file = format!("{path}.bin");
            if let Err(e) = std::fs::write(&file, bytes) {
                eprintln!("[model] Failed to save optimizer to {file}: {e}");
            } else {
                println!("[model] Optimizer saved to {file}");
            }
        }
        Err(e) => eprintln!("[model] Failed to serialize optimizer: {e}"),
    }
}

/// Load an optimizer's state from `path` (`.bin` extension appended).
/// Returns `None` on failure.
pub fn load_optimizer(
    path: &str,
    device: &<TrainBackend as Backend>::Device,
) -> Option<OptimizerAdaptor<Adam, RLNet<TrainBackend>, TrainBackend>> {
    use burn::optim::Optimizer;
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    let file = format!("{path}.bin");
    let bytes = std::fs::read(&file).ok()?;
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    match Recorder::<TrainBackend>::load(&recorder, bytes, device) {
        Ok(record) => {
            let optim = AdamConfig::new()
                .init::<TrainBackend, RLNet<TrainBackend>>()
                .load_record(record);
            println!("[model] Optimizer loaded from {file}");
            Some(optim)
        }
        Err(e) => {
            eprintln!("[model] Failed to load optimizer from {file}: {e}");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "tests/model_tests.rs"]
mod tests;
