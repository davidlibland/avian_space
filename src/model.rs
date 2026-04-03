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
    K_HOSTILE_SHIPS, K_PICKUPS, K_PLANETS, K_PROJECTILES, N_ENTITY_SLOTS, N_ENTITY_TYPES, OBS_DIM,
    PROJ_SLOT_SIZE, SELF_SIZE, SLOT_IS_PRESENT, SLOT_SIZE, SLOT_TYPE_ONEHOT, TYPE_BLOCK_SIZE,
    TYPE_BLOCK_START, TYPE_ONEHOT_SIZE,
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

/// Number of projectile slots per observation.
pub const N_PROJECTILE_SLOTS: usize = K_PROJECTILES;

/// Per-projectile feature count.
pub const PROJ_INPUT_DIM: usize = PROJ_SLOT_SIZE;

/// Total projectile floats per observation.
pub const PROJECTILES_FLAT_DIM: usize = N_PROJECTILE_SLOTS * PROJ_INPUT_DIM;

/// Number of object slots per observation.
pub const N_OBJECTS: usize = N_ENTITY_SLOTS;

/// Policy output: factored logits `turn(3) | thrust(2) | fire_primary(2) | fire_secondary(2)`.
pub const POLICY_OUTPUT_DIM: usize = 9;

/// Target-selection output: one logit per object slot plus one "no target" logit.
pub const TARGET_OUTPUT_DIM: usize = N_OBJECTS + 1;

/// Value output: one head per reward type.
pub const VALUE_OUTPUT_DIM: usize = crate::consts::N_REWARD_TYPES;

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

/// `LayerNorm → Linear(dim→dim, Xavier) → SiLU`, with residual connection.
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
        let residual = x.clone();
        let out = self.norm.forward(x);
        let out = self.fc.forward(out);
        residual + silu(out)
    }
}

type Block<B> = NetBlock<B>;

/// Number of attention heads for multi-head attention.
const N_HEADS: usize = 4;

/// Multi-head scaled dot-product attention.
///
/// * `q`: `[B, Nq, H]` queries
/// * `k`: `[B, Nk, H]` keys
/// * `v`: `[B, Nk, H]` values
/// * `mask`: optional `[B, Nk]` — 0.0 masks out that key position
///
/// Returns `[B, Nq, H]`.
fn multi_head_attention<B: Backend>(
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    mask: Option<Tensor<B, 2>>,
    n_heads: usize,
) -> Tensor<B, 3> {
    let [b, nq, h] = q.dims();
    let [_, nk, _] = k.dims();
    let head_dim = h / n_heads;
    let scale = (head_dim as f64).sqrt() as f32;

    // Reshape to [B, N, n_heads, head_dim] then transpose to [B, n_heads, N, head_dim]
    let q = q.reshape([b, nq, n_heads, head_dim]).swap_dims(1, 2); // [B, H_, Nq, D]
    let k = k.reshape([b, nk, n_heads, head_dim]).swap_dims(1, 2); // [B, H_, Nk, D]
    let v = v.reshape([b, nk, n_heads, head_dim]).swap_dims(1, 2); // [B, H_, Nk, D]

    // Scaled dot-product: [B, H_, Nq, D] @ [B, H_, D, Nk] → [B, H_, Nq, Nk]
    let scores = q.matmul(k.swap_dims(2, 3)).div_scalar(scale);

    // Apply mask if provided
    let scores = if let Some(m) = mask {
        // m: [B, Nk] → [B, 1, 1, Nk]
        let m = m.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(1);
        let neg_inf = scores.zeros_like().add_scalar(-1e9);
        scores.clone() * m.clone() + neg_inf * (m.ones_like() - m)
    } else {
        scores
    };

    let weights = softmax(scores, 3); // [B, H_, Nq, Nk]
    let out = weights.matmul(v); // [B, H_, Nq, D]

    // Transpose back and reshape: [B, Nq, H]
    out.swap_dims(1, 2).reshape([b, nq, h])
}

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
/// 1. **Entity encoder** (type-conditioned):
///    Dual embedding (core + type-specific) → 2×NetBlock (with residual).
///
/// 2. **Projectile encoder**: Linear → NetBlock (with residual).
///
/// 3. **Entity←Projectile cross-attention** (residual):
///    Entities (Q) attend to projectiles (K/V).
///    Masked so only ships and asteroids are updated (planets/pickups unchanged).
///
/// 4. **Self encoder**: 2-layer MLP → [B, H].
///
/// 5. **Self←Entity multi-head attention** (4 heads): self queries entities.
///
/// 6. **Self←Projectile multi-head attention** (4 heads): self queries projectiles.
///
/// 7. **Merge**: concat(entity_attn, proj_attn, self_enc) [B, 3H] → [B, H].
///
/// 8. **Decoder**: 2×NetBlock (with residual) + LayerNorm + SiLU → action logits.
///
/// 9. **Target head** (pointer network): Q from decoded state, K from
///    per-entity embeddings (post cross-attention) → one logit per entity + "no target".
///    Empty entity slots masked to -inf.
#[derive(Module, Debug)]
pub struct RLNet<B: Backend> {
    #[module(skip)]
    hidden_dim: usize,

    // ── Entity encoder (type-conditioned dual embedding) ──────────────────
    core_embed: Linear<B>,
    type_embed_w: Param<Tensor<B, 3>>,
    type_embed_b: Param<Tensor<B, 2>>,
    eblock1: Block<B>,
    eblock2: Block<B>,

    // ── Projectile encoder ────────────────────────────────────────────────
    proj_embed: Linear<B>,
    proj_block: Block<B>,

    // ── Entity←Projectile cross-attention (single-head, residual) ─────────
    cross_q_proj: Linear<B>,
    cross_k_proj: Linear<B>,
    cross_v_proj: Linear<B>,
    cross_norm: LayerNorm<B>,

    // ── Self encoder ──────────────────────────────────────────────────────
    self_embed: Linear<B>,
    self_fc2: Linear<B>,

    // ── Self←Entity multi-head attention ───────────────────────────────────
    ent_q_proj: Linear<B>,
    ent_k_proj: Linear<B>,
    ent_v_proj: Linear<B>,

    // ── Self←Projectile multi-head attention ──────────────────────────────
    pq_proj: Linear<B>,
    pk_proj: Linear<B>,
    pv_proj: Linear<B>,

    // ── Merge + Decoder ──────────────────────────────────────────────────
    merge_proj: Linear<B>,
    dblock1: Block<B>,
    dblock2: Block<B>,
    norm: LayerNorm<B>,
    output: Linear<B>,

    // ── Target-selection pointer head ──────────────────────────────────────
    target_q_proj: Linear<B>,
    target_k_proj: Linear<B>,
    target_no_tgt: Linear<B>,
}

impl<B: Backend> RLNet<B> {
    pub fn new(device: &B::Device, hidden_dim: usize, output_dim: usize) -> Self {
        let lin = |i, o| LinearConfig::new(i, o).init(device);
        let lin_xavier = |i, o| {
            LinearConfig::new(i, o)
                .with_initializer(Initializer::XavierNormal { gain: 1.0 })
                .init(device)
        };
        let lin_zero = |i, o| {
            LinearConfig::new(i, o)
                .with_initializer(Initializer::Zeros)
                .init(device)
        };

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

            // Entity encoder
            core_embed: lin(CORE_FEAT_SIZE, hidden_dim),
            type_embed_w: Param::from_tensor(type_w),
            type_embed_b: Param::from_tensor(type_b),
            eblock1: Block::new(device, hidden_dim),
            eblock2: Block::new(device, hidden_dim),

            // Projectile encoder
            proj_embed: lin(PROJ_INPUT_DIM, hidden_dim),
            proj_block: Block::new(device, hidden_dim),

            // Entity←Projectile cross-attention
            cross_q_proj: lin_xavier(hidden_dim, hidden_dim),
            cross_k_proj: lin_xavier(hidden_dim, hidden_dim),
            cross_v_proj: lin_xavier(hidden_dim, hidden_dim),
            cross_norm: LayerNormConfig::new(hidden_dim).init(device),

            // Self encoder
            self_embed: lin(SELF_INPUT_DIM, hidden_dim),
            self_fc2: lin(hidden_dim, hidden_dim),

            // Self←Entity multi-head attention
            ent_q_proj: lin_xavier(hidden_dim, hidden_dim),
            ent_k_proj: lin_xavier(hidden_dim, hidden_dim),
            ent_v_proj: lin_xavier(hidden_dim, hidden_dim),

            // Self←Projectile multi-head attention
            pq_proj: lin_xavier(hidden_dim, hidden_dim),
            pk_proj: lin_xavier(hidden_dim, hidden_dim),
            pv_proj: lin_xavier(hidden_dim, hidden_dim),

            // Merge + Decoder
            merge_proj: lin(hidden_dim * 3, hidden_dim),
            dblock1: Block::new(device, hidden_dim),
            dblock2: Block::new(device, hidden_dim),
            norm: LayerNormConfig::new(hidden_dim).init(device),
            output: lin_zero(hidden_dim, output_dim),

            // Target head
            target_q_proj: lin_xavier(hidden_dim, hidden_dim),
            target_k_proj: lin_xavier(hidden_dim, hidden_dim),
            target_no_tgt: lin_zero(hidden_dim, 1),
        }
    }

    /// Forward pass.
    ///
    /// * `self_feat`: `[B, SELF_INPUT_DIM]`
    /// * `obj_feat`:  `[B, N_OBJECTS, OBJECT_INPUT_DIM]`
    /// * `proj_feat`: `[B, K_PROJECTILES, PROJ_INPUT_DIM]`
    ///
    /// Returns `(action_logits [B, output_dim], target_logits [B, N_OBJECTS+1])`.
    pub fn forward(
        &self,
        self_feat: Tensor<B, 2>,
        obj_feat: Tensor<B, 3>,
        proj_feat: Tensor<B, 3>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let h = self.hidden_dim;
        let (type_onehot, is_present, core_feat, type_feat) = split_obj_feat(obj_feat);

        // ── 1. Entity embedding (type-conditioned) ───────────────────────────
        let core_enc = self.core_embed.forward(core_feat);

        let ts = TYPE_BLOCK_SIZE;
        let w_flat = self.type_embed_w.val().reshape([N_ENTITY_TYPES, h * ts]);
        let w_blended = type_onehot.clone().matmul(w_flat.unsqueeze());
        let [batch_size, n_ents, _] = w_blended.dims();
        let w_blended = w_blended.reshape([batch_size, n_ents, h, ts]);
        let b_blended = type_onehot.clone().matmul(self.type_embed_b.val().unsqueeze());
        let type_feat_col = type_feat.unsqueeze_dim::<4>(3);
        let type_enc = w_blended.matmul(type_feat_col).squeeze_dim(3) + b_blended;

        let ent_enc = core_enc + type_enc;
        let ent_enc = self.eblock1.forward(ent_enc); // residual inside
        let ent_enc = self.eblock2.forward(ent_enc); // [B, N, H]

        // ── 2. Projectile embedding ──────────────────────────────────────────
        let proj_enc = silu(self.proj_embed.forward(proj_feat.clone()));
        let proj_enc = self.proj_block.forward(proj_enc); // [B, Np, H]

        // Projectile presence mask: first float of each projectile slot.
        let proj_present: Tensor<B, 2> = proj_feat.clone().narrow(2, 0, 1).squeeze_dim::<2>(2);

        // ── 3. Entity←Projectile cross-attention (residual, masked) ──────────
        // Only ships (type 0) and asteroids (type 1) attend to projectiles.
        // Planets (type 2) and pickups (type 3) are unchanged.
        let ship_or_asteroid_mask: Tensor<B, 2> = {
            // type_onehot [B, N, 4]: columns 0=ship, 1=asteroid
            let ship_col: Tensor<B, 2> = type_onehot.clone().narrow(2, 0, 1).squeeze_dim::<2>(2);
            let ast_col: Tensor<B, 2> = type_onehot.clone().narrow(2, 1, 1).squeeze_dim::<2>(2);
            (ship_col + ast_col).clamp(0.0, 1.0)
        };

        let cross_q = self.cross_q_proj.forward(self.cross_norm.forward(ent_enc.clone()));
        let cross_k = self.cross_k_proj.forward(proj_enc.clone());
        let cross_v = self.cross_v_proj.forward(proj_enc.clone());

        // Single-head cross-attention: entities query projectiles
        let scale = (h as f64).sqrt() as f32;
        let scores = cross_q
            .matmul(cross_k.swap_dims(1, 2))
            .div_scalar(scale); // [B, N, Np]

        // Mask out absent projectiles
        let proj_mask_3d = proj_present
            .clone()
            .unsqueeze_dim::<3>(1)
            .repeat_dim(1, n_ents); // [B, N, Np]
        let neg_inf_cross =
            Tensor::<B, 3>::full([batch_size, n_ents, K_PROJECTILES], -1e9, &scores.device());
        let scores = scores.clone() * proj_mask_3d.clone()
            + neg_inf_cross * (proj_mask_3d.ones_like() - proj_mask_3d);

        let cross_weights = softmax(scores, 2); // [B, N, Np]
        let cross_out = cross_weights.matmul(cross_v); // [B, N, H]

        // Apply residual only to ships and asteroids
        let gate = ship_or_asteroid_mask.unsqueeze_dim::<3>(2); // [B, N, 1]
        let ent_enc = ent_enc + cross_out * gate; // [B, N, H]

        // ── 4. Self encoder ──────────────────────────────────────────────────
        let self_enc = silu(self.self_embed.forward(self_feat));
        let self_enc = silu(self.self_fc2.forward(self_enc)); // [B, H]

        // ── 5. Self←Entity multi-head attention ──────────────────────────────
        let sq = self_enc.clone().unsqueeze_dim::<3>(1); // [B, 1, H]
        let ent_q = self.ent_q_proj.forward(sq);
        let ent_k = self.ent_k_proj.forward(ent_enc.clone());
        let ent_v = self.ent_v_proj.forward(ent_enc.clone());
        let ent_attn = multi_head_attention(ent_q, ent_k, ent_v, None, N_HEADS)
            .squeeze_dim(1); // [B, H]

        // ── 6. Self←Projectile multi-head attention ──────────────────────────
        let pq = self.pq_proj.forward(self_enc.clone().unsqueeze_dim::<3>(1));
        let pk = self.pk_proj.forward(proj_enc.clone());
        let pv = self.pv_proj.forward(proj_enc);
        let proj_attn = multi_head_attention(pq, pk, pv, Some(proj_present), N_HEADS)
            .squeeze_dim(1); // [B, H]

        // ── 7. Merge ─────────────────────────────────────────────────────────
        let merged = Tensor::cat(vec![ent_attn, proj_attn, self_enc], 1); // [B, 3H]
        let x = silu(self.merge_proj.forward(merged)); // [B, H]

        // ── 8. Decoder → action logits ───────────────────────────────────────
        let x = self.dblock1.forward(x);
        let x = self.dblock2.forward(x);
        let x = self.norm.forward(x);
        let x = silu(x);
        let action_logits = self.output.forward(x.clone()); // [B, output_dim]

        // ── 9. Target head (pointer from decoded state to entity embeddings) ─
        let tq = self.target_q_proj.forward(x.clone()).unsqueeze_dim::<3>(1); // [B, 1, H]
        let tk = self.target_k_proj.forward(ent_enc).swap_dims(1, 2); // [B, H, N]
        let entity_logits = tq.matmul(tk).div_scalar(scale).squeeze_dim(1); // [B, N]

        let no_tgt_logit = self.target_no_tgt.forward(x); // [B, 1]
        let target_logits = Tensor::cat(vec![entity_logits, no_tgt_logit], 1); // [B, N+1]

        // Mask empty entity slots to -inf.
        let device = is_present.device();
        let ones = Tensor::<B, 2>::ones([batch_size, 1], &device);
        let mask = Tensor::cat(vec![is_present, ones], 1);
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
    /// * `proj_flat`: flattened `[batch_size × K_PROJECTILES × PROJ_INPUT_DIM]`
    ///
    /// Returns `(action_logits, target_logits)`.
    pub fn run_inference(
        &self,
        self_flat: Vec<f32>,
        obj_flat: Vec<f32>,
        proj_flat: Vec<f32>,
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
        let proj_input = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(proj_flat, [batch_size, K_PROJECTILES, PROJ_INPUT_DIM]),
            &self.device,
        );
        let (action, target) = self.net.forward(self_input, obj_input, proj_input);
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
