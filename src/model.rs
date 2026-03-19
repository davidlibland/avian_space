//! Neural-network architecture for the RL agent.
//!
//! Input shape: `[B, N_OBJECTS, NET_INPUT_DIM]` where every row contains
//! `[self_features(SELF_INPUT_DIM) | object_features(OBJECT_INPUT_DIM)]`.
//! The self features are **identical in every row**; the forward pass takes
//! row 0 for the self-encoder and ignores the rest for self-state.
//!
//! Observation → tensor conversion is handled by [`obs_to_model_input`].
//! Policy logits → [`DiscreteAction`] conversion is handled by
//! [`logits_to_discrete_action`] (greedy argmax; stochastic sampling is
//! added at training time).

use std::sync::{Arc, Mutex};

use bevy::prelude::Resource;
use burn::{
    backend::{ndarray::NdArray, Autodiff},
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
    DiscreteAction, ENTITY_SLOT_SIZE, K_ASTEROIDS, K_FRIENDLY_SHIPS, K_HOSTILE_SHIPS, K_PICKUPS,
    K_PLANETS, OBS_DIM, SELF_SIZE, TARGET_SLOT_SIZE,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Self-state feature count (matches `SELF_SIZE` in `rl_obs.rs`).
pub const SELF_INPUT_DIM: usize = 10;

/// Per-object feature count:
/// `is_present(1) + type_onehot(4) + rel_pos(2) + rel_vel(2) + hostility(1) + extra(1)`
pub const OBJECT_INPUT_DIM: usize = 11;

/// Total floats per row fed into the network (`SELF_INPUT_DIM + OBJECT_INPUT_DIM`).
pub const NET_INPUT_DIM: usize = SELF_INPUT_DIM + OBJECT_INPUT_DIM;

/// Number of object slots per observation: target + all buckets.
pub const N_OBJECTS: usize =
    1 + K_PLANETS + K_ASTEROIDS + K_HOSTILE_SHIPS + K_FRIENDLY_SHIPS + K_PICKUPS;

/// Policy output: factored logits `turn(3) | thrust(2) | fire_primary(2) | fire_secondary(2)`.
pub const POLICY_OUTPUT_DIM: usize = 9;

/// Value output: scalar state estimate.
pub const VALUE_OUTPUT_DIM: usize = 1;

/// Default hidden dimension for all network layers.
pub const HIDDEN_DIM: usize = 64;

// ---------------------------------------------------------------------------
// Backend aliases
// ---------------------------------------------------------------------------

/// CPU backend used for game-thread inference.
pub type InferBackend = NdArray;

/// CPU backend with autodiff, used by the background training thread.
pub type TrainBackend = Autodiff<NdArray>;

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
        }
    }

    /// Forward pass.
    ///
    /// `xs`: `[B, N_OBJECTS, NET_INPUT_DIM]` — every row contains
    /// `[self_features | object_features]`; self features are identical in every row.
    ///
    /// Returns `[B, output_dim]` logits.
    pub fn forward(&self, xs: Tensor<B, 3>) -> Tensor<B, 2> {
        // Split: self state from row 0, object features from all rows.
        let self_feat = xs
            .clone()
            .narrow(2, 0, SELF_INPUT_DIM)   // [B, N, SELF_INPUT_DIM]
            .narrow(1, 0, 1)                // [B, 1, SELF_INPUT_DIM]
            .squeeze_dim(1);                // [B, SELF_INPUT_DIM]
        let obj_feat = xs.narrow(2, SELF_INPUT_DIM, OBJECT_INPUT_DIM); // [B, N, OBJECT_INPUT_DIM]

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
        let v = self.v_proj.forward(obj_enc);                                // [B, N, H]
        let scores = q.matmul(k).div_scalar(scale);                          // [B, 1, N]
        let weights = softmax(scores, 2);                                    // [B, 1, N]
        let attn_out = weights.matmul(v).squeeze_dim(1);                     // [B, H]

        // Merge: concat(attn_out, self_enc) → H
        let merged = Tensor::cat(vec![attn_out, self_enc], 1); // [B, 2H]
        let x = silu(self.merge_proj.forward(merged));         // [B, H]

        // Decoder
        let x = self.dblock1.forward(x);
        let x = self.dblock2.forward(x);
        let x = self.norm.forward(x);
        let x = silu(x);
        self.output.forward(x) // [B, output_dim]
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
    /// `batch_flat`: flattened `[batch_size × N_OBJECTS × NET_INPUT_DIM]` buffer
    /// produced by calling [`obs_to_model_input`] for each observation and
    /// concatenating the results.
    ///
    /// Returns flattened `[batch_size × POLICY_OUTPUT_DIM]` logits.
    pub fn run_inference(&self, batch_flat: Vec<f32>, batch_size: usize) -> Vec<f32> {
        let input = Tensor::<InferBackend, 3>::from_data(
            TensorData::new(batch_flat, [batch_size, N_OBJECTS, NET_INPUT_DIM]),
            &self.device,
        );
        self.net
            .forward(input)
            .into_data()
            .into_vec::<f32>()
            .expect("inference logit extraction failed")
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

/// Convert a flat `OBS_DIM`=81 observation into the
/// `[N_OBJECTS × NET_INPUT_DIM]` buffer consumed by `RLNet::forward`.
///
/// Each of the `N_OBJECTS` rows contains:
/// `[self_features(10) | object_features(11)]`
///
/// where `self_features` are the same in every row, and `object_features`
/// follow the layout:
/// `is_present(1) | type_onehot(4) | rel_pos(2) | rel_vel(2) | hostility(1) | extra(1)`
pub fn obs_to_model_input(obs: &[f32]) -> Vec<f32> {
    debug_assert_eq!(obs.len(), OBS_DIM, "obs length mismatch");

    let self_feat = &obs[0..SELF_SIZE];
    let target_feat = &obs[SELF_SIZE..SELF_SIZE + TARGET_SLOT_SIZE];

    let mut rows = Vec::with_capacity(N_OBJECTS * NET_INPUT_DIM);

    // Row 0: target slot — already 11 floats in the correct object-feature layout.
    rows.extend_from_slice(self_feat);
    rows.extend_from_slice(target_feat);

    // Rows 1..: entity bucket slots.  Each 5-float slot is expanded to 11 floats.
    // Bucket order and types must match `encode_observation` in rl_obs.rs.
    let mut slot_offset = SELF_SIZE + TARGET_SLOT_SIZE;

    // (type_onehot [Ship,Asteroid,Planet,Pickup],  hostility,  slot_count)
    let bucket_specs: [([f32; 4], f32, usize); 5] = [
        ([0.0, 0.0, 1.0, 0.0],  0.0,  K_PLANETS),
        ([0.0, 1.0, 0.0, 0.0],  0.0,  K_ASTEROIDS),
        ([1.0, 0.0, 0.0, 0.0],  1.0,  K_HOSTILE_SHIPS),
        ([1.0, 0.0, 0.0, 0.0], -1.0,  K_FRIENDLY_SHIPS),
        ([0.0, 0.0, 0.0, 1.0],  0.0,  K_PICKUPS),
    ];

    for (type_onehot, hostility, count) in &bucket_specs {
        for _ in 0..*count {
            let slot = &obs[slot_offset..slot_offset + ENTITY_SLOT_SIZE];
            let is_present = if slot.iter().any(|&x| x != 0.0) { 1.0_f32 } else { 0.0 };

            rows.extend_from_slice(self_feat);      // [10] — self features
            rows.push(is_present);                  // [1]
            rows.extend_from_slice(type_onehot);    // [4]
            rows.extend_from_slice(&slot[0..4]);    // [4] rel_pos(2) + rel_vel(2)
            rows.push(*hostility);                  // [1]
            rows.push(slot[4]);                     // [1] extra (health frac / value)

            slot_offset += ENTITY_SLOT_SIZE;
        }
    }

    debug_assert_eq!(rows.len(), N_OBJECTS * NET_INPUT_DIM);
    rows
}

// ---------------------------------------------------------------------------
// Action conversion
// ---------------------------------------------------------------------------

/// Convert policy logits `[POLICY_OUTPUT_DIM=9]` to a `DiscreteAction` via
/// greedy argmax over each factored head.
///
/// Logit layout: `turn(3) | thrust(2) | fire_primary(2) | fire_secondary(2)`
pub fn logits_to_discrete_action(logits: &[f32]) -> DiscreteAction {
    debug_assert_eq!(logits.len(), POLICY_OUTPUT_DIM);
    let turn_idx = argmax(&logits[0..3]) as u8;
    let thrust_idx = argmax(&logits[3..5]) as u8;
    let fire_primary = argmax(&logits[5..7]) as u8;
    let fire_secondary = argmax(&logits[7..9]) as u8;
    (turn_idx, thrust_idx, fire_primary, fire_secondary)
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
mod tests {
    use super::*;
    use crate::rl_obs::OBS_DIM;

    // ── obs_to_model_input ───────────────────────────────────────────────────

    #[test]
    fn test_obs_to_model_input_shape() {
        let obs = vec![0.0_f32; OBS_DIM];
        let out = obs_to_model_input(&obs);
        assert_eq!(
            out.len(),
            N_OBJECTS * NET_INPUT_DIM,
            "expected {N_OBJECTS}×{NET_INPUT_DIM}={}, got {}",
            N_OBJECTS * NET_INPUT_DIM,
            out.len()
        );
    }

    #[test]
    fn test_obs_to_model_input_self_features_replicated() {
        // Give self features a distinctive pattern, everything else zero.
        let mut obs = vec![0.0_f32; OBS_DIM];
        for (i, v) in obs[0..SELF_SIZE].iter_mut().enumerate() {
            *v = (i + 1) as f32 * 0.1;
        }
        let out = obs_to_model_input(&obs);

        // Every row should start with the same SELF_INPUT_DIM values.
        let expected_self = &obs[0..SELF_INPUT_DIM];
        for row in 0..N_OBJECTS {
            let row_self = &out[row * NET_INPUT_DIM..row * NET_INPUT_DIM + SELF_INPUT_DIM];
            assert_eq!(
                row_self, expected_self,
                "self features differ in row {row}"
            );
        }
    }

    #[test]
    fn test_obs_to_model_input_target_passthrough() {
        let mut obs = vec![0.0_f32; OBS_DIM];
        // Write a distinctive pattern into the target slot.
        for (i, v) in obs[SELF_SIZE..SELF_SIZE + TARGET_SLOT_SIZE].iter_mut().enumerate() {
            *v = (i + 1) as f32 * 0.5;
        }
        let out = obs_to_model_input(&obs);

        // Row 0 object features (offset SELF_INPUT_DIM) should equal the target slot.
        let row0_obj = &out[SELF_INPUT_DIM..SELF_INPUT_DIM + OBJECT_INPUT_DIM];
        let expected = &obs[SELF_SIZE..SELF_SIZE + TARGET_SLOT_SIZE];
        assert_eq!(row0_obj, expected, "target slot not passed through to row 0");
    }

    #[test]
    fn test_obs_to_model_input_empty_slot_is_not_present() {
        // All-zero entity slot → is_present should be 0.
        let obs = vec![0.0_f32; OBS_DIM];
        let out = obs_to_model_input(&obs);

        // Every bucket row (rows 1..) starts with [self_feat | is_present=0 | ...]
        for row in 1..N_OBJECTS {
            let is_present = out[row * NET_INPUT_DIM + SELF_INPUT_DIM];
            assert_eq!(
                is_present, 0.0,
                "empty slot row {row} should have is_present=0"
            );
        }
    }

    // ── logits_to_discrete_action ────────────────────────────────────────────

    #[test]
    fn test_logits_to_discrete_action_argmax() {
        // Craft logits so each head has an obvious winner.
        // turn: [1.0, 5.0, 0.0]  → idx 1 (straight)
        // thrust: [0.0, 3.0]     → idx 1 (thrust)
        // fire_primary: [2.0, 0.0] → idx 0 (no fire)
        // fire_secondary: [0.0, 4.0] → idx 1 (fire)
        let logits = [1.0_f32, 5.0, 0.0,   0.0, 3.0,   2.0, 0.0,   0.0, 4.0];
        let (turn, thrust, fp, fs) = logits_to_discrete_action(&logits);
        assert_eq!(turn, 1, "turn should be straight");
        assert_eq!(thrust, 1, "thrust should be on");
        assert_eq!(fp, 0, "fire_primary should be off");
        assert_eq!(fs, 1, "fire_secondary should be on");
    }

    #[test]
    fn test_logits_to_discrete_action_all_valid() {
        // Zero logits → argmax picks index 0 everywhere → valid indices.
        let logits = [0.0_f32; POLICY_OUTPUT_DIM];
        let (turn, thrust, fp, fs) = logits_to_discrete_action(&logits);
        assert!(turn <= 2);
        assert!(thrust <= 1);
        assert!(fp <= 1);
        assert!(fs <= 1);
    }

    // ── RLNet forward-pass shapes ────────────────────────────────────────────

    #[test]
    fn test_rlnet_policy_output_shape() {
        let device = Default::default();
        let net = RLNet::<InferBackend>::new(&device, HIDDEN_DIM, POLICY_OUTPUT_DIM);
        let batch = 4usize;
        let input = Tensor::<InferBackend, 3>::zeros([batch, N_OBJECTS, NET_INPUT_DIM], &device);
        let out = net.forward(input);
        let shape = out.shape();
        assert_eq!(shape.dims[0], batch);
        assert_eq!(shape.dims[1], POLICY_OUTPUT_DIM);
    }

    #[test]
    fn test_rlnet_value_output_shape() {
        let device = Default::default();
        let net = RLNet::<InferBackend>::new(&device, HIDDEN_DIM, VALUE_OUTPUT_DIM);
        let batch = 3usize;
        let input = Tensor::<InferBackend, 3>::zeros([batch, N_OBJECTS, NET_INPUT_DIM], &device);
        let out = net.forward(input);
        let shape = out.shape();
        assert_eq!(shape.dims[0], batch);
        assert_eq!(shape.dims[1], VALUE_OUTPUT_DIM);
    }

    #[test]
    fn test_rlnet_output_zero_init() {
        // The output projection is zero-initialised, so a fresh net on a zero
        // input should produce all-zero logits.
        let device = Default::default();
        let net = RLNet::<InferBackend>::new(&device, HIDDEN_DIM, POLICY_OUTPUT_DIM);
        let input = Tensor::<InferBackend, 3>::zeros([1, N_OBJECTS, NET_INPUT_DIM], &device);
        let logits: Vec<f32> = net
            .forward(input)
            .into_data()
            .into_vec::<f32>()
            .expect("extraction failed");
        for (i, &v) in logits.iter().enumerate() {
            assert!(
                v.abs() < 1e-6,
                "logit[{i}] should be ~0 on zero input, got {v}"
            );
        }
    }

    // ── InferenceNet end-to-end ──────────────────────────────────────────────

    #[test]
    fn test_inference_net_produces_valid_action() {
        let inference = InferenceNet::new();
        let dummy_obs = vec![0.0_f32; OBS_DIM];
        let model_input = obs_to_model_input(&dummy_obs);
        let logits = inference.run_inference(model_input, 1);

        assert_eq!(logits.len(), POLICY_OUTPUT_DIM, "wrong logit count");
        let (turn, thrust, fp, fs) = logits_to_discrete_action(&logits);
        assert!(turn <= 2, "turn_idx out of range: {turn}");
        assert!(thrust <= 1, "thrust_idx out of range: {thrust}");
        assert!(fp <= 1, "fire_primary out of range: {fp}");
        assert!(fs <= 1, "fire_secondary out of range: {fs}");
    }

    #[test]
    fn test_inference_net_batched() {
        let inference = InferenceNet::new();
        let batch_size = 5;
        let single = obs_to_model_input(&vec![0.0_f32; OBS_DIM]);
        let batch_flat: Vec<f32> = single.iter().cloned().cycle().take(batch_size * single.len()).collect();
        let logits = inference.run_inference(batch_flat, batch_size);
        assert_eq!(logits.len(), batch_size * POLICY_OUTPUT_DIM);
    }
}
