//! Value-function helpers for PPO training.
//!
//! Provides batched (detached) multi-head value inference, bootstrap
//! recomputation, and the Huber value-function loss.

use burn::{
    prelude::*,
    tensor::{Tensor, TensorData, backend::AutodiffBackend},
};

use crate::consts::N_REWARD_TYPES;
use crate::model::{
    self, InnerTrainBackend, N_OBJECTS, OBJECT_INPUT_DIM, RLNet, SELF_INPUT_DIM, TrainBackend,
};
use crate::rl_collection::Segment;

// Maximum steps to forward through the value net at once (avoid OOM).
const VALUE_CHUNK_SIZE: usize = 2048;

// ---------------------------------------------------------------------------
// Batched multi-head value inference (detached — no gradient tape)
// ---------------------------------------------------------------------------

/// Run the value network on all observations and return per-head predictions.
///
/// Uses `.valid()` to strip the autodiff wrapper so no gradient tape is built.
/// Returns `Vec<[f32; N_REWARD_TYPES]>` of length `total_steps`.
pub fn batch_value_inference(
    value_net: &RLNet<TrainBackend>,
    self_flat: &[f32],
    obj_flat: &[f32],
    proj_flat: &[f32],
    total_steps: usize,
    _device: &<TrainBackend as Backend>::Device,
) -> Vec<[f32; N_REWARD_TYPES]> {
    use burn::module::AutodiffModule;

    let inner_net = value_net.clone().valid();
    let inner_device: <InnerTrainBackend as Backend>::Device = Default::default();
    let mut values = Vec::with_capacity(total_steps);

    for chunk_start in (0..total_steps).step_by(VALUE_CHUNK_SIZE) {
        let chunk_end = (chunk_start + VALUE_CHUNK_SIZE).min(total_steps);
        let b = chunk_end - chunk_start;

        let self_offset = chunk_start * SELF_INPUT_DIM;
        let self_slice = &self_flat[self_offset..self_offset + b * SELF_INPUT_DIM];
        let obj_offset = chunk_start * N_OBJECTS * OBJECT_INPUT_DIM;
        let obj_slice = &obj_flat[obj_offset..obj_offset + b * N_OBJECTS * OBJECT_INPUT_DIM];
        let proj_offset = chunk_start * model::PROJECTILES_FLAT_DIM;
        let proj_slice = &proj_flat[proj_offset..proj_offset + b * model::PROJECTILES_FLAT_DIM];

        let self_t = Tensor::<InnerTrainBackend, 2>::from_data(
            TensorData::new(self_slice.to_vec(), [b, SELF_INPUT_DIM]),
            &inner_device,
        );
        let obj_t = Tensor::<InnerTrainBackend, 3>::from_data(
            TensorData::new(obj_slice.to_vec(), [b, N_OBJECTS, OBJECT_INPUT_DIM]),
            &inner_device,
        );
        let proj_t = Tensor::<InnerTrainBackend, 3>::from_data(
            TensorData::new(proj_slice.to_vec(), [b, model::N_PROJECTILE_SLOTS, model::PROJ_INPUT_DIM]),
            &inner_device,
        );

        let (value_out, _, _) = inner_net.forward(self_t, obj_t, proj_t);
        // value_out: [B, N_REWARD_TYPES]
        let flat: Vec<f32> = value_out.into_data().to_vec().expect("f32 conversion");
        for row in flat.chunks_exact(N_REWARD_TYPES) {
            let mut arr = [0.0_f32; N_REWARD_TYPES];
            arr.copy_from_slice(row);
            values.push(arr);
        }
    }

    values
}

// ---------------------------------------------------------------------------
// Bootstrap recomputation
// ---------------------------------------------------------------------------

/// For segments with `bootstrap_value == Some(_)` (truncated, not terminal),
/// recompute per-head V(s_last) using the value net on the last observation.
pub fn recompute_bootstrap_values(
    segments: &mut [Segment],
    value_net: &RLNet<TrainBackend>,
    _device: &<TrainBackend as Backend>::Device,
) {
    use burn::module::AutodiffModule;

    let inner_net = value_net.clone().valid();
    let inner_device: <InnerTrainBackend as Backend>::Device = Default::default();

    for seg in segments.iter_mut() {
        if seg.bootstrap_value.is_none() {
            continue; // terminal — bootstrap stays at None
        }
        if let Some(last_t) = seg.transitions.last() {
            let (s, o) = model::split_obs(&last_t.obs);
            let self_t = Tensor::<InnerTrainBackend, 2>::from_data(
                TensorData::new(s.to_vec(), [1, SELF_INPUT_DIM]),
                &inner_device,
            );
            let obj_t = Tensor::<InnerTrainBackend, 3>::from_data(
                TensorData::new(o.to_vec(), [1, N_OBJECTS, OBJECT_INPUT_DIM]),
                &inner_device,
            );
            let proj_t = Tensor::<InnerTrainBackend, 3>::from_data(
                TensorData::new(last_t.proj_obs.clone(), [1, model::N_PROJECTILE_SLOTS, model::PROJ_INPUT_DIM]),
                &inner_device,
            );
            let (val, _, _) = inner_net.forward(self_t, obj_t, proj_t);
            // val: [1, N_REWARD_TYPES] → extract as array.
            let flat: Vec<f32> = val.into_data().to_vec().expect("f32 conversion");
            let mut bv = [0.0_f32; N_REWARD_TYPES];
            bv.copy_from_slice(&flat[..N_REWARD_TYPES]);
            seg.bootstrap_value = Some(bv);
        }
    }
}

// ---------------------------------------------------------------------------
// Value loss (Huber) — works on 2-D tensors for multi-head
// ---------------------------------------------------------------------------

/// Per-element Huber loss between predicted and target tensors, averaged.
///
/// Works for any dimensionality — operates element-wise then takes the mean.
pub fn huber_value_loss<B: AutodiffBackend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
    delta: f32,
) -> Tensor<B, 1> {
    let diff = predictions - targets;
    let abs_diff = diff.clone().abs();
    let quadratic = diff.clone() * diff * 0.5;
    let linear = abs_diff.clone() * delta - 0.5 * delta * delta;
    let mask = abs_diff.lower_equal_elem(delta).float();
    let loss = quadratic * mask.clone() + linear * (mask.ones_like() - mask);
    loss.mean()
}
