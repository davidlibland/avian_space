//! Value-function helpers for PPO training.
//!
//! Provides batched (detached) value inference, bootstrap recomputation,
//! and the Huber value-function loss used by the PPO training loop.

use burn::{
    prelude::*,
    tensor::{
        Tensor, TensorData,
        backend::AutodiffBackend,
    },
};

use crate::model::{
    self, N_OBJECTS, OBJECT_INPUT_DIM, RLNet, SELF_INPUT_DIM,
    TrainBackend,
};
use crate::rl_collection::Segment;

// Maximum steps to forward through the value net at once (avoid OOM).
const VALUE_CHUNK_SIZE: usize = 2048;

// ---------------------------------------------------------------------------
// Batched value inference (detached — no gradient tape)
// ---------------------------------------------------------------------------

/// Run the value network on all observations and return scalar predictions.
///
/// Uses `.valid()` to strip the autodiff wrapper so no gradient tape is built.
/// The result is a plain `Vec<f32>` suitable for GAE computation.
pub fn batch_value_inference(
    value_net: &RLNet<TrainBackend>,
    self_flat: &[f32],
    obj_flat: &[f32],
    total_steps: usize,
    _device: &<TrainBackend as Backend>::Device,
) -> Vec<f32> {
    use burn::module::AutodiffModule;

    let inner_net = value_net.clone().valid(); // RLNet<Wgpu>
    let inner_device = Default::default();
    let mut values = Vec::with_capacity(total_steps);

    for chunk_start in (0..total_steps).step_by(VALUE_CHUNK_SIZE) {
        let chunk_end = (chunk_start + VALUE_CHUNK_SIZE).min(total_steps);
        let b = chunk_end - chunk_start;

        let self_offset = chunk_start * SELF_INPUT_DIM;
        let self_slice = &self_flat[self_offset..self_offset + b * SELF_INPUT_DIM];
        let obj_offset = chunk_start * N_OBJECTS * OBJECT_INPUT_DIM;
        let obj_slice = &obj_flat[obj_offset..obj_offset + b * N_OBJECTS * OBJECT_INPUT_DIM];

        let self_t = Tensor::<burn::backend::wgpu::Wgpu, 2>::from_data(
            TensorData::new(self_slice.to_vec(), [b, SELF_INPUT_DIM]),
            &inner_device,
        );
        let obj_t = Tensor::<burn::backend::wgpu::Wgpu, 3>::from_data(
            TensorData::new(obj_slice.to_vec(), [b, N_OBJECTS, OBJECT_INPUT_DIM]),
            &inner_device,
        );

        let (value_out, _target_logits) = inner_net.forward(self_t, obj_t);
        // value_out: [B, 1] → squeeze to [B]
        let data = value_out.squeeze_dim::<1>(1).into_data();
        let chunk_vals: Vec<f32> = data.to_vec().expect("f32 conversion");
        values.extend_from_slice(&chunk_vals);
    }

    values
}

// ---------------------------------------------------------------------------
// Bootstrap recomputation
// ---------------------------------------------------------------------------

/// For segments with `bootstrap_value == Some(_)` (truncated, not terminal),
/// recompute V(s_last) using the value net on the last observation.
///
/// This replaces the placeholder `Some(0.0)` set by the game thread.
pub fn recompute_bootstrap_values(
    segments: &mut [Segment],
    value_net: &RLNet<TrainBackend>,
    _device: &<TrainBackend as Backend>::Device,
) {
    use burn::module::AutodiffModule;

    let inner_net = value_net.clone().valid();
    let inner_device = Default::default();

    for seg in segments.iter_mut() {
        if seg.bootstrap_value.is_none() {
            continue; // terminal — bootstrap stays at 0
        }
        if let Some(last_t) = seg.transitions.last() {
            let (s, o) = model::split_obs(&last_t.obs);
            let self_t = Tensor::<burn::backend::wgpu::Wgpu, 2>::from_data(
                TensorData::new(s.to_vec(), [1, SELF_INPUT_DIM]),
                &inner_device,
            );
            let obj_t = Tensor::<burn::backend::wgpu::Wgpu, 3>::from_data(
                TensorData::new(o.to_vec(), [1, N_OBJECTS, OBJECT_INPUT_DIM]),
                &inner_device,
            );
            let (val, _) = inner_net.forward(self_t, obj_t);
            // val is [1, 1] — flatten to scalar.
            let v: f32 = val.squeeze_dim::<1>(1).into_scalar().into();
            seg.bootstrap_value = Some(v);
        }
    }
}

// ---------------------------------------------------------------------------
// Value loss (Huber)
// ---------------------------------------------------------------------------

/// Huber loss between predicted values and regression targets.
///
/// `delta` controls the transition from quadratic to linear loss.
pub fn huber_value_loss<B: AutodiffBackend>(
    predictions: Tensor<B, 1>,
    targets: Tensor<B, 1>,
    delta: f32,
) -> Tensor<B, 1> {
    let diff = predictions - targets;
    let abs_diff = diff.clone().abs();
    let quadratic = diff.clone() * diff * 0.5;
    let linear = abs_diff.clone() * delta - 0.5 * delta * delta;
    // mask: 1 where |diff| <= delta, 0 otherwise
    let mask = abs_diff.lower_equal_elem(delta).float();
    let loss = quadratic * mask.clone() + linear * (mask.ones_like() - mask);
    loss.mean()
}
