//! PPO clipped surrogate loss and factored-action log-prob/entropy.

use burn::{
    prelude::*,
    tensor::{Tensor, TensorData, activation::log_softmax, backend::AutodiffBackend},
};

use crate::rl_obs::DiscreteAction;

/// Head descriptor: (offset into action_logits, num_classes).
const ACTION_HEADS: [(usize, usize); 4] = [
    (0, 3), // turn
    (3, 2), // thrust
    (5, 2), // fire_primary
    (7, 2), // fire_secondary
];

/// Compute per-sample log-probability and entropy for the factored action space.
///
/// Returns `(log_probs [B], entropy [B])` summed across all 6 heads
/// (4 action heads + nav + weapons target).
pub fn compute_log_probs_and_entropy<B: Backend>(
    action_logits: Tensor<B, 2>,
    nav_target_logits: Tensor<B, 2>,
    wep_target_logits: Tensor<B, 2>,
    actions: &[DiscreteAction],
    device: &B::Device,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let b = actions.len();

    let mut total_log_prob: Option<Tensor<B, 1>> = None;
    let mut total_entropy: Option<Tensor<B, 1>> = None;

    let mut accumulate_head = |logits: Tensor<B, 2>, indices: Vec<i64>| {
        let log_p = log_softmax(logits.clone(), 1);
        let idx_tensor = Tensor::<B, 2, Int>::from_data(TensorData::new(indices, [b, 1]), device);
        let gathered: Tensor<B, 1> = log_p
            .clone()
            .gather(1, idx_tensor)
            .squeeze_dim::<1>(1)
            .clamp(-20.0, 0.0);

        let p = burn::tensor::activation::softmax(logits, 1);
        let ent: Tensor<B, 1> = -(p * log_p).sum_dim(1).squeeze_dim::<1>(1);

        total_log_prob = Some(match total_log_prob.take() {
            Some(acc) => acc + gathered,
            None => gathered,
        });
        total_entropy = Some(match total_entropy.take() {
            Some(acc) => acc + ent,
            None => ent,
        });
    };

    for &(offset, num_classes) in &ACTION_HEADS {
        let head_logits = action_logits.clone().narrow(1, offset, num_classes);
        let head_action_fn: fn(&DiscreteAction) -> u8 = match offset {
            0 => |a| a.0,
            3 => |a| a.1,
            5 => |a| a.2,
            7 => |a| a.3,
            _ => unreachable!(),
        };
        let indices: Vec<i64> = actions.iter().map(|a| head_action_fn(a) as i64).collect();
        accumulate_head(head_logits, indices);
    }

    {
        let indices: Vec<i64> = actions.iter().map(|a| a.4 as i64).collect();
        accumulate_head(nav_target_logits, indices);
    }

    {
        let indices: Vec<i64> = actions.iter().map(|a| a.5 as i64).collect();
        accumulate_head(wep_target_logits, indices);
    }

    (total_log_prob.unwrap(), total_entropy.unwrap())
}

/// Diagnostics extracted from the PPO clipped loss computation.
pub struct PpoLossDiag {
    pub mean_ratio: f32,
    pub frac_clipped: f32,
}

/// PPO clipped surrogate loss. Returns `(loss, diagnostics)`.
pub fn ppo_clipped_loss<B: AutodiffBackend>(
    new_log_probs: Tensor<B, 1>,
    old_log_probs: &[f32],
    advantages: &[f32],
    clip_eps: f32,
    device: &B::Device,
) -> (Tensor<B, 1>, PpoLossDiag) {
    let b = old_log_probs.len();
    let old_lp = Tensor::<B, 1>::from_data(TensorData::new(old_log_probs.to_vec(), [b]), device);
    let adv = Tensor::<B, 1>::from_data(TensorData::new(advantages.to_vec(), [b]), device);

    let ratio = (new_log_probs - old_lp).exp();

    let ratio_data: Vec<f32> = ratio.clone().into_data().to_vec().expect("f32 conversion");
    let n = ratio_data.len() as f32;
    let mean_ratio = ratio_data.iter().sum::<f32>() / n;
    let frac_clipped = ratio_data
        .iter()
        .filter(|&&r| r < 1.0 - clip_eps || r > 1.0 + clip_eps)
        .count() as f32
        / n;

    let surr1 = ratio.clone() * adv.clone();
    let surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv;

    let stacked = Tensor::stack::<2>(vec![surr1, surr2], 1);
    let min_surr: Tensor<B, 1> = stacked.min_dim(1).squeeze_dim::<1>(1);

    let loss = -(min_surr.mean());
    let diag = PpoLossDiag {
        mean_ratio,
        frac_clipped,
    };
    (loss, diag)
}
