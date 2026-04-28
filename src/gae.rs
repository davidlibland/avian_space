//! Generalized Advantage Estimation (GAE).
//!
//! Pure-Rust implementation with no burn dependency.  Given per-step rewards,
//! done flags, value estimates, and segment boundaries, [`compute_gae`]
//! returns advantages and value-function regression targets.

use crate::consts::N_REWARD_TYPES;

/// Metadata for one contiguous segment inside a flattened batch.
pub struct SegmentInfo {
    /// Index of the first step (inclusive).
    pub start_idx: usize,
    /// Index one-past the last step (exclusive).
    pub end_idx: usize,
    /// Per-head V(s_T) for the state after the segment's last transition.
    /// All zeros when the segment ended in a terminal state (death).
    pub bootstrap_values: [f32; N_REWARD_TYPES],
}

/// Result of multi-head GAE computation.
pub struct MultiHeadGae {
    /// Per-head advantages, shape: `[total_steps][N_REWARD_TYPES]`.
    pub head_advantages: Vec<[f32; N_REWARD_TYPES]>,
    /// Per-head returns (regression targets), shape: `[total_steps][N_REWARD_TYPES]`.
    pub head_returns: Vec<[f32; N_REWARD_TYPES]>,
    /// Total advantage (sum across heads), length = `total_steps`.
    pub total_advantages: Vec<f32>,
    /// Total returns (sum across heads), length = `total_steps`.
    pub total_returns: Vec<f32>,
}

/// Compute per-head GAE advantages and value-regression targets.
///
/// # Arguments
/// * `rewards`             – per-step per-head rewards, length = total_steps
/// * `dones`               – per-step done flags, length = total_steps
/// * `values`              – per-head V(s_t) estimates, length = total_steps
/// * `segment_infos`       – describes each contiguous segment in the batch
/// * `gamma`               – discount factor
/// * `lambda`              – GAE bias-variance trade-off
/// * `reward_type_weights` – per-head weights summed into `total_*` outputs
pub fn compute_gae_multihead(
    rewards: &[[f32; N_REWARD_TYPES]],
    dones: &[bool],
    values: &[[f32; N_REWARD_TYPES]],
    segment_infos: &[SegmentInfo],
    gamma: f32,
    lambda: f32,
    reward_type_weights: &[f32; N_REWARD_TYPES],
) -> MultiHeadGae {
    let n = rewards.len();
    let mut head_advantages = vec![[0.0_f32; N_REWARD_TYPES]; n];
    let mut head_returns = vec![[0.0_f32; N_REWARD_TYPES]; n];

    for seg in segment_infos {
        for h in 0..N_REWARD_TYPES {
            let mut next_adv: f32 = 0.0;
            let mut next_value: f32 = seg.bootstrap_values[h];

            for t in (seg.start_idx..seg.end_idx).rev() {
                let not_done = if dones[t] { 0.0 } else { 1.0 };
                let delta = rewards[t][h] + gamma * next_value * not_done - values[t][h];
                let adv = delta + gamma * lambda * not_done * next_adv;
                head_advantages[t][h] = adv;
                head_returns[t][h] = adv + values[t][h];

                next_value = values[t][h];
                next_adv = adv;
            }
        }
    }

    // Compute weighted totals across heads.
    let total_advantages: Vec<f32> = head_advantages
        .iter()
        .map(|a| a.iter().zip(reward_type_weights).map(|(v, &wt)| v * wt).sum())
        .collect();
    let total_returns: Vec<f32> = head_returns
        .iter()
        .map(|r| r.iter().zip(reward_type_weights).map(|(v, &wt)| v * wt).sum())
        .collect();

    MultiHeadGae {
        head_advantages,
        head_returns,
        total_advantages,
        total_returns,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Single 3-step segment, no terminal, bootstrap = 0, single head active.
    #[test]
    fn test_gae_simple() {
        let mut rewards = vec![[0.0; N_REWARD_TYPES]; 3];
        rewards[0][0] = 1.0;
        rewards[1][0] = 1.0;
        rewards[2][0] = 1.0;
        let dones = vec![false, false, false];
        let mut values = vec![[0.0; N_REWARD_TYPES]; 3];
        values[0][0] = 0.5;
        values[1][0] = 0.5;
        values[2][0] = 0.5;
        let segments = vec![SegmentInfo {
            start_idx: 0,
            end_idx: 3,
            bootstrap_values: [0.0; N_REWARD_TYPES],
        }];
        let gamma = 0.99;
        let lambda = 0.95;

        let weights = crate::consts::REWARD_TYPE_WEIGHTS;
        let result = compute_gae_multihead(&rewards, &dones, &values, &segments, gamma, lambda, &weights);

        assert!((result.head_advantages[2][0] - 0.5).abs() < 1e-5);
        assert!((result.head_advantages[1][0] - 1.46525).abs() < 1e-4);
        assert!((result.head_advantages[0][0] - 2.373068).abs() < 1e-4);

        // Total advantages equal head 0 since other heads are zero.
        for i in 0..3 {
            assert!((result.total_advantages[i] - result.head_advantages[i][0]).abs() < 1e-6);
        }
    }

    /// Terminal at last step zeroes out bootstrapping.
    #[test]
    fn test_gae_terminal() {
        let mut rewards = vec![[0.0; N_REWARD_TYPES]; 2];
        rewards[0][0] = 1.0;
        rewards[1][0] = -1.0;
        let dones = vec![false, true];
        let mut values = vec![[0.0; N_REWARD_TYPES]; 2];
        values[0][0] = 0.5;
        values[1][0] = 0.5;
        let mut bv = [0.0; N_REWARD_TYPES];
        bv[0] = 10.0;
        let segments = vec![SegmentInfo {
            start_idx: 0,
            end_idx: 2,
            bootstrap_values: bv,
        }];

        let weights = crate::consts::REWARD_TYPE_WEIGHTS;
        let result = compute_gae_multihead(&rewards, &dones, &values, &segments, 0.99, 0.95, &weights);
        assert!((result.head_advantages[1][0] - (-1.5)).abs() < 1e-5);
        assert!((result.head_advantages[0][0] - (-0.41575)).abs() < 1e-4);
    }

    /// Two segments in one batch.
    #[test]
    fn test_gae_multi_segment() {
        let mut rewards = vec![[0.0; N_REWARD_TYPES]; 4];
        rewards[0][0] = 1.0;
        rewards[1][0] = 2.0;
        rewards[2][0] = 3.0;
        rewards[3][0] = 4.0;
        let dones = vec![false, false, false, false];
        let values = vec![[0.0; N_REWARD_TYPES]; 4];
        let segments = vec![
            SegmentInfo { start_idx: 0, end_idx: 2, bootstrap_values: [0.0; N_REWARD_TYPES] },
            SegmentInfo { start_idx: 2, end_idx: 4, bootstrap_values: [0.0; N_REWARD_TYPES] },
        ];

        let weights = crate::consts::REWARD_TYPE_WEIGHTS;
        let result = compute_gae_multihead(&rewards, &dones, &values, &segments, 1.0, 1.0, &weights);
        assert!((result.total_advantages[0] - 3.0).abs() < 1e-6);
        assert!((result.total_advantages[1] - 2.0).abs() < 1e-6);
        assert!((result.total_advantages[2] - 7.0).abs() < 1e-6);
        assert!((result.total_advantages[3] - 4.0).abs() < 1e-6);
    }

    /// Bootstrap value should add to the last step's advantage.
    #[test]
    fn test_gae_bootstrap() {
        let rewards = vec![[0.0; N_REWARD_TYPES]];
        let dones = vec![false];
        let mut values = vec![[0.0; N_REWARD_TYPES]];
        values[0][0] = 1.0;
        let mut bv = [0.0; N_REWARD_TYPES];
        bv[0] = 2.0;
        let segments = vec![SegmentInfo {
            start_idx: 0,
            end_idx: 1,
            bootstrap_values: bv,
        }];

        let weights = crate::consts::REWARD_TYPE_WEIGHTS;
        let result = compute_gae_multihead(&rewards, &dones, &values, &segments, 0.99, 0.95, &weights);
        assert!((result.head_advantages[0][0] - 0.98).abs() < 1e-5);
    }

    /// Multiple heads contribute independently to total.
    #[test]
    fn test_gae_multihead_sum() {
        let mut rewards = vec![[0.0; N_REWARD_TYPES]; 1];
        rewards[0][0] = 1.0;
        rewards[0][1] = 2.0;
        let dones = vec![false];
        let values = vec![[0.0; N_REWARD_TYPES]];
        let segments = vec![SegmentInfo {
            start_idx: 0,
            end_idx: 1,
            bootstrap_values: [0.0; N_REWARD_TYPES],
        }];

        let weights = crate::consts::REWARD_TYPE_WEIGHTS;
        let result = compute_gae_multihead(&rewards, &dones, &values, &segments, 0.99, 0.95, &weights);
        assert!((result.head_advantages[0][0] - 1.0).abs() < 1e-6);
        assert!((result.head_advantages[0][1] - 2.0).abs() < 1e-6);
        assert!((result.total_advantages[0] - 3.0).abs() < 1e-6);
    }
}
