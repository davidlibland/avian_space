//! Generalized Advantage Estimation (GAE).
//!
//! Pure-Rust implementation with no burn dependency.  Given per-step rewards,
//! done flags, value estimates, and segment boundaries, [`compute_gae`]
//! returns advantages and value-function regression targets.

/// Metadata for one contiguous segment inside a flattened batch.
pub struct SegmentInfo {
    /// Index of the first step (inclusive).
    pub start_idx: usize,
    /// Index one-past the last step (exclusive).
    pub end_idx: usize,
    /// V(s_T) for the state after the segment's last transition.
    /// 0.0 when the segment ended in a terminal state (death).
    pub bootstrap_value: f32,
}

/// Compute GAE advantages and value-regression targets.
///
/// # Arguments
/// * `rewards`       – per-step rewards,   length = total_steps
/// * `dones`         – per-step done flags, length = total_steps
/// * `values`        – V(s_t) estimates,    length = total_steps
/// * `segment_infos` – describes each contiguous segment in the batch
/// * `gamma`         – discount factor
/// * `lambda`        – GAE bias-variance trade-off
///
/// # Returns
/// `(advantages, returns)` each of length `total_steps`.
pub fn compute_gae(
    rewards: &[f32],
    dones: &[bool],
    values: &[f32],
    segment_infos: &[SegmentInfo],
    gamma: f32,
    lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = rewards.len();
    let mut advantages = vec![0.0_f32; n];
    let mut returns = vec![0.0_f32; n];

    for seg in segment_infos {
        let mut next_adv: f32 = 0.0;
        let mut next_value: f32 = seg.bootstrap_value;

        for t in (seg.start_idx..seg.end_idx).rev() {
            let not_done = if dones[t] { 0.0 } else { 1.0 };
            let delta = rewards[t] + gamma * next_value * not_done - values[t];
            let adv = delta + gamma * lambda * not_done * next_adv;
            advantages[t] = adv;
            returns[t] = adv + values[t];

            next_value = values[t];
            next_adv = adv;
        }
    }

    (advantages, returns)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Single 3-step segment, no terminal, bootstrap = 0.
    #[test]
    fn test_gae_simple() {
        let rewards = vec![1.0, 1.0, 1.0];
        let dones = vec![false, false, false];
        let values = vec![0.5, 0.5, 0.5];
        let segments = vec![SegmentInfo {
            start_idx: 0,
            end_idx: 3,
            bootstrap_value: 0.0,
        }];
        let gamma = 0.99;
        let lambda = 0.95;

        let (advantages, returns) = compute_gae(&rewards, &dones, &values, &segments, gamma, lambda);

        // Step 2 (last): delta = 1 + 0.99*0 - 0.5 = 0.5
        //   adv = 0.5
        // Step 1: delta = 1 + 0.99*0.5 - 0.5 = 0.995
        //   adv = 0.995 + 0.99*0.95*0.5 = 1.46525
        // Step 0: delta = 1 + 0.99*0.5 - 0.5 = 0.995
        //   adv = 0.995 + 0.99*0.95*1.46525 = 2.373068
        assert!((advantages[2] - 0.5).abs() < 1e-5);
        assert!((advantages[1] - 1.46525).abs() < 1e-4);
        assert!((advantages[0] - 2.373068).abs() < 1e-4);

        // Returns = advantages + values
        for i in 0..3 {
            assert!((returns[i] - (advantages[i] + values[i])).abs() < 1e-6);
        }
    }

    /// Terminal at last step zeroes out bootstrapping.
    #[test]
    fn test_gae_terminal() {
        let rewards = vec![1.0, -1.0];
        let dones = vec![false, true];
        let values = vec![0.5, 0.5];
        let segments = vec![SegmentInfo {
            start_idx: 0,
            end_idx: 2,
            bootstrap_value: 10.0, // should be ignored because done=true at step 1
        }];
        let gamma = 0.99;
        let lambda = 0.95;

        let (advantages, _) = compute_gae(&rewards, &dones, &values, &segments, gamma, lambda);

        // Step 1 (done=true): delta = -1 + 0 - 0.5 = -1.5; adv = -1.5
        // But bootstrap is 10.0 and done=true: next_value starts as 10.0 but not_done=0
        // so delta = -1 + 0.99*10*0 - 0.5 = -1.5. Correct — done masks the bootstrap.
        assert!((advantages[1] - (-1.5)).abs() < 1e-5);

        // Step 0: delta = 1 + 0.99*0.5*1 - 0.5 = 0.995;
        //   adv = 0.995 + 0.99*0.95*1*(-1.5) = 0.995 - 1.4108 = -0.4108 (approx)
        // Wait, not_done at step 1 is 0, so next_adv carries through as -1.5 but
        // at step 0 not_done=1 (dones[0]=false):
        //   delta_0 = 1 + 0.99*values[1]*1 - values[0] = 1 + 0.495 - 0.5 = 0.995
        //   adv_0 = 0.995 + 0.99*0.95*1*(-1.5) = 0.995 - 1.41075 = -0.41575
        assert!((advantages[0] - (-0.41575)).abs() < 1e-4);
    }

    /// Two segments in one batch.
    #[test]
    fn test_gae_multi_segment() {
        let rewards = vec![1.0, 2.0, 3.0, 4.0];
        let dones = vec![false, false, false, false];
        let values = vec![0.0, 0.0, 0.0, 0.0];
        let segments = vec![
            SegmentInfo { start_idx: 0, end_idx: 2, bootstrap_value: 0.0 },
            SegmentInfo { start_idx: 2, end_idx: 4, bootstrap_value: 0.0 },
        ];
        let gamma = 1.0;
        let lambda = 1.0;

        let (advantages, _) = compute_gae(&rewards, &dones, &values, &segments, gamma, lambda);

        // With gamma=1, lambda=1, values=0: advantages = discounted sum of rewards
        // Segment 0: adv[1] = 2, adv[0] = 1 + 2 = 3
        // Segment 1: adv[3] = 4, adv[2] = 3 + 4 = 7
        assert!((advantages[0] - 3.0).abs() < 1e-6);
        assert!((advantages[1] - 2.0).abs() < 1e-6);
        assert!((advantages[2] - 7.0).abs() < 1e-6);
        assert!((advantages[3] - 4.0).abs() < 1e-6);
    }

    /// Bootstrap value should add to the last step's advantage.
    #[test]
    fn test_gae_bootstrap() {
        let rewards = vec![0.0];
        let dones = vec![false];
        let values = vec![1.0];
        let segments = vec![SegmentInfo {
            start_idx: 0,
            end_idx: 1,
            bootstrap_value: 2.0,
        }];
        let gamma = 0.99;
        let lambda = 0.95;

        let (advantages, _) = compute_gae(&rewards, &dones, &values, &segments, gamma, lambda);

        // delta = 0 + 0.99*2 - 1.0 = 0.98
        assert!((advantages[0] - 0.98).abs() < 1e-5);
    }
}
