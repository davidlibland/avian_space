//! CPU-side action sampling from policy logits.

use crate::rl_obs::DiscreteAction;

use super::{POLICY_OUTPUT_DIM, TARGET_OUTPUT_DIM};

/// Convert policy logits and target logits to a `DiscreteAction` via greedy
/// argmax over each factored head.
///
/// Test-only: used by scenario tests and behaviour-cloning confusion-matrix
/// evaluation. Production rollout uses stochastic sampling.
#[cfg(test)]
pub fn logits_to_discrete_action(
    action_logits: &[f32],
    nav_target_logits: &[f32],
    wep_target_logits: &[f32],
) -> DiscreteAction {
    debug_assert_eq!(action_logits.len(), POLICY_OUTPUT_DIM);
    debug_assert_eq!(nav_target_logits.len(), TARGET_OUTPUT_DIM);
    debug_assert_eq!(wep_target_logits.len(), TARGET_OUTPUT_DIM);
    let turn_idx = argmax(&action_logits[0..3]) as u8;
    let thrust_idx = argmax(&action_logits[3..5]) as u8;
    let fire_primary = argmax(&action_logits[5..7]) as u8;
    let fire_secondary = argmax(&action_logits[7..9]) as u8;
    let nav_target_idx = argmax(nav_target_logits) as u8;
    let wep_target_idx = argmax(wep_target_logits) as u8;
    (
        turn_idx,
        thrust_idx,
        fire_primary,
        fire_secondary,
        nav_target_idx,
        wep_target_idx,
    )
}

/// Sample an action stochastically from the policy logits.
///
/// Returns `(action, log_prob)` where `log_prob` is the sum of per-head
/// log-probabilities for the sampled action.
pub fn sample_discrete_action(
    action_logits: &[f32],
    nav_target_logits: &[f32],
    wep_target_logits: &[f32],
    rng: &mut impl rand::Rng,
) -> (DiscreteAction, f32) {
    debug_assert_eq!(action_logits.len(), POLICY_OUTPUT_DIM);
    debug_assert_eq!(nav_target_logits.len(), TARGET_OUTPUT_DIM);
    debug_assert_eq!(wep_target_logits.len(), TARGET_OUTPUT_DIM);

    let mut total_log_prob: f32 = 0.0;

    let turn_idx = sample_categorical(&action_logits[0..3], rng, &mut total_log_prob);
    let thrust_idx = sample_categorical(&action_logits[3..5], rng, &mut total_log_prob);
    let fire_primary = sample_categorical(&action_logits[5..7], rng, &mut total_log_prob);
    let fire_secondary = sample_categorical(&action_logits[7..9], rng, &mut total_log_prob);
    let (nav_target_idx, wep_target_idx) = coupled_gumbel_sample(
        nav_target_logits,
        wep_target_logits,
        rng,
        &mut total_log_prob,
    );

    let action = (
        turn_idx as u8,
        thrust_idx as u8,
        fire_primary as u8,
        fire_secondary as u8,
        nav_target_idx as u8,
        wep_target_idx as u8,
    );
    (action, total_log_prob)
}

/// Sample from a categorical defined by logits (log-sum-exp stabilised).
fn sample_categorical(logits: &[f32], rng: &mut impl rand::Rng, log_prob_acc: &mut f32) -> usize {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|l| (l - max_logit).exp()).sum();
    let log_sum_exp = max_logit + exp_sum.ln();

    let u: f32 = rng.r#gen();
    let mut cumulative = 0.0_f32;
    let mut sampled = logits.len() - 1;
    for (i, &l) in logits.iter().enumerate() {
        cumulative += (l - log_sum_exp).exp();
        if u < cumulative {
            sampled = i;
            break;
        }
    }

    *log_prob_acc += (logits[sampled] - log_sum_exp).max(-20.0);
    sampled
}

/// Coupled categorical sampler using Gumbel-max with shared noise across
/// both logit vectors.
pub(super) fn coupled_gumbel_sample(
    logits_a: &[f32],
    logits_b: &[f32],
    rng: &mut impl rand::Rng,
    log_prob_acc: &mut f32,
) -> (usize, usize) {
    debug_assert_eq!(logits_a.len(), logits_b.len());
    let eps = 1e-20_f32;

    let mut best_a = (f32::NEG_INFINITY, 0usize);
    let mut best_b = (f32::NEG_INFINITY, 0usize);
    for i in 0..logits_a.len() {
        let u: f32 = rng.r#gen::<f32>().clamp(eps, 1.0 - eps);
        let g = -(-u.ln()).ln();
        let va = logits_a[i] + g;
        let vb = logits_b[i] + g;
        if va > best_a.0 {
            best_a = (va, i);
        }
        if vb > best_b.0 {
            best_b = (vb, i);
        }
    }

    let lse_a = log_sum_exp(logits_a);
    let lse_b = log_sum_exp(logits_b);
    *log_prob_acc += (logits_a[best_a.1] - lse_a).max(-20.0);
    *log_prob_acc += (logits_b[best_b.1] - lse_b).max(-20.0);
    (best_a.1, best_b.1)
}

fn log_sum_exp(logits: &[f32]) -> f32 {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|l| (l - max_logit).exp()).sum();
    max_logit + exp_sum.ln()
}

#[cfg(test)]
fn argmax(vals: &[f32]) -> usize {
    vals.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
