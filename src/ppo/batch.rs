//! Flattened training batch and segment → batch conversion.

use crate::consts::N_REWARD_TYPES;
use crate::gae::SegmentInfo;
use crate::model::{self, N_OBJECTS, OBJECT_INPUT_DIM, SELF_INPUT_DIM};
use crate::rl_collection::Segment;
use crate::rl_obs::DiscreteAction;

pub fn personality_index(p: &crate::ship::Personality) -> usize {
    match p {
        crate::ship::Personality::Miner => 0,
        crate::ship::Personality::Fighter => 1,
        crate::ship::Personality::Trader => 2,
    }
}

pub const N_PERSONALITIES: usize = 3;
pub const PERSONALITY_NAMES: [&str; N_PERSONALITIES] = ["miner", "fighter", "trader"];

/// Flattened training batch extracted from collected segments.
pub struct PpoBatch {
    pub self_flat: Vec<f32>,
    pub obj_flat: Vec<f32>,
    pub proj_flat: Vec<f32>,
    pub actions: Vec<DiscreteAction>,
    /// Rule-based (expert) actions paired with each observation, used as BC
    /// labels for the in-PPO BC auxiliary loss.
    pub rule_based_actions: Vec<DiscreteAction>,
    pub rewards: Vec<[f32; N_REWARD_TYPES]>,
    pub dones: Vec<bool>,
    /// Log π(a|s) recorded at rollout time (behaviour policy).
    pub old_log_probs: Vec<f32>,
    pub segment_infos: Vec<SegmentInfo>,
    pub total_steps: usize,
    pub personalities: Vec<usize>,
}

pub fn flatten_segments(segments: &[Segment]) -> PpoBatch {
    let total_steps: usize = segments.iter().map(|s| s.transitions.len()).sum();
    let mut self_flat = Vec::with_capacity(total_steps * SELF_INPUT_DIM);
    let mut obj_flat = Vec::with_capacity(total_steps * N_OBJECTS * OBJECT_INPUT_DIM);
    let mut proj_flat = Vec::with_capacity(total_steps * model::PROJECTILES_FLAT_DIM);
    let mut actions = Vec::with_capacity(total_steps);
    let mut rule_based_actions = Vec::with_capacity(total_steps);
    let mut rewards = Vec::with_capacity(total_steps);
    let mut dones = Vec::with_capacity(total_steps);
    let mut old_log_probs = Vec::with_capacity(total_steps);
    let mut segment_infos = Vec::with_capacity(segments.len());
    let mut personalities = Vec::with_capacity(segments.len());
    let mut idx = 0;

    for seg in segments {
        let start = idx;
        for t in &seg.transitions {
            let (s, o) = model::split_obs(&t.obs);
            self_flat.extend_from_slice(s);
            obj_flat.extend_from_slice(o);
            proj_flat.extend_from_slice(&t.proj_obs);
            actions.push(t.action);
            rule_based_actions.push(t.rule_based_action);
            rewards.push(t.rewards);
            dones.push(t.done);
            old_log_probs.push(t.log_prob);
            idx += 1;
        }
        segment_infos.push(SegmentInfo {
            start_idx: start,
            end_idx: idx,
            bootstrap_values: seg.bootstrap_value.unwrap_or([0.0; N_REWARD_TYPES]),
        });
        personalities.push(personality_index(&seg.personality));
    }

    PpoBatch {
        self_flat,
        obj_flat,
        proj_flat,
        actions,
        rule_based_actions,
        rewards,
        dones,
        old_log_probs,
        segment_infos,
        total_steps,
        personalities,
    }
}
