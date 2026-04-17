//! PPO training thread: the main update loop.

use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tensorboard_rs::summary_writer::SummaryWriter;

use burn::{
    optim::{GradientsParams, Optimizer},
    prelude::*,
    tensor::{Tensor, TensorData},
};
use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::consts::{N_REWARD_TYPES, REWARD_TYPE_NAMES};
use crate::gae;
use crate::model::{
    self, InferenceNet, N_OBJECTS, OBJECT_INPUT_DIM, RLInner, SELF_INPUT_DIM, TrainBackend,
    VALUE_OUTPUT_DIM,
};
use crate::rl_collection::Segment;
use crate::value_fn;

use super::batch::{N_PERSONALITIES, PERSONALITY_NAMES, flatten_segments, personality_index};
use super::buffer::ValueReplayBuffer;
use super::loss::{compute_log_probs_and_entropy, ppo_clipped_loss};
use super::persistence::{load_rl_buffer, load_step_counter, save_all_checkpoints};

// ---------------------------------------------------------------------------
// PPO hyperparameters
// ---------------------------------------------------------------------------

const PPO_GAMMA: f32 = 0.99;
const PPO_LAMBDA: f32 = 0.95;
const PPO_CLIP_EPS: f32 = 0.1;
const PPO_ENTROPY_COEFF: f32 = 0.01;
/// Coefficient on the behavioural-cloning auxiliary loss during PPO.
const PPO_BC_COEFF: f32 = 0.01;
const PPO_POLICY_LR: f64 = 3e-4;
const PPO_VALUE_LR: f64 = 1e-3;
const PPO_POLICY_EPOCHS: usize = 2;
const PPO_VALUE_EPOCHS: usize = 4;
const PPO_MINI_BATCH_SIZE: usize = 512;
const PPO_MIN_SEGMENTS: usize = 16;
const PPO_MAX_SEGMENTS: usize = 64;
const PPO_WEIGHT_SYNC_INTERVAL: usize = 1;
const PPO_SAVE_INTERVAL: usize = 30;
const PPO_HUBER_DELTA: f32 = 1.0;
const PPO_VALUE_BURNIN_EV_THRESHOLD: f32 = 0.3;

const VALUE_REPLAY_CAPACITY: usize = 8192;
const VALUE_REPLAY_FRACTION: f32 = 0.25;
const VALUE_REPLAY_EXTRA_BATCHES: usize = 4;

/// Spawn the PPO training thread.
pub fn spawn_ppo_training_thread(
    rl_rx: mpsc::Receiver<Segment>,
    inference_net: Arc<Mutex<InferenceNet>>,
    experiment: crate::experiments::ExperimentSetup,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let device: <TrainBackend as Backend>::Device = Default::default();
        let mut inner = RLInner::<TrainBackend>::new(&device);

        let policy_path = experiment.policy_checkpoint_path();
        let value_path = experiment.value_checkpoint_path();
        let policy_optim_path = experiment.policy_optim_checkpoint_path();
        let value_optim_path = experiment.value_optim_checkpoint_path();
        let buffer_path = experiment.rl_buffer_checkpoint_path();
        let step_counter_path = experiment.step_counter_path();

        if !experiment.is_fresh {
            if let Some(net) = model::load_training_net(&policy_path, &device) {
                inner.policy_net = Some(net);
                let bytes = model::training_net_to_bytes(inner.policy_net.as_ref().unwrap());
                if let Ok(mut lock) = inference_net.lock() {
                    lock.load_bytes(bytes);
                }
            }
            if let Some(net) =
                model::load_training_net_with_dim(&value_path, &device, VALUE_OUTPUT_DIM)
            {
                inner.value_net = Some(net);
            }
            if let Some(optim) = model::load_optimizer(&policy_optim_path, &device) {
                inner.policy_optim = optim;
            }
            if let Some(optim) = model::load_optimizer(&value_optim_path, &device) {
                inner.value_optim = optim;
            }
        }

        let mut update_cycle: usize = if !experiment.is_fresh {
            load_step_counter(&step_counter_path).unwrap_or(0)
        } else {
            0
        };
        let mut segments: Vec<Segment> = if !experiment.is_fresh {
            load_rl_buffer(&buffer_path).unwrap_or_default()
        } else {
            Vec::new()
        };
        let mut rng = thread_rng();
        let mut value_replay = ValueReplayBuffer::new(VALUE_REPLAY_CAPACITY);

        let bc_loss_fn =
            burn::nn::loss::CrossEntropyLossConfig::new().init::<TrainBackend>(&device);

        let tb_dir = format!("{}/tb", experiment.run_dir);
        std::fs::create_dir_all(&tb_dir).ok();
        let mut writer = SummaryWriter::new(&tb_dir);
        writer.add_scalar("hparams/gamma", PPO_GAMMA, 0);
        writer.add_scalar("hparams/lambda", PPO_LAMBDA, 0);
        writer.add_scalar("hparams/clip_eps", PPO_CLIP_EPS, 0);
        writer.add_scalar("hparams/entropy_coeff", PPO_ENTROPY_COEFF, 0);
        writer.add_scalar("hparams/policy_lr", PPO_POLICY_LR as f32, 0);
        writer.add_scalar("hparams/value_lr", PPO_VALUE_LR as f32, 0);
        writer.add_scalar("hparams/policy_epochs", PPO_POLICY_EPOCHS as f32, 0);
        writer.add_scalar("hparams/value_epochs", PPO_VALUE_EPOCHS as f32, 0);
        writer.add_scalar("hparams/mini_batch_size", PPO_MINI_BATCH_SIZE as f32, 0);

        println!("[ppo] Training thread started, waiting for segments...");
        let mut t_wait_start = Instant::now();

        loop {
            // ── Phase 1: Collect segments ─────────────────────────────────
            if segments.len() < PPO_MIN_SEGMENTS {
                match rl_rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(seg) => segments.push(seg),
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        println!("[ppo] Channel disconnected, saving and exiting.");
                        save_all_checkpoints(
                            &inner,
                            &segments,
                            update_cycle,
                            &policy_path,
                            &value_path,
                            &policy_optim_path,
                            &value_optim_path,
                            &buffer_path,
                            &step_counter_path,
                        );
                        break;
                    }
                }
            }
            while let Ok(seg) = rl_rx.try_recv() {
                segments.push(seg);
            }
            if segments.len() < PPO_MIN_SEGMENTS {
                continue;
            }
            if segments.len() > PPO_MAX_SEGMENTS {
                let excess = segments.len() - PPO_MAX_SEGMENTS;
                segments.drain(..excess);
            }

            let wait_secs = t_wait_start.elapsed().as_secs_f32();
            let t0 = Instant::now();
            let n_segments = segments.len();

            // ── Rollout metrics: per personality × reward_type ────────────
            let mut rew_total = [[0.0_f32; N_REWARD_TYPES]; N_PERSONALITIES];
            let mut rew_steps = [0_usize; N_PERSONALITIES];
            for seg in &segments {
                let pi = personality_index(&seg.personality);
                for t in &seg.transitions {
                    for h in 0..N_REWARD_TYPES {
                        rew_total[pi][h] += t.rewards[h];
                    }
                    rew_steps[pi] += 1;
                }
                writer.add_scalar(
                    "reward/trajectory_length",
                    seg.transitions.len() as f32,
                    update_cycle,
                );
            }
            let total_traj_steps: usize = rew_steps.iter().sum();
            for pi in 0..N_PERSONALITIES {
                let pn = PERSONALITY_NAMES[pi];
                let ns = rew_steps[pi].max(1) as f32;
                for h in 0..N_REWARD_TYPES {
                    let rn = REWARD_TYPE_NAMES[h];
                    writer.add_scalar(
                        &format!("reward_total/{pn}/{rn}"),
                        rew_total[pi][h],
                        update_cycle,
                    );
                    writer.add_scalar(
                        &format!("reward_per_step/{pn}/{rn}"),
                        rew_total[pi][h] / ns,
                        update_cycle,
                    );
                }
            }
            let total_reward: f32 = rew_total.iter().flat_map(|r| r.iter()).sum();
            let mean_reward_per_step = if total_traj_steps > 0 {
                total_reward / total_traj_steps as f32
            } else {
                0.0
            };
            writer.add_scalar("reward/mean_per_step", mean_reward_per_step, update_cycle);
            writer.add_scalar(
                "reward/mean_segment_return",
                if n_segments > 0 {
                    total_reward / n_segments as f32
                } else {
                    0.0
                },
                update_cycle,
            );

            // ── Phase 2: Recompute bootstrap values ──────────────────────
            value_fn::recompute_bootstrap_values(
                &mut segments,
                inner.value_net.as_ref().unwrap(),
                &device,
            );

            // ── Phase 3: Flatten segments into a batch ───────────────────
            let batch = flatten_segments(&segments);
            let total_steps = batch.total_steps;

            // ── Phase 3b: Target selection + entity slot stats ─────────────
            const N_NAV_TYPES: usize = 5;
            const NAV_TYPE_NAMES: [&str; N_NAV_TYPES] =
                ["ship", "asteroid", "planet", "pickup", "none"];
            const N_WEP_TYPES: usize = 5;
            const WEP_TYPE_NAMES: [&str; N_WEP_TYPES] = [
                "ship_engage",
                "ship_hostile",
                "ship_neutral",
                "asteroid",
                "none",
            ];
            const N_SLOT_TYPES: usize = 4;
            const SLOT_TYPE_NAMES_LOG: [&str; N_SLOT_TYPES] =
                ["ship", "asteroid", "planet", "pickup"];

            let mut nav_target_counts = [[0u32; N_NAV_TYPES]; N_PERSONALITIES];
            let mut wep_target_counts = [[0u32; N_WEP_TYPES]; N_PERSONALITIES];
            let mut slot_type_sums = [[0.0_f32; N_SLOT_TYPES]; N_PERSONALITIES];
            let mut pers_step_counts = [0u32; N_PERSONALITIES];
            {
                use crate::rl_obs::{
                    SHIP_IS_HOSTILE, SHIP_SHOULD_ENGAGE, SLOT_IS_PRESENT, SLOT_TYPE_ONEHOT,
                    SLOT_TYPE_SPECIFIC,
                };

                let mut step_personality = vec![0usize; total_steps];
                for (seg_idx, seg) in batch.segment_infos.iter().enumerate() {
                    let pi = batch.personalities[seg_idx];
                    for t in seg.start_idx..seg.end_idx {
                        step_personality[t] = pi;
                    }
                }

                let read_entity_type = |step: usize, slot: usize| -> usize {
                    let base = step * model::N_OBJECTS * model::OBJECT_INPUT_DIM
                        + slot * model::OBJECT_INPUT_DIM;
                    for t in 0..4 {
                        if base + SLOT_TYPE_ONEHOT + t < batch.obj_flat.len()
                            && batch.obj_flat[base + SLOT_TYPE_ONEHOT + t] > 0.5
                        {
                            return t;
                        }
                    }
                    4
                };

                let is_slot_present = |step: usize, slot: usize| -> bool {
                    let idx = step * model::N_OBJECTS * model::OBJECT_INPUT_DIM
                        + slot * model::OBJECT_INPUT_DIM
                        + SLOT_IS_PRESENT;
                    idx < batch.obj_flat.len() && batch.obj_flat[idx] > 0.5
                };

                let read_ship_hostility = |step: usize, slot: usize| -> (bool, bool) {
                    let base = step * model::N_OBJECTS * model::OBJECT_INPUT_DIM
                        + slot * model::OBJECT_INPUT_DIM;
                    let is_hostile_idx = base + SLOT_TYPE_SPECIFIC + SHIP_IS_HOSTILE;
                    let should_engage_idx = base + SLOT_TYPE_SPECIFIC + SHIP_SHOULD_ENGAGE;
                    let is_hostile = is_hostile_idx < batch.obj_flat.len()
                        && batch.obj_flat[is_hostile_idx] > 0.5;
                    let should_engage = should_engage_idx < batch.obj_flat.len()
                        && batch.obj_flat[should_engage_idx] > 0.5;
                    (is_hostile, should_engage)
                };

                for (step, action) in batch.actions.iter().enumerate() {
                    let pi = step_personality[step];
                    pers_step_counts[pi] += 1;

                    let nav_idx = action.4 as usize;
                    if nav_idx >= model::N_OBJECTS {
                        nav_target_counts[pi][4] += 1;
                    } else {
                        nav_target_counts[pi][read_entity_type(step, nav_idx).min(3)] += 1;
                    }

                    let wep_idx = action.5 as usize;
                    if wep_idx >= model::N_OBJECTS {
                        wep_target_counts[pi][4] += 1;
                    } else {
                        let etype = read_entity_type(step, wep_idx);
                        match etype {
                            0 => {
                                let (is_hostile, should_engage) =
                                    read_ship_hostility(step, wep_idx);
                                if should_engage {
                                    wep_target_counts[pi][0] += 1;
                                } else if is_hostile {
                                    wep_target_counts[pi][1] += 1;
                                } else {
                                    wep_target_counts[pi][2] += 1;
                                }
                            }
                            1 => wep_target_counts[pi][3] += 1,
                            _ => wep_target_counts[pi][4] += 1,
                        }
                    }

                    for slot in 0..model::N_OBJECTS {
                        if is_slot_present(step, slot) {
                            let etype = read_entity_type(step, slot);
                            if etype < N_SLOT_TYPES {
                                slot_type_sums[pi][etype] += 1.0;
                            }
                        }
                    }
                }
            }

            // Trader-specific diagnostics.
            let mut trader_nav_cargo_value_sum = 0.0_f32;
            let mut trader_nav_cargo_value_count = 0u32;
            let mut trader_planet_count_sum = 0.0_f32;
            let mut trader_steps = 0u32;
            {
                use crate::rl_obs::{SLOT_IS_PRESENT, SLOT_TYPE_ONEHOT, SLOT_VALUE};
                let trader_pi = 2usize;
                for (seg_idx, seg) in batch.segment_infos.iter().enumerate() {
                    if batch.personalities[seg_idx] != trader_pi {
                        continue;
                    }
                    for step in seg.start_idx..seg.end_idx {
                        trader_steps += 1;
                        let mut n_planets = 0u32;
                        for slot in 0..model::N_OBJECTS {
                            let base = step * model::N_OBJECTS * model::OBJECT_INPUT_DIM
                                + slot * model::OBJECT_INPUT_DIM;
                            let present = base + SLOT_IS_PRESENT < batch.obj_flat.len()
                                && batch.obj_flat[base + SLOT_IS_PRESENT] > 0.5;
                            let is_planet = base + SLOT_TYPE_ONEHOT + 2 < batch.obj_flat.len()
                                && batch.obj_flat[base + SLOT_TYPE_ONEHOT + 2] > 0.5;
                            if present && is_planet {
                                n_planets += 1;
                            }
                        }
                        trader_planet_count_sum += n_planets as f32;

                        let nav_idx = batch.actions[step].4 as usize;
                        if nav_idx < model::N_OBJECTS {
                            let base = step * model::N_OBJECTS * model::OBJECT_INPUT_DIM
                                + nav_idx * model::OBJECT_INPUT_DIM;
                            let is_planet = base + SLOT_TYPE_ONEHOT + 2 < batch.obj_flat.len()
                                && batch.obj_flat[base + SLOT_TYPE_ONEHOT + 2] > 0.5;
                            if is_planet {
                                let val_idx = base + SLOT_VALUE;
                                if val_idx < batch.obj_flat.len() {
                                    trader_nav_cargo_value_sum += batch.obj_flat[val_idx];
                                    trader_nav_cargo_value_count += 1;
                                }
                            }
                        }
                    }
                }
            }

            // ── Phase 4: PPO epochs ───────────────────────────────────────
            let mut epoch_policy_loss = 0.0_f32;
            let mut epoch_value_loss = 0.0_f32;
            let mut epoch_entropy = 0.0_f32;
            let mut epoch_head_max_prob = [0.0_f32; 6];
            let mut epoch_head_agreement = [0.0_f32; 6];
            let mut policy_epochs_run = 0usize;
            let mut policy_mbs_total = 0usize;
            let mut epoch_frac_clipped = 0.0_f32;
            let mut epoch_mean_ratio = 0.0_f32;
            let mut epoch_bc_loss = 0.0_f32;
            let mut epoch_bc_batches = 0_usize;
            let mut total_mini_batches = 0_usize;
            let mut first_epoch_adv_mean = 0.0_f32;
            let mut first_epoch_adv_std = 0.0_f32;
            let mut first_epoch_pers_adv_mean = [0.0_f32; N_PERSONALITIES];
            let mut first_epoch_pers_adv_std = [0.0_f32; N_PERSONALITIES];
            let mut first_epoch_explained_var = 0.0_f32;
            let mut first_epoch_head_ev = [0.0_f32; N_REWARD_TYPES];
            let mut first_epoch_head_td = [0.0_f32; N_REWARD_TYPES];

            let all_self = Tensor::<TrainBackend, 2>::from_data(
                TensorData::new(batch.self_flat.clone(), [total_steps, SELF_INPUT_DIM]),
                &device,
            );
            let all_obj = Tensor::<TrainBackend, 3>::from_data(
                TensorData::new(
                    batch.obj_flat.clone(),
                    [total_steps, N_OBJECTS, OBJECT_INPUT_DIM],
                ),
                &device,
            );
            let all_proj = Tensor::<TrainBackend, 3>::from_data(
                TensorData::new(
                    batch.proj_flat.clone(),
                    [
                        total_steps,
                        model::N_PROJECTILE_SLOTS,
                        model::PROJ_INPUT_DIM,
                    ],
                ),
                &device,
            );

            for epoch in 0..PPO_VALUE_EPOCHS {
                let values = value_fn::batch_value_inference(
                    inner.value_net.as_ref().unwrap(),
                    &batch.self_flat,
                    &batch.obj_flat,
                    &batch.proj_flat,
                    total_steps,
                    &device,
                );

                let gae_result = gae::compute_gae_multihead(
                    &batch.rewards,
                    &batch.dones,
                    &values,
                    &batch.segment_infos,
                    PPO_GAMMA,
                    PPO_LAMBDA,
                );

                let n = total_steps as f32;

                let mut step_personality = vec![0usize; total_steps];
                for (si, info) in batch.segment_infos.iter().enumerate() {
                    for k in info.start_idx..info.end_idx {
                        step_personality[k] = batch.personalities[si];
                    }
                }
                let mut per_pers_mean = [0.0_f32; N_PERSONALITIES];
                let mut per_pers_std = [1.0_f32; N_PERSONALITIES];
                let mut per_pers_count = [0usize; N_PERSONALITIES];
                for p in 0..N_PERSONALITIES {
                    let vals: Vec<f32> = gae_result
                        .total_advantages
                        .iter()
                        .zip(&step_personality)
                        .filter_map(|(a, &pp)| (pp == p).then_some(*a))
                        .collect();
                    per_pers_count[p] = vals.len();
                    if vals.is_empty() {
                        continue;
                    }
                    let m = vals.iter().sum::<f32>() / vals.len() as f32;
                    let v = vals.iter().map(|a| (a - m).powi(2)).sum::<f32>() / vals.len() as f32;
                    per_pers_mean[p] = m;
                    per_pers_std[p] = v.sqrt();
                }
                let norm_advantages: Vec<f32> = gae_result
                    .total_advantages
                    .iter()
                    .zip(&step_personality)
                    .map(|(a, &p)| (a - per_pers_mean[p]) / (per_pers_std[p] + 1e-8))
                    .collect();
                let total_count: f32 = per_pers_count.iter().sum::<usize>().max(1) as f32;
                let adv_mean: f32 = (0..N_PERSONALITIES)
                    .map(|p| per_pers_mean[p] * per_pers_count[p] as f32)
                    .sum::<f32>()
                    / total_count;
                let adv_std: f32 = (0..N_PERSONALITIES)
                    .map(|p| per_pers_std[p] * per_pers_count[p] as f32)
                    .sum::<f32>()
                    / total_count;

                if epoch == 0 {
                    first_epoch_adv_mean = adv_mean;
                    first_epoch_adv_std = adv_std;
                    first_epoch_pers_adv_mean = per_pers_mean;
                    first_epoch_pers_adv_std = per_pers_std;
                    let ret_mean: f32 = gae_result.total_returns.iter().sum::<f32>() / n;
                    let var_ret: f32 = gae_result
                        .total_returns
                        .iter()
                        .map(|r| (r - ret_mean).powi(2))
                        .sum::<f32>()
                        / n;
                    let var_adv: f32 = gae_result
                        .total_advantages
                        .iter()
                        .map(|a| a.powi(2))
                        .sum::<f32>()
                        / n;
                    first_epoch_explained_var = 1.0 - var_adv / (var_ret + 1e-8);
                    for h in 0..N_REWARD_TYPES {
                        let h_ret_mean: f32 =
                            gae_result.head_returns.iter().map(|r| r[h]).sum::<f32>() / n;
                        let h_var_ret: f32 = gae_result
                            .head_returns
                            .iter()
                            .map(|r| (r[h] - h_ret_mean).powi(2))
                            .sum::<f32>()
                            / n;
                        let h_var_adv: f32 = gae_result
                            .head_advantages
                            .iter()
                            .map(|a| a[h].powi(2))
                            .sum::<f32>()
                            / n;
                        first_epoch_head_ev[h] = 1.0 - h_var_adv / (h_var_ret + 1e-8);
                        first_epoch_head_td[h] = gae_result
                            .head_advantages
                            .iter()
                            .map(|a| a[h].abs())
                            .sum::<f32>()
                            / n;
                    }
                }

                if epoch == 0 {
                    value_replay.insert_from_batch(
                        &batch,
                        &gae_result.head_advantages,
                        &gae_result.head_returns,
                    );
                }

                let skip_policy = first_epoch_explained_var < PPO_VALUE_BURNIN_EV_THRESHOLD;

                let all_returns_flat: Vec<f32> = gae_result
                    .head_returns
                    .iter()
                    .flat_map(|r| r.iter().copied())
                    .collect();
                let all_returns = Tensor::<TrainBackend, 2>::from_data(
                    TensorData::new(all_returns_flat, [total_steps, N_REWARD_TYPES]),
                    &device,
                );

                let mut indices: Vec<usize> = (0..total_steps).collect();
                indices.shuffle(&mut rng);

                let mut head_max_prob_sum: Vec<Option<Tensor<TrainBackend, 1>>> =
                    (0..6).map(|_| None).collect();
                let mut policy_mbs_this_epoch = 0usize;

                for mb_indices in indices.chunks(PPO_MINI_BATCH_SIZE) {
                    let mb = mb_indices.len();

                    let mut mb_actions = Vec::with_capacity(mb);
                    let mut mb_old_lp = Vec::with_capacity(mb);
                    let mut mb_adv = Vec::with_capacity(mb);
                    let mut mb_rule_actions = Vec::with_capacity(mb);
                    for &i in mb_indices {
                        mb_actions.push(batch.actions[i]);
                        mb_rule_actions.push(batch.rule_based_actions[i]);
                        mb_old_lp.push(batch.old_log_probs[i]);
                        mb_adv.push(norm_advantages[i]);
                    }

                    let idx_data: Vec<i64> = mb_indices.iter().map(|&i| i as i64).collect();
                    let idx_t = Tensor::<TrainBackend, 1, Int>::from_data(
                        TensorData::new(idx_data, [mb]),
                        &device,
                    );
                    let self_t = all_self.clone().select(0, idx_t.clone());
                    let obj_t = all_obj.clone().select(0, idx_t.clone());
                    let proj_t = all_proj.clone().select(0, idx_t.clone());
                    let targets = all_returns.clone().select(0, idx_t);

                    if !skip_policy && epoch < PPO_POLICY_EPOCHS {
                        let (policy_grads, (policy_loss_s, entropy_s, bc_loss_s, diag)) = {
                            let net = inner.policy_net.as_ref().unwrap();
                            let (action_logits, nav_target_logits, wep_target_logits) =
                                net.forward(self_t.clone(), obj_t.clone(), proj_t.clone());

                            {
                                use burn::tensor::activation::softmax;
                                let slices: [(Tensor<TrainBackend, 2>, usize); 6] = [
                                    (action_logits.clone().narrow(1, 0, 3), 0),
                                    (action_logits.clone().narrow(1, 3, 2), 1),
                                    (action_logits.clone().narrow(1, 5, 2), 2),
                                    (action_logits.clone().narrow(1, 7, 2), 3),
                                    (nav_target_logits.clone(), 4),
                                    (wep_target_logits.clone(), 5),
                                ];
                                for (logit, h) in slices.iter() {
                                    let mp: Tensor<TrainBackend, 1> =
                                        softmax(logit.clone(), 1).max_dim(1).mean();
                                    head_max_prob_sum[*h] =
                                        Some(match head_max_prob_sum[*h].take() {
                                            Some(acc) => acc + mp,
                                            None => mp,
                                        });
                                }
                            }
                            let mut agree = [0u32; 6];
                            for (a, r) in mb_actions.iter().zip(mb_rule_actions.iter()) {
                                agree[0] += (a.0 == r.0) as u32;
                                agree[1] += (a.1 == r.1) as u32;
                                agree[2] += (a.2 == r.2) as u32;
                                agree[3] += (a.3 == r.3) as u32;
                                agree[4] += (a.4 == r.4) as u32;
                                agree[5] += (a.5 == r.5) as u32;
                            }
                            let n = mb_actions.len().max(1) as f32;
                            for h in 0..6 {
                                epoch_head_agreement[h] += agree[h] as f32 / n;
                            }
                            policy_mbs_this_epoch += 1;

                            let bc_loss = crate::rl_collection::compute_bc_loss_from_logits(
                                action_logits.clone(),
                                nav_target_logits.clone(),
                                wep_target_logits.clone(),
                                &mb_rule_actions,
                                &bc_loss_fn,
                                &device,
                            );

                            let (new_lp, entropy) = compute_log_probs_and_entropy(
                                action_logits,
                                nav_target_logits,
                                wep_target_logits,
                                &mb_actions,
                                &device,
                            );

                            let (policy_loss, diag) = ppo_clipped_loss(
                                new_lp,
                                &mb_old_lp,
                                &mb_adv,
                                PPO_CLIP_EPS,
                                &device,
                            );
                            let entropy_bonus = -(entropy.mean());
                            let total_policy_loss = policy_loss.clone()
                                + entropy_bonus.clone() * PPO_ENTROPY_COEFF
                                + bc_loss.clone() * PPO_BC_COEFF;

                            let raw = total_policy_loss.backward();
                            let grads = GradientsParams::from_grads(raw, net);
                            let policy_loss_s = f32::from(policy_loss.into_scalar());
                            let entropy_s = f32::from((-entropy_bonus).into_scalar());
                            let bc_loss_s = f32::from(bc_loss.into_scalar());
                            (grads, (policy_loss_s, entropy_s, bc_loss_s, diag))
                        };
                        epoch_policy_loss += policy_loss_s;
                        epoch_entropy += entropy_s;
                        epoch_frac_clipped += diag.frac_clipped;
                        epoch_mean_ratio += diag.mean_ratio;
                        epoch_bc_loss += bc_loss_s;
                        epoch_bc_batches += 1;
                        let net = inner.policy_net.take().unwrap();
                        inner.policy_net =
                            Some(inner.policy_optim.step(PPO_POLICY_LR, net, policy_grads));
                    }

                    let value_grads = {
                        let vnet = inner.value_net.as_ref().unwrap();
                        let (value_out, _, _) = vnet.forward(self_t, obj_t, proj_t);
                        let vloss = value_fn::huber_value_loss(value_out, targets, PPO_HUBER_DELTA);

                        epoch_value_loss += f32::from(vloss.clone().into_scalar());

                        let raw = vloss.backward();
                        GradientsParams::from_grads(raw, vnet)
                    };
                    let vnet = inner.value_net.take().unwrap();
                    inner.value_net = Some(inner.value_optim.step(PPO_VALUE_LR, vnet, value_grads));

                    total_mini_batches += 1;
                }

                if policy_mbs_this_epoch > 0 {
                    for h in 0..6 {
                        if let Some(acc) = head_max_prob_sum[h].take() {
                            let mp = f32::from(acc.into_scalar()) / policy_mbs_this_epoch as f32;
                            epoch_head_max_prob[h] += mp;
                        }
                    }
                    policy_epochs_run += 1;
                    policy_mbs_total += policy_mbs_this_epoch;
                }
            }

            // ── Phase 5b: Extra value training from replay buffer ────────
            if value_replay.len() > 0 {
                let replay_mb_size = (PPO_MINI_BATCH_SIZE as f32 * VALUE_REPLAY_FRACTION) as usize;
                for _ in 0..VALUE_REPLAY_EXTRA_BATCHES {
                    if let Some((r_self, r_obj, r_proj, r_ret)) =
                        value_replay.sample(replay_mb_size.max(1), &mut rng)
                    {
                        let mb = replay_mb_size.max(1);
                        let self_t = Tensor::<TrainBackend, 2>::from_data(
                            TensorData::new(r_self, [mb, SELF_INPUT_DIM]),
                            &device,
                        );
                        let obj_t = Tensor::<TrainBackend, 3>::from_data(
                            TensorData::new(r_obj, [mb, N_OBJECTS, OBJECT_INPUT_DIM]),
                            &device,
                        );
                        let proj_t = Tensor::<TrainBackend, 3>::from_data(
                            TensorData::new(
                                r_proj,
                                [mb, model::N_PROJECTILE_SLOTS, model::PROJ_INPUT_DIM],
                            ),
                            &device,
                        );
                        let value_grads = {
                            let vnet = inner.value_net.as_ref().unwrap();
                            let (value_out, _, _) = vnet.forward(self_t, obj_t, proj_t);
                            let targets = Tensor::<TrainBackend, 2>::from_data(
                                TensorData::new(r_ret, [mb, N_REWARD_TYPES]),
                                &device,
                            );
                            let vloss =
                                value_fn::huber_value_loss(value_out, targets, PPO_HUBER_DELTA);
                            let raw = vloss.backward();
                            GradientsParams::from_grads(raw, vnet)
                        };
                        let vnet = inner.value_net.take().unwrap();
                        inner.value_net =
                            Some(inner.value_optim.step(PPO_VALUE_LR, vnet, value_grads));
                    }
                }
            }

            update_cycle += 1;

            if update_cycle % PPO_WEIGHT_SYNC_INTERVAL == 0 {
                let bytes = model::training_net_to_bytes(inner.policy_net.as_ref().unwrap());
                if let Ok(mut lock) = inference_net.lock() {
                    lock.load_bytes(bytes);
                }
            }

            if update_cycle % PPO_SAVE_INTERVAL == 0 {
                save_all_checkpoints(
                    &inner,
                    &segments,
                    update_cycle,
                    &policy_path,
                    &value_path,
                    &policy_optim_path,
                    &value_optim_path,
                    &buffer_path,
                    &step_counter_path,
                );
            }

            segments.clear();

            // ── Phase 9: Diagnostics + TensorBoard ───────────────────────
            let elapsed = t0.elapsed();
            let n_mb = total_mini_batches.max(1) as f32;
            let avg_ploss = epoch_policy_loss / n_mb;
            let avg_vloss = epoch_value_loss / n_mb;
            let avg_ent = epoch_entropy / n_mb;
            let avg_clip = epoch_frac_clipped / n_mb;
            let avg_ratio = epoch_mean_ratio / n_mb;
            let steps_per_sec = total_steps as f32 / elapsed.as_secs_f32();

            writer.add_scalar("train/policy_loss", avg_ploss, update_cycle);
            writer.add_scalar("train/value_loss", avg_vloss, update_cycle);
            writer.add_scalar("train/entropy", avg_ent, update_cycle);
            writer.add_scalar("train/frac_clipped", avg_clip, update_cycle);
            writer.add_scalar("train/mean_ratio", avg_ratio, update_cycle);
            if epoch_bc_batches > 0 {
                let avg_bc = epoch_bc_loss / epoch_bc_batches as f32;
                writer.add_scalar("train/bc_loss", avg_bc, update_cycle);
            }
            if policy_epochs_run > 0 && policy_mbs_total > 0 {
                let head_names = [
                    "turn",
                    "thrust",
                    "fire_primary",
                    "fire_secondary",
                    "nav",
                    "wep",
                ];
                for h in 0..6 {
                    writer.add_scalar(
                        &format!("policy/max_prob/{}", head_names[h]),
                        epoch_head_max_prob[h] / policy_epochs_run as f32,
                        update_cycle,
                    );
                    writer.add_scalar(
                        &format!("policy/bc_agreement/{}", head_names[h]),
                        epoch_head_agreement[h] / policy_mbs_total as f32,
                        update_cycle,
                    );
                }
            }
            writer.add_scalar("train/advantage_mean", first_epoch_adv_mean, update_cycle);
            writer.add_scalar("train/advantage_std", first_epoch_adv_std, update_cycle);
            for pi in 0..N_PERSONALITIES {
                let pn = PERSONALITY_NAMES[pi];
                writer.add_scalar(
                    &format!("advantage_per_personality/{pn}/mean_raw"),
                    first_epoch_pers_adv_mean[pi],
                    update_cycle,
                );
                writer.add_scalar(
                    &format!("advantage_per_personality/{pn}/std_raw"),
                    first_epoch_pers_adv_std[pi],
                    update_cycle,
                );
            }
            writer.add_scalar(
                "train/explained_variance",
                first_epoch_explained_var,
                update_cycle,
            );
            for h in 0..N_REWARD_TYPES {
                let rn = REWARD_TYPE_NAMES[h];
                writer.add_scalar(
                    &format!("value_head/{rn}/explained_variance"),
                    first_epoch_head_ev[h],
                    update_cycle,
                );
                writer.add_scalar(
                    &format!("value_head/{rn}/mean_abs_td_error"),
                    first_epoch_head_td[h],
                    update_cycle,
                );
            }
            for pi in 0..N_PERSONALITIES {
                let pn = PERSONALITY_NAMES[pi];
                let nav_total = nav_target_counts[pi].iter().sum::<u32>().max(1) as f32;
                for tt in 0..N_NAV_TYPES {
                    writer.add_scalar(
                        &format!("nav_target_type/{pn}/{}", NAV_TYPE_NAMES[tt]),
                        nav_target_counts[pi][tt] as f32 / nav_total,
                        update_cycle,
                    );
                }
                let wep_total = wep_target_counts[pi].iter().sum::<u32>().max(1) as f32;
                for tt in 0..N_WEP_TYPES {
                    writer.add_scalar(
                        &format!("wep_target_type/{pn}/{}", WEP_TYPE_NAMES[tt]),
                        wep_target_counts[pi][tt] as f32 / wep_total,
                        update_cycle,
                    );
                }
                let n_steps = pers_step_counts[pi].max(1) as f32;
                for tt in 0..N_SLOT_TYPES {
                    writer.add_scalar(
                        &format!("slot_counts/{pn}/{}", SLOT_TYPE_NAMES_LOG[tt]),
                        slot_type_sums[pi][tt] / n_steps,
                        update_cycle,
                    );
                }
            }

            if trader_steps > 0 {
                writer.add_scalar(
                    "trader_diag/mean_visible_planets",
                    trader_planet_count_sum / trader_steps as f32,
                    update_cycle,
                );
                writer.add_scalar(
                    "trader_diag/nav_planet_frac",
                    trader_nav_cargo_value_count as f32 / trader_steps as f32,
                    update_cycle,
                );
                if trader_nav_cargo_value_count > 0 {
                    writer.add_scalar(
                        "trader_diag/mean_nav_cargo_value",
                        trader_nav_cargo_value_sum / trader_nav_cargo_value_count as f32,
                        update_cycle,
                    );
                }
            }

            let policy_skipped = first_epoch_explained_var < PPO_VALUE_BURNIN_EV_THRESHOLD;
            writer.add_scalar(
                "train/value_burnin",
                policy_skipped as u8 as f32,
                update_cycle,
            );
            let train_secs = elapsed.as_secs_f32();
            let cycle_secs = wait_secs + train_secs;
            writer.add_scalar(
                "throughput/train_steps_per_sec",
                steps_per_sec,
                update_cycle,
            );
            writer.add_scalar("throughput/train_secs", train_secs, update_cycle);
            writer.add_scalar("throughput/wait_secs", wait_secs, update_cycle);
            writer.add_scalar("throughput/cycle_secs", cycle_secs, update_cycle);
            writer.add_scalar("throughput/segments", n_segments as f32, update_cycle);
            writer.add_scalar(
                "throughput/value_replay_size",
                value_replay.len() as f32,
                update_cycle,
            );
            let data_steps_per_sec = if wait_secs > 0.01 {
                total_steps as f32 / wait_secs
            } else {
                0.0
            };
            writer.add_scalar(
                "throughput/data_steps_per_sec",
                data_steps_per_sec,
                update_cycle,
            );
            writer.add_scalar(
                "throughput/wait_fraction",
                wait_secs / cycle_secs.max(1e-6),
                update_cycle,
            );
            writer.flush();

            let burnin_tag = if policy_skipped { " [BURNIN]" } else { "" };
            println!(
                "[ppo] cycle={update_cycle:>4}  segs={n_segments:>3}  steps={total_steps:>5}  \
                 p_loss={avg_ploss:.4}  v_loss={avg_vloss:.4}  entropy={avg_ent:.3}  \
                 clip={avg_clip:.3}  expl_var={first_epoch_explained_var:.3}  \
                 wait={wait_secs:.1}s  train={train_secs:.1}s{burnin_tag}",
            );

            t_wait_start = Instant::now();
        }
    })
}
