//! PPO (Proximal Policy Optimization) training thread.
//!
//! Receives [`Segment`]s from the game thread, trains both policy and value
//! networks, and periodically syncs updated policy weights back to the
//! inference net used by the game thread.

use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tensorboard_rs::summary_writer::SummaryWriter;

use burn::{
    optim::{GradientsParams, Optimizer},
    prelude::*,
    tensor::{Tensor, TensorData, activation::log_softmax, backend::AutodiffBackend},
};
use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::gae::{self, SegmentInfo};
use crate::model::{
    self, InferenceNet, N_OBJECTS, OBJECT_INPUT_DIM, RLInner, SELF_INPUT_DIM, TrainBackend,
    VALUE_OUTPUT_DIM,
};
use crate::rl_collection::Segment;
use crate::rl_obs::DiscreteAction;
use crate::value_fn;

// ---------------------------------------------------------------------------
// PPO hyperparameters
// ---------------------------------------------------------------------------

const PPO_GAMMA: f32 = 0.99;
const PPO_LAMBDA: f32 = 0.95;
const PPO_CLIP_EPS: f32 = 0.2;
const PPO_ENTROPY_COEFF: f32 = 0.01;
const PPO_POLICY_LR: f64 = 3e-4;
const PPO_VALUE_LR: f64 = 1e-3;
const PPO_EPOCHS: usize = 4;
const PPO_MINI_BATCH_SIZE: usize = 256;
/// Minimum segments before first update (~2048 steps at 128 steps/seg).
const PPO_MIN_SEGMENTS: usize = 16;
const PPO_WEIGHT_SYNC_INTERVAL: usize = 1;
const PPO_SAVE_INTERVAL: usize = 10;
const PPO_HUBER_DELTA: f32 = 1.0;
/// Skip policy updates when explained variance is below this threshold,
/// allowing the value function to burn in before the policy starts changing.
const PPO_VALUE_BURNIN_EV_THRESHOLD: f32 = 0.8;

// ---------------------------------------------------------------------------
// Batch data structures
// ---------------------------------------------------------------------------

/// Flattened training batch extracted from collected segments.
pub struct PpoBatch {
    pub self_flat: Vec<f32>,
    pub obj_flat: Vec<f32>,
    pub actions: Vec<DiscreteAction>,
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
    /// Log π(a|s) recorded at rollout time (behaviour policy).
    pub old_log_probs: Vec<f32>,
    pub segment_infos: Vec<SegmentInfo>,
    pub total_steps: usize,
}

/// Flatten a collection of segments into contiguous arrays for training.
pub fn flatten_segments(segments: &[Segment]) -> PpoBatch {
    let total_steps: usize = segments.iter().map(|s| s.transitions.len()).sum();
    let mut self_flat = Vec::with_capacity(total_steps * SELF_INPUT_DIM);
    let mut obj_flat = Vec::with_capacity(total_steps * N_OBJECTS * OBJECT_INPUT_DIM);
    let mut actions = Vec::with_capacity(total_steps);
    let mut rewards = Vec::with_capacity(total_steps);
    let mut dones = Vec::with_capacity(total_steps);
    let mut old_log_probs = Vec::with_capacity(total_steps);
    let mut segment_infos = Vec::with_capacity(segments.len());
    let mut idx = 0;

    for seg in segments {
        let start = idx;
        for t in &seg.transitions {
            let (s, o) = model::split_obs(&t.obs);
            self_flat.extend_from_slice(s);
            obj_flat.extend_from_slice(o);
            actions.push(t.action);
            rewards.push(t.reward);
            dones.push(t.done);
            old_log_probs.push(t.log_prob);
            idx += 1;
        }
        segment_infos.push(SegmentInfo {
            start_idx: start,
            end_idx: idx,
            bootstrap_value: seg.bootstrap_value.unwrap_or(0.0),
        });
    }

    PpoBatch {
        self_flat,
        obj_flat,
        actions,
        rewards,
        dones,
        old_log_probs,
        segment_infos,
        total_steps,
    }
}

// ---------------------------------------------------------------------------
// Log-prob and entropy for factored action space
// ---------------------------------------------------------------------------

/// Head descriptor: (offset into action_logits, num_classes).
const ACTION_HEADS: [(usize, usize); 4] = [
    (0, 3), // turn
    (3, 2), // thrust
    (5, 2), // fire_primary
    (7, 2), // fire_secondary
];

/// Compute per-sample log-probability and entropy for the factored action space.
///
/// Returns `(log_probs [B], entropy [B])` where each is the sum across all 5 heads
/// (4 action heads + 1 target head).
pub fn compute_log_probs_and_entropy<B: Backend>(
    action_logits: Tensor<B, 2>, // [B, POLICY_OUTPUT_DIM=9]
    target_logits: Tensor<B, 2>, // [B, TARGET_OUTPUT_DIM=13]
    actions: &[DiscreteAction],
    device: &B::Device,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let b = actions.len();

    let mut total_log_prob: Option<Tensor<B, 1>> = None;
    let mut total_entropy: Option<Tensor<B, 1>> = None;

    // Helper: accumulate a head's log-prob and entropy.
    let mut accumulate_head = |logits: Tensor<B, 2>, indices: Vec<i64>| {
        let log_p = log_softmax(logits.clone(), 1); // [B, C]
        // Gather log-prob at taken action: build [B, 1] index tensor, gather, squeeze.
        let idx_tensor = Tensor::<B, 2, Int>::from_data(TensorData::new(indices, [b, 1]), device);
        let gathered: Tensor<B, 1> = log_p.clone().gather(1, idx_tensor).squeeze_dim::<1>(1); // [B]

        // Entropy: -sum(p * log_p, dim=1)
        let p = burn::tensor::activation::softmax(logits, 1);
        let ent: Tensor<B, 1> = -(p * log_p).sum_dim(1).squeeze_dim::<1>(1); // [B]

        total_log_prob = Some(match total_log_prob.take() {
            Some(acc) => acc + gathered,
            None => gathered,
        });
        total_entropy = Some(match total_entropy.take() {
            Some(acc) => acc + ent,
            None => ent,
        });
    };

    // 4 action heads
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

    // Target head
    {
        let indices: Vec<i64> = actions.iter().map(|a| a.4 as i64).collect();
        accumulate_head(target_logits, indices);
    }

    (total_log_prob.unwrap(), total_entropy.unwrap())
}

// ---------------------------------------------------------------------------
// PPO clipped loss
// ---------------------------------------------------------------------------

/// Diagnostics extracted from the PPO clipped loss computation.
pub struct PpoLossDiag {
    pub mean_ratio: f32,
    pub frac_clipped: f32,
}

/// Compute the PPO clipped surrogate loss (scalar, ready for `.backward()`).
///
/// Returns `(loss, diagnostics)`.
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

    // Extract ratio stats before further computation consumes it.
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

    // Element-wise min: stack to [B, 2] then min along dim 1.
    let stacked = Tensor::stack::<2>(vec![surr1, surr2], 1); // [B, 2]
    let min_surr: Tensor<B, 1> = stacked.min_dim(1).squeeze_dim::<1>(1); // [B]

    let loss = -(min_surr.mean());
    let diag = PpoLossDiag { mean_ratio, frac_clipped };
    (loss, diag)
}

// ---------------------------------------------------------------------------
// RL segment buffer persistence
// ---------------------------------------------------------------------------

/// Serialize collected segments to `path` for warm-start on resume.
fn save_rl_buffer(segments: &[Segment], path: &str) {
    use std::io::Write;
    let result = (|| -> std::io::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        // Header: segment count.
        f.write_all(&(segments.len() as u32).to_le_bytes())?;
        for seg in segments {
            // Segment header.
            f.write_all(&(seg.transitions.len() as u32).to_le_bytes())?;
            let bv = seg.bootstrap_value.unwrap_or(f32::NAN);
            f.write_all(&bv.to_le_bytes())?;
            // Transitions.
            for t in &seg.transitions {
                f.write_all(&[t.action.0, t.action.1, t.action.2, t.action.3, t.action.4])?;
                f.write_all(&t.reward.to_le_bytes())?;
                f.write_all(&[t.done as u8])?;
                f.write_all(&t.log_prob.to_le_bytes())?;
                for &v in &t.obs {
                    f.write_all(&v.to_le_bytes())?;
                }
            }
        }
        Ok(())
    })();
    match result {
        Ok(()) => println!("[ppo] Buffer saved ({} segments) → {path}", segments.len()),
        Err(e) => eprintln!("[ppo] Failed to save buffer to {path}: {e}"),
    }
}

/// Deserialize segments from `path`. Returns `None` if missing or corrupt.
fn load_rl_buffer(path: &str) -> Option<Vec<Segment>> {
    use crate::rl_obs::OBS_DIM;
    use std::io::Read;
    let mut f = std::io::BufReader::new(std::fs::File::open(path).ok()?);

    let mut u32_buf = [0u8; 4];
    f.read_exact(&mut u32_buf).ok()?;
    let n_segments = u32::from_le_bytes(u32_buf) as usize;

    let mut segments = Vec::with_capacity(n_segments);
    for _ in 0..n_segments {
        f.read_exact(&mut u32_buf).ok()?;
        let n_trans = u32::from_le_bytes(u32_buf) as usize;

        let mut bv_buf = [0u8; 4];
        f.read_exact(&mut bv_buf).ok()?;
        let bv = f32::from_le_bytes(bv_buf);
        let bootstrap_value = if bv.is_nan() { None } else { Some(bv) };

        let mut transitions = Vec::with_capacity(n_trans);
        let mut action_buf = [0u8; 5];
        let mut f32_buf = [0u8; 4];
        let mut done_buf = [0u8; 1];
        let mut obs_bytes = vec![0u8; OBS_DIM * 4];

        for _ in 0..n_trans {
            f.read_exact(&mut action_buf).ok()?;
            f.read_exact(&mut f32_buf).ok()?;
            let reward = f32::from_le_bytes(f32_buf);
            f.read_exact(&mut done_buf).ok()?;
            let done = done_buf[0] != 0;
            f.read_exact(&mut f32_buf).ok()?;
            let log_prob = f32::from_le_bytes(f32_buf);
            f.read_exact(&mut obs_bytes).ok()?;
            let obs: Vec<f32> = obs_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();

            transitions.push(crate::rl_collection::Transition {
                obs,
                action: (
                    action_buf[0],
                    action_buf[1],
                    action_buf[2],
                    action_buf[3],
                    action_buf[4],
                ),
                reward,
                done,
                log_prob,
            });
        }

        segments.push(Segment {
            entity_id: 0,
            personality: crate::ship::Personality::Fighter,
            initial_hidden: vec![],
            transitions,
            bootstrap_value,
        });
    }

    println!("[ppo] Loaded {n_segments} segments from {path}");
    Some(segments)
}

// ---------------------------------------------------------------------------
// Main training thread
// ---------------------------------------------------------------------------

/// Save all PPO training state: networks, optimizers, and segment buffer.
fn save_all_checkpoints(
    inner: &RLInner<TrainBackend>,
    segments: &[Segment],
    policy_path: &str,
    value_path: &str,
    policy_optim_path: &str,
    value_optim_path: &str,
    buffer_path: &str,
) {
    model::save_training_net(inner.policy_net.as_ref().unwrap(), policy_path);
    model::save_training_net(inner.value_net.as_ref().unwrap(), value_path);
    model::save_optimizer(&inner.policy_optim, policy_optim_path);
    model::save_optimizer(&inner.value_optim, value_optim_path);
    save_rl_buffer(segments, buffer_path);
}

/// Spawn the PPO training thread.
///
/// Mirrors [`crate::rl_collection::spawn_bc_training_thread`] in structure:
/// owns the receiver, trains on GPU, syncs weights back to the inference net.
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

        // Try to load existing checkpoints.
        if !experiment.is_fresh {
            if let Some(net) = model::load_training_net(&policy_path, &device) {
                inner.policy_net = Some(net);
                // Also push loaded weights to inference net.
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

        let mut update_cycle: usize = 0;
        let mut segments: Vec<Segment> = if !experiment.is_fresh {
            load_rl_buffer(&buffer_path).unwrap_or_default()
        } else {
            Vec::new()
        };
        let mut rng = thread_rng();

        // TensorBoard writer — logs go to the experiment run directory.
        let tb_dir = format!("{}/tb", experiment.run_dir);
        std::fs::create_dir_all(&tb_dir).ok();
        let mut writer = SummaryWriter::new(&tb_dir);
        // Log hyperparameters once.
        writer.add_scalar("hparams/gamma", PPO_GAMMA, 0);
        writer.add_scalar("hparams/lambda", PPO_LAMBDA, 0);
        writer.add_scalar("hparams/clip_eps", PPO_CLIP_EPS, 0);
        writer.add_scalar("hparams/entropy_coeff", PPO_ENTROPY_COEFF, 0);
        writer.add_scalar("hparams/policy_lr", PPO_POLICY_LR as f32, 0);
        writer.add_scalar("hparams/value_lr", PPO_VALUE_LR as f32, 0);
        writer.add_scalar("hparams/epochs", PPO_EPOCHS as f32, 0);
        writer.add_scalar("hparams/mini_batch_size", PPO_MINI_BATCH_SIZE as f32, 0);

        println!("[ppo] Training thread started, waiting for segments...");

        loop {
            // ── Phase 1: Collect segments ─────────────────────────────────
            // Block until at least one segment arrives, then drain non-blocking.
            if segments.len() < PPO_MIN_SEGMENTS {
                match rl_rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(seg) => segments.push(seg),
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        println!("[ppo] Channel disconnected, saving and exiting.");
                        save_all_checkpoints(&inner, &segments, &policy_path, &value_path,
                            &policy_optim_path, &value_optim_path, &buffer_path);
                        break;
                    }
                }
            }
            // Non-blocking drain of any additional segments.
            while let Ok(seg) = rl_rx.try_recv() {
                segments.push(seg);
            }
            if segments.len() < PPO_MIN_SEGMENTS {
                continue;
            }

            let t0 = Instant::now();
            let n_segments = segments.len();

            // ── Rollout metrics (before flattening) ──────────────────────
            let mut total_reward = 0.0_f32;
            let mut total_traj_steps = 0_usize;
            for seg in &segments {
                let seg_reward: f32 = seg.transitions.iter().map(|t| t.reward).sum();
                total_reward += seg_reward;
                total_traj_steps += seg.transitions.len();
                writer.add_scalar(
                    "reward/trajectory_length",
                    seg.transitions.len() as f32,
                    update_cycle,
                );
            }
            let mean_reward_per_step = if total_traj_steps > 0 {
                total_reward / total_traj_steps as f32
            } else {
                0.0
            };
            let mean_segment_return = if n_segments > 0 {
                total_reward / n_segments as f32
            } else {
                0.0
            };
            writer.add_scalar("reward/mean_per_step", mean_reward_per_step, update_cycle);
            writer.add_scalar("reward/mean_segment_return", mean_segment_return, update_cycle);

            // ── Phase 2: Recompute bootstrap values ──────────────────────
            value_fn::recompute_bootstrap_values(
                &mut segments,
                inner.value_net.as_ref().unwrap(),
                &device,
            );

            // ── Phase 3: Flatten segments into a batch ───────────────────
            let batch = flatten_segments(&segments);
            let total_steps = batch.total_steps;

            // ── Phase 4: PPO epochs ───────────────────────────────────────
            // Old log-probs were recorded at rollout time and are stored in
            // `batch.old_log_probs`, so no re-inference is needed.
            let mut epoch_policy_loss = 0.0_f32;
            let mut epoch_value_loss = 0.0_f32;
            let mut epoch_entropy = 0.0_f32;
            let mut epoch_frac_clipped = 0.0_f32;
            let mut epoch_mean_ratio = 0.0_f32;
            let mut total_mini_batches = 0_usize;
            // Track advantage and explained variance from the first epoch only
            // (before the value net updates change the landscape).
            let mut first_epoch_adv_mean = 0.0_f32;
            let mut first_epoch_adv_std = 0.0_f32;
            let mut first_epoch_explained_var = 0.0_f32;

            for epoch in 0..PPO_EPOCHS {
                // 5a. Fresh value estimates (detached).
                let values = value_fn::batch_value_inference(
                    inner.value_net.as_ref().unwrap(),
                    &batch.self_flat,
                    &batch.obj_flat,
                    total_steps,
                    &device,
                );

                // 5b. GAE with fresh values.
                let (advantages, returns) = gae::compute_gae(
                    &batch.rewards,
                    &batch.dones,
                    &values,
                    &batch.segment_infos,
                    PPO_GAMMA,
                    PPO_LAMBDA,
                );

                // 5c. Normalize advantages.
                let n = advantages.len() as f32;
                let adv_mean: f32 = advantages.iter().sum::<f32>() / n;
                let adv_var: f32 =
                    advantages.iter().map(|a| (a - adv_mean).powi(2)).sum::<f32>() / n;
                let adv_std = adv_var.sqrt();
                let norm_advantages: Vec<f32> = advantages
                    .iter()
                    .map(|a| (a - adv_mean) / (adv_std + 1e-8))
                    .collect();

                // Explained variance: 1 - Var(returns - V) / Var(returns)
                if epoch == 0 {
                    first_epoch_adv_mean = adv_mean;
                    first_epoch_adv_std = adv_std;
                    let return_mean: f32 = returns.iter().sum::<f32>() / n;
                    let var_returns: f32 =
                        returns.iter().map(|r| (r - return_mean).powi(2)).sum::<f32>() / n;
                    let var_residuals: f32 = advantages.iter().map(|a| a.powi(2)).sum::<f32>() / n;
                    first_epoch_explained_var = 1.0 - var_residuals / (var_returns + 1e-8);
                }

                // Skip policy updates until the value function is accurate enough.
                let skip_policy = first_epoch_explained_var < PPO_VALUE_BURNIN_EV_THRESHOLD;

                // 5d. Shuffle indices for mini-batching.
                let mut indices: Vec<usize> = (0..total_steps).collect();
                indices.shuffle(&mut rng);

                // 5e. Mini-batch updates.
                for mb_indices in indices.chunks(PPO_MINI_BATCH_SIZE) {
                    let mb = mb_indices.len();

                    // Gather mini-batch data.
                    let mut mb_self = Vec::with_capacity(mb * SELF_INPUT_DIM);
                    let mut mb_obj = Vec::with_capacity(mb * N_OBJECTS * OBJECT_INPUT_DIM);
                    let mut mb_actions = Vec::with_capacity(mb);
                    let mut mb_old_lp = Vec::with_capacity(mb);
                    let mut mb_adv = Vec::with_capacity(mb);
                    let mut mb_ret = Vec::with_capacity(mb);

                    for &i in mb_indices {
                        let s_start = i * SELF_INPUT_DIM;
                        mb_self
                            .extend_from_slice(&batch.self_flat[s_start..s_start + SELF_INPUT_DIM]);
                        let o_start = i * N_OBJECTS * OBJECT_INPUT_DIM;
                        mb_obj.extend_from_slice(
                            &batch.obj_flat[o_start..o_start + N_OBJECTS * OBJECT_INPUT_DIM],
                        );
                        mb_actions.push(batch.actions[i]);
                        mb_old_lp.push(batch.old_log_probs[i]);
                        mb_adv.push(norm_advantages[i]);
                        mb_ret.push(returns[i]);
                    }

                    // Build tensors.
                    let self_t = Tensor::<TrainBackend, 2>::from_data(
                        TensorData::new(mb_self, [mb, SELF_INPUT_DIM]),
                        &device,
                    );
                    let obj_t = Tensor::<TrainBackend, 3>::from_data(
                        TensorData::new(mb_obj, [mb, N_OBJECTS, OBJECT_INPUT_DIM]),
                        &device,
                    );

                    // ── Policy update (skipped during value burn-in) ──────
                    if !skip_policy {
                        let policy_grads = {
                            let net = inner.policy_net.as_ref().unwrap();
                            let (action_logits, target_logits) =
                                net.forward(self_t.clone(), obj_t.clone());
                            let (new_lp, entropy) = compute_log_probs_and_entropy(
                                action_logits,
                                target_logits,
                                &mb_actions,
                                &device,
                            );

                            let (policy_loss, diag) = ppo_clipped_loss(
                                new_lp, &mb_old_lp, &mb_adv, PPO_CLIP_EPS, &device,
                            );
                            let entropy_bonus = -(entropy.mean());
                            let total_policy_loss =
                                policy_loss.clone() + entropy_bonus.clone() * PPO_ENTROPY_COEFF;

                            epoch_policy_loss += f32::from(policy_loss.clone().into_scalar());
                            epoch_entropy += f32::from((-entropy_bonus.clone()).into_scalar());
                            epoch_frac_clipped += diag.frac_clipped;
                            epoch_mean_ratio += diag.mean_ratio;

                            let raw = total_policy_loss.backward();
                            GradientsParams::from_grads(raw, net)
                        };
                        let net = inner.policy_net.take().unwrap();
                        inner.policy_net =
                            Some(inner.policy_optim.step(PPO_POLICY_LR, net, policy_grads));
                    }

                    // ── Value update ──────────────────────────────────────
                    let value_grads = {
                        let vnet = inner.value_net.as_ref().unwrap();
                        let (value_out, _) = vnet.forward(self_t, obj_t);
                        let value_pred: Tensor<TrainBackend, 1> = value_out.squeeze_dim::<1>(1); // [B]
                        let targets = Tensor::<TrainBackend, 1>::from_data(
                            TensorData::new(mb_ret, [mb]),
                            &device,
                        );
                        let vloss =
                            value_fn::huber_value_loss(value_pred, targets, PPO_HUBER_DELTA);

                        epoch_value_loss += f32::from(vloss.clone().into_scalar());

                        let raw = vloss.backward();
                        GradientsParams::from_grads(raw, vnet)
                    };
                    let vnet = inner.value_net.take().unwrap();
                    inner.value_net = Some(inner.value_optim.step(PPO_VALUE_LR, vnet, value_grads));

                    total_mini_batches += 1;
                }
            }

            // ── Phase 6: Discard on-policy data ──────────────────────────
            segments.clear();
            update_cycle += 1;

            // ── Phase 7: Weight sync ─────────────────────────────────────
            if update_cycle % PPO_WEIGHT_SYNC_INTERVAL == 0 {
                let bytes = model::training_net_to_bytes(inner.policy_net.as_ref().unwrap());
                if let Ok(mut lock) = inference_net.lock() {
                    lock.load_bytes(bytes);
                }
            }

            // ── Phase 8: Checkpoint ──────────────────────────────────────
            if update_cycle % PPO_SAVE_INTERVAL == 0 {
                save_all_checkpoints(&inner, &segments, &policy_path, &value_path,
                    &policy_optim_path, &value_optim_path, &buffer_path);
            }

            // ── Phase 9: Diagnostics + TensorBoard ───────────────────────
            let elapsed = t0.elapsed();
            let n_mb = total_mini_batches.max(1) as f32;
            let avg_ploss = epoch_policy_loss / n_mb;
            let avg_vloss = epoch_value_loss / n_mb;
            let avg_ent = epoch_entropy / n_mb;
            let avg_clip = epoch_frac_clipped / n_mb;
            let avg_ratio = epoch_mean_ratio / n_mb;
            let steps_per_sec = total_steps as f32 / elapsed.as_secs_f32();

            // Training losses.
            writer.add_scalar("train/policy_loss", avg_ploss, update_cycle);
            writer.add_scalar("train/value_loss", avg_vloss, update_cycle);
            writer.add_scalar("train/entropy", avg_ent, update_cycle);
            // PPO-specific diagnostics.
            writer.add_scalar("train/frac_clipped", avg_clip, update_cycle);
            writer.add_scalar("train/mean_ratio", avg_ratio, update_cycle);
            // Advantage statistics (from first epoch, before value net updates).
            writer.add_scalar("train/advantage_mean", first_epoch_adv_mean, update_cycle);
            writer.add_scalar("train/advantage_std", first_epoch_adv_std, update_cycle);
            writer.add_scalar("train/explained_variance", first_epoch_explained_var, update_cycle);
            let policy_skipped = first_epoch_explained_var < PPO_VALUE_BURNIN_EV_THRESHOLD;
            writer.add_scalar("train/value_burnin", policy_skipped as u8 as f32, update_cycle);
            // Throughput.
            writer.add_scalar("throughput/steps_per_sec", steps_per_sec, update_cycle);
            writer.add_scalar("throughput/segments", n_segments as f32, update_cycle);
            writer.flush();

            let burnin_tag = if policy_skipped { " [BURNIN]" } else { "" };
            println!(
                "[ppo] cycle={update_cycle:>4}  segs={n_segments:>3}  steps={total_steps:>5}  \
                 p_loss={avg_ploss:.4}  v_loss={avg_vloss:.4}  entropy={avg_ent:.3}  \
                 clip={avg_clip:.3}  expl_var={first_epoch_explained_var:.3}  \
                 time={:.1}s{burnin_tag}",
                elapsed.as_secs_f32()
            );
        }
    })
}
