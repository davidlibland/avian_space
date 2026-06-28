//! Behavioural-cloning pre-training: loss helpers, replay-buffer persistence,
//! and the background training thread.

use std::collections::VecDeque;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, TensorData};
use rand::Rng;

use crate::model::{
    self, InferenceNet, N_OBJECTS, OBJECT_INPUT_DIM, RLInner, SELF_INPUT_DIM, TrainBackend,
    split_obs, training_net_to_bytes,
};
use crate::rl_obs::DiscreteAction;

use super::BCTransition;

// BC training hyperparameters
const BC_BUFFER_SIZE: usize = 32_768;
const BC_BATCH_SIZE: usize = 256;
const BC_LR: f64 = 3e-4;
const BC_WEIGHT_SYNC_INTERVAL: usize = 50;
const BC_SAVE_INTERVAL: usize = 1_000;
const BC_STEPS_PER_DRAIN: usize = 10;

/// Compute the weighted BC cross-entropy loss from pre-computed logits and a
/// slice of expert (rule-based) action labels.
pub fn compute_bc_loss_from_logits(
    action_logits: burn::tensor::Tensor<TrainBackend, 2>,
    nav_target_logits: burn::tensor::Tensor<TrainBackend, 2>,
    wep_target_logits: burn::tensor::Tensor<TrainBackend, 2>,
    rule_based_actions: &[DiscreteAction],
    loss_fn: &burn::nn::loss::CrossEntropyLoss<TrainBackend>,
    device: &<TrainBackend as Backend>::Device,
) -> burn::tensor::Tensor<TrainBackend, 1> {
    let b = rule_based_actions.len();

    let action_labels_flat: Vec<i64> = rule_based_actions
        .iter()
        .flat_map(|a| [a.0 as i64, a.1 as i64, a.2 as i64, a.3 as i64])
        .collect();
    let action_labels = burn::tensor::Tensor::<TrainBackend, 2, Int>::from_data(
        TensorData::new(action_labels_flat, [b, 4]),
        device,
    );
    let turn_t = action_labels.clone().narrow(1, 0, 1).reshape([b]);
    let thrust_t = action_labels.clone().narrow(1, 1, 1).reshape([b]);
    let fp_t = action_labels.clone().narrow(1, 2, 1).reshape([b]);
    let fs_t = action_labels.narrow(1, 3, 1).reshape([b]);

    let nav_target_labels: Vec<i64> = rule_based_actions.iter().map(|a| a.4 as i64).collect();
    let nav_target_t = burn::tensor::Tensor::<TrainBackend, 1, Int>::from_data(
        TensorData::new(nav_target_labels, [b]),
        device,
    );
    let wep_target_labels: Vec<i64> = rule_based_actions.iter().map(|a| a.5 as i64).collect();
    let wep_target_t = burn::tensor::Tensor::<TrainBackend, 1, Int>::from_data(
        TensorData::new(wep_target_labels, [b]),
        device,
    );

    let w3 = 1.0 / (3.0_f32).ln();
    let w2 = 1.0 / (2.0_f32).ln();
    let wt = 1.0 / (model::TARGET_OUTPUT_DIM as f32).ln();

    let turn_loss = loss_fn.forward(action_logits.clone().narrow(1, 0, 3), turn_t) * w3;
    let thrust_loss = loss_fn.forward(action_logits.clone().narrow(1, 3, 2), thrust_t) * w2;
    let fp_loss = loss_fn.forward(action_logits.clone().narrow(1, 5, 2), fp_t) * w2;
    let fs_loss = loss_fn.forward(action_logits.narrow(1, 7, 2), fs_t) * w2;
    let nav_tgt_loss = loss_fn.forward(nav_target_logits, nav_target_t) * wt;
    let wep_tgt_loss = loss_fn.forward(wep_target_logits, wep_target_t) * wt;

    turn_loss + thrust_loss + fp_loss + fs_loss + nav_tgt_loss + wep_tgt_loss
}

/// Compute the weighted BC cross-entropy loss over a batch of `BCTransition`s.
pub fn compute_bc_loss(
    net: &model::RLNet<TrainBackend>,
    batch: &[&BCTransition],
    loss_fn: &burn::nn::loss::CrossEntropyLoss<TrainBackend>,
    device: &<TrainBackend as Backend>::Device,
) -> burn::tensor::Tensor<TrainBackend, 1> {
    let b = batch.len();
    let mut self_flat = Vec::with_capacity(b * SELF_INPUT_DIM);
    let mut obj_flat = Vec::with_capacity(b * N_OBJECTS * OBJECT_INPUT_DIM);
    let mut proj_flat = Vec::with_capacity(b * model::PROJECTILES_FLAT_DIM);
    for t in batch {
        let (s, o) = split_obs(&t.obs);
        self_flat.extend_from_slice(s);
        obj_flat.extend_from_slice(o);
        proj_flat.extend_from_slice(&t.proj_obs);
    }
    let self_input = burn::tensor::Tensor::<TrainBackend, 2>::from_data(
        TensorData::new(self_flat, [b, SELF_INPUT_DIM]),
        device,
    );
    let obj_input = burn::tensor::Tensor::<TrainBackend, 3>::from_data(
        TensorData::new(obj_flat, [b, N_OBJECTS, OBJECT_INPUT_DIM]),
        device,
    );
    let proj_input = burn::tensor::Tensor::<TrainBackend, 3>::from_data(
        TensorData::new(
            proj_flat,
            [b, model::N_PROJECTILE_SLOTS, model::PROJ_INPUT_DIM],
        ),
        device,
    );

    let (action_logits, nav_target_logits, wep_target_logits) =
        net.forward(self_input, obj_input, proj_input);

    let rule_based: Vec<DiscreteAction> = batch.iter().map(|t| t.action).collect();
    compute_bc_loss_from_logits(
        action_logits,
        nav_target_logits,
        wep_target_logits,
        &rule_based,
        loss_fn,
        device,
    )
}

/// Spawn the behavioural-cloning pre-training thread.
pub(super) fn spawn_bc_training_thread(
    bc_rx: mpsc::Receiver<BCTransition>,
    inference_net: Arc<Mutex<InferenceNet>>,
    experiment: crate::experiments::ExperimentSetup,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let device = Default::default();
        let mut inner = RLInner::<TrainBackend>::new(&device);

        let checkpoint_path = experiment.policy_checkpoint_path();

        if !experiment.is_fresh {
            if let Some(net) = model::load_training_net(&checkpoint_path, &device) {
                inner.policy_net = Some(net);
                let bytes = training_net_to_bytes(inner.policy_net.as_ref().unwrap());
                if let Ok(mut lock) = inference_net.lock() {
                    lock.load_bytes(bytes);
                }
                println!("[bc] Resumed from checkpoint {checkpoint_path}");
            } else {
                println!("[bc] No checkpoint found at {checkpoint_path} — starting from scratch.");
            }
        } else {
            println!("[bc] Fresh run — skipping checkpoint load.");
        }

        let buffer_path = experiment.buffer_checkpoint_path();

        let mut buffer: VecDeque<BCTransition> = if !experiment.is_fresh {
            load_bc_buffer(&buffer_path)
                .map(|b| {
                    println!(
                        "[bc] Restored buffer with {} transitions from {buffer_path}",
                        b.len()
                    );
                    b
                })
                .unwrap_or_else(|| {
                    println!("[bc] No buffer checkpoint found at {buffer_path} — starting empty.");
                    VecDeque::with_capacity(BC_BUFFER_SIZE)
                })
        } else {
            println!("[bc] Fresh run — starting with empty buffer.");
            VecDeque::with_capacity(BC_BUFFER_SIZE)
        };
        let mut step = 0usize;
        let mut rng = rand::thread_rng();

        loop {
            if buffer.len() < BC_BATCH_SIZE {
                match bc_rx.recv_timeout(std::time::Duration::from_millis(50)) {
                    Ok(t) => {
                        if buffer.len() >= BC_BUFFER_SIZE {
                            buffer.pop_front();
                        }
                        buffer.push_back(t);
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        println!("[bc] Channel disconnected — saving final checkpoint.");
                        if let Some(net) = inner.policy_net.as_ref() {
                            save_bc_checkpoint(net, &checkpoint_path);
                        }
                        save_bc_buffer(&buffer, &buffer_path);
                        break;
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                }
            }
            while let Ok(t) = bc_rx.try_recv() {
                if buffer.len() >= BC_BUFFER_SIZE {
                    buffer.pop_front();
                }
                buffer.push_back(t);
            }

            if buffer.len() < BC_BATCH_SIZE {
                continue;
            }

            let loss_fn = CrossEntropyLossConfig::new().init::<TrainBackend>(&device);

            for _ in 0..BC_STEPS_PER_DRAIN {
                let n = buffer.len();

                let batch: Vec<&BCTransition> = (0..BC_BATCH_SIZE)
                    .map(|_| &buffer[rng.gen_range(0..n)])
                    .collect();

                let log_this_step = step % 100 == 0;
                let grads = {
                    let net = inner.policy_net.as_ref().unwrap();
                    let total_loss = compute_bc_loss(net, &batch, &loss_fn, &device);
                    if log_this_step {
                        let v = f32::from(total_loss.clone().into_scalar());
                        println!("[bc] step={step:>6}  loss={v:.4}  buffer={}", buffer.len());
                    }
                    let raw = total_loss.backward();
                    GradientsParams::from_grads(raw, net)
                };

                let net = inner.policy_net.take().unwrap();
                inner.policy_net = Some(inner.policy_optim.step(BC_LR, net, grads));
                step += 1;

                if step % BC_WEIGHT_SYNC_INTERVAL == 0 {
                    let bytes = training_net_to_bytes(inner.policy_net.as_ref().unwrap());
                    if let Ok(mut lock) = inference_net.lock() {
                        lock.load_bytes(bytes);
                    }
                }

                if step % BC_SAVE_INTERVAL == 0 {
                    save_bc_checkpoint(inner.policy_net.as_ref().unwrap(), &checkpoint_path);
                    save_bc_buffer(&buffer, &buffer_path);
                }
            }
        }
    })
}

fn save_bc_checkpoint(net: &model::RLNet<TrainBackend>, path: &str) {
    model::save_training_net(net, path);
}

// ---------------------------------------------------------------------------
// BC buffer persistence
// ---------------------------------------------------------------------------

fn save_bc_buffer(buffer: &VecDeque<BCTransition>, path: &str) {
    use std::io::Write;
    let Some(first) = buffer.front() else { return };
    let input_dim = first.obs.len() as u32;

    let result = (|| -> std::io::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(&input_dim.to_le_bytes())?;
        f.write_all(&(buffer.len() as u32).to_le_bytes())?;
        for t in buffer {
            f.write_all(&[
                t.action.0, t.action.1, t.action.2, t.action.3, t.action.4, t.action.5,
            ])?;
            for &v in &t.obs {
                f.write_all(&v.to_le_bytes())?;
            }
        }
        Ok(())
    })();

    if let Err(e) = result {
        eprintln!("[bc] Failed to save buffer to {path}: {e}");
    } else {
        println!("[bc] Buffer saved ({} transitions) → {path}", buffer.len());
    }
}

pub fn load_bc_buffer(path: &str) -> Option<VecDeque<BCTransition>> {
    use crate::rl_obs::OBS_DIM;
    use std::io::Read;
    let expected_dim = OBS_DIM;
    let mut f = std::io::BufReader::new(std::fs::File::open(path).ok()?);

    let mut u32_buf = [0u8; 4];
    f.read_exact(&mut u32_buf).ok()?;
    let stored_dim = u32::from_le_bytes(u32_buf) as usize;
    if stored_dim != expected_dim {
        eprintln!(
            "[bc] Buffer at {path} has dim={stored_dim}, expected {expected_dim} — discarding."
        );
        return None;
    }

    f.read_exact(&mut u32_buf).ok()?;
    let count = u32::from_le_bytes(u32_buf) as usize;

    let mut buffer = VecDeque::with_capacity(count.min(BC_BUFFER_SIZE));
    let mut action_buf = [0u8; 6];
    let mut input_bytes = vec![0u8; expected_dim * 4];

    for _ in 0..count {
        f.read_exact(&mut action_buf).ok()?;
        f.read_exact(&mut input_bytes).ok()?;
        let obs: Vec<f32> = input_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        buffer.push_back(BCTransition {
            obs,
            proj_obs: vec![0.0; crate::rl_obs::K_PROJECTILES * crate::rl_obs::PROJ_SLOT_SIZE],
            action: (
                action_buf[0],
                action_buf[1],
                action_buf[2],
                action_buf[3],
                action_buf[4],
                action_buf[5],
            ),
        });
    }
    Some(buffer)
}
