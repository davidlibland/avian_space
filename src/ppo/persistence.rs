//! Serialise / deserialise RL segment buffers and small step-counter files.

use crate::consts::N_REWARD_TYPES;
use crate::model::{self, TrainBackend};
use crate::rl_collection::Segment;

use super::batch::personality_index;

/// Serialize collected segments to `path` for warm-start on resume.
pub fn save_rl_buffer(segments: &[Segment], path: &str) {
    use std::io::Write;
    let result = (|| -> std::io::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(&(segments.len() as u32).to_le_bytes())?;
        for seg in segments {
            f.write_all(&(seg.transitions.len() as u32).to_le_bytes())?;
            match &seg.bootstrap_value {
                Some(bv) => {
                    f.write_all(&[1u8])?;
                    for &v in bv {
                        f.write_all(&v.to_le_bytes())?;
                    }
                }
                None => f.write_all(&[0u8])?,
            }
            f.write_all(&[personality_index(&seg.personality) as u8])?;
            for t in &seg.transitions {
                f.write_all(&[
                    t.action.0, t.action.1, t.action.2, t.action.3, t.action.4, t.action.5,
                ])?;
                for &r in &t.rewards {
                    f.write_all(&r.to_le_bytes())?;
                }
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
pub fn load_rl_buffer(path: &str) -> Option<Vec<Segment>> {
    use crate::rl_obs::OBS_DIM;
    use std::io::Read;
    let mut f = std::io::BufReader::new(std::fs::File::open(path).ok()?);

    let mut u32_buf = [0u8; 4];
    f.read_exact(&mut u32_buf).ok()?;
    let n_segments = u32::from_le_bytes(u32_buf) as usize;

    let personalities_map = [
        crate::ship::Personality::Miner,
        crate::ship::Personality::Fighter,
        crate::ship::Personality::Trader,
    ];

    let mut segments = Vec::with_capacity(n_segments);
    for _ in 0..n_segments {
        f.read_exact(&mut u32_buf).ok()?;
        let n_trans = u32::from_le_bytes(u32_buf) as usize;

        let mut flag = [0u8; 1];
        f.read_exact(&mut flag).ok()?;
        let bootstrap_value = if flag[0] == 1 {
            let mut bv = [0.0_f32; N_REWARD_TYPES];
            let mut bv_bytes = [0u8; 4];
            for v in &mut bv {
                f.read_exact(&mut bv_bytes).ok()?;
                *v = f32::from_le_bytes(bv_bytes);
            }
            Some(bv)
        } else {
            None
        };

        let mut pers_buf = [0u8; 1];
        f.read_exact(&mut pers_buf).ok()?;
        let personality = personalities_map
            .get(pers_buf[0] as usize)
            .cloned()
            .unwrap_or(crate::ship::Personality::Fighter);

        let mut transitions = Vec::with_capacity(n_trans);
        let mut action_buf = [0u8; 6];
        let mut f32_buf = [0u8; 4];
        let mut done_buf = [0u8; 1];
        let mut obs_bytes = vec![0u8; OBS_DIM * 4];

        for _ in 0..n_trans {
            f.read_exact(&mut action_buf).ok()?;
            let mut rewards = [0.0_f32; N_REWARD_TYPES];
            for r in &mut rewards {
                f.read_exact(&mut f32_buf).ok()?;
                *r = f32::from_le_bytes(f32_buf);
            }
            f.read_exact(&mut done_buf).ok()?;
            let done = done_buf[0] != 0;
            f.read_exact(&mut f32_buf).ok()?;
            let log_prob = f32::from_le_bytes(f32_buf);
            f.read_exact(&mut obs_bytes).ok()?;
            let obs: Vec<f32> = obs_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();

            let action = (
                action_buf[0],
                action_buf[1],
                action_buf[2],
                action_buf[3],
                action_buf[4],
                action_buf[5],
            );
            transitions.push(crate::rl_collection::Transition {
                obs,
                proj_obs: vec![0.0; model::PROJECTILES_FLAT_DIM],
                action,
                // Legacy checkpoints predate the inline BC label — fall back
                // to the executed action.
                rule_based_action: action,
                rewards,
                done,
                log_prob,
            });
        }

        segments.push(Segment {
            personality,
            transitions,
            bootstrap_value,
        });
    }

    println!("[ppo] Loaded {n_segments} segments from {path}");
    Some(segments)
}

pub fn save_step_counter(step: usize, path: &str) {
    if let Err(e) = std::fs::write(path, (step as u64).to_le_bytes()) {
        eprintln!("[ppo] Failed to save step counter to {path}: {e}");
    }
}

pub fn load_step_counter(path: &str) -> Option<usize> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 8 {
        return None;
    }
    let val = u64::from_le_bytes(bytes[..8].try_into().ok()?);
    println!("[ppo] Loaded step counter = {val} from {path}");
    Some(val as usize)
}

/// Save all PPO training state: networks, optimizers, segment buffer, step counter.
pub fn save_all_checkpoints(
    inner: &crate::model::RLInner<TrainBackend>,
    segments: &[Segment],
    update_cycle: usize,
    policy_path: &str,
    value_path: &str,
    policy_optim_path: &str,
    value_optim_path: &str,
    buffer_path: &str,
    step_counter_path: &str,
) {
    model::save_training_net(inner.policy_net.as_ref().unwrap(), policy_path);
    model::save_training_net(inner.value_net.as_ref().unwrap(), value_path);
    model::save_optimizer(&inner.policy_optim, policy_optim_path);
    model::save_optimizer(&inner.value_optim, value_optim_path);
    save_rl_buffer(segments, buffer_path);
    save_step_counter(update_cycle, step_counter_path);
}
