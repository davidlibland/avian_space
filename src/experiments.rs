//! Experiment directory management.
//!
//! Training artefacts (checkpoints, logs) are always written to
//! `experiments/run_<N>/`, where N is the highest existing run number.
//! Pass `--fresh` on the command line to start a new run in
//! `experiments/run_<N+1>/` instead of resuming.
//!
//! The only public surface this module exposes is [`setup_experiment`] and
//! the [`ExperimentSetup`] it returns.  All path-construction details stay
//! here so the rest of the codebase never hard-codes a checkpoint path.

use bevy::prelude::*;
use std::fs;
use std::path::Path;

const EXPERIMENTS_DIR: &str = "experiments";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Everything a training thread needs to know about the current run.
pub struct ExperimentSetup {
    /// Path to the run directory (e.g. `experiments/run_3`).
    pub run_dir: String,
    /// True when this is a brand-new run (no weights to load).
    pub is_fresh: bool,
}

/// Lightweight Bevy resource so game-thread systems can read/write
/// checkpoints to the current experiment directory.
#[derive(Resource, Clone)]
pub struct ExperimentDir {
    pub run_dir: String,
    pub is_fresh: bool,
}

impl ExperimentDir {
    /// Path for the serialised combat-hit statistics.
    pub fn combat_stats_path(&self) -> String {
        format!("{}/combat_stats.bin", self.run_dir)
    }
}

impl ExperimentSetup {
    /// Path prefix passed to burn's save/load helpers (burn appends `.bin`).
    ///
    /// Example: `"experiments/run_3/policy"` → file is `experiments/run_3/policy.bin`.
    pub fn policy_checkpoint_path(&self) -> String {
        format!("{}/policy", self.run_dir)
    }

    /// Path prefix for the value-network checkpoint (burn appends `.bin`).
    pub fn value_checkpoint_path(&self) -> String {
        format!("{}/value", self.run_dir)
    }

    /// Path for the serialised BC replay buffer.
    pub fn buffer_checkpoint_path(&self) -> String {
        format!("{}/bc_buffer.bin", self.run_dir)
    }

    /// Path prefix for the policy optimizer state (burn appends `.bin`).
    pub fn policy_optim_checkpoint_path(&self) -> String {
        format!("{}/policy_optim", self.run_dir)
    }

    /// Path prefix for the value optimizer state (burn appends `.bin`).
    pub fn value_optim_checkpoint_path(&self) -> String {
        format!("{}/value_optim", self.run_dir)
    }

    /// Path for the serialised RL segment buffer.
    pub fn rl_buffer_checkpoint_path(&self) -> String {
        format!("{}/rl_buffer.bin", self.run_dir)
    }

    /// Path for the serialised combat-hit statistics.
    pub fn combat_stats_checkpoint_path(&self) -> String {
        format!("{}/combat_stats.bin", self.run_dir)
    }

    /// Returns true if a checkpoint file is present on disk.
    pub fn checkpoint_exists(&self) -> bool {
        Path::new(&format!("{}.bin", self.policy_checkpoint_path())).exists()
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Determine (and create if necessary) the run directory for this process.
///
/// * `fresh = false` — resume the highest-numbered existing run; if none
///   exists, falls through to `fresh = true` behaviour.
/// * `fresh = true` — create `experiments/run_<highest+1>/` (or `run_0` if
///   this is the very first run).
pub fn setup_experiment(fresh: bool) -> ExperimentSetup {
    fs::create_dir_all(EXPERIMENTS_DIR)
        .unwrap_or_else(|e| eprintln!("[experiment] Failed to create {EXPERIMENTS_DIR}: {e}"));

    let highest_id = find_highest_run_id();

    if !fresh {
        if let Some(id) = highest_id {
            let dir = format!("{EXPERIMENTS_DIR}/run_{id}");
            println!("[experiment] Resuming run {id} from {dir}");
            return ExperimentSetup { run_dir: dir, is_fresh: false };
        }
    }

    let new_id = highest_id.map(|id| id + 1).unwrap_or(0);
    let dir = format!("{EXPERIMENTS_DIR}/run_{new_id}");
    fs::create_dir_all(&dir)
        .unwrap_or_else(|e| eprintln!("[experiment] Failed to create {dir}: {e}"));
    println!("[experiment] Starting fresh run {new_id} at {dir}");

    ExperimentSetup { run_dir: dir, is_fresh: true }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn find_highest_run_id() -> Option<u32> {
    let entries = fs::read_dir(EXPERIMENTS_DIR).ok()?;
    let mut max_id: Option<u32> = None;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(suffix) = name.strip_prefix("run_") {
            if let Ok(id) = suffix.parse::<u32>() {
                max_id = Some(max_id.map_or(id, |m: u32| m.max(id)));
            }
        }
    }
    max_id
}
