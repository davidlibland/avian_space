//! Prototype of **concurrent multi-system RL training** (Path A).
//!
//! Stands up `N` headless rollout worlds — each seeded to a different star
//! system — on `N` threads in one process, all sharing:
//!   * one inference net (`Arc<Mutex<InferenceNet>>`) the trainer syncs into, and
//!   * one segment channel (N `SyncSender`s → 1 `Receiver`),
//! feeding a single PPO trainer thread (which owns the experiment/checkpoints,
//! exactly as in single-world training).
//!
//! Each world keeps running periodic system swaps, but with a per-world **phase
//! offset** (`swap_phase_segments`) so the swaps are staggered — at any moment
//! only ~one of the N worlds is in a post-swap value-EV dip, keeping each
//! trainer batch a mostly-stationary mixture of systems while still cycling
//! through the full distribution over time.

use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use crate::config::TrainingConfig;
use crate::model::InferenceNet;
use crate::rl_collection::{Segment, SharedCollection};

/// Distinct systems to seed worlds with (must exist in star_systems.yaml).
/// `sol` first (densest population → most agents/teammates for cooperation).
const SYSTEMS: &[&str] = &[
    "sol",
    "alpha_centauri",
    "sirius",
    "procyon",
    "barnard",
    "vega",
    "rigel",
    "altair",
    "deneb",
    "tau_ceti",
];

pub fn run(n: usize, fresh: bool, training: TrainingConfig) {
    let n = n.max(1);
    println!("[mwtrain] prototype: {n} rollout worlds → 1 shared trainer");

    // Shared inference net: trainer syncs trained weights in; all worlds read it
    // for action selection. Starts random; overwritten on the trainer's first
    // weight sync (or by the checkpoint it loads on resume).
    let shared_net = Arc::new(Mutex::new(InferenceNet::new()));

    // One segment channel: N producer worlds → 1 consumer trainer.
    let (tx, rx) = mpsc::sync_channel::<Segment>(4096);

    // Single trainer (owns experiment dir + checkpoints), identical to the
    // single-world path — it just receives a mixed stream from N worlds.
    let experiment = crate::experiments::setup_experiment(fresh);
    println!(
        "[mwtrain] trainer run_dir={} fresh={}",
        experiment.run_dir, experiment.is_fresh
    );
    crate::ppo::spawn_ppo_training_thread(
        rx,
        Arc::clone(&shared_net),
        experiment,
        training.ppo.clone(),
        training.rewards.clone(),
    );

    let interval = training.ppo.system_swap_segments;
    // Dedicated scenario worlds, pinned (never swap), one per personality gap:
    // combat arena (Fighters), escort run (Fighters defend + Traders trade
    // under threat), mining belt (Miners). Remaining worlds rotate the galaxy.
    let pinned_systems: &[&str] = &["simulator", "escort", "mining"];
    let n_pinned = pinned_systems.len().min(n);
    let n_free = n.saturating_sub(n_pinned);
    let mut handles = Vec::new();
    for i in 0..n {
        let (system, pinned) = if i < n_pinned {
            (pinned_systems[i].to_string(), true)
        } else {
            (SYSTEMS[(i - n_pinned) % SYSTEMS.len()].to_string(), false)
        };
        let net = Arc::clone(&shared_net);
        let tx = tx.clone();
        let training = training.clone();
        // Evenly stagger the non-pinned worlds' first swaps across one interval.
        let swap_phase = if !pinned && interval > 0 && n_free > 0 {
            ((i - n_pinned) * interval) / n_free
        } else {
            0
        };
        handles.push(
            std::thread::Builder::new()
                .name(format!("world-{i}"))
                .spawn(move || world_loop(i, system, pinned, training, net, tx, swap_phase))
                .expect("spawn world thread"),
        );
    }
    // Drop our own sender so the channel only stays open while worlds live.
    drop(tx);

    for h in handles {
        if h.join().is_err() {
            println!("[mwtrain] !!! a world thread PANICKED (see above)");
        }
    }
}

fn world_loop(
    idx: usize,
    system: String,
    pinned: bool,
    training: TrainingConfig,
    net: Arc<Mutex<InferenceNet>>,
    tx: mpsc::SyncSender<Segment>,
    swap_phase: usize,
) {
    let shared = SharedCollection {
        inference_net: net,
        segment_tx: tx,
        swap_phase_segments: swap_phase,
        pin_system: pinned,
    };
    // RLTraining mode → RLControl + stochastic sampling; the injected
    // `SharedCollection` makes it a collection client (no own trainer /
    // experiment). disable_log=true on every world: the process-global tracing
    // subscriber can only be installed once, and the trainer logs via println.
    let mut app = crate::build_app(
        crate::AppMode::RLTraining,
        false,
        true,
        training,
        &system,
        true,
        Some(shared),
    );
    // `App::run()` normally does finish()+cleanup(); we drive update() manually.
    app.finish();
    app.cleanup();
    let tag = if pinned { " [PINNED scenario]" } else { "" };
    println!("[mwtrain] world {idx} up in '{system}'{tag} (swap_phase={swap_phase} segments)");

    loop {
        app.update();
    }
}
