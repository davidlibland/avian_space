//! Feasibility spike for **Path A** concurrent multi-system rollout.
//!
//! Stands up `N` headless game worlds — each seeded to a different star system —
//! on `N` OS threads in a single process, then steps them and reports per-world
//! init cost and step throughput. The point is to answer the make-or-break
//! questions before committing to the real design:
//!   1. Can >1 Bevy `App` coexist in one process (global singletons: log
//!      subscriber, task pools, wgpu device)?
//!   2. What does each extra world cost to stand up (full asset/render init)?
//!   3. Does throughput actually scale across threads, or do the worlds
//!      contend on Bevy's shared global compute task pool?
//!
//! Throwaway diagnostic. Uses Classic (rule-based AI) mode so it doesn't depend
//! on the policy checkpoint; the per-frame physics/population cost it exercises
//! is the same shared cost the RL rollout path would pay on top of inference.

use std::time::{Duration, Instant};

use crate::AppMode;
use crate::config::TrainingConfig;

/// How long to step each world (wall-clock) after init.
const RUN_SECS: u64 = 12;

/// Distinct star systems to spread worlds across (must exist in
/// `assets/star_systems.yaml`). `sol` first since it has the densest population.
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

struct WorldResult {
    idx: usize,
    system: String,
    init_secs: f32,
    updates: u64,
    updates_per_sec: f32,
}

pub fn run(n: usize, _app_mode: AppMode, training: TrainingConfig) {
    let n = n.max(1);
    println!(
        "[spike] standing up {n} headless world(s) on {n} thread(s), Classic mode, {RUN_SECS}s run each"
    );
    let wall0 = Instant::now();

    let handles: Vec<_> = (0..n)
        .map(|i| {
            let system = SYSTEMS[i % SYSTEMS.len()].to_string();
            let training = training.clone();
            std::thread::Builder::new()
                .name(format!("world-{i}"))
                .spawn(move || world_thread(i, system, training))
                .expect("spawn world thread")
        })
        .collect();

    let mut results = Vec::new();
    for h in handles {
        match h.join() {
            Ok(r) => results.push(r),
            Err(_) => println!("[spike] !!! a world thread PANICKED (see above)"),
        }
    }

    let wall = wall0.elapsed().as_secs_f32();
    println!("\n[spike] ===================== RESULTS =====================");
    let mut total_ups = 0.0_f32;
    let mut max_init = 0.0_f32;
    for r in &results {
        println!(
            "[spike] world {:>2}  sys={:<15}  init={:>6.2}s  updates={:>6}  {:>8.1} upd/s",
            r.idx, r.system, r.init_secs, r.updates, r.updates_per_sec
        );
        total_ups += r.updates_per_sec;
        max_init = max_init.max(r.init_secs);
    }
    println!("[spike] -----------------------------------------------------");
    println!(
        "[spike] worlds_ok={}/{}  slowest_init={:.2}s  aggregate={:.1} upd/s  wall={:.1}s",
        results.len(),
        n,
        max_init,
        total_ups,
        wall
    );
    println!(
        "[spike] (compare aggregate upd/s across N=1 vs N=2.. to gauge thread scaling / pool contention)"
    );
}

fn world_thread(idx: usize, system: String, training: TrainingConfig) -> WorldResult {
    let t_init = Instant::now();
    // Classic mode (rule-based AI, no trainer, no net). disable_log=true so the
    // worlds don't fight over the process-global tracing subscriber.
    let mut app = crate::build_app(AppMode::Classic, true, true, training, &system, true, None);
    // `App::run()` normally does finish()+cleanup() (which initialises the render
    // device etc.); since we drive update() manually we must do it ourselves.
    app.finish();
    app.cleanup();
    // Pump a handful of updates so Startup systems, async asset loading, and the
    // transition into Flying (ship spawning) settle before we start timing.
    for _ in 0..10 {
        app.update();
    }
    let init_secs = t_init.elapsed().as_secs_f32();
    println!("[spike] world {idx} ({system}) initialised in {init_secs:.2}s");

    let t_run = Instant::now();
    let mut updates = 0u64;
    while t_run.elapsed() < Duration::from_secs(RUN_SECS) {
        app.update();
        updates += 1;
    }
    let secs = t_run.elapsed().as_secs_f32();
    WorldResult {
        idx,
        system,
        init_secs,
        updates,
        updates_per_sec: updates as f32 / secs,
    }
}
