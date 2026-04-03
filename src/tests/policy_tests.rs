//! Policy behaviour tests.
//!
//! These tests load a trained checkpoint and verify that the policy makes
//! sensible decisions for hand-crafted observations, and compute a confusion
//! matrix over the BC replay buffer.
//!
//! All tests are `#[ignore]` by default — run them with:
//!
//! ```sh
//! cargo test policy -- --ignored
//! ```

use super::*;
use crate::experiments::setup_experiment;
use crate::model::{
    self, InferenceNet, N_OBJECTS, OBJECT_INPUT_DIM, POLICY_OUTPUT_DIM, SELF_INPUT_DIM,
    TARGET_OUTPUT_DIM,
};
use crate::rl_collection::{load_bc_buffer, BCTransition};
use crate::rl_obs::*;
use crate::ship::Personality;
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Load the latest experiment checkpoint, panicking with a helpful message
/// if no checkpoint exists.
fn load_latest_checkpoint() -> InferenceNet {
    let experiment = setup_experiment(false);
    let path = experiment.policy_checkpoint_path();
    model::load_inference_net(&path).unwrap_or_else(|| {
        panic!(
            "No checkpoint found at {path}.bin — run BC training first, \
             then re-run with: cargo test policy -- --ignored"
        )
    })
}

/// Load the latest BC buffer, panicking if missing.
fn load_latest_buffer() -> VecDeque<BCTransition> {
    let experiment = setup_experiment(false);
    let path = experiment.buffer_checkpoint_path();
    load_bc_buffer(&path).unwrap_or_else(|| {
        panic!(
            "No BC buffer found at {path} — run BC training first, \
             then re-run with: cargo test policy -- --ignored"
        )
    })
}

/// Build a minimal self-state (SELF_SIZE=10 floats) for a given personality.
fn self_state(personality: &Personality, health_frac: f32) -> Vec<f32> {
    let p = match personality {
        Personality::Miner => [1.0, 0.0, 0.0],
        Personality::Fighter => [0.0, 1.0, 0.0],
        Personality::Trader => [0.0, 0.0, 1.0],
    };
    vec![
        health_frac, // health
        0.5,         // speed_frac
        1.0,         // vel_cos (facing forward)
        0.0,         // vel_sin
        0.0,         // angular velocity
        0.0,         // cargo_frac
        0.5,         // ammo_frac
        p[0],        // personality
        p[1],
        p[2],
    ]
}

/// Build a single entity slot (SLOT_SIZE=25 floats).
fn make_slot(
    entity_type: u8,
    rel_pos: [f32; 2],
    rel_vel: [f32; 2],
    is_current_target: bool,
    type_specific: &[f32],
    value: f32,
) -> Vec<f32> {
    let mut slot = Vec::with_capacity(SLOT_SIZE);
    // is_present
    slot.push(1.0);
    // type_onehot
    let onehot = match entity_type {
        0 => [1.0, 0.0, 0.0, 0.0], // Ship
        1 => [0.0, 1.0, 0.0, 0.0], // Asteroid
        2 => [0.0, 0.0, 1.0, 0.0], // Planet
        3 => [0.0, 0.0, 0.0, 1.0], // Pickup
        _ => [0.0; 4],
    };
    slot.extend_from_slice(&onehot);
    slot.extend_from_slice(&rel_pos);
    slot.extend_from_slice(&rel_vel);
    slot.push(if is_current_target { 1.0 } else { 0.0 });
    // pursuit_angle, pursuit_indicator (compute simple approximation)
    let angle = rel_pos[1].atan2(rel_pos[0]);
    slot.push(angle);
    let indicator = if angle.abs() < std::f32::consts::FRAC_PI_2 {
        1.0 - (2.0 * angle / std::f32::consts::PI).powi(2)
    } else {
        0.0
    };
    slot.push(indicator);
    // fire_angle, fire_indicator (same as pursuit for simplicity)
    slot.push(angle);
    slot.push(indicator);
    // in_range
    let dist = (rel_pos[0].powi(2) + rel_pos[1].powi(2)).sqrt();
    slot.push(if dist < 800.0 { 1.0 } else { 0.0 });
    // value
    slot.push(value);
    // type_specific (pad to TYPE_SPECIFIC_SIZE)
    slot.extend_from_slice(type_specific);
    let pad = TYPE_SPECIFIC_SIZE - type_specific.len();
    slot.extend(std::iter::repeat(0.0_f32).take(pad));
    debug_assert_eq!(slot.len(), SLOT_SIZE, "slot size mismatch");
    slot
}

/// Build an empty slot (SLOT_SIZE zeros).
fn empty_slot() -> Vec<f32> {
    vec![0.0_f32; SLOT_SIZE]
}

/// Build a full flat observation from self-state and entity slots.
fn build_obs(self_state: &[f32], slots: &[Vec<f32>]) -> Vec<f32> {
    assert_eq!(self_state.len(), SELF_SIZE);
    assert_eq!(slots.len(), N_ENTITY_SLOTS);
    let mut obs = Vec::with_capacity(OBS_DIM);
    obs.extend_from_slice(self_state);
    for slot in slots {
        assert_eq!(slot.len(), SLOT_SIZE);
        obs.extend_from_slice(slot);
    }
    assert_eq!(obs.len(), OBS_DIM);
    obs
}

/// Run inference on a single observation, returning (action, target_idx).
fn infer_one(net: &InferenceNet, obs: &[f32]) -> (DiscreteAction, Vec<f32>, Vec<f32>) {
    let (s, o) = model::split_obs(obs);
    let proj = vec![0.0_f32; model::PROJECTILES_FLAT_DIM];
    let (action_logits, target_logits) =
        net.run_inference(s.to_vec(), o.to_vec(), proj, 1);
    let action = model::logits_to_discrete_action(&action_logits, &target_logits);
    (action, action_logits, target_logits)
}

/// Slot indices for each bucket.
const PLANET_SLOTS: std::ops::Range<usize> = 0..2;
const ASTEROID_SLOTS: std::ops::Range<usize> = 2..5;
const HOSTILE_SHIP_SLOTS: std::ops::Range<usize> = 5..8;
const FRIENDLY_SHIP_SLOTS: std::ops::Range<usize> = 8..10;
const PICKUP_SLOTS: std::ops::Range<usize> = 10..12;

// ---------------------------------------------------------------------------
// Hand-crafted scenario tests
// ---------------------------------------------------------------------------

/// A fighter with one hostile ship nearby should target it.
#[test]
#[ignore]
fn test_fighter_targets_hostile_ship() {
    let net = load_latest_checkpoint();
    let self_feat = self_state(&Personality::Fighter, 1.0);

    // Ship-type-specific: max_health, health, max_speed, torque, is_hostile, should_engage, personality(3)
    let hostile_ship_type_specific = vec![
        100.0, // max_health
        100.0, // health
        200.0, // max_speed
        10.0,  // torque
        1.0,   // is_hostile (they shoot at us)
        1.0,   // should_engage (we should shoot at them)
        0.0, 1.0, 0.0, // personality: Fighter
    ];

    let hostile_ship = make_slot(
        0,                                // Ship
        [500.0, 0.0],                     // directly ahead
        [0.0, 0.0],                       // stationary
        false,                            // not current target
        &hostile_ship_type_specific[..9], // truncate to TYPE_SPECIFIC_SIZE
        0.0,                              // ships don't have value
    );

    // Build obs: no target, no planets/asteroids/friendlies/pickups,
    // one hostile ship in slot 6.
    let mut slots = vec![empty_slot(); N_ENTITY_SLOTS];
    slots[HOSTILE_SHIP_SLOTS.start] = hostile_ship;

    let obs = build_obs(&self_feat, &slots);
    let (action, _action_logits, target_logits) = infer_one(&net, &obs);

    let target_idx = action.4 as usize;
    println!("Target logits: {:?}", target_logits);
    println!("Chosen target_idx: {target_idx}");
    println!(
        "Action: turn={}, thrust={}, fire_primary={}, fire_secondary={}",
        action.0, action.1, action.2, action.3
    );

    assert_eq!(
        target_idx, HOSTILE_SHIP_SLOTS.start,
        "Fighter should target the hostile ship (slot {}), but chose slot {target_idx}",
        HOSTILE_SHIP_SLOTS.start
    );
}

/// A miner with an asteroid nearby should target it.
#[test]
#[ignore]
fn test_miner_targets_asteroid() {
    let net = load_latest_checkpoint();
    let self_feat = self_state(&Personality::Miner, 1.0);

    let asteroid = make_slot(
        1,                // Asteroid
        [300.0, 100.0],   // nearby
        [0.0, 0.0],       // stationary
        false,
        &[20.0, 50.0],    // size=20, value=50
        50.0,
    );

    let mut slots = vec![empty_slot(); N_ENTITY_SLOTS];
    slots[ASTEROID_SLOTS.start] = asteroid;

    let obs = build_obs(&self_feat, &slots);
    let (action, _action_logits, target_logits) = infer_one(&net, &obs);

    let target_idx = action.4 as usize;
    println!("Target logits: {:?}", target_logits);
    println!("Chosen target_idx: {target_idx}");

    assert_eq!(
        target_idx, ASTEROID_SLOTS.start,
        "Miner should target the asteroid (slot {}), but chose slot {target_idx}",
        ASTEROID_SLOTS.start
    );
}

/// A trader with cargo and a planet nearby should target the planet.
#[test]
#[ignore]
fn test_trader_targets_planet() {
    let net = load_latest_checkpoint();
    let mut self_feat = self_state(&Personality::Trader, 1.0);
    self_feat[5] = 0.8; // cargo_frac = 80% full

    let planet = make_slot(
        2,                       // Planet
        [800.0, 0.0],            // ahead
        [0.0, 0.0],              // stationary
        false,
        &[500.0, 0.0, -100.0],  // cargo_sale_value=500, has_ammo=0, commodity_margin=-100
        500.0,
    );

    let mut slots = vec![empty_slot(); N_ENTITY_SLOTS];
    slots[PLANET_SLOTS.start] = planet;

    let obs = build_obs(&self_feat, &slots);
    let (action, _action_logits, target_logits) = infer_one(&net, &obs);

    let target_idx = action.4 as usize;
    println!("Target logits: {:?}", target_logits);
    println!("Chosen target_idx: {target_idx}");

    assert_eq!(
        target_idx, PLANET_SLOTS.start,
        "Trader should target the planet (slot {}), but chose slot {target_idx}",
        PLANET_SLOTS.start
    );
}

// ---------------------------------------------------------------------------
// BC Confusion Matrix
// ---------------------------------------------------------------------------

/// Print a confusion matrix for each action head and target selection.
///
/// Compares the trained policy's predictions against the rule-based BC labels.
#[test]
#[ignore]
fn test_bc_confusion_matrix() {
    let net = load_latest_checkpoint();
    let buffer = load_latest_buffer();

    println!("\n=== BC Confusion Matrix ===");
    println!("Buffer size: {} transitions\n", buffer.len());

    // Accumulators: [predicted][label] for each head.
    let mut turn_cm = [[0u32; 3]; 3];           // 3×3
    let mut thrust_cm = [[0u32; 2]; 2];          // 2×2
    let mut fire_primary_cm = [[0u32; 2]; 2];    // 2×2
    let mut fire_secondary_cm = [[0u32; 2]; 2];  // 2×2
    let mut target_cm = vec![vec![0u32; TARGET_OUTPUT_DIM]; TARGET_OUTPUT_DIM];

    let mut total = 0usize;

    // Batch for efficiency.
    let batch_size = 256;
    let transitions: Vec<&BCTransition> = buffer.iter().collect();

    for chunk in transitions.chunks(batch_size) {
        let bs = chunk.len();
        let mut self_flat = Vec::with_capacity(bs * SELF_INPUT_DIM);
        let mut obj_flat = Vec::with_capacity(bs * N_OBJECTS * OBJECT_INPUT_DIM);
        for t in chunk {
            let (s, o) = model::split_obs(&t.obs);
            self_flat.extend_from_slice(s);
            obj_flat.extend_from_slice(o);
        }
        let proj_flat = vec![0.0_f32; bs * model::PROJECTILES_FLAT_DIM];
        let (action_logits, target_logits) =
            net.run_inference(self_flat, obj_flat, proj_flat, bs);

        for (i, t) in chunk.iter().enumerate() {
            let al = &action_logits[i * POLICY_OUTPUT_DIM..(i + 1) * POLICY_OUTPUT_DIM];
            let tl = &target_logits[i * TARGET_OUTPUT_DIM..(i + 1) * TARGET_OUTPUT_DIM];
            let pred = model::logits_to_discrete_action(al, tl);
            let label = t.action;

            turn_cm[pred.0 as usize][label.0 as usize] += 1;
            thrust_cm[pred.1 as usize][label.1 as usize] += 1;
            fire_primary_cm[pred.2 as usize][label.2 as usize] += 1;
            fire_secondary_cm[pred.3 as usize][label.3 as usize] += 1;
            let pred_t = (pred.4 as usize).min(TARGET_OUTPUT_DIM - 1);
            let label_t = (label.4 as usize).min(TARGET_OUTPUT_DIM - 1);
            target_cm[pred_t][label_t] += 1;
            total += 1;
        }
    }

    // Print results.
    println!("Total transitions evaluated: {total}\n");

    print_confusion_matrix("Turn (0=left, 1=straight, 2=right)", &turn_cm);
    print_confusion_matrix("Thrust (0=off, 1=on)", &thrust_cm);
    print_confusion_matrix("Fire Primary (0=no, 1=yes)", &fire_primary_cm);
    print_confusion_matrix("Fire Secondary (0=no, 1=yes)", &fire_secondary_cm);
    print_confusion_matrix_dyn("Target Selection", &target_cm);

    // Overall accuracy per head.
    let turn_acc = accuracy_2d(&turn_cm);
    let thrust_acc = accuracy_2d(&thrust_cm);
    let fp_acc = accuracy_2d(&fire_primary_cm);
    let fs_acc = accuracy_2d(&fire_secondary_cm);
    let target_acc = accuracy_dyn(&target_cm);

    println!("\n=== Accuracy Summary ===");
    println!("Turn:           {turn_acc:.1}%");
    println!("Thrust:         {thrust_acc:.1}%");
    println!("Fire Primary:   {fp_acc:.1}%");
    println!("Fire Secondary: {fs_acc:.1}%");
    println!("Target:         {target_acc:.1}%");

    // Sanity check: accuracy should be above chance.
    assert!(
        turn_acc > 40.0,
        "Turn accuracy {turn_acc:.1}% is below 40% — model may not be trained"
    );
    assert!(
        thrust_acc > 55.0,
        "Thrust accuracy {thrust_acc:.1}% is below 55% — model may not be trained"
    );
}

// ---------------------------------------------------------------------------
// Printing helpers
// ---------------------------------------------------------------------------

fn print_confusion_matrix<const N: usize>(name: &str, cm: &[[u32; N]; N]) {
    println!("── {name} ──");
    print!("{:>12}", "pred\\label");
    for j in 0..N {
        print!("{j:>8}");
    }
    println!();
    for i in 0..N {
        print!("{i:>12}");
        for j in 0..N {
            print!("{:>8}", cm[i][j]);
        }
        println!();
    }
    println!();
}

fn print_confusion_matrix_dyn(name: &str, cm: &[Vec<u32>]) {
    let n = cm.len();
    // Only print rows/cols that have any counts to keep output manageable.
    let active: Vec<usize> = (0..n)
        .filter(|&i| {
            cm[i].iter().any(|&v| v > 0) || (0..n).any(|j| cm[j][i] > 0)
        })
        .collect();
    if active.is_empty() {
        println!("── {name} ── (no data)");
        return;
    }
    println!("── {name} ──");
    print!("{:>12}", "pred\\label");
    for &j in &active {
        let label = if j == n - 1 {
            "none".to_string()
        } else {
            format!("s{j}")
        };
        print!("{label:>8}");
    }
    println!();
    for &i in &active {
        let label = if i == n - 1 {
            "none".to_string()
        } else {
            format!("s{i}")
        };
        print!("{label:>12}");
        for &j in &active {
            print!("{:>8}", cm[i][j]);
        }
        println!();
    }
    println!();
}

fn accuracy_2d<const N: usize>(cm: &[[u32; N]; N]) -> f64 {
    let correct: u32 = (0..N).map(|i| cm[i][i]).sum();
    let total: u32 = cm.iter().flat_map(|row| row.iter()).sum();
    if total == 0 {
        return 0.0;
    }
    100.0 * correct as f64 / total as f64
}

fn accuracy_dyn(cm: &[Vec<u32>]) -> f64 {
    let n = cm.len();
    let correct: u32 = (0..n).map(|i| cm[i][i]).sum();
    let total: u32 = cm.iter().flat_map(|row| row.iter()).sum();
    if total == 0 {
        return 0.0;
    }
    100.0 * correct as f64 / total as f64
}

// ---------------------------------------------------------------------------
// BC target-selection data quality diagnostics
// ---------------------------------------------------------------------------

/// Find which slot (if any) has `is_current_target=1` in the observation.
fn find_is_current_target_slot(obs: &[f32]) -> Option<usize> {
    for i in 0..N_ENTITY_SLOTS {
        let slot_start = SELF_SIZE + i * SLOT_SIZE;
        if obs[slot_start + SLOT_IS_CURRENT_TARGET] > 0.5 {
            return Some(i);
        }
    }
    None
}

/// Check that the BC buffer contains non-trivial target selection training data.
///
/// Specifically, verifies that there exist transitions where:
/// 1. The output target label points to an entity that is NOT already marked
///    `is_current_target` in the input features (i.e. the AI chose a NEW target).
/// 2. The output target label is "no target" while an entity IS marked
///    `is_current_target` in the input (i.e. the AI dropped its target).
///
/// If neither case exists, the model can only learn to echo the `is_current_target`
/// flag rather than learning to actually select targets.
#[test]
#[ignore]
fn test_bc_buffer_has_nontrivial_target_data() {
    let buffer = load_latest_buffer();
    println!("\n=== BC Target Selection Data Quality ===");
    println!("Buffer size: {} transitions\n", buffer.len());

    let no_target_label = N_OBJECTS as u8; // "no target" index

    let mut total = 0usize;
    let mut target_matches_is_current = 0usize; // label == is_current_target slot
    let mut target_is_new = 0usize;             // label != is_current_target slot (new target chosen)
    let mut target_dropped = 0usize;            // label = "no target" but is_current_target exists
    let mut no_target_no_current = 0usize;      // label = "no target" and no is_current_target
    let mut label_is_none = 0usize;
    let mut label_is_entity = 0usize;
    let mut has_current_target = 0usize;

    for t in buffer.iter() {
        total += 1;
        let label_target = t.action.4;
        let current_target_slot = find_is_current_target_slot(&t.obs);

        if current_target_slot.is_some() {
            has_current_target += 1;
        }

        if label_target == no_target_label {
            label_is_none += 1;
            if current_target_slot.is_some() {
                target_dropped += 1;
            } else {
                no_target_no_current += 1;
            }
        } else {
            label_is_entity += 1;
            if current_target_slot == Some(label_target as usize) {
                target_matches_is_current += 1;
            } else {
                target_is_new += 1;
            }
        }
    }

    println!("Total transitions:                     {total}");
    println!("  Has is_current_target in input:       {has_current_target}");
    println!("  Label is entity:                      {label_is_entity}");
    println!("    ...and matches is_current_target:   {target_matches_is_current}");
    println!("    ...and is NEW target (non-trivial): {target_is_new}");
    println!("  Label is 'no target':                 {label_is_none}");
    println!("    ...while is_current_target exists:  {target_dropped}");
    println!("    ...and no is_current_target:        {no_target_no_current}");
    println!();

    let pct_new = if total > 0 {
        100.0 * target_is_new as f64 / total as f64
    } else {
        0.0
    };
    let pct_dropped = if total > 0 {
        100.0 * target_dropped as f64 / total as f64
    } else {
        0.0
    };
    println!("New-target transitions:  {target_is_new} ({pct_new:.2}%)");
    println!("Drop-target transitions: {target_dropped} ({pct_dropped:.2}%)");

    assert!(
        target_is_new > 0,
        "No transitions where the BC label selects a NEW target \
         (different from is_current_target). The model cannot learn \
         target selection from this data."
    );
    assert!(
        target_dropped > 0 || label_is_none == 0,
        "No transitions where the BC label drops the target while \
         is_current_target is set. If 'no target' labels exist, \
         some should involve dropping an existing target."
    );
}

/// Complementary test: verify that transitions exist where there IS no
/// `is_current_target` in the input but the label picks a target.
/// This is the "first target acquisition" case.
#[test]
#[ignore]
fn test_bc_buffer_has_target_acquisition_data() {
    let buffer = load_latest_buffer();
    let no_target_label = N_OBJECTS as u8;

    let mut no_current_but_label_picks = 0usize;
    let mut no_current_total = 0usize;

    for t in buffer.iter() {
        let current_target_slot = find_is_current_target_slot(&t.obs);
        if current_target_slot.is_none() {
            no_current_total += 1;
            if t.action.4 != no_target_label {
                no_current_but_label_picks += 1;
            }
        }
    }

    println!("\n=== Target Acquisition Data ===");
    println!("Transitions with no is_current_target: {no_current_total}");
    println!("  ...where label picks a target:       {no_current_but_label_picks}");

    assert!(
        no_current_but_label_picks > 0,
        "No transitions where the ship has no current target but the BC label \
         picks one. The model cannot learn initial target acquisition."
    );
}
