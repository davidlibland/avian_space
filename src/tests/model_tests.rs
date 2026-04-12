use super::*;
use crate::rl_obs::{
    self, AsteroidSlotData, CoreSlotData, EntityKind, EntitySlotData, ObsInput, PlanetSlotData,
    ShipSlotData, CORE_BLOCK_START, CORE_FEAT_SIZE, N_ENTITY_SLOTS, OBS_DIM, SLOT_IS_PRESENT,
    SLOT_TYPE_ONEHOT, TYPE_BLOCK_SIZE, TYPE_BLOCK_START, TYPE_ONEHOT_SIZE,
    SLOT_IS_NAV_TARGET, SLOT_IS_WEAPONS_TARGET,
};
use crate::ship::{Personality, Ship, ShipData};
use std::collections::HashMap;

// ── split_obs ─────────────────────────────────────────────────────────────

#[test]
fn test_split_obs_sizes() {
    let obs = vec![0.0_f32; OBS_DIM];
    let (s, o) = split_obs(&obs);
    assert_eq!(s.len(), SELF_INPUT_DIM);
    assert_eq!(o.len(), N_OBJECTS * OBJECT_INPUT_DIM);
}

#[test]
fn test_split_obs_self_features() {
    let mut obs = vec![0.0_f32; OBS_DIM];
    for (i, v) in obs[0..SELF_SIZE].iter_mut().enumerate() {
        *v = (i + 1) as f32 * 0.1;
    }
    let (s, _) = split_obs(&obs);
    assert_eq!(s, &obs[0..SELF_SIZE]);
}

#[test]
fn test_split_obs_entity_features() {
    let mut obs = vec![0.0_f32; OBS_DIM];
    // Write a distinctive pattern into slot 0.
    for (i, v) in obs[SELF_SIZE..SELF_SIZE + SLOT_SIZE].iter_mut().enumerate() {
        *v = (i + 1) as f32 * 0.5;
    }
    let (_, o) = split_obs(&obs);
    assert_eq!(
        &o[0..SLOT_SIZE],
        &obs[SELF_SIZE..SELF_SIZE + SLOT_SIZE],
    );
}

// ── Block extraction (tensor-level) ──────────────────────────────────────

/// Verify that `split_obj_feat` (used by forward()) extracts the 4 blocks
/// at the correct positions and sizes from the flat observation layout.
#[test]
fn test_obj_feat_block_extraction() {
    // Fill a dummy obs with a distinctive per-float pattern so we can verify
    // that the tensor slicing pulls the right values.
    let mut obs = vec![0.0_f32; OBS_DIM];
    // Fill entity slots with i+1 pattern (skip self-state).
    for i in 0..(N_OBJECTS * OBJECT_INPUT_DIM) {
        obs[SELF_SIZE + i] = (i + 1) as f32;
    }

    let (_, obj_slice) = split_obs(&obs);
    let device: <InferBackend as burn::prelude::Backend>::Device = Default::default();
    let obj = Tensor::<InferBackend, 3>::from_data(
        TensorData::new(obj_slice.to_vec(), [1, N_OBJECTS, OBJECT_INPUT_DIM]),
        &device,
    );

    // Use the same function that forward() calls.
    let (type_oh, is_pres, core, tspec) = split_obj_feat(obj);

    let to_vec3 = |t: Tensor<InferBackend, 3>| -> Vec<f32> {
        t.into_data().into_vec::<f32>().unwrap()
    };
    let to_vec2 = |t: Tensor<InferBackend, 2>| -> Vec<f32> {
        t.into_data().into_vec::<f32>().unwrap()
    };

    // Verify slot 0's type_onehot block (first 4 floats of slot 0).
    let th = to_vec3(type_oh);
    for i in 0..TYPE_ONEHOT_SIZE {
        let expected = (i + 1) as f32; // 1-indexed pattern
        assert_eq!(th[i], expected, "type_onehot[{i}]");
    }

    // Verify slot 0's is_present (float at position TYPE_ONEHOT_SIZE in slot).
    let ip = to_vec2(is_pres);
    assert_eq!(ip[0], (SLOT_IS_PRESENT + 1) as f32, "is_present");

    // Verify slot 0's core features start at the right offset.
    let cf = to_vec3(core);
    for i in 0..CORE_FEAT_SIZE {
        let expected = (CORE_BLOCK_START + i + 1) as f32;
        assert_eq!(cf[i], expected, "core_feat[{i}]");
    }

    // Verify slot 0's type-specific block.
    let ts = to_vec3(tspec);
    for i in 0..TYPE_BLOCK_SIZE {
        let expected = (TYPE_BLOCK_START + i + 1) as f32;
        assert_eq!(ts[i], expected, "type_feat[{i}]");
    }
}

// ── Round-trip: EntitySlotData → encode_slot → split_obj_feat ────────────

fn dummy_ship() -> Ship {
    Ship {
        ship_type: "test".to_string(),
        data: ShipData {
            max_health: 100,
            max_speed: 200.0,
            cargo_space: 50,
            thrust: 100.0,
            torque: 10.0,
            ..Default::default()
        },
        health: 50,
        cargo: HashMap::new(),
        ..Default::default()
    }
}

/// Full round-trip: construct EntitySlotData structs of each type, encode them
/// via `encode_observation`, split the resulting flat obs via `split_obs` +
/// `split_obj_feat`, and verify each block contains the expected values.
#[test]
fn test_roundtrip_entity_slot_to_blocks() {
    let ship = dummy_ship();

    let hostile_ship = EntitySlotData {
        core: CoreSlotData {
            rel_pos: [300.0, 100.0],
            rel_vel: [0.0, 0.0],
            entity_type: 0, // Ship
        },
        kind: EntityKind::Ship(ShipSlotData {
            max_health: 100.0,
            health: 80.0,
            max_speed: 250.0,
            torque: 15.0,
            is_hostile: 1.0,
            should_engage: 1.0,
            personality: Personality::Fighter,
            distressed: 0.0,
        }),
        value: 0.0,
        is_nav_target: true,
        is_weapons_target: false,
    };

    let planet = EntitySlotData {
        core: CoreSlotData {
            rel_pos: [800.0, 0.0],
            rel_vel: [0.0, 0.0],
            entity_type: 2, // Planet
        },
        kind: EntityKind::Planet(PlanetSlotData {
            cargo_sale_value: 1234.0,
            has_ammo: 1.0,
            commodity_margin: -50.0,
        }),
        value: 1234.0,
        is_nav_target: false,
        is_weapons_target: false,
    };

    let asteroid = EntitySlotData {
        core: CoreSlotData {
            rel_pos: [200.0, -50.0],
            rel_vel: [5.0, -3.0],
            entity_type: 1, // Asteroid
        },
        kind: EntityKind::Asteroid(AsteroidSlotData {
            size: 25.0,
            value: 99.0,
            collision_indicator: 0.0,
        }),
        value: 99.0,
        is_nav_target: false,
        is_weapons_target: false,
    };

    // Encode via the full observation pipeline.
    let obs = rl_obs::encode_observation(&ObsInput {
        personality: &ship.data.personality,
        ship: &ship,
        velocity: [50.0, 0.0],
        angular_velocity: 0.0,
        ship_heading: [1.0, 0.0],
        entity_slots: vec![hostile_ship, planet, asteroid],
        primary_weapon_speed: 400.0,
        primary_weapon_range: 800.0,
        credit_scale: 1000.0,
        distressed: 0.0,
        other_projectile_slots: vec![],
        own_projectile_slots: vec![],
    });
    assert_eq!(obs.len(), OBS_DIM);

    // Split into self + obj, then split obj into 4 blocks.
    let (_, obj_slice) = split_obs(&obs);
    let device: <InferBackend as burn::prelude::Backend>::Device = Default::default();
    let obj = Tensor::<InferBackend, 3>::from_data(
        TensorData::new(obj_slice.to_vec(), [1, N_OBJECTS, OBJECT_INPUT_DIM]),
        &device,
    );
    let (type_onehot, is_present, core_feat, type_feat) = split_obj_feat(obj);

    let oh: Vec<f32> = type_onehot.into_data().into_vec().unwrap();
    let ip: Vec<f32> = is_present.into_data().into_vec().unwrap();
    let cf: Vec<f32> = core_feat.into_data().into_vec().unwrap();
    let tf: Vec<f32> = type_feat.into_data().into_vec().unwrap();

    let n = N_OBJECTS;
    let t = TYPE_ONEHOT_SIZE;
    let c = CORE_FEAT_SIZE;
    let s = TYPE_BLOCK_SIZE;

    // ── Slot 0: hostile ship ────────────────────────────────────────────
    // type_onehot: Ship = [1, 0, 0, 0]
    assert_eq!(&oh[0..t], &[1.0, 0.0, 0.0, 0.0], "slot0 type_onehot");
    // is_present: 1.0
    assert_eq!(ip[0], 1.0, "slot0 is_present");
    // core_feat: rel_pos(300, 100), rel_vel(0, 0), is_nav_target(1), is_weapons_target(0), proximity, ...
    assert_eq!(cf[0 * c], 300.0, "slot0 rel_pos_x");
    assert_eq!(cf[0 * c + 1], 100.0, "slot0 rel_pos_y");
    assert_eq!(cf[0 * c + 2], 0.0, "slot0 rel_vel_x");
    assert_eq!(cf[0 * c + 3], 0.0, "slot0 rel_vel_y");
    assert_eq!(cf[0 * c + 4], 1.0, "slot0 is_nav_target");
    assert_eq!(cf[0 * c + 5], 0.0, "slot0 is_weapons_target");
    // proximity > 0
    assert!(cf[0 * c + 6] > 0.0, "slot0 proximity should be > 0");
    // type_feat: value(0.0), max_health(100), health(80), max_speed(250),
    //            torque(15), is_hostile(1), should_engage(1), personality(0,1,0)
    assert_eq!(tf[0 * s], 0.0, "slot0 value");
    assert_eq!(tf[0 * s + 1], 100.0, "slot0 max_health");
    assert_eq!(tf[0 * s + 2], 80.0, "slot0 health");
    assert_eq!(tf[0 * s + 3], 250.0, "slot0 max_speed");
    assert_eq!(tf[0 * s + 4], 15.0, "slot0 torque");
    assert_eq!(tf[0 * s + 5], 1.0, "slot0 is_hostile");
    assert_eq!(tf[0 * s + 6], 1.0, "slot0 should_engage");
    assert_eq!(&tf[0 * s + 7..0 * s + 10], &[0.0, 1.0, 0.0], "slot0 personality (Fighter)");

    // ── Slot 1: planet ──────────────────────────────────────────────────
    assert_eq!(&oh[1 * t..1 * t + t], &[0.0, 0.0, 1.0, 0.0], "slot1 type_onehot (Planet)");
    assert_eq!(ip[1], 1.0, "slot1 is_present");
    assert_eq!(cf[1 * c], 800.0, "slot1 rel_pos_x");
    assert_eq!(cf[1 * c + 4], 0.0, "slot1 is_nav_target");
    assert_eq!(cf[1 * c + 5], 0.0, "slot1 is_weapons_target");
    assert_eq!(tf[1 * s], 1234.0, "slot1 value");
    assert_eq!(tf[1 * s + 1], 1234.0, "slot1 cargo_sale_value");
    assert_eq!(tf[1 * s + 2], 1.0, "slot1 has_ammo");
    assert_eq!(tf[1 * s + 3], -50.0, "slot1 commodity_margin");
    // Remaining type-specific should be zero-padded.
    for i in 4..s {
        assert_eq!(tf[1 * s + i], 0.0, "slot1 type_specific[{i}] padding");
    }

    // ── Slot 2: asteroid ────────────────────────────────────────────────
    assert_eq!(&oh[2 * t..2 * t + t], &[0.0, 1.0, 0.0, 0.0], "slot2 type_onehot (Asteroid)");
    assert_eq!(ip[2], 1.0, "slot2 is_present");
    assert_eq!(cf[2 * c], 200.0, "slot2 rel_pos_x");
    assert_eq!(cf[2 * c + 1], -50.0, "slot2 rel_pos_y");
    assert_eq!(cf[2 * c + 2], 5.0, "slot2 rel_vel_x");
    assert_eq!(cf[2 * c + 3], -3.0, "slot2 rel_vel_y");
    assert_eq!(tf[2 * s], 99.0, "slot2 value");
    assert_eq!(tf[2 * s + 1], 25.0, "slot2 size");
    assert_eq!(tf[2 * s + 2], 99.0, "slot2 asteroid value");

    // ── Slots 3..N: empty (padded by encoder) ───────────────────────────
    for slot_idx in 3..n {
        assert_eq!(ip[slot_idx], 0.0, "slot{slot_idx} should not be present");
        // type_onehot should be all zeros
        for j in 0..t {
            assert_eq!(oh[slot_idx * t + j], 0.0, "slot{slot_idx} onehot[{j}]");
        }
        // core and type-specific should be all zeros
        for j in 0..c {
            assert_eq!(cf[slot_idx * c + j], 0.0, "slot{slot_idx} core[{j}]");
        }
        for j in 0..s {
            assert_eq!(tf[slot_idx * s + j], 0.0, "slot{slot_idx} type[{j}]");
        }
    }
}

// ── logits_to_discrete_action ────────────────────────────────────────────

#[test]
fn test_logits_to_discrete_action_argmax() {
    // Craft logits so each head has an obvious winner.
    let action_logits = [1.0_f32, 5.0, 0.0,   0.0, 3.0,   2.0, 0.0,   0.0, 4.0];
    // Target logits: slot 3 is the winner.
    let mut nav_target_logits = vec![-1e9_f32; TARGET_OUTPUT_DIM];
    nav_target_logits[3] = 5.0;
    let mut wep_target_logits = vec![-1e9_f32; TARGET_OUTPUT_DIM];
    wep_target_logits[3] = 5.0;
    let (turn, thrust, fp, fs, nav_tgt, wep_tgt) =
        logits_to_discrete_action(&action_logits, &nav_target_logits, &wep_target_logits);
    assert_eq!(turn, 1, "turn should be straight");
    assert_eq!(thrust, 1, "thrust should be on");
    assert_eq!(fp, 0, "fire_primary should be off");
    assert_eq!(fs, 1, "fire_secondary should be on");
    assert_eq!(nav_tgt, 3, "nav target should be slot 3");
    assert_eq!(wep_tgt, 3, "weapons target should be slot 3");
}

#[test]
fn test_logits_to_discrete_action_all_valid() {
    let action_logits = [0.0_f32; POLICY_OUTPUT_DIM];
    let nav_target_logits = vec![0.0_f32; TARGET_OUTPUT_DIM];
    let wep_target_logits = vec![0.0_f32; TARGET_OUTPUT_DIM];
    let (turn, thrust, fp, fs, nav_tgt, wep_tgt) =
        logits_to_discrete_action(&action_logits, &nav_target_logits, &wep_target_logits);
    assert!(turn <= 2);
    assert!(thrust <= 1);
    assert!(fp <= 1);
    assert!(fs <= 1);
    assert!((nav_tgt as usize) < TARGET_OUTPUT_DIM);
    assert!((wep_tgt as usize) < TARGET_OUTPUT_DIM);
}

// ── RLNet forward-pass shapes ────────────────────────────────────────────

/// Helper: create zero self + obj + proj tensors for testing.
fn zero_inputs(
    batch: usize,
) -> (
    Tensor<InferBackend, 2>,
    Tensor<InferBackend, 3>,
    Tensor<InferBackend, 3>,
) {
    let device = Default::default();
    let s = Tensor::<InferBackend, 2>::zeros([batch, SELF_INPUT_DIM], &device);
    let o = Tensor::<InferBackend, 3>::zeros([batch, N_OBJECTS, OBJECT_INPUT_DIM], &device);
    let p = Tensor::<InferBackend, 3>::zeros(
        [batch, N_PROJECTILE_SLOTS, PROJ_INPUT_DIM],
        &device,
    );
    (s, o, p)
}

#[test]
fn test_rlnet_policy_output_shape() {
    let device = Default::default();
    let net = RLNet::<InferBackend>::new(&device, HIDDEN_DIM, POLICY_OUTPUT_DIM);
    let batch = 4usize;
    let (s, o, p) = zero_inputs(batch);
    let (action_out, nav_target_out, wep_target_out) = net.forward(s, o, p);
    assert_eq!(action_out.shape().dims, [batch, POLICY_OUTPUT_DIM]);
    assert_eq!(nav_target_out.shape().dims, [batch, TARGET_OUTPUT_DIM]);
    assert_eq!(wep_target_out.shape().dims, [batch, TARGET_OUTPUT_DIM]);
}

#[test]
fn test_rlnet_value_output_shape() {
    let device = Default::default();
    let net = RLNet::<InferBackend>::new(&device, HIDDEN_DIM, VALUE_OUTPUT_DIM);
    let batch = 3usize;
    let (s, o, p) = zero_inputs(batch);
    let (action_out, nav_target_out, wep_target_out) = net.forward(s, o, p);
    assert_eq!(action_out.shape().dims, [batch, VALUE_OUTPUT_DIM]);
    assert_eq!(nav_target_out.shape().dims, [batch, TARGET_OUTPUT_DIM]);
    assert_eq!(wep_target_out.shape().dims, [batch, TARGET_OUTPUT_DIM]);
}

#[test]
fn test_rlnet_output_zero_init() {
    let device = Default::default();
    let net = RLNet::<InferBackend>::new(&device, HIDDEN_DIM, POLICY_OUTPUT_DIM);
    let (s, o, p) = zero_inputs(1);
    let (action_logits, _, _) = net.forward(s, o, p);
    let logits: Vec<f32> = action_logits
        .into_data()
        .into_vec::<f32>()
        .expect("extraction failed");
    for (i, &v) in logits.iter().enumerate() {
        assert!(
            v.abs() < 1e-6,
            "logit[{i}] should be ~0 on zero input, got {v}"
        );
    }
}

#[test]
fn test_rlnet_target_logits_masked() {
    let device = Default::default();
    let net = RLNet::<InferBackend>::new(&device, HIDDEN_DIM, POLICY_OUTPUT_DIM);
    let (s, o, p) = zero_inputs(1);
    let (_, nav_target_logits, _) = net.forward(s, o, p);
    let tl: Vec<f32> = nav_target_logits
        .into_data()
        .into_vec::<f32>()
        .expect("extraction failed");
    for i in 0..N_OBJECTS {
        assert!(
            tl[i] < -1e8,
            "empty slot {i} should be masked, got {}",
            tl[i]
        );
    }
    assert!(
        tl[N_OBJECTS] > -1e8,
        "no-target logit should not be masked, got {}",
        tl[N_OBJECTS]
    );
}

// ── sample_gumbel_like & coupled_gumbel_sample ──────────────────────────

#[test]
fn test_sample_gumbel_like_shape() {
    let device: <InferBackend as burn::prelude::Backend>::Device = Default::default();
    let reference = Tensor::<InferBackend, 2>::zeros([4, 8], &device);
    let g = sample_gumbel_like(&reference);
    assert_eq!(g.shape().dims, [4, 8]);
}

#[test]
fn test_sample_gumbel_like_finite() {
    let device: <InferBackend as burn::prelude::Backend>::Device = Default::default();
    let reference = Tensor::<InferBackend, 2>::zeros([10, 20], &device);
    let g = sample_gumbel_like(&reference);
    let vals: Vec<f32> = g.into_data().into_vec().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "gumbel sample[{i}] is not finite: {v}");
    }
}

#[test]
fn test_sample_gumbel_like_not_constant() {
    let device: <InferBackend as burn::prelude::Backend>::Device = Default::default();
    let reference = Tensor::<InferBackend, 2>::zeros([1, 100], &device);
    let g = sample_gumbel_like(&reference);
    let vals: Vec<f32> = g.into_data().into_vec().unwrap();
    // At least some values should differ from the first.
    let distinct = vals.iter().filter(|&&v| (v - vals[0]).abs() > 1e-6).count();
    assert!(distinct > 0, "all gumbel samples are identical — RNG broken?");
}

#[test]
fn test_coupled_gumbel_sample_identical_logits() {
    // When logits_a == logits_b the same noise is added, so argmax must match.
    let device: <InferBackend as burn::prelude::Backend>::Device = Default::default();
    let logits = Tensor::<InferBackend, 2>::zeros([32, 10], &device);
    let (a, b) = coupled_gumbel_sample(logits.clone(), logits);
    let va: Vec<i64> = a.into_data().into_vec().unwrap();
    let vb: Vec<i64> = b.into_data().into_vec().unwrap();
    assert_eq!(va, vb, "identical logits must produce identical samples");
}

#[test]
fn test_coupled_gumbel_sample_output_shapes() {
    let device: <InferBackend as burn::prelude::Backend>::Device = Default::default();
    let logits_a = Tensor::<InferBackend, 2>::zeros([5, 8], &device);
    let logits_b = Tensor::<InferBackend, 2>::zeros([5, 8], &device);
    let (a, b) = coupled_gumbel_sample(logits_a, logits_b);
    assert_eq!(a.shape().dims, [5, 1], "argmax output should be [batch, 1]");
    assert_eq!(b.shape().dims, [5, 1]);
}

#[test]
fn test_coupled_gumbel_sample_in_range() {
    let device: <InferBackend as burn::prelude::Backend>::Device = Default::default();
    let num_classes = 7;
    let logits_a = Tensor::<InferBackend, 2>::random(
        [20, num_classes],
        Distribution::Default,
        &device,
    );
    let logits_b = Tensor::<InferBackend, 2>::random(
        [20, num_classes],
        Distribution::Default,
        &device,
    );
    let (a, b) = coupled_gumbel_sample(logits_a, logits_b);
    let va: Vec<i64> = a.into_data().into_vec().unwrap();
    let vb: Vec<i64> = b.into_data().into_vec().unwrap();
    for (i, (&ai, &bi)) in va.iter().zip(vb.iter()).enumerate() {
        assert!(ai >= 0 && ai < num_classes as i64, "a[{i}]={ai} out of range");
        assert!(bi >= 0 && bi < num_classes as i64, "b[{i}]={bi} out of range");
    }
}

#[test]
fn test_coupled_gumbel_sample_dominated_logit() {
    // One logit much larger → should almost always be selected for both.
    let device: <InferBackend as burn::prelude::Backend>::Device = Default::default();
    let mut data = vec![0.0_f32; 5];
    data[2] = 100.0; // dominant
    let logits = Tensor::<InferBackend, 2>::from_data(
        TensorData::new(data, [1, 5]),
        &device,
    );
    // Run many times and check the dominant index wins.
    for _ in 0..50 {
        let (a, b) = coupled_gumbel_sample(logits.clone(), logits.clone());
        let va: Vec<i64> = a.into_data().into_vec().unwrap();
        let vb: Vec<i64> = b.into_data().into_vec().unwrap();
        assert_eq!(va[0], 2, "dominant logit should win for a");
        assert_eq!(vb[0], 2, "dominant logit should win for b");
    }
}

// ── InferenceNet end-to-end ──────────────────────────────────────────────

#[test]
fn test_inference_net_produces_valid_action() {
    let inference = InferenceNet::new();
    let obs = vec![0.0_f32; OBS_DIM];
    let proj_obs = vec![0.0_f32; N_PROJECTILE_SLOTS * PROJ_INPUT_DIM];
    let (s, o) = split_obs(&obs);
    let (action_logits, nav_target_logits, wep_target_logits) =
        inference.run_inference(s.to_vec(), o.to_vec(), proj_obs, 1);

    assert_eq!(action_logits.len(), POLICY_OUTPUT_DIM, "wrong action logit count");
    assert_eq!(nav_target_logits.len(), TARGET_OUTPUT_DIM, "wrong nav target logit count");
    assert_eq!(wep_target_logits.len(), TARGET_OUTPUT_DIM, "wrong wep target logit count");
    let (turn, thrust, fp, fs, nav_tgt, wep_tgt) =
        logits_to_discrete_action(&action_logits, &nav_target_logits, &wep_target_logits);
    assert!(turn <= 2, "turn_idx out of range: {turn}");
    assert!(thrust <= 1, "thrust_idx out of range: {thrust}");
    assert!(fp <= 1, "fire_primary out of range: {fp}");
    assert!(fs <= 1, "fire_secondary out of range: {fs}");
    assert!((nav_tgt as usize) < TARGET_OUTPUT_DIM, "nav_target_idx out of range: {nav_tgt}");
    assert!((wep_tgt as usize) < TARGET_OUTPUT_DIM, "wep_target_idx out of range: {wep_tgt}");
}

#[test]
fn test_inference_net_batched() {
    let inference = InferenceNet::new();
    let batch_size = 5;
    let obs = vec![0.0_f32; OBS_DIM];
    let (s, o) = split_obs(&obs);
    let self_flat: Vec<f32> = s.iter().cloned().cycle().take(batch_size * s.len()).collect();
    let obj_flat: Vec<f32> = o.iter().cloned().cycle().take(batch_size * o.len()).collect();
    let proj_flat: Vec<f32> = vec![0.0; batch_size * N_PROJECTILE_SLOTS * PROJ_INPUT_DIM];
    let (action_logits, nav_target_logits, wep_target_logits) =
        inference.run_inference(self_flat, obj_flat, proj_flat, batch_size);
    assert_eq!(action_logits.len(), batch_size * POLICY_OUTPUT_DIM);
    assert_eq!(nav_target_logits.len(), batch_size * TARGET_OUTPUT_DIM);
    assert_eq!(wep_target_logits.len(), batch_size * TARGET_OUTPUT_DIM);
}
