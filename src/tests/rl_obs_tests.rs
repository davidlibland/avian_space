use super::*;
use crate::ship::{Ship, ShipData};
use std::collections::HashMap;

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
        cargo: {
            let mut m = HashMap::new();
            m.insert("ore".to_string(), 10u16);
            m
        },
        ..Default::default()
    }
}

fn minimal_obs_input(ship: &Ship) -> ObsInput<'_> {
    ObsInput {
        personality: &ship.data.personality,
        ship,
        velocity: [0.0, 0.0],
        angular_velocity: 0.0,
        ship_heading: [0.0, 1.0], // facing up
        entity_slots: vec![],
        primary_weapon_speed: 400.0,
        primary_weapon_range: 800.0,
        credit_scale: 1000.0,
    }
}

#[test]
fn test_observation_shape_constant() {
    let ship = dummy_ship();

    // No nearby entities.
    let obs0 = encode_observation(&minimal_obs_input(&ship));
    assert_eq!(obs0.len(), OBS_DIM, "zero entities");

    // Some entities populated.
    let slot = EntitySlotData {
        core: CoreSlotData {
            rel_pos: [0.1, 0.2],
            rel_vel: [0.0, 0.0],
            entity_type: 0,
        },
        kind: EntityKind::Ship(ShipSlotData::default()),
        value: 1.0,
        is_current_target: false,
    };
    let mut target_slot = slot.clone();
    target_slot.is_current_target = true;
    let some_slots = vec![slot.clone(), target_slot];
    let obs1 = encode_observation(&ObsInput {
        personality: &ship.data.personality,
        ship: &ship,
        velocity: [10.0, 0.0],
        angular_velocity: 0.5,
        ship_heading: [1.0, 0.0],
        entity_slots: some_slots,
        primary_weapon_speed: 400.0,
        primary_weapon_range: 800.0,
        credit_scale: 1000.0,
    });
    assert_eq!(obs1.len(), OBS_DIM, "some entities");

    // Fully-populated.
    let obs2 = encode_observation(&ObsInput {
        personality: &ship.data.personality,
        ship: &ship,
        velocity: [0.0, 0.0],
        angular_velocity: 0.0,
        ship_heading: [0.0, 1.0],
        entity_slots: vec![slot.clone(); N_ENTITY_SLOTS],
        primary_weapon_speed: 400.0,
        primary_weapon_range: 800.0,
        credit_scale: 1000.0,
    });
    assert_eq!(obs2.len(), OBS_DIM, "full buckets");
}

#[test]
fn test_ego_centric_encoding() {
    let (sin_a, cos_a) = ego_frame_sincos([1.0_f32, 0.0]);
    let world_offset = [100.0_f32, 0.0];
    let ego = rotate_to_ego(world_offset, sin_a, cos_a);
    assert!((ego[0] - 100.0).abs() < 1e-4, "ego x should be ~100, got {}", ego[0]);
    assert!(ego[1].abs() < 1e-4, "ego y should be ~0, got {}", ego[1]);

    let (sin_a, cos_a) = ego_frame_sincos([0.0_f32, 1.0]);
    let world_offset = [0.0_f32, 100.0];
    let ego = rotate_to_ego(world_offset, sin_a, cos_a);
    assert!((ego[0] - 100.0).abs() < 1e-4, "ego x should be ~100, got {}", ego[0]);
    assert!(ego[1].abs() < 1e-4, "ego y should be ~0, got {}", ego[1]);
}

#[test]
fn test_action_to_ship_command() {
    let (thrust, turn) = discrete_to_controls(1, 0);
    assert_eq!(thrust, 1.0);
    assert_eq!(turn, -1.0);

    let (thrust, turn) = discrete_to_controls(0, 2);
    assert_eq!(thrust, 0.0);
    assert_eq!(turn, 1.0);

    let (thrust, turn) = discrete_to_controls(1, 1);
    assert_eq!(thrust, 1.0);
    assert_eq!(turn, 0.0);
}

#[test]
fn test_angle_to_discrete() {
    use std::f32::consts::PI;

    let (turn, thrust) = angle_to_discrete(PI / 2.0);
    assert_eq!(turn, 0);
    assert_eq!(thrust, 0);

    let (turn, thrust) = angle_to_discrete(0.1);
    assert_eq!(turn, 0);
    assert_eq!(thrust, 1);

    let (turn, thrust) = angle_to_discrete(-0.1);
    assert_eq!(turn, 2);
    assert_eq!(thrust, 1);

    let (turn, thrust) = angle_to_discrete(-PI / 2.0);
    assert_eq!(turn, 2);
    assert_eq!(thrust, 0);
}

#[test]
fn test_controls_to_discrete_roundtrip() {
    for thrust_idx in 0u8..=1 {
        for turn_idx in [0u8, 1, 2] {
            let (t, r) = discrete_to_controls(thrust_idx, turn_idx);
            let (t2, r2) = controls_to_discrete(t, r);
            assert_eq!(thrust_idx, t2, "thrust mismatch for ({}, {})", thrust_idx, turn_idx);
            assert_eq!(turn_idx, r2, "turn mismatch for ({}, {})", thrust_idx, turn_idx);
        }
    }
}

#[test]
fn test_personality_onehot() {
    assert_eq!(personality_onehot(&Personality::Miner), [1.0, 0.0, 0.0]);
    assert_eq!(personality_onehot(&Personality::Fighter), [0.0, 1.0, 0.0]);
    assert_eq!(personality_onehot(&Personality::Trader), [0.0, 0.0, 1.0]);
}

#[test]
fn test_action_bc_roundtrip() {
    let cases: &[(f32, f32)] = &[
        (1.0, -1.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, -1.0),
        (0.0, 0.0),
        (0.0, 1.0),
    ];
    for &(thrust, turn) in cases {
        let (thrust_idx, turn_idx) = controls_to_discrete(thrust, turn);
        let action: DiscreteAction = (turn_idx, thrust_idx, 0, 0, 0);
        let (turn_out, thrust_out, _, _, _) = action;
        let (t_back, r_back) = discrete_to_controls(thrust_out, turn_out);
        assert_eq!(t_back, thrust, "thrust mismatch for input ({thrust}, {turn}): got {t_back}");
        assert_eq!(r_back, turn, "turn mismatch for input ({thrust}, {turn}): got {r_back}");
    }
}

// ── intercept_angle / aim_indicator ─────────────────────────────────────

#[test]
fn test_intercept_angle_directly_ahead() {
    let angle = intercept_angle(200.0, [500.0, 0.0], [0.0, 0.0]).unwrap();
    assert!(angle.abs() < 1e-4, "expected ~0, got {}", angle);
}

#[test]
fn test_intercept_angle_left() {
    let angle = intercept_angle(200.0, [0.0, 500.0], [0.0, 0.0]).unwrap();
    assert!((angle - PI / 2.0).abs() < 1e-4, "expected ~π/2, got {}", angle);
}

#[test]
fn test_intercept_angle_right() {
    let angle = intercept_angle(200.0, [0.0, -500.0], [0.0, 0.0]).unwrap();
    assert!((angle + PI / 2.0).abs() < 1e-4, "expected ~−π/2, got {}", angle);
}

#[test]
fn test_intercept_angle_always_in_range() {
    let cases: &[([f32; 2], [f32; 2])] = &[
        ([100.0, 0.0], [0.0, 0.0]),
        ([0.0, 100.0], [0.0, 0.0]),
        ([-100.0, 0.0], [0.0, 0.0]),
        ([100.0, 100.0], [50.0, -50.0]),
        ([-200.0, 150.0], [-30.0, 20.0]),
        ([50.0, -300.0], [10.0, -5.0]),
    ];
    for &(pos, vel) in cases {
        if let Some(a) = intercept_angle(200.0, pos, vel) {
            assert!(a >= -PI && a <= PI, "angle {} out of [-π, π] for pos={:?} vel={:?}", a, pos, vel);
        }
    }
}

#[test]
fn test_aim_indicator_perfect() {
    let ind = aim_indicator(Some(0.0));
    assert!((ind - 1.0).abs() < 1e-5, "expected 1.0, got {}", ind);
}

#[test]
fn test_aim_indicator_perpendicular() {
    assert!(aim_indicator(Some(PI / 2.0)) < 1e-10);
    assert!(aim_indicator(Some(-PI / 2.0)) < 1e-10);
    assert_eq!(aim_indicator(Some(PI)), 0.0);
    assert_eq!(aim_indicator(Some(-PI)), 0.0);
    assert_eq!(aim_indicator(Some(PI * 0.75)), 0.0);
}

#[test]
fn test_aim_indicator_none() {
    assert_eq!(aim_indicator(None), 0.0);
}

#[test]
fn test_aim_indicator_partial() {
    let ind = aim_indicator(Some(PI / 4.0));
    assert!(ind > 0.0 && ind < 1.0, "expected (0,1), got {}", ind);
}

#[test]
fn test_aim_indicator_backwards_wrapping() {
    let ind = aim_indicator(Some(PI - 0.01));
    assert_eq!(ind, 0.0);
    let ind2 = aim_indicator(Some(-PI + 0.01));
    assert_eq!(ind2, 0.0);
}

#[test]
fn test_intercept_angle_moving_target_ahead() {
    let angle = intercept_angle(200.0, [300.0, 0.0], [50.0, 0.0]).unwrap();
    assert!(angle.abs() < 0.1, "expected near-zero angle, got {}", angle);
}

#[test]
fn test_obs_dim_matches_constant() {
    let expected = SELF_SIZE + N_ENTITY_SLOTS * SLOT_SIZE;
    assert_eq!(OBS_DIM, expected, "OBS_DIM constant does not match slot layout");

    let ship = dummy_ship();
    let obs = encode_observation(&minimal_obs_input(&ship));
    assert_eq!(obs.len(), OBS_DIM);
}

#[test]
fn test_target_pursuit_angle_ahead() {
    let ship = dummy_ship();
    // Place an asteroid directly ahead as the sole entity.
    let asteroid = EntitySlotData {
        core: CoreSlotData {
            entity_type: 1,
            rel_pos: [500.0, 0.0],
            rel_vel: [0.0, 0.0],
        },
        kind: EntityKind::Asteroid(AsteroidSlotData { size: 10.0, value: 1.0 }),
        value: 1.0,
        is_current_target: true,
    };
    let obs = encode_observation(&ObsInput {
        personality: &ship.data.personality,
        ship: &ship,
        velocity: [0.0, 0.0],
        angular_velocity: 0.0,
        ship_heading: [1.0, 0.0],
        entity_slots: vec![asteroid],
        primary_weapon_speed: 400.0,
        primary_weapon_range: 1000.0,
        credit_scale: 1000.0,
    });
    // The asteroid is in slot 0 (first and only entity).
    let slot_offset = SELF_SIZE;
    let pursuit_angle = obs[slot_offset + SLOT_PURSUIT_ANGLE];
    assert!(pursuit_angle.abs() < 1e-4, "expected ~0, got {}", pursuit_angle);
    let aim_ind = obs[slot_offset + SLOT_PURSUIT_INDICATOR];
    assert!((aim_ind - 1.0).abs() < 1e-4, "expected ~1, got {}", aim_ind);
    let in_range = obs[slot_offset + SLOT_IN_RANGE];
    assert_eq!(in_range, 1.0);
}

// ── Slot block layout tests ─────────────────────────────────────────────

/// Verify that the 4 blocks are contiguous and sum to SLOT_SIZE.
#[test]
fn test_slot_block_layout_sizes() {
    assert_eq!(
        TYPE_ONEHOT_SIZE + 1 + CORE_FEAT_SIZE + TYPE_BLOCK_SIZE,
        SLOT_SIZE,
        "block sizes must sum to SLOT_SIZE"
    );
    // Blocks are contiguous:
    assert_eq!(SLOT_TYPE_ONEHOT, 0);
    assert_eq!(SLOT_IS_PRESENT, TYPE_ONEHOT_SIZE);
    assert_eq!(CORE_BLOCK_START, SLOT_IS_PRESENT + 1);
    assert_eq!(TYPE_BLOCK_START, CORE_BLOCK_START + CORE_FEAT_SIZE);
    assert_eq!(SLOT_SIZE, TYPE_BLOCK_START + TYPE_BLOCK_SIZE);
}

/// Encode a known entity and verify each block contains the expected values.
#[test]
fn test_slot_block_extraction() {
    let ship = dummy_ship();

    // Create a hostile ship entity directly ahead at 400 units.
    let hostile_ship = EntitySlotData {
        core: CoreSlotData {
            rel_pos: [400.0, 0.0],
            rel_vel: [10.0, 5.0],
            entity_type: 0, // Ship
        },
        kind: EntityKind::Ship(ShipSlotData {
            max_health: 100.0,
            health: 75.0,
            max_speed: 200.0,
            torque: 10.0,
            is_hostile: 1.0,
            should_engage: 1.0,
            personality: Personality::Fighter,
        }),
        value: 0.0,
        is_current_target: true,
    };

    let obs = encode_observation(&ObsInput {
        personality: &ship.data.personality,
        ship: &ship,
        velocity: [0.0, 0.0],
        angular_velocity: 0.0,
        ship_heading: [1.0, 0.0],
        entity_slots: vec![hostile_ship],
        primary_weapon_speed: 400.0,
        primary_weapon_range: 800.0,
        credit_scale: 1000.0,
    });

    // Slot 0 starts at SELF_SIZE.
    let s = SELF_SIZE;

    // Block 1: type_onehot — Ship = [1, 0, 0, 0]
    assert_eq!(obs[s + SLOT_TYPE_ONEHOT], 1.0, "ship onehot[0]");
    assert_eq!(obs[s + SLOT_TYPE_ONEHOT + 1], 0.0, "ship onehot[1]");
    assert_eq!(obs[s + SLOT_TYPE_ONEHOT + 2], 0.0, "ship onehot[2]");
    assert_eq!(obs[s + SLOT_TYPE_ONEHOT + 3], 0.0, "ship onehot[3]");

    // Block 2: is_present
    assert_eq!(obs[s + SLOT_IS_PRESENT], 1.0, "is_present");

    // Block 3: core features
    assert_eq!(obs[s + SLOT_REL_POS], 400.0, "rel_pos_x");
    assert_eq!(obs[s + SLOT_REL_POS + 1], 0.0, "rel_pos_y");
    assert_eq!(obs[s + SLOT_REL_VEL], 10.0, "rel_vel_x");
    assert_eq!(obs[s + SLOT_REL_VEL + 1], 5.0, "rel_vel_y");
    assert_eq!(obs[s + SLOT_IS_CURRENT_TARGET], 1.0, "is_current_target");
    // Proximity: 500 / (400 + 500) = 0.5556
    let prox = obs[s + SLOT_PROXIMITY];
    assert!((prox - 500.0 / 900.0).abs() < 1e-4, "proximity={prox}");
    // Pursuit angle: target ahead but with lateral velocity → small nonzero angle.
    assert!(obs[s + SLOT_PURSUIT_ANGLE].abs() < 0.2, "pursuit_angle");
    // In range: dist=400 < weapon_range=800
    assert_eq!(obs[s + SLOT_IN_RANGE], 1.0, "in_range");

    // Block 4: value + type-specific
    assert_eq!(obs[s + SLOT_VALUE], 0.0, "value (ship)");
    // Type-specific for ships: max_health, health, max_speed, torque,
    // is_hostile, should_engage, personality(3)
    assert_eq!(obs[s + SLOT_TYPE_SPECIFIC + 0], 100.0, "max_health");
    assert_eq!(obs[s + SLOT_TYPE_SPECIFIC + 1], 75.0, "health");
    assert_eq!(obs[s + SLOT_TYPE_SPECIFIC + 2], 200.0, "max_speed");
    assert_eq!(obs[s + SLOT_TYPE_SPECIFIC + 3], 10.0, "torque");
    assert_eq!(obs[s + SLOT_TYPE_SPECIFIC + 4], 1.0, "is_hostile");
    assert_eq!(obs[s + SLOT_TYPE_SPECIFIC + 5], 1.0, "should_engage");
    // Fighter personality = [0, 1, 0]
    assert_eq!(obs[s + SLOT_TYPE_SPECIFIC + 6], 0.0, "personality[0]");
    assert_eq!(obs[s + SLOT_TYPE_SPECIFIC + 7], 1.0, "personality[1]");
    assert_eq!(obs[s + SLOT_TYPE_SPECIFIC + 8], 0.0, "personality[2]");
}

/// Verify that an empty slot is all zeros.
#[test]
fn test_empty_slot_all_zeros() {
    let ship = dummy_ship();
    let obs = encode_observation(&minimal_obs_input(&ship));

    // Slot 0 is empty (default entity).
    let s = SELF_SIZE;
    for i in 0..SLOT_SIZE {
        assert_eq!(
            obs[s + i], 0.0,
            "empty slot byte {i} should be 0, got {}",
            obs[s + i]
        );
    }
}

/// Verify planet type-specific block encoding.
#[test]
fn test_planet_slot_block_extraction() {
    let ship = dummy_ship();

    let planet = EntitySlotData {
        core: CoreSlotData {
            rel_pos: [1000.0, 200.0],
            rel_vel: [0.0, 0.0],
            entity_type: 2, // Planet
        },
        kind: EntityKind::Planet(PlanetSlotData {
            cargo_sale_value: 500.0,
            has_ammo: 1.0,
            commodity_margin: -100.0,
        }),
        value: 500.0,
        is_current_target: false,
    };

    let obs = encode_observation(&ObsInput {
        personality: &ship.data.personality,
        ship: &ship,
        velocity: [0.0, 0.0],
        angular_velocity: 0.0,
        ship_heading: [1.0, 0.0],
        entity_slots: vec![planet],
        primary_weapon_speed: 400.0,
        primary_weapon_range: 800.0,
        credit_scale: 1000.0,
    });

    let s = SELF_SIZE;

    // Block 1: Planet = [0, 0, 1, 0]
    assert_eq!(obs[s + SLOT_TYPE_ONEHOT + 2], 1.0, "planet onehot");

    // Block 4: value + type-specific
    assert_eq!(obs[s + SLOT_VALUE], 500.0, "value");
    assert_eq!(obs[s + SLOT_TYPE_SPECIFIC + 0], 500.0, "cargo_sale_value");
    assert_eq!(obs[s + SLOT_TYPE_SPECIFIC + 1], 1.0, "has_ammo");
    assert_eq!(obs[s + SLOT_TYPE_SPECIFIC + 2], -100.0, "commodity_margin");
    // Remaining type-specific should be zero-padded.
    for i in 3..TYPE_SPECIFIC_SIZE {
        assert_eq!(obs[s + SLOT_TYPE_SPECIFIC + i], 0.0, "type_specific[{i}] should be 0");
    }
}
