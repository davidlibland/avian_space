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
        target: None,
        nearby_planets: vec![],
        nearby_asteroids: vec![],
        nearby_hostile_ships: vec![],
        nearby_friendly_ships: vec![],
        nearby_pickups: vec![],
        primary_weapon_speed: 400.0,
        primary_weapon_range: 800.0,
    }
}

#[test]
fn test_observation_shape_constant() {
    let ship = dummy_ship();

    // No nearby entities.
    let obs0 = encode_observation(&minimal_obs_input(&ship));
    assert_eq!(obs0.len(), OBS_DIM, "zero entities");

    // One of each type.
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
    let target_slot = EntitySlotData {
        core: CoreSlotData {
            rel_pos: [0.3, 0.4],
            rel_vel: [0.0, 0.0],
            entity_type: 1,
        },
        kind: EntityKind::Asteroid(AsteroidSlotData { size: 10.0, value: 1.0 }),
        value: 1.0,
        is_current_target: true,
    };
    let obs1 = encode_observation(&ObsInput {
        personality: &ship.data.personality,
        ship: &ship,
        velocity: [10.0, 0.0],
        angular_velocity: 0.5,
        ship_heading: [1.0, 0.0],
        target: Some(target_slot),
        nearby_planets: vec![slot.clone()],
        nearby_asteroids: vec![slot.clone(), slot.clone()],
        nearby_hostile_ships: vec![slot.clone()],
        nearby_friendly_ships: vec![],
        nearby_pickups: vec![slot.clone()],
        primary_weapon_speed: 400.0,
        primary_weapon_range: 800.0,
    });
    assert_eq!(obs1.len(), OBS_DIM, "some entities");

    // Fully-populated buckets.
    let target_slot2 = EntitySlotData {
        core: CoreSlotData {
            rel_pos: [0.1, 0.0],
            rel_vel: [0.0, 0.0],
            entity_type: 0,
        },
        kind: EntityKind::Ship(ShipSlotData {
            is_hostile: 1.0,
            should_engage: 1.0,
            ..Default::default()
        }),
        value: 0.5,
        is_current_target: true,
    };
    let obs2 = encode_observation(&ObsInput {
        personality: &ship.data.personality,
        ship: &ship,
        velocity: [0.0, 0.0],
        angular_velocity: 0.0,
        ship_heading: [0.0, 1.0],
        target: Some(target_slot2),
        nearby_planets: vec![slot.clone(); K_PLANETS],
        nearby_asteroids: vec![slot.clone(); K_ASTEROIDS],
        nearby_hostile_ships: vec![slot.clone(); K_HOSTILE_SHIPS],
        nearby_friendly_ships: vec![slot.clone(); K_FRIENDLY_SHIPS],
        nearby_pickups: vec![slot.clone(); K_PICKUPS],
        primary_weapon_speed: 400.0,
        primary_weapon_range: 800.0,
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
    let obs = encode_observation(&ObsInput {
        personality: &ship.data.personality,
        ship: &ship,
        velocity: [0.0, 0.0],
        angular_velocity: 0.0,
        ship_heading: [1.0, 0.0],
        target: Some(EntitySlotData {
            core: CoreSlotData {
                entity_type: 1,
                rel_pos: [500.0, 0.0],
                rel_vel: [0.0, 0.0],
            },
            kind: EntityKind::Asteroid(AsteroidSlotData { size: 10.0, value: 1.0 }),
            value: 1.0,
            is_current_target: true,
        }),
        nearby_planets: vec![],
        nearby_asteroids: vec![],
        nearby_hostile_ships: vec![],
        nearby_friendly_ships: vec![],
        nearby_pickups: vec![],
        primary_weapon_speed: 400.0,
        primary_weapon_range: 1000.0,
    });
    let pursuit_angle = obs[SELF_SIZE + SLOT_PURSUIT_ANGLE];
    assert!(pursuit_angle.abs() < 1e-4, "expected ~0, got {}", pursuit_angle);
    let aim_ind = obs[SELF_SIZE + SLOT_PURSUIT_INDICATOR];
    assert!((aim_ind - 1.0).abs() < 1e-4, "expected ~1, got {}", aim_ind);
    let in_range = obs[SELF_SIZE + SLOT_IN_RANGE];
    assert_eq!(in_range, 1.0);
}
