use super::*;
use crate::item_universe::ItemUniverse;
use crate::rl_obs::{controls_to_discrete, discrete_to_controls};
use avian2d::prelude::{LinearVelocity, Position};
use bevy::ecs::system::SystemState;
use bevy::prelude::*;
use std::collections::HashMap;

fn empty_item_universe() -> ItemUniverse {
    ItemUniverse {
        weapons: HashMap::new(),
        ships: HashMap::new(),
        star_systems: HashMap::new(),
        outfitter_items: HashMap::new(),
        enemies: HashMap::new(),
        starting_ship: String::new(),
        starting_system: String::new(),
        commodities: HashMap::new(),
        global_average_price: HashMap::new(),
        system_commodity_best_planet_to_sell: HashMap::new(),
        system_planet_best_commodity_to_buy: HashMap::new(),
        planet_best_margin: HashMap::new(),
        planet_has_ammo_for: HashMap::new(),
        asteroid_field_expected_value: HashMap::new(),
        ship_credit_scale: HashMap::new(),
    }
}

// ── Pure-function tests ──────────────────────────────────────────────────

#[test]
fn test_angle_to_controls_all_branches() {
    // Large left (> PI/3) → hard left, no thrust
    let (turn, thrust) = angle_to_controls(PI / 2.0);
    assert_eq!(turn, -1.0);
    assert_eq!(thrust, 0.0);

    // Small left (0 < angle ≤ PI/3) → gentle left, thrust
    let (turn, thrust) = angle_to_controls(0.5);
    assert_eq!(turn, -0.5);
    assert_eq!(thrust, 1.0);

    // Small right (-PI/3 ≤ angle < 0) → gentle right, thrust
    let (turn, thrust) = angle_to_controls(-0.5);
    assert_eq!(turn, 0.5);
    assert_eq!(thrust, 1.0);

    // Large right (< -PI/3) → hard right, no thrust
    let (turn, thrust) = angle_to_controls(-PI / 2.0);
    assert_eq!(turn, 1.0);
    assert_eq!(thrust, 0.0);
}

#[test]
fn test_braking_distance_properties() {
    let (thrust, kp, kd, max_speed) = (100.0, 1.0, 2.0, 200.0_f32);

    // Zero speed → zero stopping distance
    assert_eq!(braking_distance(0.0, thrust, kp, kd, max_speed), 0.0);

    // Monotonically increasing with speed
    let d1 = braking_distance(50.0, thrust, kp, kd, max_speed);
    let d2 = braking_distance(100.0, thrust, kp, kd, max_speed);
    let d3 = braking_distance(200.0, thrust, kp, kd, max_speed);
    assert!(d1 > 0.0);
    assert!(d2 > d1);
    assert!(d3 > d2);
}

/// The most critical correctness test: verifies the (turn_idx, thrust_idx, ...)
/// tuple ordering is consistent between `store_obs_actions` (which builds the
/// DiscreteAction) and `repeat_actions` (which decodes it).
#[test]
fn test_discrete_action_ordering_roundtrip() {
    for &(thrust, turn) in &[
        (1.0_f32, -1.0_f32), // thrust + turn left
        (0.0_f32, 1.0_f32),  // no thrust + turn right
        (1.0_f32, 0.0_f32),  // thrust straight
        (0.0_f32, 0.0_f32),  // coast straight
    ] {
        // store_obs_actions: controls → DiscreteAction
        let (thrust_idx, turn_idx) = controls_to_discrete(thrust, turn);
        let action: (u8, u8, u8, u8) = (turn_idx, thrust_idx, 0, 0);

        // repeat_actions: DiscreteAction → controls
        let (decoded_turn_idx, decoded_thrust_idx, _, _) = action;
        let (rt, rr) = discrete_to_controls(decoded_thrust_idx, decoded_turn_idx);

        let expected_thrust = if thrust > 0.5 { 1.0_f32 } else { 0.0 };
        let expected_turn = if turn < -0.25 {
            -1.0_f32
        } else if turn > 0.25 {
            1.0
        } else {
            0.0
        };
        assert_eq!(
            rt, expected_thrust,
            "thrust mismatch for input ({}, {})",
            thrust, turn
        );
        assert!(
            (rr - expected_turn).abs() < 0.01,
            "turn mismatch for input ({}, {}): got {}, expected {}",
            thrust,
            turn,
            rr,
            expected_turn
        );
    }
}

// ── ECS-based tests for compute_ai_action ───────────────────────────────

#[test]
fn test_compute_ai_action_no_target() {
    let mut world = World::new();
    let mut state: SystemState<(Query<&Position>, Query<&LinearVelocity>)> =
        SystemState::new(&mut world);
    let (pos_q, vel_q) = state.get(&world);

    let ship = Ship::default();
    let result = compute_ai_action(
        &ship,
        Vec2::ZERO,
        Vec2::ZERO,
        200.0,
        &Transform::default(),
        &pos_q,
        &vel_q,
        &empty_item_universe(),
    );
    assert!(result.is_none(), "no target → should return None");
}

#[test]
fn test_compute_ai_action_target_entity_missing() {
    let mut world = World::new();
    // Spawn entity without Position component so the query will fail
    let ghost = world.spawn_empty().id();

    let mut state: SystemState<(Query<&Position>, Query<&LinearVelocity>)> =
        SystemState::new(&mut world);
    let (pos_q, vel_q) = state.get(&world);

    let mut ship = Ship::default();
    ship.target = Some(Target::Asteroid(ghost));

    let result = compute_ai_action(
        &ship,
        Vec2::ZERO,
        Vec2::ZERO,
        200.0,
        &Transform::default(),
        &pos_q,
        &vel_q,
        &empty_item_universe(),
    );
    assert!(
        result.is_none(),
        "missing target position → should return None"
    );
}

#[test]
fn test_compute_ai_action_forward_asteroid() {
    let mut world = World::new();
    // Target 500 units ahead. Default transform faces +y, so +y is "forward".
    let target = world
        .spawn((Position(Vec2::new(0.0, 500.0)), LinearVelocity(Vec2::ZERO)))
        .id();

    let mut state: SystemState<(Query<&Position>, Query<&LinearVelocity>)> =
        SystemState::new(&mut world);
    let (pos_q, vel_q) = state.get(&world);

    let mut ship = Ship::default();
    ship.target = Some(Target::Asteroid(target));

    let result = compute_ai_action(
        &ship,
        Vec2::ZERO,
        Vec2::ZERO,
        200.0,
        &Transform::default(),
        &pos_q,
        &vel_q,
        &empty_item_universe(),
    );
    let action = result.expect("valid forward target should produce an action");
    assert!(
        action.thrust > 0.5,
        "should thrust toward forward target, got {}",
        action.thrust
    );
    assert_eq!(action.reverse, 0.0, "should not brake toward an asteroid");
    assert!(
        action.weapons_to_fire.is_empty(),
        "default ship has no weapons"
    );
}

#[test]
fn test_compute_ai_action_planet_braking() {
    let mut world = World::new();
    // Planet just 50 units away
    let planet = world
        .spawn((Position(Vec2::new(0.0, 50.0)), LinearVelocity(Vec2::ZERO)))
        .id();

    let mut state: SystemState<(Query<&Position>, Query<&LinearVelocity>)> =
        SystemState::new(&mut world);
    let (pos_q, vel_q) = state.get(&world);

    let mut ship = Ship::default();
    ship.data.thrust = 100.0;
    ship.data.thrust_kp = 1.0;
    ship.data.thrust_kd = 2.0;
    ship.data.max_speed = 200.0;
    ship.data.torque = 10.0;
    ship.data.angular_drag = 1.0;
    ship.target = Some(Target::Planet(planet));

    // Moving at 500 m/s toward the planet — braking distance >> 50 units.
    // Ship faces +Y (default) and vel is +Y, so the ship is flying prograde.
    // In braking zone the AI should turn toward retrograde (-Y) and thrust.
    let vel = Vec2::new(0.0, 500.0);

    let result = compute_ai_action(
        &ship,
        Vec2::ZERO,
        vel,
        200.0,
        &Transform::default(),
        &pos_q,
        &vel_q,
        &empty_item_universe(),
    );
    let action = result.expect("close planet should produce braking action");
    assert_eq!(action.reverse, 0.0, "should use turn+thrust, not reverse");
    // Ship faces +Y, velocity is +Y, retrograde is -Y → bearing = ±PI → turn hard
    assert!(
        action.turn.abs() > 0.5,
        "should be turning toward retrograde, got turn={}",
        action.turn
    );
}

#[test]
fn test_compute_ai_action_planet_approach() {
    let mut world = World::new();
    // Planet far away — 5000 units
    let planet = world
        .spawn((Position(Vec2::new(0.0, 5000.0)), LinearVelocity(Vec2::ZERO)))
        .id();

    let mut state: SystemState<(Query<&Position>, Query<&LinearVelocity>)> =
        SystemState::new(&mut world);
    let (pos_q, vel_q) = state.get(&world);

    let mut ship = Ship::default();
    ship.data.thrust = 100.0;
    ship.data.thrust_kp = 1.0;
    ship.data.thrust_kd = 2.0;
    ship.data.max_speed = 200.0;
    ship.data.torque = 10.0;
    ship.data.angular_drag = 1.0;
    ship.target = Some(Target::Planet(planet));

    let result = compute_ai_action(
        &ship,
        Vec2::ZERO,
        Vec2::ZERO, // stationary, so braking_distance = 0
        200.0,
        &Transform::default(),
        &pos_q,
        &vel_q,
        &empty_item_universe(),
    );
    let action = result.expect("far planet should produce approach action");
    assert_eq!(action.reverse, 0.0, "should not brake when far away");
    assert!(action.thrust > 0.5, "should thrust toward far planet");
}

#[test]
fn test_compute_ai_action_planet_prepare_zone() {
    let mut world = World::new();
    let mut ship = Ship::default();
    ship.data.thrust = 100.0;
    ship.data.thrust_kp = 5.0;
    ship.data.thrust_kd = 1.0;
    ship.data.max_speed = 200.0;
    ship.data.torque = 10.0;
    ship.data.angular_drag = 3.0;

    // Ship moving at 200 m/s toward a planet.
    // braking_distance(200, 100, 5, 1, 200) ≈ 200
    // turn_margin = 200 * PI * 3 / 10 ≈ 188
    // brake_dist ≈ 388, prepare_dist ≈ 576
    // Place planet at 500 units — inside prepare zone but outside brake zone.
    let planet = world
        .spawn((Position(Vec2::new(0.0, 500.0)), LinearVelocity(Vec2::ZERO)))
        .id();
    ship.target = Some(Target::Planet(planet));

    let mut state: SystemState<(Query<&Position>, Query<&LinearVelocity>)> =
        SystemState::new(&mut world);
    let (pos_q, vel_q) = state.get(&world);

    let vel = Vec2::new(0.0, 200.0);
    let result = compute_ai_action(
        &ship,
        Vec2::ZERO,
        vel,
        200.0,
        &Transform::default(), // faces +y = same as velocity
        &pos_q,
        &vel_q,
        &empty_item_universe(),
    );
    let action = result.expect("prepare zone should produce action");
    assert_eq!(action.reverse, 0.0, "should use turn+thrust, not reverse");
    assert_eq!(
        action.thrust, 0.0,
        "should NOT thrust while in prepare zone (turning, not braking)"
    );
    // Ship faces +Y, velocity is +Y, retrograde is -Y → should be turning
    assert!(
        action.turn.abs() > 0.5,
        "should be turning toward retrograde in prepare zone, got turn={}",
        action.turn
    );
}

// ── Jump mechanic tests ──────────────────────────────────────────────────

/// `JumpingIn` ships start at `JUMP_SPEED` and should decelerate each tick.
/// Test drives `jump_in_system` directly without state guards.
#[test]
fn test_jump_in_decelerates() {
    use bevy::time::TimeUpdateStrategy;
    use std::time::Duration;

    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        // Force a known 100 ms delta so jump_in_system always has non-zero dt.
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_millis(
            100,
        )))
        .add_systems(Update, jump_in_system);

    // Spawn a JumpingIn ship moving at JUMP_SPEED.
    let entity = app
        .world_mut()
        .spawn((
            JumpingIn,
            LinearVelocity(Vec2::new(0.0, JUMP_SPEED)),
            Ship::default(),
        ))
        .id();

    app.update(); // warm up — establishes time baseline
    app.update(); // actual tick with ManualDuration delta
    let vel = app.world().get::<LinearVelocity>(entity).unwrap().0;
    assert!(
        vel.length() < JUMP_SPEED,
        "ship should have decelerated, got speed {}",
        vel.length()
    );
}

/// Once a `JumpingIn` ship reaches its normal max speed, `JumpingIn` is removed.
#[test]
fn test_jump_in_completes_when_slow_enough() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_systems(Update, jump_in_system);

    // Spawn at exactly max_speed — should remove JumpingIn on the next tick.
    let ship = Ship::default();
    let normal_speed = ship.data.max_speed;
    let entity = app
        .world_mut()
        .spawn((
            JumpingIn,
            LinearVelocity(Vec2::new(0.0, normal_speed)),
            ship,
        ))
        .id();

    app.update();
    assert!(
        app.world().get::<JumpingIn>(entity).is_none(),
        "JumpingIn should be removed once ship reaches normal speed"
    );
}

/// `ShipDistribution::sample` should return the requested number of ships.
#[test]
fn test_ship_distribution_sample_count() {
    use crate::item_universe::ShipDistribution;
    let mut types = HashMap::new();
    types.insert("fighter".to_string(), 1.0);
    types.insert("hauler".to_string(), 2.0);
    let dist = ShipDistribution {
        min: 3,
        max: 8,
        types,
    };
    let mut rng = rand::thread_rng();
    let result = dist.sample(5, &mut rng);
    assert_eq!(result.len(), 5, "should return exactly 5 ship types");
    for t in &result {
        assert!(
            t == "fighter" || t == "hauler",
            "unexpected ship type: {}",
            t
        );
    }
}

/// `ShipDistribution::sample` returns empty for an empty distribution.
#[test]
fn test_ship_distribution_sample_empty() {
    use crate::item_universe::ShipDistribution;
    let dist = ShipDistribution::default();
    let mut rng = rand::thread_rng();
    let result = dist.sample(5, &mut rng);
    assert!(
        result.is_empty(),
        "empty distribution should return no ships"
    );
}
