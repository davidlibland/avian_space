use super::*;
use crate::item_universe::ItemUniverse;
use crate::rl_obs::{controls_to_discrete, discrete_to_controls};
use avian2d::prelude::{LinearVelocity, Position};
use bevy::ecs::system::SystemState;
use std::collections::HashMap;

fn empty_item_universe() -> ItemUniverse {
    ItemUniverse {
        weapons: HashMap::new(),
        ships: HashMap::new(),
        star_systems: HashMap::new(),
        simulator_system: None,
        outfitter_items: HashMap::new(),
        enemies: HashMap::new(),
        starting_ship: String::new(),
        starting_system: String::new(),
        commodities: HashMap::new(),
        missions: HashMap::new(),
        mission_templates: HashMap::new(),
        global_average_price: HashMap::new(),
        global_minimum_price: HashMap::new(),
        system_commodity_best_planet_to_sell: HashMap::new(),
        system_planet_best_commodity_to_buy: HashMap::new(),
        planet_best_margin: HashMap::new(),
        planet_has_ammo_for: HashMap::new(),
        asteroid_field_expected_value: HashMap::new(),
        ship_credit_scale: HashMap::new(),
        allies: HashMap::new(),
    }
}

// ── Pure-function tests ──────────────────────────────────────────────────

#[test]
fn test_angle_to_controls_bang_bang_at_rest() {
    // At rest (ang_vel = 0) the stop angle is 0, so any non-zero target angle
    // triggers max torque in the direction of the target.
    let (torque, damping) = (1.0_f32, 1.0_f32);

    // Large left (> PI/3) → max left torque, no thrust (outside forward cone)
    let (turn, thrust) = angle_to_controls(PI / 2.0, 0.0, torque, damping);
    assert_eq!(turn, -1.0);
    assert_eq!(thrust, 0.0);

    // Small left (within PI/3) → max left torque + thrust (inside cone)
    let (turn, thrust) = angle_to_controls(0.5, 0.0, torque, damping);
    assert_eq!(turn, -1.0);
    assert_eq!(thrust, 1.0);

    // Small right → max right torque + thrust
    let (turn, thrust) = angle_to_controls(-0.5, 0.0, torque, damping);
    assert_eq!(turn, 1.0);
    assert_eq!(thrust, 1.0);

    // Large right (< -PI/3) → max right torque, no thrust
    let (turn, thrust) = angle_to_controls(-PI / 2.0, 0.0, torque, damping);
    assert_eq!(turn, 1.0);
    assert_eq!(thrust, 0.0);
}

#[test]
fn test_angle_to_controls_deadband() {
    // Inside ±PI/16 the controller should return turn = 0.0 and let damping
    // handle residual motion.
    let (torque, damping) = (1.0_f32, 1.0_f32);
    let (turn, _) = angle_to_controls(PI / 32.0, 0.0, torque, damping);
    assert_eq!(turn, 0.0, "small positive angle inside deadband should coast");

    let (turn, _) = angle_to_controls(-PI / 32.0, 0.0, torque, damping);
    assert_eq!(turn, 0.0, "small negative angle inside deadband should coast");
}

#[test]
fn test_angle_to_controls_coasts_when_overshoot_predicted() {
    // Already rotating fast toward the target: if braking now would still
    // overshoot (stop_angle >= target_angle), the controller should coast
    // instead of adding more torque in the same direction.
    // Pick target_angle outside the ±PI/16 deadband (~0.196).
    let torque = 1.0_f32;
    let damping = 0.1_f32; // low damping → long stopping distance
    let target_angle = 0.3_f32;
    // With these params, x_stop(0, 1.0, 1.0, 0.1) ≈ 0.47 > target, so overshoot.
    let ang_vel = 1.0_f32;

    let (turn, _thrust) = angle_to_controls(target_angle, ang_vel, torque, damping);
    assert_eq!(
        turn, 1.0,
        "should coast when max braking would still overshoot the target"
    );
}

#[test]
fn test_angle_to_controls_brakes_when_undershoot_predicted() {
    // Rotating slowly toward a positive target: max braking would stop short
    // of the target (stop_angle < target_angle), so apply max torque toward it.
    let (torque, damping) = (1.0_f32, 1.0_f32);
    // Tiny ang_vel → stop_angle ≈ 0, well below target_angle = 0.5.
    let (turn, _) = angle_to_controls(0.5, 0.01, torque, damping);
    assert_eq!(turn, -1.0, "should apply max torque toward positive target");
}

#[test]
fn test_x_stop_properties() {
    use crate::optimal_control::x_stop;
    let (a, gamma) = (100.0_f32, 1.0_f32);

    // Zero speed → zero stopping distance
    assert_eq!(x_stop(0.0, a, gamma), 0.0);

    // Stopping distance magnitude grows monotonically with speed.
    let d1 = x_stop(50.0, a, gamma).abs();
    let d2 = x_stop(100.0, a, gamma).abs();
    let d3 = x_stop(200.0, a, gamma).abs();
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
        let action: (u8, u8, u8, u8, u8, u8) = (turn_idx, thrust_idx, 0, 0, 0, 0);

        // repeat_actions: DiscreteAction → controls
        let (decoded_turn_idx, decoded_thrust_idx, _, _, _, _) = action;
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
        0.0,
        200.0,
        &Transform::default(),
        &pos_q,
        &vel_q,
        &empty_item_universe(),
        &mut rand::thread_rng(),
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
    ship.nav_target = Some(Target::Asteroid(ghost));

    let result = compute_ai_action(
        &ship,
        Vec2::ZERO,
        Vec2::ZERO,
        0.0,
        200.0,
        &Transform::default(),
        &pos_q,
        &vel_q,
        &empty_item_universe(),
        &mut rand::thread_rng(),
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
    ship.data.thrust = 100.0;
    ship.data.max_speed = 200.0;
    ship.data.torque = 10.0;
    ship.data.angular_drag = 3.0;
    ship.nav_target = Some(Target::Asteroid(target));

    // With no weapons the ship is out of firing range, so the PD pursuit
    // branch runs. For a stationary forward target at 500 units with
    // thrust=100, max_speed=200, desired acceleration saturates → thrust
    // fires every tick.
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xBEEF);
    let mut fires = 0;
    let trials = 50;
    for _ in 0..trials {
        let action = compute_ai_action(
            &ship,
            Vec2::ZERO,
            Vec2::ZERO,
            0.0,
            200.0,
            &Transform::default(),
            &pos_q,
            &vel_q,
            &empty_item_universe(),
            &mut rng,
        )
        .expect("valid forward target should produce an action");
        assert_eq!(action.reverse, 0.0, "should not brake toward an asteroid");
        assert!(
            action.weapons_to_fire.is_empty(),
            "default ship has no weapons"
        );
        if action.thrust > 0.5 {
            fires += 1;
        }
    }
    assert!(
        fires > trials / 2,
        "expected mostly-saturating thrust, got {fires}/{trials}"
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
    ship.nav_target = Some(Target::Planet(planet));

    // Moving at 500 m/s toward the planet — braking distance >> 50 units.
    // Ship faces +Y (default) and vel is +Y, so the ship is flying prograde.
    // In braking zone the AI should turn toward retrograde (-Y) and thrust.
    let vel = Vec2::new(0.0, 500.0);

    let result = compute_ai_action(
        &ship,
        Vec2::ZERO,
        vel,
        0.0,
        200.0,
        &Transform::default(),
        &pos_q,
        &vel_q,
        &empty_item_universe(),
        &mut rand::thread_rng(),
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
    // Stationary ship, far planet (5000 units) straight ahead. With the PD
    // controller tuned to `k_x = thrust / max_speed²`, the desired acceleration
    // is small compared to max thrust, so PWM thrust fires only probabilistically.
    // We verify: (a) action produced, (b) turn is zero (already aligned), and
    // (c) thrust fires at least sometimes across many trials.
    use rand::SeedableRng;

    let mut world = World::new();
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
    ship.nav_target = Some(Target::Planet(planet));

    let mut rng = rand::rngs::StdRng::seed_from_u64(0xABCD);
    let mut fires = 0;
    let trials = 200;
    for _ in 0..trials {
        let action = compute_ai_action(
            &ship,
            Vec2::ZERO,
            Vec2::ZERO,
            0.0,
            200.0,
            &Transform::default(),
            &pos_q,
            &vel_q,
            &empty_item_universe(),
            &mut rng,
        )
        .expect("far planet should produce approach action");
        assert_eq!(action.reverse, 0.0, "should not brake when far away");
        assert_eq!(action.turn, 0.0, "already aligned → turn deadband");
        if action.thrust > 0.5 {
            fires += 1;
        }
    }
    // With K_X_CONST=400 and D=5000, the PD requests ~50×a_max → clamps to
    // thrust_prob=1.0, so thrust should fire every tick.
    assert_eq!(
        fires, trials,
        "far target with saturated PD → thrust every tick, got {fires}/{trials}"
    );
}

#[test]
fn test_compute_ai_action_planet_pd_brakes_when_overshooting() {
    // Ship close to target and moving fast toward it: PD wants negative
    // acceleration (brake), so `direction` points backward, `target_angle ≈ π`
    // (or ±π via the branch-cut fix). The controller should command a turn
    // and should NOT thrust (cosine weighting zeros it out when facing away).
    let mut world = World::new();
    let planet = world
        .spawn((Position(Vec2::new(0.0, 50.0)), LinearVelocity(Vec2::ZERO)))
        .id();

    let mut state: SystemState<(Query<&Position>, Query<&LinearVelocity>)> =
        SystemState::new(&mut world);
    let (pos_q, vel_q) = state.get(&world);

    let mut ship = Ship::default();
    ship.data.thrust = 100.0;
    ship.data.max_speed = 200.0;
    ship.data.torque = 10.0;
    ship.data.angular_drag = 1.0;
    ship.nav_target = Some(Target::Planet(planet));

    // Fast flight toward target: PD says "brake", direction flips to −x.
    let vel = Vec2::new(0.0, 500.0);
    let action = compute_ai_action(
        &ship,
        Vec2::ZERO,
        vel,
        0.0,
        200.0,
        &Transform::default(),
        &pos_q,
        &vel_q,
        &empty_item_universe(),
        &mut rand::thread_rng(),
    )
    .expect("close + fast → braking action");
    assert_eq!(action.thrust, 0.0, "facing forward but PD wants to brake → no thrust");
    assert!(
        action.turn.abs() > 0.5,
        "should be turning toward retrograde, got turn={}",
        action.turn
    );
}

#[test]
fn test_compute_ai_action_planet_pd_moderate_approach() {
    // Ship moving at modest speed toward a planet just beyond the PD's
    // saturation distance. With k_x ≈ 1, the desired acceleration stays below
    // a_max, so `thrust_prob` is proportional (not saturating). Expect: turn
    // in deadband (already aligned) and thrust fires some-but-not-all ticks.
    use rand::SeedableRng;

    let mut world = World::new();
    let planet = world
        .spawn((Position(Vec2::new(0.0, 500.0)), LinearVelocity(Vec2::ZERO)))
        .id();
    let mut state: SystemState<(Query<&Position>, Query<&LinearVelocity>)> =
        SystemState::new(&mut world);
    let (pos_q, vel_q) = state.get(&world);

    let mut ship = Ship::default();
    ship.data.thrust = 100.0;
    ship.data.max_speed = 200.0;
    ship.data.torque = 10.0;
    ship.data.angular_drag = 3.0;
    ship.nav_target = Some(Target::Planet(planet));

    // Moving at 200 toward the planet: a = k_x·500 − k_v·200 = 500 − 400 = 100.
    // Exactly at a_max → thrust_prob ≈ 1.0. Use a slightly slower ship so
    // we're in the sub-saturation regime.
    let vel = Vec2::new(0.0, 150.0);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xFEED);
    let action = compute_ai_action(
        &ship,
        Vec2::ZERO,
        vel,
        0.0,
        200.0,
        &Transform::default(),
        &pos_q,
        &vel_q,
        &empty_item_universe(),
        &mut rng,
    )
    .expect("valid PD action");
    assert_eq!(action.turn, 0.0, "already aligned → turn deadband");
    assert!(matches!(action.thrust, 0.0 | 1.0), "binary thrust output");
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
