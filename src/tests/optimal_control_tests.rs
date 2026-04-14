use crate::optimal_control::{pursuit_controls_ego, x_stop};
use bevy::math::Vec2;
use rand::{Rng, SeedableRng};
use std::f32::consts::PI;

/// Numerically integrate the dynamics `dv/dt = -a·sign(v) - γ·v` and return
/// the *displacement* from the initial position until `v` reaches zero.
fn x_stop_numerical(v_now: f32, a: f32, gamma: f32) -> f32 {
    let dt = 1e-5_f32;
    let (mut x, mut v) = (0.0_f32, v_now);
    for _ in 0..2_000_000 {
        if v.abs() < 1e-6 {
            break;
        }
        let dv = (-a * v.signum() - gamma * v) * dt;
        if (v + dv).signum() != v.signum() {
            break;
        }
        v += dv;
        x += v * dt;
    }
    x
}

#[test]
fn test_x_stop() {
    let cases = [
        (2.0, 5.0, 1.0),
        (-2.0, 5.0, 1.0),
        (0.5, 3.0, 0.5),
        (5.0, 5.0, 2.0), // strong damping
        (0.1, 5.0, 0.1), // weak damping
    ];
    for (v0, a, gamma) in cases {
        let diff = (x_stop(v0, a, gamma) - x_stop_numerical(v0, a, gamma)).abs();
        assert!(diff < 1e-4, "v0={v0} a={a} gamma={gamma}: diff={diff}");
    }
}

#[test]
fn test_x_stop_zero_velocity() {
    // Zero velocity → no further displacement until stop.
    assert_eq!(x_stop(0.0, 5.0, 1.0), 0.0);
}

#[test]
fn test_x_stop_sign_symmetry() {
    let (pos, neg) = (x_stop(2.0, 4.0, 1.0), x_stop(-2.0, 4.0, 1.0));
    assert!((pos + neg).abs() < 1e-4, "pos={pos} neg={neg}");
}

// ── Landing simulation ─────────────────────────────────────────────────────

/// Minimal ship description needed for the simulation. Parsed directly from
/// `assets/ships.yaml` so this test stays in sync with live ship tuning.
#[derive(serde::Deserialize, Debug, Clone)]
struct SimShip {
    thrust: f32,
    max_speed: f32,
    torque: f32,
    #[serde(default = "default_angular_drag")]
    angular_drag: f32,
    #[serde(default = "default_thrust_kp")]
    thrust_kp: f32,
    #[serde(default = "default_thrust_kd")]
    thrust_kd: f32,
    #[serde(default)]
    radius: f32,
}

fn default_angular_drag() -> f32 {
    3.0
}
fn default_thrust_kp() -> f32 {
    5.0
}
fn default_thrust_kd() -> f32 {
    1.0
}

/// Simulate one ship approaching a stationary target from rest, under
/// `pursuit_controls_ego` + PWM thrust sampling + the same ship dynamics
/// used by [`crate::ship::ship_control_system`]. Returns the landing time in
/// seconds, or `None` if the ship failed to land within `budget`.
fn simulate_landing(ship: &SimShip, distance: f32, budget: f32, seed: u64) -> Option<f32> {
    let mut pos = Vec2::ZERO;
    let target_pos = Vec2::new(distance, 0.0);
    let mut vel = Vec2::ZERO;
    let mut heading = 0.0_f32; // radians, 0 = facing +x toward target
    let mut ang_vel = 0.0_f32;

    let dt = 0.05_f32;
    let landing_radius = ship.radius.max(5.0);
    let landing_speed = ship.max_speed * 0.1;
    let max_ticks = (budget / dt) as usize;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    for tick in 0..max_ticks {
        let forward = Vec2::new(heading.cos(), heading.sin());
        let (ca, sa) = (heading.cos(), heading.sin());
        // Rotate by −heading so `forward` maps to ego +x.
        let rotate_r = |v: Vec2| Vec2::new(v.x * ca + v.y * sa, -v.x * sa + v.y * ca);

        let local_offset = rotate_r(target_pos - pos);
        let local_rel_vel = rotate_r(-vel);

        let (turn, thrust_prob) = pursuit_controls_ego(
            local_offset,
            local_rel_vel,
            ang_vel,
            ship.torque,
            ship.angular_drag,
            ship.thrust,
            ship.max_speed,
            1.0, // landing → critical damping
        );
        let thrust = if rng.r#gen::<f32>() < thrust_prob {
            1.0_f32
        } else {
            0.0
        };

        // Linear dynamics — mirror `ship_control_system`'s PD thrust force.
        if thrust > 0.5 {
            let v_fwd = vel.dot(forward);
            let pd_force = (ship.thrust_kp * (ship.max_speed - v_fwd) - ship.thrust_kd * v_fwd)
                .clamp(0.0, ship.thrust);
            vel += forward * pd_force * dt;
        }

        // Angular dynamics — torque impulse + exponential drag.
        if turn.abs() > f32::EPSILON {
            ang_vel += -ship.torque * turn * dt;
        }
        ang_vel *= (-ship.angular_drag * dt).exp();

        pos += vel * dt;
        heading += ang_vel * dt;

        let d_remaining = (target_pos - pos).length();
        if d_remaining < landing_radius && vel.length() < landing_speed {
            return Some(tick as f32 * dt);
        }
    }
    None
}

/// Every ship in `assets/ships.yaml` should be able to land on a stationary
/// target starting from rest at a distance of `10 · max_speed`, within a
/// generous budget derived from its own dynamics. Regression test against
/// the hauler-circling / PD-too-aggressive failure mode.
#[test]
fn test_all_ships_can_land() {
    let yaml = std::fs::read_to_string("assets/ships.yaml")
        .expect("failed to read assets/ships.yaml");
    let ships: std::collections::HashMap<String, SimShip> =
        serde_yaml::from_str(&yaml).expect("failed to parse ships.yaml");
    assert!(!ships.is_empty(), "no ships loaded from assets/ships.yaml");

    let mut failures = Vec::new();
    for (name, ship) in &ships {
        let distance = 10.0 * ship.max_speed;
        // Theoretical minimum = cruise + accel/brake + one full 180° turn.
        let theoretical_min = distance / ship.max_speed
            + 2.0 * ship.max_speed / ship.thrust
            + PI * ship.angular_drag / ship.torque;
        // Generous budget: 5× theoretical minimum.
        let budget = 5.0 * theoretical_min;

        match simulate_landing(ship, distance, budget, 0xC0FFEE_u64) {
            Some(t) => println!(
                "{name}: landed in {t:.2}s (budget {budget:.2}s, minimum {theoretical_min:.2}s)",
            ),
            None => failures.push((name.clone(), budget)),
        }
    }
    assert!(
        failures.is_empty(),
        "ships failed to land within budget: {failures:?}"
    );
}
