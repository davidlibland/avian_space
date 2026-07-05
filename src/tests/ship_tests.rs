//! Tests for ship handling / movement basics: the damage→handling curve and
//! the drive-flame (exhaust) flag driven by `ShipCommand`s.

use super::*;
use bevy::time::TimeUpdateStrategy;
use std::time::Duration;

// ── handling_factor ─────────────────────────────────────────────────────────

fn ship_with_health(health: i32, max_health: i32) -> Ship {
    let mut ship = Ship::from_ship_data(
        &ShipData {
            max_health,
            ..Default::default()
        },
        "test_ship",
    );
    ship.health = health;
    ship
}

#[test]
fn handling_factor_full_health_is_one() {
    assert!((ship_with_health(100, 100).handling_factor() - 1.0).abs() < 1e-6);
}

#[test]
fn handling_factor_sublinear_sqrt_rolloff() {
    // 0.5 + 0.5*sqrt(frac): quarter health → 0.75, not 0.625 (linear would).
    let hf = ship_with_health(25, 100).handling_factor();
    assert!((hf - 0.75).abs() < 1e-6, "expected 0.75, got {hf}");
}

#[test]
fn handling_factor_floors_at_half() {
    assert!((ship_with_health(0, 100).handling_factor() - 0.5).abs() < 1e-6);
    // Negative health (overkill damage) must clamp, not NaN.
    let hf = ship_with_health(-30, 100).handling_factor();
    assert!((hf - 0.5).abs() < 1e-6, "negative health must clamp: {hf}");
}

#[test]
fn handling_factor_zero_max_health_no_panic() {
    let hf = ship_with_health(0, 0).handling_factor();
    assert!(hf.is_finite(), "max_health 0 must not divide by zero: {hf}");
}

// ── DriveActive (exhaust plume) ──────────────────────────────────────────────

fn movement_app() -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_millis(
            16,
        )))
        .add_message::<ShipCommand>()
        .add_systems(Update, ship_movement);
    // Ship::default() is the all-zero sentinel — give it real drive params so
    // the PD thrust controller actually produces force.
    let ship = Ship::from_ship_data(
        &ShipData {
            thrust: 200.0,
            max_speed: 300.0,
            torque: 20.0,
            max_health: 100,
            thrust_kp: 5.0,
            thrust_kd: 1.0,
            ..Default::default()
        },
        "test_ship",
    );
    let entity = app
        .world_mut()
        .spawn((
            ship,
            DriveActive(false),
            RigidBody::Dynamic,
            LinearVelocity(Vec2::ZERO),
            AngularVelocity(0.0),
            Transform::default(),
        ))
        .id();
    app.update(); // establish time baseline
    (app, entity)
}

fn send_command(app: &mut App, entity: Entity, thrust: f32) {
    app.world_mut()
        .write_message(ShipCommand {
            entity,
            thrust,
            turn: 0.0,
            reverse: 0.0,
        });
}

/// Regression: the plume must track the LAST command — including an all-zero
/// one. (The player input layer once skipped sending zero-input commands, so
/// the flame stayed latched on after the throttle was released.)
#[test]
fn drive_flag_follows_thrust_commands() {
    let (mut app, entity) = movement_app();

    send_command(&mut app, entity, 1.0);
    app.update();
    assert!(
        app.world().get::<DriveActive>(entity).unwrap().0,
        "thrust command must light the drive flame"
    );

    send_command(&mut app, entity, 0.0);
    app.update();
    assert!(
        !app.world().get::<DriveActive>(entity).unwrap().0,
        "zero-thrust command must clear the drive flame"
    );
}

#[test]
fn thrust_accelerates_forward() {
    let (mut app, entity) = movement_app();
    send_command(&mut app, entity, 1.0);
    app.update();
    let vel = app.world().get::<LinearVelocity>(entity).unwrap().0;
    // Transform::default() faces +Y.
    assert!(vel.y > 0.0, "forward thrust must accelerate +Y, got {vel:?}");
    assert!(vel.x.abs() < 1e-4);
}
