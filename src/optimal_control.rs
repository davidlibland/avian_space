use bevy::prelude::*;
use std::f32::consts::PI;
// This file contains code to help compute optimal control for the ships.

/// Compute the stopping position under maximum braking from (x_now, v_now).
///
/// Dynamics: dv/dt = u - gamma*v, braking control u = -a*sign(v).
/// Terminal velocity under braking: v_term = u/gamma (opposite sign to v_now).
/// Stop time: t* = (1/gamma) * ln(1 - v_now/v_term)
/// Displacement: dx = v_term*t* - (v_now - v_term)/gamma * (exp(-gamma*t*) - 1)
pub fn x_stop(v_now: f32, a: f32, gamma: f32) -> f32 {
    if v_now.abs() < 1e-6 {
        return 0.0;
    }
    // Braking control opposes current velocity
    let u = -a * v_now.signum();

    // terminal velocity under u
    let v_term = u / gamma;

    // Time to reach v=0:
    // v(t) = u/gamma + (v_now - u/gamma) * exp(-gamma * t) = 0
    // => exp(-gamma * t) = (u/gamma) / (u/gamma - v_now)
    // => t* = (1/gamma) * ln((u/gamma - v_now) / (u/gamma))
    //       = (1/gamma) * ln(1 - gamma * v_now / u)
    let t_stop = (1.0 / gamma) * (1.0 - v_now / v_term).ln();

    // x(t*) = x_now + v_term * t* + (v_now - v_term) / gamma * (1 - exp(-gamma * t*))
    // but exp(-gamma * t*) = v_term / (v_term - v_now) ... wait, let's be explicit:
    // x(t) = x_now + (u/gamma)*t - (1/gamma)*(v_now - u/gamma)*(exp(-gamma*t) - 1)
    let exp_term = (-gamma * t_stop).exp();
    v_term * t_stop - (v_now - v_term) / gamma * (exp_term - 1.0)
}

/// Bang-bang turn command toward a desired `target_angle` (ego frame).
///
/// Returns `-1.0`, `0.0`, or `+1.0` (the `turn` field of `ShipCommand`).
/// Logic: full torque toward the target until `x_stop` predicts the ship's
/// rotation momentum will carry it past, then reverse torque to brake.
/// A small deadband near zero lets angular damping bring the ship to rest.
///
/// `angular_drag` is the exponential decay rate of angular velocity
/// (matches `ShipData.angular_drag`). `torque` is the maximum braking torque.
pub fn turn_to_angle(target_angle: f32, ang_vel: f32, torque: f32, angular_drag: f32) -> f32 {
    let stop_angle = x_stop(ang_vel, torque, angular_drag);
    if -PI / 16. < target_angle && target_angle < PI / 16. {
        0.0
    } else if target_angle > 0.0 {
        if stop_angle < target_angle { -1.0 } else { 1.0 }
    } else {
        if stop_angle > target_angle { 1.0 } else { -1.0 }
    }
}

/// Ship-only control features (no target needed). Intended for use as
/// observation inputs to a learned policy, in addition to driving
/// [`pursuit_controls_ego`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ControlFeatures {
    /// Angle (rad) the ship would rotate through under maximum braking
    /// torque before coming to rest, given its current angular velocity.
    /// Same sign as `ang_vel`; zero when `ang_vel = 0`.
    pub stop_angle: f32,
}

/// Compute ship-only control features (currently just `stop_angle`).
pub fn control_features(ang_vel: f32, torque: f32, angular_drag: f32) -> ControlFeatures {
    ControlFeatures {
        stop_angle: x_stop(ang_vel, torque, angular_drag),
    }
}

/// Per-target pursuit features derived from the PD law in ego coordinates.
#[derive(Clone, Copy, Debug, Default)]
pub struct PursuitFeatures {
    /// Bearing (rad) to the PD-desired acceleration direction, in ego frame.
    /// 0 = straight ahead, positive = left, range `[−π, π]`. Tie-broken at
    /// the ±π branch cut using `ang_vel` so the value is sign-stable.
    pub target_angle: f32,
    /// `max(cos(target_angle), 0)` — fraction of applied thrust that actually
    /// contributes toward the desired direction (0 when facing away).
    pub alignment: f32,
    /// PWM-style firing probability: `alignment · |a_desired| / a_max`,
    /// clamped to `[0, 1]`.
    pub thrust_prob: f32,
}

/// Compute the PD-based pursuit features for a single target. Used by both
/// [`pursuit_controls_ego`] (to drive control) and the observation encoder
/// (as auxiliary features for a learned policy).
///
/// Args and conventions mirror [`pursuit_controls_ego`].
pub fn pursuit_features_ego(
    x_target: Vec2,
    v_target: Vec2,
    ang_vel: f32,
    torque: f32,
    angular_drag: f32,
    a_max: f32,
    v_max: f32,
    damping_factor: f32,
) -> PursuitFeatures {
    // PD gain scaled with ship dynamics. Two concerns:
    //  (a) `K_X_CONST` sets the linear-axis time scale so a typical
    //      ship (thrust=100, max_speed=200) lands at `k_x ≈ 1`,
    //      roughly matching the previous hardcoded value of 0.9.
    //  (b) The PD's settling time `1/√k_x` must not be faster than
    //      the ship's 180° turn time `π·angular_drag/torque`, or the
    //      desired-direction vector rotates faster than the ship can
    //      physically track — manifests as circling (e.g. hauler).
    const K_X_CONST: f32 = 400.0;
    let k_x_linear = K_X_CONST * a_max / (v_max * v_max).max(1e-6);
    let turn_rate = torque / (PI * angular_drag).max(1e-6);
    let k_x_angular = turn_rate * turn_rate;
    let k_x = k_x_linear.min(k_x_angular);

    let k_v = damping_factor * 2.0 * k_x.sqrt();
    // Tracking PD: a = k_x·(x_target − x_ego) + k_v·(v_target − v_ego).
    // Callers pass both as (target − ego), so both gains add.
    let acceleration = k_x * x_target + k_v * v_target;
    let accel_mag = acceleration.length();
    let direction = acceleration / accel_mag.max(1e-6);
    let raw_angle = direction.y.atan2(direction.x);

    // `atan2` has a branch cut at ±π: if the desired direction points backward,
    // tiny perturbations flip `raw_angle` between +π and −π, which flips the
    // bang-bang turn sign and causes oscillation. Commit to a consistent sign
    // near the cut, tie-broken by current angular velocity (else +π).
    let near_back = raw_angle.abs() > PI - PI / 16.0;
    let target_angle = if near_back {
        if ang_vel >= 0.0 { PI } else { -PI }
    } else {
        raw_angle
    };

    // Thrust weighted by alignment: the component of thrust that actually
    // contributes to the desired acceleration is `cos(target_angle)`. When
    // pointing behind the desired direction this is negative → clamp to 0.
    let alignment = target_angle.cos().max(0.0);
    let thrust_prob = (alignment * accel_mag / a_max.max(1e-6)).clamp(0.0, 1.0);

    PursuitFeatures {
        target_angle,
        alignment,
        thrust_prob,
    }
}

/// PD-based pursuit controller in the ego frame. Combines
/// [`control_features`] (for `stop_angle`) and [`pursuit_features_ego`]
/// (for `target_angle` + `thrust_prob`) into concrete commands.
///
/// Returns `(turn, thrust_prob)`:
/// * `turn ∈ {-1, 0, +1}` — bang-bang angular control with a small deadband
///   and a stop-angle-aware coast branch.
/// * `thrust_prob ∈ [0, 1]` — PWM-style firing probability; sampling this
///   at the call site turns a binary actuator into proportional control.
///
/// `damping_factor` scales the velocity term in the PD law. `1.0` gives
/// critical damping (landing — stop on target); smaller values let the ship
/// overshoot before turning to chase (good for pickups/pursuit).
pub fn pursuit_controls_ego(
    x_target: Vec2,
    v_target: Vec2,
    ang_vel: f32,
    torque: f32,
    angular_drag: f32,
    a_max: f32,
    v_max: f32,
    damping_factor: f32,
) -> (f32, f32) {
    let feat = pursuit_features_ego(
        x_target,
        v_target,
        ang_vel,
        torque,
        angular_drag,
        a_max,
        v_max,
        damping_factor,
    );
    let turn = turn_to_angle(feat.target_angle, ang_vel, torque, angular_drag);
    (turn, feat.thrust_prob)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "tests/optimal_control_tests.rs"]
mod tests;
