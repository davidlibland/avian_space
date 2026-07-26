//! The fauna state machine.
//!
//! Ground roamers cycle graze → wander → (bolt → watch) → graze; fliers cruise
//! a steered heading and scatter when the player walks through them.  The two
//! are different enough that they get separate functions rather than one match
//! with `if flier` scattered through it.

use avian2d::prelude::*;
use bevy::prelude::*;
use rand::{Rng, rngs::StdRng};
use std::collections::HashMap;

use super::{
    ARRIVE_DIST, CALM_DIST, DESPAWN_DIST, FLEE_RADIUS, FLY_BOB_PX, Fauna, FaunaState, FaunaWorld,
    HERD_BIAS, HomeGrid, RESPOOK_RADIUS, WANDER_TILES, tile_of,
};
use crate::surface::Walker;
use crate::surface_character::{CharacterAnim, Facing};
use crate::surface_pathfinding::SurfaceCostMap;

// ── Tuning ────────────────────────────────────────────────────────────────

/// A bolt lasts this long before the animal stops to look back.
const BOLT_SECS: std::ops::Range<f32> = 0.7..1.5;
/// How long it watches the player before settling (if he stays away).
const ALERT_SECS: std::ops::Range<f32> = 0.9..2.2;
/// How far ahead a bolt heading is tested for open ground (≈3 tiles).
const BOLT_LOOKAHEAD: f32 = 96.0;
/// One animal bolting spooks its own species within this radius.
const HERD_PANIC_RADIUS: f32 = 150.0;
/// Fraction of graze breaks that are a short shuffle rather than a full hop.
const NIBBLE_CHANCE: f64 = 0.6;
/// Length (px) of that shuffle.
const NIBBLE_PX: std::ops::Range<f32> = 10.0..28.0;
/// A wander is abandoned after this long, whether or not it arrived.
const WANDER_TIMEOUT: f32 = 6.0;

/// Typical flier turn rate (rad/s), and how long a turn persists before it
/// decays toward a new one.  Bounding the turn rate is what turns a random
/// walk into a flight path.
const FLY_TURN_RATE: f32 = 1.2;
const FLY_TURN_TAU: f32 = 0.7;
/// Fliers start steering back inside once this close to the map edge.
const FLY_MARGIN: f32 = 96.0;
/// The player walking this close scatters fliers.
const FLY_SCATTER_RADIUS: f32 = 90.0;
/// Speed multiplier while scattering.
const FLY_SCATTER_BOOST: f32 = 1.8;
/// Altitude bob rate (Hz) and the flap-driven speed pulse depth.
const FLY_BOB_HZ: f32 = 0.5;
const FLY_SURGE: f32 = 0.22;

// ── System ────────────────────────────────────────────────────────────────

pub fn run_fauna(
    mut commands: Commands,
    time: Res<Time>,
    world: Option<ResMut<FaunaWorld>>,
    walker: Query<&Transform, With<Walker>>,
    mut q: Query<
        (
            Entity,
            &mut Fauna,
            &Transform,
            &mut LinearVelocity,
            &mut CharacterAnim,
        ),
        Without<Walker>,
    >,
) {
    let Some(mut world) = world else { return };
    let FaunaWorld {
        species, grid, rng, ..
    } = &mut *world;
    let dt = time.delta_secs();
    let ppos = walker.single().ok().map(|t| t.translation.truncate());

    // Read-only pass: herd centroids (so groups drift together) and the
    // positions of anyone already bolting (so the rest of the herd catches the
    // panic instead of grazing on while a neighbour sprints past).
    let mut sums: HashMap<usize, (Vec2, f32)> = HashMap::new();
    let mut panic: HashMap<usize, Vec<Vec2>> = HashMap::new();
    for (_, fauna, tf, _, _) in q.iter() {
        if fauna.flier {
            continue;
        }
        let p = tf.translation.truncate();
        let e = sums.entry(fauna.species).or_insert((Vec2::ZERO, 0.0));
        e.0 += p;
        e.1 += 1.0;
        if fauna.state == FaunaState::Flee {
            panic.entry(fauna.species).or_default().push(p);
        }
    }
    let centroids: HashMap<usize, Vec2> = sums.iter().map(|(k, (s, n))| (*k, *s / *n)).collect();

    for (entity, mut fauna, tf, mut vel, mut anim) in &mut q {
        let pos = tf.translation.truncate();

        // Recycle anything that has drifted far off-screen.
        if let Some(pp) = ppos
            && pos.distance(pp) > DESPAWN_DIST
        {
            commands.entity(entity).despawn();
            continue;
        }

        let Some(sp) = species.get(fauna.species) else {
            continue;
        };
        if fauna.flier {
            fly(&mut fauna, pos, &mut vel, dt, ppos, grid, rng);
            continue;
        }

        let idxs = &sp.terrain_idxs;
        let spooked = match (ppos, fauna.state) {
            (Some(pp), FaunaState::Graze | FaunaState::Wander) => {
                pos.distance(pp) < FLEE_RADIUS
                    || panic
                        .get(&fauna.species)
                        .is_some_and(|ps| ps.iter().any(|p| p.distance(pos) < HERD_PANIC_RADIUS))
            }
            _ => false,
        };
        if spooked {
            start_bolt(&mut fauna, pos, ppos, idxs, grid, rng);
        }

        match fauna.state {
            FaunaState::Flee => {
                vel.0 = fauna.heading * fauna.flee_speed;
                fauna.timer.tick(time.delta());
                if fauna.timer.is_finished() {
                    fauna.state = FaunaState::Alert;
                    fauna.timer = Timer::from_seconds(rng.r#gen_range(ALERT_SECS), TimerMode::Once);
                }
            }
            FaunaState::Alert => {
                vel.0 = Vec2::ZERO;
                // Standing still means `animate_characters` will not touch the
                // facing, so point it at the player by hand: a deer looking
                // back at you is the whole point of stopping.
                let dist = match ppos {
                    Some(pp) => {
                        anim.facing = Facing::from_velocity(pp - pos);
                        pos.distance(pp)
                    }
                    None => f32::INFINITY,
                };
                if dist < RESPOOK_RADIUS {
                    start_bolt(&mut fauna, pos, ppos, idxs, grid, rng);
                } else {
                    fauna.timer.tick(time.delta());
                    if fauna.timer.is_finished() {
                        if dist > CALM_DIST {
                            fauna.state = FaunaState::Graze;
                            fauna.timer =
                                Timer::from_seconds(rng.r#gen_range(1.2..3.0), TimerMode::Once);
                        } else {
                            // Still uncomfortably close — keep watching.
                            fauna.timer =
                                Timer::from_seconds(rng.r#gen_range(ALERT_SECS), TimerMode::Once);
                        }
                    }
                }
            }
            FaunaState::Graze => {
                vel.0 = Vec2::ZERO;
                fauna.timer.tick(time.delta());
                if fauna.timer.is_finished() {
                    let centroid = centroids
                        .get(&fauna.species)
                        .copied()
                        .filter(|_| sp.group > 1);
                    match graze_break(pos, idxs, grid, rng, centroid) {
                        Some(t) => {
                            fauna.target = t;
                            fauna.state = FaunaState::Wander;
                            fauna.timer = Timer::from_seconds(WANDER_TIMEOUT, TimerMode::Once);
                        }
                        None => fauna.timer = Timer::from_seconds(1.5, TimerMode::Once),
                    }
                }
            }
            FaunaState::Wander => {
                // Give up as well as arrive: the target is picked by terrain
                // type, which says nothing about the props and buildings in
                // the way, so an animal can end up walking into a collider
                // forever waiting to "arrive".
                fauna.timer.tick(time.delta());
                let to = fauna.target - pos;
                if to.length() < ARRIVE_DIST || fauna.timer.is_finished() {
                    fauna.state = FaunaState::Graze;
                    fauna.timer = Timer::from_seconds(rng.r#gen_range(1.5..4.0), TimerMode::Once);
                    vel.0 = Vec2::ZERO;
                } else {
                    fauna.heading = to.normalize_or_zero();
                    vel.0 = fauna.heading * fauna.speed;
                }
            }
        }
    }
}

// ── Ground roamers ────────────────────────────────────────────────────────

/// Send `f` bolting away from the player (or from whatever spooked the herd).
fn start_bolt(
    f: &mut Fauna,
    pos: Vec2,
    ppos: Option<Vec2>,
    idxs: &[u32],
    grid: &HomeGrid,
    rng: &mut StdRng,
) {
    let away = ppos
        .map(|pp| (pos - pp).normalize_or_zero())
        .filter(|v| *v != Vec2::ZERO)
        .unwrap_or(f.heading);
    f.heading = bolt_heading(pos, away, idxs, grid);
    f.state = FaunaState::Flee;
    f.timer = Timer::from_seconds(rng.r#gen_range(BOLT_SECS), TimerMode::Once);
}

/// Straight away from the threat if that is open ground, else the nearest arc
/// that is.  Without the check an animal cornered against a cliff grinds into
/// the collider for the whole bolt, which reads as broken rather than scared.
fn bolt_heading(pos: Vec2, away: Vec2, idxs: &[u32], grid: &HomeGrid) -> Vec2 {
    let base = away.to_angle();
    for spread in [0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2, 1.6, -1.6] {
        let dir = Vec2::from_angle(base + spread);
        if grid.is_home_at(idxs, pos + dir * BOLT_LOOKAHEAD) {
            return dir;
        }
    }
    away
}

/// Where a grazing animal goes when it lifts its head.  Mostly a shuffle of a
/// few px (feeding along), occasionally a real hop to a new patch — an animal
/// whose every move is a 5-tile march looks like it is on patrol.
fn graze_break(
    pos: Vec2,
    idxs: &[u32],
    grid: &HomeGrid,
    rng: &mut StdRng,
    centroid: Option<Vec2>,
) -> Option<Vec2> {
    if rng.r#gen_bool(NIBBLE_CHANCE) {
        let dir = Vec2::from_angle(rng.r#gen_range(0.0..std::f32::consts::TAU));
        let t = pos + dir * rng.r#gen_range(NIBBLE_PX);
        return grid.is_home_at(idxs, t).then_some(t);
    }
    // Herd species bias the search toward the flock centroid so they regroup.
    let (mut cx, mut cy) = tile_of(pos);
    if let Some(cen) = centroid {
        let (ctx, cty) = tile_of(cen);
        cx += ((ctx - cx) as f32 * HERD_BIAS).round() as i32;
        cy += ((cty - cy) as f32 * HERD_BIAS).round() as i32;
    }
    for _ in 0..16 {
        let nx = cx + rng.r#gen_range(-WANDER_TILES..=WANDER_TILES);
        let ny = cy + rng.r#gen_range(-WANDER_TILES..=WANDER_TILES);
        if grid.is_home(idxs, nx, ny) {
            return Some(SurfaceCostMap::tile_to_world(nx as u32, ny as u32));
        }
    }
    None
}

// ── Fliers ────────────────────────────────────────────────────────────────

/// Fliers ignore terrain and never graze.  They hold a heading and steer it,
/// which is what makes their tracks read as flight; the previous version drew
/// a fresh random bearing on arrival and came out as a zigzag.
fn fly(
    f: &mut Fauna,
    pos: Vec2,
    vel: &mut LinearVelocity,
    dt: f32,
    ppos: Option<Vec2>,
    grid: &HomeGrid,
    rng: &mut StdRng,
) {
    f.phase = (f.phase + dt * FLY_BOB_HZ * std::f32::consts::TAU) % std::f32::consts::TAU;

    // Ornstein-Uhlenbeck turn: decay what we were doing toward a new random
    // rate.  Framed this way it is frame-rate independent and its stationary
    // spread is exactly FLY_TURN_RATE, so the constant means what it says.
    let decay = (-dt / FLY_TURN_TAU).exp();
    f.turn = f.turn * decay
        + rng.r#gen_range(-FLY_TURN_RATE..FLY_TURN_RATE) * (1.0 - decay * decay).sqrt();
    let mut dir = Vec2::from_angle(f.heading.to_angle() + f.turn * dt);

    // Turn back before reaching the edge rather than pressing against it.
    let he = grid.half_extent();
    let ground_y = pos.y - f.altitude;
    let inward = Vec2::new(edge_push(pos.x, he.x), edge_push(ground_y, he.y));
    if inward != Vec2::ZERO {
        dir = (dir + inward.normalize() * 2.0).normalize_or_zero();
    }

    // Walking into a cloud of butterflies should scatter it.
    let mut speed = f.speed;
    if let Some(pp) = ppos
        && pos.distance(pp) < FLY_SCATTER_RADIUS
    {
        dir = (dir + (pos - pp).normalize_or_zero() * 2.5).normalize_or_zero();
        speed *= FLY_SCATTER_BOOST;
    }

    f.heading = dir;
    // Flap-glide surge, plus a gentle climb/dive. The bob goes through the
    // VELOCITY rather than the transform because avian owns the transform —
    // writing translation here would be overwritten by the physics sync.
    let surge = 1.0 + FLY_SURGE * (f.phase * 2.0).sin();
    let climb = FLY_BOB_PX * FLY_BOB_HZ * std::f32::consts::TAU * f.phase.cos();
    vel.0 = dir * speed * surge + Vec2::Y * climb;
}

/// -1 / 0 / +1 nudge back toward the middle when within the margin of an edge.
fn edge_push(v: f32, half: f32) -> f32 {
    if v > half - FLY_MARGIN {
        -1.0
    } else if v < -half + FLY_MARGIN {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::{WORLD_HEIGHT, WORLD_WIDTH};
    use rand::SeedableRng;

    const HOME: u32 = 1;
    const WALL: u32 = 0;

    /// A world that is home terrain only WEST of `wall_x`, so a bolt heading
    /// east has to be rejected.
    fn cliff_at(wall_x: u32) -> HomeGrid {
        let mut terrain = vec![WALL; (WORLD_WIDTH * WORLD_HEIGHT) as usize];
        for ty in 0..WORLD_HEIGHT {
            for tx in 0..wall_x {
                terrain[(ty * WORLD_WIDTH + tx) as usize] = HOME;
            }
        }
        HomeGrid {
            terrain,
            w: WORLD_WIDTH,
            h: WORLD_HEIGHT,
        }
    }

    fn open() -> HomeGrid {
        HomeGrid {
            terrain: vec![HOME; (WORLD_WIDTH * WORLD_HEIGHT) as usize],
            w: WORLD_WIDTH,
            h: WORLD_HEIGHT,
        }
    }

    #[test]
    fn bolt_runs_straight_away_over_open_ground() {
        let grid = open();
        let dir = bolt_heading(Vec2::ZERO, Vec2::X, &[HOME], &grid);
        assert!(dir.dot(Vec2::X) > 0.99, "expected due east, got {dir:?}");
    }

    #[test]
    fn bolt_turns_aside_rather_than_into_a_wall() {
        // Standing just west of the cliff, with the player to the west: due
        // east is "away" but lands on wall, so the animal must veer.
        let grid = cliff_at(WORLD_WIDTH / 2);
        let pos = SurfaceCostMap::tile_to_world(WORLD_WIDTH / 2 - 2, WORLD_HEIGHT / 2);
        let dir = bolt_heading(pos, Vec2::X, &[HOME], &grid);
        assert!(
            grid.is_home_at(&[HOME], pos + dir * BOLT_LOOKAHEAD),
            "bolt heading {dir:?} lands off home terrain"
        );
        assert!(dir.dot(Vec2::X) < 0.9, "expected a veer, got {dir:?}");
    }

    #[test]
    fn graze_breaks_stay_on_home_terrain() {
        let grid = cliff_at(WORLD_WIDTH / 2);
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let pos = SurfaceCostMap::tile_to_world(WORLD_WIDTH / 2 - 3, WORLD_HEIGHT / 2);
        for _ in 0..200 {
            if let Some(t) = graze_break(pos, &[HOME], &grid, &mut rng, None) {
                assert!(grid.is_home_at(&[HOME], t), "wandered onto wall at {t:?}");
            }
        }
    }

    #[test]
    fn graze_breaks_are_mostly_short_shuffles() {
        // The point of the nibble is that an animal does not march five tiles
        // every single time it lifts its head.
        let grid = open();
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let pos = SurfaceCostMap::tile_to_world(WORLD_WIDTH / 2, WORLD_HEIGHT / 2);
        let short = (0..400)
            .filter_map(|_| graze_break(pos, &[HOME], &grid, &mut rng, None))
            .filter(|t| t.distance(pos) <= NIBBLE_PX.end)
            .count();
        assert!((180..320).contains(&short), "{short}/400 were shuffles");
    }

    #[test]
    fn fliers_are_steered_back_from_the_map_edge() {
        let half = 100.0;
        assert_eq!(edge_push(half - 10.0, half), -1.0);
        assert_eq!(edge_push(-half + 10.0, half), 1.0);
        assert_eq!(edge_push(0.0, half), 0.0);
    }
}
