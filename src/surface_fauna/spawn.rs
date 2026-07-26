//! Keeping a small fauna population alive around the player.
//!
//! Critters are spawned just off-screen and recycled once they drift far
//! enough away (see `behaviour::run_fauna`), so the world feels populated
//! without ever holding more than [`TARGET_FAUNA_COUNT`] of them.

use avian2d::prelude::*;
use bevy::prelude::*;
use rand::Rng;

use super::{
    FLY_ALTITUDE, FLY_ALTITUDE_INSIDE, FLY_Z, Fauna, FaunaState, FaunaWorld, FlierShadow,
    SPAWN_MAX_DIST, SPAWN_MIN_DIST, TARGET_FAUNA_COUNT,
};
use crate::PlayState;
use crate::surface::Walker;
use crate::surface_character::CharacterAnim;
use crate::surface_objects::depth_z;
use crate::surface_pathfinding::SurfaceCostMap;

pub fn spawn_fauna(
    mut commands: Commands,
    world: Option<ResMut<FaunaWorld>>,
    time: Res<Time>,
    walker: Query<&Transform, With<Walker>>,
    existing: Query<(), With<Fauna>>,
) {
    let Some(mut world) = world else { return };
    world.spawn_timer.tick(time.delta());
    if !world.spawn_timer.just_finished() {
        return;
    }
    if existing.iter().count() >= TARGET_FAUNA_COUNT {
        return;
    }
    let Ok(player) = walker.single() else { return };
    let ppos = player.translation.truncate();
    let world = &mut *world; // reborrow for disjoint field access (rng vs grid)

    // Pick a species, then a valid home tile off-screen from the player.
    let sp_idx = world.rng.r#gen_range(0..world.species.len());
    let idxs = world.species[sp_idx].terrain_idxs.clone();
    let (map_w, map_h) = (world.grid.w, world.grid.h);
    let mut anchor = None;
    for _ in 0..40 {
        let tx = world.rng.r#gen_range(0..map_w) as i32;
        let ty = world.rng.r#gen_range(0..map_h) as i32;
        if !world.grid.is_home(&idxs, tx, ty) {
            continue;
        }
        let wp = SurfaceCostMap::tile_to_world(tx as u32, ty as u32);
        let d = wp.distance(ppos);
        if (SPAWN_MIN_DIST..SPAWN_MAX_DIST).contains(&d) {
            anchor = Some((tx, ty));
            break;
        }
    }
    let Some((ax, ay)) = anchor else { return };

    // Herd species spawn a small cluster around the anchor.
    let group = world.species[sp_idx].group;
    for i in 0..group {
        let (tx, ty) = if i == 0 {
            (ax, ay)
        } else {
            let mut found = (ax, ay);
            for _ in 0..8 {
                let cx = ax + world.rng.r#gen_range(-2..=2);
                let cy = ay + world.rng.r#gen_range(-2..=2);
                if world.grid.is_home(&idxs, cx, cy) {
                    found = (cx, cy);
                    break;
                }
            }
            found
        };
        let ground = SurfaceCostMap::tile_to_world(tx as u32, ty as u32);
        let graze = world.rng.r#gen_range(1.2..3.5);
        let bearing = world.rng.r#gen_range(0.0..std::f32::consts::TAU);
        let phase = world.rng.r#gen_range(0.0..std::f32::consts::TAU);
        spawn_one(&mut commands, world, sp_idx, ground, graze, bearing, phase);
    }
}

/// Spawn a single critter of `sp_idx` standing on `ground`.
fn spawn_one(
    commands: &mut Commands,
    world: &FaunaWorld,
    sp_idx: usize,
    ground: Vec2,
    graze: f32,
    bearing: f32,
    phase: f32,
) {
    let sp = &world.species[sp_idx];
    let flier = sp.flier;
    let altitude = if world.scope == PlayState::Inside {
        FLY_ALTITUDE_INSIDE
    } else {
        FLY_ALTITUDE
    };
    // Fliers float above their ground track and sort over everything.
    let pos = if flier {
        Vec2::new(ground.x, ground.y + altitude)
    } else {
        ground
    };
    let z = if flier {
        FLY_Z
    } else {
        depth_z(pos.y - sp.foot_off)
    };
    let scope = world.scope.clone();
    let mut ent = commands.spawn((
        DespawnOnExit(scope.clone()),
        Fauna {
            species: sp_idx,
            speed: sp.speed,
            flee_speed: sp.flee_speed,
            foot_off: sp.foot_off,
            flier,
            state: if flier {
                FaunaState::Wander
            } else {
                FaunaState::Graze
            },
            timer: Timer::from_seconds(graze, TimerMode::Once),
            target: pos,
            heading: Vec2::from_angle(bearing),
            altitude,
            turn: 0.0,
            phase,
        },
        CharacterAnim::legacy_rpg(if flier { 0.11 } else { 0.16 }),
        RigidBody::Dynamic,
        LockedAxes::ROTATION_LOCKED,
        LinearDamping(if flier { 4.0 } else { 12.0 }),
        LinearVelocity(Vec2::ZERO),
        Sprite::from_atlas_image(
            sp.image.clone(),
            TextureAtlas {
                layout: sp.layout.clone(),
                index: 0,
            },
        ),
        Transform::from_xyz(pos.x, pos.y, z),
    ));

    // Roamers collide with the terrain; fliers pass over everything but still
    // need explicit mass (a dynamic body with no collider is mass-less → avian
    // "no mass or inertia" NaN warning).
    if flier {
        ent.insert(MassPropertiesBundle::from_shape(
            &Collider::circle(4.0),
            0.5,
        ));
        // The shadow is a child, so it tracks the flier horizontally for free
        // and is despawned along with it.  `depth_sort_fauna` owns its offset
        // and z — which way "down" points depends on the interior shear.
        ent.with_child((
            FlierShadow,
            Sprite::from_image(world.shadow.clone()),
            Transform::from_scale(Vec3::splat(sp.shadow_scale)),
        ));
    } else {
        if scope == PlayState::Inside {
            ent.insert(crate::surface::interiors::InteriorScoped);
        }
        ent.insert((
            Collider::circle(4.0),
            CollisionLayers::new(crate::GameLayer::Character, [crate::GameLayer::Surface]),
        ));
    }
}
