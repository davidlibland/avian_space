//! Civilian NPC spawning and path-following on the planet surface.
//!
//! Civilians spawn at a random building door, follow a precomputed path
//! to another building, then despawn.  On average ~2 civilians are walking
//! at any given time.

use avian2d::prelude::*;
use bevy::prelude::*;
use rand::{Rng, SeedableRng};
use serde::Deserialize;

use crate::PlayState;
use crate::surface::{BuildingKind, TILE_PX, Walker};
use crate::surface_objects::depth_z;
use crate::surface_pathfinding::SurfacePaths;

// ── Constants ────────────────────────────────────────────────────────────

/// Target average number of civilians walking at once.
const TARGET_CIVILIAN_COUNT: f32 = 2.0;

// ── Components ───────────────────────────────────────────────────────────

use crate::character_compositor::CharacterLayers;
use crate::surface_character::CharacterAnim;
use crate::surface_npc::{Behavior, Npc, NpcBehavior};

// ── Resource ─────────────────────────────────────────────────────────────

/// Spawn timer — controls spawn rate to maintain TARGET_CIVILIAN_COUNT.
#[derive(Resource)]
pub struct CivilianSpawnTimer {
    timer: Timer,
    rng: rand::rngs::StdRng,
}

// ── Public setup ─────────────────────────────────────────────────────────

/// Insert the civilian spawn timer.  Call from `setup_surface`.
/// (Sprites now come from the global [`CharacterLayers`] compositor.)
pub fn setup_civilians(commands: &mut Commands, seed: u64) {
    commands.insert_resource(CivilianSpawnTimer {
        timer: Timer::from_seconds(2.0, TimerMode::Repeating),
        rng: rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(0xC1C1_0001)),
    });
}

// ── Systems ──────────────────────────────────────────────────────────────

/// Periodically spawn civilian NPCs if below the target count.
pub fn spawn_civilians(
    mut commands: Commands,
    layers: Option<ResMut<CharacterLayers>>,
    mut images: ResMut<Assets<Image>>,
    paths_res: Option<Res<SurfacePaths>>,
    mut spawn_timer: ResMut<CivilianSpawnTimer>,
    time: Res<Time>,
    existing: Query<(), With<Npc>>,
) {
    let (Some(mut layers), Some(paths)) = (layers, paths_res) else {
        return;
    };
    if paths.paths.is_empty() || layers.items.is_empty() {
        return;
    }

    spawn_timer.timer.tick(time.delta());
    if !spawn_timer.timer.just_finished() {
        return;
    }

    let current_count = existing.iter().count();
    if current_count as f32 >= TARGET_CIVILIAN_COUNT {
        return;
    }

    let rng = &mut spawn_timer.rng;

    // Pick a random starting path for the patrol.
    let path_keys: Vec<&(BuildingKind, BuildingKind)> = paths.paths.keys().collect();
    if path_keys.is_empty() {
        return;
    }
    let key_idx = rng.r#gen_range(0..path_keys.len());
    let key = *path_keys[key_idx];
    let waypoints = paths.paths[&key].clone();
    if waypoints.is_empty() {
        return;
    }

    // Composite a fresh random civilian look (retry next tick if the layer
    // images are still loading).
    let spec = layers.random_spec(rng, "civilian");
    let Some(image) = layers.composite(&spec, &mut images) else {
        return;
    };
    let walk_speed = layers.walk_speed;

    // Convert start tile to world position.
    let (tx, ty) = waypoints[0];
    let start = crate::surface_pathfinding::SurfaceCostMap::tile_to_world(tx, ty);

    // Build a patrol behavior: walk this path, then pick another on completion.
    // The NPC behavior system will auto-despawn when the queue is empty.
    // We push two patrols so they walk two routes before despawning.
    let mut behavior = NpcBehavior::new(walk_speed);
    behavior.push(Behavior::Patrol {
        waypoints,
        current_idx: 0,
    });
    behavior.push(Behavior::Patrol {
        waypoints: Vec::new(), // will pick a random path on start
        current_idx: 0,
    });

    commands.spawn((
        DespawnOnExit(PlayState::Exploring),
        Npc,
        behavior,
        CharacterAnim::person(0.11),
        RigidBody::Dynamic,
        LockedAxes::ROTATION_LOCKED,
        crate::surface_objects::character_foot_collider(5.0),
        CollisionLayers::new(crate::GameLayer::Character, [crate::GameLayer::Surface]),
        LinearDamping(10.0),
        LinearVelocity(Vec2::ZERO),
        Sprite::from_atlas_image(
            image,
            TextureAtlas {
                layout: layers.layout.clone(),
                index: 0,
            },
        ),
        Transform::from_xyz(
            start.x,
            start.y,
            depth_z(start.y - crate::surface_objects::CHARACTER_FOOT_OFFSET),
        ),
    ));
}

/// Update NPC z for depth sorting.
pub fn depth_sort_npcs(mut npcs: Query<&mut Transform, (With<Npc>, Without<Walker>)>) {
    for mut tf in &mut npcs {
        tf.translation.z =
            depth_z(tf.translation.y - crate::surface_objects::CHARACTER_FOOT_OFFSET);
    }
}

/// Clean up civilian resources on exit.
pub fn cleanup_civilians(mut commands: Commands) {
    commands.remove_resource::<CivilianSpawnTimer>();
}
