//! Civilian NPC spawning and path-following on the planet surface.
//!
//! Civilians spawn at a random building door, follow a precomputed path
//! to another building, then despawn.  On average ~2 civilians are walking
//! at any given time.

use avian2d::prelude::*;
use bevy::prelude::*;
use rand::{Rng, SeedableRng};
use serde::Deserialize;

use crate::surface::{BuildingKind, TILE_PX, Walker};
use crate::surface_objects::depth_z;
use crate::surface_pathfinding::SurfacePaths;
use crate::PlayState;

// ── Constants ────────────────────────────────────────────────────────────

/// Target average number of civilians walking at once.
const TARGET_CIVILIAN_COUNT: f32 = 2.0;

// ── RON manifest ─────────────────────────────────────────────────────────

#[derive(Deserialize, Debug)]
struct CivilianManifest {
    sprites: Vec<String>,
    cols: u32,
    rows: u32,
    tile_w: u32,
    tile_h: u32,
    walk_speed: f32,
}

// ── Components ───────────────────────────────────────────────────────────

use crate::surface_character::CharacterAnim;
use crate::surface_npc::{Npc, NpcBehavior, Behavior};

// ── Resource ─────────────────────────────────────────────────────────────

/// Loaded civilian sprite data (inserted at surface setup).
#[derive(Resource)]
pub struct CivilianSprites {
    pub images: Vec<Handle<Image>>,
    pub layout: Handle<TextureAtlasLayout>,
    pub walk_speed: f32,
}

/// Spawn timer — controls spawn rate to maintain TARGET_CIVILIAN_COUNT.
#[derive(Resource)]
pub struct CivilianSpawnTimer {
    timer: Timer,
    rng: rand::rngs::StdRng,
}

// ── Public setup ─────────────────────────────────────────────────────────

/// Load civilian manifest and insert resources.  Call from `setup_surface`.
pub fn setup_civilians(
    commands: &mut Commands,
    asset_server: &AssetServer,
    atlas_layouts: &mut Assets<TextureAtlasLayout>,
    seed: u64,
) {
    let manifest: CivilianManifest = match std::fs::read_to_string(
        "assets/sprites/people/civilians.ron",
    )
    .ok()
    .and_then(|t| ron::from_str(&t).ok())
    {
        Some(m) => m,
        None => {
            eprintln!("[civilians] WARNING: could not load civilians.ron");
            return;
        }
    };

    let images: Vec<Handle<Image>> = manifest
        .sprites
        .iter()
        .map(|path| asset_server.load(path.clone()))
        .collect();

    let layout = atlas_layouts.add(TextureAtlasLayout::from_grid(
        UVec2::new(manifest.tile_w, manifest.tile_h),
        manifest.cols,
        manifest.rows,
        None,
        None,
    ));

    commands.insert_resource(CivilianSprites {
        images,
        layout,
        walk_speed: manifest.walk_speed,
    });

    // Average walk time across the map ≈ map_width / speed × tile_px.
    // With 64 tiles, speed 40, tile 32: ~51 seconds per crossing.
    // To maintain 2 civilians, spawn one every ~25 seconds.
    // We'll adjust dynamically based on current count.
    commands.insert_resource(CivilianSpawnTimer {
        timer: Timer::from_seconds(2.0, TimerMode::Repeating),
        rng: rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(0xC1C1_0001)),
    });
}

// ── Systems ──────────────────────────────────────────────────────────────

/// Periodically spawn civilian NPCs if below the target count.
pub fn spawn_civilians(
    mut commands: Commands,
    sprites: Option<Res<CivilianSprites>>,
    paths_res: Option<Res<SurfacePaths>>,
    mut spawn_timer: ResMut<CivilianSpawnTimer>,
    time: Res<Time>,
    existing: Query<(), With<Npc>>,
) {
    let (Some(sprites), Some(paths)) = (sprites, paths_res) else { return };
    if paths.paths.is_empty() || sprites.images.is_empty() {
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

    // Pick a random sprite.
    let sprite_idx = rng.r#gen_range(0..sprites.images.len());
    let image = sprites.images[sprite_idx].clone();

    // Convert start tile to world position.
    let (tx, ty) = waypoints[0];
    let start = crate::surface_pathfinding::SurfaceCostMap::tile_to_world(tx, ty);

    // Build a patrol behavior: walk this path, then pick another on completion.
    // The NPC behavior system will auto-despawn when the queue is empty.
    // We push two patrols so they walk two routes before despawning.
    let mut behavior = NpcBehavior::new(sprites.walk_speed);
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
        CharacterAnim::with_interval(0.2),
        RigidBody::Dynamic,
        LockedAxes::ROTATION_LOCKED,
        Collider::circle(4.0),
        CollisionLayers::new(crate::GameLayer::Character, [crate::GameLayer::Surface]),
        LinearDamping(10.0),
        LinearVelocity(Vec2::ZERO),
        Sprite::from_atlas_image(
            image,
            TextureAtlas {
                layout: sprites.layout.clone(),
                index: 0,
            },
        ),
        Transform::from_xyz(start.x, start.y, depth_z(start.y - 8.0)),
    ));
}

/// Update NPC z for depth sorting.
pub fn depth_sort_npcs(
    mut npcs: Query<&mut Transform, (With<Npc>, Without<Walker>)>,
) {
    for mut tf in &mut npcs {
        tf.translation.z = depth_z(tf.translation.y - 8.0);
    }
}

/// Clean up civilian resources on exit.
pub fn cleanup_civilians(mut commands: Commands) {
    commands.remove_resource::<CivilianSprites>();
    commands.remove_resource::<CivilianSpawnTimer>();
}
