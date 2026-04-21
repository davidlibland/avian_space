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

/// Marks a civilian NPC entity.
#[derive(Component)]
pub struct Civilian;

use crate::surface_character::CharacterAnim;

/// The path a civilian is following (tile coordinates, consumed front-to-back).
#[derive(Component)]
pub struct CivilianPath {
    waypoints: Vec<(u32, u32)>,
    current_idx: usize,
}

// ── Resource ─────────────────────────────────────────────────────────────

/// Loaded civilian sprite data (inserted at surface setup).
#[derive(Resource)]
pub struct CivilianSprites {
    images: Vec<Handle<Image>>,
    layout: Handle<TextureAtlasLayout>,
    walk_speed: f32,
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

/// Periodically spawn civilians if below the target count.
pub fn spawn_civilians(
    mut commands: Commands,
    sprites: Option<Res<CivilianSprites>>,
    paths_res: Option<Res<SurfacePaths>>,
    mut spawn_timer: ResMut<CivilianSpawnTimer>,
    time: Res<Time>,
    existing: Query<(), With<Civilian>>,
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

    // Pick a random path.
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

    // Spawn at the first waypoint.
    let (tx, ty) = waypoints[0];
    let tile_px = TILE_PX;
    let world_x = (tx as f32 - 32.0) * tile_px + tile_px / 2.0; // approximate centering
    let world_y = (ty as f32 - 32.0) * tile_px + tile_px / 2.0;
    // Use tile_to_world logic: (tx - map_w/2) * tile_px + tile_px/2
    // We don't have map_w here, but the waypoints from SurfacePaths are
    // in the same coordinate system used by tile_to_world. We need map dims.
    // Store them in SurfacePaths.

    // Actually, convert tile → world using the same formula as tile_to_world.
    // We need map_w/map_h — let's get them from the paths resource.
    // For now, use the tile coords directly and convert in the movement system.

    let start_frame = rng.r#gen_range(0u8..4);

    let initial_index = start_frame as usize; // facing down, frame 0

    // Convert start tile to world position.
    let map_w = crate::surface::WORLD_WIDTH;
    let map_h = crate::surface::WORLD_HEIGHT;
    let start_x = (tx as f32 - map_w as f32 / 2.0) * tile_px + tile_px / 2.0;
    let start_y = (ty as f32 - map_h as f32 / 2.0) * tile_px + tile_px / 2.0;

    commands.spawn((
        DespawnOnExit(PlayState::Exploring),
        Civilian,
        CivilianPath {
            waypoints,
            current_idx: 0,
        },
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
                index: initial_index,
            },
        ),
        Transform::from_xyz(start_x, start_y, depth_z(start_y - 8.0)),
    ));
}

/// Move civilians along their paths by setting velocity toward the next
/// waypoint.  Animation is handled by the shared `animate_characters` system.
pub fn move_civilians(
    mut commands: Commands,
    sprites: Option<Res<CivilianSprites>>,
    mut civilians: Query<
        (Entity, &mut CivilianPath, &Transform, &mut LinearVelocity),
        With<Civilian>,
    >,
) {
    let Some(sprites) = sprites else { return };

    let tile_px = TILE_PX;
    let speed = sprites.walk_speed;
    let map_w = crate::surface::WORLD_WIDTH;
    let map_h = crate::surface::WORLD_HEIGHT;

    for (entity, mut path, tf, mut vel) in &mut civilians {
        if path.current_idx >= path.waypoints.len() {
            commands.entity(entity).despawn();
            continue;
        }

        let (target_tx, target_ty) = path.waypoints[path.current_idx];
        let target_x = (target_tx as f32 - map_w as f32 / 2.0) * tile_px + tile_px / 2.0;
        let target_y = (target_ty as f32 - map_h as f32 / 2.0) * tile_px + tile_px / 2.0;
        let target = Vec2::new(target_x, target_y);

        let pos = tf.translation.truncate();
        let diff = target - pos;
        let dist = diff.length();

        if dist < 2.0 {
            path.current_idx += 1;
            vel.0 = Vec2::ZERO;
            continue;
        }

        vel.0 = (diff / dist) * speed;
    }
}

/// Update civilian z for depth sorting (like the walker).
pub fn depth_sort_civilians(
    mut civilians: Query<&mut Transform, (With<Civilian>, Without<Walker>)>,
) {
    for mut tf in &mut civilians {
        tf.translation.z = depth_z(tf.translation.y - 8.0);
    }
}

/// Clean up civilian resources on exit.
pub fn cleanup_civilians(mut commands: Commands) {
    commands.remove_resource::<CivilianSprites>();
    commands.remove_resource::<CivilianSpawnTimer>();
}
