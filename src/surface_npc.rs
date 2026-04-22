//! NPC behavior system for the planet surface.
//!
//! Each NPC carries an [`NpcBehavior`] component with a queue of
//! [`Behavior`] variants.  The front of the queue is the active behavior.
//! When a behavior completes it is popped, and the next one starts.
//!
//! Control systems set `LinearVelocity`; animation is handled by the
//! shared `animate_characters` system in `surface_character.rs`.
//!
//! # Behaviors
//!
//! - **Patrol**: walk a random building-to-building path (existing civilian logic).
//! - **SeekPlayer**: pathfind toward the player.  Completes when adjacent.
//! - **FleePlayer**: move away from the player.  Completes when caught.
//! - **OfferMission**: stand still with "!" marker, wait for E press.
//! - **AwaitPlayer**: stand still, complete a mission objective on contact.
//! - **Despawn**: countdown timer then remove the entity.

use std::collections::VecDeque;

use avian2d::prelude::*;
use bevy::prelude::*;

use crate::surface::{Walker, BuildingKind, TILE_PX, WORLD_WIDTH, WORLD_HEIGHT};
use crate::surface_pathfinding::{SurfaceCostMap, SurfacePaths};

// ── Behavior enum ────────────────────────────────────────────────────────

/// A single behavior in the NPC's behavior queue.
#[derive(Clone, Debug)]
pub enum Behavior {
    /// Walk a random building-to-building path.  When the path is done,
    /// this behavior completes (and can be re-pushed for continuous patrol).
    Patrol {
        /// Current waypoints (tile coords).  Empty = needs a new path.
        waypoints: Vec<(u32, u32)>,
        current_idx: usize,
    },

    /// Walk toward the player.  Completes when within `arrive_dist` tiles.
    SeekPlayer {
        /// Path to the player (recomputed periodically).
        path: Vec<(u32, u32)>,
        current_idx: usize,
        /// How often to recompute the path (seconds).
        repath_timer: Timer,
    },

    /// Run away from the player.  Completes when the player gets adjacent
    /// (i.e. the player "caught" the NPC).
    FleePlayer {
        mission_id: String,
        /// Current flee path.
        path: Vec<(u32, u32)>,
        current_idx: usize,
        repath_timer: Timer,
    },

    /// Stand still with a marker.  Wait for the player to press E while
    /// adjacent.  Completes after the player interacts.
    OfferMission {
        mission_id: String,
    },

    /// Stand still.  When the player gets adjacent, complete a mission
    /// objective and pop this behavior.
    AwaitPlayer {
        mission_id: String,
    },

    /// Countdown then despawn.
    Despawn {
        timer: Timer,
    },
}

// ── Components ───────────────────────────────────────────────────────────

/// The behavior queue driving an NPC.
#[derive(Component)]
pub struct NpcBehavior {
    pub queue: VecDeque<Behavior>,
    /// Movement speed (pixels per second).
    pub speed: f32,
}

impl NpcBehavior {
    pub fn new(speed: f32) -> Self {
        Self {
            queue: VecDeque::new(),
            speed,
        }
    }

    pub fn with_behaviors(speed: f32, behaviors: impl IntoIterator<Item = Behavior>) -> Self {
        Self {
            queue: behaviors.into_iter().collect(),
            speed,
        }
    }

    /// Push a behavior to the back of the queue.
    pub fn push(&mut self, behavior: Behavior) {
        self.queue.push_back(behavior);
    }
}

/// Marker for NPC entities (as opposed to the player Walker).
#[derive(Component)]
pub struct Npc;

/// Marks the "!" indicator sprite above a mission-giver NPC.
#[derive(Component)]
pub struct NpcMarker;

/// Currently-interacting NPC (if any).  Set when the player presses E
/// while adjacent to an NPC with `OfferMission` behavior.
#[derive(Resource, Default)]
pub struct ActiveNpcInteraction {
    pub entity: Option<Entity>,
    /// The mission ID being offered.
    pub mission_id: Option<String>,
}

// ── Constants ────────────────────────────────────────────────────────────

/// Distance (in tiles) at which SeekPlayer/AwaitPlayer considers the
/// player "adjacent" and the behavior completes.
const ADJACENT_DIST_TILES: f32 = 1.5;

/// Distance (in world units) for waypoint arrival.
const WAYPOINT_ARRIVE_DIST: f32 = 2.0;

/// Default repathing interval for SeekPlayer / FleePlayer.
const REPATH_INTERVAL: f32 = 1.0;

// ── Behavior system ──────────────────────────────────────────────────────

/// Main NPC behavior system.  Processes the front of each NPC's behavior
/// queue, sets `LinearVelocity`, and pops completed behaviors.
pub fn run_npc_behaviors(
    mut commands: Commands,
    time: Res<Time>,
    walker_q: Query<&Transform, With<Walker>>,
    cost_map: Option<Res<SurfaceCostMap>>,
    surface_paths: Option<Res<SurfacePaths>>,
    mut npcs: Query<(Entity, &mut NpcBehavior, &Transform, &mut LinearVelocity), (With<Npc>, Without<Walker>)>,
) {
    let walker_pos = walker_q.single().ok().map(|t| t.translation.truncate());

    for (entity, mut npc, tf, mut vel) in &mut npcs {
        let pos = tf.translation.truncate();
        let speed = npc.speed;

        let Some(behavior) = npc.queue.front_mut() else {
            // No behaviors left — stop and despawn.
            vel.0 = Vec2::ZERO;
            commands.entity(entity).despawn();
            continue;
        };

        match behavior {
            Behavior::Patrol { waypoints, current_idx } => {
                // If no waypoints, pick a random building-to-building path.
                if waypoints.is_empty() {
                    if let (Some(paths), Some(cm)) = (surface_paths.as_ref(), cost_map.as_ref()) {
                        use rand::Rng;
                        let mut rng = rand::thread_rng();

                        // Pick a random precomputed door-to-door path.
                        let keys: Vec<_> = paths.paths.keys().collect();
                        if let Some(&&key) = keys.get(rng.r#gen_range(0..keys.len().max(1))) {
                            let door_path = &paths.paths[&key];
                            if let Some(&first_door) = door_path.first() {
                                // Pathfind from current position to the start door.
                                let my_tile = SurfaceCostMap::world_to_tile(pos);
                                let mut full_path = if my_tile == first_door {
                                    Vec::new()
                                } else {
                                    cm.find_path(my_tile, first_door).unwrap_or_default()
                                };
                                // Append the precomputed door-to-door segment.
                                full_path.extend_from_slice(door_path);
                                *waypoints = full_path;
                                *current_idx = 0;
                            }
                        }
                    }
                    if waypoints.is_empty() {
                        // No paths available — complete patrol.
                        vel.0 = Vec2::ZERO;
                        npc.queue.pop_front();
                        continue;
                    }
                }

                // Follow waypoints.
                if *current_idx >= waypoints.len() {
                    vel.0 = Vec2::ZERO;
                    npc.queue.pop_front();
                    continue;
                }

                let (tx, ty) = waypoints[*current_idx];
                let target = SurfaceCostMap::tile_to_world(tx, ty);
                let diff = target - pos;
                if diff.length() < WAYPOINT_ARRIVE_DIST {
                    *current_idx += 1;
                    vel.0 = Vec2::ZERO;
                } else {
                    vel.0 = diff.normalize() * speed;
                }
            }

            Behavior::SeekPlayer { path, current_idx, repath_timer } => {
                let Some(wp) = walker_pos else {
                    vel.0 = Vec2::ZERO;
                    continue;
                };

                // Check if adjacent to player.
                let dist_to_player = (pos - wp).length();
                if dist_to_player < ADJACENT_DIST_TILES * TILE_PX {
                    vel.0 = Vec2::ZERO;
                    npc.queue.pop_front();
                    continue;
                }

                // Recompute path when: empty, exhausted, or timer fires.
                repath_timer.tick(time.delta());
                let needs_repath = path.is_empty()
                    || *current_idx >= path.len()
                    || repath_timer.just_finished();
                if needs_repath {
                    if let Some(cm) = cost_map.as_ref() {
                        let start = find_nearest_walkable(pos, cm);
                        let goal = find_nearest_walkable(wp, cm);
                        if let Some(new_path) = cm.find_path(start, goal) {
                            // Skip waypoints we're already past.
                            let skip = new_path.iter()
                                .position(|&(tx, ty)| {
                                    let wp = SurfaceCostMap::tile_to_world(tx, ty);
                                    (wp - pos).length() > TILE_PX * 0.5
                                })
                                .unwrap_or(0);
                            *path = new_path;
                            *current_idx = skip;
                        }
                    }
                }

                // Follow path.
                if *current_idx < path.len() {
                    let (tx, ty) = path[*current_idx];
                    let target = SurfaceCostMap::tile_to_world(tx, ty);
                    let diff = target - pos;
                    if diff.length() < WAYPOINT_ARRIVE_DIST {
                        *current_idx += 1;
                    } else {
                        vel.0 = diff.normalize() * speed;
                    }
                } else {
                    // Path exhausted and repath failed — move directly.
                    vel.0 = (wp - pos).normalize_or_zero() * speed;
                }
            }

            Behavior::FleePlayer { mission_id: _, path, current_idx, repath_timer } => {
                let Some(wp) = walker_pos else {
                    vel.0 = Vec2::ZERO;
                    continue;
                };

                // If player caught us (adjacent), complete.
                let dist_to_player = (pos - wp).length();
                if dist_to_player < ADJACENT_DIST_TILES * TILE_PX {
                    vel.0 = Vec2::ZERO;
                    // TODO: complete mission objective
                    npc.queue.pop_front();
                    continue;
                }

                // Recompute flee path when empty, exhausted, or timer fires.
                repath_timer.tick(time.delta());
                let needs_repath = path.is_empty()
                    || *current_idx >= path.len()
                    || repath_timer.just_finished();
                if needs_repath {
                    if let Some(cm) = cost_map.as_ref() {
                        let my_tile = find_nearest_walkable(pos, cm);
                        let player_tile = SurfaceCostMap::world_to_tile(wp);

                        // Pick a flee target: the tile furthest from the player
                        // among the building doors.
                        if let Some(paths) = surface_paths.as_ref() {
                            let mut best_goal = my_tile;
                            let mut best_dist = 0i32;
                            for path_tiles in paths.paths.values() {
                                let last = *path_tiles.last().unwrap_or(&my_tile);
                                let d = (last.0 as i32 - player_tile.0 as i32).abs()
                                    + (last.1 as i32 - player_tile.1 as i32).abs();
                                if d > best_dist {
                                    best_dist = d;
                                    best_goal = last;
                                }
                            }
                            if let Some(flee_path) = cm.find_path(my_tile, best_goal) {
                                *path = flee_path;
                                *current_idx = 0;
                            }
                        }
                    }
                }

                // Follow flee path.
                if *current_idx >= path.len() {
                    // Flee directly away.
                    let away = (pos - wp).normalize_or_zero();
                    vel.0 = away * speed * 1.2; // flee faster
                } else {
                    let (tx, ty) = path[*current_idx];
                    let target = SurfaceCostMap::tile_to_world(tx, ty);
                    let diff = target - pos;
                    if diff.length() < WAYPOINT_ARRIVE_DIST {
                        *current_idx += 1;
                    } else {
                        vel.0 = diff.normalize() * speed * 1.2;
                    }
                }
            }

            Behavior::OfferMission { .. } => {
                // Stand still, wait for player interaction.
                // The interaction is handled by a separate system that
                // checks for E press when adjacent to an NPC with this behavior.
                vel.0 = Vec2::ZERO;
            }

            Behavior::AwaitPlayer { .. } => {
                let Some(wp) = walker_pos else {
                    vel.0 = Vec2::ZERO;
                    continue;
                };

                vel.0 = Vec2::ZERO;

                let dist_to_player = (pos - wp).length();
                if dist_to_player < ADJACENT_DIST_TILES * TILE_PX {
                    // TODO: complete mission objective
                    npc.queue.pop_front();
                }
            }

            Behavior::Despawn { timer } => {
                vel.0 = Vec2::ZERO;
                timer.tick(time.delta());
                if timer.just_finished() {
                    commands.entity(entity).despawn();
                }
            }
        }
    }
}

// ── NPC interaction (E-press when adjacent) ──────────────────────────────

/// Detect E-press when the player is adjacent to an NPC with `OfferMission`
/// or `AwaitPlayer`.  Opens the interaction UI.
pub fn npc_interact(
    keyboard: Res<ButtonInput<KeyCode>>,
    walker_q: Query<&Transform, With<Walker>>,
    npcs: Query<(Entity, &NpcBehavior, &Transform), (With<Npc>, Without<Walker>)>,
    mut interaction: ResMut<ActiveNpcInteraction>,
) {
    // Close on Escape.
    if keyboard.just_pressed(KeyCode::Escape) {
        interaction.entity = None;
        interaction.mission_id = None;
        return;
    }

    if !keyboard.just_pressed(KeyCode::KeyE) {
        return;
    }

    // If already interacting, close.
    if interaction.entity.is_some() {
        interaction.entity = None;
        interaction.mission_id = None;
        return;
    }

    let Ok(walker_tf) = walker_q.single() else { return };
    let wp = walker_tf.translation.truncate();

    // Find the nearest NPC with OfferMission that's adjacent.
    for (entity, npc, tf) in &npcs {
        let dist = (tf.translation.truncate() - wp).length();
        if dist > ADJACENT_DIST_TILES * TILE_PX {
            continue;
        }
        // Check if the front behavior is OfferMission.
        if let Some(Behavior::OfferMission { mission_id }) = npc.queue.front() {
            interaction.entity = Some(entity);
            interaction.mission_id = Some(mission_id.clone());
            return;
        }
    }
}

/// Render the mission offer UI for the currently-interacting NPC.
pub fn npc_mission_offer_ui(
    mut egui_contexts: bevy_egui::EguiContexts,
    mut interaction: ResMut<ActiveNpcInteraction>,
    catalog: Res<crate::missions::MissionCatalog>,
    player_q: Query<&crate::ship::Ship, With<Walker>>,
    mut accept_writer: MessageWriter<crate::missions::AcceptMission>,
    mut decline_writer: MessageWriter<crate::missions::DeclineMission>,
    mut npcs: Query<&mut NpcBehavior, With<Npc>>,
    mut sfx_writer: MessageWriter<crate::sfx::SurfaceSfx>,
) {
    let Some(npc_entity) = interaction.entity else { return };
    let Some(mission_id) = interaction.mission_id.clone() else { return };
    let Some(def) = catalog.defs.get(&mission_id) else {
        interaction.entity = None;
        interaction.mission_id = None;
        return;
    };

    let Ok(ctx) = egui_contexts.ctx_mut() else { return };
    let free_cargo = player_q.single().map(|s| s.remaining_cargo_space()).unwrap_or(0);
    let required = def.required_cargo_space();
    let has_space = free_cargo >= required;

    bevy_egui::egui::Window::new("Mission Offer")
        .collapsible(false)
        .resizable(true)
        .anchor(bevy_egui::egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.label(&def.briefing);
            if required > 0 {
                ui.label(format!(
                    "Cargo required: {} units (you have {} free)",
                    required, free_cargo
                ));
            }
            ui.separator();
            ui.horizontal(|ui| {
                ui.add_enabled_ui(has_space, |ui| {
                    let btn = ui.button("Accept");
                    if btn.clicked() {
                        accept_writer.write(crate::missions::AcceptMission(mission_id.clone()));
                        // Pop the OfferMission behavior so the NPC moves on.
                        if let Ok(mut npc) = npcs.get_mut(npc_entity) {
                            npc.queue.pop_front();
                        }
                        interaction.entity = None;
                        interaction.mission_id = None;
                        sfx_writer.write(crate::sfx::SurfaceSfx::UiButton);
                    }
                    if !has_space {
                        btn.on_hover_text("Not enough free cargo space.");
                    }
                });
                if ui.button("Decline").clicked() {
                    decline_writer.write(crate::missions::DeclineMission(mission_id.clone()));
                    if let Ok(mut npc) = npcs.get_mut(npc_entity) {
                        npc.queue.pop_front();
                    }
                    interaction.entity = None;
                    interaction.mission_id = None;
                    sfx_writer.write(crate::sfx::SurfaceSfx::UiButton);
                }
            });
        });
}

/// Update the "!" marker position to float above the NPC.
pub fn update_npc_markers(
    npcs: Query<(&Transform, &NpcBehavior), With<Npc>>,
    mut markers: Query<(&mut Transform, &mut Visibility, &ChildOf), (With<NpcMarker>, Without<Npc>)>,
) {
    for (mut marker_tf, mut vis, child_of) in &mut markers {
        if let Ok((_, npc)) = npcs.get(child_of.parent()) {
            // Show marker only if front behavior is OfferMission.
            let is_offer = matches!(npc.queue.front(), Some(Behavior::OfferMission { .. }));
            *vis = if is_offer { Visibility::Inherited } else { Visibility::Hidden };
            // Float above the NPC.
            marker_tf.translation.y = 12.0;
            marker_tf.translation.z = 0.1;
        }
    }
}

// ── Spawning helpers ─────────────────────────────────────────────────────

/// Spawn a mission-giver NPC at a building door.
///
/// - `seek`: if true, the NPC walks toward the player before offering.
///   If false, the NPC stands near the building and waits.
pub fn spawn_mission_npc(
    commands: &mut Commands,
    sprites: &crate::surface_civilians::CivilianSprites,
    mission_id: &str,
    door_tile: (u32, u32),
    speed: f32,
    seek: bool,
) {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let sprite_idx = rng.r#gen_range(0..sprites.images.len());
    let image = sprites.images[sprite_idx].clone();
    let start = SurfaceCostMap::tile_to_world(door_tile.0, door_tile.1);

    let mut behavior = NpcBehavior::new(speed);
    if seek {
        behavior.push(Behavior::SeekPlayer {
            path: Vec::new(),
            current_idx: 0,
            repath_timer: Timer::from_seconds(REPATH_INTERVAL, TimerMode::Repeating),
        });
    }
    behavior.push(Behavior::OfferMission {
        mission_id: mission_id.to_string(),
    });
    // After offering, patrol away.
    behavior.push(Behavior::Patrol {
        waypoints: Vec::new(),
        current_idx: 0,
    });
    behavior.push(Behavior::Despawn {
        timer: Timer::from_seconds(2.0, TimerMode::Once),
    });

    let npc_entity = commands.spawn((
        DespawnOnExit(crate::PlayState::Exploring),
        Npc,
        behavior,
        crate::surface_character::CharacterAnim::with_interval(0.2),
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
        Transform::from_xyz(start.x, start.y, crate::surface_objects::depth_z(start.y - 8.0)),
    )).id();

    // Spawn "!" marker as a child.
    commands.entity(npc_entity).with_children(|parent| {
        parent.spawn((
            NpcMarker,
            Text2d::new("!"),
            TextFont { font_size: 20.0, ..default() },
            TextColor(Color::srgb(1.0, 0.9, 0.2)),
            Transform::from_xyz(0.0, 12.0, 0.1).with_scale(Vec3::splat(0.3)),
        ));
    });
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Find the nearest walkable tile to a world position.  If the tile under
/// `world_pos` is walkable, returns it directly.  Otherwise spirals outward
/// to find the closest passable tile.
fn find_nearest_walkable(world_pos: Vec2, cm: &SurfaceCostMap) -> (u32, u32) {
    let tile = SurfaceCostMap::world_to_tile(world_pos);
    let idx = (tile.1 * cm.width + tile.0) as usize;
    if idx < cm.data.len() && cm.data[idx] < f32::INFINITY {
        return tile;
    }
    // Spiral outward.
    for radius in 1..10u32 {
        for dy in -(radius as i32)..=(radius as i32) {
            for dx in -(radius as i32)..=(radius as i32) {
                if dx.unsigned_abs() != radius && dy.unsigned_abs() != radius {
                    continue;
                }
                let nx = (tile.0 as i32 + dx).max(0) as u32;
                let ny = (tile.1 as i32 + dy).max(0) as u32;
                if nx >= cm.width || ny >= cm.height { continue; }
                let ni = (ny * cm.width + nx) as usize;
                if ni < cm.data.len() && cm.data[ni] < f32::INFINITY {
                    return (nx, ny);
                }
            }
        }
    }
    tile // fallback
}
