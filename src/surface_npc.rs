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

    /// Follow the player indefinitely.  Never completes on its own —
    /// the NPC stays near the player until despawned or the behavior is
    /// externally popped.  Uses the same pathfinding logic as SeekPlayer.
    FollowPlayer {
        path: Vec<(u32, u32)>,
        current_idx: usize,
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
    landed_context: Res<crate::planet_ui::LandedContext>,
    mut npc_met_writer: MessageWriter<crate::missions::NpcMet>,
    mut npc_caught_writer: MessageWriter<crate::missions::NpcCaught>,
) {
    let planet_name = landed_context.planet_name.clone().unwrap_or_default();
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

            Behavior::FollowPlayer { path, current_idx, repath_timer } => {
                let Some(wp) = walker_pos else {
                    vel.0 = Vec2::ZERO;
                    continue;
                };

                let dist_to_player = (pos - wp).length();

                // When adjacent, just stop and wait (don't pop the behavior).
                if dist_to_player < ADJACENT_DIST_TILES * TILE_PX {
                    vel.0 = Vec2::ZERO;
                    // Keep repathing so we follow if the player moves away.
                    repath_timer.tick(time.delta());
                    continue;
                }

                // Same seek logic as SeekPlayer.
                repath_timer.tick(time.delta());
                let needs_repath = path.is_empty()
                    || *current_idx >= path.len()
                    || repath_timer.just_finished();
                if needs_repath {
                    if let Some(cm) = cost_map.as_ref() {
                        let start = find_nearest_walkable(pos, cm);
                        let goal = find_nearest_walkable(wp, cm);
                        if let Some(new_path) = cm.find_path(start, goal) {
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
                    vel.0 = (wp - pos).normalize_or_zero() * speed;
                }
            }

            Behavior::FleePlayer { mission_id, path, current_idx, repath_timer } => {
                let Some(wp) = walker_pos else {
                    vel.0 = Vec2::ZERO;
                    continue;
                };

                // If player caught us (adjacent), fire event and complete.
                let dist_to_player = (pos - wp).length();
                if dist_to_player < ADJACENT_DIST_TILES * TILE_PX {
                    vel.0 = Vec2::ZERO;
                    npc_caught_writer.write(crate::missions::NpcCaught {
                        planet: planet_name.clone(),
                        mission_id: mission_id.clone(),
                    });
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

            Behavior::AwaitPlayer { mission_id } => {
                let Some(wp) = walker_pos else {
                    vel.0 = Vec2::ZERO;
                    continue;
                };

                vel.0 = Vec2::ZERO;

                let dist_to_player = (pos - wp).length();
                if dist_to_player < ADJACENT_DIST_TILES * TILE_PX {
                    npc_met_writer.write(crate::missions::NpcMet {
                        planet: planet_name.clone(),
                        mission_id: mission_id.clone(),
                    });
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

// NPC interaction (E-press chat, mission offers) is in surface_npc_chat.rs.

/// Update the "!" marker position to float above the NPC.
pub fn update_npc_markers(
    npcs: Query<(&Transform, &NpcBehavior), With<Npc>>,
    mut markers: Query<(&mut Transform, &mut Visibility, &ChildOf), (With<NpcMarker>, Without<Npc>)>,
) {
    for (mut marker_tf, mut vis, child_of) in &mut markers {
        if let Ok((_, npc)) = npcs.get(child_of.parent()) {
            // Show marker for any mission-relevant behavior.
            let show = matches!(
                npc.queue.front(),
                Some(Behavior::OfferMission { .. })
                | Some(Behavior::AwaitPlayer { .. })
                | Some(Behavior::FleePlayer { .. })
                | Some(Behavior::SeekPlayer { .. })
                | Some(Behavior::FollowPlayer { .. })
            );
            *vis = if show { Visibility::Inherited } else { Visibility::Hidden };
            marker_tf.translation.y = 14.0;
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
            Transform::from_xyz(0.0, 14.0, 0.1).with_scale(Vec3::splat(0.6)),
        ));
    });
}

/// What kind of objective NPC to spawn.
pub enum ObjectiveKind {
    Meet { seek: bool },
    Catch,
}

/// Spawn an NPC for a MeetNpc or CatchNpc mission objective.
pub fn spawn_objective_npc(
    commands: &mut Commands,
    sprites: &crate::surface_civilians::CivilianSprites,
    mission_id: &str,
    door_tile: (u32, u32),
    speed: f32,
    kind: ObjectiveKind,
) {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let sprite_idx = rng.r#gen_range(0..sprites.images.len());
    let image = sprites.images[sprite_idx].clone();
    let start = SurfaceCostMap::tile_to_world(door_tile.0, door_tile.1);

    // Marker: "?" blue for meet, "!" red for catch.
    let (marker_text, marker_color) = match &kind {
        ObjectiveKind::Meet { .. } => ("?", Color::srgb(0.3, 0.8, 1.0)),
        ObjectiveKind::Catch => ("!", Color::srgb(1.0, 0.3, 0.2)),
    };

    let mut behavior = NpcBehavior::new(speed);
    match kind {
        ObjectiveKind::Meet { seek } => {
            if seek {
                behavior.push(Behavior::SeekPlayer {
                    path: Vec::new(),
                    current_idx: 0,
                    repath_timer: Timer::from_seconds(REPATH_INTERVAL, TimerMode::Repeating),
                });
            }
            behavior.push(Behavior::AwaitPlayer {
                mission_id: mission_id.to_string(),
            });
        }
        ObjectiveKind::Catch => {
            behavior.push(Behavior::FleePlayer {
                mission_id: mission_id.to_string(),
                path: Vec::new(),
                current_idx: 0,
                repath_timer: Timer::from_seconds(REPATH_INTERVAL, TimerMode::Repeating),
            });
        }
    }
    // After objective: caught NPCs follow the player, others patrol away.
    match &kind {
        ObjectiveKind::Catch => {
            // Follow the player after being caught (like an escort).
            behavior.push(Behavior::FollowPlayer {
                path: Vec::new(),
                current_idx: 0,
                repath_timer: Timer::from_seconds(REPATH_INTERVAL, TimerMode::Repeating),
            });
        }
        _ => {
            behavior.push(Behavior::Patrol {
                waypoints: Vec::new(),
                current_idx: 0,
            });
            behavior.push(Behavior::Despawn {
                timer: Timer::from_seconds(3.0, TimerMode::Once),
            });
        }
    }

    let npc_entity = commands.spawn((
        DespawnOnExit(crate::PlayState::Exploring),
        Npc,
        behavior,
        crate::surface_character::CharacterAnim::with_interval(0.18),
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

    commands.entity(npc_entity).with_children(|parent| {
        parent.spawn((
            NpcMarker,
            Text2d::new(marker_text),
            TextFont { font_size: 20.0, ..default() },
            TextColor(marker_color),
            Transform::from_xyz(0.0, 14.0, 0.1).with_scale(Vec3::splat(0.6)),
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
