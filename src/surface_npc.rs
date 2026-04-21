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

                // Periodically recompute path to player.
                repath_timer.tick(time.delta());
                if path.is_empty() || repath_timer.just_finished() {
                    if let Some(cm) = cost_map.as_ref() {
                        let start = SurfaceCostMap::world_to_tile(pos);
                        let goal = SurfaceCostMap::world_to_tile(wp);
                        if let Some(new_path) = cm.find_path(start, goal) {
                            *path = new_path;
                            *current_idx = 0;
                        }
                    }
                }

                // Follow path.
                if *current_idx >= path.len() {
                    // Path exhausted but not adjacent — will repath next tick.
                    vel.0 = (wp - pos).normalize_or_zero() * speed;
                } else {
                    let (tx, ty) = path[*current_idx];
                    let target = SurfaceCostMap::tile_to_world(tx, ty);
                    let diff = target - pos;
                    if diff.length() < WAYPOINT_ARRIVE_DIST {
                        *current_idx += 1;
                    } else {
                        vel.0 = diff.normalize() * speed;
                    }
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

                // Periodically recompute flee path (away from player).
                repath_timer.tick(time.delta());
                if path.is_empty() || repath_timer.just_finished() {
                    if let Some(cm) = cost_map.as_ref() {
                        let my_tile = SurfaceCostMap::world_to_tile(pos);
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
