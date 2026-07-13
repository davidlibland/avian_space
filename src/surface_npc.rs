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

use crate::surface::{TILE_PX, Walker};
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
    OfferMission { mission_id: String },

    /// Stand still.  When the player gets adjacent, complete a mission
    /// objective and pop this behavior.
    AwaitPlayer { mission_id: String },

    /// Countdown then despawn.
    Despawn { timer: Timer },
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

/// Tags an NPC spawned for a specific mission (offer-giver or objective NPC),
/// so `spawn_mission_npcs` can tell which missions already have their NPC on
/// the surface and stay idempotent — it runs every frame while exploring so
/// follow-up missions that start *after* landing get their NPC immediately
/// (previously they only spawned on surface entry, forcing a re-land).
#[derive(Component)]
pub struct MissionNpc(pub String);

/// A recurring character's display name (from assets/npcs.yaml), shown as the
/// conversation window title instead of the generic "Conversation".
#[derive(Component)]
pub struct NpcIdentity {
    pub name: String,
}

/// Marks the "!" indicator sprite above a mission-giver NPC.
#[derive(Component)]
pub struct NpcMarker;

// ── Constants ────────────────────────────────────────────────────────────

/// Distance (in tiles) at which SeekPlayer/AwaitPlayer considers the
/// player "adjacent" and the behavior completes.
const ADJACENT_DIST_TILES: f32 = 1.5;

/// Distance (in world units) for waypoint arrival. Physics (damping +
/// integration) never parks a body within a couple of pixels of a point —
/// the old 2px threshold made NPCs orbit waypoints and grind against the
/// walls beside them.
const WAYPOINT_ARRIVE_DIST: f32 = 10.0;

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
    mut npcs: Query<
        (Entity, &mut NpcBehavior, &Transform, &mut LinearVelocity),
        (With<Npc>, Without<Walker>),
    >,
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
            Behavior::Patrol {
                waypoints,
                current_idx,
            } => {
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
                                let my_tile = SurfaceCostMap::world_to_tile(foot(pos));
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
                if (target - foot(pos)).length() < WAYPOINT_ARRIVE_DIST {
                    *current_idx += 1;
                    vel.0 = Vec2::ZERO;
                } else {
                    vel.0 = steer_to(target, pos, speed);
                }
            }

            Behavior::SeekPlayer {
                path,
                current_idx,
                repath_timer,
            } => {
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
                let needs_repath =
                    path.is_empty() || *current_idx >= path.len() || repath_timer.just_finished();
                if needs_repath && let Some(cm) = cost_map.as_ref() {
                    let start = find_nearest_walkable(pos, cm);
                    let goal = find_nearest_walkable(wp, cm);
                    if let Some(new_path) = cm.find_path(start, goal) {
                        // Skip waypoints we're already past.
                        let skip = new_path
                            .iter()
                            .position(|&(tx, ty)| {
                                let wp = SurfaceCostMap::tile_to_world(tx, ty);
                                (wp - foot(pos)).length() > TILE_PX * 0.5
                            })
                            .unwrap_or(0);
                        *path = new_path;
                        *current_idx = skip;
                    }
                }

                // Follow path.
                if *current_idx < path.len() {
                    let (tx, ty) = path[*current_idx];
                    let target = SurfaceCostMap::tile_to_world(tx, ty);
                    if (target - foot(pos)).length() < WAYPOINT_ARRIVE_DIST {
                        *current_idx += 1;
                    } else {
                        vel.0 = steer_to(target, pos, speed);
                    }
                } else {
                    // Path exhausted / repath failed. NEVER beeline (that's
                    // how NPCs ground themselves against buildings) — nudge
                    // toward the nearest walkable tile so the next repath
                    // starts from clean ground.
                    if let Some(cm) = cost_map.as_ref() {
                        let t = find_nearest_walkable(pos, cm);
                        vel.0 = steer_to(SurfaceCostMap::tile_to_world(t.0, t.1), pos, speed * 0.6);
                    } else {
                        vel.0 = Vec2::ZERO;
                    }
                }
            }

            Behavior::FollowPlayer {
                path,
                current_idx,
                repath_timer,
            } => {
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
                let needs_repath =
                    path.is_empty() || *current_idx >= path.len() || repath_timer.just_finished();
                if needs_repath && let Some(cm) = cost_map.as_ref() {
                    let start = find_nearest_walkable(pos, cm);
                    let goal = find_nearest_walkable(wp, cm);
                    if let Some(new_path) = cm.find_path(start, goal) {
                        let skip = new_path
                            .iter()
                            .position(|&(tx, ty)| {
                                let wp = SurfaceCostMap::tile_to_world(tx, ty);
                                (wp - foot(pos)).length() > TILE_PX * 0.5
                            })
                            .unwrap_or(0);
                        *path = new_path;
                        *current_idx = skip;
                    }
                }

                if *current_idx < path.len() {
                    let (tx, ty) = path[*current_idx];
                    let target = SurfaceCostMap::tile_to_world(tx, ty);
                    if (target - foot(pos)).length() < WAYPOINT_ARRIVE_DIST {
                        *current_idx += 1;
                    } else {
                        vel.0 = steer_to(target, pos, speed);
                    }
                } else {
                    // Path exhausted / repath failed — nudge to walkable
                    // ground instead of beelining into buildings.
                    if let Some(cm) = cost_map.as_ref() {
                        let t = find_nearest_walkable(pos, cm);
                        vel.0 = steer_to(SurfaceCostMap::tile_to_world(t.0, t.1), pos, speed * 0.6);
                    } else {
                        vel.0 = Vec2::ZERO;
                    }
                }
            }

            Behavior::FleePlayer {
                mission_id,
                path,
                current_idx,
                repath_timer,
            } => {
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
                let needs_repath =
                    path.is_empty() || *current_idx >= path.len() || repath_timer.just_finished();
                if needs_repath && let Some(cm) = cost_map.as_ref() {
                    let my_tile = find_nearest_walkable(pos, cm);
                    let player_tile = SurfaceCostMap::world_to_tile(foot(wp));
                    // Sampled open-ground escapes (see pick_flee_goal) —
                    // the old door-only goals sent runners INTO
                    // buildings, frequently straight past the player.
                    let mut rng = rand::thread_rng();
                    if let Some(goal) = pick_flee_goal(cm, my_tile, player_tile, &mut rng)
                        && let Some(flee_path) = cm.find_path(my_tile, goal)
                    {
                        *path = flee_path;
                        *current_idx = 0;
                    }
                }

                // Follow flee path.
                if *current_idx >= path.len() {
                    // No path — run away along walkable ground: pick the
                    // best walkable neighbor tile in the away direction
                    // rather than pushing blindly into whatever's behind.
                    if let Some(cm) = cost_map.as_ref() {
                        let my_tile = find_nearest_walkable(pos, cm);
                        let away = (foot(pos) - foot(wp)).normalize_or_zero();
                        let mut best: Option<((u32, u32), f32)> = None;
                        for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                            let n = (
                                (my_tile.0 as i32 + dx).max(0) as u32,
                                (my_tile.1 as i32 + dy).max(0) as u32,
                            );
                            if !walkable(cm, n) {
                                continue;
                            }
                            let dir = Vec2::new(dx as f32, dy as f32);
                            let score = dir.dot(away);
                            if best.is_none_or(|(_, b)| score > b) {
                                best = Some((n, score));
                            }
                        }
                        if let Some((n, _)) = best {
                            let target = SurfaceCostMap::tile_to_world(n.0, n.1);
                            vel.0 = steer_to(target, pos, speed * 1.2);
                        } else {
                            vel.0 = Vec2::ZERO;
                        }
                    } else {
                        vel.0 = (foot(pos) - foot(wp)).normalize_or_zero() * speed * 1.2;
                    }
                } else {
                    let (tx, ty) = path[*current_idx];
                    let target = SurfaceCostMap::tile_to_world(tx, ty);
                    if (target - foot(pos)).length() < WAYPOINT_ARRIVE_DIST {
                        *current_idx += 1;
                    } else {
                        vel.0 = steer_to(target, pos, speed * 1.2);
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
    mut markers: Query<
        (&mut Transform, &mut Visibility, &ChildOf),
        (With<NpcMarker>, Without<Npc>),
    >,
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
            *vis = if show {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            };
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
    layers: &mut crate::character_compositor::CharacterLayers,
    images: &mut Assets<Image>,
    role: &str,
    identity: Option<(String, crate::character_compositor::AvatarSpec)>,
    mission_id: &str,
    door_tile: (u32, u32),
    speed: f32,
    seek: bool,
    scope: crate::PlayState,
) {
    let mut rng = rand::thread_rng();

    let spec = identity
        .as_ref()
        .map(|(_, spec)| spec.clone())
        .unwrap_or_else(|| layers.random_spec(&mut rng, role));
    let Some(image) = layers.composite(&spec, images) else {
        return; // layer images still loading; idempotent caller retries
    };
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

    let npc_entity = commands
        .spawn((
            DespawnOnExit(scope),
            Npc,
            MissionNpc(mission_id.to_string()),
            behavior,
            crate::surface_character::CharacterAnim::person(0.11),
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
                crate::surface_objects::depth_z(
                    start.y - crate::surface_objects::CHARACTER_FOOT_OFFSET,
                ),
            ),
        ))
        .id();

    if let Some((name, _)) = identity {
        commands.entity(npc_entity).insert(NpcIdentity { name });
    }

    // Spawn "!" marker as a child.
    commands.entity(npc_entity).with_children(|parent| {
        parent.spawn((
            NpcMarker,
            Text2d::new("!"),
            TextFont {
                font_size: 20.0,
                ..default()
            },
            TextColor(Color::srgb(1.0, 0.9, 0.2)),
            Transform::from_xyz(0.0, 16.0, 0.1).with_scale(Vec3::splat(0.6)),
        ));
    });
}

/// A companion friend's surface avatar (companions.yaml key). They walk
/// with the player whenever they land — same follow logic as mission NPCs.
#[derive(Component)]
pub struct CompanionAvatar(pub String);

/// Spawn a loyal friend's walking avatar, following the player around the
/// surface. Persistent in the roster sense: re-spawned on every landing
/// while the friend is enrolled.
#[allow(clippy::too_many_arguments)]
pub fn spawn_companion_avatar(
    commands: &mut Commands,
    layers: &mut crate::character_compositor::CharacterLayers,
    images: &mut Assets<Image>,
    companion_key: &str,
    identity: Option<(String, crate::character_compositor::AvatarSpec)>,
    start_tile: (u32, u32),
    speed: f32,
) {
    let mut rng = rand::thread_rng();
    let spec = identity
        .as_ref()
        .map(|(_, spec)| spec.clone())
        .unwrap_or_else(|| layers.random_spec(&mut rng, "civilian"));
    let Some(image) = layers.composite(&spec, images) else {
        return; // layer images still loading; idempotent caller retries
    };
    let start = SurfaceCostMap::tile_to_world(start_tile.0, start_tile.1);
    let behavior = NpcBehavior::with_behaviors(
        speed,
        [Behavior::FollowPlayer {
            path: Vec::new(),
            current_idx: 0,
            repath_timer: Timer::from_seconds(REPATH_INTERVAL, TimerMode::Repeating),
        }],
    );
    let npc_entity = commands
        .spawn((
            DespawnOnExit(crate::PlayState::Exploring),
            Npc,
            CompanionAvatar(companion_key.to_string()),
            behavior,
            crate::surface_character::CharacterAnim::person(0.11),
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
                crate::surface_objects::depth_z(
                    start.y - crate::surface_objects::CHARACTER_FOOT_OFFSET,
                ),
            ),
        ))
        .id();
    if let Some((name, _)) = identity {
        commands.entity(npc_entity).insert(NpcIdentity { name });
    }
}

/// What kind of objective NPC to spawn.
pub enum ObjectiveKind {
    Meet { seek: bool },
    Catch,
}

/// Spawn an NPC for a MeetNpc or CatchNpc mission objective.
pub fn spawn_objective_npc(
    commands: &mut Commands,
    layers: &mut crate::character_compositor::CharacterLayers,
    images: &mut Assets<Image>,
    role: &str,
    identity: Option<(String, crate::character_compositor::AvatarSpec)>,
    mission_id: &str,
    door_tile: (u32, u32),
    speed: f32,
    kind: ObjectiveKind,
    scope: crate::PlayState,
) -> Option<Entity> {
    let mut rng = rand::thread_rng();

    let spec = identity
        .as_ref()
        .map(|(_, spec)| spec.clone())
        .unwrap_or_else(|| layers.random_spec(&mut rng, role));
    let Some(image) = layers.composite(&spec, images) else {
        return None; // layer images still loading; idempotent caller retries
    };
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

    let npc_entity = commands
        .spawn((
            DespawnOnExit(scope),
            Npc,
            MissionNpc(mission_id.to_string()),
            behavior,
            crate::surface_character::CharacterAnim::person(0.10),
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
                crate::surface_objects::depth_z(
                    start.y - crate::surface_objects::CHARACTER_FOOT_OFFSET,
                ),
            ),
        ))
        .id();

    if let Some((name, _)) = identity {
        commands.entity(npc_entity).insert(NpcIdentity { name });
    }

    commands.entity(npc_entity).with_children(|parent| {
        parent.spawn((
            NpcMarker,
            Text2d::new(marker_text),
            TextFont {
                font_size: 20.0,
                ..default()
            },
            TextColor(marker_color),
            Transform::from_xyz(0.0, 16.0, 0.1).with_scale(Vec3::splat(0.6)),
        ));
    });
    Some(npc_entity)
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Find the nearest walkable tile to a world position.  If the tile under
/// `world_pos` is walkable, returns it directly.  Otherwise spirals outward
/// to find the closest passable tile.
/// A character's FOOT position: the collider and everything walkability-
/// related anchors 14px below the sprite centre (see CHARACTER_FOOT_OFFSET).
/// All tile math must use this, or an NPC whose sprite centre reads as a
/// walkable tile can have its feet (and collider) pressed half a tile into
/// the building row below — the classic "NPC grinding against a wall".
fn foot(pos: Vec2) -> Vec2 {
    pos - Vec2::new(0.0, crate::surface_objects::CHARACTER_FOOT_OFFSET)
}

/// Steering velocity that drives the FOOT toward a tile-centre target.
fn steer_to(target: Vec2, pos: Vec2, speed: f32) -> Vec2 {
    (target - foot(pos)).normalize_or_zero() * speed
}

/// Whether a tile is walkable on the cost map.
fn walkable(cm: &SurfaceCostMap, tile: (u32, u32)) -> bool {
    let idx = (tile.1 * cm.width + tile.0) as usize;
    idx < cm.data.len() && cm.data[idx] < f32::INFINITY
}

/// Pick a flee goal: the reachable walkable tile that best trades "far from
/// the player" against "not absurdly far from me". Samples a ring of
/// candidates around the NPC (plus a jittered fan) instead of only building
/// doors — doors made fleeing NPCs run INTO buildings, and often straight
/// past the player to reach the "far" door.
pub(crate) fn pick_flee_goal(
    cm: &SurfaceCostMap,
    npc_tile: (u32, u32),
    player_tile: (u32, u32),
    rng: &mut impl rand::Rng,
) -> Option<(u32, u32)> {
    let mut best: Option<((u32, u32), f32)> = None;
    // Candidates: 8 compass rays at fixed radii (corridors and streets are
    // axis-aligned — a pure random ring misses a 1-tile-wide lane), plus a
    // jittered ring for open ground.
    let mut candidates: Vec<(f32, f32)> = Vec::with_capacity(24 + 24);
    for (dx, dy) in [
        (1.0f32, 0.0f32),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (0.7, 0.7),
        (-0.7, 0.7),
        (0.7, -0.7),
        (-0.7, -0.7),
    ] {
        for radius in [6.0f32, 10.0, 14.0] {
            candidates.push((dx * radius, dy * radius));
        }
    }
    for _ in 0..24 {
        let angle = rng.gen_range(0.0..std::f32::consts::TAU);
        let radius = rng.gen_range(8.0..=16.0f32);
        candidates.push((angle.cos() * radius, angle.sin() * radius));
    }
    for (ox, oy) in candidates {
        let cx = (npc_tile.0 as f32 + ox).round();
        let cy = (npc_tile.1 as f32 + oy).round();
        if cx < 1.0 || cy < 1.0 || cx >= (cm.width - 1) as f32 || cy >= (cm.height - 1) as f32 {
            continue;
        }
        let cand = (cx as u32, cy as u32);
        if !walkable(cm, cand) {
            continue;
        }
        let d_player = (cand.0 as f32 - player_tile.0 as f32).abs()
            + (cand.1 as f32 - player_tile.1 as f32).abs();
        let d_npc =
            (cand.0 as f32 - npc_tile.0 as f32).abs() + (cand.1 as f32 - npc_tile.1 as f32).abs();
        // Far from the player, mildly preferring nearby escapes; a goal the
        // player stands between us and is fine ONLY if the path says so —
        // reachability is checked for the winner below.
        let score = d_player - 0.25 * d_npc;
        if best.is_none_or(|(_, b)| score > b) {
            // Verify reachability before accepting (paths route around
            // buildings, so an unreachable pocket never wins).
            if cm.find_path(npc_tile, cand).is_some() {
                best = Some((cand, score));
            }
        }
    }
    best.map(|(t, _)| t)
}

fn find_nearest_walkable(world_pos: Vec2, cm: &SurfaceCostMap) -> (u32, u32) {
    let tile = SurfaceCostMap::world_to_tile(foot(world_pos));
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
                if nx >= cm.width || ny >= cm.height {
                    continue;
                }
                let ni = (ny * cm.width + nx) as usize;
                if ni < cm.data.len() && cm.data[ni] < f32::INFINITY {
                    return (nx, ny);
                }
            }
        }
    }
    tile // fallback
}

/// A shopkeeper standing behind their counter. Pure ambience: no
/// `NpcBehavior` (an empty queue would despawn them) and no chat — the
/// counter itself opens the trade window. Static body so they hold their
/// spot behind the till.
pub fn spawn_clerk(
    commands: &mut Commands,
    layers: &mut crate::character_compositor::CharacterLayers,
    images: &mut Assets<Image>,
    tile: (u32, u32),
) {
    let mut rng = rand::thread_rng();
    let spec = layers.random_spec(&mut rng, "civilian");
    let Some(image) = layers.composite(&spec, images) else {
        return;
    };
    let pos = SurfaceCostMap::tile_to_world(tile.0, tile.1);
    commands.spawn((
        DespawnOnExit(crate::PlayState::Inside),
        crate::surface::interiors::InteriorScoped,
        crate::surface::interiors::Clerk,
        Npc,
        crate::surface_character::CharacterAnim::person(0.11),
        RigidBody::Static,
        crate::surface_objects::character_foot_collider(5.0),
        CollisionLayers::new(crate::GameLayer::Character, [crate::GameLayer::Surface]),
        Sprite::from_atlas_image(
            image,
            TextureAtlas {
                layout: layers.layout.clone(),
                index: 0,
            },
        ),
        Transform::from_xyz(
            pos.x,
            pos.y,
            crate::surface_objects::depth_z(pos.y - crate::surface_objects::CHARACTER_FOOT_OFFSET),
        ),
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// A corridor world: one open row (y=5) crossing a solid field, with an
    /// open pocket on each end. The NPC must flee ALONG the corridor away
    /// from the player, not into the walls (old door-based goals routinely
    /// pointed through the player or into buildings).
    fn corridor_map() -> SurfaceCostMap {
        let (w, h) = (40u32, 11u32);
        let mut data = vec![f32::INFINITY; (w * h) as usize];
        for x in 0..w {
            data[(5 * w + x) as usize] = 1.0; // the corridor
        }
        // open pockets at both ends
        for y in 3..8u32 {
            for x in 0..4u32 {
                data[(y * w + x) as usize] = 1.0;
                data[(y * w + (w - 1 - x)) as usize] = 1.0;
            }
        }
        SurfaceCostMap {
            data,
            width: w,
            height: h,
        }
    }

    #[test]
    fn flee_goal_runs_away_along_walkable_ground() {
        let cm = corridor_map();
        let npc = (12u32, 5u32);
        let player = (8u32, 5u32); // west of the NPC in the corridor
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let mut east_wins = 0;
        for _ in 0..10 {
            let goal = pick_flee_goal(&cm, npc, player, &mut rng)
                .expect("open corridor always offers an escape");
            // Walkable + reachable by construction; away = east of the NPC.
            assert!(walkable(&cm, goal), "flee goal must be walkable");
            assert!(
                cm.find_path(npc, goal).is_some(),
                "flee goal must be reachable"
            );
            if goal.0 > npc.0 {
                east_wins += 1;
            }
        }
        assert!(
            east_wins >= 9,
            "fleeing east (away from the player) must dominate: {east_wins}/10"
        );
    }

    #[test]
    fn flee_goal_never_picks_solid_or_unreachable_tiles() {
        let cm = corridor_map();
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        for seed_pos in [(4u32, 5u32), (20, 5), (35, 5)] {
            for _ in 0..5 {
                if let Some(goal) = pick_flee_goal(&cm, seed_pos, (2, 5), &mut rng) {
                    assert!(walkable(&cm, goal));
                    assert!(cm.find_path(seed_pos, goal).is_some());
                }
            }
        }
    }
}
