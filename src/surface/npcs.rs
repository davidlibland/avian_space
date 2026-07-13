//! Mission-giver NPCs and companion avatars on the surface.
#[allow(unused_imports)]
use super::*;

use crate::item_universe::ItemUniverse;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Spawn mission-giver NPCs from NpcOffer missions, plus objective NPCs for
/// active MeetNpc/CatchNpc missions on this planet.
///
/// Idempotent (skips missions whose NPC is already on the surface) and runs
/// every frame while Exploring: a follow-up mission that auto-starts *after*
/// the player lands (e.g. its predecessor completed on this very landing) gets
/// its NPC spawned immediately, instead of only on the next re-land.
pub(crate) fn spawn_mission_npcs(
    mut commands: Commands,
    layers: Option<ResMut<crate::character_compositor::CharacterLayers>>,
    mut images: ResMut<Assets<Image>>,
    paths: Option<Res<crate::surface_pathfinding::SurfacePaths>>,
    cost_map: Option<Res<crate::surface_pathfinding::SurfaceCostMap>>,
    mission_offers: Res<crate::missions::MissionOffers>,
    mission_catalog: Res<crate::missions::MissionCatalog>,
    mission_log: Res<crate::missions::MissionLog>,
    landed_context: Res<crate::planet_ui::LandedContext>,
    item_universe: Res<ItemUniverse>,
    existing: Query<&crate::surface_npc::MissionNpc>,
) {
    let (Some(mut layers), Some(paths), Some(cost_map)) = (layers, paths, cost_map) else {
        return;
    };
    let layers = &mut *layers;
    let walk_speed = layers.walk_speed;
    let planet_name = landed_context.planet_name.as_deref().unwrap_or("");

    let already_spawned: std::collections::HashSet<&str> =
        existing.iter().map(|m| m.0.as_str()).collect();

    let mission_ids: Vec<String> = mission_offers
        .npc
        .get(planet_name)
        .cloned()
        .unwrap_or_default();

    // Collect all door tiles from precomputed paths (deduped).
    let door_tiles: Vec<(u32, u32)> = {
        let mut set = std::collections::HashSet::new();
        for p in paths.paths.values() {
            if let Some(&first) = p.first() {
                set.insert(first);
            }
            if let Some(&last) = p.last() {
                set.insert(last);
            }
        }
        set.into_iter().collect()
    };
    if door_tiles.is_empty() {
        return;
    }

    // Map building kind name (lowercase) → door tile.
    // For each path (A→B), A's door is the first tile.
    let mut building_door: std::collections::HashMap<String, (u32, u32)> =
        std::collections::HashMap::new();
    for (&(from, _), path) in &paths.paths {
        if let Some(&first) = path.first() {
            let name = format!("{:?}", from).to_lowercase();
            building_door.entry(name).or_insert(first);
        }
    }

    use rand::Rng;
    let mut rng = rand::thread_rng();

    for mission_id in &mission_ids {
        if already_spawned.contains(mission_id.as_str()) {
            continue;
        }
        // Stale offer (procedural def pruned since the roll): an NPC offering
        // a mission that no longer exists would silently no-op on Accept.
        if !mission_catalog.defs.contains_key(mission_id.as_str()) {
            continue;
        }
        // Look up the mission def for NpcOffer config.
        let (seek, door, identity) =
            if let Some(def) = mission_catalog.defs.get(mission_id.as_str()) {
                match &def.offer {
                    crate::missions::OfferKind::NpcOffer {
                        building,
                        approach,
                        npc,
                        ..
                    } => {
                        let seek = *approach == crate::missions::NpcApproach::Seek;

                        // Givers bound to a walk-in shop wait INSIDE it
                        // (spawn_interior_npcs seats them); don't double-
                        // spawn them on the pad.
                        let inside = building
                            .as_deref()
                            .and_then(building_kind_from_name)
                            .is_some_and(crate::surface::interiors::has_interior);
                        if inside {
                            continue;
                        }

                        // Resolve building name to a door tile.
                        let door = building
                            .as_ref()
                            .and_then(|name| building_door.get(&name.to_lowercase()))
                            .copied()
                            .unwrap_or_else(|| {
                                // No building specified or not found → random door.
                                door_tiles[rng.r#gen_range(0..door_tiles.len())]
                            });

                        (seek, door, npc_identity(&item_universe, layers, npc))
                    }
                    _ => {
                        // Fallback for non-NpcOffer (shouldn't happen here).
                        let door = door_tiles[rng.r#gen_range(0..door_tiles.len())];
                        (false, door, None)
                    }
                }
            } else {
                let door = door_tiles[rng.r#gen_range(0..door_tiles.len())];
                (false, door, None)
            };

        // For waiters, pick a tile near the door (not on it).
        let spawn_tile = if !seek {
            find_nearby_tile(door, &cost_map, &paths, &mut rng)
        } else {
            door
        };

        crate::surface_npc::spawn_mission_npc(
            &mut commands,
            layers,
            &mut images,
            "civilian",
            identity,
            mission_id,
            spawn_tile,
            walk_speed,
            seek,
            PlayState::Exploring,
        );
    }

    // ── Spawn NPCs for active MeetNpc / CatchNpc objectives ─────────

    for (mission_id, def) in &mission_catalog.defs {
        if already_spawned.contains(mission_id.as_str()) {
            continue;
        }
        let status = mission_log.status(mission_id);
        if !matches!(status, crate::missions::MissionStatus::Active(_)) {
            continue;
        }
        match &def.objective {
            crate::missions::Objective::MeetNpc {
                planet,
                building,
                approach,
                npc,
                ..
            } if planet == planet_name => {
                let door = building
                    .as_ref()
                    .and_then(|name| building_door.get(&name.to_lowercase()))
                    .copied()
                    .unwrap_or_else(|| door_tiles[rng.r#gen_range(0..door_tiles.len())]);
                let seek = *approach == crate::missions::NpcApproach::Seek;
                let spawn_tile = if !seek {
                    find_nearby_tile(door, &cost_map, &paths, &mut rng)
                } else {
                    door
                };
                let _world_pos = crate::surface_pathfinding::SurfaceCostMap::tile_to_world(
                    spawn_tile.0,
                    spawn_tile.1,
                );
                crate::surface_npc::spawn_objective_npc(
                    &mut commands,
                    layers,
                    &mut images,
                    "civilian",
                    npc_identity(&item_universe, layers, npc),
                    mission_id,
                    spawn_tile,
                    walk_speed,
                    crate::surface_npc::ObjectiveKind::Meet { seek },
                );
            }
            crate::missions::Objective::CatchNpc {
                planet,
                building,
                npc,
                ..
            } if planet == planet_name => {
                let door = building
                    .as_ref()
                    .and_then(|name| building_door.get(&name.to_lowercase()))
                    .copied()
                    .unwrap_or_else(|| door_tiles[rng.r#gen_range(0..door_tiles.len())]);
                let _world_pos =
                    crate::surface_pathfinding::SurfaceCostMap::tile_to_world(door.0, door.1);
                crate::surface_npc::spawn_objective_npc(
                    &mut commands,
                    layers,
                    &mut images,
                    "civilian",
                    npc_identity(&item_universe, layers, npc),
                    mission_id,
                    door,
                    walk_speed * 1.2,
                    crate::surface_npc::ObjectiveKind::Catch,
                );
            }
            _ => {}
        }
    }
}

/// Resolve a mission's recurring-NPC reference (`npc:` in missions.yaml) to
/// a display name + consistent avatar. Unknown ids warn and fall back to the
/// anonymous random look.
/// Friends walk with you: spawn a following avatar for every enrolled
/// Companion (friends only — hired pilots stay with their ships). Runs every
/// Exploring frame, idempotent per companion key, so a friend granted while
/// landed appears immediately.
pub(crate) fn spawn_companion_avatars(
    mut commands: Commands,
    roster: Option<Res<crate::carrier::EscortRoster>>,
    item_universe: Res<ItemUniverse>,
    existing: Query<&crate::surface_npc::CompanionAvatar>,
    walker: Query<&Transform, With<Walker>>,
    cost_map: Option<Res<crate::surface_pathfinding::SurfaceCostMap>>,
    layers: Option<ResMut<crate::character_compositor::CharacterLayers>>,
    mut images: ResMut<Assets<Image>>,
) {
    let (Some(roster), Some(cost_map), Some(mut layers)) = (roster, cost_map, layers) else {
        return;
    };
    let Ok(walker_tf) = walker.single() else {
        return;
    };
    let here: std::collections::HashSet<&str> = existing.iter().map(|c| c.0.as_str()).collect();
    let walk_speed = layers.walk_speed;
    for entry in &roster.entries {
        let crate::carrier::EscortKind::Companion { name } = &entry.kind else {
            continue;
        };
        if here.contains(name.as_str()) {
            continue;
        }
        let Some(def) = item_universe.companions.get(name) else {
            continue;
        };
        // Beside the player, on walkable ground.
        let base = walker_tf.translation.truncate() + Vec2::new(TILE_PX * 1.5, 0.0);
        let tile = {
            let t = crate::surface_pathfinding::SurfaceCostMap::world_to_tile(base);
            let idx = (t.1 * cost_map.width + t.0) as usize;
            if idx < cost_map.data.len() && cost_map.data[idx] < f32::INFINITY {
                t
            } else {
                crate::surface_pathfinding::SurfaceCostMap::world_to_tile(
                    walker_tf.translation.truncate(),
                )
            }
        };
        let identity = npc_identity(&item_universe, &layers, &Some(def.npc.clone()));
        crate::surface_npc::spawn_companion_avatar(
            &mut commands,
            &mut layers,
            &mut images,
            name,
            identity,
            tile,
            walk_speed,
        );
    }
}

pub(crate) fn npc_identity(
    universe: &ItemUniverse,
    layers: &crate::character_compositor::CharacterLayers,
    npc_id: &Option<String>,
) -> Option<(String, crate::character_compositor::AvatarSpec)> {
    let id = npc_id.as_ref()?;
    let Some(def) = universe.npcs.get(id) else {
        eprintln!("[missions] WARNING: unknown npc id {id:?} (see assets/npcs.yaml)");
        return None;
    };
    let role = def.role.as_deref().unwrap_or("civilian");
    let spec = layers.spec_for_npc(id, def.avatar.as_ref(), role);
    Some((def.name.clone(), spec))
}

/// Find a walkable tile near `door` but not on it (1–10 tiles along a
/// random path from that door).  Falls back to the door itself.
pub(crate) fn find_nearby_tile(
    door: (u32, u32),
    cost_map: &crate::surface_pathfinding::SurfaceCostMap,
    paths: &crate::surface_pathfinding::SurfacePaths,
    rng: &mut impl rand::Rng,
) -> (u32, u32) {
    // Find a precomputed path that starts at this door.
    let candidates: Vec<&Vec<(u32, u32)>> = paths
        .paths
        .values()
        .filter(|p| p.first() == Some(&door) && p.len() > 2)
        .collect();

    if let Some(path) = candidates.get(rng.r#gen_range(0..candidates.len().max(1))) {
        let max_idx = path.len().min(10);
        let min_idx = 2.min(max_idx);
        if min_idx < max_idx {
            let idx = rng.r#gen_range(min_idx..max_idx);
            return path[idx];
        }
    }

    // Fallback: pathfind to a nearby random walkable tile.
    let offsets: [(i32, i32); 4] = [(0, -2), (0, 2), (-2, 0), (2, 0)];
    for &(dx, dy) in &offsets {
        let nx = (door.0 as i32 + dx).max(0) as u32;
        let ny = (door.1 as i32 + dy).max(0) as u32;
        let idx = (ny * cost_map.width + nx) as usize;
        if idx < cost_map.data.len() && cost_map.data[idx] < f32::INFINITY {
            return (nx, ny);
        }
    }

    door // absolute fallback
}

// ---------------------------------------------------------------------------
// Setup helpers
// ---------------------------------------------------------------------------

/// Convert tile coordinates to world-space center position (tilemap centered
/// at origin via `TilemapAnchor::Center`).
pub(crate) fn tile_to_world(tx: u32, ty: u32, map_w: u32, map_h: u32, tile_px: f32) -> Vec2 {
    Vec2::new(
        (tx as f32 - map_w as f32 / 2.0) * tile_px + tile_px / 2.0,
        (ty as f32 - map_h as f32 / 2.0) * tile_px + tile_px / 2.0,
    )
}

/// Building name (as used in mission `building:` fields — the Debug name
/// lowercased) → kind. None for unknown names.
pub(crate) fn building_kind_from_name(name: &str) -> Option<BuildingKind> {
    [
        BuildingKind::ShipPad,
        BuildingKind::MechanicShop,
        BuildingKind::Market,
        BuildingKind::Outfitter,
        BuildingKind::Shipyard,
        BuildingKind::Bar,
        BuildingKind::FuelStation,
        BuildingKind::Garrison,
    ]
    .into_iter()
    .find(|k| format!("{k:?}").to_lowercase() == name)
}
