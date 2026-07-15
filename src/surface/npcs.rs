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
#[allow(clippy::too_many_arguments)]
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
    chase: Res<crate::surface::interiors::MazeChase>,
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
                // Building-bound targets wait INSIDE that building now.
                if building
                    .as_deref()
                    .and_then(building_kind_from_name)
                    .is_some_and(crate::surface::interiors::has_interior)
                {
                    continue;
                }
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
                    PlayState::Exploring,
                );
            }
            crate::missions::Objective::CatchNpc {
                planet,
                building,
                npc,
                ..
            } if planet == planet_name => {
                let target_kind = building.as_deref().and_then(building_kind_from_name);
                // EVERY building-bound catch plays the staged chase now —
                // a bar pirate runs for the bar exactly like a mine
                // fugitive runs for the mine (rooms just have no shafts
                // to descend, so inside they're immediately cornered).
                let staged = target_kind.is_some_and(crate::surface::interiors::has_interior);
                if staged {
                    // The staged chase: the fugitive starts OUT HERE, near
                    // the venue, and bolts for its door when you close in
                    // (catch them in the open if you're quick). Once
                    // they're inside, the chase ledger owns them.
                    if chase.inside.contains_key(mission_id.as_str()) {
                        continue;
                    }
                    let Some(&door) = building
                        .as_ref()
                        .and_then(|name| building_door.get(&name.to_lowercase()))
                    else {
                        continue; // venue not on this world's pad
                    };
                    let spawn_tile = find_nearby_tile(door, &cost_map, &paths, &mut rng);
                    let spawned = crate::surface_npc::spawn_objective_npc(
                        &mut commands,
                        layers,
                        &mut images,
                        "civilian",
                        npc_identity(&item_universe, layers, npc),
                        mission_id,
                        spawn_tile,
                        walk_speed * 1.2,
                        crate::surface_npc::ObjectiveKind::CatchToward { goal: door },
                        PlayState::Exploring,
                    );
                    if let Some(entity) = spawned {
                        commands
                            .entity(entity)
                            .insert(crate::surface::interiors::MazeFugitive {
                                mission_id: mission_id.clone(),
                                goal: door,
                                next: crate::surface::interiors::FugitiveNext::IntoBuilding,
                            });
                    }
                    continue;
                }
                // No (resolvable) building: a pure surface chase.
                let door = building
                    .as_ref()
                    .and_then(|name| building_door.get(&name.to_lowercase()))
                    .copied()
                    .unwrap_or_else(|| door_tiles[rng.r#gen_range(0..door_tiles.len())]);
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
                    PlayState::Exploring,
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
    state: Res<State<PlayState>>,
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
        // Friends AND hired pilots walk with you (carried fighters stay
        // with the ship).
        let (name, identity) = match &entry.kind {
            crate::carrier::EscortKind::Companion { name } => {
                let Some(def) = item_universe.companions.get(name) else {
                    continue;
                };
                (
                    name,
                    npc_identity(&item_universe, &layers, &Some(def.npc.clone())),
                )
            }
            crate::carrier::EscortKind::Hired { name, .. } => {
                // Deterministic look, seeded by the pilot's name — the same
                // face that sat at the bar table.
                let spec = crate::surface_npc::anonymous_mission_spec(&layers, name, "civilian");
                (name, Some((name.clone(), spec)))
            }
            crate::carrier::EscortKind::Carried { .. } => continue,
        };
        if here.contains(name.as_str()) {
            continue;
        }
        // Beside the player, walkable, fanned out by their formation slot.
        let tile = follower_spawn_tile(&cost_map, walker_tf.translation.truncate(), name);
        let scope = state.get().clone();
        let spawned = crate::surface_npc::spawn_companion_avatar(
            &mut commands,
            &mut layers,
            &mut images,
            name,
            identity,
            tile,
            walk_speed,
            scope.clone(),
        );
        if let (Some(entity), PlayState::Inside) = (spawned, scope) {
            commands
                .entity(entity)
                .insert(crate::surface::interiors::InteriorScoped);
        }
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
        BuildingKind::Mine,
        BuildingKind::Warehouse,
        BuildingKind::Substation,
    ]
    .into_iter()
    .find(|k| format!("{k:?}").to_lowercase() == name)
}

/// A "!" hovering over a building that has a mission giver waiting
/// INSIDE — without it, indoor offers are invisible from the street.
#[derive(Component)]
pub(crate) struct BuildingOfferMarker(pub BuildingKind);

/// Keep exterior offer markers in sync with the current offers: one gold
/// "!" over each building whose interior holds at least one waiting giver.
pub(crate) fn update_building_offer_markers(
    mut commands: Commands,
    offers: Res<crate::missions::MissionOffers>,
    catalog: Res<crate::missions::MissionCatalog>,
    log: Res<crate::missions::MissionLog>,
    landed: Res<crate::planet_ui::LandedContext>,
    buildings: Query<(&Transform, &super::Building)>,
    markers: Query<(Entity, &BuildingOfferMarker)>,
) {
    let planet = landed.planet_name.clone().unwrap_or_default();
    // Which building kinds currently have an AVAILABLE indoor giver?
    let mut hot: std::collections::HashSet<BuildingKind> = Default::default();
    for id in offers.npc.get(&planet).cloned().unwrap_or_default() {
        if !matches!(log.status(&id), crate::missions::MissionStatus::Available) {
            continue;
        }
        let Some(def) = catalog.defs.get(&id) else {
            continue;
        };
        let crate::missions::types::OfferKind::NpcOffer { building, .. } = &def.offer else {
            continue;
        };
        if let Some(kind) = building.as_deref().and_then(building_kind_from_name)
            && crate::surface::interiors::has_interior(kind)
        {
            hot.insert(kind);
        }
    }
    let marked: std::collections::HashSet<BuildingKind> =
        markers.iter().map(|(_, m)| m.0).collect();
    // Remove stale markers (offer accepted / expired).
    for (entity, marker) in &markers {
        if !hot.contains(&marker.0) {
            commands.entity(entity).despawn();
        }
    }
    // Add missing ones over the building's door sensor.
    for kind in hot.difference(&marked) {
        let Some((tf, _)) = buildings.iter().find(|(_, b)| b.kind == *kind) else {
            continue;
        };
        commands.spawn((
            DespawnOnExit(PlayState::Exploring),
            BuildingOfferMarker(*kind),
            Text2d::new("!"),
            TextFont {
                font_size: 26.0,
                ..Default::default()
            },
            TextColor(Color::srgb(1.0, 0.85, 0.2)),
            Transform::from_xyz(
                tf.translation.x,
                tf.translation.y + crate::surface::TILE_PX * 1.9,
                6.0,
            )
            .with_scale(Vec3::splat(0.8)),
        ));
    }
}

/// The spawn tile for a follower appearing beside the player: walkable,
/// preferring the direction of THEIR formation slot, so simultaneous
/// spawns (friend + hire + prisoner on a scene change) fan out instead
/// of stacking on the single nearest tile.
pub(crate) fn follower_spawn_tile(
    cost_map: &crate::surface_pathfinding::SurfaceCostMap,
    walker_pos: Vec2,
    key: &str,
) -> (u32, u32) {
    let wt = crate::surface_pathfinding::SurfaceCostMap::world_to_tile(walker_pos);
    let walkable = |t: (u32, u32)| -> bool {
        let idx = (t.1 * cost_map.width + t.0) as usize;
        idx < cost_map.data.len() && cost_map.data[idx] < f32::INFINITY
    };
    let (offset, _) = crate::surface_npc::formation_params(key);
    let dir = offset.normalize_or_zero();
    // First choice: their own slot direction, nearest first.
    for r in 1i32..=4 {
        let c = (
            wt.0.saturating_add_signed((dir.x * r as f32).round() as i32),
            wt.1.saturating_add_signed((dir.y * r as f32).round() as i32),
        );
        if c != wt && walkable(c) {
            return c;
        }
    }
    // Fallback: full ring scan.
    for r in 1i32..=4 {
        for dx in -r..=r {
            for dy in -r..=r {
                if dx.abs().max(dy.abs()) != r {
                    continue;
                }
                let c = (
                    wt.0.saturating_add_signed(dx),
                    wt.1.saturating_add_signed(dy),
                );
                if c != wt && walkable(c) {
                    return c;
                }
            }
        }
    }
    wt
}
