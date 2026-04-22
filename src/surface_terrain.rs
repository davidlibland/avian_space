//! Constrained terrain generation for the planet surface.
//!
//! This module handles the full pipeline from fBm noise to final terrain:
//!
//! 1. **Generate** raw terrain via fBm (borders pinned to void/impassable)
//! 2. **Constrain** terrain near buildings to be walkable
//! 3. **Clamp** terrain gradients (max ±1 between adjacent tiles)
//! 4. **Derive** collision map from terrain indices
//! 5. **Force paths** between disconnected buildings by converting
//!    impassable tiles along the cheapest crossing to walkable terrain
//! 6. **Re-clamp** and **re-derive** after path forcing
//!
//! The key invariant maintained throughout: no two adjacent tiles
//! (8-connected) differ by more than 1 terrain index.

use crate::surface_pathfinding;
use crate::surface::BuildingKind;

/// Result of terrain generation.
pub struct GeneratedTerrain {
    /// Terrain indices per tile (row-major, bottom-up).
    pub terrain: Vec<u32>,
    /// Collision codes per tile (derived from terrain + manifest).
    pub collision: Vec<u8>,
}

/// Generate terrain and ensure all buildings are connected by walkable paths.
///
/// # Steps
/// 1. fBm terrain generation with pinned borders
/// 2. Force tiles near buildings to walkable + clamp
/// 3. Derive collision
/// 4. Check connectivity; force paths for disconnected pairs
/// 5. Re-clamp + re-derive collision
///
/// # Parameters
/// - `map_w`, `map_h`: map dimensions
/// - `n_terrain_types`: total terrain types (including void)
/// - `seed`: planet seed
/// - `collision_codes`: collision code per terrain index (from manifest)
/// - `movement_costs`: movement cost per terrain index
/// - `placed_buildings`: all building footprint rects `(x, y, w, h)`
/// - `door_positions`: `(BuildingKind, (x, y))` for each building door
/// - `solid_building_tiles`: set of (x,y) tiles that are solid building walls
pub fn generate_constrained_terrain(
    map_w: u32,
    map_h: u32,
    n_terrain_types: u32,
    seed: u64,
    fbm_scale: f32,
    fbm_octaves: u32,
    fbm_lacunarity: f32,
    fbm_gain: f32,
    collision_codes: &[u8],
    movement_costs: &[f32],
    placed_buildings: &[(u32, u32, u32, u32)],
    door_positions: &[(BuildingKind, (u32, u32))],
    solid_building_tiles: &std::collections::HashSet<(u32, u32)>,
) -> GeneratedTerrain {
    // ── Step 1: Generate raw terrain ─────────────────────────────────────
    let mut terrain = crate::fbm::generate_terrain_map(
        map_w,
        map_h,
        n_terrain_types,
        fbm_scale,
        fbm_octaves,
        fbm_lacunarity,
        fbm_gain,
        seed,
    );

    let walkable_terrain: u32 = collision_codes
        .iter()
        .position(|&c| c == 0)
        .unwrap_or(0) as u32;

    // ── Step 2: Force tiles near buildings to walkable ────────────────────
    force_walkable_near_buildings(
        &mut terrain,
        map_w,
        map_h,
        placed_buildings,
        walkable_terrain,
    );

    // ── Step 3: Derive collision ─────────────────────────────────────────
    let mut collision = derive_collision(&terrain, collision_codes);

    // ── Step 4: Check connectivity and force paths ───────────────────────
    force_paths_for_connectivity(
        &mut terrain,
        &mut collision,
        map_w,
        map_h,
        collision_codes,
        movement_costs,
        solid_building_tiles,
        door_positions,
        walkable_terrain,
    );

    GeneratedTerrain { terrain, collision }
}

/// Force tiles within `margin` of each building rect to `walkable_terrain`,
/// pin them, then re-clamp to maintain gradient constraint.
fn force_walkable_near_buildings(
    terrain: &mut Vec<u32>,
    map_w: u32,
    map_h: u32,
    placed_buildings: &[(u32, u32, u32, u32)],
    walkable_terrain: u32,
) {
    let w = map_w as usize;
    let h = map_h as usize;
    let margin = 2_i32;
    let mut pinned = vec![false; w * h];

    for &(bx, by, bw, bh) in placed_buildings {
        for dy in -margin..(bh as i32 + margin) {
            for dx in -margin..(bw as i32 + margin) {
                let x = bx as i32 + dx;
                let y = by as i32 + dy;
                if x >= 0 && y >= 0 && x < w as i32 && y < h as i32 {
                    let idx = y as usize * w + x as usize;
                    terrain[idx] = walkable_terrain;
                    pinned[idx] = true;
                }
            }
        }
    }

    // Pin border tiles so they don't get pulled down.
    for x in 0..w {
        pinned[x] = true;
        pinned[(h - 1) * w + x] = true;
    }
    for y in 0..h {
        pinned[y * w] = true;
        pinned[y * w + (w - 1)] = true;
    }

    crate::fbm::clamp_terrain_indices(terrain, w, h, &pinned);
}

/// Derive collision codes from terrain indices.
fn derive_collision(terrain: &[u32], collision_codes: &[u8]) -> Vec<u8> {
    terrain
        .iter()
        .map(|&t| *collision_codes.get(t as usize).unwrap_or(&0))
        .collect()
}

/// Find disconnected building pairs and force walkable paths between them.
///
/// Uses Dijkstra with a very high (but finite) cost for solid tiles to
/// find the cheapest crossing.  Tiles along the crossing that are
/// impassable get forced to `walkable_terrain`, then gradients are
/// re-clamped and collision is re-derived.
fn force_paths_for_connectivity(
    terrain: &mut Vec<u32>,
    collision: &mut Vec<u8>,
    map_w: u32,
    map_h: u32,
    collision_codes: &[u8],
    movement_costs: &[f32],
    solid_building_tiles: &std::collections::HashSet<(u32, u32)>,
    door_positions: &[(BuildingKind, (u32, u32))],
    walkable_terrain: u32,
) {
    // Build cost map for connectivity check.
    let cost_map = surface_pathfinding::build_cost_map(
        collision,
        terrain,
        movement_costs,
        solid_building_tiles,
        map_w,
        map_h,
    );

    // Find which pairs are disconnected.
    let mut forced_tiles: Vec<(u32, u32)> = Vec::new();

    for i in 0..door_positions.len() {
        for j in (i + 1)..door_positions.len() {
            let (kind_a, pos_a) = door_positions[i];
            let (kind_b, pos_b) = door_positions[j];

            // Try normal pathfinding first.
            if surface_pathfinding::path_exists(pos_a, pos_b, &cost_map, map_w, map_h) {
                continue;
            }

            // Disconnected — find cheapest crossing using high-cost solid tiles.
            let crossing = find_crossing_path(
                pos_a, pos_b, terrain, collision_codes, movement_costs,
                solid_building_tiles, map_w, map_h,
            );

            for (tx, ty) in &crossing {
                let idx = (*ty * map_w + *tx) as usize;
                if collision[idx] == 1 {
                    // This tile is solid — force it to walkable.
                    forced_tiles.push((*tx, *ty));
                }
            }
        }
    }

    if forced_tiles.is_empty() {
        return;
    }

    bevy::log::debug!(
        "[terrain] Forced {} tiles to walkable for connectivity",
        forced_tiles.len()
    );

    // Force the tiles and re-clamp.
    let w = map_w as usize;
    let h = map_h as usize;
    let mut pinned = vec![false; w * h];

    // Pin borders.
    for x in 0..w {
        pinned[x] = true;
        pinned[(h - 1) * w + x] = true;
    }
    for y in 0..h {
        pinned[y * w] = true;
        pinned[y * w + (w - 1)] = true;
    }

    // Pin forced path tiles to walkable.
    for &(tx, ty) in &forced_tiles {
        let idx = ty as usize * w + tx as usize;
        terrain[idx] = walkable_terrain;
        pinned[idx] = true;
    }

    // Re-clamp gradients.
    crate::fbm::clamp_terrain_indices(terrain, w, h, &pinned);

    // Re-derive collision.
    *collision = derive_collision(terrain, collision_codes);
}

/// Find a path between two points, routing through solid *terrain* tiles
/// at high cost while keeping building footprints truly impassable.
///
/// Solid terrain tiles get cost 1000 (high but finite) so Dijkstra finds
/// the crossing that goes through the fewest solid tiles.  Building
/// interiors (except doors) stay at INFINITY since they can't be fixed
/// by changing terrain.
fn find_crossing_path(
    start: (u32, u32),
    goal: (u32, u32),
    terrain: &[u32],
    collision_codes: &[u8],
    movement_costs: &[f32],
    solid_building_tiles: &std::collections::HashSet<(u32, u32)>,
    map_w: u32,
    map_h: u32,
) -> Vec<(u32, u32)> {
    let n = (map_w * map_h) as usize;
    let mut cost_map = vec![0.0f32; n];
    const SOLID_TERRAIN_PENALTY: f32 = 1000.0;

    for y in 0..map_h {
        for x in 0..map_w {
            let idx = (y * map_w + x) as usize;

            // Solid building tiles are truly impassable — can't be
            // fixed by changing terrain.
            if solid_building_tiles.contains(&(x, y)) {
                cost_map[idx] = f32::INFINITY;
                continue;
            }

            let col_code = collision_codes.get(terrain[idx] as usize).copied().unwrap_or(0);
            let base_cost = movement_costs.get(terrain[idx] as usize).copied().unwrap_or(1.0);

            if col_code == 1 {
                // Solid terrain — high cost but passable (can be fixed
                // by changing the terrain type).
                cost_map[idx] = SOLID_TERRAIN_PENALTY;
            } else {
                cost_map[idx] = base_cost;
            }
        }
    }

    surface_pathfinding::dijkstra_path_pub(start, goal, &cost_map, map_w, map_h)
        .unwrap_or_default()
}
