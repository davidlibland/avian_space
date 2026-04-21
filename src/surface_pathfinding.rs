//! Pathfinding on the surface tilemap for AI characters.
//!
//! Provides:
//! - Pre-computed building-to-building paths (`SurfacePaths`)
//! - A stored cost map (`SurfaceCostMap`) for runtime pathfinding
//! - `find_path()` for arbitrary tile-to-tile pathfinding
//! - `find_path_to_entity()` helper for "seek/flee" behaviors

use bevy::prelude::*;
use std::collections::HashMap;

use crate::surface::{BuildingKind, TILE_PX, WORLD_WIDTH, WORLD_HEIGHT};

// ── Public types ─────────────────────────────────────────────────────────

/// The cost map for the current surface, stored as a resource so any
/// system can request a path at runtime.
#[derive(Resource)]
pub struct SurfaceCostMap {
    pub data: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

impl SurfaceCostMap {
    /// Find a path from `start` to `goal` in tile coordinates.
    /// Returns `None` if unreachable.
    pub fn find_path(&self, start: (u32, u32), goal: (u32, u32)) -> Option<Vec<(u32, u32)>> {
        dijkstra_path(start, goal, &self.data, self.width, self.height)
    }

    /// Convert a world position to a tile coordinate.
    pub fn world_to_tile(world_pos: Vec2) -> (u32, u32) {
        let tx = ((world_pos.x / TILE_PX) + WORLD_WIDTH as f32 / 2.0)
            .clamp(0.0, WORLD_WIDTH as f32 - 1.0) as u32;
        let ty = ((world_pos.y / TILE_PX) + WORLD_HEIGHT as f32 / 2.0)
            .clamp(0.0, WORLD_HEIGHT as f32 - 1.0) as u32;
        (tx, ty)
    }

    /// Convert a tile coordinate to a world position (tile center).
    pub fn tile_to_world(tx: u32, ty: u32) -> Vec2 {
        Vec2::new(
            (tx as f32 - WORLD_WIDTH as f32 / 2.0) * TILE_PX + TILE_PX / 2.0,
            (ty as f32 - WORLD_HEIGHT as f32 / 2.0) * TILE_PX + TILE_PX / 2.0,
        )
    }
}

/// All pre-computed paths between building doors on the current surface.
/// Inserted as a resource when the surface is set up.
#[derive(Resource, Default)]
pub struct SurfacePaths {
    /// Paths indexed by (from_kind, to_kind).  Each entry is the sequence
    /// of tile positions from the source door to the destination door.
    pub paths: HashMap<(BuildingKind, BuildingKind), Vec<(u32, u32)>>,
}

// ── Pathfinding ──────────────────────────────────────────────────────────

/// Dijkstra on 8-connected tiles, weighted by movement cost.
///
/// `cost_map` has one entry per tile (row-major, same layout as col_data).
/// `f32::INFINITY` = impassable (solid/building wall).
/// Diagonal moves cost `√2 × tile_cost`.
///
/// Returns the shortest (cheapest) path from `start` to `goal` as a list
/// of tile coordinates (inclusive of both endpoints), or `None` if unreachable.
fn dijkstra_path(
    start: (u32, u32),
    goal: (u32, u32),
    cost_map: &[f32],
    map_w: u32,
    map_h: u32,
) -> Option<Vec<(u32, u32)>> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    if start == goal {
        return Some(vec![start]);
    }

    let n = (map_w * map_h) as usize;
    let w = map_w as i32;
    let h = map_h as i32;
    let idx = |x: u32, y: u32| (y * map_w + x) as usize;

    #[derive(PartialEq)]
    struct State { cost: f32, pos: (u32, u32) }
    impl Eq for State {}
    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
    }
    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
        }
    }

    let mut dist = vec![f32::INFINITY; n];
    let mut parent: Vec<Option<(u32, u32)>> = vec![None; n];
    let mut heap = BinaryHeap::new();

    dist[idx(start.0, start.1)] = 0.0;
    heap.push(State { cost: 0.0, pos: start });

    const DIRS: [(i32, i32); 8] = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1),
    ];
    const SQRT2: f32 = 1.414;

    while let Some(State { cost, pos: (cx, cy) }) = heap.pop() {
        if (cx, cy) == goal {
            // Reconstruct path.
            let mut path = vec![goal];
            let mut cur = goal;
            while cur != start {
                cur = parent[idx(cur.0, cur.1)].unwrap();
                path.push(cur);
            }
            path.reverse();
            return Some(path);
        }

        if cost > dist[idx(cx, cy)] {
            continue; // stale entry
        }

        for &(dx, dy) in &DIRS {
            let nx = cx as i32 + dx;
            let ny = cy as i32 + dy;
            if nx < 0 || ny < 0 || nx >= w || ny >= h {
                continue;
            }
            let (nx, ny) = (nx as u32, ny as u32);
            let ni = idx(nx, ny);
            let tile_cost = cost_map[ni];
            if tile_cost == f32::INFINITY {
                continue; // impassable
            }

            // For diagonal moves, block if either adjacent cardinal tile
            // is impassable.  This prevents corner-cutting through walls.
            if dx != 0 && dy != 0 {
                let adj_x = idx(nx, cy as u32); // horizontal neighbour
                let adj_y = idx(cx as u32, ny); // vertical neighbour
                if cost_map[adj_x] == f32::INFINITY || cost_map[adj_y] == f32::INFINITY {
                    continue;
                }
            }

            let step = if dx != 0 && dy != 0 { SQRT2 } else { 1.0 };
            let new_cost = cost + step * tile_cost;
            if new_cost < dist[ni] {
                dist[ni] = new_cost;
                parent[ni] = Some((cx, cy));
                heap.push(State { cost: new_cost, pos: (nx, ny) });
            }
        }
    }

    None // unreachable
}

// ── Public API ───────────────────────────────────────────────────────────

/// Build a cost map from terrain data and collision/movement costs.
///
/// Solid tiles and solid building tiles get `f32::INFINITY`.
/// Other tiles get their `movement_cost` value.
pub fn build_cost_map(
    col_data: &[u8],
    terrain_flat: &[u32],
    movement_costs: &[f32],
    solid_building_tiles: &std::collections::HashSet<(u32, u32)>,
    map_w: u32,
    map_h: u32,
) -> Vec<f32> {
    let n = (map_w * map_h) as usize;
    let mut cost_map = vec![f32::INFINITY; n];

    for y in 0..map_h {
        for x in 0..map_w {
            let idx = (y * map_w + x) as usize;

            // Solid terrain tiles are impassable.
            if col_data[idx] == 1 {
                continue; // stays INFINITY
            }

            // Solid building tiles are impassable.
            if solid_building_tiles.contains(&(x, y)) {
                continue; // stays INFINITY
            }

            // Use movement cost from the terrain type.
            let terrain_idx = terrain_flat[idx] as usize;
            let mc = movement_costs.get(terrain_idx).copied().unwrap_or(1.0);
            cost_map[idx] = mc;
        }
    }

    cost_map
}

/// Quick connectivity check: returns true if a path exists between two points.
pub fn path_exists(
    start: (u32, u32),
    goal: (u32, u32),
    cost_map: &[f32],
    map_w: u32,
    map_h: u32,
) -> bool {
    dijkstra_path(start, goal, cost_map, map_w, map_h).is_some()
}

/// Public wrapper around dijkstra_path for use by surface_terrain.
pub fn dijkstra_path_pub(
    start: (u32, u32),
    goal: (u32, u32),
    cost_map: &[f32],
    map_w: u32,
    map_h: u32,
) -> Option<Vec<(u32, u32)>> {
    dijkstra_path(start, goal, cost_map, map_w, map_h)
}

/// Compute shortest paths between all pairs of building doors.
///
/// `door_positions` maps `BuildingKind` → tile coordinate of the door.
/// `cost_map` has per-tile movement costs (INFINITY = impassable).
///
/// Disconnected pairs are skipped (logged as warnings). Civilians will
/// only walk routes that have valid paths.
pub fn compute_all_paths(
    door_positions: &[(BuildingKind, (u32, u32))],
    cost_map: &[f32],
    map_w: u32,
    map_h: u32,
) -> SurfacePaths {
    let mut paths = HashMap::new();

    // Compute paths between all pairs (skip disconnected).
    for i in 0..door_positions.len() {
        for j in (i + 1)..door_positions.len() {
            let (kind_a, pos_a) = door_positions[i];
            let (kind_b, pos_b) = door_positions[j];

            match dijkstra_path(pos_a, pos_b, cost_map, map_w, map_h) {
                Some(path) => {
                    let reverse: Vec<(u32, u32)> = path.iter().copied().rev().collect();
                    paths.insert((kind_a, kind_b), path);
                    paths.insert((kind_b, kind_a), reverse);
                }
                None => {
                    eprintln!(
                        "[pathfinding] No path: {:?} at ({},{}) → {:?} at ({},{}), skipping",
                        kind_a, pos_a.0, pos_a.1,
                        kind_b, pos_b.0, pos_b.1,
                    );
                }
            }
        }
    }

    SurfacePaths { paths }
}
