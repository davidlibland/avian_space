//! World tile-atlas assets and collision helpers.
//!
//! Generated tile atlases and collision maps live under
//! `assets/sprites/worlds/`. This module provides:
//! - RON asset types for loading them via `bevy_common_assets`
//! - Blob-47 bitmask helpers for auto-tiling
//! - A collision spawner that creates Avian2D colliders from collision maps
//!
//! See `scripts/tilegen.py` for the asset generator.

use avian2d::prelude::*;
use bevy::prelude::*;
use bevy_common_assets::ron::RonAssetPlugin;
use serde::Deserialize;
use std::collections::HashMap;

// ── Collision type ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CollisionType {
    Walkable = 0,
    Solid = 1,
    Slow = 2,
    Damaging = 3,
    Trigger = 4,
}

impl From<u8> for CollisionType {
    fn from(v: u8) -> Self {
        match v {
            1 => Self::Solid,
            2 => Self::Slow,
            3 => Self::Damaging,
            4 => Self::Trigger,
            _ => Self::Walkable,
        }
    }
}

// ── RON asset types ───────────────────────────────────────────────────────

#[derive(Asset, TypePath, Deserialize, Debug)]
pub struct Blob47Lut {
    pub atlas_cols: u32,
    /// 256 entries; 255 = unmapped.
    pub lut: Vec<u8>,
}

/// Collision data per tile, used by `spawn_collision_entities`.
/// No longer loaded from RON — built at runtime from fBm terrain + manifest.
pub struct CollisionMapAsset {
    pub width: u32,
    pub height: u32,
    /// `CollisionType` per tile, row-major.
    pub data: Vec<u8>,
}

#[derive(Deserialize, Debug, Clone)]
#[allow(dead_code)]
pub struct TerrainMeta {
    pub name: String,
    pub row: u32,
    pub threshold: f32,
    pub collision: u8,
    pub movement_cost: f32,
    pub damage_per_sec: f32,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
pub struct BiomeMeta {
    pub atlas: String,
    pub soft_boundaries: bool,
    pub terrains: Vec<TerrainMeta>,
}

#[derive(Asset, TypePath, Deserialize, Debug)]
#[allow(dead_code)]
pub struct WorldManifest {
    pub tile_size: u32,
    pub atlas_cols: u32,
    pub biomes: HashMap<String, BiomeMeta>,
}

// ── Building types ───────────────────────────────────────────────────────

#[derive(Deserialize, Debug)]
pub struct BuildingTemplate {
    pub name: String,
    pub style: String,
    pub width: u32,
    pub height: u32,
    #[allow(dead_code)]
    pub layer: u32,
    /// `tiles[row][col]` — exterior atlas index (0 = transparent/skip).
    pub tiles: Vec<Vec<u32>>,
    /// `(col, row)` positions of door tiles.
    pub entry_points: Vec<(u32, u32)>,
    #[allow(dead_code)]
    pub interior_offset: (u32, u32),
    #[allow(dead_code)]
    pub interior_size: (u32, u32),
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
pub struct BuildingStyleMeta {
    pub biome: String,
    pub exterior_atlas: String,
    pub interior_atlas: String,
    pub landing_pad_atlas: String,
    pub templates: Vec<String>,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
pub struct BuildingsManifest {
    pub tile_size: u32,
    pub ext_cols: u32,
    pub ext_rows: u32,
    pub int_cols: u32,
    pub int_rows: u32,
    pub blob4_lut: Vec<u32>,
    pub ext_collision: Vec<u8>,
    pub styles: HashMap<String, BuildingStyleMeta>,
}

/// Map a biome name to the building style used on that biome's planets.
pub fn biome_to_building_style(biome: &str) -> &'static str {
    match biome {
        "garden" => "colony",
        "ice" => "cryo",
        "rocky" => "extraction",
        "desert" => "outpost",
        "interior" => "station",
        _ => "colony",
    }
}

// ── Blob-47 bitmask helpers ───────────────────────────────────────────────
//
// Boris The Brave's standard ordering.
// https://www.boristhebrave.com/2013/07/14/tileset-roundup/

const TL: u8 = 1;
const T: u8 = 2;
const TR: u8 = 4;
const L: u8 = 8;
const R: u8 = 16;
const BL: u8 = 32;
const B: u8 = 64;
const BR: u8 = 128;

pub fn reduce_to_47(mut mask: u8) -> u8 {
    if mask & L == 0 || mask & T == 0 {
        mask &= !TL;
    }
    if mask & R == 0 || mask & T == 0 {
        mask &= !TR;
    }
    if mask & L == 0 || mask & B == 0 {
        mask &= !BL;
    }
    if mask & R == 0 || mask & B == 0 {
        mask &= !BR;
    }
    mask
}

pub fn compute_bitmask(
    terrain_map: &[Vec<u32>],
    x: i32,
    y: i32,
    width: i32,
    height: i32,
) -> u8 {
    let terrain = terrain_map[y as usize][x as usize];
    // A neighbour counts as "same" when its terrain index is >= ours.
    // The atlas only contains downward transitions (from higher terrain
    // rows toward lower ones), so a higher-indexed neighbour is visually
    // identical to same-terrain from this tile's perspective — the
    // higher-terrain tile handles the transition on its side.
    let same = |dx: i32, dy: i32| -> bool {
        let nx = x + dx;
        let ny = y + dy;
        if nx < 0 || ny < 0 || nx >= width || ny >= height {
            true
        } else {
            terrain_map[ny as usize][nx as usize] >= terrain
        }
    };
    // Bottom-up convention: y increases upward.
    // y+1 = up (T), y-1 = down (B).
    let mut mask = 0u8;
    if same(0, 1) { mask |= T; }
    if same(1, 0) { mask |= R; }
    if same(0, -1) { mask |= B; }
    if same(-1, 0) { mask |= L; }
    if same(-1, 1) { mask |= TL; }
    if same(1, 1) { mask |= TR; }
    if same(1, -1) { mask |= BR; }
    if same(-1, -1) { mask |= BL; }
    mask
}

// ── Tilemap builder ───────────────────────────────────────────────────────

/// Compute the `TileTextureIndex` for a given position.
///
/// `texture_index = terrain_row * atlas_cols + blob47_lut[reduce_to_47(bitmask)]`
pub fn tile_texture_index(
    terrain_map: &[Vec<u32>],
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    lut: &Blob47Lut,
) -> u32 {
    let terrain_row = terrain_map[y as usize][x as usize];
    let mask = compute_bitmask(terrain_map, x, y, width, height);
    let reduced = reduce_to_47(mask);
    let col = lut.lut[reduced as usize];
    // Safety fallback: strip diagonals if somehow unmapped, else use
    // the fully-interior tile (col 46 = mask 255).
    let col = if col == 255 {
        let cardinals_only = reduced & (T | R | B | L);
        let fallback = lut.lut[cardinals_only as usize];
        if fallback == 255 { 46 } else { fallback }
    } else {
        col
    };
    terrain_row * lut.atlas_cols + col as u32
}

// ── Planet type → biome mapping ───────────────────────────────────────────

/// Map a `planet_type` string (from YAML) to a biome name (matching
/// the keys in `world_manifest.ron`).
pub fn planet_type_to_biome(planet_type: &str) -> &'static str {
    match planet_type {
        "habitable" => "garden",
        "cloud" => "interior",
        "icy_dwarf" | "ice_giant" => "ice",
        "rocky" => "rocky",
        "desert" => "desert",
        "gas_giant" => "interior",
        _ => "rocky", // fallback
    }
}

// ── Bevy plugin ───────────────────────────────────────────────────────────

pub struct WorldAssetsPlugin;

impl Plugin for WorldAssetsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(RonAssetPlugin::<Blob47Lut>::new(&["blob47_lut.ron"]))
            .add_plugins(RonAssetPlugin::<WorldManifest>::new(&[
                "world_manifest.ron",
            ]));
    }
}

// ── Collision spawner ─────────────────────────────────────────────────────

/// Spawn Avian2D colliders from a loaded `CollisionMapAsset`.
///
/// `map_origin` is the world position of tile (0, 0). Solid tiles get
/// static `RigidBody` + `Collider`; slow/damaging/trigger tiles get
/// `Sensor` colliders with a [`TerrainSensor`] component.
/// `movement_costs` maps terrain index → movement_cost multiplier.
/// Passed through to `TerrainSensor` on sensor tiles.
pub fn spawn_collision_entities(
    commands: &mut Commands,
    col_asset: &CollisionMapAsset,
    terrain_indices: &[u32],
    movement_costs: &[f32],
    tile_size: f32,
    map_origin: Vec2,
    layers: CollisionLayers,
) {
    let half = tile_size * 0.5;
    for ty in 0..col_asset.height {
        for tx in 0..col_asset.width {
            let idx = (ty * col_asset.width + tx) as usize;
            let code = CollisionType::from(col_asset.data[idx]);
            let terrain_idx = terrain_indices.get(idx).copied().unwrap_or(0) as usize;
            let cost = movement_costs.get(terrain_idx).copied().unwrap_or(1.0);
            let world_pos = map_origin
                + Vec2::new(tx as f32 * tile_size + half, ty as f32 * tile_size + half);
            match code {
                CollisionType::Solid => {
                    commands.spawn((
                        DespawnOnExit(crate::PlayState::Exploring),
                        RigidBody::Static,
                        Collider::rectangle(tile_size, tile_size),
                        layers,
                        Transform::from_translation(world_pos.extend(0.0)),
                    ));
                }
                CollisionType::Slow | CollisionType::Damaging | CollisionType::Trigger => {
                    commands.spawn((
                        DespawnOnExit(crate::PlayState::Exploring),
                        Sensor,
                        Collider::rectangle(tile_size, tile_size),
                        layers,
                        Transform::from_translation(world_pos.extend(0.0)),
                        TerrainSensor {
                            collision_type: code,
                            movement_cost: cost,
                        },
                    ));
                }
                CollisionType::Walkable => {}
            }
        }
    }
}

/// Component on sensor tiles (slow / damaging / trigger) carrying terrain
/// properties for gameplay effects.
#[derive(Component)]
pub struct TerrainSensor {
    pub collision_type: CollisionType,
    /// Movement speed multiplier (1.0 = normal, 2.0 = half speed).
    pub movement_cost: f32,
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Boris The Brave's canonical pick_tile dictionary.
    /// https://www.boristhebrave.com/2013/07/14/tileset-roundup/
    const PICK_TILE: [(u8, u8); 47] = [
        (0, 0), (2, 1), (8, 2), (10, 3), (11, 4), (16, 5), (18, 6),
        (22, 7), (24, 8), (26, 9), (27, 10), (30, 11), (31, 12), (64, 13),
        (66, 14), (72, 15), (74, 16), (75, 17), (80, 18), (82, 19), (86, 20),
        (88, 21), (90, 22), (91, 23), (94, 24), (95, 25), (104, 26), (106, 27),
        (107, 28), (120, 29), (122, 30), (123, 31), (126, 32), (127, 33),
        (208, 34), (210, 35), (214, 36), (216, 37), (218, 38), (219, 39),
        (222, 40), (223, 41), (248, 42), (250, 43), (251, 44), (254, 45), (255, 46),
    ];

    /// Build the 256-entry LUT from the pick_tile table (same logic as tilegen.py).
    fn build_lut() -> Blob47Lut {
        let mut lut = vec![255u8; 256];
        for &(mask, col) in &PICK_TILE {
            lut[mask as usize] = col;
        }
        Blob47Lut { atlas_cols: 48, lut }
    }

    #[test]
    fn reduce_to_47_matches_pick_tile() {
        // Every reduced mask in pick_tile must be a fixed point of reduce_to_47.
        for &(mask, _col) in &PICK_TILE {
            assert_eq!(
                reduce_to_47(mask), mask,
                "pick_tile mask {mask} is not a fixed point of reduce_to_47"
            );
        }
    }

    #[test]
    fn reduce_to_47_is_idempotent() {
        for m in 0..=255u8 {
            let once = reduce_to_47(m);
            let twice = reduce_to_47(once);
            assert_eq!(once, twice, "mask {m}: reduce({m})={once}, reduce({once})={twice}");
        }
    }

    #[test]
    fn reduce_to_47_always_maps() {
        let lut = build_lut();
        for m in 0..=255u8 {
            let reduced = reduce_to_47(m);
            let col = lut.lut[reduced as usize];
            assert_ne!(col, 255, "mask {m} reduces to {reduced} which is unmapped");
        }
    }

    #[test]
    fn exactly_47_entries() {
        let lut = build_lut();
        let mapped = lut.lut.iter().filter(|&&v| v != 255).count();
        assert_eq!(mapped, 47, "expected 47 mapped entries, got {mapped}");
    }

    /// Parse a 3x3 ASCII grid into a raw 8-bit bitmask.
    ///
    /// Grid layout (rows top-to-bottom):
    ///   TL T TR
    ///   L  X  R
    ///   BL B BR
    fn parse_grid(grid: &str) -> u8 {
        let rows: Vec<&str> = grid.trim().lines().map(|l| l.trim()).collect();
        assert_eq!(rows.len(), 3, "expected 3 rows");
        let chars: Vec<Vec<char>> = rows.iter().map(|r| r.chars().collect()).collect();
        let mut mask = 0u8;
        if chars[0][0] == '#' { mask |= TL; }
        if chars[0][1] == '#' { mask |= T; }
        if chars[0][2] == '#' { mask |= TR; }
        if chars[1][0] == '#' { mask |= L; }
        if chars[1][2] == '#' { mask |= R; }
        if chars[2][0] == '#' { mask |= BL; }
        if chars[2][1] == '#' { mask |= B; }
        if chars[2][2] == '#' { mask |= BR; }
        mask
    }

    #[derive(serde::Deserialize)]
    struct TestCase {
        name: String,
        grid: String,
        mask: u8,
        column: u8,
    }

    #[test]
    fn yaml_test_cases() {
        let yaml_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("scripts/blob47_test_cases.yaml");
        let yaml_str = std::fs::read_to_string(&yaml_path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", yaml_path.display()));
        let cases: Vec<TestCase> = serde_yaml::from_str(&yaml_str)
            .expect("Failed to parse YAML test cases");

        let lut = build_lut();

        for case in &cases {
            let raw = parse_grid(&case.grid);
            let reduced = reduce_to_47(raw);
            assert_eq!(
                reduced, case.mask,
                "[{}] raw={raw}, reduced={reduced}, expected mask={}",
                case.name, case.mask
            );
            let col = lut.lut[reduced as usize];
            assert_eq!(
                col, case.column,
                "[{}] mask={reduced} → col={col}, expected {}",
                case.name, case.column
            );
        }
    }

    #[test]
    fn ron_lut_matches_rust() {
        // Load the blob47_lut.ron that tilegen.py generated and verify
        // every entry matches the Rust-side pick_tile table.
        let ron_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets/sprites/worlds/blob47_lut.ron");
        let ron_str = std::fs::read_to_string(&ron_path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", ron_path.display()));
        let ron_lut: Blob47Lut = ron::from_str(&ron_str)
            .expect("Failed to parse blob47_lut.ron");

        let rust_lut = build_lut();

        assert_eq!(
            ron_lut.atlas_cols, rust_lut.atlas_cols,
            "atlas_cols mismatch: ron={}, rust={}",
            ron_lut.atlas_cols, rust_lut.atlas_cols
        );
        assert_eq!(
            ron_lut.lut.len(), rust_lut.lut.len(),
            "LUT length mismatch: ron={}, rust={}",
            ron_lut.lut.len(), rust_lut.lut.len()
        );
        for (i, (&ron_val, &rust_val)) in ron_lut.lut.iter().zip(rust_lut.lut.iter()).enumerate() {
            assert_eq!(
                ron_val, rust_val,
                "LUT divergence at index {i}: ron={ron_val}, rust={rust_val} — \
                 regenerate with `python scripts/tilegen.py`"
            );
        }
    }

    #[test]
    fn compute_bitmask_isolated_by_higher() {
        // Center=1 surrounded by 0 (lower) → all neighbours are "different"
        let map = vec![
            vec![0, 0, 0],
            vec![0, 1, 0],
            vec![0, 0, 0],
        ];
        let mask = compute_bitmask(&map, 1, 1, 3, 3);
        assert_eq!(reduce_to_47(mask), 0, "higher tile surrounded by lower should be isolated");
    }

    #[test]
    fn compute_bitmask_lowest_terrain_always_interior() {
        // Center=0 surrounded by higher terrains → all >= 0 → all "same"
        let map = vec![
            vec![2, 1, 3],
            vec![1, 0, 2],
            vec![3, 1, 2],
        ];
        let mask = compute_bitmask(&map, 1, 1, 3, 3);
        assert_eq!(mask, 255, "lowest terrain sees all neighbours as same (>=)");
    }

    #[test]
    fn compute_bitmask_fully_interior() {
        let map = vec![
            vec![1, 1, 1],
            vec![1, 1, 1],
            vec![1, 1, 1],
        ];
        let mask = compute_bitmask(&map, 1, 1, 3, 3);
        assert_eq!(mask, 255, "fully surrounded by same terrain should have mask 255");
    }

    #[test]
    fn compute_bitmask_higher_neighbour_is_same() {
        // Bottom-up: row 0 = bottom, row 2 = top.
        // Center=1 at (1,1), top neighbour at (1,2)=2 (higher) → "same"
        // All other neighbours=0 (lower) → "different"
        let map = vec![
            vec![0, 0, 0], // row 0 = bottom
            vec![0, 1, 0], // row 1 = center
            vec![0, 2, 0], // row 2 = top
        ];
        let mask = compute_bitmask(&map, 1, 1, 3, 3);
        let reduced = reduce_to_47(mask);
        assert_eq!(reduced, T, "higher neighbour above should count as same → T bit set");
    }

    #[test]
    fn compute_bitmask_lower_neighbour_is_different() {
        // Bottom-up: row 2 = top.
        // Center=2 at (1,1), top neighbour at (1,2)=2 (same), all others=1 (lower)
        let map = vec![
            vec![1, 1, 1], // row 0 = bottom
            vec![1, 2, 1], // row 1 = center
            vec![1, 2, 1], // row 2 = top
        ];
        let mask = compute_bitmask(&map, 1, 1, 3, 3);
        let reduced = reduce_to_47(mask);
        assert_eq!(reduced, T, "only top is same/higher, rest are lower → T only");
    }
}
