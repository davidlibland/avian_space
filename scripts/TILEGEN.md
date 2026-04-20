# tilegen — tile atlas pipeline

This document explains the complete tile generation pipeline: what the Python
scripts produce, the exact layout of every output file, and how to consume
those files correctly in Bevy/Rust.

---

## Overview

The pipeline has two Python entry points:

| Script | Purpose |
|---|---|
| `tilegen.py` | Terrain atlases, collision maps, terrain maps, LUT, manifest |
| `buildings.py` | Building exterior/interior atlases, templates |
| `buildings_scifi.py` | Sci-fi variants of building styles (imports `buildings.py`) |

Run order: `tilegen.py` first (produces terrain assets), then optionally
`buildings_scifi.py` for building assets. `buildings.py` is a library module,
not run directly.

```bash
python tilegen.py --tile-size 32 --map-preview 64 --out-dir assets/worlds
python buildings_scifi.py --tile-size 32 --out-dir assets/worlds
```

All output goes to `--out-dir`. Copy the entire output directory into your
Bevy `assets/` folder.

---

## Terrain pipeline (`tilegen.py`)

### Biomes

Five biomes are defined in `ALL_BIOMES`:

| Biome name | Planet types | Terrains (low→high noise) |
|---|---|---|
| `garden` | terran, jungle | water, sand, grass, forest, mountain |
| `ice` | frozen, arctic | deep_ice, ice, snow, ice_rock, crevasse |
| `rocky` | barren, volcanic | lava, basalt, rock, dust, cliff |
| `desert` | arid, dune | saltpan, sand, hardpan, sandstone, mesa |
| `interior` | station, city | floor, plating, grate, wall, conduit |

Each biome has exactly 5 terrain types. This is important: the atlas layout
assumes exactly 5 rows.

### Noise field

Terrain assignment uses fractional Brownian motion (fBm):

```
f(x,y) = sum_{k=0}^{K-1}  gain^k * noise2(lacunarity^k * x,  lacunarity^k * y)
```

The raw fBm output is normalised to `[0, 1]`. Each `TerrainSpec` has a
`threshold` field; the terrain with the lowest threshold that the noise value
does not exceed is assigned to that tile. Thresholds are ordered low to high,
with the last terrain always having `threshold = 1.0` to catch everything.

The noise field is **stored in image-convention row order**: row 0 is the
top of the map, y increases downward. `bevy_ecs_tilemap` uses the opposite
convention (y=0 is the bottom). See the Y-axis section below.

### Output files per biome

```
{biome}_atlas.png       — visual tile sprites (see Atlas Layout below)
{biome}_collision.ron   — CollisionType per tile, row-major, top-down
{biome}_terrain.ron     — terrain row index per tile, row-major, top-down
{biome}_map_preview.png — debug image (terrain colours + collision overlay)
```

Shared files (one per run):

```
blob47_lut.ron      — 256-entry mask→column lookup table
world_manifest.ron  — all biome metadata, terrain thresholds, collision codes
world_assets.rs     — generated Rust types and helper functions (copy into project)
```

---

## Atlas layout

Each `{biome}_atlas.png` is:

```
width  = 48 * TILE_SIZE   (48 columns: 47 blob tiles + 1 padding)
height =  5 * TILE_SIZE   (5 rows: one per terrain type)
```

With `TILE_SIZE = 32`: `1536 × 160` px.

### Row = terrain type

```
row 0  →  terrain 0  (lowest noise threshold, e.g. water / lava / deep_ice)
row 1  →  terrain 1
row 2  →  terrain 2
row 3  →  terrain 3
row 4  →  terrain 4  (highest threshold, always = 1.0)
```

### Column = blob-47 autotile variant

Columns 0–46 are the 47 standard blob autotile configurations, using
**Boris The Brave's standard ordering** (sorted by reduced bitmask value).
Column 47 is a padding copy of column 46 (the fully-interior tile, all 8
neighbours same type).

Reference: https://www.boristhebrave.com/2013/07/14/tileset-roundup/

#### Neighbour bitmask

Each tile's neighbours are encoded as an 8-bit mask. Bit assignment
(Boris The Brave's standard):

```
TL=1 | T=2   | TR=4
L=8  |  X    | R=16
BL=32| B=64  | BR=128
```

A bit is **set** when the neighbour in that direction is the **same terrain type**
as the centre tile. Out-of-bounds neighbours are treated as same-type (so map
edges appear fully bordered).

#### Diagonal reduction

Before looking up the column, diagonal bits that don't affect the rendered
border are cleared. A diagonal is only relevant when **both** adjacent cardinal
neighbours are the same type — otherwise the diagonal corner is hidden behind a
cardinal edge and makes no visual difference:

```
if not left  or not top:     clear topLeft
if not right or not top:     clear topRight
if not left  or not bottom:  clear bottomLeft
if not right or not bottom:  clear bottomRight
```

This reduces 256 possible masks to 47 visually distinct cases.

#### Column assignment

The 47 valid reduced masks are sorted numerically. The atlas column is the
index in this sorted list. This is the `pick_tile` dictionary from Boris
The Brave's website:

```python
pick_tile = {
    0: 0, 2: 1, 8: 2, 10: 3, 11: 4, 16: 5, 18: 6,
    22: 7, 24: 8, 26: 9, 27: 10, 30: 11, 31: 12, 64: 13,
    66: 14, 72: 15, 74: 16, 75: 17, 80: 18, 82: 19, 86: 20,
    88: 21, 90: 22, 91: 23, 94: 24, 95: 25, 104: 26, 106: 27,
    107: 28, 120: 29, 122: 30, 123: 31, 126: 32, 127: 33,
    208: 34, 210: 35, 214: 36, 216: 37, 218: 38, 219: 39,
    222: 40, 223: 41, 248: 42, 250: 43, 251: 44, 254: 45, 255: 46,
}
```

#### Full column map (all 47 entries)

Notation: `#` = same-type neighbour, `.` = different type or absent.
Diagonal cells only shown when both adjacent cardinals are `#`.

```
Col  Mask  Description              Pattern (TL T TR / L X R / BL B BR)
───  ────  ────────────────────────  ────────────────────────────────────
 0      0  (none)                    . . .  / . X . / . . .   isolated

 1      2  T                         . # .  / . X . / . . .
 2      8  L                         . . .  / # X . / . . .
 3     10  T+L                       . # .  / # X . / . . .
 4     11  T+L+TL                    # # .  / # X . / . . .   inner TL corner
 5     16  R                         . . .  / . X # / . . .
 6     18  T+R                       . # .  / . X # / . . .
 7     22  T+R+TR                    . # #  / . X # / . . .   inner TR corner
 8     24  L+R                       . . .  / # X # / . . .   horizontal corridor
 9     26  T+L+R                     . # .  / # X # / . . .   T open-B
10     27  T+L+R+TL                  # # .  / # X # / . . .   T open-B + TL
11     30  T+L+R+TR                  . # #  / # X # / . . .   T open-B + TR
12     31  T+L+R+TL+TR              # # #  / # X # / . . .   T open-B + TL+TR

13     64  B                         . . .  / . X . / . # .
14     66  T+B                       . # .  / . X . / . # .   vertical corridor
15     72  B+L                       . . .  / # X . / . # .
16     74  T+B+L                     . # .  / # X . / . # .   T open-R
17     75  T+B+L+TL                  # # .  / # X . / . # .   T open-R + TL
18     80  B+R                       . . .  / . X # / . # .
19     82  T+B+R                     . # .  / . X # / . # .   T open-L
20     86  T+B+R+TR                  . # #  / . X # / . # .   T open-L + TR
21     88  B+L+R                     . . .  / # X # / . # .   T open-T
22     90  T+B+L+R                   . # .  / # X # / . # .   cross (no diagonals)

── All 4 cardinals set, diagonal variants ──────────────────────────────────────
23     91  T+B+L+R+TL               # # .  / # X # / . # .   +TL
24     94  T+B+L+R+TR               . # #  / # X # / . # .   +TR
25     95  T+B+L+R+TL+TR            # # #  / # X # / . # .   +TL+TR (T edge filled)

26    104  B+L+BL                    . . .  / # X . / # # .   inner BL corner
27    106  T+B+L+BL                  . # .  / # X . / # # .   T open-R + BL
28    107  T+B+L+TL+BL              # # .  / # X . / # # .   T open-R + TL+BL
29    120  B+L+R+BL                  . . .  / # X # / # # .   T open-T + BL
30    122  T+B+L+R+BL               . # .  / # X # / # # .   +BL
31    123  T+B+L+R+TL+BL            # # .  / # X # / # # .   +TL+BL
32    126  T+B+L+R+TR+BL            . # #  / # X # / # # .   +TR+BL
33    127  T+B+L+R+TL+TR+BL         # # #  / # X # / # # .   +TL+TR+BL

34    208  B+R+BR                    . . .  / . X # / . # #   inner BR corner
35    210  T+B+R+BR                  . # .  / . X # / . # #   T open-L + BR
36    214  T+B+R+TR+BR              . # #  / . X # / . # #   T open-L + TR+BR
37    216  B+L+R+BR                  . . .  / # X # / . # #   T open-T + BR
38    218  T+B+L+R+BR               . # .  / # X # / . # #   +BR
39    219  T+B+L+R+TL+BR            # # .  / # X # / . # #   +TL+BR
40    222  T+B+L+R+TR+BR            . # #  / # X # / . # #   +TR+BR
41    223  T+B+L+R+TL+TR+BR         # # #  / # X # / . # #   +TL+TR+BR

42    248  B+L+R+BL+BR              . . .  / # X # / # # #   T open-T + BL+BR
43    250  T+B+L+R+BL+BR            . # .  / # X # / # # #   +BL+BR (B edge filled)
44    251  T+B+L+R+TL+BL+BR         # # .  / # X # / # # #   +TL+BL+BR
45    254  T+B+L+R+TR+BL+BR         . # #  / # X # / # # #   +TR+BL+BR
46    255  T+B+L+R+TL+TR+BL+BR      # # #  / # X # / # # #   fully interior (all 8)
```

**Key columns to memorise:**

| Col | Situation |
|---|---|
| 0 | Isolated tile (single-tile island) |
| 1, 2, 5, 13 | Edge tiles (only one cardinal neighbour) |
| 3–4, 6–7, 15, 18, 26, 34 | Two-cardinal corners (with/without inner diagonal) |
| 8, 14 | Corridors (horizontal, vertical) |
| 22 | All-cardinal cross — common at tight terrain junctions |
| 46 | Fully surrounded — the tile used for the entire interior of a terrain area |
| 4, 7, 26, 34 | Concave inner corners — appear where a terrain region has a notch cut into it |
| 47 | Padding column — copy of col 46, never addressed by the LUT |

In a large terrain region, roughly 90% of tiles will be col 46. The edge tiles
appear at peninsulas and isolated patches. Diagonal variant columns appear at
boundaries where the terrain forms a smooth filled edge. Inner corner tiles
appear at concave corners (a terrain that wraps around another).

### Computing `TileTextureIndex`

```
texture_index = terrain_row * 48 + blob47_lut[reduce_to_47(bitmask)]
```

In Rust (using the generated helpers):

```rust
let tex_idx = tile_texture_index(
    &terrain_map,   // Vec<Vec<u32>>, terrain_map[y][x] = terrain row index
    x as i32,
    y as i32,
    map_w as i32,
    map_h as i32,
    &lut_data,      // &Blob47Lut
);
commands.spawn(TileBundle {
    texture_index: TileTextureIndex(tex_idx),
    ..default()
});
```

---

## RON file formats

### `blob47_lut.ron`

```ron
(
    atlas_cols: 48,
    lut: [ /* 256 u8 values; 255 = unmapped */ ],
)
```

Rust type: `Blob47Lut { atlas_cols: u32, lut: Vec<u8> }`

`lut[reduced_mask]` gives the atlas column. The 256-entry array is indexed
directly by the reduced mask byte — no hash lookup needed.

### `{biome}_terrain.ron`

```ron
(
    width:  64,
    height: 64,
    data: [ /* width*height u32 values, row-major, top-down */ ],
)
```

Rust type: `TerrainMapAsset { width: u32, height: u32, data: Vec<u32> }`

`data[y * width + x]` = terrain row index (0–4) for tile at column `x`, row
`y` in **image convention** (y=0 = top of map). Must be Y-flipped before use
with `bevy_ecs_tilemap` (see Y-axis section).

### `{biome}_collision.ron`

```ron
(
    width:  64,
    height: 64,
    data: [ /* width*height u8 values */ ],
)
```

Rust type: `CollisionMapAsset { width: u32, height: u32, data: Vec<u8> }`

Collision codes:

| Value | Name | Meaning |
|---|---|---|
| 0 | Walkable | No collider spawned |
| 1 | Solid | `RigidBody::Static` + `Collider::rectangle` |
| 2 | Slow | `Sensor` + `TerrainSensor`, reduces movement speed |
| 3 | Damaging | `Sensor` + `TerrainSensor`, deals `damage_per_sec` HP/s |
| 4 | Trigger | `Sensor` + `TerrainSensor`, fires interaction events |

Same row-major top-down layout as `terrain.ron`. Must be Y-flipped before
passing to `spawn_collision_entities`.

### `world_manifest.ron`

Contains biome metadata keyed by biome name. Each biome entry lists all
terrain types with their `row`, `threshold`, `collision`, `movement_cost`,
and `damage_per_sec` values. Load this once at startup and use it to look up
per-terrain game properties at runtime without hard-coding them.

---

## Y-axis: critical coordinate flip

**The RON data files use image convention: row 0 = top, y increases downward.**

**`bevy_ecs_tilemap` uses game convention: tile (0,0) = bottom-left, y increases upward.**

This means every `terrain.ron` and `collision.ron` must be Y-flipped before
use. Failure to flip results in the visual tilemap and collision bodies being
mirrored vertically — the map looks correct but walls block in the wrong
places.

Correct flip when building `terrain_map` from `TerrainMapAsset`:

```rust
let terrain_map: Vec<Vec<u32>> = (0..map_h)
    .map(|y| {
        let src_y = map_h - 1 - y;  // flip: tilemap row 0 = last data row
        (0..map_w)
            .map(|x| {
                let idx = (src_y * map_w + x) as usize;
                terrain.data.get(idx).copied().unwrap_or(0)
            })
            .collect()
    })
    .collect();
```

Correct flip for collision (build a flipped copy before calling
`spawn_collision_entities`):

```rust
let flipped_col = CollisionMapAsset {
    width:  col.width,
    height: col.height,
    data: (0..col.height)
        .flat_map(|y| {
            let src_y = col.height - 1 - y;
            let start = (src_y * col.width) as usize;
            col.data[start..start + col.width as usize].iter().copied()
        })
        .collect(),
};
```

---

## Rust types and helpers (`world_assets.rs`)

The generated `world_assets.rs` provides everything needed. Copy it into your
project. Key types and functions:

### Asset structs

```rust
// Load via RonAssetPlugin or ron::from_str
struct Blob47Lut        { atlas_cols, lut: Vec<u8> }
struct TerrainMapAsset  { width, height, data: Vec<u32> }
struct CollisionMapAsset{ width, height, data: Vec<u8>  }
struct WorldManifest    { tile_size, atlas_cols, biomes: HashMap<String, BiomeMeta> }
```

### Bitmask helpers (Boris The Brave's standard)

```rust
// Bit constants: TL=1 T=2 TR=4 L=8 R=16 BL=32 B=64 BR=128

// Compute 8-bit neighbour mask for tile at (x,y) in terrain_map
fn compute_bitmask(terrain_map, x, y, width, height) -> u8

// Clear irrelevant diagonal bits, reducing 256 cases to 47
fn reduce_to_47(mask: u8) -> u8

// Combine the above with lut lookup to get TileTextureIndex value
fn tile_texture_index(terrain_map, x, y, width, height, lut) -> u32
```

### Collision spawner

```rust
fn spawn_collision_entities(
    commands:    &mut Commands,
    col_asset:   &CollisionMapAsset,  // pass the Y-flipped copy
    tile_size:   f32,
    map_origin:  Vec2,                // world position of tile (0,0)
    layers:      CollisionLayers,     // your game's collision layer config
)
```

`map_origin` for a centred tilemap with `TilemapAnchor::Center` at world
origin is:

```rust
let map_origin = Vec2::new(
    -(map_w as f32 * tile_px / 2.0),
    -(map_h as f32 * tile_px / 2.0),
);
```

### Plugin registration

```rust
app.add_plugins(WorldAssetsPlugin);
// Registers RonAssetPlugin for: Blob47Lut, WorldManifest,
// CollisionMapAsset, BuildingsManifest, BuildingTemplate
```

### `planet_type_to_biome`

Maps planet type strings from `ItemUniverse` to biome names:

```rust
pub fn planet_type_to_biome(planet_type: &str) -> &'static str {
    match planet_type {
        "garden" | "terran" | "jungle"   => "garden",
        "ice"    | "frozen" | "arctic"   => "ice",
        "rocky"  | "barren" | "volcanic" => "rocky",
        "desert" | "arid"   | "dune"     => "desert",
        "station"| "city"   | "interior" => "interior",
        _                                => "rocky",
    }
}
```

Add this function to `world_assets.rs` — it is not currently emitted by the
generator but is called by `setup_surface`.

---

## Z-layer conventions

The tilemap and entities spawned by `setup_surface` should use these z-values:

| Layer | z | Notes |
|---|---|---|
| Tilemap | -10.0 | Well behind all gameplay entities |
| Building sensors | 0.0 | Ground-level, no visual |
| Walker | 1.0 | Above buildings |
| Building labels | 5.0 | Above walker |
| Interact prompt | 6.0 | Topmost gameplay UI |

---

## Building pipeline (`buildings.py` / `buildings_scifi.py`)

### Output files per style

```
{style}_building_exterior.png   — 6×5 = 30 tile exterior sheet
{style}_building_interior.png   — 4×4 = 16 tile interior sheet (blob-4)
buildings/{style}_{template}.ron — building template (tile grid + metadata)
buildings_manifest.ron           — style metadata + blob-4 LUT
```

### Exterior atlas layout (30 tiles, 6 cols × 5 rows)

```
Row 0: roof_nw  roof_n  roof_ne  roof_w   roof_fill  roof_e
Row 1: roof_sw  roof_s  roof_se  attic_w  attic      attic_e
Row 2: wall_nw  wall_n  wall_ne  wall_w   wall_fill  wall_e
Row 3: wall_sw  wall_s  wall_se  win_w    window     win_e
Row 4: base_nw  base_n  base_ne  door_l   door       door_r
```

Index: `row * 6 + col`. The `EXT` constants in `buildings.py` name each slot.

### Interior atlas layout (16 tiles, blob-4)

Indexed by 4-bit cardinal neighbour mask: `N=1, E=2, S=4, W=8`.
`mask == 15` (all 4 neighbours same) = pure floor tile.

```rust
// Interior wall texture index — identity mapping
fn int_tile_texture_index(mask: u8) -> u32 { mask as u32 }
```

### Building template RON

```ron
(
    name:            "small_house",
    style:           "colony",
    width:           4,
    height:          4,
    layer:           1,
    tiles:           [
        [0, 4, 4, 2],
        [6, 7, 7, 8],
        [15, 22, 16, 17],
        [24, 29, 28, 26],
    ],
    entry_points:    [(2, 3)],
    interior_offset: (1, 2),
    interior_size:   (2, 1),
)
```

- `tiles[row][col]` = exterior atlas index (0 = transparent, skip this tile)
- `entry_points` = `(col, row)` tile coords of door positions
- `interior_offset` + `interior_size` = walkable rectangle in tile coords

Use `stamp_building()` from `world_assets.rs` to place a template onto a
`TileStorage` at a given world tile anchor.

### Sci-fi styles mapped to biomes

| Style | Biome | Wall texture | Accent colour |
|---|---|---|---|
| `colony` | garden | composite panels | amber LED |
| `cryo` | ice | frosted cryo-panel | cyan |
| `extraction` | rocky | corrugated rust | red + hazard stripe |
| `station` | interior/city | conduit wall | cyan |
| `outpost` | desert | composite panels | amber heat-vent |

---

## Regenerating assets

To regenerate after changing biome parameters or tile size:

```bash
cd tilegen/
python tilegen.py --tile-size 32 --map-preview 64 --out-dir ../assets/worlds
python buildings_scifi.py --tile-size 32 --out-dir ../assets/worlds
```

The map preview size (`--map-preview 64`) controls both the debug PNG and the
size of the collision/terrain RON files. Use a power of 2 or a round number
that matches your target world size. If `--map-preview 0` is passed, preview
PNGs are skipped but RON files are still written at size 64.

Adding a new biome requires:
1. Define a new `BiomeConfig` in `tilegen.py` with 5 `TerrainSpec` entries
2. Add it to `ALL_BIOMES`
3. Add a new `BuildingStyle` in `buildings_scifi.py` with matching `biome` field
4. Add the planet type mapping in `planet_type_to_biome()` in `world_assets.rs`
5. Re-run both scripts

---

## Common errors

**Map renders but collisions are in wrong positions (mirrored vertically)**
→ Y-flip is missing. Apply the flip to both `terrain_map` and the collision
data as shown in the Y-axis section above.

**Fallback grey square appears instead of tilemap**
→ One of the three RON files failed to load. The error message from
`setup_surface` reports which (`col=`, `terrain=`, `lut=`). Check that all
files were copied to `assets/worlds/` and that the biome name resolved
correctly via `planet_type_to_biome`.

**`TileTextureIndex` out of range panic**
→ `terrain_row * 48 + col` exceeded the atlas width. This happens if
`terrain_row` is larger than 4 (only 5 rows exist) or if `col` is 255
(unmapped mask — should never occur after `reduce_to_47`). Add a debug assert:
`assert!(col != 255, "unmapped blob-47 mask: {}", reduced);`

**Buildings always in top-left cluster**
→ `find_walkable_positions` returns row-major order. Shuffle with a
planet-name-seeded RNG before iterating (see `setup_surface`).

**`planet_type_to_biome` not found**
→ This function is not emitted by `tilegen.py`. It must be added manually to
`world_assets.rs` (see the function body in the Rust types section above).
