//! Walkable planet surface plugin.
//!
//! When the player lands on a planet, this module spawns a procedurally
//! generated tilemap with buildings (Bar, Shipyard, Outfitter, Mission
//! Control, Market) that the player can walk to and interact with.

use avian2d::prelude::*;
use bevy::prelude::*;
use bevy_ecs_tilemap::prelude::*;
use bevy_egui::EguiContexts;

use crate::item_universe::ItemUniverse;
use crate::missions::{
    AbandonMission, AcceptMission, DeclineMission, MissionCatalog, MissionLog, MissionOffers,
    PlayerUnlocks, render_bar_tab, render_missions_tab,
};
use crate::planet_ui::{
    LandedContext, render_outfitter_tab, render_shipyard_tab, render_trade_tab,
};
use crate::ship::{BuyShip, Ship};
use crate::{CurrentStarSystem, GameLayer, PlayState, Player};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WALK_SPEED: f32 = 120.0;
const WALKER_RADIUS: f32 = 6.0;
const WALKER_DAMPING: f32 = 10.0;

/// Camera scale when zoomed in on the planet surface.
const SURFACE_CAMERA_SCALE: f32 = 0.5;
/// Camera scale in space (normal).
const SPACE_CAMERA_SCALE: f32 = 1.0;
/// How fast the camera zoom interpolates (per second).
const CAMERA_ZOOM_SPEED: f32 = 4.0;

/// Base path for world tile assets.
const WORLDS_DIR: &str = "sprites/worlds";

/// World dimensions in tiles. Small for testing; increase for production.
const WORLD_WIDTH: u32 = 64;
const WORLD_HEIGHT: u32 = 64;

/// Tile size in pixels (must match the atlas tile size from tilegen.py).
const TILE_PX: f32 = 32.0;

/// Extra tiles of impassable terrain around the map border so the edge
/// doesn't transition directly to empty space.
const BORDER_MARGIN: u32 = 4;

// ── fBm parameters ──────────────────────────────────────────────────────
const FBM_SCALE: f32 = 4.0;
const FBM_OCTAVES: u32 = 5;
const FBM_LACUNARITY: f32 = 2.0;
const FBM_GAIN: f32 = 0.5;
/// Number of terrain types per biome (must match the atlas row count).
const N_TERRAIN_TYPES: u32 = 5;

// ---------------------------------------------------------------------------
// Building kinds
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildingKind {
    ShipPad,
    Market,
    Outfitter,
    Shipyard,
    Bar,
    MissionControl,
}

impl BuildingKind {
    fn label(&self) -> &'static str {
        match self {
            BuildingKind::ShipPad => "Ship",
            BuildingKind::Market => "Market",
            BuildingKind::Outfitter => "Outfitter",
            BuildingKind::Shipyard => "Shipyard",
            BuildingKind::Bar => "Bar",
            BuildingKind::MissionControl => "Missions",
        }
    }

    fn color(&self) -> Color {
        match self {
            BuildingKind::ShipPad => Color::srgb(0.5, 0.5, 0.7),
            BuildingKind::Market => Color::srgb(0.9, 0.75, 0.2),
            BuildingKind::Outfitter => Color::srgb(0.7, 0.3, 0.3),
            BuildingKind::Shipyard => Color::srgb(0.3, 0.5, 0.8),
            BuildingKind::Bar => Color::srgb(0.8, 0.4, 0.1),
            BuildingKind::MissionControl => Color::srgb(0.3, 0.7, 0.4),
        }
    }
}

// ---------------------------------------------------------------------------
// Components & Resources
// ---------------------------------------------------------------------------

/// Marker for the walking character on the planet surface.
#[derive(Component)]
pub struct Walker;

// ---------------------------------------------------------------------------
// Walker animation
// ---------------------------------------------------------------------------
//
// Sprite sheet layout (RPG Maker VX standard):
//   3 columns (still, w1, w2) x 4 rows (down, left, right, up)
//   Atlas index = row * 3 + column

/// Number of columns (frames) per direction in the sprite sheet.
const SPRITE_COLS: usize = 3;

/// Direction the walker is facing. Row index in the sprite sheet.
#[derive(Clone, Copy, Default, Debug)]
enum Facing {
    #[default]
    Down = 0,
    Left = 1,
    Right = 2,
    Up = 3,
}

/// Which animation frame to display. Column index in the sprite sheet.
#[derive(Clone, Copy, Default, Debug)]
enum WalkFrame {
    #[default]
    Still = 0,
    W1 = 1,
    W2 = 2,
}

/// Compute the texture atlas index from facing direction and walk frame.
fn sprite_index(facing: Facing, frame: WalkFrame) -> usize {
    facing as usize * SPRITE_COLS + frame as usize
}

/// Loaded sprite sheet for the walker character.
#[derive(Resource)]
struct WalkerSheet {
    image: Handle<Image>,
    layout: Handle<TextureAtlasLayout>,
}

impl WalkerSheet {
    fn load(
        asset_server: &AssetServer,
        atlas_layouts: &mut Assets<TextureAtlasLayout>,
        gender: &str,
    ) -> Self {
        let image = asset_server.load(format!("sprites/people/{gender}.png"));
        let layout = atlas_layouts.add(TextureAtlasLayout::from_grid(
            UVec2::new(16, 16),
            SPRITE_COLS as u32, // columns
            4,                  // rows
            None,               // padding
            None,               // offset
        ));
        Self { image, layout }
    }
}

/// Per-walker animation state.
#[derive(Component)]
struct WalkerAnim {
    facing: Facing,
    /// Walk cycle phase: still → w1 → still → w2 → still → w1 → ...
    walk_phase: u8, // 0=still, 1=w1, 2=still, 3=w2
    walk_timer: Timer,
    is_moving: bool,
}

impl Default for WalkerAnim {
    fn default() -> Self {
        Self {
            facing: Facing::Down,
            walk_phase: 0,
            walk_timer: Timer::from_seconds(0.15, TimerMode::Repeating),
            is_moving: false,
        }
    }
}

impl WalkerAnim {
    fn current_frame(&self) -> WalkFrame {
        if !self.is_moving {
            WalkFrame::Still
        } else {
            match self.walk_phase {
                1 => WalkFrame::W1,
                3 => WalkFrame::W2,
                _ => WalkFrame::Still, // phases 0 and 2
            }
        }
    }

    fn atlas_index(&self) -> usize {
        sprite_index(self.facing, self.current_frame())
    }
}

/// Interaction zone for a building.
#[derive(Component)]
pub struct Building {
    pub kind: BuildingKind,
}

/// Which building (if any) the walker is currently overlapping.
#[derive(Resource, Default)]
pub struct NearbyBuilding(pub Option<(Entity, BuildingKind)>);

/// Which building's egui UI is currently open. `None` = walking freely.
#[derive(Resource, Default)]
pub struct ActiveBuildingUI(pub Option<BuildingKind>);

/// Current movement cost multiplier from the terrain the walker is on.
/// 1.0 = normal speed, 2.0 = half speed, etc.
#[derive(Resource)]
struct TerrainSpeedModifier(f32);

impl Default for TerrainSpeedModifier {
    fn default() -> Self {
        Self(1.0)
    }
}

/// Animated camera zoom target.
#[derive(Resource)]
struct CameraZoom {
    target: f32,
}

impl Default for CameraZoom {
    fn default() -> Self {
        Self {
            target: SPACE_CAMERA_SCALE,
        }
    }
}

/// Marker for building label text entities.
#[derive(Component)]
struct BuildingLabel;

/// Spawn a building label with a dark plaque behind for readability.
fn spawn_building_label(
    commands: &mut Commands,
    text: &str,
    pos: Vec3,
) {
    let scale = Vec3::splat(0.18);
    let char_w = 28.0 * 0.6; // approximate glyph width at font_size 28
    let plaque_w = text.len() as f32 * char_w * scale.x + 4.0;
    let plaque_h = 28.0 * scale.y + 2.5;

    // Dark plaque background.
    commands.spawn((
        DespawnOnExit(PlayState::Exploring),
        BuildingLabel,
        Sprite {
            color: Color::srgba(0.05, 0.05, 0.08, 0.85),
            custom_size: Some(Vec2::new(plaque_w, plaque_h)),
            ..default()
        },
        Transform::from_translation(pos + Vec3::new(0.0, 0.0, -0.02)),
    ));
    // White text.
    commands.spawn((
        DespawnOnExit(PlayState::Exploring),
        BuildingLabel,
        Text2d::new(text.to_string()),
        TextFont {
            font_size: 28.0,
            ..default()
        },
        TextColor(Color::srgb(1.0, 1.0, 1.0)),
        Transform::from_translation(pos).with_scale(scale),
    ));
}

/// Marker for the "Press E" prompt.
#[derive(Component)]
struct InteractPrompt;

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub fn surface_plugin(app: &mut App) {
    app.add_plugins(TilemapPlugin)
        .init_resource::<CameraZoom>()
        .init_resource::<NearbyBuilding>()
        .init_resource::<ActiveBuildingUI>()
        .init_resource::<TerrainSpeedModifier>()
        .add_systems(
            OnEnter(PlayState::Exploring),
            (setup_surface, save_on_explore),
        )
        .add_systems(OnExit(PlayState::Exploring), teardown_surface)
        .add_systems(
            Update,
            (
                walker_input,
                animate_walker,
                track_nearby_building,
                track_terrain_speed,
                building_interact,
                update_interact_prompt,
            )
                .chain()
                .run_if(in_state(PlayState::Exploring)),
        )
        .add_systems(Update, animate_camera_zoom)
        .add_systems(
            bevy_egui::EguiPrimaryContextPass,
            surface_building_ui.run_if(in_state(PlayState::Exploring)),
        )
        .add_systems(
            FixedUpdate,
            camera_follow_walker.run_if(in_state(PlayState::Exploring)),
        );
}

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------

/// Save the game when entering the Exploring state.
fn save_on_explore(game_state: Res<crate::game_save::PlayerGameState>) {
    game_state.save();
}

// ---------------------------------------------------------------------------
// Setup helpers
// ---------------------------------------------------------------------------

/// Convert tile coordinates to world-space center position (tilemap centered
/// at origin via `TilemapAnchor::Center`).
fn tile_to_world(tx: u32, ty: u32, map_w: u32, map_h: u32, tile_px: f32) -> Vec2 {
    Vec2::new(
        (tx as f32 - map_w as f32 / 2.0) * tile_px + tile_px / 2.0,
        (ty as f32 - map_h as f32 / 2.0) * tile_px + tile_px / 2.0,
    )
}

/// Determine which buildings to place based on planet data.
/// ShipPad is handled separately as a landing pad, not a building template.
fn building_kinds_for_planet(
    planet_name: &str,
    item_universe: &ItemUniverse,
    system_name: &str,
) -> Vec<BuildingKind> {
    let mut kinds = Vec::new();
    let planet_data = item_universe
        .star_systems
        .get(system_name)
        .and_then(|sys| sys.planets.get(planet_name));
    if let Some(pd) = planet_data {
        if !pd.commodities.is_empty() {
            kinds.push(BuildingKind::Market);
        }
        if !pd.outfitter.is_empty() {
            kinds.push(BuildingKind::Outfitter);
        }
        if !pd.shipyard.is_empty() {
            kinds.push(BuildingKind::Shipyard);
        }
    }
    kinds.push(BuildingKind::Bar);
    kinds.push(BuildingKind::MissionControl);
    kinds
}

/// Landing pad: all 9 tiles of the 3×3 atlas.
/// (dx, dy, atlas_index). Atlas is row-major top-down in the PNG,
/// but tile placement is bottom-up (dy=+1 = up on screen).
const PAD_TILES: [(i32, i32, usize); 9] = [
    (-1,  1, 0),  // TL corner
    ( 0,  1, 1),  // T  edge
    ( 1,  1, 2),  // TR corner
    (-1,  0, 3),  // L  edge
    ( 0,  0, 4),  // C  center
    ( 1,  0, 5),  // R  edge
    (-1, -1, 6),  // BL corner
    ( 0, -1, 7),  // B  edge
    ( 1, -1, 8),  // BR corner
];

/// Find walkable tile positions in the collision data, suitable for placing
/// buildings. Returns positions in tile coordinates, sorted by distance from
/// center, filtered to avoid the center area (where the walker spawns).
fn find_walkable_positions(
    col_data: &[u8],
    map_w: u32,
    map_h: u32,
    min_dist_from_center: f32,
) -> Vec<(u32, u32)> {
    let cx = map_w as f32 / 2.0;
    let cy = map_h as f32 / 2.0;
    let mut positions: Vec<(u32, u32, f32)> = Vec::new();

    for ty in 2..map_h.saturating_sub(2) {
        for tx in 2..map_w.saturating_sub(2) {
            let idx = (ty * map_w + tx) as usize;
            if idx < col_data.len()
                && crate::world_assets::CollisionType::from(col_data[idx])
                    == crate::world_assets::CollisionType::Walkable
            {
                let dist = ((tx as f32 - cx).powi(2) + (ty as f32 - cy).powi(2)).sqrt();
                if dist > min_dist_from_center {
                    positions.push((tx, ty, dist));
                }
            }
        }
    }
    // Sort by distance from center — buildings closer to center are easier to reach.
    positions.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    positions.into_iter().map(|(x, y, _)| (x, y)).collect()
}

// ---------------------------------------------------------------------------
// Setup / Teardown
// ---------------------------------------------------------------------------

/// Spawn the tilemap, walker, buildings, and set camera zoom on entering Exploring.
#[allow(clippy::too_many_arguments)]
fn setup_surface(
    mut commands: Commands,
    player_query: Query<Entity, With<Player>>,
    landed_context: Res<LandedContext>,
    item_universe: Res<ItemUniverse>,
    current_system: Res<crate::CurrentStarSystem>,
    mut camera_query: Query<&mut Transform, With<Camera2d>>,
    mut zoom: ResMut<CameraZoom>,
    mut atlas_layouts: ResMut<Assets<TextureAtlasLayout>>,
    asset_server: Res<AssetServer>,
    game_state: Res<crate::game_save::PlayerGameState>,
) {
    // Bug 4 fixed: get_single() is the fallible form in Bevy 0.15+
    if let Ok(ship_entity) = player_query.single() {
        commands.entity(ship_entity).insert(Visibility::Hidden);
    }

    let planet_name = landed_context.planet_name.clone().unwrap_or_default();
    let system_name = &current_system.0;

    let planet_type = item_universe
        .star_systems
        .get(system_name)
        .and_then(|sys| sys.planets.get(&planet_name))
        .map(|pd| pd.planet_type.as_str())
        .unwrap_or("rocky");
    let biome_name = crate::world_assets::planet_type_to_biome(planet_type);

    let load_ron = |filename: &str| -> Option<String> {
        std::fs::read_to_string(format!("assets/{WORLDS_DIR}/{filename}")).ok()
    };

    let lut = load_ron("blob47_lut.ron")
        .and_then(|text| ron::from_str::<crate::world_assets::Blob47Lut>(&text).ok());

    let manifest = load_ron("world_manifest.ron")
        .and_then(|text| ron::from_str::<crate::world_assets::WorldManifest>(&text).ok());

    let atlas_handle: Handle<Image> =
        asset_server.load(format!("{WORLDS_DIR}/{biome_name}_atlas.png"));

    let map_w = WORLD_WIDTH;
    let map_h = WORLD_HEIGHT;
    let tile_px = TILE_PX;

    let (col_data, placed_buildings) = if let (Some(lut_data), Some(manifest)) = (&lut, &manifest) {
        // Seed fBm from planet name for deterministic, per-planet terrain.
        let seed = planet_name
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));

        let mut terrain_flat = crate::fbm::generate_terrain_map(
            map_w,
            map_h,
            N_TERRAIN_TYPES,
            FBM_SCALE,
            FBM_OCTAVES,
            FBM_LACUNARITY,
            FBM_GAIN,
            seed,
        );

        // Derive collision and movement cost maps from terrain + manifest.
        let (collision_codes, movement_costs): (Vec<u8>, Vec<f32>) =
            if let Some(biome_meta) = manifest.biomes.get(biome_name) {
                (
                    biome_meta.terrains.iter().map(|t| t.collision).collect(),
                    biome_meta.terrains.iter().map(|t| t.movement_cost).collect(),
                )
            } else {
                (
                    vec![0; N_TERRAIN_TYPES as usize],
                    vec![1.0; N_TERRAIN_TYPES as usize],
                )
            };

        // Find the lowest walkable terrain index for this biome.
        let walkable_terrain: u32 = collision_codes
            .iter()
            .position(|&c| c == 0) // CollisionType::Walkable
            .unwrap_or(0) as u32;

        // ── Pre-tilemap building placement ─────────────────────────────
        // Find building positions on the initial terrain, then force
        // nearby tiles to walkable terrain and re-clamp before building
        // the tilemap.
        let building_kinds = building_kinds_for_planet(&planet_name, &item_universe, system_name);

        let initial_col: Vec<u8> = terrain_flat
            .iter()
            .map(|&t| *collision_codes.get(t as usize).unwrap_or(&0))
            .collect();
        let mut walkable_positions = find_walkable_positions(&initial_col, map_w, map_h, 5.0);
        {
            use rand::{SeedableRng, seq::SliceRandom};
            let seed = planet_name
                .bytes()
                .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            walkable_positions.shuffle(&mut rng);
        }

        // Load building templates so we know footprint sizes.
        let style_name = crate::world_assets::biome_to_building_style(biome_name);
        let bldg_manifest = load_ron("buildings_manifest.ron")
            .and_then(|t| ron::from_str::<crate::world_assets::BuildingsManifest>(&t).ok());
        let bldg_style = bldg_manifest
            .as_ref()
            .and_then(|m| m.styles.get(style_name));
        let ext_atlas_handle: Option<Handle<Image>> = bldg_style.map(|s| {
            asset_server.load(format!("{WORLDS_DIR}/{}", s.exterior_atlas))
        });
        let ext_layout: Option<Handle<TextureAtlasLayout>> = bldg_manifest.as_ref().map(|m| {
            atlas_layouts.add(TextureAtlasLayout::from_grid(
                UVec2::new(tile_px as u32, tile_px as u32),
                m.ext_cols,
                m.ext_rows,
                None,
                None,
            ))
        });
        let templates: Vec<crate::world_assets::BuildingTemplate> = bldg_style
            .map(|s| {
                s.templates
                    .iter()
                    .filter_map(|name| {
                        let path = format!(
                            "assets/{WORLDS_DIR}/buildings/{style_name}_{name}.ron"
                        );
                        std::fs::read_to_string(&path)
                            .ok()
                            .and_then(|t| ron::from_str(&t).ok())
                    })
                    .collect()
            })
            .unwrap_or_default();
        let ext_collision: Vec<u8> = bldg_manifest
            .as_ref()
            .map(|m| m.ext_collision.clone())
            .unwrap_or_default();
        const EXT_DOOR: u32 = 28;

        // Place the landing pad at the map center.
        let pad_cx = map_w / 2;
        let pad_cy = map_h / 2;

        let min_building_spacing = 10_u32;
        let mut placed_buildings: Vec<(u32, u32, u32, u32)> = Vec::new(); // (x, y, w, h)
        // Reserve the pad area (3x3 centered on pad_cx, pad_cy).
        placed_buildings.push((pad_cx - 1, pad_cy - 1, 3, 3));
        let mut template_idx = 0;
        let mut building_assignments: Vec<(BuildingKind, u32, u32, usize)> = Vec::new(); // (kind, x, y, template_idx)

        for kind in &building_kinds {
            let tmpl = if templates.is_empty() {
                None
            } else {
                let t = &templates[template_idx % templates.len()];
                template_idx += 1;
                Some(t)
            };
            let (bw, bh) = tmpl.map(|t| (t.width, t.height)).unwrap_or((2, 2));
            let spacing = min_building_spacing.max(bw.max(bh) + 2);

            let pos = walkable_positions.iter().find(|&&(x, y)| {
                if x + bw >= map_w || y + bh >= map_h || x < 1 || y < 1 {
                    return false;
                }
                // Check entire footprint is on walkable terrain.
                for dy in 0..bh {
                    for dx in 0..bw {
                        let idx = ((y + dy) * map_w + (x + dx)) as usize;
                        if idx >= initial_col.len() {
                            return false;
                        }
                        if initial_col[idx] == 1 { // Solid
                            return false;
                        }
                    }
                }
                placed_buildings.iter().all(|&(px, py, _, _)| {
                    let dx = (x as i32 - px as i32).unsigned_abs();
                    let dy = (y as i32 - py as i32).unsigned_abs();
                    dx + dy >= spacing
                })
            });

            if let Some(&(tx, ty)) = pos {
                let ti = if templates.is_empty() { 0 } else { (template_idx - 1) % templates.len() };
                placed_buildings.push((tx, ty, bw, bh));
                building_assignments.push((*kind, tx, ty, ti));
            }
        }

        // Force tiles within 2 of each building floor plan (and the
        // landing pad, which is the first entry) to walkable terrain,
        // then re-clamp.
        {
            let w = map_w as usize;
            let h = map_h as usize;
            let margin = 2_i32;
            let mut pinned = vec![false; w * h];

            for &(bx, by, bw, bh) in &placed_buildings {
                for dy in -margin..(bh as i32 + margin) {
                    for dx in -margin..(bw as i32 + margin) {
                        let x = bx as i32 + dx;
                        let y = by as i32 + dy;
                        if x >= 0 && y >= 0 && x < w as i32 && y < h as i32 {
                            let idx = y as usize * w + x as usize;
                            terrain_flat[idx] = walkable_terrain;
                            pinned[idx] = true;
                        }
                    }
                }
            }

            // Also pin the border tiles so they don't get pulled down.
            for x in 0..w {
                pinned[x] = true;
                pinned[(h - 1) * w + x] = true;
            }
            for y in 0..h {
                pinned[y * w] = true;
                pinned[y * w + (w - 1)] = true;
            }

            crate::fbm::clamp_terrain_indices(&mut terrain_flat, w, h, &pinned);
        }

        // Re-derive collision from the modified terrain.
        let col_data: Vec<u8> = terrain_flat
            .iter()
            .map(|&t| *collision_codes.get(t as usize).unwrap_or(&0))
            .collect();

        // Build the 2D terrain map for bitmask computation.
        // Bottom-up convention: y=0 = bottom, matching bevy_ecs_tilemap.
        let terrain_map: Vec<Vec<u32>> = (0..map_h)
            .map(|y| {
                (0..map_w)
                    .map(|x| {
                        let idx = (y * map_w + x) as usize;
                        terrain_flat.get(idx).copied().unwrap_or(0)
                    })
                    .collect()
            })
            .collect();

        // Expand the tilemap with a border margin of solid tiles so the
        // edge doesn't transition directly to empty space.
        let border = BORDER_MARGIN;
        let total_w = map_w + 2 * border;
        let total_h = map_h + 2 * border;
        let max_terrain = N_TERRAIN_TYPES - 1;
        // Fully-interior tile for the highest terrain (all neighbours same).
        let border_tex = max_terrain * lut_data.atlas_cols
            + lut_data.lut[255] as u32;

        let map_size = TilemapSize { x: total_w, y: total_h };
        let mut tile_storage = TileStorage::empty(map_size);
        let tilemap_entity = commands.spawn(DespawnOnExit(PlayState::Exploring)).id();
        let tilemap_id = TilemapId(tilemap_entity);

        for ty in 0..total_h {
            for tx in 0..total_w {
                let in_map = tx >= border && tx < border + map_w
                          && ty >= border && ty < border + map_h;
                let tex_idx = if in_map {
                    let x = tx - border;
                    let y = ty - border;
                    crate::world_assets::tile_texture_index(
                        &terrain_map,
                        x as i32,
                        y as i32,
                        map_w as i32,
                        map_h as i32,
                        lut_data,
                    )
                } else {
                    border_tex
                };
                // No y-flip: data is already bottom-up, matching TilePos.
                let tile_pos = TilePos { x: tx, y: ty };
                let tile_entity = commands
                    .spawn(TileBundle {
                        position: tile_pos,
                        tilemap_id,
                        texture_index: TileTextureIndex(tex_idx),
                        ..default()
                    })
                    .id();
                tile_storage.set(&tile_pos, tile_entity);
            }
        }

        let tile_size = TilemapTileSize {
            x: tile_px,
            y: tile_px,
        };
        let grid_size = tile_size.into();

        commands.entity(tilemap_entity).insert(TilemapBundle {
            grid_size,
            map_type: TilemapType::Square,
            size: map_size,
            storage: tile_storage,
            texture: TilemapTexture::Single(atlas_handle),
            tile_size,
            anchor: TilemapAnchor::Center,
            transform: Transform::from_xyz(0.0, 0.0, -10.0),
            ..default()
        });

        // Build expanded collision + terrain arrays that include the
        // border margin (all solid / highest terrain).
        let solid_code = 1u8; // CollisionType::Solid
        let total_len = (total_w * total_h) as usize;
        let mut expanded_col = vec![solid_code; total_len];
        let mut expanded_terrain = vec![max_terrain; total_len];
        // No y-flip: data is already bottom-up, matching bevy_ecs_tilemap.
        for ty in border..border + map_h {
            for tx in border..border + map_w {
                let src_idx = ((ty - border) * map_w + (tx - border)) as usize;
                let dst_idx = (ty * total_w + tx) as usize;
                expanded_col[dst_idx] = col_data[src_idx];
                expanded_terrain[dst_idx] = terrain_flat[src_idx];
            }
        }

        let expanded_col_asset = crate::world_assets::CollisionMapAsset {
            width: total_w,
            height: total_h,
            data: expanded_col,
        };

        let map_origin = Vec2::new(
            -(total_w as f32 * tile_px / 2.0),
            -(total_h as f32 * tile_px / 2.0),
        );
        let surface_layers = CollisionLayers::new(GameLayer::Surface, [GameLayer::Surface]);
        crate::world_assets::spawn_collision_entities(
            &mut commands,
            &expanded_col_asset,
            &expanded_terrain,
            &movement_costs,
            tile_px,
            map_origin,
            surface_layers,
        );

        // ── Spawn landing pad (plus/cross shape, 3x3 atlas) ──────────
        let pad_atlas_handle: Option<Handle<Image>> = bldg_style.map(|s| {
            asset_server.load(format!("{WORLDS_DIR}/{}", s.landing_pad_atlas))
        });
        let pad_layout: Option<Handle<TextureAtlasLayout>> = pad_atlas_handle.as_ref().map(|_| {
            atlas_layouts.add(TextureAtlasLayout::from_grid(
                UVec2::new(tile_px as u32, tile_px as u32),
                3, 3,
                None, None,
            ))
        });

        for &(dx, dy, atlas_idx) in &PAD_TILES {
            let tx = (pad_cx as i32 + dx) as u32;
            let ty = (pad_cy as i32 + dy) as u32;
            let world_pos = tile_to_world(tx, ty, map_w, map_h, tile_px);

            let mut entity = if let (Some(pad_img), Some(pad_lay)) =
                (pad_atlas_handle.as_ref(), pad_layout.as_ref())
            {
                commands.spawn((
                    DespawnOnExit(PlayState::Exploring),
                    Sprite::from_atlas_image(
                        pad_img.clone(),
                        TextureAtlas {
                            layout: pad_lay.clone(),
                            index: atlas_idx,
                        },
                    ),
                    Transform::from_xyz(world_pos.x, world_pos.y, -9.0),
                ))
            } else {
                // Fallback: flat colored sprite if atlas unavailable.
                commands.spawn((
                    DespawnOnExit(PlayState::Exploring),
                    Sprite {
                        color: Color::srgb(0.35, 0.35, 0.4),
                        custom_size: Some(Vec2::splat(tile_px)),
                        ..default()
                    },
                    Transform::from_xyz(world_pos.x, world_pos.y, -9.0),
                ))
            };

            // Center tile is the interaction sensor.
            if dx == 0 && dy == 0 {
                entity.insert((
                    Building { kind: BuildingKind::ShipPad },
                    Sensor,
                    RigidBody::Static,
                    Collider::rectangle(tile_px, tile_px),
                    CollisionEventsEnabled,
                    CollisionLayers::new(GameLayer::Surface, [GameLayer::Surface]),
                ));
            }
        }
        // Pad label — just above the center tile.
        {
            let label_world = tile_to_world(pad_cx, pad_cy, map_w, map_h, tile_px);
            spawn_building_label(
                &mut commands,
                BuildingKind::ShipPad.label(),
                Vec3::new(label_world.x, label_world.y + tile_px * 0.8, 5.0),
            );
        }

        // ── Spawn building sprites from pre-computed assignments ─────
        for &(kind, anchor_tx, anchor_ty, ti) in &building_assignments {
            let tmpl = templates.get(ti);
            let (bw, bh) = tmpl.map(|t| (t.width, t.height)).unwrap_or((2, 2));

            if let (Some(tmpl), Some(ext_img), Some(ext_lay)) =
                (tmpl, ext_atlas_handle.as_ref(), ext_layout.as_ref())
            {
                for row in 0..tmpl.height {
                    for col in 0..tmpl.width {
                        let tile_idx = tmpl.tiles[row as usize][col as usize];
                        if tile_idx == 0 {
                            continue;
                        }
                        let tx = anchor_tx + col;
                        let ty = anchor_ty + row;
                        let world_pos = tile_to_world(tx, ty, map_w, map_h, tile_px);

                        let is_door = tile_idx == EXT_DOOR;
                        let collision_code = ext_collision
                            .get(tile_idx as usize)
                            .copied()
                            .unwrap_or(1);

                        let mut entity = commands.spawn((
                            DespawnOnExit(PlayState::Exploring),
                            Sprite::from_atlas_image(
                                ext_img.clone(),
                                TextureAtlas {
                                    layout: ext_lay.clone(),
                                    index: tile_idx as usize,
                                },
                            ),
                            Transform::from_xyz(world_pos.x, world_pos.y, -5.0),
                        ));

                        if is_door {
                            entity.insert((
                                Building { kind },
                                Sensor,
                                RigidBody::Static,
                                Collider::rectangle(tile_px, tile_px),
                                CollisionEventsEnabled,
                                CollisionLayers::new(
                                    GameLayer::Surface,
                                    [GameLayer::Surface],
                                ),
                            ));
                        } else if collision_code == 1 {
                            entity.insert((
                                RigidBody::Static,
                                Collider::rectangle(tile_px, tile_px),
                                CollisionLayers::new(
                                    GameLayer::Surface,
                                    [GameLayer::Surface],
                                ),
                            ));
                        }
                    }
                }
            } else {
                let world_pos = tile_to_world(anchor_tx, anchor_ty, map_w, map_h, tile_px);
                commands.spawn((
                    DespawnOnExit(PlayState::Exploring),
                    Building { kind },
                    RigidBody::Static,
                    Collider::rectangle(tile_px * 2.0, tile_px * 2.0),
                    Sensor,
                    CollisionEventsEnabled,
                    CollisionLayers::new(GameLayer::Surface, [GameLayer::Surface]),
                    Transform::from_xyz(world_pos.x, world_pos.y, 0.0),
                ));
            }

            // Place label just above the door.
            let (door_tx, door_ty) = tmpl
                .and_then(|t| t.entry_points.first())
                .map(|&(c, r)| (anchor_tx + c, anchor_ty + r))
                .unwrap_or((anchor_tx + bw / 2, anchor_ty));
            let door_world = tile_to_world(door_tx, door_ty, map_w, map_h, tile_px);
            spawn_building_label(
                &mut commands,
                kind.label(),
                Vec3::new(door_world.x, door_world.y + tile_px * 0.8, 5.0),
            );
        }

        (col_data, placed_buildings)
    } else {
        eprintln!(
            "[surface] WARNING: could not load world data for biome '{biome_name}' \
             (lut={}, manifest={}) — falling back to plain ground",
            lut.is_some(),
            manifest.is_some(),
        );
        commands.spawn((
            DespawnOnExit(PlayState::Exploring),
            Sprite {
                color: Color::srgb(0.25, 0.25, 0.2),
                custom_size: Some(Vec2::splat(WORLD_WIDTH as f32 * tile_px)),
                ..default()
            },
            Transform::from_xyz(0.0, 0.0, -10.0),
        ));
        let n = (map_w * map_h) as usize;
        (vec![0u8; n], Vec::new())
    };

    // Spawn on the landing pad (always at map center).
    let spawn_pos = tile_to_world(map_w / 2, map_h / 2, map_w, map_h, tile_px);

    let walker_sheet = WalkerSheet::load(
        &asset_server,
        &mut atlas_layouts,
        game_state.gender.sprite_dir(),
    );
    let initial_index = sprite_index(Facing::Down, WalkFrame::Still);

    commands.spawn((
        DespawnOnExit(PlayState::Exploring),
        Walker,
        WalkerAnim::default(),
        RigidBody::Dynamic,
        LockedAxes::ROTATION_LOCKED,
        Collider::circle(WALKER_RADIUS),
        CollisionLayers::new(GameLayer::Surface, [GameLayer::Surface]),
        CollisionEventsEnabled,
        LinearDamping(WALKER_DAMPING),
        MaxLinearSpeed(WALK_SPEED),
        LinearVelocity(Vec2::ZERO),
        Sprite::from_atlas_image(
            walker_sheet.image.clone(),
            TextureAtlas {
                layout: walker_sheet.layout.clone(),
                index: initial_index,
            },
        ),
        Transform::from_xyz(spawn_pos.x, spawn_pos.y, 1.0),
    ));

    commands.insert_resource(walker_sheet);

    commands.spawn((
        DespawnOnExit(PlayState::Exploring),
        InteractPrompt,
        Text2d::new("Press E"),
        TextFont {
            font_size: 10.0,
            ..default()
        },
        TextColor(Color::srgba(1.0, 1.0, 1.0, 0.8)),
        Transform::from_xyz(0.0, -999.0, 6.0) // Bug 6: z above labels
            .with_scale(Vec3::splat(0.25)),
        Visibility::Hidden,
    ));

    if let Ok(mut cam_tf) = camera_query.single_mut() {
        // Bug 4: get_single_mut
        cam_tf.translation = Vec3::new(0.0, 0.0, cam_tf.translation.z);
    }

    zoom.target = SURFACE_CAMERA_SCALE;
}

/// Restore the player ship and reset camera zoom on exiting Exploring.
fn teardown_surface(
    mut commands: Commands,
    player_query: Query<Entity, With<Player>>,
    mut zoom: ResMut<CameraZoom>,
    mut nearby: ResMut<NearbyBuilding>,
    mut active_ui: ResMut<ActiveBuildingUI>,
    mut terrain_speed: ResMut<TerrainSpeedModifier>,
) {
    // Show the player ship again.
    if let Ok(ship_entity) = player_query.single() {
        commands.entity(ship_entity).insert(Visibility::Inherited);
    }

    // Reset surface state.
    nearby.0 = None;
    active_ui.0 = None;
    terrain_speed.0 = 1.0;

    // Trigger zoom-out.
    zoom.target = SPACE_CAMERA_SCALE;

    // Walker + tilemap + buildings auto-despawn via DespawnOnExit.
}

// ---------------------------------------------------------------------------
// Walking Input
// ---------------------------------------------------------------------------

/// WASD / arrow key movement for the walker. Frozen when a building UI is open.
/// Speed is divided by the current terrain's movement cost.
fn walker_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut walkers: Query<&mut LinearVelocity, With<Walker>>,
    active_ui: Res<ActiveBuildingUI>,
    terrain_speed: Res<TerrainSpeedModifier>,
) {
    let Ok(mut vel) = walkers.single_mut() else {
        return;
    };

    // Don't move while a building UI is open.
    if active_ui.0.is_some() {
        vel.0 = Vec2::ZERO;
        return;
    }

    let mut dir = Vec2::ZERO;
    if keyboard.any_pressed([KeyCode::KeyW, KeyCode::ArrowUp]) {
        dir.y += 1.0;
    }
    if keyboard.any_pressed([KeyCode::KeyS, KeyCode::ArrowDown]) {
        dir.y -= 1.0;
    }
    if keyboard.any_pressed([KeyCode::KeyA, KeyCode::ArrowLeft]) {
        dir.x -= 1.0;
    }
    if keyboard.any_pressed([KeyCode::KeyD, KeyCode::ArrowRight]) {
        dir.x += 1.0;
    }

    let speed = WALK_SPEED / terrain_speed.0;
    vel.0 = dir.normalize_or_zero() * speed;
}

// ---------------------------------------------------------------------------
// Walker Animation
// ---------------------------------------------------------------------------

/// Velocity threshold below which the walker is considered stationary.
const MOVE_THRESHOLD: f32 = 5.0;

/// Update walker facing direction, walk timer, and set the atlas index.
fn animate_walker(
    time: Res<Time>,
    mut walkers: Query<(&LinearVelocity, &mut WalkerAnim, &mut Sprite), With<Walker>>,
) {
    let Ok((vel, mut anim, mut sprite)) = walkers.single_mut() else {
        return;
    };

    let speed = vel.0.length();
    if speed > MOVE_THRESHOLD {
        anim.is_moving = true;
        // Pick facing from the dominant axis.
        if vel.x.abs() > vel.y.abs() {
            anim.facing = if vel.x > 0.0 {
                Facing::Right
            } else {
                Facing::Left
            };
        } else {
            anim.facing = if vel.y > 0.0 {
                Facing::Up
            } else {
                Facing::Down
            };
        }
        // Tick the walk timer and advance through 4 phases:
        // still → w1 → still → w2 → ...
        anim.walk_timer.tick(time.delta());
        if anim.walk_timer.just_finished() {
            anim.walk_phase = (anim.walk_phase + 1) % 4;
        }
    } else {
        anim.is_moving = false;
        anim.walk_phase = 0;
        anim.walk_timer.reset();
    }

    if let Some(atlas) = &mut sprite.texture_atlas {
        atlas.index = anim.atlas_index();
    }
}

// ---------------------------------------------------------------------------
// Building Interaction
// ---------------------------------------------------------------------------

/// Track which building the walker overlaps (same pattern as track_nearby_planet).
fn track_nearby_building(
    mut collision_starts: MessageReader<CollisionStart>,
    mut collision_ends: MessageReader<CollisionEnd>,
    buildings: Query<&Building>,
    walkers: Query<(), With<Walker>>,
    mut nearby: ResMut<NearbyBuilding>,
) {
    for event in collision_starts.read() {
        let (a, b) = (event.collider1, event.collider2);
        if let Some((bldg_entity, walker_present)) = match (
            buildings.get(a).ok(),
            buildings.get(b).ok(),
            walkers.get(a).ok(),
            walkers.get(b).ok(),
        ) {
            (Some(bldg), _, _, Some(_)) => Some((a, bldg)),
            (_, Some(bldg), Some(_), _) => Some((b, bldg)),
            _ => None,
        } {
            nearby.0 = Some((bldg_entity, walker_present.kind));
        }
    }
    for event in collision_ends.read() {
        let (a, b) = (event.collider1, event.collider2);
        let involves_building = buildings.contains(a) || buildings.contains(b);
        let involves_walker = walkers.contains(a) || walkers.contains(b);
        if involves_building && involves_walker {
            nearby.0 = None;
        }
    }
}

/// Track the terrain movement cost under the walker.  When the walker
/// enters/exits `TerrainSensor` zones the speed modifier updates.
fn track_terrain_speed(
    mut collision_starts: MessageReader<CollisionStart>,
    mut collision_ends: MessageReader<CollisionEnd>,
    sensors: Query<&crate::world_assets::TerrainSensor>,
    walkers: Query<(), With<Walker>>,
    mut modifier: ResMut<TerrainSpeedModifier>,
) {
    for event in collision_starts.read() {
        let (a, b) = (event.collider1, event.collider2);
        let sensor = sensors.get(a).ok().or_else(|| sensors.get(b).ok());
        let is_walker = walkers.contains(a) || walkers.contains(b);
        if let (Some(ts), true) = (sensor, is_walker) {
            modifier.0 = ts.movement_cost;
        }
    }
    for event in collision_ends.read() {
        let (a, b) = (event.collider1, event.collider2);
        let is_sensor = sensors.contains(a) || sensors.contains(b);
        let is_walker = walkers.contains(a) || walkers.contains(b);
        if is_sensor && is_walker {
            modifier.0 = 1.0; // back to normal speed
        }
    }
}

/// Handle E key to interact with buildings, Escape to close UI or return
/// to main menu.
fn building_interact(
    keyboard: Res<ButtonInput<KeyCode>>,
    nearby: Res<NearbyBuilding>,
    mut active_ui: ResMut<ActiveBuildingUI>,
    mut next_state: ResMut<NextState<PlayState>>,
    game_state: Res<crate::game_save::PlayerGameState>,
) {
    if keyboard.just_pressed(KeyCode::Escape) {
        if active_ui.0.is_some() {
            // Close building UI.
            active_ui.0 = None;
        } else {
            // No building UI open — save and return to main menu.
            game_state.save();
            next_state.set(PlayState::MainMenu);
        }
        return;
    }

    // Open building on E.
    if keyboard.just_pressed(KeyCode::KeyE) {
        if let Some((_, kind)) = nearby.0 {
            if kind == BuildingKind::ShipPad {
                // Transition to Landed for Launch/Repair UI.
                next_state.set(PlayState::Landed);
            } else {
                active_ui.0 = Some(kind);
            }
        }
    }
}

/// Show/hide the "Press E" prompt based on proximity to a building.
fn update_interact_prompt(
    nearby: Res<NearbyBuilding>,
    active_ui: Res<ActiveBuildingUI>,
    walker_query: Query<&Transform, With<Walker>>,
    mut prompt_query: Query<
        (&mut Transform, &mut Visibility),
        (With<InteractPrompt>, Without<Walker>),
    >,
) {
    let Ok((mut prompt_tf, mut prompt_vis)) = prompt_query.single_mut() else {
        return;
    };

    if nearby.0.is_some() && active_ui.0.is_none() {
        *prompt_vis = Visibility::Visible;
        if let Ok(walker_tf) = walker_query.single() {
            prompt_tf.translation.x = walker_tf.translation.x;
            prompt_tf.translation.y = walker_tf.translation.y - 12.0;
        }
    } else {
        *prompt_vis = Visibility::Hidden;
    }
}

// ---------------------------------------------------------------------------
// Building egui UIs
// ---------------------------------------------------------------------------

/// Render the appropriate egui window based on which building the player is
/// interacting with. Reuses the extracted tab renderers from planet_ui.rs.
#[allow(clippy::too_many_arguments)]
fn surface_building_ui(
    mut egui_contexts: EguiContexts,
    mut active_ui: ResMut<ActiveBuildingUI>,
    landed_context: Res<LandedContext>,
    current_system: Res<CurrentStarSystem>,
    item_universe: Res<ItemUniverse>,
    mut player_query: Query<&mut Ship, With<Player>>,
    mut buy_ship_writer: MessageWriter<BuyShip>,
    mission_log: Res<MissionLog>,
    mission_offers: Res<MissionOffers>,
    mission_catalog: Res<MissionCatalog>,
    unlocks: Res<PlayerUnlocks>,
    mut accept_writer: MessageWriter<AcceptMission>,
    mut decline_writer: MessageWriter<DeclineMission>,
    mut abandon_writer: MessageWriter<AbandonMission>,
) {
    let Some(kind) = active_ui.0 else {
        return;
    };
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };
    let planet_name = landed_context.planet_name.as_deref().unwrap_or("");
    let planet_data = item_universe
        .star_systems
        .get(&current_system.0)
        .and_then(|sys| sys.planets.get(planet_name));

    let title = kind.label();

    bevy_egui::egui::Window::new(title)
        .collapsible(false)
        .resizable(true)
        .anchor(bevy_egui::egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            match kind {
                BuildingKind::Market => {
                    if let (Ok(mut ship), Some(pd)) = (player_query.single_mut(), planet_data) {
                        render_trade_tab(ui, &mut ship, pd, &item_universe);
                    } else {
                        ui.label("No commodities available.");
                    }
                }
                BuildingKind::Outfitter => {
                    if let (Ok(mut ship), Some(pd)) = (player_query.single_mut(), planet_data) {
                        render_outfitter_tab(ui, &mut ship, pd, &item_universe, &unlocks);
                    } else {
                        ui.label("No equipment available.");
                    }
                }
                BuildingKind::Shipyard => {
                    if let (Ok(ship), Some(pd)) = (player_query.single(), planet_data) {
                        render_shipyard_tab(
                            ui,
                            &ship,
                            pd,
                            &item_universe,
                            &unlocks,
                            &mut buy_ship_writer,
                        );
                    } else {
                        ui.label("No ships for sale.");
                    }
                }
                BuildingKind::Bar => {
                    let free = player_query
                        .single()
                        .map(|s| s.remaining_cargo_space())
                        .unwrap_or(0);
                    render_bar_tab(
                        ui,
                        planet_name,
                        &mission_offers,
                        &mission_catalog,
                        free,
                        &mut accept_writer,
                        &mut decline_writer,
                    );
                }
                BuildingKind::MissionControl => {
                    let free = player_query
                        .single()
                        .map(|s| s.remaining_cargo_space())
                        .unwrap_or(0);
                    render_missions_tab(
                        ui,
                        &mission_log,
                        &mission_offers,
                        &mission_catalog,
                        free,
                        &mut accept_writer,
                        &mut abandon_writer,
                    );
                }
                BuildingKind::ShipPad => {
                    // Should not reach here — ShipPad transitions to Landed state.
                    ui.label("Return to your ship.");
                }
            }
            ui.separator();
            if ui.button("Close [Esc]").clicked() {
                active_ui.0 = None;
            }
        });
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

/// Smoothly interpolate the camera zoom via `OrthographicProjection::scale`.
fn animate_camera_zoom(
    zoom: Res<CameraZoom>,
    time: Res<Time>,
    mut cameras: Query<&mut Projection, With<Camera2d>>,
) {
    let Ok(mut proj) = cameras.single_mut() else {
        return;
    };
    let Projection::Orthographic(ref mut ortho) = *proj else {
        return;
    };
    let dt = time.delta_secs();
    let speed = CAMERA_ZOOM_SPEED * dt;
    ortho.scale = ortho.scale + (zoom.target - ortho.scale) * speed.min(1.0);
}

/// Camera follows the walker during Exploring.
fn camera_follow_walker(
    walker_query: Query<&Transform, (With<Walker>, Without<Camera2d>)>,
    mut camera_query: Query<&mut Transform, (With<Camera2d>, Without<Walker>)>,
) {
    let Ok(walker_tf) = walker_query.single() else {
        return;
    };
    let Ok(mut cam_tf) = camera_query.single_mut() else {
        return;
    };
    cam_tf.translation = cam_tf.translation.lerp(walker_tf.translation, 0.1);
}
