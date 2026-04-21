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
const SURFACE_CAMERA_SCALE: f32 = 0.8;
/// Camera scale in space (normal).
const SPACE_CAMERA_SCALE: f32 = 1.0;
/// How fast the camera zoom interpolates (per second).
const CAMERA_ZOOM_SPEED: f32 = 4.0;

/// Base path for world tile assets.
const WORLDS_DIR: &str = "sprites/worlds";

/// World dimensions in tiles. Small for testing; increase for production.
pub const WORLD_WIDTH: u32 = 64;
pub const WORLD_HEIGHT: u32 = 64;

/// Tile size in pixels (must match the atlas tile size from tilegen.py).
pub const TILE_PX: f32 = 32.0;

// ── fBm parameters ──────────────────────────────────────────────────────
const FBM_SCALE: f32 = 4.0;
const FBM_OCTAVES: u32 = 5;
const FBM_LACUNARITY: f32 = 2.0;
const FBM_GAIN: f32 = 0.5;
/// Number of terrain types per biome (must match the atlas row count).
/// Number of terrain types per biome (5 biome terrains + 1 void border).
const N_TERRAIN_TYPES: u32 = 6;

// ---------------------------------------------------------------------------
// Building kinds
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuildingKind {
    ShipPad,
    MechanicShop,
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
            BuildingKind::MechanicShop => "Mechanic",
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
            BuildingKind::MechanicShop => Color::srgb(0.8, 0.5, 0.2),
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
// Walker sprite loading
// ---------------------------------------------------------------------------

use crate::surface_character::{CharacterAnim, SPRITE_COLS};

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
            SPRITE_COLS as u32,
            4,
            None,
            None,
        ));
        Self { image, layout }
    }
}

/// Interaction zone for a building.
#[derive(Component)]
pub struct Building {
    pub kind: BuildingKind,
}

/// Marks a door sprite for depth-crossing sound detection.
/// `walker_was_behind` tracks whether the walker was behind (higher z)
/// the door last frame.
#[derive(Component)]
struct DoorSprite {
    walker_was_behind: Option<bool>,
}

/// Which building (if any) the walker is currently overlapping,
/// plus a count of active sensor overlaps (so exiting one of two
/// adjacent door tiles doesn't clear the state).
#[derive(Resource, Default)]
pub struct NearbyBuilding {
    pub current: Option<(Entity, BuildingKind)>,
    overlap_count: u32,
}

/// Which building's egui UI is currently open. `None` = walking freely.
#[derive(Resource, Default)]
pub struct ActiveBuildingUI(pub Option<BuildingKind>);

/// Current movement cost multiplier from the terrain the walker is on.
/// 1.0 = normal speed, 2.0 = half speed, etc.
#[derive(Resource)]
struct TerrainSpeedModifier(f32);

/// Per-terrain footstep data, indexed by terrain index.
#[derive(Resource, Default)]
struct FootstepData {
    /// (surface_name, volume) per terrain index.
    terrains: Vec<(String, f32)>,
    /// Terrain index per tile, flat array (map_w × map_h, bottom-up).
    terrain_map: Vec<u32>,
    map_w: u32,
    map_h: u32,
}

impl Default for TerrainSpeedModifier {
    fn default() -> Self {
        Self(1.0)
    }
}

/// The generated mini-map image handle + map dimensions, used by the HUD.
#[derive(Resource)]
pub struct SurfaceMiniMap {
    pub image: Handle<Image>,
    pub map_w: u32,
    pub map_h: u32,
    /// (tile_x, tile_y, BuildingKind) for each placed building.
    pub buildings: Vec<(u32, u32, BuildingKind)>,
    /// Landing pad center.
    pub pad_pos: (u32, u32),
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
fn spawn_building_label(commands: &mut Commands, text: &str, pos: Vec3) {
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

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub fn surface_plugin(app: &mut App) {
    app.add_plugins(TilemapPlugin)
        .init_resource::<CameraZoom>()
        .init_resource::<NearbyBuilding>()
        .init_resource::<ActiveBuildingUI>()
        .init_resource::<TerrainSpeedModifier>()
        .init_resource::<FootstepData>()
        .add_systems(
            OnEnter(PlayState::Exploring),
            (setup_surface, save_on_explore),
        )
        .add_systems(
            OnExit(PlayState::Exploring),
            (teardown_surface, crate::surface_civilians::cleanup_civilians),
        )
        .add_systems(
            Update,
            (
                walker_input,
                crate::surface_character::animate_characters,
                play_footstep,
                track_nearby_building,
                track_terrain_speed,
                building_interact,
                update_interact_prompt,
            )
                .run_if(in_state(PlayState::Exploring)),
        )
        .add_systems(
            Update,
            (
                crate::surface_objects::update_shy_objects,
                crate::surface_objects::animate_landscape_objects,
                crate::surface_objects::depth_sort_walker,
                door_depth_sound,
                crate::surface_civilians::spawn_civilians,
                crate::surface_npc::run_npc_behaviors,
                crate::surface_civilians::depth_sort_npcs,
            )
                .run_if(in_state(PlayState::Exploring)),
        )
        .add_systems(Update, animate_camera_zoom)
        .add_systems(
            bevy_egui::EguiPrimaryContextPass,
            surface_building_ui.run_if(in_state(PlayState::Exploring)),
        )
        .add_systems(
            Update,
            egui_button_click_sound.run_if(in_state(PlayState::Exploring)),
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
fn save_on_explore(
    game_state: Res<crate::game_save::PlayerGameState>,
    session_data: Res<crate::session::SessionSaveData>,
) {
    crate::game_save::write_save(&game_state, &session_data);
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
    (-1, 1, 0),  // TL corner
    (0, 1, 1),   // T  edge
    (1, 1, 2),   // TR corner
    (-1, 0, 3),  // L  edge
    (0, 0, 4),   // C  center
    (1, 0, 5),   // R  edge
    (-1, -1, 6), // BL corner
    (0, -1, 7),  // B  edge
    (1, -1, 8),  // BR corner
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
    mut comms: ResMut<crate::hud::CommsChannel>,
    mut images: ResMut<Assets<Image>>,
) {
    comms.send("");
    commands.insert_resource(ClearColor(Color::BLACK));

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

        // Derive collision and movement cost maps from terrain + manifest.
        let (collision_codes, movement_costs): (Vec<u8>, Vec<f32>) =
            if let Some(biome_meta) = manifest.biomes.get(biome_name) {
                (
                    biome_meta.terrains.iter().map(|t| t.collision).collect(),
                    biome_meta
                        .terrains
                        .iter()
                        .map(|t| t.movement_cost)
                        .collect(),
                )
            } else {
                (
                    vec![0; N_TERRAIN_TYPES as usize],
                    vec![1.0; N_TERRAIN_TYPES as usize],
                )
            };

        // Generate initial terrain+collision for building placement.
        let initial_terrain = crate::fbm::generate_terrain_map(
            map_w, map_h, N_TERRAIN_TYPES,
            FBM_SCALE, FBM_OCTAVES, FBM_LACUNARITY, FBM_GAIN, seed,
        );

        // ── Pre-tilemap building placement ─────────────────────────────
        // Find building positions on the initial terrain, then force
        // nearby tiles to walkable terrain and re-clamp before building
        // the tilemap.
        let building_kinds = building_kinds_for_planet(&planet_name, &item_universe, system_name);

        let initial_col: Vec<u8> = initial_terrain
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
        let ext_atlas_handle: Option<Handle<Image>> =
            bldg_style.map(|s| asset_server.load(format!("{WORLDS_DIR}/{}", s.exterior_atlas)));
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
                        let path = format!("assets/{WORLDS_DIR}/buildings/{style_name}_{name}.ron");
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
        // Mechanic shop: random cardinal direction, 3-6 tiles from pad.
        let (mech_x, mech_y) = {
            use rand::{Rng, SeedableRng};
            let mech_seed = planet_name
                .bytes()
                .fold(0u64, |acc, b| acc.wrapping_mul(37).wrapping_add(b as u64));
            let mut rng = rand::rngs::StdRng::seed_from_u64(mech_seed);
            let dist = rng.gen_range(3i32..=6);
            let dir = rng.gen_range(0u8..4);
            let (dx, dy) = match dir {
                0 => (dist, 0),      // right
                1 => (-dist - 3, 0), // left (offset by mechanic width)
                2 => (0, dist),      // up
                _ => (0, -dist - 2), // down (offset by mechanic height)
            };
            (
                (pad_cx as i32 + dx).max(2) as u32,
                (pad_cy as i32 + dy).max(2) as u32,
            )
        };

        let min_building_spacing = 10_u32;
        let mut placed_buildings: Vec<(u32, u32, u32, u32)> = Vec::new(); // (x, y, w, h)
        // Reserve the pad area (3x3 centered on pad_cx, pad_cy).
        placed_buildings.push((pad_cx - 1, pad_cy - 1, 3, 3));
        // Reserve the mechanic area (4×3).
        placed_buildings.push((mech_x, mech_y, 4, 3));
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
                        if initial_col[idx] == 1 {
                            // Solid
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
                let ti = if templates.is_empty() {
                    0
                } else {
                    (template_idx - 1) % templates.len()
                };
                placed_buildings.push((tx, ty, bw, bh));
                building_assignments.push((*kind, tx, ty, ti));
            }
        }

        // ── Constrain terrain: force walkable near buildings + force paths ──
        // Build door positions for pathfinding.
        let mut door_positions: Vec<(BuildingKind, (u32, u32))> = Vec::new();
        door_positions.push((BuildingKind::ShipPad, (pad_cx, pad_cy)));
        door_positions.push((BuildingKind::MechanicShop, (mech_x + 1, mech_y)));
        for &(kind, bx, by, ti) in &building_assignments {
            if let Some(tmpl) = templates.get(ti) {
                if let Some(&(dc, dr)) = tmpl.entry_points.first() {
                    door_positions.push((kind, (bx + dc, by + dr)));
                }
            }
        }

        // Build a set of solid building tiles for pathfinding.
        // Only non-transparent, non-door tiles block movement.
        let mut solid_building_tiles: std::collections::HashSet<(u32, u32)> =
            std::collections::HashSet::new();

        // Template buildings: mark each non-zero tile as solid (except doors).
        for &(_, bx, by, ti) in &building_assignments {
            if let Some(tmpl) = templates.get(ti) {
                let door_set: std::collections::HashSet<(u32, u32)> = tmpl
                    .entry_points
                    .iter()
                    .map(|&(dc, dr)| (bx + dc, by + dr))
                    .collect();
                for row in 0..tmpl.height {
                    for col in 0..tmpl.width {
                        let tile_idx = tmpl.tiles[row as usize][col as usize];
                        if tile_idx == 0 { continue; } // transparent
                        let pos = (bx + col, by + row);
                        if door_set.contains(&pos) { continue; } // door
                        solid_building_tiles.insert(pos);
                    }
                }
            }
        }

        // Mechanic building: all tiles except garage doors are solid.
        for row in 0..3u32 {
            for col in 0..4u32 {
                let atlas_row = 2 - row;
                let is_garage = atlas_row == 2 && (col == 1 || col == 2);
                if !is_garage {
                    solid_building_tiles.insert((mech_x + col, mech_y + row));
                }
            }
        }

        // Apply terrain constraints + ensure connectivity.
        let generated = crate::surface_terrain::generate_constrained_terrain(
            map_w, map_h, N_TERRAIN_TYPES, seed,
            FBM_SCALE, FBM_OCTAVES, FBM_LACUNARITY, FBM_GAIN,
            &collision_codes, &movement_costs,
            &placed_buildings, &door_positions, &solid_building_tiles,
        );
        let terrain_flat = generated.terrain;
        let col_data = generated.collision;

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

        let map_size = TilemapSize { x: map_w, y: map_h };
        let mut tile_storage = TileStorage::empty(map_size);
        let tilemap_entity = commands.spawn(DespawnOnExit(PlayState::Exploring)).id();
        let tilemap_id = TilemapId(tilemap_entity);

        for y in 0..map_h {
            for x in 0..map_w {
                let tex_idx = crate::world_assets::tile_texture_index(
                    &terrain_map,
                    x as i32,
                    y as i32,
                    map_w as i32,
                    map_h as i32,
                    lut_data,
                );
                let tile_pos = TilePos { x, y };
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

        let col_asset = crate::world_assets::CollisionMapAsset {
            width: map_w,
            height: map_h,
            data: col_data.clone(),
        };
        let map_origin = Vec2::new(
            -(map_w as f32 * tile_px / 2.0),
            -(map_h as f32 * tile_px / 2.0),
        );
        let surface_layers = CollisionLayers::new(GameLayer::Surface, [GameLayer::Surface, GameLayer::Character]);
        crate::world_assets::spawn_collision_entities(
            &mut commands,
            &col_asset,
            &terrain_flat,
            &movement_costs,
            tile_px,
            map_origin,
            surface_layers,
        );

        // ── Spawn landing pad (plus/cross shape, 3x3 atlas) ──────────
        let pad_atlas_handle: Option<Handle<Image>> =
            bldg_style.map(|s| asset_server.load(format!("{WORLDS_DIR}/{}", s.landing_pad_atlas)));
        let pad_layout: Option<Handle<TextureAtlasLayout>> = pad_atlas_handle.as_ref().map(|_| {
            atlas_layouts.add(TextureAtlasLayout::from_grid(
                UVec2::new(tile_px as u32, tile_px as u32),
                3,
                3,
                None,
                None,
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
                    Building {
                        kind: BuildingKind::ShipPad,
                    },
                    Sensor,
                    RigidBody::Static,
                    Collider::rectangle(tile_px, tile_px),
                    CollisionEventsEnabled,
                    CollisionLayers::new(GameLayer::Surface, [GameLayer::Surface, GameLayer::Character]),
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

        // ── Spawn mechanic shop next to the landing pad ──────────────
        let mech_atlas_handle: Option<Handle<Image>> =
            bldg_style.map(|s| asset_server.load(format!("{WORLDS_DIR}/{}", s.mechanic_atlas)));
        let mech_layout: Option<Handle<TextureAtlasLayout>> =
            mech_atlas_handle.as_ref().map(|_| {
                atlas_layouts.add(TextureAtlasLayout::from_grid(
                    UVec2::new(tile_px as u32, tile_px as u32),
                    4,
                    3, // MECH_COLS × MECH_ROWS
                    None,
                    None,
                ))
            });
        // Mechanic shop position was computed earlier (mech_x, mech_y).
        // All tiles in the building share the same z based on the floor row.
        let mech_floor_world = tile_to_world(mech_x, mech_y, map_w, map_h, tile_px);
        let mech_z = crate::surface_objects::depth_z(mech_floor_world.y - tile_px * 0.5);
        if let (Some(mech_img), Some(mech_lay)) = (mech_atlas_handle.as_ref(), mech_layout.as_ref())
        {
            // The mechanic atlas is 4×3, stored top-down in the PNG.
            // Row 0 in the atlas = roof (top of building).
            // We stamp bottom-up: atlas row 2 → lowest y, atlas row 0 → highest y.
            for row in 0..3u32 {
                for col in 0..4u32 {
                    let atlas_row = 2 - row; // flip: bottom-up
                    let atlas_idx = (atlas_row * 4 + col) as usize;
                    let tx = mech_x + col;
                    let ty = mech_y + row;
                    let world_pos = tile_to_world(tx, ty, map_w, map_h, tile_px);

                    // Garage door tiles (atlas row 2, cols 1-2) are sensors.
                    let is_garage = atlas_row == 2 && (col == 1 || col == 2);

                    let mut entity = commands.spawn((
                        DespawnOnExit(PlayState::Exploring),
                        Sprite::from_atlas_image(
                            mech_img.clone(),
                            TextureAtlas {
                                layout: mech_lay.clone(),
                                index: atlas_idx,
                            },
                        ),
                        Transform::from_xyz(world_pos.x, world_pos.y, mech_z),
                    ));

                    if is_garage {
                        entity.insert((
                            Building {
                                kind: BuildingKind::MechanicShop,
                            },
                            DoorSprite { walker_was_behind: None },
                            Sensor,
                            RigidBody::Static,
                            Collider::rectangle(tile_px, tile_px),
                            CollisionEventsEnabled,
                            CollisionLayers::new(GameLayer::Surface, [GameLayer::Surface, GameLayer::Character]),
                        ));
                    } else {
                        entity.insert((
                            RigidBody::Static,
                            Collider::rectangle(tile_px, tile_px),
                            CollisionLayers::new(GameLayer::Surface, [GameLayer::Surface, GameLayer::Character]),
                        ));
                    }
                }
            }
            // Mechanic label above garage door.
            let label_world = tile_to_world(mech_x + 1, mech_y, map_w, map_h, tile_px);
            spawn_building_label(
                &mut commands,
                "Mechanic",
                Vec3::new(
                    label_world.x + tile_px * 0.5,
                    label_world.y + tile_px * 0.8,
                    5.0,
                ),
            );
        }

        // ── Spawn building sprites from pre-computed assignments ─────
        for &(kind, anchor_tx, anchor_ty, ti) in &building_assignments {
            let tmpl = templates.get(ti);
            let (bw, bh) = tmpl.map(|t| (t.width, t.height)).unwrap_or((2, 2));

            // Depth-sort: all tiles in the building share z based on the floor.
            let bldg_floor_world = tile_to_world(anchor_tx, anchor_ty, map_w, map_h, tile_px);
            let bldg_z = crate::surface_objects::depth_z(bldg_floor_world.y - tile_px * 0.5);

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
                        let collision_code =
                            ext_collision.get(tile_idx as usize).copied().unwrap_or(1);

                        let mut entity = commands.spawn((
                            DespawnOnExit(PlayState::Exploring),
                            Sprite::from_atlas_image(
                                ext_img.clone(),
                                TextureAtlas {
                                    layout: ext_lay.clone(),
                                    index: tile_idx as usize,
                                },
                            ),
                            Transform::from_xyz(world_pos.x, world_pos.y, bldg_z),
                        ));

                        if is_door {
                            entity.insert((
                                Building { kind },
                                DoorSprite { walker_was_behind: None },
                                Sensor,
                                RigidBody::Static,
                                Collider::rectangle(tile_px, tile_px),
                                CollisionEventsEnabled,
                                CollisionLayers::new(GameLayer::Surface, [GameLayer::Surface, GameLayer::Character]),
                            ));
                        } else if collision_code == 1 {
                            entity.insert((
                                RigidBody::Static,
                                Collider::rectangle(tile_px, tile_px),
                                CollisionLayers::new(GameLayer::Surface, [GameLayer::Surface, GameLayer::Character]),
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
                    CollisionLayers::new(GameLayer::Surface, [GameLayer::Surface, GameLayer::Character]),
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

        // ── Build mini-map image from terrain + map_colors ──────────
        let map_colors: Vec<(u8, u8, u8)> = manifest
            .biomes
            .get(biome_name)
            .map(|b| b.terrains.iter().map(|t| t.map_color).collect())
            .unwrap_or_default();
        {
            let mut pixels = vec![255u8; (map_w * map_h * 4) as usize];
            for y in 0..map_h {
                for x in 0..map_w {
                    // Terrain data is bottom-up (y=0 = bottom), but image
                    // pixels are top-down (row 0 = top). Flip Y.
                    let src = (y * map_w + x) as usize;
                    let dst_y = map_h - 1 - y;
                    let pi = ((dst_y * map_w + x) * 4) as usize;
                    let t = terrain_flat[src] as usize;
                    let (r, g, b) = map_colors.get(t).copied().unwrap_or((128, 128, 128));
                    pixels[pi] = r;
                    pixels[pi + 1] = g;
                    pixels[pi + 2] = b;
                    pixels[pi + 3] = 255;
                }
            }
            // Helper: set a pixel on the mini-map (with Y-flip for image coords).
            let set_px = |pixels: &mut [u8], x: u32, y: u32, color: [u8; 3]| {
                if x < map_w && y < map_h {
                    let iy = map_h - 1 - y;
                    let pi = ((iy * map_w + x) * 4) as usize;
                    pixels[pi] = color[0];
                    pixels[pi + 1] = color[1];
                    pixels[pi + 2] = color[2];
                }
            };

            // Render full building footprints using per-tile colors.
            let tile_colors = bldg_style
                .map(|s| &s.ext_tile_colors)
                .filter(|c| !c.is_empty());
            for &(_, bx, by, ti) in &building_assignments {
                if let Some(tmpl) = templates.get(ti) {
                    for row in 0..tmpl.height {
                        for col in 0..tmpl.width {
                            let tile_idx = tmpl.tiles[row as usize][col as usize];
                            if tile_idx == 0 {
                                continue;
                            }
                            let (r, g, b) = tile_colors
                                .and_then(|c| c.get(tile_idx as usize))
                                .copied()
                                .unwrap_or((180, 180, 180));
                            set_px(&mut pixels, bx + col, by + row, [r, g, b]);
                        }
                    }
                    // White dot at door.
                    if let Some(&(dc, dr)) = tmpl.entry_points.first() {
                        set_px(&mut pixels, bx + dc, by + dr, [255, 255, 255]);
                    }
                }
            }
            // Pad in yellow.
            for &(dx, dy, _) in &PAD_TILES {
                set_px(
                    &mut pixels,
                    (pad_cx as i32 + dx) as u32,
                    (pad_cy as i32 + dy) as u32,
                    [255, 220, 60],
                );
            }
            // Mechanic shop.
            if let Some(mech_colors) = bldg_style
                .map(|s| &s.mechanic_tile_colors)
                .filter(|c| !c.is_empty())
            {
                for row in 0..3u32 {
                    for col in 0..4u32 {
                        let atlas_row = 2 - row;
                        let idx = (atlas_row * 4 + col) as usize;
                        let (r, g, b) = mech_colors.get(idx).copied().unwrap_or((180, 180, 180));
                        set_px(&mut pixels, mech_x + col, mech_y + row, [r, g, b]);
                    }
                }
            }

            let mut img = Image::new(
                bevy::render::render_resource::Extent3d {
                    width: map_w,
                    height: map_h,
                    depth_or_array_layers: 1,
                },
                bevy::render::render_resource::TextureDimension::D2,
                pixels,
                bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
                bevy::asset::RenderAssetUsages::MAIN_WORLD
                    | bevy::asset::RenderAssetUsages::RENDER_WORLD,
            );
            img.sampler = bevy::image::ImageSampler::nearest();

            let building_info: Vec<(u32, u32, BuildingKind)> = building_assignments
                .iter()
                .map(|&(kind, bx, by, _)| (bx, by, kind))
                .collect();
            let minimap_handle = images.add(img);
            commands.insert_resource(SurfaceMiniMap {
                image: minimap_handle,
                map_w,
                map_h,
                buildings: building_info,
                pad_pos: (pad_cx, pad_cy),
            });
        }

        // ── Compute paths between buildings for AI characters ─────────
        // (door_positions and pathfinding_rects were built above for the
        // terrain constraint step — reuse them here.)
        {
            let cost_map = crate::surface_pathfinding::build_cost_map(
                &col_data,
                &terrain_flat,
                &movement_costs,
                &solid_building_tiles,
                map_w,
                map_h,
            );

            let surface_paths = crate::surface_pathfinding::compute_all_paths(
                &door_positions,
                &cost_map,
                map_w,
                map_h,
            );
            commands.insert_resource(surface_paths);

            // Store the cost map for runtime pathfinding (seek/flee/patrol).
            commands.insert_resource(crate::surface_pathfinding::SurfaceCostMap {
                data: cost_map,
                width: map_w,
                height: map_h,
            });
        }

        // ── Store footstep data for the walking sound system ─────────
        {
            let footstep_terrains: Vec<(String, f32)> = manifest
                .biomes
                .get(biome_name)
                .map(|b| {
                    b.terrains
                        .iter()
                        .map(|t| (t.footstep_surface.clone(), t.footstep_volume))
                        .collect()
                })
                .unwrap_or_default();
            commands.insert_resource(FootstepData {
                terrains: footstep_terrains,
                terrain_map: terrain_flat.clone(),
                map_w,
                map_h,
            });
        }

        // ── Setup civilian NPCs ─────────────────────────────────────────
        crate::surface_civilians::setup_civilians(
            &mut commands,
            &asset_server,
            &mut atlas_layouts,
            seed,
        );

        // ── Spawn landscape objects (plants, creatures, etc.) ─────────
        {
            let terrain_names: Vec<String> = manifest
                .biomes
                .get(biome_name)
                .map(|b| b.terrains.iter().map(|t| t.name.clone()).collect())
                .unwrap_or_default();
            crate::surface_objects::spawn_landscape_objects(
                &mut commands,
                &asset_server,
                &mut atlas_layouts,
                &terrain_flat,
                &terrain_names,
                biome_name,
                map_w,
                map_h,
                seed,
                &placed_buildings,
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
    let initial_index = crate::surface_character::sprite_index(
        crate::surface_character::Facing::Down,
        crate::surface_character::WalkFrame::Still,
    );

    commands.spawn((
        DespawnOnExit(PlayState::Exploring),
        Walker,
        CharacterAnim::default(),
        RigidBody::Dynamic,
        LockedAxes::ROTATION_LOCKED,
        Collider::circle(WALKER_RADIUS),
        CollisionLayers::new(GameLayer::Character, [GameLayer::Surface]),
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
        Transform::from_xyz(
            spawn_pos.x,
            spawn_pos.y,
            crate::surface_objects::depth_z(spawn_pos.y - 8.0),
        ),
    ));

    commands.insert_resource(walker_sheet);

    // "Press E" prompt is now shown via the comms ticker (no floating text).

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
    *nearby = NearbyBuilding::default();
    active_ui.0 = None;
    terrain_speed.0 = 1.0;
    commands.remove_resource::<SurfaceMiniMap>();
    commands.remove_resource::<crate::surface_pathfinding::SurfacePaths>();
    commands.remove_resource::<crate::surface_pathfinding::SurfaceCostMap>();
    commands.remove_resource::<ClearColor>();

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

// Walker animation is handled by the shared `animate_characters` system
// in surface_character.rs.  The walker just needs `CharacterAnim` +
// `LinearVelocity` + `Sprite` — same as civilians.

/// Play a footstep sound when the walker's animation frame advances.
/// Play door.ogg when the walker's depth crosses a door sprite's depth
/// (i.e. the walker visually passes in front of / behind the door).
/// Play ui_open/ui_close when ActiveBuildingUI changes, and ui_button
/// on mouse clicks while a building UI is open.
fn egui_button_click_sound(
    active_ui: Res<ActiveBuildingUI>,
    mut prev_ui: Local<Option<BuildingKind>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut sfx_writer: MessageWriter<crate::sfx::SurfaceSfx>,
) {
    let current = active_ui.0;

    // Detect open/close transitions.
    if active_ui.is_changed() {
        match (*prev_ui, current) {
            (None, Some(_)) => {
                sfx_writer.write(crate::sfx::SurfaceSfx::UiOpen);
            }
            (Some(_), None) => {
                sfx_writer.write(crate::sfx::SurfaceSfx::UiClose);
            }
            _ => {}
        }
        *prev_ui = current;
    }

    // Play button click sound on any mouse press while a UI is open.
    if current.is_some() && mouse.just_released(MouseButton::Left) {
        sfx_writer.write(crate::sfx::SurfaceSfx::UiButton);
    }
}

fn door_depth_sound(
    walker_q: Query<&Transform, With<Walker>>,
    mut doors: Query<(Entity, &Transform, &mut DoorSprite), Without<Walker>>,
    nearby: Res<NearbyBuilding>,
    mut sfx_writer: MessageWriter<crate::sfx::SurfaceSfx>,
) {
    let Ok(walker_tf) = walker_q.single() else { return };
    let walker_z = walker_tf.translation.z;
    let nearby_entity = nearby.current.map(|(e, _)| e);

    for (entity, door_tf, mut door) in &mut doors {
        let door_z = door_tf.translation.z;
        let walker_behind = walker_z > door_z; // higher z = behind

        if let Some(was_behind) = door.walker_was_behind {
            // Only play if depth flipped AND walker is colliding with this door.
            if was_behind != walker_behind && nearby_entity == Some(entity) {
                sfx_writer.write(crate::sfx::SurfaceSfx::Door);
            }
        }
        door.walker_was_behind = Some(walker_behind);
    }
}

fn play_footstep(
    walkers: Query<(&CharacterAnim, &Transform), With<Walker>>,
    footstep_data: Res<FootstepData>,
    mut sfx_writer: MessageWriter<crate::sfx::SurfaceSfx>,
) {
    let Ok((anim, tf)) = walkers.single() else { return };

    if !anim.is_moving || !anim.walk_timer.just_finished() {
        return;
    }
    if anim.walk_phase != 1 && anim.walk_phase != 3 {
        return;
    }
    if footstep_data.terrains.is_empty() || footstep_data.map_w == 0 {
        return;
    }

    let tile_px = TILE_PX;
    let tx = ((tf.translation.x / tile_px) + footstep_data.map_w as f32 / 2.0) as u32;
    let ty = ((tf.translation.y / tile_px) + footstep_data.map_h as f32 / 2.0) as u32;
    let tx = tx.min(footstep_data.map_w.saturating_sub(1));
    let ty = ty.min(footstep_data.map_h.saturating_sub(1));
    let idx = (ty * footstep_data.map_w + tx) as usize;
    let terrain_idx = footstep_data.terrain_map.get(idx).copied().unwrap_or(0) as usize;
    let (surface, volume) = footstep_data
        .terrains
        .get(terrain_idx)
        .map(|(s, v)| (s.clone(), *v))
        .unwrap_or(("dull".into(), 0.3));

    sfx_writer.write(crate::sfx::SurfaceSfx::Footstep { surface, volume });
}

// ---------------------------------------------------------------------------
// Building Interaction
// ---------------------------------------------------------------------------

/// Track which building the walker overlaps.  Uses an overlap count so
/// that exiting one of two adjacent sensor tiles (e.g. mechanic garage
/// doors) doesn't prematurely clear the state.
fn track_nearby_building(
    mut collision_starts: MessageReader<CollisionStart>,
    mut collision_ends: MessageReader<CollisionEnd>,
    buildings: Query<&Building>,
    walkers: Query<(), With<Walker>>,
    mut nearby: ResMut<NearbyBuilding>,
) {
    for event in collision_starts.read() {
        let (a, b) = (event.collider1, event.collider2);
        if let Some((bldg_entity, bldg)) = match (
            buildings.get(a).ok(),
            buildings.get(b).ok(),
            walkers.get(a).ok(),
            walkers.get(b).ok(),
        ) {
            (Some(bldg), _, _, Some(_)) => Some((a, bldg)),
            (_, Some(bldg), Some(_), _) => Some((b, bldg)),
            _ => None,
        } {
            nearby.current = Some((bldg_entity, bldg.kind));
            nearby.overlap_count += 1;
        }
    }
    for event in collision_ends.read() {
        let (a, b) = (event.collider1, event.collider2);
        let involves_building = buildings.contains(a) || buildings.contains(b);
        let involves_walker = walkers.contains(a) || walkers.contains(b);
        if involves_building && involves_walker {
            nearby.overlap_count = nearby.overlap_count.saturating_sub(1);
            if nearby.overlap_count == 0 {
                nearby.current = None;
            }
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
    session_data: Res<crate::session::SessionSaveData>,
) {
    if keyboard.just_pressed(KeyCode::Escape) {
        if active_ui.0.is_some() {
            active_ui.0 = None;
        } else {
            crate::game_save::write_save(&game_state, &session_data);
            next_state.set(PlayState::MainMenu);
        }
        return;
    }

    // Open building on E.
    if keyboard.just_pressed(KeyCode::KeyE) {
        if let Some((_, kind)) = nearby.current {
            active_ui.0 = Some(kind);
        }
    }
}

/// Show "Press E to enter X" in the comms ticker when near a building.
fn update_interact_prompt(
    nearby: Res<NearbyBuilding>,
    active_ui: Res<ActiveBuildingUI>,
    mut comms: ResMut<crate::hud::CommsChannel>,
) {
    if !nearby.is_changed() && !active_ui.is_changed() {
        return;
    }
    if let (Some((_, kind)), None) = (&nearby.current, &active_ui.0) {
        comms.send(format!("Press E to enter the {}", kind.label()));
    } else if nearby.current.is_none() {
        // Only clear if the message is a "Press E" prompt.
        if comms.message.starts_with("Press E") {
            comms.send("");
        }
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
    mut next_state: ResMut<NextState<PlayState>>,
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
                    ui.label("Your ship is docked here.");
                    ui.add_space(8.0);
                    if ui.button("Launch").clicked() {
                        active_ui.0 = None;
                        next_state.set(PlayState::Flying);
                    }
                }
                BuildingKind::MechanicShop => {
                    if let Ok(mut ship) = player_query.single_mut() {
                        let max_hp = ship.data.max_health;
                        let hp = ship.health;
                        ui.label(format!("Hull: {}/{}", hp, max_hp));

                        if hp < max_hp {
                            // Repair cost: (1 - health_frac) * 5% of ship price
                            let damage_frac = 1.0 - (hp as f64 / max_hp as f64);
                            let cost = (damage_frac * 0.05 * ship.data.price as f64).ceil() as i128;
                            let cost = cost.max(1);

                            ui.add_space(4.0);
                            ui.label(format!(
                                "Repair cost: {} credits (5% of ship value per full repair)",
                                cost
                            ));

                            let can_afford = ship.credits >= cost;
                            if ui
                                .add_enabled(can_afford, bevy_egui::egui::Button::new("Repair"))
                                .clicked()
                            {
                                ship.credits -= cost;
                                ship.health = max_hp;
                                    }
                            if !can_afford {
                                ui.colored_label(
                                    bevy_egui::egui::Color32::RED,
                                    "Not enough credits.",
                                );
                            }
                        } else {
                            ui.add_space(4.0);
                            ui.label("Hull is at full integrity. No repairs needed.");
                        }

                        ui.add_space(4.0);
                        ui.label(format!("Credits: {}", ship.credits));
                    }
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
