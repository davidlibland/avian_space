//! Walkable planet surface plugin.
//!
//! When the player lands on a planet, this module spawns a procedurally
//! generated tilemap with buildings (Bar, Shipyard, Outfitter, Mission
//! Control, Market) that the player can walk to and interact with.

use avian2d::prelude::*;
use bevy::prelude::*;
use bevy_ecs_tilemap::prelude::*;

use crate::PlayState;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Player walking speed (below the run-animation threshold, 80).
const WALK_SPEED: f32 = 70.0;
/// Player speed while holding Shift (plays the run cycle).
// Shift-run: quick enough to cross the colony (and run down fleeing
// targets) without trivialising the maze chases.
const RUN_SPEED: f32 = 165.0;
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
    FuelStation,
    /// The controlling faction's war office — present only on worlds whose
    /// effective faction takes sides. War missions are posted here; it flies
    /// a flag tinted with the faction's colors.
    Garrison,
    /// Maze venue: winding backtracker tunnels on mining worlds.
    Mine,
    /// Maze venue: container-canyon aisles at trade hubs and free ports.
    Warehouse,
    /// Maze venue: BSP service level under high-tech worlds.
    Substation,
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
            BuildingKind::FuelStation => "Fuel",
            BuildingKind::Garrison => "Garrison",
            BuildingKind::Mine => "Mine",
            BuildingKind::Warehouse => "Warehouse",
            BuildingKind::Substation => "Substation",
        }
    }

    #[allow(dead_code)] // superseded by the baked 3/4 sprites; kept for minimap/fallback use
    fn color(&self) -> Color {
        match self {
            BuildingKind::ShipPad => Color::srgb(0.5, 0.5, 0.7),
            BuildingKind::MechanicShop => Color::srgb(0.8, 0.5, 0.2),
            BuildingKind::Market => Color::srgb(0.9, 0.75, 0.2),
            BuildingKind::Outfitter => Color::srgb(0.7, 0.3, 0.3),
            BuildingKind::Shipyard => Color::srgb(0.3, 0.5, 0.8),
            BuildingKind::Bar => Color::srgb(0.8, 0.4, 0.1),
            BuildingKind::FuelStation => Color::srgb(0.3, 0.8, 0.9),
            BuildingKind::Garrison => Color::srgb(0.45, 0.55, 0.35),
            BuildingKind::Mine => Color::srgb(0.5, 0.4, 0.3),
            BuildingKind::Warehouse => Color::srgb(0.6, 0.55, 0.45),
            BuildingKind::Substation => Color::srgb(0.35, 0.5, 0.55),
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

use crate::surface_character::CharacterAnim;

/// Interaction zone for a building.
#[derive(Component)]
pub struct Building {
    pub kind: BuildingKind,
}

/// Marks a door sprite for depth-crossing sound detection.
/// `walker_was_behind` tracks whether the walker was behind (higher z)
/// the door last frame.
#[derive(Component)]
pub(crate) struct DoorSprite {
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
pub(crate) struct TerrainSpeedModifier(f32);

/// Per-terrain footstep data, indexed by terrain index.
#[derive(Resource, Default)]
pub(crate) struct FootstepData {
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

/// Run condition: the player is on foot — walking the surface OR inside
/// a building. The two states share almost every walking system; gate
/// them on this instead of repeating the two-state `.or()` chain.
pub(crate) fn on_foot(state: Res<State<crate::PlayState>>) -> bool {
    matches!(
        *state.get(),
        crate::PlayState::Exploring | crate::PlayState::Inside
    )
}

/// Spawn THE walker (player avatar) at a world position, scoped to the
/// given on-foot state. One bundle for both scenes — surface and interior
/// walkers must never drift apart in physics or animation setup.
pub(crate) fn spawn_walker_at(
    commands: &mut Commands,
    layers: &mut crate::character_compositor::CharacterLayers,
    images: &mut Assets<Image>,
    avatar: &crate::character_compositor::AvatarSpec,
    spawn_pos: Vec2,
    scope: crate::PlayState,
) -> Option<Entity> {
    let walker_image = layers.composite(avatar, images)?;
    let walker_layout = layers.layout.clone();
    let walker_anim = crate::surface_character::CharacterAnim::person(0.08);
    let initial_index = walker_anim.atlas_index();
    Some(
        commands
            .spawn((
                DespawnOnExit(scope),
                Walker,
                crate::surface_objects::FootOffset(crate::surface_objects::CHARACTER_FOOT_OFFSET),
                walker_anim,
                RigidBody::Dynamic,
                LockedAxes::ROTATION_LOCKED,
                crate::surface_objects::character_foot_collider(WALKER_RADIUS),
                CollisionLayers::new(crate::GameLayer::Character, [crate::GameLayer::Surface]),
                CollisionEventsEnabled,
                LinearDamping(WALKER_DAMPING),
                MaxLinearSpeed(RUN_SPEED),
                LinearVelocity(Vec2::ZERO),
                Sprite::from_atlas_image(
                    walker_image,
                    TextureAtlas {
                        layout: walker_layout,
                        index: initial_index,
                    },
                ),
                Transform::from_xyz(
                    spawn_pos.x,
                    spawn_pos.y,
                    crate::surface_objects::depth_z(
                        spawn_pos.y - crate::surface_objects::CHARACTER_FOOT_OFFSET,
                    ),
                ),
            ))
            .id(),
    )
}

/// Set one tile's pixel on a mini-map pixel buffer. Tile rows are
/// bottom-up (world y=0 = south edge); image rows are top-down — the
/// Y-flip lives HERE and nowhere else. (It has now bitten both the
/// surface and the interior builders independently.)
pub(crate) fn minimap_set_px(
    pixels: &mut [u8],
    map_w: u32,
    map_h: u32,
    x: u32,
    y: u32,
    color: [u8; 3],
) {
    if x < map_w && y < map_h {
        let iy = map_h - 1 - y;
        let pi = ((iy * map_w + x) * 4) as usize;
        pixels[pi] = color[0];
        pixels[pi + 1] = color[1];
        pixels[pi + 2] = color[2];
        pixels[pi + 3] = 255;
    }
}

/// Wrap a finished mini-map pixel buffer into the HUD image (nearest
/// sampling, main+render world) — the single image-construction path for
/// surface and interior maps alike.
pub(crate) fn minimap_image(pixels: Vec<u8>, map_w: u32, map_h: u32) -> Image {
    let mut img = Image::new(
        bevy::render::render_resource::Extent3d {
            width: map_w,
            height: map_h,
            depth_or_array_layers: 1,
        },
        bevy::render::render_resource::TextureDimension::D2,
        pixels,
        bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
        bevy::asset::RenderAssetUsages::MAIN_WORLD | bevy::asset::RenderAssetUsages::RENDER_WORLD,
    );
    img.sampler = bevy::image::ImageSampler::nearest();
    img
}

/// The generated mini-map image handle + map dimensions, used by the HUD.
#[derive(Resource)]
pub struct SurfaceMiniMap {
    pub image: Handle<Image>,
    pub map_w: u32,
    pub map_h: u32,
    /// (tile_x, tile_y, BuildingKind) for each placed building.
    #[allow(dead_code)] // populated for debug overlays/tooling
    pub buildings: Vec<(u32, u32, BuildingKind)>,
    /// Landing pad center.
    #[allow(dead_code)]
    pub pad_pos: (u32, u32),
}

/// Animated camera zoom target.
#[derive(Resource)]
pub(crate) struct CameraZoom {
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
        .init_resource::<interiors::MazeChase>()
        .init_resource::<interiors::CaptivesInTow>()
        .init_resource::<CameraZoom>()
        .init_resource::<NearbyBuilding>()
        .init_resource::<ActiveBuildingUI>()
        .init_resource::<TerrainSpeedModifier>()
        .init_resource::<FootstepData>()
        .init_resource::<crate::surface_npc_chat::NpcChatState>()
        .add_systems(
            OnEnter(PlayState::Exploring),
            (setup_surface, save_on_explore).chain(),
        )
        .add_systems(
            OnExit(PlayState::Exploring),
            (
                save_on_explore,
                teardown_surface,
                crate::surface_civilians::cleanup_civilians,
                crate::surface_fauna::cleanup_fauna,
            )
                .chain(),
        )
        .add_systems(OnEnter(PlayState::Inside), interiors::mark_interior_dirty)
        .add_systems(
            Update,
            interiors::setup_interior.run_if(
                in_state(PlayState::Inside).and(resource_exists::<interiors::InteriorDirty>),
            ),
        )
        .add_systems(
            Update,
            (
                walker_input,
                crate::surface_character::animate_characters,
                play_footstep,
                spawn_companion_avatars,
            )
                .run_if(on_foot),
        )
        .add_systems(
            Update,
            (
                track_nearby_building,
                track_terrain_speed,
                building_interact,
                update_interact_prompt,
                spawn_mission_npcs,
                npcs::update_building_offer_markers,
            )
                .run_if(in_state(PlayState::Exploring)),
        )
        .add_systems(
            Update,
            (
                interiors::interior_interact,
                interiors::interior_interact_prompt,
                interiors::spawn_interior_npcs,
            )
                .run_if(in_state(PlayState::Inside)),
        )
        .add_systems(
            Update,
            (
                interiors::maze_fugitive_arrivals,
                interiors::record_captives,
                interiors::maintain_captive_avatars,
            )
                .run_if(on_foot),
        )
        .add_systems(
            Update,
            interiors::deliver_captives_to_garrison.run_if(in_state(PlayState::Inside)),
        )
        .add_systems(
            OnEnter(crate::PlayState::Flying),
            interiors::process_captives_on_takeoff,
        )
        .add_systems(
            Update,
            (
                crate::surface_objects::update_shy_objects,
                crate::surface_objects::animate_landscape_objects,
                door_depth_sound,
                crate::surface_civilians::spawn_civilians,
            )
                .run_if(in_state(PlayState::Exploring)),
        )
        .add_systems(
            Update,
            (
                crate::surface_fauna::spawn_fauna,
                crate::surface_fauna::run_fauna,
                crate::surface_fauna::depth_sort_fauna,
                animate_building_doors,
            )
                .run_if(on_foot),
        )
        .add_systems(
            OnExit(PlayState::Inside),
            crate::surface_fauna::cleanup_fauna,
        )
        .add_systems(
            Update,
            (
                crate::surface_objects::depth_sort_walker,
                crate::surface_npc::run_npc_behaviors,
                crate::surface_npc_chat::npc_chat_interact,
                crate::surface_npc::update_npc_markers,
                crate::surface_civilians::depth_sort_npcs,
            )
                .run_if(on_foot),
        )
        .add_systems(
            bevy_egui::EguiPrimaryContextPass,
            (interiors::display_panel_ui, interiors::hire_panel_ui)
                .run_if(in_state(PlayState::Inside)),
        )
        .add_systems(
            bevy_egui::EguiPrimaryContextPass,
            crate::surface_npc_chat::npc_chat_ui.run_if(on_foot),
        )
        .add_systems(Update, animate_camera_zoom)
        .add_systems(
            bevy_egui::EguiPrimaryContextPass,
            surface_building_ui.run_if(on_foot),
        )
        .add_systems(Update, egui_button_click_sound.run_if(on_foot))
        .add_systems(FixedUpdate, camera_follow_walker.run_if(on_foot));
}

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------

fn save_on_explore(
    game_state: Res<crate::game_save::PlayerGameState>,
    session_data: Res<crate::session::SessionSaveData>,
) {
    crate::game_save::write_save(&game_state, &session_data);
}

// ── Submodules (split from the old 2,600-line surface.rs) ────────────────────
pub(crate) mod buildings;
mod interact;
pub mod interiors;
pub mod mazes;
mod npcs;
mod setup;
mod windows;

pub(crate) use buildings::*;
pub(crate) use interact::*;
pub(crate) use npcs::*;
pub(crate) use setup::*;
pub(crate) use windows::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::item_universe::ItemUniverse;

    fn universe() -> ItemUniverse {
        let mut iu: ItemUniverse =
            crate::item_universe::parse_dir(std::path::Path::new("assets")).unwrap();
        iu.finalize();
        iu
    }

    /// The garrison follows the LIVE controller: faction worlds have one,
    /// unaligned freeports don't — until somebody takes the system.
    #[test]
    fn garrison_stands_on_faction_worlds_only() {
        let iu = universe();
        let mut galaxy = crate::galaxy::GalaxyControl::seeded_from(&iu);

        let kinds = building_kinds_for_planet("earth", &iu, "sol", &galaxy);
        assert!(
            kinds.contains(&BuildingKind::Garrison),
            "Federation garrisons Earth"
        );

        let kinds = building_kinds_for_planet("marches_freeport", &iu, "the_marches", &galaxy);
        assert!(
            !kinds.contains(&BuildingKind::Garrison),
            "no flag flies over an unaligned freeport"
        );

        // Federation takes the Marches → the garrison (and its flag) appears.
        // (update_controllers does the recompute every frame in the live game.)
        galaxy.apply_shift("the_marches", "Federation", 1.0);
        galaxy.recompute_controller("the_marches");
        let kinds = building_kinds_for_planet("marches_freeport", &iu, "the_marches", &galaxy);
        assert!(
            kinds.contains(&BuildingKind::Garrison),
            "conquest raises a garrison"
        );
    }
}
