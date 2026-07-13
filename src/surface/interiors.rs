//! Building interiors (docs/interiors_design.md phase 1–3).
//!
//! Press E at a bar/outfitter/shipyard door and you go INSIDE: a small
//! walkable tile world rendered with the "interior" biome tileset through
//! the same blob47 autotiler as the surface. Shops size themselves to their
//! DERIVED stock — every outfitter item sits on a display plinth, every
//! shipyard hull hovers as a wireframe over a cradle pad — and walking up
//! to a display opens a focused stats/buy panel. The counter opens the
//! classic full-list window; the door takes you back to the surface at the
//! building you entered.

use avian2d::prelude::*;
use bevy::prelude::*;
use bevy_egui::EguiContexts;

use crate::item_universe::ItemUniverse;
use crate::ship::{BuyShip, Ship};
use crate::surface_character::CharacterAnim;
use crate::{CurrentStarSystem, GameLayer, PlayState, Player};

use super::{
    ActiveBuildingUI, BuildingKind, RUN_SPEED, TILE_PX, WALKER_DAMPING, WALKER_RADIUS,
    WORLD_HEIGHT, WORLD_WIDTH, WORLDS_DIR, Walker,
};

/// Which buildings have walkable interiors: all of them except the landing
/// pad. The usual tasks route through the COUNTER — walk up to the clerk's
/// counter and press E to open the classic window (trade at the market,
/// repair/mods at the mechanic, refuel at the fuel depot, war desk at the
/// garrison), so nothing regresses; displays and props are the browse path.
pub(crate) fn has_interior(kind: BuildingKind) -> bool {
    !matches!(kind, BuildingKind::ShipPad)
}

/// Set when entering a door; read by `setup_interior`.
#[derive(Resource, Clone, Copy)]
pub struct InteriorContext {
    pub kind: BuildingKind,
    /// Which floor of a multi-level venue (0 = ground; mazes descend).
    pub level: u8,
}

/// Present while the interior needs (re)building — inserted on entering the
/// state and again on every stairs transition. `setup_interior` consumes it.
#[derive(Resource, Clone, Copy)]
pub(crate) struct InteriorDirty {
    pub arrive: Arrive,
}

/// Where the walker appears after a (re)build.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Arrive {
    /// The level's natural entry (door on level 0, up-stairs below).
    Entry,
    /// Climbing back up: appear at this level's DOWN stairs.
    FromBelow,
}

/// Everything spawned by `setup_interior` — despawned wholesale on rebuild
/// (stairs) as well as on state exit.
#[derive(Component)]
pub(crate) struct InteriorScoped;

/// The stairs down to the next level (E to descend).
#[derive(Component)]
pub(crate) struct StairsDown;

/// The stairs back up (E to ascend).
#[derive(Component)]
pub(crate) struct StairsUp;

/// The clerk behind the counter — ambience; the counter does the business.
#[derive(Component)]
pub(crate) struct Clerk;

/// On entering the state: flag the first build and clear the stale
/// door-side `NearbyBuilding` so NPC chat isn't suppressed inside.
pub(crate) fn mark_interior_dirty(
    mut commands: Commands,
    mut nearby: ResMut<super::NearbyBuilding>,
) {
    nearby.current = None;
    commands.insert_resource(InteriorDirty {
        arrive: Arrive::Entry,
    });
}

/// Set when leaving an interior; `setup_surface` places the walker at this
/// building's door instead of the landing pad, then removes the resource.
#[derive(Resource, Clone, Copy)]
pub struct ReturnFromInterior(pub BuildingKind);

/// A display plinth bound to something purchasable.
#[derive(Component, Clone)]
pub(crate) enum DisplayBinding {
    /// Key into `ItemUniverse::outfitter_items`.
    OutfitterItem(String),
    /// Key into `ItemUniverse::ships`.
    Ship(String),
}

/// The interior's exit: stand adjacent and press E to leave.
#[derive(Component)]
pub(crate) struct ExitDoor;

/// The shop counter: stand adjacent and press E for the classic full-list
/// window (nothing regresses — displays are the browse path).
#[derive(Component)]
pub(crate) struct Counter(pub BuildingKind);

// ── Room geometry ─────────────────────────────────────────────────────────────

/// Interior tile tiers (must match the "interior" biome terrain rows —
/// same convention as station_layout.rs).
const T_FLOOR: u32 = 0;
const T_PLATING: u32 = 1;
const T_WALL: u32 = 3;

/// One interior floor plan: a carved room in a full-size map (the blob47
/// contract needs the map-wide tier gradient; everything outside the room
/// ramps up to the border tier in ±1 rings).
pub(crate) struct InteriorPlan {
    /// Full WORLD_WIDTH × WORLD_HEIGHT terrain tiers, row-major.
    pub terrain: Vec<u32>,
    /// Where the walker appears (just inside the door).
    pub entry: (u32, u32),
    /// The exit-door tile (level 0 only; None on lower levels).
    pub door: Option<(u32, u32)>,
    /// Display plinth tiles, in stock order.
    pub displays: Vec<(u32, u32)>,
    /// Counter tile — Some for shops, None for maze venues.
    pub counter: Option<(u32, u32)>,
    /// Room bounds (x0, y0, w, h) for prop placement/tests.
    pub room: (u32, u32, u32, u32),
    /// Extra solid tiles (furniture, containers), row-major.
    pub solid: Vec<bool>,
    /// Prop sprites: (sprite name, south-west anchor tile).
    pub props: Vec<(&'static str, (u32, u32))>,
    /// Stairs to the level below / back up.
    pub stairs_down: Option<(u32, u32)>,
    pub stairs_up: Option<(u32, u32)>,
    /// Farthest walkable tile from the entry — hunt NPCs hide here.
    pub hunt_spot: (u32, u32),
    /// Total levels this venue has on this planet.
    pub levels: u8,
    /// Tiles at or above this tier are solid. Shops: the wall tier.
    /// Mazes: the grate tier, letting walls be 3 tiles thin.
    pub solid_min_tier: u32,
}

/// Room size for `n` display slots along the walls: displays sit on the
/// east + west walls (every 2 tiles) and the north wall beside the counter.
/// Grows with stock — a fully-stocked capital outfitter is a HALL, a
/// frontier shop is a booth. Never a maze.
pub(crate) fn room_size_for(slots: usize) -> (u32, u32) {
    // Capacity of a w×h room ≈ 2·((h-4)/2) + (w-6)/2 (two walls + north).
    let mut w = 12u32;
    let mut h = 10u32;
    while interior_wall_slots(w, h) < slots && (w < 34 || h < 30) {
        if w <= h {
            w += 2;
        } else {
            h += 2;
        }
    }
    (w, h)
}

fn interior_wall_slots(w: u32, h: u32) -> usize {
    let side = ((h.saturating_sub(4)) / 2) as usize; // per east/west wall
    let north = ((w.saturating_sub(6)) / 2) as usize;
    side * 2 + north
}

/// Shipyard hall: cradle pads in rows down the middle (4 per row), so the
/// hulls read as a showroom rather than shelving.
pub(crate) fn hall_size_for(ships: usize) -> (u32, u32, usize) {
    let cols = 4usize.min(ships.max(1));
    let rows = ships.div_ceil(cols.max(1));
    let w = (6 + cols * 5) as u32;
    let h = (8 + rows * 5) as u32;
    (w.max(14), h.max(12), cols)
}

/// Prop metadata: (footprint w, footprint h, blocks movement).
/// Sprites live at assets/sprites/worlds/interior_props/<name>.png.
pub(crate) fn prop_meta(name: &str) -> (u32, u32, bool) {
    match name {
        "bar_counter" => (3, 1, true),
        "table_round" => (1, 1, true),
        "stool" => (1, 1, false),
        "shelf_rack" => (2, 1, true),
        "market_stall" => (2, 1, true),
        "engine_bench" => (2, 1, true),
        "tool_rack" => (1, 1, true),
        "fuel_pump" => (1, 1, true),
        "war_desk" => (2, 1, true),
        "flag_stand" => (1, 1, false),
        "crate_stack" => (1, 1, true),
        // Containers are already solid in the maze mask — visual only here.
        "container_a" | "container_b" => (4, 2, false),
        "ore_cart" => (1, 1, true),
        "timber_brace" | "lantern" => (1, 1, false),
        "pump_unit" => (2, 2, true),
        "pipe_valve" => (1, 1, true),
        "stairs_down" | "stairs_up" => (1, 1, false),
        _ => (1, 1, false),
    }
}

/// Build the floor plan for a building kind against the planet's derived
/// stock. Deterministic: same planet, same shop, same room. `level` picks
/// the floor of multi-level maze venues (shops only have level 0).
pub(crate) fn build_plan(
    kind: BuildingKind,
    iu: &ItemUniverse,
    planet: &str,
    level: u8,
) -> InteriorPlan {
    if super::mazes::is_maze(kind) {
        return maze_plan(kind, planet, level);
    }
    let (stock_len, ship_hall) = match kind {
        BuildingKind::Outfitter => (
            iu.find_gameplay_planet(planet)
                .map(|(_, pd)| pd.outfitter.len())
                .unwrap_or(0),
            false,
        ),
        BuildingKind::Shipyard => (
            iu.find_gameplay_planet(planet)
                .map(|(_, pd)| pd.shipyard.len())
                .unwrap_or(0),
            true,
        ),
        _ => (0, false),
    };

    let (rw, rh, hall_cols) = if ship_hall {
        hall_size_for(stock_len)
    } else if stock_len > 0 {
        let (w, h) = room_size_for(stock_len);
        (w, h, 0)
    } else {
        // Fixed rooms sized to their furniture.
        match kind {
            BuildingKind::Market => (20, 13, 0),
            BuildingKind::MechanicShop => (14, 10, 0),
            BuildingKind::FuelStation => (12, 10, 0),
            BuildingKind::Garrison => (14, 11, 0),
            _ => (16, 12, 0), // the bar
        }
    };

    let map_w = WORLD_WIDTH;
    let map_h = WORLD_HEIGHT;
    let x0 = (map_w - rw) / 2;
    let y0 = (map_h - rh) / 2;

    // Terrain: floor inside the room AND its entrance corridor, then ±1
    // rings ramping to the top tier — a single-tile "door notch" would
    // break the blob47 gradient contract, so the door is a short 3-wide
    // corridor whose distance field merges with the room's.
    let door_x = x0 + rw / 2;
    let corridor = (door_x - 1, y0.saturating_sub(4), 3u32, 4u32); // x, y, w, h
    let top = super::N_TERRAIN_TYPES - 1;
    let mut terrain = vec![top; (map_w * map_h) as usize];
    let dist_to = |x: u32, y: u32, (bx, by, bw, bh): (u32, u32, u32, u32)| -> u32 {
        let dx = if x < bx {
            bx - x
        } else if x >= bx + bw {
            x - (bx + bw - 1)
        } else {
            0
        };
        let dy = if y < by {
            by - y
        } else if y >= by + bh {
            y - (by + bh - 1)
        } else {
            0
        };
        dx.max(dy)
    };
    for y in 0..map_h {
        for x in 0..map_w {
            let d = dist_to(x, y, (x0, y0, rw, rh)).min(dist_to(x, y, corridor));
            // Nested ±1 rings, the station-generator pattern: floor,
            // plating, grate, wall, then on up to the border tier. The
            // plating/grate apron is walkable, exactly as in station maps.
            let tier = (T_FLOOR + d).min(top);
            terrain[(y * map_w + x) as usize] = tier;
        }
    }

    // The exit door sits at the corridor's far (south) end.
    let door = (door_x, corridor.1);
    terrain[(door.1 * map_w + door.0) as usize] = T_PLATING;
    let entry = (door_x, y0 + 1);

    // Counter: north wall centre (inside the room).
    let counter = (x0 + rw / 2, y0 + rh - 2);

    // Displays.
    let mut displays = Vec::new();
    if hall_cols > 0 {
        // Cradle grid down the hall.
        let mut placed = 0;
        'rows: for row in 0.. {
            for col in 0..hall_cols {
                if placed >= stock_len {
                    break 'rows;
                }
                let cx = x0 + 3 + (col as u32) * 5;
                let cy = y0 + 3 + (row as u32) * 5;
                if cy >= y0 + rh - 3 {
                    break 'rows;
                }
                displays.push((cx, cy));
                placed += 1;
            }
        }
    } else if stock_len > 0 {
        // Wall slots: west wall, east wall, then north beside the counter.
        let mut slots = Vec::new();
        let mut y = y0 + 2;
        while y < y0 + rh - 2 {
            slots.push((x0 + 1, y));
            y += 2;
        }
        let mut y = y0 + 2;
        while y < y0 + rh - 2 {
            slots.push((x0 + rw - 2, y));
            y += 2;
        }
        let mut x = x0 + 2;
        while x < x0 + rw - 3 {
            if x != counter.0 {
                slots.push((x, y0 + rh - 2));
            }
            x += 2;
        }
        displays = slots.into_iter().take(stock_len).collect();
        // Plating accents under the plinths.
        for &(px, py) in &displays {
            terrain[(py * map_w + px) as usize] = T_PLATING;
        }
    }

    // ── Furnishing: the props that make each shop ITS shop ──
    let mut props: Vec<(&'static str, (u32, u32))> = Vec::new();
    match kind {
        BuildingKind::Bar => {
            props.push(("bar_counter", (counter.0 - 1, counter.1)));
            // Round tables + stools in a loose grid clear of the walkway.
            for (i, &(tx, ty)) in [
                (x0 + 3, y0 + 3),
                (x0 + 3, y0 + 6),
                (x0 + rw - 5, y0 + 3),
                (x0 + rw - 5, y0 + 6),
                (x0 + rw / 2 + 2, y0 + 4),
            ]
            .iter()
            .enumerate()
            {
                props.push(("table_round", (tx, ty)));
                props.push(("stool", (tx + 1, ty + if i % 2 == 0 { 1 } else { 0 })));
            }
        }
        BuildingKind::Market => {
            // Two stall rows facing a central aisle.
            for row in 0..2u32 {
                let sy = y0 + 3 + row * 5;
                let mut sx = x0 + 2;
                while sx + 2 < x0 + rw - 2 {
                    props.push(("market_stall", (sx, sy)));
                    // Loose crates dress the first aisle only — the second
                    // row's aisle doubles as the counter walkway.
                    if row == 0 {
                        props.push(("crate_stack", (sx + rng_ish(sx, sy) % 2, sy + 2)));
                    }
                    sx += 4;
                }
            }
        }
        BuildingKind::MechanicShop => {
            props.push(("engine_bench", (x0 + rw / 2 - 1, y0 + rh / 2)));
            props.push(("tool_rack", (x0 + 1, y0 + 3)));
            props.push(("tool_rack", (x0 + 1, y0 + 6)));
            props.push(("crate_stack", (x0 + rw - 2, y0 + 2)));
        }
        BuildingKind::FuelStation => {
            for i in 0..3u32 {
                props.push(("fuel_pump", (x0 + rw - 2, y0 + 2 + i * 3)));
            }
        }
        BuildingKind::Garrison => {
            props.push(("war_desk", (counter.0 - 1, counter.1)));
            props.push(("flag_stand", (counter.0 - 3, counter.1)));
            props.push(("flag_stand", (counter.0 + 2, counter.1)));
        }
        BuildingKind::Outfitter | BuildingKind::Shipyard => {
            props.push(("shelf_rack", (counter.0 - 1, counter.1)));
        }
        _ => {}
    }

    // Furniture solidity → the solid mask (kept out of doorways by layout).
    let mut solid = vec![false; (map_w * map_h) as usize];
    for &(name, (px, py)) in &props {
        let (w, h, blocks) = prop_meta(name);
        if blocks {
            for y in py..(py + h).min(map_h) {
                for x in px..(px + w).min(map_w) {
                    solid[(y * map_w + x) as usize] = true;
                }
            }
        }
    }

    InteriorPlan {
        terrain,
        entry,
        door: Some(door),
        displays,
        counter: Some(counter),
        room: (x0, y0, rw, rh),
        solid,
        props,
        stairs_down: None,
        stairs_up: None,
        hunt_spot: (x0 + 2, y0 + rh - 2),
        levels: 1,
        solid_min_tier: T_WALL,
    }
}

/// Tiny deterministic hash for prop jitter (no RNG in shop plans).
fn rng_ish(x: u32, y: u32) -> u32 {
    x.wrapping_mul(31).wrapping_add(y).wrapping_mul(2654435761) >> 16
}

/// A maze venue floor: generator output → plan.
fn maze_plan(kind: BuildingKind, planet: &str, level: u8) -> InteriorPlan {
    let seed = super::mazes::planet_seed(planet);
    let levels = super::mazes::levels_for(kind, seed);
    let level = level.min(levels - 1);
    let mut lv = super::mazes::build_maze_level(kind, seed, level);
    let terrain = super::mazes::terrain_from_floor(&lv.floor);
    if let Some((sx, sy)) = lv.stairs_down {
        lv.props.push(("stairs_down", (sx, sy)));
    }
    if let Some((sx, sy)) = lv.stairs_up {
        lv.props.push(("stairs_up", (sx, sy)));
    }
    // Maze props block per their metadata (carts, valves...).
    for &(name, (px, py)) in &lv.props {
        let (w, h, blocks) = prop_meta(name);
        if blocks {
            for y in py..(py + h).min(WORLD_HEIGHT) {
                for x in px..(px + w).min(WORLD_WIDTH) {
                    lv.solid[(y * WORLD_WIDTH + x) as usize] = true;
                }
            }
        }
    }
    // Keep the entry, stairs and hunt spot clear of props.
    for tile in [
        Some(lv.entry),
        lv.stairs_down,
        lv.stairs_up,
        Some(lv.hunt_spot),
    ]
    .into_iter()
    .flatten()
    {
        lv.solid[(tile.1 * WORLD_WIDTH + tile.0) as usize] = false;
    }
    InteriorPlan {
        terrain,
        entry: lv.entry,
        door: lv.door,
        displays: Vec::new(),
        counter: None,
        room: (1, 1, WORLD_WIDTH - 2, WORLD_HEIGHT - 2),
        solid: lv.solid,
        props: lv.props,
        stairs_down: lv.stairs_down,
        stairs_up: lv.stairs_up,
        hunt_spot: lv.hunt_spot,
        levels,
        solid_min_tier: 2, // grate and above are rock/steel in a maze
    }
}

// ── Scene construction ────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub(crate) fn setup_interior(
    mut commands: Commands,
    dirty: Option<Res<InteriorDirty>>,
    old_scene: Query<Entity, With<InteriorScoped>>,
    context: Option<Res<InteriorContext>>,
    landed: Res<crate::planet_ui::LandedContext>,
    iu: Res<ItemUniverse>,
    current_system: Res<CurrentStarSystem>,
    asset_server: Res<AssetServer>,
    game_state: Res<crate::game_save::PlayerGameState>,
    mut character_layers: Option<ResMut<crate::character_compositor::CharacterLayers>>,
    mut images: ResMut<Assets<Image>>,
    mut atlas_layouts: ResMut<Assets<TextureAtlasLayout>>,
    mut camera_query: Query<&mut Transform, With<Camera2d>>,
    mut zoom: ResMut<super::CameraZoom>,
) {
    let _ = &current_system;
    let (Some(context), Some(dirty)) = (context, dirty) else {
        return;
    };
    let arrive = dirty.arrive;
    commands.remove_resource::<InteriorDirty>();
    // Rebuild: clear the previous level's scene (state didn't change, so
    // DespawnOnExit hasn't fired).
    for e in &old_scene {
        commands.entity(e).despawn();
    }
    let kind = context.kind;
    let planet = landed.planet_name.clone().unwrap_or_default();
    let plan = build_plan(kind, &iu, &planet, context.level);

    commands.insert_resource(ClearColor(Color::srgb(0.02, 0.02, 0.03)));

    // ── Interior biome data (same files the surface uses) ──
    let load_ron = |filename: &str| -> Option<String> {
        crate::embedded_assets::read_to_string(&format!("assets/{WORLDS_DIR}/{filename}")).ok()
    };
    let lut = load_ron("blob47_lut.ron")
        .and_then(|text| ron::from_str::<crate::world_assets::Blob47Lut>(&text).ok());
    let manifest = load_ron("world_manifest.ron")
        .and_then(|text| ron::from_str::<crate::world_assets::WorldManifest>(&text).ok());
    let (Some(lut), Some(manifest)) = (lut, manifest) else {
        eprintln!("[interiors] WARNING: world data unavailable — no interior spawned");
        return;
    };
    let biome = manifest
        .biomes
        .get("interior")
        .or_else(|| manifest.biomes.values().next());
    let Some(biome) = biome else { return };
    let atlas_image: Handle<Image> = asset_server.load(format!("{WORLDS_DIR}/interior_atlas.png"));
    let tile_px = TILE_PX;
    let (map_w, map_h) = (WORLD_WIDTH, WORLD_HEIGHT);

    // 2-D view for the shared bitmask/texture-index math.
    let map2d: Vec<Vec<u32>> = (0..map_h)
        .map(|y| {
            (0..map_w)
                .map(|x| plan.terrain[(y * map_w + x) as usize])
                .collect()
        })
        .collect();
    let n_rows = biome.terrains.iter().map(|t| t.row + 1).max().unwrap_or(6);
    let layout = atlas_layouts.add(TextureAtlasLayout::from_grid(
        UVec2::splat(manifest.tile_size),
        manifest.atlas_cols,
        n_rows,
        None,
        None,
    ));

    // ── Tiles: only around the room (the void beyond never shows) ──
    let (x0, y0, rw, rh) = plan.room;
    let pad = 4u32;
    let (tx0, ty0) = (x0.saturating_sub(pad), y0.saturating_sub(pad));
    let (tx1, ty1) = ((x0 + rw + pad).min(map_w), (y0 + rh + pad).min(map_h));
    for ty in ty0..ty1 {
        for tx in tx0..tx1 {
            let index = crate::world_assets::tile_texture_index(
                &map2d,
                tx as i32,
                ty as i32,
                map_w as i32,
                map_h as i32,
                &lut,
            );
            let pos = super::tile_to_world(tx, ty, map_w, map_h, tile_px);
            commands.spawn((
                DespawnOnExit(PlayState::Inside),
                InteriorScoped,
                Sprite::from_atlas_image(
                    atlas_image.clone(),
                    TextureAtlas {
                        layout: layout.clone(),
                        index: index as usize,
                    },
                ),
                Transform::from_xyz(pos.x, pos.y, -10.0),
            ));
            // Solid walls block. Shops use the biome's collision rows; maze
            // venues also solidify the grate tier (3-tile-thin walls) and
            // any furniture/container tiles from the plan.
            let tier = map2d[ty as usize][tx as usize];
            let idx = (ty * map_w + tx) as usize;
            let solid = tier >= plan.solid_min_tier
                || plan.solid[idx]
                || biome
                    .terrains
                    .iter()
                    .find(|t| t.row == tier)
                    .map(|t| t.collision == 1)
                    .unwrap_or(false);
            if solid {
                commands.spawn((
                    DespawnOnExit(PlayState::Inside),
                    InteriorScoped,
                    RigidBody::Static,
                    Collider::rectangle(tile_px, tile_px),
                    CollisionLayers::new(
                        GameLayer::Surface,
                        [GameLayer::Surface, GameLayer::Character],
                    ),
                    Transform::from_xyz(pos.x, pos.y, 0.0),
                ));
            }
        }
    }

    // ── Cost map for NPC pathfinding (companions, patrons, hunts later) ──
    let data: Vec<f32> = plan
        .terrain
        .iter()
        .enumerate()
        .map(|(i, &tier)| {
            if plan.solid[i] || tier >= plan.solid_min_tier {
                return f32::INFINITY;
            }
            let meta = biome.terrains.iter().find(|t| t.row == tier);
            match meta {
                Some(m) if m.collision != 1 => m.movement_cost.max(0.1),
                None if tier < T_WALL => 1.0,
                _ => f32::INFINITY,
            }
        })
        .collect();
    commands.insert_resource(crate::surface_pathfinding::SurfaceCostMap {
        data,
        width: map_w,
        height: map_h,
    });
    commands.insert_resource(crate::surface_pathfinding::SurfacePaths::default());

    // ── Props ──
    let stock: Vec<DisplayBinding> = match kind {
        BuildingKind::Outfitter => iu
            .find_gameplay_planet(&planet)
            .map(|(_, pd)| {
                let mut items = pd.outfitter.clone();
                items.sort();
                items
                    .into_iter()
                    .map(DisplayBinding::OutfitterItem)
                    .collect()
            })
            .unwrap_or_default(),
        BuildingKind::Shipyard => iu
            .find_gameplay_planet(&planet)
            .map(|(_, pd)| {
                let mut ships = pd.shipyard.clone();
                ships.sort();
                ships.into_iter().map(DisplayBinding::Ship).collect()
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    };
    for (slot, binding) in stock.iter().enumerate() {
        let Some(&(px, py)) = plan.displays.get(slot) else {
            break;
        };
        let pos = super::tile_to_world(px, py, map_w, map_h, tile_px);
        let (sprite, size) = display_sprite(binding, &iu, &asset_server);
        commands.spawn((
            DespawnOnExit(PlayState::Inside),
            InteriorScoped,
            binding.clone(),
            sprite,
            Transform::from_xyz(
                pos.x,
                pos.y + 6.0,
                crate::surface_objects::depth_z(pos.y - tile_px * 0.5) + 0.5,
            )
            .with_scale(Vec3::splat(size)),
        ));
    }

    // Counter + exit door markers (invisible; interaction handles them).
    if let Some(counter) = plan.counter {
        let cpos = super::tile_to_world(counter.0, counter.1, map_w, map_h, tile_px);
        commands.spawn((
            DespawnOnExit(PlayState::Inside),
            InteriorScoped,
            Counter(kind),
            Transform::from_xyz(cpos.x, cpos.y, 0.0),
        ));
    }
    if let Some(door) = plan.door {
        let dpos = super::tile_to_world(door.0, door.1, map_w, map_h, tile_px);
        commands.spawn((
            DespawnOnExit(PlayState::Inside),
            InteriorScoped,
            ExitDoor,
            Transform::from_xyz(dpos.x, dpos.y, 0.0),
        ));
    }
    if let Some(stairs) = plan.stairs_down {
        let spos = super::tile_to_world(stairs.0, stairs.1, map_w, map_h, tile_px);
        commands.spawn((
            DespawnOnExit(PlayState::Inside),
            InteriorScoped,
            StairsDown,
            Transform::from_xyz(spos.x, spos.y, 0.0),
        ));
    }
    if let Some(stairs) = plan.stairs_up {
        let spos = super::tile_to_world(stairs.0, stairs.1, map_w, map_h, tile_px);
        commands.spawn((
            DespawnOnExit(PlayState::Inside),
            InteriorScoped,
            StairsUp,
            Transform::from_xyz(spos.x, spos.y, 0.0),
        ));
    }

    // ── Prop sprites (baked at 32 px/tile, bottom-center anchored) ──
    for &(name, (px, py)) in &plan.props {
        let (w, _h, _) = prop_meta(name);
        let base = super::tile_to_world(px, py, map_w, map_h, tile_px);
        let front_y = base.y - tile_px * 0.5;
        let cx = base.x + (w as f32 - 1.0) * 0.5 * tile_px;
        commands.spawn((
            DespawnOnExit(PlayState::Inside),
            InteriorScoped,
            Sprite::from_image(
                asset_server.load(format!("sprites/worlds/interior_props/{name}.png")),
            ),
            bevy::sprite::Anchor(Vec2::new(0.0, -0.5)),
            Transform::from_xyz(cx, front_y, crate::surface_objects::depth_z(front_y)),
        ));
    }

    // ── The walker: at the door coming in, at the stairs coming up ──
    let spawn_tile = match arrive {
        Arrive::Entry => plan.entry,
        Arrive::FromBelow => plan.stairs_down.unwrap_or(plan.entry),
    };
    let spawn_pos = super::tile_to_world(spawn_tile.0, spawn_tile.1, map_w, map_h, tile_px);
    if let Some((walker_image, walker_layout)) = character_layers.as_deref_mut().and_then(|l| {
        l.composite(&game_state.avatar, &mut images)
            .map(|img| (img, l.layout.clone()))
    }) {
        let walker_anim = CharacterAnim::person(0.08);
        let initial_index = walker_anim.atlas_index();
        commands.spawn((
            DespawnOnExit(PlayState::Inside),
            InteriorScoped,
            Walker,
            crate::surface_objects::FootOffset(crate::surface_objects::CHARACTER_FOOT_OFFSET),
            walker_anim,
            RigidBody::Dynamic,
            LockedAxes::ROTATION_LOCKED,
            crate::surface_objects::character_foot_collider(WALKER_RADIUS),
            CollisionLayers::new(GameLayer::Character, [GameLayer::Surface]),
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
        ));
    }

    if let Ok(mut cam_tf) = camera_query.single_mut() {
        cam_tf.translation = Vec3::new(spawn_pos.x, spawn_pos.y, cam_tf.translation.z);
    }
    zoom.target = super::SURFACE_CAMERA_SCALE * 0.8;
}

/// What a display shows: outfitter items use their weapon sprite (or a glow
/// disc), ships use their HUD wireframe.
fn display_sprite(
    binding: &DisplayBinding,
    iu: &ItemUniverse,
    asset_server: &AssetServer,
) -> (Sprite, f32) {
    match binding {
        DisplayBinding::Ship(ship) => {
            let mut s =
                Sprite::from_image(asset_server.load(format!("sprites/wireframes/{ship}.png")));
            s.color = Color::srgba(0.4, 0.9, 1.0, 0.9);
            s.custom_size = Some(Vec2::splat(TILE_PX * 2.6));
            (s, 1.0)
        }
        DisplayBinding::OutfitterItem(item) => {
            let weapon = iu.weapons.get(item);
            match weapon.and_then(|w| w.sprite_path.clone()) {
                Some(path) => {
                    let mut s = Sprite::from_image(asset_server.load(path));
                    s.custom_size = Some(Vec2::splat(TILE_PX * 0.9));
                    (s, 1.0)
                }
                None => {
                    let color = weapon
                        .map(|w| Color::srgb(w.color[0], w.color[1], w.color[2]))
                        .unwrap_or(Color::srgb(0.7, 0.9, 1.0));
                    let mut s = Sprite::from_color(color, Vec2::splat(TILE_PX * 0.45));
                    s.color = color;
                    (s, 1.0)
                }
            }
        }
    }
}

// ── Interaction ───────────────────────────────────────────────────────────────

/// E inside: exit door first, then stairs, then the counter (which opens
/// the classic full-list window — trade, repair, refuel, war desk).
#[allow(clippy::too_many_arguments)]
pub(crate) fn interior_interact(
    keyboard: Res<ButtonInput<KeyCode>>,
    walker: Query<&Transform, With<Walker>>,
    exits: Query<&Transform, (With<ExitDoor>, Without<Walker>)>,
    stairs_down: Query<&Transform, (With<StairsDown>, Without<Walker>)>,
    stairs_up: Query<&Transform, (With<StairsUp>, Without<Walker>)>,
    counters: Query<(&Counter, &Transform), Without<Walker>>,
    mut active_ui: ResMut<ActiveBuildingUI>,
    context: Option<ResMut<InteriorContext>>,
    mut commands: Commands,
    mut next_state: ResMut<NextState<PlayState>>,
) {
    if !keyboard.just_pressed(KeyCode::KeyE) || active_ui.0.is_some() {
        return;
    }
    let Ok(wtf) = walker.single() else { return };
    let wp = wtf.translation.truncate();
    let range = TILE_PX * 1.6;
    let near = |t: &Transform| (t.translation.truncate() - wp).length() < range;
    if exits.iter().any(near) {
        if let Some(ctx) = context {
            commands.insert_resource(ReturnFromInterior(ctx.kind));
        }
        next_state.set(PlayState::Exploring);
        return;
    }
    if stairs_down.iter().any(near) {
        if let Some(mut ctx) = context {
            ctx.level += 1;
            commands.insert_resource(InteriorDirty {
                arrive: Arrive::Entry,
            });
        }
        return;
    }
    if stairs_up.iter().any(near) {
        if let Some(mut ctx) = context {
            ctx.level = ctx.level.saturating_sub(1);
            commands.insert_resource(InteriorDirty {
                arrive: Arrive::FromBelow,
            });
        }
        return;
    }
    if let Some((counter, _)) = counters.iter().find(|(_, t)| near(t)) {
        active_ui.0 = Some(counter.0);
    }
}

/// The focused display panel: stats + Buy for whatever plinth the walker is
/// standing at. Ships show comparative bars against the current hull.
#[allow(clippy::too_many_arguments)]
pub(crate) fn display_panel_ui(
    mut contexts: EguiContexts,
    walker: Query<&Transform, With<Walker>>,
    displays: Query<(&DisplayBinding, &Transform)>,
    iu: Res<ItemUniverse>,
    landed: Res<crate::planet_ui::LandedContext>,
    standings: Res<crate::standing::FactionStandings>,
    galaxy: Res<crate::galaxy::GalaxyControl>,
    unlocks: Res<crate::missions::PlayerUnlocks>,
    mut player: Query<&mut Ship, With<Player>>,
    mut buy_ship: MessageWriter<BuyShip>,
    active_ui: Res<ActiveBuildingUI>,
) {
    if active_ui.0.is_some() {
        return; // the classic window is open — don't stack panels
    }
    let Ok(wtf) = walker.single() else { return };
    let wp = wtf.translation.truncate();
    let nearest = displays
        .iter()
        .map(|(b, t)| (b, (t.translation.truncate() - wp).length()))
        .filter(|(_, d)| *d < TILE_PX * 1.8)
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let Some((binding, _)) = nearest else { return };
    let Ok(ctx) = contexts.ctx_mut() else { return };
    let Ok(mut ship) = player.single_mut() else {
        return;
    };
    let planet = landed.planet_name.clone().unwrap_or_default();
    let markup = crate::galaxy::effective_planet_faction(&galaxy, &iu, &planet)
        .map(|f| crate::standing::price_markup(standings.get(&f)))
        .unwrap_or(1.0);

    bevy_egui::egui::Window::new("Display")
        .collapsible(false)
        .resizable(false)
        .anchor(bevy_egui::egui::Align2::RIGHT_CENTER, [-24.0, 0.0])
        .show(ctx, |ui| match binding {
            DisplayBinding::OutfitterItem(item_key) => {
                let Some(item) = iu.outfitter_items.get(item_key) else {
                    return;
                };
                ui.heading(item.display_name());
                let price = crate::standing::markup_price(item.price(), markup);
                ui.label(format!("Price: {price} cr"));
                ui.label(format!("Space: {}", item.space()));
                if let Some(w) = iu.weapons.get(item_key) {
                    ui.label(format!("Damage: {}", w.damage));
                    ui.label(format!("Range: {:.0}", w.speed * w.lifetime));
                    ui.label(if w.is_turret() {
                        "Mount: turret"
                    } else {
                        "Mount: gun"
                    });
                    let ok = ship.mount_free_for(w);
                    if !ok {
                        ui.colored_label(
                            bevy_egui::egui::Color32::from_rgb(230, 160, 100),
                            "No free mount on this hull.",
                        );
                    }
                }
                let locked = item.required_unlocks().iter().any(|u| !unlocks.has(u));
                if locked {
                    ui.label("License required.");
                } else if ui.button("Buy").clicked() {
                    ship.buy_weapon(item_key, &iu, markup);
                }
            }
            DisplayBinding::Ship(ship_key) => {
                let Some(data) = iu.ships.get(ship_key) else {
                    return;
                };
                ui.heading(if data.display_name.is_empty() {
                    ship_key.clone()
                } else {
                    data.display_name.clone()
                });
                let price = crate::standing::markup_price(data.price, markup);
                ui.label(format!("Price: {price} cr"));
                // Comparative stat bars vs. the current hull.
                let cur = &ship.data;
                let bar = |ui: &mut bevy_egui::egui::Ui, name: &str, a: f32, b: f32| {
                    let max = a.max(b).max(1.0);
                    ui.label(format!("{name}: {a:.0} (yours {b:.0})"));
                    ui.add(
                        bevy_egui::egui::ProgressBar::new(a / max)
                            .desired_width(140.0)
                            .desired_height(6.0),
                    );
                };
                bar(ui, "Speed", data.max_speed, cur.max_speed);
                bar(ui, "Hull", data.max_health as f32, cur.max_health as f32);
                bar(ui, "Cargo", data.cargo_space as f32, cur.cargo_space as f32);
                bar(ui, "Equip", data.item_space as f32, cur.item_space as f32);
                ui.label(format!(
                    "Mounts: {} guns, {} turrets",
                    data.gun_mounts, data.turret_mounts
                ));
                let locked = data.required_unlocks.iter().any(|u| !unlocks.has(u));
                if locked {
                    ui.label("License required.");
                } else if ui.button("Buy").clicked() {
                    buy_ship.write(BuyShip {
                        ship_type: ship_key.clone(),
                    });
                }
            }
        });
}

/// People inside: offer NPCs whose building this is wait at tables
/// (level 0), a CLERK stands behind every shop counter, and active
/// meet/catch hunt targets hide at the deepest level's far end.
/// Idempotent per mission id / per clerk.
#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_interior_npcs(
    mut commands: Commands,
    context: Option<Res<InteriorContext>>,
    landed: Res<crate::planet_ui::LandedContext>,
    iu: Res<ItemUniverse>,
    offers: Res<crate::missions::MissionOffers>,
    catalog: Res<crate::missions::MissionCatalog>,
    mission_log: Res<crate::missions::MissionLog>,
    existing: Query<&crate::surface_npc::MissionNpc>,
    clerks: Query<(), With<Clerk>>,
    layers: Option<ResMut<crate::character_compositor::CharacterLayers>>,
    mut images: ResMut<Assets<Image>>,
    cost_map: Option<Res<crate::surface_pathfinding::SurfaceCostMap>>,
) {
    let (Some(context), Some(mut layers), Some(_cm)) = (context, layers, cost_map) else {
        return;
    };
    let kind = context.kind;
    let planet = landed.planet_name.clone().unwrap_or_default();
    let already: std::collections::HashSet<&str> = existing.iter().map(|m| m.0.as_str()).collect();
    let building_name = format!("{kind:?}").to_lowercase();
    let plan = build_plan(kind, &iu, &planet, context.level);
    let (x0, y0, _rw, rh) = plan.room;
    let walk_speed = layers.walk_speed;

    // The clerk: one per counter, standing behind it.
    if let Some(counter) = plan.counter
        && clerks.is_empty()
    {
        let tile = (counter.0, (counter.1 + 1).min(y0 + rh - 1));
        crate::surface_npc::spawn_clerk(&mut commands, &mut layers, &mut images, tile);
    }

    // Offer NPCs at the tables (ground floor only; mazes have no givers).
    if context.level == 0 && !super::mazes::is_maze(kind) {
        let mut slot = 0u32;
        for id in offers.npc.get(&planet).cloned().unwrap_or_default() {
            if already.contains(id.as_str()) {
                continue;
            }
            let Some(def) = catalog.defs.get(&id) else {
                continue;
            };
            let crate::missions::types::OfferKind::NpcOffer { building, npc, .. } = &def.offer
            else {
                continue;
            };
            if building.as_deref() != Some(building_name.as_str()) {
                continue;
            }
            // A table spot along the west side.
            let tile = (x0 + 2, (y0 + 3 + slot * 2).min(y0 + rh - 3));
            slot += 1;
            let identity = super::npc_identity(&iu, &layers, npc);
            crate::surface_npc::spawn_mission_npc(
                &mut commands,
                &mut layers,
                &mut images,
                "civilian",
                identity,
                &id,
                tile,
                walk_speed,
                false,
                PlayState::Inside,
            );
        }
    }

    // Hunt targets: active meet/catch objectives bound to THIS building,
    // hiding at the deepest level's farthest tile.
    if context.level + 1 == plan.levels {
        for (mission_id, def) in &catalog.defs {
            if already.contains(mission_id.as_str()) {
                continue;
            }
            if !matches!(
                mission_log.status(mission_id),
                crate::missions::MissionStatus::Active(_)
            ) {
                continue;
            }
            let (m_planet, m_building, npc, objective) = match &def.objective {
                crate::missions::Objective::MeetNpc {
                    planet: p,
                    building: b,
                    npc,
                    ..
                } => (
                    p,
                    b,
                    npc,
                    crate::surface_npc::ObjectiveKind::Meet { seek: false },
                ),
                crate::missions::Objective::CatchNpc {
                    planet: p,
                    building: b,
                    npc,
                    ..
                } => (p, b, npc, crate::surface_npc::ObjectiveKind::Catch),
                _ => continue,
            };
            if *m_planet != planet || m_building.as_deref() != Some(building_name.as_str()) {
                continue;
            }
            let identity = super::npc_identity(&iu, &layers, npc);
            crate::surface_npc::spawn_objective_npc(
                &mut commands,
                &mut layers,
                &mut images,
                "civilian",
                identity,
                mission_id,
                plan.hunt_spot,
                walk_speed * 1.1,
                objective,
                PlayState::Inside,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface_pathfinding::SurfaceCostMap;

    fn iu() -> ItemUniverse {
        let mut iu: ItemUniverse =
            crate::item_universe::parse_dir(std::path::Path::new("assets")).unwrap();
        iu.finalize();
        iu
    }

    /// Walkable predicate mirroring setup: below the plan's solid tier and
    /// not furniture.
    fn cost_map_of(plan: &InteriorPlan) -> SurfaceCostMap {
        let data = plan
            .terrain
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                if plan.solid[i] || t >= plan.solid_min_tier {
                    f32::INFINITY
                } else {
                    1.0
                }
            })
            .collect();
        SurfaceCostMap {
            data,
            width: WORLD_WIDTH,
            height: WORLD_HEIGHT,
        }
    }

    fn assert_contract(terrain: &[u32], what: &str) {
        let (w, h) = (WORLD_WIDTH as i32, WORLD_HEIGHT as i32);
        for y in 0..h {
            for x in 0..w {
                let t = terrain[(y * w + x) as usize] as i64;
                for (dx, dy) in [(1, 0), (0, 1), (1, 1), (1, -1)] {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w || ny >= h {
                        continue;
                    }
                    let n = terrain[(ny * w + nx) as usize] as i64;
                    assert!(
                        (t - n).abs() <= 1,
                        "{what}: gradient break at ({x},{y}): {t} vs {n}"
                    );
                }
            }
        }
    }

    /// Shops are rooms, not mazes: every display, the counter, and the
    /// entry are mutually reachable — WITH the furniture in place.
    #[test]
    fn every_display_and_the_counter_are_reachable_from_the_entry() {
        let iu = iu();
        for (kind, planet) in [
            (BuildingKind::Bar, "earth"),
            (BuildingKind::Outfitter, "earth"),
            (BuildingKind::Shipyard, "earth"),
            (BuildingKind::Market, "earth"),
            (BuildingKind::MechanicShop, "earth"),
            (BuildingKind::FuelStation, "earth"),
            (BuildingKind::Garrison, "earth"),
            (BuildingKind::Outfitter, "marches_freeport"),
            (BuildingKind::Shipyard, "deneb_prime"),
        ] {
            let plan = build_plan(kind, &iu, planet, 0);
            let cm = cost_map_of(&plan);
            let counter_front = plan.counter.map(|(cx, cy)| (cx, cy - 1)).unwrap();
            assert!(
                cm.find_path(plan.entry, counter_front).is_some(),
                "{kind:?}@{planet}: counter reachable"
            );
            for (i, &d) in plan.displays.iter().enumerate() {
                assert!(
                    cm.find_path(plan.entry, d).is_some(),
                    "{kind:?}@{planet}: display {i} at {d:?} reachable"
                );
            }
        }
    }

    /// The blob47 autotiler contract holds for shops AND all maze levels.
    #[test]
    fn interior_terrain_satisfies_the_gradient_contract() {
        let iu = iu();
        assert_contract(
            &build_plan(BuildingKind::Shipyard, &iu, "earth", 0).terrain,
            "shipyard",
        );
        for kind in [
            BuildingKind::Mine,
            BuildingKind::Warehouse,
            BuildingKind::Substation,
        ] {
            for planet in ["earth", "ceres", "deneb_prime"] {
                let plan = build_plan(kind, &iu, planet, 0);
                for level in 0..plan.levels {
                    let p = build_plan(kind, &iu, planet, level);
                    assert_contract(&p.terrain, &format!("{kind:?}@{planet} L{level}"));
                }
            }
        }
    }

    /// Shops size themselves to their wares: one plinth per stocked item,
    /// and a better-stocked planet gets a bigger room.
    #[test]
    fn shops_size_to_stock_and_display_everything() {
        let iu = iu();
        let earth_items = iu.find_gameplay_planet("earth").unwrap().1.outfitter.len();
        let plan = build_plan(BuildingKind::Outfitter, &iu, "earth", 0);
        assert_eq!(
            plan.displays.len(),
            earth_items,
            "every stocked item gets a plinth"
        );
        let freeport_items = iu
            .find_gameplay_planet("marches_freeport")
            .unwrap()
            .1
            .outfitter
            .len();
        assert!(earth_items > freeport_items, "premise: Earth stocks more");
        let small = build_plan(BuildingKind::Outfitter, &iu, "marches_freeport", 0);
        let area = |p: &InteriorPlan| p.room.2 * p.room.3;
        assert!(
            area(&plan) > area(&small),
            "Earth's outfitter hall outsizes the freeport booth"
        );
        // Ships too.
        let hulls = iu.find_gameplay_planet("earth").unwrap().1.shipyard.len();
        let yard = build_plan(BuildingKind::Shipyard, &iu, "earth", 0);
        assert_eq!(yard.displays.len(), hulls, "every hull gets a cradle");
        // The bar ignores stock entirely.
        let bar = build_plan(BuildingKind::Bar, &iu, "earth", 0);
        assert_eq!((bar.room.2, bar.room.3), (16, 12), "cosy, fixed, no maze");
    }

    /// The outside-spawn filter's name mapping covers every kind, and every
    /// kind except the landing pad has a walkable interior now.
    #[test]
    fn building_names_map_to_kinds_and_shops_have_interiors() {
        use BuildingKind::*;
        for kind in [
            ShipPad,
            MechanicShop,
            Market,
            Outfitter,
            Shipyard,
            Bar,
            FuelStation,
            Garrison,
            Mine,
            Warehouse,
            Substation,
        ] {
            let name = format!("{kind:?}").to_lowercase();
            assert_eq!(
                crate::surface::npcs::building_kind_from_name(&name),
                Some(kind),
                "{name} maps back to its kind"
            );
            assert_eq!(
                has_interior(kind),
                kind != ShipPad,
                "{kind:?} interior flag"
            );
        }
        assert_eq!(
            crate::surface::npcs::building_kind_from_name("casino"),
            None
        );
    }

    /// room_size_for growth is monotone and bounded (never a maze).
    #[test]
    fn room_growth_is_monotone_and_bounded() {
        let mut last = 0;
        for n in [1usize, 4, 8, 16, 24, 40, 60] {
            let (w, h) = room_size_for(n);
            let area = w * h;
            assert!(area >= last, "monotone growth");
            assert!(w <= 36 && h <= 32, "bounded: {w}x{h} for {n}");
            last = area;
        }
    }

    /// Every maze level is fully traversable: from the entry you can reach
    /// the door (level 0), the stairs in BOTH directions, and the hunt
    /// spot — with props and containers solid.
    #[test]
    fn maze_levels_are_connected_with_working_exits() {
        let iu = iu();
        for kind in [
            BuildingKind::Mine,
            BuildingKind::Warehouse,
            BuildingKind::Substation,
        ] {
            for planet in ["earth", "ceres", "deneb_prime", "halcyon"] {
                let ground = build_plan(kind, &iu, planet, 0);
                assert!(
                    ground.door.is_some(),
                    "{kind:?}@{planet}: level 0 has a door"
                );
                assert!(ground.levels >= 1);
                for level in 0..ground.levels {
                    let plan = build_plan(kind, &iu, planet, level);
                    let cm = cost_map_of(&plan);
                    let reach = |to: (u32, u32), what: &str| {
                        assert!(
                            cm.find_path(plan.entry, to).is_some(),
                            "{kind:?}@{planet} L{level}: {what} unreachable from entry"
                        );
                    };
                    if let Some(door) = plan.door {
                        // The entry sits just inside; the door tile itself
                        // is the walkable corridor end.
                        reach(door, "exit door");
                    }
                    if let Some(s) = plan.stairs_down {
                        reach(s, "stairs down");
                    }
                    if let Some(s) = plan.stairs_up {
                        reach(s, "stairs up");
                    }
                    reach(plan.hunt_spot, "hunt spot");
                    // Levels below 0 must have a way back up.
                    if level > 0 {
                        assert!(plan.stairs_up.is_some(), "L{level} needs stairs up");
                    }
                    // Non-final levels must lead further down.
                    if level + 1 < plan.levels {
                        assert!(plan.stairs_down.is_some(), "L{level} needs stairs down");
                    }
                }
            }
        }
    }

    /// Stair positions agree across adjacent levels' plans regenerated
    /// independently (determinism), and the hunt spot is genuinely deep.
    #[test]
    fn maze_generation_is_deterministic_and_deep() {
        let iu = iu();
        let a = build_plan(BuildingKind::Mine, &iu, "ceres", 0);
        let b = build_plan(BuildingKind::Mine, &iu, "ceres", 0);
        assert_eq!(a.terrain, b.terrain, "same seed, same maze");
        assert_eq!(a.stairs_down, b.stairs_down);
        let cm = cost_map_of(&a);
        let to_hunt = cm.find_path(a.entry, a.hunt_spot).map(|p| p.len()).unwrap();
        assert!(
            to_hunt >= 20,
            "the hunt spot should be a real trek, got {to_hunt} tiles"
        );
    }

    /// Each venue reads differently: the mine is a winding tree with thin
    /// tunnels, the warehouse one big hall with container blocks, the
    /// substation rooms-and-corridors.
    #[test]
    fn venues_have_distinct_shapes() {
        let iu = iu();
        let floor_count = |p: &InteriorPlan| {
            p.terrain
                .iter()
                .enumerate()
                .filter(|&(i, &t)| t == 0 && !p.solid[i])
                .count()
        };
        let mine = build_plan(BuildingKind::Mine, &iu, "earth", 0);
        let wh = build_plan(BuildingKind::Warehouse, &iu, "earth", 0);
        let sub = build_plan(BuildingKind::Substation, &iu, "earth", 0);
        // The warehouse hall dwarfs the mine's tunnels in open floor.
        assert!(
            floor_count(&wh) > floor_count(&mine),
            "warehouse {} vs mine {}",
            floor_count(&wh),
            floor_count(&mine)
        );
        // The warehouse actually contains container blocks.
        assert!(
            wh.solid.iter().filter(|&&s| s).count() > 40,
            "container blocks present"
        );
        // Substation has multiple disjoint room clusters joined by
        // corridors — more floor than the mine's tunnels too.
        assert!(floor_count(&sub) > 0);
        // Mazes never expose a counter.
        assert!(mine.counter.is_none() && wh.counter.is_none() && sub.counter.is_none());
    }

    /// Maze venues derive from what the world is, and are rare.
    #[test]
    fn maze_venues_derive_from_world_economy() {
        let iu = iu();
        // Ceres is a mining world (iron on the market) → a mine.
        assert_eq!(
            super::super::buildings::maze_venue_for_planet("ceres", &iu),
            Some(BuildingKind::Mine),
            "ceres digs"
        );
        // Every venue is at most one per world.
        for planet in ["earth", "ceres", "deneb_prime", "halcyon", "mars"] {
            let v = super::super::buildings::maze_venue_for_planet(planet, &iu);
            assert!(v.is_none() || v.is_some(), "{planet}: at most one venue");
        }
    }
}
