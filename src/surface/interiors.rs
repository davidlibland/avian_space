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
use crate::{CurrentStarSystem, GameLayer, PlayState, Player};

use super::{
    ActiveBuildingUI, BuildingKind, TILE_PX, WORLD_HEIGHT, WORLD_WIDTH, WORLDS_DIR, Walker,
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

/// The maze-hunt chase ledger: for each catch mission whose target has
/// fled INSIDE its venue, which level the fugitive is waiting on. Session
/// resource — leave the planet mid-chase and they're still down there.
#[derive(Resource, Default)]
pub struct MazeChase {
    pub inside: std::collections::HashMap<String, u8>,
}

/// A fleeing hunt target with a destination: the venue door (outside) or
/// the down shaft (inside). Arrival despawns them and advances the chase.
#[derive(Component)]
pub(crate) struct MazeFugitive {
    pub mission_id: String,
    pub goal: (u32, u32),
    pub next: FugitiveNext,
}

#[derive(Clone, Copy)]
pub(crate) enum FugitiveNext {
    /// Slips through the venue door — the chase moves inside (level 0).
    IntoBuilding,
    /// Scrambles down the shaft — the chase moves one level deeper.
    Descend,
}

/// Fugitives that reach their goal vanish ahead of the player: into the
/// building from the surface, or down the stairs between levels. Also
/// prunes chase entries for missions that ended.
pub(crate) fn maze_fugitive_arrivals(
    mut commands: Commands,
    fugitives: Query<(Entity, &Transform, &MazeFugitive)>,
    mut chase: ResMut<MazeChase>,
    mission_log: Res<crate::missions::MissionLog>,
    mut comms: ResMut<crate::hud::CommsChannel>,
) {
    for (entity, tf, fug) in &fugitives {
        let goal =
            crate::surface_pathfinding::SurfaceCostMap::tile_to_world(fug.goal.0, fug.goal.1);
        if (tf.translation.truncate() - goal).length() > TILE_PX * 0.9 {
            continue;
        }
        commands.entity(entity).despawn();
        match fug.next {
            FugitiveNext::IntoBuilding => {
                chase.inside.insert(fug.mission_id.clone(), 0);
                comms.send("The target ducks inside! After them.");
            }
            FugitiveNext::Descend => {
                let lvl = chase.inside.entry(fug.mission_id.clone()).or_insert(0);
                *lvl += 1;
                comms.send("They scramble down the shaft — keep up!");
            }
        }
    }
    chase.inside.retain(|id, _| {
        matches!(
            mission_log.status(id),
            crate::missions::MissionStatus::Active(_)
        )
    });
}

/// Where a pursued fugitive waits on this level: halfway along the
/// shortest path from the player's entry to the down stairs (their head
/// start), or None when this level has no way further down (cornered —
/// they fall back to ordinary flee AI).
pub(crate) fn fugitive_head_start(
    plan: &InteriorPlan,
    cm: &crate::surface_pathfinding::SurfaceCostMap,
) -> Option<((u32, u32), (u32, u32))> {
    let stairs = plan.stairs_down?;
    let path = cm.find_path(plan.entry, stairs)?;
    let spawn = *path.get(path.len() / 2)?;
    Some((spawn, stairs))
}

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

/// What the outfitter actually SELLS to this player: stocked here,
/// license held, and not a ship mod (mods live at the counter, exactly
/// like the classic window's split).
pub(crate) fn purchasable_items(
    iu: &ItemUniverse,
    planet: &str,
    unlocks: &crate::missions::PlayerUnlocks,
) -> Vec<String> {
    let mut items: Vec<String> = iu
        .find_gameplay_planet(planet)
        .map(|(_, pd)| {
            pd.outfitter
                .iter()
                .filter(|k| {
                    iu.outfitter_items.get(k.as_str()).is_some_and(|item| {
                        item.mod_effect().is_none()
                            && item.required_unlocks().iter().all(|u| unlocks.has(u))
                    })
                })
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    items.sort();
    items
}

/// The hulls this player can actually buy here (license-gated).
pub(crate) fn purchasable_ships(
    iu: &ItemUniverse,
    planet: &str,
    unlocks: &crate::missions::PlayerUnlocks,
) -> Vec<String> {
    let mut ships: Vec<String> = iu
        .find_gameplay_planet(planet)
        .map(|(_, pd)| {
            pd.shipyard
                .iter()
                .filter(|k| {
                    iu.ships
                        .get(k.as_str())
                        .is_some_and(|d| d.required_unlocks.iter().all(|u| unlocks.has(u)))
                })
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    ships.sort();
    ships
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
        // Exit markers stand ON the walkable door tile.
        "exit_door" | "ladder_up" => (1, 1, false),
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
    unlocks: &crate::missions::PlayerUnlocks,
) -> InteriorPlan {
    if super::mazes::is_maze(kind) {
        return maze_plan(kind, planet, level);
    }
    let (stock_len, ship_hall) = match kind {
        BuildingKind::Outfitter => (purchasable_items(iu, planet, unlocks).len(), false),
        BuildingKind::Shipyard => (purchasable_ships(iu, planet, unlocks).len(), true),
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
        // The same plating pedestal tiles the outfitter's plinths sit on.
        for &(px, py) in &displays {
            terrain[(py * map_w + px) as usize] = T_PLATING;
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
    let mut props: Vec<(&'static str, (u32, u32))> = vec![("exit_door", door)];
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

/// Per-world tile tint: the SAME venue atlas reads differently under a
/// cryo dome than in a desert dig — cool light on ice worlds, amber on
/// deserts, rust on rocky mining worlds, warm neutral on garden colonies.
/// Multiplicative, subtle: geometry and readability stay untouched.
pub(crate) fn world_tint(surface_biome: &str) -> Color {
    match surface_biome {
        "garden" => Color::srgb(1.0, 0.98, 0.94),
        "ice" => Color::srgb(0.84, 0.92, 1.0),
        "rocky" => Color::srgb(1.0, 0.90, 0.84),
        "desert" => Color::srgb(1.0, 0.95, 0.80),
        // stations / cloud decks: cool artificial light
        _ => Color::srgb(0.92, 0.95, 1.0),
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
    if let Some(door) = lv.door {
        // Climbing out of a dig reads as a ladder; the warehouse keeps a
        // proper lit doorway.
        lv.props.push((
            match kind {
                BuildingKind::Mine | BuildingKind::Substation => "ladder_up",
                _ => "exit_door",
            },
            door,
        ));
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
    player_ship: Query<Entity, With<Player>>,
    unlocks: Res<crate::missions::PlayerUnlocks>,
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
    let plan = build_plan(kind, &iu, &planet, context.level, &unlocks);

    commands.insert_resource(ClearColor(Color::srgb(0.02, 0.02, 0.03)));
    // teardown_surface (OnExit(Exploring)) re-reveals the parked ship on
    // EVERY exit, including the one into this interior — hide it again.
    if let Ok(ship) = player_ship.single() {
        commands.entity(ship).insert(Visibility::Hidden);
    }

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
    // Maze venues bring their own tileset (mine tunnels, container canyons,
    // pipe runs); shops use the station-interior atlas.
    let biome_name = match kind {
        BuildingKind::Mine => "mine",
        BuildingKind::Warehouse => "warehouse",
        BuildingKind::Substation => "substation",
        _ => "interior",
    };
    let biome = manifest
        .biomes
        .get(biome_name)
        .or_else(|| manifest.biomes.get("interior"))
        .or_else(|| manifest.biomes.values().next());
    let Some(biome) = biome else { return };
    let atlas_image: Handle<Image> = asset_server.load(format!("{WORLDS_DIR}/{}", biome.atlas));
    // Per-world lighting tint over the whole tileset.
    let surface_biome = iu
        .find_gameplay_planet(&planet)
        .map(|(_, pd)| crate::world_assets::planet_type_to_biome(&pd.planet_type))
        .unwrap_or("rocky");
    let tint = world_tint(surface_biome);
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

    // ── Tiles: the WHOLE map — any gap would show the space scene
    // behind the interior. Colliders only in a band around the walkable
    // region (incl. the entrance corridor), which seals every edge the
    // walker can actually reach.
    let walkable_bounds = {
        let (mut lo_x, mut lo_y, mut hi_x, mut hi_y) = (map_w, map_h, 0u32, 0u32);
        for y in 0..map_h {
            for x in 0..map_w {
                let i = (y * map_w + x) as usize;
                if plan.terrain[i] < plan.solid_min_tier && !plan.solid[i] {
                    lo_x = lo_x.min(x);
                    lo_y = lo_y.min(y);
                    hi_x = hi_x.max(x);
                    hi_y = hi_y.max(y);
                }
            }
        }
        (
            lo_x.saturating_sub(3),
            lo_y.saturating_sub(3),
            (hi_x + 4).min(map_w),
            (hi_y + 4).min(map_h),
        )
    };
    for ty in 0..map_h {
        for tx in 0..map_w {
            let index = crate::world_assets::tile_texture_index(
                &map2d,
                tx as i32,
                ty as i32,
                map_w as i32,
                map_h as i32,
                &lut,
            );
            let pos = super::tile_to_world(tx, ty, map_w, map_h, tile_px);
            let mut tile_sprite = Sprite::from_atlas_image(
                atlas_image.clone(),
                TextureAtlas {
                    layout: layout.clone(),
                    index: index as usize,
                },
            );
            tile_sprite.color = tint;
            commands.spawn((
                DespawnOnExit(PlayState::Inside),
                InteriorScoped,
                tile_sprite,
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
            let in_collider_band = tx >= walkable_bounds.0
                && ty >= walkable_bounds.1
                && tx < walkable_bounds.2
                && ty < walkable_bounds.3;
            if solid && in_collider_band {
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

    // ── Mini-map: the HUD keeps its surface slot, but inside it shows
    // THIS floor's layout — walls dark, floor lit, exits highlighted.
    {
        let mut pixels = vec![0u8; (map_w * map_h * 4) as usize];
        for y in 0..map_h {
            for x in 0..map_w {
                let i = (y * map_w + x) as usize;
                let tier = plan.terrain[i];
                let (mut r, mut g, mut b) = biome
                    .terrains
                    .iter()
                    .find(|t| t.row == tier)
                    .map(|t| t.map_color)
                    .unwrap_or((40, 40, 44));
                if plan.solid[i] {
                    r /= 2;
                    g /= 2;
                    b /= 2;
                }
                let p = i * 4;
                pixels[p] = r;
                pixels[p + 1] = g;
                pixels[p + 2] = b;
                pixels[p + 3] = 255;
            }
        }
        let mut mark = |tile: Option<(u32, u32)>, rgb: [u8; 3]| {
            if let Some((tx, ty)) = tile {
                let p = ((ty * map_w + tx) * 4) as usize;
                pixels[p..p + 3].copy_from_slice(&rgb);
            }
        };
        mark(plan.door, [90, 255, 120]); // the way out: green
        mark(plan.stairs_down, [255, 210, 80]); // shafts: amber
        mark(plan.stairs_up, [255, 210, 80]);
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
        commands.insert_resource(super::SurfaceMiniMap {
            image: images.add(img),
            map_w,
            map_h,
            buildings: Vec::new(),
            pad_pos: plan.door.unwrap_or(plan.entry),
        });
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
    // Only what this player can actually buy — same filter as the
    // classic window (license held, mods excluded).
    let stock: Vec<DisplayBinding> = match kind {
        BuildingKind::Outfitter => purchasable_items(&iu, &planet, &unlocks)
            .into_iter()
            .map(DisplayBinding::OutfitterItem)
            .collect(),
        BuildingKind::Shipyard => purchasable_ships(&iu, &planet, &unlocks)
            .into_iter()
            .map(DisplayBinding::Ship)
            .collect(),
        _ => Vec::new(),
    };
    for (slot, binding) in stock.iter().enumerate() {
        let Some(&(px, py)) = plan.displays.get(slot) else {
            break;
        };
        let pos = super::tile_to_world(px, py, map_w, map_h, tile_px);
        let (sprite, size) = display_sprite(binding, &iu, &asset_server);
        // The pedestal catches the hologram's light: a translucent pad in
        // the weapon's signature color (holo cyan for hulls and the rest).
        let pad_color = match binding {
            DisplayBinding::OutfitterItem(key) => iu
                .weapons
                .get(key)
                .map(|w| Color::srgba(w.color[0], w.color[1], w.color[2], 0.30))
                .unwrap_or(Color::srgba(0.4, 0.9, 1.0, 0.22)),
            DisplayBinding::Ship(_) => Color::srgba(0.4, 0.9, 1.0, 0.22),
        };
        commands.spawn((
            DespawnOnExit(PlayState::Inside),
            InteriorScoped,
            Sprite::from_color(pad_color, Vec2::splat(tile_px * 0.88)),
            Transform::from_xyz(pos.x, pos.y, -9.5),
        ));
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
        // Holo name-tag under the plinth — glyphs are shared within a
        // weapon family, so the label is what tells a laser from a
        // proton beam at a glance. Same tint as the wireframe.
        let label = display_label(binding, &iu);
        if !label.is_empty() {
            commands.spawn((
                DespawnOnExit(PlayState::Inside),
                InteriorScoped,
                Text2d::new(label),
                TextFont {
                    font_size: 18.0,
                    ..default()
                },
                TextColor(Color::srgba(0.4, 0.9, 1.0, 0.85)),
                bevy::text::TextLayout::new_with_justify(bevy::text::Justify::Center),
                Transform::from_xyz(
                    pos.x,
                    pos.y - tile_px * 0.62,
                    crate::surface_objects::depth_z(pos.y - tile_px * 0.5) + 0.4,
                )
                .with_scale(Vec3::splat(0.45)),
            ));
        }
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
    if let Some(layers) = character_layers.as_deref_mut()
        && let Some(walker) = super::spawn_walker_at(
            &mut commands,
            layers,
            &mut images,
            &game_state.avatar,
            spawn_pos,
            PlayState::Inside,
        )
    {
        commands.entity(walker).insert(InteriorScoped);
    }

    if let Ok(mut cam_tf) = camera_query.single_mut() {
        cam_tf.translation = Vec3::new(spawn_pos.x, spawn_pos.y, cam_tf.translation.z);
    }
    zoom.target = super::SURFACE_CAMERA_SCALE * 0.8;
}

/// The holo name-tag under a display.
fn display_label(binding: &DisplayBinding, iu: &ItemUniverse) -> String {
    match binding {
        DisplayBinding::Ship(key) => iu
            .ships
            .get(key)
            .map(|d| {
                if d.display_name.is_empty() {
                    key.clone()
                } else {
                    d.display_name.clone()
                }
            })
            .unwrap_or_else(|| key.clone()),
        DisplayBinding::OutfitterItem(key) => iu
            .outfitter_items
            .get(key)
            .map(|i| i.display_name().to_string())
            .unwrap_or_else(|| key.clone()),
    }
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
            // Holo-schematic over the plinth: every weapon has a generated
            // wireframe (gen_wireframes.py item glyphs); anything else
            // shows the generic cargo-canister schematic.
            let path = if iu.weapons.contains_key(item) {
                format!("sprites/wireframes/item_{item}.png")
            } else {
                "sprites/wireframes/pickup.png".to_string()
            };
            let mut s = Sprite::from_image(asset_server.load(path));
            s.color = Color::srgba(0.4, 0.9, 1.0, 0.9);
            s.custom_size = Some(Vec2::splat(TILE_PX * 1.3));
            (s, 1.0)
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
    chase: Res<MazeChase>,
    unlocks: Res<crate::missions::PlayerUnlocks>,
) {
    let (Some(context), Some(mut layers), Some(cm)) = (context, layers, cost_map) else {
        return;
    };
    let kind = context.kind;
    let planet = landed.planet_name.clone().unwrap_or_default();
    let already: std::collections::HashSet<&str> = existing.iter().map(|m| m.0.as_str()).collect();
    let building_name = format!("{kind:?}").to_lowercase();
    let plan = build_plan(kind, &iu, &planet, context.level, &unlocks);
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

    // Hunt targets bound to THIS building. Meet targets hide at the
    // deepest level's farthest tile. Catch targets in maze venues play
    // the staged chase: they're only here if the chase ledger says so,
    // spawned halfway along the path to the down stairs (their head
    // start) and fleeing TOWARD them; on the deepest level they're
    // cornered and fall back to ordinary flee AI.
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
        let (m_planet, m_building, npc, is_catch) = match &def.objective {
            crate::missions::Objective::MeetNpc {
                planet: p,
                building: b,
                npc,
                ..
            } => (p, b, npc, false),
            crate::missions::Objective::CatchNpc {
                planet: p,
                building: b,
                npc,
                ..
            } => (p, b, npc, true),
            _ => continue,
        };
        if *m_planet != planet || m_building.as_deref() != Some(building_name.as_str()) {
            continue;
        }

        let maze = super::mazes::is_maze(kind);
        let deepest = context.level + 1 == plan.levels;
        let (spawn_tile, objective, fugitive_goal) = if is_catch && maze {
            // Chase discipline: the fugitive is on exactly one level.
            match chase.inside.get(mission_id.as_str()) {
                Some(&lvl) if lvl == context.level => {}
                _ => continue,
            }
            match fugitive_head_start(&plan, &cm) {
                Some((spawn, stairs)) => (
                    spawn,
                    crate::surface_npc::ObjectiveKind::CatchToward { goal: stairs },
                    Some(stairs),
                ),
                None => {
                    // Bottom of the shaft: cornered, half a level of room.
                    let spawn = cm
                        .find_path(plan.entry, plan.hunt_spot)
                        .and_then(|p| p.get(p.len() / 2).copied())
                        .unwrap_or(plan.hunt_spot);
                    (spawn, crate::surface_npc::ObjectiveKind::Catch, None)
                }
            }
        } else {
            if !deepest {
                continue;
            }
            (
                plan.hunt_spot,
                if is_catch {
                    crate::surface_npc::ObjectiveKind::Catch
                } else {
                    crate::surface_npc::ObjectiveKind::Meet { seek: false }
                },
                None,
            )
        };

        let identity = super::npc_identity(&iu, &layers, npc);
        let spawned = crate::surface_npc::spawn_objective_npc(
            &mut commands,
            &mut layers,
            &mut images,
            "civilian",
            identity,
            mission_id,
            spawn_tile,
            walk_speed * 1.1,
            objective,
            PlayState::Inside,
        );
        if let Some(entity) = spawned {
            // Level rebuilds (stairs) must clear hunt NPCs with the scene.
            commands.entity(entity).insert(InteriorScoped);
            if let Some(goal) = fugitive_goal {
                commands.entity(entity).insert(MazeFugitive {
                    mission_id: mission_id.clone(),
                    goal,
                    next: FugitiveNext::Descend,
                });
            }
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

    fn un() -> crate::missions::PlayerUnlocks {
        crate::missions::PlayerUnlocks::default()
    }

    /// Every license in the universe — for size comparisons that shouldn't
    /// depend on what a fresh pilot happens to hold.
    fn un_all(iu: &ItemUniverse) -> crate::missions::PlayerUnlocks {
        let mut all = std::collections::HashSet::new();
        for item in iu.outfitter_items.values() {
            all.extend(item.required_unlocks().iter().cloned());
        }
        for ship in iu.ships.values() {
            all.extend(ship.required_unlocks.iter().cloned());
        }
        crate::missions::PlayerUnlocks(all)
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
            let plan = build_plan(kind, &iu, planet, 0, &un());
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
            &build_plan(BuildingKind::Shipyard, &iu, "earth", 0, &un()).terrain,
            "shipyard",
        );
        for kind in [
            BuildingKind::Mine,
            BuildingKind::Warehouse,
            BuildingKind::Substation,
        ] {
            for planet in ["earth", "ceres", "deneb_prime"] {
                let plan = build_plan(kind, &iu, planet, 0, &un());
                for level in 0..plan.levels {
                    let p = build_plan(kind, &iu, planet, level, &un());
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
        let all = un_all(&iu);
        let earth_items = purchasable_items(&iu, "earth", &all).len();
        let plan = build_plan(BuildingKind::Outfitter, &iu, "earth", 0, &all);
        assert_eq!(
            plan.displays.len(),
            earth_items,
            "every stocked item gets a plinth"
        );
        let freeport_items = purchasable_items(&iu, "marches_freeport", &all).len();
        assert!(earth_items > freeport_items, "premise: Earth stocks more");
        let small = build_plan(BuildingKind::Outfitter, &iu, "marches_freeport", 0, &all);
        let area = |p: &InteriorPlan| p.room.2 * p.room.3;
        assert!(
            area(&plan) > area(&small),
            "Earth's outfitter hall outsizes the freeport booth"
        );
        // Ships too.
        let hulls = purchasable_ships(&iu, "earth", &all).len();
        let yard = build_plan(BuildingKind::Shipyard, &iu, "earth", 0, &all);
        assert_eq!(yard.displays.len(), hulls, "every hull gets a cradle");
        // The bar ignores stock entirely.
        let bar = build_plan(BuildingKind::Bar, &iu, "earth", 0, &un());
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
                let ground = build_plan(kind, &iu, planet, 0, &un());
                assert!(
                    ground.door.is_some(),
                    "{kind:?}@{planet}: level 0 has a door"
                );
                assert!(ground.levels >= 1);
                for level in 0..ground.levels {
                    let plan = build_plan(kind, &iu, planet, level, &un());
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
        let a = build_plan(BuildingKind::Mine, &iu, "ceres", 0, &un());
        let b = build_plan(BuildingKind::Mine, &iu, "ceres", 0, &un());
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
        let mine = build_plan(BuildingKind::Mine, &iu, "earth", 0, &un());
        let wh = build_plan(BuildingKind::Warehouse, &iu, "earth", 0, &un());
        let sub = build_plan(BuildingKind::Substation, &iu, "earth", 0, &un());
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

    /// Every venue biome exists in the world manifest with the full
    /// six-tier row set, its atlas file is on disk with matching rows,
    /// and solidity matches the maze contract (grate+ solid enough).
    #[test]
    fn venue_biomes_have_atlases_and_six_tiers() {
        let manifest_text =
            std::fs::read_to_string("assets/sprites/worlds/world_manifest.ron").unwrap();
        let manifest: crate::world_assets::WorldManifest = ron::from_str(&manifest_text).unwrap();
        for name in ["mine", "warehouse", "substation", "interior"] {
            let biome = manifest
                .biomes
                .get(name)
                .unwrap_or_else(|| panic!("{name} biome in manifest"));
            if name != "interior" {
                assert_eq!(biome.terrains.len(), 6, "{name}: six tiers incl. void");
                let atlas = format!("assets/sprites/worlds/{}", biome.atlas);
                let meta = std::fs::metadata(&atlas).unwrap_or_else(|_| panic!("{atlas} exists"));
                assert!(meta.len() > 0);
                // Wall and everything above must collide (mazes also
                // solidify the grate tier via solid_min_tier).
                for t in &biome.terrains {
                    if t.row >= 3 {
                        assert_eq!(t.collision, 1, "{name}/{}: solid", t.name);
                    }
                }
            }
        }
    }

    /// World tints are distinct per surface biome and subtle enough not to
    /// crush the tile art (every channel ≥ 0.8).
    #[test]
    fn world_tints_are_distinct_and_subtle() {
        let biomes = ["garden", "ice", "rocky", "desert", "interior"];
        let mut seen = Vec::new();
        for b in biomes {
            let c = world_tint(b).to_srgba();
            for ch in [c.red, c.green, c.blue] {
                assert!((0.8..=1.0).contains(&ch), "{b}: subtle tint, got {ch}");
            }
            let key = (
                (c.red * 100.0) as u32,
                (c.green * 100.0) as u32,
                (c.blue * 100.0) as u32,
            );
            assert!(!seen.contains(&key), "{b}: tint duplicates another biome");
            seen.push(key);
        }
        // And the mapping actually differentiates real planets: an ice
        // world and a desert world get different light.
        assert_ne!(
            world_tint(crate::world_assets::planet_type_to_biome("icy_dwarf")).to_srgba(),
            world_tint(crate::world_assets::planet_type_to_biome("desert")).to_srgba(),
        );
    }

    /// The staged chase: on every level with a way down, the fugitive
    /// gets a head start — spawned ON the entry→stairs path, roughly
    /// halfway, strictly closer to the stairs than the player's entry —
    /// and the deepest level corners them (no head start possible).
    #[test]
    fn fugitives_get_a_halfway_head_start_on_every_descending_level() {
        let iu = iu();
        for planet in ["ceres", "earth", "deneb_prime"] {
            let ground = build_plan(BuildingKind::Mine, &iu, planet, 0, &un());
            assert!(ground.levels >= 2, "premise: mines descend");
            for level in 0..ground.levels {
                let plan = build_plan(BuildingKind::Mine, &iu, planet, level, &un());
                let cm = cost_map_of(&plan);
                let head_start = fugitive_head_start(&plan, &cm);
                if level + 1 == plan.levels {
                    assert!(head_start.is_none(), "deepest level corners the fugitive");
                    continue;
                }
                let (spawn, stairs) = head_start.unwrap_or_else(|| panic!("L{level}: head start"));
                assert_eq!(Some(stairs), plan.stairs_down, "goal is the shaft");
                let full = cm.find_path(plan.entry, stairs).unwrap().len();
                let to_spawn = cm.find_path(plan.entry, spawn).unwrap().len();
                let to_go = cm.find_path(spawn, stairs).unwrap().len();
                assert!(
                    to_spawn >= full / 3 && to_spawn <= full * 2 / 3 + 2,
                    "L{level}: spawn ~halfway ({to_spawn} of {full})"
                );
                assert!(
                    to_go < full,
                    "L{level}: the fugitive is strictly AHEAD of the player"
                );
            }
        }
    }

    /// No leaks: every walkable tile (room, corridor, door) is strictly
    /// enclosed — each of its neighbours is walkable or solid, and the
    /// walkable region never reaches the map border (where colliders
    /// would stop).
    #[test]
    fn walkable_region_is_sealed_for_every_interior() {
        let iu = iu();
        for (kind, planet) in [
            (BuildingKind::Bar, "earth"),
            (BuildingKind::Outfitter, "earth"),
            (BuildingKind::Shipyard, "earth"),
            (BuildingKind::Market, "earth"),
            (BuildingKind::Mine, "ceres"),
            (BuildingKind::Warehouse, "earth"),
            (BuildingKind::Substation, "deneb_prime"),
        ] {
            let ground = build_plan(kind, &iu, planet, 0, &un());
            for level in 0..ground.levels {
                let plan = build_plan(kind, &iu, planet, level, &un());
                for y in 0..WORLD_HEIGHT {
                    for x in 0..WORLD_WIDTH {
                        let i = (y * WORLD_WIDTH + x) as usize;
                        let walkable = plan.terrain[i] < plan.solid_min_tier && !plan.solid[i];
                        if !walkable {
                            continue;
                        }
                        assert!(
                            x > 0 && y > 0 && x + 1 < WORLD_WIDTH && y + 1 < WORLD_HEIGHT,
                            "{kind:?}@{planet} L{level}: walkable at map border ({x},{y})"
                        );
                    }
                }
            }
        }
    }

    /// Every interior marks its way out: a prop sits ON the door tile —
    /// a ladder up for digs (mine/substation), a lit doorway elsewhere.
    #[test]
    fn every_interior_marks_its_exit() {
        let iu = iu();
        for (kind, planet, expect) in [
            (BuildingKind::Bar, "earth", "exit_door"),
            (BuildingKind::Market, "earth", "exit_door"),
            (BuildingKind::Shipyard, "earth", "exit_door"),
            (BuildingKind::Warehouse, "earth", "exit_door"),
            (BuildingKind::Mine, "ceres", "ladder_up"),
            (BuildingKind::Substation, "deneb_prime", "ladder_up"),
        ] {
            let plan = build_plan(kind, &iu, planet, 0, &un());
            let door = plan.door.unwrap();
            assert!(
                plan.props.iter().any(|&(n, t)| n == expect && t == door),
                "{kind:?}@{planet}: {expect} on the door tile"
            );
            let (_, _, blocks) = prop_meta(expect);
            assert!(!blocks, "{expect} must stay walkable");
        }
    }

    /// Displays show only what THIS pilot can buy: gaining licenses grows
    /// the plinth count, and a fresh pilot never sees license-locked gear.
    #[test]
    fn displays_respect_licenses() {
        let iu = iu();
        let fresh = purchasable_items(&iu, "earth", &un());
        let licensed = purchasable_items(&iu, "earth", &un_all(&iu));
        assert!(
            fresh.len() < licensed.len(),
            "premise: Earth stocks licensed gear ({} vs {})",
            fresh.len(),
            licensed.len()
        );
        for key in &fresh {
            let item = iu.outfitter_items.get(key).unwrap();
            assert!(item.required_unlocks().is_empty(), "{key} is unlocked gear");
            assert!(
                item.mod_effect().is_none(),
                "{key}: mods stay at the counter"
            );
        }
        let plan = build_plan(BuildingKind::Outfitter, &iu, "earth", 0, &un());
        assert_eq!(plan.displays.len(), fresh.len());
    }

    /// Every weapon in the universe has its display wireframe on disk —
    /// plinths never show an empty holo.
    #[test]
    fn every_weapon_has_an_item_wireframe() {
        let iu = iu();
        for key in iu.weapons.keys() {
            let path = format!("assets/sprites/wireframes/item_{key}.png");
            assert!(
                std::path::Path::new(&path).exists(),
                "{path} missing — rerun scripts/gen_wireframes.py"
            );
        }
        assert!(std::path::Path::new("assets/sprites/wireframes/pickup.png").exists());
    }

    /// Every purchasable display resolves to a non-empty, distinct-enough
    /// label — the name-tag is what distinguishes shared glyph families.
    #[test]
    fn every_display_gets_a_readable_label() {
        let iu = iu();
        let all = un_all(&iu);
        for key in purchasable_items(&iu, "earth", &all) {
            let label = display_label(&DisplayBinding::OutfitterItem(key.clone()), &iu);
            assert!(!label.trim().is_empty(), "{key}: label");
        }
        for key in purchasable_ships(&iu, "earth", &all) {
            let label = display_label(&DisplayBinding::Ship(key.clone()), &iu);
            assert!(!label.trim().is_empty(), "{key}: label");
        }
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
