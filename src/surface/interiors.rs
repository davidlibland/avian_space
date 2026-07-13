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

/// Which buildings have walkable interiors (the rest keep their windows).
pub(crate) fn has_interior(kind: BuildingKind) -> bool {
    matches!(
        kind,
        BuildingKind::Bar | BuildingKind::Outfitter | BuildingKind::Shipyard
    )
}

/// Set when entering a door; read by `setup_interior`.
#[derive(Resource, Clone, Copy)]
pub struct InteriorContext {
    pub kind: BuildingKind,
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
    /// The exit-door tile (south wall centre).
    pub door: (u32, u32),
    /// Display plinth tiles, in stock order.
    pub displays: Vec<(u32, u32)>,
    /// Counter tile (north wall centre).
    pub counter: (u32, u32),
    /// Room bounds (x0, y0, w, h) for prop placement/tests.
    pub room: (u32, u32, u32, u32),
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

/// Build the floor plan for a building kind against the planet's derived
/// stock. Deterministic: same planet, same shop, same room.
pub(crate) fn build_plan(kind: BuildingKind, iu: &ItemUniverse, planet: &str) -> InteriorPlan {
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
        _ => (0, false), // bar: fixed cosy room
    };

    let (rw, rh, hall_cols) = if ship_hall {
        hall_size_for(stock_len)
    } else if stock_len > 0 {
        let (w, h) = room_size_for(stock_len);
        (w, h, 0)
    } else {
        (16, 12, 0) // the bar
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

    InteriorPlan {
        terrain,
        entry,
        door,
        displays,
        counter,
        room: (x0, y0, rw, rh),
    }
}

// ── Scene construction ────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub(crate) fn setup_interior(
    mut commands: Commands,
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
    let Some(context) = context else { return };
    let kind = context.kind;
    let planet = landed.planet_name.clone().unwrap_or_default();
    let plan = build_plan(kind, &iu, &planet);

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
                Sprite::from_atlas_image(
                    atlas_image.clone(),
                    TextureAtlas {
                        layout: layout.clone(),
                        index: index as usize,
                    },
                ),
                Transform::from_xyz(pos.x, pos.y, -10.0),
            ));
            // Solid walls (tier ≥ wall) block; the door notch is plating.
            let tier = map2d[ty as usize][tx as usize];
            let solid = biome
                .terrains
                .iter()
                .find(|t| t.row == tier)
                .map(|t| t.collision == 1)
                .unwrap_or(tier >= T_WALL);
            if solid {
                commands.spawn((
                    DespawnOnExit(PlayState::Inside),
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
        .map(|&tier| {
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
    let cpos = super::tile_to_world(plan.counter.0, plan.counter.1, map_w, map_h, tile_px);
    commands.spawn((
        DespawnOnExit(PlayState::Inside),
        Counter(kind),
        Transform::from_xyz(cpos.x, cpos.y, 0.0),
    ));
    let dpos = super::tile_to_world(plan.door.0, plan.door.1, map_w, map_h, tile_px);
    commands.spawn((
        DespawnOnExit(PlayState::Inside),
        ExitDoor,
        Transform::from_xyz(dpos.x, dpos.y, 0.0),
    ));

    // ── The walker, just inside the door ──
    let spawn_pos = super::tile_to_world(plan.entry.0, plan.entry.1, map_w, map_h, tile_px);
    if let Some((walker_image, walker_layout)) = character_layers.as_deref_mut().and_then(|l| {
        l.composite(&game_state.avatar, &mut images)
            .map(|img| (img, l.layout.clone()))
    }) {
        let walker_anim = CharacterAnim::person(0.08);
        let initial_index = walker_anim.atlas_index();
        commands.spawn((
            DespawnOnExit(PlayState::Inside),
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

/// E inside: exit door first, then the counter (classic window).
pub(crate) fn interior_interact(
    keyboard: Res<ButtonInput<KeyCode>>,
    walker: Query<&Transform, With<Walker>>,
    exits: Query<&Transform, (With<ExitDoor>, Without<Walker>)>,
    counters: Query<(&Counter, &Transform), Without<Walker>>,
    mut active_ui: ResMut<ActiveBuildingUI>,
    context: Option<Res<InteriorContext>>,
    mut commands: Commands,
    mut next_state: ResMut<NextState<PlayState>>,
) {
    if !keyboard.just_pressed(KeyCode::KeyE) || active_ui.0.is_some() {
        return;
    }
    let Ok(wtf) = walker.single() else { return };
    let wp = wtf.translation.truncate();
    let range = TILE_PX * 1.6;
    if exits
        .iter()
        .any(|t| (t.translation.truncate() - wp).length() < range)
    {
        if let Some(ctx) = context {
            commands.insert_resource(ReturnFromInterior(ctx.kind));
        }
        next_state.set(PlayState::Exploring);
        return;
    }
    if let Some((counter, _)) = counters
        .iter()
        .find(|(_, t)| (t.translation.truncate() - wp).length() < range)
    {
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

/// Offer NPCs whose building has an interior spawn INSIDE it; the bar's
/// patrons come with them. Idempotent per mission id (same as the surface
/// spawner). Companion avatars arrive via the shared system.
pub(crate) fn spawn_interior_npcs(
    mut commands: Commands,
    context: Option<Res<InteriorContext>>,
    landed: Res<crate::planet_ui::LandedContext>,
    iu: Res<ItemUniverse>,
    offers: Res<crate::missions::MissionOffers>,
    catalog: Res<crate::missions::MissionCatalog>,
    existing: Query<&crate::surface_npc::MissionNpc>,
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
    let plan = build_plan(kind, &iu, &planet);
    let (x0, y0, rw, rh) = plan.room;
    let mut slot = 0u32;
    for id in offers.npc.get(&planet).cloned().unwrap_or_default() {
        if already.contains(id.as_str()) {
            continue;
        }
        let Some(def) = catalog.defs.get(&id) else {
            continue;
        };
        let crate::missions::types::OfferKind::NpcOffer { building, npc, .. } = &def.offer else {
            continue;
        };
        if building.as_deref() != Some(building_name.as_str()) {
            continue;
        }
        // A table spot along the west side.
        let tile = (x0 + 2, (y0 + 3 + slot * 2).min(y0 + rh - 3));
        slot += 1;
        let identity = super::npc_identity(&iu, &layers, npc);
        let walk_speed = layers.walk_speed;
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
    let _ = rw;
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

    /// Walkable predicate mirroring setup: below the wall tier walks.
    fn cost_map_of(plan: &InteriorPlan) -> SurfaceCostMap {
        let data = plan
            .terrain
            .iter()
            .map(|&t| if t < T_WALL { 1.0 } else { f32::INFINITY })
            .collect();
        SurfaceCostMap {
            data,
            width: WORLD_WIDTH,
            height: WORLD_HEIGHT,
        }
    }

    /// Shops are rooms, not mazes: every display, the counter, and the
    /// entry are mutually reachable, and the entry sits by the door notch.
    #[test]
    fn every_display_and_the_counter_are_reachable_from_the_entry() {
        let iu = iu();
        for (kind, planet) in [
            (BuildingKind::Bar, "earth"),
            (BuildingKind::Outfitter, "earth"),
            (BuildingKind::Shipyard, "earth"),
            (BuildingKind::Outfitter, "marches_freeport"),
            (BuildingKind::Shipyard, "deneb_prime"),
        ] {
            let plan = build_plan(kind, &iu, planet);
            let cm = cost_map_of(&plan);
            assert!(
                cm.find_path(plan.entry, plan.counter).is_some(),
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

    /// The blob47 autotiler contract: no two 8-connected neighbors differ
    /// by more than one tier.
    #[test]
    fn interior_terrain_satisfies_the_gradient_contract() {
        let iu = iu();
        let plan = build_plan(BuildingKind::Shipyard, &iu, "earth");
        let (w, h) = (WORLD_WIDTH as i32, WORLD_HEIGHT as i32);
        for y in 0..h {
            for x in 0..w {
                let t = plan.terrain[(y * w + x) as usize] as i64;
                for (dx, dy) in [(1, 0), (0, 1), (1, 1), (1, -1)] {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w || ny >= h {
                        continue;
                    }
                    let n = plan.terrain[(ny * w + nx) as usize] as i64;
                    assert!(
                        (t - n).abs() <= 1,
                        "gradient break at ({x},{y}): {t} vs {n}"
                    );
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
        let plan = build_plan(BuildingKind::Outfitter, &iu, "earth");
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
        let small = build_plan(BuildingKind::Outfitter, &iu, "marches_freeport");
        let area = |p: &InteriorPlan| p.room.2 * p.room.3;
        assert!(
            area(&plan) > area(&small),
            "Earth's outfitter hall outsizes the freeport booth"
        );
        // Ships too.
        let hulls = iu.find_gameplay_planet("earth").unwrap().1.shipyard.len();
        let yard = build_plan(BuildingKind::Shipyard, &iu, "earth");
        assert_eq!(yard.displays.len(), hulls, "every hull gets a cradle");
        // The bar ignores stock entirely.
        let bar = build_plan(BuildingKind::Bar, &iu, "earth");
        assert_eq!((bar.room.2, bar.room.3), (16, 12), "cosy, fixed, no maze");
    }

    /// The outside-spawn filter's name mapping covers every kind, and
    /// exactly the walk-in shops report an interior.
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
        ] {
            let name = format!("{kind:?}").to_lowercase();
            assert_eq!(
                crate::surface::npcs::building_kind_from_name(&name),
                Some(kind),
                "{name} maps back to its kind"
            );
            assert_eq!(
                has_interior(kind),
                matches!(kind, Bar | Outfitter | Shipyard),
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
}
