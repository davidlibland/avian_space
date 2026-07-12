//! Building placement, 3/4 sprite layers, doors, and the faction flag.
#[allow(unused_imports)]
use super::*;

use crate::PlayState;
use crate::item_universe::ItemUniverse;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Determine which buildings to place based on planet data.
/// ShipPad is handled separately as a landing pad, not a building template.
pub(crate) fn building_kinds_for_planet(
    planet_name: &str,
    item_universe: &ItemUniverse,
    system_name: &str,
    galaxy: &crate::galaxy::GalaxyControl,
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
    // Every landable colony has a fuel station — you must be able to refuel
    // wherever you set down (or you'd risk being stranded with an empty tank).
    kinds.push(BuildingKind::FuelStation);
    // The war office follows the LIVE controller: only factions that take
    // sides garrison their worlds, so the building appears/disappears when a
    // system changes hands (and never on unaligned freeports).
    if crate::galaxy::effective_planet_faction(galaxy, item_universe, planet_name)
        .is_some_and(|f| item_universe.faction_takes_sides(&f))
    {
        kinds.push(BuildingKind::Garrison);
    }
    kinds
}

/// Template (footprint + door) used to place each building kind. Chosen so the
/// baked 3/4 sprite — which assumes this footprint — fits. Every style provides
/// these three (cryo/station were given the missing two).
/// Door (walkable, sensor) tiles for a building: the template's entry
/// points — except the SHIPYARD, whose baked 3/4 sprite is an open hull-bay:
/// its whole front row is a walkable threshold apart from the corner pillars.
pub(crate) fn door_tiles_for(
    kind: BuildingKind,
    tmpl: &crate::world_assets::BuildingTemplate,
) -> Vec<(u32, u32)> {
    if kind == BuildingKind::Shipyard && tmpl.width >= 3 {
        (1..tmpl.width - 1).map(|c| (c, 0)).collect()
    } else {
        tmpl.entry_points.clone()
    }
}

pub(crate) fn kind_template(kind: BuildingKind) -> &'static str {
    match kind {
        BuildingKind::Outfitter => "small_house",   // 4×4
        BuildingKind::FuelStation => "small_house", // 4×4 (booth + forecourt)
        BuildingKind::Shipyard => "large_building", // 8×6
        _ => "medium_house",                        // 6×5 — Market, Bar (+ fallback)
    }
}

/// Baked 3/4 sprite name (assets/sprites/worlds/buildings3d/<style>_<func>.png).
pub(crate) fn kind_func(kind: BuildingKind) -> &'static str {
    match kind {
        BuildingKind::Market => "market",
        BuildingKind::Outfitter => "outfitter",
        BuildingKind::Shipyard => "shipyard",
        BuildingKind::Bar => "bar",
        BuildingKind::MechanicShop => "mechanic",
        BuildingKind::ShipPad => "pad",
        BuildingKind::FuelStation => "fuel_station",
        BuildingKind::Garrison => "garrison",
    }
}

/// World position of a footprint's front-centre on the ground (south edge of the
/// door row), where the 3/4 sprite's anchor is pinned.
pub(crate) fn footprint_front_center(
    tx: u32,
    ty: u32,
    w: u32,
    map_w: u32,
    map_h: u32,
    tile_px: f32,
) -> Vec2 {
    let base = tile_to_world(tx, ty, map_w, map_h, tile_px);
    Vec2::new(
        base.x + (w as f32 - 1.0) * 0.5 * tile_px,
        base.y - tile_px * 0.5,
    )
}

/// Spawn a 3/4 building as two depth-split layers so a player stands framed in
/// its doorway: `_back` (surfaces farther than the door plane) draws behind the
/// player; `_front` (surfaces nearer, with the doorway cut open) draws over the
/// player so they show through the opening, framed by the jambs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_building_3d(
    commands: &mut Commands,
    asset_server: &AssetServer,
    style: &str,
    func: &str,
    anchor: (f32, f32),
    fc: Vec2,
    scale: f32,
    tile_px: f32,
    props: &[crate::world_assets::BuildingPropSprite],
) {
    // _floor: the building's own floor + thresholds (walkable). ALWAYS below the
    // player (and above terrain at -10), so the player walks over it and it hides
    // the ground under the footprint / through the doorway.
    commands.spawn((
        DespawnOnExit(PlayState::Exploring),
        Sprite::from_image(
            asset_server.load(format!("{WORLDS_DIR}/buildings3d/{style}_{func}_floor.png")),
        ),
        bevy::sprite::Anchor(Vec2::new(anchor.0, anchor.1)),
        Transform::from_xyz(fc.x, fc.y, crate::surface_objects::depth_z(fc.y) - 2.0)
            .with_scale(Vec3::splat(scale)),
    ));
    commands.spawn((
        DespawnOnExit(PlayState::Exploring),
        Sprite::from_image(
            asset_server.load(format!("{WORLDS_DIR}/buildings3d/{style}_{func}_back.png")),
        ),
        bevy::sprite::Anchor(Vec2::new(anchor.0, anchor.1)),
        Transform::from_xyz(fc.x, fc.y, crate::surface_objects::depth_z(fc.y + tile_px))
            .with_scale(Vec3::splat(scale)),
    ));
    // _front sorts NATURALLY at the building's front-wall depth: a player
    // whose feet are SOUTH of the wall line draws over it (their head can
    // overlap the facade without being clipped), and a player whose feet
    // cross into the doorway (north of the line) sorts underneath, framed by
    // the jambs. The old +8 "always above the player" lift clipped the
    // 32px sprites' heads a full tile before the doorway. The cost: front-
    // protruding props (awnings, the mechanic's engine) draw under a player
    // standing hard against them — their tiles are solid, so it's marginal.
    commands.spawn((
        DespawnOnExit(PlayState::Exploring),
        Sprite::from_image(
            asset_server.load(format!("{WORLDS_DIR}/buildings3d/{style}_{func}_front.png")),
        ),
        bevy::sprite::Anchor(Vec2::new(anchor.0, anchor.1)),
        Transform::from_xyz(fc.x, fc.y, crate::surface_objects::depth_z(fc.y))
            .with_scale(Vec3::splat(scale)),
    ));
    // South-protruding props (market crates, the mechanic's engine, fuel
    // pumps, the garrison's monument gun) spawn as INDIVIDUAL objects, each
    // depth-sorted at its own ground line — a player weaving between the
    // mechanic's toolbox and armor plate sorts correctly against each.
    // Overhead pieces (porch, hanging sign) carry their SUPPORT line as dy.
    for prop in props {
        commands.spawn((
            DespawnOnExit(PlayState::Exploring),
            Sprite::from_image(asset_server.load(format!(
                "{WORLDS_DIR}/buildings3d/{style}_{func}_prop_{}.png",
                prop.name
            ))),
            bevy::sprite::Anchor(Vec2::new(prop.anchor.0, prop.anchor.1)),
            Transform::from_xyz(
                fc.x,
                fc.y,
                crate::surface_objects::depth_z(fc.y - prop.dy * tile_px),
            )
            .with_scale(Vec3::splat(scale)),
        ));
    }

    // Animated roll-up door (over the facade): overlays _front exactly (same
    // anchor/pos), and rolls open when the player nears. Only door buildings.
    if matches!(
        func,
        "market" | "outfitter" | "bar" | "mechanic" | "fuel_station" | "garrison"
    ) {
        let frames: Vec<Handle<Image>> = (0..4)
            .map(|k| {
                asset_server.load(format!(
                    "{WORLDS_DIR}/buildings3d/{style}_{func}_door{k}.png"
                ))
            })
            .collect();
        commands.spawn((
            DespawnOnExit(PlayState::Exploring),
            Sprite::from_image(frames[0].clone()),
            bevy::sprite::Anchor(Vec2::new(anchor.0, anchor.1)),
            // A hair BENEATH the facade — or beneath the VESTIBULE on cryo
            // buildings, whose airlock (and its door) juts forward of the
            // wall: the panel sits in whichever recess it was baked into.
            Transform::from_xyz(
                fc.x,
                fc.y,
                crate::surface_objects::depth_z(
                    fc.y - props
                        .iter()
                        .find(|p| p.name == "vest")
                        .map_or(0.0, |p| p.dy)
                        * tile_px,
                ) - 0.0004,
            )
            .with_scale(Vec3::splat(scale)),
            BuildingDoor {
                frames,
                door_pos: fc,
                openness: 0.0,
            },
        ));
    }
}

/// An animated roll-up building door. `openness` 0 (closed) → 1 (open) eases
/// toward open when the player is within range; the sprite swaps to the matching
/// baked frame.
#[derive(Component)]
pub(crate) struct BuildingDoor {
    frames: Vec<Handle<Image>>,
    door_pos: Vec2,
    openness: f32,
}

/// Roll building doors open when the player is near, closed when far.
pub(crate) fn animate_building_doors(
    time: Res<Time>,
    walker: Query<&Transform, With<Walker>>,
    mut doors: Query<(&mut BuildingDoor, &mut Sprite)>,
) {
    let Ok(player) = walker.single() else { return };
    let pp = player.translation.truncate();
    const OPEN_RADIUS: f32 = TILE_PX * 2.5;
    for (mut door, mut sprite) in &mut doors {
        let target = if door.door_pos.distance(pp) < OPEN_RADIUS {
            1.0
        } else {
            0.0
        };
        let step = time.delta_secs() * 5.0;
        door.openness += (target - door.openness).clamp(-step, step);
        door.openness = door.openness.clamp(0.0, 1.0);
        let n = door.frames.len();
        if n == 0 {
            continue;
        }
        let frame = ((door.openness * (n - 1) as f32).round() as usize).min(n - 1);
        if sprite.image != door.frames[frame] {
            sprite.image = door.frames[frame].clone();
        }
    }
}

/// Landing pad: the 9 tiles of the 3×3 pad footprint, as (dx, dy) offsets from
/// the centre (used to clear/flatten terrain under the pad).
pub(crate) const PAD_TILES: [(i32, i32, usize); 9] = [
    (-1, 1, 0),
    (0, 1, 1),
    (1, 1, 2),
    (-1, 0, 3),
    (0, 0, 4),
    (1, 0, 5),
    (-1, -1, 6),
    (0, -1, 7),
    (1, -1, 8),
];

/// Find walkable tile positions in the collision data, suitable for placing
/// buildings. Returns positions in tile coordinates, sorted by distance from
/// center, filtered to avoid the center area (where the walker spawns).
pub(crate) fn find_walkable_positions(
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
