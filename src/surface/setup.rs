//! Surface world construction and teardown: terrain, tilemap, buildings, paths.
#[allow(unused_imports)]
use super::*;

use crate::item_universe::ItemUniverse;
use crate::planet_ui::LandedContext;
use crate::{GameLayer, PlayState, Player};

/// Spawn the tilemap, walker, buildings, and set camera zoom on entering Exploring.
#[allow(clippy::too_many_arguments)]
pub(crate) fn setup_surface(
    mut commands: Commands,
    return_from_interior: Option<Res<crate::surface::interiors::ReturnFromInterior>>,
    player_query: Query<Entity, With<Player>>,
    landed_context: Res<LandedContext>,
    item_universe: Res<ItemUniverse>,
    galaxy: Res<crate::galaxy::GalaxyControl>,
    current_system: Res<crate::CurrentStarSystem>,
    mut camera_query: Query<&mut Transform, With<Camera2d>>,
    mut zoom: ResMut<CameraZoom>,
    mut atlas_layouts: ResMut<Assets<TextureAtlasLayout>>,
    asset_server: Res<AssetServer>,
    game_state: Res<crate::game_save::PlayerGameState>,
    mut comms: ResMut<crate::hud::CommsChannel>,
    mut images: ResMut<Assets<Image>>,
    mut character_layers: Option<ResMut<crate::character_compositor::CharacterLayers>>,
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
        crate::embedded_assets::read_to_string(&format!("assets/{WORLDS_DIR}/{filename}")).ok()
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

    let (_col_data, _placed_buildings, door_positions) = if let (Some(lut_data), Some(manifest)) =
        (&lut, &manifest)
    {
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

        // Generate initial terrain+collision for building placement. The
        // biome picks its field generator: organic fBm noise (default) or
        // the designed corridor/room station layout for interiors.
        let generator = manifest
            .biomes
            .get(biome_name)
            .map(|b| b.generator.as_str())
            .unwrap_or("organic");
        let initial_terrain = if generator == "station" {
            crate::station_layout::generate_station_map(map_w, map_h, N_TERRAIN_TYPES, seed)
        } else {
            crate::fbm::generate_terrain_map(
                map_w,
                map_h,
                N_TERRAIN_TYPES,
                FBM_SCALE,
                FBM_OCTAVES,
                FBM_LACUNARITY,
                FBM_GAIN,
                seed,
            )
        };

        // ── Pre-tilemap building placement ─────────────────────────────
        // Find building positions on the initial terrain, then force
        // nearby tiles to walkable terrain and re-clamp before building
        // the tilemap.
        let building_kinds =
            building_kinds_for_planet(&planet_name, &item_universe, system_name, &galaxy);

        let initial_col: Vec<u8> = initial_terrain
            .iter()
            .map(|&t| *collision_codes.get(t as usize).unwrap_or(&0))
            .collect();
        let mut walkable_positions = find_walkable_positions(&initial_col, map_w, map_h, 5.0);
        // Keep only positions within the inner portion of the map (closer
        // to the landing pad) so buildings cluster near the center.
        let max_building_radius = (map_w.min(map_h) as f32 * 0.4).max(15.0);
        let cx = map_w as f32 / 2.0;
        let cy = map_h as f32 / 2.0;
        walkable_positions.retain(|&(x, y)| {
            let dist = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
            dist < max_building_radius
        });
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
        let _ext_atlas_handle: Option<Handle<Image>> =
            bldg_style.map(|s| asset_server.load(format!("{WORLDS_DIR}/{}", s.exterior_atlas)));
        let _ext_layout: Option<Handle<TextureAtlasLayout>> = bldg_manifest.as_ref().map(|m| {
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
                        crate::embedded_assets::read_to_string(&path)
                            .ok()
                            .and_then(|t| ron::from_str(&t).ok())
                    })
                    .collect()
            })
            .unwrap_or_default();
        // Baked 3/4 building sprites (replace the flat exterior tiles, visually).
        let b3d = load_ron("buildings3d_manifest.ron")
            .and_then(|t| ron::from_str::<crate::world_assets::Buildings3dManifest>(&t).ok());

        // Place the landing pad at the map center.
        let pad_cx = map_w / 2;
        let pad_cy = map_h / 2;
        // Fuel station shares the mechanic's "fixed beside the pad" placement,
        // mirrored to the opposite side of the pad. Resolve its footprint up front.
        let fuel_ti = templates
            .iter()
            .position(|t| t.name == kind_template(BuildingKind::FuelStation))
            .unwrap_or(0);
        let (fuel_bw, fuel_bh) = templates
            .get(fuel_ti)
            .map(|t| (t.width, t.height))
            .unwrap_or((4, 4));
        // Mechanic shop: random cardinal direction, 3-6 tiles from pad. The fuel
        // station is placed on the opposite side of the pad from the mechanic.
        let (mech_x, mech_y, fuel_x, fuel_y) = {
            use rand::{Rng, SeedableRng};
            let mech_seed = planet_name
                .bytes()
                .fold(0u64, |acc, b| acc.wrapping_mul(37).wrapping_add(b as u64));
            let mut rng = rand::rngs::StdRng::seed_from_u64(mech_seed);
            let dist = rng.gen_range(3i32..=6);
            // Never place the mechanic SOUTH of the pad: it draws in front and its
            // tall roof extends north over the pad, hiding the ship. Only
            // right / left / north (where the pad stays in front, unobscured).
            let dir = rng.gen_range(0u8..3);
            let (mdx, mdy) = match dir {
                0 => (dist + 2, 0),  // right
                1 => (-dist - 5, 0), // left (offset by the 6-wide mechanic)
                _ => (0, dist + 2),  // up / north
            };
            // Fuel station: opposite side of the pad. The fuel booth is short, so
            // the south case doesn't hide the ship the way the tall mechanic would.
            let (fdx, fdy) = match dir {
                0 => (-dist - fuel_bw as i32 - 1, 0), // mech right -> fuel left
                1 => (dist + 2, 0),                   // mech left  -> fuel right
                _ => (0, -dist - fuel_bh as i32 - 1), // mech north -> fuel south
            };
            (
                (pad_cx as i32 + mdx).max(2) as u32,
                (pad_cy as i32 + mdy).max(2) as u32,
                (pad_cx as i32 + fdx).max(2) as u32,
                (pad_cy as i32 + fdy).max(2) as u32,
            )
        };

        let min_building_spacing = 14_u32;
        let mut placed_buildings: Vec<(u32, u32, u32, u32)> = Vec::new(); // (x, y, w, h)
        // Reserve the pad area (3x3 centered on pad_cx, pad_cy).
        placed_buildings.push((pad_cx - 1, pad_cy - 1, 3, 3));
        // Reserve the mechanic area (6×4 — matches the 3/4 sprite footprint).
        placed_buildings.push((mech_x.saturating_sub(1), mech_y, 6, 4));
        let mut building_assignments: Vec<(BuildingKind, u32, u32, usize)> = Vec::new(); // (kind, x, y, template_idx)
        // Fuel station: fixed beside the pad opposite the mechanic, rather than
        // scattered with the other buildings (skipped in the loop below). Reusing
        // building_assignments/placed_buildings means the existing door, solid-tile,
        // terrain, minimap and interaction code all handle it like any other building.
        placed_buildings.push((fuel_x, fuel_y, fuel_bw, fuel_bh));
        building_assignments.push((BuildingKind::FuelStation, fuel_x, fuel_y, fuel_ti));

        for kind in &building_kinds {
            if *kind == BuildingKind::FuelStation {
                continue; // placed beside the pad opposite the mechanic, above.
            }
            // Semantic: each kind gets the template its 3/4 sprite was built for.
            let ti_opt = templates
                .iter()
                .position(|t| t.name == kind_template(*kind))
                .or(if templates.is_empty() { None } else { Some(0) });
            let tmpl = ti_opt.map(|i| &templates[i]);
            let (bw, bh) = tmpl.map(|t| (t.width, t.height)).unwrap_or((2, 2));
            let spacing = min_building_spacing.max(bw.max(bh) + 2);

            let valid = |x: u32, y: u32| {
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
            };
            // The BAR anchors the port's social life (offers, hires,
            // rumors): of all valid spots, it takes the one nearest the
            // pad so a fresh landing is a short walk from the action.
            let pos = if *kind == BuildingKind::Bar {
                walkable_positions
                    .iter()
                    .filter(|&&(x, y)| valid(x, y))
                    .min_by_key(|&&(x, y)| {
                        let cx = x as i64 + bw as i64 / 2 - pad_cx as i64;
                        let cy = y as i64 + bh as i64 / 2 - pad_cy as i64;
                        cx * cx + cy * cy
                    })
            } else {
                walkable_positions.iter().find(|&&(x, y)| valid(x, y))
            };

            if let Some(&(tx, ty)) = pos {
                placed_buildings.push((tx, ty, bw, bh));
                building_assignments.push((*kind, tx, ty, ti_opt.unwrap_or(0)));
            }
        }

        // ── Constrain terrain: force walkable near buildings + force paths ──
        // Build door positions for pathfinding.
        let mut door_positions: Vec<(BuildingKind, (u32, u32))> = Vec::new();
        door_positions.push((BuildingKind::ShipPad, (pad_cx, pad_cy)));
        door_positions.push((BuildingKind::MechanicShop, (mech_x + 1, mech_y)));
        for &(kind, bx, by, ti) in &building_assignments {
            if let Some(tmpl) = templates.get(ti)
                && let Some(&(dc, dr)) = tmpl.entry_points.first()
            {
                door_positions.push((kind, (bx + dc, by + dr)));
            }
        }

        // Build a set of solid building tiles for pathfinding.
        // Only non-transparent, non-door tiles block movement.
        let mut solid_building_tiles: std::collections::HashSet<(u32, u32)> =
            std::collections::HashSet::new();

        // Template buildings: mark each non-zero tile as solid (except doors).
        for &(kind, bx, by, ti) in &building_assignments {
            if let Some(tmpl) = templates.get(ti) {
                let door_set: std::collections::HashSet<(u32, u32)> = door_tiles_for(kind, tmpl)
                    .into_iter()
                    .map(|(dc, dr)| (bx + dc, by + dr))
                    .collect();
                for row in 0..tmpl.height {
                    for col in 0..tmpl.width {
                        // Guard: template RON with a short tiles grid must
                        // not panic every surface entry.
                        let Some(tile_idx) = tmpl
                            .tiles
                            .get(row as usize)
                            .and_then(|r| r.get(col as usize))
                            .copied()
                        else {
                            continue;
                        };
                        if tile_idx == 0 {
                            continue;
                        } // transparent
                        let pos = (bx + col, by + row);
                        if door_set.contains(&pos) {
                            continue;
                        } // door
                        solid_building_tiles.insert(pos);
                    }
                }
            }
        }

        // Mechanic building: pathfinding solids must MATCH the physics
        // colliders (6 cols × 4 rows centred on the sprite, garage doors
        // open) — the old 4×3 patch left 14 tiles A*-walkable but
        // physically walled, and NPCs ground against them.
        for row in 0..4u32 {
            for col in 0..6u32 {
                let tx_i = mech_x as i32 - 1 + col as i32;
                if tx_i < 0 {
                    continue;
                }
                let tx = tx_i as u32;
                let ty = mech_y + row;
                let is_garage = row == 0 && (tx == mech_x + 1 || tx == mech_x + 2);
                if !is_garage {
                    solid_building_tiles.insert((tx, ty));
                }
            }
        }

        // The repair engine sits front-left of the door. Mark its tiles
        // impassable just like the building tiles so character pathfinding
        // routes around it. TWO rows: its visible base reaches ~1.6 screen
        // tiles south of the wall (the gun's plinth is shallower — one row).
        let engine_tiles = [
            (mech_x.saturating_sub(1), mech_y.saturating_sub(1)),
            (mech_x, mech_y.saturating_sub(1)),
            (mech_x.saturating_sub(1), mech_y.saturating_sub(2)),
            (mech_x, mech_y.saturating_sub(2)),
        ];
        for &t in &engine_tiles {
            solid_building_tiles.insert(t);
        }

        // The garrison's monument gun (plinth + cannon) blocks a 2x1 row
        // front-left of its door — too big to walk through, same treatment
        // (and same one-row depth) as the mechanic's engine.
        let gun_tiles: Vec<(u32, u32)> = building_assignments
            .iter()
            .filter(|(kind, ..)| *kind == BuildingKind::Garrison)
            .flat_map(|&(_, bx, by, _)| {
                [(bx, by.saturating_sub(1)), (bx + 1, by.saturating_sub(1))]
            })
            .collect();
        for &t in &gun_tiles {
            solid_building_tiles.insert(t);
        }

        // Apply terrain constraints + ensure connectivity, starting from the
        // same base field used for building placement (organic or station).
        let generated = crate::surface_terrain::generate_constrained_terrain(
            map_w,
            map_h,
            initial_terrain.clone(),
            &collision_codes,
            &movement_costs,
            &placed_buildings,
            &door_positions,
            &solid_building_tiles,
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
        let surface_layers = CollisionLayers::new(
            GameLayer::Surface,
            [GameLayer::Surface, GameLayer::Character],
        );
        crate::world_assets::spawn_collision_entities(
            &mut commands,
            &col_asset,
            &terrain_flat,
            &movement_costs,
            tile_px,
            map_origin,
            surface_layers,
        );

        // ── Spawn landing pad — the baked 3/4 pad sprite, laid flat below the
        // player (it's walkable: you land on it and walk to the take-off sensor) ──
        if let Some(spr) = b3d.as_ref().and_then(|m| {
            m.sprites
                .iter()
                .find(|s| s.style == style_name && s.func == "pad")
        }) {
            let fc = footprint_front_center(pad_cx - 1, pad_cy - 1, spr.w, map_w, map_h, tile_px);
            let scale = tile_px / b3d.as_ref().map(|m| m.px_per_tile).unwrap_or(tile_px);
            commands.spawn((
                DespawnOnExit(PlayState::Exploring),
                Sprite::from_image(
                    asset_server.load(format!("{WORLDS_DIR}/buildings3d/{}_pad.png", style_name)),
                ),
                bevy::sprite::Anchor(Vec2::new(spr.anchor.0, spr.anchor.1)),
                Transform::from_xyz(fc.x, fc.y, -9.0).with_scale(Vec3::splat(scale)),
            ));
        }
        // The WHOLE pad is the take-off sensor: standing anywhere on it,
        // E means "launch" — it must outrank any chatty NPC loitering
        // there (npc chat already defers to a nearby building).
        {
            let world_pos = tile_to_world(pad_cx, pad_cy, map_w, map_h, tile_px);
            commands.spawn((
                DespawnOnExit(PlayState::Exploring),
                Building {
                    kind: BuildingKind::ShipPad,
                },
                Sensor,
                RigidBody::Static,
                Collider::rectangle(tile_px * 3.4, tile_px * 3.4),
                CollisionEventsEnabled,
                CollisionLayers::new(
                    GameLayer::Surface,
                    [GameLayer::Surface, GameLayer::Character],
                ),
                Transform::from_xyz(world_pos.x, world_pos.y, -9.0),
            ));
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
        let _ = (mech_atlas_handle.as_ref(), mech_layout.as_ref(), mech_z);
        {
            // Collision footprint matches the 6-wide × 4-deep 3/4 sprite (centred
            // on mech_x+1.5, the sprite's anchor). All solid except the two
            // centre-front garage-door tiles, so the whole building blocks.
            let base_col = mech_x as i32 - 1; // 6 cols: mech_x-1 .. mech_x+4
            for row in 0..4i32 {
                for col in 0..6i32 {
                    let tx_i = base_col + col;
                    let ty_i = mech_y as i32 + row;
                    if tx_i < 0 || ty_i < 0 || tx_i as u32 >= map_w || ty_i as u32 >= map_h {
                        continue;
                    }
                    let tx = tx_i as u32;
                    let ty = ty_i as u32;
                    let world_pos = tile_to_world(tx, ty, map_w, map_h, tile_px);
                    let is_garage = row == 0 && (tx == mech_x + 1 || tx == mech_x + 2);

                    let mut entity = commands.spawn((
                        DespawnOnExit(PlayState::Exploring),
                        Transform::from_xyz(world_pos.x, world_pos.y, mech_z),
                    ));
                    if is_garage {
                        entity.insert((
                            Building {
                                kind: BuildingKind::MechanicShop,
                            },
                            DoorSprite {
                                walker_was_behind: None,
                            },
                            Sensor,
                            RigidBody::Static,
                            Collider::rectangle(tile_px, tile_px),
                            CollisionEventsEnabled,
                            CollisionLayers::new(
                                GameLayer::Surface,
                                [GameLayer::Surface, GameLayer::Character],
                            ),
                        ));
                    } else {
                        entity.insert((
                            RigidBody::Static,
                            Collider::rectangle(tile_px, tile_px),
                            CollisionLayers::new(
                                GameLayer::Surface,
                                [GameLayer::Surface, GameLayer::Character],
                            ),
                        ));
                    }
                }
            }
            // 3/4 mechanic sprite (6-wide; overhangs the 4×3 collision footprint).
            if let Some(spr) = b3d.as_ref().and_then(|m| {
                m.sprites
                    .iter()
                    .find(|s| s.style == style_name && s.func == "mechanic")
            }) {
                let fc = footprint_front_center(mech_x, mech_y, 4, map_w, map_h, tile_px);
                let scale = tile_px / b3d.as_ref().map(|m| m.px_per_tile).unwrap_or(tile_px);
                spawn_building_3d(
                    &mut commands,
                    &asset_server,
                    style_name,
                    "mechanic",
                    (spr.anchor.0, spr.anchor.1),
                    fc,
                    scale,
                    tile_px,
                    &spr.props,
                );
            }
            // The repair engine sits front-left of the door. Its tiles were marked
            // impassable up front (see `engine_tiles`) so pathfinding routes around
            // it; drop a matching per-tile collider on each — exactly like a building
            // tile — so the player is physically blocked where it's solid.
            for &(tx, ty) in &engine_tiles {
                let wp = tile_to_world(tx, ty, map_w, map_h, tile_px);
                commands.spawn((
                    DespawnOnExit(PlayState::Exploring),
                    RigidBody::Static,
                    Collider::rectangle(tile_px, tile_px),
                    CollisionLayers::new(
                        GameLayer::Surface,
                        [GameLayer::Surface, GameLayer::Character],
                    ),
                    Transform::from_xyz(wp.x, wp.y, 0.0),
                ));
            }
            // The garrison's monument gun blocks like the engine does.
            for &(tx, ty) in &gun_tiles {
                let wp = tile_to_world(tx, ty, map_w, map_h, tile_px);
                commands.spawn((
                    DespawnOnExit(PlayState::Exploring),
                    RigidBody::Static,
                    Collider::rectangle(tile_px, tile_px),
                    CollisionLayers::new(
                        GameLayer::Surface,
                        [GameLayer::Surface, GameLayer::Character],
                    ),
                    Transform::from_xyz(wp.x, wp.y, 0.0),
                ));
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
            let (bw, _bh) = tmpl.map(|t| (t.width, t.height)).unwrap_or((2, 2));

            // Depth-sort: all tiles in the building share z based on the floor.
            let bldg_floor_world = tile_to_world(anchor_tx, anchor_ty, map_w, map_h, tile_px);
            let bldg_z = crate::surface_objects::depth_z(bldg_floor_world.y - tile_px * 0.5);

            if let Some(tmpl) = tmpl {
                // Footprint collision + the door sensor stay on the grid; the flat
                // exterior tiles are no longer drawn (the 3/4 sprite replaces them).
                // Every footprint tile is solid except the door tiles — the 3/4
                // sprite is a solid building with a centred doorway. Doors come
                // from the template's `entry_points` (not tile values), so the
                // collision matches the floorplan + sprite exactly.
                let door_set: std::collections::HashSet<(u32, u32)> =
                    door_tiles_for(kind, tmpl).into_iter().collect();
                for row in 0..tmpl.height {
                    for col in 0..tmpl.width {
                        let tx = anchor_tx + col;
                        let ty = anchor_ty + row;
                        let world_pos = tile_to_world(tx, ty, map_w, map_h, tile_px);

                        let mut entity = commands.spawn((
                            DespawnOnExit(PlayState::Exploring),
                            Transform::from_xyz(world_pos.x, world_pos.y, bldg_z),
                        ));

                        if door_set.contains(&(col, row)) {
                            entity.insert((
                                Building { kind },
                                DoorSprite {
                                    walker_was_behind: None,
                                },
                                Sensor,
                                RigidBody::Static,
                                Collider::rectangle(tile_px, tile_px),
                                CollisionEventsEnabled,
                                CollisionLayers::new(
                                    GameLayer::Surface,
                                    [GameLayer::Surface, GameLayer::Character],
                                ),
                            ));
                        } else {
                            entity.insert((
                                RigidBody::Static,
                                Collider::rectangle(tile_px, tile_px),
                                CollisionLayers::new(
                                    GameLayer::Surface,
                                    [GameLayer::Surface, GameLayer::Character],
                                ),
                            ));
                        }
                    }
                }

                // The 3/4 building, as two depth layers (lower behind the player,
                // _top over the player) so they read standing in the doorway.
                if let Some(spr) = b3d.as_ref().and_then(|m| {
                    let func = kind_func(kind);
                    m.sprites
                        .iter()
                        .find(|s| s.style == style_name && s.func == func)
                }) {
                    let func = kind_func(kind);
                    let fc =
                        footprint_front_center(anchor_tx, anchor_ty, spr.w, map_w, map_h, tile_px);
                    let scale = tile_px / b3d.as_ref().map(|m| m.px_per_tile).unwrap_or(tile_px);
                    spawn_building_3d(
                        &mut commands,
                        &asset_server,
                        style_name,
                        func,
                        (spr.anchor.0, spr.anchor.1),
                        fc,
                        scale,
                        tile_px,
                        &spr.props,
                    );
                    // The garrison flies the CONTROLLING faction's colors: a
                    // grayscale cloth sprite tinted at runtime, pinned to the
                    // baked flagpole (model: pole at (w/2+0.55, -d/2+0.3),
                    // finial z≈4.2; screen offset = (x, Δdepth·sin50° +
                    // z·cos50°) in tiles — see build_garrison in
                    // scripts/ship3d/buildings3d.py).
                    if kind == BuildingKind::Garrison
                        && let Some(color) = crate::galaxy::effective_planet_faction(
                            &galaxy,
                            &item_universe,
                            &planet_name,
                        )
                        .and_then(|f| item_universe.factions.get(&f))
                        .map(|fd| fd.color)
                    {
                        let offset = Vec2::new(4.25, 2.61) * tile_px;
                        commands.spawn((
                            DespawnOnExit(PlayState::Exploring),
                            Sprite {
                                image: asset_server
                                    .load(format!("{WORLDS_DIR}/buildings3d/flag.png")),
                                color: Color::srgb_u8(color[0], color[1], color[2]),
                                custom_size: Some(Vec2::new(1.3, 0.84) * tile_px),
                                ..default()
                            },
                            // The cloth rides just above the pole, which
                            // is its own prop object at its own depth.
                            Transform::from_xyz(
                                fc.x + offset.x,
                                fc.y + offset.y,
                                crate::surface_objects::depth_z(
                                    fc.y - spr
                                        .props
                                        .iter()
                                        .find(|p| p.name == "flag")
                                        .map_or(0.0, |p| p.dy)
                                        * tile_px,
                                ) + 0.0004,
                            ),
                        ));
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
                    CollisionLayers::new(
                        GameLayer::Surface,
                        [GameLayer::Surface, GameLayer::Character],
                    ),
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
            let set_px = |pixels: &mut [u8], x: u32, y: u32, color: [u8; 3]| {
                crate::surface::minimap_set_px(pixels, map_w, map_h, x, y, color);
            };

            // Render full building footprints using per-tile colors.
            let tile_colors = bldg_style
                .map(|s| &s.ext_tile_colors)
                .filter(|c| !c.is_empty());
            for &(_, bx, by, ti) in &building_assignments {
                if let Some(tmpl) = templates.get(ti) {
                    for row in 0..tmpl.height {
                        for col in 0..tmpl.width {
                            // Guard: template RON with a short tiles grid must
                            // not panic every surface entry.
                            let Some(tile_idx) = tmpl
                                .tiles
                                .get(row as usize)
                                .and_then(|r| r.get(col as usize))
                                .copied()
                            else {
                                continue;
                            };
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

            let img = crate::surface::minimap_image(pixels, map_w, map_h);

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
            let names: Vec<String> = manifest
                .biomes
                .get(biome_name)
                .map(|b| b.terrains.iter().map(|t| t.name.clone()).collect())
                .unwrap_or_default();
            commands.insert_resource(FootstepData {
                terrains: footstep_terrains,
                names,
                terrain_map: terrain_flat.clone(),
                map_w,
                map_h,
            });
        }

        // ── Setup civilian NPCs ─────────────────────────────────────────
        crate::surface_civilians::setup_civilians(&mut commands, seed);

        // ── Spawn landscape objects (plants, creatures, etc.) ─────────
        {
            let terrain_names: Vec<String> = manifest
                .biomes
                .get(biome_name)
                .map(|b| b.terrains.iter().map(|t| t.name.clone()).collect())
                .unwrap_or_default();
            // Keep objects off the footprints AND a clearance band — especially
            // the 2 tiles in front (south), where a tall object would overlap the
            // building facade — plus 1 tile on the other sides.
            let object_exclusion: Vec<(u32, u32, u32, u32)> = placed_buildings
                .iter()
                .map(|&(bx, by, bw, bh)| {
                    let nx = bx.saturating_sub(1);
                    let ny = by.saturating_sub(2);
                    (nx, ny, bw + 2, (by + bh + 1).saturating_sub(ny))
                })
                .collect();
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
                &object_exclusion,
            );

            // ── Setup roaming fauna (deer, rabbit, …) ────────────────
            crate::surface_fauna::setup_fauna(
                &mut commands,
                &asset_server,
                &mut atlas_layouts,
                &terrain_flat,
                &terrain_names,
                biome_name,
                map_w,
                map_h,
                seed,
                PlayState::Exploring,
            );
        }

        (col_data, placed_buildings, door_positions)
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
        (vec![0u8; n], Vec::new(), Vec::new())
    };

    // Spawn on the landing pad (map center) — unless the player just walked
    // out of a building's interior, in which case they appear at ITS door.
    let mut spawn_pos = tile_to_world(map_w / 2, map_h / 2, map_w, map_h, tile_px);
    if let Some(ret) = return_from_interior {
        if let Some(&(_, (dtx, dty))) = door_positions.iter().find(|(k, _)| *k == ret.0) {
            spawn_pos = tile_to_world(dtx, dty.saturating_sub(1), map_w, map_h, tile_px);
        }
        commands.remove_resource::<crate::surface::interiors::ReturnFromInterior>();
    }

    let Some(layers) = character_layers.as_deref_mut() else {
        eprintln!("[surface] WARNING: character layers unavailable — no walker spawned");
        return;
    };
    if crate::surface::spawn_walker_at(
        &mut commands,
        layers,
        &mut images,
        &game_state.avatar,
        spawn_pos,
        PlayState::Exploring,
    )
    .is_none()
    {
        eprintln!("[surface] WARNING: avatar sheet not composited — no walker spawned");
        return;
    }

    // "Press E" prompt is now shown via the comms ticker (no floating text).

    if let Ok(mut cam_tf) = camera_query.single_mut() {
        // Bug 4: get_single_mut
        cam_tf.translation = Vec3::new(0.0, 0.0, cam_tf.translation.z);
    }

    zoom.target = SURFACE_CAMERA_SCALE;
}

/// Restore the player ship and reset camera zoom on exiting Exploring.
pub(crate) fn teardown_surface(
    mut commands: Commands,
    player_query: Query<Entity, With<Player>>,
    mut zoom: ResMut<CameraZoom>,
    mut nearby: ResMut<NearbyBuilding>,
    mut active_ui: ResMut<ActiveBuildingUI>,
    mut terrain_speed: ResMut<TerrainSpeedModifier>,
    mut npc_chat: ResMut<crate::surface_npc_chat::NpcChatState>,
) {
    // Show the player ship again.
    if let Ok(ship_entity) = player_query.single() {
        commands.entity(ship_entity).insert(Visibility::Inherited);
    }

    // Reset surface state.
    *nearby = NearbyBuilding::default();
    active_ui.0 = None;
    terrain_speed.0 = 1.0;
    *npc_chat = crate::surface_npc_chat::NpcChatState::default();
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
