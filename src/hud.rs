use bevy::prelude::*;

use crate::asteroids::Asteroid;
use crate::item_universe::ItemUniverse;
use crate::pickups::Pickup;
use crate::planets::Planet;
use crate::ship::Target;
use crate::utils::safe_despawn;
use crate::{CurrentStarSystem, PlayState, Player, Ship};

const RADAR_SIZE: f32 = 144.0;
const DOT_SIZE: f32 = 4.0;
// How far from center to place the chevron tip (px), leaving room for border
const CHEVRON_EDGE: f32 = 52.0;

#[derive(Component)]
struct HealthBarFill;

#[derive(Component)]
struct CreditsText;

#[derive(Component)]
struct CargoText;

#[derive(Component)]
struct TargetText;

#[derive(Component)]
struct SecondaryWeaponText;

#[derive(Component)]
struct CommsContainer;

#[derive(Component)]
struct CommsText;

#[derive(Component)]
struct RadarDisplay;

/// Marks HUD elements that should be hidden during surface exploration.
#[derive(Component)]
struct SpaceOnlyHud;

/// Marks HUD elements that should only be visible during surface exploration.
#[derive(Component)]
struct SurfaceOnlyHud;

/// Marks the mini-map image node so we can update it.
#[derive(Component)]
struct MiniMapImage;

/// Marks the player dot on the mini-map.
#[derive(Component)]
struct MiniMapPlayerDot;

#[derive(Component)]
struct RadarDot;

#[derive(Resource)]
struct RadarEntity(Entity);

const COMMS_SCROLL_SPEED: f32 = 40.0;
const COMMS_INITIAL_PAUSE: f32 = 2.0;
const COMMS_LOOP_PAUSE: f32 = 1.0;

/// Resource holding the comms ticker state. Use [`CommsChannel::send`] to
/// display a message on the HUD ticker-tape.
#[derive(Resource, Default)]
pub struct CommsChannel {
    pub message: String,
    scroll_offset: f32,
    pause_timer: f32,
    /// Set to `true` by the ticker after the first full scroll pass (or
    /// immediately if the text fits without scrolling).  External systems
    /// can poll this to know when to refresh the message.
    pub cycle_complete: bool,
}

impl CommsChannel {
    pub fn send(&mut self, msg: impl Into<String>) {
        self.message = msg.into();
        self.scroll_offset = 0.0;
        self.pause_timer = COMMS_INITIAL_PAUSE;
        self.cycle_complete = false;
    }
}

#[derive(Resource)]
pub struct RadarConfig {
    pub range: f32,
}

pub struct HudPlugin {
    pub radar_range: f32,
}

impl Default for HudPlugin {
    fn default() -> Self {
        Self {
            radar_range: 2000.0,
        }
    }
}

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(RadarConfig {
            range: self.radar_range,
        })
        .init_resource::<CommsChannel>()
        .add_systems(Startup, spawn_hud)
        .add_systems(
            Update,
            (
                update_health_bar,
                update_cargo_credits,
                update_comms_ticker,
            ),
        )
        .add_systems(
            Update,
            (
                update_target_display,
                update_secondary_weapon,
            )
                .run_if(not(in_state(crate::PlayState::Exploring))),
        )
        .add_systems(
            Update,
            (update_radar_dots, draw_target_reticle).run_if(in_state(PlayState::Flying)),
        )
        .add_systems(
            Update,
            toggle_space_hud_visibility
                .run_if(state_changed::<PlayState>),
        )
        .add_systems(
            Update,
            update_minimap_player_dot
                .run_if(in_state(PlayState::Exploring)),
        );
    }
}

fn spawn_hud(mut commands: Commands) {
    let mut radar_id = Entity::PLACEHOLDER;

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                right: Val::Px(16.0),
                top: Val::Px(16.0),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Stretch,
                row_gap: Val::Px(6.0),
                width: Val::Px(144.0),
                ..default()
            },
            Transform::default(),
        ))
        .with_children(|root| {
            // ── Radar ────────────────────────────────────────────────────
            radar_id = root
                .spawn((
                    RadarDisplay,
                    SpaceOnlyHud,
                    Node {
                        width: Val::Px(RADAR_SIZE),
                        height: Val::Px(RADAR_SIZE),
                        border: UiRect::all(Val::Px(2.0)),
                        border_radius: BorderRadius::all(Val::Px(72.0)), // full circle
                        overflow: Overflow::clip(),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.0, 0.06, 0.0, 0.85)),
                    BorderColor::all(Color::srgb(0.1, 0.8, 0.3)),
                    // Required so children with Transform have a GlobalTransform ancestor.
                    Transform::default(),
                ))
                .id();

            // ── Mini-map (surface only, hidden by default) ──────────────
            root.spawn((
                SurfaceOnlyHud,
                Node {
                    width: Val::Px(RADAR_SIZE),
                    height: Val::Px(RADAR_SIZE),
                    border: UiRect::all(Val::Px(2.0)),
                    overflow: Overflow::clip(),
                    display: Display::None,
                    ..default()
                },
                BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.85)),
                BorderColor::all(Color::srgb(0.3, 0.7, 0.9)),
                Visibility::Hidden,
            ))
            .with_children(|map_root| {
                // The mini-map image (updated when entering Exploring).
                map_root.spawn((
                    MiniMapImage,
                    ImageNode::default(),
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Percent(100.0),
                        ..default()
                    },
                ));
                // Player position dot.
                map_root.spawn((
                    MiniMapPlayerDot,
                    Node {
                        position_type: PositionType::Absolute,
                        width: Val::Px(6.0),
                        height: Val::Px(6.0),
                        border_radius: BorderRadius::all(Val::Px(3.0)),
                        left: Val::Px(RADAR_SIZE / 2.0 - 3.0),
                        top: Val::Px(RADAR_SIZE / 2.0 - 3.0),
                        ..default()
                    },
                    BackgroundColor(Color::srgb(1.0, 1.0, 0.2)),
                ));
            });

            // ── Health bar ───────────────────────────────────────────────
            root.spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(18.0),
                    border: UiRect::all(Val::Px(2.0)),
                    padding: UiRect::all(Val::Px(2.0)),
                    border_radius: BorderRadius::all(Val::Px(4.0)),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.15, 0.0, 0.0, 0.85)),
                BorderColor::all(Color::srgb(0.8, 0.15, 0.15)),
            ))
            .with_children(|bar| {
                bar.spawn((
                    HealthBarFill,
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Percent(100.0),
                        border_radius: BorderRadius::all(Val::Px(2.0)),
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.85, 0.15, 0.15)),
                ));
            });

            // ── Secondary weapon ─────────────────────────────────────────
            root.spawn((
                SpaceOnlyHud,
                Node {
                    padding: UiRect::axes(Val::Px(4.0), Val::Px(2.0)),
                    border_radius: BorderRadius::all(Val::Px(3.0)),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.6)),
            ))
            .with_children(|w| {
                w.spawn((
                    SecondaryWeaponText,
                    Text::new("Secondary: None"),
                    TextFont {
                        font_size: 13.0,
                        ..default()
                    },
                    TextColor(Color::srgb(0.5, 0.5, 0.55)),
                ));
            });

            // ── Target ───────────────────────────────────────────────────
            root.spawn((
                SpaceOnlyHud,
                Node {
                    padding: UiRect::axes(Val::Px(4.0), Val::Px(2.0)),
                    border_radius: BorderRadius::all(Val::Px(3.0)),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.6)),
            ))
            .with_children(|w| {
                w.spawn((
                    TargetText,
                    Text::new("Target: None"),
                    TextFont {
                        font_size: 13.0,
                        ..default()
                    },
                    TextColor(Color::srgb(1.0, 0.5, 0.2)),
                ));
            });

            // ── Comms ticker ─────────────────────────────────────────────
            root.spawn((
                CommsContainer,
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(18.0),
                    padding: UiRect::axes(Val::Px(4.0), Val::Px(2.0)),
                    border_radius: BorderRadius::all(Val::Px(3.0)),
                    overflow: Overflow::clip(),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.0, 0.04, 0.0, 0.7)),
            ))
            .with_children(|w| {
                w.spawn((
                    CommsText,
                    Text::new(""),
                    TextFont {
                        font_size: 12.0,
                        ..default()
                    },
                    TextColor(Color::srgb(0.2, 1.0, 0.4)),
                    TextLayout::new_with_no_wrap(),
                    Node {
                        position_type: PositionType::Absolute,
                        left: Val::Px(4.0),
                        top: Val::Px(2.0),
                        ..default()
                    },
                ));
            });

            // ── Credits ──────────────────────────────────────────────────
            root.spawn((
                Node {
                    padding: UiRect::axes(Val::Px(4.0), Val::Px(2.0)),
                    border_radius: BorderRadius::all(Val::Px(3.0)),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.6)),
            ))
            .with_children(|w| {
                w.spawn((
                    CreditsText,
                    Text::new("Credits: 0"),
                    TextFont {
                        font_size: 13.0,
                        ..default()
                    },
                    TextColor(Color::srgb(0.9, 0.85, 0.3)),
                ));
            });

            // ── Cargo space ───────────────────────────────────────────────
            root.spawn((
                Node {
                    padding: UiRect::axes(Val::Px(4.0), Val::Px(2.0)),
                    border_radius: BorderRadius::all(Val::Px(3.0)),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.6)),
            ))
            .with_children(|w| {
                w.spawn((
                    CargoText,
                    Text::new("Free: 10"),
                    TextFont {
                        font_size: 13.0,
                        ..default()
                    },
                    TextColor(Color::srgb(0.6, 0.85, 0.9)),
                ));
            });
        });

    commands.insert_resource(RadarEntity(radar_id));
}

fn update_health_bar(
    player_query: Query<&Ship, With<Player>>,
    mut fill_query: Query<&mut Node, With<HealthBarFill>>,
) {
    let Ok(ship) = player_query.single() else {
        return;
    };
    let Ok(mut node) = fill_query.single_mut() else {
        return;
    };
    let pct = (ship.health as f32 / ship.data.max_health as f32 * 100.0).clamp(0.0, 100.0);
    node.width = Val::Percent(pct);
}

fn update_radar_dots(
    mut commands: Commands,
    radar: Res<RadarEntity>,
    config: Res<RadarConfig>,
    time: Res<Time>,
    player_query: Query<(Entity, &Transform, &Ship), With<Player>>,
    ships_query: Query<(Entity, &Transform, &Ship), (With<Ship>, Without<Player>)>,
    planets_query: Query<(Entity, &Transform), With<Planet>>,
    dots_query: Query<Entity, With<RadarDot>>,
) {
    let Ok((player_entity, player_tf, player_ship)) = player_query.single() else {
        return;
    };
    let player_pos = player_tf.translation.truncate();
    let half = RADAR_SIZE / 2.0;

    // Targeted entity for blink effect
    let targeted_entity: Option<Entity> = match &player_ship.nav_target {
        Some(Target::Ship(e)) => Some(*e),
        Some(Target::Planet(e)) => Some(*e),
        _ => None,
    };

    // Blink: visible for ~0.4s, hidden for ~0.4s
    let blink_visible = (time.elapsed_secs() * 2.5).floor() as i32 % 2 == 0;

    // Collect planet data before the with_children closure
    let planet_data: Vec<(Entity, Vec2)> = planets_query
        .iter()
        .map(|(e, tf)| (e, tf.translation.truncate()))
        .collect();

    for entity in dots_query.iter() {
        safe_despawn(&mut commands, entity);
    }

    commands.entity(radar.0).with_children(|parent| {
        // ── Player indicator (^) at radar center ─────────────────────────
        // Rotate ^ to match the player's heading. ^ points screen-up by
        // default, so we compute the screen-space angle of the forward
        // vector and adjust for the π/2 offset vs. the > chevron formula.
        let forward = (player_tf.rotation * Vec3::Y).truncate();
        let indicator_angle = -forward.y.atan2(forward.x) + std::f32::consts::FRAC_PI_2;
        parent
            .spawn((
                RadarDot,
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Px(half),
                    top: Val::Px(half),
                    ..default()
                },
                UiTransform::from_rotation(Rot2::radians(indicator_angle)),
            ))
            .with_children(|inner| {
                inner.spawn((
                    RadarDot,
                    Text::new("^"),
                    TextFont {
                        font_size: 12.0,
                        ..default()
                    },
                    TextColor(Color::srgb(0.0, 1.0, 0.3)),
                    Node {
                        position_type: PositionType::Absolute,
                        left: Val::Px(-4.0),
                        top: Val::Px(-6.0),
                        ..default()
                    },
                ));
            });

        // ── Ships ────────────────────────────────────────────────────────
        for (ship_entity, ship_tf, ship) in ships_query.iter() {
            let offset = ship_tf.translation.truncate() - player_pos;
            if offset.length() > config.range {
                continue;
            }

            let is_targeted = targeted_entity == Some(ship_entity);
            if is_targeted && !blink_visible {
                continue; // blink by skipping this frame
            }

            let rx = half + (offset.x / config.range) * half - DOT_SIZE / 2.0;
            let ry = half - (offset.y / config.range) * half - DOT_SIZE / 2.0;

            let targets_player =
                matches!(&ship.weapons_target, Some(Target::Ship(e)) if *e == player_entity);
            let color = if targets_player {
                Color::srgb(1.0, 0.15, 0.15) // red: hostile
            } else {
                Color::WHITE
            };

            parent.spawn((RadarDot, dot_bundle(rx, ry, color)));
        }

        // ── Planets (blue dots) ──────────────────────────────────────────
        let mut any_planet_visible = false;
        for &(planet_entity, pos) in &planet_data {
            let offset = pos - player_pos;
            if offset.length() > config.range {
                continue;
            }
            any_planet_visible = true;

            let is_targeted = targeted_entity == Some(planet_entity);
            if is_targeted && !blink_visible {
                continue; // blink by skipping this frame
            }

            let rx = half + (offset.x / config.range) * half - DOT_SIZE / 2.0;
            let ry = half - (offset.y / config.range) * half - DOT_SIZE / 2.0;
            parent.spawn((RadarDot, dot_bundle(rx, ry, Color::srgb(0.3, 0.5, 1.0))));
        }

        // ── Off-radar planet chevron ─────────────────────────────────────
        // When no planets appear on the radar, show a ">" at the edge that
        // is rotated to point toward the mean planet location.
        if !any_planet_visible && !planet_data.is_empty() {
            let mean: Vec2 =
                planet_data.iter().map(|(_, p)| *p).sum::<Vec2>() / planet_data.len() as f32;
            let offset = mean - player_pos;
            if offset.length() > f32::EPSILON {
                let dir = offset.normalize();
                let cx = half + dir.x * CHEVRON_EDGE;
                let cy = half - dir.y * CHEVRON_EDGE; // screen Y is flipped

                // Bevy's from_rotation_z is CCW in world space (+Y up), which
                // renders CW on screen (+Y down). Negate to get the correct
                // screen-space visual direction.
                let angle = -dir.y.atan2(dir.x);

                // Use a zero-sized container node as the rotation pivot, then
                // offset the ">" text inside it so its glyph centre sits at
                // (cx, cy). This keeps the transform anchor at (cx, cy) rather
                // than at the text's top-left corner.
                parent
                    .spawn((
                        RadarDot,
                        Node {
                            position_type: PositionType::Absolute,
                            left: Val::Px(cx),
                            top: Val::Px(cy),
                            ..default()
                        },
                        UiTransform::from_rotation(Rot2::radians(angle)),
                    ))
                    .with_children(|inner| {
                        inner.spawn((
                            RadarDot,
                            Text::new(">"),
                            TextFont {
                                font_size: 16.0,
                                ..default()
                            },
                            TextColor(Color::srgb(0.4, 0.6, 1.0)),
                            Node {
                                position_type: PositionType::Absolute,
                                left: Val::Px(-5.0),
                                top: Val::Px(-8.0),
                                ..default()
                            },
                        ));
                    });
            }
        }
    });
}

fn update_cargo_credits(
    player_query: Query<&Ship, With<Player>>,
    mut credits_query: Query<&mut Text, (With<CreditsText>, Without<CargoText>)>,
    mut cargo_query: Query<&mut Text, (With<CargoText>, Without<CreditsText>)>,
) {
    let Ok(ship) = player_query.single() else {
        return;
    };
    if let Ok(mut text) = credits_query.single_mut() {
        **text = format!("Credits: {}", ship.credits);
    }
    if let Ok(mut text) = cargo_query.single_mut() {
        let free = ship.remaining_cargo_space();
        **text = format!("Free: {}/{}", free, ship.data.cargo_space);
    }
}

fn update_secondary_weapon(
    player_query: Query<&Ship, With<Player>>,
    mut text_query: Query<&mut Text, With<SecondaryWeaponText>>,
    item_universe: Res<ItemUniverse>,
) {
    let Ok(ship) = player_query.single() else {
        return;
    };
    let Ok(mut text) = text_query.single_mut() else {
        return;
    };
    **text = match &ship.weapon_systems.selected_secondary {
        Some(name) => {
            if let Some(ws) = ship.weapon_systems.secondary.get(name) {
                let display = item_universe
                    .outfitter_items
                    .get(name)
                    .map(|i| i.display_name())
                    .unwrap_or(name);
                match ws.ammo_quantity {
                    Some(qty) => format!("Sec: {} ({})", display, qty),
                    None => format!("Sec: {}", display),
                }
            } else {
                "Secondary: None".to_string()
            }
        }
        None => "Secondary: None".to_string(),
    };
}

fn update_target_display(
    player_query: Query<(Entity, &Ship), With<Player>>,
    ships_query: Query<&Ship, Without<Player>>,
    mission_targets: Query<&crate::missions::MissionTarget>,
    asteroids_query: Query<&crate::asteroids::Asteroid>,
    planets_query: Query<&Planet>,
    pickups_query: Query<&crate::pickups::Pickup>,
    current_system: Res<crate::CurrentStarSystem>,
    item_universe: Res<ItemUniverse>,
    mut text_query: Query<&mut Text, With<TargetText>>,
) {
    let Ok((player_entity, player_ship)) = player_query.single() else {
        return;
    };
    let Ok(mut text) = text_query.single_mut() else {
        return;
    };
    **text = match &player_ship.nav_target {
        Some(Target::Ship(entity)) => {
            if let Ok(target_ship) = ships_query.get(*entity) {
                let is_hostile =
                    matches!(&target_ship.weapons_target, Some(Target::Ship(e)) if *e == player_entity);
                // Prefer mission target display name over generic ship type.
                let display = mission_targets
                    .get(*entity)
                    .map(|mt| mt.display_name.as_str())
                    .unwrap_or(&target_ship.data.display_name);
                let max_health = target_ship.data.max_health.max(1) as f32;
                let health_pct = (target_ship.health as f32 / max_health * 100.0).round();
                if is_hostile {
                    format!("Target: {} ({}%) [HOSTILE]", display, health_pct)
                } else {
                    format!("Target: {} ({}%)", display, health_pct)
                }
            } else {
                "Target: None".to_string()
            }
        }
        Some(Target::Asteroid(entity)) => {
            if let Ok(asteroid) = asteroids_query.get(*entity) {
                format!("Target: Asteroid (size {:.0})", asteroid.size)
            } else {
                "Target: None".to_string()
            }
        }
        Some(Target::Planet(entity)) => {
            if let Ok(planet) = planets_query.get(*entity) {
                let display = item_universe
                    .star_systems
                    .get(&current_system.0)
                    .and_then(|sys| sys.planets.get(&planet.0))
                    .map(|p| p.display_name.as_str())
                    .unwrap_or(&planet.0);
                format!("Target: {}", display)
            } else {
                "Target: None".to_string()
            }
        }
        Some(Target::Pickup(entity)) => {
            if let Ok(pickup) = pickups_query.get(*entity) {
                let commodity_display = item_universe
                    .commodities
                    .get(&pickup.commodity)
                    .map(|c| c.display_name.as_str())
                    .unwrap_or(&pickup.commodity);
                format!("Target: {} (x{})", commodity_display, pickup.quantity)
            } else {
                "Target: None".to_string()
            }
        }
        None => "Target: None".to_string(),
    };
}

/// Draw 4 corner brackets around the player's current nav_target.
fn draw_target_reticle(
    mut gizmos: Gizmos,
    player: Query<&Ship, With<Player>>,
    transforms: Query<&Transform>,
    ships: Query<&Ship, Without<Player>>,
    asteroids: Query<&Asteroid>,
    planets: Query<&Planet>,
    pickups: Query<&Pickup>,
    item_universe: Res<ItemUniverse>,
    current_system: Res<CurrentStarSystem>,
) {
    let Ok(player_ship) = player.single() else {
        return;
    };
    let Some(target) = &player_ship.nav_target else {
        return;
    };
    let entity = target.get_entity();
    let Ok(tf) = transforms.get(entity) else {
        return;
    };

    // Half-side of the bracket box, sized to the target with padding.
    let radius = match target {
        Target::Ship(_) => ships.get(entity).map(|s| s.data.radius).unwrap_or(20.0),
        Target::Asteroid(_) => asteroids.get(entity).map(|a| a.size).unwrap_or(20.0),
        Target::Planet(_) => planets
            .get(entity)
            .ok()
            .and_then(|p| {
                item_universe
                    .star_systems
                    .get(&current_system.0)
                    .and_then(|sys| sys.planets.get(&p.0))
            })
            .map(|pd| pd.radius)
            .unwrap_or(80.0),
        Target::Pickup(_) => pickups.get(entity).map(|_| 8.0).unwrap_or(8.0),
    };
    let half = radius * 1.4;
    let arm = half * 0.35;
    let center = tf.translation.truncate();
    let color = Color::srgb(0.4, 1.0, 0.6);

    // 4 corners × 2 line segments each = 8 lines forming L-shapes.
    for &(sx, sy) in &[(-1.0_f32, 1.0_f32), (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)] {
        let corner = center + Vec2::new(sx * half, sy * half);
        gizmos.line_2d(corner, corner + Vec2::new(-sx * arm, 0.0), color);
        gizmos.line_2d(corner, corner + Vec2::new(0.0, -sy * arm), color);
    }
}

fn update_comms_ticker(
    time: Res<Time>,
    mut channel: ResMut<CommsChannel>,
    mut text_query: Query<(&mut Text, &mut Node, &ComputedNode), With<CommsText>>,
    container_query: Query<&ComputedNode, (With<CommsContainer>, Without<CommsText>)>,
) {
    let Ok((mut text, mut node, text_computed)) = text_query.single_mut() else {
        return;
    };
    let Ok(container_computed) = container_query.single() else {
        return;
    };

    // Sync displayed text with resource
    if **text != channel.message {
        **text = channel.message.clone();
        channel.scroll_offset = 0.0;
        node.left = Val::Px(4.0);
    }

    if channel.message.is_empty() {
        return;
    }

    let text_width = text_computed.size().x;
    let container_width = container_computed.size().x;
    let overflow = text_width - container_width;

    if overflow <= 0.0 {
        // Text fits — show static, left-aligned
        node.left = Val::Px(4.0);
        channel.cycle_complete = true;
        return;
    }

    // Pause before (re)starting scroll
    if channel.pause_timer > 0.0 {
        channel.pause_timer -= time.delta_secs();
        return;
    }

    // Advance scroll
    channel.scroll_offset += COMMS_SCROLL_SPEED * time.delta_secs();

    // Once the end of the text has scrolled into view (+ a small gap), loop
    if channel.scroll_offset > overflow + 30.0 {
        channel.scroll_offset = 0.0;
        channel.pause_timer = COMMS_LOOP_PAUSE;
        channel.cycle_complete = true;
    }

    node.left = Val::Px(4.0 - channel.scroll_offset);
}

/// Update the mini-map player dot based on walker world position.
fn update_minimap_player_dot(
    walker_q: Query<&Transform, With<crate::surface::Walker>>,
    minimap: Option<Res<crate::surface::SurfaceMiniMap>>,
    mut dot_q: Query<&mut Node, With<MiniMapPlayerDot>>,
) {
    let (Some(minimap), Ok(walker_tf), Ok(mut dot)) =
        (minimap, walker_q.single(), dot_q.single_mut())
    else {
        return;
    };
    let map_w = minimap.map_w as f32;
    let map_h = minimap.map_h as f32;
    let tile_px = crate::surface::TILE_PX;
    // Walker world pos → tile coords.
    let wx = walker_tf.translation.x;
    let wy = walker_tf.translation.y;
    let tx = wx / tile_px + map_w / 2.0;
    let ty = wy / tile_px + map_h / 2.0;
    // Tile coords → pixel position in the mini-map display.
    let display_size = RADAR_SIZE - 4.0; // account for border
    let px = (tx / map_w * display_size).clamp(0.0, display_size - 6.0);
    // Mini-map image is top-down (row 0 = bottom-up y=0 = bottom of map),
    // but UI y increases downward, so flip.
    let py = ((1.0 - ty / map_h) * display_size).clamp(0.0, display_size - 6.0);
    dot.left = Val::Px(px);
    dot.top = Val::Px(py);
}

fn dot_bundle(left: f32, top: f32, color: Color) -> impl Bundle {
    (
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(left),
            top: Val::Px(top),
            width: Val::Px(DOT_SIZE),
            height: Val::Px(DOT_SIZE),
            border_radius: BorderRadius::all(Val::Px(DOT_SIZE / 2.0)),
            ..default()
        },
        BackgroundColor(color),
    )
}

/// Toggle HUD element visibility based on whether we're in Exploring state.
/// Uses `Display::None` instead of `Visibility::Hidden` so hidden elements
/// don't occupy layout space.
fn toggle_space_hud_visibility(
    state: Res<State<PlayState>>,
    mut space_q: Query<(&mut Visibility, &mut Node), (With<SpaceOnlyHud>, Without<SurfaceOnlyHud>)>,
    mut surface_q: Query<(&mut Visibility, &mut Node), (With<SurfaceOnlyHud>, Without<SpaceOnlyHud>)>,
    mut minimap_q: Query<&mut ImageNode, With<MiniMapImage>>,
    minimap_res: Option<Res<crate::surface::SurfaceMiniMap>>,
) {
    let exploring = *state.get() == PlayState::Exploring;
    for (mut vis, mut node) in &mut space_q {
        if exploring {
            *vis = Visibility::Hidden;
            node.display = Display::None;
        } else {
            *vis = Visibility::Inherited;
            node.display = Display::Flex;
        }
    }
    for (mut vis, mut node) in &mut surface_q {
        if exploring {
            *vis = Visibility::Inherited;
            node.display = Display::Flex;
        } else {
            *vis = Visibility::Hidden;
            node.display = Display::None;
        }
    }
    // Set the mini-map image when entering Exploring.
    if exploring {
        if let (Some(minimap), Ok(mut img_node)) = (minimap_res, minimap_q.single_mut()) {
            img_node.image = minimap.image.clone();
        }
    }
}
