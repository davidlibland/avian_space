use bevy::prelude::*;

use crate::planets::Planet;
use crate::ship::Target;
use crate::utils::safe_despawn;
use crate::{PlayState, Player, Ship};

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
struct RadarDisplay;

#[derive(Component)]
struct RadarDot;

#[derive(Resource)]
struct RadarEntity(Entity);

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
        .add_systems(Startup, spawn_hud)
        .add_systems(
            Update,
            (
                update_health_bar,
                update_cargo_credits,
                update_target_display,
                update_secondary_weapon,
            ),
        )
        .add_systems(
            Update,
            update_radar_dots.run_if(in_state(PlayState::Flying)),
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
    let targeted_entity: Option<Entity> = match &player_ship.target {
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
                matches!(&ship.target, Some(Target::Ship(e)) if *e == player_entity);
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
                match ws.ammo_quantity {
                    Some(qty) => format!("Sec: {} ({})", name, qty),
                    None => format!("Sec: {}", name),
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
    mut text_query: Query<&mut Text, With<TargetText>>,
) {
    let Ok((player_entity, player_ship)) = player_query.single() else {
        return;
    };
    let Ok(mut text) = text_query.single_mut() else {
        return;
    };
    **text = match &player_ship.target {
        Some(Target::Ship(entity)) => {
            if let Ok(target_ship) = ships_query.get(*entity) {
                let is_hostile =
                    matches!(&target_ship.target, Some(Target::Ship(e)) if *e == player_entity);
                if is_hostile {
                    format!("Target: {} [HOSTILE]", target_ship.ship_type)
                } else {
                    format!("Target: {}", target_ship.ship_type)
                }
            } else {
                "Target: None".to_string()
            }
        }
        _ => "Target: None".to_string(),
    };
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
