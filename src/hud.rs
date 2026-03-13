use bevy::prelude::*;

use crate::planets::Planet;
use crate::{GameState, Player, Ship};

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
        .add_systems(Update, (update_health_bar, update_cargo_credits))
        .add_systems(
            Update,
            update_radar_dots.run_if(in_state(GameState::Flying)),
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

            // ── Primary weapon ───────────────────────────────────────────
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
                    Text::new("Primary:  Laser"),
                    TextFont {
                        font_size: 13.0,
                        ..default()
                    },
                    TextColor(Color::srgb(0.9, 0.85, 0.5)),
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
                    Text::new("Secondary: None"),
                    TextFont {
                        font_size: 13.0,
                        ..default()
                    },
                    TextColor(Color::srgb(0.5, 0.5, 0.55)),
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
    player_query: Query<&Transform, With<Player>>,
    ships_query: Query<&Transform, (With<Ship>, Without<Player>)>,
    planets_query: Query<&Transform, With<Planet>>,
    dots_query: Query<Entity, With<RadarDot>>,
) {
    let Ok(player_tf) = player_query.single() else {
        return;
    };
    let player_pos = player_tf.translation.truncate();
    let half = RADAR_SIZE / 2.0;

    // Collect planet world positions up front so we can check visibility
    // and compute the mean outside the with_children closure.
    let planet_positions: Vec<Vec2> = planets_query
        .iter()
        .map(|tf| tf.translation.truncate())
        .collect();

    for entity in dots_query.iter() {
        commands.entity(entity).despawn();
    }

    commands.entity(radar.0).with_children(|parent| {
        // ── Ships (white dots) ───────────────────────────────────────────
        for ship_tf in ships_query.iter() {
            let offset = ship_tf.translation.truncate() - player_pos;
            if offset.length() > config.range {
                continue;
            }
            let rx = half + (offset.x / config.range) * half - DOT_SIZE / 2.0;
            let ry = half - (offset.y / config.range) * half - DOT_SIZE / 2.0;
            parent.spawn((RadarDot, dot_bundle(rx, ry, Color::WHITE)));
        }

        // ── Planets (blue dots) ──────────────────────────────────────────
        let mut any_planet_visible = false;
        for &pos in &planet_positions {
            let offset = pos - player_pos;
            if offset.length() > config.range {
                continue;
            }
            any_planet_visible = true;
            let rx = half + (offset.x / config.range) * half - DOT_SIZE / 2.0;
            let ry = half - (offset.y / config.range) * half - DOT_SIZE / 2.0;
            parent.spawn((RadarDot, dot_bundle(rx, ry, Color::srgb(0.3, 0.5, 1.0))));
        }

        // ── Off-radar planet chevron ─────────────────────────────────────
        // When no planets appear on the radar, show a ">" at the edge that
        // is rotated to point toward the mean planet location.
        if !any_planet_visible && !planet_positions.is_empty() {
            let mean: Vec2 =
                planet_positions.iter().copied().sum::<Vec2>() / planet_positions.len() as f32;
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
        let free = ship.data.cargo_space - ship.cargo.values().sum::<u16>();
        **text = format!("Free: {}/{}", free, ship.data.cargo_space);
    }
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
