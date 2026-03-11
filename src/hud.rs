use bevy::prelude::*;

use crate::{Player, Ship};

#[derive(Component)]
struct HealthBarFill;

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_hud)
            .add_systems(Update, update_health_bar);
    }
}

fn spawn_hud(mut commands: Commands) {
    // Root: absolute top-right column
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            right: Val::Px(16.0),
            top: Val::Px(16.0),
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Stretch,
            row_gap: Val::Px(6.0),
            width: Val::Px(144.0),
            ..default()
        })
        .with_children(|root| {
            // ── Radar ────────────────────────────────────────────────────
            root.spawn((
                Node {
                    width: Val::Px(144.0),
                    height: Val::Px(144.0),
                    border: UiRect::all(Val::Px(2.0)),
                    border_radius: BorderRadius::all(Val::Px(72.0)), // full circle
                    ..default()
                },
                BackgroundColor(Color::srgba(0.0, 0.06, 0.0, 0.85)),
                BorderColor::all(Color::srgb(0.1, 0.8, 0.3)),
            ));

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
        });
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
    let pct = (ship.health as f32 / ship.max_health as f32 * 100.0).clamp(0.0, 100.0);
    node.width = Val::Percent(pct);
}
