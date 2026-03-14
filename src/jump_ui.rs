use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use egui::{Align2, Color32, FontId, Pos2, Sense, Stroke};
use std::collections::HashSet;

use crate::{CurrentStarSystem, GameState, TravelContext, item_universe::ItemUniverse};

// Canvas is larger than the visible viewport — the ScrollArea lets the user pan around.
const CANVAS_W: f32 = 1400.0;
const CANVAS_H: f32 = 900.0;
// Map origin (0,0) is placed at this pixel in the canvas.
const CANVAS_CENTER_X: f32 = 700.0;
const CANVAS_CENTER_Y: f32 = 450.0;
// Visible viewport of the scroll area.
const VIEWPORT_W: f32 = 600.0;
const VIEWPORT_H: f32 = 400.0;

// Initial scroll so the origin (sol) appears near the center of the viewport.
const INITIAL_SCROLL_X: f32 = CANVAS_CENTER_X - VIEWPORT_W * 0.5;
const INITIAL_SCROLL_Y: f32 = CANVAS_CENTER_Y - VIEWPORT_H * 0.5;

const NODE_R: f32 = 3.5;   // visual dot radius
const CLICK_R: f32 = 12.0; // invisible click hit radius

#[derive(Resource, Default)]
struct JumpUiOpen {
    open: bool,
    scroll_initialized: bool,
}

pub fn jump_ui_plugin(app: &mut App) {
    app.init_resource::<JumpUiOpen>()
        .add_systems(
            Update,
            toggle_jump_ui.run_if(in_state(GameState::Flying)),
        )
        .add_systems(
            EguiPrimaryContextPass,
            jump_ui.run_if(in_state(GameState::Flying)),
        );
}

fn toggle_jump_ui(keyboard: Res<ButtonInput<KeyCode>>, mut state: ResMut<JumpUiOpen>) {
    if keyboard.just_pressed(KeyCode::KeyJ) {
        state.open = !state.open;
        if state.open {
            // Reset scroll so the map re-centers on the origin (current system area).
            state.scroll_initialized = false;
        }
    }
}

fn jump_ui(
    mut egui_contexts: EguiContexts,
    mut ui_state: ResMut<JumpUiOpen>,
    mut game_state: ResMut<NextState<GameState>>,
    mut travel_ctx: ResMut<TravelContext>,
    current_system: Res<CurrentStarSystem>,
    item_universe: Res<ItemUniverse>,
) {
    if !ui_state.open {
        return;
    }
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };

    let neighbors: HashSet<String> = item_universe
        .star_systems
        .get(&current_system.0)
        .map(|s| s.connections.iter().cloned().collect())
        .unwrap_or_default();

    let mut jump_target: Option<String> = None;
    let mut close = false;

    egui::Window::new("Star Map")
        .collapsible(false)
        .resizable(true)
        .default_size([VIEWPORT_W + 16.0, VIEWPORT_H + 60.0])
        .show(ctx, |ui| {
            let mut scroll = egui::ScrollArea::both().id_salt("star_map_scroll");
            if !ui_state.scroll_initialized {
                scroll = scroll.scroll_offset(egui::Vec2::new(INITIAL_SCROLL_X, INITIAL_SCROLL_Y));
                ui_state.scroll_initialized = true;
            }

            scroll.show(ui, |ui| {
                let (response, painter) =
                    ui.allocate_painter(egui::Vec2::new(CANVAS_W, CANVAS_H), Sense::click());

                let origin = response.rect.min;

                // Convert a map-space Vec2 to screen Pos2.
                let to_screen = |mp: bevy::math::Vec2| -> Pos2 {
                    Pos2::new(
                        origin.x + CANVAS_CENTER_X + mp.x,
                        origin.y + CANVAS_CENTER_Y + mp.y,
                    )
                };

                // Background
                painter.rect_filled(response.rect, 0.0, Color32::from_rgb(4, 4, 18));

                // Edges — draw each undirected edge once (smaller name → larger name lexicographically)
                for (sys_name, sys) in &item_universe.star_systems {
                    let from = to_screen(sys.map_position);
                    for conn in &sys.connections {
                        if conn > sys_name {
                            if let Some(other) = item_universe.star_systems.get(conn) {
                                let to = to_screen(other.map_position);
                                painter.line_segment(
                                    [from, to],
                                    Stroke::new(1.0, Color32::from_rgb(40, 50, 90)),
                                );
                            }
                        }
                    }
                }

                // Collect pointer state once.
                let hover_pos = ctx.pointer_hover_pos();
                let click_pos = if response.clicked() {
                    response.interact_pointer_pos()
                } else {
                    None
                };

                // Nodes
                for (sys_name, sys) in &item_universe.star_systems {
                    let pos = to_screen(sys.map_position);
                    let is_current = *sys_name == current_system.0;
                    let is_neighbor = neighbors.contains(sys_name);

                    // Hover glow for jumpable neighbors
                    if is_neighbor {
                        let hovered =
                            hover_pos.map_or(false, |hp| (hp - pos).length() < CLICK_R);
                        if hovered {
                            painter.circle_filled(
                                pos,
                                NODE_R + 5.0,
                                Color32::from_rgba_unmultiplied(255, 220, 80, 50),
                            );
                            painter.circle_stroke(
                                pos,
                                NODE_R + 4.0,
                                Stroke::new(1.5, Color32::from_rgb(255, 220, 80)),
                            );
                        }
                    }

                    // Dot fill
                    let fill = if is_current {
                        Color32::from_rgb(100, 190, 255)
                    } else if is_neighbor {
                        Color32::from_rgb(200, 210, 230)
                    } else {
                        Color32::from_rgb(90, 95, 120)
                    };

                    painter.circle_filled(pos, NODE_R, fill);

                    // Subtle border only on current and neighbors
                    if is_current || is_neighbor {
                        painter.circle_stroke(
                            pos,
                            NODE_R,
                            Stroke::new(1.0, Color32::from_rgb(160, 170, 210)),
                        );
                    }

                    // Name label
                    let display = sys_name.replace('_', " ");
                    painter.text(
                        Pos2::new(pos.x, pos.y + NODE_R + 5.0),
                        Align2::CENTER_TOP,
                        &display,
                        FontId::proportional(11.0),
                        if is_current {
                            Color32::from_rgb(140, 210, 255)
                        } else {
                            Color32::from_rgb(170, 175, 200)
                        },
                    );

                    // Click to jump
                    if is_neighbor {
                        if let Some(cp) = click_pos {
                            if (cp - pos).length() < CLICK_R {
                                jump_target = Some(sys_name.clone());
                            }
                        }
                    }
                }
            });

            ui.separator();
            if ui.button("Close  [J]").clicked() {
                close = true;
            }
        });

    if let Some(dest) = jump_target {
        travel_ctx.destination = dest;
        travel_ctx.timer = Timer::from_seconds(3.0, TimerMode::Once);
        game_state.set(GameState::Traveling);
        ui_state.open = false;
    }
    if close {
        ui_state.open = false;
    }
}
