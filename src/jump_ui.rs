use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use egui::{Align2, Color32, FontId, Pos2, Sense, Stroke};
use std::collections::HashSet;

use crate::{
    CurrentStarSystem, PlayState, TravelContext, TravelPhase,
    game_save::PlayerGameState, item_universe::ItemUniverse,
    missions::{MissionCatalog, MissionLog},
    missions::types::{MissionStatus, Objective},
};

// Canvas is larger than the visible viewport — the ScrollArea lets the user pan around.
const CANVAS_W: f32 = 1400.0;
const CANVAS_H: f32 = 900.0;
// Map origin (0,0) is placed at this pixel in the canvas.
const CANVAS_CENTER_X: f32 = 700.0;
const CANVAS_CENTER_Y: f32 = 450.0;
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
            toggle_jump_ui.run_if(in_state(PlayState::Flying)),
        )
        .add_systems(
            EguiPrimaryContextPass,
            (
                jump_ui.run_if(in_state(PlayState::Flying)),
                jump_flash.run_if(in_state(PlayState::Traveling)),
            ),
        );
}

fn toggle_jump_ui(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<JumpUiOpen>,
    mut virtual_time: ResMut<Time<Virtual>>,
) {
    if keyboard.just_pressed(KeyCode::KeyJ) {
        state.open = !state.open;
        if state.open {
            state.scroll_initialized = false;
            virtual_time.pause();
        } else {
            virtual_time.unpause();
        }
    }
}

/// Full-screen hyperspace flash drawn over everything while in Traveling state.
fn jump_flash(mut egui_contexts: EguiContexts, travel_ctx: Res<TravelContext>) {
    if travel_ctx.phase != TravelPhase::Flashing {
        return;
    }
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };

    let t = travel_ctx.flash_t; // 0..1
    // Triangle envelope: fade in 0→0.5, fade out 0.5→1.
    let alpha = if t < 0.5 { t * 2.0 } else { 2.0 - t * 2.0 };
    let screen = ctx.viewport_rect();
    let center = screen.center();

    egui::Area::new(egui::Id::new("jump_flash"))
        .order(egui::Order::Foreground)
        .fixed_pos(egui::Pos2::ZERO)
        .interactable(false)
        .show(ctx, |ui| {
            let p = ui.painter();

            // Expanding radial burst — a circle that races outward from screen center.
            let burst_r = t * screen.size().max_elem() * 1.3;
            let burst_alpha = ((1.0 - t) * 230.0) as u8;
            p.circle_filled(center, burst_r, Color32::from_white_alpha(burst_alpha));

            // Bright ring at the leading edge of the burst.
            let ring_width = (1.0 - t) * 12.0 + 2.0;
            p.circle_stroke(
                center,
                burst_r,
                Stroke::new(ring_width, Color32::from_white_alpha((alpha * 255.0) as u8)),
            );

            // Overall white-out overlay — peaks at the midpoint.
            p.rect_filled(screen, 0.0, Color32::from_white_alpha((alpha * 200.0) as u8));
        });
}

/// Return the target star system for an objective, looking up the planet's
/// system when necessary.
fn objective_system<'a>(obj: &'a Objective, item_universe: &'a ItemUniverse) -> Option<&'a str> {
    match obj {
        Objective::TravelToSystem { system } => Some(system.as_str()),
        Objective::CollectPickups { system, .. } => Some(system.as_str()),
        Objective::DestroyShips { system, .. } => Some(system.as_str()),
        Objective::LandOnPlanet { planet } => {
            // Scan all systems for the planet.
            item_universe
                .star_systems
                .iter()
                .find(|(_, sys)| sys.planets.contains_key(planet))
                .map(|(name, _)| name.as_str())
        }
    }
}

fn jump_ui(
    mut egui_contexts: EguiContexts,
    mut ui_state: ResMut<JumpUiOpen>,
    mut travel_ctx: ResMut<TravelContext>,
    current_system: Res<CurrentStarSystem>,
    item_universe: Res<ItemUniverse>,
    game_state: Res<PlayerGameState>,
    mission_log: Res<MissionLog>,
    mission_catalog: Res<MissionCatalog>,
    primary_window: Query<&Window, With<PrimaryWindow>>,
    mut virtual_time: ResMut<Time<Virtual>>,
) {
    if !ui_state.open {
        return;
    }
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };

    let jumpable_neighbors: HashSet<String> = item_universe
        .star_systems
        .get(&current_system.0)
        .map(|s| s.connections.iter().cloned().collect())
        .unwrap_or_default();

    let visited: &HashSet<String> = &game_state.visited_systems;
    // Greyed discovery ring: strictly one jump from any visited system.
    let mut discovered: HashSet<String> = HashSet::new();
    for v in visited {
        if let Some(sys) = item_universe.star_systems.get(v) {
            for conn in &sys.connections {
                if !visited.contains(conn) {
                    discovered.insert(conn.clone());
                }
            }
        }
    }

    // Systems with active mission objectives.
    let mission_systems: HashSet<String> = mission_log
        .statuses
        .iter()
        .filter_map(|(id, status)| {
            if let MissionStatus::Active(_) = status {
                let def = mission_catalog.defs.get(id)?;
                objective_system(&def.objective, &item_universe)
                    .map(|s| s.to_string())
            } else {
                None
            }
        })
        .collect();

    // Size the map window proportionally to the primary window.
    let (win_w, win_h) = primary_window
        .single()
        .map(|w| (w.width(), w.height()))
        .unwrap_or((1280.0, 720.0));
    let map_w = (win_w * 0.85).max(400.0);
    let map_h = (win_h * 0.70).max(250.0);

    let mut jump_target: Option<String> = None;
    let mut close = false;

    egui::Window::new("Star Map")
        .collapsible(false)
        .resizable(true)
        .default_size([map_w, map_h])
        .max_size([map_w, map_h])
        .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            let mut scroll = egui::ScrollArea::both().id_salt("star_map_scroll");
            if !ui_state.scroll_initialized {
                let map_pos = item_universe
                    .star_systems
                    .get(&current_system.0)
                    .map(|s| s.map_position)
                    .unwrap_or_default();
                let offset_x = CANVAS_CENTER_X + map_pos.x - map_w * 0.5;
                let offset_y = CANVAS_CENTER_Y + map_pos.y - map_h * 0.5;
                scroll = scroll.scroll_offset(egui::Vec2::new(offset_x, offset_y));
                ui_state.scroll_initialized = true;
            }

            scroll.show(ui, |ui| {
                let (response, painter) =
                    ui.allocate_painter(egui::Vec2::new(CANVAS_W, CANVAS_H), Sense::click());

                let origin = response.rect.min;
                let to_screen = |mp: bevy::math::Vec2| -> Pos2 {
                    Pos2::new(
                        origin.x + CANVAS_CENTER_X + mp.x,
                        origin.y + CANVAS_CENTER_Y + mp.y,
                    )
                };

                painter.rect_filled(response.rect, 0.0, Color32::from_rgb(4, 4, 18));

                let is_visible = |name: &String| -> bool {
                    visited.contains(name) || discovered.contains(name)
                };

                // Edges — only when both endpoints are visible AND at least one is visited.
                for (sys_name, sys) in &item_universe.star_systems {
                    if !is_visible(sys_name) {
                        continue;
                    }
                    let from = to_screen(sys.map_position);
                    for conn in &sys.connections {
                        if conn > sys_name
                            && is_visible(conn)
                            && (visited.contains(sys_name) || visited.contains(conn))
                        {
                            if let Some(other) = item_universe.star_systems.get(conn) {
                                painter.line_segment(
                                    [from, to_screen(other.map_position)],
                                    Stroke::new(1.0, Color32::from_rgb(40, 50, 90)),
                                );
                            }
                        }
                    }
                }

                let hover_pos = ctx.pointer_hover_pos();
                let click_pos = if response.clicked() {
                    response.interact_pointer_pos()
                } else {
                    None
                };

                // Nodes
                for (sys_name, sys) in &item_universe.star_systems {
                    if !is_visible(sys_name) {
                        continue;
                    }
                    let pos = to_screen(sys.map_position);
                    let is_current = *sys_name == current_system.0;
                    let is_visited = visited.contains(sys_name);
                    let is_jumpable = jumpable_neighbors.contains(sys_name);

                    if is_jumpable {
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

                    let fill = if is_current {
                        Color32::from_rgb(100, 190, 255)
                    } else if is_visited {
                        Color32::from_rgb(200, 210, 230)
                    } else {
                        // Discovered but unvisited — greyed.
                        Color32::from_rgb(80, 85, 100)
                    };

                    painter.circle_filled(pos, NODE_R, fill);
                    if is_current || is_visited {
                        painter.circle_stroke(
                            pos,
                            NODE_R,
                            Stroke::new(1.0, Color32::from_rgb(160, 170, 210)),
                        );
                    }

                    if is_visited {
                        painter.text(
                            Pos2::new(pos.x, pos.y + NODE_R + 5.0),
                            Align2::CENTER_TOP,
                            &sys.display_name,
                            FontId::proportional(11.0),
                            if is_current {
                                Color32::from_rgb(140, 210, 255)
                            } else {
                                Color32::from_rgb(170, 175, 200)
                            },
                        );
                    }

                    // Red mission marker for systems with active objectives.
                    if mission_systems.contains(sys_name) {
                        let half = NODE_R + 6.0;
                        let rect = egui::Rect::from_center_size(
                            pos,
                            egui::Vec2::splat(half * 2.0),
                        );
                        painter.rect_stroke(
                            rect,
                            1.0,
                            Stroke::new(1.5, Color32::from_rgb(220, 50, 50)),
                            egui::StrokeKind::Outside,
                        );
                    }

                    if is_jumpable {
                        if let Some(cp) = click_pos {
                            if (cp - pos).length() < CLICK_R {
                                jump_target = Some(sys_name.clone());
                            }
                        }
                    }
                }

                // Mission markers for hidden (unvisited & undiscovered) systems —
                // show a dim red diamond at the map position so the player has a
                // directional hint when scrolling.
                for sys_name in &mission_systems {
                    if visited.contains(sys_name) || discovered.contains(sys_name) {
                        continue; // already drawn above
                    }
                    if let Some(sys) = item_universe.star_systems.get(sys_name) {
                        let pos = to_screen(sys.map_position);
                        let s = NODE_R + 4.0;
                        let points = vec![
                            Pos2::new(pos.x, pos.y - s),
                            Pos2::new(pos.x + s, pos.y),
                            Pos2::new(pos.x, pos.y + s),
                            Pos2::new(pos.x - s, pos.y),
                        ];
                        painter.add(egui::Shape::convex_polygon(
                            points,
                            Color32::from_rgba_unmultiplied(180, 40, 40, 120),
                            Stroke::new(1.0, Color32::from_rgb(200, 60, 60)),
                        ));
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
        travel_ctx.phase = TravelPhase::Accelerating;
        ui_state.open = false;
        virtual_time.unpause();
    }
    if close {
        ui_state.open = false;
        virtual_time.unpause();
    }
}
