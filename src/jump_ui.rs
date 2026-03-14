use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use egui::{Align2, Color32, FontId, Pos2, Sense, Stroke};
use std::collections::{HashMap, HashSet, VecDeque};

use crate::{CurrentStarSystem, GameState, TravelContext, item_universe::ItemUniverse};

#[derive(Resource, Default)]
struct JumpUiOpen(bool);

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

fn toggle_jump_ui(keyboard: Res<ButtonInput<KeyCode>>, mut open: ResMut<JumpUiOpen>) {
    if keyboard.just_pressed(KeyCode::KeyJ) {
        open.0 = !open.0;
    }
}

/// BFS layout: current system at center, neighbors at radius_step, their neighbors at 2*radius_step, etc.
fn star_map_positions(
    universe: &ItemUniverse,
    current: &str,
    center: Pos2,
    radius_step: f32,
) -> HashMap<String, Pos2> {
    let mut positions: HashMap<String, Pos2> = HashMap::new();
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<(String, usize)> = VecDeque::new();
    // layer_nodes[k] = names of nodes at BFS depth k
    let mut layer_nodes: Vec<Vec<String>> = vec![vec![current.to_string()]];

    positions.insert(current.to_string(), center);
    visited.insert(current.to_string());
    queue.push_back((current.to_string(), 0));

    while let Some((name, depth)) = queue.pop_front() {
        if let Some(sys) = universe.star_systems.get(&name) {
            let next_depth = depth + 1;
            for neighbor in &sys.connections {
                if visited.contains(neighbor) {
                    continue;
                }
                visited.insert(neighbor.clone());
                queue.push_back((neighbor.clone(), next_depth));
                while layer_nodes.len() <= next_depth {
                    layer_nodes.push(vec![]);
                }
                layer_nodes[next_depth].push(neighbor.clone());
            }
        }
    }

    for (layer_idx, layer) in layer_nodes.iter().enumerate() {
        if layer_idx == 0 {
            continue;
        }
        let r = layer_idx as f32 * radius_step;
        let count = layer.len();
        for (i, name) in layer.iter().enumerate() {
            // Start at top (-π/2) and distribute evenly
            let angle = (i as f32 / count as f32) * std::f32::consts::TAU
                - std::f32::consts::FRAC_PI_2;
            positions.insert(
                name.clone(),
                Pos2::new(center.x + r * angle.cos(), center.y + r * angle.sin()),
            );
        }
    }

    positions
}

fn jump_ui(
    mut egui_contexts: EguiContexts,
    mut open: ResMut<JumpUiOpen>,
    mut state: ResMut<NextState<GameState>>,
    mut travel_ctx: ResMut<TravelContext>,
    current_system: Res<CurrentStarSystem>,
    item_universe: Res<ItemUniverse>,
) {
    if !open.0 {
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
        .resizable(false)
        .default_size([500.0, 440.0])
        .show(ctx, |ui| {
            let map_size = egui::Vec2::new(480.0, 380.0);
            let (response, painter) = ui.allocate_painter(map_size, Sense::click());
            let rect = response.rect;
            let center = rect.center();

            let positions = star_map_positions(&item_universe, &current_system.0, center, 130.0);

            // Background
            painter.rect_filled(rect, 6.0, Color32::from_rgb(5, 5, 20));

            // Edges — draw all edges (deduplicated by only drawing when sys_name < conn lexicographically)
            for (sys_name, sys) in &item_universe.star_systems {
                if let Some(&from) = positions.get(sys_name) {
                    for conn in &sys.connections {
                        if conn > sys_name {
                            if let Some(&to) = positions.get(conn) {
                                painter.line_segment(
                                    [from, to],
                                    Stroke::new(1.5, Color32::from_rgb(50, 60, 110)),
                                );
                            }
                        }
                    }
                }
            }

            let node_r = 22.0f32;
            let label_offset = node_r + 10.0;

            // Pointer positions for hover highlight and click
            let hover_pos = ctx.pointer_hover_pos();
            let click_pos = if response.clicked() {
                response.interact_pointer_pos()
            } else {
                None
            };

            // Nodes
            for (sys_name, &pos) in &positions {
                let is_current = *sys_name == current_system.0;
                let is_neighbor = neighbors.contains(sys_name);

                let fill = if is_current {
                    Color32::from_rgb(80, 170, 255) // light blue
                } else if is_neighbor {
                    Color32::from_rgb(200, 205, 225)
                } else {
                    Color32::from_rgb(55, 55, 80)
                };

                // Glow ring on hover (only for jump-able neighbors)
                if is_neighbor {
                    let hovered = hover_pos.map_or(false, |hp| (hp - pos).length() < node_r + 6.0);
                    if hovered {
                        painter.circle_filled(pos, node_r + 6.0, Color32::from_rgba_unmultiplied(255, 220, 80, 60));
                        painter.circle_stroke(pos, node_r + 4.0, Stroke::new(2.0, Color32::from_rgb(255, 220, 80)));
                    }
                }

                painter.circle_filled(pos, node_r, fill);
                painter.circle_stroke(
                    pos,
                    node_r,
                    Stroke::new(1.5, Color32::from_rgb(140, 150, 200)),
                );

                // Name label
                let display = sys_name.replace('_', " ");
                painter.text(
                    Pos2::new(pos.x, pos.y + label_offset),
                    Align2::CENTER_TOP,
                    &display,
                    FontId::proportional(13.0),
                    Color32::from_rgb(210, 215, 240),
                );

                // Click on a neighbor → jump
                if is_neighbor {
                    if let Some(cp) = click_pos {
                        if (cp - pos).length() < node_r {
                            jump_target = Some(sys_name.clone());
                        }
                    }
                }
            }

            ui.separator();
            if ui.button("Close  [J]").clicked() {
                close = true;
            }
        });

    if let Some(dest) = jump_target {
        travel_ctx.destination = dest;
        travel_ctx.timer = Timer::from_seconds(3.0, TimerMode::Once);
        state.set(GameState::Traveling);
        open.0 = false;
    }
    if close {
        open.0 = false;
    }
}
