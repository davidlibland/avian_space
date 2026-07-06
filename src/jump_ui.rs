use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use egui::{Align2, Color32, FontId, Pos2, Sense, Stroke};
use std::collections::HashSet;

use crate::session::SessionResourceExt;
use crate::{
    CurrentStarSystem, PlayState, TravelContext, TravelPhase,
    game_save::PlayerGameState,
    item_universe::{ItemUniverse, StarSystem},
    missions::{MissionCatalog, MissionLog},
    missions::types::{MissionStatus, Objective},
};

// The scrollable canvas is sized at runtime to the galaxy's actual extent (see
// the map-bounds computation below), so every system is reachable by panning no
// matter how far the map expands. This padding rings the outermost systems.
const MAP_PAD: f32 = 120.0;
const NODE_R: f32 = 3.5;   // visual dot radius
const CLICK_R: f32 = 12.0; // invisible click hit radius

/// Map factions shown on the star map (Free Frontier's spaced label, others as-is).
const LEGEND_FACTIONS: &[&str] = &[
    "Federation", "Rebel", "FreeFrontier", "Helios", "Bastion", "Order", "Precursor",
];

/// A faction's signature colour on the star map. Keep in sync with the design
/// bible / faction palette. Unknown or contested space reads as neutral grey.
fn faction_color(faction: &str) -> Color32 {
    match faction {
        "Federation" => Color32::from_rgb(78, 120, 196),
        "Rebel" => Color32::from_rgb(90, 190, 110),
        "FreeFrontier" => Color32::from_rgb(228, 198, 92),
        "Helios" => Color32::from_rgb(84, 200, 230),
        "Bastion" => Color32::from_rgb(188, 64, 56),
        "Order" => Color32::from_rgb(162, 110, 214),
        "Precursor" => Color32::from_rgb(196, 84, 206),
        "Independent" => Color32::from_rgb(176, 166, 150),
        _ => Color32::from_rgb(120, 125, 140), // contested / unknown
    }
}

fn faction_label(faction: &str) -> &str {
    match faction {
        "FreeFrontier" => "Free Frontier",
        other => other,
    }
}

/// Lerp a colour toward the map void — used to dim discovered-but-unvisited systems.
fn dim(c: Color32, t: f32) -> Color32 {
    let l = |a: u8, b: u8| (a as f32 * (1.0 - t) + b as f32 * t) as u8;
    Color32::from_rgb(l(c.r(), 4), l(c.g(), 4), l(c.b(), 18))
}

/// The controlling faction of a system: the faction holding its planets, or — for
/// uninhabited systems — the single faction that dominates its spawns (so the
/// Precursor deep reads Precursor). Contested borderlands (e.g. the Drift, an even
/// Federation/Rebel split) return "" and read neutral.
fn system_faction(sys: &StarSystem, iu: &ItemUniverse) -> String {
    use std::collections::HashMap;
    let mut by: HashMap<&str, f32> = HashMap::new();
    for p in sys.planets.values() {
        if !p.faction.is_empty() {
            *by.entry(p.faction.as_str()).or_default() += 1.0;
        }
    }
    if by.is_empty() {
        for (name, weight) in &sys.ships.types {
            if let Some(f) = iu.ships.get(name).and_then(|s| s.faction.as_deref()) {
                if f != "Pirate" && f != "Merchant" {
                    *by.entry(f).or_default() += *weight;
                }
            }
        }
    }
    let mut ranked: Vec<(&str, f32)> = by.into_iter().collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(b.0))
    });
    match ranked.as_slice() {
        [(f, _)] => f.to_string(),
        [(f, w0), (_, w1), ..] if *w0 > *w1 * 1.4 => f.to_string(),
        _ => String::new(),
    }
}

#[derive(Resource, Default)]
pub struct JumpUiOpen {
    pub open: bool,
    scroll_initialized: bool,
}

impl crate::session::SessionResource for JumpUiOpen {
    type SaveData = ();
    fn new_session(_: &crate::item_universe::ItemUniverse) -> Self { Self::default() }
    fn from_save(_: (), _: &crate::item_universe::ItemUniverse) -> Self { Self::default() }
}

pub fn jump_ui_plugin(app: &mut App) {
    app.init_session_resource::<JumpUiOpen>()
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
        Objective::LandOnPlanet { planet }
        | Objective::MeetNpc { planet, .. }
        | Objective::CatchNpc { planet, .. } => {
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
    mut player_ship: Query<&mut crate::ship::Ship, With<crate::Player>>,
) {
    if !ui_state.open {
        return;
    }
    // Fuel (jumps remaining) drives whether a jump is allowed + the readout.
    let (fuel, fuel_cap) = player_ship
        .single()
        .map(|s| (s.fuel, s.data.fuel_capacity))
        .unwrap_or((0, 0));
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

    // Bounds of every system in map-space, so the scrollable canvas is exactly
    // large enough to hold the whole galaxy (the expanded map ran off the old
    // fixed canvas, leaving far systems unreachable by scrolling).
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for sys in item_universe.star_systems.values() {
        min_x = min_x.min(sys.map_position.x);
        min_y = min_y.min(sys.map_position.y);
        max_x = max_x.max(sys.map_position.x);
        max_y = max_y.max(sys.map_position.y);
    }
    if !min_x.is_finite() {
        (min_x, min_y, max_x, max_y) = (0.0, 0.0, 0.0, 0.0);
    }
    let canvas_w = (max_x - min_x) + MAP_PAD * 2.0;
    let canvas_h = (max_y - min_y) + MAP_PAD * 2.0;

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
                let offset_x = (map_pos.x - min_x) + MAP_PAD - map_w * 0.5;
                let offset_y = (map_pos.y - min_y) + MAP_PAD - map_h * 0.5;
                scroll = scroll
                    .scroll_offset(egui::Vec2::new(offset_x.max(0.0), offset_y.max(0.0)));
                ui_state.scroll_initialized = true;
            }

            scroll.show(ui, |ui| {
                let (response, painter) =
                    ui.allocate_painter(egui::Vec2::new(canvas_w, canvas_h), Sense::click());

                let origin = response.rect.min;
                let to_screen = |mp: bevy::math::Vec2| -> Pos2 {
                    Pos2::new(
                        origin.x + (mp.x - min_x) + MAP_PAD,
                        origin.y + (mp.y - min_y) + MAP_PAD,
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

                    let base = faction_color(&system_faction(sys, &item_universe));
                    let fill = if is_current || is_visited {
                        base
                    } else {
                        // Discovered but unvisited — faction colour dimmed toward the void.
                        dim(base, 0.55)
                    };

                    painter.circle_filled(pos, NODE_R, fill);
                    if is_current {
                        // "You are here" — bright white ring.
                        painter.circle_stroke(pos, NODE_R + 1.5, Stroke::new(2.0, Color32::WHITE));
                    } else if is_visited {
                        painter.circle_stroke(pos, NODE_R, Stroke::new(1.0, dim(base, 0.35)));
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
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 12.0;
                for fac in LEGEND_FACTIONS {
                    ui.colored_label(faction_color(fac), faction_label(fac));
                }
            });
            ui.label(format!("Fuel: {fuel}/{fuel_cap} jumps"));
            if fuel == 0 {
                ui.colored_label(
                    Color32::from_rgb(220, 80, 80),
                    "Out of fuel — land at a mechanic to refuel.",
                );
            }
            if ui.button("Close  [J]").clicked() {
                close = true;
            }
        });

    // Only jump if there's fuel; a jump burns one unit.
    if let Some(dest) = jump_target {
        if fuel > 0 {
            if let Ok(mut ship) = player_ship.single_mut() {
                ship.fuel = ship.fuel.saturating_sub(1);
            }
            travel_ctx.destination = dest;
            travel_ctx.phase = TravelPhase::Accelerating;
            ui_state.open = false;
            virtual_time.unpause();
        }
    }
    if close {
        ui_state.open = false;
        virtual_time.unpause();
    }
}
