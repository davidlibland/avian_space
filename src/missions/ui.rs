use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};

use super::events::*;
use super::log::{MissionCatalog, MissionLog, MissionOffers};
use super::types::*;
use crate::Player;
use crate::game_save::PlayerGameState;
use crate::item_universe::ItemUniverse;
use crate::pickups::PickupDrop;
use crate::ship::Ship;

/// Render the list of active missions into any `egui::Ui`.
/// When `abandon` is provided, an Abandon button is shown for each mission.
pub fn render_active_missions(
    ui: &mut egui::Ui,
    log: &MissionLog,
    catalog: &MissionCatalog,
    mut abandon: Option<&mut MessageWriter<AbandonMission>>,
) {
    ui.heading("Active");
    let mut any_active = false;
    // Sort by id: statuses is a HashMap, and unordered iteration made the
    // mission list shuffle whenever an entry was inserted.
    let mut entries: Vec<_> = log.statuses.iter().collect();
    entries.sort_by_key(|(a, _)| *a);
    for (id, status) in entries {
        if let MissionStatus::Active(progress) = status {
            any_active = true;
            let Some(def) = catalog.defs.get(id) else {
                continue;
            };
            ui.group(|ui| {
                ui.label(&def.briefing);
                ui.label(format_objective(&def.objective, progress));
                if let Some(ref mut w) = abandon
                    && ui.button("Abandon").clicked()
                {
                    w.write(AbandonMission(id.clone()));
                }
            });
        }
    }
    if !any_active {
        ui.label("(No active missions.)");
    }
}

/// Render the Bar tab inside the planet UI window.
/// Just the postings — the bar's compact job board (active missions live
/// in the mission log, one I-press away).
pub fn render_job_board(
    ui: &mut egui::Ui,
    offers: &MissionOffers,
    catalog: &MissionCatalog,
    player_free_cargo: u16,
    accept: &mut MessageWriter<AcceptMission>,
) {
    ui.heading("Job board");
    if offers.tab.is_empty() {
        ui.label("(No postings on the board.)");
    } else {
        for id in &offers.tab {
            let Some(def) = catalog.defs.get(id) else {
                continue;
            };
            render_offer(ui, id, def, player_free_cargo, accept);
        }
    }
}

fn render_offer(
    ui: &mut egui::Ui,
    id: &str,
    def: &MissionDef,
    player_free_cargo: u16,
    accept: &mut MessageWriter<AcceptMission>,
) {
    let required = def.required_cargo_space();
    let has_space = player_free_cargo >= required;
    ui.group(|ui| {
        ui.label(&def.briefing);
        if required > 0 {
            ui.label(format!(
                "Cargo required: {} units (you have {} free)",
                required, player_free_cargo
            ));
        }
        ui.add_enabled_ui(has_space, |ui| {
            let btn = ui.button("Accept");
            if btn.clicked() {
                accept.write(AcceptMission(id.to_string()));
            }
            if !has_space {
                btn.on_hover_text("Not enough free cargo space.");
            }
        });
    });
}

fn format_objective(obj: &Objective, progress: &ObjectiveProgress) -> String {
    match obj {
        Objective::TravelToSystem { system } => format!("Travel to the {} system.", system),
        Objective::LandOnPlanet { planet } => format!("Land on {}.", planet),
        Objective::CollectPickups {
            commodity,
            system,
            quantity,
        } => {
            let have = progress.collected;
            format!(
                "Collect {} units of {} in the {} system ({}/{}).",
                quantity, commodity, system, have, quantity
            )
        }
        Objective::MeetNpc {
            planet,
            npc_name,
            hint,
            ..
        } => match hint {
            Some(hint) => format!("Find {} — {}.", npc_name, hint),
            None => format!("Meet {} on {}.", npc_name, planet),
        },
        Objective::CatchNpc {
            planet,
            npc_name,
            hint,
            ..
        } => match hint {
            Some(hint) => format!("Hunt down {} — {}.", npc_name, hint),
            None => format!("Catch {} on {}.", npc_name, planet),
        },
        Objective::DestroyShips {
            system,
            count,
            target_name,
            collect,
            ..
        } => {
            let mut s = format!(
                "Destroy {} {} in the {} system ({}/{}).",
                count, target_name, system, progress.destroyed, count
            );
            if let Some(req) = collect {
                s += &format!(
                    " Collect {} {} ({}/{}).",
                    req.quantity, req.commodity, progress.collected, req.quantity
                );
            }
            s
        }
    }
}

/// FIFO queue of mission messages to surface to the player. A queue (not a
/// single slot) because several can land in the same instant — e.g. a mission
/// completes on landing and its Auto follow-up starts one frame later; with a
/// single slot the follow-up's briefing would overwrite the completion text
/// (or vice versa) and the player would never see one of them.
#[derive(Resource, Default)]
pub struct MissionToast {
    pub queue: std::collections::VecDeque<String>,
}

impl MissionToast {
    pub fn push(&mut self, text: impl Into<String>) {
        let text = text.into();
        // Dedup back-to-back identical messages (e.g. re-emitted on load).
        if self.queue.back().map(String::as_str) != Some(text.as_str()) {
            self.queue.push_back(text);
        }
    }
}

impl crate::session::SessionResource for MissionToast {
    type SaveData = ();
    fn new_session(_: &crate::item_universe::ItemUniverse) -> Self {
        Self::default()
    }
    fn from_save(_: (), _: &crate::item_universe::ItemUniverse) -> Self {
        Self::default()
    }
}

/// Egui overlay that surfaces the latest mission success/failure message.
/// Runs every frame in `EguiPrimaryContextPass`; registered only in
/// non-headless mode (see `missions_ui_plugin`).
pub fn render_toast(mut egui_contexts: EguiContexts, mut toast: ResMut<MissionToast>) {
    let Some(text) = toast.queue.front().cloned() else {
        return;
    };
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };
    let mut open = true;
    let remaining = toast.queue.len() - 1;
    egui::Window::new("Mission Update")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 40.0))
        .open(&mut open)
        .show(ctx, |ui| {
            ui.label(text);
            let label = if remaining > 0 {
                format!("Next ({remaining} more)")
            } else {
                "Dismiss".to_string()
            };
            // Keyboard focus is globally surrendered outside the main menu
            // (see drop_egui_keyboard_focus), so Space (= FIRE) can never
            // "click" this button and swallow unread messages.
            if ui.button(label).clicked() {
                toast.queue.pop_front();
            }
        });
    if !open {
        toast.queue.pop_front();
    }
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub enum MissionLogTab {
    #[default]
    Missions,
    Story,
    Info,
    Cargo,
}

#[derive(Resource, Default)]
pub struct MissionLogOpen {
    pub open: bool,
    pub tab: MissionLogTab,
}

impl crate::session::SessionResource for MissionLogOpen {
    type SaveData = ();
    fn new_session(_: &crate::item_universe::ItemUniverse) -> Self {
        Self::default()
    }
    fn from_save(_: (), _: &crate::item_universe::ItemUniverse) -> Self {
        Self::default()
    }
}

fn toggle_mission_log(keyboard: Res<ButtonInput<KeyCode>>, mut state: ResMut<MissionLogOpen>) {
    // Pausing is derived from `open` by sync_ui_pause (main.rs).
    if keyboard.just_pressed(KeyCode::KeyI) {
        state.open = !state.open;
    }
}

fn render_mission_log(
    mut egui_contexts: EguiContexts,
    mut state: ResMut<MissionLogOpen>,
    log: Res<MissionLog>,
    catalog: Res<MissionCatalog>,
    unlocks: Res<super::log::PlayerUnlocks>,
    item_universe: Res<ItemUniverse>,
    game_state: Res<PlayerGameState>,
    mut player_query: Query<(&mut Ship, &Transform), With<Player>>,
    mut abandon: MessageWriter<AbandonMission>,
    mut pickup_drop: MessageWriter<PickupDrop>,
    standings: Res<crate::standing::FactionStandings>,
) {
    if !state.open {
        return;
    }
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };
    let Ok((mut ship, transform)) = player_query.single_mut() else {
        return;
    };
    let mut close = false;
    egui::Window::new("Pilot Info")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut state.tab, MissionLogTab::Missions, "Missions");
                ui.selectable_value(&mut state.tab, MissionLogTab::Story, "Story");
                ui.selectable_value(&mut state.tab, MissionLogTab::Info, "Info");
                ui.selectable_value(&mut state.tab, MissionLogTab::Cargo, "Cargo");
            });
            ui.separator();
            match state.tab {
                MissionLogTab::Missions => {
                    render_active_missions(ui, &log, &catalog, Some(&mut abandon));
                }
                MissionLogTab::Story => {
                    render_story_tab(ui, &log, &unlocks, &item_universe);
                }
                MissionLogTab::Info => {
                    render_info_tab(ui, &ship, &game_state, &item_universe, &standings);
                }
                MissionLogTab::Cargo => {
                    let forward = (transform.rotation * Vec3::Y).xy();
                    let drop_distance = ship.data.radius + crate::pickups::PICKUP_RADIUS + 5.0;
                    let location = transform.translation.xy() - forward * drop_distance;
                    render_cargo_tab(ui, &mut ship, location, &item_universe, &mut pickup_drop);
                }
            }
            ui.separator();
            if ui.button("Close  [I]").clicked() {
                close = true;
            }
        });
    if close {
        state.open = false;
    }
}

/// Player-facing, partially-obscured storyline flow chart. Faction-colored
/// layered DAG: completed missions read fully, the next available shows as a
/// blank in faction color, unmet-requirement missions stay hidden, and closed
/// branches appear locked. See [`super::story_chart`].
fn render_story_tab(
    ui: &mut egui::Ui,
    log: &MissionLog,
    unlocks: &super::log::PlayerUnlocks,
    universe: &ItemUniverse,
) {
    use super::story_chart::{NodeUi, build_story_graph};

    let graph = build_story_graph(log, unlocks, universe);
    if graph.nodes.is_empty() {
        ui.label("No storylines discovered yet. Take work from the people you\nmeet on planet surfaces to begin a chain.");
        return;
    }

    // Legend.
    ui.horizontal_wrapped(|ui| {
        let chip = |ui: &mut egui::Ui, label: &str, col: egui::Color32| {
            let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
            ui.painter().rect_filled(rect, 2.0, col);
            ui.label(label);
            ui.add_space(8.0);
        };
        chip(ui, "done", egui::Color32::from_rgb(90, 170, 100));
        chip(ui, "active", egui::Color32::from_rgb(230, 200, 90));
        chip(ui, "next", egui::Color32::from_gray(110));
        chip(ui, "failed", egui::Color32::from_rgb(170, 70, 70));
        chip(ui, "closed", egui::Color32::from_gray(60));
    });
    ui.separator();

    // Layout metrics (chart-space pixels).
    const NW: f32 = 128.0;
    const NH: f32 = 34.0;
    const COL_GAP: f32 = 40.0;
    const ROW_GAP: f32 = 10.0;
    let width = graph.cols as f32 * (NW + COL_GAP) + COL_GAP;
    let height = graph.rows.max(1) as f32 * (NH + ROW_GAP) + ROW_GAP;

    let node_at = |col: u32, row: u32| -> egui::Rect {
        let x = COL_GAP + col as f32 * (NW + COL_GAP);
        let y = ROW_GAP + row as f32 * (NH + ROW_GAP);
        egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(NW, NH))
    };

    egui::ScrollArea::both()
        .max_height(460.0)
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let (canvas, _) =
                ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
            let origin = canvas.min.to_vec2();
            let p = ui.painter_at(canvas);

            // Edges first (under nodes).
            for e in &graph.edges {
                let a = node_at(graph.nodes[e.from].col, graph.nodes[e.from].row).translate(origin);
                let b = node_at(graph.nodes[e.to].col, graph.nodes[e.to].row).translate(origin);
                let col = if e.satisfied {
                    egui::Color32::from_gray(130)
                } else {
                    egui::Color32::from_gray(70)
                };
                p.line_segment(
                    [
                        egui::pos2(a.max.x, a.center().y),
                        egui::pos2(b.min.x, b.center().y),
                    ],
                    egui::Stroke::new(1.5, col),
                );
            }

            // Nodes.
            for n in &graph.nodes {
                let rect = node_at(n.col, n.row).translate(origin);
                let base = egui::Color32::from_rgb(n.color[0], n.color[1], n.color[2]);
                let (fill, stroke, text_col) = match n.ui {
                    NodeUi::Completed => (
                        base,
                        egui::Stroke::new(1.0, egui::Color32::from_gray(20)),
                        text_on(base),
                    ),
                    NodeUi::Active => (
                        base,
                        egui::Stroke::new(2.5, egui::Color32::from_rgb(240, 210, 90)),
                        text_on(base),
                    ),
                    // Next: faction hue retained but muted + no text.
                    NodeUi::Next => (
                        mute(base, 0.32),
                        egui::Stroke::new(1.0, mute(base, 0.6)),
                        egui::Color32::TRANSPARENT,
                    ),
                    NodeUi::Failed => (
                        mute(base, 0.4),
                        egui::Stroke::new(1.5, egui::Color32::from_rgb(150, 60, 60)),
                        egui::Color32::from_gray(150),
                    ),
                    // Closed branch: barely-there hue, locked.
                    NodeUi::Impossible => (
                        mute(base, 0.15),
                        egui::Stroke::new(1.0, egui::Color32::from_gray(70)),
                        egui::Color32::TRANSPARENT,
                    ),
                };
                p.rect(rect, 5.0, fill, stroke, egui::StrokeKind::Inside);

                if n.ui.shows_name() {
                    p.text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        &n.label,
                        egui::FontId::proportional(11.0),
                        text_col,
                    );
                    if !n.grants.is_empty() {
                        let g: Vec<String> = n
                            .grants
                            .iter()
                            .map(|u| u.replace("_license", "").replace('_', " "))
                            .collect();
                        p.text(
                            egui::pos2(rect.center().x, rect.max.y - 5.0),
                            egui::Align2::CENTER_CENTER,
                            format!("★ {}", g.join(", ")),
                            egui::FontId::proportional(8.0),
                            egui::Color32::from_rgb(240, 210, 110),
                        );
                    }
                } else if matches!(n.ui, NodeUi::Next) {
                    p.text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "?",
                        egui::FontId::proportional(15.0),
                        mute(base, 0.9),
                    );
                } else if matches!(n.ui, NodeUi::Impossible) {
                    p.text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "\u{1f512}",
                        egui::FontId::proportional(12.0),
                        egui::Color32::from_gray(90),
                    );
                }
            }
        });
}

/// Desaturate + darken a color toward grey by `1.0 - keep` (keep in [0,1]).
fn mute(c: egui::Color32, keep: f32) -> egui::Color32 {
    let grey = 40.0;
    let mix = |v: u8| (v as f32 * keep + grey * (1.0 - keep)) as u8;
    egui::Color32::from_rgb(mix(c.r()), mix(c.g()), mix(c.b()))
}

/// Pick black or white text for contrast against `bg`.
fn text_on(bg: egui::Color32) -> egui::Color32 {
    let lum = 0.299 * bg.r() as f32 + 0.587 * bg.g() as f32 + 0.114 * bg.b() as f32;
    if lum > 140.0 {
        egui::Color32::from_gray(15)
    } else {
        egui::Color32::from_gray(235)
    }
}

fn render_info_tab(
    ui: &mut egui::Ui,
    ship: &Ship,
    game_state: &PlayerGameState,
    item_universe: &ItemUniverse,
    standings: &crate::standing::FactionStandings,
) {
    render_standings(ui, standings);
    ui.heading("Pilot");
    egui::Grid::new("pilot_info_grid")
        .num_columns(2)
        .striped(true)
        .show(ui, |ui| {
            ui.label("Name:");
            ui.label(&game_state.pilot_name);
            ui.end_row();
            ui.label("Credits:");
            ui.label(ship.credits.to_string());
            ui.end_row();
        });

    ui.add_space(6.0);
    ui.heading("Ship");
    egui::Grid::new("ship_stats_grid")
        .num_columns(2)
        .striped(true)
        .show(ui, |ui| {
            ui.label("Type:");
            ui.label(&ship.data.display_name);
            ui.end_row();
            ui.label("Hull:");
            ui.label(format!("{} / {}", ship.health, ship.max_health()));
            ui.end_row();
            ui.label("Max speed:");
            ui.label(format!(
                "{} m/s",
                (ship.data.max_speed * ship.mod_stats.speed_mult) as i32
            ));
            ui.end_row();
            ui.label("Thrust:");
            ui.label(format!(
                "{} N",
                (ship.data.thrust * ship.mod_stats.thrust_mult) as i32
            ));
            ui.end_row();
            ui.label("Turning torque:");
            ui.label(format!(
                "{} N·m",
                (ship.data.torque * ship.mod_stats.torque_mult) as i32
            ));
            ui.end_row();
            ui.label("Cargo space:");
            ui.label(format!(
                "{} / {} t",
                ship.data.cargo_space - ship.remaining_cargo_space(),
                ship.data.cargo_space
            ));
            ui.end_row();
            ui.label("Item slots:");
            ui.label(format!(
                "{} / {}",
                ship.data.item_space as i32 - ship.remaining_item_space(),
                ship.data.item_space
            ));
            ui.end_row();
        });

    ui.add_space(6.0);
    ui.heading("Equipped");
    let mut entries: Vec<(&String, &crate::weapons::WeaponSystem)> =
        ship.weapon_systems.iter_all().collect();
    if entries.is_empty() {
        ui.label("(Nothing equipped.)");
    } else {
        entries.sort_by(|a, b| a.0.cmp(b.0));
        egui::Grid::new("equipped_grid")
            .num_columns(3)
            .striped(true)
            .show(ui, |ui| {
                ui.strong("Item");
                ui.strong("Count");
                ui.strong("Ammo");
                ui.end_row();
                for (key, ws) in entries {
                    let display = item_universe
                        .outfitter_items
                        .get(key)
                        .map(|i| i.display_name())
                        .unwrap_or(key);
                    ui.label(display);
                    ui.label(ws.number.to_string());
                    ui.label(match ws.ammo_quantity {
                        Some(q) => q.to_string(),
                        None => "n/a".to_string(),
                    });
                    ui.end_row();
                }
            });
    }
}

fn render_cargo_tab(
    ui: &mut egui::Ui,
    ship: &mut Ship,
    location: Vec2,
    item_universe: &ItemUniverse,
    pickup_drop: &mut MessageWriter<PickupDrop>,
) {
    let used = ship.data.cargo_space - ship.remaining_cargo_space();
    ui.label(format!(
        "Cargo hold: {} / {} t ({} free)",
        used,
        ship.data.cargo_space,
        ship.remaining_cargo_space()
    ));
    ui.separator();

    let mut entries: Vec<(String, u16)> = ship
        .cargo
        .iter()
        .filter(|(_, q)| **q > 0)
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    if entries.is_empty() {
        ui.label("(Cargo hold is empty.)");
        return;
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    let mut to_jettison: Option<String> = None;
    egui::Grid::new("cargo_grid")
        .num_columns(4)
        .striped(true)
        .show(ui, |ui| {
            ui.strong("Commodity");
            ui.strong("Tons");
            ui.strong("Reserved");
            ui.label("");
            ui.end_row();
            for (commodity, qty) in &entries {
                let reserved = *ship.reserved_cargo.get(commodity).unwrap_or(&0);
                let droppable = qty.saturating_sub(reserved);
                let display = item_universe
                    .commodities
                    .get(commodity)
                    .map(|c| c.display_name.as_str())
                    .unwrap_or(commodity);
                ui.label(display);
                ui.label(qty.to_string());
                if reserved > 0 {
                    ui.label(reserved.to_string());
                } else {
                    ui.label("-");
                }
                ui.add_enabled_ui(droppable > 0, |ui| {
                    let btn = ui.button("Jettison 1 t");
                    if btn.clicked() {
                        to_jettison = Some(commodity.clone());
                    }
                    if droppable == 0 {
                        btn.on_hover_text("All units are mission-locked.");
                    }
                });
                ui.end_row();
            }
        });

    if let Some(commodity) = to_jettison
        && ship.jettison_cargo(&commodity)
    {
        pickup_drop.write(PickupDrop {
            location,
            commodity,
            quantity: 1,
        });
    }
}

pub fn missions_ui_plugin(app: &mut App) {
    use crate::session::SessionResourceExt;
    use bevy_egui::EguiPrimaryContextPass;
    app.init_session_resource::<MissionToast>()
        .init_session_resource::<MissionLogOpen>()
        .add_systems(
            Update,
            (
                toggle_mission_log.run_if(not(in_state(crate::PlayState::MainMenu))),
                drain_completion_toasts,
            ),
        )
        .add_systems(
            EguiPrimaryContextPass,
            (
                render_toast,
                render_mission_log.run_if(not(in_state(crate::PlayState::MainMenu))),
            ),
        );
}

pub fn drain_completion_toasts(
    mut completed: MessageReader<MissionCompleted>,
    mut failed: MessageReader<MissionFailed>,
    mut started: MessageReader<MissionStarted>,
    catalog: Res<MissionCatalog>,
    mut toast: ResMut<MissionToast>,
) {
    for MissionCompleted(id) in completed.read() {
        if let Some(def) = catalog.defs.get(id) {
            toast.push(def.success_text.clone());
        }
    }
    for MissionFailed(id) in failed.read() {
        if let Some(def) = catalog.defs.get(id) {
            toast.push(def.failure_text.clone());
        }
    }
    // Auto missions start themselves — there is no offer dialog, so unless we
    // surface the briefing here the player never sees the opening message
    // (it was previously only visible in the mission log).
    for MissionStarted(id) in started.read() {
        if let Some(def) = catalog.defs.get(id)
            && matches!(def.offer, OfferKind::Auto)
        {
            toast.push(def.briefing.clone());
        }
    }
}

/// Faction standings summary for the info tab.
fn render_standings(ui: &mut egui::Ui, standings: &crate::standing::FactionStandings) {
    use crate::standing::{ARREST_THRESHOLD, ENGAGE_THRESHOLD};
    ui.heading("Faction Standing");
    if standings.0.is_empty() {
        ui.label("(No faction has an opinion of you yet.)");
    } else {
        let mut entries: Vec<(&String, &f32)> = standings.0.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        egui::Grid::new("standings_grid")
            .num_columns(3)
            .striped(true)
            .show(ui, |ui| {
                for (faction, &value) in entries {
                    let (label, color) = if value <= ARREST_THRESHOLD {
                        ("wanted", egui::Color32::from_rgb(235, 60, 60))
                    } else if value <= ENGAGE_THRESHOLD {
                        ("hostile", egui::Color32::from_rgb(230, 120, 60))
                    } else if value < 0.0 {
                        ("distrusted", egui::Color32::from_rgb(220, 190, 90))
                    } else if value > 20.0 {
                        ("trusted", egui::Color32::from_rgb(90, 220, 120))
                    } else {
                        ("neutral", egui::Color32::GRAY)
                    };
                    ui.label(faction);
                    ui.label(format!("{value:+.0}"));
                    ui.colored_label(color, label);
                    ui.end_row();
                }
            });
    }
    ui.add_space(6.0);
    ui.separator();
}
