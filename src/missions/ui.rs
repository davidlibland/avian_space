use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use super::events::*;
use super::log::{MissionCatalog, MissionLog, MissionOffers};
use super::types::*;
use crate::game_save::PlayerGameState;
use crate::item_universe::ItemUniverse;
use crate::pickups::PickupDrop;
use crate::ship::Ship;
use crate::Player;

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
    for (id, status) in &log.statuses {
        if let MissionStatus::Active(progress) = status {
            any_active = true;
            let Some(def) = catalog.defs.get(id) else {
                continue;
            };
            ui.group(|ui| {
                ui.label(&def.briefing);
                ui.label(format_objective(&def.objective, progress));
                if let Some(ref mut w) = abandon {
                    if ui.button("Abandon").clicked() {
                        w.write(AbandonMission(id.clone()));
                    }
                }
            });
        }
    }
    if !any_active {
        ui.label("(No active missions.)");
    }
}

/// Render the Bar tab inside the planet UI window.
pub fn render_missions_tab(
    ui: &mut egui::Ui,
    log: &MissionLog,
    offers: &MissionOffers,
    catalog: &MissionCatalog,
    player_free_cargo: u16,
    accept: &mut MessageWriter<AcceptMission>,
    abandon: &mut MessageWriter<AbandonMission>,
) {
    render_active_missions(ui, log, catalog, Some(abandon));

    ui.separator();
    ui.heading("Available");
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
        Objective::MeetNpc { planet, npc_name, .. } => {
            format!("Meet {} on {}.", npc_name, planet)
        }
        Objective::CatchNpc { planet, npc_name, .. } => {
            format!("Catch {} on {}.", npc_name, planet)
        }
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

#[derive(Resource, Default)]
pub struct MissionToast {
    pub text: Option<String>,
}

impl crate::session::SessionResource for MissionToast {
    type SaveData = ();
    fn new_session(_: &crate::item_universe::ItemUniverse) -> Self { Self::default() }
    fn from_save(_: (), _: &crate::item_universe::ItemUniverse) -> Self { Self::default() }
}

/// Egui overlay that surfaces the latest mission success/failure message.
/// Runs every frame in `EguiPrimaryContextPass`; registered only in
/// non-headless mode (see `missions_ui_plugin`).
pub fn render_toast(mut egui_contexts: EguiContexts, mut toast: ResMut<MissionToast>) {
    let Some(text) = toast.text.clone() else {
        return;
    };
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };
    let mut open = true;
    egui::Window::new("Mission Update")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 40.0))
        .open(&mut open)
        .show(ctx, |ui| {
            ui.label(text);
            if ui.button("Dismiss").clicked() {
                toast.text = None;
            }
        });
    if !open {
        toast.text = None;
    }
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub enum MissionLogTab {
    #[default]
    Missions,
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
    fn new_session(_: &crate::item_universe::ItemUniverse) -> Self { Self::default() }
    fn from_save(_: (), _: &crate::item_universe::ItemUniverse) -> Self { Self::default() }
}

fn toggle_mission_log(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<MissionLogOpen>,
    mut virtual_time: ResMut<Time<Virtual>>,
) {
    if keyboard.just_pressed(KeyCode::KeyI) {
        state.open = !state.open;
        if state.open {
            virtual_time.pause();
        } else {
            virtual_time.unpause();
        }
    }
}

fn render_mission_log(
    mut egui_contexts: EguiContexts,
    mut state: ResMut<MissionLogOpen>,
    log: Res<MissionLog>,
    catalog: Res<MissionCatalog>,
    item_universe: Res<ItemUniverse>,
    game_state: Res<PlayerGameState>,
    mut player_query: Query<(&mut Ship, &Transform), With<Player>>,
    mut abandon: MessageWriter<AbandonMission>,
    mut pickup_drop: MessageWriter<PickupDrop>,
    mut virtual_time: ResMut<Time<Virtual>>,
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
                ui.selectable_value(&mut state.tab, MissionLogTab::Info, "Info");
                ui.selectable_value(&mut state.tab, MissionLogTab::Cargo, "Cargo");
            });
            ui.separator();
            match state.tab {
                MissionLogTab::Missions => {
                    render_active_missions(ui, &log, &catalog, Some(&mut abandon));
                }
                MissionLogTab::Info => {
                    render_info_tab(ui, &ship, &game_state, &item_universe);
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
        virtual_time.unpause();
    }
}

fn render_info_tab(
    ui: &mut egui::Ui,
    ship: &Ship,
    game_state: &PlayerGameState,
    item_universe: &ItemUniverse,
) {
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
            ui.label(format!("{} / {}", ship.health, ship.data.max_health));
            ui.end_row();
            ui.label("Max speed:");
            ui.label(format!("{} m/s", ship.data.max_speed as i32));
            ui.end_row();
            ui.label("Thrust:");
            ui.label(format!("{} N", ship.data.thrust as i32));
            ui.end_row();
            ui.label("Turning torque:");
            ui.label(format!("{} N·m", ship.data.torque as i32));
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

    if let Some(commodity) = to_jettison {
        if ship.jettison_cargo(&commodity) {
            pickup_drop.write(PickupDrop {
                location,
                commodity,
                quantity: 1,
            });
        }
    }
}

pub fn missions_ui_plugin(app: &mut App) {
    use bevy_egui::EguiPrimaryContextPass;
    use crate::session::SessionResourceExt;
    app.init_session_resource::<MissionToast>()
        .init_session_resource::<MissionLogOpen>()
        .add_systems(
            Update,
            (
                toggle_mission_log.run_if(in_state(crate::PlayState::Flying)),
                drain_completion_toasts,
            ),
        )
        .add_systems(
            EguiPrimaryContextPass,
            (
                render_toast,
                render_mission_log.run_if(in_state(crate::PlayState::Flying)),
            ),
        );
}

pub fn drain_completion_toasts(
    mut completed: MessageReader<MissionCompleted>,
    mut failed: MessageReader<MissionFailed>,
    catalog: Res<MissionCatalog>,
    mut toast: ResMut<MissionToast>,
) {
    for MissionCompleted(id) in completed.read() {
        if let Some(def) = catalog.defs.get(id) {
            toast.text = Some(def.success_text.clone());
        }
    }
    for MissionFailed(id) in failed.read() {
        if let Some(def) = catalog.defs.get(id) {
            toast.text = Some(def.failure_text.clone());
        }
    }
}
