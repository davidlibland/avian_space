use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use super::events::*;
use super::log::{MissionCatalog, MissionLog, MissionOffers};
use super::types::*;

/// Render the Missions tab inside the planet UI window.
pub fn render_missions_tab(
    ui: &mut egui::Ui,
    log: &MissionLog,
    offers: &MissionOffers,
    catalog: &MissionCatalog,
    player_free_cargo: u16,
    accept: &mut MessageWriter<AcceptMission>,
    abandon: &mut MessageWriter<AbandonMission>,
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
                if ui.button("Abandon").clicked() {
                    abandon.write(AbandonMission(id.clone()));
                }
            });
        }
    }
    if !any_active {
        ui.label("(No active missions.)");
    }

    ui.separator();
    ui.heading("Available");
    if offers.tab.is_empty() {
        ui.label("(No postings on the board.)");
    } else {
        for id in &offers.tab {
            let Some(def) = catalog.defs.get(id) else {
                continue;
            };
            render_offer(ui, id, def, player_free_cargo, accept, None);
        }
    }
}

/// Render the Bar tab (for the currently-landed planet).
pub fn render_bar_tab(
    ui: &mut egui::Ui,
    planet_name: &str,
    offers: &MissionOffers,
    catalog: &MissionCatalog,
    player_free_cargo: u16,
    accept: &mut MessageWriter<AcceptMission>,
    decline: &mut MessageWriter<DeclineMission>,
) {
    ui.heading("The Bar");
    ui.label("Patrons hunch over drinks. Some glance up as you enter.");
    ui.separator();
    let Some(ids) = offers.bar.get(planet_name) else {
        ui.label("(Nobody has work for you tonight.)");
        return;
    };
    if ids.is_empty() {
        ui.label("(Nobody has work for you tonight.)");
        return;
    }
    for id in ids {
        let Some(def) = catalog.defs.get(id) else {
            continue;
        };
        render_offer(ui, id, def, player_free_cargo, accept, Some(decline));
    }
}

fn render_offer(
    ui: &mut egui::Ui,
    id: &str,
    def: &MissionDef,
    player_free_cargo: u16,
    accept: &mut MessageWriter<AcceptMission>,
    decline: Option<&mut MessageWriter<DeclineMission>>,
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
        ui.horizontal(|ui| {
            ui.add_enabled_ui(has_space, |ui| {
                let btn = ui.button("Accept");
                if btn.clicked() {
                    accept.write(AcceptMission(id.to_string()));
                }
                if !has_space {
                    btn.on_hover_text("Not enough free cargo space.");
                }
            });
            if let Some(decline) = decline {
                if ui.button("Decline").clicked() {
                    decline.write(DeclineMission(id.to_string()));
                }
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
    }
}

#[derive(Resource, Default)]
pub struct MissionToast {
    pub text: Option<String>,
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

pub fn missions_ui_plugin(app: &mut App) {
    use bevy_egui::EguiPrimaryContextPass;
    app.add_systems(EguiPrimaryContextPass, render_toast);
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
