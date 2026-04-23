use crate::game_save::{Gender, PlayerGameState, list_saves, load_save};
use crate::item_universe::ItemUniverse;
use crate::session::PendingSessionLoad;
use crate::{CurrentStarSystem, PlayState};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass};

// ── Local state ───────────────────────────────────────────────────────────────

#[derive(Resource, Default)]
struct MainMenuState {
    new_pilot_name: String,
    new_pilot_gender: Gender,
    saves: Vec<String>,
}

// ── Systems ───────────────────────────────────────────────────────────────────

fn refresh_saves(mut menu_state: ResMut<MainMenuState>) {
    menu_state.saves = list_saves();
}

/// Clear any UI-pause that was active when the player bailed out of Flying.
/// Otherwise `Time<Virtual>` stays paused across a load, freezing physics
/// while real-time systems (input, audio) keep running — producing the
/// "thruster sfx plays but the ship doesn't move" symptom.
fn reset_pause_on_menu(mut virtual_time: ResMut<Time<Virtual>>) {
    virtual_time.unpause();
}

fn main_menu_ui(
    mut commands: Commands,
    mut egui_contexts: EguiContexts,
    mut menu_state: ResMut<MainMenuState>,
    mut game_state: ResMut<PlayerGameState>,
    mut current_system: ResMut<CurrentStarSystem>,
    mut next_state: ResMut<NextState<PlayState>>,
    item_universe: Res<ItemUniverse>,
) {
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };

    egui::CentralPanel::default().show(ctx, |ui| {
        ui.vertical_centered(|ui| {
            ui.add_space(80.0);

            ui.heading(egui::RichText::new("AVIAN SPACE").size(48.0));
            ui.add_space(48.0);

            // ── New pilot ─────────────────────────────────────────────────
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.set_min_width(300.0);
                ui.label("New Pilot");
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut menu_state.new_pilot_name)
                            .hint_text("Pilot name…")
                            .desired_width(200.0),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Gender:");
                    ui.radio_value(&mut menu_state.new_pilot_gender, Gender::Boy, "Boy");
                    ui.radio_value(&mut menu_state.new_pilot_gender, Gender::Girl, "Girl");
                });
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    let ready = !menu_state.new_pilot_name.trim().is_empty();
                    if ui.add_enabled(ready, egui::Button::new("Create")).clicked() {
                        let name = menu_state.new_pilot_name.trim().to_string();
                        let gender = menu_state.new_pilot_gender;
                        *game_state =
                            PlayerGameState::new_pilot(&name, gender, &item_universe);
                        current_system.0 = game_state.current_star_system.clone();
                        // No PendingSessionLoad — session resources start fresh
                        // via their new_session() defaults.
                        next_state.set(PlayState::Flying);
                    }
                });
            });

            // ── Load pilot ────────────────────────────────────────────────
            if !menu_state.saves.is_empty() {
                ui.add_space(24.0);
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.set_min_width(300.0);
                    ui.label("Load Pilot");
                    ui.add_space(4.0);
                    let saves = menu_state.saves.clone();
                    for save_name in &saves {
                        if ui
                            .add_sized([300.0, 28.0], egui::Button::new(save_name))
                            .clicked()
                        {
                            if let Some(save) = load_save(save_name) {
                                current_system.0 = save.current_star_system.clone();
                                // Store the resources map for session resources to
                                // consume on entering Flying.
                                commands.insert_resource(PendingSessionLoad {
                                    resources: save.resources.clone(),
                                });
                                *game_state =
                                    PlayerGameState::from_save(&save, &item_universe);
                                next_state.set(PlayState::Flying);
                            }
                        }
                    }
                });
            }
        });
    });
}

// ── Plugin ────────────────────────────────────────────────────────────────────

pub fn main_menu_plugin(app: &mut App) {
    app.init_resource::<MainMenuState>()
        .add_systems(
            OnEnter(PlayState::MainMenu),
            (refresh_saves, reset_pause_on_menu),
        )
        .add_systems(
            EguiPrimaryContextPass,
            main_menu_ui.run_if(in_state(PlayState::MainMenu)),
        );
}
