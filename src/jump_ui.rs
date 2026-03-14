use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};

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

fn toggle_jump_ui(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut open: ResMut<JumpUiOpen>,
) {
    if keyboard.just_pressed(KeyCode::KeyJ) {
        open.0 = !open.0;
    }
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
    let Some(system) = item_universe.star_systems.get(&current_system.0) else {
        return;
    };

    let connections: Vec<String> = system.connections.clone();

    egui::Window::new("Jump to System")
        .collapsible(false)
        .resizable(false)
        .show(ctx, |ui| {
            if connections.is_empty() {
                ui.label("No jump routes from this system.");
            }
            for dest in &connections {
                ui.horizontal(|ui| {
                    ui.label(dest.replace('_', " ").to_uppercase());
                    if ui.button("Jump").clicked() {
                        travel_ctx.destination = dest.clone();
                        travel_ctx.timer =
                            Timer::from_seconds(3.0, TimerMode::Once);
                        state.set(GameState::Traveling);
                        open.0 = false;
                    }
                });
            }
            ui.separator();
            if ui.button("Cancel").clicked() {
                open.0 = false;
            }
        });
}
