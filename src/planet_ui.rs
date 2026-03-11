use crate::{GameState, Player, Ship};
use avian2d::prelude::{Physics, PhysicsTime};
use bevy::prelude::*;
use bevy_egui::EguiContexts;

#[derive(Resource)]
pub struct LandedContext {
    pub planet: Option<Entity>,
    pub active_tab: PlanetTab,
}

#[derive(PartialEq)]
pub enum PlanetTab {
    Trade,
    Shipyard,
    Bar,
    Missions,
}

pub fn planet_ui(
    mut egui_contexts: EguiContexts,
    mut state: ResMut<NextState<GameState>>,
    mut landed: ResMut<LandedContext>,
    mut player_query: Query<&mut Ship, With<Player>>,
) {
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };
    egui::Window::new("Docked").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut landed.active_tab, PlanetTab::Trade, "Trade");
            ui.selectable_value(&mut landed.active_tab, PlanetTab::Shipyard, "Shipyard");
            ui.selectable_value(&mut landed.active_tab, PlanetTab::Bar, "Bar");
            ui.selectable_value(&mut landed.active_tab, PlanetTab::Missions, "Missions");
        });
        ui.separator();
        ui.horizontal(|ui| {
            if ui.button("Repair").clicked() {
                if let Ok(mut ship) = player_query.single_mut() {
                    ship.health = ship.max_health;
                }
            }
            if ui.button("Launch").clicked() {
                state.set(GameState::Flying);
            }
        });
        // match ctx.active_tab {
        //     PlanetTab::Trade => trade_ui(ui /* ... */),
        //     PlanetTab::Shipyard => shipyard_ui(ui /* ... */),
        //     PlanetTab::Bar => bar_ui(ui /* ... */),
        //     PlanetTab::Missions => missions_ui(ui /* ... */),
        // }
    });
}

pub fn pause_physics(mut time: ResMut<Time<Physics>>) {
    time.pause();
}

pub fn unpause_physics(mut time: ResMut<Time<Physics>>) {
    time.unpause();
}
