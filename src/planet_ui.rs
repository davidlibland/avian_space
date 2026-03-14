use crate::{
    CurrentStarSystem, GameState, Player, Ship, item_universe::ItemUniverse,
};
use avian2d::prelude::{Physics, PhysicsTime, Position};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass};

pub fn planet_ui_plugin(app: &mut App) {
    app.add_plugins(EguiPlugin::default())
        .insert_resource(LandedContext::default())
        .add_systems(
            EguiPrimaryContextPass,
            planet_ui.run_if(in_state(GameState::Landed)),
        )
        .add_systems(OnEnter(GameState::Landed), pause_physics)
        .add_systems(OnExit(GameState::Landed), unpause_physics)
        .add_systems(OnEnter(GameState::Flying), place_player_at_launch_site);
}

#[derive(Resource, Default)]
pub struct LandedContext {
    /// Name of the planet the player is docked at.
    pub planet_name: Option<String>,
    /// World position of the planet — used to place the ship on launch.
    pub planet_position: Option<Vec2>,
    pub active_tab: PlanetTab,
}

#[derive(PartialEq, Default)]
pub enum PlanetTab {
    #[default]
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
    current_system: Res<CurrentStarSystem>,
    item_universe: Res<ItemUniverse>,
) {
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };
    let Some(current_system) = item_universe.star_systems.get(&current_system.0) else {
        return;
    };
    let Some(planet_name) = &landed.planet_name.clone() else {
        return;
    };
    let Some(planet) = current_system.planets.get(planet_name) else {
        return;
    };
    egui::Window::new(format!("Docked on {}", planet_name)).show(ctx, |ui| {
        ui.label(&planet.description);
        ui.separator();
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
                    ship.health = ship.data.max_health;
                }
            }
            if ui.button("Launch").clicked() {
                state.set(GameState::Flying);
            }
        });
        match landed.active_tab {
            PlanetTab::Trade => {
                if let Ok(mut ship) = player_query.single_mut() {
                    ui.label(format!("Credits: {}", ship.credits));
                    ui.separator();
                    egui::Grid::new("trade_grid")
                        .num_columns(5)
                        .striped(true)
                        .show(ui, |ui| {
                            ui.strong("Commodity");
                            ui.strong("Price");
                            ui.strong("Cargo");
                            ui.label("");
                            ui.label("");
                            ui.end_row();
                            let commodities: Vec<(String, u16)> = planet
                                .commodities
                                .iter()
                                .map(|(k, v)| (k.clone(), *v))
                                .collect();
                            for (commodity, price) in commodities {
                                let qty = *ship.cargo.get(&commodity).unwrap_or(&0);
                                ui.label(&commodity);
                                ui.label(price.to_string());
                                ui.label(qty.to_string());
                                if ui.button("Buy").clicked() {
                                    ship.buy_cargo(&commodity, 1, price as u128);
                                }
                                if ui.button("Sell").clicked() {
                                    ship.sell_cargo(&commodity, 1, price as u128);
                                }
                                ui.end_row();
                            }
                        });
                }
            }
            _ => {}
        }
    });
}

/// When re-entering Flying from a landing, place the ship at the planet and clear the context.
fn place_player_at_launch_site(
    mut landed: ResMut<LandedContext>,
    mut player_query: Query<(&mut Transform, &mut Position), With<Player>>,
) {
    if let Some(pos) = landed.planet_position.take() {
        if let Ok((mut tf, mut physics_pos)) = player_query.single_mut() {
            tf.translation = pos.extend(tf.translation.z);
            physics_pos.0 = pos;
        }
    }
    landed.planet_name = None;
}

pub fn pause_physics(mut time: ResMut<Time<Physics>>) {
    time.pause();
}

pub fn unpause_physics(mut time: ResMut<Time<Physics>>) {
    time.unpause();
}
