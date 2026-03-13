use crate::{
    CurrentStarSystem, GameState, Player, Ship, item_universe::ItemUniverse, planets::Planet,
};
use avian2d::prelude::{Physics, PhysicsTime};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass};

pub fn planet_ui_plugin(app: &mut App) {
    app.add_plugins(EguiPlugin::default())
        .insert_resource::<LandedContext>(LandedContext {
            planet: None,
            active_tab: PlanetTab::Trade,
        })
        .add_systems(
            EguiPrimaryContextPass,
            planet_ui.run_if(in_state(GameState::Landed)),
        )
        .add_systems(OnEnter(GameState::Landed), pause_physics)
        .add_systems(OnExit(GameState::Landed), unpause_physics);
}

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
    planet_query: Query<&Planet>,
    current_system: Res<CurrentStarSystem>,
    item_universe: Res<ItemUniverse>,
) {
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };
    let Some(current_system) = item_universe.star_systems.get(&current_system.0) else {
        return;
    };
    let Some(planet_name) = landed
        .planet
        .and_then(|e| planet_query.get(e).map(|p| p.0.clone()).ok())
    else {
        return;
    };
    // Lookup the planet data:
    let Some(planet) = current_system.planets.get(&planet_name) else {
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
