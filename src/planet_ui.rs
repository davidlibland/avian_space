use crate::{
    CurrentStarSystem, PlayState, Player, Ship, WeaponSystems, item_universe::ItemUniverse,
    ship::BuyShip,
};
use avian2d::prelude::{LinearVelocity, Physics, PhysicsTime, Position};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass};

pub fn planet_ui_plugin(app: &mut App) {
    app.add_plugins(EguiPlugin::default())
        .insert_resource(LandedContext::default())
        .add_systems(
            EguiPrimaryContextPass,
            planet_ui.run_if(in_state(PlayState::Landed)),
        )
        .add_systems(OnEnter(PlayState::Landed), pause_physics)
        .add_systems(OnExit(PlayState::Landed), unpause_physics)
        .add_systems(OnEnter(PlayState::Flying), place_player_at_launch_site);
}

#[derive(Resource, Default)]
pub struct LandedContext {
    /// Name of the planet the player is docked at.
    pub planet_name: Option<String>,
    pub active_tab: PlanetTab,
}

#[derive(PartialEq, Default)]
pub enum PlanetTab {
    #[default]
    Trade,
    Shipyard,
    Outfitter,
    Bar,
    Missions,
}

pub fn planet_ui(
    mut egui_contexts: EguiContexts,
    mut state: ResMut<NextState<PlayState>>,
    mut landed: ResMut<LandedContext>,
    mut player_query: Query<(&mut Ship, &mut WeaponSystems), With<Player>>,
    mut buy_ship_writer: MessageWriter<BuyShip>,
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
            ui.selectable_value(&mut landed.active_tab, PlanetTab::Outfitter, "Outfitter");
            ui.selectable_value(&mut landed.active_tab, PlanetTab::Bar, "Bar");
            ui.selectable_value(&mut landed.active_tab, PlanetTab::Missions, "Missions");
        });
        ui.separator();
        ui.horizontal(|ui| {
            if ui.button("Repair").clicked() {
                if let Ok((mut ship, _)) = player_query.single_mut() {
                    ship.health = ship.data.max_health;
                }
            }
            if ui.button("Launch").clicked() {
                state.set(PlayState::Flying);
            }
        });
        match landed.active_tab {
            PlanetTab::Trade => {
                if let Ok((mut ship, _)) = player_query.single_mut() {
                    ui.label(format!("Credits: {}", ship.credits));
                    ui.separator();
                    egui::Grid::new("trade_grid")
                        .num_columns(6)
                        .striped(true)
                        .show(ui, |ui| {
                            ui.strong("Commodity");
                            ui.strong("Price");
                            ui.strong("Market");
                            ui.strong("Cargo");
                            ui.label("");
                            ui.label("");
                            ui.end_row();
                            let mut commodities: Vec<(String, i128)> = planet
                                .commodities
                                .iter()
                                .map(|(k, v)| (k.clone(), *v))
                                .collect();
                            commodities.sort_by(|a, b| a.0.cmp(&b.0));
                            for (commodity, price) in commodities {
                                let qty = *ship.cargo.get(&commodity).unwrap_or(&0);
                                ui.label(&commodity);
                                ui.label(price.to_string());
                                // Price indicator vs. global average
                                if let Some(&avg) =
                                    item_universe.global_average_price.get(&commodity)
                                {
                                    let ratio = price as f64 / avg;
                                    let (label, color) = if ratio < 0.6 {
                                        ("very cheap", egui::Color32::from_rgb(50, 220, 50))
                                    } else if ratio < 0.85 {
                                        ("cheap", egui::Color32::from_rgb(150, 230, 150))
                                    } else if ratio > 1.6 {
                                        ("very expensive", egui::Color32::from_rgb(230, 60, 60))
                                    } else if ratio > 1.15 {
                                        ("expensive", egui::Color32::from_rgb(230, 160, 100))
                                    } else {
                                        ("average", egui::Color32::GRAY)
                                    };
                                    ui.colored_label(color, label);
                                } else {
                                    ui.label("-");
                                }
                                ui.label(qty.to_string());
                                if ui.button("Buy").clicked() {
                                    ship.buy_cargo(&commodity, 1, price);
                                }
                                if ui.button("Sell").clicked() {
                                    ship.sell_cargo(&commodity, 1, price);
                                }
                                ui.end_row();
                            }
                        });
                }
            }
            PlanetTab::Outfitter => {
                if let Ok((mut ship, mut weapon_systems)) = player_query.single_mut() {
                    ui.label(format!("Credits: {}", ship.credits));
                    ui.label(format!(
                        "Free space: {}/{}",
                        ship.remaining_item_space(),
                        ship.data.item_space
                    ));
                    ui.separator();
                    egui::Grid::new("outfitter_grid")
                        .num_columns(6)
                        .striped(true)
                        .show(ui, |ui| {
                            ui.strong("Item");
                            ui.strong("Price");
                            ui.strong("Space");
                            ui.strong("Owned");
                            ui.label("");
                            ui.label("");
                            ui.end_row();
                            let items: Vec<(String, i128, u16)> = planet
                                .outfitter
                                .iter()
                                .filter_map(|k| {
                                    item_universe
                                        .outfitter_items
                                        .get(k)
                                        .map(|item| (k.clone(), item.price(), item.space()))
                                })
                                .collect();
                            for (item, price, space) in items {
                                let owned = weapon_systems
                                    .primary
                                    .get(&item)
                                    .map(|ws| ws.number)
                                    .unwrap_or(0);
                                ui.label(&item);
                                ui.label(price.to_string());
                                ui.label(space.to_string());
                                ui.label(owned.to_string());
                                if ui.button("Buy").clicked() {
                                    weapon_systems.buy_weapon(&item, &mut ship, &item_universe);
                                }
                                if ui.button("Sell").clicked() {
                                    weapon_systems.sell_weapon(&item, &mut ship, &item_universe);
                                }
                                ui.end_row();
                            }
                        });
                }
            }
            PlanetTab::Shipyard => {
                let player_credits = player_query
                    .single()
                    .map(|(s, _)| s.credits)
                    .unwrap_or(0);
                let player_ship_type = player_query
                    .single()
                    .map(|(s, _)| s.ship_type.clone())
                    .unwrap_or_default();
                ui.label(format!("Credits: {}", player_credits));
                ui.separator();
                if planet.shipyard.is_empty() {
                    ui.label("No ships for sale here.");
                } else {
                    egui::Grid::new("shipyard_grid")
                        .num_columns(7)
                        .striped(true)
                        .show(ui, |ui| {
                            ui.strong("Ship");
                            ui.strong("Price");
                            ui.strong("Speed");
                            ui.strong("Health");
                            ui.strong("Cargo");
                            ui.strong("Slots");
                            ui.label("");
                            ui.end_row();
                            let ships: Vec<(String, _)> = planet
                                .shipyard
                                .iter()
                                .filter_map(|k| {
                                    item_universe.ships.get(k).map(|d| (k.clone(), d.clone()))
                                })
                                .collect();
                            for (ship_type, data) in ships {
                                let is_current = ship_type == player_ship_type;
                                let can_afford = player_credits >= data.price;
                                if is_current {
                                    ui.strong(&ship_type);
                                } else {
                                    ui.label(&ship_type);
                                }
                                ui.label(format!("${}", data.price));
                                ui.label(format!("{}", data.max_speed as i32));
                                ui.label(format!("{}", data.max_health));
                                ui.label(format!("{}", data.cargo_space));
                                ui.label(format!("{}", data.item_space));
                                if is_current {
                                    ui.label("(current)");
                                } else {
                                    ui.add_enabled_ui(can_afford, |ui| {
                                        if ui.button("Buy").clicked() {
                                            buy_ship_writer.write(BuyShip {
                                                ship_type: ship_type.clone(),
                                            });
                                        }
                                    });
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

/// When re-entering Flying from a landing, place the ship at the planet's YAML position
/// and zero its velocity. Only runs when planet_name is set (i.e. coming from a landing,
/// not from a jump).
fn place_player_at_launch_site(
    mut landed: ResMut<LandedContext>,
    mut player_query: Query<(&mut Transform, &mut Position, &mut LinearVelocity), With<Player>>,
    mut camera_query: Query<&mut Transform, (With<Camera2d>, Without<Player>)>,
    current_system: Res<CurrentStarSystem>,
    item_universe: Res<ItemUniverse>,
) {
    let Some(planet_name) = landed.planet_name.take() else {
        return;
    };
    let Some(system) = item_universe.star_systems.get(&current_system.0) else {
        return;
    };
    let Some(planet_data) = system.planets.get(&planet_name) else {
        return;
    };
    let pos = planet_data.location;
    if let Ok((mut tf, mut physics_pos, mut vel)) = player_query.single_mut() {
        tf.translation = pos.extend(tf.translation.z);
        physics_pos.0 = pos;
        vel.0 = Vec2::ZERO;
        if let Ok(mut cam_tf) = camera_query.single_mut() {
            cam_tf.translation = pos.extend(cam_tf.translation.z);
        }
    }
}

pub fn pause_physics(mut time: ResMut<Time<Physics>>) {
    time.pause();
}

pub fn unpause_physics(mut time: ResMut<Time<Physics>>) {
    time.unpause();
}
