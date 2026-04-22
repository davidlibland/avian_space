use crate::{
    CurrentStarSystem, PlayState, Player, Ship, item_universe::ItemUniverse, ship::BuyShip,
};
use crate::session::SessionResourceExt;
use crate::ship_anim::{ANIM_MIN_SCALE, PLANET_ANIM_DURATION, ScalingUp, image_size};
use avian2d::prelude::{LinearVelocity, Position};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass};

pub fn planet_ui_plugin(app: &mut App) {
    app.add_plugins(EguiPlugin::default())
        .init_session_resource::<LandedContext>()
        .add_systems(
            EguiPrimaryContextPass,
            ship_pad_ui.run_if(in_state(PlayState::Landed)),
        )
        .add_systems(OnEnter(PlayState::Landed), pause_physics)
        .add_systems(OnExit(PlayState::Landed), unpause_physics)
        .add_systems(OnEnter(PlayState::Flying), place_player_at_launch_site);
}

#[derive(Resource, Default)]
pub struct LandedContext {
    /// Name of the planet the player is docked at.
    pub planet_name: Option<String>,
}

impl crate::session::SessionResource for LandedContext {
    type SaveData = ();
    fn new_session(_: &crate::item_universe::ItemUniverse) -> Self { Self::default() }
    fn from_save(_: (), _: &crate::item_universe::ItemUniverse) -> Self { Self::default() }
}

/// Ship-pad UI shown during `Landed` state — just Repair + Launch.
///
/// The full Trade/Outfitter/Shipyard/Bar/Missions UIs are now handled by
/// [`crate::surface::surface_building_ui`] during the `Exploring` state.
fn ship_pad_ui(
    mut egui_contexts: EguiContexts,
    mut state: ResMut<NextState<PlayState>>,
    landed: Res<LandedContext>,
    mut player_query: Query<&mut Ship, With<Player>>,
    item_universe: Res<ItemUniverse>,
    current_system: Res<CurrentStarSystem>,
) {
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };
    let display_name = landed
        .planet_name
        .as_ref()
        .and_then(|name| {
            item_universe
                .star_systems
                .get(&current_system.0)
                .and_then(|sys| sys.planets.get(name))
                .map(|pd| pd.display_name.as_str())
        })
        .unwrap_or("Unknown");

    egui::Window::new(format!("Ship Pad - {}", display_name))
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            if let Ok(ship) = player_query.single() {
                ui.label(format!(
                    "Hull: {}/{}",
                    ship.health, ship.data.max_health
                ));
                ui.label(format!("Credits: {}", ship.credits));
            }
            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("Repair").clicked() {
                    if let Ok(mut ship) = player_query.single_mut() {
                        ship.health = ship.data.max_health;
                    }
                }
                if ui.button("Launch").clicked() {
                    state.set(PlayState::Flying);
                }
                if ui.button("Back to Surface").clicked() {
                    state.set(PlayState::Exploring);
                }
            });
        });
}

/// When re-entering Flying from a landing, place the ship at the planet's YAML position
/// and zero its velocity. Only runs when planet_name is set (i.e. coming from a landing,
/// not from a jump).
fn place_player_at_launch_site(
    mut commands: Commands,
    mut landed: ResMut<LandedContext>,
    mut player_query: Query<
        (Entity, &mut Transform, &mut Position, &mut LinearVelocity, &mut Sprite),
        With<Player>,
    >,
    mut camera_query: Query<&mut Transform, (With<Camera2d>, Without<Player>)>,
    current_system: Res<CurrentStarSystem>,
    item_universe: Res<ItemUniverse>,
    images: Res<Assets<Image>>,
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
    if let Ok((entity, mut tf, mut physics_pos, mut vel, mut sprite)) =
        player_query.single_mut()
    {
        tf.translation = pos.extend(tf.translation.z);
        physics_pos.0 = pos;
        vel.0 = Vec2::ZERO;
        if let Ok(mut cam_tf) = camera_query.single_mut() {
            cam_tf.translation = pos.extend(cam_tf.translation.z);
        }

        // Start take-off scale-up animation.
        let full_size = image_size(&sprite, &images);
        sprite.custom_size = Some(full_size * ANIM_MIN_SCALE);
        commands.entity(entity).insert(ScalingUp {
            timer: Timer::from_seconds(PLANET_ANIM_DURATION, TimerMode::Once),
            full_size,
        });
    }
}

pub fn pause_physics(mut time: ResMut<Time<Virtual>>) {
    time.pause();
}

pub fn unpause_physics(mut time: ResMut<Time<Virtual>>) {
    time.unpause();
}

// ---------------------------------------------------------------------------
// Extracted tab renderers (reused by surface::surface_building_ui)
// ---------------------------------------------------------------------------

use crate::planets::PlanetData;
use bevy_egui::egui;

/// Render the Trade tab content into an egui Ui.
pub fn render_trade_tab(
    ui: &mut egui::Ui,
    ship: &mut Ship,
    planet: &PlanetData,
    item_universe: &ItemUniverse,
) {
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
                let commodity_display = item_universe
                    .commodities
                    .get(&commodity)
                    .map(|c| c.display_name.as_str())
                    .unwrap_or(&commodity);
                ui.label(commodity_display);
                ui.label(price.to_string());
                if let Some(&avg) = item_universe.global_average_price.get(&commodity) {
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

    // Show cargo items that this planet doesn't trade, sell-only at minimum price.
    let mut extra_cargo: Vec<(String, u16)> = ship
        .cargo
        .iter()
        .filter(|(k, v)| **v > 0 && !planet.commodities.contains_key(*k))
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    extra_cargo.sort_by(|a, b| a.0.cmp(&b.0));
    if !extra_cargo.is_empty() {
        ui.separator();
        ui.label("Other cargo (sell only):");
        egui::Grid::new("trade_extra_grid")
            .num_columns(4)
            .striped(true)
            .show(ui, |ui| {
                ui.strong("Commodity");
                ui.strong("Sell Price");
                ui.strong("Cargo");
                ui.label("");
                ui.end_row();
                for (commodity, qty) in &extra_cargo {
                    let sell_price = item_universe
                        .global_minimum_price
                        .get(commodity)
                        .copied()
                        .unwrap_or(1);
                    let commodity_display = item_universe
                        .commodities
                        .get(commodity)
                        .map(|c| c.display_name.as_str())
                        .unwrap_or(commodity);
                    ui.label(commodity_display);
                    ui.label(sell_price.to_string());
                    ui.label(qty.to_string());
                    if ui.button("Sell").clicked() {
                        ship.sell_cargo(commodity, 1, sell_price);
                    }
                    ui.end_row();
                }
            });
    }
}

/// Render the Outfitter tab content into an egui Ui.
pub fn render_outfitter_tab(
    ui: &mut egui::Ui,
    ship: &mut Ship,
    planet: &PlanetData,
    item_universe: &ItemUniverse,
    unlocks: &crate::missions::PlayerUnlocks,
) {
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
            ui.strong("Ammo");
            ui.label("");
            ui.label("");
            ui.end_row();
            let items: Vec<(String, i128, u16)> = planet
                .outfitter
                .iter()
                .filter_map(|k| {
                    item_universe.outfitter_items.get(k).and_then(|item| {
                        let locked = item.required_unlocks().iter().any(|u| !unlocks.has(u));
                        if locked {
                            None
                        } else {
                            Some((k.clone(), item.price(), item.space()))
                        }
                    })
                })
                .collect();
            for (item, price, space) in items {
                let (owned, ammo) = ship
                    .weapon_systems
                    .find_weapon(&item)
                    .map(|ws| (ws.number, ws.ammo_quantity))
                    .unwrap_or((0, None));
                let item_display = item_universe
                    .outfitter_items
                    .get(&item)
                    .map(|i| i.display_name())
                    .unwrap_or(&item);
                ui.label(item_display);
                ui.label(price.to_string());
                ui.label(space.to_string());
                ui.label(owned.to_string());
                if ui.button("Buy").clicked() {
                    ship.buy_weapon(&item, &item_universe);
                }
                if ui.button("Sell").clicked() {
                    ship.sell_weapon(&item, &item_universe);
                }
                ui.label(match ammo {
                    Some(qty) => qty.to_string(),
                    _ => "n/a".to_string(),
                });
                if ui.button("Buy").clicked() {
                    ship.buy_ammo(&item, &item_universe);
                }
                if ui.button("Sell").clicked() {
                    ship.sell_ammo(&item, &item_universe);
                }
                ui.end_row();
            }
        });
}

/// Render the Shipyard tab content into an egui Ui.
pub fn render_shipyard_tab(
    ui: &mut egui::Ui,
    ship: &Ship,
    planet: &PlanetData,
    item_universe: &ItemUniverse,
    unlocks: &crate::missions::PlayerUnlocks,
    buy_ship_writer: &mut MessageWriter<BuyShip>,
) {
    ui.label(format!("Credits: {}", ship.credits));
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
                        item_universe.ships.get(k).and_then(|d| {
                            let locked = d.required_unlocks.iter().any(|u| !unlocks.has(u));
                            if locked {
                                None
                            } else {
                                Some((k.clone(), d.clone()))
                            }
                        })
                    })
                    .collect();
                let trade_in = ship.trade_in_value(item_universe);
                for (ship_type, data) in ships {
                    let is_current = ship_type == ship.ship_type;
                    let net_cost = data.price - trade_in;
                    let can_afford = ship.credits >= net_cost;
                    if is_current {
                        ui.strong(&data.display_name);
                    } else {
                        ui.label(&data.display_name);
                    }
                    if is_current {
                        ui.label(format!("${}", data.price));
                    } else {
                        ui.label(format!("${} (net ${})", data.price, net_cost));
                    }
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
