//! The building egui windows (market/outfitter/shipyard/bar/mechanic/garrison).
#[allow(unused_imports)]
use super::*;
use bevy_egui::EguiContexts;

use crate::item_universe::ItemUniverse;
use crate::missions::{
    AbandonMission, AcceptMission, MissionCatalog, MissionLog, MissionOffers, PlayerUnlocks,
    render_missions_tab,
};
use crate::planet_ui::{
    LandedContext, render_mods_section, render_outfitter_tab, render_shipyard_tab, render_trade_tab,
};
use crate::ship::{BuyShip, Ship};
use crate::{CurrentStarSystem, PlayState, Player};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Render the appropriate egui window based on which building the player is
/// interacting with. Reuses the extracted tab renderers from planet_ui.rs.
#[allow(clippy::too_many_arguments)]
/// Price multiplier for a planet from the player's standing with its
/// EFFECTIVE faction (the live controller of its system).
pub(crate) fn planet_markup(
    standings: &crate::standing::FactionStandings,
    galaxy: &crate::galaxy::GalaxyControl,
    iu: &crate::item_universe::ItemUniverse,
    planet_name: &str,
) -> f32 {
    crate::galaxy::effective_planet_faction(galaxy, iu, planet_name)
        .map(|f| crate::standing::price_markup(standings.get(&f)))
        .unwrap_or(1.0)
}

pub(crate) fn surface_building_ui(
    mut egui_contexts: EguiContexts,
    mut active_ui: ResMut<ActiveBuildingUI>,
    landed_context: Res<LandedContext>,
    current_system: Res<CurrentStarSystem>,
    item_universe: Res<ItemUniverse>,
    mut player_query: Query<&mut Ship, With<Player>>,
    mut buy_ship_writer: MessageWriter<BuyShip>,
    missions: (
        Res<MissionLog>,
        Res<MissionOffers>,
        Res<MissionCatalog>,
        Res<PlayerUnlocks>,
    ),
    standings: Res<crate::standing::FactionStandings>,
    galaxy: Res<crate::galaxy::GalaxyControl>,
    mut accept_writer: MessageWriter<AcceptMission>,
    mut abandon_writer: MessageWriter<AbandonMission>,
    mut escort_roster: Option<ResMut<crate::carrier::EscortRoster>>,
    mut next_state: ResMut<NextState<PlayState>>,
) {
    let (mission_log, mission_offers, mission_catalog, unlocks) = missions;
    let Some(kind) = active_ui.0 else {
        return;
    };
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };
    let planet_name = landed_context.planet_name.as_deref().unwrap_or("");
    let planet_data = item_universe
        .star_systems
        .get(&current_system.0)
        .and_then(|sys| sys.planets.get(planet_name));

    let title = kind.label();

    bevy_egui::egui::Window::new(title)
        .collapsible(false)
        .resizable(true)
        .anchor(bevy_egui::egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            match kind {
                BuildingKind::Market => {
                    if let (Ok(mut ship), Some(pd)) = (player_query.single_mut(), planet_data) {
                        let markup =
                            planet_markup(&standings, &galaxy, &item_universe, planet_name);
                        render_trade_tab(ui, &mut ship, pd, &item_universe, markup);
                    } else {
                        ui.label("No commodities available.");
                    }
                }
                BuildingKind::Outfitter => {
                    if let (Ok(mut ship), Some(pd)) = (player_query.single_mut(), planet_data) {
                        let markup =
                            planet_markup(&standings, &galaxy, &item_universe, planet_name);
                        render_outfitter_tab(ui, &mut ship, pd, &item_universe, &unlocks, markup);
                    } else {
                        ui.label("No equipment available.");
                    }
                }
                BuildingKind::Shipyard => {
                    if let (Ok(ship), Some(pd)) = (player_query.single(), planet_data) {
                        let markup =
                            planet_markup(&standings, &galaxy, &item_universe, planet_name);
                        render_shipyard_tab(
                            ui,
                            ship,
                            pd,
                            &item_universe,
                            &unlocks,
                            &mut buy_ship_writer,
                            markup,
                        );
                    } else {
                        ui.label("No ships for sale.");
                    }
                }
                BuildingKind::Bar => {
                    let free = player_query
                        .single()
                        .map(|s| s.remaining_cargo_space())
                        .unwrap_or(0);
                    render_missions_tab(
                        ui,
                        &mission_log,
                        &mission_offers,
                        &mission_catalog,
                        free,
                        &mut accept_writer,
                        &mut abandon_writer,
                    );
                    // The wingman desk: companions, rejoins, pilots for hire.
                    if let (Ok(mut ship), Some(roster)) =
                        (player_query.single_mut(), escort_roster.as_deref_mut())
                    {
                        crate::companions::render_companions_section(
                            ui,
                            &mut ship,
                            roster,
                            &item_universe,
                            planet_name,
                        );
                    }
                }
                BuildingKind::ShipPad => {
                    ui.label("Your ship is docked here.");
                    ui.add_space(8.0);
                    if ui.button("Launch").clicked() {
                        active_ui.0 = None;
                        next_state.set(PlayState::Flying);
                    }
                }
                BuildingKind::MechanicShop => {
                    if let Ok(mut ship) = player_query.single_mut() {
                        let max_hp = ship.max_health();
                        let hp = ship.health;
                        ui.label(format!("Hull: {}/{}", hp, max_hp));

                        if hp < max_hp {
                            // Repair cost: (1 - health_frac) * 5% of ship price
                            let damage_frac = 1.0 - (hp as f64 / max_hp as f64);
                            let cost = (damage_frac * 0.05 * ship.data.price as f64).ceil() as i128;
                            let cost = cost.max(1);

                            ui.add_space(4.0);
                            ui.label(format!(
                                "Repair cost: {} credits (5% of ship value per full repair)",
                                cost
                            ));

                            let can_afford = ship.credits >= cost;
                            if ui
                                .add_enabled(can_afford, bevy_egui::egui::Button::new("Repair"))
                                .clicked()
                            {
                                ship.credits -= cost;
                                ship.health = max_hp;
                            }
                            if !can_afford {
                                ui.colored_label(
                                    bevy_egui::egui::Color32::RED,
                                    "Not enough credits.",
                                );
                            }
                        } else {
                            ui.add_space(4.0);
                            ui.label("Hull is at full integrity. No repairs needed.");
                        }

                        // Hull modification bench — the mechanic's trade.
                        if let Some(pd) = planet_data {
                            let markup =
                                planet_markup(&standings, &galaxy, &item_universe, planet_name);
                            render_mods_section(
                                ui,
                                &mut ship,
                                pd,
                                &item_universe,
                                &unlocks,
                                markup,
                            );
                        }

                        ui.add_space(4.0);
                        ui.label(format!("Credits: {}", ship.credits));
                    }
                }
                BuildingKind::Garrison => {
                    let faction = crate::galaxy::effective_planet_faction(
                        &galaxy,
                        &item_universe,
                        planet_name,
                    );
                    match faction {
                        Some(f) => {
                            let standing = standings.get(&f);
                            ui.label(format!("{} war office.", f));
                            ui.label(format!("Your standing: {:+.0}", standing));
                            ui.add_space(6.0);
                            if standing >= 10.0 {
                                ui.label(
                                    "The duty officer posts commissions outside when a \
                                     front needs freelance pilots. Fight for the flag and \
                                     the flag remembers.",
                                );
                            } else {
                                ui.colored_label(
                                    bevy_egui::egui::Color32::from_rgb(230, 160, 100),
                                    "The duty officer looks through you. War work goes to \
                                     TRUSTED pilots — run their missions, hunt their \
                                     enemies, come back at +10.",
                                );
                            }
                        }
                        None => {
                            ui.label("The office stands empty — nobody holds this world.");
                        }
                    }
                }
                BuildingKind::FuelStation => {
                    if let Ok(mut ship) = player_query.single_mut() {
                        let fuel = ship.fuel;
                        let cap = ship.data.fuel_capacity;
                        ui.label(format!("Fuel: {}/{} jumps", fuel, cap));
                        if fuel < cap {
                            let per_unit = (ship.data.price / 100).max(20);
                            let cost = (per_unit * (cap - fuel) as i128).max(1);
                            ui.add_space(4.0);
                            ui.label(format!("Refuel cost: {} credits", cost));
                            let can_afford = ship.credits >= cost;
                            if ui
                                .add_enabled(can_afford, bevy_egui::egui::Button::new("Refuel"))
                                .clicked()
                            {
                                ship.credits -= cost;
                                ship.fuel = cap;
                            }
                            if !can_afford {
                                ui.colored_label(
                                    bevy_egui::egui::Color32::RED,
                                    "Not enough credits.",
                                );
                            }
                        } else {
                            ui.add_space(4.0);
                            ui.label("Tank is full.");
                        }
                        ui.add_space(4.0);
                        ui.label(format!("Credits: {}", ship.credits));
                    }
                }
            }
            ui.separator();
            if ui.button("Close [Esc]").clicked() {
                active_ui.0 = None;
            }
        });
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------
