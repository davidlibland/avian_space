//! The building egui windows (market/outfitter/shipyard/bar/mechanic/garrison).
#[allow(unused_imports)]
use super::*;
use bevy_egui::EguiContexts;

use crate::item_universe::ItemUniverse;
use crate::missions::{AcceptMission, MissionCatalog, MissionLog, MissionOffers, PlayerUnlocks};
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

/// Everything the building windows read from the mission/faction layer —
/// bundled so the system signature stays legible (see [lints] policy note).
#[derive(bevy::ecs::system::SystemParam)]
pub(crate) struct BuildingUiContext<'w> {
    mission_log: Res<'w, MissionLog>,
    mission_offers: Res<'w, MissionOffers>,
    mission_catalog: Res<'w, MissionCatalog>,
    unlocks: Res<'w, PlayerUnlocks>,
    standings: Res<'w, crate::standing::FactionStandings>,
    galaxy: Res<'w, crate::galaxy::GalaxyControl>,
}

pub(crate) fn surface_building_ui(
    mut egui_contexts: EguiContexts,
    mut active_ui: ResMut<ActiveBuildingUI>,
    landed_context: Res<LandedContext>,
    current_system: Res<CurrentStarSystem>,
    item_universe: Res<ItemUniverse>,
    mut player_query: Query<&mut Ship, With<Player>>,
    mut buy_ship_writer: MessageWriter<BuyShip>,
    ctx: BuildingUiContext,
    mut accept_writer: MessageWriter<AcceptMission>,
    mut escort_roster: Option<ResMut<crate::carrier::EscortRoster>>,
    mut next_state: ResMut<NextState<PlayState>>,
) {
    let BuildingUiContext {
        mission_log,
        mission_offers,
        mission_catalog,
        unlocks,
        standings,
        galaxy,
    } = ctx;
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
                    // The compact job board (postings only — active
                    // missions live in the log, one I-press away)...
                    let free = player_query
                        .single()
                        .map(|s| s.remaining_cargo_space())
                        .unwrap_or(0);
                    crate::missions::ui::render_job_board(
                        ui,
                        &mission_offers,
                        &mission_catalog,
                        free,
                        &mut accept_writer,
                    );
                    ui.separator();
                    // ...the bartender's rumors about who in the room has
                    // work (those missions live on the NPCs at the tables)...
                    let fallen = escort_roster
                        .as_deref()
                        .map(|r| r.fallen.clone())
                        .unwrap_or_default();
                    crate::companions::render_bartender_rumors(
                        ui,
                        &mission_offers,
                        &mission_catalog,
                        &mission_log,
                        &item_universe,
                        planet_name,
                        &fallen,
                    );
                    // The wingman desk: your flight, dismissals, rejoins.
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
                    // ── Wing service: repair & rearm your companions and
                    // hired escorts, billed to YOUR account (carried
                    // fighters service automatically when they dock). ──
                    if let (Some(roster), Ok(mut ship)) =
                        (escort_roster.as_deref_mut(), player_query.single_mut())
                    {
                        let wing: Vec<usize> = roster
                            .entries
                            .iter()
                            .enumerate()
                            .filter(|(_, e)| {
                                matches!(
                                    e.kind,
                                    crate::carrier::EscortKind::Companion { .. }
                                        | crate::carrier::EscortKind::Hired { .. }
                                )
                            })
                            .map(|(i, _)| i)
                            .collect();
                        if !wing.is_empty() {
                            ui.heading("Wing service");
                            let mut total = 0i128;
                            for &i in &wing {
                                let e = &roster.entries[i];
                                let max = item_universe
                                    .ships
                                    .get(&e.ship_type)
                                    .map(|d| d.max_health)
                                    .unwrap_or(1)
                                    .max(1);
                                let cost = crate::carrier::service_cost(e, &item_universe);
                                total += cost;
                                ui.label(format!(
                                    "{} — hull {}/{}{}",
                                    match &e.kind {
                                        crate::carrier::EscortKind::Companion { name } =>
                                            item_universe
                                                .companions
                                                .get(name)
                                                .map(|d| d.name.clone())
                                                .unwrap_or_else(|| name.clone()),
                                        crate::carrier::EscortKind::Hired { name, .. } =>
                                            name.clone(),
                                        _ => e.ship_type.clone(),
                                    },
                                    e.health,
                                    max,
                                    if cost > 0 {
                                        format!(" ({cost} cr to service)")
                                    } else {
                                        " (ready)".to_string()
                                    }
                                ));
                            }
                            if total > 0
                                && ui.button(format!("Service wing ({total} cr)")).clicked()
                            {
                                let mut credits = ship.credits;
                                for &i in &wing {
                                    crate::carrier::service_entry(
                                        &mut roster.entries[i],
                                        &item_universe,
                                        &mut credits,
                                    );
                                }
                                ship.credits = credits;
                            }
                            ui.separator();
                        }
                    }
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
                BuildingKind::Mine | BuildingKind::Warehouse | BuildingKind::Substation => {
                    // Maze venues have no counter window — everything
                    // happens inside. Unreachable in practice.
                    ui.label("Nothing to do here.");
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
                            let per_unit = ship.fuel_price_per_unit();
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
