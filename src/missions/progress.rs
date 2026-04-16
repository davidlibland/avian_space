use bevy::prelude::*;
use rand::Rng;

use super::events::*;
use super::log::{MissionCatalog, MissionLog, MissionOffers, PlayerUnlocks};
use super::types::*;
use crate::item_universe::ItemUniverse;
use crate::ship::Ship;
use crate::Player;

/// Startup: copy the static missions from ItemUniverse into the catalog.
pub fn init_catalog(universe: Res<ItemUniverse>, mut catalog: ResMut<MissionCatalog>) {
    catalog.defs = universe.missions.clone();
}

/// Run on startup (and on any mission status change) to:
///   • flip Locked → Available when preconditions are met
///   • auto-start Auto-offered missions that are now Available
pub fn update_locked_to_available(
    mut log: ResMut<MissionLog>,
    catalog: Res<MissionCatalog>,
    unlocks: Res<PlayerUnlocks>,
    mut started: MessageWriter<MissionStarted>,
) {
    loop {
        let mut changed = false;
        let ids: Vec<String> = catalog.defs.keys().cloned().collect();
        for id in ids {
            let Some(def) = catalog.defs.get(&id) else {
                continue;
            };
            let status = log.status(&id);
            if matches!(
                status,
                MissionStatus::Available
                    | MissionStatus::Active(_)
                    | MissionStatus::Completed
                    | MissionStatus::Failed
            ) {
                continue;
            }
            if !preconditions_met(&def.preconditions, &log, &unlocks) {
                continue;
            }
            match def.offer {
                OfferKind::Auto => {
                    log.set(
                        &id,
                        MissionStatus::Active(initial_progress(&def.objective)),
                    );
                    started.write(MissionStarted(id.clone()));
                    changed = true;
                }
                _ => {
                    log.set(&id, MissionStatus::Available);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
}

pub(crate) fn preconditions_met(
    pres: &[Precondition],
    log: &MissionLog,
    unlocks: &PlayerUnlocks,
) -> bool {
    pres.iter().all(|p| match p {
        Precondition::Completed { mission } => {
            matches!(log.status(mission), MissionStatus::Completed)
        }
        Precondition::Failed { mission } => {
            matches!(log.status(mission), MissionStatus::Failed)
        }
        Precondition::HasUnlock { name } => unlocks.has(name),
    })
}

fn initial_progress(_obj: &Objective) -> ObjectiveProgress {
    ObjectiveProgress::default()
}

/// React to MissionStarted by applying the mission's start_effects.
pub fn apply_start_effects(
    mut reader: MessageReader<MissionStarted>,
    catalog: Res<MissionCatalog>,
    mut player_q: Query<&mut Ship, With<Player>>,
) {
    for MissionStarted(id) in reader.read() {
        let Some(def) = catalog.defs.get(id) else {
            continue;
        };
        let Ok(mut ship) = player_q.single_mut() else {
            continue;
        };
        for effect in &def.start_effects {
            match effect {
                StartEffect::LoadCargo {
                    commodity,
                    quantity,
                    reserved,
                } => {
                    *ship.cargo.entry(commodity.clone()).or_insert(0) += quantity;
                    if *reserved {
                        *ship
                            .reserved_cargo
                            .entry(commodity.clone())
                            .or_insert(0) += quantity;
                    }
                }
            }
        }
    }
}

/// Handles Accept/Decline/Abandon button presses from the UI.
/// Refuses Accept if the player doesn't have enough cargo space.
pub fn handle_ui_actions(
    mut accept: MessageReader<AcceptMission>,
    mut decline: MessageReader<DeclineMission>,
    mut abandon: MessageReader<AbandonMission>,
    mut log: ResMut<MissionLog>,
    catalog: Res<MissionCatalog>,
    mut offers: ResMut<MissionOffers>,
    mut started: MessageWriter<MissionStarted>,
    mut failed: MessageWriter<MissionFailed>,
    player_q: Query<&Ship, With<Player>>,
) {
    for AcceptMission(id) in accept.read() {
        let Some(def) = catalog.defs.get(id) else {
            continue;
        };
        if !matches!(log.status(id), MissionStatus::Available) {
            continue;
        }
        // Cargo-space gate — refuse the accept entirely if the hold can't
        // hold the LoadCargo start effects. The UI should already disable
        // the button, but we defend in depth so programmatic paths (auto
        // acceptance, future tests) can't overload the hold.
        if let Ok(ship) = player_q.single() {
            if def.required_cargo_space() > ship.remaining_cargo_space() {
                continue;
            }
        }
        log.set(id, MissionStatus::Active(initial_progress(&def.objective)));
        started.write(MissionStarted(id.clone()));
        remove_from_offers(&mut offers, id);
    }
    for DeclineMission(id) in decline.read() {
        remove_from_offers(&mut offers, id);
    }
    for AbandonMission(id) in abandon.read() {
        if matches!(log.status(id), MissionStatus::Active(_)) {
            log.set(id, MissionStatus::Failed);
            failed.write(MissionFailed(id.clone()));
        }
    }
}

fn remove_from_offers(offers: &mut MissionOffers, id: &str) {
    offers.tab.retain(|m| m != id);
    for v in offers.bar.values_mut() {
        v.retain(|m| m != id);
    }
}

// ── Objective progress ─────────────────────────────────────────────────────

/// Check the mission's `requires` clauses against the player's current
/// state. Returns true if all requirements are satisfied.
fn requirements_met(def: &MissionDef, ship: &Ship, unlocks: &PlayerUnlocks) -> bool {
    def.requires.iter().all(|req| match req {
        CompletionRequirement::HasCargo {
            commodity,
            quantity,
        } => ship.cargo.get(commodity).copied().unwrap_or(0) >= *quantity,
        CompletionRequirement::HasUnlock { name } => unlocks.has(name),
    })
}

/// Transition a mission from Active to either Completed or Failed depending
/// on whether its `requires` clauses pass. Returns true if the mission
/// moved to Completed (so callers can skip any further progress updates).
fn resolve_active_mission(
    id: &str,
    def: &MissionDef,
    ship: Option<&Ship>,
    unlocks: &PlayerUnlocks,
    log: &mut MissionLog,
    completed: &mut MessageWriter<MissionCompleted>,
    failed: &mut MessageWriter<MissionFailed>,
) -> bool {
    let ok = ship
        .map(|s| requirements_met(def, s, unlocks))
        .unwrap_or(true);
    if ok {
        log.set(id, MissionStatus::Completed);
        completed.write(MissionCompleted(id.to_string()));
        true
    } else {
        log.set(id, MissionStatus::Failed);
        failed.write(MissionFailed(id.to_string()));
        false
    }
}

pub fn advance_travel_objectives(
    mut reader: MessageReader<PlayerEnteredSystem>,
    mut log: ResMut<MissionLog>,
    catalog: Res<MissionCatalog>,
    unlocks: Res<PlayerUnlocks>,
    player_q: Query<&Ship, With<Player>>,
    mut completed: MessageWriter<MissionCompleted>,
    mut failed: MessageWriter<MissionFailed>,
) {
    let ship = player_q.single().ok();
    for PlayerEnteredSystem { system } in reader.read() {
        for (id, def) in &catalog.defs {
            if !matches!(log.status(id), MissionStatus::Active(_)) {
                continue;
            }
            if let Objective::TravelToSystem { system: target } = &def.objective {
                if target == system {
                    resolve_active_mission(id, def, ship, &unlocks, &mut log, &mut completed, &mut failed);
                }
            }
        }
    }
}

pub fn advance_land_objectives(
    mut reader: MessageReader<PlayerLandedOnPlanet>,
    mut log: ResMut<MissionLog>,
    catalog: Res<MissionCatalog>,
    unlocks: Res<PlayerUnlocks>,
    player_q: Query<&Ship, With<Player>>,
    mut completed: MessageWriter<MissionCompleted>,
    mut failed: MessageWriter<MissionFailed>,
) {
    let ship = player_q.single().ok();
    for PlayerLandedOnPlanet { planet } in reader.read() {
        for (id, def) in &catalog.defs {
            if !matches!(log.status(id), MissionStatus::Active(_)) {
                continue;
            }
            if let Objective::LandOnPlanet { planet: target } = &def.objective {
                if target == planet {
                    resolve_active_mission(id, def, ship, &unlocks, &mut log, &mut completed, &mut failed);
                }
            }
        }
    }
}

pub fn advance_collect_objectives(
    mut reader: MessageReader<PickupCollected>,
    mut log: ResMut<MissionLog>,
    catalog: Res<MissionCatalog>,
    unlocks: Res<PlayerUnlocks>,
    player_q: Query<&Ship, With<Player>>,
    mut completed: MessageWriter<MissionCompleted>,
    mut failed: MessageWriter<MissionFailed>,
) {
    let ship = player_q.single().ok();
    for event in reader.read() {
        let ids: Vec<String> = catalog.defs.keys().cloned().collect();
        for id in ids {
            let Some(def) = catalog.defs.get(&id) else {
                continue;
            };
            let Objective::CollectPickups {
                commodity,
                system,
                quantity,
            } = &def.objective
            else {
                continue;
            };
            if commodity != &event.commodity || system != &event.system {
                continue;
            }
            let MissionStatus::Active(progress) = log.status(&id) else {
                continue;
            };
            let new_have = progress.collected.saturating_add(event.quantity);
            if new_have >= *quantity {
                resolve_active_mission(&id, def, ship, &unlocks, &mut log, &mut completed, &mut failed);
            } else {
                log.set(
                    &id,
                    MissionStatus::Active(ObjectiveProgress {
                        collected: new_have,
                    }),
                );
            }
        }
    }
}

/// When MissionFailed fires (abandon or failed requirement), strip any
/// cargo the mission loaded at start (reserved or not), since the
/// player never actually paid for it — otherwise cancelling a delivery
/// mission would be a free cargo exploit.
pub fn finalize_failures(
    mut reader: MessageReader<MissionFailed>,
    catalog: Res<MissionCatalog>,
    mut player_q: Query<&mut Ship, With<Player>>,
) {
    for MissionFailed(id) in reader.read() {
        let Some(def) = catalog.defs.get(id) else {
            continue;
        };
        let Ok(mut ship) = player_q.single_mut() else {
            continue;
        };
        for effect in &def.start_effects {
            match effect {
                StartEffect::LoadCargo {
                    commodity, quantity, ..
                } => {
                    let held = ship.cargo.get(commodity).copied().unwrap_or(0);
                    let remove = held.min(*quantity);
                    let new_held = held - remove;
                    if new_held == 0 {
                        ship.cargo.remove(commodity);
                        ship.cargo_cost.remove(commodity);
                    } else {
                        ship.cargo.insert(commodity.clone(), new_held);
                    }
                    // Clear / cap the reserved counter.
                    if let Some(r) = ship.reserved_cargo.get_mut(commodity) {
                        *r = (*r).saturating_sub(*quantity).min(new_held);
                        if *r == 0 {
                            ship.reserved_cargo.remove(commodity);
                        }
                    }
                }
            }
        }
    }
}

/// When MissionCompleted fires, apply completion_effects and clear any
/// reserved cargo belonging to this mission's start_effects.
pub fn finalize_completions(
    mut reader: MessageReader<MissionCompleted>,
    catalog: Res<MissionCatalog>,
    mut unlocks: ResMut<PlayerUnlocks>,
    mut player_q: Query<&mut Ship, With<Player>>,
) {
    for MissionCompleted(id) in reader.read() {
        let Some(def) = catalog.defs.get(id) else {
            continue;
        };
        let Ok(mut ship) = player_q.single_mut() else {
            continue;
        };
        for effect in &def.start_effects {
            if let StartEffect::LoadCargo {
                commodity,
                quantity,
                reserved: true,
            } = effect
            {
                let entry = ship.reserved_cargo.entry(commodity.clone()).or_insert(0);
                *entry = entry.saturating_sub(*quantity);
                if *entry == 0 {
                    ship.reserved_cargo.remove(commodity);
                }
            }
        }
        for effect in &def.completion_effects {
            match effect {
                CompletionEffect::RemoveCargo {
                    commodity,
                    quantity,
                } => {
                    let held = ship.cargo.get(commodity).copied().unwrap_or(0);
                    let remove = held.min(*quantity);
                    if remove > 0 {
                        let new_held = held - remove;
                        if new_held == 0 {
                            ship.cargo.remove(commodity);
                            ship.cargo_cost.remove(commodity);
                        } else {
                            ship.cargo.insert(commodity.clone(), new_held);
                        }
                        // Cap reserved to what remains.
                        if let Some(r) = ship.reserved_cargo.get_mut(commodity) {
                            *r = (*r).min(new_held);
                            if *r == 0 {
                                ship.reserved_cargo.remove(commodity);
                            }
                        }
                    }
                }
                CompletionEffect::Pay { credits } => {
                    ship.credits = ship.credits.saturating_add(*credits as i128);
                }
                CompletionEffect::GrantUnlock { name } => {
                    unlocks.0.insert(name.clone());
                }
            }
        }
    }
}

// ── Offer rolling ──────────────────────────────────────────────────────────

pub fn roll_offers_on_land(
    mut reader: MessageReader<PlayerLandedOnPlanet>,
    log: Res<MissionLog>,
    unlocks: Res<PlayerUnlocks>,
    universe: Res<ItemUniverse>,
    mut catalog: ResMut<MissionCatalog>,
    mut offers: ResMut<MissionOffers>,
) {
    let mut rng = rand::thread_rng();
    for PlayerLandedOnPlanet { planet } in reader.read() {
        // Static missions currently Available.
        let mut tab: Vec<(String, f32)> = Vec::new();
        let mut bar: Vec<(String, f32)> = Vec::new();
        for (id, def) in &catalog.defs {
            if !matches!(log.status(id), MissionStatus::Available) {
                continue;
            }
            match &def.offer {
                OfferKind::Tab { weight } => tab.push((id.clone(), *weight)),
                OfferKind::Bar {
                    planet: p,
                    weight,
                } if p == planet => bar.push((id.clone(), *weight)),
                _ => {}
            }
        }
        let mut rolled_tab = roll(&tab, &mut rng);
        let mut rolled_bar = roll(&bar, &mut rng);

        // Procedurally generate mission instances from templates.
        for (template_id, template) in &universe.mission_templates {
            if !preconditions_met(template.preconditions(), &log, &unlocks) {
                continue;
            }
            let chain = instantiate_template(
                template_id,
                template,
                planet,
                &universe,
                &mut rng,
            );
            if chain.is_empty() {
                continue;
            }
            // The first element is the "entry point" offer; additional
            // elements are followup stages that auto-start via their
            // preconditions referencing earlier stages' generated ids.
            let (entry_id, entry_def) = &chain[0];
            let placement = match &entry_def.offer {
                OfferKind::Tab { weight } => {
                    if rng.gen_range(0.0..1.0) < weight.clamp(0.0, 1.0) {
                        Some(&mut rolled_tab)
                    } else {
                        None
                    }
                }
                OfferKind::Bar { planet: p, weight } if p == planet => {
                    if rng.gen_range(0.0..1.0) < weight.clamp(0.0, 1.0) {
                        Some(&mut rolled_bar)
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some(dest) = placement {
                dest.push(entry_id.clone());
                for (id, def) in chain {
                    catalog.defs.insert(id, def);
                }
            }
        }

        offers.tab = rolled_tab;
        offers.bar.insert(planet.clone(), rolled_bar);
    }
}

fn roll(candidates: &[(String, f32)], rng: &mut impl Rng) -> Vec<String> {
    candidates
        .iter()
        .filter(|(_, w)| rng.gen_range(0.0..1.0) < w.clamp(0.0, 1.0))
        .map(|(id, _)| id.clone())
        .collect()
}

/// Produce one or more concrete `(id, MissionDef)`s from a template. The
/// first entry is the "entry point" that gets offered to the player; any
/// trailing entries are followup stages that auto-start when their
/// preconditions (referencing earlier stages' ids) are met.
/// Returns an empty vec if the template can't be satisfied (e.g. no
/// eligible destination / system in the universe).
fn instantiate_template(
    template_id: &str,
    template: &MissionTemplate,
    offer_planet: &str,
    universe: &ItemUniverse,
    rng: &mut impl Rng,
) -> Vec<(String, MissionDef)> {
    match template {
        MissionTemplate::Delivery {
            briefing,
            success_text,
            failure_text,
            offer,
            preconditions: _,
            commodity_pool,
            quantity_range,
            pay_range,
            reserved,
        } => {
            if commodity_pool.is_empty() {
                return Vec::new();
            }
            let commodity = commodity_pool[rng.gen_range(0..commodity_pool.len())].clone();
            let quantity = rand_in_range_u16(rng, *quantity_range);
            let pay = rand_in_range_i128(rng, *pay_range);

            // Pick a destination planet from the whole universe, != offer_planet.
            let destinations: Vec<(String, String)> = universe
                .star_systems
                .iter()
                .flat_map(|(_sys_id, sys)| {
                    sys.planets
                        .iter()
                        .map(|(pid, p)| (pid.clone(), p.display_name.clone()))
                })
                .filter(|(pid, _)| pid != offer_planet)
                .collect();
            if destinations.is_empty() {
                return Vec::new();
            }
            let (dest_id, dest_display) =
                destinations[rng.gen_range(0..destinations.len())].clone();

            let vars = [
                ("{commodity}", commodity.as_str().to_string()),
                ("{quantity}", quantity.to_string()),
                ("{pay}", pay.to_string()),
                ("{planet}", dest_id.clone()),
                ("{planet_display}", dest_display),
            ];
            let def = MissionDef {
                briefing: subst(briefing, &vars),
                success_text: subst(success_text, &vars),
                failure_text: subst(failure_text, &vars),
                preconditions: Vec::new(),
                offer: offer.clone(),
                start_effects: vec![StartEffect::LoadCargo {
                    commodity: commodity.clone(),
                    quantity,
                    reserved: *reserved,
                }],
                objective: Objective::LandOnPlanet {
                    planet: dest_id.clone(),
                },
                requires: vec![CompletionRequirement::HasCargo {
                    commodity: commodity.clone(),
                    quantity,
                }],
                completion_effects: vec![
                    CompletionEffect::RemoveCargo {
                        commodity,
                        quantity,
                    },
                    CompletionEffect::Pay { credits: pay },
                ],
            };
            vec![(gen_id(template_id, rng), def)]
        }
        MissionTemplate::CollectFromAsteroidField {
            briefing,
            success_text,
            failure_text,
            offer,
            preconditions: _,
            quantity_range,
            pay_range,
        } => {
            // Collect all (system_id, system_display, commodity) tuples drawn
            // from asteroid fields across the universe, dedup by pair.
            let mut candidates: Vec<(String, String, String)> = Vec::new();
            for (sys_id, sys) in &universe.star_systems {
                let mut seen: std::collections::HashSet<&String> =
                    std::collections::HashSet::new();
                for field in &sys.astroid_fields {
                    for commodity in field.commodities.keys() {
                        if seen.insert(commodity) {
                            candidates.push((
                                sys_id.clone(),
                                sys.display_name.clone(),
                                commodity.clone(),
                            ));
                        }
                    }
                }
            }
            if candidates.is_empty() {
                return Vec::new();
            }
            let (sys_id, sys_display, commodity) =
                candidates[rng.gen_range(0..candidates.len())].clone();
            let quantity = rand_in_range_u16(rng, *quantity_range);
            let pay = rand_in_range_i128(rng, *pay_range);

            let vars = [
                ("{commodity}", commodity.clone()),
                ("{quantity}", quantity.to_string()),
                ("{pay}", pay.to_string()),
                ("{system}", sys_id.clone()),
                ("{system_display}", sys_display),
            ];
            let def = MissionDef {
                briefing: subst(briefing, &vars),
                success_text: subst(success_text, &vars),
                failure_text: subst(failure_text, &vars),
                preconditions: Vec::new(),
                offer: offer.clone(),
                start_effects: Vec::new(),
                objective: Objective::CollectPickups {
                    commodity,
                    system: sys_id,
                    quantity,
                },
                requires: Vec::new(),
                completion_effects: vec![CompletionEffect::Pay { credits: pay }],
            };
            vec![(gen_id(template_id, rng), def)]
        }
        MissionTemplate::CollectThenDeliver {
            stage1_briefing,
            stage1_success_text,
            stage1_failure_text,
            stage2_briefing,
            stage2_success_text,
            stage2_failure_text,
            offer,
            preconditions: _,
            quantity_range,
            pay_range,
        } => {
            // Pick a (system, commodity) from asteroid fields.
            let mut candidates: Vec<(String, String, String)> = Vec::new();
            for (sys_id, sys) in &universe.star_systems {
                let mut seen: std::collections::HashSet<&String> =
                    std::collections::HashSet::new();
                for field in &sys.astroid_fields {
                    for c in field.commodities.keys() {
                        if seen.insert(c) {
                            candidates.push((
                                sys_id.clone(),
                                sys.display_name.clone(),
                                c.clone(),
                            ));
                        }
                    }
                }
            }
            // Pick a destination planet != the collection system's planets
            // is not essential; any planet across the universe is fine.
            let destinations: Vec<(String, String)> = universe
                .star_systems
                .iter()
                .flat_map(|(_sys_id, sys)| {
                    sys.planets
                        .iter()
                        .map(|(pid, p)| (pid.clone(), p.display_name.clone()))
                })
                .filter(|(pid, _)| pid != offer_planet)
                .collect();
            if candidates.is_empty() || destinations.is_empty() {
                return Vec::new();
            }
            let (sys_id, sys_display, commodity) =
                candidates[rng.gen_range(0..candidates.len())].clone();
            let (dest_id, dest_display) =
                destinations[rng.gen_range(0..destinations.len())].clone();
            let quantity = rand_in_range_u16(rng, *quantity_range);
            let pay = rand_in_range_i128(rng, *pay_range);

            let stage1_id = gen_id(template_id, rng);
            let stage2_id = gen_id(template_id, rng);

            let vars = [
                ("{commodity}", commodity.clone()),
                ("{quantity}", quantity.to_string()),
                ("{pay}", pay.to_string()),
                ("{system}", sys_id.clone()),
                ("{system_display}", sys_display),
                ("{planet}", dest_id.clone()),
                ("{planet_display}", dest_display),
            ];

            let stage1 = MissionDef {
                briefing: subst(stage1_briefing, &vars),
                success_text: subst(stage1_success_text, &vars),
                failure_text: subst(stage1_failure_text, &vars),
                preconditions: Vec::new(),
                offer: offer.clone(),
                start_effects: Vec::new(),
                objective: Objective::CollectPickups {
                    commodity: commodity.clone(),
                    system: sys_id.clone(),
                    quantity,
                },
                requires: Vec::new(),
                completion_effects: Vec::new(),
            };
            let stage2 = MissionDef {
                briefing: subst(stage2_briefing, &vars),
                success_text: subst(stage2_success_text, &vars),
                failure_text: subst(stage2_failure_text, &vars),
                // Stage 2 is gated on stage 1's generated id, so it stays
                // Locked until the catalog sees stage 1 completed.
                preconditions: vec![Precondition::Completed {
                    mission: stage1_id.clone(),
                }],
                offer: OfferKind::Auto,
                start_effects: Vec::new(),
                objective: Objective::LandOnPlanet {
                    planet: dest_id.clone(),
                },
                requires: vec![CompletionRequirement::HasCargo {
                    commodity: commodity.clone(),
                    quantity,
                }],
                completion_effects: vec![
                    CompletionEffect::RemoveCargo {
                        commodity,
                        quantity,
                    },
                    CompletionEffect::Pay { credits: pay },
                ],
            };
            vec![(stage1_id, stage1), (stage2_id, stage2)]
        }
    }
}

fn rand_in_range_u16(rng: &mut impl Rng, range: (u16, u16)) -> u16 {
    let (lo, hi) = (range.0.min(range.1), range.0.max(range.1));
    if lo == hi {
        lo
    } else {
        rng.gen_range(lo..=hi)
    }
}

fn rand_in_range_i128(rng: &mut impl Rng, range: (i64, i64)) -> i64 {
    let (lo, hi) = (range.0.min(range.1), range.0.max(range.1));
    if lo == hi {
        lo
    } else {
        rng.gen_range(lo..=hi)
    }
}

fn gen_id(template_id: &str, rng: &mut impl Rng) -> String {
    let n: u64 = rng.r#gen();
    format!("{}__{:016x}", template_id, n)
}

fn subst(s: &str, vars: &[(&str, String)]) -> String {
    let mut out = s.to_string();
    for (k, v) in vars {
        out = out.replace(k, v);
    }
    out
}
