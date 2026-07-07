//! Cross-reference validation for the item-universe asset files.
//!
//! Called once at startup after all YAML files have been loaded and derived
//! tables computed.  Every check emits a `warn!` so the developer sees the
//! problem in the console without crashing the game.

use bevy::prelude::*;
use std::collections::{HashSet, VecDeque};
use std::path::Path;

use crate::item_universe::{ItemUniverse, OutfitterItem};
use crate::missions::types::{
    CompletionEffect, CompletionRequirement, MissionTemplate, Objective, OfferKind,
    Precondition, StartEffect,
};
use crate::planets::PlanetData;
use crate::ship::{Personality, ShipData};

/// Run every validation pass and log warnings for anything broken.
pub fn validate(iu: &ItemUniverse) {
    validate_faction_references(iu);
    validate_ship_weapon_space(iu);
    validate_uncolonized_planets(iu);
    validate_star_system_references(iu);
    validate_outfitter_items_reference_weapons(iu);
    validate_ship_base_weapons(iu);
    validate_weapon_carrier_bays(iu);
    validate_missions(iu);
    validate_mission_templates(iu);
    validate_ship_steering(iu);
    for problem in collect_problems(iu) {
        warn!("asset check: {problem}");
    }
}

/// Maximum rotation (degrees) the RL agent should be able to commit to in a
/// single decision step at full turn input from rest. Above this threshold,
/// 3-level discrete steering becomes too coarse to aim precisely.
const MAX_FIRST_TICK_DEG: f32 = 60.0;

fn validate_ship_steering(iu: &ItemUniverse) {
    use crate::rl_collection::RL_STEP_PERIOD;
    use crate::ship::first_tick_rotation_rad;
    for (ship_name, ship_data) in &iu.ships {
        let rad = first_tick_rotation_rad(ship_data.torque, ship_data.angular_drag, RL_STEP_PERIOD);
        let deg = rad.to_degrees();
        if deg > MAX_FIRST_TICK_DEG {
            warn!(
                "Ship \"{ship_name}\" rotates {deg:.1}° in one RL tick \
                 ({RL_STEP_PERIOD}s) at full turn from rest (torque={}, \
                 angular_drag={}); exceeds {MAX_FIRST_TICK_DEG}° steering \
                 budget — raise angular_drag in assets/ships.yaml",
                ship_data.torque, ship_data.angular_drag,
            );
        }
    }
}

/// Every faction named by a planet, system, ship, or enemies/allies entry
/// must exist in the faction registry (assets/factions.yaml) — a typo'd
/// faction silently behaves like a real (unknown) territorial power.
fn validate_faction_references(iu: &ItemUniverse) {
    let check = |kind: &str, whom: &str, f: &str| {
        if !f.is_empty() && !iu.factions.contains_key(f) {
            warn!("{kind} \"{whom}\" references faction \"{f}\" missing from factions.yaml");
        }
    };
    for (sys_name, sys) in &iu.star_systems {
        check("system", sys_name, &sys.faction);
        for (p_name, p) in &sys.planets {
            check("planet", p_name, &p.faction);
        }
    }
    for (ship_name, ship) in &iu.ships {
        if let Some(f) = &ship.faction {
            check("ship", ship_name, f);
        }
        for f in &ship.sold_by {
            check("ship sold_by", ship_name, f);
        }
    }
    for (f, enemies) in &iu.enemies {
        check("enemies.yaml", f, f);
        for e in enemies {
            check("enemies.yaml", f, e);
        }
    }
    for (f, allies) in &iu.allies {
        check("allies.yaml", f, f);
        for a in allies {
            check("allies.yaml", f, a);
        }
    }
}

// ── helpers ─────────────────────────────────────────────────────────────────

/// Collect every planet name across all star systems.
fn all_planet_names(iu: &ItemUniverse) -> HashSet<&str> {
    iu.star_systems
        .values()
        .flat_map(|sys| sys.planets.keys())
        .map(|s| s.as_str())
        .collect()
}

// ── existing checks (moved from ItemUniverse::validate) ─────────────────────

fn validate_ship_weapon_space(iu: &ItemUniverse) {
    use crate::weapons::WeaponSystems;
    for (ship_name, ship_data) in &iu.ships {
        let consumed: i32 = WeaponSystems::build(&ship_data.base_weapons, iu)
            .iter_all()
            .map(|(_, s)| s.space_consumed())
            .sum();
        if consumed > ship_data.item_space as i32 {
            warn!(
                "Ship \"{ship_name}\" base weapons consume {consumed} item space \
                 but ship only has {} — check assets/ships.yaml",
                ship_data.item_space
            );
        }
    }
}

fn validate_uncolonized_planets(iu: &ItemUniverse) {
    for (sys_name, system) in &iu.star_systems {
        for (planet_name, planet) in &system.planets {
            if planet.uncolonized && !planet.commodities.is_empty() {
                warn!(
                    "Planet \"{planet_name}\" in \"{sys_name}\" is uncolonized \
                     but has commodities — these will be inaccessible"
                );
            }
            if planet.uncolonized && !planet.outfitter.is_empty() {
                warn!(
                    "Planet \"{planet_name}\" in \"{sys_name}\" is uncolonized \
                     but has an outfitter — this will be inaccessible"
                );
            }
            if planet.uncolonized && !planet.shipyard.is_empty() {
                warn!(
                    "Planet \"{planet_name}\" in \"{sys_name}\" is uncolonized \
                     but has a shipyard — this will be inaccessible"
                );
            }
        }
    }
}

// ── new cross-reference checks ──────────────────────────────────────────────

/// 1. Everything sold/referenced in star_systems.yaml must exist in the
///    corresponding top-level definition file.
fn validate_star_system_references(iu: &ItemUniverse) {
    for (sys_name, system) in &iu.star_systems {
        // System connections
        for conn in &system.connections {
            if !iu.star_systems.contains_key(conn) {
                warn!(
                    "System \"{sys_name}\" has connection \"{conn}\" which is \
                     not a defined star system — check assets/star_systems.yaml"
                );
            }
        }

        for (planet_name, planet) in &system.planets {
            // Commodities sold on planet
            for commodity in planet.commodities.keys() {
                if !iu.commodities.contains_key(commodity) {
                    warn!(
                        "Planet \"{planet_name}\" in \"{sys_name}\" sells commodity \
                         \"{commodity}\" which is not defined in commodities.yaml"
                    );
                }
            }
            // Outfitter items sold on planet
            for item in &planet.outfitter {
                if !iu.outfitter_items.contains_key(item) {
                    warn!(
                        "Planet \"{planet_name}\" in \"{sys_name}\" sells outfitter \
                         item \"{item}\" which is not defined in outfitter_items.yaml"
                    );
                }
            }
            // Ships sold on planet
            for ship in &planet.shipyard {
                if !iu.ships.contains_key(ship) {
                    warn!(
                        "Planet \"{planet_name}\" in \"{sys_name}\" sells ship \
                         \"{ship}\" which is not defined in ships.yaml"
                    );
                }
            }
        }

        // Asteroid field commodities
        for (idx, field) in system.astroid_fields.iter().enumerate() {
            for commodity in field.commodities.keys() {
                if !iu.commodities.contains_key(commodity) {
                    warn!(
                        "Asteroid field #{idx} in \"{sys_name}\" drops commodity \
                         \"{commodity}\" which is not defined in commodities.yaml"
                    );
                }
            }
        }

        // Ship distribution types
        for ship_type in system.ships.types.keys() {
            if !iu.ships.contains_key(ship_type) {
                warn!(
                    "System \"{sys_name}\" ship distribution references ship type \
                     \"{ship_type}\" which is not defined in ships.yaml"
                );
            }
        }
    }
}

/// 2. Every outfitter item's weapon_type must exist in weapons.yaml.
fn validate_outfitter_items_reference_weapons(iu: &ItemUniverse) {
    for (item_name, item) in &iu.outfitter_items {
        let wtype = match item {
            OutfitterItem::PrimaryWeapon { weapon_type, .. } => weapon_type,
            OutfitterItem::SecondaryWeapon { weapon_type, .. } => weapon_type,
            OutfitterItem::ShipMod { .. } => continue, // no projectile weapon
        };
        if !iu.weapons.contains_key(wtype) {
            warn!(
                "Outfitter item \"{item_name}\" references weapon_type \"{wtype}\" \
                 which is not defined in weapons.yaml"
            );
        }
    }
}

/// 3. Every weapon listed in a ship's base_weapons must exist in weapons.yaml.
fn validate_ship_base_weapons(iu: &ItemUniverse) {
    for (ship_name, ship_data) in &iu.ships {
        for weapon_key in ship_data.base_weapons.keys() {
            if !iu.weapons.contains_key(weapon_key) {
                warn!(
                    "Ship \"{ship_name}\" has base weapon \"{weapon_key}\" \
                     which is not defined in weapons.yaml"
                );
            }
            // Also check that there is a corresponding outfitter item (needed
            // for space/price lookups).
            if !iu.outfitter_items.contains_key(weapon_key) {
                warn!(
                    "Ship \"{ship_name}\" has base weapon \"{weapon_key}\" \
                     which has no matching entry in outfitter_items.yaml"
                );
            }
        }
    }
}

/// Weapons with carrier-bay behavior must reference a valid ship type.
fn validate_weapon_carrier_bays(iu: &ItemUniverse) {
    for (weapon_name, weapon) in &iu.weapons {
        if let Some(bay_ship) = weapon.carrier_bay() {
            if !iu.ships.contains_key(bay_ship) {
                warn!(
                    "Weapon \"{weapon_name}\" has carrier bay ship \"{bay_ship}\" \
                     which is not defined in ships.yaml"
                );
            }
        }
    }
}

// ── mission validation ──────────────────────────────────────────────────────

/// 4. All planets, systems, commodities, ships, and mission cross-references
///    in missions.yaml must exist.
fn validate_missions(iu: &ItemUniverse) {
    let planets = all_planet_names(iu);

    for (mission_id, def) in &iu.missions {
        validate_preconditions(&def.preconditions, mission_id, "mission", iu);
        for ship in &def.squadron {
            if !iu.ships.contains_key(ship) {
                warn!(
                    "mission \"{mission_id}\" squadron references ship type \"{ship}\" \
                     which is not defined in ships.yaml"
                );
            }
        }
        validate_offer(&def.offer, mission_id, "mission", &planets);
        validate_objective(&def.objective, mission_id, "mission", iu, &planets);
        validate_start_effects(&def.start_effects, mission_id, "mission", iu);
        validate_completion_requirements(&def.requires, mission_id, "mission", iu);
        validate_completion_effects(&def.completion_effects, mission_id, "mission", iu);
    }
}

fn validate_mission_templates(iu: &ItemUniverse) {
    let planets = all_planet_names(iu);

    for (tmpl_id, tmpl) in &iu.mission_templates {
        let label = "mission template";

        // Preconditions (shared by all variants)
        validate_preconditions(tmpl.preconditions(), tmpl_id, label, iu);

        match tmpl {
            MissionTemplate::Delivery {
                offer,
                commodity_pool,
                ..
            } => {
                validate_offer(offer, tmpl_id, label, &planets);
                for c in commodity_pool {
                    if !iu.commodities.contains_key(c) {
                        warn!(
                            "Mission template \"{tmpl_id}\" commodity_pool contains \
                             \"{c}\" which is not defined in commodities.yaml"
                        );
                    }
                }
            }
            MissionTemplate::CollectFromAsteroidField { offer, .. } => {
                validate_offer(offer, tmpl_id, label, &planets);
            }
            MissionTemplate::CollectThenDeliver { offer, .. } => {
                validate_offer(offer, tmpl_id, label, &planets);
            }
            MissionTemplate::BountyHunt {
                offer,
                ship_type_pool,
                ..
            } => {
                validate_offer(offer, tmpl_id, label, &planets);
                for s in ship_type_pool {
                    if !iu.ships.contains_key(s) {
                        warn!(
                            "Mission template \"{tmpl_id}\" ship_type_pool contains \
                             \"{s}\" which is not defined in ships.yaml"
                        );
                    }
                }
            }
            MissionTemplate::War { .. } => {}
            MissionTemplate::Covert { action, .. } => {
                if let crate::missions::types::CovertAction::Smuggle { commodity, .. } = action {
                    if !iu.commodities.contains_key(commodity) {
                        warn!(
                            "Mission template \"{tmpl_id}\" smuggle commodity \
                             \"{commodity}\" is not defined in commodities.yaml"
                        );
                    }
                }
            }
            MissionTemplate::Arrest {
                service_commodity, ..
            } => {
                if !iu.commodities.contains_key(service_commodity) {
                    warn!(
                        "Mission template \"{tmpl_id}\" service_commodity \"{service_commodity}\" \
                         is not defined in commodities.yaml"
                    );
                }
            }
            MissionTemplate::CatchThief {
                offer,
                ship_type_pool,
                commodity_pool,
                ..
            } => {
                validate_offer(offer, tmpl_id, label, &planets);
                for s in ship_type_pool {
                    if !iu.ships.contains_key(s) {
                        warn!(
                            "Mission template \"{tmpl_id}\" ship_type_pool contains \
                             \"{s}\" which is not defined in ships.yaml"
                        );
                    }
                }
                for c in commodity_pool {
                    if !iu.commodities.contains_key(c) {
                        warn!(
                            "Mission template \"{tmpl_id}\" commodity_pool contains \
                             \"{c}\" which is not defined in commodities.yaml"
                        );
                    }
                }
            }
        }
    }
}

// ── sub-field validators ────────────────────────────────────────────────────

fn validate_preconditions(
    preconds: &[Precondition],
    id: &str,
    label: &str,
    iu: &ItemUniverse,
) {
    for p in preconds {
        match p {
            Precondition::Completed { mission } | Precondition::Failed { mission } => {
                if !iu.missions.contains_key(mission) {
                    warn!(
                        "{label} \"{id}\" has precondition referencing mission \
                         \"{mission}\" which is not defined in missions.yaml"
                    );
                }
            }
            Precondition::HasUnlock { .. } => {}
        }
    }
}

fn validate_offer(
    offer: &OfferKind,
    id: &str,
    label: &str,
    planets: &HashSet<&str>,
) {
    if let OfferKind::NpcOffer { planet, .. } = offer {
        if !planets.contains(planet.as_str()) {
            warn!(
                "{label} \"{id}\" NpcOffer references planet \"{planet}\" \
                 which does not exist in any star system"
            );
        }
    }
}

fn validate_objective(
    obj: &Objective,
    id: &str,
    label: &str,
    iu: &ItemUniverse,
    planets: &HashSet<&str>,
) {
    match obj {
        Objective::TravelToSystem { system } => {
            if !iu.star_systems.contains_key(system) {
                warn!(
                    "{label} \"{id}\" objective references system \"{system}\" \
                     which is not defined in star_systems.yaml"
                );
            }
        }
        Objective::LandOnPlanet { planet } => {
            if !planets.contains(planet.as_str()) {
                warn!(
                    "{label} \"{id}\" objective references planet \"{planet}\" \
                     which does not exist in any star system"
                );
            }
        }
        Objective::CollectPickups {
            commodity, system, ..
        } => {
            if !iu.star_systems.contains_key(system) {
                warn!(
                    "{label} \"{id}\" objective references system \"{system}\" \
                     which is not defined in star_systems.yaml"
                );
            }
            if !iu.commodities.contains_key(commodity) {
                warn!(
                    "{label} \"{id}\" objective references commodity \"{commodity}\" \
                     which is not defined in commodities.yaml"
                );
            }
        }
        Objective::MeetNpc { planet, .. } | Objective::CatchNpc { planet, .. } => {
            if !planets.contains(planet.as_str()) {
                warn!(
                    "{label} \"{id}\" objective references planet \"{planet}\" \
                     which does not exist in any star system"
                );
            }
        }
        Objective::DestroyShips {
            system,
            ship_type,
            count,
            collect,
            ..
        } => {
            if !iu.star_systems.contains_key(system) {
                warn!(
                    "{label} \"{id}\" objective references system \"{system}\" \
                     which is not defined in star_systems.yaml"
                );
            }
            if !iu.ships.contains_key(ship_type) {
                warn!(
                    "{label} \"{id}\" objective references ship_type \"{ship_type}\" \
                     which is not defined in ships.yaml"
                );
            }
            if let Some(req) = collect {
                if !iu.commodities.contains_key(&req.commodity) {
                    warn!(
                        "{label} \"{id}\" objective collect references commodity \
                         \"{}\" which is not defined in commodities.yaml",
                        req.commodity
                    );
                }
                // The `collect.quantity` units are distributed across the
                // spawned ships by `spawn_mission_targets`; the first
                // `quantity % count` ships carry one extra unit. Verify the
                // per-ship share fits in the ship_type's cargo hold so the
                // player can actually collect the full quantity.
                if *count > 0 {
                    if let Some(ship_data) = iu.ships.get(ship_type) {
                        let hold = ship_data.cargo_space;
                        let per_ship_max = req.quantity.div_ceil(*count as u16);
                        if per_ship_max > hold {
                            warn!(
                                "{label} \"{id}\" objective collect requires {} units of \
                                 \"{}\" across {} \"{ship_type}\" ships — each ship must \
                                 carry up to {per_ship_max} units, but ship_type cargo \
                                 hold is only {hold}. Reduce collect.quantity, raise count, \
                                 or pick a ship with more cargo_space.",
                                req.quantity, req.commodity, count,
                            );
                        }
                    }
                }
            }
        }
    }
}

fn validate_start_effects(
    effects: &[StartEffect],
    id: &str,
    label: &str,
    iu: &ItemUniverse,
) {
    for e in effects {
        match e {
            StartEffect::LoadCargo { commodity, .. } => {
                if !iu.commodities.contains_key(commodity) {
                    warn!(
                        "{label} \"{id}\" start effect references commodity \
                         \"{commodity}\" which is not defined in commodities.yaml"
                    );
                }
            }
        }
    }
}

fn validate_completion_requirements(
    reqs: &[CompletionRequirement],
    id: &str,
    label: &str,
    iu: &ItemUniverse,
) {
    for r in reqs {
        match r {
            CompletionRequirement::HasCargo { commodity, .. } => {
                if !iu.commodities.contains_key(commodity) {
                    warn!(
                        "{label} \"{id}\" completion requirement references \
                         commodity \"{commodity}\" which is not defined in \
                         commodities.yaml"
                    );
                }
            }
            CompletionRequirement::HasUnlock { .. } => {}
        }
    }
}

fn validate_completion_effects(
    effects: &[CompletionEffect],
    id: &str,
    label: &str,
    iu: &ItemUniverse,
) {
    for e in effects {
        match e {
            CompletionEffect::RemoveCargo { commodity, .. } => {
                if !iu.commodities.contains_key(commodity) {
                    warn!(
                        "{label} \"{id}\" completion effect references commodity \
                         \"{commodity}\" which is not defined in commodities.yaml"
                    );
                }
            }
            CompletionEffect::Pay { .. }
            | CompletionEffect::GrantUnlock { .. }
            | CompletionEffect::AdjustStanding { .. } => {}
            CompletionEffect::ShiftInfluence { system, .. } => {
                if !iu.star_systems.contains_key(system) {
                    warn!(
                        "{label} \"{id}\" ShiftInfluence references system \"{system}\" \
                         which is not defined in star_systems.yaml"
                    );
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Strict, TEST-ENFORCED asset checks.
//
// The `warn!` passes above are advisory (logged, non-fatal). The checks below
// return a list of problems instead: `validate()` logs them at load time, and
// the asset-load test asserts the list is EMPTY — so a broken asset fails CI/dev
// rather than silently shipping. Covers the invariants the game relies on:
//   * merchant-heavy systems have a real within-system trade route,
//   * every ship/planet has its in-game + HUD-wireframe sprites,
//   * every star system is reachable from the start,
//   * missions are offered/fulfilled on landable planets at buildings that exist.
// ─────────────────────────────────────────────────────────────────────────────

/// Training-only systems folded in by `materialize_training_systems`; they are
/// intentionally disconnected from the campaign jump graph, so skip them in the
/// reachability check.
const TRAINING_SYSTEMS: &[&str] = &ItemUniverse::TRAINING_SYSTEM_KEYS;

/// A planet is "landable" (has a surface with buildings) iff it is colonised,
/// which the data models as having a non-empty commodity market.
fn is_landable(pd: &PlanetData) -> bool {
    !pd.commodities.is_empty()
}

/// Whether `building` (a mission offer/objective building name) is present on a
/// landable planet. bar / fuel / mechanic exist on every colony; market needs a
/// market; shipyard / outfitter need that specific facility.
fn planet_has_building(pd: &PlanetData, building: &str) -> bool {
    match building {
        "market" | "bar" | "fuel_station" | "mechanicshop" => is_landable(pd),
        // The garrison exists only on faction-held worlds (live controller),
        // so static content should use it sparingly; landable is the floor.
        "garrison" => is_landable(pd),
        "outfitter" => !pd.outfitter.is_empty(),
        "shipyard" => !pd.shipyard.is_empty(),
        _ => false, // unknown building name => typo
    }
}

fn find_planet<'a>(iu: &'a ItemUniverse, name: &str) -> Option<&'a PlanetData> {
    iu.find_gameplay_planet(name).map(|(_, p)| p)
}

/// Run all strict checks and collect human-readable problems (empty == good).
pub fn collect_problems(iu: &ItemUniverse) -> Vec<String> {
    let mut p = Vec::new();
    check_base_weapons_fit_mounts(iu, &mut p);
    check_trade_routes(iu, &mut p);
    check_sprites_exist(iu, &mut p);
    check_reachability(iu, &mut p);
    check_mission_coherence(iu, &mut p);
    check_unlock_obtainability(iu, &mut p);
    check_ship_weapons_buyable(iu, &mut p);
    check_everything_sold_somewhere(iu, &mut p);
    check_fenced_carrier_items(iu, &mut p);
    check_mission_graph(iu, &mut p);
    p
}

/// Every hull's factory loadout must fit its own gun/turret mounts —
/// otherwise the limits would be a lie the moment an AI ship spawns.
fn check_base_weapons_fit_mounts(iu: &ItemUniverse, p: &mut Vec<String>) {
    for (ship_name, data) in &iu.ships {
        let mut guns = 0u32;
        let mut turrets = 0u32;
        for (weapon_name, (count, _)) in &data.base_weapons {
            let Some(w) = iu.weapons.get(weapon_name) else {
                continue; // dangling reference — validate_ship_base_weapons reports it
            };
            if !w.uses_mount() {
                continue;
            }
            if w.is_turret() {
                turrets += *count as u32;
            } else {
                guns += *count as u32;
            }
        }
        if guns > data.gun_mounts as u32 {
            p.push(format!(
                "ship '{ship_name}': base loadout uses {guns} gun mounts but the hull has {}",
                data.gun_mounts
            ));
        }
        if turrets > data.turret_mounts as u32 {
            p.push(format!(
                "ship '{ship_name}': base loadout uses {turrets} turret mounts but the hull has {}",
                data.turret_mounts
            ));
        }
    }
}

/// An outfitter item sold UNIVERSALLY (empty `factions`) whose only carriers
/// are faction-fenced hulls is almost certainly a mistake — e.g. the pirate
/// corvette bay on sale at Earth while the corvettes themselves are fenced
/// through rebel yards. Warn so the item gets fenced like its carriers.
fn check_fenced_carrier_items(iu: &ItemUniverse, p: &mut Vec<String>) {
    for (item_name, item) in &iu.outfitter_items {
        if !item.factions().is_empty() {
            continue; // already fenced
        }
        let carriers: Vec<&ShipData> = iu
            .ships
            .values()
            .filter(|d| d.base_weapons.contains_key(item_name))
            .collect();
        if carriers.is_empty() {
            continue; // generic gear nobody ships with — universal is fine
        }
        let planet_factions: HashSet<&str> = iu
            .star_systems
            .values()
            .flat_map(|s| s.planets.values())
            .map(|pd| pd.faction.as_str())
            .filter(|f| iu.faction_takes_sides(f))
            .collect();
        let all_fenced = carriers.iter().all(|d| {
            if !d.sold_by.is_empty() {
                return true;
            }
            d.faction
                .as_deref()
                .is_some_and(|f| planet_factions.contains(f))
        });
        if !all_fenced {
            continue;
        }
        let fences: HashSet<&str> = carriers
            .iter()
            .flat_map(|d| {
                if d.sold_by.is_empty() {
                    d.faction.iter().map(String::as_str).collect::<Vec<_>>()
                } else {
                    d.sold_by.iter().map(String::as_str).collect()
                }
            })
            .collect();
        // Only the unambiguous case: every carrier is fenced to ONE faction.
        // Gear spanning several factions' hulls is genuinely generic.
        if fences.len() == 1 {
            p.push(format!(
                "outfitter item '{item_name}' is sold universally but only \
                 {fences:?}-fenced hulls carry it — fence the item's \
                 `factions` to match its carriers"
            ));
        }
    }
}

/// Every ship and outfitter item must be purchasable SOMEWHERE: stocked on at
/// least one landable planet, in a reachable (non-training) system, at a
/// planet that actually has the right building (a non-empty shipyard/outfitter
/// list IS the building, per `planet_has_building`). Licences may still gate
/// the purchase — obtainability of those is checked separately. Run after
/// `derive_market_catalogs`, so tech-level/faction coverage gaps surface here.
fn check_everything_sold_somewhere(iu: &ItemUniverse, p: &mut Vec<String>) {
    let mut sold_ships: HashSet<&str> = HashSet::new();
    let mut sold_items: HashSet<&str> = HashSet::new();
    for (sys_name, sys) in &iu.star_systems {
        if TRAINING_SYSTEMS.contains(&sys_name.as_str()) {
            continue;
        }
        for pd in sys.planets.values() {
            if !is_landable(pd) {
                continue;
            }
            sold_ships.extend(pd.shipyard.iter().map(String::as_str));
            sold_items.extend(pd.outfitter.iter().map(String::as_str));
        }
    }
    for name in iu.ships.keys() {
        if !sold_ships.contains(name.as_str()) {
            p.push(format!(
                "ship '{name}' is sold nowhere — no landable planet's tech level \
                 + faction stocks it (check tech_level/sold_by in ships.yaml)"
            ));
        }
    }
    for name in iu.outfitter_items.keys() {
        if !sold_items.contains(name.as_str()) {
            p.push(format!(
                "outfitter item '{name}' is sold nowhere — no landable planet's \
                 tech level + faction stocks it (check tech_level/factions)"
            ));
        }
    }
}

/// Buying a ship must imply being able to buy every weapon it spawns with.
/// For each ship sold anywhere, each of its base weapons must be sold at the
/// outfitter, and the unlocks that outfitter entry needs must be guaranteed
/// owned by the time the ship is buyable — i.e. reachable along the mission
/// grant graph from the ship's own `required_unlocks`. Mirrors
/// `scripts/validate_progression.py`.
fn check_ship_weapons_buyable(iu: &ItemUniverse, p: &mut Vec<String>) {
    use std::collections::HashMap;
    // mission -> unlocks it grants ; mission -> prerequisite missions
    let mut grants: HashMap<&str, HashSet<&str>> = HashMap::new();
    let mut prereqs: HashMap<&str, Vec<&str>> = HashMap::new();
    for (mid, def) in &iu.missions {
        let mut g = HashSet::new();
        for eff in &def.completion_effects {
            if let CompletionEffect::GrantUnlock { name } = eff {
                g.insert(name.as_str());
            }
        }
        let pre = def
            .preconditions
            .iter()
            .filter_map(|pc| match pc {
                Precondition::Completed { mission } if iu.missions.contains_key(mission) => {
                    Some(mission.as_str())
                }
                _ => None,
            })
            .collect();
        grants.insert(mid.as_str(), g);
        prereqs.insert(mid.as_str(), pre);
    }
    // held_after[m] = unlocks guaranteed owned once m is completed (fixpoint).
    let mut held: HashMap<&str, HashSet<&str>> = grants.clone();
    let keys: Vec<&str> = held.keys().copied().collect();
    let mut changed = true;
    while changed {
        changed = false;
        for &m in &keys {
            let mut new = grants[m].clone();
            for pr in &prereqs[m] {
                if let Some(h) = held.get(pr) {
                    new.extend(h.iter().copied());
                }
            }
            if new.len() != held[m].len() {
                held.insert(m, new);
                changed = true;
            }
        }
    }
    // reach(u): unlocks guaranteed already owned whenever u is owned =
    // intersection over every mission that grants u of held_after[m].
    let reach = |u: &str| -> HashSet<&str> {
        let mut acc: Option<HashSet<&str>> = None;
        for (&m, g) in &grants {
            if g.contains(u) {
                acc = Some(match acc {
                    None => held[m].clone(),
                    Some(a) => a.intersection(&held[m]).copied().collect(),
                });
            }
        }
        acc.unwrap_or_default()
    };
    let mut sold_ships: HashSet<&str> = HashSet::new();
    for sys in iu.star_systems.values() {
        for pd in sys.planets.values() {
            sold_ships.extend(pd.shipyard.iter().map(String::as_str));
        }
    }
    for s in &sold_ships {
        let Some(ship) = iu.ships.get(*s) else { continue };
        // Everything guaranteed owned once this ship is buyable.
        let mut have: HashSet<&str> = ship.required_unlocks.iter().map(String::as_str).collect();
        for u in &ship.required_unlocks {
            have.extend(reach(u));
        }
        for w in ship.base_weapons.keys() {
            match iu.outfitter_items.get(w) {
                None => p.push(format!(
                    "ship '{s}' spawns with weapon '{w}' which is not sold at any \
                     outfitter — a buyer could never replace it"
                )),
                Some(item) => {
                    for u in item.required_unlocks() {
                        if !have.contains(u.as_str()) {
                            p.push(format!(
                                "ship '{s}' spawns with weapon '{w}', but buying that \
                                 weapon needs unlock '{u}' which owning '{s}' does not \
                                 guarantee"
                            ));
                        }
                    }
                }
            }
        }
    }
}

/// The mission dependency graph must be a DAG whose `completed`/`failed`
/// preconditions all reference defined missions — otherwise a mission can never
/// be started (a dangling prerequisite, or a cycle that never resolves).
fn check_mission_graph(iu: &ItemUniverse, p: &mut Vec<String>) {
    for (id, def) in &iu.missions {
        for pc in &def.preconditions {
            if let Precondition::Completed { mission } | Precondition::Failed { mission } = pc {
                if !iu.missions.contains_key(mission) {
                    p.push(format!(
                        "mission '{id}': precondition references unknown mission '{mission}'"
                    ));
                }
            }
        }
    }
    let mut color: std::collections::HashMap<&str, u8> = std::collections::HashMap::new();
    for id in iu.missions.keys() {
        if color.get(id.as_str()).copied().unwrap_or(0) == 0 {
            mission_cycle_dfs(id.as_str(), iu, &mut color, p);
        }
    }
}

fn mission_cycle_dfs<'a>(
    id: &'a str,
    iu: &'a ItemUniverse,
    color: &mut std::collections::HashMap<&'a str, u8>,
    p: &mut Vec<String>,
) {
    color.insert(id, 1); // grey = on the current DFS stack
    if let Some(def) = iu.missions.get(id) {
        for pc in &def.preconditions {
            if let Precondition::Completed { mission } = pc {
                if !iu.missions.contains_key(mission.as_str()) {
                    continue;
                }
                match color.get(mission.as_str()).copied().unwrap_or(0) {
                    1 => p.push(format!(
                        "mission precondition cycle: '{id}' depends on '{mission}', \
                         which loops back — neither can ever be started"
                    )),
                    0 => mission_cycle_dfs(mission.as_str(), iu, color, p),
                    _ => {}
                }
            }
        }
    }
    color.insert(id, 2); // black = fully explored
}

/// Every ship/weapon actually SOLD in a shipyard or outfitter that is gated by a
/// `required_unlock` must have that unlock granted by some mission — otherwise
/// the player can see it for sale but can never buy it.
fn check_unlock_obtainability(iu: &ItemUniverse, p: &mut Vec<String>) {
    let mut granted: HashSet<&str> = HashSet::new();
    for def in iu.missions.values() {
        for eff in &def.completion_effects {
            if let CompletionEffect::GrantUnlock { name } = eff {
                granted.insert(name.as_str());
            }
        }
    }
    let mut sold_ships: HashSet<&str> = HashSet::new();
    let mut sold_items: HashSet<&str> = HashSet::new();
    for sys in iu.star_systems.values() {
        for pd in sys.planets.values() {
            sold_ships.extend(pd.shipyard.iter().map(String::as_str));
            sold_items.extend(pd.outfitter.iter().map(String::as_str));
        }
    }
    for s in &sold_ships {
        if let Some(ship) = iu.ships.get(*s) {
            for u in &ship.required_unlocks {
                if !granted.contains(u.as_str()) {
                    p.push(format!(
                        "ship '{s}' is sold but its unlock '{u}' is never granted \
                         by any mission — the player can never buy it"
                    ));
                }
            }
        }
    }
    for i in &sold_items {
        if let Some(item) = iu.outfitter_items.get(*i) {
            for u in item.required_unlocks() {
                if !granted.contains(u.as_str()) {
                    p.push(format!(
                        "outfitter item '{i}' is sold but its unlock '{u}' is never \
                         granted by any mission — the player can never buy it"
                    ));
                }
            }
        }
    }
}

/// Systems where more than 10% of spawn traffic is merchants must have a valid
/// within-system trade route (>= 2 colonised planets). A rare merchant passing
/// through a border system (<= 10%) is fine. Miners likewise need ore to mine.
fn check_trade_routes(iu: &ItemUniverse, p: &mut Vec<String>) {
    for (name, sys) in &iu.star_systems {
        if TRAINING_SYSTEMS.contains(&name.as_str()) {
            continue;
        }
        let total: f32 = sys.ships.types.values().copied().sum();
        if total <= 0.0 {
            continue;
        }
        let is_personality = |ship: &str, want: &Personality| {
            iu.ships.get(ship).map(|s| &s.personality) == Some(want)
        };
        let merchant: f32 = sys
            .ships
            .types
            .iter()
            .filter(|(n, _)| is_personality(n, &Personality::Trader))
            .map(|(_, w)| *w)
            .sum();
        if merchant / total > 0.10 {
            let colonised = sys.planets.values().filter(|pd| is_landable(pd)).count();
            if colonised < 2 {
                p.push(format!(
                    "system '{name}': {:.0}% merchant traffic but {colonised} colonised \
                     planet(s) — traders have no within-system route to run",
                    merchant / total * 100.0
                ));
            }
        }
        let has_miner = sys
            .ships
            .types
            .keys()
            .any(|n| is_personality(n, &Personality::Miner));
        if has_miner && sys.astroid_fields.is_empty() {
            p.push(format!(
                "system '{name}': spawns miner ships but has no asteroid fields to mine"
            ));
        }
    }
}

/// Every ship needs its in-game atlas sprite and a HUD target wireframe; every
/// planet type and the generic asteroid/pickup targets need a wireframe too.
fn check_sprites_exist(iu: &ItemUniverse, p: &mut Vec<String>) {
    for (name, ship) in &iu.ships {
        if !Path::new("assets").join(&ship.sprite_path).exists() {
            p.push(format!(
                "ship '{name}': missing in-game sprite assets/{}",
                ship.sprite_path
            ));
        }
        let wf = format!("assets/sprites/wireframes/{name}.png");
        if !Path::new(&wf).exists() {
            p.push(format!("ship '{name}': missing HUD wireframe {wf}"));
        }
    }
    let mut planet_types = HashSet::new();
    for sys in iu.star_systems.values() {
        for (pname, pd) in &sys.planets {
            // in-game sprite (planets load sprites/planets/<name>.png at startup)
            let sprite = format!("assets/sprites/planets/{pname}.png");
            if !Path::new(&sprite).exists() {
                p.push(format!("planet '{pname}': missing in-game sprite {sprite}"));
            }
            if !pd.planet_type.is_empty() {
                planet_types.insert(pd.planet_type.clone());
            }
        }
    }
    for pt in &planet_types {
        let wf = format!("assets/sprites/wireframes/planet_{pt}.png");
        if !Path::new(&wf).exists() {
            p.push(format!(
                "planet type '{pt}': missing HUD wireframe {wf}"
            ));
        }
    }
    for generic in ["asteroid", "pickup"] {
        let wf = format!("assets/sprites/wireframes/{generic}.png");
        if !Path::new(&wf).exists() {
            p.push(format!("missing generic HUD wireframe {wf}"));
        }
    }
}

/// Every (non-training) star system must be reachable from the start system via
/// the jump-connection graph, or the player can never get there.
fn check_reachability(iu: &ItemUniverse, p: &mut Vec<String>) {
    let start = if iu.star_systems.contains_key(&iu.starting_system) {
        iu.starting_system.clone()
    } else {
        match iu.star_systems.keys().next() {
            Some(s) => s.clone(),
            None => return,
        }
    };
    let mut seen = HashSet::new();
    let mut queue = VecDeque::new();
    seen.insert(start.clone());
    queue.push_back(start);
    while let Some(s) = queue.pop_front() {
        if let Some(sys) = iu.star_systems.get(&s) {
            for c in &sys.connections {
                if iu.star_systems.contains_key(c) && seen.insert(c.clone()) {
                    queue.push_back(c.clone());
                }
            }
        }
    }
    for name in iu.star_systems.keys() {
        if !TRAINING_SYSTEMS.contains(&name.as_str()) && !seen.contains(name) {
            p.push(format!(
                "system '{name}' is unreachable from start system \
                 '{}' (no jump path)",
                iu.starting_system
            ));
        }
    }
}

/// Missions must be offered, and have surface objectives, on landable planets at
/// buildings that actually exist there.
fn check_mission_coherence(iu: &ItemUniverse, p: &mut Vec<String>) {
    for (id, def) in &iu.missions {
        // offer location
        if let OfferKind::NpcOffer {
            planet, building, ..
        } = &def.offer
        {
            if let Some(pd) = find_planet(iu, planet) {
                check_surface(id, "offered on", planet, pd, building.as_deref(), p);
            }
        }
        // surface objectives
        match &def.objective {
            Objective::LandOnPlanet { planet } => {
                if let Some(pd) = find_planet(iu, planet) {
                    if !is_landable(pd) {
                        p.push(format!(
                            "mission '{id}': land objective '{planet}' is \
                             uncolonised — nowhere to set down"
                        ));
                    }
                }
            }
            Objective::MeetNpc {
                planet, building, ..
            }
            | Objective::CatchNpc {
                planet, building, ..
            } => {
                if let Some(pd) = find_planet(iu, planet) {
                    check_surface(id, "has a surface objective on", planet, pd, building.as_deref(), p);
                }
            }
            _ => {}
        }
    }
}

fn check_surface(
    id: &str,
    verb: &str,
    planet: &str,
    pd: &PlanetData,
    building: Option<&str>,
    p: &mut Vec<String>,
) {
    if !is_landable(pd) {
        p.push(format!(
            "mission '{id}': {verb} '{planet}' which is uncolonised — can't land there"
        ));
        return;
    }
    if let Some(b) = building {
        if !planet_has_building(pd, b) {
            p.push(format!(
                "mission '{id}': {verb} '{planet}' at building '{b}', \
                 which that planet does not have"
            ));
        }
    }
}
