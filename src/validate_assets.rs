//! Cross-reference validation for the item-universe asset files.
//!
//! Called once at startup after all YAML files have been loaded and derived
//! tables computed.  Every check emits a `warn!` so the developer sees the
//! problem in the console without crashing the game.

use bevy::prelude::*;
use std::collections::HashSet;

use crate::item_universe::{ItemUniverse, OutfitterItem};
use crate::missions::types::{
    CompletionEffect, CompletionRequirement, MissionTemplate, Objective, OfferKind,
    Precondition, StartEffect,
};

/// Run every validation pass and log warnings for anything broken.
pub fn validate(iu: &ItemUniverse) {
    validate_ship_weapon_space(iu);
    validate_uncolonized_planets(iu);
    validate_star_system_references(iu);
    validate_outfitter_items_reference_weapons(iu);
    validate_ship_base_weapons(iu);
    validate_weapon_carrier_bays(iu);
    validate_missions(iu);
    validate_mission_templates(iu);
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

/// Weapons with carrier_bay must reference a valid ship type.
fn validate_weapon_carrier_bays(iu: &ItemUniverse) {
    for (weapon_name, weapon) in &iu.weapons {
        if let Some(ref bay_ship) = weapon.carrier_bay {
            if !iu.ships.contains_key(bay_ship) {
                warn!(
                    "Weapon \"{weapon_name}\" has carrier_bay \"{bay_ship}\" \
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
            CompletionEffect::Pay { .. } | CompletionEffect::GrantUnlock { .. } => {}
        }
    }
}
