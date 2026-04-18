//! Communications channel — drives the HUD ticker based on game state.
//!
//! Watches the player's nav_target and posts context-appropriate messages
//! to [`CommsChannel`](crate::hud::CommsChannel).  Ship messages vary by
//! personality, distress state, nav/weapons targets, and cargo.

use bevy::prelude::*;

use crate::asteroids::Asteroid;
use crate::hud::CommsChannel;
use crate::item_universe::ItemUniverse;
use crate::pickups::Pickup;
use crate::planets::Planet;
use crate::ship::{Distressed, Personality, ShipHostility, Target};
use crate::{CurrentStarSystem, PlayState, Player, Ship};

pub fn comms_plugin(app: &mut App) {
    app.add_systems(
        Update,
        update_comms_from_nav_target.run_if(in_state(PlayState::Flying)),
    );
}

// ---------------------------------------------------------------------------
// Nav-target change detection
// ---------------------------------------------------------------------------

#[derive(Resource, Default)]
struct PrevNavTarget(Option<Entity>);

fn update_comms_from_nav_target(
    mut comms: ResMut<CommsChannel>,
    mut prev: Local<PrevNavTarget>,
    player_query: Query<(&Ship, &ShipHostility), With<Player>>,
    planets_query: Query<&Planet>,
    ships_query: Query<(&Ship, &Distressed, &ShipHostility), Without<Player>>,
    asteroids_query: Query<&Asteroid>,
    pickups_query: Query<&Pickup>,
    item_universe: Res<ItemUniverse>,
    current_system: Res<CurrentStarSystem>,
    time: Res<Time>,
) {
    let Ok((player_ship, player_hostility)) = player_query.single() else {
        return;
    };

    let current_entity = player_ship.nav_target.as_ref().map(|t| t.get_entity());

    let target_changed = current_entity != prev.0;
    if !target_changed && !comms.cycle_complete {
        return;
    }
    prev.0 = current_entity;

    let Some(target) = &player_ship.nav_target else {
        comms.send("");
        return;
    };

    let entity = target.get_entity();

    match target {
        Target::Planet(_) => {
            let Ok(planet) = planets_query.get(entity) else {
                comms.send("");
                return;
            };
            let msg = planet_message(
                &planet.0,
                player_hostility,
                &item_universe,
                &current_system.0,
            );
            comms.send(msg);
        }
        Target::Ship(_) => {
            let Ok((ship, distressed, hostility)) = ships_query.get(entity) else {
                comms.send("");
                return;
            };
            let msg = ship_message(
                entity,
                ship,
                distressed,
                hostility,
                &ships_query,
                &planets_query,
                &asteroids_query,
                &pickups_query,
                &item_universe,
                &current_system.0,
                &time,
            );
            comms.send(msg);
        }
        _ => {
            comms.send("");
        }
    }
}

// ---------------------------------------------------------------------------
// Planet messages (unchanged logic, extracted)
// ---------------------------------------------------------------------------

fn planet_message(
    planet_key: &str,
    player_hostility: &ShipHostility,
    iu: &ItemUniverse,
    system: &str,
) -> String {
    let Some(sys) = iu.star_systems.get(system) else {
        return String::new();
    };
    let Some(pd) = sys.planets.get(planet_key) else {
        return String::new();
    };
    let name = &pd.display_name;

    if pd.uncolonized {
        format!("{name}: No colony detected.")
    } else if !pd.faction.is_empty()
        && player_hostility.0.get(&pd.faction).copied().unwrap_or(0.0) > 0.0
    {
        format!("{name}: Docking access denied.")
    } else {
        format!("Welcome to {name}. Press L to land.")
    }
}

// ---------------------------------------------------------------------------
// Ship messages
// ---------------------------------------------------------------------------

/// Seed a deterministic-but-varied index from an entity and the current second.
/// The message stays stable for ~4 seconds so it doesn't flicker.
fn msg_index(entity: Entity, time: &Time, pool_len: usize) -> usize {
    if pool_len == 0 {
        return 0;
    }
    let coarse_time = (time.elapsed_secs() / 4.0) as u32;
    let idx = entity.to_bits() as u32;
    let hash = idx.wrapping_mul(2654435761).wrapping_add(coarse_time);
    hash as usize % pool_len
}

fn pick<'a>(pool: &[&'a str], entity: Entity, time: &Time) -> &'a str {
    pool[msg_index(entity, time, pool.len())]
}

/// Resolve a Target to a human-readable description, or None.
fn describe_target(
    target: &Option<Target>,
    ships: &Query<(&Ship, &Distressed, &ShipHostility), Without<Player>>,
    planets: &Query<&Planet>,
    _asteroids: &Query<&Asteroid>,
    pickups: &Query<&Pickup>,
    iu: &ItemUniverse,
    system: &str,
) -> Option<String> {
    let t = target.as_ref()?;
    match t {
        Target::Ship(e) => ships
            .get(*e)
            .ok()
            .map(|(s, _, _)| s.data.display_name.clone()),
        Target::Planet(e) => planets.get(*e).ok().and_then(|p| {
            iu.star_systems
                .get(system)
                .and_then(|sys| sys.planets.get(&p.0))
                .map(|pd| pd.display_name.clone())
        }),
        Target::Asteroid(_) => Some("an asteroid".into()),
        Target::Pickup(e) => pickups.get(*e).ok().map(|p| {
            let name = iu
                .commodities
                .get(&p.commodity)
                .map(|c| c.display_name.as_str())
                .unwrap_or(&p.commodity);
            format!("{} cargo", name)
        }),
    }
}

/// Look up the most valuable commodity this ship is carrying, by display name.
fn primary_cargo(ship: &Ship, iu: &ItemUniverse) -> Option<String> {
    ship.cargo
        .iter()
        .max_by_key(|(_, qty)| **qty)
        .map(|(commodity, _)| {
            iu.commodities
                .get(commodity)
                .map(|c| c.display_name.clone())
                .unwrap_or_else(|| commodity.clone())
        })
}

/// Best commodity to buy at a given planet (looked up from precomputed maps).
#[allow(dead_code)]
fn best_buy_at_planet(planet_key: &str, iu: &ItemUniverse, system: &str) -> Option<String> {
    iu.system_planet_best_commodity_to_buy
        .get(system)
        .and_then(|m| m.get(planet_key))
        .and_then(|commodity| {
            iu.commodities
                .get(commodity)
                .map(|c| c.display_name.clone())
        })
}

/// Best planet to sell a commodity (looked up from precomputed maps).
#[allow(dead_code)]
fn best_sell_planet(commodity: &str, iu: &ItemUniverse, system: &str) -> Option<String> {
    iu.system_commodity_best_planet_to_sell
        .get(system)
        .and_then(|m| m.get(commodity))
        .and_then(|planet_key| {
            iu.star_systems
                .get(system)
                .and_then(|sys| sys.planets.get(planet_key))
                .map(|pd| pd.display_name.clone())
        })
}

#[allow(clippy::too_many_arguments)]
fn ship_message(
    entity: Entity,
    ship: &Ship,
    distressed: &Distressed,
    _hostility: &ShipHostility,
    ships: &Query<(&Ship, &Distressed, &ShipHostility), Without<Player>>,
    planets: &Query<&Planet>,
    asteroids: &Query<&Asteroid>,
    pickups: &Query<&Pickup>,
    iu: &ItemUniverse,
    system: &str,
    time: &Time,
) -> String {
    let is_distressed = distressed.level > 0.15;

    let nav_desc = describe_target(
        &ship.nav_target,
        ships,
        planets,
        asteroids,
        pickups,
        iu,
        system,
    );
    let wpn_desc = describe_target(
        &ship.weapons_target,
        ships,
        planets,
        asteroids,
        pickups,
        iu,
        system,
    );

    // ── Distressed (any personality) ────────────────────────────────────
    if is_distressed {
        if let Some(ref attacker) = wpn_desc {
            let pool: &[&str] = match ship.data.personality {
                Personality::Trader => &[
                    "Mayday! {attacker} is firing on us! Any ships nearby?",
                    "We're under attack by {attacker}! Need assistance!",
                    "Taking heavy fire from {attacker}! Cargo isn't worth dying for!",
                ],
                Personality::Fighter => &[
                    "Engaged with {attacker}: could use some backup!",
                    "{attacker} is putting up a fight. Shields failing!",
                    "In combat with {attacker}. Situation critical!",
                ],
                Personality::Miner => &[
                    "Help! {attacker} jumped us while mining! We're unarmed!",
                    "Mayday! {attacker} attacking our mining vessel!",
                    "{attacker} is on us! Dropping cargo and running!",
                ],
            };
            return pick(pool, entity, time).replace("{attacker}", attacker);
        }

        if let Some(ref dest) = nav_desc {
            let pool: &[&str] = match ship.data.personality {
                Personality::Trader => &[
                    "Hull critical: limping to {dest}. Stay clear.",
                    "Barely holding together. Making for {dest}.",
                ],
                Personality::Fighter => &[
                    "Pulling back to {dest}. Shields are gone.",
                    "Disengaging: headed for {dest} for repairs.",
                ],
                Personality::Miner => &[
                    "Damaged! Heading to {dest} for emergency repairs.",
                    "Hull breach! Making best speed to {dest}.",
                ],
            };
            return pick(pool, entity, time).replace("{dest}", dest);
        }

        let pool = &[
            "Mayday, mayday! Any ships: we need help!",
            "Hull integrity failing! Requesting immediate assistance!",
        ];
        return pick(pool, entity, time).to_string();
    }

    // ── Not distressed — personality-specific messages ───────────────────
    match ship.data.personality {
        Personality::Trader => trader_message(entity, ship, nav_desc, wpn_desc, iu, system, time),
        Personality::Fighter => fighter_message(entity, ship, nav_desc, wpn_desc, iu, system, time),
        Personality::Miner => miner_message(entity, ship, nav_desc, wpn_desc, iu, system, time),
    }
}

fn trader_message(
    entity: Entity,
    ship: &Ship,
    nav_desc: Option<String>,
    wpn_desc: Option<String>,
    iu: &ItemUniverse,
    _system: &str,
    time: &Time,
) -> String {
    let name = &ship.data.display_name;
    let cargo_full = ship.remaining_cargo_space() == 0;
    let cargo_empty = ship.cargo.is_empty();

    // Has a weapons target — worried about hostiles
    if let Some(ref hostile) = wpn_desc {
        let pool = &[
            "Keep your distance: {hostile} has been causing trouble around here.",
            "Watch out for {hostile}. They've been aggressive today.",
            "Heads up: {hostile} in the area. We're keeping weapons hot.",
        ];
        return pick(pool, entity, time).replace("{hostile}", hostile);
    }

    // Heading to a planet
    if let Some(ref dest) = nav_desc {
        if matches!(&ship.nav_target, Some(Target::Planet(_))) {
            if cargo_full {
                if let Some(cargo_name) = primary_cargo(ship, iu) {
                    let pool = &[
                        "Hold's packed with {cargo}. {dest} better have good prices.",
                        "Full load of {cargo}: headed to {dest} to sell.",
                        "Running {cargo} to {dest}. Margins look decent.",
                    ];
                    return pick(pool, entity, time)
                        .replace("{cargo}", &cargo_name)
                        .replace("{dest}", dest);
                }
                let pool = &[
                    "Hold's full. Making a run to {dest}.",
                    "Cargo bay's bursting: off to {dest} to sell.",
                ];
                return pick(pool, entity, time).replace("{dest}", dest);
            }

            if cargo_empty {
                let pool = &[
                    "Heading to {dest} to pick up a load.",
                    "Empty hold. {dest} should have something worth buying.",
                    "Making for {dest}: time to restock.",
                ];
                return pick(pool, entity, time).replace("{dest}", dest);
            }

            // Partial cargo
            let pool = &[
                "En route to {dest}. Still have room in the hold.",
                "Headed to {dest} for another load.",
            ];
            return pick(pool, entity, time).replace("{dest}", dest);
        }
    }

    // Fallback
    let pool = &[
        "{name}: Just making a living out here.",
        "{name}: Quiet run so far. Hope it stays that way.",
        "{name}: Clear skies and steady credits.",
    ];
    pick(pool, entity, time).replace("{name}", name)
}

fn fighter_message(
    entity: Entity,
    ship: &Ship,
    nav_desc: Option<String>,
    wpn_desc: Option<String>,
    _iu: &ItemUniverse,
    _system: &str,
    time: &Time,
) -> String {
    let name = &ship.data.display_name;
    let faction = ship.data.faction.as_deref().unwrap_or("patrol");

    // Engaging a target
    if let Some(ref target_name) = wpn_desc {
        let pool = &[
            "Engaging {target}. Stay out of the firing line.",
            "{target} in our sights. Moving to intercept.",
            "Weapons free on {target}. All units be advised.",
            "Closing on {target}. This won't take long.",
        ];
        return pick(pool, entity, time).replace("{target}", target_name);
    }

    // Heading somewhere
    if let Some(ref dest) = nav_desc {
        let pool = &[
            "{faction} patrol: en route to {dest}.",
            "Heading to {dest}: sector looks clear.",
            "{faction} vessel on patrol. Next stop: {dest}.",
        ];
        return pick(pool, entity, time)
            .replace("{dest}", dest)
            .replace("{faction}", faction);
    }

    // Idle
    let pool = &[
        "{faction} patrol. All clear in this sector.",
        "{name} on station. Nothing to report.",
        "{faction} vessel standing by.",
    ];
    pick(pool, entity, time)
        .replace("{faction}", faction)
        .replace("{name}", name)
}

fn miner_message(
    entity: Entity,
    ship: &Ship,
    nav_desc: Option<String>,
    wpn_desc: Option<String>,
    iu: &ItemUniverse,
    _system: &str,
    time: &Time,
) -> String {
    let name = &ship.data.display_name;

    // Under threat
    if let Some(ref hostile) = wpn_desc {
        let pool = &[
            "{hostile} getting too close. We're just miners!",
            "Keeping an eye on {hostile}. Hope they move along.",
        ];
        return pick(pool, entity, time).replace("{hostile}", hostile);
    }

    // Mining an asteroid
    if matches!(&ship.nav_target, Some(Target::Asteroid(_))) {
        if let Some(cargo_name) = primary_cargo(ship, iu) {
            let pool = &[
                "Mining ops underway. Good {cargo} yields today.",
                "Breaking rock: finding decent {cargo} deposits.",
                "Lasers hot on this asteroid. {cargo} readings look promising.",
            ];
            return pick(pool, entity, time).replace("{cargo}", &cargo_name);
        }
        let pool = &[
            "Scanning this rock. Looks promising.",
            "Mining ops underway. Steady work out here.",
        ];
        return pick(pool, entity, time).to_string();
    }

    // Collecting a pickup
    if let Some(Target::Pickup(_)) = &ship.nav_target {
        let pool = &[
            "Cargo detected: moving to collect.",
            "Pickup on sensors. Adjusting course.",
        ];
        return pick(pool, entity, time).to_string();
    }

    // Heading to a planet (probably to sell)
    if let Some(ref dest) = nav_desc {
        if !ship.cargo.is_empty() {
            if let Some(cargo_name) = primary_cargo(ship, iu) {
                let pool = &[
                    "Hold's got a good haul of {cargo}. Heading to {dest} to sell.",
                    "Hauling {cargo} to {dest}. Not bad for a day's work.",
                ];
                return pick(pool, entity, time)
                    .replace("{cargo}", &cargo_name)
                    .replace("{dest}", dest);
            }
        }
        let pool = &[
            "Headed back to {dest}. Need to refuel.",
            "Making for {dest}. Time to sell what we've got.",
        ];
        return pick(pool, entity, time).replace("{dest}", dest);
    }

    // Fallback
    let pool = &[
        "{name}: Prospecting. Quiet out here.",
        "{name}: Just another day in the belt.",
    ];
    pick(pool, entity, time).replace("{name}", name)
}
