//! The galactic-war front generator (docs/galactic_war_design.md §4–5).
//!
//! A *front* is a jump-graph edge between systems whose live controllers are
//! enemies (per enemies.yaml). When the player lands in a front system held
//! by a faction they're in good standing with (≥ `min_standing` per
//! template), the generator instantiates war missions from the War/Covert
//! mission templates — ad-hoc, like arrests — **tiered by how contested the
//! target is**: lopsided fronts get raids and covert ops, fronts near the
//! control threshold get squadron battles and the decisive push. Completion
//! effects carry `ShiftInfluence`, so each front reads as a campaign that
//! the existing galaxy machinery resolves.
//!
//! An ambient drift nudges both sides of every active front on each landing,
//! so the war moves without the player; their missions are the decisive
//! strokes.

use bevy::prelude::*;
use rand::Rng;

use crate::PlayState;
use crate::galaxy::{CONTROL_GAIN, GalaxyControl};
use crate::item_universe::ItemUniverse;
use crate::missions::types::{
    CompletionEffect, CompletionRequirement, CovertAction, MissionDef, NpcApproach, Objective,
    OfferKind, StartEffect,
};
use crate::missions::{
    MissionCatalog, MissionLog, MissionOffers, MissionStatus, MissionTemplate,
    PlayerLandedOnPlanet,
};
use crate::standing::FactionStandings;

// ── Tuning ───────────────────────────────────────────────────────────────────

/// Ambient drift magnitude per landing per active front (both sides roll).
const DRIFT_PER_LANDING: f32 = 0.015;
/// Max war missions offered per landing.
const MAX_WAR_OFFERS: usize = 2;

// ── Fronts + tiers ───────────────────────────────────────────────────────────

/// One warring border: the sponsor holds `home`; `target` is the adjacent
/// contestable system the campaign is fought over — either held by an enemy
/// directly, or an UNALIGNED buffer zone that borders one (faction cores are
/// mostly separated by neutral space, so most wars start as a race for the
/// no-man's-land between them).
#[derive(Debug, Clone, PartialEq)]
pub struct Front {
    pub sponsor: String,
    pub home: String,
    pub enemy: String,
    pub target: String,
}

/// All active fronts under the live galaxy.
pub fn detect_fronts(iu: &ItemUniverse, galaxy: &GalaxyControl) -> Vec<Front> {
    let mut fronts = Vec::new();
    for (home, sys) in &iu.star_systems {
        let Some(sponsor) = galaxy.controller(home) else {
            continue;
        };
        let enemies = iu.enemies.get(sponsor).cloned().unwrap_or_default();
        for target in &sys.connections {
            if !iu
                .star_systems
                .get(target)
                .map(|s| s.contestable)
                .unwrap_or(false)
            {
                continue;
            }
            let enemy = match galaxy.controller(target) {
                // Direct border: the enemy holds the target.
                Some(holder) if enemies.contains(&holder.to_string()) => holder.to_string(),
                Some(_) => continue, // friendly/neutral-held: no front
                // Buffer zone: uncontrolled, but an enemy sits on its far side.
                None => {
                    let Some(far_enemy) = iu
                        .star_systems
                        .get(target)
                        .into_iter()
                        .flat_map(|s| s.connections.iter())
                        .filter_map(|n| galaxy.controller(n))
                        .find(|f| enemies.contains(&f.to_string()))
                    else {
                        continue;
                    };
                    far_enemy.to_string()
                }
            };
            fronts.push(Front {
                sponsor: sponsor.to_string(),
                home: home.clone(),
                enemy,
                target: target.clone(),
            });
        }
    }
    fronts
}

/// Campaign tier from the SPONSOR's progress on the target (works for both
/// enemy-held systems and empty buffer zones):
/// 1 = no foothold (raids, covert ops), 2 = a contested push (battles),
/// 3 = the threshold in sight (the decisive push).
pub fn front_tier(galaxy: &GalaxyControl, front: &Front) -> u8 {
    let sponsor_share = galaxy.influence_of(&front.target, &front.sponsor);
    if sponsor_share < 0.25 {
        1
    } else if sponsor_share < 0.45 {
        2
    } else {
        3
    }
}

/// The cheapest fighters a faction fields, up to `max_tech` (battle targets
/// and squadron wings). Sorted for determinism.
fn faction_fighters(iu: &ItemUniverse, faction: &str, max_tech: u8) -> Vec<String> {
    let mut ships: Vec<(&String, u8, i128)> = iu
        .ships
        .iter()
        .filter(|(_, d)| {
            d.faction.as_deref() == Some(faction)
                && d.personality == crate::ship::Personality::Fighter
                && d.tech_level <= max_tech
        })
        .map(|(n, d)| (n, d.tech_level, d.price))
        .collect();
    ships.sort_by_key(|(_, t, p)| (*t, *p));
    ships.into_iter().map(|(n, _, _)| n.clone()).collect()
}

/// A landable enemy planet in the target system (covert-op venue).
fn enemy_planet(iu: &ItemUniverse, system: &str) -> Option<(String, String)> {
    let sys = iu.star_systems.get(system)?;
    let mut planets: Vec<(&String, &crate::planets::PlanetData)> = sys
        .planets
        .iter()
        .filter(|(_, p)| !p.commodities.is_empty())
        .collect();
    planets.sort_by_key(|(n, _)| (*n).clone());
    planets
        .first()
        .map(|(n, p)| ((*n).clone(), p.display_name.clone()))
}

fn subst(s: &str, vars: &[(&str, String)]) -> String {
    let mut out = s.to_string();
    for (k, v) in vars {
        out = out.replace(k, v);
    }
    out
}

// ── Systems ──────────────────────────────────────────────────────────────────

/// Offer war missions when the player lands in a front system whose sponsor
/// they're in good standing with. Runs after the normal offer roll (which
/// resets the planet's offer lists on landing).
#[allow(clippy::too_many_arguments)]
fn offer_war_missions(
    mut reader: MessageReader<PlayerLandedOnPlanet>,
    iu: Res<ItemUniverse>,
    galaxy: Res<GalaxyControl>,
    standings: Res<FactionStandings>,
    mut catalog: ResMut<MissionCatalog>,
    log: Res<MissionLog>,
    mut offers: ResMut<MissionOffers>,
    mut counter: Local<u32>,
) {
    let mut rng = rand::thread_rng();
    for PlayerLandedOnPlanet { planet } in reader.read() {
        let Some((system, _)) = iu.find_gameplay_planet(planet) else {
            continue;
        };
        let system = system.to_string();
        // Fronts this landing can hire for: sponsor holds the landing system.
        let fronts: Vec<Front> = detect_fronts(&iu, &galaxy)
            .into_iter()
            .filter(|f| f.home == system)
            .collect();
        if fronts.is_empty() {
            continue;
        }
        // One open war mission per front at a time.
        let open_war = |catalog: &MissionCatalog, front: &Front| {
            catalog.defs.iter().any(|(id, def)| {
                id.starts_with("war__")
                    && def.shift_target() == Some(front.target.as_str())
                    && matches!(log.status(id), MissionStatus::Active(_) | MissionStatus::Available)
            })
        };
        let mut offered = 0usize;
        for front in &fronts {
            if offered >= MAX_WAR_OFFERS || open_war(&catalog, front) {
                continue;
            }
            let tier = front_tier(&galaxy, front);
            // Candidate templates for this tier, standing permitting.
            let templates: Vec<(&String, &MissionTemplate)> = iu
                .mission_templates
                .iter()
                .filter(|(_, t)| match t {
                    MissionTemplate::War {
                        tier: t_tier,
                        min_standing,
                        ..
                    }
                    | MissionTemplate::Covert {
                        tier: t_tier,
                        min_standing,
                        ..
                    } => *t_tier == tier && standings.get(&front.sponsor) >= *min_standing,
                    _ => false,
                })
                .collect();
            let mut templates = templates;
            templates.sort_by_key(|(id, _)| (*id).clone());
            let Some((tmpl_id, tmpl)) = templates
                .get(rng.gen_range(0..templates.len().max(1)))
                .copied()
            else {
                continue;
            };
            *counter += 1;
            let id = format!("war__{}__{:04}", tmpl_id, *counter);
            if let Some(def) = instantiate_war_mission(tmpl, front, &iu, &galaxy, &mut rng) {
                catalog.defs.insert(id.clone(), def);
                offers
                    .npc
                    .entry(planet.clone())
                    .or_default()
                    .push(id.clone());
                offers.considered.insert(id);
                offered += 1;
            }
        }
    }
}

/// Build a concrete mission from a War/Covert template for a front.
fn instantiate_war_mission(
    tmpl: &MissionTemplate,
    front: &Front,
    iu: &ItemUniverse,
    galaxy: &GalaxyControl,
    rng: &mut impl Rng,
) -> Option<MissionDef> {
    let tier = front_tier(galaxy, front);
    let system_display = iu
        .star_systems
        .get(&front.target)
        .map(|s| s.display_name.clone())
        .unwrap_or_else(|| front.target.clone());
    match tmpl {
        MissionTemplate::War {
            briefing,
            success_text,
            failure_text,
            attack,
            squadron_size,
            count_range,
            influence_delta,
            pay_range,
            ..
        } => {
            // Battles are fought where the influence moves: the contestable
            // target for attacks, and (only if contestable) the sponsor's own
            // system for defenses.
            let battle_system = if *attack {
                front.target.clone()
            } else {
                if !iu
                    .star_systems
                    .get(&front.home)
                    .map(|s| s.contestable)
                    .unwrap_or(false)
                {
                    return None;
                }
                front.home.clone()
            };
            let shift_system = battle_system.clone();
            // Invaders/defenders: the enemy's fighters, heavier at higher tiers.
            let hostiles = faction_fighters(iu, &front.enemy, 1 + tier);
            let ship_type = hostiles.last()?.clone();
            let count = rng.gen_range(count_range.0..=count_range.1.max(count_range.0));
            let pay = rng.gen_range(pay_range.0..=pay_range.1.max(pay_range.0));
            let squadron = faction_fighters(iu, &front.sponsor, 2)
                .first()
                .map(|s| vec![s.clone(); *squadron_size as usize])
                .unwrap_or_default();
            let vars = [
                ("{faction}", front.sponsor.clone()),
                ("{enemy}", front.enemy.clone()),
                ("{system_display}", system_display),
                ("{count}", count.to_string()),
                ("{pay}", pay.to_string()),
            ];
            Some(MissionDef {
                briefing: subst(briefing, &vars),
                success_text: subst(success_text, &vars),
                failure_text: subst(failure_text, &vars),
                preconditions: Vec::new(),
                offer: OfferKind::NpcOffer {
                    planet: String::new(), // placed directly into offers
                    weight: 1.0,
                    building: Some("bar".to_string()),
                    approach: NpcApproach::Seek,
                },
                start_effects: Vec::new(),
                objective: Objective::DestroyShips {
                    system: battle_system,
                    ship_type,
                    count,
                    target_name: format!("{} Warfleet", front.enemy),
                    hostile: true,
                    collect: None,
                },
                requires: Vec::new(),
                completion_effects: vec![
                    CompletionEffect::Pay { credits: pay },
                    CompletionEffect::ShiftInfluence {
                        system: shift_system,
                        faction: front.sponsor.clone(),
                        delta: *influence_delta,
                    },
                ],
                squadron,
            })
        }
        MissionTemplate::Covert {
            briefing,
            success_text,
            failure_text,
            action,
            influence_delta,
            pay_range,
            enemy_standing_penalty,
            ..
        } => {
            let (planet_id, planet_display) = enemy_planet(iu, &front.target)?;
            let pay = rng.gen_range(pay_range.0..=pay_range.1.max(pay_range.0));
            let mut vars = vec![
                ("{faction}", front.sponsor.clone()),
                ("{enemy}", front.enemy.clone()),
                ("{system_display}", system_display),
                ("{planet_display}", planet_display),
                ("{pay}", pay.to_string()),
            ];
            let (objective, start_effects, requires, mut effects): (
                Objective,
                Vec<StartEffect>,
                Vec<CompletionRequirement>,
                Vec<CompletionEffect>,
            ) = match action {
                CovertAction::Smuggle {
                    commodity,
                    quantity,
                } => {
                    vars.push(("{commodity}", commodity.clone()));
                    vars.push(("{quantity}", quantity.to_string()));
                    (
                        Objective::LandOnPlanet {
                            planet: planet_id.clone(),
                        },
                        vec![StartEffect::LoadCargo {
                            commodity: commodity.clone(),
                            quantity: *quantity,
                            reserved: true,
                        }],
                        vec![CompletionRequirement::HasCargo {
                            commodity: commodity.clone(),
                            quantity: *quantity,
                        }],
                        vec![CompletionEffect::RemoveCargo {
                            commodity: commodity.clone(),
                            quantity: *quantity,
                        }],
                    )
                }
                CovertAction::MeetContact { npc_name } => (
                    Objective::MeetNpc {
                        planet: planet_id.clone(),
                        npc_name: npc_name.clone(),
                        building: None,
                        approach: NpcApproach::Seek,
                    },
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                ),
                CovertAction::CatchOfficial { npc_name } => (
                    Objective::CatchNpc {
                        planet: planet_id.clone(),
                        npc_name: npc_name.clone(),
                        building: None,
                    },
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                ),
            };
            effects.push(CompletionEffect::Pay { credits: pay });
            effects.push(CompletionEffect::ShiftInfluence {
                system: front.target.clone(),
                faction: front.sponsor.clone(),
                delta: *influence_delta,
            });
            if *enemy_standing_penalty > 0.0 {
                effects.push(CompletionEffect::AdjustStanding {
                    faction: front.enemy.clone(),
                    delta: -enemy_standing_penalty,
                });
            }
            Some(MissionDef {
                briefing: subst(briefing, &vars),
                success_text: subst(success_text, &vars),
                failure_text: subst(failure_text, &vars),
                preconditions: Vec::new(),
                offer: OfferKind::NpcOffer {
                    planet: String::new(),
                    weight: 1.0,
                    building: Some("bar".to_string()),
                    approach: NpcApproach::Wait,
                },
                start_effects,
                objective,
                requires,
                completion_effects: effects,
                squadron: Vec::new(),
            })
        }
        _ => None,
    }
}

/// The war moves without the player: on each landing, every active front's
/// contested target drifts a little for BOTH sides.
fn war_drift(
    mut reader: MessageReader<PlayerLandedOnPlanet>,
    iu: Res<ItemUniverse>,
    mut galaxy: ResMut<GalaxyControl>,
) {
    let mut rng = rand::thread_rng();
    let mut landings = 0;
    for _ in reader.read() {
        landings += 1;
    }
    if landings == 0 {
        return;
    }
    let fronts = detect_fronts(&iu, &galaxy);
    for front in fronts {
        for faction in [&front.sponsor, &front.enemy] {
            let delta = rng.gen_range(0.0..DRIFT_PER_LANDING) * landings as f32;
            galaxy.apply_shift(&front.target, faction, delta);
        }
    }
}

pub fn war_plugin(app: &mut App) {
    app.add_systems(
        Update,
        (
            offer_war_missions.after(crate::missions::progress::roll_offers_on_land),
            war_drift,
        )
            .run_if(not(in_state(PlayState::MainMenu))),
    );
}

#[cfg(test)]
#[path = "tests/war_tests.rs"]
mod tests;
