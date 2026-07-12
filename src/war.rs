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
/// A faction only trusts you with WAR work after this many completed
/// missions for them — standing alone can be bought with a few deliveries;
/// moving borders takes a record.
pub const WAR_SERVICE_MIN: u32 = 5;

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

/// The cheapest freighter a faction fields (CutSupply targets).
fn faction_freighter(iu: &ItemUniverse, faction: &str) -> Option<String> {
    iu.ships
        .iter()
        .filter(|(_, d)| {
            d.faction.as_deref() == Some(faction)
                && d.personality == crate::ship::Personality::Trader
        })
        .min_by_key(|(n, d)| (d.price, (*n).clone()))
        .map(|(n, _)| n.clone())
}

/// A landable planet in `system` (covert-op venue on the target side, the
/// drop-off on the sponsor side for extractions).
fn landable_planet(iu: &ItemUniverse, system: &str) -> Option<(String, String)> {
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
    service: Res<crate::standing::FactionServiceRecord>,
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
            // The sponsor has to KNOW you before the war desk talks:
            // completed missions for the faction, not just standing.
            if service.get(&front.sponsor) < WAR_SERVICE_MIN {
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
            if templates.is_empty() {
                continue;
            }
            // Start at a random template but fall through the rest: some
            // can't instantiate for some fronts (covert ops need a landable
            // enemy world, which empty buffer zones lack).
            let start = rng.gen_range(0..templates.len());
            for k in 0..templates.len() {
                let (tmpl_id, tmpl) = templates[(start + k) % templates.len()];
                let Some((def, follow_up)) =
                    instantiate_war_mission(tmpl, front, &iu, &galaxy, &mut rng)
                else {
                    continue;
                };
                // The counter is session-local, but ACTIVE war missions are
                // persisted with the pilot and re-enter the catalog on load —
                // a fresh session's counter would reuse their ids and
                // silently overwrite the player's active mission with a
                // different front's. Skip past anything that exists.
                *counter += 1;
                let mut id = format!("war__{}__{:04}", tmpl_id, *counter);
                while catalog.defs.contains_key(&id)
                    || catalog.defs.contains_key(&format!("{id}__return"))
                {
                    *counter += 1;
                    id = format!("war__{}__{:04}", tmpl_id, *counter);
                }
                if let Some(mut follow) = follow_up {
                    follow
                        .preconditions
                        .push(crate::missions::types::Precondition::Completed {
                            mission: id.clone(),
                        });
                    catalog.defs.insert(format!("{id}__return"), follow);
                }
                catalog.defs.insert(id.clone(), def);
                offers
                    .npc
                    .entry(planet.clone())
                    .or_default()
                    .push(id.clone());
                offers.considered.insert(id);
                offered += 1;
                break;
            }
        }
    }
}

/// Build a concrete mission from a War/Covert template for a front. Two-stage
/// covert actions (Extract, Propaganda) also return a follow-up def that the
/// generator files precondition-locked on the primary and auto-starting.
fn instantiate_war_mission(
    tmpl: &MissionTemplate,
    front: &Front,
    iu: &ItemUniverse,
    galaxy: &GalaxyControl,
    rng: &mut impl Rng,
) -> Option<(MissionDef, Option<MissionDef>)> {
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
            Some((
                MissionDef {
                    briefing: subst(briefing, &vars),
                    success_text: subst(success_text, &vars),
                    failure_text: subst(failure_text, &vars),
                    preconditions: Vec::new(),
                    offer: OfferKind::NpcOffer {
                        planet: String::new(), // placed directly into offers
                        weight: 1.0,
                        // Overt war work is OFFICIAL: the duty officer waits
                        // at the faction's garrison.
                        building: Some("garrison".to_string()),
                        approach: NpcApproach::Wait,
                        npc: None,
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
                    faction: None,
                },
                None,
            ))
        }
        MissionTemplate::Covert {
            briefing,
            success_text,
            failure_text,
            action,
            influence_delta,
            pay_range,
            enemy_standing_penalty,
            merchant_standing_penalty,
            stage2_briefing,
            stage2_success,
            ..
        } => {
            let (planet_id, planet_display) = landable_planet(iu, &front.target)?;
            let pay = rng.gen_range(pay_range.0..=pay_range.1.max(pay_range.0));
            let mut vars = vec![
                ("{faction}", front.sponsor.clone()),
                ("{enemy}", front.enemy.clone()),
                ("{system_display}", system_display),
                ("{planet_display}", planet_display),
                ("{pay}", pay.to_string()),
                ("{cost}", pay.abs().to_string()),
            ];
            // Two-stage actions split the influence shift: a taste on the
            // primary (also what keys open_war's one-per-front bookkeeping),
            // the bulk on the follow-up.
            let mut stage2: Option<(Objective, Vec<CompletionRequirement>)> = None;
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
                        hint: None,
                        planet: planet_id.clone(),
                        npc_name: npc_name.clone(),
                        building: None,
                        approach: NpcApproach::Seek,
                        npc: None,
                    },
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                ),
                CovertAction::CatchOfficial { npc_name } => (
                    Objective::CatchNpc {
                        hint: None,
                        planet: planet_id.clone(),
                        npc_name: npc_name.clone(),
                        building: None,
                        npc: None,
                    },
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                ),
                CovertAction::CutSupply { count } => {
                    let freighter = faction_freighter(iu, &front.enemy)?;
                    vars.push(("{count}", count.to_string()));
                    (
                        Objective::DestroyShips {
                            system: front.target.clone(),
                            ship_type: freighter,
                            count: *count,
                            target_name: format!("{} Supply Convoy", front.enemy),
                            hostile: false, // freighters run, they don't hunt
                            collect: None,
                        },
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                    )
                }
                CovertAction::Bribe { npc_name } => (
                    // The negative pay_range IS the bribe — Pay below charges it.
                    Objective::MeetNpc {
                        hint: None,
                        planet: planet_id.clone(),
                        npc_name: npc_name.clone(),
                        building: None,
                        approach: NpcApproach::Wait,
                        npc: None,
                    },
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                ),
                CovertAction::Extract { npc_name } => {
                    let (home_planet, home_display) = landable_planet(iu, &front.home)?;
                    vars.push(("{dest_display}", home_display));
                    stage2 = Some((
                        Objective::LandOnPlanet { planet: home_planet },
                        Vec::new(),
                    ));
                    (
                        Objective::MeetNpc {
                        hint: None,
                            planet: planet_id.clone(),
                            npc_name: npc_name.clone(),
                            building: None,
                            approach: NpcApproach::Seek,
                            npc: None,
                        },
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                    )
                }
                CovertAction::Propaganda {
                    commodity,
                    quantity,
                    npc_name,
                } => {
                    vars.push(("{commodity}", commodity.clone()));
                    vars.push(("{quantity}", quantity.to_string()));
                    stage2 = Some((
                        Objective::MeetNpc {
                        hint: None,
                            planet: planet_id.clone(),
                            npc_name: npc_name.clone(),
                            building: None,
                            approach: NpcApproach::Seek,
                            npc: None,
                        },
                        Vec::new(),
                    ));
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
            };
            // Payout, influence, and standing costs land on the FINAL stage;
            // a two-stage primary carries a quarter of the influence shift
            // (the defector agreeing to leave already hurts them).
            let mut final_effects = vec![
                CompletionEffect::Pay { credits: pay },
                CompletionEffect::ShiftInfluence {
                    system: front.target.clone(),
                    faction: front.sponsor.clone(),
                    delta: if stage2.is_some() {
                        influence_delta * 0.75
                    } else {
                        *influence_delta
                    },
                },
            ];
            if *enemy_standing_penalty > 0.0 {
                final_effects.push(CompletionEffect::AdjustStanding {
                    faction: front.enemy.clone(),
                    delta: -enemy_standing_penalty,
                });
            }
            if *merchant_standing_penalty > 0.0 {
                final_effects.push(CompletionEffect::AdjustStanding {
                    faction: "Merchant".to_string(),
                    delta: -merchant_standing_penalty,
                });
            }
            let (stage1_effects, follow_up) = match stage2 {
                None => {
                    effects.extend(final_effects);
                    (effects, None)
                }
                Some((objective2, requires2)) => {
                    effects.push(CompletionEffect::ShiftInfluence {
                        system: front.target.clone(),
                        faction: front.sponsor.clone(),
                        delta: influence_delta * 0.25,
                    });
                    let follow = MissionDef {
                        briefing: subst(stage2_briefing, &vars),
                        success_text: subst(stage2_success, &vars),
                        failure_text: subst(failure_text, &vars),
                        // Precondition on the primary is filled in by the
                        // generator, which knows the assigned mission id.
                        preconditions: Vec::new(),
                        offer: OfferKind::Auto,
                        start_effects: Vec::new(),
                        objective: objective2,
                        requires: requires2,
                        completion_effects: final_effects,
                        squadron: Vec::new(),
                        faction: None,
                    };
                    (effects, Some(follow))
                }
            };
            Some((
                MissionDef {
                    briefing: subst(briefing, &vars),
                    success_text: subst(success_text, &vars),
                    failure_text: subst(failure_text, &vars),
                    preconditions: Vec::new(),
                    offer: OfferKind::NpcOffer {
                        planet: String::new(),
                        weight: 1.0,
                        // Covert work is DENIABLE: a stranger sidles up in the
                        // bar, never anyone in a uniform.
                        building: Some("bar".to_string()),
                        approach: NpcApproach::Seek,
                        npc: None,
                    },
                    start_effects,
                    objective,
                    requires,
                    completion_effects: stage1_effects,
                    squadron: Vec::new(),
                    faction: None,
                },
                follow_up,
            ))
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
