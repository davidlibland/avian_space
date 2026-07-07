//! Signed faction standing for the player.
//!
//! Standing is a number in [-100, 100] per faction (0 = neutral, unknown
//! factions are 0). It moves in response to the same gameplay messages the
//! rest of the game already emits — no other module needs to know about it:
//!
//!   * `ScoreHit` (intentional, player-sourced): hitting a faction's ship
//!     lowers standing with that faction; hitting a faction's ENEMY inside a
//!     system that faction controls raises it a little.
//!   * `MissionCompleted`: finishing a mission offered on a faction's planet
//!     raises standing with that faction; missions can also carry explicit
//!     `CompletionEffect::AdjustStanding` effects.
//!   * `PlayerLandedOnPlanet`: landing on a faction's planet with standing at
//!     or below the arrest threshold generates a procedural "arrest" mission —
//!     enforcers seek the player out on the surface, and settling with them
//!     (a MeetNpc objective) levies a fine and restores standing to just
//!     above hostile.
//!
//! Consequences are DERIVED from standing (per the ECS paradigm):
//!   * standing < 0        → price markup at that faction's planets,
//!   * standing ≤ ENGAGE   → the faction's ships engage the player (via the
//!                           existing `ShipHostility` machinery),
//!   * standing ≤ ARREST   → arrest mission on landing.

use bevy::prelude::*;
use std::collections::HashMap;

use crate::item_universe::ItemUniverse;
use crate::missions::types::{CompletionEffect, MissionDef, NpcApproach, Objective, OfferKind};
use crate::missions::{
    MissionCatalog, MissionCompleted, MissionLog, MissionStatus, PlayerLandedOnPlanet,
};
use crate::ship::{ScoreHit, Ship, ShipHostility};
use crate::{CurrentStarSystem, PlayState, Player};

// ── Tuning ───────────────────────────────────────────────────────────────────

/// Standing lost per intentional hit on a faction's ship.
pub const HIT_PENALTY: f32 = 2.0;
/// Standing gained with a system's controlling faction per intentional hit on
/// one of that faction's enemies inside the system.
pub const ENEMY_HIT_BONUS: f32 = 0.5;
/// Standing gained with the offering planet's faction on mission completion.
pub const MISSION_BONUS: f32 = 6.0;
/// At or below this, the faction's ships engage the player on sight.
pub const ENGAGE_THRESHOLD: f32 = -10.0;
/// At or below this, landing on the faction's planets triggers an arrest.
pub const ARREST_THRESHOLD: f32 = -40.0;
/// Paying the arrest fine restores standing to this value (just above the
/// engage threshold — you bought your way back to "watched, not shot at").
pub const POST_ARREST_STANDING: f32 = -5.0;

// ── Resource ─────────────────────────────────────────────────────────────────

/// The player's signed standing per faction. Empty map = neutral everywhere.
#[derive(Resource, Default, Clone)]
pub struct FactionStandings(pub HashMap<String, f32>);

impl FactionStandings {
    pub fn get(&self, faction: &str) -> f32 {
        self.0.get(faction).copied().unwrap_or(0.0)
    }
    pub fn adjust(&mut self, faction: &str, delta: f32) {
        let v = self.0.entry(faction.to_string()).or_insert(0.0);
        *v = (*v + delta).clamp(-100.0, 100.0);
    }
}

impl crate::session::SessionResource for FactionStandings {
    type SaveData = HashMap<String, f32>;
    fn new_session(_: &ItemUniverse) -> Self {
        Self::default()
    }
    fn to_save(&self) -> Self::SaveData {
        self.0.clone()
    }
    fn from_save(data: Self::SaveData, _: &ItemUniverse) -> Self {
        Self(data)
    }
}

// ── Derived values ───────────────────────────────────────────────────────────

/// Price multiplier at a faction's planets: 1.0 at neutral-or-better, rising
/// linearly to 1.75× at standing −100.
pub fn price_markup(standing: f32) -> f32 {
    if standing >= 0.0 {
        1.0
    } else {
        1.0 + (-standing / 100.0) * 0.75
    }
}

/// Markup applied to a buy price (rounded up — the house never rounds in
/// your favor).
pub fn markup_price(price: i128, markup: f32) -> i128 {
    if markup <= 1.0 {
        price
    } else {
        (price as f64 * markup as f64).ceil() as i128
    }
}

/// The faction a planet answers to, if it's a real faction the standing
/// system tracks (Independent worlds don't care who you've angered).
pub fn planet_faction(planet: &crate::planets::PlanetData) -> Option<&str> {
    match planet.faction.as_str() {
        "" | "Independent" => None,
        f => Some(f),
    }
}

/// The faction controlling a star system: the most common tracked faction
/// among its planets. None for unclaimed / fully independent systems.
pub fn controlling_faction(iu: &ItemUniverse, system: &str) -> Option<String> {
    let sys = iu.star_systems.get(system)?;
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for planet in sys.planets.values() {
        if let Some(f) = planet_faction(planet) {
            *counts.entry(f).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .max_by_key(|(_, n)| *n)
        .map(|(f, _)| f.to_string())
}

// ── Systems ──────────────────────────────────────────────────────────────────

/// Standing consequences of the player's intentional weapon hits.
fn standing_on_hits(
    mut reader: MessageReader<ScoreHit>,
    player: Query<Entity, With<Player>>,
    ships: Query<&Ship>,
    mut standings: ResMut<FactionStandings>,
    current_system: Res<CurrentStarSystem>,
    iu: Res<ItemUniverse>,
) {
    let Ok(player_entity) = player.single() else {
        return;
    };
    for event in reader.read() {
        let ScoreHit::OnShip { source, target } = event else {
            continue;
        };
        if *source != player_entity {
            continue;
        }
        // Intentional hits only — same rule as faction-hostility contagion.
        let on_target = ships
            .get(*source)
            .ok()
            .and_then(|s| s.weapons_target.as_ref())
            .map(|t| t.get_entity() == *target)
            .unwrap_or(false);
        if !on_target {
            continue;
        }
        let Some(victim_faction) = ships.get(*target).ok().and_then(|t| t.data.faction.clone())
        else {
            continue;
        };
        standings.adjust(&victim_faction, -HIT_PENALTY);

        // Helping the local power fight its enemies earns a little goodwill.
        if let Some(controller) = controlling_faction(&iu, &current_system.0) {
            if controller != victim_faction
                && iu
                    .enemies
                    .get(&controller)
                    .is_some_and(|es| es.contains(&victim_faction))
            {
                standings.adjust(&controller, ENEMY_HIT_BONUS);
            }
        }
    }
}

/// Standing rewards for completed missions: an implicit bonus with the
/// faction whose planet offered the mission, plus any explicit
/// `AdjustStanding` completion effects the mission carries.
fn standing_on_mission_complete(
    mut reader: MessageReader<MissionCompleted>,
    catalog: Res<MissionCatalog>,
    iu: Res<ItemUniverse>,
    mut standings: ResMut<FactionStandings>,
) {
    for MissionCompleted(id) in reader.read() {
        let Some(def) = catalog.defs.get(id) else {
            continue;
        };
        if let OfferKind::NpcOffer { planet, .. } = &def.offer {
            let faction = iu
                .star_systems
                .values()
                .find_map(|s| s.planets.get(planet))
                .and_then(|p| planet_faction(p).map(String::from));
            if let Some(f) = faction {
                standings.adjust(&f, MISSION_BONUS);
            }
        }
        for effect in &def.completion_effects {
            if let CompletionEffect::AdjustStanding { faction, delta } = effect {
                standings.adjust(faction, *delta);
            }
        }
    }
}

/// Derive the player's `ShipHostility` from standing: factions at or below
/// the engage threshold hunt the player via the existing engagement
/// machinery; recovering standing (missions, fines) calls them off.
fn derive_player_hostility(
    standings: Res<FactionStandings>,
    mut player: Query<&mut ShipHostility, With<Player>>,
) {
    let Ok(mut hostility) = player.single_mut() else {
        return;
    };
    let derived: HashMap<String, f32> = standings
        .0
        .iter()
        .filter(|(_, s)| **s <= ENGAGE_THRESHOLD)
        .map(|(f, s)| (f.clone(), (-*s / 10.0).max(1.0)))
        .collect();
    if hostility.0 != derived {
        hostility.0 = derived;
    }
}

/// `{var}` substitution for arrest-template text.
fn subst(s: &str, vars: &[(&str, String)]) -> String {
    let mut out = s.to_string();
    for (k, v) in vars {
        out = out.replace(k, v);
    }
    out
}

/// Pick a bounty target for a penal mission: a fighter belonging to one of
/// `faction`'s enemies (cheapest, for a fair fight). Falls back to the pirate
/// corvette — someone is always raiding somebody.
fn penal_bounty_ship(iu: &ItemUniverse, faction: &str) -> String {
    let enemy_factions = iu.enemies.get(faction).cloned().unwrap_or_default();
    iu.ships
        .iter()
        .filter(|(_, d)| {
            d.faction
                .as_ref()
                .is_some_and(|f| enemy_factions.contains(f))
                && d.personality == crate::ship::Personality::Fighter
        })
        .min_by_key(|(_, d)| d.price)
        .map(|(name, _)| name.clone())
        .unwrap_or_else(|| "pirate_corvette".to_string())
}

/// A landable planet of the same faction to run community service to
/// (falls back to any other landable planet).
fn penal_service_destination(
    iu: &ItemUniverse,
    faction: &str,
    exclude: &str,
) -> Option<(String, String)> {
    let landable = |p: &crate::planets::PlanetData| !p.commodities.is_empty();
    let candidates: Vec<(&String, &crate::planets::PlanetData)> = iu
        .star_systems
        .iter()
        .filter(|(sys, _)| !ItemUniverse::TRAINING_SYSTEM_KEYS.contains(&sys.as_str()))
        .flat_map(|(_, s)| s.planets.iter())
        .filter(|(pid, p)| pid.as_str() != exclude && landable(p))
        .collect();
    candidates
        .iter()
        .find(|(_, p)| planet_faction(p) == Some(faction))
        .or_else(|| candidates.first())
        .map(|(pid, p)| ((*pid).clone(), p.display_name.clone()))
}

/// Landing on a hostile faction's world gets you arrested: enforcers seek the
/// player out on the surface (the arrest itself), which opens THREE parallel
/// resolution missions — pay the fine at the clerk, fly a penal bounty, or
/// run a community-service delivery. Completing any one restores standing
/// (see `close_arrest_case`, which cancels the other two). Everything
/// downstream is stock mission machinery: briefing toasts, idempotent NPC
/// spawns, MeetNpc/DestroyShips/LandOnPlanet objectives, completion effects.
fn arrest_on_landing(
    mut reader: MessageReader<PlayerLandedOnPlanet>,
    mut standings: ResMut<FactionStandings>,
    iu: Res<ItemUniverse>,
    current_system: Res<CurrentStarSystem>,
    mut catalog: ResMut<MissionCatalog>,
    log: Res<MissionLog>,
    mut counter: Local<u32>,
) {
    // NB: ResMut<FactionStandings> only for read + the change-tick touch at
    // the end; the actual standing restoration happens via mission effects.
    for PlayerLandedOnPlanet { planet } in reader.read() {
        let Some(pd) = iu.star_systems.values().find_map(|s| s.planets.get(planet)) else {
            continue;
        };
        let Some(faction) = planet_faction(pd).map(String::from) else {
            continue;
        };
        let standing = standings.get(&faction);
        if standing > ARREST_THRESHOLD {
            continue;
        }
        // One open case at a time: the player must resolve (or abandon) the
        // previous arrest before a new one is filed.
        let case_open = catalog.defs.keys().any(|id| {
            id.starts_with("arrest__") && matches!(log.status(id), MissionStatus::Active(_))
        });
        if case_open {
            continue;
        }
        // All text and tuning comes from the (single) Arrest mission template
        // in assets/mission_templates.yaml — regenerated per arrest with the
        // concrete faction / fine / destination substituted in.
        let Some(crate::missions::MissionTemplate::Arrest {
            meet_briefing,
            meet_success,
            meet_failure,
            fine_briefing,
            fine_success,
            fine_failure,
            bounty_briefing,
            bounty_success,
            bounty_failure,
            service_briefing,
            service_success,
            service_failure,
            fine_base,
            fine_per_standing,
            bounty_count,
            service_commodity,
            service_quantity,
        }) = iu
            .mission_templates
            .values()
            .find(|t| matches!(t, crate::missions::MissionTemplate::Arrest { .. }))
        else {
            warn!("no Arrest mission template in assets/mission_templates.yaml — skipping arrest");
            continue;
        };
        *counter += 1;
        let n = *counter;
        let meet_id = format!("arrest__{faction}__{n:04}__meet");

        let fine: i64 = fine_base + (-standing as i64) * fine_per_standing;
        let service_dest = penal_service_destination(&iu, &faction, planet);
        let vars = [
            ("{faction}", faction.clone()),
            ("{fine}", fine.to_string()),
            ("{planet_display}", pd.display_name.clone()),
            (
                "{dest_display}",
                service_dest
                    .as_ref()
                    .map(|(_, d)| d.clone())
                    .unwrap_or_default(),
            ),
            ("{count}", bounty_count.to_string()),
            ("{quantity}", service_quantity.to_string()),
            ("{commodity}", service_commodity.clone()),
        ];
        let restore = CompletionEffect::AdjustStanding {
            faction: faction.clone(),
            delta: POST_ARREST_STANDING - standing,
        };
        let after_meet = vec![crate::missions::types::Precondition::Completed {
            mission: meet_id.clone(),
        }];

        // ── The arrest itself: enforcers cross the pad to you ──
        catalog.defs.insert(
            meet_id.clone(),
            MissionDef {
                briefing: subst(meet_briefing, &vars),
                success_text: subst(meet_success, &vars),
                failure_text: subst(meet_failure, &vars),
                preconditions: Vec::new(),
                offer: OfferKind::Auto,
                start_effects: Vec::new(),
                objective: Objective::MeetNpc {
                    planet: planet.clone(),
                    npc_name: format!("the {faction} enforcers"),
                    building: None,
                    approach: NpcApproach::Seek,
                },
                requires: Vec::new(),
                completion_effects: Vec::new(),
            },
        );

        // ── Resolution 1: pay the fine at the clerk ──
        catalog.defs.insert(
            format!("arrest__{faction}__{n:04}__fine"),
            MissionDef {
                briefing: subst(fine_briefing, &vars),
                success_text: subst(fine_success, &vars),
                failure_text: subst(fine_failure, &vars),
                preconditions: after_meet.clone(),
                offer: OfferKind::Auto,
                start_effects: Vec::new(),
                objective: Objective::MeetNpc {
                    planet: planet.clone(),
                    npc_name: "the fines clerk".to_string(),
                    building: Some("market".to_string()),
                    approach: NpcApproach::Wait,
                },
                requires: Vec::new(),
                completion_effects: vec![CompletionEffect::Pay { credits: -fine }, restore.clone()],
            },
        );

        // ── Resolution 2: penal bounty — hunt the faction's enemies ──
        catalog.defs.insert(
            format!("arrest__{faction}__{n:04}__bounty"),
            MissionDef {
                briefing: subst(bounty_briefing, &vars),
                success_text: subst(bounty_success, &vars),
                failure_text: subst(bounty_failure, &vars),
                preconditions: after_meet.clone(),
                offer: OfferKind::Auto,
                start_effects: Vec::new(),
                objective: Objective::DestroyShips {
                    system: current_system.0.clone(),
                    ship_type: penal_bounty_ship(&iu, &faction),
                    count: *bounty_count,
                    target_name: "Warrant Targets".to_string(),
                    hostile: true,
                    collect: None,
                },
                requires: Vec::new(),
                completion_effects: vec![restore.clone()],
            },
        );

        // ── Resolution 3: community service — a relief run on your own coin ──
        if let Some((dest_id, _)) = service_dest {
            catalog.defs.insert(
                format!("arrest__{faction}__{n:04}__service"),
                MissionDef {
                    briefing: subst(service_briefing, &vars),
                    success_text: subst(service_success, &vars),
                    failure_text: subst(service_failure, &vars),
                    preconditions: after_meet,
                    offer: OfferKind::Auto,
                    start_effects: Vec::new(),
                    objective: Objective::LandOnPlanet { planet: dest_id },
                    requires: vec![crate::missions::types::CompletionRequirement::HasCargo {
                        commodity: service_commodity.clone(),
                        quantity: *service_quantity,
                    }],
                    completion_effects: vec![
                        CompletionEffect::RemoveCargo {
                            commodity: service_commodity.clone(),
                            quantity: *service_quantity,
                        },
                        restore,
                    ],
                },
            );
        }
        // Touch standings so downstream change-gated systems re-run.
        standings.adjust(&faction, 0.0);
    }
}

/// Completing ANY arrest-resolution mission closes the whole case: the other
/// open resolutions are silently retired (no failure toast / effects — the
/// status flip alone means they stop being offered, tracked, and spawned).
fn close_arrest_case(
    mut completed: MessageReader<MissionCompleted>,
    catalog: Res<MissionCatalog>,
    mut log: ResMut<MissionLog>,
) {
    for MissionCompleted(id) in completed.read() {
        let Some(case) = id
            .strip_suffix("__fine")
            .or_else(|| id.strip_suffix("__bounty"))
            .or_else(|| id.strip_suffix("__service"))
        else {
            continue;
        };
        if !case.starts_with("arrest__") {
            continue;
        }
        for other in catalog.defs.keys() {
            if other != id
                && other.starts_with(case)
                && matches!(log.status(other), MissionStatus::Active(_))
            {
                log.set(other, MissionStatus::Failed);
            }
        }
    }
}

pub fn standing_plugin(app: &mut App) {
    use crate::session::SessionResourceExt;
    app.init_session_resource::<FactionStandings>().add_systems(
        Update,
        (
            standing_on_hits,
            standing_on_mission_complete,
            arrest_on_landing,
            close_arrest_case,
            derive_player_hostility.run_if(resource_changed::<FactionStandings>),
        )
            .run_if(not(in_state(PlayState::MainMenu))),
    );
}

#[cfg(test)]
#[path = "tests/standing_tests.rs"]
mod tests;
