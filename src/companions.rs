//! Companions — loyal friends and hired escorts (docs/companions_design.md).
//!
//! A companion is one [`EscortRoster`] entry plus one `companions.yaml`
//! record; everything else derives. Friends are granted by missions
//! (`CompletionEffect::GrantCompanion`) and are PERMADEATH — the roster's
//! `fallen` ledger blocks a re-grant forever. Hires are bought at the bar
//! from a pool derived from the local shipyard, and are replaceable.
//! Temperament maps onto the escort AI; chatter rides the comms ticker.

use bevy::prelude::*;
use std::collections::HashMap;

use crate::carrier::{Escort, EscortKind, EscortRoster, PersistentEscort};
use crate::item_universe::ItemUniverse;
use crate::missions::{MissionCatalog, MissionCompleted};
use crate::ship::{Ship, ShipHostility};
use crate::{PlayState, Player};

/// Total simultaneous companions (friends + hires). Carried bay fighters
/// don't count. Kept small so formations, orders and chatter stay readable.
pub const MAX_COMPANIONS: usize = 3;

/// Hire fee as a fraction of hull price.
pub const HIRE_FEE_FRACTION: f64 = 0.3;

/// Minimum seconds between chatter lines.
const CHATTER_COOLDOWN: f32 = 45.0;
/// Seconds of quiet before an idle line may fire.
const IDLE_CHATTER_AFTER: f32 = 150.0;

// ── Registry (assets/companions.yaml) ────────────────────────────────────────

/// A named companion the player can befriend through missions.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct CompanionDef {
    /// Display name ("Vex Marlowe").
    pub name: String,
    /// Key into assets/npcs.yaml — face, avatar, chat title.
    pub npc: String,
    /// Hull they fly.
    pub ship_type: String,
    /// Where they go when dismissed (and where they can be re-recruited).
    pub home_planet: String,
    pub temperament: Temperament,
    #[serde(default)]
    pub bio: String,
    /// Event key → lines. Known keys: kill, player_hit, jump_in, idle.
    #[serde(default)]
    pub chatter: HashMap<String, Vec<String>>,
}

/// How a companion fights. One enum; the escort AI derives the rest.
#[derive(
    Component, Clone, Copy, PartialEq, Eq, Debug, serde::Deserialize, serde::Serialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Temperament {
    /// Hunts anything hostile in a wide radius; chases kills.
    Aggressive,
    /// Guards the player: engages whatever is shooting at them.
    Protective,
    /// Disengages below 40% hull and holds formation until patched up.
    Cautious,
}

impl Temperament {
    pub fn parse(s: &str) -> Self {
        match s {
            "aggressive" => Temperament::Aggressive,
            "cautious" => Temperament::Cautious,
            _ => Temperament::Protective,
        }
    }

    pub fn key(&self) -> &'static str {
        match self {
            Temperament::Aggressive => "aggressive",
            Temperament::Protective => "protective",
            Temperament::Cautious => "cautious",
        }
    }
}

/// Cautious companions that broke off at low hull; cleared above 70%.
#[derive(Component)]
pub struct CautiousRetreat;

// ── Grants + the permadeath ledger ────────────────────────────────────────────

/// Award friends when their granting mission completes. The roster's
/// `fallen` ledger makes death permanent: a fallen friend is never
/// re-granted, and an enrolled/parked friend is never duplicated.
fn grant_companions_on_mission_complete(
    mut reader: MessageReader<MissionCompleted>,
    catalog: Res<MissionCatalog>,
    iu: Res<ItemUniverse>,
    mut roster: ResMut<EscortRoster>,
    toast: Option<ResMut<crate::missions::ui::MissionToast>>,
) {
    let mut toast = toast;
    for MissionCompleted(id) in reader.read() {
        let Some(def) = catalog.defs.get(id) else {
            continue;
        };
        for effect in &def.completion_effects {
            let crate::missions::types::CompletionEffect::GrantCompanion { companion } = effect
            else {
                continue;
            };
            let Some(cdef) = iu.companions.get(companion) else {
                warn!("GrantCompanion: unknown companion '{companion}'");
                continue;
            };
            if roster.fallen.contains(companion)
                || roster.is_enrolled(companion)
                || roster.parked.contains(companion)
            {
                continue;
            }
            let health = iu
                .ships
                .get(&cdef.ship_type)
                .map(|d| d.max_health)
                .unwrap_or(100);
            roster.add(
                cdef.ship_type.clone(),
                EscortKind::Companion {
                    name: companion.clone(),
                },
                health,
            );
            if let Some(toast) = toast.as_deref_mut() {
                toast.push(format!(
                    "{} joins your wing — {}",
                    cdef.name,
                    cdef.bio.clone()
                ));
            }
        }
    }
}

// ── Hire pool (derived per planet) ────────────────────────────────────────────

/// One pilot for hire at a bar.
#[derive(Clone, Debug, PartialEq)]
pub struct HireOffer {
    pub pilot_name: String,
    pub ship_type: String,
    pub temperament: Temperament,
    pub fee: i128,
}

const HIRE_FIRST_NAMES: &[&str] = &[
    "Joss", "Mara", "Deke", "Ilsa", "Rook", "Tamsin", "Cole", "Nadia", "Pike", "Vera", "Ash",
    "Juno", "Silas", "Petra", "Moss", "Katya",
];
const HIRE_SURNAMES: &[&str] = &[
    "Calloway", "Iri", "Stroud", "Vance", "Okafor", "Reyes", "Holt", "Drummond", "Sarti",
    "Kesler", "Nyx", "Ferrow",
];

/// Pilots for hire at this planet's bar: fighter hulls from the LOCAL
/// derived shipyard catalog, seeded by planet name so the same faces wait
/// at the same bar. Empty when the planet has no shipyard.
pub fn hire_pool(iu: &ItemUniverse, planet_name: &str) -> Vec<HireOffer> {
    use rand::{Rng, SeedableRng};
    let Some((_, pd)) = iu.find_gameplay_planet(planet_name) else {
        return Vec::new();
    };
    let mut hulls: Vec<&String> = pd
        .shipyard
        .iter()
        .filter(|k| {
            iu.ships
                .get(*k)
                .is_some_and(|d| d.personality == crate::ship::Personality::Fighter)
        })
        .collect();
    hulls.sort();
    if hulls.is_empty() {
        return Vec::new();
    }
    let seed = planet_name
        .bytes()
        .fold(0xC0FFEEu64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut offers = Vec::new();
    let count = hulls.len().min(3);
    for i in 0..count {
        let hull = hulls[rng.gen_range(0..hulls.len())].clone();
        let data = &iu.ships[&hull];
        let temperament = match i {
            0 => Temperament::Protective,
            1 => Temperament::Aggressive,
            _ => Temperament::Cautious,
        };
        let pilot_name = format!(
            "{} {}",
            HIRE_FIRST_NAMES[rng.gen_range(0..HIRE_FIRST_NAMES.len())],
            HIRE_SURNAMES[rng.gen_range(0..HIRE_SURNAMES.len())]
        );
        offers.push(HireOffer {
            pilot_name,
            ship_type: hull,
            temperament,
            fee: (data.price as f64 * HIRE_FEE_FRACTION).ceil() as i128,
        });
    }
    offers
}

/// Companions currently enrolled (friends + hires) — the cap counts these.
pub fn companion_count(roster: &EscortRoster) -> usize {
    roster
        .entries
        .iter()
        .filter(|e| {
            matches!(
                e.kind,
                EscortKind::Companion { .. } | EscortKind::Hired { .. }
            )
        })
        .count()
}

// ── Temperament → escort AI ──────────────────────────────────────────────────

const AGGRESSIVE_ENGAGE_RANGE: f32 = 1400.0;
const PROTECTIVE_ENGAGE_RANGE: f32 = 1800.0;
const CAUTIOUS_BREAK_FRAC: f32 = 0.4;
const CAUTIOUS_RESUME_FRAC: f32 = 0.7;

/// Give companions combat autonomy according to their temperament. Plain
/// escorts only fight when ordered; a companion is a PERSON:
/// - aggressive: engages any ship hostile to the player within range;
/// - protective: engages whatever is targeting the player;
/// - cautious: breaks off below 40% hull and holds formation until 70%.
#[allow(clippy::type_complexity)]
fn apply_temperament(
    mut commands: Commands,
    player: Query<(Entity, &ShipHostility, &avian2d::prelude::Position), With<Player>>,
    mut companions: Query<
        (
            Entity,
            &Escort,
            &Temperament,
            &Ship,
            &mut crate::carrier::EscortMode,
            &avian2d::prelude::Position,
            Has<CautiousRetreat>,
        ),
        Without<crate::carrier::DockingEscort>,
    >,
    hostiles: Query<
        (Entity, &Ship, &ShipHostility, &avian2d::prelude::Position),
        (With<crate::ai_ships::AIShip>, Without<Escort>),
    >,
) {
    use crate::carrier::EscortMode;
    use crate::ship::Target;
    let Ok((player_entity, player_hostility, player_pos)) = player.single() else {
        return;
    };
    for (entity, escort, temperament, ship, mut mode, pos, retreating) in &mut companions {
        if escort.mother != player_entity {
            continue;
        }
        let max_health = ship.max_health().max(1);
        let frac = ship.health as f32 / max_health as f32;

        // Cautious: break off at low hull, resume above the hysteresis band.
        if *temperament == Temperament::Cautious {
            if retreating {
                if frac >= CAUTIOUS_RESUME_FRAC {
                    commands.entity(entity).remove::<CautiousRetreat>();
                } else {
                    if !matches!(*mode, EscortMode::Escort) {
                        *mode = EscortMode::Escort;
                    }
                    continue;
                }
            } else if frac < CAUTIOUS_BREAK_FRAC {
                commands.entity(entity).insert(CautiousRetreat);
                *mode = EscortMode::Escort;
                continue;
            }
        }

        // Already fighting or ordered elsewhere: leave the order alone.
        if !matches!(*mode, EscortMode::Escort) {
            continue;
        }

        let engage_range = match temperament {
            Temperament::Aggressive => AGGRESSIVE_ENGAGE_RANGE,
            Temperament::Protective => PROTECTIVE_ENGAGE_RANGE,
            Temperament::Cautious => AGGRESSIVE_ENGAGE_RANGE * 0.6,
        };

        let candidate = hostiles
            .iter()
            .filter(|(_, hship, hhost, hpos)| {
                let threat_to_player = hship.should_engage(player_hostility)
                    || hship
                        .weapons_target
                        .as_ref()
                        .is_some_and(|t| t.get_entity() == player_entity);
                let _ = hhost;
                let anchor = match temperament {
                    // Protective ranges from the PLAYER; others from self.
                    Temperament::Protective => player_pos.0,
                    _ => pos.0,
                };
                threat_to_player && (hpos.0 - anchor).length() <= engage_range
            })
            .min_by(|a, b| {
                let da = (a.3 .0 - pos.0).length();
                let db = (b.3 .0 - pos.0).length();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
        if let Some((target, ..)) = candidate {
            *mode = EscortMode::Attack {
                target: Target::Ship(target),
            };
        }
    }
}

// ── Chatter ──────────────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct ChatterState {
    cooldown: f32,
    idle: f32,
}

impl Default for ChatterState {
    fn default() -> Self {
        Self {
            cooldown: 0.0,
            idle: 0.0,
        }
    }
}

/// Small line pools for hired pilots, by temperament.
fn hired_lines(t: Temperament, event: &str) -> &'static [&'static str] {
    match (t, event) {
        (Temperament::Aggressive, "kill") => &["Paid in full.", "Next."],
        (Temperament::Aggressive, "jump_in") => &["Formation. Try to keep up."],
        (Temperament::Protective, "player_hit") => &["Covering you — break off!"],
        (Temperament::Protective, "jump_in") => &["On your wing."],
        (Temperament::Cautious, "player_hit") => &["That looked expensive. Careful."],
        (Temperament::Cautious, "jump_in") => &["I don't like this system already."],
        _ => &[],
    }
}

/// One rate-limited voice at a time, into the comms ticker.
#[allow(clippy::too_many_arguments)]
fn companion_chatter(
    time: Res<Time>,
    mut state: ResMut<ChatterState>,
    iu: Res<ItemUniverse>,
    roster: Res<EscortRoster>,
    live: Query<(&PersistentEscort, &avian2d::prelude::Position), With<Escort>>,
    player: Query<Entity, With<Player>>,
    mut entered: MessageReader<crate::missions::PlayerEnteredSystem>,
    mut damaged: MessageReader<crate::ship::DamageShip>,
    mut destroyed: MessageReader<crate::missions::ShipDestroyed>,
    destroyed_pos: Query<&avian2d::prelude::Position>,
    comms: Option<ResMut<crate::hud::CommsChannel>>,
) {
    let Some(mut comms) = comms else { return };
    state.cooldown = (state.cooldown - time.delta_secs()).max(0.0);
    state.idle += time.delta_secs();
    let player_entity = player.single().ok();

    // Event priority: kill > player_hit > jump_in > idle.
    let mut event: Option<&str> = None;
    for d in destroyed.read() {
        // Attribution heuristic: a companion close to the wreck takes credit.
        let near = destroyed_pos.get(d.entity).ok().map(|p| p.0);
        if let Some(wreck) = near {
            if live.iter().any(|(_, p)| (p.0 - wreck).length() < 700.0) {
                event = Some("kill");
            }
        }
    }
    for d in damaged.read() {
        if Some(d.entity) == player_entity {
            event.get_or_insert("player_hit");
        }
    }
    if entered.read().next().is_some() {
        event.get_or_insert("jump_in");
    }
    if event.is_none() && state.idle >= IDLE_CHATTER_AFTER {
        event = Some("idle");
    }
    let Some(event) = event else { return };
    if state.cooldown > 0.0 {
        return;
    }

    // Pick a speaker among LIVE companions with a line for this event.
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut lines: Vec<(String, String)> = Vec::new(); // (speaker, line)
    for (pe, _) in &live {
        let Some(entry) = roster.entries.iter().find(|e| e.id == pe.0) else {
            continue;
        };
        match &entry.kind {
            EscortKind::Companion { name } => {
                if let Some(def) = iu.companions.get(name) {
                    if let Some(pool) = def.chatter.get(event) {
                        if !pool.is_empty() {
                            let line = pool[rng.gen_range(0..pool.len())].clone();
                            lines.push((def.name.clone(), line));
                        }
                    }
                }
            }
            EscortKind::Hired { name, temperament } => {
                let pool = hired_lines(Temperament::parse(temperament), event);
                if !pool.is_empty() {
                    lines.push((name.clone(), pool[rng.gen_range(0..pool.len())].to_string()));
                }
            }
            EscortKind::Carried { .. } => {}
        }
    }
    if let Some((speaker, line)) = lines.into_iter().next() {
        comms.send(format!("{speaker}: \"{line}\""));
        state.cooldown = CHATTER_COOLDOWN;
        state.idle = 0.0;
    }
}

/// The bar's wingman desk: current companions (dismiss), parked friends at
/// THIS planet (rejoin), and the derived pilots-for-hire pool.
pub fn render_companions_section(
    ui: &mut bevy_egui::egui::Ui,
    ship: &mut Ship,
    roster: &mut EscortRoster,
    iu: &ItemUniverse,
    planet_name: &str,
) {
    use bevy_egui::egui;
    ui.separator();
    ui.heading("Wingmen");
    let count = companion_count(roster);
    ui.label(format!("Flight: {count}/{MAX_COMPANIONS} companions"));

    // Current companions with a Dismiss control.
    let current: Vec<(u64, String, String, i32)> = roster
        .entries
        .iter()
        .filter_map(|e| match &e.kind {
            EscortKind::Companion { name } => iu
                .companions
                .get(name)
                .map(|d| (e.id, d.name.clone(), e.ship_type.clone(), e.health)),
            EscortKind::Hired { name, .. } => {
                Some((e.id, name.clone(), e.ship_type.clone(), e.health))
            }
            EscortKind::Carried { .. } => None,
        })
        .collect();
    for (id, name, hull, health) in current {
        ui.horizontal(|ui| {
            ui.label(format!("{name} — {hull} ({health} hp)"));
            if ui
                .button("Dismiss")
                .on_hover_text("Hires leave for good; friends go home and can rejoin there.")
                .clicked()
            {
                roster.dismiss(id);
            }
        });
    }

    // Parked friends whose home is THIS planet can rejoin for free.
    let here: Vec<String> = roster
        .parked
        .iter()
        .filter(|key| {
            iu.companions
                .get(*key)
                .is_some_and(|d| d.home_planet == planet_name)
        })
        .cloned()
        .collect();
    for key in here {
        let Some(def) = iu.companions.get(&key) else { continue };
        ui.horizontal(|ui| {
            ui.label(format!("{} is here, nursing a drink.", def.name));
            let can = companion_count(roster) < MAX_COMPANIONS;
            if ui
                .add_enabled(can, egui::Button::new("Rejoin"))
                .clicked()
            {
                let health = iu
                    .ships
                    .get(&def.ship_type)
                    .map(|d| d.max_health)
                    .unwrap_or(100);
                roster.parked.remove(&key);
                roster.add(
                    def.ship_type.clone(),
                    EscortKind::Companion { name: key.clone() },
                    health,
                );
            }
        });
    }

    // Pilots for hire (derived from the local shipyard).
    let pool = hire_pool(iu, planet_name);
    if !pool.is_empty() {
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new("Pilots for hire")
                .color(egui::Color32::from_rgb(200, 200, 210)),
        );
        for offer in pool {
            // One copy of each pilot: skip if already flying with us.
            let taken = roster.entries.iter().any(|e| {
                matches!(&e.kind, EscortKind::Hired { name, .. } if *name == offer.pilot_name)
            });
            if taken {
                continue;
            }
            ui.horizontal(|ui| {
                ui.label(format!(
                    "{} — {} ({}) · {} cr",
                    offer.pilot_name,
                    offer.ship_type,
                    offer.temperament.key(),
                    offer.fee
                ));
                let can = companion_count(roster) < MAX_COMPANIONS
                    && ship.credits >= offer.fee;
                if ui.add_enabled(can, egui::Button::new("Hire")).clicked() {
                    ship.credits -= offer.fee;
                    let health = iu
                        .ships
                        .get(&offer.ship_type)
                        .map(|d| d.max_health)
                        .unwrap_or(100);
                    roster.add(
                        offer.ship_type.clone(),
                        EscortKind::Hired {
                            name: offer.pilot_name.clone(),
                            temperament: offer.temperament.key().to_string(),
                        },
                        health,
                    );
                }
            });
        }
    }
}

pub fn companions_plugin(app: &mut App) {
    app.init_resource::<ChatterState>().add_systems(
        Update,
        (
            grant_companions_on_mission_complete.run_if(not(in_state(PlayState::MainMenu))),
            (apply_temperament, companion_chatter).run_if(in_state(PlayState::Flying)),
        ),
    );
}

#[cfg(test)]
#[path = "tests/companions_tests.rs"]
mod tests;
