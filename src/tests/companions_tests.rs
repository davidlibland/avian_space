//! Companions: registry, grants + the permadeath ledger, the hire pool,
//! temperament AI, chatter rate-limiting, and dismissal semantics.

use super::*;
use crate::carrier::{Escort, EscortKind, EscortMode, EscortRoster};
use crate::missions::types::{CompletionEffect, MissionDef, Objective, OfferKind};
use crate::missions::{MissionCatalog, MissionCompleted};
use crate::ship::{Ship, ShipData, ShipHostility, Target};
use crate::{CurrentStarSystem, Player};
use avian2d::prelude::Position;
use std::path::Path;

fn universe() -> ItemUniverse {
    let mut iu: ItemUniverse = crate::item_universe::parse_dir(Path::new("assets")).unwrap();
    iu.finalize();
    iu
}

// ── Registry ─────────────────────────────────────────────────────────────────

#[test]
fn every_friend_exists_and_is_granted_by_an_arc() {
    let iu = universe();
    // The full cast — update when someone new joins the galaxy.
    let cast = [
        "vex_marlowe",
        "oak_adaora",
        "sable_dune",
        "brother_cassian",
        "tinny",
        "yara_brakespear",
        "chandra_vale",
        "mirelle_ossan",
        "ismene_kore",
        "aldous_rook",
        "whisper",
        "pip_harlan",
        "saoirse_quill",
        "jonah_wren",
    ];
    assert_eq!(iu.companions.len(), cast.len(), "cast list is exhaustive");
    for key in cast {
        let def = iu
            .companions
            .get(key)
            .unwrap_or_else(|| panic!("{key} in companions.yaml"));
        assert!(iu.npcs.contains_key(&def.npc), "{key}: face in npcs.yaml");
        assert!(iu.ships.contains_key(&def.ship_type), "{key}: hull exists");
        assert!(
            iu.find_gameplay_planet(&def.home_planet).is_some(),
            "{key}: home exists"
        );
        // Every friend has at least a jump_in line — silence is a bug.
        assert!(
            def.chatter.get("jump_in").is_some_and(|v| !v.is_empty()),
            "{key}: has a jump_in line"
        );
        // Some mission grants them.
        assert!(
            iu.missions
                .values()
                .any(|m| m.completion_effects.iter().any(|e| matches!(
                    e,
                    CompletionEffect::GrantCompanion { companion } if companion == key
                ))),
            "{key}: granted by an arc"
        );
    }
}

// ── Grants + permadeath ──────────────────────────────────────────────────────

fn grant_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(universe())
        .init_resource::<EscortRoster>()
        .init_resource::<MissionCatalog>()
        .add_message::<MissionCompleted>()
        .add_systems(Update, super::grant_companions_on_mission_complete);
    let def = MissionDef {
        briefing: "b".into(),
        success_text: "s".into(),
        failure_text: "f".into(),
        preconditions: vec![],
        offer: OfferKind::Auto,
        start_effects: vec![],
        objective: Objective::TravelToSystem {
            system: "sol".into(),
        },
        requires: vec![],
        completion_effects: vec![CompletionEffect::GrantCompanion {
            companion: "vex_marlowe".into(),
        }],
        squadron: vec![],
        faction: None,
        hint: None,
    };
    app.world_mut()
        .resource_mut::<MissionCatalog>()
        .defs
        .insert("pledge".into(), def);
    app
}

fn complete(app: &mut App) {
    app.world_mut()
        .write_message(MissionCompleted("pledge".into()));
    app.update();
}

#[test]
fn grant_enrolls_the_friend_once_and_death_is_permanent() {
    let mut app = grant_app();
    complete(&mut app);
    {
        let roster = app.world().resource::<EscortRoster>();
        assert_eq!(roster.entries.len(), 1, "Vex joins the wing");
        assert!(roster.is_enrolled("vex_marlowe"));
    }
    // Completing again never duplicates.
    complete(&mut app);
    assert_eq!(app.world().resource::<EscortRoster>().entries.len(), 1);

    // She dies: fallen ledger blocks any re-grant forever.
    {
        let mut roster = app.world_mut().resource_mut::<EscortRoster>();
        let id = roster.entries[0].id;
        roster.record_death(id);
        assert!(roster.entries.is_empty());
        assert!(roster.fallen.contains("vex_marlowe"));
    }
    complete(&mut app);
    assert!(
        app.world().resource::<EscortRoster>().entries.is_empty(),
        "the fallen do not return"
    );
}

#[test]
fn dismissal_parks_friends_and_discards_hires() {
    let mut roster = EscortRoster::default();
    let friend = roster.add(
        "fed_patrol".into(),
        EscortKind::Companion {
            name: "vex_marlowe".into(),
        },
        80,
    );
    let hire = roster.add(
        "corvette".into(),
        EscortKind::Hired {
            name: "Joss Calloway".into(),
            temperament: "aggressive".into(),
        },
        60,
    );
    roster.dismiss(friend);
    roster.dismiss(hire);
    assert!(roster.entries.is_empty());
    assert!(
        roster.parked.contains("vex_marlowe"),
        "friends go home, re-recruitable"
    );
    assert!(roster.fallen.is_empty(), "dismissal is not death");
    // A parked friend is not re-granted by mission completion either.
    let mut app = grant_app();
    app.world_mut()
        .resource_mut::<EscortRoster>()
        .parked
        .insert("vex_marlowe".into());
    complete(&mut app);
    assert!(app.world().resource::<EscortRoster>().entries.is_empty());
}

#[test]
fn roster_save_round_trips_both_ledgers() {
    use crate::session::SessionResource;
    let mut r = EscortRoster::default();
    r.add(
        "fed_patrol".into(),
        EscortKind::Companion {
            name: "vex_marlowe".into(),
        },
        55,
    );
    r.fallen.insert("tinny".into());
    r.parked.insert("sable_dune".into());
    let restored = EscortRoster::from_save(r.to_save(), &universe());
    assert!(restored.is_enrolled("vex_marlowe"));
    assert!(restored.fallen.contains("tinny"), "the dead stay dead");
    assert!(
        restored.parked.contains("sable_dune"),
        "the parked stay home"
    );
}

// ── Hire pool ────────────────────────────────────────────────────────────────

#[test]
fn hire_pool_is_deterministic_fighters_only_and_fee_priced() {
    let iu = universe();
    let a = super::hire_pool(&iu, "earth", &Default::default());
    let b = super::hire_pool(&iu, "earth", &Default::default());
    assert_eq!(a, b, "same faces wait at the same bar");
    assert!(!a.is_empty(), "Earth's shipyard staffs the bar");
    for offer in &a {
        let data = &iu.ships[&offer.ship_type];
        assert_eq!(data.personality, crate::ship::Personality::Fighter);
        assert_eq!(
            offer.fee,
            (data.price as f64 * super::HIRE_FEE_FRACTION).ceil() as i128,
            "fee is the hire fraction of hull price"
        );
    }
    // Nowhere, nobody.
    assert!(
        super::hire_pool(&iu, "not_a_planet", &Default::default()).is_empty(),
        "unknown planets hire nobody"
    );
}

// ── Temperament ──────────────────────────────────────────────────────────────

fn temperament_app() -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(CurrentStarSystem("sol".into()))
        .add_systems(Update, super::apply_temperament);
    let player = app
        .world_mut()
        .spawn((
            Ship::from_ship_data(&ShipData::default(), "p"),
            ShipHostility::default(),
            Player,
            Position(Vec2::ZERO),
        ))
        .id();
    (app, player)
}

fn spawn_companion(
    app: &mut App,
    player: Entity,
    temperament: super::Temperament,
    health: i32,
) -> Entity {
    let mut ship = Ship::from_ship_data(
        &ShipData {
            max_health: 100,
            ..Default::default()
        },
        "c",
    );
    ship.health = health;
    app.world_mut()
        .spawn((
            Escort {
                mother: player,
                anchor: Entity::PLACEHOLDER,
            },
            temperament,
            ship,
            EscortMode::Escort,
            Position(Vec2::new(100.0, 0.0)),
        ))
        .id()
}

fn spawn_threat(app: &mut App, player: Entity, dist: f32) -> Entity {
    let mut ship = Ship::from_ship_data(&ShipData::default(), "h");
    ship.weapons_target = Some(Target::Ship(player));
    app.world_mut()
        .spawn((
            crate::ai_ships::AIShip {
                personality: crate::ship::Personality::Fighter,
            },
            ship,
            ShipHostility::default(),
            Position(Vec2::new(dist, 0.0)),
        ))
        .id()
}

#[test]
fn protective_companions_engage_whatever_targets_the_player() {
    let (mut app, player) = temperament_app();
    let companion = spawn_companion(&mut app, player, super::Temperament::Protective, 100);
    let threat = spawn_threat(&mut app, player, 900.0);
    app.update();
    match app.world().get::<EscortMode>(companion) {
        Some(EscortMode::Attack { target }) => assert_eq!(target.get_entity(), threat),
        other => panic!("protective must engage the player's attacker: {other:?}"),
    }
}

#[test]
fn cautious_companions_break_off_at_low_hull_and_resume_when_patched() {
    let (mut app, player) = temperament_app();
    let companion = spawn_companion(&mut app, player, super::Temperament::Cautious, 30);
    let _threat = spawn_threat(&mut app, player, 300.0);
    app.update();
    assert!(
        matches!(
            app.world().get::<EscortMode>(companion),
            Some(EscortMode::Escort)
        ),
        "30% hull: holds formation instead of engaging"
    );
    assert!(
        app.world()
            .get::<super::CautiousRetreat>(companion)
            .is_some()
    );

    // Mid-band (50%): still retreating — hysteresis, not a hard line.
    app.world_mut().get_mut::<Ship>(companion).unwrap().health = 50;
    app.update();
    assert!(
        app.world()
            .get::<super::CautiousRetreat>(companion)
            .is_some()
    );

    // Patched to 80%: back in the fight.
    app.world_mut().get_mut::<Ship>(companion).unwrap().health = 80;
    app.update();
    app.update();
    assert!(
        app.world()
            .get::<super::CautiousRetreat>(companion)
            .is_none()
    );
    assert!(
        matches!(
            app.world().get::<EscortMode>(companion),
            Some(EscortMode::Attack { .. })
        ),
        "healed cautious companion re-engages the threat"
    );
}

#[test]
fn player_orders_override_temperament() {
    let (mut app, player) = temperament_app();
    let companion = spawn_companion(&mut app, player, super::Temperament::Aggressive, 100);
    let _threat = spawn_threat(&mut app, player, 400.0);
    // The player ordered a hold — temperament must not undo an explicit
    // non-Escort order.
    *app.world_mut().get_mut::<EscortMode>(companion).unwrap() = EscortMode::Dock;
    app.update();
    assert!(
        matches!(
            app.world().get::<EscortMode>(companion),
            Some(EscortMode::Dock)
        ),
        "explicit orders stand"
    );
}

// ── Chatter ──────────────────────────────────────────────────────────────────

#[test]
fn chatter_speaks_once_then_rate_limits() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(universe())
        .init_resource::<EscortRoster>()
        .init_resource::<super::ChatterState>()
        .insert_resource(crate::hud::CommsChannel::default())
        .add_message::<crate::missions::PlayerEnteredSystem>()
        .add_message::<crate::ship::DamageShip>()
        .add_message::<crate::missions::ShipDestroyed>()
        .add_systems(Update, super::companion_chatter);
    let player = app.world_mut().spawn(Player).id();
    // A live Vex with a roster entry.
    let id = app.world_mut().resource_mut::<EscortRoster>().add(
        "fed_patrol".into(),
        EscortKind::Companion {
            name: "vex_marlowe".into(),
        },
        100,
    );
    app.world_mut().spawn((
        Escort {
            mother: player,
            anchor: Entity::PLACEHOLDER,
        },
        crate::carrier::PersistentEscort(id),
        Position(Vec2::ZERO),
    ));

    app.world_mut()
        .write_message(crate::missions::PlayerEnteredSystem {
            system: "sol".into(),
        });
    app.update();
    let first = app
        .world()
        .resource::<crate::hud::CommsChannel>()
        .message
        .clone();
    assert!(
        first.starts_with("Vex Marlowe:"),
        "she greets the new system: {first}"
    );

    // Another event inside the cooldown stays quiet.
    app.world_mut()
        .resource_mut::<crate::hud::CommsChannel>()
        .send("SENTINEL");
    app.world_mut()
        .write_message(crate::missions::PlayerEnteredSystem {
            system: "barnard".into(),
        });
    app.update();
    assert_eq!(
        app.world().resource::<crate::hud::CommsChannel>().message,
        "SENTINEL",
        "rate limit holds the channel"
    );
}

/// The merchant-line friends fly real cargo vessels: befriending them is a
/// fleet-hold upgrade, not just a gun.
#[test]
fn merchant_line_friends_fly_cargo_vessels() {
    let iu = universe();
    for (key, min_hold) in [
        ("chandra_vale", 30u16), // freighter
        ("mirelle_ossan", 60),   // hauler
        ("saoirse_quill", 12),   // courier
        ("pip_harlan", 8),       // shuttle
    ] {
        let def = &iu.companions[key];
        let data = &iu.ships[&def.ship_type];
        assert!(
            data.cargo_space >= min_hold,
            "{key} flies a {} with only {} hold",
            def.ship_type,
            data.cargo_space
        );
        assert_eq!(
            data.personality,
            crate::ship::Personality::Trader,
            "{key}: cargo friends fly trader hulls"
        );
    }
}

#[cfg(test)]
mod hire_replacement {
    /// A dead hire doesn't rise from the dead: their name goes to the
    /// fallen ledger and the same SEAT (hull, temperament, fee) is taken
    /// by the next name in the deterministic sequence. Supply recovers;
    /// people don't.
    #[test]
    fn fallen_hires_are_replaced_not_resurrected() {
        let mut iu: crate::item_universe::ItemUniverse =
            crate::item_universe::parse_dir(std::path::Path::new("assets")).unwrap();
        iu.finalize();
        let planet = "earth";
        let empty = Default::default();
        let before = crate::companions::hire_pool(&iu, planet, &empty);
        assert!(!before.is_empty(), "premise: earth hires pilots");
        let dead = before[0].pilot_name.clone();

        let mut fallen = std::collections::HashSet::new();
        fallen.insert(dead.clone());
        let after = crate::companions::hire_pool(&iu, planet, &fallen);
        assert_eq!(before.len(), after.len(), "the seat is refilled");
        assert!(
            !after.iter().any(|o| o.pilot_name == dead),
            "{dead} stays dead"
        );
        // The seat keeps its terms: same hull + fee lineup, only names move.
        let terms = |pool: &[crate::companions::HireOffer]| {
            pool.iter()
                .map(|o| (o.ship_type.clone(), o.fee))
                .collect::<Vec<_>>()
        };
        assert_eq!(terms(&before), terms(&after), "hull and fee are the seat's");
        // Deterministic: same ledger, same replacement.
        let again = crate::companions::hire_pool(&iu, planet, &fallen);
        assert_eq!(
            after.iter().map(|o| &o.pilot_name).collect::<Vec<_>>(),
            again.iter().map(|o| &o.pilot_name).collect::<Vec<_>>(),
        );
        // Hired deaths actually reach the ledger.
        let mut roster = crate::carrier::EscortRoster::default();
        let id = roster.add(
            "fighter".into(),
            crate::carrier::EscortKind::Hired {
                name: "Joss Pike".into(),
                temperament: "aggressive".into(),
            },
            10,
        );
        roster.record_death(id);
        assert!(roster.fallen.contains("Joss Pike"));
    }
}

#[cfg(test)]
mod fugitive_identity {
    /// Anonymous mission NPCs keep one face: the fallback avatar spec is
    /// seeded by mission id, so the maze fugitive looks the same on every
    /// level and every landing — and different missions get different
    /// strangers.
    #[test]
    fn anonymous_spec_is_stable_per_mission() {
        let Some(layers) = crate::character_compositor::CharacterLayers::load_for_tests() else {
            return; // layer assets unavailable in this environment
        };
        let a1 =
            crate::surface_npc::anonymous_mission_spec(&layers, "claim_jumper_mars", "civilian");
        let a2 =
            crate::surface_npc::anonymous_mission_spec(&layers, "claim_jumper_mars", "civilian");
        assert_eq!(a1, a2, "same mission, same face");
        let b = crate::surface_npc::anonymous_mission_spec(&layers, "manifest_ghost", "civilian");
        assert_ne!(a1, b, "different missions, different strangers");
    }
}

#[cfg(test)]
mod captive_conversion {
    use bevy::prelude::*;

    /// Catching a target converts the caught BODY into the captive (gains
    /// the CompanionAvatar key, loses the mission tag) — and because the
    /// convert and respawn phases live in ONE system sharing one presence
    /// set, no duplicate prisoner can spawn beside it, not even in the
    /// conversion frame (the exact race the playtest caught twice).
    #[test]
    fn caught_npc_converts_in_place_no_twin() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(bevy::state::app::StatesPlugin);
        app.init_state::<crate::PlayState>();
        app.init_resource::<crate::missions::MissionCatalog>()
            .init_resource::<crate::surface::interiors::CaptivesInTow>()
            .init_resource::<crate::hud::CommsChannel>()
            .init_resource::<bevy::asset::Assets<bevy::image::Image>>()
            .insert_resource({
                let mut iu: crate::item_universe::ItemUniverse =
                    crate::item_universe::parse_dir(std::path::Path::new("assets")).unwrap();
                iu.finalize();
                iu
            })
            .insert_resource(crate::surface_pathfinding::SurfaceCostMap {
                data: vec![1.0; 64 * 64],
                width: 64,
                height: 64,
            })
            .add_message::<crate::missions::NpcCaught>()
            .add_systems(Update, crate::surface::interiors::manage_captives);
        if let Some(layers) = crate::character_compositor::CharacterLayers::load_for_tests() {
            // With layers + cost map + walker present, phase 2 is LIVE:
            // it reaches the presence check every frame, so a conversion
            // that failed to count as "present" would spawn the twin here
            // — the exact race the playtest caught twice.
            app.insert_resource(layers);
        }

        // A catch mission in the catalog...
        let def = crate::missions::MissionDef {
            briefing: String::new(),
            success_text: String::new(),
            failure_text: String::new(),
            preconditions: Vec::new(),
            offer: crate::missions::types::OfferKind::Auto,
            start_effects: Vec::new(),
            objective: crate::missions::Objective::CatchNpc {
                planet: "triton".into(),
                npc_name: "the pirate".into(),
                building: Some("bar".into()),
                hint: None,
                npc: None,
            },
            requires: Vec::new(),
            completion_effects: Vec::new(),
            squadron: Vec::new(),
            faction: None,
            hint: None,
        };
        app.world_mut()
            .resource_mut::<crate::missions::MissionCatalog>()
            .defs
            .insert("m_catch".into(), def);

        // ...its caught body in the world, and the player walker.
        let body = app
            .world_mut()
            .spawn(crate::surface_npc::MissionNpc("m_catch".into()))
            .id();
        app.world_mut()
            .spawn((crate::surface::Walker, Transform::from_xyz(0.0, 0.0, 0.0)));

        app.world_mut().write_message(crate::missions::NpcCaught {
            planet: "triton".into(),
            mission_id: "m_catch".into(),
        });
        app.update();
        app.update(); // a second frame must not double-record either

        let tow = app
            .world()
            .resource::<crate::surface::interiors::CaptivesInTow>();
        assert_eq!(tow.captives.len(), 1, "recorded once");
        let e = app.world().entity(body);
        assert!(
            e.get::<crate::surface_npc::CompanionAvatar>()
                .is_some_and(|c| c.0 == "m_catch"),
            "the caught body IS the captive"
        );
        assert!(
            e.get::<crate::surface_npc::MissionNpc>().is_none(),
            "mission tag removed"
        );
        let mut q = app
            .world_mut()
            .query::<&crate::surface_npc::CompanionAvatar>();
        assert_eq!(
            q.iter(app.world()).count(),
            1,
            "exactly one captive body — no twin"
        );
    }
}

#[cfg(test)]
mod formation {
    /// Followers get DISTINCT formation slots and paces — that's what
    /// keeps a friend, a hire and a prisoner from marching as one sprite.
    #[test]
    fn formation_params_are_distinct_and_sane() {
        let keys = ["Joss Pike", "jonah_wren", "claim_jumper_mars", "Vera Moss"];
        let mut offsets = Vec::new();
        for k in keys {
            let (off, pace) = crate::surface_npc::formation_params(k);
            let r = off.length() / crate::surface::TILE_PX;
            assert!((1.0..=2.8).contains(&r), "{k}: slot radius {r} in range");
            assert!((0.85..=1.15).contains(&pace), "{k}: pace {pace} in range");
            // Deterministic.
            assert_eq!(off, crate::surface_npc::formation_params(k).0);
            offsets.push(off);
        }
        for i in 0..offsets.len() {
            for j in (i + 1)..offsets.len() {
                assert!(
                    (offsets[i] - offsets[j]).length() > crate::surface::TILE_PX * 0.5,
                    "slots {i}/{j} too close: {:?} vs {:?}",
                    offsets[i],
                    offsets[j]
                );
            }
        }
    }
}
