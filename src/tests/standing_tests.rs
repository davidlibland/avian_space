//! Faction-standing tests: derived thresholds, price markup, the arrest
//! mission chain (generated from the Arrest template, resolved through the
//! REAL mission systems), and the standing gate on mission offers.

use super::*;
use crate::Player;
use crate::missions::types::{CompletionEffect, MissionStatus, Objective, OfferKind};
use crate::missions::{
    MissionCatalog, MissionCompleted, MissionLog, MissionOffers, NpcMet, PlayerLandedOnPlanet,
    PlayerUnlocks,
};
use crate::ship::{Ship, ShipData, ShipHostility};
use bevy::prelude::*;
use std::path::Path;

fn universe() -> ItemUniverse {
    crate::item_universe::parse_dir(Path::new("assets")).expect("assets/ must parse")
}

// ── Pure derived values ──────────────────────────────────────────────────────

#[test]
fn markup_curve() {
    assert_eq!(price_markup(50.0), 1.0);
    assert_eq!(price_markup(0.0), 1.0);
    assert!((price_markup(-40.0) - 1.30).abs() < 1e-6);
    assert!((price_markup(-100.0) - 1.75).abs() < 1e-6);
    assert_eq!(markup_price(100, 1.0), 100);
    assert_eq!(markup_price(100, 1.30), 130);
    assert_eq!(markup_price(101, 1.30), 132, "rounds up, never down");
}

#[test]
fn standings_clamp() {
    let mut s = FactionStandings::default();
    s.adjust("Federation", -500.0);
    assert_eq!(s.get("Federation"), -100.0);
    s.adjust("Federation", 1000.0);
    assert_eq!(s.get("Federation"), 100.0);
    assert_eq!(s.get("Unknown"), 0.0);
}

// ── Arrest chain, end to end through the real mission systems ────────────────

fn arrest_app() -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(universe())
        .insert_resource(crate::CurrentStarSystem("sol".to_string()))
        .init_resource::<FactionStandings>()
        .init_resource::<MissionLog>()
        .init_resource::<MissionCatalog>()
        .init_resource::<MissionOffers>()
        .init_resource::<PlayerUnlocks>()
        .add_message::<PlayerLandedOnPlanet>()
        .add_message::<crate::missions::PlayerEnteredSystem>()
        .add_message::<crate::missions::PickupCollected>()
        .add_message::<crate::missions::ShipDestroyed>()
        .add_message::<crate::missions::AcceptMission>()
        .add_message::<crate::missions::DeclineMission>()
        .add_message::<crate::missions::AbandonMission>()
        .add_message::<crate::missions::MissionStarted>()
        .add_message::<MissionCompleted>()
        .add_message::<crate::missions::MissionFailed>()
        .add_message::<NpcMet>()
        .add_message::<crate::missions::NpcCaught>()
        .add_message::<crate::ship::ScoreHit>()
        .add_systems(
            Update,
            (
                arrest_on_landing,
                crate::missions::progress::update_locked_to_available,
                crate::missions::progress::advance_land_objectives,
                crate::missions::progress::advance_meet_npc_objectives,
                crate::missions::progress::advance_destroy_objectives,
                crate::missions::progress::finalize_completions,
                crate::missions::progress::finalize_failures,
                standing_on_mission_complete,
                close_arrest_case,
            )
                .chain(),
        );
    let player = app
        .world_mut()
        .spawn((
            Ship::from_ship_data(
                &ShipData {
                    cargo_space: 20,
                    max_health: 100,
                    ..Default::default()
                },
                "test_ship",
            ),
            ShipHostility::default(),
            Player,
        ))
        .id();
    (app, player)
}

fn land(app: &mut App, planet: &str) {
    app.world_mut().write_message(PlayerLandedOnPlanet {
        planet: planet.to_string(),
    });
    app.update();
    app.update(); // let update_locked_to_available auto-start the arrest
}

fn arrest_ids(app: &mut App) -> Vec<String> {
    let mut ids: Vec<String> = app
        .world()
        .resource::<MissionCatalog>()
        .defs
        .keys()
        .filter(|id| id.starts_with("arrest__"))
        .cloned()
        .collect();
    ids.sort();
    ids
}

#[test]
fn arrest_generates_case_and_fine_resolution_restores_standing() {
    let (mut app, player) = arrest_app();
    app.world_mut()
        .resource_mut::<FactionStandings>()
        .adjust("Federation", -50.0);

    // Earth is a Federation world. All four defs are filed at once; the
    // resolutions stay Locked (preconditions on the meet) until the arrest
    // actually happens.
    land(&mut app, "earth");
    let ids = arrest_ids(&mut app);
    assert_eq!(ids.len(), 4, "meet + fine + bounty + service: {ids:?}");
    let meet_id = ids.iter().find(|i| i.ends_with("__meet")).unwrap().clone();
    assert!(matches!(
        app.world().resource::<MissionLog>().status(&meet_id),
        MissionStatus::Active(_)
    ));
    for id in ids.iter().filter(|i| !i.ends_with("__meet")) {
        assert!(
            matches!(app.world().resource::<MissionLog>().status(id), MissionStatus::Locked),
            "resolutions stay locked until the enforcers reach you: {id}"
        );
    }

    // The enforcers reach the player → the case opens with 3 resolutions.
    app.world_mut().write_message(NpcMet {
        planet: "earth".to_string(),
        mission_id: meet_id.clone(),
    });
    app.update();
    app.update();
    let ids = arrest_ids(&mut app);
    let fine_id = ids.iter().find(|i| i.ends_with("__fine")).unwrap().clone();
    let bounty_id = ids.iter().find(|i| i.ends_with("__bounty")).unwrap().clone();
    let service_id = ids
        .iter()
        .find(|i| i.ends_with("__service"))
        .unwrap()
        .clone();
    for id in [&fine_id, &bounty_id, &service_id] {
        assert!(
            matches!(app.world().resource::<MissionLog>().status(id), MissionStatus::Active(_)),
            "all three resolutions auto-start: {id}"
        );
    }

    // Pay the fine (meet the clerk) → standing restored, others retired.
    let credits_before = app.world().get::<Ship>(player).unwrap().credits;
    app.world_mut().write_message(NpcMet {
        planet: "earth".to_string(),
        mission_id: fine_id.clone(),
    });
    app.update();
    app.update();

    let standing = app
        .world()
        .resource::<FactionStandings>()
        .get("Federation");
    assert!(
        (standing - POST_ARREST_STANDING).abs() < 1e-3,
        "fine restores standing to just above hostile, got {standing}"
    );
    let credits_after = app.world().get::<Ship>(player).unwrap().credits;
    // fine = 3000 + 50*50 = 5500 (from the template's fine_base/fine_per_standing)
    assert_eq!(credits_before - credits_after, 5500, "fine charged");
    for id in [&bounty_id, &service_id] {
        assert!(
            matches!(app.world().resource::<MissionLog>().status(id), MissionStatus::Failed),
            "sibling resolutions retire when the case closes: {id}"
        );
    }
    // No re-arrest while nothing changed… standing is above the threshold now.
    land(&mut app, "earth");
    assert_eq!(arrest_ids(&mut app).len(), 4, "no new case filed");
}

#[test]
fn no_arrest_above_threshold_or_on_unaligned_worlds() {
    let (mut app, _player) = arrest_app();
    app.world_mut()
        .resource_mut::<FactionStandings>()
        .adjust("Federation", -20.0); // hostile-ish but above arrest
    land(&mut app, "earth");
    assert!(arrest_ids(&mut app).is_empty());

    app.world_mut()
        .resource_mut::<FactionStandings>()
        .adjust("Federation", -60.0); // now -80: arrestable
    land(&mut app, "pluto"); // pluto: no faction / independent
    assert!(
        arrest_ids(&mut app).is_empty(),
        "unaligned worlds don't arrest"
    );
}

// ── Offer gating ─────────────────────────────────────────────────────────────

#[test]
fn negative_standing_blocks_offers() {
    let iu = universe();
    let mut standings = FactionStandings::default();
    assert!(
        crate::missions::progress::offers_allowed(&standings, &iu, "earth"),
        "neutral standing → offers"
    );
    standings.adjust("Federation", -1.0);
    assert!(
        !crate::missions::progress::offers_allowed(&standings, &iu, "earth"),
        "any negative standing → no offers on that faction's worlds"
    );
    assert!(
        crate::missions::progress::offers_allowed(&standings, &iu, "pluto"),
        "independent worlds don't care"
    );
}

// ── Hostility derivation ─────────────────────────────────────────────────────

#[test]
fn hostility_derived_from_standing_thresholds() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .init_resource::<FactionStandings>()
        .add_systems(Update, derive_player_hostility);
    let player = app
        .world_mut()
        .spawn((ShipHostility::default(), Player))
        .id();

    app.world_mut()
        .resource_mut::<FactionStandings>()
        .adjust("Federation", ENGAGE_THRESHOLD + 1.0); // above threshold
    app.update();
    assert!(
        app.world()
            .get::<ShipHostility>(player)
            .unwrap()
            .0
            .is_empty(),
        "above the engage threshold no faction hunts the player"
    );

    app.world_mut()
        .resource_mut::<FactionStandings>()
        .adjust("Federation", -20.0); // now well below
    app.update();
    assert!(
        app.world()
            .get::<ShipHostility>(player)
            .unwrap()
            .0
            .contains_key("Federation"),
        "below the engage threshold the faction engages"
    );

    // Recovering standing calls them off.
    app.world_mut()
        .resource_mut::<FactionStandings>()
        .adjust("Federation", 100.0);
    app.update();
    assert!(
        app.world()
            .get::<ShipHostility>(player)
            .unwrap()
            .0
            .is_empty(),
        "restored standing clears the hunt"
    );
}
