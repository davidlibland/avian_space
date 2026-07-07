//! Escort/squadron tests: mission squadrons muster once in the battle system,
//! stand down with the mission, and (unlike carried escorts) never dock.

use super::*;
use crate::missions::types::{
    CompletionEffect, MissionDef, MissionStatus, Objective, ObjectiveProgress, OfferKind,
};
use crate::missions::{MissionCatalog, MissionCompleted, MissionFailed, MissionLog};
use crate::ship::{Ship, ShipData};
use crate::{CurrentStarSystem, Player};

fn battle_mission(system: &str, squadron: &[&str]) -> MissionDef {
    MissionDef {
        briefing: "b".into(),
        success_text: "s".into(),
        failure_text: "f".into(),
        preconditions: vec![],
        offer: OfferKind::Auto,
        start_effects: vec![],
        objective: Objective::DestroyShips {
            system: system.into(),
            ship_type: "pirate_corvette".into(),
            count: 3,
            target_name: "Invaders".into(),
            hostile: true,
            collect: None,
        },
        requires: vec![],
        completion_effects: vec![CompletionEffect::Pay { credits: 1 }],
        squadron: squadron.iter().map(|s| s.to_string()).collect(),
        faction: None,
    }
}

fn squadron_app(current_system: &str) -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .init_resource::<MissionLog>()
        .init_resource::<MissionCatalog>()
        .insert_resource(CurrentStarSystem(current_system.to_string()))
        .add_message::<SpawnEscort>()
        .add_systems(Update, spawn_mission_squadrons);
    let player = app
        .world_mut()
        .spawn((
            Ship::from_ship_data(&ShipData::default(), "test"),
            Player,
            Position(Vec2::ZERO),
        ))
        .id();
    (app, player)
}

fn drain_spawns(app: &mut App) -> Vec<SpawnEscort> {
    app.world_mut()
        .resource_mut::<Messages<SpawnEscort>>()
        .drain()
        .collect()
}

#[test]
fn squadron_musters_once_in_the_battle_system() {
    let (mut app, player) = squadron_app("procyon");
    app.world_mut()
        .resource_mut::<MissionCatalog>()
        .defs
        .insert("battle".into(), battle_mission("procyon", &["fighter", "fighter", "corvette"]));
    app.world_mut().resource_mut::<MissionLog>().set(
        "battle",
        MissionStatus::Active(ObjectiveProgress::default()),
    );

    app.update();
    let spawns = drain_spawns(&mut app);
    assert_eq!(spawns.len(), 3, "one escort per squadron entry");
    for s in &spawns {
        assert_eq!(s.mother, player);
        assert!(s.carried.is_none(), "squadron wings are not carried");
        assert_eq!(s.mission.as_deref(), Some("battle"));
    }
    // The wing arrives in a ring around the player, not stacked on one point
    // (the spawner honors SpawnEscort.position).
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        assert!(
            spawns[a].position.distance(spawns[b].position) > 1.0,
            "wingmen spawn at distinct formation slots"
        );
    }

    // Never re-mustered — losses are real.
    app.update();
    app.update();
    assert_eq!(drain_spawns(&mut app).len(), 0, "no re-muster on later frames");
    let MissionStatus::Active(p) = app.world().resource::<MissionLog>().status("battle") else {
        panic!()
    };
    assert!(p.squadron_spawned);
}

#[test]
fn squadron_waits_for_the_battle_system() {
    let (mut app, _player) = squadron_app("sol"); // player NOT in procyon
    app.world_mut()
        .resource_mut::<MissionCatalog>()
        .defs
        .insert("battle".into(), battle_mission("procyon", &["fighter"]));
    app.world_mut().resource_mut::<MissionLog>().set(
        "battle",
        MissionStatus::Active(ObjectiveProgress::default()),
    );
    app.update();
    assert_eq!(
        drain_spawns(&mut app).len(),
        0,
        "the wing musters only in the battle system"
    );
}

#[test]
fn squadron_stands_down_when_the_mission_ends() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_message::<MissionCompleted>()
        .add_message::<MissionFailed>()
        .add_systems(
            Update,
            crate::missions::progress::despawn_targets_on_failure,
        );
    let wing: Vec<Entity> = (0..3)
        .map(|_| app.world_mut().spawn(MissionSquadron("battle".into())).id())
        .collect();
    let other = app
        .world_mut()
        .spawn(MissionSquadron("other_mission".into()))
        .id();

    app.world_mut()
        .write_message(MissionCompleted("battle".into()));
    app.update();
    app.update(); // let despawn commands apply

    for e in wing {
        assert!(
            app.world().get_entity(e).is_err(),
            "wing despawns on mission completion"
        );
    }
    assert!(
        app.world().get_entity(other).is_ok(),
        "other missions' wings are untouched"
    );
}

#[test]
fn dock_order_falls_back_to_formation_for_squadrons() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_message::<crate::sfx::EscortSfx>()
        .add_systems(Update, player_escort_input);
    let player = app
        .world_mut()
        .spawn((Ship::from_ship_data(&ShipData::default(), "t"), Player))
        .id();
    let carried = app
        .world_mut()
        .spawn((
            Escort { mother: player },
            CarriedBy {
                weapon_type: "fighter_bay".into(),
            },
            EscortMode::Escort,
        ))
        .id();
    let wing = app
        .world_mut()
        .spawn((Escort { mother: player }, EscortMode::Escort))
        .id();

    let mut input = ButtonInput::<KeyCode>::default();
    input.press(KeyCode::KeyB);
    app.insert_resource(input);
    app.update();

    assert!(
        matches!(app.world().get::<EscortMode>(carried), Some(EscortMode::Dock)),
        "carried escorts obey the dock order"
    );
    assert!(
        matches!(app.world().get::<EscortMode>(wing), Some(EscortMode::Escort)),
        "squadron wings can't dock — they hold formation instead"
    );
}

/// Escort-order sounds only play when the player actually commands escorts —
/// and the dock cue only when something can actually dock.
#[test]
fn order_sounds_require_escorts() {
    use crate::sfx::EscortSfx;
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_message::<EscortSfx>()
        .add_systems(Update, player_escort_input);
    let player = app
        .world_mut()
        .spawn((Ship::from_ship_data(&ShipData::default(), "t"), Player))
        .id();
    let press = |app: &mut App, key: KeyCode| {
        let mut input = ButtonInput::<KeyCode>::default();
        input.press(key);
        app.insert_resource(input);
        app.update();
        app.world_mut()
            .resource_mut::<Messages<EscortSfx>>()
            .drain()
            .collect::<Vec<_>>()
    };

    // No escorts: silence.
    assert!(press(&mut app, KeyCode::KeyB).is_empty());
    assert!(press(&mut app, KeyCode::KeyN).is_empty());

    // A wings-only flight: B acknowledges with the formation cue, not dock.
    app.world_mut()
        .spawn((Escort { mother: player }, EscortMode::Escort));
    assert_eq!(press(&mut app, KeyCode::KeyB), vec![EscortSfx::Escort]);

    // With a carried escort present, B is a real dock order.
    app.world_mut().spawn((
        Escort { mother: player },
        CarriedBy {
            weapon_type: "fighter_bay".into(),
        },
        EscortMode::Escort,
    ));
    assert_eq!(press(&mut app, KeyCode::KeyB), vec![EscortSfx::Dock]);
}
