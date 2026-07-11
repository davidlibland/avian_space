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

// ── Persistent escorts: the roster ───────────────────────────────────────────
//
// Three escort kinds, three lifetimes:
//   squadron wings  — system-bound, never rostered (tested above);
//   carried escorts — persist across jumps/landings, retire by DOCKING;
//   companions      — persist across jumps/landings, retire only by DYING.
mod roster {
    use super::*;
    use crate::session::SessionResource;

    fn iu() -> crate::item_universe::ItemUniverse {
        let mut iu: crate::item_universe::ItemUniverse =
            crate::item_universe::parse_dir(std::path::Path::new("assets")).unwrap();
        iu.finalize();
        iu
    }

    fn spawn_app() -> (App, Entity) {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, bevy::asset::AssetPlugin::default()))
            .init_asset::<Image>()
            .insert_resource(iu())
            .insert_resource(CurrentStarSystem("sol".to_string()))
            .init_resource::<EscortRoster>()
            .add_message::<SpawnEscort>()
            .add_message::<crate::sfx::EscortSfx>()
            .add_message::<crate::explosions::TriggerJumpFlash>()
            .add_systems(Update, (respawn_roster_escorts, spawn_escort_ships).chain());
        let player = app
            .world_mut()
            .spawn((
                Ship::from_ship_data(&ShipData::default(), "fighter"),
                Player,
                Position(Vec2::ZERO),
                avian2d::prelude::LinearVelocity(Vec2::ZERO),
                Transform::default(),
            ))
            .id();
        (app, player)
    }

    fn roster(app: &App) -> &EscortRoster {
        app.world().resource::<EscortRoster>()
    }

    #[test]
    fn player_bay_launch_enrolls_and_ai_launch_does_not() {
        let (mut app, player) = spawn_app();
        let ai_carrier = app
            .world_mut()
            .spawn((
                Ship::from_ship_data(&ShipData::default(), "pirate_carrier"),
                Position(Vec2::new(500.0, 0.0)),
                avian2d::prelude::LinearVelocity(Vec2::ZERO),
                Transform::default(),
            ))
            .id();
        for mother in [player, ai_carrier] {
            app.world_mut().write_message(SpawnEscort {
                mother,
                ship_type: "fighter".into(),
                carried: Some("fighter_bay".into()),
                position: Vec2::ZERO,
                mission: None,
                roster: None,
            });
        }
        app.update();
        let (entry_id, entry_kind) = {
            let entries = &roster(&app).entries;
            assert_eq!(entries.len(), 1, "only the PLAYER's launch enrolls");
            (entries[0].id, entries[0].kind.clone())
        };
        assert_eq!(
            entry_kind,
            EscortKind::Carried {
                weapon_type: "fighter_bay".into()
            }
        );
        // The live entity is linked to the entry.
        let mut linked = app
            .world_mut()
            .query::<(&PersistentEscort, &Escort)>();
        let links: Vec<_> = linked.iter(app.world()).collect();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].0 .0, entry_id);
        assert_eq!(links[0].1.mother, player);
    }

    #[test]
    fn squadron_wings_never_enroll() {
        let (mut app, player) = spawn_app();
        app.world_mut().write_message(SpawnEscort {
            mother: player,
            ship_type: "fighter".into(),
            carried: None,
            position: Vec2::ZERO,
            mission: Some("battle".into()),
            roster: None,
        });
        app.update();
        assert!(roster(&app).entries.is_empty());
        let mut q = app.world_mut().query::<&PersistentEscort>();
        assert_eq!(q.iter(app.world()).count(), 0);
    }

    #[test]
    fn roster_respawns_the_flight_after_a_world_reset() {
        let (mut app, _player) = spawn_app();
        // A carried fighter and a loyal companion survive from "last system".
        let (carried_id, companion_id) = {
            let mut r = app.world_mut().resource_mut::<EscortRoster>();
            (
                r.add(
                    "fighter".into(),
                    EscortKind::Carried {
                        weapon_type: "fighter_bay".into(),
                    },
                    37,
                ),
                r.add(
                    "corvette".into(),
                    EscortKind::Companion {
                        name: "Vex Marlowe".into(),
                    },
                    64,
                ),
            )
        };
        app.update(); // respawn writes SpawnEscort
        app.update(); // spawner materializes them

        let mut q = app
            .world_mut()
            .query::<(&PersistentEscort, &Ship, Option<&CarriedBy>, Option<&MissionSquadron>)>();
        let escorts: Vec<_> = q.iter(app.world()).collect();
        assert_eq!(escorts.len(), 2, "both roster entries re-materialize");
        let carried = escorts.iter().find(|(p, ..)| p.0 == carried_id).unwrap();
        let companion = escorts.iter().find(|(p, ..)| p.0 == companion_id).unwrap();
        assert!(carried.2.is_some(), "carried escorts can dock again");
        assert!(
            companion.2.is_none(),
            "companions have no bay — they never dock"
        );
        assert!(
            carried.3.is_none() && companion.3.is_none(),
            "roster escorts are NOT mission squadrons (mission cleanup can't touch them)"
        );
        // Hull damage came through the jump.
        assert_eq!(carried.1.health, 37);
        assert_eq!(companion.1.health, 64);

        // No duplicates on later frames.
        app.update();
        app.update();
        let mut q = app.world_mut().query::<&PersistentEscort>();
        assert_eq!(q.iter(app.world()).count(), 2, "no duplicate respawns");

        // Simulate the jump: the per-system entities die, the roster lives.
        let entities: Vec<Entity> = {
            let mut q = app.world_mut().query_filtered::<Entity, With<PersistentEscort>>();
            q.iter(app.world()).collect()
        };
        for e in entities {
            app.world_mut().despawn(e);
        }
        app.update();
        app.update();
        let mut q = app.world_mut().query::<&PersistentEscort>();
        assert_eq!(
            q.iter(app.world()).count(),
            2,
            "the flight re-forms in the next system"
        );
    }

    #[test]
    fn health_syncs_into_the_roster() {
        let (mut app, _player) = spawn_app();
        app.add_systems(PostUpdate, sync_roster_health);
        let id = app.world_mut().resource_mut::<EscortRoster>().add(
            "fighter".into(),
            EscortKind::Companion {
                name: "Okonkwo".into(),
            },
            100,
        );
        app.update();
        app.update();
        // Battle damage on the live entity…
        let entity = {
            let mut q = app.world_mut().query_filtered::<Entity, With<PersistentEscort>>();
            q.single(app.world()).unwrap()
        };
        app.world_mut().get_mut::<Ship>(entity).unwrap().health = 41;
        app.update();
        // …lands in the roster, so it survives the next jump/save.
        assert_eq!(
            roster(&app).entries.iter().find(|e| e.id == id).unwrap().health,
            41
        );
    }

    #[test]
    fn dock_retires_the_entry_and_replenishes_the_bay() {
        let (mut app, player) = spawn_app();
        app.add_systems(PostUpdate, animate_escort_dock);
        // Give the mother a fighter bay with 0 rounds left.
        {
            let iu = iu();
            let mut ship = app.world_mut().get_mut::<Ship>(player).unwrap();
            ship.weapon_systems = crate::weapons::WeaponSystems::build(
                &std::collections::HashMap::from([("fighter_bay".to_string(), (1u8, Some(0u32)))]),
                &iu,
            );
        }
        let id = app.world_mut().resource_mut::<EscortRoster>().add(
            "fighter".into(),
            EscortKind::Carried {
                weapon_type: "fighter_bay".into(),
            },
            80,
        );
        // A docking escort right on top of the mother (dist 0 → despawn frame).
        let escort = app
            .world_mut()
            .spawn((
                Escort { mother: player },
                CarriedBy {
                    weapon_type: "fighter_bay".into(),
                },
                PersistentEscort(id),
                DockingEscort::for_tests(120.0, Vec2::splat(16.0)),
                EscortMode::Dock,
                Position(Vec2::ZERO),
                Sprite::default(),
                Ship::from_ship_data(&ShipData::default(), "fighter"),
                MaxLinearSpeed(100.0),
            ))
            .id();
        app.update();
        app.update();
        assert!(
            app.world().get_entity(escort).is_err(),
            "docked escort despawns"
        );
        assert!(
            roster(&app).entries.is_empty(),
            "docking retires the roster entry — it's back in the bay"
        );
        let mut ship = app.world_mut().get_mut::<Ship>(player).unwrap();
        assert_eq!(
            ship.weapon_systems
                .find_weapon("fighter_bay")
                .unwrap()
                .ammo_quantity,
            Some(1),
            "the bay gets its round back"
        );
    }

    #[test]
    fn death_retires_the_entry() {
        let (mut app, _player) = spawn_app();
        app.insert_resource(crate::ModelMode::Eval)
            .insert_resource(crate::config::RewardConfig::default())
            .add_message::<crate::ship::DamageShip>()
            .add_message::<crate::explosions::TriggerExplosion>()
            .add_message::<crate::pickups::PickupDrop>()
            .add_message::<crate::rl_collection::RLShipDied>()
            .add_message::<crate::rl_collection::RLReward>()
            .add_message::<crate::missions::ShipDestroyed>()
            .add_systems(PostUpdate, crate::ship::apply_damage);
        let id = app.world_mut().resource_mut::<EscortRoster>().add(
            "fighter".into(),
            EscortKind::Companion {
                name: "Brakespear".into(),
            },
            100,
        );
        app.update();
        app.update();
        let entity = {
            let mut q = app.world_mut().query_filtered::<Entity, With<PersistentEscort>>();
            q.single(app.world()).unwrap()
        };
        app.world_mut().write_message(crate::ship::DamageShip {
            entity,
            damage: 10_000.0,
        });
        app.update();
        assert!(
            roster(&app).entries.is_empty(),
            "a dead escort leaves the roster for good"
        );
        assert!(roster(&app).entries.iter().all(|e| e.id != id));
    }

    #[test]
    fn roster_round_trips_through_the_save() {
        let mut r = EscortRoster::default();
        r.add(
            "fighter".into(),
            EscortKind::Carried {
                weapon_type: "fighter_bay".into(),
            },
            55,
        );
        r.add(
            "corvette".into(),
            EscortKind::Companion {
                name: "Sable Dune".into(),
            },
            72,
        );
        let saved = r.to_save();
        let iu = iu();
        let restored = EscortRoster::from_save(saved, &iu);
        assert_eq!(restored.entries.len(), 2);
        assert_eq!(restored.entries[0].health, 55);
        assert_eq!(
            restored.entries[1].kind,
            EscortKind::Companion {
                name: "Sable Dune".into()
            }
        );
        // Fresh runtime ids, all distinct.
        assert_ne!(restored.entries[0].id, restored.entries[1].id);
    }
}
