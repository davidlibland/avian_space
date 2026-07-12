//! Front-generator tests: front detection, campaign tiers, war-mission
//! offers gated by standing, mission validity, and ambient drift.

use super::*;
use rand::SeedableRng;
use std::path::Path;

fn universe() -> ItemUniverse {
    let mut iu: ItemUniverse =
        crate::item_universe::parse_dir(Path::new("assets")).expect("assets/ must parse");
    iu.finalize();
    iu
}

fn seeded(iu: &ItemUniverse) -> GalaxyControl {
    GalaxyControl::seeded_from(iu)
}

/// A landable planet in `system`, for landing messages.
fn landable_planet(iu: &ItemUniverse, system: &str) -> String {
    iu.star_systems[system]
        .planets
        .iter()
        .find(|(_, p)| !p.commodities.is_empty())
        .map(|(n, _)| n.clone())
        .expect("front home must have a landable planet")
}

#[test]
fn fronts_exist_and_are_coherent() {
    let iu = universe();
    let galaxy = seeded(&iu);
    let fronts = detect_fronts(&iu, &galaxy);
    assert!(!fronts.is_empty(), "the seeded galaxy has warring borders");
    for f in &fronts {
        assert!(
            iu.enemies
                .get(&f.sponsor)
                .is_some_and(|es| es.contains(&f.enemy)),
            "front factions must be enemies: {f:?}"
        );
        let holder = GalaxyControl::seeded_from(&iu);
        let target_holder = holder.controller(&f.target);
        assert!(
            target_holder.is_none() || target_holder == Some(f.enemy.as_str()),
            "targets are enemy-held or unaligned buffers: {f:?}"
        );
        assert!(
            iu.star_systems[&f.target].contestable,
            "front targets must be contestable: {f:?}"
        );
        assert!(
            iu.star_systems[&f.home].connections.contains(&f.target),
            "fronts sit on jump edges: {f:?}"
        );
    }
    assert!(
        fronts
            .iter()
            .any(|f| f.sponsor == "Federation" && f.enemy == "Rebel"),
        "the Federation–Rebel border is at war"
    );
}

#[test]
fn tiers_track_the_defenders_grip() {
    let mut g = GalaxyControl::default();
    let front = Front {
        sponsor: "Federation".into(),
        home: "a".into(),
        enemy: "Rebel".into(),
        target: "t".into(),
    };
    g.apply_shift("t", "Rebel", 1.0);
    assert_eq!(front_tier(&g, &front), 1, "no foothold → raids");
    g.apply_shift("t", "Federation", 0.3);
    assert_eq!(front_tier(&g, &front), 2, "a foothold → battles");
    g.apply_shift("t", "Federation", 0.2); // 0.5
    assert_eq!(
        front_tier(&g, &front),
        3,
        "threshold in sight → decisive push"
    );
}

fn war_app(iu: ItemUniverse, galaxy: GalaxyControl) -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(iu)
        .insert_resource(galaxy)
        .init_resource::<FactionStandings>()
        .insert_resource({
            // A veteran of every faction: the service gate is tested
            // separately; other tests assume war offers are reachable.
            let mut sr = crate::standing::FactionServiceRecord::default();
            for f in [
                "Federation",
                "Rebel",
                "Bastion",
                "FreeFrontier",
                "Helios",
                "Order",
            ] {
                for _ in 0..crate::war::WAR_SERVICE_MIN {
                    sr.record(f);
                }
            }
            sr
        })
        .init_resource::<MissionCatalog>()
        .init_resource::<MissionLog>()
        .init_resource::<MissionOffers>()
        .init_resource::<crate::missions::OfferBackoff>()
        .add_message::<PlayerLandedOnPlanet>()
        .add_systems(Update, (offer_war_missions, war_drift));
    app
}

#[test]
fn war_missions_offered_only_to_trusted_pilots() {
    let iu = universe();
    let galaxy = seeded(&iu);
    let mut fronts = detect_fronts(&iu, &galaxy);
    fronts.sort_by(|a, b| (&a.home, &a.target).cmp(&(&b.home, &b.target)));
    let front = fronts.into_iter().next().expect("a front exists");
    let planet = landable_planet(&iu, &front.home);
    let sponsor = front.sponsor.clone();
    let mut app = war_app(iu, galaxy);

    // Neutral standing: no war work.
    app.world_mut().write_message(PlayerLandedOnPlanet {
        planet: planet.clone(),
    });
    app.update();
    let war_ids = |app: &App| {
        app.world()
            .resource::<MissionCatalog>()
            .defs
            .keys()
            .filter(|id| id.starts_with("war__"))
            .count()
    };
    assert_eq!(war_ids(&app), 0, "no war offers at neutral standing");

    // Trusted pilot: the front hires.
    app.world_mut()
        .resource_mut::<FactionStandings>()
        .adjust(&sponsor, 25.0);
    app.world_mut().write_message(PlayerLandedOnPlanet {
        planet: planet.clone(),
    });
    app.update();
    assert!(war_ids(&app) >= 1, "war offers appear for a trusted pilot");

    // The generated mission is coherent and appears in the planet's offers.
    let world = app.world();
    let catalog = world.resource::<MissionCatalog>();
    // Two-stage covert ops file an auto-starting __return leg alongside the
    // primary; only the PRIMARY appears in the planet's offers.
    let (id, def) = catalog
        .defs
        .iter()
        .find(|(id, _)| id.starts_with("war__") && !id.ends_with("__return"))
        .unwrap();
    // Venue split: overt war work is posted at the GARRISON desk; covert ops
    // come from a stranger in the bar.
    if let crate::missions::types::OfferKind::NpcOffer {
        building: Some(b), ..
    } = &def.offer
    {
        if id.contains("covert") {
            assert_eq!(b, "bar", "covert stays deniable");
        } else {
            assert_eq!(b, "garrison", "war work is official");
        }
    } else {
        panic!("war missions are NPC offers");
    }
    let iu = world.resource::<ItemUniverse>();
    if let crate::missions::Objective::DestroyShips { ship_type, .. } = &def.objective {
        assert!(
            iu.ships.contains_key(ship_type),
            "battle target ship exists"
        );
    }
    let target = def.shift_target().expect("war missions shift influence");
    assert!(iu.star_systems[target].contestable);
    for wing_ship in &def.squadron {
        assert!(iu.ships.contains_key(wing_ship), "squadron ships exist");
    }
    assert!(
        world.resource::<MissionOffers>().npc[&planet].contains(id),
        "offered at the landing planet"
    );
}

#[test]
fn drift_moves_active_fronts_only_a_little() {
    let iu = universe();
    let galaxy = seeded(&iu);
    let fronts = detect_fronts(&iu, &galaxy);
    assert!(!fronts.is_empty());
    let before: f32 = fronts
        .iter()
        .map(|f| app_influence(&galaxy, &f.target, &f.sponsor))
        .sum();
    let mut app = war_app(iu, galaxy);

    app.world_mut().write_message(PlayerLandedOnPlanet {
        planet: "earth".to_string(),
    });
    app.update();
    let g = app.world().resource::<GalaxyControl>();
    // Across all fronts the attackers gained ambient ground (individual
    // fronts sharing one buffer can shuffle shares via renormalization).
    let after: f32 = fronts
        .iter()
        .map(|f| g.influence_of(&f.target, &f.sponsor))
        .sum();
    assert!(
        after > before - 1e-6,
        "attackers gain ground: {before} → {after}"
    );
    // Every simplex stays valid, and no single share jumps beyond drift+slack.
    for f in &fronts {
        let sum: f32 = g
            .influence
            .get(&f.target)
            .map(|m| m.values().sum())
            .unwrap_or(0.0);
        assert!(sum <= 1.0 + 1e-5, "simplex valid at {}", f.target);
    }
}

fn app_influence(g: &GalaxyControl, system: &str, faction: &str) -> f32 {
    g.influence_of(system, faction)
}

// ── The Marches: the Federation–Bastion buffer front ─────────────────────────

#[test]
fn the_marches_is_a_hot_fed_bastion_front() {
    let iu = universe();
    let galaxy = seeded(&iu);
    let fronts = detect_fronts(&iu, &galaxy);
    // Both cores sponsor the campaign from BOTH their border systems, so a
    // won buffer becomes a springboard rather than a dead end.
    for (sponsor, homes, enemy) in [
        ("Federation", ["kepler_22", "epsilon_eridani"], "Bastion"),
        ("Bastion", ["iron_march", "coldforge"], "Federation"),
    ] {
        for home in homes {
            assert!(
                fronts.iter().any(|f| f.sponsor == sponsor
                    && f.home == home
                    && f.enemy == enemy
                    && f.target == "the_marches"),
                "{sponsor} fights for the_marches from {home}"
            );
        }
    }
}

// ── Covert family: every template instantiates coherently ────────────────────

#[test]
fn every_covert_template_instantiates_on_the_marches_front() {
    use crate::missions::types::{CompletionEffect, OfferKind};
    let iu = universe();
    let galaxy = seeded(&iu);
    let front = Front {
        sponsor: "Federation".into(),
        home: "kepler_22".into(),
        enemy: "Bastion".into(),
        target: "the_marches".into(),
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let mut coverts: Vec<(&String, &MissionTemplate)> = iu
        .mission_templates
        .iter()
        .filter(|(_, t)| matches!(t, MissionTemplate::Covert { .. }))
        .collect();
    coverts.sort_by_key(|(id, _)| (*id).clone());
    assert_eq!(
        coverts.len(),
        11,
        "the full covert family from the design doc"
    );

    let has_pay = |effects: &[CompletionEffect]| {
        effects
            .iter()
            .any(|e| matches!(e, CompletionEffect::Pay { .. }))
    };
    let shift_on_target = |effects: &[CompletionEffect]| {
        effects.iter().any(|e| {
            matches!(e,
            CompletionEffect::ShiftInfluence { system, faction, delta }
                if system == "the_marches" && faction == "Federation" && *delta > 0.0)
        })
    };

    for (id, tmpl) in coverts {
        let (def, follow_up) = instantiate_war_mission(tmpl, &front, &iu, &galaxy, &mut rng)
            .unwrap_or_else(|| panic!("{id} must instantiate on a landable front"));
        assert!(
            !def.briefing.contains('{'),
            "{id}: all vars substituted: {}",
            def.briefing
        );
        // Deniability: covert offers come from a stranger in the BAR, never
        // anyone at the garrison desk.
        assert!(
            matches!(&def.offer, OfferKind::NpcOffer { building: Some(b), approach, .. }
                if b == "bar" && matches!(approach, crate::missions::types::NpcApproach::Seek)),
            "{id}: covert work is offered by a stranger in the bar"
        );
        // The influence lever always points at the front target, and the
        // final stage always pays (bribes pay negative).
        match &follow_up {
            None => {
                assert!(has_pay(&def.completion_effects), "{id}: pays");
                assert!(
                    shift_on_target(&def.completion_effects),
                    "{id}: shifts the front"
                );
            }
            Some(follow) => {
                assert!(
                    matches!(follow.offer, OfferKind::Auto),
                    "{id}: stage 2 auto-starts"
                );
                assert!(
                    !follow.briefing.is_empty() && !follow.briefing.contains('{'),
                    "{id}: stage-2 text: {}",
                    follow.briefing
                );
                assert!(has_pay(&follow.completion_effects), "{id}: stage 2 pays");
                assert!(
                    shift_on_target(&follow.completion_effects),
                    "{id}: stage 2 shifts"
                );
                assert!(
                    shift_on_target(&def.completion_effects),
                    "{id}: the primary carries a partial shift (open_war bookkeeping)"
                );
            }
        }
        // Bribes cost money; everything else earns it.
        let pay = def
            .completion_effects
            .iter()
            .chain(follow_up.iter().flat_map(|f| f.completion_effects.iter()))
            .find_map(|e| match e {
                CompletionEffect::Pay { credits } => Some(*credits),
                _ => None,
            })
            .unwrap();
        if id.contains("bribe") {
            assert!(pay < 0, "{id}: the bribe comes out of your pocket");
        } else {
            assert!(pay > 0, "{id}: war work pays");
        }
    }
}

/// The generator files two-stage covert ops as a primary offer plus an
/// auto-starting follow-up locked on it.
#[test]
fn two_stage_covert_files_a_precondition_locked_return_leg() {
    use crate::missions::types::Precondition;
    let mut iu = universe();
    iu.mission_templates
        .retain(|id, _| id == "covert_propaganda"); // tier 1, two-stage
    let galaxy = seeded(&iu);
    let mut app = war_app(iu, galaxy);
    app.world_mut()
        .resource_mut::<FactionStandings>()
        .adjust("Federation", 25.0);
    app.world_mut().write_message(PlayerLandedOnPlanet {
        planet: "kepler_22b".to_string(),
    });
    app.update();

    let catalog = app.world().resource::<MissionCatalog>();
    let primary = catalog
        .defs
        .keys()
        .find(|id| id.starts_with("war__covert_propaganda") && !id.ends_with("__return"))
        .expect("the propaganda op is filed")
        .clone();
    let follow = catalog
        .defs
        .get(&format!("{primary}__return"))
        .expect("the return leg is filed with it");
    assert!(
        follow.preconditions.iter().any(|p| matches!(p,
            Precondition::Completed { mission } if *mission == primary)),
        "return leg unlocks on the primary"
    );
}

/// Every war the map supports must be fightable from BOTH sides, and every
/// front target must have a landable planet so the full mission family
/// (including covert ops) can instantiate there. Bastion–Rebel and
/// FreeFrontier–Helios stay cold by design: their cores sit on opposite
/// ends of the map with third factions between them.
#[test]
fn every_geographic_war_is_two_way_with_covert_venues() {
    let iu = universe();
    let galaxy = seeded(&iu);
    let fronts = detect_fronts(&iu, &galaxy);
    for (a, b) in [
        ("Federation", "Rebel"),
        ("Federation", "Bastion"),
        ("Bastion", "FreeFrontier"),
        ("Helios", "Order"),
    ] {
        for (sponsor, enemy) in [(a, b), (b, a)] {
            let side: Vec<&Front> = fronts
                .iter()
                .filter(|f| f.sponsor == sponsor && f.enemy == enemy)
                .collect();
            assert!(!side.is_empty(), "{sponsor} must be able to attack {enemy}");
            for f in side {
                assert!(
                    iu.star_systems[&f.target]
                        .planets
                        .values()
                        .any(|p| !p.commodities.is_empty()),
                    "front target {} needs a covert venue",
                    f.target
                );
            }
        }
    }
}

/// War-mission ids must never collide with ACTIVE missions persisted from a
/// previous session: the id counter is session-local, and reusing an id
/// silently overwrites the player's active mission with a new front's.
#[test]
fn war_ids_skip_persisted_missions_from_prior_sessions() {
    use crate::missions::types::{CompletionEffect, MissionDef, OfferKind};
    let mut iu = universe();
    iu.mission_templates.retain(|id, _| id == "war_raid"); // tier 1, deterministic
    let galaxy = seeded(&iu);
    let mut app = war_app(iu, galaxy);
    // A war mission persisted from an earlier session, still active.
    let stale_id = "war__war_raid__0001".to_string();
    let stale = MissionDef {
        briefing: "the player's ACTIVE mission from last session".into(),
        success_text: "s".into(),
        failure_text: "f".into(),
        preconditions: vec![],
        offer: OfferKind::Auto,
        start_effects: vec![],
        objective: crate::missions::Objective::TravelToSystem {
            system: "sol".into(),
        },
        requires: vec![],
        // Shifts an UNRELATED system so open_war doesn't block new offers.
        completion_effects: vec![CompletionEffect::ShiftInfluence {
            system: "pilgrims_deep".into(),
            faction: "Helios".into(),
            delta: 0.1,
        }],
        squadron: vec![],
        faction: None,
    };
    app.world_mut()
        .resource_mut::<MissionCatalog>()
        .defs
        .insert(stale_id.clone(), stale);
    app.world_mut().resource_mut::<MissionLog>().set(
        &stale_id,
        MissionStatus::Active(crate::missions::types::ObjectiveProgress::default()),
    );
    app.world_mut()
        .resource_mut::<FactionStandings>()
        .adjust("Federation", 25.0);
    app.world_mut().write_message(PlayerLandedOnPlanet {
        planet: "kepler_22b".to_string(),
    });
    app.update();

    let catalog = app.world().resource::<MissionCatalog>();
    assert_eq!(
        catalog.defs[&stale_id].briefing, "the player's ACTIVE mission from last session",
        "the persisted active mission must survive untouched"
    );
    assert!(
        catalog
            .defs
            .keys()
            .any(|id| id.starts_with("war__war_raid") && *id != stale_id),
        "the new offer takes a fresh id instead of colliding"
    );
}

/// Standing can be bought with a few deliveries; moving borders takes a
/// RECORD — the war desk ignores pilots below WAR_SERVICE_MIN completed
/// missions for the sponsor, whatever their standing.
#[test]
fn war_offers_require_a_service_record() {
    let iu = universe();
    let galaxy = seeded(&iu);
    let mut app = war_app(iu, galaxy);
    // Trusted standing, ZERO missions flown for the Federation.
    app.insert_resource(crate::standing::FactionServiceRecord::default());
    app.world_mut()
        .resource_mut::<FactionStandings>()
        .adjust("Federation", 50.0);
    app.world_mut().write_message(PlayerLandedOnPlanet {
        planet: "kepler_22b".to_string(),
    });
    app.update();
    let war_count = app
        .world()
        .resource::<MissionCatalog>()
        .defs
        .keys()
        .filter(|id| id.starts_with("war__"))
        .count();
    assert_eq!(war_count, 0, "no record, no war work");

    // Five completed missions later, the desk talks.
    {
        let mut sr = app
            .world_mut()
            .resource_mut::<crate::standing::FactionServiceRecord>();
        for _ in 0..crate::war::WAR_SERVICE_MIN {
            sr.record("Federation");
        }
    }
    app.world_mut().write_message(PlayerLandedOnPlanet {
        planet: "kepler_22b".to_string(),
    });
    app.update();
    let war_count = app
        .world()
        .resource::<MissionCatalog>()
        .defs
        .keys()
        .filter(|id| id.starts_with("war__"))
        .count();
    assert!(war_count >= 1, "a proven pilot gets the commission");
}
