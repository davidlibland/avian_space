//! Galactic-control tests: simplex math, controller hysteresis, seeding,
//! market/traffic re-derivation, and the ShiftInfluence mission effect driven
//! through the real systems.

use super::*;
use crate::missions::types::{CompletionEffect, MissionDef, Objective, OfferKind};
use crate::missions::{MissionCatalog, MissionCompleted};
use std::path::Path;

fn universe() -> ItemUniverse {
    let mut iu: ItemUniverse =
        crate::item_universe::parse_dir(Path::new("assets")).expect("assets/ must parse");
    iu.finalize();
    iu
}

// ── Simplex math ─────────────────────────────────────────────────────────────

#[test]
fn shift_draws_from_unaligned_then_other_factions() {
    let mut g = GalaxyControl::default();
    // 0.5 Federation, 0.5 unaligned.
    g.apply_shift("s", "Federation", 0.5);
    assert!((g.influence_of("s", "Federation") - 0.5).abs() < 1e-5);

    // +0.3 Rebel fits in the unaligned remainder — Federation untouched.
    g.apply_shift("s", "Rebel", 0.3);
    assert!((g.influence_of("s", "Federation") - 0.5).abs() < 1e-5);
    assert!((g.influence_of("s", "Rebel") - 0.3).abs() < 1e-5);

    // +0.4 more Rebel: 0.2 comes from unaligned, the excess from Federation.
    g.apply_shift("s", "Rebel", 0.4);
    let fed = g.influence_of("s", "Federation");
    let reb = g.influence_of("s", "Rebel");
    assert!((reb - 0.7).abs() < 1e-5, "rebel {reb}");
    assert!(fed < 0.5 - 1e-6, "federation must shrink, got {fed}");
    assert!(fed + reb <= 1.0 + 1e-5, "simplex must stay valid");

    // Decrease just releases share back to unaligned.
    g.apply_shift("s", "Rebel", -0.5);
    assert!((g.influence_of("s", "Rebel") - 0.2).abs() < 1e-5);
    let sum: f32 = g.influence.get("s").unwrap().values().sum();
    assert!(sum < 1.0, "released share becomes unaligned");
}

#[test]
fn shift_clamps_and_prunes() {
    let mut g = GalaxyControl::default();
    g.apply_shift("s", "Rebel", 5.0);
    assert!((g.influence_of("s", "Rebel") - 1.0).abs() < 1e-6);
    g.apply_shift("s", "Rebel", -9.0);
    assert_eq!(g.influence_of("s", "Rebel"), 0.0);
    assert!(
        !g.influence.get("s").unwrap().contains_key("Rebel"),
        "zeroed entries are pruned"
    );
}

// ── Controller hysteresis ────────────────────────────────────────────────────

#[test]
fn control_gained_at_60_kept_to_50() {
    let mut g = GalaxyControl::default();

    g.apply_shift("s", "Rebel", 0.55);
    assert!(g.recompute_controller("s").is_none(), "0.55 < gain threshold");
    assert_eq!(g.controller("s"), None);

    g.apply_shift("s", "Rebel", 0.1); // 0.65 → gain
    assert_eq!(g.recompute_controller("s"), Some(Some("Rebel".into())));
    assert_eq!(g.controller("s"), Some("Rebel"));

    g.apply_shift("s", "Rebel", -0.1); // 0.55 → still held (hysteresis)
    assert!(g.recompute_controller("s").is_none());
    assert_eq!(g.controller("s"), Some("Rebel"));

    g.apply_shift("s", "Rebel", -0.1); // 0.45 → lost, contested
    assert_eq!(g.recompute_controller("s"), Some(None));
    assert_eq!(g.controller("s"), None);
}

#[test]
fn challenger_takes_over_only_at_gain_threshold() {
    let mut g = GalaxyControl::default();
    g.apply_shift("s", "Federation", 1.0);
    g.recompute_controller("s");
    assert_eq!(g.controller("s"), Some("Federation"));

    // Rebel pushes to 0.62; Federation drops to ~0.38 (below keep).
    g.apply_shift("s", "Rebel", 0.62);
    assert_eq!(
        g.recompute_controller("s"),
        Some(Some("Rebel".into())),
        "challenger above GAIN takes over once the holder is below KEEP"
    );
}

// ── Seeding + effective factions ─────────────────────────────────────────────

#[test]
fn seeded_galaxy_matches_static_assets() {
    let iu = universe();
    let g = GalaxyControl::seeded_from(&iu);
    assert_eq!(g.controller("sol"), Some("Federation"));
    assert_eq!(g.controller("procyon"), Some("Rebel"));
    assert!((g.influence_of("sol", "Federation") - 1.0).abs() < 1e-6);
    assert!(
        !g.influence.contains_key("simulator"),
        "training systems stay out of the galaxy"
    );

    assert_eq!(
        effective_planet_faction(&g, &iu, "earth").as_deref(),
        Some("Federation")
    );
    assert_eq!(
        effective_planet_faction(&g, &iu, "pluto"),
        None,
        "independent enclaves stay independent"
    );
}

// ── Market + traffic re-derivation ───────────────────────────────────────────

#[test]
fn control_flip_restocks_markets() {
    let mut iu = universe();
    let procyon_prime = |iu: &ItemUniverse| {
        iu.find_gameplay_planet("procyon_prime").unwrap().1.clone()
    };
    // Rebel-held: rebel gear on the shelves.
    let before = procyon_prime(&iu);
    assert!(before.outfitter.contains(&"javelin".to_string()));
    assert!(before.shipyard.contains(&"rebel_fighter".to_string()));

    // Federation takes the system.
    iu.rederive_system_market("procyon", Some("Federation"));
    let flipped = procyon_prime(&iu);
    assert!(!flipped.outfitter.contains(&"javelin".to_string()));
    assert!(!flipped.shipyard.contains(&"rebel_fighter".to_string()));
    assert!(flipped.shipyard.contains(&"fed_patrol".to_string()));
    assert!(
        flipped.outfitter.contains(&"laser".to_string()),
        "universal gear always stocked"
    );

    // Contested: markets degrade to universal-only.
    iu.rederive_system_market("procyon", None);
    let contested = procyon_prime(&iu);
    assert!(!contested.shipyard.contains(&"fed_patrol".to_string()));
    assert!(!contested.shipyard.contains(&"rebel_fighter".to_string()));
    assert!(contested.outfitter.contains(&"laser".to_string()));
    assert!(
        contested.shipyard.contains(&"shuttle".to_string()),
        "universal hulls still sold while contested"
    );
}

#[test]
fn derived_traffic_reflects_presence_and_wealth() {
    let iu = universe();
    let sol = &iu.star_systems.get("sol").unwrap().ships;
    assert!(!sol.types.is_empty(), "derived distribution must be non-empty");
    assert!(sol.min >= 1 && sol.min <= sol.max && sol.max <= 8);
    assert!(
        sol.types.contains_key("fed_patrol"),
        "controller's patrols present at home"
    );
    assert!(
        sol.types.contains_key("freighter"),
        "universal traders everywhere"
    );
    assert!(
        !sol.types.contains_key("helios_titan"),
        "distant factions don't patrol Sol"
    );
    // Small ships outweigh capitals within the same faction.
    let procyon = &iu.star_systems.get("procyon").unwrap().ships;
    let fighter = procyon.types.get("rebel_fighter").copied().unwrap_or(0.0);
    let carrier = procyon.types.get("rebel_carrier").copied().unwrap_or(0.0);
    assert!(
        fighter > carrier,
        "tech weighting: fighters ({fighter}) outnumber carriers ({carrier})"
    );

    // Pirates follow wealth in unaligned space.
    let drift = &iu.star_systems.get("drift").unwrap().ships;
    assert!(
        drift.types.keys().any(|k| {
            iu.ships.get(k).and_then(|s| s.faction.as_deref()) == Some("Pirate")
        }),
        "unaligned systems crawl with pirates: {:?}",
        drift.types.keys().collect::<Vec<_>>()
    );

    // The precursor rift keeps its authored distribution.
    let rift = iu.star_systems.get("the_rift").unwrap();
    assert!(rift.authored_traffic);
    assert!(rift.ships.types.keys().all(|k| k.starts_with("precursor_")));
}

#[test]
fn presence_bleeds_across_borders() {
    let iu = universe();
    let mut g = GalaxyControl::seeded_from(&iu);
    // Rebels seize the drift (adjacent to sol): rebel patrols should start
    // appearing in Sol's traffic via λ-propagation.
    let mut iu2 = universe();
    let before = iu2
        .star_systems
        .get("sol")
        .unwrap()
        .ships
        .types
        .get("rebel_fighter")
        .copied()
        .unwrap_or(0.0);
    g.apply_shift("drift", "Rebel", 1.0);
    iu2.rederive_ship_presence(&g);
    let after = iu2
        .star_systems
        .get("sol")
        .unwrap()
        .ships
        .types
        .get("rebel_fighter")
        .copied()
        .unwrap_or(0.0);
    assert!(
        after > before,
        "rebel presence next door must bleed into Sol: {before} → {after}"
    );
}

// ── ShiftInfluence through the real mission systems ──────────────────────────

#[test]
fn mission_shift_flips_system_and_restocks() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(universe())
        .init_resource::<MissionCatalog>()
        .add_message::<MissionCompleted>()
        .add_message::<SystemControlChanged>()
        .add_systems(
            Update,
            (
                influence_on_mission_complete,
                update_controllers.run_if(resource_changed::<GalaxyControl>),
                rederive_on_control_change,
            )
                .chain(),
        );
    let seeded = GalaxyControl::seeded_from(app.world().resource::<ItemUniverse>());
    app.insert_resource(seeded);

    let mk_mission = |system: &str, delta: f32| MissionDef {
        briefing: "b".into(),
        success_text: "s".into(),
        failure_text: "f".into(),
        preconditions: vec![],
        offer: OfferKind::Auto,
        start_effects: vec![],
        objective: Objective::TravelToSystem { system: "sol".into() },
        requires: vec![],
        completion_effects: vec![CompletionEffect::ShiftInfluence {
            system: system.into(),
            faction: "Federation".into(),
            delta,
        }],
        squadron: Vec::new(),
    };
    // Two war missions push Federation influence in (Rebel) procyon.
    for (id, delta) in [("war1", 0.5), ("war2", 0.35)] {
        app.world_mut()
            .resource_mut::<MissionCatalog>()
            .defs
            .insert(id.into(), mk_mission("procyon", delta));
    }
    // Also try to grab Sol for the Rebels — non-contestable, must be ignored.
    app.world_mut().resource_mut::<MissionCatalog>().defs.insert(
        "illegal".into(),
        MissionDef {
            completion_effects: vec![CompletionEffect::ShiftInfluence {
                system: "sol".into(),
                faction: "Rebel".into(),
                delta: 1.0,
            }],
            ..mk_mission("sol", 0.0)
        },
    );

    app.world_mut().write_message(MissionCompleted("war1".into()));
    app.update();
    app.update();
    {
        let g = app.world().resource::<GalaxyControl>();
        assert_eq!(
            g.controller("procyon"),
            Some("Rebel"),
            "0.5 Federation vs 0.5 Rebel — holder keeps at exactly KEEP"
        );
    }

    app.world_mut().write_message(MissionCompleted("war2".into()));
    app.update();
    app.update();
    {
        let g = app.world().resource::<GalaxyControl>();
        assert_eq!(g.controller("procyon"), Some("Federation"), "system flips");
    }
    // Markets restocked under the new controller.
    let iu = app.world().resource::<ItemUniverse>();
    let pp = iu
        .star_systems
        .get("procyon")
        .unwrap()
        .planets
        .get("procyon_prime")
        .unwrap();
    assert!(!pp.outfitter.contains(&"javelin".to_string()));
    assert!(pp.shipyard.contains(&"fed_patrol".to_string()));

    // The illegal grab on Sol did nothing.
    app.world_mut().write_message(MissionCompleted("illegal".into()));
    app.update();
    app.update();
    let g = app.world().resource::<GalaxyControl>();
    assert_eq!(g.controller("sol"), Some("Federation"));
    assert_eq!(g.influence_of("sol", "Rebel"), 0.0);
}
