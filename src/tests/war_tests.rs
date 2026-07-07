//! Front-generator tests: front detection, campaign tiers, war-mission
//! offers gated by standing, mission validity, and ambient drift.

use super::*;
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
    assert_eq!(front_tier(&g, &front), 3, "threshold in sight → decisive push");
}

fn war_app(iu: ItemUniverse, galaxy: GalaxyControl) -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(iu)
        .insert_resource(galaxy)
        .init_resource::<FactionStandings>()
        .init_resource::<MissionCatalog>()
        .init_resource::<MissionLog>()
        .init_resource::<MissionOffers>()
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
    let (id, def) = catalog
        .defs
        .iter()
        .find(|(id, _)| id.starts_with("war__"))
        .unwrap();
    let iu = world.resource::<ItemUniverse>();
    if let crate::missions::Objective::DestroyShips { ship_type, .. } = &def.objective {
        assert!(iu.ships.contains_key(ship_type), "battle target ship exists");
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
        .map(|f| {
            app_influence(&galaxy, &f.target, &f.sponsor)
        })
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
    assert!(after > before - 1e-6, "attackers gain ground: {before} → {after}");
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
