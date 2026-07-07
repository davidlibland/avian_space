use std::collections::HashMap;

use super::log::{MissionCatalog, MissionLog, PlayerUnlocks};
use super::progress::{preconditions_met, requirements_met};
use super::types::*;
use crate::ship::{Ship, ShipData};

// ── Helpers ─────────────────────────────────────────────────────────────────

fn dummy_def() -> MissionDef {
    MissionDef {
        briefing: String::new(),
        success_text: String::new(),
        failure_text: String::new(),
        preconditions: Vec::new(),
        offer: OfferKind::Auto,
        start_effects: Vec::new(),
        objective: Objective::TravelToSystem {
            system: "sol".into(),
        },
        requires: Vec::new(),
        completion_effects: Vec::new(),
    }
}

fn ship_with_cargo(cargo_space: u16, cargo: &[(&str, u16)]) -> Ship {
    let mut ship = Ship::from_ship_data(
        &ShipData {
            cargo_space,
            ..Default::default()
        },
        "test_ship",
    );
    for (commodity, qty) in cargo {
        ship.cargo.insert(commodity.to_string(), *qty);
    }
    ship
}

// ── MissionLog ──────────────────────────────────────────────────────────────

#[test]
fn log_default_status_is_locked() {
    let log = MissionLog::default();
    assert_eq!(log.status("nonexistent"), MissionStatus::Locked);
}

#[test]
fn log_set_and_get() {
    let mut log = MissionLog::default();
    log.set("m1", MissionStatus::Completed);
    assert_eq!(log.status("m1"), MissionStatus::Completed);
    log.set("m1", MissionStatus::Failed);
    assert_eq!(log.status("m1"), MissionStatus::Failed);
}

// ── PlayerUnlocks ───────────────────────────────────────────────────────────

#[test]
fn unlocks_empty_has_nothing() {
    let unlocks = PlayerUnlocks::default();
    assert!(!unlocks.has("anything"));
}

#[test]
fn unlocks_insert_and_check() {
    let mut unlocks = PlayerUnlocks::default();
    unlocks.0.insert("mining_license".into());
    assert!(unlocks.has("mining_license"));
    assert!(!unlocks.has("other"));
}

#[test]
fn unlocks_idempotent() {
    let mut unlocks = PlayerUnlocks::default();
    unlocks.0.insert("x".into());
    unlocks.0.insert("x".into());
    assert_eq!(unlocks.0.len(), 1);
}

// ── preconditions_met ───────────────────────────────────────────────────────

#[test]
fn empty_preconditions_always_met() {
    let log = MissionLog::default();
    let unlocks = PlayerUnlocks::default();
    assert!(preconditions_met(&[], &log, &unlocks));
}

#[test]
fn completed_precondition_met_when_completed() {
    let mut log = MissionLog::default();
    log.set("m1", MissionStatus::Completed);
    let unlocks = PlayerUnlocks::default();
    let pres = vec![Precondition::Completed {
        mission: "m1".into(),
    }];
    assert!(preconditions_met(&pres, &log, &unlocks));
}

#[test]
fn completed_precondition_not_met_when_active() {
    let mut log = MissionLog::default();
    log.set(
        "m1",
        MissionStatus::Active(ObjectiveProgress::default()),
    );
    let unlocks = PlayerUnlocks::default();
    let pres = vec![Precondition::Completed {
        mission: "m1".into(),
    }];
    assert!(!preconditions_met(&pres, &log, &unlocks));
}

#[test]
fn completed_precondition_not_met_when_locked() {
    let log = MissionLog::default();
    let unlocks = PlayerUnlocks::default();
    let pres = vec![Precondition::Completed {
        mission: "m1".into(),
    }];
    assert!(!preconditions_met(&pres, &log, &unlocks));
}

#[test]
fn failed_precondition_met_when_failed() {
    let mut log = MissionLog::default();
    log.set("m1", MissionStatus::Failed);
    let unlocks = PlayerUnlocks::default();
    let pres = vec![Precondition::Failed {
        mission: "m1".into(),
    }];
    assert!(preconditions_met(&pres, &log, &unlocks));
}

#[test]
fn failed_precondition_not_met_when_completed() {
    let mut log = MissionLog::default();
    log.set("m1", MissionStatus::Completed);
    let unlocks = PlayerUnlocks::default();
    let pres = vec![Precondition::Failed {
        mission: "m1".into(),
    }];
    assert!(!preconditions_met(&pres, &log, &unlocks));
}

#[test]
fn has_unlock_precondition_met() {
    let log = MissionLog::default();
    let mut unlocks = PlayerUnlocks::default();
    unlocks.0.insert("mining_license".into());
    let pres = vec![Precondition::HasUnlock {
        name: "mining_license".into(),
    }];
    assert!(preconditions_met(&pres, &log, &unlocks));
}

#[test]
fn has_unlock_precondition_not_met() {
    let log = MissionLog::default();
    let unlocks = PlayerUnlocks::default();
    let pres = vec![Precondition::HasUnlock {
        name: "mining_license".into(),
    }];
    assert!(!preconditions_met(&pres, &log, &unlocks));
}

#[test]
fn mixed_preconditions_all_must_pass() {
    let mut log = MissionLog::default();
    log.set("m1", MissionStatus::Completed);
    let mut unlocks = PlayerUnlocks::default();
    unlocks.0.insert("flag".into());

    let pres = vec![
        Precondition::Completed {
            mission: "m1".into(),
        },
        Precondition::HasUnlock {
            name: "flag".into(),
        },
    ];
    assert!(preconditions_met(&pres, &log, &unlocks));

    // Remove one — should fail.
    let pres_missing = vec![
        Precondition::Completed {
            mission: "m1".into(),
        },
        Precondition::HasUnlock {
            name: "missing".into(),
        },
    ];
    assert!(!preconditions_met(&pres_missing, &log, &unlocks));
}

// ── requirements_met ────────────────────────────────────────────────────────

#[test]
fn empty_requirements_always_met() {
    let def = dummy_def();
    let ship = ship_with_cargo(10, &[]);
    let unlocks = PlayerUnlocks::default();
    assert!(requirements_met(&def, &ship, &unlocks));
}

#[test]
fn has_cargo_requirement_met() {
    let mut def = dummy_def();
    def.requires = vec![CompletionRequirement::HasCargo {
        commodity: "iron".into(),
        quantity: 5,
    }];
    let ship = ship_with_cargo(20, &[("iron", 5)]);
    let unlocks = PlayerUnlocks::default();
    assert!(requirements_met(&def, &ship, &unlocks));
}

#[test]
fn has_cargo_requirement_excess_ok() {
    let mut def = dummy_def();
    def.requires = vec![CompletionRequirement::HasCargo {
        commodity: "iron".into(),
        quantity: 5,
    }];
    let ship = ship_with_cargo(20, &[("iron", 10)]);
    let unlocks = PlayerUnlocks::default();
    assert!(requirements_met(&def, &ship, &unlocks));
}

#[test]
fn has_cargo_requirement_not_met_insufficient() {
    let mut def = dummy_def();
    def.requires = vec![CompletionRequirement::HasCargo {
        commodity: "iron".into(),
        quantity: 5,
    }];
    let ship = ship_with_cargo(20, &[("iron", 4)]);
    let unlocks = PlayerUnlocks::default();
    assert!(!requirements_met(&def, &ship, &unlocks));
}

#[test]
fn has_cargo_requirement_not_met_wrong_commodity() {
    let mut def = dummy_def();
    def.requires = vec![CompletionRequirement::HasCargo {
        commodity: "iron".into(),
        quantity: 5,
    }];
    let ship = ship_with_cargo(20, &[("gold", 10)]);
    let unlocks = PlayerUnlocks::default();
    assert!(!requirements_met(&def, &ship, &unlocks));
}

#[test]
fn has_cargo_requirement_not_met_empty_hold() {
    let mut def = dummy_def();
    def.requires = vec![CompletionRequirement::HasCargo {
        commodity: "iron".into(),
        quantity: 1,
    }];
    let ship = ship_with_cargo(20, &[]);
    let unlocks = PlayerUnlocks::default();
    assert!(!requirements_met(&def, &ship, &unlocks));
}

#[test]
fn has_unlock_requirement_met() {
    let mut def = dummy_def();
    def.requires = vec![CompletionRequirement::HasUnlock {
        name: "license".into(),
    }];
    let ship = ship_with_cargo(10, &[]);
    let mut unlocks = PlayerUnlocks::default();
    unlocks.0.insert("license".into());
    assert!(requirements_met(&def, &ship, &unlocks));
}

#[test]
fn has_unlock_requirement_not_met() {
    let mut def = dummy_def();
    def.requires = vec![CompletionRequirement::HasUnlock {
        name: "license".into(),
    }];
    let ship = ship_with_cargo(10, &[]);
    let unlocks = PlayerUnlocks::default();
    assert!(!requirements_met(&def, &ship, &unlocks));
}

#[test]
fn mixed_requirements_all_must_pass() {
    let mut def = dummy_def();
    def.requires = vec![
        CompletionRequirement::HasCargo {
            commodity: "iron".into(),
            quantity: 3,
        },
        CompletionRequirement::HasUnlock {
            name: "flag".into(),
        },
    ];
    let ship = ship_with_cargo(20, &[("iron", 5)]);
    let mut unlocks = PlayerUnlocks::default();
    unlocks.0.insert("flag".into());
    assert!(requirements_met(&def, &ship, &unlocks));

    // Missing unlock → fails.
    let empty_unlocks = PlayerUnlocks::default();
    assert!(!requirements_met(&def, &ship, &empty_unlocks));

    // Missing cargo → fails.
    let empty_ship = ship_with_cargo(20, &[]);
    assert!(!requirements_met(&def, &empty_ship, &unlocks));
}

// ── required_cargo_space ────────────────────────────────────────────────────

#[test]
fn no_effects_no_space() {
    let def = dummy_def();
    assert_eq!(def.required_cargo_space(), 0);
}

#[test]
fn load_cargo_counts() {
    let mut def = dummy_def();
    def.start_effects = vec![
        StartEffect::LoadCargo {
            commodity: "food".into(),
            quantity: 10,
            reserved: true,
        },
        StartEffect::LoadCargo {
            commodity: "iron".into(),
            quantity: 5,
            reserved: false,
        },
    ];
    assert_eq!(def.required_cargo_space(), 15);
}

#[test]
fn collect_pickups_objective_counts() {
    let mut def = dummy_def();
    def.objective = Objective::CollectPickups {
        commodity: "iron".into(),
        system: "sol".into(),
        quantity: 8,
    };
    assert_eq!(def.required_cargo_space(), 8);
}

#[test]
fn destroy_ships_collect_counts() {
    let mut def = dummy_def();
    def.objective = Objective::DestroyShips {
        system: "sol".into(),
        ship_type: "pirate".into(),
        count: 3,
        target_name: "Pirates".into(),
        hostile: true,
        collect: Some(CollectRequirement {
            commodity: "iron".into(),
            quantity: 5,
        }),
    };
    assert_eq!(def.required_cargo_space(), 5);
}

#[test]
fn destroy_ships_no_collect_zero_space() {
    let mut def = dummy_def();
    def.objective = Objective::DestroyShips {
        system: "sol".into(),
        ship_type: "pirate".into(),
        count: 3,
        target_name: "Pirates".into(),
        hostile: true,
        collect: None,
    };
    assert_eq!(def.required_cargo_space(), 0);
}

#[test]
fn load_cargo_plus_collect_objective_additive() {
    let mut def = dummy_def();
    def.start_effects = vec![StartEffect::LoadCargo {
        commodity: "food".into(),
        quantity: 10,
        reserved: true,
    }];
    def.objective = Objective::CollectPickups {
        commodity: "iron".into(),
        system: "sol".into(),
        quantity: 5,
    };
    assert_eq!(def.required_cargo_space(), 15);
}

#[test]
fn travel_and_land_objectives_zero_space() {
    let mut def = dummy_def();
    def.objective = Objective::TravelToSystem {
        system: "sol".into(),
    };
    assert_eq!(def.required_cargo_space(), 0);
    def.objective = Objective::LandOnPlanet {
        planet: "earth".into(),
    };
    assert_eq!(def.required_cargo_space(), 0);
}

// ── Serde round-trips ───────────────────────────────────────────────────────

#[test]
fn mission_status_serde_roundtrip() {
    let statuses = vec![
        MissionStatus::Locked,
        MissionStatus::Available,
        MissionStatus::Active(ObjectiveProgress {
            collected: 3,
            destroyed: 2,
        }),
        MissionStatus::Completed,
        MissionStatus::Failed,
    ];
    for s in &statuses {
        let yaml = serde_yaml::to_string(s).unwrap();
        let restored: MissionStatus = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(*s, restored, "Failed roundtrip for {s:?}");
    }
}

#[test]
fn objective_progress_default_is_zeroed() {
    let p = ObjectiveProgress::default();
    assert_eq!(p.collected, 0);
    assert_eq!(p.destroyed, 0);
}

#[test]
fn mission_def_yaml_roundtrip_travel() {
    let def = MissionDef {
        briefing: "Go to sol.".into(),
        success_text: "Done.".into(),
        failure_text: "Failed.".into(),
        preconditions: vec![Precondition::Completed {
            mission: "intro".into(),
        }],
        offer: OfferKind::Auto,
        start_effects: Vec::new(),
        objective: Objective::TravelToSystem {
            system: "sol".into(),
        },
        requires: Vec::new(),
        completion_effects: vec![CompletionEffect::Pay { credits: 1000 }],
    };
    let yaml = serde_yaml::to_string(&def).unwrap();
    let restored: MissionDef = serde_yaml::from_str(&yaml).unwrap();
    assert_eq!(def, restored);
}

#[test]
fn mission_def_yaml_roundtrip_land_with_requires() {
    let def = MissionDef {
        briefing: "Deliver.".into(),
        success_text: "Done.".into(),
        failure_text: "Failed.".into(),
        preconditions: Vec::new(),
        offer: OfferKind::NpcOffer {
            planet: "earth".into(),
            weight: 0.5,
            building: None,
            approach: Default::default(),
        },
        start_effects: vec![StartEffect::LoadCargo {
            commodity: "food".into(),
            quantity: 10,
            reserved: true,
        }],
        objective: Objective::LandOnPlanet {
            planet: "mars".into(),
        },
        requires: vec![CompletionRequirement::HasCargo {
            commodity: "food".into(),
            quantity: 10,
        }],
        completion_effects: vec![
            CompletionEffect::RemoveCargo {
                commodity: "food".into(),
                quantity: 10,
            },
            CompletionEffect::Pay { credits: 5000 },
            CompletionEffect::GrantUnlock {
                name: "flag".into(),
            },
        ],
    };
    let yaml = serde_yaml::to_string(&def).unwrap();
    let restored: MissionDef = serde_yaml::from_str(&yaml).unwrap();
    assert_eq!(def, restored);
}

#[test]
fn mission_def_yaml_roundtrip_collect() {
    let def = MissionDef {
        briefing: "Collect.".into(),
        success_text: "Done.".into(),
        failure_text: "Failed.".into(),
        preconditions: vec![Precondition::HasUnlock {
            name: "mining".into(),
        }],
        offer: OfferKind::Tab { weight: 0.4 },
        start_effects: Vec::new(),
        objective: Objective::CollectPickups {
            commodity: "iron".into(),
            system: "sol".into(),
            quantity: 5,
        },
        requires: Vec::new(),
        completion_effects: Vec::new(),
    };
    let yaml = serde_yaml::to_string(&def).unwrap();
    let restored: MissionDef = serde_yaml::from_str(&yaml).unwrap();
    assert_eq!(def, restored);
}

#[test]
fn mission_def_yaml_roundtrip_destroy_ships() {
    let def = MissionDef {
        briefing: "Kill.".into(),
        success_text: "Done.".into(),
        failure_text: "Failed.".into(),
        preconditions: Vec::new(),
        offer: OfferKind::Auto,
        start_effects: Vec::new(),
        objective: Objective::DestroyShips {
            system: "sol".into(),
            ship_type: "pirate_corvette".into(),
            count: 3,
            target_name: "Pirates".into(),
            hostile: true,
            collect: Some(CollectRequirement {
                commodity: "iron".into(),
                quantity: 5,
            }),
        },
        requires: Vec::new(),
        completion_effects: vec![CompletionEffect::Pay { credits: 10000 }],
    };
    let yaml = serde_yaml::to_string(&def).unwrap();
    let restored: MissionDef = serde_yaml::from_str(&yaml).unwrap();
    assert_eq!(def, restored);
}

#[test]
fn mission_def_yaml_roundtrip_destroy_ships_no_collect() {
    let def = MissionDef {
        briefing: "Kill.".into(),
        success_text: "Done.".into(),
        failure_text: "Failed.".into(),
        preconditions: Vec::new(),
        offer: OfferKind::Auto,
        start_effects: Vec::new(),
        objective: Objective::DestroyShips {
            system: "drift".into(),
            ship_type: "pirate_corvette".into(),
            count: 1,
            target_name: "Pirate".into(),
            hostile: false,
            collect: None,
        },
        requires: Vec::new(),
        completion_effects: Vec::new(),
    };
    let yaml = serde_yaml::to_string(&def).unwrap();
    let restored: MissionDef = serde_yaml::from_str(&yaml).unwrap();
    assert_eq!(def, restored);
}

// ── Mission state machine scenarios ─────────────────────────────────────────


#[test]
fn scenario_gated_mission_unlocks_after_prerequisite_completes() {
    let mut log = MissionLog::default();
    let unlocks = PlayerUnlocks::default();

    let pres = vec![Precondition::Completed {
        mission: "intro".into(),
    }];

    // Stage 2 is locked because intro is not yet done.
    assert!(!preconditions_met(&pres, &log, &unlocks));

    // Complete the intro.
    log.set("intro", MissionStatus::Completed);

    // Now stage 2 preconditions are met.
    assert!(preconditions_met(&pres, &log, &unlocks));
}

#[test]
fn scenario_auto_mission_chain() {
    let mut log = MissionLog::default();
    let unlocks = PlayerUnlocks::default();

    // Stage 1: no preconditions, Auto.
    let stage1 = MissionDef {
        preconditions: Vec::new(),
        offer: OfferKind::Auto,
        ..dummy_def()
    };

    // Stage 2: gated on stage 1 completion, Auto.
    let stage2 = MissionDef {
        preconditions: vec![Precondition::Completed {
            mission: "s1".into(),
        }],
        offer: OfferKind::Auto,
        ..dummy_def()
    };

    // Stage 1 preconditions met → auto-start.
    assert!(preconditions_met(&stage1.preconditions, &log, &unlocks));

    // Stage 2 not yet.
    assert!(!preconditions_met(&stage2.preconditions, &log, &unlocks));

    // Complete stage 1.
    log.set("s1", MissionStatus::Completed);

    // Stage 2 preconditions now met → auto-start.
    assert!(preconditions_met(&stage2.preconditions, &log, &unlocks));
}

#[test]
fn scenario_failed_mission_can_gate_branch() {
    let mut log = MissionLog::default();
    let unlocks = PlayerUnlocks::default();

    let branch = vec![Precondition::Failed {
        mission: "main".into(),
    }];

    assert!(!preconditions_met(&branch, &log, &unlocks));
    log.set("main", MissionStatus::Completed);
    assert!(!preconditions_met(&branch, &log, &unlocks));
    log.set("main", MissionStatus::Failed);
    assert!(preconditions_met(&branch, &log, &unlocks));
}

#[test]
fn scenario_completion_requirement_prevents_completion() {
    let mut def = dummy_def();
    def.requires = vec![CompletionRequirement::HasCargo {
        commodity: "food".into(),
        quantity: 10,
    }];
    let ship_insufficient = ship_with_cargo(20, &[("food", 5)]);
    let ship_sufficient = ship_with_cargo(20, &[("food", 10)]);
    let unlocks = PlayerUnlocks::default();

    assert!(!requirements_met(&def, &ship_insufficient, &unlocks));
    assert!(requirements_met(&def, &ship_sufficient, &unlocks));
}



// ── Ship.reserved_cargo integration ─────────────────────────────────────────

#[test]
fn sell_cargo_respects_reserved() {
    let mut ship = ship_with_cargo(20, &[("food", 10)]);
    ship.reserved_cargo.insert("food".into(), 8);
    ship.sell_cargo("food", 5, 100);
    // Can only sell 10 - 8 = 2 units (sellable), so selling 5 caps at 2.
    assert_eq!(*ship.cargo.get("food").unwrap(), 8);
}

#[test]
fn sell_cargo_all_reserved_sells_nothing() {
    let mut ship = ship_with_cargo(20, &[("food", 10)]);
    ship.reserved_cargo.insert("food".into(), 10);
    ship.sell_cargo("food", 10, 100);
    assert_eq!(*ship.cargo.get("food").unwrap(), 10);
    assert_eq!(ship.credits, 10000); // unchanged (from_ship_data default)
}

// ── YAML asset loading ──────────────────────────────────────────────────────

#[test]
fn assets_missions_yaml_parses() {
    let text =
        std::fs::read_to_string("assets/missions.yaml").expect("failed to read missions.yaml");
    let missions: HashMap<String, MissionDef> =
        serde_yaml::from_str(&text).expect("failed to parse missions.yaml");
    assert!(
        !missions.is_empty(),
        "missions.yaml should contain at least one mission"
    );
    // Spot-check a known mission.
    assert!(
        missions.contains_key("deliver_wheat_intro"),
        "expected deliver_wheat_intro in missions.yaml"
    );
}

#[test]
fn assets_mission_templates_yaml_parses() {
    let text = std::fs::read_to_string("assets/mission_templates.yaml")
        .expect("failed to read mission_templates.yaml");
    let templates: HashMap<String, MissionTemplate> =
        serde_yaml::from_str(&text).expect("failed to parse mission_templates.yaml");
    assert!(
        !templates.is_empty(),
        "mission_templates.yaml should contain at least one template"
    );
}

// ── Template instantiation picks fulfillable targets ────────────────────────
//
// Regression tests for procedurally-generated missions that could not be
// completed: destinations on uncolonised (unlandable) planets, and systems /
// asteroid fields inside the unreachable RL-training systems.

mod template_targets {
    use super::super::progress::instantiate_template;
    use super::*;
    use crate::asteroids::AsteroidFieldData;
    use crate::item_universe::{ItemUniverse, StarSystem};
    use crate::planets::PlanetData;
    use bevy::math::Vec2;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn planet(landable: bool) -> PlanetData {
        PlanetData {
            display_name: String::new(),
            planet_type: String::new(),
            tech_level: 0,
            explicit_outfitter: Vec::new(),
            explicit_shipyard: Vec::new(),
            uncolonized: !landable,
            faction: String::new(),
            sprite_handle: Default::default(),
            location: Vec2::ZERO,
            description: String::new(),
            commodities: if landable {
                HashMap::from([("food".to_string(), 10_i128)])
            } else {
                HashMap::new()
            },
            outfitter: vec![],
            shipyard: vec![],
            radius: 50.0,
            color: [0.0; 3],
        }
    }

    fn system(planets: &[(&str, bool)], field_commodity: Option<&str>) -> StarSystem {
        StarSystem {
            faction: String::new(),
            contestable: false,
            authored_traffic: false,
            display_name: String::new(),
            map_position: Vec2::ZERO,
            connections: vec![],
            planets: planets
                .iter()
                .map(|(n, landable)| (n.to_string(), planet(*landable)))
                .collect(),
            astroid_fields: field_commodity
                .map(|c| {
                    vec![AsteroidFieldData {
                        location: Vec2::ZERO,
                        radius: 100.0,
                        number: 5,
                        commodities: HashMap::from([(c.to_string(), 1.0)]),
                    }]
                })
                .unwrap_or_default(),
            ships: Default::default(),
        }
    }

    /// sol: earth/mars/venus landable + barren unlandable + iron field.
    /// simulator (a TRAINING system): landable sim_world + a GOLD field —
    /// gold exists nowhere else, so any "gold" pick means a training leak.
    fn mini_universe() -> ItemUniverse {
        ItemUniverse {
            weapons: HashMap::new(),
            ships: HashMap::new(),
            star_systems: HashMap::from([
                (
                    "sol".to_string(),
                    system(
                        &[
                            ("earth", true),
                            ("mars", true),
                            ("venus", true),
                            ("barren", false),
                        ],
                        Some("iron"),
                    ),
                ),
                (
                    "simulator".to_string(),
                    system(&[("sim_world", true)], Some("gold")),
                ),
            ]),
            simulator_system: None,
            escort_system: None,
            mining_system: None,
            outfitter_items: HashMap::new(),
            commodities: HashMap::new(),
            missions: HashMap::new(),
            mission_templates: HashMap::new(),
            global_average_price: HashMap::new(),
            global_minimum_price: HashMap::new(),
            system_commodity_best_planet_to_sell: HashMap::new(),
            system_planet_best_commodity_to_buy: HashMap::new(),
            planet_best_margin: HashMap::new(),
            planet_has_ammo_for: HashMap::new(),
            asteroid_field_expected_value: HashMap::new(),
            ship_credit_scale: HashMap::new(),
            starting_system: "sol".to_string(),
            starting_ship: "shuttle".to_string(),
            enemies: HashMap::new(),
            allies: HashMap::new(),
        }
    }

    fn delivery_template() -> MissionTemplate {
        MissionTemplate::Delivery {
            briefing: "take {quantity} {commodity} to {planet_display}".into(),
            success_text: "ok".into(),
            failure_text: "fail".into(),
            offer: OfferKind::Tab { weight: 1.0 },
            preconditions: vec![],
            commodity_pool: vec!["food".into()],
            quantity_range: (5, 10),
            pay_range: (100, 200),
            reserved: true,
        }
    }

    #[test]
    fn delivery_destination_is_landable_and_reachable() {
        let iu = mini_universe();
        for seed in 0..200 {
            let mut rng = StdRng::seed_from_u64(seed);
            let chain = instantiate_template("t", &delivery_template(), "earth", &iu, &mut rng);
            assert_eq!(chain.len(), 1);
            let Objective::LandOnPlanet { planet } = &chain[0].1.objective else {
                panic!("delivery must produce a LandOnPlanet objective");
            };
            assert!(
                planet == "mars" || planet == "venus",
                "seed {seed}: destination '{planet}' must be a landable, reachable \
                 planet other than the offer planet (not barren/sim_world/earth)"
            );
        }
    }

    #[test]
    fn delivery_unsatisfiable_returns_empty_not_panic() {
        // Only landable planet is the offer planet itself.
        let mut iu = mini_universe();
        iu.star_systems.insert(
            "sol".to_string(),
            system(&[("earth", true), ("barren", false)], None),
        );
        iu.star_systems.remove("simulator");
        let mut rng = StdRng::seed_from_u64(7);
        let chain = instantiate_template("t", &delivery_template(), "earth", &iu, &mut rng);
        assert!(chain.is_empty(), "no valid destination → no mission");
    }

    #[test]
    fn collect_field_never_in_training_system() {
        let iu = mini_universe();
        let template = MissionTemplate::CollectFromAsteroidField {
            briefing: "mine {quantity} {commodity} in {system_display}".into(),
            success_text: "ok".into(),
            failure_text: "fail".into(),
            offer: OfferKind::Tab { weight: 1.0 },
            preconditions: vec![],
            quantity_range: (3, 6),
            pay_range: (100, 200),
        };
        for seed in 0..200 {
            let mut rng = StdRng::seed_from_u64(seed);
            let chain = instantiate_template("t", &template, "earth", &iu, &mut rng);
            assert_eq!(chain.len(), 1);
            let Objective::CollectPickups {
                system, commodity, ..
            } = &chain[0].1.objective
            else {
                panic!("collect template must produce CollectPickups");
            };
            assert_eq!(system, "sol", "seed {seed}: unreachable training system picked");
            assert_eq!(commodity, "iron", "seed {seed}: training-only commodity picked");
        }
    }

    #[test]
    fn bounty_system_is_reachable() {
        let iu = mini_universe();
        let template = MissionTemplate::BountyHunt {
            briefing: "kill {count} {target_name} in {system_display}".into(),
            success_text: "ok".into(),
            failure_text: "fail".into(),
            offer: OfferKind::Tab { weight: 1.0 },
            preconditions: vec![],
            ship_type_pool: vec!["pirate".into()],
            count_range: (1, 3),
            pay_range: (100, 200),
            target_name: "Pirates".into(),
        };
        for seed in 0..200 {
            let mut rng = StdRng::seed_from_u64(seed);
            let chain = instantiate_template("t", &template, "earth", &iu, &mut rng);
            assert_eq!(chain.len(), 1);
            let Objective::DestroyShips { system, .. } = &chain[0].1.objective else {
                panic!("bounty template must produce DestroyShips");
            };
            assert_eq!(system, "sol", "seed {seed}: bounty sent to training system");
        }
    }

    #[test]
    fn catch_thief_planets_landable_distinct_and_chained() {
        let iu = mini_universe();
        let template = MissionTemplate::CatchThief {
            stage1_briefing: "s1".into(),
            stage1_success_text: "ok".into(),
            stage1_failure_text: "fail".into(),
            stage2_briefing: "s2".into(),
            stage2_success_text: "ok".into(),
            stage2_failure_text: "fail".into(),
            stage3_briefing: "s3".into(),
            stage3_success_text: "ok".into(),
            stage3_failure_text: "fail".into(),
            offer: OfferKind::Tab { weight: 1.0 },
            preconditions: vec![],
            ship_type_pool: vec!["pirate".into()],
            target_name: "The Thief".into(),
            commodity_pool: vec!["food".into()],
            quantity_range: (2, 4),
            pay_range: (100, 200),
        };
        for seed in 0..200 {
            let mut rng = StdRng::seed_from_u64(seed);
            let chain = instantiate_template("t", &template, "earth", &iu, &mut rng);
            assert_eq!(chain.len(), 3);
            let (s1_id, s1) = &chain[0];
            let (s2_id, s2) = &chain[1];
            let (_s3_id, s3) = &chain[2];

            let Objective::DestroyShips { system, .. } = &s1.objective else {
                panic!("stage 1 must be DestroyShips");
            };
            assert_eq!(system, "sol");

            let Objective::LandOnPlanet { planet: deliver } = &s2.objective else {
                panic!("stage 2 must be LandOnPlanet");
            };
            let Objective::CatchNpc { planet: chase, .. } = &s3.objective else {
                panic!("stage 3 must be CatchNpc");
            };
            for p in [deliver, chase] {
                assert!(
                    p == "mars" || p == "venus",
                    "seed {seed}: '{p}' must be landable, reachable, != offer planet"
                );
            }
            assert_ne!(deliver, chase, "seed {seed}: deliver/chase must differ");

            // Follow-ups gate on the previous stage's generated id.
            assert_eq!(
                s2.preconditions,
                vec![Precondition::Completed {
                    mission: s1_id.clone()
                }]
            );
            assert_eq!(
                s3.preconditions,
                vec![Precondition::Completed {
                    mission: s2_id.clone()
                }]
            );
        }
    }
}

// ── Toast queue ──────────────────────────────────────────────────────────────

mod toast_queue {
    use super::super::ui::MissionToast;

    #[test]
    fn queue_preserves_order_no_overwrite() {
        // Regression: a single-slot toast dropped one message whenever a
        // completion and an auto-start briefing landed in the same instant.
        let mut toast = MissionToast::default();
        toast.push("mission complete");
        toast.push("new mission briefing");
        assert_eq!(toast.queue.len(), 2);
        assert_eq!(toast.queue.front().map(String::as_str), Some("mission complete"));
        toast.queue.pop_front();
        assert_eq!(
            toast.queue.front().map(String::as_str),
            Some("new mission briefing")
        );
    }

    #[test]
    fn back_to_back_duplicates_dedup() {
        let mut toast = MissionToast::default();
        toast.push("same");
        toast.push("same");
        assert_eq!(toast.queue.len(), 1);
        toast.push("other");
        toast.push("same");
        assert_eq!(toast.queue.len(), 3, "only adjacent duplicates dedup");
    }
}

// ── Mission runtime: drive the REAL systems through messages ────────────────
//
// These exercise the actual Update-chain systems (accept flow, start effects,
// objective advancement, completion/failure effects) in a minimal App — not
// hand-mutated `log.set` reproductions of their logic.

mod runtime {
    use super::super::events::*;
    use super::super::log::{MissionCatalog, MissionLog, MissionOffers, PlayerUnlocks};
    use super::super::progress;
    use super::*;
    use crate::Player;
    use bevy::prelude::*;

    /// App wired with the mission chain (same order as `missions_plugin`) and
    /// a player ship with the given cargo space.
    fn missions_app(cargo_space: u16) -> (App, Entity) {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<MissionLog>()
            .init_resource::<MissionCatalog>()
            .init_resource::<MissionOffers>()
            .init_resource::<PlayerUnlocks>()
            .add_message::<PlayerLandedOnPlanet>()
            .add_message::<PlayerEnteredSystem>()
            .add_message::<PickupCollected>()
            .add_message::<ShipDestroyed>()
            .add_message::<AcceptMission>()
            .add_message::<DeclineMission>()
            .add_message::<AbandonMission>()
            .add_message::<MissionStarted>()
            .add_message::<MissionCompleted>()
            .add_message::<MissionFailed>()
            .add_message::<NpcMet>()
            .add_message::<NpcCaught>()
            .add_systems(
                Update,
                (
                    progress::update_locked_to_available,
                    progress::handle_ui_actions,
                    progress::apply_start_effects,
                    progress::advance_travel_objectives,
                    progress::advance_land_objectives,
                    progress::advance_destroy_objectives,
                    progress::advance_destroy_collect,
                    progress::finalize_completions,
                    progress::finalize_failures,
                )
                    .chain(),
            );
        let player = app
            .world_mut()
            .spawn((ship_with_cargo(cargo_space, &[]), Player))
            .id();
        (app, player)
    }

    fn insert_mission(app: &mut App, id: &str, def: MissionDef) {
        app.world_mut()
            .resource_mut::<MissionCatalog>()
            .defs
            .insert(id.to_string(), def);
    }

    fn status(app: &mut App, id: &str) -> MissionStatus {
        app.world().resource::<MissionLog>().status(id)
    }

    fn delivery_def() -> MissionDef {
        MissionDef {
            briefing: "b".into(),
            success_text: "s".into(),
            failure_text: "f".into(),
            preconditions: vec![],
            offer: OfferKind::Tab { weight: 1.0 },
            start_effects: vec![StartEffect::LoadCargo {
                commodity: "food".into(),
                quantity: 10,
                reserved: true,
            }],
            objective: Objective::LandOnPlanet {
                planet: "mars".into(),
            },
            requires: vec![CompletionRequirement::HasCargo {
                commodity: "food".into(),
                quantity: 10,
            }],
            completion_effects: vec![
                CompletionEffect::RemoveCargo {
                    commodity: "food".into(),
                    quantity: 10,
                },
                CompletionEffect::Pay { credits: 5000 },
                CompletionEffect::GrantUnlock {
                    name: "test_license".into(),
                },
            ],
        }
    }

    #[test]
    fn accept_starts_mission_and_loads_reserved_cargo() {
        let (mut app, player) = missions_app(20);
        insert_mission(&mut app, "m", delivery_def());
        app.update(); // update_locked_to_available flips Locked → Available
        assert!(matches!(status(&mut app, "m"), MissionStatus::Available));

        app.world_mut().write_message(AcceptMission("m".into()));
        app.update();
        assert!(matches!(status(&mut app, "m"), MissionStatus::Active(_)));
        let ship = app.world().get::<Ship>(player).unwrap();
        assert_eq!(ship.cargo.get("food"), Some(&10), "start effect loads cargo");
        assert_eq!(
            ship.reserved_cargo.get("food"),
            Some(&10),
            "mission cargo is reserved (unsellable)"
        );
    }

    #[test]
    fn accept_refused_when_hold_too_small() {
        // The defence-in-depth cargo gate in handle_ui_actions, not the UI.
        let (mut app, player) = missions_app(5); // < the 10 the mission loads
        insert_mission(&mut app, "m", delivery_def());
        app.update();
        app.world_mut().write_message(AcceptMission("m".into()));
        app.update();
        assert!(
            matches!(status(&mut app, "m"), MissionStatus::Available),
            "accept must be refused — the hold cannot fit the start cargo"
        );
        let ship = app.world().get::<Ship>(player).unwrap();
        assert!(ship.cargo.is_empty(), "no cargo may be loaded on refusal");
    }

    #[test]
    fn landing_completes_delivery_and_applies_all_effects() {
        let (mut app, player) = missions_app(20);
        insert_mission(&mut app, "m", delivery_def());
        app.update();
        app.world_mut().write_message(AcceptMission("m".into()));
        app.update();

        app.world_mut().write_message(PlayerLandedOnPlanet {
            planet: "mars".into(),
        });
        app.update();

        assert!(matches!(status(&mut app, "m"), MissionStatus::Completed));
        let ship = app.world().get::<Ship>(player).unwrap();
        assert_eq!(ship.cargo.get("food"), None, "delivered cargo removed");
        assert_eq!(ship.reserved_cargo.get("food"), None, "reservation cleared");
        // from_ship_data starts ships at 10_000 credits; the mission pays 5_000.
        assert_eq!(ship.credits, 15_000, "payment applied on top of starting credits");
        assert!(
            app.world().resource::<PlayerUnlocks>().has("test_license"),
            "grant_unlock applied"
        );
    }

    #[test]
    fn landing_without_cargo_fails_the_delivery() {
        let (mut app, player) = missions_app(20);
        insert_mission(&mut app, "m", delivery_def());
        app.update();
        app.world_mut().write_message(AcceptMission("m".into()));
        app.update();
        // Lose the cargo (jettison/sell equivalent), then land.
        app.world_mut().get_mut::<Ship>(player).unwrap().cargo.clear();
        app.world_mut().write_message(PlayerLandedOnPlanet {
            planet: "mars".into(),
        });
        app.update();
        assert!(
            matches!(status(&mut app, "m"), MissionStatus::Failed),
            "landing at the destination without the goods fails the mission"
        );
    }

    #[test]
    fn abandon_strips_mission_cargo_but_keeps_own() {
        // Anti-exploit: accept a delivery, abandon it, keep the free cargo.
        let (mut app, player) = missions_app(20);
        insert_mission(&mut app, "m", delivery_def());
        // Player already owns 3 food of their own.
        app.world_mut()
            .get_mut::<Ship>(player)
            .unwrap()
            .cargo
            .insert("food".into(), 3);
        app.update();
        app.world_mut().write_message(AcceptMission("m".into()));
        app.update(); // now holds 3 own + 10 mission food

        app.world_mut().write_message(AbandonMission("m".into()));
        app.update();

        assert!(matches!(status(&mut app, "m"), MissionStatus::Failed));
        let ship = app.world().get::<Ship>(player).unwrap();
        assert_eq!(
            ship.cargo.get("food"),
            Some(&3),
            "mission cargo stripped, player's own 3 kept"
        );
        assert!(
            ship.reserved_cargo.get("food").is_none(),
            "reservation fully cleared"
        );
    }

    #[test]
    fn auto_chain_cascades_through_real_transitions() {
        // A completed prerequisite must flip an unlock-gated mission to
        // Available and an Auto follow-up to Active (with MissionStarted) —
        // via update_locked_to_available's fixpoint, not hand-set statuses.
        let (mut app, _player) = missions_app(20);
        let mut auto_follow = dummy_def();
        auto_follow.preconditions = vec![Precondition::Completed {
            mission: "first".into(),
        }];
        auto_follow.offer = OfferKind::Auto;
        let mut offered_follow = dummy_def();
        offered_follow.preconditions = vec![Precondition::Completed {
            mission: "first".into(),
        }];
        offered_follow.offer = OfferKind::Tab { weight: 1.0 };

        let mut first = delivery_def();
        first.requires.clear();
        first.start_effects.clear();
        first.completion_effects.clear();
        insert_mission(&mut app, "first", first);
        insert_mission(&mut app, "auto_follow", auto_follow);
        insert_mission(&mut app, "offered_follow", offered_follow);

        app.update();
        assert!(matches!(status(&mut app, "auto_follow"), MissionStatus::Locked));

        app.world_mut().write_message(AcceptMission("first".into()));
        app.update();
        app.world_mut().write_message(PlayerLandedOnPlanet {
            planet: "mars".into(),
        });
        app.update(); // completes "first"
        app.update(); // next pass flips the followups

        assert!(
            matches!(status(&mut app, "auto_follow"), MissionStatus::Active(_)),
            "Auto follow-up must self-start once its prerequisite completes"
        );
        assert!(
            matches!(status(&mut app, "offered_follow"), MissionStatus::Available),
            "offered follow-up becomes Available (not started)"
        );
    }

    #[test]
    fn destroy_with_collect_requires_both() {
        // Replaces the old tautological scenario test: this drives
        // advance_destroy_objectives / advance_destroy_collect with real
        // ShipDestroyed / PickupCollected messages.
        let (mut app, _player) = missions_app(20);
        let mut def = dummy_def();
        def.objective = Objective::DestroyShips {
            system: "sol".into(),
            ship_type: "pirate".into(),
            count: 2,
            target_name: "Raiders".into(),
            hostile: true,
            collect: Some(CollectRequirement {
                commodity: "iron".into(),
                quantity: 3,
            }),
        };
        def.offer = OfferKind::Auto;
        insert_mission(&mut app, "m", def);
        app.update(); // auto-starts

        let kill = |app: &mut App| {
            let e = Entity::PLACEHOLDER;
            app.world_mut().write_message(ShipDestroyed {
                entity: e,
                mission_target: Some(MissionTarget {
                    mission_id: "m".into(),
                    display_name: "Raider".into(),
                    always_targets_player: true,
                }),
            });
            app.update();
        };
        let collect = |app: &mut App, qty: u16| {
            app.world_mut().write_message(PickupCollected {
                commodity: "iron".into(),
                quantity: qty,
                system: "sol".into(),
            });
            app.update();
        };

        kill(&mut app);
        kill(&mut app);
        match status(&mut app, "m") {
            MissionStatus::Active(p) => {
                assert_eq!(p.destroyed, 2, "kills tracked");
                assert_eq!(p.collected, 0);
            }
            s => panic!("kills alone must not complete the mission: {s:?}"),
        }

        collect(&mut app, 2);
        assert!(
            matches!(status(&mut app, "m"), MissionStatus::Active(_)),
            "partial collect must not complete"
        );
        collect(&mut app, 1);
        assert!(
            matches!(status(&mut app, "m"), MissionStatus::Completed),
            "kills + full collect completes"
        );
    }

    #[test]
    fn new_offer_rolls_mid_visit_exactly_once() {
        // roll_new_offers_while_landed: a mission that becomes Available
        // while the player is landed gets one roll on the spot (weight 1.0 →
        // guaranteed offer) and is not re-added on later frames.
        let mut app = App::new();
        let mut app_galaxy: Option<crate::galaxy::GalaxyControl> = None;
        app.add_plugins(MinimalPlugins)
            .init_resource::<MissionLog>()
            .init_resource::<MissionCatalog>()
            .init_resource::<MissionOffers>()
            .init_resource::<PlayerUnlocks>()
            .init_resource::<crate::standing::FactionStandings>()
            .insert_resource({
                let mut iu = crate::item_universe::parse_dir::<crate::item_universe::ItemUniverse>(
                    std::path::Path::new("assets"),
                )
                .expect("assets/ must parse");
                iu.finalize();
                app_galaxy = Some(crate::galaxy::GalaxyControl::seeded_from(&iu));
                iu
            })
            .insert_resource(crate::planet_ui::LandedContext {
                planet_name: Some("earth".into()),
            })
            .add_systems(Update, progress::roll_new_offers_while_landed);
        app.insert_resource(app_galaxy.take().unwrap());

        let mut def = dummy_def();
        def.offer = OfferKind::NpcOffer {
            planet: "earth".into(),
            weight: 1.0,
            building: None,
            approach: NpcApproach::Wait,
        };
        app.world_mut()
            .resource_mut::<MissionCatalog>()
            .defs
            .insert("m".into(), def);
        app.world_mut()
            .resource_mut::<MissionLog>()
            .set("m", MissionStatus::Available);

        app.update();
        let offers = app.world().resource::<MissionOffers>();
        assert_eq!(
            offers.npc.get("earth").map(|v| v.as_slice()),
            Some(&["m".to_string()][..]),
            "newly Available NpcOffer must appear without re-landing"
        );

        app.update();
        app.update();
        let offers = app.world().resource::<MissionOffers>();
        assert_eq!(
            offers.npc.get("earth").map(|v| v.len()),
            Some(1),
            "one roll per visit — never duplicated on later frames"
        );
    }
}
