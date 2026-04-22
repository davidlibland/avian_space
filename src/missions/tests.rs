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
fn scenario_locked_to_available_to_active_to_completed() {
    let mut log = MissionLog::default();
    let unlocks = PlayerUnlocks::default();
    let mut catalog = MissionCatalog::default();

    let def = MissionDef {
        briefing: "Intro".into(),
        success_text: "Done".into(),
        failure_text: "Fail".into(),
        preconditions: Vec::new(),
        offer: OfferKind::Tab { weight: 1.0 },
        start_effects: Vec::new(),
        objective: Objective::TravelToSystem {
            system: "sol".into(),
        },
        requires: Vec::new(),
        completion_effects: Vec::new(),
    };
    catalog.defs.insert("m1".into(), def);

    // Initially Locked.
    assert_eq!(log.status("m1"), MissionStatus::Locked);

    // Preconditions are met (empty) → should become Available.
    let def = catalog.defs.get("m1").unwrap();
    assert!(preconditions_met(&def.preconditions, &log, &unlocks));

    // Simulate: set to Available.
    log.set("m1", MissionStatus::Available);
    assert_eq!(log.status("m1"), MissionStatus::Available);

    // Accept: set to Active.
    log.set(
        "m1",
        MissionStatus::Active(ObjectiveProgress::default()),
    );
    assert!(matches!(log.status("m1"), MissionStatus::Active(_)));

    // Complete.
    log.set("m1", MissionStatus::Completed);
    assert_eq!(log.status("m1"), MissionStatus::Completed);
}

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

#[test]
fn scenario_destroy_progress_tracking() {
    let mut log = MissionLog::default();
    log.set(
        "bounty",
        MissionStatus::Active(ObjectiveProgress {
            destroyed: 0,
            collected: 0,
        }),
    );

    // Simulate killing 1 of 3.
    let MissionStatus::Active(p) = log.status("bounty") else {
        panic!();
    };
    assert_eq!(p.destroyed, 0);

    log.set(
        "bounty",
        MissionStatus::Active(ObjectiveProgress {
            destroyed: 1,
            ..p
        }),
    );
    let MissionStatus::Active(p) = log.status("bounty") else {
        panic!();
    };
    assert_eq!(p.destroyed, 1);

    // Kill 2 more.
    log.set(
        "bounty",
        MissionStatus::Active(ObjectiveProgress {
            destroyed: 3,
            ..p
        }),
    );
    let MissionStatus::Active(p) = log.status("bounty") else {
        panic!();
    };
    assert_eq!(p.destroyed, 3);
}

#[test]
fn scenario_destroy_with_collect_both_must_complete() {
    let def = MissionDef {
        objective: Objective::DestroyShips {
            system: "sol".into(),
            ship_type: "pirate".into(),
            count: 2,
            target_name: "Pirates".into(),
            hostile: true,
            collect: Some(CollectRequirement {
                commodity: "iron".into(),
                quantity: 5,
            }),
        },
        ..dummy_def()
    };

    // Kills done but not enough collected.
    let progress = ObjectiveProgress {
        destroyed: 2,
        collected: 3,
    };
    let Objective::DestroyShips { count, collect, .. } = &def.objective else {
        panic!();
    };
    let kills_done = progress.destroyed >= *count;
    let collect_done = collect
        .as_ref()
        .map(|req| progress.collected >= req.quantity)
        .unwrap_or(true);
    assert!(kills_done);
    assert!(!collect_done);

    // Now enough collected.
    let progress2 = ObjectiveProgress {
        destroyed: 2,
        collected: 5,
    };
    let collect_done2 = collect
        .as_ref()
        .map(|req| progress2.collected >= req.quantity)
        .unwrap_or(true);
    assert!(kills_done);
    assert!(collect_done2);
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
