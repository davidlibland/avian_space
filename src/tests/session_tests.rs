use std::collections::{HashMap, HashSet};

use crate::game_save::{Gender, PilotSave, PlayerGameState};
use crate::item_universe::{ItemUniverse, StarSystem};
use crate::missions::{
    MissionCatalog, MissionCatalogSave, MissionLog, MissionLogSave, MissionOffers, PlayerUnlocks,
};
use crate::missions::types::{
    MissionDef, MissionStatus, Objective, ObjectiveProgress, OfferKind,
};
use super::{PendingSessionLoad, SessionResource, SessionSaveData};
use crate::ship::{Personality, ShipData};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn basic_universe() -> ItemUniverse {
    let shuttle = ShipData {
        display_name: String::new(),
        thrust: 200.0,
        max_speed: 300.0,
        torque: 20.0,
        max_health: 100,
        cargo_space: 10,
        item_space: 5,
        base_weapons: HashMap::new(),
        sprite_path: "shuttle.png".to_string(),
        sprite_handle: Default::default(),
        radius: 10.0,
        price: 1000,
        personality: Personality::Trader,
        faction: None,
        required_unlocks: Vec::new(),
        angular_drag: 3.0,
        thrust_kp: 5.0,
        thrust_kd: 1.0,
        reverse_kp: 20.0,
        reverse_kd: 1.5,
    };
    let base_mission = MissionDef {
        briefing: "base".into(),
        success_text: "ok".into(),
        failure_text: "fail".into(),
        preconditions: vec![],
        offer: OfferKind::Auto,
        start_effects: vec![],
        objective: Objective::TravelToSystem {
            system: "sol".into(),
        },
        requires: vec![],
        completion_effects: vec![],
    };
    ItemUniverse {
        weapons: HashMap::new(),
        ships: HashMap::from([("shuttle".to_string(), shuttle)]),
        star_systems: HashMap::from([(
            "sol".to_string(),
            StarSystem {
                display_name: String::new(),
                map_position: bevy::math::Vec2::ZERO,
                connections: vec![],
                planets: HashMap::new(),
                astroid_fields: vec![],
                ships: Default::default(),
            },
        )]),
        outfitter_items: HashMap::new(),
        commodities: HashMap::new(),
        missions: HashMap::from([("base_quest".to_string(), base_mission)]),
        mission_templates: HashMap::new(),
        global_average_price: HashMap::new(),
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

fn sample_mission_def(name: &str) -> MissionDef {
    MissionDef {
        briefing: format!("{name} briefing"),
        success_text: "ok".into(),
        failure_text: "fail".into(),
        preconditions: vec![],
        offer: OfferKind::Auto,
        start_effects: vec![],
        objective: Objective::TravelToSystem {
            system: "sol".into(),
        },
        requires: vec![],
        completion_effects: vec![],
    }
}

// ── MissionLog to_save / from_save ───────────────────────────────────────

#[test]
fn mission_log_filters_locked_on_save() {
    let mut log = MissionLog::default();
    log.set("quest_a", MissionStatus::Active(ObjectiveProgress::default()));
    log.set("quest_b", MissionStatus::Locked);
    log.set("quest_c", MissionStatus::Completed);

    let saved = log.to_save();
    // Locked statuses should be stripped.
    assert!(!saved.0.contains_key("quest_b"), "Locked status should not be saved");
    assert_eq!(saved.0.len(), 2);
    assert!(saved.0.contains_key("quest_a"));
    assert!(saved.0.contains_key("quest_c"));
}

#[test]
fn mission_log_roundtrip() {
    let iu = basic_universe();
    let mut log = MissionLog::default();
    log.set("quest_a", MissionStatus::Active(ObjectiveProgress::default()));
    log.set("quest_b", MissionStatus::Completed);
    log.set("quest_c", MissionStatus::Failed);

    let saved = log.to_save();
    let yaml = serde_yaml::to_string(&saved).unwrap();
    let restored_data: MissionLogSave = serde_yaml::from_str(&yaml).unwrap();
    let restored = MissionLog::from_save(restored_data, &iu);

    assert_eq!(restored.status("quest_a"), log.status("quest_a"));
    assert_eq!(restored.status("quest_b"), MissionStatus::Completed);
    assert_eq!(restored.status("quest_c"), MissionStatus::Failed);
}

// ── MissionCatalog to_save / from_save ───────────────────────────────────

#[test]
fn mission_catalog_new_session_has_base_defs() {
    let iu = basic_universe();
    let catalog = MissionCatalog::new_session(&iu);
    assert!(catalog.defs.contains_key("base_quest"));
}

#[test]
fn mission_catalog_roundtrip_merges_with_base() {
    let iu = basic_universe();

    // Simulate a catalog that has base + a procedural mission.
    let mut catalog = MissionCatalog::new_session(&iu);
    catalog
        .defs
        .insert("procedural_1".to_string(), sample_mission_def("procedural_1"));

    let saved = catalog.to_save();
    let yaml = serde_yaml::to_string(&saved).unwrap();
    let restored_data: MissionCatalogSave = serde_yaml::from_str(&yaml).unwrap();
    let restored = MissionCatalog::from_save(restored_data, &iu);

    // Should have both base and procedural defs.
    assert!(restored.defs.contains_key("base_quest"));
    assert!(restored.defs.contains_key("procedural_1"));
}

#[test]
fn mission_catalog_to_save_excludes_base_defs() {
    let iu = basic_universe();
    let mut catalog = MissionCatalog::new_session(&iu);
    catalog
        .defs
        .insert("procedural_1".to_string(), sample_mission_def("procedural_1"));

    let saved = catalog.to_save();
    // Base def should NOT be in the save snapshot.
    assert!(
        !saved.0.contains_key("base_quest"),
        "Base defs should be excluded from save"
    );
    // Procedural def should be saved.
    assert!(saved.0.contains_key("procedural_1"));
}

#[test]
fn mission_catalog_base_keys_populated() {
    let iu = basic_universe();
    let catalog = MissionCatalog::new_session(&iu);
    assert!(catalog.base_keys.contains("base_quest"));
    assert_eq!(catalog.base_keys.len(), iu.missions.len());
}

#[test]
fn mission_catalog_from_save_preserves_base_keys() {
    let iu = basic_universe();
    let mut saved = HashMap::new();
    saved.insert("procedural_1".to_string(), sample_mission_def("procedural_1"));
    let catalog = MissionCatalog::from_save(MissionCatalogSave(saved), &iu);

    assert!(catalog.base_keys.contains("base_quest"));
    assert!(!catalog.base_keys.contains("procedural_1"));
}

// ── prune_dead_chains ────────────────────────────────────────────────────

#[test]
fn prune_keeps_active_procedural_missions() {
    let iu = basic_universe();
    let mut catalog = MissionCatalog::new_session(&iu);
    catalog
        .defs
        .insert("proc_active".to_string(), sample_mission_def("active"));

    let mut log = MissionLog::default();
    log.set(
        "proc_active",
        MissionStatus::Active(ObjectiveProgress::default()),
    );

    catalog.prune_dead_chains(&log);
    assert!(catalog.defs.contains_key("proc_active"));
    assert!(catalog.defs.contains_key("base_quest")); // base always kept
}

#[test]
fn prune_removes_completed_procedural_missions() {
    let iu = basic_universe();
    let mut catalog = MissionCatalog::new_session(&iu);
    catalog
        .defs
        .insert("proc_done".to_string(), sample_mission_def("done"));

    let mut log = MissionLog::default();
    log.set("proc_done", MissionStatus::Completed);

    catalog.prune_dead_chains(&log);
    assert!(
        !catalog.defs.contains_key("proc_done"),
        "Completed procedural mission should be pruned"
    );
    assert!(catalog.defs.contains_key("base_quest")); // base always kept
}

#[test]
fn prune_removes_unaccepted_available_missions() {
    let iu = basic_universe();
    let mut catalog = MissionCatalog::new_session(&iu);
    catalog
        .defs
        .insert("proc_avail".to_string(), sample_mission_def("offered"));

    let mut log = MissionLog::default();
    log.set("proc_avail", MissionStatus::Available);

    catalog.prune_dead_chains(&log);
    assert!(
        !catalog.defs.contains_key("proc_avail"),
        "Available procedural mission with no active chain should be pruned"
    );
}

#[test]
fn prune_keeps_chain_with_active_member() {
    use crate::missions::types::Precondition;

    let iu = basic_universe();
    let mut catalog = MissionCatalog::new_session(&iu);

    // Stage 1: active
    catalog
        .defs
        .insert("stage1".to_string(), sample_mission_def("stage1"));

    // Stage 2: locked, depends on stage1
    let mut stage2_def = sample_mission_def("stage2");
    stage2_def.preconditions = vec![Precondition::Completed {
        mission: "stage1".to_string(),
    }];
    catalog.defs.insert("stage2".to_string(), stage2_def);

    let mut log = MissionLog::default();
    log.set(
        "stage1",
        MissionStatus::Active(ObjectiveProgress::default()),
    );
    // stage2 is Locked (default — not in log)

    catalog.prune_dead_chains(&log);
    assert!(
        catalog.defs.contains_key("stage1"),
        "Active mission should be kept"
    );
    assert!(
        catalog.defs.contains_key("stage2"),
        "Locked mission in chain with active member should be kept"
    );
}

#[test]
fn prune_removes_entire_dead_chain() {
    use crate::missions::types::Precondition;

    let iu = basic_universe();
    let mut catalog = MissionCatalog::new_session(&iu);

    // Stage 1: completed
    catalog
        .defs
        .insert("stage1".to_string(), sample_mission_def("stage1"));

    // Stage 2: completed, depends on stage1
    let mut stage2_def = sample_mission_def("stage2");
    stage2_def.preconditions = vec![Precondition::Completed {
        mission: "stage1".to_string(),
    }];
    catalog.defs.insert("stage2".to_string(), stage2_def);

    let mut log = MissionLog::default();
    log.set("stage1", MissionStatus::Completed);
    log.set("stage2", MissionStatus::Completed);

    catalog.prune_dead_chains(&log);
    assert!(!catalog.defs.contains_key("stage1"));
    assert!(!catalog.defs.contains_key("stage2"));
}

#[test]
fn prune_never_removes_base_defs() {
    let iu = basic_universe();
    let catalog_before = MissionCatalog::new_session(&iu);
    let base_count = catalog_before.defs.len();

    let mut catalog = catalog_before;
    let log = MissionLog::default(); // all missions Locked

    catalog.prune_dead_chains(&log);
    assert_eq!(
        catalog.defs.len(),
        base_count,
        "Base defs should never be pruned"
    );
}

// ── PlayerUnlocks to_save / from_save ────────────────────────────────────

#[test]
fn player_unlocks_roundtrip() {
    let iu = basic_universe();
    let unlocks = PlayerUnlocks(HashSet::from([
        "warp_drive".to_string(),
        "cloaking".to_string(),
    ]));

    let saved = unlocks.to_save();
    let yaml = serde_yaml::to_string(&saved).unwrap();
    let restored_data: HashSet<String> = serde_yaml::from_str(&yaml).unwrap();
    let restored = PlayerUnlocks::from_save(restored_data, &iu);

    assert!(restored.has("warp_drive"));
    assert!(restored.has("cloaking"));
    assert!(!restored.has("shields"));
}

// ── Ephemeral resources ──────────────────────────────────────────────────

#[test]
fn ephemeral_resource_has_no_save_key() {
    assert!(MissionOffers::SAVE_KEY.is_none());
}

#[test]
fn ephemeral_new_session_is_default() {
    let iu = basic_universe();
    let offers = MissionOffers::new_session(&iu);
    assert!(offers.tab.is_empty());
    assert!(offers.bar.is_empty());
}

// ── Full pipeline: session data → PilotSave YAML → session data ─────────

#[test]
fn full_pipeline_roundtrip() {
    let iu = basic_universe();

    // 1. Build session resource state.
    let mut log = MissionLog::default();
    log.set("quest_a", MissionStatus::Active(ObjectiveProgress::default()));
    log.set("quest_b", MissionStatus::Completed);

    let mut catalog = MissionCatalog::new_session(&iu);
    catalog
        .defs
        .insert("dynamic_1".to_string(), sample_mission_def("dynamic_1"));

    let unlocks = PlayerUnlocks(HashSet::from(["warp_drive".to_string()]));

    // 2. Build SessionSaveData buffer (simulating what sync_save_data does).
    let mut session_data = SessionSaveData::default();
    session_data.resources.insert(
        MissionLog::SAVE_KEY.unwrap().to_string(),
        serde_yaml::to_value(&log.to_save()).unwrap(),
    );
    session_data.resources.insert(
        MissionCatalog::SAVE_KEY.unwrap().to_string(),
        serde_yaml::to_value(&catalog.to_save()).unwrap(),
    );
    session_data.resources.insert(
        PlayerUnlocks::SAVE_KEY.unwrap().to_string(),
        serde_yaml::to_value(&unlocks.to_save()).unwrap(),
    );

    // 3. Build PilotSave and serialise to YAML.
    let game_state = PlayerGameState::new_pilot("TestPilot", Gender::Boy, &iu);
    let save = game_state.to_save(&session_data);
    let yaml = serde_yaml::to_string(&save).unwrap();

    // 4. Deserialise back.
    let loaded: PilotSave = serde_yaml::from_str(&yaml).unwrap();
    assert_eq!(loaded.pilot_name, "TestPilot");
    assert!(!loaded.resources.is_empty(), "resources map should be populated");

    // 5. Simulate PendingSessionLoad → restore each resource.
    let pending = PendingSessionLoad {
        resources: loaded.resources,
    };

    let log_data: MissionLogSave = serde_yaml::from_value(
        pending.resources.get("mission_statuses").unwrap().clone(),
    )
    .unwrap();
    let restored_log = MissionLog::from_save(log_data, &iu);
    assert_eq!(
        restored_log.status("quest_a"),
        MissionStatus::Active(ObjectiveProgress::default())
    );
    assert_eq!(restored_log.status("quest_b"), MissionStatus::Completed);
    // quest_c was never set — should be Locked (default).
    assert_eq!(restored_log.status("quest_c"), MissionStatus::Locked);

    let catalog_data: MissionCatalogSave = serde_yaml::from_value(
        pending.resources.get("active_mission_defs").unwrap().clone(),
    )
    .unwrap();
    let restored_catalog = MissionCatalog::from_save(catalog_data, &iu);
    assert!(restored_catalog.defs.contains_key("base_quest"));
    assert!(restored_catalog.defs.contains_key("dynamic_1"));

    let unlocks_data: HashSet<String> = serde_yaml::from_value(
        pending.resources.get("unlocks").unwrap().clone(),
    )
    .unwrap();
    let restored_unlocks = PlayerUnlocks::from_save(unlocks_data, &iu);
    assert!(restored_unlocks.has("warp_drive"));
}

// ── Edge cases ───────────────────────────────────────────────────────────

#[test]
fn empty_resources_map_survives_yaml() {
    let iu = basic_universe();
    let game_state = PlayerGameState::new_pilot("EmptyPilot", Gender::Girl, &iu);
    let session_data = SessionSaveData::default();
    let save = game_state.to_save(&session_data);

    let yaml = serde_yaml::to_string(&save).unwrap();
    // Empty resources map should be omitted (skip_serializing_if).
    assert!(
        !yaml.contains("resources:"),
        "Empty resources should not appear in YAML"
    );

    let loaded: PilotSave = serde_yaml::from_str(&yaml).unwrap();
    assert!(loaded.resources.is_empty());
}

#[test]
fn unknown_keys_in_resources_are_preserved() {
    // A save file with an unknown key should not break deserialization.
    let yaml = r#"
pilot_name: "Alice"
current_star_system: "sol"
ship_type: "shuttle"
health: 100
cargo: {}
credits: 5000
weapon_loadout: {}
enemies: {}
resources:
  unknown_future_resource:
    some_field: 42
"#;
    let save: PilotSave = serde_yaml::from_str(yaml).unwrap();
    assert!(save.resources.contains_key("unknown_future_resource"));
}

#[test]
fn missing_resource_key_falls_back_to_default() {
    let iu = basic_universe();
    // PendingSessionLoad with no "mission_statuses" key.
    let pending = PendingSessionLoad {
        resources: HashMap::new(),
    };

    // Simulating what load_session_data does:
    let data = pending
        .resources
        .get("mission_statuses")
        .and_then(|v| serde_yaml::from_value::<MissionLogSave>(v.clone()).ok())
        .unwrap_or_default();
    let log = MissionLog::from_save(data, &iu);
    assert!(log.statuses.is_empty(), "Missing key should produce empty log");
}

#[test]
fn corrupt_resource_value_falls_back_to_default() {
    let iu = basic_universe();
    // A resources map with a corrupt value for mission_statuses.
    let mut resources = HashMap::new();
    resources.insert(
        "mission_statuses".to_string(),
        serde_yaml::Value::String("not a valid mission log".to_string()),
    );
    let pending = PendingSessionLoad { resources };

    let data = pending
        .resources
        .get("mission_statuses")
        .and_then(|v| serde_yaml::from_value::<MissionLogSave>(v.clone()).ok())
        .unwrap_or_default();
    let log = MissionLog::from_save(data, &iu);
    assert!(
        log.statuses.is_empty(),
        "Corrupt value should fall back to default"
    );
}

#[test]
fn partial_save_loads_independently() {
    let iu = basic_universe();
    // Only unlocks present, no mission_statuses or active_mission_defs.
    let mut resources = HashMap::new();
    resources.insert(
        "unlocks".to_string(),
        serde_yaml::to_value(&HashSet::from(["shields".to_string()])).unwrap(),
    );
    let pending = PendingSessionLoad { resources };

    // Unlocks should restore.
    let unlocks_data: HashSet<String> = serde_yaml::from_value(
        pending.resources.get("unlocks").unwrap().clone(),
    )
    .unwrap();
    let unlocks = PlayerUnlocks::from_save(unlocks_data, &iu);
    assert!(unlocks.has("shields"));

    // Mission log should fall back to default.
    let log_data = pending
        .resources
        .get("mission_statuses")
        .and_then(|v| serde_yaml::from_value::<MissionLogSave>(v.clone()).ok())
        .unwrap_or_default();
    let log = MissionLog::from_save(log_data, &iu);
    assert!(log.statuses.is_empty());
}
