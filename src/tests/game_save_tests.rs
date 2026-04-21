use super::*;
use crate::session::SessionSaveData;

fn sample_save() -> PilotSave {
    PilotSave {
        pilot_name: "Zara".to_string(),
        gender: Gender::default(),
        current_star_system: "sol".to_string(),
        ship_type: "shuttle".to_string(),
        health: 75,
        cargo: [("food".to_string(), 5u16), ("metal".to_string(), 10u16)]
            .into_iter()
            .collect(),
        credits: 50_000,
        weapon_loadout: [("laser".to_string(), (2u8, None))].into_iter().collect(),
        enemies: HashMap::from([("Trader".to_string(), 1.0)]),
        visited_systems: HashSet::new(),
        reserved_cargo: HashMap::new(),
        resources: HashMap::new(),
    }
}

fn basic_item_universe() -> ItemUniverse {
    use crate::item_universe::StarSystem;
    use crate::ship::{Personality, ShipData};
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
    ItemUniverse {
        weapons: HashMap::new(),
        ships: HashMap::from([("shuttle".to_string(), shuttle)]),
        star_systems: HashMap::from([(
            "sol".to_string(),
            StarSystem {
                display_name: String::new(),
                map_position: Vec2::ZERO,
                connections: vec![],
                planets: HashMap::new(),
                astroid_fields: vec![],
                ships: Default::default(),
            },
        )]),
        outfitter_items: HashMap::new(),
        commodities: HashMap::new(),
        missions: HashMap::new(),
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

// ── Serialisation ─────────────────────────────────────────────────────────

#[test]
fn pilot_save_yaml_roundtrip() {
    let original = sample_save();
    let yaml = serde_yaml::to_string(&original).expect("serialise");
    let restored: PilotSave = serde_yaml::from_str(&yaml).expect("deserialise");
    assert_eq!(original, restored);
}

#[test]
fn yaml_contains_expected_keys() {
    let yaml = serde_yaml::to_string(&sample_save()).unwrap();
    for key in &[
        "pilot_name",
        "current_star_system",
        "ship_type",
        "health",
        "cargo",
        "credits",
        "weapon_loadout",
    ] {
        assert!(yaml.contains(key), "YAML missing key: {key}");
    }
}

// ── PlayerGameState construction ──────────────────────────────────────────

#[test]
fn new_pilot_defaults() {
    let state = PlayerGameState::new_pilot("Zara", Gender::default(), &basic_item_universe());
    assert_eq!(state.pilot_name, "Zara");
    assert_eq!(state.current_star_system, "sol");
    assert_eq!(state.player_ship.credits, 10_000);
    assert!(state.weapon_loadout.is_empty());
}

#[test]
fn to_save_captures_state() {
    let mut state = PlayerGameState::new_pilot("Rex", Gender::default(), &basic_item_universe());
    state.player_ship.health = 42;
    state.player_ship.credits = 7_777;
    state.player_ship.cargo.insert("ore".to_string(), 3);
    state
        .weapon_loadout
        .insert("missile".to_string(), (1, Some(4)));
    state.current_star_system = "proxima".to_string();

    let session_data = SessionSaveData::default();
    let save = state.to_save(&session_data);
    assert_eq!(save.pilot_name, "Rex");
    assert_eq!(save.health, 42);
    assert_eq!(save.credits, 7_777);
    assert_eq!(save.cargo.get("ore"), Some(&3));
    assert_eq!(save.weapon_loadout.get("missile"), Some(&(1, Some(4))));
    assert_eq!(save.current_star_system, "proxima");
}

#[test]
fn from_save_restores_all_fields() {
    let save = sample_save();
    let iu = basic_item_universe();
    let state = PlayerGameState::from_save(&save, &iu);

    assert_eq!(state.pilot_name, save.pilot_name);
    assert_eq!(state.current_star_system, save.current_star_system);
    assert_eq!(state.player_ship.ship_type, save.ship_type);
    assert_eq!(state.player_ship.health, save.health);
    assert_eq!(state.player_ship.cargo, save.cargo);
    assert_eq!(state.player_ship.credits, save.credits);
    assert_eq!(state.weapon_loadout, save.weapon_loadout);
}

#[test]
fn from_save_target_is_none() {
    // Entity handles must never be persisted.
    let state = PlayerGameState::from_save(&sample_save(), &basic_item_universe());
    assert!(state.player_ship.nav_target.is_none());
    assert!(state.player_ship.weapons_target.is_none());
}

// ── to_save / from_save round-trip ────────────────────────────────────────

#[test]
fn state_to_save_and_back() {
    let mut original =
        PlayerGameState::new_pilot("Lyra", Gender::default(), &basic_item_universe());
    original.player_ship.health = 60;
    original.player_ship.credits = 12_345;
    original.player_ship.cargo.insert("fuel".to_string(), 4);
    original
        .weapon_loadout
        .insert("plasma".to_string(), (3, None));
    original.current_star_system = "tau_ceti".to_string();

    let session_data = SessionSaveData::default();
    let save = original.to_save(&session_data);
    let iu = basic_item_universe();
    let restored = PlayerGameState::from_save(&save, &iu);

    assert_eq!(restored.pilot_name, original.pilot_name);
    assert_eq!(restored.current_star_system, original.current_star_system);
    assert_eq!(restored.player_ship.health, original.player_ship.health);
    assert_eq!(restored.player_ship.credits, original.player_ship.credits);
    assert_eq!(restored.player_ship.cargo, original.player_ship.cargo);
    assert_eq!(restored.weapon_loadout, original.weapon_loadout);
}

// ── write_save guard ─────────────────────────────────────────────────────

#[test]
fn write_save_is_noop_for_unnamed_pilot() {
    // Should not panic or create any files.
    let state = PlayerGameState::default();
    let session_data = SessionSaveData::default();
    assert!(state.pilot_name.is_empty());
    write_save(&state, &session_data); // must not panic
}

// ── Filesystem roundtrip ──────────────────────────────────────────────────

#[test]
fn yaml_file_roundtrip() {
    let tmp = std::env::temp_dir().join("avian_space_test_saves");
    std::fs::create_dir_all(&tmp).unwrap();
    let path = tmp.join("test_pilot.yaml");

    let original = sample_save();
    std::fs::write(&path, serde_yaml::to_string(&original).unwrap()).unwrap();
    let loaded: PilotSave =
        serde_yaml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();

    assert_eq!(original, loaded);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&tmp);
}

