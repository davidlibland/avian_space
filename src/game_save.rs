use crate::item_universe::ItemUniverse;
use crate::ship::{Ship, ship_bundle_from_pilot};
use crate::{CurrentStarSystem, PlayState, Player};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

// ── Serialisable save ────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct PilotSave {
    pub pilot_name: String,
    pub current_star_system: String,
    pub ship_type: String,
    pub health: i32,
    pub cargo: HashMap<String, u16>,
    pub credits: i128,
    /// weapon_type → count owned
    pub weapon_loadout: HashMap<String, (u8, Option<u32>)>,
    pub enemies: HashMap<String, f32>,
}

// ── In-memory resource ───────────────────────────────────────────────────────

/// Mirrors the player's persistent state; synced from ECS every frame.
/// Also the source of truth when spawning / restoring the player ship.
#[derive(Resource, Clone, Default)]
pub struct PlayerGameState {
    pub pilot_name: String,
    pub current_star_system: String,
    pub player_ship: Ship,
    pub weapon_loadout: HashMap<String, (u8, Option<u32>)>,
}

impl PlayerGameState {
    pub fn new_pilot(name: &str, item_universe: &ItemUniverse) -> Self {
        let starting_ship = item_universe.starting_ship.clone();
        let starting_ship_data = item_universe
            .ships
            .get(&starting_ship)
            .expect("Can't find the starting ship.");
        Self {
            pilot_name: name.to_string(),
            current_star_system: item_universe.starting_system.clone(),
            player_ship: Ship::from_ship_data(starting_ship_data, &starting_ship),
            weapon_loadout: HashMap::new(),
        }
    }

    pub fn from_save(save: PilotSave, item_universe: &ItemUniverse) -> Self {
        let ship_data = item_universe
            .ships
            .get(&save.ship_type)
            .cloned()
            .expect("Ship type not found in item universe — save file may be corrupt or from an incompatible version");

        // Sanity check: saved cargo must fit in the ship's cargo hold.
        let cargo_used: u16 = save.cargo.values().sum();
        if cargo_used > ship_data.cargo_space {
            warn!(
                "Pilot \"{}\": saved cargo ({cargo_used}) exceeds ship cargo space ({}) \
                 — save may be corrupt",
                save.pilot_name, ship_data.cargo_space
            );
        }

        // Sanity check: saved weapon loadout must fit in the ship's item space.
        let loadout_space: i32 = save
            .weapon_loadout
            .iter()
            .filter_map(|(weapon_type, &(count, ammo_qty))| {
                item_universe.outfitter_items.get(weapon_type).map(|item| {
                    let base = item.space() as i32 * count as i32;
                    let ammo = match item {
                        crate::item_universe::OutfitterItem::SecondaryWeapon {
                            ammo_space, ..
                        } => ammo_qty.unwrap_or(0) as i32 * *ammo_space as i32,
                        _ => 0,
                    };
                    base + ammo
                })
            })
            .sum();
        if loadout_space > ship_data.item_space as i32 {
            warn!(
                "Pilot \"{}\": saved weapon loadout consumes {loadout_space} item space \
                 but ship only has {} — save may be corrupt",
                save.pilot_name, ship_data.item_space
            );
        }

        let ship = Ship {
            ship_type: save.ship_type.clone(),
            data: ship_data,
            health: save.health,
            cargo: save.cargo.clone(),
            credits: save.credits,
            target: None,
            weapon_systems: Default::default(),
            enemies: save.enemies.clone(),
        };
        Self {
            pilot_name: save.pilot_name,
            current_star_system: save.current_star_system,
            player_ship: ship,
            weapon_loadout: save.weapon_loadout,
        }
    }

    fn save_path(&self) -> PathBuf {
        PathBuf::from("pilots").join(format!("{}.yaml", self.pilot_name))
    }

    fn to_save(&self) -> PilotSave {
        PilotSave {
            pilot_name: self.pilot_name.clone(),
            current_star_system: self.current_star_system.clone(),
            ship_type: self.player_ship.ship_type.clone(),
            health: self.player_ship.health,
            cargo: self.player_ship.cargo.clone(),
            credits: self.player_ship.credits,
            weapon_loadout: self.weapon_loadout.clone(),
            enemies: self.player_ship.enemies.clone(),
        }
    }

    pub fn save(&self) {
        if self.pilot_name.is_empty() {
            return;
        }
        let path = self.save_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match serde_yaml::to_string(&self.to_save()) {
            Ok(s) => {
                if let Err(e) = std::fs::write(&path, &s) {
                    error!("Failed to write pilot save to {path:?}: {e}");
                }
            }
            Err(e) => error!("Failed to serialise pilot save: {e}"),
        }
    }
}

// ── File helpers (pub so main_menu can use them) ─────────────────────────────

pub fn list_saves() -> Vec<String> {
    let Ok(entries) = std::fs::read_dir("pilots") else {
        return vec![];
    };
    let mut names: Vec<String> = entries
        .filter_map(|e| {
            let entry = e.ok()?;
            let raw = entry.file_name();
            let name = raw.to_string_lossy();
            name.ends_with(".yaml")
                .then(|| name.trim_end_matches(".yaml").to_string())
        })
        .collect();
    names.sort();
    names
}

pub fn load_save(pilot_name: &str) -> Option<PilotSave> {
    let path = PathBuf::from("pilots").join(format!("{pilot_name}.yaml"));
    let data = std::fs::read_to_string(&path).ok()?;
    serde_yaml::from_str(&data).ok()
}

// ── Systems ───────────────────────────────────────────────────────────────────

/// Syncs ECS Player data → PlayerGameState each frame.
fn sync_player_state(
    player_query: Query<&Ship, With<Player>>,
    mut game_state: ResMut<PlayerGameState>,
    current_system: Res<CurrentStarSystem>,
) {
    if let Ok(ship) = player_query.single() {
        game_state.weapon_loadout = ship
            .weapon_systems
            .iter_all()
            .map(|(k, v)| (k.clone(), (v.number, v.ammo_quantity)))
            .collect();
        game_state.player_ship = ship.clone();
    }
    game_state.current_star_system = current_system.0.clone();
}

/// Spawns the player ship on the first entry into Flying (from the main menu).
/// Subsequent Landed → Flying transitions skip this because the player already exists.
fn spawn_player_on_enter_flying(
    mut commands: Commands,
    player_query: Query<Entity, With<Player>>,
    game_state: Res<PlayerGameState>,
    asset_server: Res<AssetServer>,
    item_universe: Res<ItemUniverse>,
) {
    if !player_query.is_empty() || game_state.pilot_name.is_empty() {
        return;
    }
    let bundle = ship_bundle_from_pilot(
        game_state.player_ship.clone(),
        &game_state.weapon_loadout,
        &asset_server,
        &item_universe,
        Vec2::ZERO,
    );
    commands.spawn((Player, bundle));
}

/// Saves to disk whenever the player lands on a planet.
fn save_pilot(game_state: Res<PlayerGameState>) {
    game_state.save();
}

// ── Plugin ────────────────────────────────────────────────────────────────────

pub fn game_save_plugin(app: &mut App) {
    app.init_resource::<PlayerGameState>()
        .add_systems(Update, sync_player_state)
        .add_systems(OnEnter(PlayState::Flying), spawn_player_on_enter_flying)
        // Save when landing/taking off
        .add_systems(OnEnter(PlayState::Landed), save_pilot)
        .add_systems(OnExit(PlayState::Landed), save_pilot);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_save() -> PilotSave {
        PilotSave {
            pilot_name: "Zara".to_string(),
            current_star_system: "sol".to_string(),
            ship_type: "shuttle".to_string(),
            health: 75,
            cargo: [("food".to_string(), 5u16), ("metal".to_string(), 10u16)]
                .into_iter()
                .collect(),
            credits: 50_000,
            weapon_loadout: [("laser".to_string(), (2u8, None))].into_iter().collect(),
            enemies: HashMap::from([("Trader".to_string(), 1.0)]),
        }
    }

    fn empty_item_universe() -> ItemUniverse {
        ItemUniverse {
            weapons: HashMap::new(),
            ships: HashMap::new(),
            star_systems: HashMap::new(),
            outfitter_items: HashMap::new(),
            commodities: HashMap::new(),
            global_average_price: HashMap::new(),
            system_commodity_best_planet_to_sell: HashMap::new(),
            system_planet_best_commodity_to_buy: HashMap::new(),
            starting_system: "sol".to_string(),
            starting_ship: "shuttle".to_string(),
            enemies: HashMap::new(),
        }
    }

    fn basic_item_universe() -> ItemUniverse {
        use crate::item_universe::StarSystem;
        use crate::ship::{Personality, ShipData};
        let shuttle = ShipData {
            thrust: 200.0,
            max_speed: 300.0,
            torque: 20.0,
            max_health: 100,
            cargo_space: 10,
            item_space: 5,
            base_weapons: HashMap::new(),
            sprite_path: "shuttle.png".to_string(),
            radius: 10.0,
            price: 1000,
            personality: Personality::Trader,
            faction: None,
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
                    map_position: Vec2::ZERO,
                    connections: vec![],
                    planets: HashMap::new(),
                    astroid_fields: vec![],
                    ships: Default::default(),
                },
            )]),
            outfitter_items: HashMap::new(),
            commodities: HashMap::new(),
            global_average_price: HashMap::new(),
            system_commodity_best_planet_to_sell: HashMap::new(),
            system_planet_best_commodity_to_buy: HashMap::new(),
            starting_system: "sol".to_string(),
            starting_ship: "shuttle".to_string(),
            enemies: HashMap::new(),
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
        let state = PlayerGameState::new_pilot("Zara", &basic_item_universe());
        assert_eq!(state.pilot_name, "Zara");
        assert_eq!(state.current_star_system, "sol");
        assert_eq!(state.player_ship.credits, 100_000);
        assert!(state.weapon_loadout.is_empty());
    }

    #[test]
    fn to_save_captures_state() {
        let mut state = PlayerGameState::new_pilot("Rex", &basic_item_universe());
        state.player_ship.health = 42;
        state.player_ship.credits = 7_777;
        state.player_ship.cargo.insert("ore".to_string(), 3);
        state
            .weapon_loadout
            .insert("missile".to_string(), (1, Some(4)));
        state.current_star_system = "proxima".to_string();

        let save = state.to_save();
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
        let state = PlayerGameState::from_save(save.clone(), &iu);

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
        let state = PlayerGameState::from_save(sample_save(), &basic_item_universe());
        assert!(state.player_ship.target.is_none());
    }

    // ── to_save / from_save round-trip ────────────────────────────────────────

    #[test]
    fn state_to_save_and_back() {
        let mut original = PlayerGameState::new_pilot("Lyra", &basic_item_universe());
        original.player_ship.health = 60;
        original.player_ship.credits = 12_345;
        original.player_ship.cargo.insert("fuel".to_string(), 4);
        original
            .weapon_loadout
            .insert("plasma".to_string(), (3, None));
        original.current_star_system = "tau_ceti".to_string();

        let save = original.to_save();
        let iu = basic_item_universe();
        let restored = PlayerGameState::from_save(save, &iu);

        assert_eq!(restored.pilot_name, original.pilot_name);
        assert_eq!(restored.current_star_system, original.current_star_system);
        assert_eq!(restored.player_ship.health, original.player_ship.health);
        assert_eq!(restored.player_ship.credits, original.player_ship.credits);
        assert_eq!(restored.player_ship.cargo, original.player_ship.cargo);
        assert_eq!(restored.weapon_loadout, original.weapon_loadout);
    }

    // ── save() guard ──────────────────────────────────────────────────────────

    #[test]
    fn save_is_noop_for_unnamed_pilot() {
        // Should not panic or create any files.
        let state = PlayerGameState::default();
        assert!(state.pilot_name.is_empty());
        state.save(); // must not panic
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
}
