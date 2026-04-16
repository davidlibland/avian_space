use crate::item_universe::ItemUniverse;
use crate::missions::types::{MissionDef, MissionStatus};
use crate::missions::{MissionCatalog, MissionLog, PlayerUnlocks};
use crate::ship::{Ship, ship_bundle_from_pilot};
use crate::{CurrentStarSystem, PlayState, Player};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
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
    #[serde(default)]
    pub visited_systems: HashSet<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub reserved_cargo: HashMap<String, u16>,
    /// id → status for every non-default status (Active/Completed/Failed).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub mission_statuses: HashMap<String, MissionStatus>,
    /// Defs for currently-Active missions only; merged into MissionCatalog on load.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub active_mission_defs: HashMap<String, MissionDef>,
    #[serde(default, skip_serializing_if = "HashSet::is_empty")]
    pub unlocks: HashSet<String>,
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
    pub visited_systems: HashSet<String>,
    pub mission_statuses: HashMap<String, MissionStatus>,
    pub active_mission_defs: HashMap<String, MissionDef>,
    pub unlocks: HashSet<String>,
}

impl PlayerGameState {
    pub fn new_pilot(name: &str, item_universe: &ItemUniverse) -> Self {
        let starting_ship = item_universe.starting_ship.clone();
        let starting_ship_data = item_universe
            .ships
            .get(&starting_ship)
            .expect("Can't find the starting ship.");
        let mut visited_systems = HashSet::new();
        visited_systems.insert(item_universe.starting_system.clone());
        Self {
            pilot_name: name.to_string(),
            current_star_system: item_universe.starting_system.clone(),
            player_ship: Ship::from_ship_data(starting_ship_data, &starting_ship),
            weapon_loadout: HashMap::new(),
            visited_systems,
            mission_statuses: HashMap::new(),
            active_mission_defs: HashMap::new(),
            unlocks: HashSet::new(),
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
            cargo_cost: HashMap::new(),
            reserved_cargo: save.reserved_cargo.clone(),
            recent_landings: HashMap::new(),
            credits: save.credits,
            nav_target: None,
            weapons_target: None,
            weapon_systems: Default::default(),
            enemies: save.enemies.clone(),
        };
        let mut visited_systems = save.visited_systems;
        visited_systems.insert(save.current_star_system.clone());
        Self {
            pilot_name: save.pilot_name,
            current_star_system: save.current_star_system,
            player_ship: ship,
            weapon_loadout: save.weapon_loadout,
            visited_systems,
            mission_statuses: save.mission_statuses,
            active_mission_defs: save.active_mission_defs,
            unlocks: save.unlocks,
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
            visited_systems: self.visited_systems.clone(),
            reserved_cargo: self.player_ship.reserved_cargo.clone(),
            mission_statuses: self.mission_statuses.clone(),
            active_mission_defs: self.active_mission_defs.clone(),
            unlocks: self.unlocks.clone(),
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
    mission_log: Res<MissionLog>,
    mission_catalog: Res<MissionCatalog>,
    unlocks: Res<PlayerUnlocks>,
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
    game_state.visited_systems.insert(current_system.0.clone());

    // Mission state: keep only non-default statuses; preserve defs for the
    // subset that are Active (the only ones whose defs we need across loads).
    game_state.mission_statuses = mission_log
        .statuses
        .iter()
        .filter(|(_, s)| !matches!(s, MissionStatus::Locked))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    game_state.unlocks = unlocks.0.clone();
    game_state.active_mission_defs = mission_log
        .statuses
        .iter()
        .filter_map(|(id, status)| {
            matches!(status, MissionStatus::Active(_))
                .then(|| mission_catalog.defs.get(id).map(|d| (id.clone(), d.clone())))
                .flatten()
        })
        .collect();
}

/// Runs once on the first Flying entry of a pilot session — restores saved
/// mission state into the ECS resources. Gated on `player_query.is_empty()`
/// so it only fires when the player ship is about to be spawned (i.e. after
/// pilot selection), not on every Landed → Flying transition.
fn restore_mission_state_on_load(
    player_query: Query<Entity, With<Player>>,
    game_state: Res<PlayerGameState>,
    mut mission_log: ResMut<MissionLog>,
    mut mission_catalog: ResMut<MissionCatalog>,
    mut unlocks: ResMut<PlayerUnlocks>,
) {
    if !player_query.is_empty() || game_state.pilot_name.is_empty() {
        return;
    }
    mission_log.statuses = game_state.mission_statuses.clone();
    for (id, def) in &game_state.active_mission_defs {
        mission_catalog.defs.insert(id.clone(), def.clone());
    }
    unlocks.0 = game_state.unlocks.clone();
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
        .add_systems(
            OnEnter(PlayState::Flying),
            (restore_mission_state_on_load, spawn_player_on_enter_flying).chain(),
        )
        // Save when landing/taking off
        .add_systems(OnEnter(PlayState::Landed), save_pilot)
        .add_systems(OnExit(PlayState::Landed), save_pilot);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests/game_save_tests.rs"]
mod tests;
