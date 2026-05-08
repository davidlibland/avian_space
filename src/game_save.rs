use crate::item_universe::ItemUniverse;
use crate::session::{PendingSessionLoad, SessionSaveData};
use crate::ship::{Ship, ship_bundle_from_pilot};
use crate::{CurrentStarSystem, PlayState, Player};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

// ── Gender ───────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[serde(rename_all = "snake_case")]
pub enum Gender {
    #[default]
    Boy,
    Girl,
}

impl Gender {
    /// Sprite directory name under `assets/people/`.
    pub fn sprite_dir(&self) -> &'static str {
        match self {
            Gender::Boy => "boy",
            Gender::Girl => "girl",
        }
    }
}

// ── Serialisable save ────────────────────────────────────────────────────────

/// On-disk format for a pilot save file.
///
/// Ship and pilot-identity fields live at the top level (managed by
/// `PlayerGameState`).  Everything else — missions, unlocks, etc. — lives in
/// the `resources` map, keyed by each `SessionResource::SAVE_KEY`.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct PilotSave {
    pub pilot_name: String,
    #[serde(default)]
    pub gender: Gender,
    pub current_star_system: String,
    pub ship_type: String,
    pub health: i32,
    pub cargo: HashMap<String, u16>,
    pub credits: i128,
    /// weapon_type → (count, ammo_quantity)
    pub weapon_loadout: HashMap<String, (u8, Option<u32>)>,
    pub enemies: HashMap<String, f32>,
    #[serde(default)]
    pub visited_systems: HashSet<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub reserved_cargo: HashMap<String, u16>,
    /// Per-resource save data, keyed by `SessionResource::SAVE_KEY`.
    /// Populated automatically by the session-resource infrastructure.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub resources: HashMap<String, serde_yaml::Value>,
}

// ── In-memory resource ───────────────────────────────────────────────────────

/// The player's ship + identity state.  Synced from the live ECS `Ship`
/// component every frame.  Session resources (missions, unlocks, …) are
/// managed independently via the `SessionResource` trait.
#[derive(Resource, Clone, Default)]
pub struct PlayerGameState {
    pub pilot_name: String,
    pub gender: Gender,
    pub current_star_system: String,
    pub player_ship: Ship,
    pub weapon_loadout: HashMap<String, (u8, Option<u32>)>,
    pub visited_systems: HashSet<String>,
}

impl PlayerGameState {
    pub fn new_pilot(name: &str, gender: Gender, item_universe: &ItemUniverse) -> Self {
        let starting_ship = item_universe.starting_ship.clone();
        let starting_ship_data = item_universe
            .ships
            .get(&starting_ship)
            .expect("Can't find the starting ship.");
        let mut visited_systems = HashSet::new();
        visited_systems.insert(item_universe.starting_system.clone());
        Self {
            pilot_name: name.to_string(),
            gender,
            current_star_system: item_universe.starting_system.clone(),
            player_ship: Ship::from_ship_data(starting_ship_data, &starting_ship),
            weapon_loadout: HashMap::new(),
            visited_systems,
        }
    }

    pub fn from_save(save: &PilotSave, item_universe: &ItemUniverse) -> Self {
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
            allies: Vec::new(),
            nav_target: None,
            weapons_target: None,
            weapon_systems: Default::default(),
            enemies: save.enemies.clone(),
        };
        let mut visited_systems = save.visited_systems.clone();
        visited_systems.insert(save.current_star_system.clone());
        Self {
            pilot_name: save.pilot_name.clone(),
            gender: save.gender,
            current_star_system: save.current_star_system.clone(),
            player_ship: ship,
            weapon_loadout: save.weapon_loadout.clone(),
            visited_systems,
        }
    }

    fn save_path(&self) -> PathBuf {
        pilots_dir().join(format!("{}.yaml", self.pilot_name))
    }

    pub(crate) fn to_save(&self, session_data: &SessionSaveData) -> PilotSave {
        PilotSave {
            pilot_name: self.pilot_name.clone(),
            gender: self.gender,
            current_star_system: self.current_star_system.clone(),
            ship_type: self.player_ship.ship_type.clone(),
            health: self.player_ship.health,
            cargo: self.player_ship.cargo.clone(),
            credits: self.player_ship.credits,
            weapon_loadout: self.weapon_loadout.clone(),
            enemies: self.player_ship.enemies.clone(),
            visited_systems: self.visited_systems.clone(),
            reserved_cargo: self.player_ship.reserved_cargo.clone(),
            resources: session_data.resources.clone(),
        }
    }
}

/// Write the current game state + session resource data to disk.
pub fn write_save(game_state: &PlayerGameState, session_data: &SessionSaveData) {
    if game_state.pilot_name.is_empty() {
        return;
    }
    let path = game_state.save_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let save = game_state.to_save(session_data);
    match serde_yaml::to_string(&save) {
        Ok(s) => {
            if let Err(e) = std::fs::write(&path, &s) {
                error!("Failed to write pilot save to {path:?}: {e}");
            }
        }
        Err(e) => error!("Failed to serialise pilot save: {e}"),
    }
}

// ── File helpers (pub so main_menu can use them) ─────────────────────────────

/// Where pilot save files live.
///
/// * With the `bundle` feature on (distributable .app build): user-writable
///   `~/Library/Application Support/AvianSpace/pilots/`. Falls back to
///   `./pilots/` if `$HOME` is unset.
/// * Without the feature (dev / `cargo run`): `./pilots/` relative to CWD,
///   matching the existing repo-local behaviour.
fn pilots_dir() -> PathBuf {
    #[cfg(feature = "bundle")]
    {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join("Library/Application Support/AvianSpace/pilots");
        }
    }
    PathBuf::from("pilots")
}

pub fn list_saves() -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(pilots_dir()) else {
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
    let path = pilots_dir().join(format!("{pilot_name}.yaml"));
    let data = std::fs::read_to_string(&path).ok()?;
    serde_yaml::from_str(&data).ok()
}

// ── Systems ──────────────────────────────────────────────────────────────────

/// Syncs the live ECS Ship component → PlayerGameState each frame.
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
    game_state.visited_systems.insert(current_system.0.clone());
}

/// Spawns the player ship on the first entry into Flying (from the main menu).
/// Subsequent Landed → Flying transitions skip this because the player already exists.
fn spawn_player_on_enter_flying(
    mut commands: Commands,
    player_query: Query<Entity, With<Player>>,
    game_state: Res<PlayerGameState>,
    item_universe: Res<ItemUniverse>,
) {
    if !player_query.is_empty() || game_state.pilot_name.is_empty() {
        return;
    }
    let bundle = ship_bundle_from_pilot(
        game_state.player_ship.clone(),
        &game_state.weapon_loadout,
        &item_universe,
        Vec2::ZERO,
    );
    commands.spawn((Player, bundle));
}

/// Saves to disk whenever the player lands on / takes off from a planet.
fn save_pilot(game_state: Res<PlayerGameState>, session_data: Res<SessionSaveData>) {
    write_save(&game_state, &session_data);
}

/// Despawn the player entity and reset PlayerGameState when returning to the
/// main menu.  Session resources are reset independently by their own
/// `init_session_resource` registrations.
fn cleanup_on_enter_menu(
    mut commands: Commands,
    player_query: Query<Entity, With<Player>>,
    mut game_state: ResMut<PlayerGameState>,
    mut session_data: ResMut<SessionSaveData>,
) {
    for entity in &player_query {
        crate::utils::safe_despawn(&mut commands, entity);
    }
    *game_state = PlayerGameState::default();
    session_data.resources.clear();
}

/// Remove the `PendingSessionLoad` resource after session resources have
/// consumed it on the first Flying entry.
fn consume_pending_load(mut commands: Commands, pending: Option<Res<PendingSessionLoad>>) {
    if pending.is_some() {
        commands.remove_resource::<PendingSessionLoad>();
    }
}

// ── Plugin ───────────────────────────────────────────────────────────────────

pub fn game_save_plugin(app: &mut App) {
    app.init_resource::<PlayerGameState>()
        .add_systems(
            Update,
            sync_player_state.run_if(not(in_state(PlayState::MainMenu))),
        )
        .add_systems(
            OnEnter(PlayState::Flying),
            // consume_pending_load runs last so session resources can read it.
            (spawn_player_on_enter_flying, consume_pending_load).chain(),
        )
        .add_systems(OnEnter(PlayState::MainMenu), cleanup_on_enter_menu)
        // Save when landing/taking off
        .add_systems(OnEnter(PlayState::Landed), save_pilot)
        .add_systems(OnExit(PlayState::Landed), save_pilot);
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests/game_save_tests.rs"]
mod tests;
