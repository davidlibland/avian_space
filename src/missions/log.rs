use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use super::types::{MissionDef, MissionStatus};
use crate::item_universe::ItemUniverse;
use crate::session::SessionResource;

// ── MissionLog ───────────────────────────────────────────────────────────────

/// Per-player mission state. Plain data so it's trivial to serialize later.
#[derive(Resource, Default, Debug)]
pub struct MissionLog {
    pub statuses: HashMap<String, MissionStatus>,
}

impl MissionLog {
    pub fn status(&self, id: &str) -> MissionStatus {
        self.statuses.get(id).cloned().unwrap_or_default()
    }
    pub fn set(&mut self, id: &str, status: MissionStatus) {
        self.statuses.insert(id.to_string(), status);
    }
}

/// Save snapshot for MissionLog — only non-default (non-Locked) statuses.
#[derive(Serialize, Deserialize, Default, Clone, Debug)]
pub struct MissionLogSave(pub HashMap<String, MissionStatus>);

impl SessionResource for MissionLog {
    type SaveData = MissionLogSave;
    const SAVE_KEY: Option<&'static str> = Some("mission_statuses");

    fn new_session(_universe: &ItemUniverse) -> Self {
        Self::default()
    }

    fn to_save(&self) -> Self::SaveData {
        MissionLogSave(
            self.statuses
                .iter()
                .filter(|(_, s)| !matches!(s, MissionStatus::Locked))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        )
    }

    fn from_save(data: Self::SaveData, _universe: &ItemUniverse) -> Self {
        Self {
            statuses: data.0,
        }
    }
}

// ── MissionCatalog ───────────────────────────────────────────────────────────

/// All mission definitions currently known to the game. At startup this is
/// populated from `ItemUniverse.missions`; at runtime, procedural templates
/// append freshly-rolled instances. Single source of truth for all
/// mission-def lookups inside this module.
#[derive(Resource, Default, Debug)]
pub struct MissionCatalog {
    pub defs: HashMap<String, MissionDef>,
}

/// Save snapshot for MissionCatalog — only the defs for currently-active
/// missions (base/static defs are re-loaded from ItemUniverse on restore).
#[derive(Serialize, Deserialize, Default, Clone, Debug)]
pub struct MissionCatalogSave(pub HashMap<String, MissionDef>);

impl SessionResource for MissionCatalog {
    type SaveData = MissionCatalogSave;
    const SAVE_KEY: Option<&'static str> = Some("active_mission_defs");

    fn new_session(universe: &ItemUniverse) -> Self {
        Self {
            defs: universe.missions.clone(),
        }
    }

    fn to_save(&self) -> Self::SaveData {
        // We save ALL defs; on load we merge them back on top of the base set.
        // In practice only active (procedural) defs that aren't in the base set
        // matter — but including everything is safe and simpler.
        MissionCatalogSave(self.defs.clone())
    }

    fn from_save(data: Self::SaveData, universe: &ItemUniverse) -> Self {
        let mut defs = universe.missions.clone();
        defs.extend(data.0);
        Self { defs }
    }
}

// ── PlayerUnlocks ────────────────────────────────────────────────────────────

/// Named unlock flags the player has accumulated. Read by ship/outfitter
/// listings (to hide locked entries) and by mission preconditions /
/// completion requirements. Populated by `CompletionEffect::GrantUnlock`.
#[derive(Resource, Default, Debug, Clone)]
pub struct PlayerUnlocks(pub HashSet<String>);

impl PlayerUnlocks {
    pub fn has(&self, name: &str) -> bool {
        self.0.contains(name)
    }
}

impl SessionResource for PlayerUnlocks {
    type SaveData = HashSet<String>;
    const SAVE_KEY: Option<&'static str> = Some("unlocks");

    fn new_session(_universe: &ItemUniverse) -> Self {
        Self::default()
    }

    fn to_save(&self) -> Self::SaveData {
        self.0.clone()
    }

    fn from_save(data: Self::SaveData, _universe: &ItemUniverse) -> Self {
        Self(data)
    }
}

// ── MissionOffers ────────────────────────────────────────────────────────────

/// Currently-rolled mission offers, refreshed on landing. Separate from the
/// log because offers are ephemeral (not persisted across landings).
#[derive(Resource, Default, Debug)]
pub struct MissionOffers {
    /// Missions offered in the tab of the currently-landed planet.
    pub tab: Vec<String>,
    /// planet → list of mission ids currently available in that planet's bar.
    pub bar: HashMap<String, Vec<String>>,
}

impl SessionResource for MissionOffers {
    type SaveData = ();

    fn new_session(_universe: &ItemUniverse) -> Self {
        Self::default()
    }

    fn from_save(_data: (), _universe: &ItemUniverse) -> Self {
        Self::default()
    }
}
