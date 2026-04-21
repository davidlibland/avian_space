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
///
/// `base_keys` tracks which IDs came from the static `ItemUniverse` data.
/// These are excluded from saves (they're restored by `new_session`) and
/// from pruning.
#[derive(Resource, Default, Debug)]
pub struct MissionCatalog {
    pub defs: HashMap<String, MissionDef>,
    /// IDs that originate from `ItemUniverse.missions` (never pruned or saved).
    pub base_keys: HashSet<String>,
}

/// Save snapshot for MissionCatalog — only procedural defs (base defs are
/// re-loaded from `ItemUniverse` on restore).
#[derive(Serialize, Deserialize, Default, Clone, Debug)]
pub struct MissionCatalogSave(pub HashMap<String, MissionDef>);

impl MissionCatalog {
    /// Remove procedural defs that are no longer needed.
    ///
    /// A procedural def is **needed** if it is part of a chain where at least
    /// one member is currently `Active`.  "Part of a chain" means it is
    /// Active itself, or is reachable from an Active mission by following
    /// precondition references (`Completed { mission }` / `Failed { mission }`).
    ///
    /// Everything else — completed chains, failed chains, offered-but-never-
    /// accepted missions — is pruned.  Fresh procedural missions are rolled on
    /// each landing, so dropped Available/Locked defs are naturally replaced.
    ///
    /// Call this at the start of `roll_offers_on_land`, after
    /// `update_locked_to_available` has already fired.
    pub fn prune_dead_chains(&mut self, log: &MissionLog) {
        use super::types::Precondition;

        // 1. Seed: procedural defs whose status is Active.
        let mut needed: HashSet<String> = self
            .defs
            .keys()
            .filter(|id| !self.base_keys.contains(*id))
            .filter(|id| matches!(log.status(id), MissionStatus::Active(_)))
            .cloned()
            .collect();

        // 2. Expand: any procedural def whose preconditions reference a needed
        //    ID is itself needed (handles multi-stage chains).
        let mut changed = true;
        while changed {
            changed = false;
            for (id, def) in &self.defs {
                if needed.contains(id) || self.base_keys.contains(id) {
                    continue;
                }
                let refs_needed = def.preconditions.iter().any(|p| match p {
                    Precondition::Completed { mission } | Precondition::Failed { mission } => {
                        needed.contains(mission)
                    }
                    _ => false,
                });
                if refs_needed {
                    needed.insert(id.clone());
                    changed = true;
                }
            }
        }

        // 3. Remove procedural defs that aren't needed.
        self.defs
            .retain(|id, _| self.base_keys.contains(id) || needed.contains(id));
    }
}

impl SessionResource for MissionCatalog {
    type SaveData = MissionCatalogSave;
    const SAVE_KEY: Option<&'static str> = Some("active_mission_defs");

    fn new_session(universe: &ItemUniverse) -> Self {
        let base_keys = universe.missions.keys().cloned().collect();
        Self {
            defs: universe.missions.clone(),
            base_keys,
        }
    }

    fn to_save(&self) -> Self::SaveData {
        // Only save procedural defs — base defs are restored from ItemUniverse.
        MissionCatalogSave(
            self.defs
                .iter()
                .filter(|(id, _)| !self.base_keys.contains(*id))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        )
    }

    fn from_save(data: Self::SaveData, universe: &ItemUniverse) -> Self {
        let base_keys = universe.missions.keys().cloned().collect();
        let mut defs = universe.missions.clone();
        defs.extend(data.0);
        Self { defs, base_keys }
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
