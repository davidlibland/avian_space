use bevy::prelude::*;
use std::collections::{HashMap, HashSet};

use super::types::{MissionDef, MissionStatus};

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

/// All mission definitions currently known to the game. At startup this is
/// populated from `ItemUniverse.missions`; at runtime, procedural templates
/// append freshly-rolled instances. Single source of truth for all
/// mission-def lookups inside this module.
#[derive(Resource, Default, Debug)]
pub struct MissionCatalog {
    pub defs: HashMap<String, MissionDef>,
}

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

/// Currently-rolled mission offers, refreshed on landing. Separate from the
/// log because offers are ephemeral (not persisted across landings).
#[derive(Resource, Default, Debug)]
pub struct MissionOffers {
    /// Missions offered in the tab of the currently-landed planet.
    pub tab: Vec<String>,
    /// planet → list of mission ids currently available in that planet's bar.
    pub bar: HashMap<String, Vec<String>>,
}
