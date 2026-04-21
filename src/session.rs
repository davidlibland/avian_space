//! Session-resource infrastructure.
//!
//! A **session resource** is any ECS `Resource` whose lifetime is tied to a
//! single pilot session.  Registering one via [`SessionResourceExt::init_session_resource`]
//! gives you three things for free:
//!
//! * **Reset** — the resource is re-initialised when entering `MainMenu`.
//! * **Save**  — if [`SessionResource::SAVE_KEY`] is `Some`, the resource's
//!   [`to_save`](SessionResource::to_save) data is collected into the pilot
//!   save file automatically.
//! * **Load**  — on entering `Flying` after a pilot is selected, saved data is
//!   fed back through [`from_save`](SessionResource::from_save).

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::item_universe::ItemUniverse;
use crate::PlayState;

// ── Trait ────────────────────────────────────────────────────────────────────

/// Implement this for any `Resource` whose lifetime is one pilot session.
pub trait SessionResource: Resource + Send + Sync + 'static {
    /// The serialisable snapshot type.  Use `()` for ephemeral resources that
    /// are reset on pilot switch but never persisted.
    type SaveData: Serialize + for<'de> Deserialize<'de> + Default;

    /// Key under which this resource is stored in the save file.
    /// `None` means the resource is ephemeral (reset but not saved).
    const SAVE_KEY: Option<&'static str> = None;

    /// Create a fresh resource for a brand-new pilot.
    fn new_session(universe: &ItemUniverse) -> Self;

    /// Snapshot the current live state for saving.
    /// Only called when `SAVE_KEY` is `Some`.
    fn to_save(&self) -> Self::SaveData {
        Default::default()
    }

    /// Restore from a previously-saved snapshot.
    fn from_save(data: Self::SaveData, universe: &ItemUniverse) -> Self;
}

// ── Buffer resources ─────────────────────────────────────────────────────────

/// Collects serialised snapshots from every persisted session resource.
/// Kept up-to-date by per-resource `sync_save_data` systems so that the
/// save function can read it at any time without exclusive world access.
#[derive(Resource, Default, Clone)]
pub struct SessionSaveData {
    pub resources: HashMap<String, serde_yaml::Value>,
}

/// Holds the `resources` map from a just-loaded pilot save file.
/// Consumed (and removed) by per-resource `load_session_data` systems on
/// the first entry into `Flying`.
#[derive(Resource, Default, Clone)]
pub struct PendingSessionLoad {
    pub resources: HashMap<String, serde_yaml::Value>,
}

// ── Generic systems ──────────────────────────────────────────────────────────

/// Reset a session resource to its new-session state.
fn reset_resource<R: SessionResource>(mut res: ResMut<R>, universe: Res<ItemUniverse>) {
    *res = R::new_session(&universe);
}

/// Sync a persisted session resource's save data into the shared buffer.
/// Runs every frame but only serialises when the resource has changed.
fn sync_save_data<R: SessionResource>(res: Res<R>, mut buf: ResMut<SessionSaveData>) {
    if !res.is_changed() {
        return;
    }
    if let Some(key) = R::SAVE_KEY {
        if let Ok(value) = serde_yaml::to_value(&res.to_save()) {
            buf.resources.insert(key.to_string(), value);
        }
    }
}

/// On entering `Flying`, restore a session resource from the pending load.
/// Only fires when `PendingSessionLoad` exists (i.e. a pilot was just selected).
fn load_session_data<R: SessionResource>(
    mut res: ResMut<R>,
    pending: Option<Res<PendingSessionLoad>>,
    universe: Res<ItemUniverse>,
) {
    let Some(pending) = pending else { return };
    if let Some(key) = R::SAVE_KEY {
        let data = pending
            .resources
            .get(key)
            .and_then(|v| serde_yaml::from_value::<R::SaveData>(v.clone()).ok())
            .unwrap_or_default();
        *res = R::from_save(data, &universe);
    } else {
        // Ephemeral — just ensure it's fresh.
        *res = R::new_session(&universe);
    }
}

// ── App extension ────────────────────────────────────────────────────────────

pub trait SessionResourceExt {
    /// Register a session resource.  Gives you automatic reset on `MainMenu`
    /// entry, save-data buffering, and restore-on-load.
    fn init_session_resource<R: SessionResource>(&mut self) -> &mut Self;
}

impl SessionResourceExt for App {
    fn init_session_resource<R: SessionResource>(&mut self) -> &mut Self {
        // Initialise from universe data (overwritten on pilot selection).
        // `item_universe_plugin` must be registered before any plugin that
        // calls this method.
        if !self.world().contains_resource::<R>() {
            let universe = self.world().resource::<ItemUniverse>();
            let res = R::new_session(universe);
            self.insert_resource(res);
        }

        // Reset when returning to the main menu.
        self.add_systems(OnEnter(PlayState::MainMenu), reset_resource::<R>);

        // Restore from save when entering Flying (after pilot selection).
        self.add_systems(OnEnter(PlayState::Flying), load_session_data::<R>);

        if R::SAVE_KEY.is_some() {
            // Keep save buffer up-to-date.
            self.add_systems(
                Update,
                sync_save_data::<R>.run_if(not(in_state(PlayState::MainMenu))),
            );
        }

        self
    }
}

// ── Plugin ───────────────────────────────────────────────────────────────────

/// Registers the shared buffer resources.  Individual session resources are
/// registered by their owning plugins via `init_session_resource`.
pub fn session_plugin(app: &mut App) {
    app.init_resource::<SessionSaveData>()
        .init_resource::<PendingSessionLoad>();
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests/session_tests.rs"]
mod tests;
