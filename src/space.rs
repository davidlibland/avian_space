//! Consolidated space-flight plugin.
//!
//! Groups all plugins that are specific to the in-space `Flying` state into a
//! single registration point. Individual modules remain in their own files;
//! this module just wires them together.

use bevy::prelude::*;

use crate::ai_ships::ai_ship_bundle;
use crate::asteroids::asteroid_plugin;
use crate::carrier::carrier_plugin;
use crate::pickups::pickup_plugin;
use crate::ship::ship_plugin;
use crate::ship_anim::ship_anim_plugin;
use crate::weapons::weapons_plugin;

/// Registers all space-flight gameplay plugins.
///
/// These plugins contain systems that run during `PlayState::Flying` (and
/// sometimes `Traveling`). Data-loading, save/load, missions, rendering, and
/// RL-training plugins are registered separately because they serve multiple
/// states.
pub fn space_plugin(app: &mut App) {
    app.add_plugins((
        ship_plugin,
        weapons_plugin,
        asteroid_plugin,
        ship_anim_plugin,
        carrier_plugin,
        ai_ship_bundle,
        pickup_plugin,
    ));
}
