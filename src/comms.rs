//! Communications channel — drives the HUD ticker based on game state.
//!
//! Watches the player's nav_target and posts context-appropriate messages
//! to [`CommsChannel`](crate::hud::CommsChannel).

use bevy::prelude::*;

use crate::hud::CommsChannel;
use crate::item_universe::ItemUniverse;
use crate::planets::Planet;
use crate::ship::ShipHostility;
use crate::{CurrentStarSystem, PlayState, Player, Ship};

pub fn comms_plugin(app: &mut App) {
    app.add_systems(
        Update,
        update_comms_from_nav_target.run_if(in_state(PlayState::Flying)),
    );
}

/// Track the previous nav_target so we only send a message when it changes.
#[derive(Resource, Default)]
struct PrevNavTarget(Option<Entity>);

fn update_comms_from_nav_target(
    mut comms: ResMut<CommsChannel>,
    mut prev: Local<PrevNavTarget>,
    player_query: Query<(&Ship, &ShipHostility), With<Player>>,
    planets_query: Query<&Planet>,
    item_universe: Res<ItemUniverse>,
    current_system: Res<CurrentStarSystem>,
) {
    let Ok((ship, hostility)) = player_query.single() else {
        return;
    };

    let current_entity = ship.nav_target.as_ref().map(|t| t.get_entity());

    // Only react when the target changes
    if current_entity == prev.0 {
        return;
    }
    prev.0 = current_entity;

    // If target cleared, clear comms
    let Some(target) = &ship.nav_target else {
        comms.send("");
        return;
    };

    // Only handle planet targets for now
    let entity = target.get_entity();
    let Ok(planet) = planets_query.get(entity) else {
        comms.send("");
        return;
    };

    let Some(system) = item_universe.star_systems.get(&current_system.0) else {
        return;
    };
    let Some(planet_data) = system.planets.get(&planet.0) else {
        return;
    };

    let name = &planet_data.display_name;

    if planet_data.uncolonized {
        comms.send(format!("{name}: No colony detected."));
    } else if !planet_data.faction.is_empty()
        && hostility
            .0
            .get(&planet_data.faction)
            .copied()
            .unwrap_or(0.0)
            > 0.0
    {
        comms.send(format!("{name}: Docking access denied."));
    } else {
        comms.send(format!("Welcome to {name}. Press L to land."));
    }
}
