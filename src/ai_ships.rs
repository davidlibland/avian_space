// Some AI for the ships
use crate::CurrentStarSystem;
use crate::item_universe::ItemUniverse;
use crate::ship::ship_bundle;
use bevy::prelude::*;
use rand::Rng;

#[derive(Component)]
struct AIShip;

pub fn ai_ship_bundle(app: &mut App) {
    app.add_systems(Startup, spawn_ai_ships);
}

pub fn spawn_ai_ships(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    asset_server: Res<AssetServer>,
    item_universe: Res<ItemUniverse>,
    star_system: Res<CurrentStarSystem>,
) {
    if let Some(system_data) = item_universe.star_systems.get(&star_system.0) {
        let mut rng = rand::thread_rng();
        for _ in (0..system_data.ships) {
            let x = rng.gen_range(-1000.0..1000.0);
            let y = rng.gen_range(-1000.0..1000.0);
            // Player
            commands.spawn((
                AIShip, // Mark the player
                ship_bundle(&asset_server, &item_universe, Vec2::new(x, y)),
            ));
        }
    }
}
