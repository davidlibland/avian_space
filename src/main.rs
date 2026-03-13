use avian2d::{math::*, prelude::*};
use bevy::{math::VectorSpace, prelude::*};

mod asteroids;
mod hud;
mod item_universe;
mod planet_ui;
mod planets;
use planets::planets_plugin;
mod ai_ships;
mod ship;
mod starfield;
mod utils;
mod weapons;
use ai_ships::ai_ship_bundle;
use asteroids::{Asteroid, asteroid_plugin, build_asteroid_field, shatter_asteroid};
use hud::HudPlugin;
use item_universe::{ItemUniverse, item_universe_plugin};
use planet_ui::planet_ui_plugin;
use ship::{Ship, ShipCommand, ship_bundle, ship_plugin};
use starfield::StarfieldPlugin;
use weapons::{FireCommand, Projectile, weapons_plugin};

use crate::weapons::WeaponSystems;

// Define your boundary
const BOUNDS: f32 = 10000.0;

#[derive(States, Default, PartialEq, Eq, Hash, Clone, Debug)]
enum GameState {
    #[default]
    Flying,
    Landed,
}

#[derive(PhysicsLayer, Default)]
pub enum GameLayer {
    Ship,
    Weapon,
    Asteroid,
    Planet,
    Radar,
    #[default]
    Other,
}

#[derive(Resource)]
pub struct CurrentStarSystem(pub String);

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            // Add physics plugins and specify a units-per-meter scaling factor, 1 meter = 20 pixels.
            // The unit allows the engine to tune its parameters for the scale of the world, improving stability.
            PhysicsPlugins::default().with_length_unit(20.0),
            planet_ui_plugin,
            StarfieldPlugin::default(),
            HudPlugin::default(),
            ship_plugin,
            weapons_plugin,
            item_universe_plugin,
            planets_plugin,
            asteroid_plugin,
            ai_ship_bundle,
        ))
        .init_state::<GameState>()
        .insert_resource(CurrentStarSystem("sol".to_string())) // Set custom gravity
        .insert_resource(Gravity(Vec2::NEG_Y * 0.0)) // Set custom gravity
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (keyboard_input, collision_system).run_if(in_state(GameState::Flying)),
        )
        .run();
}

#[derive(Component)]
pub struct Player;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    asset_server: Res<AssetServer>,
    item_universe: Res<ItemUniverse>,
    star_system: Res<CurrentStarSystem>,
) {
    // Player
    commands.spawn((
        Player, // Mark the player
        ship_bundle(&asset_server, &item_universe, Vec2::ZERO),
    ));

    if let Some(system_data) = item_universe.star_systems.get(&star_system.0) {
        for field in system_data.astroid_fields.iter() {
            // Asteroids
            build_asteroid_field(&mut commands, &mut meshes, &mut materials, &field);
        }
    }

    // Camera
    commands.spawn(Camera2d);
}

/// Sends [`MovementAction`] events based on keyboard input.
fn keyboard_input(
    mut writer: MessageWriter<ShipCommand>,
    mut weapons_writer: MessageWriter<FireCommand>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    player_query: Query<(Entity, &WeaponSystems), With<Player>>,
) {
    let Ok((player_entity, weapons)) = player_query.single() else {
        return; // Player not spawned yet
    };

    let left = keyboard_input.any_pressed([KeyCode::KeyA, KeyCode::ArrowLeft]);
    let right = keyboard_input.any_pressed([KeyCode::KeyD, KeyCode::ArrowRight]);
    let up = keyboard_input.any_pressed([KeyCode::KeyW, KeyCode::ArrowUp]);
    let down = keyboard_input.any_pressed([KeyCode::KeyS, KeyCode::ArrowDown]);

    let horizontal = right as i8 - left as i8;
    let turn = horizontal as Scalar;

    let vertical = up as i8;
    let thrust = vertical as Scalar;
    let reverse = (down as i8) as Scalar;

    // Only send if there's actual input
    if thrust.abs() > f32::EPSILON || turn.abs() > f32::EPSILON || reverse.abs() > f32::EPSILON {
        writer.write(ShipCommand {
            entity: player_entity,
            thrust,
            turn,
            reverse,
        });
    }

    let fire = keyboard_input.any_pressed([KeyCode::Space]);
    if fire {
        for specific in weapons.primary.iter() {
            if specific.cooldown.just_finished() {
                weapons_writer.write(FireCommand {
                    ship: player_entity,
                    weapon_type: specific.weapon_type.clone(),
                });
            }
        }
    }
}

fn collision_system(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut collisions: MessageReader<CollisionStart>,
    asteroids: Query<(&Asteroid, &Transform, &LinearVelocity)>,
    mut ships: Query<&mut Ship>,
    weapons: Query<&Projectile>,
) {
    for event in collisions.read() {
        let (a, b) = (event.collider1, event.collider2);

        // Determine which entity is the asteroid and which is the ship,
        // handling both orderings since Avian2D can emit either way.
        let asteroid_ship_entity = if asteroids.contains(a) && ships.contains(b) {
            Some((a, b))
        } else if asteroids.contains(b) && ships.contains(a) {
            Some((b, a))
        } else {
            None
        };

        if let Some((asteroid_entity, ship_entity)) = asteroid_ship_entity {
            shatter_asteroid(
                &mut commands,
                &mut meshes,
                &mut materials,
                &asteroid_entity,
                &asteroids,
            );
            if let Ok(mut ship) = ships.get_mut(ship_entity) {
                // Send a damage Event:
                ship.health -= 10;
                ship.health = if ship.health < 0 { 0 } else { ship.health };
            }
        }

        // Determine which entity is the asteroid and which is the ship,
        // handling both orderings since Avian2D can emit either way.
        let asteroid_weapon_entity = if asteroids.contains(a) && weapons.contains(b) {
            Some((a, b))
        } else if asteroids.contains(b) && weapons.contains(a) {
            Some((b, a))
        } else {
            None
        };

        if let Some((asteroid_entity, weapon_entity)) = asteroid_weapon_entity {
            shatter_asteroid(
                &mut commands,
                &mut meshes,
                &mut materials,
                &asteroid_entity,
                &asteroids,
            );
            commands.entity(weapon_entity).despawn();
        }
    }
}
