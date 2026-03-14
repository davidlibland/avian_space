use avian2d::{math::*, prelude::*};
use bevy::{math::VectorSpace, prelude::*};

mod asteroids;
mod hud;
mod item_universe;
mod jump_ui;
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
use jump_ui::jump_ui_plugin;
use planet_ui::planet_ui_plugin;
use planets::NearbyPlanet;
use ship::{Ship, ShipCommand, ship_bundle, ship_plugin};
use starfield::StarfieldPlugin;
use utils::safe_despawn;
use weapons::{FireCommand, Projectile, weapon_lifetime, weapons_plugin};

use crate::weapons::WeaponSystems;

// Define your boundary
const BOUNDS: f32 = 10000.0;

#[derive(States, Default, PartialEq, Eq, Hash, Clone, Debug)]
pub enum GameState {
    #[default]
    Flying,
    Landed,
    Traveling,
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

/// Acceleration to apply each second during the pre-jump boost (units/s²).
const JUMP_ACCEL: f32 = 2500.0;
/// Target speed the ship reaches before the hyperspace flash triggers.
const JUMP_SPEED: f32 = 1600.0;
/// Duration of the hyperspace flash (seconds).
const FLASH_DURATION: f32 = 0.7;

#[derive(Default, PartialEq, Clone, Debug)]
pub enum TravelPhase {
    #[default]
    Idle,
    /// Ship is accelerating toward JUMP_SPEED (still in Flying state).
    Accelerating,
    /// Hyperspace flash is playing (in Traveling state).
    Flashing,
}

#[derive(Resource, Default)]
pub struct TravelContext {
    pub destination: String,
    pub phase: TravelPhase,
    /// 0.0..=1.0, advances during Flashing.
    pub flash_t: f32,
    /// Normal MaxLinearSpeed saved before boosting, restored on arrival.
    pub saved_max_speed: f32,
}

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            // Add physics plugins and specify a units-per-meter scaling factor, 1 meter = 20 pixels.
            // The unit allows the engine to tune its parameters for the scale of the world, improving stability.
            PhysicsPlugins::default().with_length_unit(20.0),
            planet_ui_plugin,
            jump_ui_plugin,
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
        .init_resource::<TravelContext>()
        .insert_resource(CurrentStarSystem("sol".to_string()))
        .insert_resource(Gravity(Vec2::NEG_Y * 0.0))
        .add_systems(Startup, setup)
        .add_systems(
            OnEnter(GameState::Flying),
            (spawn_asteroids, set_arrival_velocity),
        )
        .add_systems(OnEnter(GameState::Traveling), reset_nearby_planet)
        .add_systems(
            Update,
            (keyboard_input, collision_system, accelerate_for_jump)
                .run_if(in_state(GameState::Flying)),
        )
        .add_systems(
            Update,
            ApplyDeferred
                .after(collision_system)
                .before(weapon_lifetime),
        )
        .add_systems(Update, travel_system.run_if(in_state(GameState::Traveling)))
        .run();
}

#[derive(Component)]
pub struct Player;

fn setup(mut commands: Commands, asset_server: Res<AssetServer>, item_universe: Res<ItemUniverse>) {
    // Player (persists across system travel — not StateScoped)
    commands.spawn((
        Player,
        ship_bundle(&asset_server, &item_universe, Vec2::ZERO),
    ));

    // Camera (persists across system travel)
    commands.spawn(Camera2d);
}

fn spawn_asteroids(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    item_universe: Res<ItemUniverse>,
    star_system: Res<CurrentStarSystem>,
) {
    if let Some(system_data) = item_universe.star_systems.get(&star_system.0) {
        for field in system_data.astroid_fields.iter() {
            build_asteroid_field(&mut commands, &mut meshes, &mut materials, field);
        }
    }
}

fn reset_nearby_planet(mut nearby: ResMut<NearbyPlanet>) {
    nearby.0 = None;
}

/// Runs during Flying state: boosts the player ship toward JUMP_SPEED, then triggers Traveling.
fn accelerate_for_jump(
    mut ctx: ResMut<TravelContext>,
    mut player_q: Query<
        (&Transform, &mut LinearVelocity, &mut MaxLinearSpeed, &Ship),
        With<Player>,
    >,
    time: Res<Time>,
    mut next_state: ResMut<NextState<GameState>>,
) {
    if ctx.phase != TravelPhase::Accelerating {
        return;
    }
    let Ok((transform, mut vel, mut max_speed, ship)) = player_q.single_mut() else {
        return;
    };
    // Save the normal cap on the first tick.
    if ctx.saved_max_speed == 0.0 {
        ctx.saved_max_speed = ship.data.max_speed;
    }

    let forward = (transform.rotation * Vec3::Y).xy();
    let new_speed = (vel.0.length() + JUMP_ACCEL * time.delta_secs()).min(JUMP_SPEED);
    // Drive in the ship's nose direction so the jump always looks intentional.
    vel.0 = forward * new_speed;
    // Raise the Avian2d velocity cap to match — otherwise physics clamps us back.
    max_speed.0 = new_speed;

    if new_speed >= JUMP_SPEED {
        ctx.phase = TravelPhase::Flashing;
        ctx.flash_t = 0.0;
        next_state.set(GameState::Traveling);
    }
}

/// Runs during Traveling state: advances the hyperspace flash and handles the system switch.
fn travel_system(
    mut ctx: ResMut<TravelContext>,
    mut current_system: ResMut<CurrentStarSystem>,
    mut state: ResMut<NextState<GameState>>,
    time: Res<Time>,
) {
    if ctx.phase != TravelPhase::Flashing {
        return;
    }
    ctx.flash_t = (ctx.flash_t + time.delta_secs() / FLASH_DURATION).min(1.0);

    // Switch the star system at the visual peak of the flash (midpoint).
    if ctx.flash_t >= 0.5 {
        current_system.0 = ctx.destination.clone();
    }

    if ctx.flash_t >= 1.0 {
        ctx.phase = TravelPhase::Idle;
        state.set(GameState::Flying);
    }
}

/// Runs OnEnter(Flying): if we just completed a jump, restore max speed and set arrival velocity.
fn set_arrival_velocity(
    mut ctx: ResMut<TravelContext>,
    mut player_q: Query<(&Transform, &mut LinearVelocity, &mut MaxLinearSpeed), With<Player>>,
) {
    // Only act when returning from a jump (saved_max_speed was set by accelerate_for_jump).
    if ctx.saved_max_speed == 0.0 {
        return;
    }
    let Ok((transform, mut vel, mut max_speed)) = player_q.single_mut() else {
        return;
    };
    let normal_speed = ctx.saved_max_speed;
    max_speed.0 = normal_speed;
    vel.0 = (transform.rotation * Vec3::Y).xy() * normal_speed;
    ctx.saved_max_speed = 0.0; // clear so this doesn't re-run on planet launches
}

/// Sends [`MovementAction`] events based on keyboard input.
fn keyboard_input(
    mut writer: MessageWriter<ShipCommand>,
    mut weapons_writer: MessageWriter<FireCommand>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut player_query: Query<(Entity, &mut WeaponSystems), With<Player>>,
) {
    let Ok((player_entity, mut weapons)) = player_query.single_mut() else {
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
        for specific in weapons.primary.values() {
            weapons_writer.write(FireCommand {
                ship: player_entity,
                weapon_type: specific.weapon_type.clone(),
            });
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
    use std::collections::HashSet;
    let mut shattered: HashSet<Entity> = HashSet::new();
    let mut despawned_weapons: HashSet<Entity> = HashSet::new();

    for event in collisions.read() {
        let (a, b) = (event.collider1, event.collider2);

        let asteroid_ship_entity = if asteroids.contains(a) && ships.contains(b) {
            Some((a, b))
        } else if asteroids.contains(b) && ships.contains(a) {
            Some((b, a))
        } else {
            None
        };

        if let Some((asteroid_entity, ship_entity)) = asteroid_ship_entity {
            if shattered.insert(asteroid_entity) {
                shatter_asteroid(
                    &mut commands,
                    &mut meshes,
                    &mut materials,
                    &asteroid_entity,
                    &asteroids,
                );
            }
            if let Ok(mut ship) = ships.get_mut(ship_entity) {
                ship.health -= 10;
                ship.health = ship.health.max(0);
            }
        }

        let asteroid_weapon_entity = if asteroids.contains(a) && weapons.contains(b) {
            Some((a, b))
        } else if asteroids.contains(b) && weapons.contains(a) {
            Some((b, a))
        } else {
            None
        };

        if let Some((asteroid_entity, weapon_entity)) = asteroid_weapon_entity {
            if shattered.insert(asteroid_entity) {
                shatter_asteroid(
                    &mut commands,
                    &mut meshes,
                    &mut materials,
                    &asteroid_entity,
                    &asteroids,
                );
            }
            if despawned_weapons.insert(weapon_entity) {
                safe_despawn(&mut commands, weapon_entity);
            }
        }
    }
}
