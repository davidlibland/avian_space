use avian2d::{math::*, prelude::*};
use bevy::prelude::*;

mod asteroids;
mod carrier;
mod comms;
mod experiments;
mod explosions;
mod fbm;
mod game_save;
mod hud;
mod item_universe;
mod jump_ui;
mod main_menu;
mod missions;
mod planet_ui;
mod planets;
use planets::planets_plugin;
mod ai_ships;
mod consts;
mod gae;
mod model;
mod optimal_control;
mod pickups;
mod ppo;
mod rl_collection;
mod rl_obs;
mod session;
mod sfx;
mod ship;
mod ship_anim;
mod space;
mod starfield;
mod surface;
mod surface_character;
mod surface_civilians;
mod surface_npc;
mod surface_objects;
mod surface_pathfinding;
mod surface_terrain;
mod utils;
mod value_fn;
mod weapons;
mod world_assets;

#[cfg(test)]
#[path = "tests/policy_tests.rs"]
mod policy_tests;

use asteroids::{Asteroid, ShatterAsteroid, build_asteroid_field};
use explosions::explosions_plugin;
use game_save::game_save_plugin;
use hud::HudPlugin;
use item_universe::{ItemUniverse, item_universe_plugin};
use jump_ui::jump_ui_plugin;
use main_menu::main_menu_plugin;
use missions::{PlayerEnteredSystem, missions_plugin, missions_ui_plugin};
use planet_ui::planet_ui_plugin;
use planets::NearbyPlanet;
use rl_collection::RLCollectionPlugin;
use ship::{DamageShip, ScoreHit, Ship, ShipCommand, Target};
use starfield::StarfieldPlugin;
use utils::safe_despawn;
use weapons::{FireCommand, Projectile, weapon_lifetime};

// Define your boundary
const BOUNDS: f32 = 10000.0;

#[derive(States, Default, PartialEq, Eq, Hash, Clone, Debug)]
pub enum PlayState {
    #[default]
    MainMenu,
    Flying,
    /// Player is walking around the planet surface (tilemap world).
    Exploring,
    /// Player is at the ship pad — egui Launch/Repair UI, physics paused.
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
    Pickup,
    /// Static surface geometry: building walls, terrain colliders.
    Surface,
    /// Characters on the surface (walker + civilians).  Collides with
    /// Surface but not with other Characters.
    Character,
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

/// Top-level AI control / training mode selected via command-line flags.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AppMode {
    /// AI ships use rule-based control only; no training thread.
    Classic,
    /// AI ships use the RL neural net for control; no training thread.
    Inference,
    /// AI ships use rule-based control; BC training thread is started.
    BCTraining,
    /// AI ships use the RL neural net for control; RL training thread is started.
    RLTraining,
}

/// Parsed command-line arguments.
pub struct AppArgs {
    pub mode: AppMode,
    /// When true, ignore any existing checkpoint and start a fresh run.
    pub fresh: bool,
    /// When true, run without a window or renderer for faster training.
    pub headless: bool,
}

fn parse_args() -> AppArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut mode = AppMode::BCTraining; // default: preserves previous behaviour
    let mut fresh = false;
    let mut headless = false;
    for arg in &args[1..] {
        match arg.as_str() {
            "--classic" => mode = AppMode::Classic,
            "--inference" => mode = AppMode::Inference,
            "--bc-training" | "--bc" => mode = AppMode::BCTraining,
            "--rl-training" | "--rl" => mode = AppMode::RLTraining,
            "--fresh" => fresh = true,
            "--headless" => headless = true,
            _ => {}
        }
    }
    AppArgs {
        mode,
        fresh,
        headless,
    }
}

#[derive(Resource)]
enum ModelMode {
    Training,
    Eval,
}

fn main() {
    let AppArgs {
        mode: app_mode,
        fresh,
        headless,
    } = parse_args();

    let mut app = App::new();

    // ── Core plugins (rendering vs headless) ─────────────────────────────
    if headless {
        use bevy::app::ScheduleRunnerPlugin;
        use bevy::time::TimeUpdateStrategy;
        use bevy::window::WindowPlugin;

        // DefaultPlugins with no window and no exit-on-close, plus a headless
        // schedule runner.  The render pipeline still initialises (to satisfy
        // asset-type registrations) but draws nothing because there is no
        // surface to present to.
        app.add_plugins(
            DefaultPlugins
                .build()
                .disable::<bevy::winit::WinitPlugin>()
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: bevy::window::ExitCondition::DontExit,
                    ..default()
                }),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(
            std::time::Duration::from_millis(1),
        ))
        // Advance 50ms of game time per frame (~10-25x real-time).
        .insert_resource(TimeUpdateStrategy::ManualDuration(
            std::time::Duration::from_millis(50),
        ));
    } else {
        // Game uses pixel-scale world units (radii in tens, distances in hundreds-to-thousands).
        // Default spatial scale (1.0) attenuates anything beyond a few units to silence — shrink
        // it so spatial sounds remain audible across the play area.
        app.add_plugins(DefaultPlugins.set(bevy::audio::AudioPlugin {
            default_spatial_scale: bevy::audio::SpatialScale::new(0.003),
            ..default()
        }));
    }

    // ── Physics ──────────────────────────────────────────────────────────
    app.add_plugins(PhysicsPlugins::default().with_length_unit(20.0));

    // ── Data loading (must come before plugins that use init_session_resource) ──
    app.add_plugins((item_universe_plugin, session::session_plugin));

    // ── Rendering-dependent plugins (skipped in headless) ────────────────
    if headless {
        app.add_plugins(starfield::ToroidalWrapPlugin::default());
        // Register resources that skipped plugins would normally provide
        // but that game-logic systems still depend on.
        app.insert_resource(crate::planet_ui::LandedContext::default());
    } else {
        app.add_plugins((
            planet_ui_plugin,
            jump_ui_plugin,
            StarfieldPlugin::default(),
            HudPlugin::default(),
            comms::comms_plugin,
            main_menu_plugin,
            missions_ui_plugin,
            sfx::sfx_plugin,
        ));
    }
    // Explosions plugin registers messages used by game logic (asteroid shatter),
    // so it's needed even in headless mode (particles just won't render).
    app.add_plugins(explosions_plugin);
    app.add_plugins(world_assets::WorldAssetsPlugin);

    // ── Game-logic plugins ──────────────────────────────────────────────
    app.add_plugins((
        space::space_plugin,    // consolidated space-flight systems
        planets_plugin,         // planet spawning (shared)
        RLCollectionPlugin {
            mode: app_mode,
            fresh,
        },
        game_save_plugin,
        missions_plugin,
        surface::surface_plugin, // walkable planet surface
    ));

    // ── State and resources ──────────────────────────────────────────────
    app.init_state::<PlayState>()
        .init_resource::<TravelContext>()
        .insert_resource(CurrentStarSystem("sol".to_string()))
        .insert_resource(Gravity(Vec2::NEG_Y * 0.0))
        .insert_resource(match app_mode {
            AppMode::BCTraining | AppMode::RLTraining => ModelMode::Training,
            AppMode::Classic | AppMode::Inference => ModelMode::Eval,
        });

    // ── Startup systems ──────────────────────────────────────────────────
    if !headless {
        app.add_systems(Startup, setup); // spawns Camera2d
    }

    // In headless mode, skip the main menu and jump straight to Flying.
    if headless {
        app.add_systems(Startup, |mut next_state: ResMut<NextState<PlayState>>| {
            next_state.set(PlayState::Flying);
        });
    }

    // ── Flying-state systems ─────────────────────────────────────────────
    app.add_systems(
        OnEnter(PlayState::Flying),
        (spawn_asteroids, set_arrival_velocity),
    )
    .add_systems(OnEnter(PlayState::Traveling), reset_nearby_planet);

    if headless {
        // No player → only collision_system (no keyboard_input / accelerate_for_jump).
        app.add_systems(Update, collision_system.run_if(in_state(PlayState::Flying)));
    } else {
        app.add_systems(
            Update,
            (keyboard_input, collision_system, accelerate_for_jump)
                .run_if(in_state(PlayState::Flying)),
        )
        .add_systems(
            Update,
            escape_to_menu.run_if(in_state(PlayState::Flying)),
        );
    }

    app.add_systems(
        Update,
        ApplyDeferred
            .after(collision_system)
            .before(weapon_lifetime),
    )
    .add_systems(Update, travel_system.run_if(in_state(PlayState::Traveling)))
    .run();
}

#[derive(Component)]
pub struct Player;

fn setup(mut commands: Commands) {
    // Camera (persists across all states) — also the spatial audio listener.
    commands.spawn((Camera2d, SpatialListener::default()));
    // Player ship is spawned by game_save_plugin when Flying state is first entered,
    // after the pilot has been selected or loaded from the main menu.
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
            build_asteroid_field(
                &mut commands,
                &mut meshes,
                &mut materials,
                field,
                &item_universe,
            );
        }
    }
}

fn reset_nearby_planet(mut nearby: ResMut<NearbyPlanet>) {
    nearby.0 = None;
}

/// Runs during Flying state: boosts the player ship toward JUMP_SPEED, then triggers Traveling.
fn accelerate_for_jump(
    mut commands: Commands,
    mut ctx: ResMut<TravelContext>,
    mut player_q: Query<
        (
            Entity,
            &Transform,
            &mut LinearVelocity,
            &mut MaxLinearSpeed,
            &Ship,
        ),
        With<Player>,
    >,
    time: Res<Time>,
    mut next_state: ResMut<NextState<PlayState>>,
) {
    if ctx.phase != TravelPhase::Accelerating {
        return;
    }
    let Ok((entity, transform, mut vel, mut max_speed, ship)) = player_q.single_mut() else {
        return;
    };
    // Save the normal cap on the first tick and disable collisions.
    if ctx.saved_max_speed == 0.0 {
        ctx.saved_max_speed = ship.data.max_speed;
        let density = 2.0;
        let r = ship.data.radius;
        let mass_val = density * std::f32::consts::PI * r * r;
        let inertia_val = 0.5 * mass_val * r * r;
        commands.entity(entity).insert((
            Sensor,
            Mass(mass_val),
            AngularInertia(inertia_val),
            AngularVelocity(0.0),
        ));
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
        next_state.set(PlayState::Traveling);
    }
}

/// Runs during Traveling state: advances the hyperspace flash and handles the system switch.
fn travel_system(
    mut ctx: ResMut<TravelContext>,
    mut current_system: ResMut<CurrentStarSystem>,
    mut state: ResMut<NextState<PlayState>>,
    mut entered_writer: MessageWriter<PlayerEnteredSystem>,
    time: Res<Time>,
) {
    if ctx.phase != TravelPhase::Flashing {
        return;
    }
    let was_flashing_past_half = ctx.flash_t >= 0.5;
    ctx.flash_t = (ctx.flash_t + time.delta_secs() / FLASH_DURATION).min(1.0);

    // Switch the star system at the visual peak of the flash (midpoint).
    if ctx.flash_t >= 0.5 && !was_flashing_past_half {
        current_system.0 = ctx.destination.clone();
        entered_writer.write(PlayerEnteredSystem {
            system: current_system.0.clone(),
        });
    }

    if ctx.flash_t >= 1.0 {
        ctx.phase = TravelPhase::Idle;
        state.set(PlayState::Flying);
    }
}

/// Runs OnEnter(Flying): if we just completed a jump, restore max speed and set arrival velocity.
fn set_arrival_velocity(
    mut commands: Commands,
    mut ctx: ResMut<TravelContext>,
    mut player_q: Query<
        (Entity, &Transform, &mut LinearVelocity, &mut MaxLinearSpeed),
        With<Player>,
    >,
) {
    // Only act when returning from a jump (saved_max_speed was set by accelerate_for_jump).
    if ctx.saved_max_speed == 0.0 {
        return;
    }
    let Ok((entity, transform, mut vel, mut max_speed)) = player_q.single_mut() else {
        return;
    };
    let normal_speed = ctx.saved_max_speed;
    max_speed.0 = normal_speed;
    vel.0 = (transform.rotation * Vec3::Y).xy() * normal_speed;
    ctx.saved_max_speed = 0.0; // clear so this doesn't re-run on planet launches
    commands
        .entity(entity)
        .remove::<(Sensor, Mass, AngularInertia)>();
}

/// Save the game and return to the main menu when Escape is pressed.
fn escape_to_menu(
    keyboard: Res<ButtonInput<KeyCode>>,
    game_state: Res<game_save::PlayerGameState>,
    session_data: Res<session::SessionSaveData>,
    mut next_state: ResMut<NextState<PlayState>>,
) {
    if keyboard.just_pressed(KeyCode::Escape) {
        game_save::write_save(&game_state, &session_data);
        next_state.set(PlayState::MainMenu);
    }
}

/// Compute (turn, thrust) to fly toward the ballistic intercept point of the
/// player's current target. Returns `None` if there's no target, no target
/// position available, or no firing solution exists.
///
/// Uses the first primary weapon's projectile speed for the intercept math;
/// falls back to the straight bearing if the player has no primary weapons.
fn compute_intercept_command(
    player_ship: &Ship,
    player_tf: &Transform,
    player_vel: Vec2,
    player_ang_vel: f32,
    target_pos_query: &Query<&Position>,
    target_vel_query: &Query<&LinearVelocity>,
    item_universe: &ItemUniverse,
) -> Option<Scalar> {
    use std::f32::consts::PI;
    let target_e = player_ship.nav_target.as_ref()?.get_entity();
    let target_pos = target_pos_query.get(target_e).ok()?.0;
    let target_vel = target_vel_query
        .get(target_e)
        .map(|v| v.0)
        .unwrap_or(Vec2::ZERO);

    // World → ego frame: forward = +x.
    let ship_dir = (player_tf.rotation * Vec3::Y).xy();
    let frame_angle = -ship_dir.y.atan2(ship_dir.x);
    let (sin_a, cos_a) = frame_angle.sin_cos();
    let rotate_r = |v: Vec2| Vec2::new(v.x * cos_a - v.y * sin_a, v.x * sin_a + v.y * cos_a);

    let player_pos = player_tf.translation.truncate();
    let local_offset = rotate_r(target_pos - player_pos);
    let local_rel_vel = rotate_r(target_vel - player_vel);

    // Pick a projectile speed from any of the player's primary weapons.
    let proj_speed = player_ship
        .weapon_systems
        .primary
        .keys()
        .filter_map(|wt| item_universe.weapons.get(wt))
        .filter(|w| !w.guided)
        .map(|w| w.speed)
        .next();

    let target_angle = match proj_speed {
        Some(speed) => utils::angle_to_hit(speed, &local_offset, &local_rel_vel)?,
        // No primary weapon → just aim straight at the target's current position.
        None => local_offset.y.atan2(local_offset.x),
    };
    // Wrap to [-PI, PI] so the bang-bang controller picks the shorter arc.
    let target_angle = (target_angle + PI).rem_euclid(2.0 * PI) - PI;

    let turn = optimal_control::turn_to_angle(
        target_angle,
        player_ang_vel,
        player_ship.data.torque,
        player_ship.data.angular_drag,
    );
    Some(turn)
}

/// Sends [`MovementAction`] events based on keyboard input.
fn keyboard_input(
    mut writer: MessageWriter<ShipCommand>,
    mut weapons_writer: MessageWriter<FireCommand>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut player_query: Query<
        (
            Entity,
            &Transform,
            &LinearVelocity,
            &AngularVelocity,
            &mut Ship,
        ),
        With<Player>,
    >,
    enemy_ships_query: Query<
        (Entity, &Transform, Option<&missions::MissionTarget>, &ship::ShipHostility),
        (With<Ship>, Without<Player>),
    >,
    asteroids_query: Query<(Entity, &Transform), With<Asteroid>>,
    pickups_query: Query<(Entity, &Transform), With<pickups::Pickup>>,
    planets_query: Query<(Entity, &Transform, &planets::Planet)>,
    target_pos_query: Query<&Position>,
    target_vel_query: Query<&LinearVelocity>,
    current_star_system: Res<CurrentStarSystem>,
    item_universe: Res<ItemUniverse>,
) {
    let Ok((player_entity, player_tf, player_vel, player_ang_vel, mut player_ship)) =
        player_query.single_mut()
    else {
        return; // Player not spawned yet
    };

    // Tab: cycle through ship targets
    if keyboard_input.just_pressed(KeyCode::Tab) {
        let mut entities: Vec<Entity> = enemy_ships_query.iter().map(|(e, ..)| e).collect();
        entities.sort();
        if !entities.is_empty() {
            let current = match &player_ship.nav_target {
                Some(Target::Ship(e)) => Some(*e),
                _ => None,
            };
            let next = match current {
                None => entities[0],
                Some(cur) => entities
                    .iter()
                    .position(|&e| e == cur)
                    .map(|idx| entities[(idx + 1) % entities.len()])
                    .unwrap_or(entities[0]),
            };
            player_ship.nav_target = Some(Target::Ship(next));
        }
    }

    // R: target nearest mission target → nearest hostile → nearest ship
    if keyboard_input.just_pressed(KeyCode::KeyR) {
        let player_pos = player_tf.translation.truncate();
        let nearest_by = |iter: Box<dyn Iterator<Item = Entity> + '_>| -> Option<Entity> {
            iter.filter_map(|e| {
                enemy_ships_query
                    .get(e)
                    .ok()
                    .map(|(_, tf, ..)| (e, (tf.translation.truncate() - player_pos).length()))
            })
            .min_by(|(_, da), (_, db)| da.partial_cmp(db).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(e, _)| e)
        };

        // 1. Nearest mission target
        let target = nearest_by(Box::new(
            enemy_ships_query
                .iter()
                .filter(|(_, _, mt, _)| mt.is_some())
                .map(|(e, ..)| e),
        ))
        // 2. Nearest hostile ship
        .or_else(|| {
            nearest_by(Box::new(
                enemy_ships_query
                    .iter()
                    .filter(|(_, _, _, hostility)| player_ship.should_engage(hostility))
                    .map(|(e, ..)| e),
            ))
        })
        // 3. Nearest any ship
        .or_else(|| {
            nearest_by(Box::new(enemy_ships_query.iter().map(|(e, ..)| e)))
        });

        if let Some(entity) = target {
            player_ship.nav_target = Some(Target::Ship(entity));
        }
    }

    // Q: select nearest asteroid target
    if keyboard_input.just_pressed(KeyCode::KeyQ) {
        let player_pos = player_tf.translation.truncate();
        let nearest = asteroids_query
            .iter()
            .map(|(e, tf)| (e, (tf.translation.truncate() - player_pos).length()))
            .min_by(|(_, da), (_, db)| da.partial_cmp(db).unwrap_or(std::cmp::Ordering::Equal));
        if let Some((entity, _)) = nearest {
            player_ship.nav_target = Some(Target::Asteroid(entity));
        }
    }

    // P: select nearest pickup
    if keyboard_input.just_pressed(KeyCode::KeyP) {
        let player_pos = player_tf.translation.truncate();
        let nearest = pickups_query
            .iter()
            .map(|(e, tf)| (e, (tf.translation.truncate() - player_pos).length()))
            .min_by(|(_, da), (_, db)| da.partial_cmp(db).unwrap_or(std::cmp::Ordering::Equal));
        if let Some((entity, _)) = nearest {
            player_ship.nav_target = Some(Target::Pickup(entity));
        }
    }

    // [ / ]: cycle through planets in the current system
    if let Some(current_system) = item_universe.star_systems.get(&current_star_system.0) {
        let cycle_planets = keyboard_input.just_pressed(KeyCode::BracketLeft) as i32
            - keyboard_input.just_pressed(KeyCode::BracketRight) as i32;
        if cycle_planets != 0 {
            let mut entities: Vec<Entity> = planets_query
                .iter()
                .filter(|(_, _, p)| {
                    // Only include planets that can be landed on.
                    current_system
                        .planets
                        .get(&p.0)
                        .map(|pdata| !pdata.uncolonized)
                        .unwrap_or(false)
                })
                .map(|(e, _, _)| e)
                .collect();
            entities.sort();
            if !entities.is_empty() {
                let current = match &player_ship.nav_target {
                    Some(Target::Planet(e)) => Some(*e),
                    _ => None,
                };
                let n = entities.len() as i32;
                let idx = match current {
                    None => 0,
                    Some(cur) => entities.iter().position(|&e| e == cur).unwrap_or(0) as i32,
                };
                // `[` goes backward (cycle_planets = +1), `]` goes forward (-1).
                let next_idx = ((idx - cycle_planets).rem_euclid(n)) as usize;
                player_ship.nav_target = Some(Target::Planet(entities[next_idx]));
            }
        }
    }

    // Mirror nav_target onto weapons_target for combat-relevant targets;
    // clear it otherwise so weapons don't aim at the previous selection.
    match player_ship.nav_target {
        Some(Target::Ship(_) | Target::Asteroid(_)) => {
            player_ship.weapons_target = player_ship.nav_target.clone();
        }
        _ => {
            player_ship.weapons_target = None;
        }
    }

    // A (held): autopilot — turn toward the ballistic intercept angle for the
    // current target, and thrust forward when roughly aimed.
    let intercept = keyboard_input.pressed(KeyCode::KeyA);
    let auto_turn_cmd = if intercept {
        compute_intercept_command(
            &player_ship,
            player_tf,
            player_vel.0,
            player_ang_vel.0,
            &target_pos_query,
            &target_vel_query,
            &item_universe,
        )
    } else {
        None
    };

    let left = keyboard_input.any_pressed([KeyCode::ArrowLeft]);
    let right = keyboard_input.any_pressed([KeyCode::ArrowRight]);
    let up = keyboard_input.any_pressed([KeyCode::ArrowUp]);
    let down = keyboard_input.any_pressed([KeyCode::ArrowDown]);

    let horizontal = right as i8 - left as i8;
    let mut turn = horizontal as Scalar;

    let thrust = (up as i8) as Scalar;
    let reverse = (down as i8) as Scalar;

    // Intercept autopilot overrides turn/thrust when the player isn't already
    // giving arrow-key input on those axes. Arrow keys always win so the
    // player can course-correct while holding I.
    if let Some(auto_turn) = auto_turn_cmd {
        if turn.abs() < f32::EPSILON {
            turn = auto_turn;
        }
    }

    // Only send if there's actual input
    if thrust.abs() > f32::EPSILON || turn.abs() > f32::EPSILON || reverse.abs() > f32::EPSILON {
        writer.write(ShipCommand {
            entity: player_entity,
            thrust,
            turn,
            reverse,
        });
    }

    let fire_primary = keyboard_input.any_pressed([KeyCode::Space]);
    if fire_primary {
        let target_e = player_ship.weapons_target.as_ref().map(|t| t.get_entity());
        for weapon_type in player_ship.weapon_systems.primary.keys() {
            weapons_writer.write(FireCommand {
                ship: player_entity,
                weapon_type: weapon_type.clone(),
                target: target_e,
            });
        }
    }

    let select_secondary = keyboard_input.any_pressed([KeyCode::KeyW]);
    if select_secondary {
        player_ship.weapon_systems.increment_secondary();
    }

    let fire_secondary = keyboard_input.any_pressed([KeyCode::ShiftLeft]);
    if fire_secondary {
        if let Some(selected) = &player_ship.weapon_systems.selected_secondary {
            weapons_writer.write(FireCommand {
                ship: player_entity,
                weapon_type: selected.clone(),
                target: player_ship.weapons_target.clone().map(|t| t.get_entity()), // guided missiles auto-acquire the nearest enemy
            });
        }
    }
}

fn collision_system(
    mut collisions: MessageReader<CollisionStart>,
    mut damage_writer: MessageWriter<DamageShip>,
    mut shatter_writer: MessageWriter<ShatterAsteroid>,
    mut score_hit_writer: MessageWriter<ScoreHit>,
    mut commands: Commands,
    asteroids: Query<(), With<Asteroid>>,
    ships: Query<&Ship, Without<Sensor>>,
    weapons: Query<&Projectile>,
    item_universe: Res<ItemUniverse>,
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
                shatter_writer.write(ShatterAsteroid(asteroid_entity));
            }
            damage_writer.write(DamageShip {
                entity: ship_entity,
                damage: 10.0,
            });
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
                shatter_writer.write(ShatterAsteroid(asteroid_entity));
            }
            if despawned_weapons.insert(weapon_entity) {
                safe_despawn(&mut commands, weapon_entity);
            }
            if let Ok(projectile) = weapons.get(weapon_entity) {
                score_hit_writer.write(ScoreHit::OnAsteroid {
                    source: projectile.owner,
                });
            }
        }

        let weapon_ship_entity = if weapons.contains(a) && ships.contains(b) {
            Some((a, b))
        } else if weapons.contains(b) && ships.contains(a) {
            Some((b, a))
        } else {
            None
        };

        if let Some((weapon_entity, ship_entity)) = weapon_ship_entity {
            if let Ok(projectile) = weapons.get(weapon_entity) {
                if projectile.owner == ship_entity {
                    continue;
                }
                if despawned_weapons.insert(weapon_entity) {
                    safe_despawn(&mut commands, weapon_entity);
                    let damage = item_universe
                        .weapons
                        .get(&projectile.weapon_type)
                        .map_or(0.0, |w| w.damage as f32);
                    damage_writer.write(DamageShip {
                        entity: ship_entity,
                        damage,
                    });
                    score_hit_writer.write(ScoreHit::OnShip {
                        source: projectile.owner,
                        target: ship_entity,
                    });
                }
            }
        }
    }
}
