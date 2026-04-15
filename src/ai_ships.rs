use std::f32::consts::PI;

// Some AI for the ships
use crate::asteroids::Asteroid;
use crate::item_universe::ItemUniverse;
use crate::optimal_control::{pursuit_controls_ego, x_stop};
use crate::pickups::Pickup;
use crate::planets::Planet;
use crate::rl_collection::{RLAgent, RLReward, RLShipJumped, build_rl_ship_jumped};
use crate::ship::{Personality, Ship, ShipCommand, ShipHostility, Target, ship_bundle};
use crate::utils::{angle_indicator, angle_to_hit};
use crate::weapons::FireCommand;
use crate::{CurrentStarSystem, GameLayer, PlayState};
use avian2d::prelude::*;
use bevy::prelude::*;
use rand::Rng;

const DETECTION_RADIUS: f32 = 2000.;
const LANDING_RADIUS: f32 = 150.;
const LANDING_SPEED: f32 = 30.;

/// Speed at which a jumping-in ship enters the system (matches player jump speed).
const JUMP_SPEED: f32 = 1600.0;
/// Jump-in deceleration rate — ship slows from JUMP_SPEED to normal max speed.
const JUMP_IN_DECEL: f32 = 800.0;
/// Jump-out acceleration rate — ship accelerates to JUMP_SPEED then disappears.
const JUMP_OUT_ACCEL: f32 = 2500.0;
/// Radius (units) at which a jumping-in ship is placed at the system edge.
const JUMP_IN_RADIUS: f32 = 6000.0;
/// How often (seconds) to check whether ship population needs adjusting.
const POPULATION_CHECK_INTERVAL: f32 = 5.0;

#[derive(Component)]
pub struct AIShip {
    pub personality: Personality,
}

/// Marks a ship that has just jumped into the system and is decelerating
/// from `JUMP_SPEED` down to its normal `max_speed`.
#[derive(Component)]
pub struct JumpingIn;

/// Marks a ship that is accelerating out of the system.
/// Once it reaches `JUMP_SPEED` it is despawned (with a flash and RL flush).
#[derive(Component)]
pub struct JumpingOut;

/// Periodic timer that checks whether the AI ship population needs adjusting.
#[derive(Resource)]
pub struct ShipPopulationTimer(pub Timer);

pub fn ai_ship_bundle(app: &mut App) {
    app.add_systems(OnEnter(crate::PlayState::Flying), spawn_ai_ships)
        .insert_resource(ShipPopulationTimer(Timer::from_seconds(
            POPULATION_CHECK_INTERVAL,
            TimerMode::Repeating,
        )))
        .add_systems(
            Update,
            (
                classic_ai_target_selection,
                classic_ai_control,
                land_ship,
                jump_in_system,
                jump_out_system,
                manage_ship_population,
            )
                .chain()
                .run_if(in_state(PlayState::Flying)),
        );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Explicit mass properties for a ship turned into a `Sensor`.
/// Sensor colliders don't contribute to mass, so we provide it manually
/// to avoid Avian2D "no mass" warnings on dynamic rigid bodies.
fn sensor_mass_for_ship(radius: f32) -> (Mass, AngularInertia) {
    let density = 2.0; // matches ColliderDensity in ShipBundle
    let mass_val = density * PI * radius * radius;
    let inertia_val = 0.5 * mass_val * radius * radius;
    (Mass(mass_val), AngularInertia(inertia_val))
}

// ---------------------------------------------------------------------------
// Spawn helpers
// ---------------------------------------------------------------------------

/// Spawn an AI ship at a given position (used for initial system load).
fn spawn_ai_ship_at(
    commands: &mut Commands,
    asset_server: &Res<AssetServer>,
    item_universe: &Res<ItemUniverse>,
    ship_type: &str,
    pos: Vec2,
) {
    let bundle = ship_bundle(ship_type, asset_server, item_universe, pos);
    let personality = bundle.get_personality();
    let mut rng = rand::thread_rng();
    let angle = rng.gen_range(0.0..std::f32::consts::TAU);
    commands
        .spawn((
            DespawnOnExit(PlayState::Flying),
            AIShip {
                personality: personality.clone(),
            },
            RLAgent::new(personality),
            bundle,
        ))
        .insert(Transform::from_xyz(pos.x, pos.y, 0.0).with_rotation(Quat::from_rotation_z(angle)))
        .with_child((
            Collider::circle(DETECTION_RADIUS),
            Sensor,
            CollisionLayers::new(
                GameLayer::Radar,
                [GameLayer::Planet, GameLayer::Asteroid, GameLayer::Ship],
            ),
        ));
}

/// Spawn an AI ship that jumps in from the system edge.
///
/// The ship appears at a random point on a circle of radius `JUMP_IN_RADIUS`,
/// moving at `JUMP_SPEED` toward the origin, and carries `JumpingIn` until it
/// decelerates to its normal maximum speed.
fn spawn_ai_ship_jumping_in(
    commands: &mut Commands,
    asset_server: &Res<AssetServer>,
    item_universe: &Res<ItemUniverse>,
    jump_flash_writer: &mut MessageWriter<crate::explosions::TriggerJumpFlash>,
    ship_type: &str,
) {
    let mut rng = rand::thread_rng();
    let theta = rng.gen_range(0.0_f32..(2.0 * PI));
    let edge_pos = Vec2::new(theta.cos(), theta.sin()) * JUMP_IN_RADIUS;
    // Velocity points roughly toward the system center with a small random spread.
    let inbound_dir = -edge_pos.normalize();
    let spread_angle = rng.gen_range(-0.2_f32..0.2_f32);
    let (sa, ca) = spread_angle.sin_cos();
    let dir = Vec2::new(
        inbound_dir.x * ca - inbound_dir.y * sa,
        inbound_dir.x * sa + inbound_dir.y * ca,
    );
    let vel = dir * JUMP_SPEED;

    // Small flash at entry point.
    jump_flash_writer.write(crate::explosions::TriggerJumpFlash {
        location: edge_pos,
        size: 8.0,
    });

    let bundle = ship_bundle(ship_type, asset_server, item_universe, edge_pos);
    let personality = bundle.get_personality();
    let radius = item_universe
        .ships
        .get(ship_type)
        .map(|s| s.radius)
        .unwrap_or(20.0);
    let (mass, inertia) = sensor_mass_for_ship(radius);
    commands
        .spawn((
            DespawnOnExit(PlayState::Flying),
            AIShip {
                personality: personality.clone(),
            },
            RLAgent::new(personality),
            JumpingIn,
            Sensor,
            mass,
            inertia,
            bundle,
        ))
        .insert(LinearVelocity(vel))
        .with_child((
            Collider::circle(DETECTION_RADIUS),
            Sensor,
            CollisionLayers::new(
                GameLayer::Radar,
                [GameLayer::Planet, GameLayer::Asteroid, GameLayer::Ship],
            ),
        ));
}

pub fn spawn_ai_ships(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    item_universe: Res<ItemUniverse>,
    star_system: Res<CurrentStarSystem>,
) {
    if let Some(system_data) = item_universe.star_systems.get(&star_system.0) {
        let dist = &system_data.ships;
        if dist.types.is_empty() {
            return;
        }
        let mut rng = rand::thread_rng();
        let count = rng.gen_range(dist.min..=dist.max.max(dist.min));
        let ship_types = dist.sample(count, &mut rng);
        for ship_type in ship_types {
            let x = rng.gen_range(-1000.0..1000.0);
            let y = rng.gen_range(-1000.0..1000.0);
            spawn_ai_ship_at(
                &mut commands,
                &asset_server,
                &item_universe,
                &ship_type,
                Vec2::new(x, y),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Jump-in / jump-out systems
// ---------------------------------------------------------------------------

/// Decelerate ships that just jumped in until they reach their normal max speed.
fn jump_in_system(
    mut commands: Commands,
    time: Res<Time>,
    mut ships: Query<(Entity, &mut LinearVelocity, &Ship), With<JumpingIn>>,
) {
    let dt = time.delta_secs();
    for (entity, mut vel, ship) in ships.iter_mut() {
        let speed = vel.0.length();
        let target = ship.data.max_speed;
        if speed <= target {
            commands
                .entity(entity)
                .remove::<(JumpingIn, Sensor, Mass, AngularInertia)>();
            // Clamp to max_speed so MaxLinearSpeed constraint isn't violated.
            if speed > f32::EPSILON {
                vel.0 = vel.0.normalize() * target;
            }
        } else {
            let new_speed = (speed - JUMP_IN_DECEL * dt).max(target);
            if speed > f32::EPSILON {
                vel.0 = vel.0.normalize() * new_speed;
            }
        }
    }
}

/// Accelerate ships that are jumping out.  When they reach `JUMP_SPEED` they
/// are despawned with a small flash (and their RL segment is flushed as
/// non-terminal).
fn jump_out_system(
    mut commands: Commands,
    time: Res<Time>,
    mut ships: Query<
        (
            Entity,
            &Transform,
            &mut LinearVelocity,
            &mut MaxLinearSpeed,
            Option<&RLAgent>,
        ),
        With<JumpingOut>,
    >,
    mut jump_flash_writer: MessageWriter<crate::explosions::TriggerJumpFlash>,
    mut rl_jumped_writer: MessageWriter<RLShipJumped>,
) {
    let dt = time.delta_secs();
    for (entity, transform, mut vel, mut max_speed, rl_agent) in ships.iter_mut() {
        let forward = (transform.rotation * Vec3::Y).xy();
        let new_speed = (vel.0.length() + JUMP_OUT_ACCEL * dt).min(JUMP_SPEED);
        vel.0 = forward * new_speed;
        // Keep the physics cap above the current speed.
        max_speed.0 = new_speed;

        if new_speed >= JUMP_SPEED {
            // Flush RL segment as non-terminal before despawning.
            if let Some(agent) = rl_agent {
                let ev = build_rl_ship_jumped(entity, agent);
                rl_jumped_writer.write(ev);
            }
            jump_flash_writer.write(crate::explosions::TriggerJumpFlash {
                location: transform.translation.xy(),
                size: 8.0,
            });
            crate::utils::safe_despawn(&mut commands, entity);
        }
    }
}

// ---------------------------------------------------------------------------
// Population management
// ---------------------------------------------------------------------------

/// Periodically enforce the min/max ship-count bounds for the current system.
///
/// - Below `min`: pick a random ship type from the distribution and jump one in.
/// - Above `max`: pick a random non-jumping ship and trigger a jump-out.
fn manage_ship_population(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    item_universe: Res<ItemUniverse>,
    star_system: Res<CurrentStarSystem>,
    time: Res<Time>,
    mut timer: ResMut<ShipPopulationTimer>,
    ai_ships: Query<(Entity, &Ship), (With<AIShip>, Without<JumpingOut>)>,
    all_ai_ships: Query<Entity, With<AIShip>>,
    mut jump_flash_writer: MessageWriter<crate::explosions::TriggerJumpFlash>,
) {
    if !timer.0.tick(time.delta()).just_finished() {
        return;
    }
    let Some(system_data) = item_universe.star_systems.get(&star_system.0) else {
        return;
    };
    let dist = &system_data.ships;
    if dist.types.is_empty() {
        return;
    }

    let total = all_ai_ships.iter().count();
    let mut rng = rand::thread_rng();

    if total < dist.min {
        // Spawn one ship jumping in.
        let types = dist.sample(1, &mut rng);
        if let Some(ship_type) = types.into_iter().next() {
            spawn_ai_ship_jumping_in(
                &mut commands,
                &asset_server,
                &item_universe,
                &mut jump_flash_writer,
                &ship_type,
            );
        }
    } else if total > dist.max {
        // Pick a random ship (not already jumping out) and trigger jump-out.
        let candidates: Vec<(Entity, &Ship)> = ai_ships.iter().collect();
        if !candidates.is_empty() {
            let idx = rng.gen_range(0..candidates.len());
            let (e, ship) = candidates[idx];
            let (mass, inertia) = sensor_mass_for_ship(ship.data.radius);
            commands
                .entity(e)
                .insert((JumpingOut, Sensor, mass, inertia, AngularVelocity(0.0)));
        }
    }
}

// ---------------------------------------------------------------------------
// AI control helpers (unchanged)
// ---------------------------------------------------------------------------

/// Maps a local-frame bearing angle (from `angle_to_hit`) to (turn, thrust) commands.
/// Positive angle = target is to the left; negative = to the right.
fn angle_to_controls(
    target_angle: f32,
    angular_velocity: f32,
    torque: f32,
    damping: f32,
) -> (f32, f32) {
    let stop_angle = x_stop(angular_velocity, torque, damping); // angle at which max braking would stop rotation
    let turn = if -PI / 16. < target_angle && target_angle < PI / 16. {
        0.0 // small angle → let damping handle it
    } else if target_angle > 0.0 {
        if stop_angle < target_angle {
            -1.0 // max left torque
        } else {
            1.0 // max right torque
        }
    } else {
        if stop_angle > target_angle {
            1.0 // max right torque
        } else {
            -1.0 // max left torque
        }
    };
    let thrust = if target_angle > -PI / 3. && target_angle < PI / 3. {
        1.0
    } else {
        0.0
    };
    return (turn, thrust);
}

/// The action chosen by the rule-based AI for a single decision step.
pub struct RawAIAction {
    pub thrust: f32,
    pub turn: f32,
    pub reverse: f32,
    /// All weapon types to fire, paired with their optional target entity.
    pub weapons_to_fire: Vec<(String, Option<Entity>)>,
}

/// Compute the rule-based AI action for a ship given its current target.
///
/// Returns `None` when the ship has no target or the target entity is missing
/// (caller should clear `ship.nav_target`). Returns `Some` with a coast action
/// when a valid target exists but no firing solution is available.
pub fn compute_ai_action(
    ship: &Ship,
    pos: Vec2,
    vel: Vec2,
    ang_vel: f32,
    max_speed: f32,
    transform: &Transform,
    all_positions: &Query<&Position>,
    all_velocities: &Query<&LinearVelocity>,
    item_universe: &ItemUniverse,
    rng: &mut impl rand::Rng,
) -> Option<RawAIAction> {
    let target = ship.nav_target.as_ref()?;

    let ship_dir = (transform.rotation * Vec3::Y).xy();
    let frame_angle = -ship_dir.y.atan2(ship_dir.x);
    let (sin_a, cos_a) = frame_angle.sin_cos();
    let rotate_r = |v: Vec2| Vec2::new(v.x * cos_a - v.y * sin_a, v.x * sin_a + v.y * cos_a);

    match target {
        Target::Asteroid(target_e) | Target::Ship(target_e) => {
            let target_pos = all_positions.get(*target_e).ok()?;
            let target_vel = all_velocities
                .get(*target_e)
                .map(|v| v.0)
                .unwrap_or(Vec2::ZERO);

            let offset = target_pos.0 - pos;
            let local_offset = rotate_r(offset);
            let local_rel_vel = rotate_r(target_vel - vel);

            // Compute firing cutoff and aim-speed from this ship's unguided
            // weapons. Guided weapons don't need the ship to aim, so they're
            // excluded. Using medians keeps both values anchored to a typical
            // weapon rather than an outlier long-range piece.
            fn median_sorted(mut xs: Vec<f32>) -> Option<f32> {
                if xs.is_empty() {
                    return None;
                }
                xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mid = xs.len() / 2;
                Some(if xs.len() % 2 == 0 {
                    0.5 * (xs[mid - 1] + xs[mid])
                } else {
                    xs[mid]
                })
            }
            let unguided: Vec<&crate::weapons::Weapon> = ship
                .weapon_systems
                .iter_all()
                .filter_map(|(wt, _)| item_universe.weapons.get(wt))
                .filter(|w| !w.guided)
                .collect();
            let median_range =
                median_sorted(unguided.iter().map(|w| w.range()).collect()).unwrap_or(0.0);
            let median_speed =
                median_sorted(unguided.iter().map(|w| w.speed).collect()).unwrap_or(max_speed);
            // 75% of median range: nudge the ship a bit closer before
            // committing to coast+aim.
            let firing_cutoff = 0.75 * median_range;
            let in_firing_range = local_offset.length() < firing_cutoff;

            // If within firing range: coast (no thrust) and aim using the
            // ballistic lead angle computed at the typical projectile speed.
            // Otherwise: pursue using the PD controller.
            let (turn, thrust) = if in_firing_range {
                let turn = if let Some(hit_angle) =
                    angle_to_hit(median_speed, &local_offset, &local_rel_vel)
                {
                    angle_to_controls(hit_angle, ang_vel, ship.data.torque, ship.data.angular_drag)
                        .0
                } else {
                    0.0
                };
                (turn, 0.0)
            } else {
                // Out-of-range pursuit: modest damping so the ship stays
                // near the target long enough to fire, but isn't forced to
                // match its velocity precisely.
                const PURSUIT_DAMPING: f32 = 0.4;
                let (turn, thrust_prob) = pursuit_controls_ego(
                    local_offset,
                    local_rel_vel,
                    ang_vel,
                    ship.data.torque,
                    ship.data.angular_drag,
                    ship.data.thrust,
                    ship.data.max_speed,
                    PURSUIT_DAMPING,
                );
                let thrust = if rng.r#gen::<f32>() < thrust_prob {
                    1.0
                } else {
                    0.0
                };
                (turn, thrust)
            };

            let mut weapons_to_fire = Vec::new();
            for (weapon_type, _) in ship.weapon_systems.iter_all() {
                let Some(weapon) = item_universe.weapons.get(weapon_type) else {
                    continue;
                };
                if local_offset.length() < weapon.range() {
                    let fire_angle = angle_to_hit(weapon.speed, &local_offset, &local_rel_vel);
                    let residual = fire_angle.map(|a| {
                        let wrapped = (a + PI).rem_euclid(2.0 * PI) - PI;
                        wrapped - wrapped.clamp(-weapon.aimable_arc, weapon.aimable_arc)
                    });
                    if weapon.guided || angle_indicator(residual) > 0.5 {
                        weapons_to_fire.push((weapon_type.clone(), Some(*target_e)));
                    }
                }
            }

            Some(RawAIAction {
                thrust,
                turn,
                reverse: 0.0,
                weapons_to_fire,
            })
        }

        Target::Pickup(target_e) => {
            let target_pos = all_positions.get(*target_e).ok()?;
            let target_vel = all_velocities
                .get(*target_e)
                .map(|v| v.0)
                .unwrap_or(Vec2::ZERO);

            let local_offset = rotate_r(target_pos.0 - pos);
            let local_rel_vel = rotate_r(target_vel - vel);
            // Pickups: fly through at speed — just need proximity, not a stop.
            const PICKUP_DAMPING: f32 = 0.2;
            let (turn, thrust_prob) = pursuit_controls_ego(
                local_offset,
                local_rel_vel,
                ang_vel,
                ship.data.torque,
                ship.data.angular_drag,
                ship.data.thrust,
                ship.data.max_speed,
                PICKUP_DAMPING,
            );
            // PWM-style firing: expected thrust ≈ requested acceleration.
            let thrust = if rng.r#gen::<f32>() < thrust_prob {
                1.0
            } else {
                0.0
            };
            Some(RawAIAction {
                thrust,
                turn,
                reverse: 0.0,
                weapons_to_fire: vec![],
            })
        }

        Target::Planet(target_e) => {
            let target_pos = all_positions.get(*target_e).ok()?;
            let target_vel = all_velocities
                .get(*target_e)
                .map(|v| v.0)
                .unwrap_or(Vec2::ZERO);

            let local_offset = rotate_r(target_pos.0 - pos);
            let local_rel_vel = rotate_r(target_vel - vel);
            // Planet landing: full critical damping — must stop on target.
            const LANDING_DAMPING: f32 = 1.0;
            let (turn, thrust_prob) = pursuit_controls_ego(
                local_offset,
                local_rel_vel,
                ang_vel,
                ship.data.torque,
                ship.data.angular_drag,
                ship.data.thrust,
                ship.data.max_speed,
                LANDING_DAMPING,
            );
            // PWM-style firing: expected thrust ≈ requested acceleration.
            let thrust = if rng.r#gen::<f32>() < thrust_prob {
                1.0
            } else {
                0.0
            };
            Some(RawAIAction {
                thrust,
                turn,
                reverse: 0.0,
                weapons_to_fire: vec![],
            })
        }
    }
}

/// Validate and assign targets for non-RLAgent AI ships.
///
/// RLAgent ships are always skipped — their target selection is handled by
/// `choose_target_slot` (BC mode) or the RL policy (RLControl mode).
pub fn classic_ai_target_selection(
    spatial_query: SpatialQuery,
    mut ships: Query<
        (Entity, &Position, &mut Ship, &AIShip, Option<&RLAgent>),
        Without<JumpingOut>,
    >,
    all_positions: Query<&Position>,
    planet_marker: Query<(), With<Planet>>,
    asteroid_marker: Query<(), With<Asteroid>>,
    ship_marker: Query<(), With<Ship>>,
    pickup_marker: Query<(), With<Pickup>>,
    planet_names: Query<(Entity, &Planet)>,
    ship_factions: Query<&ShipHostility>,
    current_star_system: Res<CurrentStarSystem>,
    item_universe: Res<ItemUniverse>,
) {
    for (entity, position, mut ship, ai_ship, rl_agent) in &mut ships {
        // RLAgent ships handle targeting via rl_step / choose_target_slot.
        if rl_agent.is_some() {
            continue;
        }
        // 1. Validate existing target: clear if entity gone or (for combat targets) out of range.
        if let Some(ref tgt) = ship.nav_target.clone() {
            let target_entity = tgt.get_entity();
            let valid = match tgt {
                Target::Planet(_) => all_positions.get(target_entity).is_ok(),
                Target::Ship(_) | Target::Asteroid(_) | Target::Pickup(_) => all_positions
                    .get(target_entity)
                    .map(|p| (p.0 - position.0).length() <= DETECTION_RADIUS)
                    .unwrap_or(false),
            };
            if !valid {
                ship.nav_target = None;
            }
        }

        // 2. If no target, pick one.
        if ship.nav_target.is_none() {
            // Ships with cargo override personality and head to the best sell planet.
            // Traders sell immediately; Miners/Fighters wait until holds are ≥75% full.
            let cargo_used: u16 = ship.cargo.values().sum();
            let should_sell = match ai_ship.personality {
                Personality::Trader => cargo_used > 0,
                _ => false, //cargo_used * 4 >= ship.data.cargo_space * 3,
            };
            if should_sell {
                let system_name = &current_star_system.0;
                if let Some(commodity_to_planet) = item_universe
                    .system_commodity_best_planet_to_sell
                    .get(system_name)
                {
                    let system_data = item_universe.star_systems.get(system_name);
                    // Pick the sell-planet that maximises total value of our cargo.
                    let best = ship
                        .cargo
                        .iter()
                        .filter_map(|(commodity, &qty)| {
                            let planet_name = commodity_to_planet.get(commodity)?;
                            let price = system_data?
                                .planets
                                .get(planet_name)?
                                .commodities
                                .get(commodity)?;
                            let planet_entity = planet_names
                                .iter()
                                .find(|(_, p)| &p.0 == planet_name)
                                .map(|(e, _)| e)?;
                            Some((planet_entity, qty as i128 * price))
                        })
                        .max_by_key(|(_, value)| *value);
                    if let Some((planet_entity, _)) = best {
                        ship.nav_target = Some(Target::Planet(planet_entity));
                    }
                }
            }

            // If still no target, use personality-based selection.
            if ship.nav_target.is_none() {
                let filter = SpatialQueryFilter::from_mask([
                    GameLayer::Planet,
                    GameLayer::Asteroid,
                    GameLayer::Ship,
                    GameLayer::Pickup,
                ])
                .with_excluded_entities([entity]);

                let mut hits: Vec<(Entity, f32)> = spatial_query
                    .shape_intersections(
                        &Collider::circle(DETECTION_RADIUS),
                        position.0,
                        0.0,
                        &filter,
                    )
                    .into_iter()
                    .filter_map(|hit| {
                        all_positions
                            .get(hit)
                            .ok()
                            .map(|p| (hit, (p.0 - position.0).length_squared()))
                    })
                    .collect();

                hits.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                let nearest_pickup = hits
                    .iter()
                    .find(|(e, _)| pickup_marker.get(*e).is_ok())
                    .copied();
                let nearest_asteroid = hits
                    .iter()
                    .find(|(e, _)| asteroid_marker.get(*e).is_ok())
                    .copied();
                let nearest_planet = hits
                    .iter()
                    .find(|(e, _)| planet_marker.get(*e).is_ok())
                    .copied();
                let nearest_ship = hits
                    .iter()
                    .find(|(e, _)| {
                        ship_marker.get(*e).is_ok()
                            && ship_factions
                                .get(*e)
                                .map(|f| ship.should_engage(f))
                                .unwrap_or(false)
                    })
                    .copied();

                // Pickup only preempts the natural target if it's closer.
                let pickup_closer_than = |natural: Option<(Entity, f32)>| -> Option<Target> {
                    match (nearest_pickup, natural) {
                        (Some((p, pd)), Some((_, nd))) if pd < nd => Some(Target::Pickup(p)),
                        (Some((p, _)), None) => Some(Target::Pickup(p)),
                        _ => None,
                    }
                };

                ship.nav_target = match ai_ship.personality {
                    Personality::Miner => {
                        // Miners grab any nearby pickup first.
                        nearest_pickup
                            .map(|(e, _)| Target::Pickup(e))
                            .or_else(|| nearest_asteroid.map(|(e, _)| Target::Asteroid(e)))
                            .or_else(|| nearest_planet.map(|(e, _)| Target::Planet(e)))
                    }
                    Personality::Fighter => pickup_closer_than(nearest_ship)
                        .or_else(|| nearest_ship.map(|(e, _)| Target::Ship(e)))
                        .or_else(|| nearest_planet.map(|(e, _)| Target::Planet(e))),
                    Personality::Trader => pickup_closer_than(nearest_planet)
                        .or_else(|| nearest_planet.map(|(e, _)| Target::Planet(e))),
                };
            }

            // Sync weapons_target: shoot at nav_target when it's a ship or
            // asteroid, otherwise clear (nothing to shoot at for planets/pickups).
            ship.weapons_target = match &ship.nav_target {
                Some(t @ Target::Ship(_)) | Some(t @ Target::Asteroid(_)) => Some(t.clone()),
                _ => None,
            };
        }
    }
}

/// Compute and apply rule-based actions for AI ships.
///
/// Compute and apply rule-based actions for non-RLAgent AI ships.
///
/// RLAgent ships are always skipped — they are driven by `rl_step` and
/// `repeat_actions`. Target selection is performed by
/// [`classic_ai_target_selection`].
pub fn classic_ai_control(
    mut ship_writer: MessageWriter<ShipCommand>,
    mut weapons_writer: MessageWriter<FireCommand>,
    mut ships: Query<
        (
            Entity,
            &Position,
            &LinearVelocity,
            &AngularVelocity,
            &MaxLinearSpeed,
            &Transform,
            &mut Ship,
            &AIShip,
            Option<&RLAgent>,
        ),
        Without<JumpingOut>,
    >,
    all_positions: Query<&Position>,
    velocities: Query<&LinearVelocity>,
    item_universe: Res<ItemUniverse>,
) {
    for (
        entity,
        position,
        ship_vel,
        ang_vel,
        max_speed,
        ship_transform,
        mut ship,
        _ai_ship,
        rl_agent,
    ) in &mut ships
    {
        // RLAgent ships are driven by rl_step + repeat_actions in both modes.
        if rl_agent.is_some() {
            continue;
        }

        // Act on the current target (assigned by classic_ai_target_selection).
        if ship.nav_target.is_none() {
            continue;
        }
        match compute_ai_action(
            &*ship,
            position.0,
            ship_vel.0,
            ang_vel.0,
            max_speed.0,
            ship_transform,
            &all_positions,
            &velocities,
            &item_universe,
            &mut rand::thread_rng(),
        ) {
            Some(action) => {
                ship_writer.write(ShipCommand {
                    entity,
                    thrust: action.thrust,
                    turn: action.turn,
                    reverse: action.reverse,
                });
                for (weapon_type, fire_target) in action.weapons_to_fire {
                    weapons_writer.write(FireCommand {
                        ship: entity,
                        weapon_type,
                        target: fire_target,
                    });
                }
            }
            None => {
                ship.nav_target = None;
            }
        }
    }
}

fn land_ship(
    time: Res<Time>,
    planets: Query<(&Planet, &Position)>,
    mut ships: Query<(
        Entity,
        &mut Ship,
        &Position,
        &LinearVelocity,
        &AIShip,
        Option<&RLAgent>,
    )>,
    item_universe: Res<ItemUniverse>,
    current_star_system: Res<CurrentStarSystem>,
    mut rl_reward_writer: MessageWriter<RLReward>,
    mut commands: Commands,
    ship_factions: Query<(Entity, &ShipHostility, &Position), With<Ship>>,
) {
    let mut rng = rand::thread_rng();
    for (ship_entity, mut ship, ship_pos, vel, ai_ship, rl_agent) in ships.iter_mut() {
        match ship.nav_target {
            Some(Target::Planet(planet_entity)) => {
                // ── Landed ───────────────────────────────────────────────────
                if vel.length() < LANDING_SPEED {
                    let Ok((planet, planet_pos)) = planets.get(planet_entity) else {
                        continue;
                    };
                    if (planet_pos.0 - ship_pos.0).length() > LANDING_RADIUS {
                        continue;
                    }
                    // Resolve planet data for reward computation and transactions.
                    let planet_name = planet.0.clone();
                    let system_name = current_star_system.0.clone();
                    let planet_data = item_universe
                        .star_systems
                        .get(&system_name)
                        .and_then(|s| s.planets.get(&planet_name));

                    // ── Landing reward for RLAgent ships ──────────────────
                    if rl_agent.is_some() {
                        use crate::consts::*;
                        let cargo_held: u16 = ship.cargo.values().sum();
                        let cargo_cap = ship.data.cargo_space.max(1) as f32;
                        let cargo_frac = cargo_held as f32 / cargo_cap;
                        let health_frac = ship.health as f32 / ship.data.max_health.max(1) as f32;

                        let mut reward = 0.0_f32;

                        // Any personality: low health → incentive to heal.
                        reward += (1.0 - health_frac) * LANDING_LOW_HEALTH;

                        match ai_ship.personality {
                            Personality::Trader => {
                                // Can sell: has cargo to unload.
                                if cargo_held > 0 {
                                    reward += cargo_frac * LANDING_TRADER_CAN_SELL;
                                }
                                // Can buy: has cargo space AND credits for at least something.
                                let can_buy = ship.remaining_cargo_space() > 0
                                    && ship.credits > 0
                                    && planet_data
                                        .map(|pd| !pd.commodities.is_empty())
                                        .unwrap_or(false);
                                if can_buy {
                                    reward += LANDING_TRADER_CAN_BUY;
                                }
                            }
                            Personality::Fighter => {
                                // Can rearm: planet sells ammo for a weapon we carry.
                                let can_rearm = planet_data
                                    .map(|pd| {
                                        ship.weapon_systems
                                            .secondary
                                            .keys()
                                            .any(|wt| pd.outfitter.contains(wt) && ship.credits > 0)
                                    })
                                    .unwrap_or(false);
                                if can_rearm {
                                    reward += LANDING_FIGHTER_CAN_REARM;
                                }
                                // Cargo full → sell it off.
                                if ship.remaining_cargo_space() == 0 {
                                    reward += LANDING_FIGHTER_CARGO_FULL;
                                }
                            }
                            Personality::Miner => {
                                // Can sell: has cargo to unload.
                                if cargo_held > 0 {
                                    reward += cargo_frac * LANDING_MINER_CAN_SELL;
                                }
                            }
                        }

                        // Targeting bonus.
                        let targeting_mult = if matches!(ship.nav_target, Some(Target::Planet(e)) if e == planet_entity)
                        {
                            LANDING_ON_TARGET_MULTIPLIER
                        } else {
                            LANDING_OFF_TARGET_MULTIPLIER
                        };
                        reward *= targeting_mult;

                        if reward > 0.0 {
                            rl_reward_writer.write(RLReward {
                                entity: ship_entity,
                                reward,
                                reward_type: crate::consts::REWARD_LANDING,
                            });
                        }
                    }

                    // Snapshot health fraction BEFORE repair — used for
                    // health-gated bonuses fired at landing / cargo_sold.
                    let h_frac_at_landing =
                        ship.health as f32 / ship.data.max_health.max(1) as f32;
                    if rl_agent.is_some() {
                        rl_reward_writer.write(RLReward {
                            entity: ship_entity,
                            reward: crate::consts::HEALTH_BONUS_PER_EVENT * h_frac_at_landing,
                            reward_type: crate::consts::REWARD_HEALTH_GATED,
                        });
                    }

                    // Landed successfully, so clear the target
                    if matches!(ship.nav_target, Some(Target::Planet(e)) if e == planet_entity) {
                        ship.nav_target = None;
                    }

                    // Record this landing so the recent-visited cooldown masks
                    // the planet from nav-target selection until it expires.
                    let now = time.elapsed_secs();
                    let expire_at = now + crate::consts::LANDING_COOLDOWN_SECS;
                    ship.recent_landings.insert(planet_name.clone(), expire_at);
                    // Opportunistic prune of expired entries to keep the map small.
                    ship.recent_landings.retain(|_, &mut t| t > now);

                    // Repair the ship:
                    ship.health = ship.data.max_health;

                    // Sell all cargo at the planet's listed prices.
                    if let Some(planet_data) = planet_data {
                        let credits_before = ship.credits;

                        for (commodity, qty) in ship.clone().cargo.iter() {
                            if let Some(&price) = planet_data.commodities.get(commodity) {
                                ship.sell_cargo(commodity, *qty, price);
                            }
                        }

                        // Reward: credits earned normalised by ship credit scale.
                        let credits_earned = (ship.credits - credits_before).max(0) as f32;
                        if rl_agent.is_some() && credits_earned > 0.0 {
                            use crate::consts::*;
                            let credit_scale = item_universe
                                .ship_credit_scale
                                .get(&ship.ship_type)
                                .copied()
                                .unwrap_or(1.0);
                            let credit_frac = credits_earned / credit_scale;
                            let personality_weight = match ai_ship.personality {
                                Personality::Fighter => CARGO_SOLD_FIGHTER,
                                Personality::Miner => CARGO_SOLD_MINER,
                                Personality::Trader => CARGO_SOLD_TRADER,
                            };
                            rl_reward_writer.write(RLReward {
                                entity: ship_entity,
                                reward: credit_frac * personality_weight,
                                reward_type: crate::consts::REWARD_CARGO_SOLD,
                            });
                            // Health bonus uses pre-repair health (captured above).
                            rl_reward_writer.write(RLReward {
                                entity: ship_entity,
                                reward: crate::consts::HEALTH_BONUS_PER_EVENT * h_frac_at_landing,
                                reward_type: crate::consts::REWARD_HEALTH_GATED,
                            });
                        }

                        // Traders buy the commodity with the best discount here.
                        if matches!(ai_ship.personality, Personality::Trader) {
                            if let Some(commodity) = item_universe
                                .system_planet_best_commodity_to_buy
                                .get(&system_name)
                                .and_then(|m| m.get(&planet_name))
                            {
                                if let Some(&price) = planet_data.commodities.get(commodity) {
                                    ship.buy_cargo(commodity, u16::MAX, price);
                                }
                            }
                        }

                        // Buy Ammo:
                        for weapon_type in ship.clone().weapon_systems.secondary.keys() {
                            if planet_data.outfitter.contains(weapon_type) {
                                ship.buy_max_ammo(weapon_type, &item_universe);
                            }
                        }
                    }

                    // Fighters: if no hostile ships are visible, jump out.
                    let jump_out_bundle = {
                        let (mass, inertia) = sensor_mass_for_ship(ship.data.radius);
                        (JumpingOut, Sensor, mass, inertia, AngularVelocity(0.0))
                    };
                    if matches!(ai_ship.personality, Personality::Fighter) {
                        let has_hostile =
                            ship_factions.iter().any(|(other_e, other_h, other_pos)| {
                                other_e != ship_entity
                                    && ship.should_engage(other_h)
                                    && (other_pos.0 - ship_pos.0).length() <= DETECTION_RADIUS
                            });
                        if !has_hostile {
                            commands.entity(ship_entity).try_insert(jump_out_bundle);
                        }
                    } else {
                        let choose_to_jump = rng.gen_bool(0.1);
                        if choose_to_jump {
                            commands.entity(ship_entity).try_insert(jump_out_bundle);
                        }
                    }
                }
            }
            _ => (),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "tests/ai_ships_tests.rs"]
mod tests;
