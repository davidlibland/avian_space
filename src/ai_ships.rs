use std::f32::consts::PI;

// Some AI for the ships
use crate::asteroids::Asteroid;
use crate::item_universe::ItemUniverse;
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

/// Distance needed to stop from `speed` using the ship's thrust PD controller,
/// assuming the ship is already retrograde-aligned.
///
/// The PD force when braking (forward = retrograde, so forward_speed = −v) is:
///   a(v) = (kp·(max_speed + v) + kd·v).clamp(0, thrust)
///         = (kp·max_speed + (kp+kd)·v).clamp(0, thrust)
///
/// Stopping distance = ∫₀^v  v / a(v) dv
fn braking_distance(speed: f32, thrust: f32, kp: f32, kd: f32, max_speed: f32) -> f32 {
    let pd_base = kp * max_speed; // a(v) at v = 0
    let pd_slope = kp + kd; // da/dv
    // Speed below which PD force is below its thrust ceiling:
    let v_crossover = (thrust - pd_base) / pd_slope;

    if v_crossover <= 0.0 {
        // PD force always saturates → constant deceleration at `thrust`
        speed * speed / (2.0 * thrust)
    } else if speed <= v_crossover {
        // Entirely in the soft-PD regime: a(v) = pd_base + pd_slope·v
        pd_soft_distance(speed, pd_slope, pd_base)
    } else {
        // Constant-thrust phase (speed → v_crossover) + soft-PD phase (v_crossover → 0)
        let d_hard = (speed * speed - v_crossover * v_crossover) / (2.0 * thrust);
        let d_soft = pd_soft_distance(v_crossover, pd_slope, pd_base);
        d_hard + d_soft
    }
}

/// ∫₀^v  x / (pd_base + pd_slope·x) dx  (exact closed form)
fn pd_soft_distance(v: f32, pd_slope: f32, pd_base: f32) -> f32 {
    v / pd_slope - (pd_base / (pd_slope * pd_slope)) * (1.0 + pd_slope * v / pd_base).ln()
}

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
    commands
        .spawn((
            DespawnOnExit(PlayState::Flying),
            AIShip {
                personality: personality.clone(),
            },
            RLAgent::new(personality),
            bundle,
        ))
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
    explosion_writer: &mut MessageWriter<crate::explosions::TriggerExplosion>,
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
    explosion_writer.write(crate::explosions::TriggerExplosion {
        location: edge_pos,
        size: 8.0,
    });

    let bundle = ship_bundle(ship_type, asset_server, item_universe, edge_pos);
    let personality = bundle.get_personality();
    commands
        .spawn((
            DespawnOnExit(PlayState::Flying),
            AIShip {
                personality: personality.clone(),
            },
            RLAgent::new(personality),
            JumpingIn,
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
            commands.entity(entity).remove::<JumpingIn>();
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
    mut explosion_writer: MessageWriter<crate::explosions::TriggerExplosion>,
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
            explosion_writer.write(crate::explosions::TriggerExplosion {
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
    ai_ships: Query<Entity, (With<AIShip>, Without<JumpingOut>)>,
    all_ai_ships: Query<Entity, With<AIShip>>,
    mut explosion_writer: MessageWriter<crate::explosions::TriggerExplosion>,
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
                &mut explosion_writer,
                &ship_type,
            );
        }
    } else if total > dist.max {
        // Pick a random ship (not already jumping out) and trigger jump-out.
        let candidates: Vec<Entity> = ai_ships.iter().collect();
        if !candidates.is_empty() {
            let idx = rng.gen_range(0..candidates.len());
            commands.entity(candidates[idx]).insert(JumpingOut);
        }
    }
}

// ---------------------------------------------------------------------------
// AI control helpers (unchanged)
// ---------------------------------------------------------------------------

/// Maps a local-frame bearing angle (from `angle_to_hit`) to (turn, thrust) commands.
/// Positive angle = target is to the left; negative = to the right.
fn angle_to_controls(angle: f32) -> (f32, f32) {
    if angle > PI / 3. {
        (-1.0, 0.0)
    } else if angle > 0.0 {
        (-0.5, 1.0)
    } else if angle > -PI / 3. {
        (0.5, 1.0)
    } else {
        (1.0, 0.0)
    }
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
/// (caller should clear `ship.target`). Returns `Some` with a coast action
/// when a valid target exists but no firing solution is available.
pub fn compute_ai_action(
    ship: &Ship,
    pos: Vec2,
    vel: Vec2,
    max_speed: f32,
    transform: &Transform,
    all_positions: &Query<&Position>,
    all_velocities: &Query<&LinearVelocity>,
    item_universe: &ItemUniverse,
) -> Option<RawAIAction> {
    let target = ship.target.as_ref()?;

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
            let local_target_vel = rotate_r(target_vel);

            let (turn, thrust) =
                if let Some(hit_angle) = angle_to_hit(max_speed, &local_offset, &local_rel_vel) {
                    angle_to_controls(hit_angle)
                } else {
                    (0.0, 0.0) // no solution → coast
                };

            let mut weapons_to_fire = Vec::new();
            for (weapon_type, _) in ship.weapon_systems.iter_all() {
                let Some(weapon) = item_universe.weapons.get(weapon_type) else {
                    continue;
                };
                if local_offset.length() < weapon.range() {
                    let fire_angle = angle_to_hit(weapon.speed, &local_offset, &local_target_vel);
                    if weapon.guided || angle_indicator(fire_angle) > 0.5 {
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
            let local_offset = rotate_r(target_pos.0 - pos);
            let (turn, thrust) =
                if let Some(angle) = angle_to_hit(max_speed, &local_offset, &Vec2::ZERO) {
                    angle_to_controls(angle)
                } else {
                    (0.0, 0.0)
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
            let offset = target_pos.0 - pos;
            let dist = offset.length();
            let speed = vel.length();

            let d = &ship.data;
            let stop_dist =
                braking_distance(speed, d.thrust, d.thrust_kp, d.thrust_kd, d.max_speed);
            let turn_margin = speed * PI * d.angular_drag / d.torque;
            // Core braking zone: must be decelerating.
            let brake_dist = stop_dist + turn_margin;
            // Outer prepare zone: start the retrograde turn early so the ship
            // is already aligned by the time it reaches the braking zone.
            let prepare_dist = brake_dist + turn_margin;

            if dist < prepare_dist && speed > f32::EPSILON {
                // Retrograde bearing in ego frame — reuses the same angle
                // convention as navigation so angle_to_controls works directly.
                let retro_ego = rotate_r(-vel);
                let retro_angle = retro_ego.y.atan2(retro_ego.x);

                if dist < brake_dist {
                    // Braking zone: turn toward retrograde AND thrust when
                    // reasonably aligned (angle_to_controls gives thrust=1
                    // when within ±60° of the target bearing).
                    let (turn, thrust) = angle_to_controls(retro_angle);
                    Some(RawAIAction {
                        thrust,
                        turn,
                        reverse: 0.0,
                        weapons_to_fire: vec![],
                    })
                } else {
                    // Prepare zone: turn toward retrograde, no thrust yet.
                    let (turn, _) = angle_to_controls(retro_angle);
                    Some(RawAIAction {
                        thrust: 0.0,
                        turn,
                        reverse: 0.0,
                        weapons_to_fire: vec![],
                    })
                }
            } else {
                // Far from planet (or nearly stopped): approach normally.
                let local_offset = rotate_r(offset);
                let (turn, thrust) =
                    if let Some(angle) = angle_to_hit(max_speed, &local_offset, &Vec2::ZERO) {
                        angle_to_controls(angle)
                    } else {
                        (0.0, 0.0)
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
        if let Some(ref tgt) = ship.target.clone() {
            let target_entity = tgt.get_entity();
            let valid = match tgt {
                Target::Planet(_) => all_positions.get(target_entity).is_ok(),
                Target::Ship(_) | Target::Asteroid(_) | Target::Pickup(_) => all_positions
                    .get(target_entity)
                    .map(|p| (p.0 - position.0).length() <= DETECTION_RADIUS)
                    .unwrap_or(false),
            };
            if !valid {
                ship.target = None;
            }
        }

        // 2. If no target, pick one.
        if ship.target.is_none() {
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
                        ship.target = Some(Target::Planet(planet_entity));
                    }
                }
            }

            // If still no target, use personality-based selection.
            if ship.target.is_none() {
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
                    .map(|(e, _)| *e);
                let nearest_asteroid = hits
                    .iter()
                    .find(|(e, _)| asteroid_marker.get(*e).is_ok())
                    .map(|(e, _)| *e);
                let nearest_planet = hits
                    .iter()
                    .find(|(e, _)| planet_marker.get(*e).is_ok())
                    .map(|(e, _)| *e);
                let nearest_ship = hits
                    .iter()
                    .find(|(e, _)| {
                        ship_marker.get(*e).is_ok()
                            && ship_factions
                                .get(*e)
                                .map(|f| ship.should_engage(f))
                                .unwrap_or(false)
                    })
                    .map(|(e, _)| *e);

                // All personalities grab nearby pickups first.
                ship.target = if let Some(pickup) = nearest_pickup {
                    Some(Target::Pickup(pickup))
                } else {
                    match ai_ship.personality {
                        Personality::Miner => nearest_asteroid
                            .map(Target::Asteroid)
                            .or_else(|| nearest_planet.map(Target::Planet)),
                        Personality::Fighter => nearest_ship
                            .map(Target::Ship)
                            .or_else(|| nearest_planet.map(Target::Planet)),
                        Personality::Trader => nearest_planet.map(Target::Planet),
                    }
                };
            }
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
    for (entity, position, ship_vel, max_speed, ship_transform, mut ship, _ai_ship, rl_agent) in
        &mut ships
    {
        // RLAgent ships are driven by rl_step + repeat_actions in both modes.
        if rl_agent.is_some() {
            continue;
        }

        // Act on the current target (assigned by classic_ai_target_selection).
        if ship.target.is_none() {
            continue;
        }
        match compute_ai_action(
            &*ship,
            position.0,
            ship_vel.0,
            max_speed.0,
            ship_transform,
            &all_positions,
            &velocities,
            &item_universe,
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
                ship.target = None;
            }
        }
    }
}

fn land_ship(
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
        match ship.target {
            Some(Target::Planet(planet_entity)) => {
                // ── Landed ───────────────────────────────────────────────────
                if vel.length() < LANDING_SPEED {
                    let Ok((planet, planet_pos)) = planets.get(planet_entity) else {
                        continue;
                    };
                    if (planet_pos.0 - ship_pos.0).length() > LANDING_RADIUS {
                        continue;
                    }
                    // Landed on the target planet → bonus reward for RLAgent ships.
                    if rl_agent.is_some() {
                        let personality_weight = match ai_ship.personality {
                            Personality::Fighter => 0.2,
                            Personality::Miner => 0.5,
                            Personality::Trader => 0.8,
                        };
                        rl_reward_writer.write(RLReward {
                            entity: ship_entity,
                            reward: personality_weight,
                        });
                    }

                    // Landed successfully, so clear the target
                    if matches!(ship.target, Some(Target::Planet(e)) if e == planet_entity) {
                        // Get a new target
                        ship.target = None;
                    }

                    // Repair the ship:
                    ship.health = ship.data.max_health;

                    // Sell all cargo at the planet's listed prices.
                    let planet_name = planet.0.clone();
                    let system_name = current_star_system.0.clone();
                    if let Some(planet_data) = item_universe
                        .star_systems
                        .get(&system_name)
                        .and_then(|s| s.planets.get(&planet_name))
                    {
                        // Compute cargo value before selling (for reward normalisation).
                        let total_cargo_before: u16 = ship.cargo.values().sum();

                        for (commodity, qty) in ship.clone().cargo.iter() {
                            if let Some(&price) = planet_data.commodities.get(commodity) {
                                ship.sell_cargo(commodity, *qty, price);
                            }
                        }

                        // Reward: fraction of cargo sold (normalised by cargo space).
                        if rl_agent.is_some() && total_cargo_before > 0 {
                            let sold_frac =
                                total_cargo_before as f32 / ship.data.cargo_space.max(1) as f32;
                            let personality_weight = match ai_ship.personality {
                                Personality::Fighter => 0.1,
                                Personality::Miner => 0.8,
                                Personality::Trader => 1.0,
                            };
                            rl_reward_writer.write(RLReward {
                                entity: ship_entity,
                                reward: sold_frac * personality_weight,
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
                    if matches!(ai_ship.personality, Personality::Fighter) {
                        let has_hostile =
                            ship_factions.iter().any(|(other_e, other_h, other_pos)| {
                                other_e != ship_entity
                                    && ship.should_engage(other_h)
                                    && (other_pos.0 - ship_pos.0).length() <= DETECTION_RADIUS
                            });
                        if !has_hostile {
                            commands.entity(ship_entity).insert(JumpingOut);
                        }
                    } else {
                        let choose_to_jump = rng.gen_bool(0.1);
                        if choose_to_jump {
                            commands.entity(ship_entity).insert(JumpingOut);
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
