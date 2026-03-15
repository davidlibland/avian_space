use std::f32::consts::PI;

// Some AI for the ships
use crate::asteroids::Asteroid;
use crate::item_universe::ItemUniverse;
use crate::pickups::Pickup;
use crate::planets::Planet;
use crate::ship::{Personality, Ship, ShipCommand, Target, ship_bundle};
use crate::utils::{angle_indicator, angle_to_hit};
use crate::weapons::FireCommand;
use crate::{CurrentStarSystem, GameLayer, PlayState};
use avian2d::prelude::*;
use bevy::prelude::*;
use rand::Rng;

const DETECTION_RADIUS: f32 = 2000.;
const LANDING_RADIUS: f32 = 150.;
const LANDING_SPEED: f32 = 30.;

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
    pub target: Option<Target>,
}

pub fn ai_ship_bundle(app: &mut App) {
    app.add_systems(OnEnter(crate::PlayState::Flying), spawn_ai_ships)
        .add_systems(Update, simple_ai_control);
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
        for (ship_type, count) in system_data.ships.iter() {
            for _ in 0..*count {
                let x = rng.gen_range(-1000.0..1000.0);
                let y = rng.gen_range(-1000.0..1000.0);
                let ship_bundle =
                    ship_bundle(ship_type, &asset_server, &item_universe, Vec2::new(x, y));
                commands
                    .spawn((
                        DespawnOnExit(PlayState::Flying),
                        AIShip {
                            personality: ship_bundle.get_personality(),
                            target: None,
                        },
                        ship_bundle,
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
        }
    }
}

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

pub fn simple_ai_control(
    mut ship_writer: MessageWriter<ShipCommand>,
    mut weapons_writer: MessageWriter<FireCommand>,
    spatial_query: SpatialQuery,
    mut ships: Query<(
        Entity,
        &Position,
        &LinearVelocity,
        &MaxLinearSpeed,
        &Transform,
        &mut Ship,
        &mut AIShip,
    )>,
    all_positions: Query<&Position>,
    velocities: Query<&LinearVelocity>,
    planet_marker: Query<(), With<Planet>>,
    asteroid_marker: Query<(), With<Asteroid>>,
    ship_marker: Query<(), With<Ship>>,
    pickup_marker: Query<(), With<Pickup>>,
    planet_names: Query<(Entity, &Planet)>,
    current_star_system: Res<CurrentStarSystem>,
    item_universe: Res<ItemUniverse>,
) {
    for (entity, position, ship_vel, max_speed, ship_transform, mut ship, mut ai_ship) in &mut ships
    {
        // 1. Validate existing target: clear if entity gone or (for combat targets) out of range.
        if let Some(ref tgt) = ai_ship.target.clone() {
            let target_entity = match tgt {
                Target::Ship(e) | Target::Asteroid(e) | Target::Planet(e) | Target::Pickup(e) => *e,
            };
            let valid = match tgt {
                Target::Planet(_) => all_positions.get(target_entity).is_ok(),
                Target::Ship(_) | Target::Asteroid(_) | Target::Pickup(_) => all_positions
                    .get(target_entity)
                    .map(|p| (p.0 - position.0).length() <= DETECTION_RADIUS)
                    .unwrap_or(false),
            };
            if !valid {
                ai_ship.target = None;
            }
        }

        // 2. If no target, pick one.
        if ai_ship.target.is_none() {
            // Ships with cargo override personality and head to the best sell planet.
            // Traders sell immediately; Miners/Fighters wait until holds are ≥75% full.
            let cargo_used: u16 = ship.cargo.values().sum();
            let should_sell = match ai_ship.personality {
                Personality::Trader => cargo_used > 0,
                _ => cargo_used * 4 >= ship.data.cargo_space * 3,
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
                        ai_ship.target = Some(Target::Planet(planet_entity));
                    }
                }
            }

            // If still no target, use personality-based selection.
            if ai_ship.target.is_none() {
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
                    .find(|(e, _)| ship_marker.get(*e).is_ok())
                    .map(|(e, _)| *e);

                // All personalities grab nearby pickups first.
                ai_ship.target = if let Some(pickup) = nearest_pickup {
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

        // 3. Act on the current target.
        let Some(ref target) = ai_ship.target.clone() else {
            continue;
        };

        // Rotation that maps ship-forward (+Y) onto the local +X axis.
        let ship_dir = ship_transform.rotation * Vec3::Y;
        let frame_angle = -ship_dir.y.atan2(ship_dir.x);
        let (sin_a, cos_a) = frame_angle.sin_cos();
        let rotate_r = |v: &Vec2| Vec2::new(v.x * cos_a - v.y * sin_a, v.x * sin_a + v.y * cos_a);

        match target {
            // ── Pursue and destroy ──────────────────────────────────────────
            Target::Asteroid(target_e) | Target::Ship(target_e) => {
                let Ok(target_pos) = all_positions.get(*target_e) else {
                    ai_ship.target = None;
                    continue;
                };
                let target_vel = velocities.get(*target_e).map(|v| v.0).unwrap_or(Vec2::ZERO);

                let offset = target_pos.0 - position.0;
                let local_offset = rotate_r(&offset);
                let local_rel_vel = rotate_r(&(target_vel - ship_vel.0));
                let local_target_vel = rotate_r(&target_vel);

                let Some(hit_angle) = angle_to_hit(max_speed.0, &local_offset, &local_rel_vel)
                else {
                    continue;
                };
                let (turn, thrust) = angle_to_controls(hit_angle);
                ship_writer.write(ShipCommand {
                    entity,
                    thrust,
                    turn,
                    reverse: 0.,
                });

                for (weapon_type, _) in ship.weapon_systems.iter_all() {
                    let Some(weapon) = item_universe.weapons.get(weapon_type) else {
                        continue;
                    };
                    if local_offset.length() < weapon.range() {
                        let fire_angle =
                            angle_to_hit(weapon.speed, &local_offset, &local_target_vel);
                        if weapon.guided || angle_indicator(fire_angle) > 0.5 {
                            weapons_writer.write(FireCommand {
                                ship: entity,
                                weapon_type: weapon_type.clone(),
                                target: Some(*target_e),
                            });
                        }
                    }
                }
            }

            // ── Collect pickup ──────────────────────────────────────────────
            Target::Pickup(target_e) => {
                let Ok(target_pos) = all_positions.get(*target_e) else {
                    ai_ship.target = None;
                    continue;
                };
                let local_offset = rotate_r(&(target_pos.0 - position.0));
                let Some(bearing_angle) = angle_to_hit(max_speed.0, &local_offset, &Vec2::ZERO)
                else {
                    continue;
                };
                let (turn, thrust) = angle_to_controls(bearing_angle);
                ship_writer.write(ShipCommand {
                    entity,
                    thrust,
                    turn,
                    reverse: 0.,
                });
            }

            // ── Fly to planet, land, trade ───────────────────────────────────
            Target::Planet(target_e) => {
                let Ok(target_pos) = all_positions.get(*target_e) else {
                    ai_ship.target = None;
                    continue;
                };
                let offset = target_pos.0 - position.0;
                let dist = offset.length();
                let speed = ship_vel.0.length();

                // ── Landed ───────────────────────────────────────────────────
                if dist < LANDING_RADIUS && speed < LANDING_SPEED {
                    // Sell all cargo at the planet's listed prices.
                    if let Ok((_, planet)) = planet_names.get(*target_e) {
                        let planet_name = planet.0.clone();
                        let system_name = current_star_system.0.clone();
                        if let Some(planet_data) = item_universe
                            .star_systems
                            .get(&system_name)
                            .and_then(|s| s.planets.get(&planet_name))
                        {
                            for (commodity, qty) in ship.clone().cargo.iter() {
                                if let Some(&price) = planet_data.commodities.get(commodity) {
                                    ship.sell_cargo(commodity, *qty, price);
                                }
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
                        }
                    }
                    ai_ship.target = None;
                    continue;
                }

                // ── Approach / brake ─────────────────────────────────────────
                let d = &ship.data;
                let stop_dist =
                    braking_distance(speed, d.thrust, d.thrust_kp, d.thrust_kd, d.max_speed);
                let turn_margin = speed * PI * d.angular_drag / d.torque;
                let trigger_dist = stop_dist + turn_margin;

                if dist < trigger_dist {
                    // Turn retrograde; once aligned, thrust to decelerate.
                    let ship_forward = ship_dir.xy();
                    let thrust = if speed > f32::EPSILON {
                        let retrograde = -ship_vel.0 / speed;
                        if ship_forward.dot(retrograde) > 0.7 {
                            1.0
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };
                    ship_writer.write(ShipCommand {
                        entity,
                        thrust,
                        turn: 0.,
                        reverse: 1.,
                    });
                } else {
                    // Steer toward the planet and thrust.
                    let local_offset = rotate_r(&offset);
                    let Some(bearing_angle) = angle_to_hit(max_speed.0, &local_offset, &Vec2::ZERO)
                    else {
                        continue;
                    };
                    let (turn, thrust) = angle_to_controls(bearing_angle);
                    ship_writer.write(ShipCommand {
                        entity,
                        thrust,
                        turn,
                        reverse: 0.,
                    });
                }
            }
        }
    }
}
