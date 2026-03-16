use std::f32::consts::PI;

// Some AI for the ships
use crate::asteroids::Asteroid;
use crate::item_universe::ItemUniverse;
use crate::pickups::Pickup;
use crate::planets::Planet;
use crate::rl_collection::{AIPlayMode, RLAgent, RLReward};
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

pub fn ai_ship_bundle(app: &mut App) {
    app.add_systems(OnEnter(crate::PlayState::Flying), spawn_ai_ships)
        .add_systems(Update, (simple_ai_control, land_ship));
}

pub fn spawn_ai_ships(
    mut commands: Commands,
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
                let personality = ship_bundle.get_personality();
                commands
                    .spawn((
                        DespawnOnExit(PlayState::Flying),
                        AIShip {
                            personality: personality.clone(),
                        },
                        RLAgent::new(personality),
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
                    let fire_angle =
                        angle_to_hit(weapon.speed, &local_offset, &local_target_vel);
                    if weapon.guided || angle_indicator(fire_angle) > 0.5 {
                        weapons_to_fire.push((weapon_type.clone(), Some(*target_e)));
                    }
                }
            }

            Some(RawAIAction { thrust, turn, reverse: 0.0, weapons_to_fire })
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
            Some(RawAIAction { thrust, turn, reverse: 0.0, weapons_to_fire: vec![] })
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
            let trigger_dist = stop_dist + turn_margin;

            if dist < trigger_dist {
                let ship_forward = ship_dir;
                let thrust = if speed > f32::EPSILON {
                    let retrograde = -vel / speed;
                    if ship_forward.dot(retrograde) > 0.7 { 1.0 } else { 0.0 }
                } else {
                    0.0
                };
                Some(RawAIAction { thrust, turn: 0.0, reverse: 1.0, weapons_to_fire: vec![] })
            } else {
                let local_offset = rotate_r(offset);
                let (turn, thrust) =
                    if let Some(angle) = angle_to_hit(max_speed, &local_offset, &Vec2::ZERO) {
                        angle_to_controls(angle)
                    } else {
                        (0.0, 0.0)
                    };
                Some(RawAIAction { thrust, turn, reverse: 0.0, weapons_to_fire: vec![] })
            }
        }
    }
}

pub fn simple_ai_control(
    mut ship_writer: MessageWriter<ShipCommand>,
    mut weapons_writer: MessageWriter<FireCommand>,
    spatial_query: SpatialQuery,
    mode: Res<AIPlayMode>,
    mut ships: Query<(
        Entity,
        &Position,
        &LinearVelocity,
        &MaxLinearSpeed,
        &Transform,
        &mut Ship,
        &AIShip,
        Option<&RLAgent>,
    )>,
    all_positions: Query<&Position>,
    velocities: Query<&LinearVelocity>,
    planet_marker: Query<(), With<Planet>>,
    asteroid_marker: Query<(), With<Asteroid>>,
    ship_marker: Query<(), With<Ship>>,
    pickup_marker: Query<(), With<Pickup>>,
    planet_names: Query<(Entity, &Planet)>,
    ship_factions: Query<&ShipHostility>,
    current_star_system: Res<CurrentStarSystem>,
    item_universe: Res<ItemUniverse>,
) {
    for (entity, position, ship_vel, max_speed, ship_transform, mut ship, ai_ship, rl_agent) in &mut ships {
        // In RLControl mode, RLAgent ships are driven by rl_step instead.
        if *mode == AIPlayMode::RLControl && rl_agent.is_some() {
            continue;
        }

        // 1. Validate existing target: clear if entity gone or (for combat targets) out of range.
        if let Some(ref tgt) = ship.target.clone() {
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

        // 3. Act on the current target.
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
    // mut collisions: MessageReader<CollisionStart>,
    planets: Query<(&Planet, &Position)>,
    ships: Query<(Entity, &mut Ship, &Position, &LinearVelocity, &AIShip, Option<&RLAgent>)>,
    item_universe: Res<ItemUniverse>,
    current_star_system: Res<CurrentStarSystem>,
    mut rl_reward_writer: MessageWriter<RLReward>,
) {
    for (ship_entity, mut ship, ship_pos, vel, ai_ship, rl_agent) in ships {
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
                        let cargo_value_before: i128 = ship
                            .cargo
                            .iter()
                            .filter_map(|(c, &qty)| {
                                planet_data.commodities.get(c).map(|&p| qty as i128 * p)
                            })
                            .sum();
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
mod tests {
    use super::*;
    use crate::item_universe::ItemUniverse;
    use crate::rl_obs::{controls_to_discrete, discrete_to_controls};
    use avian2d::prelude::{LinearVelocity, Position};
    use bevy::ecs::system::SystemState;
    use bevy::prelude::*;
    use std::collections::HashMap;

    fn empty_item_universe() -> ItemUniverse {
        ItemUniverse {
            weapons: HashMap::new(),
            ships: HashMap::new(),
            star_systems: HashMap::new(),
            outfitter_items: HashMap::new(),
            enemies: HashMap::new(),
            starting_ship: String::new(),
            starting_system: String::new(),
            commodities: HashMap::new(),
            global_average_price: HashMap::new(),
            system_commodity_best_planet_to_sell: HashMap::new(),
            system_planet_best_commodity_to_buy: HashMap::new(),
        }
    }

    // ── Pure-function tests ──────────────────────────────────────────────────

    #[test]
    fn test_angle_to_controls_all_branches() {
        // Large left (> PI/3) → hard left, no thrust
        let (turn, thrust) = angle_to_controls(PI / 2.0);
        assert_eq!(turn, -1.0);
        assert_eq!(thrust, 0.0);

        // Small left (0 < angle ≤ PI/3) → gentle left, thrust
        let (turn, thrust) = angle_to_controls(0.5);
        assert_eq!(turn, -0.5);
        assert_eq!(thrust, 1.0);

        // Small right (-PI/3 ≤ angle < 0) → gentle right, thrust
        let (turn, thrust) = angle_to_controls(-0.5);
        assert_eq!(turn, 0.5);
        assert_eq!(thrust, 1.0);

        // Large right (< -PI/3) → hard right, no thrust
        let (turn, thrust) = angle_to_controls(-PI / 2.0);
        assert_eq!(turn, 1.0);
        assert_eq!(thrust, 0.0);
    }

    #[test]
    fn test_braking_distance_properties() {
        let (thrust, kp, kd, max_speed) = (100.0, 1.0, 2.0, 200.0_f32);

        // Zero speed → zero stopping distance
        assert_eq!(braking_distance(0.0, thrust, kp, kd, max_speed), 0.0);

        // Monotonically increasing with speed
        let d1 = braking_distance(50.0, thrust, kp, kd, max_speed);
        let d2 = braking_distance(100.0, thrust, kp, kd, max_speed);
        let d3 = braking_distance(200.0, thrust, kp, kd, max_speed);
        assert!(d1 > 0.0);
        assert!(d2 > d1);
        assert!(d3 > d2);
    }

    /// The most critical correctness test: verifies the (turn_idx, thrust_idx, ...)
    /// tuple ordering is consistent between `store_obs_actions` (which builds the
    /// DiscreteAction) and `repeat_actions` (which decodes it).
    #[test]
    fn test_discrete_action_ordering_roundtrip() {
        for &(thrust, turn) in &[
            (1.0_f32, -1.0_f32), // thrust + turn left
            (0.0_f32, 1.0_f32),  // no thrust + turn right
            (1.0_f32, 0.0_f32),  // thrust straight
            (0.0_f32, 0.0_f32),  // coast straight
        ] {
            // store_obs_actions: controls → DiscreteAction
            let (thrust_idx, turn_idx) = controls_to_discrete(thrust, turn);
            let action: (u8, u8, u8, u8) = (turn_idx, thrust_idx, 0, 0);

            // repeat_actions: DiscreteAction → controls
            let (decoded_turn_idx, decoded_thrust_idx, _, _) = action;
            let (rt, rr) = discrete_to_controls(decoded_thrust_idx, decoded_turn_idx);

            let expected_thrust = if thrust > 0.5 { 1.0_f32 } else { 0.0 };
            let expected_turn = if turn < -0.25 {
                -1.0_f32
            } else if turn > 0.25 {
                1.0
            } else {
                0.0
            };
            assert_eq!(rt, expected_thrust, "thrust mismatch for input ({}, {})", thrust, turn);
            assert!(
                (rr - expected_turn).abs() < 0.01,
                "turn mismatch for input ({}, {}): got {}, expected {}",
                thrust,
                turn,
                rr,
                expected_turn
            );
        }
    }

    // ── ECS-based tests for compute_ai_action ───────────────────────────────

    #[test]
    fn test_compute_ai_action_no_target() {
        let mut world = World::new();
        let mut state: SystemState<(Query<&Position>, Query<&LinearVelocity>)> =
            SystemState::new(&mut world);
        let (pos_q, vel_q) = state.get(&world);

        let ship = Ship::default();
        let result = compute_ai_action(
            &ship,
            Vec2::ZERO,
            Vec2::ZERO,
            200.0,
            &Transform::default(),
            &pos_q,
            &vel_q,
            &empty_item_universe(),
        );
        assert!(result.is_none(), "no target → should return None");
    }

    #[test]
    fn test_compute_ai_action_target_entity_missing() {
        let mut world = World::new();
        // Spawn entity without Position component so the query will fail
        let ghost = world.spawn_empty().id();

        let mut state: SystemState<(Query<&Position>, Query<&LinearVelocity>)> =
            SystemState::new(&mut world);
        let (pos_q, vel_q) = state.get(&world);

        let mut ship = Ship::default();
        ship.target = Some(Target::Asteroid(ghost));

        let result = compute_ai_action(
            &ship,
            Vec2::ZERO,
            Vec2::ZERO,
            200.0,
            &Transform::default(),
            &pos_q,
            &vel_q,
            &empty_item_universe(),
        );
        assert!(result.is_none(), "missing target position → should return None");
    }

    #[test]
    fn test_compute_ai_action_forward_asteroid() {
        let mut world = World::new();
        // Target 500 units ahead. Default transform faces +y, so +y is "forward".
        let target = world
            .spawn((Position(Vec2::new(0.0, 500.0)), LinearVelocity(Vec2::ZERO)))
            .id();

        let mut state: SystemState<(Query<&Position>, Query<&LinearVelocity>)> =
            SystemState::new(&mut world);
        let (pos_q, vel_q) = state.get(&world);

        let mut ship = Ship::default();
        ship.target = Some(Target::Asteroid(target));

        let result = compute_ai_action(
            &ship,
            Vec2::ZERO,
            Vec2::ZERO,
            200.0,
            &Transform::default(),
            &pos_q,
            &vel_q,
            &empty_item_universe(),
        );
        let action = result.expect("valid forward target should produce an action");
        assert!(
            action.thrust > 0.5,
            "should thrust toward forward target, got {}",
            action.thrust
        );
        assert_eq!(action.reverse, 0.0, "should not brake toward an asteroid");
        assert!(
            action.weapons_to_fire.is_empty(),
            "default ship has no weapons"
        );
    }

    #[test]
    fn test_compute_ai_action_planet_braking() {
        let mut world = World::new();
        // Planet just 50 units away
        let planet = world
            .spawn((Position(Vec2::new(0.0, 50.0)), LinearVelocity(Vec2::ZERO)))
            .id();

        let mut state: SystemState<(Query<&Position>, Query<&LinearVelocity>)> =
            SystemState::new(&mut world);
        let (pos_q, vel_q) = state.get(&world);

        let mut ship = Ship::default();
        ship.data.thrust = 100.0;
        ship.data.thrust_kp = 1.0;
        ship.data.thrust_kd = 2.0;
        ship.data.max_speed = 200.0;
        ship.data.torque = 10.0;
        ship.data.angular_drag = 1.0;
        ship.target = Some(Target::Planet(planet));

        // Moving at 500 m/s toward the planet — braking distance >> 50 units
        let vel = Vec2::new(0.0, 500.0);

        let result = compute_ai_action(
            &ship,
            Vec2::ZERO,
            vel,
            200.0,
            &Transform::default(),
            &pos_q,
            &vel_q,
            &empty_item_universe(),
        );
        let action = result.expect("close planet should produce braking action");
        assert_eq!(action.reverse, 1.0, "should be in braking (reverse) mode");
    }

    #[test]
    fn test_compute_ai_action_planet_approach() {
        let mut world = World::new();
        // Planet far away — 5000 units
        let planet = world
            .spawn((Position(Vec2::new(0.0, 5000.0)), LinearVelocity(Vec2::ZERO)))
            .id();

        let mut state: SystemState<(Query<&Position>, Query<&LinearVelocity>)> =
            SystemState::new(&mut world);
        let (pos_q, vel_q) = state.get(&world);

        let mut ship = Ship::default();
        ship.data.thrust = 100.0;
        ship.data.thrust_kp = 1.0;
        ship.data.thrust_kd = 2.0;
        ship.data.max_speed = 200.0;
        ship.data.torque = 10.0;
        ship.data.angular_drag = 1.0;
        ship.target = Some(Target::Planet(planet));

        let result = compute_ai_action(
            &ship,
            Vec2::ZERO,
            Vec2::ZERO, // stationary, so braking_distance = 0
            200.0,
            &Transform::default(),
            &pos_q,
            &vel_q,
            &empty_item_universe(),
        );
        let action = result.expect("far planet should produce approach action");
        assert_eq!(action.reverse, 0.0, "should not brake when far away");
        assert!(action.thrust > 0.5, "should thrust toward far planet");
    }
}
