use std::f32::consts::PI;

// Some AI for the ships
use crate::CurrentStarSystem;
use crate::GameLayer;
use crate::asteroids::Asteroid;
use crate::item_universe::ItemUniverse;
use crate::planets::Planet;
use crate::ship::{Ship, ShipCommand, ship_bundle};
use crate::utils::{angle_indicator, angle_to_hit};
use crate::weapons::FireCommand;
use crate::weapons::WeaponSystems;
use avian2d::prelude::*;
use bevy::prelude::*;
use rand::Rng;

const DETECTION_RADIUS: f32 = 2000.;

#[derive(Component)]
pub struct AIShip;

pub fn ai_ship_bundle(app: &mut App) {
    app.add_systems(Startup, spawn_ai_ships)
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
        for _ in (0..system_data.ships) {
            let x = rng.gen_range(-1000.0..1000.0);
            let y = rng.gen_range(-1000.0..1000.0);
            // Player
            commands
                .spawn((
                    AIShip, // Mark the player
                    ship_bundle(&asset_server, &item_universe, Vec2::new(x, y)),
                ))
                .with_child((
                    Collider::circle(DETECTION_RADIUS),
                    Sensor,
                    CollisionLayers::new(
                        GameLayer::Radar,
                        [GameLayer::Planet, GameLayer::Asteroid, GameLayer::Ship],
                    ),
                    // ProximitySensor, // marker so we can query this child
                ));
        }
    }
}

pub fn simple_ai_control(
    mut ship_writer: MessageWriter<ShipCommand>,
    mut weapons_writer: MessageWriter<FireCommand>,
    spatial_query: SpatialQuery,
    ships: Query<
        (
            Entity,
            &Position,
            &LinearVelocity,
            &MaxLinearSpeed,
            &Transform,
            &WeaponSystems,
        ),
        With<AIShip>,
    >,
    space_objects: Query<(Entity, &Position, &LinearVelocity)>,
    type_query: Query<(Option<&Planet>, Option<&Asteroid>, Option<&Ship>)>,
    item_universe: Res<ItemUniverse>,
) {
    for (entity, position, ship_vel, max_speed, ship_transform, weapons) in &ships {
        // Only search within detection radius, only hit relevant layers
        let filter = SpatialQueryFilter::from_mask([
            GameLayer::Planet,
            GameLayer::Asteroid,
            GameLayer::Ship,
        ])
        .with_excluded_entities([entity]); // exclude self

        let mut hits: Vec<(Entity, Vec2, Vec2)> = spatial_query
            .shape_intersections(
                &Collider::circle(DETECTION_RADIUS),
                position.0,
                0.0,
                &filter,
            )
            .into_iter()
            .filter_map(|hit| {
                // We need positions — query them separately
                space_objects
                    .get(hit)
                    .ok()
                    .map(|(e, p, vel)| (e, (p.0 - position.0).clone(), vel.0))
            })
            .collect();

        // Sort once, then partition by type
        hits.sort_unstable_by(|a, b| {
            a.1.length_squared()
                .partial_cmp(&b.1.length_squared())
                .unwrap()
        });

        let nearest_planet = hits
            .iter()
            .find(|(e, _, _)| type_query.get(*e).map_or(false, |(p, _, _)| p.is_some()));
        let nearest_asteroid = hits
            .iter()
            .find(|(e, _, _)| type_query.get(*e).map_or(false, |(_, a, _)| a.is_some()));
        let nearest_ship = hits
            .iter()
            .find(|(e, _, _)| type_query.get(*e).map_or(false, |(_, _, s)| s.is_some()));

        // Simply target the nearest ship or asteroid, fly towards it, and shoot it.
        let Some((asteroid, asteroid_offset, asteroid_vel)) = nearest_asteroid else {
            continue;
        };
        let ship_dir = ship_transform.rotation * Vec3::Y;

        // Build rotation R that maps ship_dir onto the +X axis.
        let angle = -ship_dir.y.atan2(ship_dir.x);
        let (sin_a, cos_a) = angle.sin_cos();
        let rotate_r = |v: &Vec2| Vec2::new(v.x * cos_a - v.y * sin_a, v.x * sin_a + v.y * cos_a);
        let asteroid_offset = rotate_r(asteroid_offset);
        let asteroid_rel_vel = rotate_r(&(asteroid_vel - ship_vel.0));
        let asteroid_vel = rotate_r(asteroid_vel);
        let maybe_angle = angle_to_hit((max_speed).0, &asteroid_offset, &asteroid_rel_vel);
        let Some(angle) = maybe_angle else {
            continue;
        };
        let (turn, thrust) = if angle > PI / 3. {
            (-1.0, 0.0)
        } else if angle > 0.0 {
            (-0.5, 1.0)
        } else if angle > -PI / 3. {
            (0.5, 1.0)
        } else {
            (1.0, 0.0)
        };
        ship_writer.write(ShipCommand {
            entity,
            thrust,
            turn,
            reverse: 0.,
        });

        for specific in weapons.primary.iter() {
            if specific.cooldown.just_finished() {
                let Some(weapon) = item_universe.weapons.get(&specific.weapon_type) else {
                    continue;
                };
                if asteroid_offset.length() < weapon.range() {
                    let fire_angle = angle_to_hit(weapon.speed, &asteroid_offset, &asteroid_vel);
                    let fire_indicator = angle_indicator(fire_angle);
                    if fire_indicator > 0.5 {
                        weapons_writer.write(FireCommand {
                            ship: entity,
                            weapon_type: specific.weapon_type.clone(),
                        });
                    }
                }
            }
        }
    }
}
