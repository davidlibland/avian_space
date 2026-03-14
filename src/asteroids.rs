use crate::{GameLayer, GameState};
use crate::utils::{polygon_mesh, random_velocity};
use avian2d::prelude::*;
use bevy::math::FloatPow;
use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

pub fn asteroid_plugin(app: &mut App) {
    app.add_systems(Update, asteroid_field_gravity);
}

const ASTEROID_VELOCITY: f32 = 50.0;

#[derive(Component)]
pub struct Asteroid {
    size: f32,
    field: Entity,
}

#[derive(Deserialize, Serialize)]
pub struct AsteroidFieldData {
    pub location: Vec2,
    pub radius: f32,
    pub number: usize,
}

#[derive(Component)]
pub struct AsteroidField {
    pub radius: f32,
    pub number: usize,
}

impl AsteroidField {
    fn gmass(&self) -> f32 {
        return self.radius * ASTEROID_VELOCITY.powi(2);
    }
}

pub fn spawn_asteroid(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
    field: Entity,
    size: f32,
    pos: Vec2,
    vel: Vec2,
) -> Entity {
    let mut rng = rand::thread_rng();
    let segments = rng.gen_range(5..10);

    // Build jagged polygon
    let mut verts: Vec<Vec2> = Vec::new();
    for i in 0..segments {
        let angle = (i as f32 / segments as f32) * std::f32::consts::TAU;
        let r = size * rng.gen_range(0.75..1.25);
        verts.push(Vec2::new(angle.cos() * r, angle.sin() * r));
    }
    let rot = rng.gen_range(-(0.1 * std::f32::consts::PI)..(0.1 * std::f32::consts::PI));

    let mesh = polygon_mesh(&verts);
    // Asteroids
    commands
        .spawn((
            DespawnOnExit(GameState::Flying),
            Asteroid { size, field },
            Mesh2d(meshes.add(mesh)),
            MeshMaterial2d(materials.add(Color::srgb(0.5, 0.5, 0.5))),
            Transform::from_xyz(pos.x, pos.y, 0.0),
            Collider::circle(size),
            CollisionLayers::new(
                GameLayer::Asteroid,
                [
                    GameLayer::Ship,
                    GameLayer::Weapon,
                    GameLayer::Asteroid,
                    GameLayer::Radar,
                ],
            ),
            LinearVelocity(vel),
            AngularVelocity(rot),
            RigidBody::Dynamic,
            ColliderDensity(0.5),
            CollisionEventsEnabled,
            Restitution::new(1.0),
        ))
        .id()
}

pub fn build_asteroid_field(
    mut commands: &mut Commands,
    mut meshes: &mut ResMut<Assets<Mesh>>,
    mut materials: &mut ResMut<Assets<ColorMaterial>>,
    field_data: &AsteroidFieldData,
) {
    // Asteroids
    let field = commands
        .spawn((
            DespawnOnExit(GameState::Flying),
            AsteroidField {
                radius: field_data.radius,
                number: field_data.number,
            },
            Transform::from_xyz(field_data.location.x, field_data.location.y, 0.0),
        ))
        .id();
    let mut rng = rand::thread_rng();
    for _ in 0..field_data.number {
        let r = rng.gen_range((field_data.radius * 0.5)..(field_data.radius * 1.5));
        let theta = rng.gen_range(0.0..(2.0 * std::f32::consts::PI));
        let (s, c) = theta.sin_cos();
        let x = r * c;
        let y = r * s;
        let size: f32 = rng.gen_range(15f32..30f32);
        let v = (field_data.radius / r).sqrt() * ASTEROID_VELOCITY;
        let vx = -s * v;
        let vy = c * v;
        let vel = Vec2 { x: vx, y: vy } + random_velocity(ASTEROID_VELOCITY * 0.3);
        // Possibly orbit in other direction.
        let vel = if rng.gen_bool(0.5) { vel } else { -vel };
        spawn_asteroid(
            &mut commands,
            &mut meshes,
            &mut materials,
            field,
            size,
            Vec2 { x, y } + field_data.location,
            vel,
        );
    }
}

pub fn shatter_asteroid(
    mut commands: &mut Commands,
    mut meshes: &mut ResMut<Assets<Mesh>>,
    mut materials: &mut ResMut<Assets<ColorMaterial>>,
    asteroid_entity: &Entity,
    asteroids: &Query<(&Asteroid, &Transform, &LinearVelocity)>,
) {
    if let Ok((asteroid, transform, vel)) = asteroids.get(*asteroid_entity) {
        let mut rng = rand::thread_rng();
        let pos = transform.translation.truncate();
        let size = asteroid.size;
        let field = asteroid.field;
        // Remove the asteroid:
        commands.entity(*asteroid_entity).despawn();
        if size > 10.0 {
            for _ in 0..2 {
                let new_size = rng.gen_range((size * 0.3)..(size * 0.8));
                let new_vel = vel.0 + random_velocity(vel.0.length() * size / new_size);
                let offset = new_size * new_vel / new_vel.length();
                spawn_asteroid(
                    &mut commands,
                    &mut meshes,
                    &mut materials,
                    field,
                    new_size,
                    pos + offset,
                    new_vel,
                );
            }
        }
    }
}

// Apply gravity towards the center of the asteroid field
pub fn asteroid_field_gravity(
    asteroids: Query<(Entity, &Asteroid, &Transform, &ComputedMass), With<RigidBody>>,
    fields: Query<(&AsteroidField, &Transform)>,
    mut forces: Query<Forces>,
) {
    for (asteroid_entity, asteroid, asteroid_transform, mass) in asteroids.iter() {
        let Ok((field, field_transform)) = fields.get(asteroid.field) else {
            continue;
        };
        let Ok(mut force) = forces.get_mut(asteroid_entity) else {
            continue;
        };
        let gmass = field.gmass();
        let offset = (field_transform.translation.xy() - asteroid_transform.translation.xy());
        let force_strength = mass.value() * gmass / offset.length().squared();

        force.apply_force(offset.normalize() * force_strength);
    }
}
