use crate::Layer;
use crate::utils::{polygon_mesh, random_velocity};
use avian2d::prelude::*;
use bevy::prelude::*;
use rand::Rng;

#[derive(Component)]
pub struct Asteroid {
    size: f32,
}
pub fn spawn_asteroid(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
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
            Asteroid { size },
            Mesh2d(meshes.add(mesh)),
            MeshMaterial2d(materials.add(Color::srgb(0.5, 0.5, 0.5))),
            Transform::from_xyz(pos.x, pos.y, 0.0),
            Collider::circle(size),
            CollisionLayers::new(Layer::Asteroid, [Layer::Ship, Layer::Weapon]),
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
    density: f32,
    bound: f32,
) {
    let mut rng = rand::thread_rng();
    let count = (density * (bound / 450.).powi(2)) as usize;
    for _ in 0..count {
        let size: f32 = rng.gen_range(15f32..30f32);
        let x = rng.gen_range(-bound..bound);
        let y = rng.gen_range(-bound..bound);
        spawn_asteroid(
            &mut commands,
            &mut meshes,
            &mut materials,
            size,
            Vec2 { x, y },
            random_velocity(10.0),
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
                    new_size,
                    pos + offset,
                    new_vel,
                );
            }
        }
    }
}
