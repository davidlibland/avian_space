use bevy::prelude::*;
use rand::Rng;

use crate::PlayState;

pub fn explosions_plugin(app: &mut App) {
    app.add_message::<TriggerExplosion>()
        .add_message::<TriggerJumpFlash>()
        .add_systems(
            Update,
            (trigger_explosions, trigger_jump_flashes).run_if(in_state(PlayState::Flying)),
        )
        .add_systems(Update, tick_particles.run_if(in_state(PlayState::Flying)));
}

#[derive(Event, Message)]
pub struct TriggerExplosion {
    pub location: Vec2,
    pub size: f32,
}

#[derive(Event, Message)]
pub struct TriggerJumpFlash {
    pub location: Vec2,
    pub size: f32,
}

#[derive(Component)]
struct Particle {
    lifetime: f32,
    max_lifetime: f32,
    velocity: Vec2,
}

fn trigger_explosions(
    mut commands: Commands,
    mut reader: MessageReader<TriggerExplosion>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let mut rng = rand::thread_rng();

    for event in reader.read() {
        let particle_count = (event.size * 2.0) as usize;
        let max_speed = event.size * 3.0;
        let max_lifetime = 0.4 + event.size * 0.01;

        for _ in 0..particle_count {
            let angle: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
            let speed: f32 = rng.gen_range(max_speed * 0.2..max_speed);
            let velocity = Vec2::new(angle.cos(), angle.sin()) * speed;

            let lifetime = rng.gen_range(max_lifetime * 0.5..max_lifetime);
            let size = rng.gen_range(1.5..4.0) * (event.size / 20.0).clamp(0.5, 3.0);

            // Mix between orange core and yellow/white sparks
            let t: f32 = rng.gen_range(0.0..1.0);
            let color = Color::srgb(1.0, rng.gen_range(0.3..0.9) * (1.0 - t * 0.5), t * 0.2);

            let offset = Vec2::new(
                rng.gen_range(-event.size * 0.3..event.size * 0.3),
                rng.gen_range(-event.size * 0.3..event.size * 0.3),
            );
            let pos = event.location + offset;

            commands.spawn((
                DespawnOnExit(PlayState::Flying),
                Particle {
                    lifetime,
                    max_lifetime: lifetime,
                    velocity,
                },
                Mesh2d(meshes.add(Circle::new(size))),
                MeshMaterial2d(materials.add(ColorMaterial::from_color(color))),
                Transform::from_xyz(pos.x, pos.y, 1.0),
            ));
        }
    }
}

fn trigger_jump_flashes(
    mut commands: Commands,
    mut reader: MessageReader<TriggerJumpFlash>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let mut rng = rand::thread_rng();

    for event in reader.read() {
        let particle_count = (event.size * 3.0) as usize;
        let max_speed = event.size * 4.0;
        let max_lifetime = 0.3 + event.size * 0.015;

        for _ in 0..particle_count {
            let angle: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
            let speed: f32 = rng.gen_range(max_speed * 0.3..max_speed);
            let velocity = Vec2::new(angle.cos(), angle.sin()) * speed;

            let lifetime = rng.gen_range(max_lifetime * 0.4..max_lifetime);
            let size = rng.gen_range(1.0..3.0) * (event.size / 20.0).clamp(0.5, 2.5);

            // White-blue palette for hyperspace flash
            let t: f32 = rng.gen_range(0.0..1.0);
            let color = Color::srgb(
                0.7 + 0.3 * t,
                0.8 + 0.2 * t,
                1.0,
            );

            let offset = Vec2::new(
                rng.gen_range(-event.size * 0.2..event.size * 0.2),
                rng.gen_range(-event.size * 0.2..event.size * 0.2),
            );
            let pos = event.location + offset;

            commands.spawn((
                DespawnOnExit(PlayState::Flying),
                Particle {
                    lifetime,
                    max_lifetime: lifetime,
                    velocity,
                },
                Mesh2d(meshes.add(Circle::new(size))),
                MeshMaterial2d(materials.add(ColorMaterial::from_color(color))),
                Transform::from_xyz(pos.x, pos.y, 1.0),
            ));
        }
    }
}

fn tick_particles(
    mut commands: Commands,
    time: Res<Time>,
    mut particles: Query<(
        Entity,
        &mut Particle,
        &mut Transform,
        &MeshMaterial2d<ColorMaterial>,
    )>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let dt = time.delta_secs();
    for (entity, mut particle, mut transform, material_handle) in &mut particles {
        particle.lifetime -= dt;
        if particle.lifetime <= 0.0 {
            commands.entity(entity).despawn();
            continue;
        }

        // Decelerate and move
        let t = particle.lifetime / particle.max_lifetime;
        transform.translation.x += particle.velocity.x * t * dt;
        transform.translation.y += particle.velocity.y * t * dt;

        // Fade out
        if let Some(mat) = materials.get_mut(material_handle) {
            mat.color = mat.color.with_alpha(t * t);
        }
    }
}
