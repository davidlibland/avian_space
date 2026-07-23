use avian2d::prelude::LinearVelocity;
use bevy::prelude::*;
use rand::Rng;

use crate::PlayState;
use crate::ship::Ship;

pub fn explosions_plugin(app: &mut App) {
    app.add_message::<TriggerExplosion>()
        .add_message::<TriggerJumpFlash>()
        .add_systems(
            Update,
            (
                trigger_explosions,
                trigger_jump_flashes,
                emit_weapon_particles,
                attach_ship_auras,
            )
                .run_if(in_state(PlayState::Flying)),
        )
        .add_systems(Update, tick_particles.run_if(in_state(PlayState::Flying)));
}

/// Give ships whose data declares an `aura:` their particle emitter (once).
/// Idempotent over all spawn paths — AI spawner, escorts, the player.
fn attach_ship_auras(
    mut commands: Commands,
    ships: Query<(Entity, &crate::ship::Ship), Without<ParticleEmitter>>,
) {
    attach_ship_auras_impl(&mut commands, &ships);
}

#[cfg(test)]
pub fn attach_ship_auras_for_tests(
    mut commands: Commands,
    ships: Query<(Entity, &crate::ship::Ship), Without<ParticleEmitter>>,
) {
    attach_ship_auras_impl(&mut commands, &ships);
}

fn attach_ship_auras_impl(
    commands: &mut Commands,
    ships: &Query<(Entity, &crate::ship::Ship), Without<ParticleEmitter>>,
) {
    for (entity, ship) in ships {
        if let Some(fx) = &ship.data.aura {
            commands.entity(entity).insert(ParticleEmitter {
                fx: fx.clone(),
                color: fx.color.unwrap_or([0.8, 0.6, 1.0]),
                accum: 0.0,
            });
        }
    }
}

// ── Weapon particle trails ───────────────────────────────────────────────────

/// Attached by `weapon_fire` to projectiles/decoys whose weapon declares a
/// `particles:` block — a continuous emitter riding the entity. The flare's
/// "slow firework" look is entirely data: low speed, long particle life.
#[derive(Component)]
pub struct ParticleEmitter {
    pub fx: crate::weapons::ParticleFx,
    /// Base color (from the fx override or the weapon color).
    pub color: [f32; 3],
    /// Fractional particles carried between frames.
    pub accum: f32,
}

/// Emit particles from live weapon emitters (drift/fade is `tick_particles`).
fn emit_weapon_particles(
    mut commands: Commands,
    time: Res<Time>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut emitters: Query<(&Transform, &mut ParticleEmitter)>,
) {
    let mut rng = rand::thread_rng();
    let dt = time.delta_secs();
    for (transform, mut emitter) in &mut emitters {
        emitter.accum += emitter.fx.rate * dt;
        let n = emitter.accum.floor() as usize;
        if n == 0 {
            continue;
        }
        emitter.accum -= n as f32;
        let [r, g, b] = emitter.color;
        for _ in 0..n {
            let angle: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
            let speed: f32 = rng.gen_range(emitter.fx.speed * 0.25..emitter.fx.speed.max(0.26));
            let lifetime =
                rng.gen_range(emitter.fx.particle_lifetime * 0.5..emitter.fx.particle_lifetime);
            let size = rng.gen_range(emitter.fx.size * 0.6..emitter.fx.size * 1.3);
            // Slight per-particle color jitter so the plume shimmers.
            let jitter = rng.gen_range(0.85..1.15);
            let color = Color::srgb(
                (r * jitter).min(1.0),
                (g * jitter).min(1.0),
                (b * jitter).min(1.0),
            );
            commands.spawn((
                DespawnOnExit(PlayState::Flying),
                Particle {
                    lifetime,
                    max_lifetime: lifetime,
                    velocity: Vec2::new(angle.cos(), angle.sin()) * speed,
                },
                Mesh2d(meshes.add(Circle::new(size))),
                MeshMaterial2d(materials.add(ColorMaterial::from_color(color))),
                Transform::from_xyz(transform.translation.x, transform.translation.y, 0.5),
            ));
        }
    }
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

/// Spawn a single free-flying spark particle (fades and decelerates via
/// `tick_particles`). Shared with the fuel-shimmer effect.
pub fn spawn_spark(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    pos: Vec2,
    velocity: Vec2,
    size: f32,
    color: Color,
    lifetime: f32,
) {
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
            let color = Color::srgb(0.7 + 0.3 * t, 0.8 + 0.2 * t, 1.0);

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

// ── Damage smoke ────────────────────────────────────────────────────────────
// Ships emit translucent smoke once damaged past 50%, with the emission rate
// (and density/darkness) scaling with how badly they're hurt. Registered only
// in non-headless mode so it adds no cost to headless RL training.

/// Below this health fraction a ship starts smoking.
const SMOKE_THRESHOLD: f32 = 0.5;
/// Smoke puffs per second at maximum damage (health ≈ 0).
const SMOKE_MAX_RATE: f32 = 22.0;

#[derive(Component)]
struct Smoke {
    lifetime: f32,
    max_lifetime: f32,
    velocity: Vec2,
    max_alpha: f32,
    growth: f32,
    base_size: f32,
}

pub fn ship_smoke_plugin(app: &mut App) {
    app.add_systems(
        Update,
        (emit_ship_smoke, tick_smoke).run_if(in_state(PlayState::Flying)),
    );
}

fn emit_ship_smoke(
    mut commands: Commands,
    time: Res<Time>,
    asset_server: Res<AssetServer>,
    ships: Query<(&Ship, &Transform, Option<&LinearVelocity>)>,
) {
    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }
    let tex = asset_server.load("sprites/effects/smoke.png");
    let mut rng = rand::thread_rng();
    for (ship, transform, vel) in &ships {
        if ship.health <= 0 {
            continue;
        }
        let frac = ship.health as f32 / ship.data.max_health.max(1) as f32;
        if frac >= SMOKE_THRESHOLD {
            continue;
        }
        // 0 at the 50% threshold → 1 near death; drives rate, density, darkness.
        let damage = ((SMOKE_THRESHOLD - frac) / SMOKE_THRESHOLD).clamp(0.0, 1.0);

        // Emit at a rate proportional to the damage (fractional counts via a
        // Bernoulli remainder so low rates still emit smoothly).
        let expected = damage * SMOKE_MAX_RATE * dt;
        let mut count = expected.floor() as u32;
        if rng.gen_range(0.0..1.0) < expected.fract() {
            count += 1;
        }
        if count == 0 {
            continue;
        }
        let radius = ship.data.radius.max(4.0);
        let ship_vel = vel.map(|v| v.0).unwrap_or(Vec2::ZERO);
        let pos = transform.translation.truncate();
        for _ in 0..count {
            let off = Vec2::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)) * radius * 0.5;
            let gray = 0.52 - 0.30 * damage; // darker, sootier with more damage
            let max_alpha = 0.16 + 0.24 * damage; // translucent; denser with damage
            let base_size = radius * rng.gen_range(0.5..0.9);
            let a: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
            // drift outward a little, trailing slightly behind the ship's motion
            let velocity = Vec2::new(a.cos(), a.sin()) * rng.gen_range(6.0..18.0) - ship_vel * 0.12;
            let lifetime = rng.gen_range(0.8..1.6);
            commands.spawn((
                DespawnOnExit(PlayState::Flying),
                Smoke {
                    lifetime,
                    max_lifetime: lifetime,
                    velocity,
                    max_alpha,
                    growth: rng.gen_range(0.8..1.6),
                    base_size,
                },
                Sprite {
                    image: tex.clone(),
                    color: Color::srgba(gray, gray, gray * 1.03, max_alpha),
                    custom_size: Some(Vec2::splat(base_size)),
                    ..default()
                },
                Transform::from_xyz(pos.x + off.x, pos.y + off.y, 0.7),
            ));
        }
    }
}

/// Drift, expand and fade the smoke puffs, despawning when spent.
fn tick_smoke(
    mut commands: Commands,
    time: Res<Time>,
    mut smoke: Query<(Entity, &mut Smoke, &mut Transform, &mut Sprite)>,
) {
    let dt = time.delta_secs();
    for (entity, mut s, mut transform, mut sprite) in &mut smoke {
        s.lifetime -= dt;
        if s.lifetime <= 0.0 {
            commands.entity(entity).despawn();
            continue;
        }
        let t = s.lifetime / s.max_lifetime; // 1 → 0
        let age = 1.0 - t;
        transform.translation.x += s.velocity.x * t * dt;
        transform.translation.y += s.velocity.y * t * dt;
        sprite.custom_size = Some(Vec2::splat(s.base_size * (1.0 + s.growth * age)));
        // fade in briefly, then fade out
        let fade = if age < 0.15 { age / 0.15 } else { t };
        sprite.color = sprite.color.with_alpha(s.max_alpha * fade);
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
