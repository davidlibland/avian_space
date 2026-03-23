use crate::pickups::PickupDrop;
use crate::utils::{polygon_mesh, random_velocity, safe_despawn};
use crate::{GameLayer, PlayState};
use avian2d::prelude::*;
use bevy::math::FloatPow;
use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::PI;

/// Max units dropped per unit of asteroid size when shattered.
const ASTEROID_DROP_SCALE: f32 = 0.2;
const ASTEROID_VELOCITY: f32 = 50.0;
const ASTEROID_MAX_VELOCITY: f32 = ASTEROID_VELOCITY * 10.;
/// How long (seconds) it takes a respawned asteroid to grow to full size.
const ASTEROID_GROW_DURATION: f32 = 4.0;
/// How often (seconds) we check if asteroid fields need repopulation.
const ASTEROID_RESPAWN_INTERVAL: f32 = 8.0;

pub fn asteroid_plugin(app: &mut App) {
    app.add_message::<ShatterAsteroid>()
        .insert_resource(AsteroidRespawnTimer(Timer::from_seconds(
            ASTEROID_RESPAWN_INTERVAL,
            TimerMode::Repeating,
        )))
        .add_systems(Update, asteroid_field_gravity)
        .add_systems(Update, handle_shatter.run_if(in_state(PlayState::Flying)))
        .add_systems(
            Update,
            (grow_asteroids, respawn_asteroids).run_if(in_state(PlayState::Flying)),
        );
}

/// Marks an asteroid that is still materialising from nothing.
/// The entity's Transform scale runs 0→1 over `ASTEROID_GROW_DURATION` seconds.
/// A `Sensor` component prevents collisions during this phase.
#[derive(Component)]
pub struct GrowingAsteroid {
    /// 0.0 (just spawned) → 1.0 (fully grown).
    pub progress: f32,
}

/// Periodic timer for checking asteroid-field populations.
#[derive(Resource)]
pub struct AsteroidRespawnTimer(pub Timer);

#[derive(Event, Message)]
pub struct ShatterAsteroid(pub Entity);

#[derive(Component)]
pub struct Asteroid {
    pub size: f32,
    field: Entity,
}

#[derive(Deserialize, Serialize)]
pub struct AsteroidFieldData {
    pub location: Vec2,
    pub radius: f32,
    pub number: usize,
    #[serde(default = "default_asteroid_commodities")]
    pub commodities: HashMap<String, f32>,
}

fn default_asteroid_commodities() -> HashMap<String, f32> {
    let mut m = HashMap::new();
    m.insert("iron".to_string(), 1.0);
    m
}

#[derive(Component)]
pub struct AsteroidField {
    pub radius: f32,
    pub number: usize,
    pub commodities: HashMap<String, f32>,
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
            DespawnOnExit(PlayState::Flying),
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
            MaxLinearSpeed(ASTEROID_MAX_VELOCITY),
            MaxAngularSpeed(3.0 * PI),
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
            DespawnOnExit(PlayState::Flying),
            AsteroidField {
                radius: field_data.radius,
                number: field_data.number,
                commodities: field_data.commodities.clone(),
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
        safe_despawn(&mut commands, *asteroid_entity);
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

fn handle_shatter(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut reader: MessageReader<ShatterAsteroid>,
    mut explosion_writer: MessageWriter<crate::explosions::TriggerExplosion>,
    mut pickup_writer: MessageWriter<PickupDrop>,
    asteroids: Query<(&Asteroid, &Transform, &LinearVelocity)>,
    fields: Query<&AsteroidField>,
) {
    use std::collections::HashSet;
    let mut rng = rand::thread_rng();
    let mut shattered: HashSet<Entity> = HashSet::new();
    for ShatterAsteroid(entity) in reader.read() {
        if !shattered.insert(*entity) {
            continue;
        }
        if let Ok((asteroid, transform, _)) = asteroids.get(*entity) {
            explosion_writer.write(crate::explosions::TriggerExplosion {
                location: transform.translation.xy(),
                size: asteroid.size,
            });
            let commodity = fields
                .get(asteroid.field)
                .ok()
                .and_then(|f| {
                    let total: f32 = f.commodities.values().sum();
                    if total <= 0.0 {
                        return None;
                    }
                    let mut roll = rng.gen_range(0.0..total);
                    for (c, &w) in &f.commodities {
                        roll -= w;
                        if roll <= 0.0 {
                            return Some(c.clone());
                        }
                    }
                    f.commodities.keys().next().cloned()
                })
                .unwrap_or_else(|| "iron".to_string());
            let max_qty = ((asteroid.size * ASTEROID_DROP_SCALE) as u16).max(1);
            let qty = rng.gen_range(1..=max_qty);
            pickup_writer.write(PickupDrop {
                location: transform.translation.xy(),
                commodity,
                quantity: qty,
            });
        }
        shatter_asteroid(
            &mut commands,
            &mut meshes,
            &mut materials,
            entity,
            &asteroids,
        );
    }
}

/// Spawn an asteroid that grows in from nothing over `ASTEROID_GROW_DURATION`.
/// It starts as a `Sensor` (no physics collisions) and becomes solid when fully grown.
pub fn spawn_growing_asteroid(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
    field: Entity,
    size: f32,
    pos: Vec2,
    vel: Vec2,
) {
    let mut rng = rand::thread_rng();
    let segments = rng.gen_range(5..10);
    let mut verts: Vec<Vec2> = Vec::new();
    for i in 0..segments {
        let angle = (i as f32 / segments as f32) * std::f32::consts::TAU;
        let r = size * rng.gen_range(0.75..1.25);
        verts.push(Vec2::new(angle.cos() * r, angle.sin() * r));
    }
    let rot = rng.gen_range(-(0.1 * std::f32::consts::PI)..(0.1 * std::f32::consts::PI));
    let mesh = polygon_mesh(&verts);

    // Split across two inserts because Bevy's tuple Bundle limit is 15 items.
    commands
        .spawn((
            DespawnOnExit(PlayState::Flying),
            Asteroid { size, field },
            GrowingAsteroid { progress: 0.0 },
            Mesh2d(meshes.add(mesh)),
            MeshMaterial2d(materials.add(Color::srgb(0.5, 0.5, 0.5))),
            // Start invisible-small; grow_asteroids will lerp scale toward 1.
            Transform::from_xyz(pos.x, pos.y, 0.0).with_scale(Vec3::splat(0.01)),
            Collider::circle(size),
            // Sensor disables collision while growing so it doesn't disrupt nearby objects.
            Sensor,
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
        ))
        .insert((
            ColliderDensity(0.5),
            CollisionEventsEnabled,
            Restitution::new(1.0),
            MaxLinearSpeed(ASTEROID_MAX_VELOCITY),
            MaxAngularSpeed(3.0 * PI),
        ));
}

/// Advance the growth of all `GrowingAsteroid` entities.
/// When fully grown the entity becomes a real solid collider.
fn grow_asteroids(
    mut commands: Commands,
    time: Res<Time>,
    mut growing: Query<(Entity, &mut GrowingAsteroid, &mut Transform)>,
) {
    let dt = time.delta_secs();
    for (entity, mut growing, mut transform) in growing.iter_mut() {
        growing.progress =
            (growing.progress + dt / ASTEROID_GROW_DURATION).min(1.0);
        let scale = growing.progress;
        transform.scale = Vec3::splat(scale);

        if growing.progress >= 1.0 {
            // Fully grown: remove Sensor so it participates in physics.
            commands.entity(entity).remove::<GrowingAsteroid>().remove::<Sensor>();
        }
    }
}

/// Periodically check each asteroid field and spawn new growing asteroids
/// if the live count falls below the field's target number.
fn respawn_asteroids(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut timer: ResMut<AsteroidRespawnTimer>,
    time: Res<Time>,
    fields: Query<(Entity, &AsteroidField, &Transform)>,
    asteroids: Query<&Asteroid>,
) {
    if !timer.0.tick(time.delta()).just_finished() {
        return;
    }
    let mut rng = rand::thread_rng();

    for (field_entity, field, field_transform) in fields.iter() {
        // Count live asteroids belonging to this field (including growing ones).
        let live_count = asteroids.iter().filter(|a| a.field == field_entity).count();
        let target = field.number;
        if live_count >= target {
            continue;
        }
        // Spawn one new asteroid per tick to avoid frame spikes.
        let field_pos = field_transform.translation.xy();
        let r = rng.gen_range((field.radius * 0.5)..(field.radius * 1.5));
        let theta = rng.gen_range(0.0..(2.0 * std::f32::consts::PI));
        let (s, c) = theta.sin_cos();
        let pos = field_pos + Vec2::new(r * c, r * s);
        let size: f32 = rng.gen_range(15f32..30f32);
        let v = (field.radius / r).sqrt() * ASTEROID_VELOCITY;
        let vel = Vec2::new(-s * v, c * v) + random_velocity(ASTEROID_VELOCITY * 0.3);
        let vel = if rng.gen_bool(0.5) { vel } else { -vel };
        spawn_growing_asteroid(
            &mut commands,
            &mut meshes,
            &mut materials,
            field_entity,
            size,
            pos,
            vel,
        );
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
        let offset = field_transform.translation.xy() - asteroid_transform.translation.xy();
        let force_strength = mass.value() * gmass / offset.length().squared();

        force.apply_force(offset.normalize() * force_strength);
    }
}
