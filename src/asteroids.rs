use crate::item_universe::ItemUniverse;
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

const ASTEROID_VELOCITY: f32 = 50.0;
const ASTEROID_MAX_VELOCITY: f32 = ASTEROID_VELOCITY * 10.;
/// How long (seconds) it takes a respawned asteroid to grow to full size.
const ASTEROID_GROW_DURATION: f32 = 4.0;
/// How often (seconds) we check if asteroid fields need repopulation.
const ASTEROID_RESPAWN_INTERVAL: f32 = 1.0;
/// How long (seconds) an asteroid takes to fade away.
const ASTEROID_FADE_DURATION: f32 = 1.0;
/// Fade if distance-from-field-center < this fraction of field radius.
const ASTEROID_FADE_INNER_FRAC: f32 = 0.15;
/// Fade if distance-from-field-center > this fraction of field radius.
const ASTEROID_FADE_OUTER_FRAC: f32 = 3.0;

/// Number of baked asteroid shapes (rock_<i>.png / dep_<i>.png) and tumble
/// frames per shape — MUST match scripts/ship3d/asteroid_gen.py.
const ASTEROID_SHAPES: usize = 12;
const ASTEROID_TUMBLE_FRAMES: usize = 64;
/// On-screen sprite size = radius × this. The baked rock fills ~0.9 of the
/// frame's half-extent (ortho 2.2), so this maps the tile to ~2·radius.
const ASTEROID_SPRITE_SCALE: f32 = 2.44;

/// Loaded rock + deposit tumble atlases and their shared layout.
#[derive(Resource, Default)]
pub struct AsteroidAtlases {
    rocks: Vec<Handle<Image>>,
    deps: Vec<Handle<Image>>,
    layout: Option<Handle<TextureAtlasLayout>>,
}

/// Time-based tumble clock; the same phase/speed on the rock and its deposit
/// child keeps them frame-synced.
#[derive(Component, Clone)]
pub struct AsteroidTumble {
    phase: f32,
    speed: f32,
}

fn load_asteroid_atlases(
    asset_server: Res<AssetServer>,
    layouts: Option<ResMut<Assets<TextureAtlasLayout>>>,
    mut commands: Commands,
) {
    // 8×8 grid of 128px tiles (64 tumble frames). Optional layout = headless-safe.
    let layout = layouts.map(|mut l| {
        l.add(TextureAtlasLayout::from_grid(
            UVec2::splat(128),
            8,
            8,
            None,
            None,
        ))
    });
    let mut rocks = Vec::new();
    let mut deps = Vec::new();
    for i in 0..ASTEROID_SHAPES {
        rocks.push(asset_server.load(format!("sprites/asteroids/rock_{i}.png")));
        deps.push(asset_server.load(format!("sprites/asteroids/dep_{i}.png")));
    }
    commands.insert_resource(AsteroidAtlases {
        rocks,
        deps,
        layout,
    });
}

/// Build an atlas sprite (or plain image fallback when headless / no layout).
fn atlas_sprite(
    handle: &Handle<Image>,
    layout: &Option<Handle<TextureAtlasLayout>>,
    px: f32,
    tint: Color,
) -> Sprite {
    let mut s = match layout {
        Some(l) => Sprite::from_atlas_image(
            handle.clone(),
            TextureAtlas {
                layout: l.clone(),
                index: 0,
            },
        ),
        None => Sprite::from_image(handle.clone()),
    };
    s.color = tint;
    s.custom_size = Some(Vec2::splat(px));
    s
}

/// Weighted-random pick of one commodity colour for an asteroid's deposits, so
/// a mixed field shows a mix of single-ore asteroids the player can pick from.
fn pick_tint(colors: &[([f32; 3], f32)], rng: &mut impl Rng) -> Color {
    if colors.is_empty() {
        return Color::srgb(0.7, 0.6, 0.45);
    }
    let total: f32 = colors.iter().map(|(_, w)| w).sum();
    let chosen = if total <= 0.0 {
        colors[0].0
    } else {
        let mut roll = rng.gen_range(0.0..total);
        let mut pick = colors[0].0;
        for (c, w) in colors {
            roll -= w;
            if roll <= 0.0 {
                pick = *c;
                break;
            }
        }
        pick
    };
    Color::srgb(chosen[0], chosen[1], chosen[2])
}

/// Spawn the rock + tinted deposit-child sprites with a synced tumble clock.
/// Shared by `spawn_asteroid` and `spawn_growing_asteroid`.
fn asteroid_visual_bundle(
    atlases: &AsteroidAtlases,
    size: f32,
    colors: &[([f32; 3], f32)],
    rng: &mut impl Rng,
) -> (Sprite, AsteroidTumble, usize, Color) {
    let shape = rng.gen_range(0..atlases.rocks.len());
    let tumble = AsteroidTumble {
        phase: rng.gen_range(0.0..1.0),
        speed: rng.gen_range(0.04..0.12),
    };
    let px = size * ASTEROID_SPRITE_SCALE;
    let rock = atlas_sprite(&atlases.rocks[shape], &atlases.layout, px, Color::WHITE);
    let tint = pick_tint(colors, rng);
    (rock, tumble, shape, tint)
}

/// Advance every asteroid's (and deposit child's) tumble frame over time. The
/// physics `AngularVelocity` separately spins it in-plane.
fn tumble_asteroids(time: Res<Time>, mut q: Query<(&AsteroidTumble, &mut Sprite)>) {
    let t = time.elapsed_secs();
    for (tumble, mut sprite) in &mut q {
        if let Some(atlas) = sprite.texture_atlas.as_mut() {
            let frac = (t * tumble.speed + tumble.phase).rem_euclid(1.0);
            atlas.index =
                ((frac * ASTEROID_TUMBLE_FRAMES as f32) as usize) % ASTEROID_TUMBLE_FRAMES;
        }
    }
}

pub fn asteroid_plugin(app: &mut App) {
    app.add_message::<ShatterAsteroid>()
        .insert_resource(AsteroidRespawnTimer(Timer::from_seconds(
            ASTEROID_RESPAWN_INTERVAL,
            TimerMode::Repeating,
        )))
        .init_resource::<AsteroidAtlases>()
        .add_systems(Startup, load_asteroid_atlases)
        // FixedUpdate, NOT Update: avian consumes accumulated forces once
        // per physics step (FixedPostUpdate). Applying per rendered frame
        // made gravity framerate-dependent — and while the game was PAUSED
        // (physics stopped, Update still running) the force banked up and
        // fired all at once on unpause, blasting the field apart.
        .add_systems(FixedUpdate, asteroid_field_gravity)
        .add_systems(Update, handle_shatter.run_if(in_state(PlayState::Flying)))
        .add_systems(
            Update,
            (
                grow_asteroids,
                respawn_asteroids,
                fade_asteroids,
                tumble_asteroids,
                check_asteroid_bounds,
                asteroid_planet_collisions,
            )
                .run_if(in_state(PlayState::Flying)),
        );
}

/// Marks an asteroid that is fading out of existence (shrinking, then despawn).
/// Will not drop pickups or spawn children.
#[derive(Component)]
pub struct FadingAsteroid {
    pub progress: f32,
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
pub struct ShatterAsteroid {
    pub entity: Entity,
    /// Whether the shatter drops commodity pickups. `true` for weapon hits
    /// (the intended way to mine); `false` for ship-ramming, so a miner must
    /// actually shoot an asteroid to get any reward from it.
    pub drop_pickups: bool,
}

#[derive(Component)]
pub struct Asteroid {
    pub size: f32,
    pub field: Entity,
    /// A hydrogen-ice rock: shimmers (via `FuelShimmer`) and always drops
    /// `fuel` when shot, so a dry ship can mine its way home.
    pub fuel: bool,
}

/// Roll whether a rock spawned in `commodities` is a fuel rock, using the
/// field's `fuel` weight as the probability (weight is a 0..1-ish share).
pub(crate) fn rolls_fuel(commodities: &HashMap<String, f32>, rng: &mut impl Rng) -> bool {
    let w = commodities
        .get(crate::ship::FUEL_COMMODITY)
        .copied()
        .unwrap_or(0.0);
    w > 0.0 && rng.gen_range(0.0..1.0) < w.min(1.0)
}

#[derive(Clone, Deserialize, Serialize)]
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
        self.radius * ASTEROID_VELOCITY.powi(2)
    }
}

/// Build weighted commodity colors from a field's commodities map and the item universe.
fn commodity_colors(
    commodities: &HashMap<String, f32>,
    item_universe: &ItemUniverse,
) -> Vec<([f32; 3], f32)> {
    commodities
        .iter()
        .filter_map(|(name, &weight)| {
            item_universe
                .commodities
                .get(name)
                .map(|c| (c.color, weight))
        })
        .collect()
}

/// Spawn small irregular polygon "crystal" children on an asteroid's surface.
#[allow(dead_code)] // detached when ore visuals moved to sprites; kept for a rich-vein event
fn spawn_ore_crystals(
    parent: &mut ChildSpawnerCommands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
    asteroid_size: f32,
    colors: &[([f32; 3], f32)],
) {
    if colors.is_empty() {
        return;
    }
    let mut rng = rand::thread_rng();
    let max_crystals = (asteroid_size / 5.0).ceil() as usize;
    let num_crystals = rng.gen_range(0..=max_crystals);
    let total_weight: f32 = colors.iter().map(|(_, w)| w).sum();
    if total_weight <= 0.0 {
        return;
    }

    for _ in 0..num_crystals {
        // Pick a random commodity color weighted by field weights
        let mut roll = rng.gen_range(0.0..total_weight);
        let mut color = colors[0].0;
        for &(c, w) in colors {
            roll -= w;
            if roll <= 0.0 {
                color = c;
                break;
            }
        }

        // Small irregular polygon (3-5 vertices) for crystal shape
        let crystal_size = asteroid_size * rng.gen_range(0.15..0.30);
        let crystal_segments = rng.gen_range(3..=5);
        let mut verts: Vec<Vec2> = Vec::new();
        for i in 0..crystal_segments {
            let angle = (i as f32 / crystal_segments as f32) * std::f32::consts::TAU;
            let r = crystal_size * rng.gen_range(0.6..1.4);
            verts.push(Vec2::new(angle.cos() * r, angle.sin() * r));
        }

        // Position on asteroid surface (within the asteroid radius)
        let place_angle = rng.gen_range(0.0..std::f32::consts::TAU);
        let place_r = asteroid_size * rng.gen_range(0.2..0.75);
        let offset = Vec2::new(place_angle.cos() * place_r, place_angle.sin() * place_r);

        let mesh = polygon_mesh(&verts);
        let [r, g, b] = color;
        parent.spawn((
            Mesh2d(meshes.add(mesh)),
            MeshMaterial2d(materials.add(Color::srgb(r, g, b))),
            Transform::from_xyz(offset.x, offset.y, 0.01),
        ));
    }
}

pub fn spawn_asteroid(
    commands: &mut Commands,
    atlases: &AsteroidAtlases,
    field: Entity,
    size: f32,
    pos: Vec2,
    vel: Vec2,
    colors: &[([f32; 3], f32)],
    fuel: bool,
) -> Entity {
    let mut rng = rand::thread_rng();
    let rot = rng.gen_range(-(0.1 * PI)..(0.1 * PI));
    let px = size * ASTEROID_SPRITE_SCALE;
    let (rock, tumble, shape, tint) = asteroid_visual_bundle(atlases, size, colors, &mut rng);

    let mut entity_cmd = commands.spawn((
        DespawnOnExit(PlayState::Flying),
        Asteroid { size, field, fuel },
        rock,
        tumble.clone(),
        Transform::from_xyz(pos.x, pos.y, 0.0),
        Collider::circle(size),
        CollisionLayers::new(
            GameLayer::Asteroid,
            [
                GameLayer::Ship,
                GameLayer::Weapon,
                GameLayer::Asteroid,
                GameLayer::Planet,
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
    ));
    entity_cmd.with_children(|parent| {
        // tinted deposit overlay, frame-synced via the same tumble clock
        parent.spawn((
            atlas_sprite(&atlases.deps[shape], &atlases.layout, px, tint),
            tumble.clone(),
            Transform::from_xyz(0.0, 0.0, 0.01),
        ));
    });
    if fuel {
        entity_cmd.insert(crate::fuel::FuelShimmer);
    }
    entity_cmd.id()
}

pub fn build_asteroid_field(
    commands: &mut Commands,
    atlases: &AsteroidAtlases,
    field_data: &AsteroidFieldData,
    item_universe: &ItemUniverse,
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
            RigidBody::Static,
            Transform::from_xyz(field_data.location.x, field_data.location.y, 0.0),
        ))
        .id();
    let colors = commodity_colors(&field_data.commodities, item_universe);
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
        let fuel = rolls_fuel(&field_data.commodities, &mut rng);
        spawn_asteroid(
            commands,
            atlases,
            field,
            size,
            Vec2 { x, y } + field_data.location,
            vel,
            &colors,
            fuel,
        );
    }
}

pub fn shatter_asteroid(
    commands: &mut Commands,
    atlases: &AsteroidAtlases,
    asteroid_entity: &Entity,
    asteroids: &Query<(&Asteroid, &Transform, &LinearVelocity)>,
    colors: &[([f32; 3], f32)],
) {
    if let Ok((asteroid, transform, vel)) = asteroids.get(*asteroid_entity) {
        let mut rng = rand::thread_rng();
        let pos = transform.translation.truncate();
        let size = asteroid.size;
        let field = asteroid.field;
        // Remove the asteroid:
        safe_despawn(commands, *asteroid_entity);
        if size > 10.0 {
            for _ in 0..2 {
                let new_size = rng.gen_range((size * 0.3)..(size * 0.8));
                let new_vel = vel.0 + random_velocity(vel.0.length() * size / new_size);
                let offset = new_size * new_vel / new_vel.length();
                spawn_asteroid(
                    commands,
                    atlases,
                    field,
                    new_size,
                    pos + offset,
                    new_vel,
                    colors,
                    asteroid.fuel,
                );
            }
        }
    }
}

fn handle_shatter(
    mut commands: Commands,
    atlases: Res<AsteroidAtlases>,
    mut reader: MessageReader<ShatterAsteroid>,
    mut explosion_writer: MessageWriter<crate::explosions::TriggerExplosion>,
    mut pickup_writer: MessageWriter<PickupDrop>,
    item_universe: Res<ItemUniverse>,
    reward_cfg: Res<crate::config::RewardConfig>,
    asteroids: Query<(&Asteroid, &Transform, &LinearVelocity)>,
    fields: Query<&AsteroidField>,
) {
    use std::collections::HashSet;
    let mut rng = rand::thread_rng();
    let mut shattered: HashSet<Entity> = HashSet::new();
    for ShatterAsteroid {
        entity,
        drop_pickups,
    } in reader.read()
    {
        if !shattered.insert(*entity) {
            continue;
        }
        if let Ok((asteroid, transform, _)) = asteroids.get(*entity) {
            explosion_writer.write(crate::explosions::TriggerExplosion {
                location: transform.translation.xy(),
                size: asteroid.size,
            });
            // Pickups only drop when an asteroid is shot (weapon hit). Ramming
            // shatters the rock but yields nothing, so miners must actually
            // shoot to mine (and thereby earn the asteroid_hit reward).
            if *drop_pickups {
                // Fuel rocks always drop fuel; ordinary rocks roll the field's
                // commodities with `fuel` EXCLUDED, so an iron rock never
                // surprise-drops fuel (fuel comes only from shimmering rocks).
                let commodity = if asteroid.fuel {
                    crate::ship::FUEL_COMMODITY.to_string()
                } else {
                    fields
                        .get(asteroid.field)
                        .ok()
                        .and_then(|f| {
                            let total: f32 = f
                                .commodities
                                .iter()
                                .filter(|(c, _)| c.as_str() != crate::ship::FUEL_COMMODITY)
                                .map(|(_, w)| *w)
                                .sum();
                            if total <= 0.0 {
                                return None;
                            }
                            let mut roll = rng.gen_range(0.0..total);
                            for (c, &w) in &f.commodities {
                                if c.as_str() == crate::ship::FUEL_COMMODITY {
                                    continue;
                                }
                                roll -= w;
                                if roll <= 0.0 {
                                    return Some(c.clone());
                                }
                            }
                            None
                        })
                        .unwrap_or_else(|| "iron".to_string())
                };
                let max_qty = ((asteroid.size * reward_cfg.asteroid_drop_scale) as u16).max(1);
                let qty = rng.gen_range(1..=max_qty);
                pickup_writer.write(PickupDrop {
                    location: transform.translation.xy(),
                    commodity,
                    quantity: qty,
                });
            }
        }
        let colors = asteroids
            .get(*entity)
            .ok()
            .and_then(|(a, _, _)| fields.get(a.field).ok())
            .map(|f| commodity_colors(&f.commodities, &item_universe))
            .unwrap_or_default();
        shatter_asteroid(&mut commands, &atlases, entity, &asteroids, &colors);
    }
}

/// Spawn an asteroid that grows in from nothing over `ASTEROID_GROW_DURATION`.
/// It starts as a `Sensor` (no physics collisions) and becomes solid when fully grown.
pub fn spawn_growing_asteroid(
    commands: &mut Commands,
    atlases: &AsteroidAtlases,
    field: Entity,
    size: f32,
    pos: Vec2,
    vel: Vec2,
    colors: &[([f32; 3], f32)],
    fuel: bool,
) {
    let mut rng = rand::thread_rng();
    let rot = rng.gen_range(-(0.1 * PI)..(0.1 * PI));
    let px = size * ASTEROID_SPRITE_SCALE;
    let (rock, tumble, shape, tint) = asteroid_visual_bundle(atlases, size, colors, &mut rng);

    // Sensor colliders don't contribute to mass, so we provide it explicitly.
    // Values match what Avian2D would compute from ColliderDensity(0.5) + Collider::circle(size):
    //   mass            = density × π × r²
    //   angular_inertia = ½ × mass × r²   (solid disk)
    let mass_value = 0.5 * PI * size * size;
    let inertia_value = 0.5 * mass_value * size * size;

    // Split across two inserts because Bevy's tuple Bundle limit is 15 items.
    let grown = commands
        .spawn((
            DespawnOnExit(PlayState::Flying),
            Asteroid { size, field, fuel },
            GrowingAsteroid { progress: 0.0 },
            rock,
            tumble.clone(),
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
                    GameLayer::Planet,
                    GameLayer::Radar,
                ],
            ),
            LinearVelocity(vel),
            AngularVelocity(rot),
            RigidBody::Dynamic,
        ))
        .insert((
            // Explicit mass so Avian2D doesn't warn about a massless dynamic body.
            Mass(mass_value),
            AngularInertia(inertia_value),
            ColliderDensity(0.5),
            CollisionEventsEnabled,
            Restitution::new(1.0),
            MaxLinearSpeed(ASTEROID_MAX_VELOCITY),
            MaxAngularSpeed(3.0 * PI),
        ))
        .with_children(|parent| {
            parent.spawn((
                atlas_sprite(&atlases.deps[shape], &atlases.layout, px, tint),
                tumble.clone(),
                Transform::from_xyz(0.0, 0.0, 0.01),
            ));
        })
        .id();
    if fuel {
        commands.entity(grown).insert(crate::fuel::FuelShimmer);
    }
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
        growing.progress = (growing.progress + dt / ASTEROID_GROW_DURATION).min(1.0);
        let scale = growing.progress;
        transform.scale = Vec3::splat(scale);

        if growing.progress >= 1.0 {
            // Fully grown: remove Sensor so it participates in physics.
            if let Ok(mut entity_cmd) = commands.get_entity(entity) {
                entity_cmd
                    .try_remove::<GrowingAsteroid>()
                    .try_remove::<Sensor>();
            }
        }
    }
}

/// Periodically check each asteroid field and spawn new growing asteroids
/// if the live count falls below the field's target number.
fn respawn_asteroids(
    mut commands: Commands,
    atlases: Res<AsteroidAtlases>,
    mut timer: ResMut<AsteroidRespawnTimer>,
    time: Res<Time>,
    item_universe: Res<ItemUniverse>,
    fields: Query<(Entity, &AsteroidField, &Transform)>,
    asteroids: Query<&Asteroid, Without<FadingAsteroid>>,
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
        let colors = commodity_colors(&field.commodities, &item_universe);
        let field_pos = field_transform.translation.xy();
        for _ in 0..(target - live_count) {
            let r = rng.gen_range((field.radius * 0.5)..(field.radius * 1.5));
            let theta = rng.gen_range(0.0..(2.0 * std::f32::consts::PI));
            let (s, c) = theta.sin_cos();
            let pos = field_pos + Vec2::new(r * c, r * s);
            let size: f32 = rng.gen_range(15f32..30f32);
            let v = (field.radius / r).sqrt() * ASTEROID_VELOCITY;
            let vel = Vec2::new(-s * v, c * v) + random_velocity(ASTEROID_VELOCITY * 0.3);
            let vel = if rng.gen_bool(0.5) { vel } else { -vel };
            let fuel = rolls_fuel(&field.commodities, &mut rng);
            spawn_growing_asteroid(
                &mut commands,
                &atlases,
                field_entity,
                size,
                pos,
                vel,
                &colors,
                fuel,
            );
        }
    }
}

// Apply gravity towards the center of the asteroid field
pub fn asteroid_field_gravity(
    asteroids: Query<
        (Entity, &Asteroid, &Transform, &ComputedMass),
        (With<RigidBody>, Without<FadingAsteroid>),
    >,
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

fn start_fade(commands: &mut Commands, entity: Entity, size: f32) {
    // Sensor disables collider mass contribution; provide explicit mass to avoid NaN warnings.
    let mass_value = 0.5 * PI * size * size;
    let inertia_value = 0.5 * mass_value * size * size;
    if let Ok(mut entity_cmd) = commands.get_entity(entity) {
        entity_cmd.try_insert((
            FadingAsteroid { progress: 0.0 },
            Sensor,
            Mass(mass_value),
            AngularInertia(inertia_value),
        ));
    }
}

/// Start fading any asteroid that's drifted too close to or too far from its field center.
fn check_asteroid_bounds(
    mut commands: Commands,
    asteroids: Query<
        (Entity, &Asteroid, &Transform),
        (Without<FadingAsteroid>, Without<GrowingAsteroid>),
    >,
    fields: Query<&Transform, With<AsteroidField>>,
    field_data: Query<&AsteroidField>,
) {
    for (entity, asteroid, transform) in asteroids.iter() {
        let Ok(field_transform) = fields.get(asteroid.field) else {
            continue;
        };
        let Ok(field) = field_data.get(asteroid.field) else {
            continue;
        };
        let dist = (transform.translation.xy() - field_transform.translation.xy()).length();
        let inner = field.radius * ASTEROID_FADE_INNER_FRAC;
        let outer = field.radius * ASTEROID_FADE_OUTER_FRAC;
        if dist < inner || dist > outer {
            start_fade(&mut commands, entity, asteroid.size);
        }
    }
}

/// Silently shatter asteroids that collide with planets — splits like a normal
/// shatter but emits no pickups or explosion.
fn asteroid_planet_collisions(
    mut commands: Commands,
    atlases: Res<AsteroidAtlases>,
    mut collision_starts: MessageReader<CollisionStart>,
    mut explosion_writer: MessageWriter<crate::explosions::TriggerExplosion>,
    asteroid_marker: Query<(), (With<Asteroid>, Without<FadingAsteroid>)>,
    asteroid_data: Query<(&Asteroid, &Transform, &LinearVelocity)>,
    fields: Query<&AsteroidField>,
    item_universe: Res<ItemUniverse>,
    planets: Query<(), With<crate::planets::Planet>>,
) {
    for event in collision_starts.read() {
        let (a, b) = (event.collider1, event.collider2);
        let asteroid_entity = if asteroid_marker.contains(a) && planets.contains(b) {
            Some(a)
        } else if asteroid_marker.contains(b) && planets.contains(a) {
            Some(b)
        } else {
            None
        };
        if let Some(entity) = asteroid_entity {
            let colors = asteroid_data
                .get(entity)
                .ok()
                .and_then(|(a, _, _)| fields.get(a.field).ok())
                .map(|f| commodity_colors(&f.commodities, &item_universe))
                .unwrap_or_default();
            if let Ok((asteroid, transform, _)) = asteroid_data.get(entity) {
                explosion_writer.write(crate::explosions::TriggerExplosion {
                    location: transform.translation.xy(),
                    size: asteroid.size,
                });
            }
            shatter_asteroid(&mut commands, &atlases, &entity, &asteroid_data, &colors);
        }
    }
}

/// Advance FadingAsteroid progress; shrink and despawn when done.
fn fade_asteroids(
    mut commands: Commands,
    time: Res<Time>,
    mut fading: Query<(Entity, &mut FadingAsteroid, &mut Transform)>,
) {
    let dt = time.delta_secs();
    for (entity, mut fade, mut transform) in fading.iter_mut() {
        fade.progress = (fade.progress + dt / ASTEROID_FADE_DURATION).min(1.0);
        let scale = (1.0 - fade.progress).max(0.0);
        transform.scale = Vec3::splat(scale);
        if fade.progress >= 1.0 {
            safe_despawn(&mut commands, entity);
        }
    }
}
