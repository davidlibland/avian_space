use avian2d::{math::*, prelude::*};
use bevy::prelude::*;
use bevy::{
    asset::RenderAssetUsages, mesh::Indices, prelude::*, render::render_resource::PrimitiveTopology,
};
use rand::Rng;

mod starfield;
use starfield::{Star, StarfieldPlugin, WorldOffset};

// Define your boundary
const BOUNDS: f32 = 900.0;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            // Add physics plugins and specify a units-per-meter scaling factor, 1 meter = 20 pixels.
            // The unit allows the engine to tune its parameters for the scale of the world, improving stability.
            PhysicsPlugins::default().with_length_unit(20.0),
            StarfieldPlugin,
        ))
        .insert_resource(Gravity(Vec2::NEG_Y * 0.0)) // Set custom gravity
        .init_resource::<WorldOffset>()
        .add_message::<ShipCommand>()
        .add_systems(Startup, setup)
        .add_systems(
            FixedUpdate,
            (keyboard_input, ship_movement)
                .chain()
                .before(PhysicsSystems::StepSimulation),
        )
        .add_systems(Update, collision_system)
        .add_systems(
            Update,
            (screen_wrapping_system, recenter_world)
                .chain()
                .after(PhysicsSystems::Writeback),
        )
        .run();
}

#[derive(Component)]
struct Player;

#[derive(Component)]
struct Asteroid {
    size: f32,
}

#[derive(Component, Clone)]
pub struct Ship {
    pub thrust: Scalar,       // N — forward force
    pub max_speed: Scalar,    // m/s — speed cap
    pub torque: Scalar,       // N·m — maximum turning torque
    pub angular_drag: Scalar, // s⁻¹ — exponential decay rate for angular velocity
    // PD gains for thrust: F = kp*(v_target - v) - kd*dv
    pub thrust_kp: Scalar,
    pub thrust_kd: Scalar,
    // PD gains for reverse heading correction
    pub reverse_kp: Scalar,
    pub reverse_kd: Scalar,
}

impl Default for Ship {
    fn default() -> Self {
        Self {
            thrust: 200.0,
            max_speed: 300.0,
            torque: 20.0,
            angular_drag: 3.0,
            thrust_kp: 5.0,
            thrust_kd: 1.0,
            reverse_kp: 20.0,
            reverse_kd: 1.5,
        }
    }
}

pub fn build_ship_mesh() -> Mesh {
    // Nose up, two wings back
    let verts: Vec<Vec2> = vec![
        Vec2::new(0.0, 18.0),    // tip
        Vec2::new(-12.0, -12.0), // left wing
        Vec2::new(0.0, -6.0),    // tail indent
        Vec2::new(12.0, -12.0),  // right wing
    ];
    polygon_mesh(&verts)
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let ship = Ship::default();
    // Player
    commands.spawn((
        Player, // Mark the player
        ship.clone(),
        Mesh2d(meshes.add(build_ship_mesh())),
        MeshMaterial2d(materials.add(Color::srgb(0.2, 0.7, 0.9))),
        Transform::from_xyz(0.0, 0.0, 0.0),
        Collider::circle(15.),
        RigidBody::Dynamic,
        ColliderDensity(2.0),
        CollisionEventsEnabled,
        AngularDamping(ship.angular_drag), // equivalent to angular_drag = 3.0
        MaxLinearSpeed(ship.max_speed),    // Restitution::new(1.5),
    ));

    let mut rng = rand::thread_rng();
    for _ in 0..20 {
        let size: f32 = rng.gen_range(15f32..30f32);
        let x = rng.gen_range(-BOUNDS..BOUNDS);
        let y = rng.gen_range(-BOUNDS..BOUNDS);
        spawn_asteroid(
            &mut commands,
            &mut meshes,
            &mut materials,
            size,
            Vec2 { x, y },
            random_velocity(10.0),
        );
    }

    // Camera
    commands.spawn(Camera2d);
}

pub fn polygon_mesh(verts: &[Vec2]) -> Mesh {
    // use bevy::render::mesh::{Indices, PrimitiveTopology};
    // use bevy::render::render_asset::RenderAssetUsages;

    let positions: Vec<[f32; 3]> = verts.iter().map(|v| [v.x, v.y, 0.0]).collect();
    let n = verts.len();
    // Fan triangulation from vertex 0
    let mut indices: Vec<u32> = Vec::new();
    for i in 1..(n as u32 - 1) {
        indices.push(0);
        indices.push(i);
        indices.push(i + 1);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn spawn_asteroid(
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
            LinearVelocity(vel),
            AngularVelocity(rot),
            RigidBody::Dynamic,
            ColliderDensity(0.5),
            CollisionEventsEnabled,
            Restitution::new(1.0),
        ))
        .id()
}

/// Sends [`MovementAction`] events based on keyboard input.
fn keyboard_input(
    mut writer: MessageWriter<ShipCommand>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    player_query: Query<Entity, With<Player>>,
) {
    let Ok(player_entity) = player_query.single() else {
        return; // Player not spawned yet
    };

    let left = keyboard_input.any_pressed([KeyCode::KeyA, KeyCode::ArrowLeft]);
    let right = keyboard_input.any_pressed([KeyCode::KeyD, KeyCode::ArrowRight]);
    let up = keyboard_input.any_pressed([KeyCode::KeyW, KeyCode::ArrowUp]);
    let down = keyboard_input.any_pressed([KeyCode::KeyS, KeyCode::ArrowDown]);

    let horizontal = right as i8 - left as i8;
    let turn = horizontal as Scalar;

    let vertical = up as i8;
    let thrust = vertical as Scalar;
    let reverse = (down as i8) as Scalar;

    // Only send if there's actual input
    if thrust.abs() > f32::EPSILON || turn.abs() > f32::EPSILON || reverse.abs() > f32::EPSILON {
        writer.write(ShipCommand {
            entity: player_entity,
            thrust,
            turn,
            reverse,
        });
    }
}

#[derive(Event, Message)]
struct ShipCommand {
    entity: Entity,
    thrust: Scalar,
    turn: Scalar,
    reverse: Scalar,
}

// Physics system reads ShipCommand messages - runs in FixedUpdate before physics
fn ship_movement(
    mut reader: MessageReader<ShipCommand>,
    time: Res<Time>,
    mut query: Query<
        (&mut LinearVelocity, &mut AngularVelocity, &Transform, &Ship),
        With<RigidBody>,
    >,
) {
    for cmd in reader.read() {
        let Ok((mut velocity, mut ang_vel, transform, ship)) = query.get_mut(cmd.entity) else {
            continue;
        };
        let dt = time.delta_secs();

        let forward = (transform.rotation * Vec3::Y).xy();
        // let velocity = forces.linear_velocity();
        // let ang_vel = forces.angular_velocity();
        let speed = velocity.length();

        if cmd.thrust.abs() > f32::EPSILON {
            let forward_speed = velocity.dot(forward);
            let speed_deficit = ship.max_speed - forward_speed;
            let pd_force = (ship.thrust_kp * speed_deficit - ship.thrust_kd * forward_speed)
                .clamp(0.0, ship.thrust);
            (*velocity).0 += forward * pd_force * cmd.thrust * dt;
            // forces.apply_linear_impulse(forward * pd_force * cmd.thrust * dt);
        }

        // if speed > ship.max_speed {
        //     let excess = speed - ship.max_speed;
        //     (*velocity).0 += -velocity.0.normalize() * excess * 50.0 * dt
        //     // forces.apply_linear_impulse(-velocity.normalize() * excess * 50.0 * dt);
        // }

        if cmd.turn.abs() > f32::EPSILON {
            (*ang_vel).0 += -ship.torque * cmd.turn * dt;
            // forces.apply_angular_impulse(-ship.torque * cmd.turn * dt);
        }

        let new_ang_vel = ang_vel.0 * (-ship.angular_drag * dt).exp();
        (*ang_vel).0 = new_ang_vel;
        // *forces.angular_velocity_mut() = new_ang_vel;

        if cmd.reverse.abs() > f32::EPSILON && speed > f32::EPSILON {
            let retrograde = -velocity.normalize();
            let angle_err = forward.angle_to(retrograde);
            let pd_torque = (ship.reverse_kp * angle_err - ship.reverse_kd * new_ang_vel)
                .clamp(-ship.torque, ship.torque);
            (*ang_vel).0 += pd_torque * cmd.reverse * dt;
            // forces.apply_angular_impulse(pd_torque * cmd.reverse * dt);
        }
    }
}

fn recenter_world(
    mut player_query: Query<(&mut Position, &mut Transform), With<Player>>,
    mut physics_query: Query<(&mut Position, &mut Transform), (With<RigidBody>, Without<Player>)>,
    mut visual_query: Query<
        &mut Transform,
        (
            Without<RigidBody>,
            Without<Player>,
            Without<Camera2d>,
            Without<Star>,
        ),
    >,
    mut world_offset: ResMut<WorldOffset>,
) {
    let Ok((mut player_pos, mut player_transform)) = player_query.single_mut() else {
        return;
    };

    let offset = player_pos.0;
    world_offset.0 = offset; // ← write offset for starfield

    if offset == Vec2::ZERO {
        return;
    }

    player_pos.0 = Vec2::ZERO;
    player_transform.translation = player_transform.translation.with_xy(Vec2::ZERO);

    for (mut pos, mut transform) in physics_query.iter_mut() {
        pos.0 -= offset;
        transform.translation -= offset.extend(0.0);
    }
    for mut transform in visual_query.iter_mut() {
        transform.translation -= offset.extend(0.0);
    }
}

fn screen_wrapping_system(mut query: Query<(&mut Transform, &mut Position), With<RigidBody>>) {
    for (mut transform, mut position) in query.iter_mut() {
        let mut wrapped = false;
        let mut new_translation = transform.translation;

        // Wrap X
        if transform.translation.x > BOUNDS {
            new_translation.x = -BOUNDS;
            wrapped = true;
        } else if transform.translation.x < -BOUNDS {
            new_translation.x = BOUNDS;
            wrapped = true;
        }

        // Wrap Y
        if transform.translation.y > BOUNDS {
            new_translation.y = -BOUNDS;
            wrapped = true;
        } else if transform.translation.y < -BOUNDS {
            new_translation.y = BOUNDS;
            wrapped = true;
        }

        if wrapped {
            // Update Bevy Transform
            transform.translation = new_translation;
            // Update Avian Position component
            position.0 = new_translation.xy();
        }
    }
}

fn collision_system(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut collisions: MessageReader<CollisionStart>,
    asteroids: Query<(&Asteroid, &Transform, &LinearVelocity)>,
    ships: Query<&Player>,
) {
    let mut rng = rand::thread_rng();
    for event in collisions.read() {
        let (a, b) = (event.collider1, event.collider2);

        // Determine which entity is the asteroid and which is the ship,
        // handling both orderings since Avian2D can emit either way.
        let asteroid_entity = if asteroids.contains(a) && ships.contains(b) {
            Some(a)
        } else if asteroids.contains(b) && ships.contains(a) {
            Some(b)
        } else {
            None
        };

        if let Some(asteroid_entity) = asteroid_entity {
            if let Ok((asteroid, transform, vel)) = asteroids.get(asteroid_entity) {
                let pos = transform.translation.truncate();
                let size = asteroid.size;
                commands.entity(asteroid_entity).despawn();
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
    }
}

pub fn random_velocity(speed: f32) -> Vec2 {
    let mut rng = rand::thread_rng();
    let angle = rng.gen_range(0.0_f32..(2.0 * std::f32::consts::PI));
    Vec2::new(angle.cos(), angle.sin()) * speed
}
