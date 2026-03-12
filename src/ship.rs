use crate::utils::polygon_mesh;
use crate::weapons::WeaponSystems;
use crate::{GameState, Layer};
use avian2d::{math::*, prelude::*};
use bevy::prelude::*;

pub fn ship_plugin(app: &mut App) {
    app.add_message::<ShipCommand>().add_systems(
        FixedUpdate,
        ship_movement.run_if(in_state(GameState::Flying)),
    );
    // app.init_resource::<MyCustomResource>();
    // app.add_systems(Update, (do_some_things, do_other_things));
}

#[derive(Event, Message)]
pub struct ShipCommand {
    pub entity: Entity,
    pub thrust: Scalar,
    pub turn: Scalar,
    pub reverse: Scalar,
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
    pub max_health: i32,
    pub health: i32,
}

#[derive(Bundle)]
pub struct ShipBundle {
    ship: Ship,
    sprite: Sprite,
    transform: Transform,
    body: RigidBody,
    angular_damping: AngularDamping,
    max_speed: MaxLinearSpeed,
    collider: Collider,
    colider_density: ColliderDensity,
    collision_events: CollisionEventsEnabled,
    layer: CollisionLayers,
    weapons: WeaponSystems,
}

pub fn ship_bundle(asset_server: &Res<AssetServer>) -> ShipBundle {
    let ship = Ship::default();
    ShipBundle {
        ship: ship.clone(),
        // Mesh2d(meshes.add(build_ship_mesh())),
        // MeshMaterial2d(materials.add(Color::srgb(0.2, 0.7, 0.9))),
        sprite: Sprite::from_image(asset_server.load("spaceship.png")),
        transform: Transform::from_xyz(0.0, 0.0, 0.0),
        body: RigidBody::Dynamic,
        angular_damping: AngularDamping(ship.angular_drag), // equivalent to angular_drag = 3.0
        max_speed: MaxLinearSpeed(ship.max_speed),          // Restitution::new(1.5),
        collider: Collider::circle(15.),
        colider_density: ColliderDensity(2.0),
        collision_events: CollisionEventsEnabled,
        layer: CollisionLayers::new(Layer::Ship, [Layer::Weapon, Layer::Asteroid, Layer::Planet]),
        weapons: WeaponSystems::default(),
    }
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
            max_health: 100,
            health: 100,
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
