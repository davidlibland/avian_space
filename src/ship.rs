use crate::item_universe::ItemUniverse;
use crate::utils::polygon_mesh;
use crate::weapons::{WeaponSystem, WeaponSystems};
use crate::{GameLayer, GameState};
use avian2d::{math::*, prelude::*};
use bevy::prelude::*;
use std::collections::HashMap;

pub fn ship_plugin(app: &mut App) {
    app.add_message::<ShipCommand>()
        .add_message::<DamageShip>()
        .add_systems(
            FixedUpdate,
            ship_movement.run_if(in_state(GameState::Flying)),
        )
        .add_systems(Update, apply_damage.run_if(in_state(GameState::Flying)));
}

#[derive(Event, Message)]
pub struct ShipCommand {
    pub entity: Entity,
    pub thrust: Scalar,
    pub turn: Scalar,
    pub reverse: Scalar,
}

#[derive(Event, Message)]
pub struct DamageShip {
    pub entity: Entity,
    pub damage: Scalar,
}

#[derive(Clone)]
pub struct ShipData {
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
    pub cargo_space: u16,
    pub item_space: u16,
}

impl Default for ShipData {
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
            cargo_space: 10,
            item_space: 5,
        }
    }
}

#[derive(Component, Clone)]
pub struct Ship {
    pub data: ShipData,
    pub health: i32,
    pub cargo: HashMap<String, u16>, // Map from commodities to quantity
    pub credits: i128,
    pub consumed_item_space: u16, // Space in the ship filled with outfitter items
}

impl Default for Ship {
    fn default() -> Self {
        let data = ShipData::default();
        Self {
            data: data.clone(),
            health: data.max_health,
            cargo: HashMap::new(),
            credits: 10000,
            consumed_item_space: 0,
        }
    }
}

impl Ship {
    pub fn remaining_item_space(&self) -> u16 {
        return self
            .data
            .item_space
            .saturating_sub(self.consumed_item_space);
    }
    fn current_cargo(&self) -> u16 {
        self.cargo.values().sum()
    }
    pub fn remaining_cargo_space(&self) -> u16 {
        return self.data.cargo_space.saturating_sub(self.current_cargo());
    }
    fn add_cargo(&mut self, commodity: &str, quantity_desired: u16) -> u16 {
        let quantity_added = std::cmp::max(quantity_desired, self.remaining_cargo_space());
        *self.cargo.entry(commodity.to_string()).or_insert(0) += quantity_added;
        return quantity_added;
    }
    pub fn sell_cargo(&mut self, commodity: &str, quantity: u16, price: i128) {
        let quantity = std::cmp::min(*self.cargo.get(commodity).unwrap_or(&0u16), quantity);
        *self.cargo.entry(commodity.to_string()).or_insert(0) -= quantity;
        self.credits += (quantity as i128) * price;
        if let std::collections::hash_map::Entry::Occupied(entry) =
            self.cargo.entry(commodity.to_string())
        {
            if *entry.get() <= 0 {
                entry.remove_entry(); // Removes the key-value pair
            }
        }
    }
    pub fn buy_cargo(&mut self, commodity: &str, quantity_desired: u16, price: i128) {
        let quantity_desired = std::cmp::min(quantity_desired, (self.credits / price) as u16);
        let quantity_added = self.add_cargo(commodity, quantity_desired);
        self.credits -= (quantity_added as i128) * price;
    }
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

pub fn ship_bundle(
    asset_server: &Res<AssetServer>,
    item_universe: &Res<ItemUniverse>,
    pos: Vec2,
) -> ShipBundle {
    let ship = Ship::default();
    let primary_weapons: HashMap<String, WeaponSystem> =
        match WeaponSystem::from_type("laser", 1, &item_universe.weapons) {
            Some(weapon_system) => HashMap::from([("laser".to_string(), weapon_system)]),
            _ => HashMap::new(),
        };
    ShipBundle {
        ship: ship.clone(),
        // mesh: Mesh2d(meshes.add(build_ship_mesh())),
        // material: MeshMaterial2d(materials.add(Color::srgb(0.2, 0.7, 0.9))),
        sprite: Sprite::from_image(asset_server.load("spaceship.png")),
        transform: Transform::from_xyz(pos.x, pos.y, 0.0),
        body: RigidBody::Dynamic,
        angular_damping: AngularDamping(ship.data.angular_drag), // equivalent to angular_drag = 3.0
        max_speed: MaxLinearSpeed(ship.data.max_speed),          // Restitution::new(1.5),
        collider: Collider::circle(15.),
        colider_density: ColliderDensity(2.0),
        collision_events: CollisionEventsEnabled,
        layer: CollisionLayers::new(
            GameLayer::Ship,
            [
                GameLayer::Weapon,
                GameLayer::Asteroid,
                GameLayer::Planet,
                GameLayer::Radar,
            ],
        ),
        weapons: WeaponSystems {
            primary: primary_weapons,
        },
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
            let speed_deficit = ship.data.max_speed - forward_speed;
            let pd_force = (ship.data.thrust_kp * speed_deficit
                - ship.data.thrust_kd * forward_speed)
                .clamp(0.0, ship.data.thrust);
            (*velocity).0 += forward * pd_force * cmd.thrust * dt;
            // forces.apply_linear_impulse(forward * pd_force * cmd.thrust * dt);
        }

        // if speed > ship.max_speed {
        //     let excess = speed - ship.max_speed;
        //     (*velocity).0 += -velocity.0.normalize() * excess * 50.0 * dt
        //     // forces.apply_linear_impulse(-velocity.normalize() * excess * 50.0 * dt);
        // }

        if cmd.turn.abs() > f32::EPSILON {
            (*ang_vel).0 += -ship.data.torque * cmd.turn * dt;
            // forces.apply_angular_impulse(-ship.torque * cmd.turn * dt);
        }

        let new_ang_vel = ang_vel.0 * (-ship.data.angular_drag * dt).exp();
        (*ang_vel).0 = new_ang_vel;
        // *forces.angular_velocity_mut() = new_ang_vel;

        if cmd.reverse.abs() > f32::EPSILON && speed > f32::EPSILON {
            let retrograde = -velocity.normalize();
            let angle_err = forward.angle_to(retrograde);
            let pd_torque = (ship.data.reverse_kp * angle_err - ship.data.reverse_kd * new_ang_vel)
                .clamp(-ship.data.torque, ship.data.torque);
            (*ang_vel).0 += pd_torque * cmd.reverse * dt;
            // forces.apply_angular_impulse(pd_torque * cmd.reverse * dt);
        }
    }
}

fn apply_damage(
    mut commands: Commands,
    mut reader: MessageReader<DamageShip>,
    mut ships: Query<(&mut Ship, &Transform)>,
    ai_ships: Query<(), With<crate::ai_ships::AIShip>>,
    mut explosion_writer: MessageWriter<crate::explosions::TriggerExplosion>,
) {
    for event in reader.read() {
        let Ok((mut ship, transform)) = ships.get_mut(event.entity) else {
            continue;
        };
        ship.health = (ship.health - event.damage as i32).max(0);
        if ship.health == 0 && ai_ships.contains(event.entity) {
            explosion_writer.write(crate::explosions::TriggerExplosion {
                location: transform.translation.xy(),
                size: 20.0,
            });
            crate::utils::safe_despawn(&mut commands, event.entity);
        }
    }
}
