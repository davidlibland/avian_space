use crate::item_universe::ItemUniverse;
use crate::utils::polygon_mesh;
use crate::weapons::{WeaponSystem, WeaponSystems};
use crate::{GameLayer, GameState};
use avian2d::{math::*, prelude::*};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub fn ship_plugin(app: &mut App) {
    app.add_message::<ShipCommand>()
        .add_message::<DamageShip>()
        .add_message::<BuyShip>()
        .add_systems(
            FixedUpdate,
            ship_movement.run_if(in_state(GameState::Flying)),
        )
        .add_systems(Update, apply_damage.run_if(in_state(GameState::Flying)))
        .add_systems(Update, handle_buy_ship);
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

#[derive(Event, Message)]
pub struct BuyShip {
    pub ship_type: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum Personality {
    Trader,  // Likes to trade, traveling from planet to planet
    Fighter, // Will attack other ships
    Miner,   // Will mine asteroids
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ShipData {
    pub thrust: Scalar,    // N — forward force
    pub max_speed: Scalar, // m/s — speed cap
    pub torque: Scalar,    // N·m — maximum turning torque
    pub max_health: i32,
    pub cargo_space: u16,
    pub item_space: u16,
    pub base_weapons: HashMap<String, u8>, // A list of the basic weapons this ship starts with, along with counts
    pub sprite_path: String,
    pub radius: f32,
    pub price: i128,
    pub personality: Personality,

    // Some serde defaults (typically not defined in the serialized data):
    pub angular_drag: Scalar, // s⁻¹ — exponential decay rate for angular velocity
    // PD gains for thrust: F = kp*(v_target - v) - kd*dv
    pub thrust_kp: Scalar,
    pub thrust_kd: Scalar,
    // PD gains for reverse heading correction
    pub reverse_kp: Scalar,
    pub reverse_kd: Scalar,
}

impl Default for ShipData {
    fn default() -> Self {
        Self {
            thrust: 200.0,
            max_speed: 300.0,
            torque: 20.0,
            max_health: 100,
            cargo_space: 10,
            item_space: 5,
            base_weapons: HashMap::new(),
            sprite_path: "shuttle.png".to_string(),
            radius: 10.0,
            price: 1000,
            personality: Personality::Trader,
            // Defaults
            angular_drag: 3.0,
            thrust_kp: 5.0,
            thrust_kd: 1.0,
            reverse_kp: 20.0,
            reverse_kd: 1.5,
        }
    }
}

#[derive(Clone)]
pub enum Target {
    Ship(Entity),
    Planet(Entity),
    Asteroid(Entity),
    Pickup(Entity),
}

#[derive(Component, Clone)]
pub struct Ship {
    pub ship_type: String, // The type of the ship
    pub data: ShipData, // The ship data, can be looked up in the item universe, but stored here for convenience
    pub health: i32,
    pub cargo: HashMap<String, u16>, // Map from commodities to quantity
    pub credits: i128,
    pub consumed_item_space: u16, // Space in the ship filled with outfitter items
    // An optional target for the ship.
    // Traders will tend to target planets,
    // Miners will tend to target asteroids,
    // Fighters will tend to target ships
    pub target: Option<Target>,
}

impl Default for Ship {
    fn default() -> Self {
        let data = ShipData::default();
        Self {
            ship_type: "shuttle".to_string(),
            data: data.clone(),
            health: data.max_health,
            cargo: HashMap::new(),
            credits: 100000,
            consumed_item_space: 0,
            target: None,
        }
    }
}

impl Ship {
    pub fn from_ship_data(data: &ShipData, ship_type: &str) -> Self {
        Self {
            ship_type: ship_type.to_string(),
            data: data.clone(),
            health: data.max_health,
            cargo: HashMap::new(),
            credits: 100000,
            consumed_item_space: 0,
            target: None,
        }
    }
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
        let quantity_added = std::cmp::min(quantity_desired, self.remaining_cargo_space());
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

impl ShipBundle {
    pub fn get_personality(&self) -> Personality {
        self.ship.data.personality.clone()
    }
}

pub fn ship_bundle(
    ship_type: &str,
    asset_server: &Res<AssetServer>,
    item_universe: &Res<ItemUniverse>,
    pos: Vec2,
) -> ShipBundle {
    let default_data = ShipData::default();
    let ship_data = item_universe.ships.get(ship_type).unwrap_or(&default_data);
    let mut ship = Ship::from_ship_data(ship_data, ship_type);
    ship.consumed_item_space = ship_data
        .base_weapons
        .iter()
        .filter_map(|(weapon_type, &count)| {
            item_universe
                .outfitter_items
                .get(weapon_type)
                .map(|item| item.space() * count as u16)
        })
        .sum();
    let mut primary_weapons: HashMap<String, WeaponSystem> = HashMap::new();
    for (weapon_type, count) in ship_data.base_weapons.iter() {
        if let Some(weapon_system) =
            WeaponSystem::from_type(&weapon_type, *count, &item_universe.weapons)
        {
            primary_weapons
                .entry(weapon_type.clone())
                .insert_entry(weapon_system);
        }
    }
    ShipBundle {
        ship: ship.clone(),
        // mesh: Mesh2d(meshes.add(build_ship_mesh())),
        // material: MeshMaterial2d(materials.add(Color::srgb(0.2, 0.7, 0.9))),
        sprite: Sprite::from_image(asset_server.load(ship_data.sprite_path.to_string())),
        transform: Transform::from_xyz(pos.x, pos.y, 0.0),
        body: RigidBody::Dynamic,
        angular_damping: AngularDamping(ship.data.angular_drag), // equivalent to angular_drag = 3.0
        max_speed: MaxLinearSpeed(ship.data.max_speed),          // Restitution::new(1.5),
        collider: Collider::circle(ship_data.radius),
        colider_density: ColliderDensity(2.0),
        collision_events: CollisionEventsEnabled,
        layer: CollisionLayers::new(
            GameLayer::Ship,
            [
                GameLayer::Weapon,
                GameLayer::Asteroid,
                GameLayer::Planet,
                GameLayer::Radar,
                GameLayer::Pickup,
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
    mut pickup_writer: MessageWriter<crate::pickups::PickupDrop>,
) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for event in reader.read() {
        let Ok((mut ship, transform)) = ships.get_mut(event.entity) else {
            continue;
        };
        ship.health = (ship.health - event.damage as i32).max(0);
        if ship.health == 0 && ai_ships.contains(event.entity) {
            let location = transform.translation.xy();
            explosion_writer.write(crate::explosions::TriggerExplosion {
                location,
                size: 20.0,
            });
            for (commodity, &qty) in &ship.cargo {
                if qty > 0 {
                    let drop_qty = rng.gen_range(1..=qty);
                    pickup_writer.write(crate::pickups::PickupDrop {
                        location,
                        commodity: commodity.clone(),
                        quantity: drop_qty,
                    });
                }
            }
            crate::utils::safe_despawn(&mut commands, event.entity);
        }
    }
}

fn handle_buy_ship(
    mut commands: Commands,
    mut reader: MessageReader<BuyShip>,
    mut player_query: Query<(Entity, &mut Ship, &mut WeaponSystems), With<crate::Player>>,
    item_universe: Res<ItemUniverse>,
    asset_server: Res<AssetServer>,
) {
    for event in reader.read() {
        let Ok((entity, mut ship, mut weapons)) = player_query.single_mut() else {
            continue;
        };
        let Some(new_data) = item_universe.ships.get(&event.ship_type) else {
            continue;
        };
        if ship.credits < new_data.price {
            continue;
        }
        ship.credits -= new_data.price;

        // Build new weapon systems from the ship's base loadout.
        let mut primary = HashMap::new();
        for (weapon_type, count) in &new_data.base_weapons {
            if let Some(ws) = WeaponSystem::from_type(weapon_type, *count, &item_universe.weapons) {
                primary.insert(weapon_type.clone(), ws);
            }
        }
        *weapons = WeaponSystems { primary };

        // Replace the ship component, preserving credits and cargo (capped to new space).
        let mut new_ship = Ship::from_ship_data(new_data, &event.ship_type);
        new_ship.credits = ship.credits;
        new_ship.consumed_item_space = 0;
        for (commodity, qty) in &ship.cargo {
            let space_left = new_ship
                .data
                .cargo_space
                .saturating_sub(new_ship.cargo.values().sum::<u16>());
            let transfer = (*qty).min(space_left);
            if transfer > 0 {
                *new_ship.cargo.entry(commodity.clone()).or_insert(0) += transfer;
            }
        }
        *ship = new_ship;

        // Replace physics/render components.
        commands.entity(entity).insert((
            Sprite::from_image(asset_server.load(new_data.sprite_path.clone())),
            Collider::circle(new_data.radius),
            MaxLinearSpeed(new_data.max_speed),
            AngularDamping(new_data.angular_drag),
        ));
    }
}
