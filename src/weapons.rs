use crate::item_universe::ItemUniverse;
use crate::ship::Ship;
use crate::utils::safe_despawn;
use crate::{GameLayer, PlayState};
use avian2d::prelude::*;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, hash_map::Entry};

pub fn weapons_plugin(app: &mut App) {
    app.add_message::<FireCommand>().add_systems(
        Update,
        (
            weapon_fire,
            weapon_lifetime,
            weapon_system_cooldown,
            missile_guidance,
            cleanup_tracer_slots,
        ),
    );
}

#[derive(Event, Message)]
pub struct FireCommand {
    pub ship: Entity,
    pub weapon_type: String,
    pub target: Option<Entity>,
}

/// A weapon fired by any ship.
///
/// `owner` is `None` for player weapons, or `Some(owner_entity)` for
/// enemy lasers so that hits can be credited to the correct trajectory.
#[derive(Component)]
pub struct Projectile {
    pub lifetime: f32,
    pub owner: Entity,
    pub weapon_type: String,
}

/// Marks a projectile as a tracer — tracked by the RL observation system
/// across its full lifetime.
#[derive(Component)]
pub struct Tracer;

/// Per-ship resource tracking which tracer slots are in use.
/// Each slot is `None` (free) or `Some(projectile_entity)`.
#[derive(Component)]
pub struct TracerSlots {
    pub slots: Vec<Option<Entity>>,
    /// Round-robin counter across weapon types for even distribution.
    next_weapon_idx: usize,
}

impl TracerSlots {
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: vec![None; capacity],
            next_weapon_idx: 0,
        }
    }

    /// Find the next free slot. Returns `None` if all slots are occupied.
    pub fn next_free_slot(&self) -> Option<u8> {
        self.slots.iter().position(|s| s.is_none()).map(|i| i as u8)
    }

    /// Assign a projectile to a slot.
    pub fn assign(&mut self, slot: u8, entity: Entity) {
        self.slots[slot as usize] = Some(entity);
    }

    /// Check if a weapon type should get the next tracer, cycling through
    /// weapon types uniformly. `weapon_names` is the ordered list of weapon
    /// types this ship has.
    pub fn should_trace(&mut self, weapon_type: &str, weapon_names: &[&str]) -> bool {
        if weapon_names.is_empty() || self.next_free_slot().is_none() {
            return false;
        }
        let target_weapon = weapon_names[self.next_weapon_idx % weapon_names.len()];
        if weapon_type == target_weapon {
            self.next_weapon_idx += 1;
            true
        } else {
            false
        }
    }
}

#[derive(Clone)]
pub struct WeaponSystems {
    pub primary: HashMap<String, WeaponSystem>,
    pub secondary: HashMap<String, WeaponSystem>,
    pub selected_secondary: Option<String>,
}
impl WeaponSystems {
    pub fn find_weapon_entry(&mut self, weapon_type: &str) -> Entry<'_, String, WeaponSystem> {
        let primary_entry = self.primary.entry(weapon_type.to_string());
        match primary_entry {
            Entry::Occupied(_) => primary_entry,
            Entry::Vacant(_) => self.secondary.entry(weapon_type.to_string()),
        }
    }
    pub fn find_weapon(&mut self, weapon_type: &str) -> Option<&mut WeaponSystem> {
        match self.find_weapon_entry(weapon_type) {
            Entry::Vacant(_) => None,
            Entry::Occupied(view) => Some(view.into_mut()),
        }
    }
    pub fn build(
        layout: &HashMap<String, (u8, Option<u32>)>,
        item_universe: &ItemUniverse,
    ) -> Self {
        let mut primary: HashMap<String, WeaponSystem> = HashMap::new();
        let mut secondary: HashMap<String, WeaponSystem> = HashMap::new();
        for (weapon_type, &(count, ammo_quantity)) in layout.iter() {
            if let Some(ws) =
                WeaponSystem::from_type(weapon_type, count, ammo_quantity, &item_universe)
            {
                if ws.ammo_quantity.is_some() {
                    secondary.insert(weapon_type.clone(), ws);
                } else {
                    primary.insert(weapon_type.clone(), ws);
                }
            }
        }
        WeaponSystems {
            primary,
            secondary: secondary,
            selected_secondary: None,
        }
    }
    pub fn increment_secondary(&mut self) {
        let mut sorted_names: Vec<String> = self.secondary.keys().map(String::from).collect();
        sorted_names.sort();
        if let Some(current) = &self.selected_secondary {
            // Get the next secondary weapon:
            self.selected_secondary = sorted_names
                .iter()
                .position(|s| s == current)
                .map(|x| (x + 1) % sorted_names.len())
                .and_then(|i| sorted_names.get(i))
                .map(String::from);
        } else {
            // Just get the first secondary weapon
            self.selected_secondary = sorted_names.first().map(String::from);
        }
    }
    pub fn iter_all(&self) -> impl Iterator<Item = (&String, &WeaponSystem)> {
        self.primary.iter().chain(self.secondary.iter())
    }
    pub fn iter_all_mut(&mut self) -> impl Iterator<Item = (&String, &mut WeaponSystem)> {
        self.primary.iter_mut().chain(self.secondary.iter_mut())
    }
}

impl Default for WeaponSystems {
    fn default() -> Self {
        WeaponSystems {
            primary: HashMap::new(),
            secondary: HashMap::new(),
            selected_secondary: None,
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Ammo {
    pub space: u32,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Weapon {
    pub name: String,
    pub lifetime: f32,
    pub speed: f32,
    pub cooldown: f32,
    pub color: [f32; 3],
    pub damage: i16,
    pub ammo: Option<Ammo>,
    #[serde(default)]
    pub guided: bool,
    #[serde(default)]
    pub turn_rate: f32,
    /// Optional path to a sprite image. When set, the projectile uses this
    /// image instead of a plain colored rectangle.
    #[serde(default)]
    pub sprite_path: Option<String>,
}

#[derive(Component)]
pub struct GuidedMissile {
    pub target: Option<Entity>,
    pub turn_rate: f32,
}

impl Weapon {
    pub fn range(&self) -> f32 {
        self.speed * self.lifetime
    }
}
#[derive(Clone)]
pub struct WeaponSystem {
    pub weapon: Weapon,
    pub cooldown: Timer,
    pub space_per_system: i32,
    pub number: u8,
    pub ammo_quantity: Option<u32>,
}

impl WeaponSystem {
    pub fn from_type(
        weapon_type: &str,
        number: u8,
        ammo_quantity: Option<u32>,
        item_universe: &ItemUniverse,
    ) -> Option<Self> {
        let Some(weapon) = item_universe.weapons.get(weapon_type) else {
            return None;
        };
        // Lookup the space used:
        let Some(outfitter_item) = item_universe.outfitter_items.get(weapon_type) else {
            return None;
        };
        if number == 0 {
            return None;
        }
        return Some(WeaponSystem {
            weapon: weapon.clone(),
            cooldown: Timer::from_seconds(weapon.cooldown / number as f32, TimerMode::Once),
            number: number,
            space_per_system: outfitter_item.space() as i32,
            ammo_quantity: weapon.ammo.clone().map(|_| ammo_quantity.unwrap_or(0)),
        });
    }
    pub fn space_consumed(&self) -> i32 {
        let base_space = self.space_per_system * self.number as i32;
        let ammo_space = self
            .ammo_quantity
            .and_then(|q| self.weapon.ammo.clone().map(|a| q * a.space));
        return base_space + ammo_space.unwrap_or(0) as i32;
    }
}

pub fn weapon_fire(
    mut reader: MessageReader<FireCommand>,
    mut commands: Commands,
    mut ships: Query<(
        &Transform,
        &mut Ship,
        &LinearVelocity,
        Option<&mut TracerSlots>,
    )>,
    item_universe: Res<ItemUniverse>,
    asset_server: Res<AssetServer>,
) {
    for cmd in reader.read() {
        let Ok((ship_transform, mut ship, linear_velocity, tracer_slots)) = ships.get_mut(cmd.ship)
        else {
            continue;
        };
        let Some(specific) = ship.weapon_systems.find_weapon(&cmd.weapon_type) else {
            continue;
        };
        // If we have no ammo, then don't fire.
        if specific.ammo_quantity.map(|n| n == 0).unwrap_or(false) {
            continue;
        }
        // Check that the weapon is ready to fire:
        if !specific.cooldown.is_finished() {
            continue;
        }
        specific.cooldown.reset();
        let Some(weapon) = item_universe.weapons.get(&cmd.weapon_type) else {
            continue;
        };
        let forward = ship_transform.rotation * Vec3::Y;
        let tip = ship_transform.translation + forward * 20.0;
        let vel = forward.truncate() * weapon.speed + linear_velocity.0;
        let [r, g, b] = weapon.color;
        let sprite = if let Some(path) = &weapon.sprite_path {
            Sprite::from_image(asset_server.load(path.clone()))
        } else {
            Sprite {
                color: Color::srgb(r, g, b),
                custom_size: Some(Vec2::new(3.0, 12.0)),
                ..default()
            }
        };
        let mut entity_cmd = commands.spawn((
            DespawnOnExit(PlayState::Flying),
            Projectile {
                lifetime: weapon.lifetime,
                owner: cmd.ship,
                weapon_type: cmd.weapon_type.clone(),
            },
            Collider::circle(10.),
            CollisionLayers::new(GameLayer::Weapon, [GameLayer::Asteroid, GameLayer::Ship]),
            RigidBody::Dynamic,
            // Swept CCD prevents fast projectiles from tunneling through ships
            // at the 50ms headless training timestep (laser travels 25 units/step
            // vs 9-unit corvette radius).
            SweptCcd::LINEAR,
            LinearVelocity(vel),
            Transform::from_translation(tip.xy().extend(-0.3))
                .with_rotation(ship_transform.rotation),
            sprite,
        ));
        // Decrease the ammo:
        specific.ammo_quantity = specific.ammo_quantity.map(|x| x - 1);
        if weapon.guided {
            entity_cmd.insert(GuidedMissile {
                target: cmd.target,
                turn_rate: weapon.turn_rate,
            });
        }

        // Assign tracer if this ship has TracerSlots and a slot is free.
        if let Some(mut ts) = tracer_slots {
            let weapon_names: Vec<&str> = ship
                .weapon_systems
                .iter_all()
                .map(|(name, _)| name.as_str())
                .collect();
            if ts.should_trace(&cmd.weapon_type, &weapon_names) {
                if let Some(slot) = ts.next_free_slot() {
                    let proj_entity = entity_cmd.id();
                    entity_cmd.insert(Tracer);
                    ts.assign(slot, proj_entity);
                }
            }
        }
    }
}

pub(crate) fn weapon_lifetime(
    mut commands: Commands,
    time: Res<Time>,
    mut query: Query<(Entity, &mut Projectile)>,
) {
    let dt = time.delta_secs();
    for (entity, mut weapon) in &mut query {
        weapon.lifetime -= dt;
        if weapon.lifetime <= 0.0 {
            safe_despawn(&mut commands, entity);
        }
    }
}

/// Free tracer slots whose projectile entities no longer exist.
fn cleanup_tracer_slots(
    mut ships: Query<&mut TracerSlots>,
    projectiles: Query<(), With<Projectile>>,
) {
    for mut ts in &mut ships {
        for slot in &mut ts.slots {
            if let Some(e) = *slot {
                if projectiles.get(e).is_err() {
                    *slot = None;
                }
            }
        }
    }
}

pub fn weapon_system_cooldown(time: Res<Time>, mut query: Query<&mut Ship>) {
    for mut ship in &mut query {
        for (_, specific) in ship.weapon_systems.iter_all_mut() {
            specific
                .cooldown
                .tick(time.delta() * (specific.number as u32));
        }
    }
}

fn missile_guidance(
    time: Res<Time>,
    mut missiles: Query<
        (&mut Transform, &mut LinearVelocity, &mut GuidedMissile),
        With<Projectile>,
    >,
    all_positions: Query<&Position>,
) {
    let dt = time.delta_secs();
    for (mut transform, mut vel, mut missile) in &mut missiles {
        // If we have no target, just fly straight.
        let Some(target) = missile.target else {
            continue;
        };
        let Ok(target_pos) = all_positions.get(target) else {
            missile.target = None;
            continue;
        };

        let speed = vel.0.length();
        if speed < f32::EPSILON {
            continue;
        }
        let current_dir = vel.0 / speed;
        let to_target = (target_pos.0 - transform.translation.xy()).normalize_or_zero();

        let angle = current_dir.angle_to(to_target);
        let max_turn = missile.turn_rate * dt;
        let turn = angle.clamp(-max_turn, max_turn);
        let (sin_t, cos_t) = turn.sin_cos();
        let new_dir = Vec2::new(
            current_dir.x * cos_t - current_dir.y * sin_t,
            current_dir.x * sin_t + current_dir.y * cos_t,
        );
        vel.0 = new_dir * speed;

        // Rotate sprite to match travel direction (+Y is forward).
        let visual_angle = new_dir.x.atan2(new_dir.y);
        transform.rotation = Quat::from_rotation_z(-visual_angle);
    }
}
