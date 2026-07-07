use crate::carrier::SpawnEscort;
use crate::item_universe::ItemUniverse;
use crate::ship::Ship;
use crate::utils::safe_despawn;
use crate::{GameLayer, PlayState};
use avian2d::prelude::*;
use bevy::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{HashMap, hash_map::Entry};

pub fn weapons_plugin(app: &mut App) {
    app.add_message::<FireCommand>()
        .add_message::<WeaponFired>()
        .add_message::<DecoyDeployed>()
        .add_systems(
            Update,
            (
                auto_deploy_decoys.before(weapon_fire),
                weapon_fire,
                decoy_missiles.after(weapon_fire),
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

#[derive(Event, Message)]
pub struct WeaponFired {
    pub ship: Entity,
    pub weapon_type: String,
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

/// What a weapon *does* when fired. One enum instead of independent
/// `guided`/`turn_rate`/`carrier_bay` flags, so mutually exclusive behaviors
/// can't be combined into nonsense (a guided carrier bay) and adding a new
/// behavior is one variant + one `match` arm in `weapon_fire`.
#[derive(Debug, Deserialize, Serialize, Clone, Default, PartialEq)]
pub enum WeaponBehavior {
    /// Plain projectile flying straight until its lifetime expires.
    #[default]
    Ballistic,
    /// Homing missile: launches with partial velocity inheritance, drags back
    /// to cruise speed, and steers toward its target at `turn_rate` rad/s.
    Guided { turn_rate: f32 },
    /// Launches an escort ship of this type instead of a projectile.
    CarrierBay { ship_type: String },
    /// Countermeasure: spawns a decoy that each guided missile currently
    /// homing on the launcher retargets to with probability `strength`.
    Decoy { strength: f32 },
}

/// Optional particle visual for a weapon's projectile (or decoy): a
/// continuous emitter rides the entity. `replace_sprite` hides the plain
/// sprite so the particles ARE the visual (the flare look).
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ParticleFx {
    /// Particles emitted per second.
    pub rate: f32,
    /// Seconds each particle lives (upper bound; actual is 0.5–1×).
    pub particle_lifetime: f32,
    /// Max radial speed of emitted particles (slow = gentle expansion).
    pub speed: f32,
    /// Base particle radius in pixels.
    pub size: f32,
    /// Color override; defaults to the weapon color.
    #[serde(default)]
    pub color: Option<[f32; 3]>,
    /// Hide the projectile sprite — the particles are the whole visual.
    #[serde(default)]
    pub replace_sprite: bool,
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
    pub behavior: WeaponBehavior,
    /// Optional particle trail/plume for the projectile or decoy.
    #[serde(default)]
    pub particles: Option<ParticleFx>,
    /// Half-angle of the weapon's aiming arc, in radians. Zero means a fixed
    /// forward-firing weapon; `PI` represents a full turret. Deserialized
    /// from degrees in asset files.
    #[serde(default, deserialize_with = "deserialize_arc_degrees")]
    pub aimable_arc: f32,
    /// Optional path to a sprite image. When set, the projectile uses this
    /// image instead of a plain colored rectangle.
    #[serde(default)]
    pub sprite_path: Option<String>,
    #[serde(skip)]
    pub sprite_handle: Option<Handle<Image>>,
    /// Optional path to a fire sound effect, played when the weapon fires.
    #[serde(default)]
    pub sound_path: Option<String>,
    #[serde(skip)]
    pub sound_handle: Option<Handle<AudioSource>>,
    #[serde(default)]
    pub display_name: String,
}

/// Fraction of the launching ship's velocity a *guided* missile keeps at launch.
/// (Unguided shots inherit it fully, so you can lead targets with ballistic fire.)
/// Partial inheritance + drag-to-cruise lets a fast ship eventually outrun a homing
/// missile instead of it riding the launcher's chase velocity forever.
const MISSILE_LAUNCH_INHERIT: f32 = 0.7;
/// The launch-speed excess decays to this fraction of its initial value by the end
/// of the missile's lifetime — this drives the drag rate (see `missile_guidance`).
const MISSILE_SETTLE_FRACTION: f32 = 0.05;

/// A deployed countermeasure flare (see [`WeaponBehavior::Decoy`]).
#[derive(Component)]
pub struct Decoy;

/// Fired when a decoy flare is deployed; `decoy_missiles` rolls each inbound
/// missile against `strength` and retargets the spoofed ones to the flare.
#[derive(Event, Message)]
pub struct DecoyDeployed {
    pub owner: Entity,
    pub flare: Entity,
    pub strength: f32,
}

/// Each guided missile currently homing on the flare's owner gets ONE roll:
/// with probability `strength` it retargets to the flare. When the flare
/// burns out (Projectile lifetime), `missile_guidance`'s dead-target handling
/// leaves the missile flying ballistic — permanently spoofed. Event-driven:
/// deploying at the right moment matters, and missiles fired *after* the
/// flare ignore it.
fn decoy_missiles(
    mut reader: MessageReader<DecoyDeployed>,
    mut missiles: Query<&mut GuidedMissile>,
) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for DecoyDeployed {
        owner,
        flare,
        strength,
    } in reader.read()
    {
        for mut missile in &mut missiles {
            if missile.target == Some(*owner) && rng.r#gen_range(0.0..1.0) < *strength {
                missile.target = Some(*flare);
            }
        }
    }
}

/// Range (world units) at which an AI ship reacts to an inbound missile by
/// deploying a decoy.
const AI_DECOY_RANGE: f32 = 550.0;

/// AI ships with a ready decoy weapon deploy it automatically when a guided
/// missile homing on them closes within range — the countermeasure analog of
/// `auto_launch_carrier_bays`. The player always deploys manually.
fn auto_deploy_decoys(
    ships: Query<(Entity, &Ship, &Position), Without<crate::Player>>,
    missiles: Query<(&GuidedMissile, &Position), With<Projectile>>,
    mut fire_writer: MessageWriter<FireCommand>,
) {
    for (entity, ship, pos) in &ships {
        // A ready decoy launcher (off cooldown, ammo left)?
        let Some(decoy_type) = ship.weapon_systems.iter_all().find_map(|(name, ws)| {
            (matches!(ws.weapon.behavior, WeaponBehavior::Decoy { .. })
                && ws.cooldown.is_finished()
                && !ws.ammo_quantity.map(|n| n == 0).unwrap_or(false))
            .then(|| name.clone())
        }) else {
            continue;
        };
        let inbound = missiles.iter().any(|(m, mp)| {
            m.target == Some(entity) && mp.0.distance_squared(pos.0) < AI_DECOY_RANGE * AI_DECOY_RANGE
        });
        if inbound {
            fire_writer.write(FireCommand {
                ship: entity,
                weapon_type: decoy_type,
                target: None,
            });
        }
    }
}

#[derive(Component)]
pub struct GuidedMissile {
    pub target: Option<Entity>,
    pub turn_rate: f32,
    /// The missile's own cruise speed (`weapon.speed`); the inherited launch boost
    /// drags back down to this over the missile's lifetime.
    pub cruise_speed: f32,
    /// Exponential drag rate (1/s) toward `cruise_speed`, from solving
    /// dv/dt = -r·(v − cruise) for r given the settle boundary condition.
    pub drag_rate: f32,
}

impl Weapon {
    pub fn range(&self) -> f32 {
        self.speed * self.lifetime
    }
    /// Particle emitter component for this weapon's projectile, when declared.
    pub fn particle_emitter(&self) -> Option<crate::explosions::WeaponParticleEmitter> {
        self.particles
            .as_ref()
            .map(|fx| crate::explosions::WeaponParticleEmitter {
                fx: fx.clone(),
                color: fx.color.unwrap_or(self.color),
                accum: 0.0,
            })
    }
    pub fn is_guided(&self) -> bool {
        matches!(self.behavior, WeaponBehavior::Guided { .. })
    }
    /// The escort ship type this weapon launches, when it's a carrier bay.
    pub fn carrier_bay(&self) -> Option<&str> {
        match &self.behavior {
            WeaponBehavior::CarrierBay { ship_type } => Some(ship_type),
            _ => None,
        }
    }
}

fn deserialize_arc_degrees<'de, D: Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
    let degrees = f32::deserialize(d)?;
    Ok(degrees.to_radians())
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
            warn!("Weapon type '{weapon_type}' not found in weapons.yaml");
            return None;
        };
        // Lookup the space used:
        let Some(outfitter_item) = item_universe.outfitter_items.get(weapon_type) else {
            warn!("Weapon type '{weapon_type}' not found in outfitter_items.yaml");
            return None;
        };
        if number == 0 {
            return None;
        }
        // Full cooldown as the timer duration: `weapon_system_cooldown` already
        // ticks the timer `number`× faster, which is the one place copy count
        // scales fire rate (N copies → rate N/cooldown). Scaling the duration
        // here as well double-counted (N baked-in copies fired every
        // cooldown/N²) and diverged from weapons bought one at a time
        // (`buy_weapon` bumps `number` without rebuilding the timer).
        let mut cooldown = Timer::from_seconds(weapon.cooldown, TimerMode::Once);
        // Start ready — a fresh ship / fresh purchase shouldn't wait out a
        // full cooldown (up to ~9 s for missiles) before its first shot.
        // (tick(), not set_elapsed(): only tick() updates the finished flag.)
        let duration = cooldown.duration();
        cooldown.tick(duration);
        return Some(WeaponSystem {
            weapon: weapon.clone(),
            cooldown,
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
    mut writer: MessageWriter<WeaponFired>,
    mut escort_writer: MessageWriter<SpawnEscort>,
    mut decoy_writer: MessageWriter<DecoyDeployed>,
    mut commands: Commands,
    mut ships: Query<(
        &Transform,
        &Position,
        &mut Ship,
        &LinearVelocity,
        Option<&mut TracerSlots>,
    )>,
    target_kinematics: Query<(&Position, &LinearVelocity)>,
    item_universe: Res<ItemUniverse>,
) {
    for cmd in reader.read() {
        let Ok((ship_transform, ship_pos, mut ship, linear_velocity, tracer_slots)) =
            ships.get_mut(cmd.ship)
        else {
            continue;
        };
        let ship_radius = ship.data.radius;
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
        // Validate the weapon type BEFORE burning the cooldown — a stale or
        // mistyped key must not silently eat the shot.
        let Some(weapon) = item_universe.weapons.get(&cmd.weapon_type) else {
            continue;
        };
        specific.cooldown.reset();

        // ── Carrier bay: spawn an escort ship instead of a projectile ───
        if let WeaponBehavior::CarrierBay { ship_type } = &weapon.behavior {
            specific.ammo_quantity = specific.ammo_quantity.map(|x| x - 1);
            // Use physics Position (not rendering Transform) so the escort
            // spawns at the carrier's actual location, not last frame's visual.
            let forward = (ship_transform.rotation * Vec3::Y).xy();
            let spawn_pos = ship_pos.0 + forward * (ship_radius + 30.0);
            escort_writer.write(SpawnEscort {
                mother: cmd.ship,
                ship_type: ship_type.clone(),
                carried: Some(cmd.weapon_type.clone()),
                position: spawn_pos,
                mission: None,
            });
            writer.write(WeaponFired {
                ship: cmd.ship,
                weapon_type: cmd.weapon_type.clone(),
            });
            continue;
        }

        // ── Decoy flare: drop a countermeasure aft and try to spoof every
        // missile currently homing on the launcher ───────────────────────
        if let WeaponBehavior::Decoy { strength } = weapon.behavior {
            specific.ammo_quantity = specific.ammo_quantity.map(|x| x - 1);
            let backward = -(ship_transform.rotation * Vec3::Y).xy();
            // The flare drops behind the ship and mostly sheds its velocity,
            // so it separates cleanly from the launcher's flight path.
            let vel = linear_velocity.0 * 0.4 + backward * weapon.speed;
            let spawn_pos = ship_pos.0 + backward * (ship_radius + 8.0);
            let [r, g, b] = weapon.color;
            let hide_sprite = weapon
                .particles
                .as_ref()
                .is_some_and(|fx| fx.replace_sprite);
            let sprite = if hide_sprite {
                // The particle plume IS the visual.
                Sprite {
                    color: Color::NONE,
                    custom_size: Some(Vec2::splat(1.0)),
                    ..default()
                }
            } else if let Some(handle) = &weapon.sprite_handle {
                Sprite::from_image(handle.clone())
            } else {
                Sprite {
                    color: Color::srgb(r, g, b),
                    custom_size: Some(Vec2::splat(6.0)),
                    ..default()
                }
            };
            let mut flare_cmd = commands.spawn((
                DespawnOnExit(PlayState::Flying),
                Projectile {
                    lifetime: weapon.lifetime,
                    owner: cmd.ship,
                    weapon_type: cmd.weapon_type.clone(),
                },
                Decoy,
                RigidBody::Dynamic,
                // Collides with nothing — missiles chase it (they home on
                // its Position) until it burns out.
                Collider::circle(3.0),
                CollisionLayers::new(GameLayer::Weapon, LayerMask::NONE),
                LinearVelocity(vel),
                LinearDamping(1.2),
                Transform::from_translation(spawn_pos.extend(-0.3)),
                sprite,
            ));
            if let Some(emitter) = weapon.particle_emitter() {
                flare_cmd.insert(emitter);
            }
            let flare = flare_cmd.id();
            decoy_writer.write(DecoyDeployed {
                owner: cmd.ship,
                flare,
                strength,
            });
            writer.write(WeaponFired {
                ship: cmd.ship,
                weapon_type: cmd.weapon_type.clone(),
            });
            continue;
        }

        let aim_offset = if weapon.aimable_arc > 0.0 {
            cmd.target
                .and_then(|e| {
                    let (tp, tv) = target_kinematics.get(e).ok()?;
                    let ship_pos = ship_transform.translation.truncate();
                    let ship_dir = (ship_transform.rotation * Vec3::Y).xy();
                    let frame_angle = -ship_dir.y.atan2(ship_dir.x);
                    let (sin_a, cos_a) = frame_angle.sin_cos();
                    let rotate_r =
                        |v: Vec2| Vec2::new(v.x * cos_a - v.y * sin_a, v.x * sin_a + v.y * cos_a);
                    let local_offset = rotate_r(tp.0 - ship_pos);
                    let local_rel_vel = rotate_r(tv.0 - linear_velocity.0);
                    let a =
                        crate::utils::angle_to_hit(weapon.speed, &local_offset, &local_rel_vel)?;
                    let wrapped = (a + std::f32::consts::PI).rem_euclid(2.0 * std::f32::consts::PI)
                        - std::f32::consts::PI;
                    Some(wrapped.clamp(-weapon.aimable_arc, weapon.aimable_arc))
                })
                .unwrap_or(0.0)
        } else {
            0.0
        };
        let aim_rot = Quat::from_rotation_z(aim_offset);
        let fire_rotation = ship_transform.rotation * aim_rot;
        let forward = fire_rotation * Vec3::Y;
        const PROJECTILE_RADIUS: f32 = 10.0;
        const SPAWN_MARGIN: f32 = 1.0;
        let spawn_offset = ship_radius + PROJECTILE_RADIUS + SPAWN_MARGIN;
        let tip = ship_transform.translation + forward * spawn_offset;
        // Guided missiles keep only part of the launcher's velocity (the rest drags
        // off over their lifetime, see GuidedMissile); unguided shots inherit it
        // fully so ballistic fire can lead a moving target.
        let inherit = if weapon.is_guided() {
            MISSILE_LAUNCH_INHERIT
        } else {
            1.0
        };
        let vel = forward.truncate() * weapon.speed + linear_velocity.0 * inherit;
        let [r, g, b] = weapon.color;
        let sprite = if let Some(handle) = &weapon.sprite_handle {
            Sprite::from_image(handle.clone())
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
            Collider::circle(PROJECTILE_RADIUS),
            CollisionLayers::new(GameLayer::Weapon, [GameLayer::Asteroid, GameLayer::Ship]),
            RigidBody::Dynamic,
            // Swept CCD prevents fast projectiles from tunneling through ships
            // at the 50ms headless training timestep (laser travels 25 units/step
            // vs 9-unit corvette radius).
            SweptCcd::LINEAR,
            LinearVelocity(vel),
            Transform::from_translation(tip.xy().extend(-0.3)).with_rotation(fire_rotation),
            sprite,
        ));
        // Decrease the ammo:
        specific.ammo_quantity = specific.ammo_quantity.map(|x| x - 1);
        if let Some(emitter) = weapon.particle_emitter() {
            entity_cmd.insert(emitter);
        }
        if let WeaponBehavior::Guided { turn_rate } = weapon.behavior {
            entity_cmd.insert(GuidedMissile {
                target: cmd.target,
                turn_rate,
                cruise_speed: weapon.speed,
                // Solve dv/dt = -r·(v − cruise): the excess decays to
                // MISSILE_SETTLE_FRACTION by t = lifetime, so r = ln(1/ε)/lifetime.
                drag_rate: (1.0 / MISSILE_SETTLE_FRACTION).ln() / weapon.lifetime.max(0.01),
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
        writer.write(WeaponFired {
            ship: cmd.ship,
            weapon_type: cmd.weapon_type.clone(),
        });
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
        let speed = vel.0.length();
        if speed < f32::EPSILON {
            continue;
        }
        // Drag the inherited launch boost back toward the missile's own cruise speed
        // — analytic step of dv/dt = -drag_rate·(v − cruise). Applies whether or not
        // we currently have a target, so a homing missile fired from a chasing ship
        // can be outrun once the boost bleeds off.
        let dragged =
            missile.cruise_speed + (speed - missile.cruise_speed) * (-missile.drag_rate * dt).exp();

        let current_dir = vel.0 / speed;
        // Steer toward the target if we still have one; otherwise keep heading.
        let new_dir = match missile.target.and_then(|t| all_positions.get(t).ok()) {
            Some(target_pos) => {
                let to_target = (target_pos.0 - transform.translation.xy()).normalize_or_zero();
                let angle = current_dir.angle_to(to_target);
                let turn = angle.clamp(-missile.turn_rate * dt, missile.turn_rate * dt);
                let (sin_t, cos_t) = turn.sin_cos();
                Vec2::new(
                    current_dir.x * cos_t - current_dir.y * sin_t,
                    current_dir.x * sin_t + current_dir.y * cos_t,
                )
            }
            None => {
                missile.target = None;
                current_dir
            }
        };
        vel.0 = new_dir * dragged;

        // Rotate sprite to match travel direction (+Y is forward).
        let visual_angle = new_dir.x.atan2(new_dir.y);
        transform.rotation = Quat::from_rotation_z(-visual_angle);
    }
}

#[cfg(test)]
#[path = "tests/weapons_tests.rs"]
mod tests;
