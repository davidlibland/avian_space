/// RL trajectory collection plugin.
///
/// Adds the `RLAgent` component to AI ships and collects (observation, action, reward)
/// transitions. Completed segments are sent via an mpsc channel to a trainer thread.
///
/// # Modes
/// - `AIPlayMode::BehavioralCloning`: `simple_ai_control` runs normally for non-RLAgent ships;
///   for `RLAgent` ships `rl_step` uses the same rule-based logic and records (obs, action) pairs
///   via the `BCSender` channel.
/// - `AIPlayMode::RLControl`: `simple_ai_control` is excluded for `RLAgent` ships; `rl_step`
///   performs inference (stub: rule-based until a model is integrated) and records full
///   (obs, action, reward, done) transitions via the `RLSender` channel.
///
/// # Action space
/// Factored discrete: (thrust_idx, turn_idx, fire_primary, fire_secondary)
///   - thrust_idx:    0=no-thrust, 1=thrust
///   - turn_idx:      0=left, 1=straight, 2=right
///   - fire_primary:  0=no, 1=yes
///   - fire_secondary: 0=no, 1=yes  (which weapon fires is chosen deterministically)
/// Total: 2×3×2×2 = 24 factored actions.
///
/// # Decision rate
/// `rl_step` is gated by a 4 Hz timer. Between ticks, `repeat_actions` re-emits the last
/// chosen action every Update frame so the ship keeps moving.
use std::sync::mpsc;

use avian2d::prelude::*;
use bevy::prelude::*;

use crate::ai_ships::{AIShip, compute_ai_action};
use crate::asteroids::Asteroid;
use crate::item_universe::ItemUniverse;
use crate::pickups::Pickup;
use crate::planets::Planet;
use crate::rl_obs::{
    self, DiscreteAction, EntitySlotData, ObsInput, TargetSlotData, DETECTION_RADIUS,
    K_ASTEROIDS, K_FRIENDLY_SHIPS, K_HOSTILE_SHIPS, K_PICKUPS, K_PLANETS,
};
use crate::ship::{Personality, Ship, ShipCommand, ShipHostility, Target};
use crate::weapons::FireCommand;
use crate::{GameLayer, PlayState};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of decision-steps per trajectory segment.
pub const RL_SEGMENT_LEN: usize = 128;

/// Decision rate in seconds.
const RL_STEP_PERIOD: f32 = 0.25; // 4 Hz

// ---------------------------------------------------------------------------
// Types sent to the trainer thread
// ---------------------------------------------------------------------------

/// A single (obs, action, reward, done) tuple.
#[derive(Clone)]
pub struct Transition {
    pub obs: Vec<f32>,
    pub action: DiscreteAction,
    pub reward: f32,
    /// True when the ship died at this step (terminal state).
    pub done: bool,
}

/// A fixed-length segment of transitions, ready for PPO / recurrent-PPO training.
pub struct Segment {
    /// Entity bits — stable ID for logging and debugging.
    pub entity_id: u64,
    pub personality: Personality,
    /// LSTM/GRU hidden state at the start of this segment (zeros on first segment).
    pub initial_hidden: Vec<f32>,
    pub transitions: Vec<Transition>,
    /// None when the last transition is terminal; Some(V(s_T)) when truncated.
    pub bootstrap_value: Option<f32>,
}

/// A (obs, action) pair for behavioural-cloning pre-training.
pub struct BCTransition {
    pub obs: Vec<f32>,
    pub action: DiscreteAction,
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// Sender side of the PPO trajectory channel.
#[derive(Resource)]
pub struct RLSender(pub mpsc::SyncSender<Segment>);

/// Receiver side of the PPO trajectory channel.
/// Stored as a `NonSend` resource because `Receiver` is not `Sync`.
pub struct RLReceiver(pub mpsc::Receiver<Segment>);

/// Sender side of the BC data channel.
#[derive(Resource)]
pub struct BCSender(pub mpsc::SyncSender<BCTransition>);

/// Receiver side of the BC data channel.
/// Stored as a `NonSend` resource because `Receiver` is not `Sync`.
pub struct BCReceiver(pub mpsc::Receiver<BCTransition>);

/// Controls whether AI ships are driven by the rule-based system (for BC collection)
/// or by the RL policy.
#[derive(Resource, Default, PartialEq, Eq, Clone, Copy, Debug)]
pub enum AIPlayMode {
    /// Rule-based `simple_ai_control` runs for RLAgent ships; observations and
    /// rule-based actions are recorded for behavioural-cloning training.
    #[default]
    BehavioralCloning,
    /// RLAgent ships are controlled by the RL policy (currently rule-based stub).
    RLControl,
}

/// 4 Hz decision-step timer.
#[derive(Resource)]
struct RLStepTimer(Timer);

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// Emitted by any game system to reward or penalise an `RLAgent` ship.
///
/// Multiple events per step are accumulated and summed into the transition's
/// reward field.
#[derive(Event, Message, Clone)]
pub struct RLReward {
    pub entity: Entity,
    pub reward: f32,
}

/// Emitted by `apply_damage` (in ship.rs) just before an `RLAgent` ship is
/// despawned, carrying the current segment so it can be flushed with done=true.
#[derive(Event, Message)]
pub struct RLShipDied {
    pub entity: Entity,
    pub entity_id: u64,
    pub personality: Personality,
    pub initial_hidden: Vec<f32>,
    pub transitions: Vec<Transition>,
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/// Added to every AI ship we want to train.
#[derive(Component)]
pub struct RLAgent {
    pub personality: Personality,
    /// Placeholder LSTM/GRU hidden state (zeros until a model is integrated).
    pub hidden_state: Vec<f32>,
    /// Observation from the previous decision step.
    pub last_obs: Option<Vec<f32>>,
    /// Action chosen at the previous decision step.
    pub last_action: Option<DiscreteAction>,
    /// Sum of rewards since the last decision step.
    pub accumulated_reward: f32,
    /// Transitions collected since the last segment flush.
    pub segment_buffer: Vec<Transition>,
    /// Hidden state at the start of the current segment (for BPTT).
    pub segment_initial_hidden: Vec<f32>,
    /// Number of decision steps since the last segment flush.
    pub steps_since_flush: usize,
}

impl RLAgent {
    pub fn new(personality: Personality) -> Self {
        Self {
            personality,
            hidden_state: vec![0.0; 64], // placeholder size
            last_obs: None,
            last_action: None,
            accumulated_reward: 0.0,
            segment_buffer: Vec::with_capacity(RL_SEGMENT_LEN),
            segment_initial_hidden: vec![0.0; 64],
            steps_since_flush: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct RLCollectionPlugin;

impl Plugin for RLCollectionPlugin {
    fn build(&self, app: &mut App) {
        let (rl_tx, rl_rx) = mpsc::sync_channel::<Segment>(1024);
        let (bc_tx, bc_rx) = mpsc::sync_channel::<BCTransition>(65536);

        app.insert_resource(RLSender(rl_tx))
            .insert_non_send_resource(RLReceiver(rl_rx))
            .insert_resource(BCSender(bc_tx))
            .insert_non_send_resource(BCReceiver(bc_rx))
            .init_resource::<AIPlayMode>()
            .insert_resource(RLStepTimer(Timer::from_seconds(
                RL_STEP_PERIOD,
                TimerMode::Repeating,
            )))
            .add_message::<RLReward>()
            .add_message::<RLShipDied>()
            .add_systems(
                Update,
                (
                    accumulate_rewards,
                    rl_step,
                    repeat_actions,
                    handle_rl_ship_died,
                )
                    .chain()
                    .run_if(in_state(PlayState::Flying)),
            );
    }
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Read all `RLReward` events and add them to the corresponding agent's
/// accumulated reward.
fn accumulate_rewards(
    mut reward_events: MessageReader<RLReward>,
    mut agents: Query<&mut RLAgent>,
) {
    for ev in reward_events.read() {
        if let Ok(mut agent) = agents.get_mut(ev.entity) {
            agent.accumulated_reward += ev.reward;
        }
    }
}

/// Main RL decision step — runs every Update frame but only acts when the 4 Hz
/// timer fires.
///
/// For each `RLAgent` ship:
/// 1. Build an ego-centric observation (sub-fn `build_all_observations`).
/// 2. Choose an action (rule-based for both BC and RL stub; sub-fn `run_policy`).
/// 3. Store the previous (obs, action, reward) transition and update agent state
///    (sub-fn `store_obs_actions`).
fn rl_step(
    time: Res<Time>,
    mut timer: ResMut<RLStepTimer>,
    mut agents: Query<(
        Entity,
        &mut RLAgent,
        &mut Ship,
        &Position,
        &LinearVelocity,
        &AngularVelocity,
        &MaxLinearSpeed,
        &Transform,
        &ShipHostility,
    )>,
    // Queries for nearby entities (non-overlapping with the agent query).
    all_positions: Query<&Position>,
    all_velocities: Query<&LinearVelocity>,
    planet_query: Query<(Entity, &Planet)>,
    asteroid_query: Query<Entity, With<Asteroid>>,
    pickup_query: Query<Entity, With<Pickup>>,
    ship_query: Query<(Entity, &Ship, &ShipHostility), (With<AIShip>, Without<RLAgent>)>,
    spatial_query: SpatialQuery,
    item_universe: Res<ItemUniverse>,
    mode: Res<AIPlayMode>,
    rl_sender: Res<RLSender>,
    bc_sender: Res<BCSender>,
) {
    if !timer.0.tick(time.delta()).just_finished() {
        return;
    }

    // ── Sub-function 1: build observations for all agents ──────────────────
    let obs_data: Vec<(Entity, Vec<f32>)> = build_all_observations(
        &agents,
        &all_positions,
        &all_velocities,
        &planet_query,
        &asteroid_query,
        &pickup_query,
        &ship_query,
        &spatial_query,
        &item_universe,
    );

    // ── Sub-function 2 (future): run batched model inference here ───────────
    // (When a burn model is integrated, replace the identity pass-through
    // below with a real forward pass over obs_data.)

    // ── Sub-function 3: store transitions and update agent state ───────────
    store_obs_actions(
        &obs_data,
        &mut agents,
        &all_positions,
        &all_velocities,
        &item_universe,
        &rl_sender,
        &bc_sender,
        &mode,
    );
}

/// Re-emit the last chosen action every `Update` frame so the ship continues
/// moving between 4 Hz decision steps.
/// Only active in `RLControl` mode — in `BehavioralCloning` mode `simple_ai_control`
/// drives the ship directly.
fn repeat_actions(
    mode: Res<AIPlayMode>,
    agents: Query<(Entity, &RLAgent, &Ship)>,
    mut ship_writer: MessageWriter<ShipCommand>,
    mut fire_writer: MessageWriter<FireCommand>,
) {
    if *mode != AIPlayMode::RLControl {
        return;
    }
    for (entity, agent, ship) in &agents {
        let Some(action) = agent.last_action else {
            continue;
        };
        let (turn_idx, thrust_idx, fire_primary, fire_secondary) = action;
        let (thrust, turn) = rl_obs::discrete_to_controls(thrust_idx, turn_idx);
        ship_writer.write(ShipCommand {
            entity,
            thrust,
            turn,
            reverse: 0.0,
        });
        let target_entity = ship.target.as_ref().map(|t| t.get_entity());
        let target_is_ship = matches!(ship.target, Some(Target::Ship(_)));
        if fire_primary == 1 {
            for weapon_type in ship.weapon_systems.primary.keys() {
                fire_writer.write(FireCommand {
                    ship: entity,
                    weapon_type: weapon_type.clone(),
                    target: target_entity,
                });
            }
        }
        if fire_secondary == 1 {
            if let Some((wtype, _)) = rl_obs::select_secondary_weapon(ship, target_is_ship) {
                fire_writer.write(FireCommand {
                    ship: entity,
                    weapon_type: wtype.to_string(),
                    target: target_entity,
                });
            }
        }
    }
}

/// Flush a terminal segment when an `RLShipDied` event is received.
///
/// The event is emitted in `apply_damage` (ship.rs) before the entity is
/// despawned. Because `Commands` are deferred, the entity is still alive when
/// this system runs.
fn handle_rl_ship_died(
    mut events: MessageReader<RLShipDied>,
    rl_sender: Res<RLSender>,
) {
    for ev in events.read() {
        let mut transitions = ev.transitions.clone();
        // Mark the last transition as terminal.
        if let Some(last) = transitions.last_mut() {
            last.done = true;
        }
        let segment = Segment {
            entity_id: ev.entity_id,
            personality: ev.personality.clone(),
            initial_hidden: ev.initial_hidden.clone(),
            transitions,
            bootstrap_value: None, // terminal — no bootstrapping needed
        };
        let _ = rl_sender.0.try_send(segment);
    }
}

// ---------------------------------------------------------------------------
// Sub-functions called from rl_step
// ---------------------------------------------------------------------------

/// Build ego-centric observations for every `RLAgent` ship.
///
/// Returns a vec of `(entity, observation_vec, rule_based_action)` tuples.
/// The rule-based action is used directly in BC mode and as a stub in RL mode.
fn build_all_observations(
    agents: &Query<(
        Entity,
        &mut RLAgent,
        &mut Ship,
        &Position,
        &LinearVelocity,
        &AngularVelocity,
        &MaxLinearSpeed,
        &Transform,
        &ShipHostility,
    )>,
    all_positions: &Query<&Position>,
    all_velocities: &Query<&LinearVelocity>,
    planet_query: &Query<(Entity, &Planet)>,
    asteroid_query: &Query<Entity, With<Asteroid>>,
    pickup_query: &Query<Entity, With<Pickup>>,
    ship_query: &Query<(Entity, &Ship, &ShipHostility), (With<AIShip>, Without<RLAgent>)>,
    spatial_query: &SpatialQuery,
    item_universe: &ItemUniverse,
) -> Vec<(Entity, Vec<f32>)> {
    let mut results = Vec::new();

    for (entity, agent, ship, pos, vel, ang_vel, max_speed, transform, self_hostility) in
        agents.iter()
    {
        // Ego-frame rotation parameters.
        let ship_dir = (transform.rotation * Vec3::Y).xy();
        let heading = [ship_dir.x, ship_dir.y];
        let (sin_a, cos_a) = rl_obs::ego_frame_sincos(heading);

        // Spatial query: all entities within DETECTION_RADIUS.
        let filter = SpatialQueryFilter::from_mask([
            GameLayer::Planet,
            GameLayer::Asteroid,
            GameLayer::Ship,
            GameLayer::Pickup,
        ])
        .with_excluded_entities([entity]);

        let nearby_hits: Vec<(Entity, f32)> = spatial_query
            .shape_intersections(
                &Collider::circle(DETECTION_RADIUS),
                pos.0,
                0.0,
                &filter,
            )
            .into_iter()
            .filter_map(|hit| {
                all_positions
                    .get(hit)
                    .ok()
                    .map(|p| (hit, (p.0 - pos.0).length_squared()))
            })
            .collect();

        // Helper: build EntitySlotData for a single entity.
        let make_slot = |e: Entity, extra: f32| -> EntitySlotData {
            let world_offset = all_positions
                .get(e)
                .map(|p| p.0 - pos.0)
                .unwrap_or(Vec2::ZERO);
            let world_rel_vel = all_velocities
                .get(e)
                .map(|v| v.0 - vel.0)
                .unwrap_or(Vec2::ZERO);
            let rel_pos = rl_obs::rotate_to_ego(
                [world_offset.x, world_offset.y],
                sin_a,
                cos_a,
            );
            let rel_vel = rl_obs::rotate_to_ego(
                [world_rel_vel.x, world_rel_vel.y],
                sin_a,
                cos_a,
            );
            EntitySlotData {
                rel_pos: [
                    rel_pos[0] / DETECTION_RADIUS,
                    rel_pos[1] / DETECTION_RADIUS,
                ],
                rel_vel: [rel_vel[0] / 300.0, rel_vel[1] / 300.0],
                extra,
            }
        };

        // Buckets.
        let mut planets: Vec<(Entity, f32)> = nearby_hits
            .iter()
            .filter(|(e, _)| planet_query.contains(*e))
            .cloned()
            .collect();
        planets.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut asteroids: Vec<(Entity, f32)> = nearby_hits
            .iter()
            .filter(|(e, _)| asteroid_query.contains(*e))
            .cloned()
            .collect();
        asteroids.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut hostile_ships: Vec<(Entity, f32)> = nearby_hits
            .iter()
            .filter(|(e, _)| {
                ship_query
                    .get(*e)
                    .map(|(_, other_ship, _)| ship.should_engage(&ShipHostility(other_ship.enemies.clone())))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        hostile_ships.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut friendly_ships: Vec<(Entity, f32)> = nearby_hits
            .iter()
            .filter(|(e, _)| {
                ship_query
                    .get(*e)
                    .map(|(_, other_ship, _)| !ship.should_engage(&ShipHostility(other_ship.enemies.clone())))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        friendly_ships.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut pickups: Vec<(Entity, f32)> = nearby_hits
            .iter()
            .filter(|(e, _)| pickup_query.contains(*e))
            .cloned()
            .collect();
        pickups.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let nearby_planets: Vec<EntitySlotData> = planets
            .iter()
            .take(K_PLANETS)
            .map(|(e, _)| make_slot(*e, 1.0))
            .collect();
        let nearby_asteroids: Vec<EntitySlotData> = asteroids
            .iter()
            .take(K_ASTEROIDS)
            .map(|(e, _)| make_slot(*e, 1.0))
            .collect();
        let nearby_hostile_ships: Vec<EntitySlotData> = hostile_ships
            .iter()
            .take(K_HOSTILE_SHIPS)
            .map(|(e, _)| {
                let health_frac = ship_query
                    .get(*e)
                    .map(|(_, s, _)| s.health as f32 / s.data.max_health.max(1) as f32)
                    .unwrap_or(1.0);
                make_slot(*e, health_frac)
            })
            .collect();
        let nearby_friendly_ships: Vec<EntitySlotData> = friendly_ships
            .iter()
            .take(K_FRIENDLY_SHIPS)
            .map(|(e, _)| {
                let health_frac = ship_query
                    .get(*e)
                    .map(|(_, s, _)| s.health as f32 / s.data.max_health.max(1) as f32)
                    .unwrap_or(1.0);
                make_slot(*e, health_frac)
            })
            .collect();
        let nearby_pickups: Vec<EntitySlotData> = pickups
            .iter()
            .take(K_PICKUPS)
            .map(|(e, _)| make_slot(*e, 1.0))
            .collect();

        // Target slot.
        let target_slot = build_target_slot(&ship, pos, vel, all_positions, all_velocities, sin_a, cos_a, &ship_query);

        let obs_input = ObsInput {
            personality: &agent.personality,
            ship: &ship,
            velocity: [vel.x, vel.y],
            angular_velocity: ang_vel.0,
            ship_heading: heading,
            target: target_slot,
            nearby_planets,
            nearby_asteroids,
            nearby_hostile_ships,
            nearby_friendly_ships,
            nearby_pickups,
        };
        let obs = rl_obs::encode_observation(&obs_input);

        results.push((entity, obs));
    }

    results
}

/// Build the target slot data for a ship's current target.
fn build_target_slot(
    ship: &Ship,
    pos: &Position,
    vel: &LinearVelocity,
    all_positions: &Query<&Position>,
    all_velocities: &Query<&LinearVelocity>,
    sin_a: f32,
    cos_a: f32,
    ship_query: &Query<(Entity, &Ship, &ShipHostility), (With<AIShip>, Without<RLAgent>)>,
) -> Option<TargetSlotData> {
    let target = ship.target.as_ref()?;
    let target_entity = target.get_entity();
    let target_pos = all_positions.get(target_entity).ok()?;
    let target_vel = all_velocities
        .get(target_entity)
        .map(|v| v.0)
        .unwrap_or(Vec2::ZERO);

    let world_offset = target_pos.0 - pos.0;
    let world_rel_vel = target_vel - vel.0;

    let rel_pos_ego = rl_obs::rotate_to_ego(
        [world_offset.x, world_offset.y],
        sin_a,
        cos_a,
    );
    let rel_vel_ego = rl_obs::rotate_to_ego(
        [world_rel_vel.x, world_rel_vel.y],
        sin_a,
        cos_a,
    );

    let (entity_type, hostility, value_hint) = match target {
        Target::Ship(e) => {
            let hostile = ship_query
                .get(*e)
                .map(|(_, other, _)| ship.should_engage(&ShipHostility(other.enemies.clone())))
                .unwrap_or(false);
            (0u8, if hostile { 1.0 } else { -1.0 }, 1.0)
        }
        Target::Asteroid(_) => (1, 0.0, 1.0),
        Target::Planet(_) => (2, 0.0, 1.0),
        Target::Pickup(_) => (3, 0.0, 1.0),
    };

    Some(TargetSlotData {
        entity_type,
        rel_pos: [
            rel_pos_ego[0] / DETECTION_RADIUS,
            rel_pos_ego[1] / DETECTION_RADIUS,
        ],
        rel_vel: [rel_vel_ego[0] / 300.0, rel_vel_ego[1] / 300.0],
        hostility,
        value_hint,
    })
}

/// Store the previous (obs, action, reward) transition for each agent, update
/// agent state with the new observation and action, and flush full segments.
fn store_obs_actions(
    decisions: &[(Entity, Vec<f32>)],
    agents: &mut Query<(
        Entity,
        &mut RLAgent,
        &mut Ship,
        &Position,
        &LinearVelocity,
        &AngularVelocity,
        &MaxLinearSpeed,
        &Transform,
        &ShipHostility,
    )>,
    all_positions: &Query<&Position>,
    all_velocities: &Query<&LinearVelocity>,
    item_universe: &ItemUniverse,
    rl_sender: &RLSender,
    bc_sender: &BCSender,
    mode: &AIPlayMode,
) {
    for (entity, obs) in decisions {
        let Ok((_, mut agent, mut ship, pos, vel, _, max_speed, transform, _)) =
            agents.get_mut(*entity)
        else {
            continue;
        };

        // Compute the rule-based action using the same logic as simple_ai_control.
        let accurate_action = if let Some(raw) = compute_ai_action(
            &*ship,
            pos.0,
            vel.0,
            max_speed.0,
            transform,
            all_positions,
            all_velocities,
            item_universe,
        ) {
            let (thrust_idx, turn_idx) = rl_obs::controls_to_discrete(raw.thrust, raw.turn);
            let fire_primary: u8 = if raw
                .weapons_to_fire
                .iter()
                .any(|(w, _)| ship.weapon_systems.primary.contains_key(w))
            {
                1
            } else {
                0
            };
            let fire_secondary: u8 = if raw
                .weapons_to_fire
                .iter()
                .any(|(w, _)| ship.weapon_systems.secondary.contains_key(w))
            {
                1
            } else {
                0
            };
            (turn_idx, thrust_idx, fire_primary, fire_secondary)
        } else {
            (1, 1, 0, 0) // no target → coast straight
        };

        // Store the PREVIOUS step's transition (obs recorded at t-1, action taken
        // at t-1, rewards accumulated between t-1 and t).
        if let (Some(last_obs), Some(last_action)) =
            (agent.last_obs.clone(), agent.last_action)
        {
            let reward = agent.accumulated_reward;
            // Add health-fraction as a per-step reward signal.
            let health_reward = ship.health as f32 / ship.data.max_health.max(1) as f32;
            let personality_health_weight = match agent.personality {
                Personality::Fighter => 0.3,
                Personality::Miner | Personality::Trader => 0.5,
            };
            let total_reward = reward + health_reward * personality_health_weight;

            let transition = Transition {
                obs: last_obs,
                action: last_action,
                reward: total_reward,
                done: false,
            };
            agent.segment_buffer.push(transition.clone());
            agent.accumulated_reward = 0.0;
            agent.steps_since_flush += 1;

            // Also send BC data.
            if *mode == AIPlayMode::BehavioralCloning {
                let bc = BCTransition {
                    obs: transition.obs.clone(),
                    action: accurate_action,
                };
                let _ = bc_sender.0.try_send(bc);
            }

            // Flush if segment is full.
            if agent.steps_since_flush >= RL_SEGMENT_LEN {
                let segment = Segment {
                    entity_id: entity.to_bits(),
                    personality: agent.personality.clone(),
                    initial_hidden: agent.segment_initial_hidden.clone(),
                    transitions: agent.segment_buffer.drain(..).collect(),
                    bootstrap_value: Some(0.0), // TODO: replace with V(s_T) from model
                };
                let _ = rl_sender.0.try_send(segment);
                agent.segment_initial_hidden = agent.hidden_state.clone();
                agent.steps_since_flush = 0;
            }
        }

        // Update agent with new observation and action.
        agent.last_obs = Some(obs.clone());
        agent.last_action = Some(accurate_action);
    }
}

// ---------------------------------------------------------------------------
// Public helper to build RLShipDied from apply_damage
// ---------------------------------------------------------------------------

/// Build the `RLShipDied` event data from an `RLAgent` component before despawn.
pub fn build_rl_ship_died(entity: Entity, agent: &RLAgent) -> RLShipDied {
    RLShipDied {
        entity,
        entity_id: entity.to_bits(),
        personality: agent.personality.clone(),
        initial_hidden: agent.segment_initial_hidden.clone(),
        transitions: agent.segment_buffer.clone(),
    }
}
