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
use std::collections::VecDeque;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use avian2d::prelude::*;
use bevy::prelude::*;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::{Int, TensorData};
use rand::Rng;

use crate::ai_ships::{AIShip, compute_ai_action};
use crate::asteroids::Asteroid;
use crate::item_universe::ItemUniverse;
use crate::model::{
    self, HIDDEN_DIM, InferenceNet, N_OBJECTS, NET_INPUT_DIM, POLICY_OUTPUT_DIM, RLInner,
    RLResource, TrainBackend, obs_to_model_input, training_net_to_bytes,
};
use crate::pickups::Pickup;
use crate::planets::Planet;
use crate::rl_obs::{
    self, DETECTION_RADIUS, DiscreteAction, EntitySlotData, K_ASTEROIDS, K_FRIENDLY_SHIPS,
    K_HOSTILE_SHIPS, K_PICKUPS, K_PLANETS, ObsInput, TargetSlotData,
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

// BC training hyperparameters
/// Maximum number of `BCTransition`s kept in the replay buffer.
const BC_BUFFER_SIZE: usize = 32_768;
/// Mini-batch size for each BC gradient step.
const BC_BATCH_SIZE: usize = 256;
/// Adam learning rate for BC pre-training.
const BC_LR: f64 = 3e-4;
/// Push weights to the inference net every N gradient steps.
const BC_WEIGHT_SYNC_INTERVAL: usize = 50;
/// Save a checkpoint to disk every N gradient steps.
const BC_SAVE_INTERVAL: usize = 1_000;
/// Number of gradient steps to run per drain cycle when the buffer is full.
/// Higher values keep the compute busy between data-drain pauses.
const BC_STEPS_PER_DRAIN: usize = 10;

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

/// A (model_input, action) pair for behavioural-cloning pre-training.
///
/// `model_input` already holds the output of `obs_to_model_input` —
/// i.e. the `[N_OBJECTS × NET_INPUT_DIM]` flat tensor row.  The transform
/// is done once at collection time so the training thread pays no per-sample
/// CPU cost per gradient step.
pub struct BCTransition {
    pub model_input: Vec<f32>,
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

/// Emitted just before an `RLAgent` ship jumps out of the system.
/// The segment is flushed as a truncated (non-terminal) trajectory.
#[derive(Event, Message)]
pub struct RLShipJumped {
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

pub struct RLCollectionPlugin {
    pub mode: crate::AppMode,
    pub fresh: bool,
}

impl Plugin for RLCollectionPlugin {
    fn build(&self, app: &mut App) {
        let (rl_tx, rl_rx) = mpsc::sync_channel::<Segment>(1024);
        let (bc_tx, bc_rx) = mpsc::sync_channel::<BCTransition>(65536);

        // Create RLResource first so we can clone the Arc before inserting it.
        let rl_resource = RLResource::new();
        let inference_net_arc = Arc::clone(&rl_resource.inference_net);

        // Resolve the run directory once; training threads own it from here.
        let experiment = crate::experiments::setup_experiment(self.fresh);

        // Spawn training threads only for the modes that need them.
        match self.mode {
            crate::AppMode::BCTraining => {
                spawn_bc_training_thread(bc_rx, inference_net_arc, experiment);
            }
            crate::AppMode::RLTraining => {
                // TODO: spawn PPO training thread here.
                // For now the channel has no consumer; data is silently dropped.
                drop(bc_rx);
            }
            crate::AppMode::Classic | crate::AppMode::Inference => {
                // No training — channels have no consumer; data is silently dropped.
                drop(bc_rx);
            }
        }

        // Map the app mode to the in-game AI control mode.
        let ai_play_mode = match self.mode {
            crate::AppMode::Classic | crate::AppMode::BCTraining => AIPlayMode::BehavioralCloning,
            crate::AppMode::Inference | crate::AppMode::RLTraining => AIPlayMode::RLControl,
        };

        app.insert_resource(RLSender(rl_tx))
            .insert_non_send_resource(RLReceiver(rl_rx))
            .insert_resource(BCSender(bc_tx))
            .insert_resource(rl_resource)
            .insert_resource(ai_play_mode)
            .insert_resource(RLStepTimer(Timer::from_seconds(
                RL_STEP_PERIOD,
                TimerMode::Repeating,
            )))
            .add_message::<RLReward>()
            .add_message::<RLShipDied>()
            .add_message::<RLShipJumped>()
            .add_systems(
                Update,
                (
                    accumulate_rewards,
                    rl_step,
                    repeat_actions,
                    handle_rl_ship_died,
                    handle_rl_ship_jumped,
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
fn accumulate_rewards(mut reward_events: MessageReader<RLReward>, mut agents: Query<&mut RLAgent>) {
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
    rl_resource: Res<RLResource>,
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

    // ── Sub-function 2: run batched model inference (RLControl mode only) ──
    // In BehavioralCloning mode the rule-based action from compute_ai_action
    // is used instead, so we skip the model entirely.
    let model_actions: Option<Vec<DiscreteAction>> =
        if *mode == AIPlayMode::RLControl && !obs_data.is_empty() {
            let batch_size = obs_data.len();
            let mut batch_flat =
                Vec::with_capacity(batch_size * model::N_OBJECTS * model::NET_INPUT_DIM);
            for (_, obs) in &obs_data {
                batch_flat.extend(model::obs_to_model_input(obs));
            }
            let inference = rl_resource.inference_net.lock().unwrap();
            let logits = inference.run_inference(batch_flat, batch_size);
            let actions = logits
                .chunks(model::POLICY_OUTPUT_DIM)
                .map(model::logits_to_discrete_action)
                .collect();
            Some(actions)
        } else {
            None
        };

    // ── Sub-function 3: store transitions and update agent state ───────────
    store_obs_actions(
        &obs_data,
        model_actions.as_deref(),
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
fn handle_rl_ship_died(mut events: MessageReader<RLShipDied>, rl_sender: Res<RLSender>) {
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

/// Flush a truncated segment when an `RLShipJumped` event is received.
///
/// Unlike death, the last transition is NOT marked as terminal (`done = false`).
/// `bootstrap_value` is left as `None` — the jump ends the trajectory without
/// a future-value estimate.
fn handle_rl_ship_jumped(mut events: MessageReader<RLShipJumped>, rl_sender: Res<RLSender>) {
    for ev in events.read() {
        if ev.transitions.is_empty() {
            continue;
        }
        let segment = Segment {
            entity_id: ev.entity_id,
            personality: ev.personality.clone(),
            initial_hidden: ev.initial_hidden.clone(),
            transitions: ev.transitions.clone(),
            bootstrap_value: None,
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
            .shape_intersections(&Collider::circle(DETECTION_RADIUS), pos.0, 0.0, &filter)
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
            let rel_pos = rl_obs::rotate_to_ego([world_offset.x, world_offset.y], sin_a, cos_a);
            let rel_vel = rl_obs::rotate_to_ego([world_rel_vel.x, world_rel_vel.y], sin_a, cos_a);
            EntitySlotData {
                rel_pos: [rel_pos[0] / DETECTION_RADIUS, rel_pos[1] / DETECTION_RADIUS],
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
                    .map(|(_, other_ship, _)| {
                        ship.should_engage(&ShipHostility(other_ship.enemies.clone()))
                    })
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
                    .map(|(_, other_ship, _)| {
                        !ship.should_engage(&ShipHostility(other_ship.enemies.clone()))
                    })
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
        let target_slot = build_target_slot(
            &ship,
            pos,
            vel,
            all_positions,
            all_velocities,
            sin_a,
            cos_a,
            &ship_query,
        );

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

    let rel_pos_ego = rl_obs::rotate_to_ego([world_offset.x, world_offset.y], sin_a, cos_a);
    let rel_vel_ego = rl_obs::rotate_to_ego([world_rel_vel.x, world_rel_vel.y], sin_a, cos_a);

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
///
/// `model_actions` — when `Some`, contains one `DiscreteAction` per entry in
/// `decisions` (same order). Used as the executed action in `RLControl` mode.
/// In `BehavioralCloning` mode (or when `None`) the rule-based action from
/// `compute_ai_action` is used for both execution and BC labels.
fn store_obs_actions(
    decisions: &[(Entity, Vec<f32>)],
    model_actions: Option<&[DiscreteAction]>,
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
    for (idx, (entity, obs)) in decisions.iter().enumerate() {
        let Ok((_, mut agent, mut ship, pos, vel, _, max_speed, transform, _)) =
            agents.get_mut(*entity)
        else {
            continue;
        };

        // Always compute the rule-based action — used as BC label and as the
        // fallback when no model action is available.
        let rule_based_action = if let Some(raw) = compute_ai_action(
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

        // In RLControl mode use the model's action; otherwise use rule-based.
        let executed_action: DiscreteAction = if *mode == AIPlayMode::RLControl {
            model_actions
                .and_then(|ma| ma.get(idx))
                .copied()
                .unwrap_or(rule_based_action)
        } else {
            rule_based_action
        };
        // BC label is always the rule-based action (what simple_ai_control would do).
        let accurate_action = rule_based_action;

        // Store the PREVIOUS step's transition (obs recorded at t-1, action taken
        // at t-1, rewards accumulated between t-1 and t).
        if let (Some(last_obs), Some(last_action)) = (agent.last_obs.clone(), agent.last_action) {
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

            // Also send BC data.  Pre-process obs → model_input here so the
            // training thread pays zero per-sample transform cost per step.
            if *mode == AIPlayMode::BehavioralCloning {
                let bc = BCTransition {
                    model_input: obs_to_model_input(&transition.obs),
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
        agent.last_action = Some(executed_action);
    }
}

// ---------------------------------------------------------------------------
// BC training thread
// ---------------------------------------------------------------------------

/// Spawn the behavioural-cloning pre-training thread.
///
/// The thread owns `bc_rx` and continuously:
/// 1. Drains `BCTransition`s into a circular replay buffer.
/// 2. Samples random mini-batches and runs Adam gradient steps, minimising
///    cross-entropy loss on all four action heads simultaneously.
/// 3. Every [`BC_WEIGHT_SYNC_INTERVAL`] steps pushes serialised weights to
///    `inference_net` so the game thread always uses the latest policy.
/// 4. Every [`BC_SAVE_INTERVAL`] steps saves a checkpoint to [`BC_SAVE_PATH`]`.bin`.
///
/// On startup the thread looks for an existing checkpoint and, if found, loads
/// it into both the training net and the inference net before the first
/// gradient step.
fn spawn_bc_training_thread(
    bc_rx: mpsc::Receiver<BCTransition>,
    inference_net: Arc<Mutex<InferenceNet>>,
    experiment: crate::experiments::ExperimentSetup,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let device = Default::default(); // <TrainBackend as Backend>::Device
        let mut inner = RLInner::<TrainBackend>::new(&device);

        let checkpoint_path = experiment.policy_checkpoint_path();

        // Try to load an existing BC checkpoint into the training net.
        if !experiment.is_fresh {
            if let Some(net) = model::load_training_net(&checkpoint_path, &device) {
                inner.policy_net = Some(net);
                // Sync the loaded weights to the inference net immediately.
                let bytes = training_net_to_bytes(inner.policy_net.as_ref().unwrap());
                if let Ok(mut lock) = inference_net.lock() {
                    lock.load_bytes(bytes);
                }
                println!("[bc] Resumed from checkpoint {checkpoint_path}");
            } else {
                println!("[bc] No checkpoint found at {checkpoint_path} — starting from scratch.");
            }
        } else {
            println!("[bc] Fresh run — skipping checkpoint load.");
        }

        let buffer_path = experiment.buffer_checkpoint_path();

        // Try to restore a previously saved replay buffer so training starts hot.
        let mut buffer: VecDeque<BCTransition> = if !experiment.is_fresh {
            load_bc_buffer(&buffer_path)
                .map(|b| {
                    println!("[bc] Restored buffer with {} transitions from {buffer_path}", b.len());
                    b
                })
                .unwrap_or_else(|| {
                    println!("[bc] No buffer checkpoint found at {buffer_path} — starting empty.");
                    VecDeque::with_capacity(BC_BUFFER_SIZE)
                })
        } else {
            println!("[bc] Fresh run — starting with empty buffer.");
            VecDeque::with_capacity(BC_BUFFER_SIZE)
        };
        let mut step = 0usize;
        let mut rng = rand::thread_rng();

        loop {
            // ── Drain incoming data ──────────────────────────────────────────
            // Only block waiting for new transitions when the buffer is too
            // small to train.  Once we have enough data, drain non-blocking
            // and go straight to gradient steps.
            if buffer.len() < BC_BATCH_SIZE {
                match bc_rx.recv_timeout(std::time::Duration::from_millis(50)) {
                    Ok(t) => {
                        if buffer.len() >= BC_BUFFER_SIZE {
                            buffer.pop_front();
                        }
                        buffer.push_back(t);
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        println!("[bc] Channel disconnected — saving final checkpoint.");
                        if let Some(net) = inner.policy_net.as_ref() {
                            save_bc_checkpoint(net, &checkpoint_path);
                        }
                        save_bc_buffer(&buffer, &buffer_path);
                        break;
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                }
            }
            // Always drain the rest of the channel non-blocking.
            while let Ok(t) = bc_rx.try_recv() {
                if buffer.len() >= BC_BUFFER_SIZE {
                    buffer.pop_front();
                }
                buffer.push_back(t);
            }

            if buffer.len() < BC_BATCH_SIZE {
                continue;
            }

            // ── One-time setup hoisted above the inner loop ──────────────────
            let loss_fn = CrossEntropyLossConfig::new().init::<TrainBackend>(&device);

            // ── Multiple gradient steps per drain cycle ──────────────────────
            // Amortises the drain overhead: run BC_STEPS_PER_DRAIN steps
            // before going back to check for new data.
            for _ in 0..BC_STEPS_PER_DRAIN {
                let n = buffer.len();

                // ── Build mini-batch ─────────────────────────────────────────
                // model_input is already the processed [N_OBJECTS × NET_INPUT_DIM]
                // flat row — no per-sample transform needed here.
                let batch: Vec<&BCTransition> = (0..BC_BATCH_SIZE)
                    .map(|_| &buffer[rng.gen_range(0..n)])
                    .collect();

                let flat: Vec<f32> = batch
                    .iter()
                    .flat_map(|t| t.model_input.iter().copied())
                    .collect();
                let input = burn::tensor::Tensor::<TrainBackend, 3>::from_data(
                    TensorData::new(flat, [BC_BATCH_SIZE, N_OBJECTS, NET_INPUT_DIM]),
                    &device,
                );

                // Build one [B, 4] label tensor and slice per head — four
                // Vec builds collapsed into one.
                let labels_flat: Vec<i64> = batch
                    .iter()
                    .flat_map(|t| {
                        [
                            t.action.0 as i64,
                            t.action.1 as i64,
                            t.action.2 as i64,
                            t.action.3 as i64,
                        ]
                    })
                    .collect();
                let labels = burn::tensor::Tensor::<TrainBackend, 2, Int>::from_data(
                    TensorData::new(labels_flat, [BC_BATCH_SIZE, 4]),
                    &device,
                );
                let turn_t   = labels.clone().narrow(1, 0, 1).reshape([BC_BATCH_SIZE]);
                let thrust_t = labels.clone().narrow(1, 1, 1).reshape([BC_BATCH_SIZE]);
                let fp_t     = labels.clone().narrow(1, 2, 1).reshape([BC_BATCH_SIZE]);
                let fs_t     = labels.narrow(1, 3, 1).reshape([BC_BATCH_SIZE]);

                // ── Forward + loss ───────────────────────────────────────────
                let grads = {
                    let net = inner.policy_net.as_ref().unwrap();
                    let logits = net.forward(input); // [B, 9]

                    let turn_loss = loss_fn.forward(logits.clone().narrow(1, 0, 3), turn_t);
                    let thrust_loss = loss_fn.forward(logits.clone().narrow(1, 3, 2), thrust_t);
                    let fp_loss = loss_fn.forward(logits.clone().narrow(1, 5, 2), fp_t);
                    let fs_loss = loss_fn.forward(logits.narrow(1, 7, 2), fs_t);
                    let total_loss = turn_loss + thrust_loss + fp_loss + fs_loss;

                    if step % 100 == 0 {
                        let v = total_loss.clone().into_scalar();
                        println!("[bc] step={step:>6}  loss={v:.4}  buffer={}", buffer.len());
                    }

                    let raw = total_loss.backward();
                    GradientsParams::from_grads(raw, net)
                };

                let net = inner.policy_net.take().unwrap();
                inner.policy_net = Some(inner.policy_optim.step(BC_LR, net, grads));
                step += 1;

                // ── Sync weights to inference net ────────────────────────────
                if step % BC_WEIGHT_SYNC_INTERVAL == 0 {
                    let bytes = training_net_to_bytes(inner.policy_net.as_ref().unwrap());
                    if let Ok(mut lock) = inference_net.lock() {
                        lock.load_bytes(bytes);
                    }
                }

                // ── Periodic checkpoint ──────────────────────────────────────
                if step % BC_SAVE_INTERVAL == 0 {
                    save_bc_checkpoint(inner.policy_net.as_ref().unwrap(), &checkpoint_path);
                    save_bc_buffer(&buffer, &buffer_path);
                }
            }
        }
    })
}

fn save_bc_checkpoint(net: &model::RLNet<TrainBackend>, path: &str) {
    model::save_training_net(net, path);
}

// ---------------------------------------------------------------------------
// BC buffer persistence
// ---------------------------------------------------------------------------
//
// Binary format (little-endian):
//   [u32] obs_dim   — number of f32s per observation
//   [u32] count     — number of transitions
//   for each transition:
//     [u8; 4]       — action (turn, thrust, fire_primary, fire_secondary)
//     [f32; obs_dim] — flat observation

/// Serialise the replay buffer to `path`.  Silent on error (training continues).
fn save_bc_buffer(buffer: &VecDeque<BCTransition>, path: &str) {
    use std::io::Write;
    let Some(first) = buffer.front() else { return };
    let input_dim = first.model_input.len() as u32;

    let result = (|| -> std::io::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(&input_dim.to_le_bytes())?;
        f.write_all(&(buffer.len() as u32).to_le_bytes())?;
        for t in buffer {
            f.write_all(&[t.action.0, t.action.1, t.action.2, t.action.3])?;
            for &v in &t.model_input {
                f.write_all(&v.to_le_bytes())?;
            }
        }
        Ok(())
    })();

    if let Err(e) = result {
        eprintln!("[bc] Failed to save buffer to {path}: {e}");
    } else {
        println!("[bc] Buffer saved ({} transitions) → {path}", buffer.len());
    }
}

/// Deserialise a previously saved replay buffer from `path`.
/// Returns `None` if the file is missing, corrupt, or has a mismatched
/// `model_input` dimension (e.g. saved before the obs→model_input change).
fn load_bc_buffer(path: &str) -> Option<VecDeque<BCTransition>> {
    use std::io::Read;
    let expected_dim = N_OBJECTS * NET_INPUT_DIM;
    let mut f = std::io::BufReader::new(std::fs::File::open(path).ok()?);

    let mut u32_buf = [0u8; 4];
    f.read_exact(&mut u32_buf).ok()?;
    let stored_dim = u32::from_le_bytes(u32_buf) as usize;
    if stored_dim != expected_dim {
        eprintln!(
            "[bc] Buffer at {path} has dim={stored_dim}, expected {expected_dim} — discarding."
        );
        return None;
    }

    f.read_exact(&mut u32_buf).ok()?;
    let count = u32::from_le_bytes(u32_buf) as usize;

    let mut buffer = VecDeque::with_capacity(count.min(BC_BUFFER_SIZE));
    let mut action_buf = [0u8; 4];
    let mut input_bytes = vec![0u8; expected_dim * 4];

    for _ in 0..count {
        f.read_exact(&mut action_buf).ok()?;
        f.read_exact(&mut input_bytes).ok()?;
        let model_input: Vec<f32> = input_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        buffer.push_back(BCTransition {
            model_input,
            action: (action_buf[0], action_buf[1], action_buf[2], action_buf[3]),
        });
    }
    Some(buffer)
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

/// Build the `RLShipJumped` event data from an `RLAgent` component before a
/// voluntary jump-out.  The trajectory is flushed as truncated (non-terminal).
pub fn build_rl_ship_jumped(entity: Entity, agent: &RLAgent) -> RLShipJumped {
    RLShipJumped {
        entity,
        entity_id: entity.to_bits(),
        personality: agent.personality.clone(),
        initial_hidden: agent.segment_initial_hidden.clone(),
        transitions: agent.segment_buffer.clone(),
    }
}
