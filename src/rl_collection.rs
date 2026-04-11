/// RL trajectory collection plugin.
///
/// Adds the `RLAgent` component to AI ships and collects (observation, action, reward)
/// transitions. Completed segments are sent via an mpsc channel to a trainer thread.
///
/// # Modes
/// - `AIPlayMode::BehavioralCloning`: `rl_step` computes rule-based actions for `RLAgent` ships
///   and records (obs, action) pairs via `BCSender`. `repeat_actions` drives the ships every frame.
/// - `AIPlayMode::RLControl`: `rl_step` performs model inference and records full
///   (obs, action, reward, done) transitions via `RLSender`. `repeat_actions` drives the ships.
///
/// In both modes, `classic_ai_control` only runs for non-RLAgent ships.
///
/// # Action space
/// Factored discrete: (turn_idx, thrust_idx, fire_primary, fire_secondary, target_idx)
///   - turn_idx:      0=left, 1=straight, 2=right
///   - thrust_idx:    0=no-thrust, 1=thrust
///   - fire_primary:  0=no, 1=yes
///   - fire_secondary: 0=no, 1=yes  (which weapon fires is chosen deterministically)
///   - target_idx:    0..N_OBJECTS-1 = entity slot, N_OBJECTS = "no target"
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

use crate::CurrentStarSystem;
use crate::ai_ships::{AIShip, compute_ai_action};
use crate::asteroids::{Asteroid, AsteroidField};
use crate::item_universe::ItemUniverse;
use crate::model::{
    self, HIDDEN_DIM, InferenceNet, N_OBJECTS, OBJECT_INPUT_DIM, POLICY_OUTPUT_DIM, RLInner,
    RLResource, SELF_INPUT_DIM, TrainBackend, split_obs, training_net_to_bytes,
};
use crate::pickups::Pickup;
use crate::planets::Planet;
use crate::rl_obs::{
    self, AsteroidSlotData, CoreSlotData, DETECTION_RADIUS, DiscreteAction, EntityKind,
    EntitySlotData, K_ASTEROIDS, K_FRIENDLY_SHIPS, K_HOSTILE_SHIPS, K_OTHER_PROJECTILES, K_PICKUPS,
    K_PLANETS, ObsInput, PickupSlotData, PlanetSlotData, ProjectileSlotData, SELF_SIZE, SLOT_SIZE,
    ShipSlotData,
};
use crate::ship::{Personality, Ship, ShipCommand, ShipHostility, Target};
use crate::weapons::{FireCommand, GuidedMissile, Projectile, Tracer, TracerSlots};
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

const OVERRIDE_TRADER_NAV_TARGET: bool = false;

// ---------------------------------------------------------------------------
// Types sent to the trainer thread
// ---------------------------------------------------------------------------

/// A single (obs, action, rewards, done) tuple.
#[derive(Clone)]
pub struct Transition {
    pub obs: Vec<f32>,
    /// Flat projectile features `[K_PROJECTILES × PROJ_SLOT_SIZE]`.
    pub proj_obs: Vec<f32>,
    pub action: DiscreteAction,
    /// Per-reward-type rewards, indexed by `consts::REWARD_*`.
    pub rewards: [f32; crate::consts::N_REWARD_TYPES],
    /// True when the ship died at this step (terminal state).
    pub done: bool,
    /// Sum of per-head log π(a|s) at the time the action was sampled.
    /// `0.0` for rule-based actions (BC mode).
    pub log_prob: f32,
}

/// A fixed-length segment of transitions, ready for PPO / recurrent-PPO training.
pub struct Segment {
    /// Entity bits — stable ID for logging and debugging.
    pub entity_id: u64,
    pub personality: Personality,
    /// LSTM/GRU hidden state at the start of this segment (zeros on first segment).
    pub initial_hidden: Vec<f32>,
    pub transitions: Vec<Transition>,
    /// None when the last transition is terminal; `Some([V_1(s_T), ..])` when truncated.
    pub bootstrap_value: Option<[f32; crate::consts::N_REWARD_TYPES]>,
}

/// A (observation, action) pair for behavioural-cloning pre-training.
///
/// `obs` holds the flat `OBS_DIM` observation vector. The training thread
/// splits it into self-features and entity-features via [`model::split_obs`].
pub struct BCTransition {
    pub obs: Vec<f32>,
    /// Flat projectile features `[K_PROJECTILES × PROJ_SLOT_SIZE]`.
    pub proj_obs: Vec<f32>,
    pub action: DiscreteAction,
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// Sender side of the PPO trajectory channel.
#[derive(Resource)]
pub struct RLSender(pub mpsc::SyncSender<Segment>);

/// Sender side of the BC data channel.
#[derive(Resource)]
pub struct BCSender(pub mpsc::SyncSender<BCTransition>);

/// Controls whether AI ships are driven by the rule-based system (for BC collection)
/// or by the RL policy.
#[derive(Resource, Default, PartialEq, Eq, Clone, Copy, Debug)]
pub enum AIPlayMode {
    /// `rl_step` computes rule-based actions for RLAgent ships; `repeat_actions`
    /// drives them. Observations and actions are recorded for BC training.
    #[default]
    BehavioralCloning,
    /// RLAgent ships are controlled by the RL policy (currently rule-based stub).
    RLControl,
}

/// 4 Hz decision-step timer.
#[derive(Resource)]
struct RLStepTimer(Timer);

/// Engagement shaping ramp: linearly increases from 0 to 1 over RAMP_DURATION
/// seconds of game time, preventing sudden reward distribution changes.
#[derive(Resource)]
struct EngagementRamp {
    elapsed: f32,
}

/// Duration (in game-time seconds) over which engagement rewards ramp from 0→1.
/// TODO: Remove this ramp once training has stabilised with the new engagement
/// rewards. Set `engagement_ramp` to a constant 1.0 and delete this resource.
const ENGAGEMENT_RAMP_DURATION: f32 = 7200.0; // 2 hours of game time

impl EngagementRamp {
    fn factor(&self) -> f32 {
        (self.elapsed / ENGAGEMENT_RAMP_DURATION).clamp(0.0, 1.0)
    }
}

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
    /// Index into `consts::REWARD_TYPE_NAMES` identifying the reward channel.
    pub reward_type: usize,
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
    /// Projectile observation from the previous decision step.
    pub last_proj_obs: Option<Vec<f32>>,
    /// Action chosen at the previous decision step.
    pub last_action: Option<DiscreteAction>,
    /// Log-probability of the action under the policy at sampling time.
    pub last_log_prob: f32,
    /// Per-type accumulated rewards since the last decision step.
    pub accumulated_rewards: [f32; crate::consts::N_REWARD_TYPES],
    /// Transitions collected since the last segment flush.
    pub segment_buffer: Vec<Transition>,
    /// Hidden state at the start of the current segment (for BPTT).
    pub segment_initial_hidden: Vec<f32>,
    /// Number of decision steps since the last segment flush.
    pub steps_since_flush: usize,
    /// Maps each observation entity slot to the `Target` it represents.
    /// Length = `N_OBJECTS`: slot 0 = current target, slots 1.. = nearby buckets.
    /// `None` for empty/unused slots.
    pub slot_targets: Vec<Option<Target>>,
}

impl RLAgent {
    pub fn new(personality: Personality) -> Self {
        Self {
            personality,
            hidden_state: vec![0.0; 64], // placeholder size
            last_obs: None,
            last_proj_obs: None,
            last_action: None,
            last_log_prob: 0.0,
            accumulated_rewards: [0.0; crate::consts::N_REWARD_TYPES],
            segment_buffer: Vec::with_capacity(RL_SEGMENT_LEN),
            segment_initial_hidden: vec![0.0; 64],
            steps_since_flush: 0,
            slot_targets: vec![None; model::N_OBJECTS],
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

        // Make the experiment directory available to game-thread systems.
        app.insert_resource(crate::experiments::ExperimentDir {
            run_dir: experiment.run_dir.clone(),
            is_fresh: experiment.is_fresh,
        });

        // Spawn training threads only for the modes that need them.
        match self.mode {
            crate::AppMode::BCTraining => {
                spawn_bc_training_thread(bc_rx, inference_net_arc, experiment);
                drop(rl_rx);
            }
            crate::AppMode::RLTraining => {
                crate::ppo::spawn_ppo_training_thread(rl_rx, inference_net_arc, experiment);
                drop(bc_rx);
            }
            crate::AppMode::Inference => {
                // Load the checkpoint into the inference net so the game thread
                // uses trained weights rather than a random initialisation.
                if !experiment.is_fresh {
                    let checkpoint_path = experiment.policy_checkpoint_path();
                    if let Some(loaded) = model::load_inference_net(&checkpoint_path) {
                        *inference_net_arc.lock().unwrap() = loaded;
                        println!("[inference] Loaded policy from {checkpoint_path}");
                    } else {
                        eprintln!(
                            "[inference] WARNING: no checkpoint found at {checkpoint_path} \
                             — running with random weights!"
                        );
                    }
                }
                drop(bc_rx);
                drop(rl_rx);
            }
            crate::AppMode::Classic => {
                // No training — channels have no consumer; data is silently dropped.
                drop(bc_rx);
                drop(rl_rx);
            }
        }

        // Map the app mode to the in-game AI control mode.
        let ai_play_mode = match self.mode {
            crate::AppMode::Classic | crate::AppMode::BCTraining => AIPlayMode::BehavioralCloning,
            crate::AppMode::Inference | crate::AppMode::RLTraining => AIPlayMode::RLControl,
        };

        app.insert_resource(RLSender(rl_tx))
            .insert_resource(BCSender(bc_tx))
            .insert_resource(rl_resource)
            .insert_resource(ai_play_mode)
            .insert_resource(RLStepTimer(Timer::from_seconds(
                RL_STEP_PERIOD,
                TimerMode::Repeating,
            )))
            .insert_resource(EngagementRamp { elapsed: 0.0 })
            .add_message::<RLReward>()
            .add_message::<RLShipDied>()
            .add_message::<RLShipJumped>()
            .add_systems(
                Update,
                (
                    accumulate_rewards,
                    (rl_step, repeat_actions).chain(),
                    handle_rl_ship_died,
                    handle_rl_ship_jumped,
                )
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
            if ev.reward_type < crate::consts::N_REWARD_TYPES {
                agent.accumulated_rewards[ev.reward_type] += ev.reward;
            }
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
    mut engagement_ramp: ResMut<EngagementRamp>,
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
        &crate::ship::Distressed,
        &TracerSlots,
    )>,
    // Queries for nearby entities (non-overlapping with the agent query).
    all_positions: Query<&Position>,
    all_velocities: Query<&LinearVelocity>,
    entity_queries: (
        Query<(Entity, &Planet)>,
        Query<(Entity, &Asteroid)>,
        Query<&AsteroidField>,
        Query<(Entity, &Pickup)>,
        // Ship data for non-RLAgent ships (disjoint with the mutable `agents` query).
        Query<(Entity, &Ship, &ShipHostility), (With<AIShip>, Without<RLAgent>)>,
    ),
    // ShipHostility for ALL ships — used to bucket nearby ships as hostile/friendly.
    // Only reads ShipHostility (not Ship), so disjoint with the `agents` query.
    all_ship_factions: Query<&ShipHostility, With<Ship>>,
    all_distressed: Query<&crate::ship::Distressed>,
    all_transforms: Query<&Transform>,
    projectile_query: Query<(&Projectile, Option<&GuidedMissile>)>,
    spatial_query: SpatialQuery,
    resources: (
        Res<ItemUniverse>,
        Res<CurrentStarSystem>,
        Res<AIPlayMode>,
        Res<RLSender>,
        Res<BCSender>,
        Res<RLResource>,
    ),
) {
    let (planet_query, asteroid_query, asteroid_field_query, pickup_query, ship_query) =
        &entity_queries;
    let (item_universe, current_system, mode, rl_sender, bc_sender, rl_resource) = &resources;
    if !timer.0.tick(time.delta()).just_finished() {
        return;
    }
    // Tick the engagement ramp (accumulates game time).
    engagement_ramp.elapsed += RL_STEP_PERIOD;
    let eng_ramp = engagement_ramp.factor();

    // ── Sub-function 1: build observations for all agents ──────────────────
    let obs_data: Vec<(Entity, Vec<f32>, Vec<f32>, Vec<Option<Target>>)> = build_all_observations(
        &agents,
        &all_positions,
        &all_velocities,
        &all_transforms,
        planet_query,
        asteroid_query,
        asteroid_field_query,
        pickup_query,
        ship_query,
        &all_ship_factions,
        &all_distressed,
        &projectile_query,
        &spatial_query,
        item_universe,
        current_system,
    );

    // ── Sub-function 2: run batched model inference (RLControl mode only) ──
    // In BehavioralCloning mode the rule-based action from compute_ai_action
    // is used instead, so we skip the model entirely.
    let model_actions: Option<Vec<(DiscreteAction, f32)>> =
        if **mode == AIPlayMode::RLControl && !obs_data.is_empty() {
            let batch_size = obs_data.len();
            let mut self_flat = Vec::with_capacity(batch_size * model::SELF_INPUT_DIM);
            let mut obj_flat = Vec::with_capacity(batch_size * model::ENTITIES_FLAT_DIM);
            let mut proj_flat = Vec::with_capacity(batch_size * model::PROJECTILES_FLAT_DIM);
            for (_, obs, proj_obs, _) in &obs_data {
                let (s, o) = model::split_obs(obs);
                self_flat.extend_from_slice(s);
                obj_flat.extend_from_slice(o);
                proj_flat.extend_from_slice(proj_obs);
            }
            let inference = rl_resource.inference_net.lock().unwrap();
            let (action_logits, nav_target_logits, wep_target_logits) =
                inference.run_inference(self_flat, obj_flat, proj_flat, batch_size);
            let mut rng = rand::thread_rng();
            let actions = action_logits
                .chunks(model::POLICY_OUTPUT_DIM)
                .zip(nav_target_logits.chunks(model::TARGET_OUTPUT_DIM))
                .zip(wep_target_logits.chunks(model::TARGET_OUTPUT_DIM))
                .map(|((al, ntl), wtl)| model::sample_discrete_action(al, ntl, wtl, &mut rng))
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
        item_universe,
        rl_sender,
        bc_sender,
        mode,
        eng_ramp,
    );
}

/// Re-emit the last chosen action every `Update` frame so the ship continues
/// moving between 4 Hz decision steps.
/// Active in both `RLControl` and `BehavioralCloning` modes — `classic_ai_control`
/// only drives non-RLAgent ships.
fn repeat_actions(
    agents: Query<(Entity, &RLAgent, &Ship)>,
    mut ship_writer: MessageWriter<ShipCommand>,
    mut fire_writer: MessageWriter<FireCommand>,
) {
    for (entity, agent, ship) in &agents {
        let Some(action) = agent.last_action else {
            continue;
        };
        let (turn_idx, thrust_idx, fire_primary, fire_secondary, _nav_idx, _wep_idx) = action;
        let (thrust, turn) = rl_obs::discrete_to_controls(thrust_idx, turn_idx);
        ship_writer.write(ShipCommand {
            entity,
            thrust,
            turn,
            reverse: 0.0,
        });
        // Weapons fire at the weapons_target, not the nav_target.
        let wep_entity = ship.weapons_target.as_ref().map(|t| t.get_entity());
        let wep_is_ship = matches!(ship.weapons_target, Some(Target::Ship(_)));
        if fire_primary == 1 {
            for weapon_type in ship.weapon_systems.primary.keys() {
                fire_writer.write(FireCommand {
                    ship: entity,
                    weapon_type: weapon_type.clone(),
                    target: wep_entity,
                });
            }
        }
        if fire_secondary == 1 {
            if let Some((wtype, _)) = rl_obs::select_secondary_weapon(ship, wep_is_ship) {
                fire_writer.write(FireCommand {
                    ship: entity,
                    weapon_type: wtype.to_string(),
                    target: wep_entity,
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
/// Returns a vec of `(entity, observation_vec, slot_targets)` tuples.
/// `slot_targets` maps each entity slot index to the `Target` it represents
/// (or `None` for empty slots), enabling the policy's target selection action.
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
        &crate::ship::Distressed,
        &TracerSlots,
    )>,
    all_positions: &Query<&Position>,
    all_velocities: &Query<&LinearVelocity>,
    all_transforms: &Query<&Transform>,
    planet_query: &Query<(Entity, &Planet)>,
    asteroid_query: &Query<(Entity, &Asteroid)>,
    asteroid_field_query: &Query<&AsteroidField>,
    pickup_query: &Query<(Entity, &Pickup)>,
    ship_query: &Query<(Entity, &Ship, &ShipHostility), (With<AIShip>, Without<RLAgent>)>,
    all_ship_factions: &Query<&ShipHostility, With<Ship>>,
    all_distressed: &Query<&crate::ship::Distressed>,
    projectile_query: &Query<(&Projectile, Option<&GuidedMissile>)>,
    spatial_query: &SpatialQuery,
    item_universe: &ItemUniverse,
    current_system: &CurrentStarSystem,
) -> Vec<(Entity, Vec<f32>, Vec<f32>, Vec<Option<Target>>)> {
    let mut results = Vec::new();

    for (
        entity,
        agent,
        ship,
        pos,
        vel,
        ang_vel,
        max_speed,
        transform,
        self_hostility,
        self_distressed,
        self_tracer_slots,
    ) in agents.iter()
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
                let hit_pos = all_positions
                    .get(hit)
                    .map(|p| p.0)
                    .or_else(|_| all_transforms.get(hit).map(|t| t.translation.truncate()))
                    .ok()?;
                Some((hit, (hit_pos - pos.0).length_squared()))
            })
            .collect();

        // Helper: compute ego-frame core data for any entity.
        let make_core = |e: Entity, entity_type: u8| -> CoreSlotData {
            let entity_pos = all_positions
                .get(e)
                .map(|p| p.0)
                .or_else(|_| all_transforms.get(e).map(|t| t.translation.truncate()))
                .unwrap_or(pos.0);
            let world_offset = entity_pos - pos.0;
            let world_rel_vel = all_velocities
                .get(e)
                .map(|v| v.0 - vel.0)
                .unwrap_or(Vec2::ZERO);
            CoreSlotData {
                rel_pos: rl_obs::rotate_to_ego([world_offset.x, world_offset.y], sin_a, cos_a),
                rel_vel: rl_obs::rotate_to_ego([world_rel_vel.x, world_rel_vel.y], sin_a, cos_a),
                entity_type,
            }
        };

        // Precompute planet trade data for this system.
        let system_name = &current_system.0;
        let planet_margins = item_universe.planet_best_margin.get(system_name);

        // Compute asteroid field expected values for this system.
        let field_values = item_universe.asteroid_field_expected_value.get(system_name);

        // Cargo sale value helper: total value of selling ship's cargo at a planet.
        let cargo_sale_value = |planet_name: &str| -> f32 {
            let planet_data = item_universe
                .star_systems
                .get(system_name)
                .and_then(|sys| sys.planets.get(planet_name));
            let Some(pd) = planet_data else { return 0.0 };
            ship.cargo
                .iter()
                .map(|(commodity, &qty)| {
                    pd.commodities.get(commodity).copied().unwrap_or(0) as f32 * qty as f32
                })
                .sum()
        };

        // Buckets — sort by distance within each type.
        // Planets: include ALL planets in the system (not just nearby), since
        // there are few and they're critical for navigation/trading.
        // Note: static RigidBody entities may only have Transform, not Position.
        let all_transforms: &Query<&Transform> = all_transforms;
        let mut planets: Vec<(Entity, f32)> = planet_query
            .iter()
            .filter_map(|(e, _)| {
                // Try Position first (dynamic bodies), fall back to Transform.
                let planet_pos = all_positions
                    .get(e)
                    .map(|p| p.0)
                    .or_else(|_| all_transforms.get(e).map(|t| t.translation.truncate()))
                    .ok()?;
                Some((e, (planet_pos - pos.0).length()))
            })
            .collect();
        planets.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut asteroids: Vec<(Entity, f32)> = nearby_hits
            .iter()
            .filter(|(e, _)| asteroid_query.get(*e).is_ok())
            .cloned()
            .collect();
        asteroids.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut hostile_ships: Vec<(Entity, f32)> = nearby_hits
            .iter()
            .filter(|(e, _)| {
                all_ship_factions
                    .get(*e)
                    .map(|hostility| ship.should_engage(hostility))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        hostile_ships.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut friendly_ships: Vec<(Entity, f32)> = nearby_hits
            .iter()
            .filter(|(e, _)| {
                all_ship_factions
                    .get(*e)
                    .map(|hostility| !ship.should_engage(hostility))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        friendly_ships.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut pickups: Vec<(Entity, f32)> = nearby_hits
            .iter()
            .filter(|(e, _)| pickup_query.get(*e).is_ok())
            .cloned()
            .collect();
        pickups.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Build EntitySlotData for each bucket.
        let mut nearby_planets: Vec<EntitySlotData> = planets
            .iter()
            .take(K_PLANETS)
            .map(|(e, _)| {
                let core = make_core(*e, 2); // Planet
                let planet_name = planet_query
                    .get(*e)
                    .map(|(_, p)| p.0.as_str())
                    .unwrap_or("");
                let csv = cargo_sale_value(planet_name);
                let has_ammo = item_universe
                    .planet_has_ammo_for
                    .get(planet_name)
                    .map(|set| set.contains(&ship.ship_type))
                    .unwrap_or(false);
                let margin = planet_margins
                    .and_then(|m| m.get(planet_name))
                    .copied()
                    .unwrap_or(0.0) as f32;
                EntitySlotData {
                    core,
                    kind: EntityKind::Planet(PlanetSlotData {
                        cargo_sale_value: csv,
                        has_ammo: if has_ammo { 1.0 } else { 0.0 },
                        commodity_margin: margin,
                    }),
                    value: csv,
                    is_nav_target: false,
                    is_weapons_target: false,
                }
            })
            .collect();

        let mut nearby_asteroids: Vec<EntitySlotData> = asteroids
            .iter()
            .take(K_ASTEROIDS)
            .map(|(e, _)| {
                let core = make_core(*e, 1); // Asteroid
                let (size, ev) = asteroid_query
                    .get(*e)
                    .map(|(_, ast)| {
                        // Expected value: field EV * expected qty from this asteroid.
                        let field_ev = asteroid_field_query
                            .get(ast.field)
                            .ok()
                            .and_then(|_| {
                                // Find field index in system config to look up precomputed EV.
                                // Fallback: use global average if not found.
                                field_values.and_then(|fv| fv.first().copied())
                            })
                            .unwrap_or(0.0);
                        let expected_qty = (ast.size * 0.2 * 0.5).max(0.5); // rough E[qty]
                        (ast.size, (field_ev * expected_qty as f64) as f32)
                    })
                    .unwrap_or((0.0, 0.0));
                let col_ind =
                    rl_obs::collision_indicator(core.rel_pos, core.rel_vel, ship.data.radius);
                EntitySlotData {
                    core,
                    kind: EntityKind::Asteroid(AsteroidSlotData {
                        size,
                        value: ev,
                        collision_indicator: col_ind,
                    }),
                    value: ev,
                    is_nav_target: false,
                    is_weapons_target: false,
                }
            })
            .collect();

        // Build ship slot data — tries ship_query (non-RLAgent) first, then
        // falls back to the agents query (RLAgent ships appearing as nearby entities).
        let make_ship_slot = |e: Entity| -> EntitySlotData {
            let core = make_core(e, 0); // Ship
            // Get the other ship's data from either query.
            let other_ship_ref: Option<(&Ship, &ShipHostility)> =
                ship_query.get(e).map(|(_, s, h)| (s, h)).ok().or_else(|| {
                    agents
                        .get(e)
                        .map(|(_, _, s, _, _, _, _, _, h, _, _)| (s as &Ship, h as &ShipHostility))
                        .ok()
                });
            let (ship_data, value) = other_ship_ref
                .map(|(other_ship, other_hostility)| {
                    let is_hostile =
                        if other_ship.should_engage(&ShipHostility(ship.enemies.clone())) {
                            1.0
                        } else {
                            -1.0
                        };
                    let should_engage = if ship.should_engage(other_hostility) {
                        1.0
                    } else {
                        -1.0
                    };
                    let data = item_universe.ships.get(&other_ship.ship_type);
                    let other_distressed = all_distressed.get(e).map(|d| d.level).unwrap_or(0.0);
                    let sd = ShipSlotData {
                        max_health: data.map(|d| d.max_health as f32).unwrap_or(0.0),
                        health: other_ship.health as f32,
                        max_speed: data.map(|d| d.max_speed as f32).unwrap_or(0.0),
                        torque: data.map(|d| d.torque as f32).unwrap_or(0.0),
                        is_hostile,
                        should_engage,
                        personality: data.map(|d| d.personality.clone()).unwrap_or_default(),
                        distressed: other_distressed,
                    };
                    (sd, 0.0_f32)
                })
                .unwrap_or_default();
            EntitySlotData {
                core,
                kind: EntityKind::Ship(ship_data),
                value,
                is_nav_target: false,
                is_weapons_target: false,
            }
        };

        let mut nearby_hostile_ships: Vec<EntitySlotData> = hostile_ships
            .iter()
            .take(K_HOSTILE_SHIPS)
            .map(|(e, _)| make_ship_slot(*e))
            .collect();
        let mut nearby_friendly_ships: Vec<EntitySlotData> = friendly_ships
            .iter()
            .take(K_FRIENDLY_SHIPS)
            .map(|(e, _)| make_ship_slot(*e))
            .collect();

        let mut nearby_pickups: Vec<EntitySlotData> = pickups
            .iter()
            .take(K_PICKUPS)
            .map(|(e, _)| {
                let core = make_core(*e, 3); // Pickup
                let pv = pickup_query
                    .get(*e)
                    .map(|(_, p)| {
                        let avg = item_universe
                            .global_average_price
                            .get(&p.commodity)
                            .copied()
                            .unwrap_or(0.0);
                        (avg * p.quantity as f64) as f32
                    })
                    .unwrap_or(0.0);
                EntitySlotData {
                    core,
                    kind: EntityKind::Pickup(PickupSlotData { value: pv }),
                    value: pv,
                    is_nav_target: false,
                    is_weapons_target: false,
                }
            })
            .collect();

        // Build the unified entity_slots vec and slot_targets mapping.
        //
        // Phase 1: add up to K entities of each type (no padding).
        // Phase 2: fill remaining capacity with the personality's preferred types.
        let nav_entity = ship.nav_target.as_ref().map(|t| t.get_entity());
        let wep_entity = ship.weapons_target.as_ref().map(|t| t.get_entity());
        let mut entity_slots = Vec::with_capacity(model::N_OBJECTS);
        let mut slot_targets: Vec<Option<Target>> = Vec::with_capacity(model::N_OBJECTS);

        // Helper: push up to `max_count` entities, returns how many were added.
        let mut push_up_to = |slot_data: &mut [EntitySlotData],
                              entity_list: &[(Entity, f32)],
                              max_count: usize,
                              make_target: fn(Entity) -> Target|
         -> usize {
            let cap = max_count.min(entity_list.len());
            let room = model::N_OBJECTS - entity_slots.len();
            let n = cap.min(room);
            for i in 0..n {
                let (e, _) = entity_list[i];
                if let Some(slot) = slot_data.get_mut(i) {
                    if nav_entity == Some(e) {
                        slot.is_nav_target = true;
                    }
                    if wep_entity == Some(e) {
                        slot.is_weapons_target = true;
                    }
                    entity_slots.push(slot.clone());
                    slot_targets.push(Some(make_target(e)));
                }
            }
            n
        };

        // Phase 1: each type gets up to its base allocation.
        let n_planets = push_up_to(&mut nearby_planets, &planets, K_PLANETS, Target::Planet);
        let n_asteroids = push_up_to(
            &mut nearby_asteroids,
            &asteroids,
            K_ASTEROIDS,
            Target::Asteroid,
        );
        let n_hostile = push_up_to(
            &mut nearby_hostile_ships,
            &hostile_ships,
            K_HOSTILE_SHIPS,
            Target::Ship,
        );
        let n_friendly = push_up_to(
            &mut nearby_friendly_ships,
            &friendly_ships,
            K_FRIENDLY_SHIPS,
            Target::Ship,
        );
        let n_pickups = push_up_to(&mut nearby_pickups, &pickups, K_PICKUPS, Target::Pickup);

        // Phase 2: fill remaining slots with personality-preferred entities.
        // Each call continues from where phase 1 left off (skip already-added).
        // Use usize::MAX as max_count since push_up_to clamps to remaining room.
        match agent.personality {
            Personality::Trader => {
                push_up_to(
                    &mut nearby_planets[n_planets..],
                    &planets[n_planets..],
                    usize::MAX,
                    Target::Planet,
                );
            }
            Personality::Miner => {
                push_up_to(
                    &mut nearby_asteroids[n_asteroids..],
                    &asteroids[n_asteroids..],
                    usize::MAX,
                    Target::Asteroid,
                );
                push_up_to(
                    &mut nearby_planets[n_planets..],
                    &planets[n_planets..],
                    usize::MAX,
                    Target::Planet,
                );
            }
            Personality::Fighter => {
                push_up_to(
                    &mut nearby_hostile_ships[n_hostile..],
                    &hostile_ships[n_hostile..],
                    usize::MAX,
                    Target::Ship,
                );
                push_up_to(
                    &mut nearby_friendly_ships[n_friendly..],
                    &friendly_ships[n_friendly..],
                    usize::MAX,
                    Target::Ship,
                );
            }
        }

        // Pad slot_targets to N_OBJECTS (entity_slots may be shorter — the
        // encoder handles padding with zero slots).
        while slot_targets.len() < model::N_OBJECTS {
            slot_targets.push(None);
        }
        debug_assert!(entity_slots.len() <= model::N_OBJECTS);
        debug_assert_eq!(slot_targets.len(), model::N_OBJECTS);

        // Primary weapon speed / range for fire-lead angle in the observation.
        let (primary_weapon_speed, primary_weapon_range) = ship
            .weapon_systems
            .primary
            .values()
            .next()
            .map(|ws| (ws.weapon.speed, ws.weapon.range()))
            .unwrap_or((0.0, 0.0));

        let credit_scale = item_universe
            .ship_credit_scale
            .get(&ship.ship_type)
            .copied()
            .unwrap_or(1.0);

        // -- Projectile slots -------------------------------------------------
        // Helper to build a ProjectileSlotData from a projectile entity.
        let make_proj_slot = |e: Entity, is_ours: f32| -> Option<ProjectileSlotData> {
            let (proj, guided) = projectile_query.get(e).ok()?;
            let p = all_positions.get(e).ok()?;
            let v = all_velocities.get(e).ok()?;
            let world_offset = [p.x - pos.x, p.y - pos.y];
            let world_rel_vel = [v.x - vel.x, v.y - vel.y];
            let weapon = item_universe.weapons.get(&proj.weapon_type);
            Some(ProjectileSlotData {
                rel_pos: rl_obs::rotate_to_ego(world_offset, sin_a, cos_a),
                rel_vel: rl_obs::rotate_to_ego(world_rel_vel, sin_a, cos_a),
                is_ours,
                is_guided: if guided.is_some() { 1.0 } else { 0.0 },
                speed: weapon.map(|w| w.speed).unwrap_or(0.0),
                damage: weapon.map(|w| w.damage as f32).unwrap_or(0.0),
                lifetime_remaining: proj.lifetime,
                lifetime_max: weapon.map(|w| w.lifetime).unwrap_or(1.0),
                target_radius: ship.data.radius,
            })
        };

        // Other ships' projectiles: nearest K_OTHER_PROJECTILES.
        let proj_filter =
            SpatialQueryFilter::from_mask([GameLayer::Weapon]).with_excluded_entities([entity]);
        let mut nearby_other_projs: Vec<(Entity, f32)> = spatial_query
            .shape_intersections(
                &Collider::circle(DETECTION_RADIUS),
                pos.0,
                0.0,
                &proj_filter,
            )
            .into_iter()
            .filter_map(|hit| {
                let proj = projectile_query.get(hit).ok()?;
                if proj.0.owner == entity {
                    return None;
                }
                let p = all_positions.get(hit).ok()?;
                Some((hit, (p.0 - pos.0).length_squared()))
            })
            .collect();
        nearby_other_projs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let other_projectile_slots: Vec<ProjectileSlotData> = nearby_other_projs
            .iter()
            .take(K_OTHER_PROJECTILES)
            .filter_map(|(e, _)| make_proj_slot(*e, 0.0))
            .collect();

        // Own projectiles: tracer-tagged projectiles in their assigned slot order.
        let mut own_projectile_slots: Vec<ProjectileSlotData> = Vec::new();
        for slot_entry in &self_tracer_slots.slots {
            if let Some(proj_entity) = slot_entry {
                if let Some(slot_data) = make_proj_slot(*proj_entity, 1.0) {
                    own_projectile_slots.push(slot_data);
                }
            }
        }

        let obs_input = ObsInput {
            personality: &agent.personality,
            ship: &ship,
            velocity: [vel.x, vel.y],
            angular_velocity: ang_vel.0,
            ship_heading: heading,
            entity_slots,
            primary_weapon_speed,
            primary_weapon_range,
            credit_scale,
            distressed: self_distressed.level,
            other_projectile_slots,
            own_projectile_slots,
        };
        let obs = rl_obs::encode_observation(&obs_input);
        let max_speed = ship.data.max_speed.max(f32::EPSILON);
        let proj_obs = rl_obs::encode_projectiles(
            &obs_input.other_projectile_slots,
            &obs_input.own_projectile_slots,
            max_speed,
        );

        results.push((entity, obs, proj_obs, slot_targets));
    }

    results
}

/// Choose the best target slot index for a ship, mirroring the priority logic
/// in `classic_ai_target_selection`.
///
/// Reads entity types and hostility directly from the observation's slot
/// features (via named offsets), so this function does not depend on the
/// order of entity buckets.
///
/// Priority:
/// 1. Traders with cargo → planet with highest value
/// 2. All personalities → nearest pickup
/// 3. Miner → nearest asteroid → nearest planet
///    Fighter → nearest hostile ship → nearest planet
///    Trader → nearest planet
///
/// Returns the slot index, or `N_OBJECTS` for "no target".
fn choose_target_slot(personality: &Personality, has_cargo: bool, obs: &[f32]) -> u8 {
    use rl_obs::*;
    let no_target = model::N_OBJECTS as u8;
    let n = model::N_OBJECTS;

    // Read slot features from the flat observation.
    let slot_base = |i: usize| SELF_SIZE + i * SLOT_SIZE;
    let is_present = |i: usize| obs[slot_base(i) + SLOT_IS_PRESENT] > 0.5;
    let type_onehot =
        |i: usize| &obs[slot_base(i) + SLOT_TYPE_ONEHOT..slot_base(i) + SLOT_TYPE_ONEHOT + 4];
    let is_ship = |i: usize| type_onehot(i)[0] > 0.5;
    let is_asteroid = |i: usize| type_onehot(i)[1] > 0.5;
    let is_planet = |i: usize| type_onehot(i)[2] > 0.5;
    let is_pickup = |i: usize| type_onehot(i)[3] > 0.5;
    let should_engage = |i: usize| obs[slot_base(i) + SLOT_TYPE_SPECIFIC + 5] > 0.5;
    let value = |i: usize| obs[slot_base(i) + SLOT_VALUE];

    // Helper: find the first present slot matching a predicate.
    let first_matching = |pred: &dyn Fn(usize) -> bool| -> Option<u8> {
        (0..n).find(|&i| is_present(i) && pred(i)).map(|i| i as u8)
    };

    // 1. Traders with cargo → planet with highest value (= cargo_sale_value).
    if has_cargo && matches!(personality, Personality::Trader) {
        let best_planet = (0..n)
            .filter(|&i| is_present(i) && is_planet(i))
            .max_by(|&a, &b| {
                value(a)
                    .partial_cmp(&value(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        if let Some(idx) = best_planet {
            return idx as u8;
        }
    }

    // 2. All personalities: nearest pickup first.
    if let Some(idx) = first_matching(&|i| is_pickup(i)) {
        return idx;
    }

    // 3. Personality-based fallback.
    match personality {
        Personality::Miner => first_matching(&|i| is_asteroid(i))
            .or_else(|| first_matching(&|i| is_planet(i)))
            .unwrap_or(no_target),
        Personality::Fighter => first_matching(&|i| is_ship(i) && should_engage(i))
            .or_else(|| first_matching(&|i| is_planet(i)))
            .unwrap_or(no_target),
        Personality::Trader => first_matching(&|i| is_planet(i)).unwrap_or(no_target),
    }
}

/// Compute the per-step nav-target reward from the flat observation vector.
///
/// Rewards ships for having the correct navigation target for their personality,
/// gated by health and distress level.
fn compute_nav_target_reward(personality: &Personality, obs: &[f32]) -> (f32, bool) {
    use crate::consts::*;
    use rl_obs::*;

    let health_frac = obs[0];
    let cargo_frac = obs[5];
    let distressed = obs[8]; // distressed level (index 8 in self-state)

    if health_frac < 0.3 || distressed > 0.5 {
        return (0.0, false); // too damaged/distressed to get nav reward
    }

    let slot_base = |i: usize| SELF_SIZE + i * SLOT_SIZE;
    let n = model::N_OBJECTS;

    for i in 0..n {
        let base = slot_base(i);
        if obs[base + SLOT_IS_PRESENT] < 0.5 {
            continue;
        }
        if obs[base + SLOT_IS_NAV_TARGET] < 0.5 {
            continue;
        }

        let is_asteroid = obs[base + SLOT_TYPE_ONEHOT + 1] > 0.5;
        let is_planet = obs[base + SLOT_TYPE_ONEHOT + 2] > 0.5;
        let is_pickup = obs[base + SLOT_TYPE_ONEHOT + 3] > 0.5;
        let is_ship = obs[base + SLOT_TYPE_ONEHOT] > 0.5;

        let (raw, is_good) = match personality {
            Personality::Trader => {
                if is_planet {
                    (NAV_TARGET_TRADER_PLANET, true)
                } else {
                    (0.0, false)
                }
            }
            Personality::Miner => {
                if cargo_frac < 0.95 && (is_asteroid || is_pickup) {
                    (NAV_TARGET_MINER_RESOURCE, true)
                } else if cargo_frac >= 0.5 && is_planet {
                    (NAV_TARGET_MINER_PLANET * cargo_frac, true)
                } else {
                    (0.0, false)
                }
            }
            Personality::Fighter => {
                if is_ship || is_asteroid {
                    (NAV_TARGET_FIGHTER, true)
                } else {
                    (0.0, false)
                }
            }
        };
        return (raw, is_good);
    }
    (0.0, false)
}

/// Compute the per-step weapons-target reward from the flat observation vector.
///
/// Fighters: rewarded for targeting anything hostile or should_engage.
/// Miners/Traders: rewarded for targeting anything hostile (defensive only).
fn compute_weapons_target_reward(personality: &Personality, obs: &[f32]) -> f32 {
    use crate::consts::*;
    use rl_obs::*;

    let slot_base = |i: usize| SELF_SIZE + i * SLOT_SIZE;
    let n = model::N_OBJECTS;

    for i in 0..n {
        let base = slot_base(i);
        if obs[base + SLOT_IS_PRESENT] < 0.5 {
            continue;
        }
        if obs[base + SLOT_IS_WEAPONS_TARGET] < 0.5 {
            continue;
        }

        let is_hostile = obs[base + SLOT_TYPE_SPECIFIC + 4] > 0.5;
        let should_engage = obs[base + SLOT_TYPE_SPECIFIC + 5] > 0.5;

        let raw = match personality {
            Personality::Fighter => {
                if is_hostile || should_engage {
                    WEAPONS_TARGET_FIGHTER
                } else {
                    0.0
                }
            }
            Personality::Miner | Personality::Trader => {
                if is_hostile {
                    WEAPONS_TARGET_DEFENSIVE
                } else {
                    0.0
                }
            }
        };
        return raw;
    }
    0.0
}

/// Compute the planet approach + velocity matching + braking reward.
///
/// All inputs are in the ego-centric frame (from the observation vector).
/// - `rp`: rel_pos (ego frame, points from ship toward target)
/// - `rv`: rel_vel (ego frame, = target_vel − ship_vel)
/// - `vel_cos`, `vel_sin`: ship velocity direction in ego frame (from self-state)
/// - `ship_speed`: ship speed (from self-state speed_frac × max_speed, or abs value)
fn compute_planet_approach_reward(
    rp: [f32; 2],
    rv: [f32; 2],
    dist: f32,
    vel_cos: f32,
    vel_sin: f32,
    ship_speed: f32,
    is_thrusting: bool,
) -> f32 {
    use crate::consts::*;
    let mut r = 0.0_f32;

    // Closing speed: rate at which distance decreases.
    // d(dist)/dt = dot(rv, rp) / dist, so closing = −dot(rv, rp) / dist.
    let approach_cos = if dist > 1e-3 {
        let closing = -(rv[0] * rp[0] + rv[1] * rp[1]) / dist;
        // Normalise to [-1, 1] by dividing by ship speed.
        (closing / (ship_speed + 1e-3)).clamp(-1.0, 1.0)
    } else {
        0.0
    };

    // Approach reward: incentivises closing on planet outside threshold.
    let approach = 1.0
        + if dist > MOVEMENT_THRESHOLD_DIST {
            (approach_cos - 1.0) / 2.0 * (1.0 - (MOVEMENT_THRESHOLD_DIST / dist).min(1.0))
        } else {
            0.0
        };
    // Velocity matching: incentivises matching planet speed within threshold.
    let rel_speed = (rv[0] * rv[0] + rv[1] * rv[1]).sqrt();
    let vel_match = if dist < MOVEMENT_THRESHOLD_DIST {
        let frac = (MOVEMENT_THRESHOLD_DIST - dist) / MOVEMENT_THRESHOLD_DIST;
        MOVEMENT_VEL_SCALE / (rel_speed + MOVEMENT_VEL_SCALE) * frac
    } else {
        0.0
    };
    r += (approach + vel_match) * MOVEMENT_STEP_WEIGHT;

    // Braking: retrograde facing + thrust when close and approaching.
    if dist < BRAKING_THRESHOLD_DIST && approach_cos > 0.0 {
        if ship_speed < 30.0 {
            r += BRAKING_SLOW_REWARD * BRAKING_STEP_WEIGHT;
        }
        if ship_speed > 1.0 {
            // In ego frame, ship heading is (0, 1). Ship velocity direction
            // is (vel_sin, vel_cos). Retrograde = facing opposite to velocity:
            // retro_cos = dot((0,1), -(vel_sin, vel_cos) / speed) = -vel_cos / speed
            // But vel_cos is already cos(angle between heading and velocity),
            // so retro_cos = -vel_cos when normalised.
            let retro_cos = -vel_cos; // 1.0 = fully retrograde
            let retro_frac = ((retro_cos + 1.0) / 2.0).clamp(0.0, 1.0);
            r += BRAKING_RETROGRADE_REWARD * retro_frac * BRAKING_STEP_WEIGHT;
            if retro_cos > BRAKING_RETROGRADE_COS_THRESH && is_thrusting {
                r += BRAKING_THRUST_REWARD * BRAKING_STEP_WEIGHT;
            }
        }
    }
    r
}

/// Compute the full movement + engagement reward from the obs vector.
///
/// Purely obs-based — no ECS queries needed.
fn compute_movement_reward(obs: &[f32], last_action: DiscreteAction, max_speed: f32, engagement_ramp: f32) -> f32 {
    use crate::consts::*;
    use rl_obs::*;

    let slot_base = |i: usize| SELF_SIZE + i * SLOT_SIZE;

    // Find nav and weapons target slots.
    let mut nav_slot: Option<usize> = None;
    let mut wep_slot: Option<usize> = None;
    for i in 0..model::N_OBJECTS {
        let base = slot_base(i);
        if obs[base + SLOT_IS_PRESENT] < 0.5 {
            continue;
        }
        if obs[base + SLOT_IS_NAV_TARGET] > 0.5 {
            nav_slot = Some(i);
        }
        if obs[base + SLOT_IS_WEAPONS_TARGET] > 0.5 {
            wep_slot = Some(i);
        }
    }

    let Some(ni) = nav_slot else { return 0.0 };
    let nb = slot_base(ni);

    let is_planet = obs[nb + SLOT_TYPE_ONEHOT + 2] > 0.5;
    let is_ship = obs[nb + SLOT_TYPE_ONEHOT] > 0.5;
    let is_asteroid = obs[nb + SLOT_TYPE_ONEHOT + 1] > 0.5;
    let is_pickup = obs[nb + SLOT_TYPE_ONEHOT + 3] > 0.5;
    let is_combatable = is_ship || is_asteroid;
    let pursuit_ind = obs[nb + SLOT_PURSUIT_INDICATOR];
    let fire_ind = obs[nb + SLOT_FIRE_INDICATOR];
    let in_range = obs[nb + SLOT_IN_RANGE] > 0.5;
    let nav_and_wep_match = wep_slot == nav_slot;

    let rp = [obs[nb + SLOT_REL_POS], obs[nb + SLOT_REL_POS + 1]];
    let rv = [obs[nb + SLOT_REL_VEL], obs[nb + SLOT_REL_VEL + 1]];
    let dist = (rp[0] * rp[0] + rp[1] * rp[1]).sqrt();

    let mut reward = 0.0_f32;

    // 1. Weapons focus (ramped).
    if !is_combatable || nav_and_wep_match {
        reward += WEAPONS_FOCUS_REWARD * engagement_ramp;
    }

    // 2. Proximity: always applies.
    reward += MOVEMENT_LENGTH_SCALE / (dist + MOVEMENT_LENGTH_SCALE) * MOVEMENT_STEP_WEIGHT;

    // 3. Planet: approach + vel_match + braking.
    if is_planet {
        let vel_cos = obs[2];
        let vel_sin = obs[3];
        let speed_frac = obs[1];
        let ship_speed = speed_frac * max_speed;
        reward += compute_planet_approach_reward(rp, rv, dist, vel_cos, vel_sin, ship_speed, last_action.1 == 1);
    }

    // 4. Pickup: pursuit indicator (ramped).
    if is_pickup {
        reward += pursuit_ind * PICKUP_PURSUIT_WEIGHT * engagement_ramp;
    }

    // 5. Ship/asteroid: pursuit or fire indicator (ramped, only when nav=weapons).
    if is_combatable && nav_and_wep_match {
        if in_range {
            reward += fire_ind * COMBAT_FIRE_WEIGHT * engagement_ramp;
        } else {
            reward += pursuit_ind * COMBAT_PURSUIT_WEIGHT * engagement_ramp;
        }
    }

    reward
}

/// `model_actions` — when `Some`, contains one `(DiscreteAction, log_prob)` per
/// entry in `decisions` (same order). Used as the executed action in `RLControl`
/// mode. In `BehavioralCloning` mode (or when `None`) the rule-based action
/// from `compute_ai_action` is used for both execution and BC labels.
fn store_obs_actions(
    decisions: &[(Entity, Vec<f32>, Vec<f32>, Vec<Option<Target>>)],
    model_actions: Option<&[(DiscreteAction, f32)]>,
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
        &crate::ship::Distressed,
        &TracerSlots,
    )>,
    all_positions: &Query<&Position>,
    all_velocities: &Query<&LinearVelocity>,
    item_universe: &ItemUniverse,
    rl_sender: &RLSender,
    bc_sender: &BCSender,
    mode: &AIPlayMode,
    engagement_ramp: f32,
) {
    for (idx, (entity, obs, proj_obs, slot_targets)) in decisions.iter().enumerate() {
        let Ok((_, mut agent, mut ship, pos, vel, _, max_speed, transform, _, self_distressed, _)) =
            agents.get_mut(*entity)
        else {
            continue;
        };

        // Choose target first so compute_ai_action can act on it.
        let has_cargo = ship.cargo.values().sum::<u16>() > 0;
        let target_idx = choose_target_slot(&agent.personality, has_cargo, obs);
        if (target_idx as usize) < model::N_OBJECTS {
            let tgt = slot_targets[target_idx as usize].clone();
            // Weapons target = nav target when it's something we'd shoot at.
            ship.weapons_target = match &tgt {
                Some(Target::Ship(_)) | Some(Target::Asteroid(_)) => tgt.clone(),
                _ => None,
            };
            ship.nav_target = tgt;
        } else {
            ship.nav_target = None;
            ship.weapons_target = None;
        }

        // Compute the rule-based action — used as BC label and as the
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
            (
                turn_idx,
                thrust_idx,
                fire_primary,
                fire_secondary,
                target_idx,
                target_idx, // rule-based: weapons target = nav target
            )
        } else {
            (1, 1, 0, 0, target_idx, target_idx)
        };

        // In RLControl mode use the model's action; otherwise use rule-based.
        let (executed_action, action_log_prob): (DiscreteAction, f32) =
            if *mode == AIPlayMode::RLControl {
                model_actions
                    .and_then(|ma| ma.get(idx))
                    .copied()
                    .unwrap_or((rule_based_action, 0.0))
            } else {
                (rule_based_action, 0.0)
            };
        // Apply both targets from the action.
        if *mode == AIPlayMode::RLControl {
            let nav_idx = executed_action.4 as usize;
            if nav_idx < model::N_OBJECTS {
                ship.nav_target = slot_targets[nav_idx].clone();
            } else {
                ship.nav_target = None;
            }
            let wep_idx = executed_action.5 as usize;
            if wep_idx < model::N_OBJECTS {
                ship.weapons_target = slot_targets[wep_idx].clone();
            } else {
                ship.weapons_target = None;
            }

            // EXPERIMENT: Override trader nav target to best cargo-value planet.
            if OVERRIDE_TRADER_NAV_TARGET && agent.personality == Personality::Trader {
                let best_planet_idx = choose_target_slot(&Personality::Trader, true, obs);
                if (best_planet_idx as usize) < model::N_OBJECTS {
                    ship.nav_target = slot_targets[best_planet_idx as usize].clone();
                }
            }
        }

        // Store slot mapping on the agent for use by repeat_actions.
        agent.slot_targets = slot_targets.clone();

        // Store the PREVIOUS step's transition (obs recorded at t-1, action taken
        // at t-1, rewards accumulated between t-1 and t).
        let last_proj = agent.last_proj_obs.clone().unwrap_or_default();
        if let (Some(last_obs), Some(last_action)) = (agent.last_obs.clone(), agent.last_action) {
            // Build per-type reward array from accumulated events + health signal.
            let mut rewards = agent.accumulated_rewards;
            let health_reward = ship.health as f32 / ship.data.max_health.max(1) as f32;
            let personality_health_weight = match agent.personality {
                Personality::Fighter => crate::consts::HEALTH_STEP_FIGHTER,
                Personality::Miner | Personality::Trader => crate::consts::HEALTH_STEP_MINER_TRADER,
            };
            rewards[crate::consts::REWARD_HEALTH] += health_reward * personality_health_weight;
            let (nav_target_reward, good_nav_target) =
                compute_nav_target_reward(&agent.personality, obs);
            rewards[crate::consts::REWARD_NAV_TARGET] += nav_target_reward;

            rewards[crate::consts::REWARD_WEAPONS_TARGET] +=
                compute_weapons_target_reward(&agent.personality, obs);

            // Movement + engagement reward.
            if good_nav_target {
                rewards[crate::consts::REWARD_MOVEMENT] += compute_movement_reward(
                    obs,
                    last_action,
                    max_speed.0,
                    engagement_ramp,
                );
            }
            let transition = Transition {
                obs: last_obs,
                proj_obs: last_proj.clone(),
                action: last_action,
                rewards,
                done: false,
                log_prob: agent.last_log_prob,
            };
            agent.segment_buffer.push(transition.clone());
            agent.accumulated_rewards = [0.0; crate::consts::N_REWARD_TYPES];
            agent.steps_since_flush += 1;

            // Also send BC data.  Pre-process obs → model_input here so the
            // training thread pays zero per-sample transform cost per step.
            // NOTE: use `transition.action` (= last_action, the rule-based action
            // computed at step t-1 for obs[t-1]), NOT `accurate_action` (which is
            // the rule-based action for the CURRENT step t).  Pairing obs[t-1]
            // with action[t] would create a systematic off-by-one mismatch.
            if *mode == AIPlayMode::BehavioralCloning {
                let bc = BCTransition {
                    obs: transition.obs.clone(),
                    proj_obs: transition.proj_obs.clone(),
                    action: transition.action,
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
                    bootstrap_value: Some([0.0; crate::consts::N_REWARD_TYPES]),
                };
                let _ = rl_sender.0.try_send(segment);
                agent.segment_initial_hidden = agent.hidden_state.clone();
                agent.steps_since_flush = 0;
            }
        }

        // Update agent with new observation and action.
        agent.last_obs = Some(obs.clone());
        agent.last_proj_obs = Some(proj_obs.clone());
        agent.last_action = Some(executed_action);
        agent.last_log_prob = action_log_prob;
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
                    println!(
                        "[bc] Restored buffer with {} transitions from {buffer_path}",
                        b.len()
                    );
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
                let batch: Vec<&BCTransition> = (0..BC_BATCH_SIZE)
                    .map(|_| &buffer[rng.gen_range(0..n)])
                    .collect();

                let mut self_flat = Vec::with_capacity(BC_BATCH_SIZE * SELF_INPUT_DIM);
                let mut obj_flat = Vec::with_capacity(BC_BATCH_SIZE * N_OBJECTS * OBJECT_INPUT_DIM);
                let mut proj_flat = Vec::with_capacity(BC_BATCH_SIZE * model::PROJECTILES_FLAT_DIM);
                for t in &batch {
                    let (s, o) = split_obs(&t.obs);
                    self_flat.extend_from_slice(s);
                    obj_flat.extend_from_slice(o);
                    proj_flat.extend_from_slice(&t.proj_obs);
                }
                let self_input = burn::tensor::Tensor::<TrainBackend, 2>::from_data(
                    TensorData::new(self_flat, [BC_BATCH_SIZE, SELF_INPUT_DIM]),
                    &device,
                );
                let obj_input = burn::tensor::Tensor::<TrainBackend, 3>::from_data(
                    TensorData::new(obj_flat, [BC_BATCH_SIZE, N_OBJECTS, OBJECT_INPUT_DIM]),
                    &device,
                );
                let proj_input = burn::tensor::Tensor::<TrainBackend, 3>::from_data(
                    TensorData::new(
                        proj_flat,
                        [
                            BC_BATCH_SIZE,
                            model::N_PROJECTILE_SLOTS,
                            model::PROJ_INPUT_DIM,
                        ],
                    ),
                    &device,
                );

                // Build one [B, 4] label tensor and slice per head — four
                // Vec builds collapsed into one.
                // Action labels: [B, 4] for turn/thrust/fire_primary/fire_secondary.
                let action_labels_flat: Vec<i64> = batch
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
                let action_labels = burn::tensor::Tensor::<TrainBackend, 2, Int>::from_data(
                    TensorData::new(action_labels_flat, [BC_BATCH_SIZE, 4]),
                    &device,
                );
                let turn_t = action_labels
                    .clone()
                    .narrow(1, 0, 1)
                    .reshape([BC_BATCH_SIZE]);
                let thrust_t = action_labels
                    .clone()
                    .narrow(1, 1, 1)
                    .reshape([BC_BATCH_SIZE]);
                let fp_t = action_labels
                    .clone()
                    .narrow(1, 2, 1)
                    .reshape([BC_BATCH_SIZE]);
                let fs_t = action_labels.narrow(1, 3, 1).reshape([BC_BATCH_SIZE]);

                // Target labels: nav and weapons target indices.
                let nav_target_labels: Vec<i64> = batch.iter().map(|t| t.action.4 as i64).collect();
                let nav_target_t = burn::tensor::Tensor::<TrainBackend, 1, Int>::from_data(
                    TensorData::new(nav_target_labels, [BC_BATCH_SIZE]),
                    &device,
                );
                let wep_target_labels: Vec<i64> = batch.iter().map(|t| t.action.5 as i64).collect();
                let wep_target_t = burn::tensor::Tensor::<TrainBackend, 1, Int>::from_data(
                    TensorData::new(wep_target_labels, [BC_BATCH_SIZE]),
                    &device,
                );

                // ── Forward + loss ────────���─────────────────────────────────���
                let grads = {
                    let net = inner.policy_net.as_ref().unwrap();
                    let (action_logits, nav_target_logits, wep_target_logits) =
                        net.forward(self_input, obj_input, proj_input);

                    // Weight each head's loss inversely by log(num_classes) so
                    // the gradient contribution from each head is roughly equal
                    // at initialisation.  Without this, the 13-class target heads
                    // dominate and the 2/3-class action heads barely learn.
                    let w3 = 1.0 / (3.0_f32).ln(); // turn (3 classes)
                    let w2 = 1.0 / (2.0_f32).ln(); // thrust/fire (2 classes)
                    let wt = 1.0 / (model::TARGET_OUTPUT_DIM as f32).ln(); // target (13 classes)

                    let turn_loss =
                        loss_fn.forward(action_logits.clone().narrow(1, 0, 3), turn_t) * w3;
                    let thrust_loss =
                        loss_fn.forward(action_logits.clone().narrow(1, 3, 2), thrust_t) * w2;
                    let fp_loss = loss_fn.forward(action_logits.clone().narrow(1, 5, 2), fp_t) * w2;
                    let fs_loss = loss_fn.forward(action_logits.narrow(1, 7, 2), fs_t) * w2;
                    let nav_tgt_loss = loss_fn.forward(nav_target_logits, nav_target_t) * wt;
                    let wep_tgt_loss = loss_fn.forward(wep_target_logits, wep_target_t) * wt;
                    let total_loss =
                        turn_loss + thrust_loss + fp_loss + fs_loss + nav_tgt_loss + wep_tgt_loss;

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
//     [u8; 5]       — action (turn, thrust, fire_primary, fire_secondary, target_idx)
//     [f32; obs_dim] — flat observation

/// Serialise the replay buffer to `path`.  Silent on error (training continues).
fn save_bc_buffer(buffer: &VecDeque<BCTransition>, path: &str) {
    use std::io::Write;
    let Some(first) = buffer.front() else { return };
    let input_dim = first.obs.len() as u32;

    let result = (|| -> std::io::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(&input_dim.to_le_bytes())?;
        f.write_all(&(buffer.len() as u32).to_le_bytes())?;
        for t in buffer {
            f.write_all(&[
                t.action.0, t.action.1, t.action.2, t.action.3, t.action.4, t.action.5,
            ])?;
            for &v in &t.obs {
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
/// observation dimension.
pub(crate) fn load_bc_buffer(path: &str) -> Option<VecDeque<BCTransition>> {
    use crate::rl_obs::OBS_DIM;
    use std::io::Read;
    let expected_dim = OBS_DIM;
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
    let mut action_buf = [0u8; 6];
    let mut input_bytes = vec![0u8; expected_dim * 4];

    for _ in 0..count {
        f.read_exact(&mut action_buf).ok()?;
        f.read_exact(&mut input_bytes).ok()?;
        let obs: Vec<f32> = input_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        buffer.push_back(BCTransition {
            obs,
            proj_obs: vec![0.0; crate::rl_obs::K_PROJECTILES * crate::rl_obs::PROJ_SLOT_SIZE],
            action: (
                action_buf[0],
                action_buf[1],
                action_buf[2],
                action_buf[3],
                action_buf[4],
                action_buf[5],
            ),
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
