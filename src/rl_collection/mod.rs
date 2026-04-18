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
use std::sync::mpsc;
use std::sync::Arc;

use avian2d::prelude::*;
use bevy::prelude::*;

mod bc;
#[allow(unused_imports)]
pub use bc::{compute_bc_loss, compute_bc_loss_from_logits, load_bc_buffer};

use crate::CurrentStarSystem;
use crate::ai_ships::{AIShip, compute_ai_action};
use crate::asteroids::{Asteroid, AsteroidField};
use crate::item_universe::ItemUniverse;
use crate::model::{self, RLResource};
use crate::pickups::Pickup;
use crate::planets::Planet;
use crate::rl_obs::{
    self, AsteroidSlotData, CoreSlotData, DETECTION_RADIUS, DiscreteAction, EntityKind,
    EntitySlotData, K_ASTEROIDS, K_FRIENDLY_SHIPS, K_HOSTILE_SHIPS, K_OTHER_PROJECTILES, K_PICKUPS,
    K_PLANETS, ObsInput, PickupSlotData, PlanetSlotData, ProjectileSlotData,
    ShipSlotData,
};
use crate::ship::{Personality, Ship, ShipCommand, ShipHostility, Target};
use crate::weapons::{FireCommand, GuidedMissile, Projectile, TracerSlots};
use crate::{GameLayer, PlayState};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of decision-steps per trajectory segment.
pub const RL_SEGMENT_LEN: usize = 128;

/// Decision rate in seconds.
const RL_STEP_PERIOD: f32 = 0.25; // 4 Hz

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
    /// Rule-based (expert) action at the same `obs`, used as the BC label
    /// for the in-PPO BC auxiliary loss.  Computed every step regardless of
    /// whether the executed `action` came from the policy or the rule-based AI.
    pub rule_based_action: DiscreteAction,
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
    pub personality: Personality,
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
    pub personality: Personality,
    pub transitions: Vec<Transition>,
}

/// Emitted just before an `RLAgent` ship jumps out of the system.
/// The segment is flushed as a truncated (non-terminal) trajectory.
#[derive(Event, Message)]
pub struct RLShipJumped {
    pub personality: Personality,
    pub transitions: Vec<Transition>,
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/// Added to every AI ship we want to train.
#[derive(Component)]
pub struct RLAgent {
    pub personality: Personality,
    /// Observation from the previous decision step.
    pub last_obs: Option<Vec<f32>>,
    /// Projectile observation from the previous decision step.
    pub last_proj_obs: Option<Vec<f32>>,
    /// Action chosen at the previous decision step.
    pub last_action: Option<DiscreteAction>,
    /// Rule-based (expert) action at the previous decision step — used as the
    /// BC label regardless of whether the executed action came from the policy
    /// or the rule-based AI.
    pub last_rule_based_action: Option<DiscreteAction>,
    /// Log-probability of the action under the policy at sampling time.
    pub last_log_prob: f32,
    /// Per-type accumulated rewards since the last decision step.
    pub accumulated_rewards: [f32; crate::consts::N_REWARD_TYPES],
    /// Transitions collected since the last segment flush.
    pub segment_buffer: Vec<Transition>,
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
            last_obs: None,
            last_proj_obs: None,
            last_action: None,
            last_rule_based_action: None,
            last_log_prob: 0.0,
            accumulated_rewards: [0.0; crate::consts::N_REWARD_TYPES],
            segment_buffer: Vec::with_capacity(RL_SEGMENT_LEN),
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
                bc::spawn_bc_training_thread(bc_rx, inference_net_arc, experiment);
                drop(rl_rx);
            }
            crate::AppMode::RLTraining => {
                crate::ppo::spawn_ppo_training_thread(rl_rx, inference_net_arc, experiment);
                // BC labels now travel inline with each `Transition`, so the
                // separate BC channel is unused during RL training.
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
        // Ship data for non-RLAgent ships (includes AI ships AND the player).
        // Disjoint with the mutable `agents` query via Without<RLAgent>.
        Query<(Entity, &Ship, &ShipHostility), Without<RLAgent>>,
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
        time.elapsed_secs(),
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
        // Weapons fire only when the ship has a valid weapons_target.
        // This prevents firing at neutral/friendly ships that happen to be
        // in front because they're the nav_target.
        let wep_entity = ship.weapons_target.as_ref().map(|t| t.get_entity());
        let wep_is_ship = matches!(ship.weapons_target, Some(Target::Ship(_)));
        if fire_primary == 1 && wep_entity.is_some() {
            for weapon_type in ship.weapon_systems.primary.keys() {
                fire_writer.write(FireCommand {
                    ship: entity,
                    weapon_type: weapon_type.clone(),
                    target: wep_entity,
                });
            }
        }
        if fire_secondary == 1 && wep_entity.is_some() {
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
            personality: ev.personality.clone(),
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
            personality: ev.personality.clone(),
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
    ship_query: &Query<(Entity, &Ship, &ShipHostility), Without<RLAgent>>,
    all_ship_factions: &Query<&ShipHostility, With<Ship>>,
    all_distressed: &Query<&crate::ship::Distressed>,
    projectile_query: &Query<(&Projectile, Option<&GuidedMissile>)>,
    spatial_query: &SpatialQuery,
    item_universe: &ItemUniverse,
    current_system: &CurrentStarSystem,
    current_time_secs: f32,
) -> Vec<(Entity, Vec<f32>, Vec<f32>, Vec<Option<Target>>)> {
    let mut results = Vec::new();

    for (
        entity,
        agent,
        ship,
        pos,
        vel,
        ang_vel,
        _max_speed,
        transform,
        _self_hostility,
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

        // Cargo profit value helper: profit (sale value - acquisition cost) of
        // selling the ship's current cargo at a planet. Pickups have zero
        // recorded cost basis so they show their full sale value as profit.
        let cargo_profit_value = |planet_name: &str| -> f32 {
            let planet_data = item_universe
                .star_systems
                .get(system_name)
                .and_then(|sys| sys.planets.get(planet_name));
            let Some(pd) = planet_data else { return 0.0 };
            ship.cargo
                .iter()
                .map(|(commodity, &qty)| {
                    let sale = pd.commodities.get(commodity).copied().unwrap_or(0) as f32
                        * qty as f32;
                    let cost = ship.cargo_cost.get(commodity).copied().unwrap_or(0) as f32;
                    sale - cost
                })
                .sum()
        };

        // Buckets — sort by distance within each type.
        // Planets: include ALL planets in the system (not just nearby), since
        // there are few and they're critical for navigation/trading.
        // Note: static RigidBody entities may only have Transform, not Position.
        let all_transforms: &Query<&Transform> = all_transforms;
        let system_data = item_universe.star_systems.get(system_name);
        let mut planets: Vec<(Entity, f32)> = planet_query
            .iter()
            .filter_map(|(e, planet)| {
                // Skip planets the ship can't land on (uncolonized or hostile).
                if let Some(sd) = system_data {
                    if let Some(pd) = sd.planets.get(&planet.0) {
                        if pd.uncolonized {
                            return None;
                        }
                        if !pd.faction.is_empty()
                            && _self_hostility
                                .0
                                .get(&pd.faction)
                                .copied()
                                .unwrap_or(0.0)
                                > 0.0
                        {
                            return None;
                        }
                    }
                }
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
                let cpv = cargo_profit_value(planet_name);
                let has_ammo = item_universe
                    .planet_has_ammo_for
                    .get(planet_name)
                    .map(|set| set.contains(&ship.ship_type))
                    .unwrap_or(false);
                let margin = planet_margins
                    .and_then(|m| m.get(planet_name))
                    .copied()
                    .unwrap_or(0.0) as f32;
                let recently_visited = ship
                    .recent_landings
                    .get(planet_name)
                    .map(|&t| t > current_time_secs)
                    .unwrap_or(false);
                EntitySlotData {
                    core,
                    kind: EntityKind::Planet(PlanetSlotData {
                        cargo_profit_value: cpv,
                        has_ammo: if has_ammo { 1.0 } else { 0.0 },
                        commodity_margin: margin,
                        is_recently_visited: if recently_visited { 1.0 } else { 0.0 },
                    }),
                    value: cpv,
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
                    // Their primary weapon's range + fire rate (first primary, if any).
                    let (other_primary_range, other_primary_fire_rate) = other_ship
                        .weapon_systems
                        .primary
                        .values()
                        .next()
                        .map(|ws| {
                            let dur = ws.cooldown.duration().as_secs_f32().max(f32::EPSILON);
                            (ws.weapon.range(), 1.0 / dur)
                        })
                        .unwrap_or((0.0, 0.0));
                    // 1.0 iff their weapons_target's entity is us.
                    let is_targeting_me = match &other_ship.weapons_target {
                        Some(t) => {
                            if t.get_entity() == entity {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        None => 0.0,
                    };
                    let sd = ShipSlotData {
                        max_health: data.map(|d| d.max_health as f32).unwrap_or(0.0),
                        health: other_ship.health as f32,
                        max_speed: data.map(|d| d.max_speed as f32).unwrap_or(0.0),
                        torque: data.map(|d| d.torque as f32).unwrap_or(0.0),
                        is_hostile,
                        should_engage,
                        personality: data.map(|d| d.personality.clone()).unwrap_or_default(),
                        distressed: other_distressed,
                        primary_weapon_range: other_primary_range,
                        is_targeting_me,
                        thrust: data.map(|d| d.thrust).unwrap_or(0.0),
                        primary_fire_rate: other_primary_fire_rate,
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
        push_up_to(&mut nearby_pickups, &pickups, K_PICKUPS, Target::Pickup);

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

        // Primary weapon speed / range / damage / cooldown / fire_rate for self.
        let (
            primary_weapon_speed,
            primary_weapon_range,
            primary_weapon_damage,
            primary_cooldown_frac,
            primary_fire_rate,
        ) = ship
            .weapon_systems
            .primary
            .values()
            .next()
            .map(|ws| {
                let dur = ws.cooldown.duration().as_secs_f32().max(f32::EPSILON);
                let frac = ws.cooldown.remaining_secs() / dur;
                (
                    ws.weapon.speed,
                    ws.weapon.range(),
                    ws.weapon.damage as f32,
                    frac,
                    1.0 / dur,
                )
            })
            .unwrap_or((0.0, 0.0, 0.0, 0.0, 0.0));

        // Secondary weapon stats — choose based on current weapons_target type.
        let wep_target_is_ship = matches!(ship.weapons_target, Some(Target::Ship(_)));
        let (
            secondary_weapon_range,
            secondary_weapon_damage,
            secondary_weapon_speed,
            secondary_cooldown_frac,
            secondary_fire_rate,
        ) = rl_obs::select_secondary_weapon(&ship, wep_target_is_ship)
            .and_then(|(wname, _ammo)| ship.weapon_systems.secondary.get(wname))
            .map(|ws| {
                let dur = ws.cooldown.duration().as_secs_f32().max(f32::EPSILON);
                let frac = ws.cooldown.remaining_secs() / dur;
                (
                    ws.weapon.range(),
                    ws.weapon.damage as f32,
                    ws.weapon.speed,
                    frac,
                    1.0 / dur,
                )
            })
            .unwrap_or((0.0, 0.0, 0.0, 0.0, 0.0));

        // Current nav/wep target types for the self observation one-hots.
        let target_type_code = |t: &Option<Target>| -> u8 {
            use rl_obs::{
                TARGET_TYPE_IDX_ASTEROID, TARGET_TYPE_IDX_NONE, TARGET_TYPE_IDX_PICKUP,
                TARGET_TYPE_IDX_PLANET, TARGET_TYPE_IDX_SHIP,
            };
            match t {
                Some(Target::Ship(_)) => TARGET_TYPE_IDX_SHIP as u8,
                Some(Target::Asteroid(_)) => TARGET_TYPE_IDX_ASTEROID as u8,
                Some(Target::Planet(_)) => TARGET_TYPE_IDX_PLANET as u8,
                Some(Target::Pickup(_)) => TARGET_TYPE_IDX_PICKUP as u8,
                None => TARGET_TYPE_IDX_NONE as u8,
            }
        };
        let nav_target_type = target_type_code(&ship.nav_target);
        let weapons_target_type = target_type_code(&ship.weapons_target);

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
            // is_tracking_me: 1.0 if guided missile has us as target.
            let is_tracking_me = guided
                .and_then(|g| g.target)
                .map(|t| if t == entity { 1.0 } else { 0.0 })
                .unwrap_or(0.0);
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
                is_tracking_me,
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
            primary_weapon_damage,
            primary_cooldown_frac,
            secondary_weapon_range,
            secondary_weapon_damage,
            secondary_weapon_speed,
            secondary_cooldown_frac,
            primary_fire_rate,
            secondary_fire_rate,
            nav_target_type,
            weapons_target_type,
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
    let is_nav_target = |i: usize| obs[slot_base(i) + SLOT_IS_NAV_TARGET] > 0.5;
    let type_onehot =
        |i: usize| &obs[slot_base(i) + SLOT_TYPE_ONEHOT..slot_base(i) + SLOT_TYPE_ONEHOT + 4];
    let is_ship = |i: usize| type_onehot(i)[0] > 0.5;
    let is_asteroid = |i: usize| type_onehot(i)[1] > 0.5;
    let is_planet = |i: usize| type_onehot(i)[2] > 0.5;
    let is_pickup = |i: usize| type_onehot(i)[3] > 0.5;
    let should_engage =
        |i: usize| obs[slot_base(i) + SLOT_TYPE_SPECIFIC + SHIP_SHOULD_ENGAGE] > 0.5;
    let value = |i: usize| obs[slot_base(i) + SLOT_VALUE];

    // Helper: find the first present slot matching a predicate.
    let first_matching = |pred: &dyn Fn(usize) -> bool| -> Option<u8> {
        (0..n).find(|&i| is_present(i) && pred(i)).map(|i| i as u8)
    };

    // Ship-level state that exempts a planet from the "must be profitable"
    // filter: low health (needs repair), or free cargo space (can buy more).
    let low_health = obs[SELF_HEALTH_FRAC] < 0.5;
    let has_free_cargo = obs[SELF_CARGO_FRAC] < 1.0;
    let planet_has_ammo =
        |i: usize| obs[slot_base(i) + SLOT_TYPE_SPECIFIC + PLANET_HAS_AMMO] > 0.5;
    let planet_profit =
        |i: usize| obs[slot_base(i) + SLOT_TYPE_SPECIFIC + PLANET_CARGO_PROFIT_VALUE];
    let planet_recently_visited =
        |i: usize| obs[slot_base(i) + SLOT_TYPE_SPECIFIC + PLANET_IS_RECENTLY_VISITED] > 0.5;
    // A planet is a viable trader destination if selling its current cargo
    // there would yield positive profit, OR if the ship has another reason to
    // visit (repair, refill ammo, or load more cargo). The recent-visited
    // cooldown blocks re-landing regardless of the above reasons.
    let planet_viable = |i: usize| {
        !planet_recently_visited(i)
            && (planet_profit(i) > 0.0 || low_health || planet_has_ammo(i) || has_free_cargo)
    };

    // 1. Traders with cargo → planet with highest value (= cargo_profit_value).
    if has_cargo && matches!(personality, Personality::Trader) {
        let best_planet = (0..n)
            .filter(|&i| is_present(i) && is_planet(i) && planet_viable(i))
            .max_by(|&a, &b| {
                value(a)
                    .partial_cmp(&value(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        if let Some(idx) = best_planet {
            return idx as u8;
        }
    }

    // 2. Sticky: keep the current target if it is still present in the
    // observation (i.e. within detection range) AND still valid for this
    // personality's task.  This prevents churn when multiple candidate
    // targets are in range.
    let current = (0..n).find(|&i| is_present(i) && is_nav_target(i));
    if let Some(i) = current {
        let still_valid = match personality {
            // Planets are always filtered through `planet_viable` to match the
            // model's nav-target mask — otherwise BC labels can point at a
            // masked logit (recently-visited / unprofitable) and blow up the
            // cross-entropy.
            Personality::Miner => {
                is_pickup(i) || is_asteroid(i) || (is_planet(i) && planet_viable(i))
            }
            Personality::Fighter => {
                (is_ship(i) && should_engage(i))
                    || is_pickup(i)
                    || (is_planet(i) && planet_viable(i))
            }
            Personality::Trader => (is_planet(i) && planet_viable(i)) || is_pickup(i),
        };
        if still_valid {
            return i as u8;
        }
    }

    // 3. Personality-based fallback.
    match personality {
        Personality::Miner => first_matching(&|i| is_pickup(i))
            .or_else(|| first_matching(&|i| is_asteroid(i)))
            .or_else(|| first_matching(&|i| is_planet(i) && planet_viable(i)))
            .unwrap_or(no_target),
        Personality::Fighter => first_matching(&|i| is_ship(i) && should_engage(i))
            .or_else(|| first_matching(&|i| is_pickup(i)))
            .or_else(|| first_matching(&|i| is_planet(i) && planet_viable(i)))
            .unwrap_or(no_target),
        Personality::Trader => first_matching(&|i| is_planet(i) && planet_viable(i))
            .or_else(|| first_matching(&|i| is_pickup(i)))
            .unwrap_or(no_target),
    }
}


/// `model_actions` — when `Some`, contains one `(DiscreteAction, log_prob)` per
/// entry in `decisions` (same order). Used as the executed action in `RLControl`
/// mode. In `BehavioralCloning` mode (or when `None`) the rule-based action
/// from `compute_ai_action` is used for both execution and BC labels.
/// Mix visible allies' rewards into `rewards`, scaled by personality.
///
/// Only considers entities in `slot_targets` that are `Target::Ship` AND
/// appear in `reward_snapshots` (i.e. are RLAgent ships) AND whose faction
/// is in the observer's `allies` list.  `health_raw` is excluded (diagnostic
/// channel with weight 0).
pub(crate) fn mix_ally_rewards(
    rewards: &mut [f32; crate::consts::N_REWARD_TYPES],
    slot_targets: &[Option<Target>],
    allies: &[String],
    self_entity: Entity,
    personality: &Personality,
    reward_snapshots: &std::collections::HashMap<
        Entity,
        ([f32; crate::consts::N_REWARD_TYPES], Option<String>),
    >,
) {
    use crate::consts::*;
    let alpha = match personality {
        Personality::Fighter => REWARD_SHARING_FIGHTER,
        Personality::Miner => REWARD_SHARING_MINER,
        Personality::Trader => REWARD_SHARING_TRADER,
    };
    if alpha <= 0.0 || allies.is_empty() {
        return;
    }
    let mut ally_reward_sum = [0.0_f32; N_REWARD_TYPES];
    let mut ally_count = 0u32;
    for target in slot_targets.iter() {
        if let Some(Target::Ship(ally_e)) = target {
            if *ally_e == self_entity {
                continue;
            }
            if let Some((ally_rewards, ally_faction)) = reward_snapshots.get(ally_e) {
                let is_ally = ally_faction
                    .as_ref()
                    .map(|f| allies.contains(f))
                    .unwrap_or(false);
                if is_ally {
                    for ch in 0..N_REWARD_TYPES {
                        if ch != REWARD_HEALTH_RAW {
                            ally_reward_sum[ch] += ally_rewards[ch];
                        }
                    }
                    ally_count += 1;
                }
            }
        }
    }
    if ally_count > 0 {
        let scale = alpha / ally_count as f32;
        for ch in 0..N_REWARD_TYPES {
            rewards[ch] += scale * ally_reward_sum[ch];
        }
    }
}

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
) {
    // Pre-pass: snapshot accumulated_rewards and faction for every RLAgent.
    // This must happen before the mutable loop so reward sharing can read
    // allies' rewards without borrow conflicts.
    let reward_snapshots: std::collections::HashMap<Entity, ([f32; crate::consts::N_REWARD_TYPES], Option<String>)> =
        agents
            .iter()
            .map(|(e, agent, s, _, _, _, _, _, _, _, _)| {
                (e, (agent.accumulated_rewards, s.data.faction.clone()))
            })
            .collect();

    for (idx, (entity, obs, proj_obs, slot_targets)) in decisions.iter().enumerate() {
        let Ok((
            _,
            mut agent,
            mut ship,
            pos,
            vel,
            ang_vel,
            max_speed,
            transform,
            _,
            _self_distressed,
            _,
        )) = agents.get_mut(*entity)
        else {
            continue;
        };

        // Choose target first so compute_ai_action can act on it.
        let has_cargo = ship.cargo.values().sum::<u16>() > 0;
        let target_idx = choose_target_slot(&agent.personality, has_cargo, obs);
        let wep_target_idx: u8 = if (target_idx as usize) < model::N_OBJECTS {
            let tgt = slot_targets[target_idx as usize].clone();
            // Weapons target = nav target when it's an asteroid or a
            // hostile/should-engage ship; otherwise the nearest such ship (slots
            // are ordered nearest-first), else None.
            use rl_obs::*;
            let slot_base = |i: usize| SELF_SIZE + i * SLOT_SIZE;
            let engageable = |i: usize| {
                obs[slot_base(i) + SLOT_IS_PRESENT] > 0.5
                    && obs[slot_base(i) + SLOT_TYPE_ONEHOT] > 0.5
                    && (obs[slot_base(i) + SLOT_TYPE_SPECIFIC + SHIP_IS_HOSTILE] > 0.5
                        || obs[slot_base(i) + SLOT_TYPE_SPECIFIC + SHIP_SHOULD_ENGAGE] > 0.5)
            };
            let (wep_tgt, wep_idx) = match &tgt {
                Some(Target::Asteroid(_)) => (tgt.clone(), target_idx),
                Some(Target::Ship(_)) if engageable(target_idx as usize) => {
                    (tgt.clone(), target_idx)
                }
                _ => match (0..model::N_OBJECTS).find(|&i| engageable(i)) {
                    Some(i) => (slot_targets[i].clone(), i as u8),
                    None => (None, model::N_OBJECTS as u8),
                },
            };
            ship.weapons_target = wep_tgt;
            ship.nav_target = tgt;
            wep_idx
        } else {
            ship.nav_target = None;
            ship.weapons_target = None;
            model::N_OBJECTS as u8
        };

        // Compute the rule-based action — used as BC label and as the
        // fallback when no model action is available.
        let rule_based_action = if let Some(raw) = compute_ai_action(
            &*ship,
            pos.0,
            vel.0,
            ang_vel.0,
            max_speed.0,
            transform,
            all_positions,
            all_velocities,
            item_universe,
            &mut rand::thread_rng(),
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
                wep_target_idx,
            )
        } else {
            (1, 1, 0, 0, target_idx, wep_target_idx)
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
                // DEBUG: check if the model selected a non-engageable ship as wep target
                if let Some(ref tgt) = slot_targets[wep_idx] {
                    use rl_obs::*;
                    let sb = SELF_SIZE + wep_idx * SLOT_SIZE;
                    let is_ship_v = obs[sb + SLOT_TYPE_ONEHOT];
                    let hostile_v = obs[sb + SLOT_TYPE_SPECIFIC + SHIP_IS_HOSTILE];
                    let engage_v = obs[sb + SLOT_TYPE_SPECIFIC + SHIP_SHOULD_ENGAGE];
                    if is_ship_v > 0.5 && hostile_v < 0.5 && engage_v < 0.5 {
                        eprintln!(
                            "[BUG] {:?} selected NEUTRAL ship {:?} as wep_target! \
                             slot={wep_idx} is_hostile={hostile_v:.1} should_engage={engage_v:.1} \
                             bucket_type={}",
                            entity,
                            tgt.get_entity(),
                            if wep_idx < rl_obs::K_PLANETS { "planet" }
                            else if wep_idx < rl_obs::K_PLANETS + rl_obs::K_ASTEROIDS { "asteroid" }
                            else if wep_idx < rl_obs::K_PLANETS + rl_obs::K_ASTEROIDS + rl_obs::K_HOSTILE_SHIPS { "hostile_ship" }
                            else if wep_idx < rl_obs::K_PLANETS + rl_obs::K_ASTEROIDS + rl_obs::K_HOSTILE_SHIPS + rl_obs::K_FRIENDLY_SHIPS { "friendly_ship" }
                            else { "pickup" },
                        );
                    }
                }
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
            // Build per-type reward array from accumulated events.  Health is
            // not a per-step signal — it's emitted by the event writers,
            // scaled by the firing ship's `health/max_health` at event time.
            // Separately, `REWARD_HEALTH_RAW` is written per step with weight
            // 0.0 so its value head learns to predict expected future health
            // without influencing the policy.
            let mut rewards = agent.accumulated_rewards;
            rewards[crate::consts::REWARD_HEALTH_RAW] =
                ship.health as f32 / ship.data.max_health.max(1) as f32;

            // Reward sharing: mix visible allies' rewards.
            mix_ally_rewards(
                &mut rewards,
                slot_targets,
                &ship.allies,
                *entity,
                &agent.personality,
                &reward_snapshots,
            );

            let transition = Transition {
                obs: last_obs,
                proj_obs: last_proj.clone(),
                action: last_action,
                rule_based_action: agent
                    .last_rule_based_action
                    .unwrap_or(last_action),
                rewards,
                done: false,
                log_prob: agent.last_log_prob,
            };
            agent.segment_buffer.push(transition.clone());
            agent.accumulated_rewards = [0.0; crate::consts::N_REWARD_TYPES];
            agent.steps_since_flush += 1;

            // Also send BC data.  The BC label is the rule-based action at the
            // same step as `last_obs` — available in every mode because we stash
            // it into `last_rule_based_action` whenever a decision is made.
            // In BC mode this equals `transition.action`; in RL mode it differs
            // (transition.action is the policy's action) — BC still trains
            // against the expert label.
            if let Some(rule_action) = agent.last_rule_based_action {
                let bc = BCTransition {
                    obs: transition.obs.clone(),
                    proj_obs: transition.proj_obs.clone(),
                    action: rule_action,
                };
                let _ = bc_sender.0.try_send(bc);
            }

            // Flush if segment is full.
            if agent.steps_since_flush >= RL_SEGMENT_LEN {
                let segment = Segment {
                    personality: agent.personality.clone(),
                    transitions: agent.segment_buffer.drain(..).collect(),
                    bootstrap_value: Some([0.0; crate::consts::N_REWARD_TYPES]),
                };
                let _ = rl_sender.0.try_send(segment);
                agent.steps_since_flush = 0;
            }
        }

        // Update agent with new observation and action.
        agent.last_obs = Some(obs.clone());
        agent.last_proj_obs = Some(proj_obs.clone());
        agent.last_action = Some(executed_action);
        agent.last_rule_based_action = Some(rule_based_action);
        agent.last_log_prob = action_log_prob;
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Public helper to build RLShipDied from apply_damage
// ---------------------------------------------------------------------------

/// Build the `RLShipDied` event data from an `RLAgent` component before despawn.
pub fn build_rl_ship_died(_entity: Entity, agent: &RLAgent) -> RLShipDied {
    RLShipDied {
        personality: agent.personality.clone(),
        transitions: agent.segment_buffer.clone(),
    }
}

/// Build the `RLShipJumped` event data from an `RLAgent` component before a
/// voluntary jump-out.  The trajectory is flushed as truncated (non-terminal).
pub fn build_rl_ship_jumped(_entity: Entity, agent: &RLAgent) -> RLShipJumped {
    RLShipJumped {
        personality: agent.personality.clone(),
        transitions: agent.segment_buffer.clone(),
    }
}


#[cfg(test)]
#[path = "../tests/rl_collection_tests.rs"]
mod tests;
