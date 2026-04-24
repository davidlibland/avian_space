# Source Layout

Avian Space is a 2D space game built with Bevy and Avian2D physics, featuring
AI ships trained via reinforcement learning (PPO) with a behavioural-cloning
pre-training stage. The player flies between star systems, trades commodities,
mines asteroids, fights hostile ships, and completes missions.

## Entry Point

- **main.rs** — Application entry point. Parses CLI flags (`--classic`,
  `--inference`, `--bc-training`, `--rl-training`, `--fresh`, `--headless`),
  assembles the Bevy `App` with all plugins, defines top-level game states
  (`PlayState`: MainMenu, Flying, Landed, Traveling), physics layers
  (`GameLayer`), hyperspace travel logic, player keyboard input, and the
  collision system that dispatches damage/shatter/score-hit events.

## Core Gameplay

- **ship.rs** — The `Ship` component and `ShipData` (stats loaded from YAML).
  Handles ship movement physics (PD-controlled thrust/torque), damage
  application, death/respawn, buying/selling cargo and weapons, faction
  hostility tracking, and RL reward signals for combat hits. Defines the
  `ShipCommand`, `DamageShip`, `ScoreHit`, and `BuyShip` message types.

- **weapons.rs** — Weapon definitions (`Weapon`, `WeaponSystem`,
  `WeaponSystems`), projectile spawning with aimable arcs and guided missiles,
  cooldown management, tracer slot tracking for RL observations, and carrier
  bay integration. Defines `FireCommand` and `Projectile`.

- **planets.rs** — Planet spawning from YAML data, player proximity detection
  via collision sensors, landing input (L key), and the landing scale-down
  animation trigger. Defines `Planet`, `NearbyPlanet`, and `PlanetData`.

- **asteroids.rs** — Asteroid field spawning with orbital gravity, shattering
  into smaller fragments with commodity drops, respawn/grow/fade lifecycle,
  and planet collision handling. Defines `Asteroid`, `AsteroidField`, and
  `ShatterAsteroid`.

- **pickups.rs** — Floating commodity pickups dropped by shattered asteroids
  and destroyed ships. Handles spawn, collection on ship contact, and RL
  reward emission.

- **carrier.rs** — Carrier bay system: ships with carrier-bay weapons spawn
  escort ships instead of projectiles. Escorts mirror the mother's targets,
  return and re-dock when idle, and become independent if the mother is
  destroyed. Includes launch/dock animations.

- **explosions.rs** — Particle-based explosion and hyperspace jump flash
  effects. Registers `TriggerExplosion` and `TriggerJumpFlash` messages
  consumed by both visual and audio systems.

- **ship_anim.rs** — Reusable sprite scale-up/scale-down animation components
  (`ScalingUp`, `ScalingDown`) used by planet landing/takeoff and carrier
  escort launch/dock sequences. Emits `ScaleUpFinished`/`ScaleDownFinished`
  messages.

## AI & Control

- **ai_ships.rs** — AI ship lifecycle: spawning (initial + jump-in/jump-out),
  population management, rule-based target selection and control
  (`compute_ai_action`), planet landing/trading/takeoff cycle, and the
  `AIShip` component. Rule-based AI uses ballistic intercept for aiming and
  a PD pursuit controller for navigation.

- **optimal_control.rs** — Analytical control helpers: bang-bang turn-to-angle
  with stopping-distance prediction (`x_stop`, `turn_to_angle`), PD-based
  pursuit controller (`pursuit_controls_ego`), and observation features
  (`control_features`, `pursuit_features_ego`).

- **utils.rs** — Shared utilities: safe entity despawn, title-case string
  conversion, polygon mesh generation, random velocity, ballistic intercept
  angle computation (`angle_to_hit`), and aim indicator.

## Reinforcement Learning

- **rl_collection/mod.rs** — RL trajectory collection plugin. Adds the
  `RLAgent` component to AI ships and runs a 4 Hz decision loop (`rl_step`)
  that builds ego-centric observations, runs policy inference (or rule-based
  fallback), records transitions, and flushes completed segments to the
  training thread via mpsc channels. Handles both BehavioralCloning and
  RLControl modes. Also implements target selection (`choose_target_slot`),
  reward accumulation, ally reward sharing, and death/jump segment flushing.

- **rl_collection/bc.rs** — Behavioural-cloning training thread: maintains a
  replay buffer of (observation, expert-action) pairs, trains the policy
  network with weighted cross-entropy loss, and periodically syncs weights to
  the game-thread inference net. Includes buffer serialisation for warm restarts.

- **rl_obs.rs** — Observation encoding specification. Defines the full
  observation layout (self-state block + entity slots + projectile slots),
  all named feature offsets, bucket sizes, the `DiscreteAction` type, and
  the `encode_observation`/`encode_projectiles` functions that produce flat
  float vectors from structured `ObsInput` data.

- **consts.rs** — All RL reward constants and tuning parameters in one place.
  Defines reward channel indices, per-personality reward weights for combat,
  mining, trading, landing, and pickup events, plus the multi-head reward
  type weight array.

## Neural Network & Training

- **model/mod.rs** — Top-level module for the neural network. Defines
  dimension constants (observation sizes, policy/value/target output dims,
  hidden dim), backend type aliases (`InferBackend` = NdArray CPU,
  `TrainBackend` = Autodiff<Wgpu>), and the `RLResource` Bevy resource
  that holds the `Arc<Mutex<InferenceNet>>` shared between game and training
  threads.

- **model/net.rs** — Network architecture (`RLNet`): type-conditioned entity
  embedding with learned per-type weight matrices, multi-head attention over
  entity and projectile encodings, self-state embedding, merged decoder
  blocks, and pointer-network heads for nav/weapons target selection with
  viability masking (planet profitability, hostility filtering).

- **model/inference.rs** — `InferenceNet` wrapper for synchronous game-thread
  forward passes, plus checkpoint save/load helpers for both inference and
  training backends.

- **model/sampling.rs** — CPU-side action sampling from policy logits:
  categorical sampling with log-sum-exp stabilisation, coupled Gumbel-max
  sampling for correlated nav/weapons target selection, and greedy argmax
  (test-only).

- **model/training.rs** — Training-thread state (`RLInner`): holds policy and
  value networks with their Adam optimizers. Includes optimizer
  serialisation/deserialisation.

- **ppo/mod.rs** — PPO training module entry point.

- **ppo/train.rs** — Main PPO training loop: collects segments from the game
  thread, computes multi-head GAE advantages, runs clipped surrogate policy
  updates with entropy bonus and BC auxiliary loss, trains the multi-head
  value network with Huber loss, maintains a priority replay buffer for
  extra value training, syncs weights to the inference net, saves periodic
  checkpoints, and logs extensive diagnostics to TensorBoard.

- **ppo/batch.rs** — Flattens collected `Segment`s into contiguous
  `PpoBatch` arrays (observations, actions, rewards, log-probs) suitable for
  mini-batch training.

- **ppo/buffer.rs** — Fixed-capacity priority replay buffer for auxiliary
  value-function training, ordered by max absolute advantage.

- **ppo/loss.rs** — PPO clipped surrogate loss computation and factored
  log-probability/entropy calculation across all 6 action heads.

- **ppo/persistence.rs** — Serialisation of RL segment buffers and step
  counters for warm-restart across training runs.

- **gae.rs** — Pure-Rust multi-head Generalized Advantage Estimation (GAE).
  Computes per-reward-channel advantages and value regression targets with
  configurable gamma/lambda, respecting segment boundaries and bootstrap
  values.

- **value_fn.rs** — Value-function helpers: batched detached multi-head
  inference, bootstrap value recomputation for truncated segments, and
  Huber value loss.

## Data & Configuration

- **item_universe.rs** — Loads and indexes the entire game universe from YAML
  asset files: weapons, ships, star systems (with planets, asteroid fields,
  ship distributions), outfitter items, commodities, faction enemies/allies,
  missions, and mission templates. Computes derived trade maps (best
  buy/sell planets, commodity margins), ammo availability, asteroid field
  expected values, and per-ship credit scales. Also handles sprite and sound
  preloading.

- **experiments.rs** — Training experiment directory management
  (`experiments/run_N/`). Resolves checkpoint paths for policy, value,
  optimizer, and buffer files. Supports `--fresh` for new runs vs. resuming
  the latest.

- **session.rs** — Session-resource infrastructure. See the **Session
  Resources** section below for full details.

- **game_save.rs** — Player save/load system. `PlayerGameState` holds only
  pilot identity and the Ship component mirror; all other per-pilot state
  (missions, unlocks, UI) is managed by the `SessionResource` trait.
  Serialises to YAML files in `pilots/`; the save file has top-level ship
  fields plus a `resources` map populated by session resources.

## Session Resources

Any ECS `Resource` whose lifetime is tied to a single pilot session should
implement the `SessionResource` trait (defined in `session.rs`) and be
registered with `app.init_session_resource::<R>()` instead of
`app.init_resource::<R>()`.

### What the trait provides

Registering via `init_session_resource` gives three behaviours for free:

1. **Reset** — the resource is re-initialised via `new_session(universe)` when
   entering `MainMenu` (i.e. on pilot switch, escape, or death).
2. **Save** — if the resource declares a `SAVE_KEY`, its `to_save()` output is
   serialised into the pilot save file's `resources` map every time the
   resource changes.
3. **Load** — on entering `Flying` after pilot selection, saved data is fed
   back through `from_save(data, universe)`.

### The trait

```rust
pub trait SessionResource: Resource + Send + Sync + 'static {
    /// Serialisable snapshot type.  Use `()` for ephemeral resources.
    type SaveData: Serialize + Deserialize + Default;

    /// Key in the save file's `resources` map.  `None` = not persisted.
    const SAVE_KEY: Option<&'static str> = None;

    /// Fresh state for a brand-new pilot.
    fn new_session(universe: &ItemUniverse) -> Self;

    /// Snapshot live state for saving (only called when SAVE_KEY is Some).
    fn to_save(&self) -> Self::SaveData { Default::default() }

    /// Restore from a saved snapshot.
    fn from_save(data: Self::SaveData, universe: &ItemUniverse) -> Self;
}
```

### Adding a new session resource

1. Implement `SessionResource` on your struct (in the file that defines it).
2. Call `app.init_session_resource::<YourResource>()` in the owning plugin.

That's it — reset, save, and load are handled automatically.

**Ephemeral example** (reset on pilot switch, not saved):

```rust
impl SessionResource for CommsChannel {
    type SaveData = ();
    fn new_session(_: &ItemUniverse) -> Self { Self::default() }
    fn from_save(_: (), _: &ItemUniverse) -> Self { Self::default() }
}
```

**Persisted example** (saved to disk):

```rust
impl SessionResource for MissionLog {
    type SaveData = MissionLogSave;
    const SAVE_KEY: Option<&'static str> = Some("mission_statuses");

    fn new_session(_: &ItemUniverse) -> Self { Self::default() }

    fn to_save(&self) -> Self::SaveData {
        MissionLogSave(self.statuses.iter()
            .filter(|(_, s)| !matches!(s, MissionStatus::Locked))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect())
    }

    fn from_save(data: Self::SaveData, _: &ItemUniverse) -> Self {
        Self { statuses: data.0 }
    }
}
```

### What is NOT a session resource

- **`PlayerGameState`** — the Ship component mirror. It manages pilot
  identity and the live ship sync; it is reset by its own `OnEnter(MainMenu)`
  system in `game_save.rs`.  Ship data lives at the top level of the save
  file, not in the `resources` map.
- **Static config** — `ItemUniverse`, asset handles, audio resources.  These
  are global and never reset.
- **Engine/physics** — Bevy and Avian2D internals.

### Current session resources

| Resource | Module | Persisted | Key |
|---|---|---|---|
| `MissionLog` | missions/log | Yes | `mission_statuses` |
| `MissionCatalog` | missions/log | Yes | `active_mission_defs` |
| `PlayerUnlocks` | missions/log | Yes | `unlocks` |
| `MissionOffers` | missions/log | No | — |
| `MissionToast` | missions/ui | No | — |
| `MissionLogOpen` | missions/ui | No | — |
| `CommsChannel` | hud | No | — |
| `JumpUiOpen` | jump_ui | No | — |
| `NearbyPlanet` | planets | No | — |
| `LandedContext` | planet_ui | No | — |

## Missions

- **missions/mod.rs** — Mission plugin wiring: registers all mission
  resources, messages, and systems.

- **missions/types.rs** — Mission data model: `MissionDef` (briefing,
  objectives, preconditions, start effects, completion requirements and
  effects), `MissionTemplate` (procedural recipes: Delivery,
  CollectFromAsteroidField, CollectThenDeliver, BountyHunt),
  `MissionStatus`, `ObjectiveProgress`, and `MissionTarget` component.

- **missions/events.rs** — Game-world events consumed by the mission system:
  `PlayerLandedOnPlanet`, `PlayerEnteredSystem`, `PickupCollected`,
  `ShipDestroyed`, plus UI interaction events (Accept/Decline/Abandon/
  Started/Completed/Failed).

- **missions/log.rs** — `MissionLog` (per-player status map),
  `MissionCatalog` (all known mission defs), `MissionOffers` (ephemeral
  per-landing offers), and `PlayerUnlocks` (named unlock flags). All four
  implement `SessionResource` for automatic reset and (where appropriate)
  save/load.

- **missions/progress.rs** — Mission state machine: precondition checking,
  Locked-to-Available transitions, auto-start for Auto missions, objective
  advancement (travel, land, collect, destroy), completion/failure
  resolution with cargo effects and rewards, procedural template
  instantiation with placeholder substitution, and mission-target ship
  spawning.

- **missions/ui.rs** — Egui-based mission UI: active mission list with
  abandon buttons, available mission offers (accept), bar tab renderer,
  mission log overlay (I key), and completion/failure toast notifications.
  NPC-offered missions are accepted via surface NPC dialog
  (`surface_npc_chat`), not through this UI.

- **missions/tests.rs** — Unit tests for mission types, preconditions,
  requirements, cargo space calculations, YAML round-trips, and state
  machine scenarios.

## UI & Presentation

- **main_menu.rs** — Main menu screen (egui): new pilot creation and saved
  pilot loading.

- **planet_ui.rs** — Landed-state planet UI (egui): trade commodities,
  outfitter (buy/sell weapons and ammo), shipyard (buy ships), and bar tab
  (active mission log + available tab-offered contracts). Handles launch
  with take-off animation.

- **jump_ui.rs** — Star map overlay (egui): scrollable node graph of star
  systems with visited/discovered/jumpable states, mission objective markers,
  and click-to-jump. Includes the hyperspace flash full-screen overlay.

- **hud.rs** — In-flight HUD: radar display with parallax dots and target
  blinking, health bar, credits/cargo readout, target display, secondary
  weapon selector, comms ticker, and target reticle gizmos.

- **comms.rs** — Communications channel: generates context-appropriate radio
  chatter based on the player's nav target, varying by ship personality,
  distress state, cargo, and destination.

- **starfield.rs** — Multi-layer parallax starfield background with toroidal
  wrapping, origin-shift system to prevent floating-point drift, and camera
  follow.

- **sfx.rs** — Sound effects: weapon fire (per-weapon sounds), explosions
  (size-tiered), thruster loop, hyperspace jump, pickup collection, landing,
  target lock chime, and hostile lock-on warning.

## Tests

Test files live in `src/tests/` and are included via `#[path]` attributes:

- **tests/ai_ships_tests.rs** — AI ship control tests.
- **tests/game_save_tests.rs** — Save/load round-trip tests.
- **tests/item_universe_tests.rs** — Asset parsing validation.
- **tests/model_tests.rs** — Neural network forward-pass tests.
- **tests/optimal_control_tests.rs** — Control math tests.
- **tests/policy_tests.rs** — End-to-end policy inference tests.
- **tests/rl_collection_tests.rs** — RL collection and target selection tests.
- **tests/rl_obs_tests.rs** — Observation encoding tests.
- **tests/utils_tests.rs** — Utility function tests.
