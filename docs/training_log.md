# Training Log

## Architecture (run_15)
- Dual target: `nav_target` + `weapons_target` with two pointer heads
- 8 reward channels: landing, cargo_sold, weapon_hit, pickup, health, nav_target, weapons_target, movement
- Projectile←Entity cross-attention (projectiles attend to entities)
- BC loss weighted by `1/log(num_classes)`
- Movement reward: always positive, `proximity * vel_match * STEP_WEIGHT`, gated by `good_nav_target`
- Approach reward: scales with direction toward target outside threshold radius

## BC Training (run_15)
- Turn 93.9%, Thrust 96.1%, Target 98.8% after ~25 min
- Loss weighting critical for turn accuracy
- Projectile attention direction critical for action head learning

## RL Training (run_15)

### Bug fixes applied during training
1. **Distressed index bug** (from run_13): `obs[SELF_SIZE-1]` was trader personality flag, not distressed. Fixed to `obs[8]`.
2. **Trader nav_target too narrow**: Traders only got nav_target for planets (2 slots). Added pickups as valid targets (4 slots total). This was the key fix — trader nav_target went from ~0 to 0.001 effective.
3. **Approach `.max(1.0)` bug**: Should have been `.min(1.0)`. Fixed so approach term actually varies with heading direction.

### Reward balance at convergence (cycle 419, ~60 min RL)
| | health(×0.1) | weapon_hit | nav_target(×0.1) | movement(×0.1) | TOTAL |
|---|---|---|---|---|---|
| fighter | 0.018 | **0.002** | **0.003** | **0.002** | 0.026 |
| miner | 0.033 | 0.000 | **0.003** | **0.003** | 0.038 |
| trader | 0.029 | 0.000 | **0.001** | **0.001** | 0.031 |

### Training metrics at convergence
- Clip fraction: 0.05-0.08 (excellent)
- Entropy: 1.54-1.55 (stable)
- Total EV: 0.67-0.89
- Movement head EV: 0.91-0.94 (well learned)
- Policy loss: 0.004-0.008 (incremental)

### Key learnings
1. **BC loss weighting** (`1/log(C)`) essential — without it, 13-class target heads dominate
2. **Projectile attention direction** matters — Entity←Proj contaminates entity stream, Proj←Entity keeps entities clean
3. **Valid target count per personality** is critical — traders needed pickups as valid nav targets (2 slots → 4)
4. **Movement shaping works** — when gated by `good_nav_target`, it provides a strong dense gradient
5. **Entropy coeff 0.05** maintains exploration without destabilizing
6. **Landing/cargo_sold never emerged** — the approach→slow→land chain is too long for exploration to discover. May need curriculum, stronger velocity-matching near planets, or BC anchoring for landing states.

### Hyperparameters (final)
```
PPO_CLIP_EPS = 0.1
PPO_ENTROPY_COEFF = 0.05
PPO_POLICY_LR = 3e-4
PPO_VALUE_LR = 1e-3
PPO_POLICY_EPOCHS = 2
PPO_VALUE_EPOCHS = 4
PPO_MINI_BATCH_SIZE = 1024
PPO_MAX_SEGMENTS = 256
PPO_VALUE_BURNIN_EV_THRESHOLD = 0.3
REWARD_TYPE_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1]
MOVEMENT_STEP_WEIGHT = 0.05
MOVEMENT_LENGTH_SCALE = 200.0
MOVEMENT_THRESHOLD_DIST = 300.0
MOVEMENT_VEL_SCALE = 50.0
```
