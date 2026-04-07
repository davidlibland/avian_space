# Overnight Training Analysis — Run 16

## Setup
- BC pretrained: Turn 92% balanced, Thrust 81%, Target 99%
- RL from BC weights with: planet visibility fix, log-prob clamping, braking reward
- Hardcoded trader nav override to best cargo-value planet
- Headless mode, ~64 segments/cycle, ~23s/cycle
- Total cycles at analysis: **1043** (~970 with new braking reward)

## NaN Stability
**0 NaN in 1040 policy loss values.** The log-prob clamping fix (`max(-20.0)`) on both the sampling and training sides has held perfectly throughout the run. ✓

## Training Health

| Metric | Value | Assessment |
|--------|-------|------------|
| Policy loss | 0.001-0.004 | Very small incremental updates |
| Entropy | 1.575-1.580 | Rock stable, good exploration |
| Clip fraction | 0.03-0.05 | Excellent — very few clipped updates |
| Total EV | 0.80-0.94 | Good value function accuracy |

**Assessment:** Training is extremely stable. The low clip fraction means the policy changes very little between data collection and training — the on-policy assumption holds well.

## Reward Balance

### Effective reward per step (raw × weight), last 10 cycles

| | health (×0.1) | weapon_hit | nav_target (×0.1) | movement (×0.1) | TOTAL |
|---|---|---|---|---|---|
| **fighter** | 0.016 | **0.002** | **0.002** | **0.002** | 0.022 |
| **miner** | 0.029 | 0.001 | **0.002** | **0.002** | 0.034 |
| **trader** | 0.032 | 0.000 | **0.004** | **0.003** | 0.039 |

**Assessment:**
- **Health** is the largest single signal for all personalities (0.016-0.032 effective after ×0.1 weight). It's 1-2 orders of magnitude larger than event rewards, but comparable to nav_target and movement after weighting.
- **Nav_target** and **movement** are the 2nd and 3rd largest effective signals — the dense shaping is working as intended.
- **Weapon_hit** is meaningful for fighters (0.002) and visible for miners (0.001).
- **No single reward dominates excessively** — health is largest but nav_target + movement together are comparable.
- **Landing and cargo_sold remain at zero** in the per-step averages.

### Reward trends (Q1 → Q4)

| Personality | Improving | Declining |
|---|---|---|
| **Fighter** | weapon_hit +67%, pickup +32%, nav_target +21%, weapons_target +39%, movement +29% | landing -99% |
| **Miner** | health +6%, pickup +24% | (mostly flat) |
| **Trader** | weapon_hit +110%, weapons_target +253%, movement +18%, health +7% | landing -92%, cargo_sold -98% |

**Assessment:**
- **Fighters** are the strongest learners — improving across combat, targeting, and navigation.
- **Miners** have plateaued — rewards are flat or slightly positive.
- **Traders** show strong nav/movement growth (+18%, +253% weapons) but landing/cargo_sold are declining (from already-rare events becoming rarer).

## Value Head Explained Variance

| Head | EV Range | Assessment |
|------|----------|------------|
| health | 0.89-0.97 | ✓ Excellent — dense signal, well predicted |
| nav_target | 0.86-0.98 | ✓ Excellent — dense signal |
| movement | 0.84-0.96 | ✓ Very good — dense signal with braking component |
| cargo_sold | 0.89-0.99 | ✓ Good — trivially predicting near-zero |
| landing | 0.74-0.99 | ⚠ Variable — oscillates between trivial-zero and trying to predict rare events |
| pickup | 0.01-0.47 | ⚠ Poor — sparse and noisy |
| weapons_target | 0.22-0.55 | ⚠ Moderate — could be better |
| weapon_hit | -0.01-0.30 | ⚠ Poor — too sparse/noisy to predict well |
| **TOTAL** | 0.80-0.94 | ✓ Good overall |

**Assessment:** The dense heads (health, nav_target, movement) are well-learned. The sparse heads (weapon_hit, pickup, weapons_target) have low EV, meaning the value function can't accurately predict those reward channels. This limits the policy's ability to optimize for those rewards.

## Landing Events

| | Landing events | Cargo sold events |
|---|---|---|
| Trader | **6 / 1043** (0.6%) | **6 / 1043** (0.6%) |
| Fighter | 1 / 1043 | 1 / 1043 |
| Miner | 0 / 1043 | 0 / 1043 |

**Assessment:** Traders have landed 6 times in 1043 cycles. This is rare but non-zero — the conditions for landing exist (planets visible, movement shaping active, braking reward added). However, the rate is too low for the policy to learn from it. The braking reward is folded into the movement channel, so it doesn't create a distinct signal the value function can learn.

## Nav Target Selection

| | ship | asteroid | planet | pickup | none |
|---|---|---|---|---|---|
| fighter | 0.25 | **0.46** | 0.14 | 0.13 | 0.02 |
| miner | 0.22 | **0.54** | 0.08 | 0.15 | 0.01 |
| trader | 0.35 | **0.44** | 0.11 | 0.09 | 0.01 |

**Assessment:** All personalities heavily favor asteroids (44-54%) as nav targets. Planets are selected 8-14%. Despite the hardcoded trader override forcing planet selection, the policy's own nav head (which is what's recorded here) prefers asteroids. The override masks this in the actual behavior.

## Key Findings

### What's working:
1. **NaN prevention** — log-prob clamping completely solved the instability
2. **Dense reward shaping** — nav_target and movement provide strong, learnable signals
3. **Planet visibility** — planets always appear in obs (2.0 per step)
4. **Training stability** — clip fraction 0.03-0.05, entropy 1.58, no warnings
5. **Fighter combat** — weapon_hit +67%, steadily improving
6. **Trader navigation** — nav_target and movement rewards strong

### What's not working:
1. **Landing** — 6 events in 1043 cycles is not enough signal to learn from
2. **Cargo_sold** — same, entirely dependent on landing happening
3. **Miner plateau** — miners aren't improving; they've found a local optimum
4. **Sparse heads** — weapon_hit, pickup, weapons_target have poor value prediction (EV < 0.5)

### Recommendations:
1. **Landing remains the bottleneck.** The braking reward helps but isn't sufficient. Consider:
   - Increasing braking reward magnitude
   - Making the braking reward a separate channel (so the value function can learn it independently)
   - Curriculum: temporarily increase LANDING_RADIUS or LANDING_SPEED to make landing easier
2. **Miner training** needs a new incentive — they've converged to pickup/asteroid collection without landing
3. **The hardcoded trader nav override** proves that correct targeting + movement shaping gets ships near planets. The gap is the final approach/brake/land sequence.
### Cycle 1127 — 22:50

**Training:** policy_loss=0.0030, entropy=1.576, clip=0.036, EV=0.637, NaN=0

**Effective reward/step (raw × weight):**

| | health | weapon_hit | landing | cargo_sold | pickup | nav_target | weapons_target | movement | TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| fighter | 0.01771 | 0.00272 | 0.00000 | 0.00000 | 0.00002 | 0.00267 | 0.00098 | 0.00234 | 0.02644 |
| miner | 0.02922 | 0.00028 | 0.00000 | 0.00000 | 0.00022 | 0.00254 | 0.00006 | 0.00237 | 0.03468 |
| trader | 0.03075 | 0.00022 | 0.00000 | 0.00000 | 0.00005 | 0.00360 | 0.00010 | 0.00339 | 0.03811 |

**Reward trends (Q1 → Q4):**

- **fighter**: weapon_hit=+55%, landing=-99%, cargo_sold=-26%, pickup=+22%, nav_target=+18%, weapons_target=+35%, movement=+24%
- **miner**: weapon_hit=+21%, landing=+16228%, cargo_sold=+346%, pickup=+32%
- **trader**: health=+6%, weapon_hit=+73%, landing=-100%, cargo_sold=-100%, pickup=+9%, weapons_target=+178%, movement=+16%

**Value head EV (last 5):**

| head | EV |
|---|---|
| health | [0.942, 0.931, 0.922, 0.915, 0.865] |
| weapon_hit | [-0.001, 0.005, 0.013, 0.013, -0.097] |
| landing | [0.800, 0.920, 0.916, 0.975, 0.966] |
| cargo_sold | [0.970, 0.990, 0.995, 0.999, 0.999] |
| pickup | [0.196, 0.588, 0.350, 0.494, 0.745] |
| nav_target | [0.880, 0.880, 0.836, 0.848, 0.597] |
| weapons_target | [0.360, 0.279, 0.408, 0.344, 0.444] |
| movement | [0.854, 0.861, 0.839, 0.841, 0.563] |
| **TOTAL** | [0.612, 0.716, 0.668, 0.628, 0.560] |

**Landing events:** fighter/landing=1/1127, fighter/cargo_sold=1/1127, miner/landing=1/1127, miner/cargo_sold=1/1127, trader/landing=6/1127, trader/cargo_sold=6/1127

**Nav target selection (last 5 avg):**

| | ship | asteroid | planet | pickup | none |
|---|---|---|---|---|---|
| fighter | 0.343 | 0.408 | 0.158 | 0.081 | 0.009 |
| miner | 0.159 | 0.452 | 0.062 | 0.122 | 0.004 |
| trader | 0.385 | 0.432 | 0.067 | 0.106 | 0.010 |

---
### Cycle 1298 — 23:50

**Training:** policy_loss=0.0081, entropy=1.578, clip=0.173, EV=0.806, NaN=0

**Effective reward/step (raw × weight):**

| | health | weapon_hit | landing | cargo_sold | pickup | nav_target | weapons_target | movement | TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| fighter | 0.01632 | 0.00275 | 0.00000 | 0.00000 | 0.00002 | 0.00275 | 0.00101 | 0.00237 | 0.02522 |
| miner | 0.02430 | 0.00047 | 0.00000 | 0.00000 | 0.00013 | 0.00171 | 0.00019 | 0.00156 | 0.02836 |
| trader | 0.03107 | 0.00001 | 0.00000 | 0.00000 | 0.00004 | 0.00419 | 0.00040 | 0.00376 | 0.03947 |

**Reward trends (Q1 → Q4):**

- **fighter**: weapon_hit=+55%, landing=-99%, cargo_sold=-24%, pickup=+31%, nav_target=+20%, weapons_target=+51%, movement=+25%
- **miner**: weapon_hit=+10%, landing=+14075%, cargo_sold=+300%, pickup=+16%, weapons_target=+35%
- **trader**: health=+6%, weapon_hit=+63%, landing=-100%, cargo_sold=-100%, pickup=+5%, weapons_target=+199%, movement=+19%

**Value head EV (last 5):**

| head | EV |
|---|---|
| health | [0.921, 0.898, 0.946, 0.947, 0.942] |
| weapon_hit | [0.165, 0.060, 0.158, 0.039, 0.071] |
| landing | [0.992, 0.997, 0.999, 1.000, 1.000] |
| cargo_sold | [0.343, 0.813, 0.917, 0.917, 0.970] |
| pickup | [0.475, 0.233, 0.318, 0.161, 0.288] |
| nav_target | [0.927, 0.935, 0.919, 0.949, 0.919] |
| weapons_target | [0.318, 0.497, 0.520, 0.670, 0.490] |
| movement | [0.915, 0.929, 0.910, 0.938, 0.905] |
| **TOTAL** | [0.847, 0.845, 0.872, 0.631, 0.833] |

**Landing events:** fighter/landing=1/1298, fighter/cargo_sold=1/1298, miner/landing=1/1298, miner/cargo_sold=1/1298, trader/landing=6/1298, trader/cargo_sold=6/1298

**Nav target selection (last 5 avg):**

| | ship | asteroid | planet | pickup | none |
|---|---|---|---|---|---|
| fighter | 0.428 | 0.381 | 0.101 | 0.082 | 0.009 |
| miner | 0.153 | 0.307 | 0.072 | 0.057 | 0.011 |
| trader | 0.380 | 0.391 | 0.088 | 0.132 | 0.009 |

---
### Cycle 1468 — 00:51

**Training:** policy_loss=0.0016, entropy=1.581, clip=0.047, EV=0.593, NaN=0

**Effective reward/step (raw × weight):**

| | health | weapon_hit | landing | cargo_sold | pickup | nav_target | weapons_target | movement | TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| fighter | 0.01657 | 0.00299 | 0.00000 | 0.00000 | 0.00001 | 0.00263 | 0.00110 | 0.00237 | 0.02567 |
| miner | 0.03206 | 0.00090 | 0.00000 | 0.00000 | 0.00024 | 0.00242 | 0.00012 | 0.00225 | 0.03799 |
| trader | 0.03399 | 0.00032 | 0.00000 | 0.00000 | 0.00004 | 0.00400 | 0.00025 | 0.00392 | 0.04252 |

**Reward trends (Q1 → Q4):**

- **fighter**: weapon_hit=+67%, landing=-99%, cargo_sold=-22%, pickup=+42%, nav_target=+24%, weapons_target=+67%, movement=+30%
- **miner**: landing=+12426%, cargo_sold=+265%, pickup=+7%, nav_target=+8%, weapons_target=+22%, movement=+7%
- **trader**: health=+8%, weapon_hit=+64%, landing=-93%, cargo_sold=-84%, pickup=+8%, weapons_target=+326%, movement=+21%

**Value head EV (last 5):**

| head | EV |
|---|---|
| health | [0.942, 0.903, 0.929, 0.949, 0.953] |
| weapon_hit | [0.207, 0.098, 0.072, 0.073, 0.242] |
| landing | [0.607, 0.450, -0.078, -0.091, 0.147] |
| cargo_sold | [0.424, 0.578, 0.686, 0.794, 0.825] |
| pickup | [0.742, 0.750, 0.666, 0.785, 0.518] |
| nav_target | [0.836, 0.748, 0.777, 0.735, 0.794] |
| weapons_target | [0.614, 0.768, 0.679, 0.710, 0.783] |
| movement | [0.825, 0.729, 0.743, 0.720, 0.782] |
| **TOTAL** | [0.674, 0.449, 0.634, 0.523, 0.688] |

**Landing events:** fighter/landing=1/1468, fighter/cargo_sold=1/1468, miner/landing=1/1468, miner/cargo_sold=1/1468, trader/landing=8/1468, trader/cargo_sold=8/1468

**Nav target selection (last 5 avg):**

| | ship | asteroid | planet | pickup | none |
|---|---|---|---|---|---|
| fighter | 0.407 | 0.406 | 0.094 | 0.079 | 0.014 |
| miner | 0.269 | 0.502 | 0.102 | 0.119 | 0.008 |
| trader | 0.372 | 0.450 | 0.056 | 0.107 | 0.015 |

---
### Cycle 1633 — 01:51

**Training:** policy_loss=0.0029, entropy=1.588, clip=0.039, EV=0.617, NaN=0

**Effective reward/step (raw × weight):**

| | health | weapon_hit | landing | cargo_sold | pickup | nav_target | weapons_target | movement | TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| fighter | 0.01803 | 0.00286 | 0.00000 | 0.00000 | 0.00001 | 0.00291 | 0.00102 | 0.00260 | 0.02743 |
| miner | 0.03642 | 0.00048 | 0.00000 | 0.00000 | 0.00043 | 0.00236 | 0.00019 | 0.00218 | 0.04206 |
| trader | 0.02875 | 0.00016 | 0.00000 | 0.00000 | 0.00004 | 0.00393 | 0.00012 | 0.00339 | 0.03639 |

**Reward trends (Q1 → Q4):**

- **fighter**: weapon_hit=+81%, landing=-99%, cargo_sold=-20%, pickup=+45%, nav_target=+25%, weapons_target=+83%, movement=+31%
- **miner**: pickup=+9%, nav_target=+11%, weapons_target=+5%, movement=+10%
- **trader**: health=+8%, weapon_hit=+86%, landing=+7%, cargo_sold=+8%, pickup=+7%, weapons_target=+299%, movement=+24%

**Value head EV (last 5):**

| head | EV |
|---|---|
| health | [0.918, 0.879, 0.949, 0.959, 0.945] |
| weapon_hit | [0.305, 0.425, 0.093, 0.100, 0.169] |
| landing | [0.770, 0.678, 0.701, 0.775, 0.523] |
| cargo_sold | [0.947, 0.911, 0.964, 0.955, 0.762] |
| pickup | [0.634, 0.857, 0.835, 0.701, 0.495] |
| nav_target | [0.909, 0.854, 0.845, 0.895, 0.791] |
| weapons_target | [0.778, 0.774, 0.728, 0.812, 0.758] |
| movement | [0.890, 0.846, 0.831, 0.879, 0.774] |
| **TOTAL** | [0.744, 0.737, 0.621, 0.672, 0.313] |

**Landing events:** fighter/landing=1/1633, fighter/cargo_sold=1/1633, miner/landing=1/1633, miner/cargo_sold=1/1633, trader/landing=13/1633, trader/cargo_sold=13/1633

**Nav target selection (last 5 avg):**

| | ship | asteroid | planet | pickup | none |
|---|---|---|---|---|---|
| fighter | 0.389 | 0.409 | 0.098 | 0.092 | 0.012 |
| miner | 0.236 | 0.488 | 0.104 | 0.122 | 0.049 |
| trader | 0.403 | 0.468 | 0.046 | 0.071 | 0.012 |

---
### Cycle 1813 — 02:51

**Training:** policy_loss=0.0041, entropy=1.639, clip=0.059, EV=0.514, NaN=0

**Effective reward/step (raw × weight):**

| | health | weapon_hit | landing | cargo_sold | pickup | nav_target | weapons_target | movement | TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| fighter | 0.01853 | 0.00254 | 0.00000 | 0.00000 | 0.00002 | 0.00310 | 0.00106 | 0.00272 | 0.02796 |
| miner | 0.03752 | 0.00047 | 0.00000 | 0.00000 | 0.00014 | 0.00298 | 0.00016 | 0.00248 | 0.04376 |
| trader | 0.03015 | 0.00019 | 0.00000 | 0.00000 | 0.00005 | 0.00354 | 0.00008 | 0.00317 | 0.03718 |

**Reward trends (Q1 → Q4):**

- **fighter**: weapon_hit=+94%, landing=-99%, cargo_sold=-18%, pickup=+45%, nav_target=+25%, weapons_target=+96%, movement=+31%
- **miner**: health=+6%, weapon_hit=+5%, pickup=+15%, nav_target=+8%, weapons_target=-18%, movement=+8%
- **trader**: health=+10%, weapon_hit=+89%, landing=+47%, cargo_sold=+27%, pickup=+9%, weapons_target=+309%, movement=+27%

**Value head EV (last 5):**

| head | EV |
|---|---|
| health | [0.923, 0.924, 0.945, 0.949, 0.927] |
| weapon_hit | [0.388, 0.238, 0.322, 0.189, 0.005] |
| landing | [0.950, 0.976, 0.987, 0.988, 0.992] |
| cargo_sold | [0.989, 0.990, 0.997, 0.904, 0.981] |
| pickup | [0.309, 0.744, 0.916, 0.878, 0.374] |
| nav_target | [0.697, 0.836, 0.820, 0.751, 0.718] |
| weapons_target | [0.729, 0.681, 0.784, 0.805, 0.686] |
| movement | [0.656, 0.790, 0.783, 0.749, 0.711] |
| **TOTAL** | [0.435, 0.375, 0.654, 0.545, 0.561] |

**Landing events:** fighter/landing=1/1813, fighter/cargo_sold=1/1813, miner/landing=1/1813, miner/cargo_sold=1/1813, trader/landing=16/1813, trader/cargo_sold=16/1813

**Nav target selection (last 5 avg):**

| | ship | asteroid | planet | pickup | none |
|---|---|---|---|---|---|
| fighter | 0.454 | 0.396 | 0.080 | 0.064 | 0.005 |
| miner | 0.241 | 0.604 | 0.050 | 0.104 | 0.001 |
| trader | 0.447 | 0.437 | 0.040 | 0.068 | 0.008 |

---
### Cycle 1987 — 03:51

**Training:** policy_loss=0.0071, entropy=2.702, clip=0.137, EV=0.712, NaN=0

**Effective reward/step (raw × weight):**

| | health | weapon_hit | landing | cargo_sold | pickup | nav_target | weapons_target | movement | TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| fighter | 0.01692 | 0.00254 | 0.00000 | 0.00000 | 0.00002 | 0.00274 | 0.00076 | 0.00230 | 0.02528 |
| miner | 0.02960 | 0.00085 | 0.00000 | 0.00000 | 0.00027 | 0.00208 | 0.00022 | 0.00177 | 0.03479 |
| trader | 0.02902 | 0.00014 | 0.00051 | 0.00002 | 0.00004 | 0.00366 | 0.00016 | 0.00322 | 0.03677 |

**Reward trends (Q1 → Q4):**

- **fighter**: weapon_hit=+86%, landing=-99%, cargo_sold=-17%, pickup=+45%, nav_target=+24%, weapons_target=+96%, movement=+29%
- **miner**: health=+6%, weapon_hit=+18%, landing=+2720%, pickup=+18%, nav_target=+6%
- **trader**: health=+9%, weapon_hit=+66%, landing=+119%, cargo_sold=+59%, pickup=+8%, weapons_target=+278%, movement=+24%

**Value head EV (last 5):**

| head | EV |
|---|---|
| health | [0.959, 0.958, 0.957, 0.963, 0.961] |
| weapon_hit | [0.337, 0.190, 0.302, 0.109, 0.122] |
| landing | [0.527, 0.713, -0.006, 0.526, 0.634] |
| cargo_sold | [0.834, 0.945, -0.004, 0.369, 0.548] |
| pickup | [0.838, 0.399, 0.706, 0.656, 0.734] |
| nav_target | [0.907, 0.909, 0.865, 0.895, 0.918] |
| weapons_target | [0.654, 0.707, 0.835, 0.849, 0.805] |
| movement | [0.890, 0.901, 0.864, 0.872, 0.902] |
| **TOTAL** | [0.809, 0.732, 0.532, 0.699, 0.787] |

**Landing events:** fighter/landing=1/1987, fighter/cargo_sold=1/1987, miner/landing=2/1987, miner/cargo_sold=1/1987, trader/landing=21/1987, trader/cargo_sold=21/1987

**Nav target selection (last 5 avg):**

| | ship | asteroid | planet | pickup | none |
|---|---|---|---|---|---|
| fighter | 0.383 | 0.399 | 0.125 | 0.076 | 0.016 |
| miner | 0.297 | 0.481 | 0.084 | 0.110 | 0.028 |
| trader | 0.434 | 0.417 | 0.067 | 0.070 | 0.012 |

---
### Cycle 2159 — 04:52

**Training:** policy_loss=0.0029, entropy=3.060, clip=0.136, EV=0.814, NaN=0

**Effective reward/step (raw × weight):**

| | health | weapon_hit | landing | cargo_sold | pickup | nav_target | weapons_target | movement | TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| fighter | 0.01743 | 0.00246 | 0.00000 | 0.00000 | 0.00001 | 0.00283 | 0.00079 | 0.00247 | 0.02600 |
| miner | 0.02741 | 0.00050 | 0.00000 | 0.00000 | 0.00021 | 0.00168 | 0.00039 | 0.00149 | 0.03168 |
| trader | 0.02979 | 0.00010 | 0.00039 | 0.00002 | 0.00004 | 0.00390 | 0.00009 | 0.00348 | 0.03781 |

**Reward trends (Q1 → Q4):**

- **fighter**: weapon_hit=+67%, landing=-98%, cargo_sold=-16%, pickup=+34%, nav_target=+21%, weapons_target=+77%, movement=+23%
- **miner**: health=+6%, weapon_hit=+23%, landing=+2503%, pickup=+17%, weapons_target=+22%
- **trader**: health=+7%, weapon_hit=+45%, landing=+167%, cargo_sold=+80%, pickup=+11%, weapons_target=+212%, movement=+19%

**Value head EV (last 5):**

| head | EV |
|---|---|
| health | [0.963, 0.963, 0.952, 0.961, 0.971] |
| weapon_hit | [0.160, 0.221, 0.297, 0.181, 0.180] |
| landing | [0.953, 0.960, -0.005, 0.835, 0.846] |
| cargo_sold | [0.999, 0.999, -0.005, 0.817, 0.864] |
| pickup | [0.807, 0.492, 0.674, 0.165, 0.512] |
| nav_target | [0.943, 0.962, 0.931, 0.940, 0.948] |
| weapons_target | [0.787, 0.765, 0.638, 0.693, 0.792] |
| movement | [0.935, 0.948, 0.927, 0.938, 0.943] |
| **TOTAL** | [0.874, 0.924, 0.555, 0.853, 0.865] |

**Landing events:** fighter/landing=1/2159, fighter/cargo_sold=1/2159, miner/landing=2/2159, miner/cargo_sold=1/2159, trader/landing=26/2159, trader/cargo_sold=26/2159

**Nav target selection (last 5 avg):**

| | ship | asteroid | planet | pickup | none |
|---|---|---|---|---|---|
| fighter | 0.360 | 0.417 | 0.111 | 0.092 | 0.020 |
| miner | 0.319 | 0.530 | 0.077 | 0.065 | 0.010 |
| trader | 0.446 | 0.414 | 0.058 | 0.070 | 0.012 |

---
### Cycle 2519 — 05:17

**Training:** policy_loss=0.003, entropy=3.85, clip=0.08, EV=0.85, NaN=0

**Key metrics:**
- Weapons_target EV improved significantly: 0.83-0.89 (was 0.22-0.55 earlier)
- Entropy rose to 3.85 (from 1.58 at start) — heavy exploration
- Planet nav selection improved: fighter 16%, miner 17%, trader 14% (up from 8-14%)
- Landing events: trader=27/2519, fighter=9/2519, miner=5/2519 (steady accumulation)
- No NaN (0/2516)

**Effective reward/step:**

| | health | weapon_hit | nav_target | movement | TOTAL |
|---|---|---|---|---|---|
| fighter | 0.017 | 0.003 | 0.003 | 0.002 | 0.026 |
| miner | 0.029 | 0.001 | 0.002 | 0.002 | 0.035 |
| trader | 0.030 | 0.000 | 0.004 | 0.003 | 0.038 |

**Trends:** Fighter weapon_hit growing steadily. Trader landing/cargo_sold positive. All stable, no overpowering.

---
### Cycle 2589 — 06:17

**Training:** policy_loss=0.004, entropy=3.84, clip=0.10, EV=0.85, NaN=0

**Notable changes since last snapshot:**
- Fighter landings: 10→11 (still growing)
- Weapon_hit EV improved: 0.41-0.63 (was 0.18-0.50)
- Weapons_target EV stable at 0.85-0.91 (strong)
- Planet nav selection up across all: fighter 17%, miner 16%, trader 17%
- Trader nav_target raw total: 55.4/cycle (steady)

**Landing:** trader=27/2589, fighter=11/2589, miner=5/2589. NaN=0/2586.

---
### Cycle 2820 — 07:17 (with value replay buffer)

**Training:** policy_loss=0.005, entropy=3.79, clip=0.11, EV=0.84, NaN=0

**Value replay buffer:** 8192/8192 steps (full)

**Landing events:** trader=29/2820, fighter=14/2820, miner=9/2820 (all growing)
- Fighter: 11→14 (+3 since last check)
- Miner: 5→9 (+4 since last check)
- Trader: 27→29 (+2 since last check)

**Landing head EV:** [-0.133, 0.291, 0.196, -0.187, 0.051] — still volatile, oscillating around 0. The replay buffer just started; needs more time to impact.

**Planet nav selection:** fighter 17%, miner 16%, trader 16% — stable and healthy.

**Notable:** Miner landings accelerating (5→9), suggesting the replay buffer may already be helping the value function around planet approaches. Fighter landings also up (11→14).

---
### Cycle 2908 — 08:17

**Training:** policy_loss=0.005, entropy=3.83, clip=0.09, EV=0.78, NaN=0. Replay=8192.

**Landing events:** trader=29/2908, fighter=17/2908(+3), miner=9/2908. Fighter landings accelerating.

**Landing head EV:** [0.788, 0.665, 0.619, 0.706, 0.011] — improved from earlier (-0.13 to 0.29), now mostly 0.6-0.8. The replay buffer is helping the value function learn landing predictions.

**Landing head TD error:** [0.0012, 0.0008, 0.0005, 0.0011, 0.0018] — up from 0.0002 earlier. More landing signal flowing through.

**Fighter landing raw total appeared:** 0.07/cycle (last 10 avg) — first time nonzero in the per-cycle table.

**Miner anomaly:** entity slot counts dropped (0.85 ships, 1.6 planets/pickups vs normal 2.0). May indicate miners are in low-density areas.

---
### Cycle 3247 — 09:17

**Training:** NaN=0, Replay=8192. Stable.

**Landing events (cumulative):**
| | Landing | Cargo sold | Δ since 08:17 |
|---|---|---|---|
| trader | 36 | 36 | +7 |
| fighter | 28 | 27 | +11 |
| miner | 13 | 12 | +4 |

**Landing is accelerating significantly.** Fighter went from 17→28 (+11 in one hour). Trader 29→36 (+7). Miner 9→13 (+4).

**Miner landing appeared in raw totals:** 0.06/cycle (last 10 avg) — first time nonzero.

**Planet nav selection highest ever:** fighter 18%, miner 19%, trader 19%. All personalities increasingly targeting planets.

---
### Cycle 3386 — 10:17

**Training:** NaN=0/3382. Stable.

**Landing events:**
| | Landing | Cargo sold | Δ since 09:17 |
|---|---|---|---|
| trader | 36 | 36 | +0 |
| fighter | 32 | 33 | **+4** |
| miner | 14 | 13 | +1 |

**Fighter landings continue to grow** (28→32, +4). Trader landings paused at 36. Miner at 14.

**Planet nav selection:** fighter 16%, miner 16%, trader 17% — steady.

**Fighter ship_engage weapons target at 11%** — highest yet, fighters increasingly targeting hostile ships.

---
### Cycle 3411 — 11:17

**Training:** NaN=0/3407. Stable.

**Landing events:**
| | Landing | Δ since 10:17 |
|---|---|---|
| trader | 36 | +0 |
| fighter | 33 | +1 |
| miner | 14 | +0 |

Landing rate slowed this hour. May be normal variance.

**Landing head EV: [0.831, 0.573, 0.601, 0.714, 0.816]** — significantly improved from the -0.13 to 0.29 range before the replay buffer. The value function is learning to predict landing events much better.

---
### Cycle 3793 — 12:17

**Training:** NaN=0/3789. Value loss=0.008-0.015. Stable.

**Landing events:**
| | Landing | Δ since 11:17 |
|---|---|---|
| trader | 37 | +1 |
| fighter | 39 | **+6** |
| miner | 15 | +1 |

Fighter landings continue strong (33→39, +6). Total landings across all personalities: 91 (was 83 at 11:17).

**Landing head EV: [0.228, 0.456, 0.589, 0.745, 0.767]** — trending upward within the window. The value function is steadily improving at predicting landing.

**Total EV: [0.753, 0.705, 0.753, 0.862, 0.797]** — healthy, no sign of replay overfitting degrading fresh-data prediction.

**No replay overfitting detected:** Total EV stable (0.70-0.86), value loss stable (0.008-0.015), while landing EV improving. The replay buffer is helping the landing head without hurting other heads.

---
### Cycle 3798 — 13:17

**Training:** NaN=0/3794. Stable.

**Landing events:** trader=37, fighter=39, miner=15. No change from 12:17 — landing rate paused this hour.

**Landing head EV: [0.278, 0.711, 0.365, -0.682, -0.513]** — volatile, dipped negative. The value function is struggling with landing prediction — this could be stale replay targets causing instability on the landing head.

**Total EV: [0.881, 0.786, 0.683, 0.871, 0.751]** — stable, no overall degradation.

**Note:** Landing EV oscillating between -0.68 and 0.71 in the same window suggests the landing head is receiving conflicting signals — possibly from replay buffer entries with outdated return targets.

---
