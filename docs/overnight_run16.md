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
### Cycle 4346 — 14:17

**Training:** NaN=0/4342. Entropy=3.88. Stable.

**Landing events:**
| | Landing | Δ since 13:17 |
|---|---|---|
| trader | 40 | +3 |
| fighter | 44 | **+5** |
| miner | 22 | **+7** |

**All personalities landing more.** Miner had the biggest jump (+7, from 15→22). Total landings: 106 (was 91 at 13:17, +15 in one hour).

**Landing head EV: [0.816, 0.611, 0.767, 0.809, 0.651]** — recovered from the -0.68 dip, back in the 0.6-0.8 range. The oscillation was transient.

**Total EV: [0.749, 0.726, 0.877, 0.817, 0.818]** — healthy, no replay degradation.

**Landing rate summary (cumulative):**
- Hour 09: 83 total → Hour 10: 83 → Hour 11: 91 → Hour 12: 91 → Hour 13: 91 → Hour 14: **106**
- Rate accelerating again after a pause.

---
### Cycle 4367 — 15:17

**Training:** NaN=0/4363. Stable.

**Landing events:**
| | Landing | Δ since 14:17 |
|---|---|---|
| trader | 40 | +0 |
| fighter | 45 | +1 |
| miner | 22 | +0 |

Quiet hour for landing (+1 total). Normal variance — landings come in bursts.

**Landing EV: [0.556, 0.395, 0.426, 0.881, 0.681]** — variable but generally positive (0.4-0.9 range).

**Total EV: [0.644, 0.613, 0.653, 0.466, 0.674]** — lower than usual. One cycle dipped to 0.47. Worth watching — could be transient noise or early sign of value function instability.

**Cumulative landing totals:** trader=40, fighter=45, miner=22. Total=107.

---
### Cycle 4586 — 16:17

**Training:** NaN=0/4582. Stable.

**Landing events:**
| | Landing | Δ since 15:17 |
|---|---|---|
| trader | 42 | +2 |
| fighter | 46 | +1 |
| miner | 24 | +2 |

Total landings: 112 (+5 this hour). Steady accumulation across all personalities.

**Landing head EV: [0.737, 0.858, 0.784, 0.634, 0.854]** — mostly 0.6-0.9, much more stable than earlier oscillations.

**Total EV: [0.772, 0.836, 0.724, 0.832, 0.848]** — healthy.

**Cumulative landing trajectory:**
| Hour | 09:17 | 10:17 | 11:17 | 12:17 | 13:17 | 14:17 | 15:17 | 16:17 |
|---|---|---|---|---|---|---|---|---|
| Total | 83 | 83 | 91 | 91 | 91 | 106 | 107 | **112** |

---
### Cycle 4863 — 17:17

**Training:** NaN=0/4859. Stable.

**Landing events:**
| | Landing | Δ since 16:17 |
|---|---|---|
| trader | 45 | +3 |
| fighter | 49 | +3 |
| miner | 27 | +3 |

Total landings: 121 (+9 this hour). Even distribution across all personalities (+3 each).

**Landing EV: [0.242, 0.660, 0.346, 0.641, 0.281]** — still oscillating but positive.

**Landing rate per hour (recent):**
| Hour | 14:17 | 15:17 | 16:17 | 17:17 |
|---|---|---|---|---|
| Δ landings | +15 | +1 | +5 | **+9** |

---
### Cycle 4911 — 18:17

**Training:** NaN=0/4907. Stable.

**Landing events:**
| | Landing | Δ since 17:17 |
|---|---|---|
| trader | 45 | +0 |
| fighter | 52 | +3 |
| miner | 27 | +0 |

Total landings: 124 (+3 this hour, all fighters).

**Landing EV: [-0.273, 0.528, -0.007, 0.399, -0.002]** — dipped negative again. Landing head continues to oscillate. This is likely due to stale replay targets — the replay buffer stores returns from older policy states.

**Total EV: [0.815, 0.767, 0.706, 0.804, 0.751]** — stable, no overall degradation.

**Summary:** Training stable, landing accumulating at ~5-10/hour average. Fighter is the most frequent lander (52 total). Landing head EV volatile but not causing training issues.

---
### Cycle 4929 — 21:03

**Training:** policy_loss=0.0008, entropy=3.852, clip=0.066, EV=0.809, NaN=0

**Landing:** fighter=53/4929, miner=29/4929, trader=46/4929 (total=128)

**Landing EV:** [0.055, 0.098, 0.650, -0.870, 0.502]

**Effective reward/step (last 20):**

- **fighter**: health=0.0169, weapon_hit=0.0020, nav_target=0.0022, weapons_target=0.0007, movement=0.0018
- **miner**: health=0.0293, nav_target=0.0020, movement=0.0018
- **trader**: health=0.0295, nav_target=0.0037, movement=0.0031

---
### Cycle 5132 — 22:30

**Training:** policy_loss=0.0004, entropy=3.867, clip=0.064, EV=0.615, NaN=0

**Landing:** fighter=62/5132, miner=40/5132, trader=50/5132 (total=152)

**Landing EV:** [0.292, -0.068, -0.003, -0.491, -2.452]

**Effective reward/step (last 20):**

- **fighter**: health=0.0174, weapon_hit=0.0026, nav_target=0.0022, weapons_target=0.0006, movement=0.0018
- **miner**: health=0.0266, nav_target=0.0017, movement=0.0015
- **trader**: health=0.0307, nav_target=0.0041, movement=0.0033

---
### Cycle 6595 — 22:47 (thorough check)

**Training health:** NaN=0/6591. Entropy=3.88. Clip=0.08. Total EV=0.75-0.85. Stable.

**Landing events — significant growth:**
| | Landing | Cargo sold | Rate |
|---|---|---|---|
| fighter | **117** | 123 | 1.8% |
| miner | **67** | 64 | 1.0% |
| trader | **65** | 65 | 1.0% |
| **TOTAL** | **249** | 252 | — |

Landing more than doubled since last thorough check (was 124 at 18:17). Fighter is the most frequent lander. All personalities participating.

**Reward dominance (% of effective total):**
| | health | weapon_hit | nav_target | movement | pickup | weapons_tgt |
|---|---|---|---|---|---|---|
| fighter | 22% | **19%** | **28%** | 23% | 2% | 7% |
| miner | 27% | 4% | **21%** | 18% | **28%** | 1% |
| trader | 30% | 0% | **38%** | **28%** | 3% | 0% |

**Assessment:** Well-balanced. No single reward dominates excessively. Health is 22-30% (was 74-85% in early runs). Each personality has distinct reward profiles matching their role.

**Growth (first 200 → last 200 cycles):**
- **Fighter:** weapon_hit=+30%, **landing=+501%**, cargo_sold=+353%, pickup=+10%
- **Miner:** health=+6%, pickup=+26%, weapons_target=+29%
- **Trader:** weapon_hit=+39%, weapons_target=+114%

**Fighter landing growth is exceptional (+501%).** Miner growth is moderate but steady. Trader landing growth is flat (-8%) — traders land at a steady rate but aren't accelerating.

**Value head EV:**
| Head | Last 5 | Assessment |
|------|--------|------------|
| health | 0.92-0.96 | ✓ Excellent |
| nav_target | 0.92-0.96 | ✓ Excellent |
| movement | 0.90-0.94 | ✓ Very good |
| landing | -0.12 to 0.59 | ⚠ Volatile — replay buffer helps but still oscillating |
| cargo_sold | 0.32-0.73 | ⚠ Moderate |
| weapons_target | 0.19-0.60 | ⚠ Declining from 0.83-0.89 earlier |
| pickup | 0.03-0.29 | ⚠ Poor — too sparse |
| weapon_hit | -0.11 to 0.15 | ⚠ Poor — too sparse/stochastic |
| **TOTAL** | 0.75-0.85 | ✓ Healthy |

**Concerns:**
1. **weapons_target EV declined** from 0.83-0.89 to 0.19-0.60. The value function is losing accuracy on this head. May be due to replay buffer displacing weapons_target training signal.
2. **weapon_hit EV consistently near zero** — this head never learned well, the signal is too stochastic.
3. **GPU throughput variable** (131-366 steps/sec) — possibly thermal throttling or background load.

**No action needed.** Training is healthy, landing is growing, rewards are balanced. Continue monitoring.

---
### Cycle 6602 — 07:07

**Training:** policy_loss=0.0010, entropy=3.832, clip=0.062, EV=0.759, NaN=0

**Landing:** fighter=117/6602, miner=67/6602, trader=65/6602 (total=249)

**Landing EV:** [0.627, 0.468, 0.226, 0.793, 0.521]

**Effective reward/step (last 20):**

- **fighter**: health=0.0167, weapon_hit=0.0017, nav_target=0.0021, weapons_target=0.0005, movement=0.0017
- **miner**: health=0.0275, weapon_hit=0.0005, nav_target=0.0020, movement=0.0017
- **trader**: health=0.0289, nav_target=0.0039, movement=0.0028

---
### Cycle 6625 — 23:47 (thorough check)

**Training health:** NaN=0/6621. Entropy=3.82. Clip=0.09. Total EV=0.59-0.83. Stable.

**Landing events:**
| | Landing | Cargo sold |
|---|---|---|
| fighter | 117 | 123 |
| miner | 67 | 64 |
| trader | 65 | 65 |
| **TOTAL** | **249** | **252** |

No new landings since 22:47 check (same counts). Landing rate has paused — may resume in subsequent hours.

**Reward dominance:**
| | health | weapon_hit | nav_target | movement | pickup | weapons_tgt |
|---|---|---|---|---|---|---|
| fighter | 22% | **21%** | **27%** | 22% | 1% | 7% |
| miner | 29% | 4% | **21%** | 17% | **25%** | 3% |
| trader | 28% | 1% | **40%** | **27%** | 3% | 0% |

**Assessment:** Well-balanced, consistent with prior checks. Each personality maintains distinct reward profiles.

**Growth (first 200 → last 200):**
- Fighter: weapon_hit=+27%, **landing=+501%**, cargo_sold=+353%
- Miner: pickup=+30%, weapons_target=+44%
- Trader: weapon_hit=+35%, weapons_target=+80%

**Value head EV:**
| Head | Last 5 | Trend |
|------|--------|-------|
| health | 0.91-0.96 | ✓ Stable |
| nav_target | 0.90-0.97 | ✓ Stable |
| movement | 0.88-0.96 | ✓ Stable |
| weapons_target | 0.39-0.72 | ⚠ Recovering from earlier dip |
| landing | -0.00 to 0.63 | ⚠ Volatile but mostly positive |
| cargo_sold | 0.43-0.76 | → Moderate |
| weapon_hit | 0.06-0.70 | ⚠ Highly variable |
| pickup | -0.05 to 0.18 | ⚠ Poor |
| **TOTAL** | 0.59-0.83 | → Healthy |

**Concerns:**
1. Landing rate paused at 249 — may be normal variance (landings come in bursts)
2. Total EV lower end (0.59) — some cycles have poor value prediction
3. Pickup head EV near zero — this head isn't learning

**No action needed.** Training stable, rewards balanced, landing accumulated. Continue overnight.

---
### Cycle 6758 — 08:05

**Training:** policy_loss=0.0004, entropy=3.769, clip=0.056, EV=0.743, NaN=0

**Landing:** fighter=120/6758, miner=68/6758, trader=65/6758 (total=253)

**Landing EV:** [0.920, 0.347, 0.009, -0.173, 0.411]

**Effective reward/step (last 20):**

- **fighter**: health=0.0167, weapon_hit=0.0020, nav_target=0.0020, movement=0.0016
- **miner**: health=0.0212, nav_target=0.0015, movement=0.0012
- **trader**: health=0.0277, nav_target=0.0037, movement=0.0025

---
### Cycle 7455 — 12:04

**Training:** policy_loss=0.0001, entropy=3.079, clip=0.041, EV=0.490, NaN=0

**Landing:** fighter=149/7455, miner=85/7455, trader=75/7455 (total=309)

**Landing EV:** [0.456, 0.208, -0.459, 0.198, -0.455]

**Effective reward/step (last 20):**

- **fighter**: health=0.0178, weapon_hit=0.0020, nav_target=0.0021, weapons_target=0.0006, movement=0.0017
- **miner**: health=0.0332, weapon_hit=0.0006, nav_target=0.0023, movement=0.0020
- **trader**: health=0.0294, nav_target=0.0039, movement=0.0031

---
### Cycle 7190 — 00:47

**Training:** NaN=0/7186. Entropy=3.79. Clip=0.07. Stable.

**Landing events:** fighter=143, miner=79, trader=70. **Total=292** (+43 since 22:47).

**Landing growing steadily across all personalities.** Fighter +26, miner +12, trader +5 since last thorough check.

**Growth:** Fighter landing=+596%, cargo_sold=+626%. Miner landing=+301518% (from near-zero). Fighter weapon_hit=+58%.

**Landing EV: [0.398, -0.325, 0.210, -0.044, -0.219]** — dipped negative again. The oscillation continues — replay buffer helps but stale targets cause instability.

**Total EV: [0.47, 0.47, 0.80, 0.78, 0.73]** — some low cycles (0.47).

---
### Cycle 7463 — 02:47

**Training:** NaN=0/7459. Entropy=3.86. Stable.

**Landing events:** fighter=149(+6), miner=85(+6), trader=75(+5). **Total=309** (+17 since 00:47).

**Reward balance:** Unchanged — fighter weapon_hit=24%, nav_target=26%; miner pickup=27%, nav_target=21%; trader nav_target=38%, movement=27%. Well-balanced.

**Landing EV: [0.737, 0.023, 0.390, -0.830, 0.382]** — still highly volatile, one cycle hit -0.83.

**Value head summary:**
- Dense heads stable: health 0.90-0.94, nav_target 0.84-0.93, movement 0.85-0.92
- weapons_target recovering: 0.50-0.77
- Landing/cargo_sold volatile: oscillating between -0.83 and 0.74
- weapon_hit/pickup remain poor (<0.34)

**No action needed.** Landing continues accumulating (~10-20/hour). Training stable, no NaN, rewards balanced.

---
### Cycle 7469 — 12:09

**Training:** policy_loss=0.0002, entropy=3.863, clip=0.073, EV=0.601, NaN=0

**Landing:** fighter=149/7469, miner=85/7469, trader=75/7469 (total=309)

**Landing EV:** [-0.147, 0.420, 0.350, 0.566, 0.502]

**Effective reward/step (last 20):**

- **fighter**: health=0.0177, weapon_hit=0.0024, nav_target=0.0022, weapons_target=0.0007, movement=0.0018
- **miner**: health=0.0251, weapon_hit=0.0006, nav_target=0.0017, movement=0.0014
- **trader**: health=0.0294, nav_target=0.0037, movement=0.0030

---
### Cycle 7571 — 03:47 (thorough check)

**Training:** NaN=0/7567. Entropy=3.87. Clip=0.07. Stable.

**Landing events:**
| | Landing | Cargo sold | Δ since 02:47 |
|---|---|---|---|
| fighter | 154 | 160 | +5 |
| miner | 86 | 83 | +1 |
| trader | 75 | 75 | +0 |
| **TOTAL** | **315** | **318** | **+6** |

Landing continues at ~6/hour. Fighter remains the most frequent lander. Trader has plateaued at 75.

**Reward dominance:**
| | health | weapon_hit | nav_target | movement | pickup | weapons_tgt |
|---|---|---|---|---|---|---|
| fighter | 19% | **27%** | 24% | 19% | 1% | 9% |
| miner | 30% | 7% | 21% | 18% | **21%** | 3% |
| trader | 29% | 1% | **36%** | **30%** | 3% | 0% |

**Notable:** Fighter weapon_hit has risen to 27% dominance (was 19-21%). Fighters are becoming more combat-focused. Fighter weapons_target also up to 9%.

**Growth (first 200 → last 200):**
- Fighter: weapon_hit=+58%, **landing=+421%**, cargo_sold=+355%, weapons_target=+22%
- Miner: weapon_hit=+24%, **landing=+76861%**, weapons_target=+86%, pickup=+16%
- Trader: weapon_hit=+59%, weapons_target=+146%

**Value head EV:**
| Head | Last 5 | Assessment |
|------|--------|------------|
| health | 0.92-0.94 | ✓ Stable |
| nav_target | 0.90-0.93 | ✓ Stable |
| movement | 0.89-0.92 | ✓ Stable |
| cargo_sold | 0.43-0.76 | → Moderate, variable |
| weapons_target | 0.49-0.71 | → Moderate |
| landing | 0.03-0.58 | ⚠ Still volatile |
| weapon_hit | 0.13-0.37 | ⚠ Poor |
| pickup | 0.03-0.17 | ⚠ Poor |
| **TOTAL** | 0.61-0.78 | → OK |

**Assessment:** Training is healthy and rewards are well-balanced. Fighter combat dominance is increasing (+27% weapon_hit share). Landing accumulates steadily. No concerns — continue training.

---
### Cycle 7629 — 13:11

**Training:** policy_loss=0.0001, entropy=3.867, clip=0.075, EV=0.625, NaN=0

**Landing:** fighter=154/7629, miner=88/7629, trader=75/7629 (total=317)

**Landing EV:** [0.503, 0.382, 0.648, -0.349, -0.595]

**Effective reward/step (last 20):**

- **fighter**: health=0.0173, weapon_hit=0.0021, nav_target=0.0021, weapons_target=0.0007, movement=0.0017
- **miner**: health=0.0300, nav_target=0.0024, movement=0.0020
- **trader**: health=0.0314, nav_target=0.0038, movement=0.0030

---
### Cycle 7631 — 04:47 (with engagement shaping)

**Training:** NaN=0/7626. Entropy=3.87. Clip=0.08. Stable.

**Landing events:** fighter=154, miner=88(+2), trader=75. Total=317 (+2 since 03:47). Engagement shaping just started — too early to see landing impact.

**Reward dominance:**
| | health | weapon_hit | nav_target | movement | pickup | weapons_tgt |
|---|---|---|---|---|---|---|
| fighter | 20% | **25%** | 25% | 20% | 2% | 8% |
| miner | 30% | 4% | 23% | 19% | **21%** | 3% |
| trader | 29% | 2% | **35%** | **29%** | 4% | 1% |

Balance unchanged from before engagement shaping — the new rewards feed into the existing movement channel and haven't had time to shift proportions yet.

**Growth:** Fighter weapon_hit=+57%, landing=+301%. Miner landing=+102957%. Trader weapon_hit=+80%, weapons_target=+189%.

**Value head EV:**
- Dense heads stable: health 0.87-0.97, nav_target 0.84-0.95, movement 0.79-0.94
- Landing volatile: -0.60 to 0.65 — still oscillating
- weapons_target improving: 0.49-0.74

**No concerns.** Continue training.

---
### Cycle 8426 — 17:45

**Training:** policy_loss=0.0001, entropy=3.831, clip=0.063, EV=0.590, NaN=0

**Landing:** fighter=162/8426, miner=94/8426, trader=79/8426 (total=335)

**Landing EV:** [0.061, 0.707, 0.482, 0.097, -0.043]

**Effective reward/step (last 20):**

- **fighter**: health=0.0171, weapon_hit=0.0018, nav_target=0.0019, weapons_target=0.0006
- **miner**: health=0.0304, weapon_hit=0.0007, nav_target=0.0022, movement=0.0009
- **trader**: health=0.0304, nav_target=0.0029, movement=0.0034

---
### Cycle 9092 — 06:47 ⚠ TRAINING COLLAPSED

**CRITICAL: Value function collapsed after engagement shaping change.**

- Entropy=0.000, Clip=0.000 — policy in permanent burn-in
- Health EV: 0.04-0.21 (was 0.92+)
- Landing EV: -4.08 (catastrophic)
- Cargo_sold EV: -7.62 (catastrophic)
- Total EV: 0.00-0.19

**Root cause:** Engagement shaping changed the movement reward distribution. The value replay buffer contained 8192 steps with returns computed under the old reward structure, causing the value function to receive conflicting training signals and diverge.

**Lesson learned:** When changing reward computation, the replay buffer must be cleared or recomputed. Stale return targets from the old reward structure poison the value function.

---
### Cycle 9109 — 07:47 ⚠ STILL COLLAPSED

Training remains in collapsed state. Total EV: -0.03. Entropy: 0.000. Policy in permanent burn-in. No recovery without intervention.

**Action needed:** Restore from pre-engagement-shaping backup or start fresh.

---
### Recovery — 08:30

**Restored from backup 20260408_0804** (cycle ~7571, pre-engagement-shaping).
- Cleared replay buffer (rl_buffer.bin deleted)
- Added engagement ramp: 0→1 linearly over 2 hours of game time (7200s)
- Ramped rewards: WEAPONS_FOCUS, PICKUP_PURSUIT, COMBAT_PURSUIT, COMBAT_FIRE
- Non-ramped (unchanged): proximity, planet approach/braking

**Lesson learned:** Never change reward structure with stale replay buffer. The ramp should prevent the value function collapse by giving it time to adapt.

---
### Cycle 9309 — 22:05

**Training:** policy_loss=0.0001, entropy=0.755, clip=0.008, EV=0.097, NaN=0

**Landing:** fighter=174/9309, miner=99/9309, trader=84/9309 (total=357)

**Landing EV:** [0.003, 0.104, -0.052, -1.821, 0.416]

**Effective reward/step (last 20):**

- **fighter**: health=0.0172, weapon_hit=0.0020, nav_target=0.0020, weapons_target=0.0006
- **miner**: health=0.0283, nav_target=0.0020, movement=0.0009
- **trader**: health=0.0309, nav_target=0.0008, movement=0.0008

---
### Cycle 9379 — 09:47 ⚠ VALUE FUNCTION DEGRADING

**Training:** NaN=0/9372. Entropy=0.755 (dropped from 3.8). Clip=0.008.

**Value function partially collapsed again:**
- Total EV: 0.02-0.38 (was 0.7-0.9)
- Health EV: 0.17-0.38 (was 0.92+)
- Cargo_sold EV: -10.8 (catastrophic)
- Movement EV: -0.18 to 0.33 (was 0.9+)

**Landing:** fighter=177, miner=100, trader=84. Total=361.

**Trader health dominance jumped to 58%** — other rewards declining.

**Likely cause:** The engagement ramp + obs-only refactoring changed the reward values enough to destabilize the value function again. The `ship_speed = speed_frac * 200.0` approximation in the obs-only braking reward may produce different values than the exact speed used before, causing a distribution shift in the movement reward.

---
### Cycle 10670 — 10:47

**Training:** NaN=0. Total EV: -0.04 to 0.38. Entropy oscillating (0→3.78 — burn-in toggling).

**Still degraded but showing signs of partial recovery** — one cycle hit entropy 3.78 and EV 0.38.

**Landing:** fighter=193, miner=103, trader=85. Total=381 (+20 since 09:47).

**Status:** Value function degraded from engagement ramp + speed approximation bug. The max_speed fix is compiled but not deployed. Need to restore from backup and restart.

---
### Cycle 10711 — 07:00

**Training:** policy_loss=0.0001, entropy=0.755, clip=0.008, EV=0.097, NaN=0

**Landing:** fighter=194/10711, miner=103/10711, trader=86/10711 (total=383)

**Landing EV:** [0.003, 0.104, -0.052, -1.821, 0.416]

**Effective reward/step (last 20):**

- **fighter**: health=0.0172, weapon_hit=0.0020, nav_target=0.0020, weapons_target=0.0006
- **miner**: health=0.0283, nav_target=0.0020, movement=0.0009
- **trader**: health=0.0309, nav_target=0.0008, movement=0.0008

---
### Restart — 06:53

**Restored from backup 20260408_0804 (3rd time), cleared replay buffer.**
- Fixed: `ship_speed = speed_frac * max_speed` (exact) instead of `* 200.0` (approximation)
- Engagement ramp active: 0→1 over 2 hours game time
- Training running (PID 50429)
- Will check back after a few hours of training

---
### Cycle 10875 — 08:06

**Training:** policy_loss=0.0001, entropy=0.755, clip=0.008, EV=0.097, NaN=0

**Landing:** fighter=196/10875, miner=105/10875, trader=87/10875 (total=388)

**Landing EV:** [0.003, 0.104, -0.052, -1.821, 0.416]

**Effective reward/step (last 20):**

- **fighter**: health=0.0172, weapon_hit=0.0020, nav_target=0.0020, weapons_target=0.0006
- **miner**: health=0.0283, nav_target=0.0020, movement=0.0009
- **trader**: health=0.0309, nav_target=0.0008, movement=0.0008

---
### Cycle ~7800 (566 post-restart) — 10:47 ✓ STABLE

**Training healthy after max_speed fix + engagement ramp restart.**

**Key metrics (post-restart only):**
- Total EV: **0.57-0.62** — stable, not degrading ✓
- Entropy: **3.79-3.81** — stable ✓
- Clip: **0.04-0.07** — excellent ✓
- NaN: 0 ✓

**Value head EV:**
- health: 0.90-0.94 ✓ (was 0.17-0.38 during collapse)
- nav_target: 0.83-0.90 ✓ (was -0.09 during collapse)
- movement: 0.83-0.91 ✓ (was -0.18 during collapse)
- weapons_target: 0.50-0.61 ✓
- landing: 0.00-0.60 — volatile but positive ✓

**Landing (post-restart only):** fighter=8, miner=4, trader=7. Total=19 in 566 cycles (3.4%).

**Assessment:** The max_speed fix resolved the value function collapse. The engagement ramp is working — rewards are changing gradually without destabilising the value function. Dense heads (health, nav_target, movement) have recovered to pre-collapse levels.

---
### Cycle 11264 — 18:38

**Training:** policy_loss=0.0001, entropy=0.755, clip=0.008, EV=0.097, NaN=0

**Landing:** fighter=201/11264, miner=107/11264, trader=92/11264 (total=400)

**Landing EV:** [0.003, 0.104, -0.052, -1.821, 0.416]

**Effective reward/step (last 20):**

- **fighter**: health=0.0172, weapon_hit=0.0020, nav_target=0.0020, weapons_target=0.0006
- **miner**: health=0.0283, nav_target=0.0020, movement=0.0009
- **trader**: health=0.0309, nav_target=0.0008, movement=0.0008

---
### Cycle ~7850 (620 post-restart) — 11:47 ✓

**Training health:** NaN=0/619. Entropy=3.78. Clip=0.04. Stable.

**Landing (post-restart):** fighter=8, miner=4, trader=7. Total=19/620 (3.1%).

**Reward dominance:**
| | health | weapon_hit | nav_target | movement | pickup | weapons_tgt |
|---|---|---|---|---|---|---|
| fighter | 27% | 25% | **31%** | 7% | 1% | 8% |
| miner | 34% | 6% | **25%** | 11% | **23%** | — |
| trader | **57%** | 2% | 15% | 16% | 9% | 1% |

**Trader health dominance at 57%** — higher than ideal. Trader nav_target (15%) and movement (16%) are lower than pre-collapse (was 38% and 29%). The engagement ramp is still early (~1.5 hrs into 2hr ramp), so engagement rewards haven't reached full strength yet.

**Value head EV:**
- health: 0.92-0.96 ✓
- nav_target: 0.86-0.94 ✓
- movement: 0.82-0.95 ✓
- landing: 0.40-0.77 — improved, less volatile ✓
- weapons_target: 0.53-0.73 ✓
- cargo_sold: 0.36-0.71 — moderate
- weapon_hit: -0.02 to 0.36 — still poor
- **Total EV: 0.55-0.82** — healthy ✓

**Assessment:** Stable training. Dense heads recovered fully. Engagement ramp proceeding without destabilisation. Trader dominance is a concern but should improve as engagement rewards reach full strength. No action needed.

---
### Cycle 11442 — 19:42

**Training:** policy_loss=0.0001, entropy=0.755, clip=0.008, EV=0.097, NaN=0

**Landing:** fighter=202/11442, miner=107/11442, trader=92/11442 (total=401)

**Landing EV:** [0.003, 0.104, -0.052, -1.821, 0.416]

**Effective reward/step (last 20):**

- **fighter**: health=0.0172, weapon_hit=0.0020, nav_target=0.0020, weapons_target=0.0006
- **miner**: health=0.0283, nav_target=0.0020, movement=0.0009
- **trader**: health=0.0309, nav_target=0.0008, movement=0.0008

---
### Cycle ~7980 (752 post-restart) — 19:47

**Training health:** NaN=0/751. Entropy=3.78. Clip=0.11. Stable.

**Landing (post-restart):** fighter=9(+1), miner=4, trader=7. Total=20/752 (2.7%).

**Reward dominance:**
| | health | weapon_hit | nav_target | movement | pickup | weapons_tgt |
|---|---|---|---|---|---|---|
| fighter | 31% | 21% | **32%** | 7% | 2% | 7% |
| miner | 31% | 8% | **22%** | 11% | **28%** | — |
| trader | **59%** | 2% | 16% | 16% | 6% | 1% |

**Engagement ramp now complete** (>2 hours since restart). Trader health dominance at 59% — the engagement rewards didn't significantly shift trader balance. Trader nav_target (16%) and movement (16%) are present but lower than pre-collapse (was 38%/29%).

**Value head EV:**
- Dense heads stable: health 0.91-0.94, nav_target 0.82-0.93, movement 0.73-0.92
- Landing volatile: -0.52 to 0.44 — still oscillating
- weapons_target dropping: 0.37-0.51 (was 0.50-0.73 last check)
- **Total EV: 0.59-0.77** — healthy

**Growth (first 200 → last 200 post-restart):**
- Fighter landing -75% — landing declining in recent cycles
- Miner movement +14%, weapon_hit +18% — miners improving
- Trader nav_target +9%, movement +6% — slight improvement

**Concerns:**
1. Trader health dominance (59%) — not improving despite engagement ramp reaching full strength
2. Fighter landing declining (-75%) — was positive before
3. weapons_target EV dropping

**No immediate action needed** but the engagement shaping doesn't seem to have significantly improved trader reward balance compared to pre-engagement-shaping training.

---
### Cycle 11582 — 20:34

**Training:** policy_loss=0.0001, entropy=0.755, clip=0.008, EV=0.097, NaN=0

**Landing:** fighter=204/11582, miner=107/11582, trader=92/11582 (total=403)

**Landing EV:** [0.003, 0.104, -0.052, -1.821, 0.416]

**Effective reward/step (last 20):**

- **fighter**: health=0.0172, weapon_hit=0.0020, nav_target=0.0020, weapons_target=0.0006
- **miner**: health=0.0283, nav_target=0.0020, movement=0.0009
- **trader**: health=0.0309, nav_target=0.0008, movement=0.0008

---
### Cycle ~8120 (892 post-restart) — 20:47

**Training health:** NaN=0/891. Entropy=3.03 (dropped from 3.78). Clip=0.05. Stable but entropy declining.

**Landing (post-restart):** fighter=11(+2), miner=4, trader=7. Total=22/892 (2.5%).

**Reward dominance:**
| | health | weapon_hit | nav_target | movement | pickup | weapons_tgt |
|---|---|---|---|---|---|---|
| fighter | 27% | 26% | **30%** | 7% | 2% | 8% |
| miner | 34% | 5% | **23%** | 10% | **27%** | 1% |
| trader | **54%** | 2% | 17% | 18% | 7% | 3% |

Trader health dominance improved slightly (59% → 54%). Trader nav_target up (16% → 17%), movement up (16% → 18%).

**Value head EV:**
- health: 0.93-0.96 ✓
- nav_target: 0.50-0.85 — variable, some low cycles
- movement: 0.86-0.92 ✓
- landing: 0.05-0.72 — volatile
- **Total EV: 0.10-0.64** — one cycle dipped to 0.10, concerning

**Growth (post-restart, first 200 → last 200):**
- Fighter landing -39% — still declining
- Miner landing -100%, cargo_sold -78% — both dropping
- Trader nav_target +7% — slight positive

**Concerns:**
1. **Entropy dropping** (3.78 → 3.03) — policy becoming more deterministic. May need monitoring.
2. **Total EV volatile** — one cycle at 0.10 is low
3. **Landing declining** for fighters and miners in growth metrics

**No action yet** but entropy decline bears watching. If it drops below 1.0, the policy may collapse.

---
### Cycle 11704 — 21:21

**Training:** policy_loss=0.0001, entropy=0.755, clip=0.008, EV=0.097, NaN=0

**Landing:** fighter=205/11704, miner=107/11704, trader=93/11704 (total=405)

**Landing EV:** [0.003, 0.104, -0.052, -1.821, 0.416]

**Effective reward/step (last 20):**

- **fighter**: health=0.0172, weapon_hit=0.0020, nav_target=0.0020, weapons_target=0.0006
- **miner**: health=0.0283, nav_target=0.0020, movement=0.0009
- **trader**: health=0.0309, nav_target=0.0008, movement=0.0008

---
### Cycle ~8240 (1014 post-restart) — 21:47

**Training health:** NaN=0/1013. Entropy=3.81 (recovered from 3.03 dip). Clip=0.06. Stable.

**Landing (post-restart):** fighter=12(+1), miner=4, trader=8(+1). Total=24/1014 (2.4%).

**Reward dominance:** Unchanged from prior check. Fighter balanced (26-29% across health/weapon_hit/nav_target). Trader health-heavy at 58%.

**Value head EV:**
- health: 0.92-0.95 ✓
- nav_target: 0.88-0.93 ✓
- movement: 0.75-0.93 ✓ (one low cycle)
- landing: -0.54 to 0.81 — very volatile, hit 0.81 in latest cycle ✓
- cargo_sold: -1.50 to 0.60 — wild swings, stale replay targets
- weapons_target: 0.24-0.51 — moderate
- **Total EV: 0.44-0.64** — OK

**Entropy recovered** to 3.81 (was 3.03 last check). The dip was transient.

**Growth:** Miner weapons_target +22%, movement +9%. Trader nav_target +8%, movement +7%. Mostly steady.

**Assessment:** Training stable and healthy. Entropy recovered. Landing continues accumulating slowly. No action needed. Continue overnight.

---
### Cycle 11817 — 22:07

**Training:** policy_loss=0.0001, entropy=0.755, clip=0.008, EV=0.097, NaN=0

**Landing:** fighter=208/11817, miner=107/11817, trader=94/11817 (total=409)

**Landing EV:** [0.003, 0.104, -0.052, -1.821, 0.416]

**Effective reward/step (last 20):**

- **fighter**: health=0.0172, weapon_hit=0.0020, nav_target=0.0020, weapons_target=0.0006
- **miner**: health=0.0283, nav_target=0.0020, movement=0.0009
- **trader**: health=0.0309, nav_target=0.0008, movement=0.0008

---
### Cycle ~8350 (1127 post-restart) — 22:47

**Training health:** NaN=0/1126. Entropy=3.05. Clip=0.05. Stable.

**Landing (post-restart):** fighter=15(+3), miner=4, trader=9(+1). Total=28/1127 (2.5%).

**Reward dominance:**
| | health | weapon_hit | nav_target | movement | pickup | weapons_tgt |
|---|---|---|---|---|---|---|
| fighter | 27% | **28%** | 27% | 6% | 2% | **11%** |
| miner | 30% | 3% | **23%** | 12% | **29%** | 3% |
| trader | **53%** | 2% | 17% | 19% | 7% | 2% |

**Notable:** Fighter weapons_target up to 11% (was 7-8%). Trader health dropping (59% → 53%). Miner pickup at 29% — highest.

**Growth (post-restart, first 200 → last 200):**
- **Fighter landing +67%** — reversed the decline! cargo_sold +22%
- Miner weapons_target +30%, movement +14%
- Trader nav_target +12%, movement +14%

**Value head EV — improving trends:**
- landing: 0.72-0.93 — **best ever!** (was -0.5 to 0.8 volatile)
- cargo_sold: 0.56-0.92 — also improving
- health: 0.85-0.95 ✓
- nav_target: 0.85-0.95 ✓
- movement: 0.66-0.88 — one low cycle
- weapons_target: 0.66-0.75 ✓
- **Total EV: 0.25-0.62** — variable, some low cycles

**Entropy:** 3.05 — still lower than the 3.78 peak but stable, not collapsing.

**Assessment:** Training is healthy and improving. Landing head EV at its best levels ever. Fighter landing growth reversed to positive (+67%). The engagement ramp completed successfully without destabilising training. Continue overnight.

---
### Cycle 13505 — 08:06

**Training:** policy_loss=0.0007, entropy=3.819, clip=0.074, EV=0.810, NaN=0

**Landing:** fighter=226/13505, miner=112/13505, trader=103/13505 (total=441)

**Landing EV:** [0.934, -0.288, 0.488, 0.601, 0.671]

**Effective reward/step (last 20):**

- **fighter**: health=0.0171, weapon_hit=0.0012, nav_target=0.0019, weapons_target=0.0007
- **miner**: health=0.0280, nav_target=0.0019, movement=0.0008
- **trader**: health=0.0279, nav_target=0.0008, movement=0.0008

---

### Cycle 13542 — 08:04 (thorough analysis)

**Training health:** NaN=0/13534, entropy=3.81, clip=0.07, total EV last 5: [0.899, 0.891, 0.626, 0.890, 0.905]. Throughput steady at ~24s/cycle, ~340 steps/sec GPU.

**Landing counts (cumulative):** fighter=228, miner=112, trader=104 (total=444). Only +3 since the previous check (~5,200 cycles ago) — landing rate has effectively stalled near zero.

**Reward dominance (last 20):**
| | health | weapon_hit | nav_target | weapons_tgt | movement | pickup |
|---|---|---|---|---|---|---|
| fighter | 27% | 24% | **31%** | 10% | 7% | 2% |
| miner | **37%** | 6% | 25% | 2% | 13% | 17% |
| trader | **59%** | 2% | 14% | 3% | 15% | 7% |

**Growth (first 200 → last 200):**
- **fighter**: weapon_hit +21%, landing +274%, cargo_sold +270%, nav_target -11%, **movement -76%**
- **miner**: landing +47840%, cargo_sold +1973%, pickup +19%, weapons_target +35%, **movement -49%**
- **trader**: weapon_hit +31%, **landing -88%, cargo_sold -94%, nav_target -80%, movement -71%**

**Value head EV (last 5):**
| head | last 5 |
|---|---|
| health | 0.984, 0.984, 0.989, 0.987, 0.983 ✓ |
| weapon_hit | 0.497, 0.565, 0.638, 0.315, 0.591 |
| **landing** | 0.755, 0.621, 0.854, 0.801, **0.009** |
| **cargo_sold** | 0.670, 0.532, 0.808, 0.818, **-0.832** |
| pickup | 0.472, 0.559, 0.683, 0.345, 0.101 |
| nav_target | 0.956, 0.961, 0.954, 0.970, 0.944 ✓ |
| weapons_target | 0.832, 0.853, 0.722, 0.631, 0.810 ✓ |
| movement | 0.871, 0.933, 0.927, 0.950, 0.922 ✓ |

**Concerns:**
1. **Landing/cargo_sold heads destabilising again.** Both held strong (0.7-0.9) for 4 cycles then collapsed in the last cycle (landing→0.009, cargo_sold→-0.832). Same pattern as the previous collapse — the rare-event heads remain fragile and likely suffering from stale replay targets when sparse landing events redistribute.
2. **Trader regression is severe.** Trader has lost most of its planet-trading signal: nav_target -80%, landing -88%, cargo_sold -94%, movement -71%. Trader is now overwhelmingly dominated by passive health reward (59%) — it has effectively given up on active strategies.
3. **Movement reward decay across all personalities** (-49% to -79%). Agents are spending less time approaching nav targets — likely correlated with the trader regression.
4. **Landing rate stalled.** Only 3 new landing events in ~5200 cycles since the previous check. The post-restart improvement has not translated into sustained landing accumulation.

**Positives:**
- Total EV is back to healthy levels (0.79-0.91 over last 5).
- Fighter retains the broad reward balance (27/24/31/10/7) and landing growth (+274%).
- Miner landing growth (+47840%) and cargo_sold (+1973%) remain very strong year-over-run.
- nav_target, movement, weapons_target value heads are all stable and high-EV.
- No NaNs.

**Assessment:** The fighter and miner are still progressing, but the trader has clearly regressed and the landing/cargo_sold heads continue to be unstable. The next collapse cycle is concerning — if total EV drops below 0.5 again, restoring from the 22:05 backup (or this 08:04 backup) and clearing the replay buffer should be considered. For now, training continues; another check in an hour will show whether this is a transient blip or a sustained collapse.

---
### Cycle 13668 — 09:06

**Training:** policy_loss=0.0001, entropy=3.814, clip=0.081, EV=0.678, NaN=0

**Landing:** fighter=230/13668, miner=113/13668, trader=106/13668 (total=449)

**Landing EV:** [0.260, 0.000, 0.488, 0.886, 0.926]

**Effective reward/step (last 20):**

- **fighter**: health=0.0176, weapon_hit=0.0019, nav_target=0.0020, weapons_target=0.0006
- **miner**: health=0.0284, nav_target=0.0019, movement=0.0008
- **trader**: health=0.0272, nav_target=0.0007, movement=0.0008

---

### Cycle 14034 — 09:04 (thorough analysis)

**Training health:** NaN=0/14026. Entropy=3.79–3.85. Clip=0.06–0.08. Total EV last 5: [0.733, 0.763, 0.774, 0.895, 0.680] — healthy on average but with continued single-cycle dips. Throughput ~26s/cycle.

**Landing counts (cumulative, deltas vs last hour):**
- fighter: 236 (+8)
- miner: 115 (+3)
- trader: 109 (+5)
- total: 460 (+16)

Landing rate over the last hour ≈ 16/500 cycles = 3.2% — noticeably better than the previous hour's stall (+3 in 5200). Trader actually got 5 new landings, the most of any personality this hour.

**Reward dominance (last 20):**
| | health | weapon_hit | nav_target | weapons_tgt | movement | pickup |
|---|---|---|---|---|---|---|
| fighter | 26% | **28%** | 28% | 9% | 6% | 2% |
| miner | 31% | 7% | 25% | — | 11% | **26%** |
| trader | **55%** | — | 16% | 3% | 17% | 9% |

Trader health dominance has dropped slightly (59% → 55%) and pickup share rose to 9%. Fighter weapon_hit is the highest channel (28%).

**Growth (first 200 → last 200):**
- **fighter**: weapon_hit +36%, **landing +68%, cargo_sold +158%**, pickup +13%, nav_target -13%, movement -77%
- **miner**: pickup +13%, weapons_target +11%, movement -53%
- **trader**: weapon_hit +34%, **landing -45% (was -88% last hour)**, cargo_sold -79% (was -94%), nav_target -79%, **weapons_target +185%**, movement -69%

Trader regression is **partially recovering**: landing growth went from -88% → -45% and cargo_sold from -94% → -79%. Still negative but the trajectory has reversed.

**Value head EV (last 5):**
| head | last 5 |
|---|---|
| health | 0.991, 0.984, 0.988, 0.984, 0.988 ✓ |
| weapon_hit | 0.244, 0.152, 0.339, 0.556, 0.427 |
| **landing** | 0.033, **-0.261**, 0.791, 0.578, 0.663 — recovering from collapse |
| **cargo_sold** | 0.271, 0.713, 0.862, **-0.161**, 0.754 — bouncing, mostly recovered |
| pickup | 0.242, 0.317, 0.263, 0.755, 0.675 |
| nav_target | 0.965, 0.941, 0.961, 0.969, 0.966 ✓ |
| weapons_target | 0.778, 0.585, 0.754, 0.820, 0.776 ✓ |
| movement | 0.954, 0.914, 0.947, 0.939, 0.937 ✓ |

**Concerns:**
1. **Landing/cargo_sold heads remain volatile.** Both bounced in and out of negative territory in the last 5 cycles. They are recovering but each individual cycle's value targets are noisy because the events themselves are sparse.
2. **Trader still bottom-heavy.** 55% health dominance is too high; trader has not yet rediscovered planet trading at any meaningful level. Movement reward still -69% from start.
3. **Movement reward decay across all personalities** (-52% to -79% from start) — this is now a stable feature of the run rather than a transient. Likely the policy has shifted away from "approach nav target" behavior in favor of combat / pickup chasing.

**Positives:**
- **Trader regression is reversing** — landing -88%→-45%, cargo_sold -94%→-79%, nav_target -69% (was -80%).
- **Landing accumulation has resumed**: +16 landings this hour vs +3 last hour. Trader contributed 5/16.
- **Fighter landing/cargo_sold growth is strong**: +68% / +158%.
- The previous-cycle landing collapse (0.009) recovered to 0.66 within 4 cycles — the value head is self-healing without intervention.
- Health, nav_target, weapons_target, movement value heads are all rock-solid (>0.78 across last 5).
- No NaN, training stable, total EV averaging ~0.78.

**Assessment:** The transient landing/cargo_sold collapses observed last hour did not develop into a sustained collapse — they've already bounced back. Landing accumulation has resumed at 3% per cycle (better than the previous hour). Trader is showing the first signs of recovering its planet-trading signal. **No intervention warranted** — training continues healthy. Watch for further trader recovery and continued landing accumulation in the next check.

---
### Cycle 14054 — 11:35

**Training:** policy_loss=0.0004, entropy=3.819, clip=0.080, EV=0.863, NaN=0

**Landing:** fighter=238/14054, miner=115/14054, trader=109/14054 (total=462)

**Landing EV:** [0.948, 0.866, 0.921, 0.863, 0.876]

**Effective reward/step (last 20):**

- **fighter**: health=0.0164, weapon_hit=0.0016, nav_target=0.0019, weapons_target=0.0008
- **miner**: health=0.0291, nav_target=0.0022, movement=0.0010
- **trader**: health=0.0273, nav_target=0.0008, movement=0.0009

---

### Cycle 15016 — 11:33 (thorough analysis)

**Backup:** experiments/run_16/backups/20260410_1133

**Training health:** NaN=0/15008. Entropy=3.78–3.83. Clip=0.06–0.10. Total EV last 5 (from monitor at cycle 14054): [0.904, 0.857, 0.868, 0.904, 0.784]. From the detail snapshot at 15016: [0.794, 0.746, 0.697, 0.842, **0.593**]. The detail snapshot's last value (0.593) is the lowest in the last 5 — slightly soft but well above the 0.5 intervention threshold. Throughput steady at ~26s/cycle.

**Landing counts (cumulative, deltas vs last hour):**
- fighter: 265 (+29)
- miner: 123 (+8)
- trader: 115 (+6)
- total: 503 (+43)

Landing rate this hour ≈ 43/980 ≈ **4.4%** — best hourly rate of the run! Fighter alone produced 29 landings, more than the previous two hours combined.

**Reward dominance (last 20):**
| | health | weapon_hit | nav_target | weapons_tgt | movement | pickup |
|---|---|---|---|---|---|---|
| fighter | 25% | **31%** | 27% | 9% | 6% | 2% |
| miner | 34% | 4% | 23% | 2% | 11% | **27%** |
| trader | **59%** | 3% | 15% | 2% | 16% | 5% |

Fighter weapon_hit jumped to 31% (was 28%) — engaging more aggressively. Trader health dominance climbed back to 59% from 55% last hour. Miner pickup share holding at 27%.

**Growth (first 200 → last 200):**
- **fighter**: weapon_hit +23%, **landing +155% (was +68%)**, cargo_sold +158%, pickup +16%, nav_target -12%, weapons_target +10%, movement -76%
- **miner**: **landing +39339%, cargo_sold +2481%**, pickup +16%, weapons_target +27%, movement -51%
- **trader**: weapon_hit +74%, **landing -80% (was -45%), cargo_sold -77%**, nav_target -79%, **weapons_target +250%**, movement -69%

**Fighter landing growth more than doubled** this hour (+68 → +155%), consistent with the +29 landings observed. **Trader regression has resumed** — landing growth went from -45% back to -80%, partially undoing last hour's recovery. Trader weapons_target +250% suggests trader is increasingly engaging in combat instead of trading.

**Value head EV (last 5, from cycle 15016 snapshot):**
| head | last 5 |
|---|---|
| health | 0.992, 0.989, 0.991, 0.992, 0.982 ✓ |
| weapon_hit | 0.497, 0.594, 0.508, 0.733, 0.497 |
| **landing** | 0.471, 0.707, 0.377, 0.478, **0.208** — degrading |
| **cargo_sold** | 0.668, **-0.876**, **-0.363**, 0.423, 0.737 — volatile |
| pickup | 0.255, 0.426, 0.336, 0.335, 0.162 |
| nav_target | 0.979, 0.965, 0.968, 0.955, 0.932 ✓ |
| weapons_target | 0.863, 0.879, 0.872, 0.877, 0.872 ✓ |
| movement | 0.962, 0.943, 0.925, 0.968, 0.910 ✓ |

**Concerns:**
1. **Landing head EV degrading.** Mean of last 5: 0.448 (was 0.55 last hour). Trending down despite landings increasing — value head can't keep up with the changing distribution.
2. **cargo_sold head volatile** with two negative-EV cycles (-0.876, -0.363) in the last 5. Not collapsing but not stabilising either.
3. **Trader regression resumed** (landing -45% → -80%, weapons_target +250%). The trader appears to be drifting toward a "fighter-lite" policy: targeting hostile ships and engaging instead of trading.
4. **Movement reward decay** persists at -51% to -79%. The policy has fully shifted away from approach behavior — landing is happening but it's a side-effect of combat/pursuit rather than an intentional approach.
5. **Pickup head EV is weak** (0.16-0.45). Despite pickup being 27% of miner reward, the value head can't predict it well.

**Positives:**
- **Best landing hour ever**: +43 landings (4.4% rate). Fighter +29, miner +8, trader +6 — every personality contributed.
- **Fighter landing growth +155%** — more than doubled from last hour.
- **No NaN**, total EV averaging 0.74 (still healthy).
- Health, nav_target, weapons_target, movement value heads remain rock-solid (>0.87 each).
- Miner landing growth still at +39,339% from baseline.
- Throughput stable.

**Assessment:** Mixed picture. Landing accumulation is at its best rate of the run (43/hour, fighter contributing 29), confirming the policy is producing real landing behavior. However, the landing/cargo_sold value heads are degrading even as the rewards become more frequent — and trader is regressing back toward a combat-focused policy. **No intervention warranted yet** (total EV well above 0.5, NaN-free, landings increasing). Continue monitoring; if landing head EV falls below 0.0 sustained over multiple cycles, or trader landing growth approaches -100%, consider restoring from the 11:33 backup and clearing the replay buffer.

---
### Cycle 15075 — 18:06

**Training:** policy_loss=0.0010, entropy=3.801, clip=0.090, EV=0.766, NaN=0

**Landing:** fighter=266/15075, miner=123/15075, trader=115/15075 (total=504)

**Landing EV:** [-2.611, -3.889, -2.158, 0.333, 0.198]

**Effective reward/step (last 20):**

- **fighter**: health=0.0175, weapon_hit=0.0018, nav_target=0.0019, weapons_target=0.0006
- **miner**: health=0.0281, nav_target=0.0019, movement=0.0009
- **trader**: health=0.0290, nav_target=0.0008, movement=0.0008

---

### Cycle 15653 — 18:03 (thorough analysis)

**Backup:** experiments/run_16/backups/20260410_1803

**Training health:** NaN=0/15645. Entropy=3.78–3.84. Clip=0.07–0.12. Total EV last 5: [0.574, 0.716, 0.529, 0.780, 0.777] — mostly healthy but with the second cycle dropping to 0.53. Throughput ~27s/cycle.

**Landing counts (cumulative, deltas vs last hour):**
- fighter: 292 (+27)
- miner: 135 (+12)
- trader: 121 (+6)
- total: 548 (+45)

Landing rate this hour ≈ 45/637 ≈ **7.1%** — **new best hourly rate**. Fighter steady at ~27/hour, miner improved to 12 (from 8), trader stable at 6.

**Reward dominance (last 20):**
| | health | weapon_hit | nav_target | weapons_tgt | movement | pickup |
|---|---|---|---|---|---|---|
| fighter | 25% | 27% | **29%** | 10% | 7% | 2% |
| miner | 33% | 4% | 26% | — | 12% | **23%** |
| trader | **57%** | 4% | 13% | 3% | 15% | 8% |

Fighter balance unchanged. Trader health dominance held at 57%. Miner distributing reward across more channels.

**Growth (first 200 → last 200):**
- **fighter**: weapon_hit **+62%**, **landing +617%**, cargo_sold **+467%**, pickup +23%, weapons_target +33%, movement -75%
- **miner**: **landing +364,028%**, cargo_sold +4064%, pickup +14%, **weapons_target +107%**, movement -53%
- **trader**: weapon_hit +42%, **landing -78%**, cargo_sold -88%, nav_target -81%, **weapons_target +330%**, movement -71%

**Fighter landing growth exploded**: +155% → **+617%**. Miner landing at +364,028% — literally 3600× baseline. Trader regression is stable at the lower floor (landing -78%).

**Value head EV (last 5):**
| head | last 5 |
|---|---|
| health | 0.980, 0.980, 0.978, 0.981, 0.988 ✓ |
| weapon_hit | 0.046, -0.031, 0.094, 0.249, 0.099 — **weak** |
| **landing** | -0.326, 0.150, 0.305, -0.452, **-0.789** — **significantly degraded** |
| **cargo_sold** | -2.666, -3.878, -0.674, 0.387, **-3.365** — **severely degraded** |
| pickup | 0.070, 0.380, 0.090, 0.086, 0.587 |
| nav_target | 0.960, 0.951, 0.955, 0.947, 0.962 ✓ |
| weapons_target | 0.717, 0.738, 0.830, 0.770, 0.703 ✓ |
| movement | 0.951, 0.954, 0.920, 0.902, 0.945 ✓ |

**Concerns:**
1. **Landing and cargo_sold value heads are genuinely broken now.** Landing mean of last 5: **-0.22** (was +0.45 last hour). cargo_sold mean of last 5: **-2.04**. Not a transient bounce — these heads are failing to fit the increasing landing frequency. The bursty, step-count-dependent returns are out-of-distribution for a head that trained on long stretches of zero.
2. **weapon_hit head EV has also collapsed** to near zero / slightly negative. This is new — was 0.5-0.7 last hour. Likely fighter's rapid increase in combat engagement is shifting the weapon_hit return distribution.
3. **Total EV is healthy on average (0.68)** because health/nav_target/weapons_target/movement dominate the weighting and are all >0.70. But the sparse heads are degrading.
4. **Trader permanent regression**: weapons_target +330%, landing -78%. Trader has fully converted to a combat policy.

**Positives:**
- **Best landing hour yet**: +45 landings (7.1% rate). Third consecutive hour of accelerating landing activity (+16 → +43 → +45).
- **Fighter landing growth +617%** — still accelerating rapidly.
- **Miner landing +364,028%**, cargo_sold +4064% — miners are now reliably completing the full trade loop.
- No NaN, throughput stable.
- Health, nav_target, weapons_target, movement heads still rock-solid.

**Assessment:** Paradoxical state — **behavior is improving rapidly (best landing rate, fighter +617% growth) while the value heads for landing/cargo_sold/weapon_hit are collapsing**. This is consistent with the classic PPO issue where the value function lags behind a rapidly changing policy on sparse/bursty rewards. Because the dense heads (health, nav_target, movement) dominate the weighted total, PPO's advantage signal is still reasonable and landings keep increasing.

**Decision:** Total EV is at 0.68 average — still above the 0.5 intervention threshold. Behavior is actively improving (best landing hour yet). **Do not intervene.** Continue training; the sparse heads may auto-heal as they did 2 hours ago (landing bounced from -0.26 → +0.66 within a few cycles). If in the next hour: total EV drops below 0.5 sustained, OR landings start decreasing, OR fighter landing growth reverses → restore from the 18:03 backup and clear the replay buffer.

---
### Cycle 15667 — 21:51

**Training:** policy_loss=0.0008, entropy=3.811, clip=0.081, EV=0.602, NaN=0

**Landing:** fighter=292/15667, miner=135/15667, trader=122/15667 (total=549)

**Landing EV:** [-0.115, -0.589, -0.249, 0.238, 0.001]

**Effective reward/step (last 20):**

- **fighter**: health=0.0167, weapon_hit=0.0027, nav_target=0.0019, weapons_target=0.0007
- **miner**: health=0.0282, weapon_hit=0.0006, nav_target=0.0020, movement=0.0009
- **trader**: health=0.0311, nav_target=0.0007, movement=0.0008

---

### Cycle 15761 — 21:49 (thorough analysis)

**Backup:** experiments/run_16/backups/20260410_2149

**⚠ Cycle throughput anomaly:** At 18:03 we were at 15653 cycles; now at 15761 ≈ 3h46m later. That's only **108 cycles** over ~3.75 hours — far below the expected ~485 cycles at 28s/cycle. Training either stalled for a period or has hit a major throughput regression. Monitor's last-5 cycle time still reports 28s/cycle (healthy), so the stall was transient rather than ongoing. **Worth investigating** whether something paused training during the afternoon (machine sleep, SSH disconnect, OS update, etc.).

**Training health:** NaN=0/15753. Entropy=3.80–3.82. Clip=0.06–0.10. Total EV last 5: [0.694, 0.711, 0.765, 0.638, 0.793] — healthy on average. Monitor's slightly older snapshot shows one dip to 0.419.

**Landing counts (cumulative, deltas vs last hour):**
- fighter: 296 (+4)
- miner: 139 (+4)
- trader: 124 (+3)
- total: 559 (+11)

Only +11 new landings this "hour" — but over just 108 cycles, that's a rate of **10.2%**, which would actually be the best per-cycle rate of the run. The absolute count is low solely because of the throughput anomaly.

**Reward dominance (last 20):**
| | health | weapon_hit | nav_target | weapons_tgt | movement | pickup |
|---|---|---|---|---|---|---|
| fighter | 28% | 23% | **30%** | 10% | 7% | 2% |
| miner | 34% | 6% | 23% | 2% | 11% | **23%** |
| trader | **56%** | 1% | 17% | 1% | **19%** | 5% |

Trader movement dominance climbed from 15% → 19% — the trader is spending more effort on approach behavior, first positive trader signal in hours. Fighter weapon_hit share dipped from 27% → 23%.

**Growth (first 200 → last 200):**
- **fighter**: weapon_hit +43%, **landing +372%** (was +617% last hour — *down*), cargo_sold +219%, weapons_target +11%, movement -76%
- **miner**: **landing +153,299%** (was +364,028% — *down*), cargo_sold +1731%, pickup +21%, weapons_target +78%, movement -50%
- **trader**: weapon_hit +42%, landing -85%, cargo_sold -97%, nav_target -81%, weapons_target +195%, movement -71%

Fighter and miner landing growth have both *declined* from last hour (617% → 372%, 364,028% → 153,299%). Since these are first-200 vs last-200 rolling windows, a decline means the very recent 200 cycles are performing worse than the peak 200 from earlier — the landing behavior is stabilising or slightly regressing.

**Value head EV (last 5):**
| head | last 5 |
|---|---|
| health | 0.985, 0.990, 0.990, 0.984, 0.975 ✓ |
| weapon_hit | 0.164, 0.133, 0.017, 0.528, 0.308 — still weak |
| **landing** | -0.315, -0.158, **-1.047**, -0.682, **0.021** — consistently negative, slight recovery |
| **cargo_sold** | 0.522, 0.400, **-2.959**, -0.931, 0.078 — volatile, recovering |
| pickup | 0.276, 0.313, 0.179, 0.035, 0.120 |
| nav_target | 0.952, 0.965, 0.967, 0.933, 0.949 ✓ |
| weapons_target | 0.701, 0.693, 0.686, 0.780, 0.764 ✓ |
| movement | 0.949, 0.942, 0.930, 0.919, 0.892 ✓ |

**Concerns:**
1. **⚠ Major throughput slowdown.** Only 108 cycles in ~3.75 hours vs expected ~485. Training was paused or severely slowed. Current throughput looks healthy, so it may be resolved, but worth investigating.
2. **Landing value head persistently in the red.** Mean of last 5: -0.44 (was -0.22 at 18:03, +0.45 at 11:33). Three consecutive hours of degradation. Not bouncing back yet.
3. **cargo_sold head volatile** with occasional large negative cycles (-2.959). Trying to recover but unstable.
4. **Fighter landing growth declining** (+617% → +372%). Still positive but decelerating.
5. **Miner landing growth also declining** (+364,028% → +153,299%). Same pattern.
6. **Trader still permanently regressed** — landing -85%, cargo_sold -97%, nav_target -81%.

**Positives:**
- **Per-cycle landing rate still excellent** (10.2% of the 108 cycles produced a landing).
- **Total EV still healthy** (avg 0.72 over last 5, range 0.64-0.79).
- **First positive trader signal**: trader movement share climbed to 19% (from 15-17% previous hours). Trader is spending more effort approaching nav targets, even if not closing the loop.
- **No NaN.** Health, nav_target, movement, weapons_target value heads still >0.89.
- Throughput has recovered to ~28s/cycle.

**Assessment:** The landing value head has been in negative territory for 3+ hours now without fully recovering, which is different from previous bounces. Fighter/miner landing growth is decelerating. The first-200-vs-last-200 growth decline for fighter (+617% → +372%) is notable because it suggests the *rate* of landing is starting to slow — not catastrophic, but the upward trend has plateaued.

**Decision:** Do NOT restore from backup yet. Total EV is still healthy (0.72 avg), landings are still happening (10% rate per cycle), no NaN. The sparse value heads are struggling but not causing total collapse. Continue monitoring. If next check shows: (a) total EV below 0.5 sustained, (b) absolute landing count doesn't increase, or (c) the throughput issue recurs — then restore from 11:33 backup (which had the best landing head EV of recent hours).

**Action item:** Check if there was a system pause/sleep/disconnect between 18:03 and 21:49 that caused the throughput gap.

---
### Cycle 15791 — 23:06

**Training:** policy_loss=0.0003, entropy=3.832, clip=0.088, EV=0.569, NaN=0

**Landing:** fighter=298/15791, miner=140/15791, trader=124/15791 (total=562)

**Landing EV:** [-0.179, 0.091, 0.158, 0.333, -0.098]

**Effective reward/step (last 20):**

- **fighter**: health=0.0170, weapon_hit=0.0023, nav_target=0.0018, weapons_target=0.0006
- **miner**: health=0.0276, nav_target=0.0021, movement=0.0010
- **trader**: health=0.0300, nav_target=0.0009, movement=0.0010

---

### Cycle 17075 — 23:04 (thorough analysis)

**Backup:** experiments/run_16/backups/20260410_2304

**Throughput recovered:** +1314 cycles since 21:49 (15761 → 17075) over ~1h15m = ~1050 cycles/hour. Monitor shows ~22.5s/cycle, ~375 steps/sec GPU — **faster than any previous hour**. The earlier throughput gap is fully resolved.

**Training health:** NaN=0/17067. Entropy=3.80–3.85. Clip=0.08–0.10. Total EV last 5: [0.583, 0.802, 0.807, 0.538, 0.696] — healthy on average (0.69). Still some single-cycle dips.

**Landing counts (cumulative, deltas vs last hour):**
- fighter: 352 (+56)
- miner: 165 (+26)
- trader: 132 (+8)
- total: 649 (+90)

**+90 landings in ~1h15m** — rate normalised to ~72/hour, **best of the run**. Fighter +56 is the biggest one-hour fighter contribution yet. Miner rate tripled from +8 to +26.

**Reward dominance (last 20):**
| | health | weapon_hit | nav_target | weapons_tgt | movement | pickup |
|---|---|---|---|---|---|---|
| fighter | 28% | 24% | **29%** | 11% | 7% | 2% |
| miner | 31% | 7% | 22% | — | 10% | **29%** |
| trader | **54%** | 2% | 17% | 3% | **18%** | 6% |

**Trader health dominance dropped** 59% → 56% → 54% over the last three hours — the lowest trader health share observed since the restart. **Trader movement share up to 18%** (from 15% → 17% → 19% → 18%). Trader is now putting substantive effort into navigation rather than passively accumulating health reward.

**Growth (first 200 → last 200):**
- **fighter**: weapon_hit +22%, **landing +118%** (was +372% → now declining), cargo_sold +185%, nav_target -15%, movement -77%
- **miner**: weapon_hit -11%, pickup +6%, movement -51% — landing/cargo_sold below the 5% growth threshold this cycle (noise)
- **trader**: weapon_hit +10%, landing -81%, cargo_sold -89%, nav_target -77%, **weapons_target +153%** (was +195%), movement -68%

Fighter landing growth declined again (+372% → +118%) — but remember these are first-200 vs last-200 windows, and the early cycles include the lowest-landing portions of the run, so "decline" here just reflects that recent rate is closer to the long-term average rather than the peak. Importantly, the **absolute landing count keeps rising** (+56 fighter this hour).

**Value head EV (last 5):**
| head | last 5 |
|---|---|
| health | 0.932, 0.983, 0.987, 0.976, 0.976 ✓ |
| weapon_hit | 0.423, 0.480, 0.336, 0.499, 0.551 — **recovering** |
| **landing** | -2.266, 0.377, 0.259, 0.464, **0.410** — **recovered!** (mean excl. outlier: +0.378) |
| **cargo_sold** | -5.318, -0.582, -1.037, 0.074, **-1.887** — still volatile |
| pickup | 0.133, 0.279, 0.664, 0.262, 0.122 |
| nav_target | 0.892, 0.956, 0.971, 0.972, 0.935 ✓ |
| weapons_target | 0.785, 0.767, 0.823, 0.700, 0.860 ✓ |
| movement | 0.699, 0.935, 0.946, 0.926, 0.877 ✓ |

**Concerns:**
1. **cargo_sold head still volatile** with a -5.32 outlier and -1.89 in the last cycle. Not recovering as quickly as landing.
2. **Trader still permanently below baseline** on landing/cargo_sold/nav_target — even though its behavioral share of movement/nav is increasing.
3. **Fighter landing growth declining** (+617 → +372 → +118%) — three consecutive hours of decline, suggesting the landing rate has plateaued rather than continuing to accelerate.

**Positives:**
- **Best throughput of the run** (~22.5s/cycle, ~375 steps/sec) — whatever caused the earlier slowdown is gone.
- **Best landing hour of the run**: +90 landings (effectively +72/hour normalised). Fighter +56, miner +26 — miner rate tripled.
- **Landing value head has recovered** from -0.44 last hour to +0.38 this hour (excluding the -2.27 outlier at the start of the window).
- **Trader health dominance trending down** (59→56→54%) and trader movement share trending up (15→17→19→18%). First signs of trader behavioral improvement, even if the trade loop isn't closing.
- **Total EV healthy** (avg 0.69).
- No NaN.
- weapon_hit head has recovered from ~0.15 mean to 0.46 mean.

**Assessment:** Strong hour. The throughput gap from earlier resolved, and the recovery came with the **best landing rate of the run** (+90 landings, fighter +56). The landing value head — which had been negative for 3+ hours — has now recovered into positive territory. cargo_sold is still struggling, but landing, weapon_hit, and total EV are all improving. Trader is showing its first real behavioral shifts post-restart. **No intervention warranted** — continue training. Landing growth deceleration is concerning on paper but contradicted by the absolute count increase.

---
### Cycle 17084 — 06:11

**Training:** policy_loss=0.0004, entropy=3.789, clip=0.087, EV=0.731, NaN=0

**Landing:** fighter=352/17084, miner=165/17084, trader=132/17084 (total=649)

**Landing EV:** [0.325, 0.482, 0.467, 0.641, 0.726]

**Effective reward/step (last 20):**

- **fighter**: health=0.0164, weapon_hit=0.0018, nav_target=0.0017, weapons_target=0.0007
- **miner**: health=0.0238, weapon_hit=0.0006, nav_target=0.0018, movement=0.0008
- **trader**: health=0.0290, nav_target=0.0009, movement=0.0009

---

### Cycle 17086 — 06:08 (thorough analysis)

**Backup:** experiments/run_16/backups/20260411_0608

**⚠⚠ MAJOR STALL:** Only **11 cycles** completed in ~7 hours since the 23:04 check (17075 → 17086). Expected ~850 cycles at 25s/cycle. Training was effectively paused — the machine almost certainly slept overnight. The last-5 cycle times in the monitor report 27–29s each, so the *most recent* handful of cycles completed normally; the 6-7h gap is in the middle of the interval.

**Training health:** NaN=0/17078. Entropy=3.77–3.80. Clip=0.07–0.10. Total EV last 5: [0.692, 0.835, 0.759, 0.620, 0.736] — healthy (avg 0.73). Training itself is fine; it just wasn't running for most of the night.

**Landing counts (cumulative, deltas vs last check):**
- fighter: 352 (+0)
- miner: 165 (+0)
- trader: 132 (+0)
- total: 649 (+0)

**No new landings** — because essentially no new cycles ran. This is consistent with the stall rather than a behavioral collapse.

**Reward dominance (last 20):**
| | health | weapon_hit | nav_target | weapons_tgt | movement | pickup |
|---|---|---|---|---|---|---|
| fighter | 26% | **29%** | 27% | 11% | 6% | 2% |
| miner | 31% | 8% | 24% | 2% | 11% | **25%** |
| trader | **54%** | 2% | 17% | 2% | **18%** | 7% |

Fighter weapon_hit climbed to 29% (highest yet). Trader health held at 54%, movement at 18% — trader behavioral improvement still intact.

**⚠ Miner population collapse.** The monitor reports miner mean slot counts of just **ship=0.38, asteroid=0.52, planet=0.40, pickup=0.40** — about a quarter of the usual values. Miner raw reward totals also plunged: health 36.14 → **5.31**, nav_target 28.68 → **3.88**, movement 13.51 → **1.61**, pickup 3.68 → **0.24**. Miner nav-target selection fractions (0.056/0.048/0.049/0.033/0.014) sum to only ~0.20 instead of the usual ~1.0, meaning **miners are almost entirely absent from the simulation** for the window being summarised.

This appears to be a side-effect of the stall: the "last 10 cycles" / "last 20 cycles" windows are thinly populated because so few cycles have been logged recently, and whatever system transient coincided with the stall may have temporarily removed miner agents.

**Growth (first 200 → last 200):**
- **fighter**: weapon_hit +24%, landing +118%, cargo_sold +185%, nav_target -16%, movement -77%
- **miner**: weapon_hit -12%, nav_target -5%, movement -52% (landing/cargo_sold below 5% threshold — dropped out of noise)
- **trader**: landing -81%, cargo_sold -89%, pickup -8%, nav_target -77%, weapons_target +158%, movement -68%

**Value head EV (last 5):**
| head | last 5 |
|---|---|
| health | 0.987, 0.986, 0.980, 0.977, 0.983 ✓ |
| weapon_hit | 0.566, 0.607, 0.516, 0.296, 0.232 |
| **landing** | 0.467, 0.641, 0.726, **-0.561**, 0.268 — avg +0.31 |
| **cargo_sold** | **-1.715**, 0.653, -0.485, -0.438, 0.069 — still volatile |
| pickup | 0.390, 0.318, 0.356, 0.553, 0.463 — healthiest it has been! |
| nav_target | 0.965, 0.976, 0.961, 0.962, 0.977 ✓ |
| weapons_target | 0.759, 0.758, 0.817, 0.710, 0.705 ✓ |
| movement | 0.972, 0.971, 0.946, 0.935, 0.933 ✓ |

**Concerns:**
1. **⚠⚠ Training stalled for ~7 hours overnight.** Only 11 cycles of actual progress. The machine likely slept. This is the second throughput anomaly in two checks — whatever is causing system pauses is recurring.
2. **⚠ Miner population / activity crash.** Miner slot counts dropped to ~25% of normal. Miner reward totals dropped 5-7× across every channel. It's possible this is a sampling artefact from the thin window, but it could also be that miners are dying and not respawning. **Worth investigating once more cycles complete** — if the miner slot counts are still low at the next check with more cycles elapsed, something is actually wrong.
3. **cargo_sold still volatile** (-1.72 outlier in the last 5). Hasn't cleanly stabilised.
4. **Landing value head slipped back** from +0.38 mean last check to +0.31 with one -0.56 outlier. Not catastrophic but not fully recovered.

**Positives:**
- **Pickup value head healthiest of the run** (avg 0.42, was ~0.20-0.27 in prior checks).
- **Health, nav_target, weapons_target, movement value heads all rock-solid** (>0.70).
- **No NaN**, total EV avg 0.73.
- **Trader behavioral improvement holding**: health 54%, movement 18%, weapons_target still elevated (+158%) but weapons_target share only 2%, so it's a stored policy preference rather than currently dominant reward.
- **Fighter weapon_hit dominance** climbed to 29% — fighter is genuinely engaging in combat.

**Assessment:** The training health is fine when it's running — the issue is that it *wasn't* running for most of the last 7 hours. **Highest priority next:** verify that training is actively progressing, and if the miner slot counts / reward totals haven't recovered by the next check, dig into whether miners are actually spawning correctly.

**Decision:** Do not intervene on the training side. The value heads and reward balance are all within normal ranges for this run. The stall is a system-level issue (likely machine sleep), not a training failure. Continue monitoring; the next check will show whether cycles accumulate normally and whether miner activity recovers.

**Action items:**
1. Check if the machine went to sleep overnight and consider setting `caffeinate` or similar to prevent it.
2. If the miner slot counts are still ~0.4 at the next hourly check with a full hour of cycles, this is a real miner population issue — investigate ai_ships spawn logic or map state.

---
### Cycle 17100 — 08:05

**Training:** policy_loss=0.0008, entropy=3.813, clip=0.064, EV=0.731, NaN=0

**Landing:** fighter=352/17100, miner=165/17100, trader=132/17100 (total=649)

**Landing EV:** [0.362, 0.200, 0.185, 0.265, -0.220]

**Effective reward/step (last 20):**

- **fighter**: health=0.0160, weapon_hit=0.0019, nav_target=0.0016, weapons_target=0.0006
- **miner**: health=0.0218, nav_target=0.0015, movement=0.0007
- **trader**: health=0.0290, nav_target=0.0009, movement=0.0010

---

### Cycle 17126 — 07:13 (final hourly check before stop)

**Backup:** experiments/run_16/backups/20260411_0713

**Throughput:** 17086 → 17126 = **+40 cycles** in ~1h05m. That's only ~37 cycles/hour vs the expected ~140-150/hour. Something is still slowing the run, though it's not a complete stall like overnight. Cycle times in the monitor are 21–38s with one 38s outlier.

**Training health:** NaN=0/17118. Entropy=3.80–3.83. Clip=0.06–0.07. Total EV last 5: [0.832, 0.718, 0.576, 0.524, 0.524] — **trending down**, last two cycles at the 0.52 floor.

**Landing counts (cumulative, deltas vs last check):**
- fighter: 353 (+1)
- miner: 166 (+1)
- trader: 132 (+0)
- total: 651 (+2)

Only +2 landings in 40 cycles — about a 5% rate, lower than the recent 7-10% peak.

**Reward dominance (last 20):**
| | health | weapon_hit | nav_target | weapons_tgt | movement | pickup |
|---|---|---|---|---|---|---|
| fighter | 24% | **34%** | 26% | 9% | 6% | 2% |
| miner | 30% | 10% | 19% | 2% | 9% | **30%** |
| trader | **54%** | 3% | 15% | 4% | 16% | 7% |

Fighter weapon_hit dominance climbed to **34%** — highest yet, fighter is heavily combat-focused. Miner pickup share at 30%.

**Miner population recovered:** slot counts back to ship=1.43, asteroid=2.91, planet=2.0, pickup=2.0 — confirms the previous miner crash was a sampling artifact from the thin window, not a real population issue.

**Growth (first 200 → last 200):**
- **fighter**: weapon_hit +29%, **landing +177%, cargo_sold +209%**, pickup +5%, nav_target -16%, movement -77%
- **miner**: weapon_hit -6%, **landing +44962%**, cargo_sold +806%, pickup +8%, nav_target -7%, weapons_target +6%, movement -54%
- **trader**: weapon_hit +21%, landing -81%, cargo_sold -89%, nav_target -77%, **weapons_target +193%**, movement -68%

**Value head EV (last 5):**
| head | last 5 |
|---|---|
| health | 0.983, 0.981, 0.984, 0.986, 0.982 ✓ |
| weapon_hit | 0.154, 0.441, 0.262, 0.238, 0.363 |
| **landing** | 0.463, -0.179, 0.651, 0.252, 0.463 — avg +0.33 |
| **cargo_sold** | 0.057, 0.248, -0.369, -0.211, **-2.549** — degrading again |
| pickup | 0.350, 0.507, 0.322, 0.318, 0.270 |
| nav_target | 0.965, 0.936, 0.957, 0.973, 0.942 ✓ |
| weapons_target | 0.508, 0.582, 0.580, 0.678, 0.580 |
| movement | 0.946, 0.953, 0.930, 0.957, 0.914 ✓ |

**Concerns:**
1. **Total EV dropped to 0.52 in the last two cycles** (was 0.83 → 0.72 → 0.58 → 0.52 → 0.52). Approaching the 0.5 intervention threshold.
2. **cargo_sold head spiking negative** (-2.55 in last cycle). Hasn't stabilised over the run.
3. **Throughput still depressed** (~37 cycles/hour, about 25% of normal).
4. **weapons_target EV softened** to 0.58 mean (was 0.78 earlier in the run).

**Positives:**
- **No NaN.**
- **Miner population recovered** — confirms previous crash was a sampling artifact.
- **Fighter weapon_hit dominance at 34%** — highest of the run.
- **Health, nav_target, movement** value heads all >0.91 — dense rewards remain rock-solid.

**Final overnight totals (cycles 8350 post-restart → 17126):**
- Fighter: 226 → 353 (+127 landings)
- Miner: 112 → 166 (+54 landings)
- Trader: 103 → 132 (+29 landings)
- **Total: 441 → 651 (+210 landings over the overnight run)**

---

### STOP requested at 07:13

User requested training stop. The training process should be stopped externally; this script does not directly control the training process. Final state: cycle 17126, NaN-free, total EV trending down to 0.52 over last 5 cycles. Best checkpoint of the recent run is likely backups/20260410_2304 (cycle ~17075, when total EV was 0.69 avg and landing rate was at its peak).


### 07:47 — Cron cancelled

User requested training stop at 07:13. Cycle counter at 07:47 = 17136 (+10 since stop request), so training is still running but the user's external stop has not yet taken effect.

The recurring hourly check cron has been cancelled in this session. No further automated hourly checks will run.

