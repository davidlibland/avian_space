# Economic Loop (Miners) — Learnings & Methodology

How we got miners to actually **mine → fill → land → sell for a profit**, the dead
ends, the diagnostic methodology, and the tooling left behind. Branch:
`coop-attention`, experiment: `run_4`.

> Status: economic loop being driven to convergence by `scripts/miner_fill.sh`.
> The **Final converged config** section is filled in when that daemon logs `DONE`.

---

## The problem

Miners (`Personality::Miner`, faction `Merchant`) would shoot asteroids but never
ran a profitable loop — `cargo_sold` sat at ~0.00002/step for a long time. Getting
them to mine, fill a hold, fly to a planet, and sell turned out to be surprisingly
hard, and most of our first guesses were wrong.

## What actually mattered (the durable lessons)

1. **Reward *structure* beats reward *magnitude*.** `asteroid_hit` rewarded the
   *shot regardless of cargo*, so the optimal policy was "shoot continuously,
   ignore the ore." A full miner kept shooting (still paid per hit) instead of
   selling. Lowering `asteroid_hit` from 6.5→3.5→…→1.0 helped only once it fell
   **below** the collect reward; the lesson is to reward the *outcome you want*
   (ore collected & sold), not a proxy step (hitting rocks).

2. **BC can only teach what the expert *demonstrates in reached states*.** A high
   `bc_coeff` failed at first because the `asteroid_miner`'s 25-unit hold almost
   never reached the 80% sell trigger, so the rule-based expert almost never
   *demonstrated* the sell-route — high BC just cloned shooting. nav-agreement of
   0.97 was misleading: agents agree with the expert in the common (not-full)
   states; the disagreement is concentrated in the *rare* "time to sell" decision.

3. **A small hold forces the loop.** The fix that unstuck it was a new ship, the
   **`prospector`** (cargo 6 vs 25), which fills fast and *must* ferry ore to a
   planet to sell. Half the miners in the mining belt are prospectors.

4. **Fill rate is a real constraint.** A 25-unit hold takes ~14k–48k steps to fill
   (≈2.5 ore/pickup, one laser hit ≈ one drop). Ships persist across RL segments
   (`RL_SEGMENT_LEN=512` is just the training window) and the mining belt is pinned
   (no swap) with no jump exits, and miners rarely die — so resets/swaps/death were
   **not** the blocker; the hold was simply too big to fill before they gave up.
   We made `asteroid_drop_scale` a runtime knob and started it high (4.0) so a
   prospector fills in ~1 hit, then titrate it back down.

5. **The decisive diagnosis came from *behavioral* instrumentation, not reward
   curves.** Reward/step is confounded (e.g. fill-shaping lowers the number while
   behavior improves). Logging *what ships do at landing* settled it (below).

## The decisive diagnosis (expert vs policy A/B)

Added gated `[DIAG]` landing logging (`DIAG_LANDINGS=1`) + a `--starting-system`
flag, then compared the pure rule-based **expert** (`--classic`) against the
**trained policy** on the mining system, measuring hold fullness *at landing*:

| metric (miners) | Expert | Trained policy |
|---|---|---|
| mean cargo_frac at landing | **0.97** | **0.49** |
| % full (≥0.8) | 98% | 43% |
| % that actually sell | 99% | 56% |
| % heading to a planet (sell-route) | 100% | 98% |
| trader fill (reference) | 0.93 | 0.91 |

**Conclusions:**
- The expert is a *near-perfect* miner → not an expert/environment/reward-path bug.
- RL learned **traders** almost perfectly → RL *can* learn land-and-sell.
- RL **miners make premature, half-loaded sell trips**: targeting is right (98%
  head to a planet) but they arrive ~half full. **Root cause:** a trader fills
  *instantly* by buying; a miner fills through a *slow multi-step mining chain*.
  The policy learned "go sell" but not the **patience to fill first** — a
  credit-assignment / fill-discipline gap, not a target-choice gap.

## The fix: superlinear fill shaping → hand off to RL

- New config knob **`cargo_sold_fill_exponent`**: the sale reward is scaled by
  `cargo_frac^(exponent-1)`, so a full hold pays disproportionately more than
  partial loads (full ≈ 3× two half-loads at exp=2.5). It's global but
  *self-targeting* — full-selling traders are ~unaffected (mult≈1); half-full
  miners get docked, teaching "fill before selling."
- Curriculum (`scripts/miner_fill.sh`): **CLIMB** the exponent until miner mean
  fill-at-landing clears `FILL_TARGET=0.75`, then **HANDOFF** — ramp `bc_coeff`
  back *down* toward 0.1 so rewards (not imitation) drive the policy, verifying
  fill persists. (RL is the right long-term driver; BC is scaffolding.)
- Result: **when miners sell, they sell ~84% full** (prospectors **0.94**, near the
  expert's 0.97; asteroid_miners 0.82), and ~53% of landings are sales. The loop
  largely works. The earlier "fill ~0.5" alarm was a **confounded metric** —
  averaging fill over *all* landings, ~47% of which are *empty* repair/incidental
  stops (fill ≈ 0), which caps the all-landings mean near 0.55 no matter how full
  the sells are. Measuring fill on **sell** landings (credits>0) is the correct
  signal and shows the fill-discipline fix succeeded.

## Reusable methodology

- **Measure behavior, not just reward.** Gated `[DIAG]`/`[COOP]` `println!`
  samplers (env-flag → log lines → offline aggregation) are cheap and decisive.
- **Expert vs policy A/B.** `--classic` (pure rule-based) vs the live policy on a
  *dedicated single system* (`--starting-system`) isolates "is the expert weak?"
  from "is RL failing to imitate/learn it?" — this is what cracked the case.
- **Autonomous tuning daemons.** Self-contained `nohup` shell daemons poll
  metrics (tbparse + log aggregation), edit `training_config.yaml`, restart
  `run_4`, and log every decision to a run-dir `.md` — no per-step approval, and
  they survive the session. See `scripts/miner_*.sh`.
- **Pitfall:** `--inference` single-world loads a *random* net (no checkpoint
  load) — use the live multiworld run to evaluate the trained policy.
- **Pitfall (condition your metric):** "mean fill at landing" over *all* landings
  is confounded by empty repair/incidental stops and plateaus ~0.55 even when
  every sale is full. Condition on the event you care about — fill on *sell*
  landings (credits>0). The same applies broadly: average an outcome over the
  decisions that actually produce it, not over all events.

## Tooling / knobs added (reproducibility)

- `training_config.yaml`: `asteroid_drop_scale` (was a const), `cargo_sold_fill_exponent`.
- Ships: `prospector` (`assets/ships.yaml`, sprite via `scripts/generate_sprites.py`);
  added to the mining belt (half the miners) and the regular galaxy (spawns +
  shipyards where the miner is sold or asteroids are above-median).
- CLI: `--starting-system`. Instrumentation: `DIAG_LANDINGS`, `DIAG_COOP`.
- Daemons: `scripts/miner_fill.sh` (active), superseded `miner_bc_ramp.sh`,
  `miner_profit_v2.sh`, `miner_dropscale.sh`.

## Final converged config (CONVERGED 2026-06-24 18:38, `run_4`)

The `miner_fill.sh` curriculum finished: CLIMB the fill-exponent until sell-fill
cleared 0.75 (14:48), then HANDOFF — ramp `bc_coeff` 0.50→0.40→0.30→0.20→**0.10**
(floor), verifying sell-fill held at each step. **It held the whole way down**
(0.79–0.86), so the loop is now **RL-driven** with only a light BC anchor.

- `cargo_sold_fill_exponent`: **3.0** (superlinear fill shaping)
- `bc_coeff` (after handoff): **0.10** (the floor — RL drives; BC is a faint nudge)
- `asteroid_drop_scale`: **4.0** (titration toward a smaller floor was deferred —
  fast fill keeps the loop tight; can be lowered later if desired)
- Other miner weights: `asteroid_hit_miner=1.0`, `pickup_reward_miner=1.7`,
  `cargo_sold_miner=1.5`
- **Converged behavior:** miners that sell land **~0.83 full** (prospectors
  **~0.94**, ≈ the rule-based expert's 0.97); sellrate ~0.3–0.5 of landings.
  nav-agreement stayed **~0.95–0.99 even at `bc_coeff=0.10`** — the policy
  *internalized* the mine→fill→land→sell loop rather than leaning on BC.

**Bottom line:** the miner economic loop is solved — RL (not imitation) drives
miners to fill their holds and sell them ~full. *Residual / future polish:* a
chunk of landings still carry no cargo (repair/incidental stops), so **sell
frequency** (not fill) is the remaining efficiency lever if we want miners to
spend more of their time on profitable trips.

## Cooperation (deferred — the harder problem)

We can now *measure* it (`[COOP]` sampler: focus-fire, ally grouping, escort
coverage, threat response). Finding: the policy's cooperation is around/slightly
below the rule-based baseline and **does not improve on its own** — the
economic-focused training mildly crowds it out. Growing it will need its own
lever: raise `reward_sharing_*` weights, or build the Phase 2/3 inter-agent
attention architecture (the branch's original aim). Tracked separately.
