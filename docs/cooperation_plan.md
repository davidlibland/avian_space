# Cooperation: plan, implementation & measurement

The longer challenge (vs the economic loop). Goal: get same-team ships to *actually*
cooperate — focus-fire, escort/defend merchants, and have merchants defend
themselves — and to **measure** it rather than infer from the reward-share proxy.

## Where cooperation comes from (two ingredients)
1. **Incentive** that *demands* cooperation. Reward-sharing alone (`reward_sharing_*`)
   was blunt → flat, mediocre cooperation.
2. **Capability** to coordinate. We have a hand-aggregated team-context obs block
   (`SELF_TEAM_*`); not yet a learned inter-agent attention layer or a centralized
   critic.

The incentive was the bigger gap, so we started there.

## Implemented (2026-06-26)
- **Merchant self-defense (#1)** — `choose_target_slot` rule 0: a *distressed*
  Trader/Miner (`obs[SELF_DISTRESSED] > 0.3`) under attack breaks off its economic
  task and targets its attacker (the ship `is_targeting_me`, else nearest
  hostile/should-engage). Returning that ship as the **nav** target makes the
  weapons-target logic lock onto it too → it faces and fires back. It's in the
  **rule-based expert**, so BC teaches it. *Verified:* pure-expert `merch_def` went
  from ~0 to **0.94** (94% of threatened merchants return fire).
  - Foundations confirmed: pirates already register `is_hostile=1` to merchants
    (`enemies.yaml: Merchant:[Pirate]`), and an *intentional* hit flips the
    victim's `should_engage` on the attacker (`score_hits`, gated to `on_target`
    so stray/friendly fire doesn't turn allies hostile).
- **Cooperative-assist reward (#2)** — config `cooperative_assist_bonus` (0.5): a
  Fighter that hits a ship which is itself attacking a **nearby ally** (within
  `ASSIST_RADIUS=800`) earns a bonus on its ship-hit reward. This is escort +
  threat-interception + focus-fire unified into "kill what's hurting your
  teammate." Reward-driven, so it works at any `bc_coeff`.
- **Metric (#4)** — `[COOP]` sampler gained `merch_def` (fraction of threatened
  merchants returning fire). See Measurement below.

> Note: the self-defense rule lives in the BC expert, so `bc_coeff` was bumped
> 0.10 → 0.30 to teach it (cold-start: at 0.10 the policy won't try it often enough
> to get the reinforcing reward). Wean back down once it's learned (teach-then-
> handoff, like the economic loop). The assist reward needs no BC.

## Centralized critic / decentralized policy (CTDE) — implementation plan

### Where we are (verified in code)
- The **value net** is a *separate* `RLNet` instance: `RLNet::new(device,
  VALUE_HIDDEN_DIM, VALUE_OUTPUT_DIM)` (`model/training.rs`) — so we can change its
  architecture **without touching the policy net**.
- It runs *per-agent on the agent's own obs*: `value_fn::batch_value_inference`
  takes `self_flat`/`obj_flat`/`proj_flat` (assembled from each step's
  `Transition.obs`/`proj_obs` via `ppo/buffer.rs`) and calls
  `RLNet::forward(self_t, obj_t, proj_t)`.
- The obs already contains **in-range** ally slots (`friendly_ships`) + a
  `SELF_TEAM_*` summary, and the net already has **entity attention** (`ent_q/k/v`)
  over object slots. So the critic is *locally* team-aware — but it never sees
  out-of-range allies or a holistic team state, so advantages for cooperative
  rewards (assist, shared outcomes, "we ganged up and killed the pirate") are
  high-variance.
- Rollout builds obs for **all agents in a world together** (`build_all_observations`),
  so a per-world team state is cheap to compute there.

### Recommended: **A — team-pooled value input** (minimal, true CTDE)
Condition the value net (only) on a per-faction **team-state vector** pooled over
*all* same-faction agents in the world. Policy net is unchanged → decentralized
execution. The team vector is needed only at *training* time.

**Step 1 — compute the team-state vector** (`rl_collection::build_all_observations`).
After gathering the world's RLAgent ships, compute per faction a fixed-size
`TEAM_STATE_DIM` (~16–24) permutation-invariant, frame-invariant summary:
- `n_allies` (norm by ref), `mean/min health_frac`, `frac_distressed`,
  `frac_engaging` (has a hostile weapons-target), `mean cargo_frac`,
  positional **spread** (std of positions — translation-invariant), `n_enemies`
  in system / mean enemy proximity, `mean ally–nearest-enemy distance`,
  `n_allies_with_shared_target` (focus-fire potential).
Mean-pool ⇒ permutation-invariant + variable-count-safe. Computed once per faction
per step, shared by all that faction's agents.

**Step 2 — store it** in `Transition` and `Segment` (add `pub team_state:
Vec<f32>` next to `obs`), thread through the flush
(`rl_collection` ~L2069/L2090) and the trainer flattening (`ppo/buffer.rs`: add a
`team_flat` alongside `self_flat`/`obj_flat`/`proj_flat`; `StepData` gains
`team_feat`). Also store the **last-step** team vector (for
`recompute_bootstrap_values`).

**Step 3 — value-net architecture** (`model/net.rs`). Add an optional team branch
to `RLNet`:
- new field `team_embed: Option<Linear<B>>` (built only when a `team_input: bool`
  ctor arg is true; policy passes false, value passes true);
- `forward(self_t, obj_t, proj_t, team: Option<Tensor<B,2>>)` — when `team` is
  `Some`, `silu(team_embed(team))` is **added into the merged trunk** just before
  the `output` head (same place the self/entity/proj streams merge). Policy calls
  with `team = None` → identical behaviour to today.
- `VALUE_*` head dims unchanged. ~`TEAM_STATE_DIM×hidden + hidden` extra params.

**Step 4 — pass it through training** (`value_fn.rs`): `batch_value_inference`,
the gradient value-loss path, and `recompute_bootstrap_values` each gain a
`team_flat` arg, build `team_t: [B, TEAM_STATE_DIM]`, and pass `Some(team_t)`.
Policy forward (inference + PPO policy loss) passes `None`.

**Step 5 — execution unchanged**: the game-thread inference uses the **policy**
net (`team = None`), so no team state is needed at runtime → decentralized
execution preserved. This is the CTDE property.

**Gating & retrain.** Put it behind a config flag `ppo.ctde_enabled` (default
false) so it merges safely and we can A/B. Because the value-net *architecture*
changes, the old `value.bin` won't load — **re-init the value net** (keep
`policy.bin`, which is what execution uses) and let it re-warm (value burn-in
handles this); the policy keeps training throughout.

**Effort:** medium (net.rs + value_fn + Transition/Segment/buffer plumbing + one
config flag). **Risk:** more value params; team summary must be informative; only
pays off because cooperative rewards now exist to credit.

**A/B success criteria:** vs the non-CTDE run — (1) higher explained variance on
the `ship_hit`/assist channels, (2) lower advantage variance, (3) the `[COOP]`
behaviors (focus_fire, threat_resp, escort_cov, merch_def, merchant `ship_hit`)
rise faster / higher. Same expert-vs-policy escort A/B harness.

### Later — C: attention critic
Replace the mean-pool with masked **attention over per-ally embeddings** in the
value path (reuse the existing `ent_q/k/v` machinery, applied to the full team set
rather than just in-range slots). More expressive; shares code with the eventual
Phase-2 *policy* attention. Keep execution decentralized (attend only in critic).

### Not first — B: full global-state MAPPO critic
Concatenate every agent's full state into a global vector. Powerful but heavy and
awkward with variable agent counts; A (pooled) gets most of the benefit for far
less. Revisit only if A's pooled summary proves too lossy.

## Other levers (queued)
- **Friendly-fire obs feature.** Fed/Rebel fighters *are* penalized for hitting
  merchants (neutral-hit penalty in `score_hits`), but the obs has no explicit
  "ally in my line of fire" signal — only ally positions + own target (inferable,
  not explicit). Add an angular line-of-fire / friendly-fire-risk feature so they
  learn to hold fire / reposition.
- **Phase 2 inter-agent attention (policy).** The branch's original aim — learned
  attention over allies in the *policy* for emergent coordination. Biggest lift;
  do after the reward + critic make cooperation demanded and well-credited.
- **Make cooperation necessary** (environment): tanky pirates requiring focus-fire,
  or waves that overwhelm an unescorted merchant.

## Measurement (already in place)
`DIAG_COOP=1` → `[COOP]` sampler (per system, every 5s). Methodology = expert
(`--classic --starting-system escort`) vs live policy A/B.
- `focus_fire` — engaging fighters whose target an ally also attacks.
- `allies_near` / `enemies_near` — local grouping / outnumbering.
- `escort_cov` / `mean_def_dist` — merchants with a defender in range / tightness.
- `threatened` / `threat_resp` — threatened merchants with a defender nearby.
- `merch_def` — **new**: threatened merchants returning fire (self-defense).
Outcome metrics from TB: per-personality `ship_hit` (miner/trader going positive =
they fight back; fighter assist shows in its ship_hit).
Baselines: expert `merch_def≈0.94`; policy cooperation was `focus_fire≈0.42`,
`escort_cov≈0.31`, `threat_resp≈0.25` (below expert) — targets to beat.
