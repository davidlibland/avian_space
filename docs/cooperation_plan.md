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

## Centralized critic / decentralized policy (CTDE) — plan
**Today:** the `value_net` is an `RLNet` run *per-agent on the agent's own obs*
(`value_fn.rs`), which includes only the `SELF_TEAM_*` *summary*. So it is mildly
team-aware but **not** a centralized critic — it never sees the full joint/team
state. Cooperative rewards (assist, shared outcomes) have notoriously bad credit
assignment from a single agent's view; a centralized critic fixes that **at
training time only**, leaving execution decentralized (policy unchanged).

**Recommended approach — A: team-pooled value input (minimal CTDE).**
- Compute, at obs-build time (`build_all_observations`, which already processes all
  agents in a world together), a **per-faction pooled team embedding** — a
  permutation-invariant mean (or attention) pool over *all* same-faction allies'
  self-states (health, pos/vel, distress, weapons-target, personality), not just
  the in-range summary.
- Feed that pooled vector **only to the value net** (concatenate into the value
  trunk). The **policy net keeps its current per-agent obs** → decentralized
  execution preserved (CTDE).
- Plumbing: the trainer recomputes the value on stored transitions (`value_fn.rs`),
  so the pooled team vector must be **stored in the segment** alongside the obs (or
  stored ally states to re-pool). Extend `Transition`/`Segment` with the team
  vector.
- Net change: value trunk takes an extra input; `VALUE_*` heads unchanged. Retrain
  (or fine-tune from the current checkpoint).
- Effort: **medium**. Risk: more value params; pooling must handle variable agent
  counts (mean-pool or masked attention); only pays off because we now have a
  cooperative reward to credit.

**Later — C: attention critic.** Replace the mean-pool with a set-transformer /
masked-attention pool over allies in the value net. More expressive; a stepping
stone to (and shares code with) the Phase-2 *policy* attention. Keep execution
decentralized by attending only in the critic.

**Not recommended first — B: full global-state MAPPO critic.** Concatenate every
agent's full state; powerful but heavy and awkward with variable agent counts.
Start with A.

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
