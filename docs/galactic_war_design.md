# Galactic War — design (implemented 2026-07)

Systems change hands between factions through player-driven missions; the
galaxy's markets, standings, traffic, and mission offers all follow
automatically.

**Status**: phases 1–3 and the news/map polish of phase 4 are BUILT
(src/galaxy.rs, src/war.rs, the Escort/CarriedBy split in src/carrier.rs,
War/Covert/Arrest templates in assets/mission_templates.yaml). Two design
notes from implementation: fronts are usually fought over the UNALIGNED
buffer systems separating faction cores (direct enemy borders are rare on
the real map), and campaign tiers key on the SPONSOR's progress toward the
control threshold. The FULL espionage family (§5) is implemented: all 11
mission shapes as covert templates, including two-stage extractions
(primary NpcOffer + auto-starting `__return` leg, precondition-locked on
the primary, effects on the final stage), cut-supply freighter hunts with
a Merchant standing cost, and negative-pay bribes. The Federation–Bastion
war got its venue: `the_marches`, an unaligned contestable buffer wired to
contestable systems on both sides (epsilon_eridani / coldforge) so a won
buffer becomes a springboard. Remaining phase-4 ideas (war-scarcity
prices, the planted-intel *weakening hook* — the mission itself exists —
and Elo logging) stay future work.

The guiding principle is the one the codebase already lives by: *state is
data, consequences are derived.* Faction control is one number per
(system, faction); everything else — markets, spawns, arrests, prices,
offers, war missions — derives from it through machinery that already
exists.

---

## 1. Control model: an influence simplex per system

```
GalaxyControl (session resource, persisted like FactionStandings):
    system → { faction → influence }    // entries ≥ 0, sum ≤ 1
                                        // remainder = "unaligned"
```

* **Controlled**: top faction ≥ `CONTROL_GAIN` (0.6) → effective controller.
* **Contested**: nobody qualifies → the system is effectively neutral.
* **Hysteresis**: control is *gained* at 0.6 but only *lost* below 0.5, so a
  see-sawing front doesn't flip-flop ownership every mission.
* Seeded at session start from the static assets (`system.faction` → 1.0).
  The YAML remains the *initial* galaxy; the session resource is the *live*
  one (same overlay pattern as standings vs. ship data).

**`contestable: true` per system, default false.** Sol and the faction
capitals are never contested — story arcs hard-reference their planets
("meet the recruiter on Helios Prime"), and letting the war flip them
produces narrative absurdity. Wars are fought over the frontier.

### Derived consequences (all existing machinery)

| Consequence | Mechanism |
|---|---|
| Markets restock | effective seller faction = controller, or **Independent when contested** → contested systems stock only universal gear (war zones degrade to surplus). Re-run `derive_market_catalogs` for the affected system on a `SystemControlChanged` message. |
| Standings follow | `controlling_faction()` reads GalaxyControl — hit bonuses, arrests, price markups, offer gating all track the new owner with zero new code. |
| Traffic shifts | ship presence derives from influence (see §2) — contested systems spawn both sides, and `enemies.yaml` hostility makes them **fight each other live**. The war is visible without scripting a battle. |
| Star map | jump UI colors systems by controller; contested = hatched/neutral. |

### Moving influence

* `CompletionEffect::ShiftInfluence { system, faction, delta }` — applied by
  the galaxy module watching `MissionCompleted` (the `AdjustStanding`
  pattern; missions stay decoupled).
* **Ambient drift**: a small random walk on both sides of each active front
  per landing/day — the war moves without the player; player missions are
  the decisive strokes.

---

## 2. Dynamic ship presence (replaces authored `ships:` maps)

The same simplification as the market derivation: per-system hand-curated
spawn tables become derived quantities. Ships already carry every input the
model needs (`faction`, `personality`, `tech_level`).

```
population(s)   = traffic_scale(s) × ( k_p · Σ planet tech levels
                                     + k_a · #asteroid fields
                                     + k_c · #jump connections )

merchants(s)    ∝ base + commerce(s)          // commerce = landable planets
                                              //  (≥2 → real routes) + connections
miners(s)       ∝ asteroid fields (Σ field.number)
pirates(s)      ∝ merchant traffic            // predators follow wealth;
                                              // scaled by the UNALIGNED share
combat(F, s)    ∝ presence(F, s)              // per faction F, see below
```

**Faction presence with λ-propagation** (fronts get depth):

```
presence(F, s) = influence(F, s)
               + λ  · Σ_{n ∈ neighbors(s)}   influence(F, n)
               + λ² · Σ_{n ∈ 2-jump(s)}      influence(F, n)
λ ≈ 0.4, then normalize across factions
```

A war next door bleeds enemy patrols into border systems *before* they're
contested — and "cut the supply lines" espionage becomes mechanically real:
lowering a neighbor's influence reduces spillover here. Recomputed only on
influence change (change-detection gated), from precomputed neighbor lists.

**Within-faction ship mix** — no new data for v1:

```
weight(ship) = presence_weight / 2^(tech_level − 1)
```

Small ships common, capitals rare, automatically consistent with the market
model. `presence_weight` is an optional per-ship override (default 1.0).
Role buckets come from the existing `personality` (Trader/Miner/Fighter);
faction merchants and miners use the same presence formula as combat ships.

**v3 experiment — Elo-rated combat weights**: the RL infrastructure already
logs kill outcomes, so per-ship-type ratings come nearly free. Caveat: pure
Elo makes *winning* designs more common (militaries field what wins), which
can snowball; game balance likely wants an underdog term or rating-as-price
rather than rating-as-frequency. Log first, decide later.

**Guardrails**
* Training systems (simulator/escort/mining) keep their authored `ships:`
  distributions — RL worlds must not shift under the trainer. Any system
  with an explicit `ships:` block keeps it (authored = override, exactly
  like explicit outfitter/shipyard entries in the market derivation).
* Validator: `check_trade_routes` (merchants need ≥2 colonized planets)
  holds by construction under derivation but stays as the regression net;
  add a check that derived populations are non-degenerate (no system with
  traffic but zero possible spawn types).

---

## 3. Escorts: split "carried" out of the escort concept

`CarrierEscort { mother, weapon_type }` currently conflates two ideas:

* `Escort { mother }` — follow/attack/orders (B/N/M command layer,
  `update_escort_modes`, orphaning — all reusable unchanged).
* `CarriedBy { weapon_type }` — *only* the dock/replenish behaviors gate on
  this.

**Squadron escorts** = `Escort` without `CarriedBy`: fly with the player,
take orders, cannot dock, despawn at mission end (mission-target cleanup
pattern, tagged with the mission id). Spawned by a `spawn_mission_escorts`
system (the `spawn_mission_targets` analog) when a mission with a
`squadron:` spec is active and the player is in the battle system.

---

## 4. War missions: one template family, tiered per front

`DestroyShips` + `spawn_mission_targets` already does "spawn N hostiles,
count kills". The four battle types are one `WarBattle` mission template
(the `Arrest` pattern: ad-hoc, regenerated, all text data-driven with
substitution vars) parameterized by side/scale/squadron:

| Mission | Parameterization |
|---|---|
| Raid | enemy system, small count, no squadron |
| Rapid defense | friendly system (invaders = mission targets), no squadron |
| Defensive battle | friendly system, big count, `squadron:` support |
| Offensive battle | enemy system, big count, squadron |

**Front generator** (the arrest-generator pattern): a front = a jump-graph
edge between enemy controllers (per `enemies.yaml`). The generator offers
war missions at faction worlds near the front, **tiered by how contested
the front is**: lopsided influence → raids; closing on the threshold →
squadron battles; at the threshold → a decisive battle that flips the
system. Each front reads as a campaign with an arc, for free.

**Standing is the war credential**: war missions require standing ≥ +10
with the hiring faction. Fighting for A tanks standing with B (automatic
via hit penalties), locking the player out of B's markets galaxy-wide.
Choosing a side has real, fully-derived consequences.

---

## 5. Espionage: loyalty shifts without fleets

Mechanical identity: little-to-no combat, offered by the *sponsor* at their
worlds, executed in *enemy* space, failure costs standing with the target —
and bad standing in enemy space already risks the **arrest flow**, so the
espionage tension is pre-built. All map to existing objective kinds:

| Mission | Objective machinery |
|---|---|
| Capture a government figure | `catch_npc` |
| Steal military defense docs | `meet_npc` / `collect` |
| Gather evidence of corruption | `meet_npc` chain |
| Instill revolutionary ideas | delivery + `meet_npc` chain |
| Arm the partisans | delivery of weapons_parts to enemy world |
| Extract the defector | 2-stage meet + land-home (CatchThief shape) |
| Cut the supply lines | `destroy_ships` on enemy *freighters* (influence shift + Merchant standing hit — a moral cost) |
| Bribe the governor | negative-Pay mission: spend credits for a quiet, large shift |
| Hearts and minds | relief delivery to a contested world |
| Plant false intelligence | delivery; v2 hook — enemy's next defense mission spawns weaker |
| Rescue POWs | catch/meet + return; standing *and* influence |

---

## 6. Build order (each phase ships independently)

1. **Influence substrate** — GalaxyControl + thresholds/hysteresis +
   `ShiftInfluence` + market re-derivation on change + **derived ship
   presence (§2)** + star-map/info-tab display. Already fun with no war
   missions: contested systems visibly skirmish.
2. **Squadron escorts** — `Escort`/`CarriedBy` split + `squadron:` spawn
   and cleanup.
3. **War templates + front generator** — battle tiers + espionage family +
   ambient drift.
4. **Polish** — galactic news toasts ("Federation seizes Vega"),
   war-scarcity price effects, planted-intel hooks, Elo logging.

Testing follows the established patterns: pure math (thresholds,
hysteresis, λ-propagation, population formulas), derivation unit tests
(market restock on flip, spawn mixes), and runtime-fixture mission chains
for the generator (the arrest tests are the template).

## 7. The war map (2026-07)

Every mutual-enemy pair with plausible geography has a two-way front whose
target holds a landable covert venue:

| War | Fronts (sponsor: home → target) |
|---|---|
| Federation–Rebel | sol→alpha_centauri, barnard→drift / sirius→alpha_centauri, procyon→drift (venues: centauri_post, drift_station) |
| Federation–Bastion | kepler_22 & epsilon_eridani → the_marches / iron_march & coldforge → the_marches (venue: marches_freeport) |
| Bastion–FreeFrontier | iron_march → drumlin & the_barrens / drumlin & lowmark → the_barrens (venue: barrens_hold) |
| Helios–Order | the_foundry → pilgrims_deep / the_threshold → pilgrims_deep (venue: pilgrims_rest) |

Venue texture: alpha_centauri keeps a planeted outpost (it's a star
system); drift's venue is a REMOTE STATION (a landable habitat, no world);
the_gyre and the_long_dark are truly empty wilderness — unaligned pirate
pockets no front targets, so they need no venue at all.

Bastion–Rebel and FreeFrontier–Helios stay COLD by design: their cores sit
on opposite ends of the map with third factions between them — the enmity
lives in the data (hit bonuses, hostility) but no front generates.
`every_geographic_war_is_two_way_with_covert_venues` pins the table.

Traffic tuning (2026-07): COMBAT_PER_PRESENCE 2.2→3.5, pirate share
0.35→0.5, mercenaries 0.25→0.3, population clamp 3-8→4-10, λ 0.4→0.5,
population check 5s→4s — denser skies, deeper cross-border patrols, more
frequent skirmishes.

## 8. Open questions

* Does *anything* happen at galactic domination, or is it a sandbox state?
  (Lean: news toast + achievement-flavor, no mechanical end.)
* Economy under war: should contested systems get commodity scarcity
  multipliers? (v2 flag, one function.)
* Should AI factions launch offensives that flip systems with no player
  involvement, or does ambient drift only *pressure* fronts the player
  opened? (Lean: drift-only until playtesting says the galaxy feels static.)
* λ, k_p/k_a/k_c, and role-mix constants want a tuning pass with the
  traffic monitor once derived spawns are in.
