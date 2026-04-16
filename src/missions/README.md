# Missions module

All mission logic is self-contained here. The rest of the codebase only emits
general gameplay events (`PlayerLandedOnPlanet`, `PlayerEnteredSystem`,
`PickupCollected`) and accepts no mission concepts — so new mission features
should almost always be implementable without touching other modules.

## Authoring missions

Missions live in `assets/missions.yaml` (or any `assets/missions/*.yaml` —
subdirectories are merged by the universal asset loader). The top-level keys
become mission ids:

```yaml
deliver_wheat_intro:
  briefing: "The colony on Mars needs grain."
  success_text: "The governor thanks you for the delivery."
  failure_text: "The grain is spoiled. The colony goes hungry."
  preconditions: []
  offer: { kind: bar, planet: earth, weight: 1.0 }
  start_effects:
    - { kind: load_cargo, commodity: wheat, quantity: 10, reserved: true }
  objective: { kind: land_on_planet, planet: mars }
  completion_effects:
    - { kind: remove_cargo, commodity: wheat, quantity: 10 }

mars_followup:
  briefing: "The governor wants you to scout Proxima."
  success_text: "Scouting complete."
  failure_text: "Unreachable."
  preconditions:
    - { kind: completed, mission: deliver_wheat_intro }
  offer: { kind: auto }
  objective: { kind: travel_to_system, system: proxima }
```

### Offer kinds

- `auto`: starts as soon as preconditions are met (story beats)
- `tab { weight: f32 }`: appears in the Missions tab with Bernoulli
  probability `weight` when the player lands anywhere
- `bar { planet: String, weight: f32 }`: appears in that planet's Bar with
  probability `weight` when the player lands there

### Objective kinds

- `travel_to_system { system }`
- `land_on_planet { planet }`
- `collect_pickups { commodity, system, quantity }` — only pickups collected
  in the named system count; purchased cargo does not

### Start effects

- `load_cargo { commodity, quantity, reserved? }` — adds cargo on mission
  accept. `reserved: true` locks the cargo so it can't be sold/dropped until
  the mission completes or fails.

### Completion requirements

`requires` is a list of conditions that must hold the moment the objective
would otherwise complete. If any fails, the mission moves to **Failed**
(not Completed) and its `failure_text` is surfaced.

- `has_cargo { commodity, quantity }` — player must hold at least `quantity`
  units of `commodity`.
- `has_unlock { name }` — the named unlock must be granted.

### Abandon / failure cleanup

When a mission is abandoned or fails, any cargo it loaded at start (via
`start_effects.load_cargo`, reserved or not) is stripped from the player's
hold. Otherwise "accept delivery → cancel → sell cargo" would be a free
exploit.

### Completion effects

- `remove_cargo { commodity, quantity }` — removes up to `quantity` units
  (reserved or not).
- `pay { credits }` — adds credits to the player's balance.
- `grant_unlock { name }` — adds a named flag to `PlayerUnlocks`. Ships
  (`ShipData.required_unlocks`) and outfitter items
  (`OutfitterItem::*.required_unlocks`) with that flag in their required list
  will then start appearing in the shipyard / outfitter UI. Multiple
  missions can grant the same unlock; the set is idempotent.

### Preconditions

- `completed { mission }` / `failed { mission }`
- `has_unlock { name }` — mission only becomes Available once the named
  unlock flag has been granted.

### Acceptance gating

A mission's `Accept` button is disabled when the player's free cargo space
is less than `sum(LoadCargo.quantity)` across the mission's `start_effects`.
`handle_ui_actions` enforces the same check (defence in depth) so
programmatic accepts can't overload the hold.

## Procedural missions (templates)

Templates live in `assets/mission_templates.yaml` as
`HashMap<String, MissionTemplate>`. Each time the player lands, every
template is Bernoulli-rolled against its `offer.weight`; hits are
instantiated into concrete `MissionDef`s with a fresh random id (format:
`{template_id}__{16-hex}`) and added to the appropriate offer list (tab or
the landing planet's bar).

### Template kinds

**`delivery`** — loads a random commodity (from `commodity_pool`) into your
hold and asks you to land at a random destination planet (not the offer
planet). Pays on delivery.

```yaml
random_delivery:
  kind: delivery
  briefing: "Take {quantity} units of {commodity} to {planet_display} for {pay} credits."
  success_text: "..."
  failure_text: "..."
  offer: { kind: tab, weight: 0.4 }
  preconditions: []            # optional; same shape as mission preconditions
  commodity_pool: [food, water, electronics]
  quantity_range: [5, 15]
  pay_range: [500, 2500]
  reserved: true
```

Templates accept the same `preconditions` list as hand-authored missions
(`completed { mission }` / `failed { mission }`). The roll at landing time
skips a template whose preconditions aren't met, so procedural content can
be gated behind story beats.

**`collect_then_deliver`** — two-stage: collect N units of a random commodity
from the asteroid fields of a random system, then deliver them to a random
destination planet. Pays on final delivery. Emitted as two linked
`MissionDef`s sharing the same rolled parameters; stage 2 uses
`offer: auto` with a `completed` precondition referencing stage 1's
generated id, so it auto-starts as soon as stage 1 finishes.

```yaml
collect_then_deliver:
  kind: collect_then_deliver
  stage1_briefing: "Pull {quantity} units of {commodity} from {system_display}."
  stage1_success_text: "Now take it to {planet_display}."
  stage1_failure_text: "..."
  stage2_briefing: "Deliver the {commodity} to {planet_display}."
  stage2_success_text: "..."
  stage2_failure_text: "..."
  offer: { kind: tab, weight: 0.2 }
  quantity_range: [3, 8]
  pay_range: [2000, 5000]
```

**`collect_from_asteroid_field`** — asks you to travel to a random system
that contains asteroid fields and collect pickups of a commodity found in
those fields. Pays on return.

```yaml
random_asteroid_collection:
  kind: collect_from_asteroid_field
  briefing: "Bring back {quantity} units of {commodity} from {system_display}."
  success_text: "..."
  failure_text: "..."
  offer: { kind: tab, weight: 0.4 }
  quantity_range: [3, 10]
  pay_range: [400, 2000]
```

### Placeholders in template text

`{commodity}` · `{quantity}` · `{pay}` · `{planet}` · `{planet_display}` ·
`{system}` · `{system_display}` — substituted at instantiation time into
`briefing`, `success_text`, and `failure_text`.

### Adding a new template kind

1. Add a variant to `MissionTemplate` in [types.rs](types.rs) with the
   tunables it needs (and remember to include it in `preconditions()`).
2. Add a match arm to `instantiate_template` in [progress.rs](progress.rs)
   that builds a `Vec<(id, MissionDef)>` (pick random commodity / planet /
   system / pay, fill in placeholders, produce objective + effects).
3. Document the new kind in this file.

### Multi-stage / chained missions

`instantiate_template` returns `Vec<(id, MissionDef)>`. The first entry is
the "entry point" — it gets offered to the player. Any additional entries
are followup stages inserted straight into the catalog (not offered) with
`offer: auto` and `preconditions: [completed: <earlier_id>]` — the normal
`update_locked_to_available` loop will auto-start them once their gate is
met. This is how `collect_then_deliver` works: stage 1 is the offered
collection mission, stage 2 is a `Locked` auto-start land-at-planet mission
whose precondition references stage 1's freshly-generated id.

This is the same pattern authors can use for static missions: write stage 2
with `offer: auto` and `preconditions: [completed: stage1_id]`. Templates
just do the same thing with random ids generated atomically.

## Extending

All mission state flows through `MissionLog` (per-player statuses) and
`MissionOffers` (currently-rolled tab/bar offers). Logic is split into
`progress.rs` systems that each react to a single event type.

### Add a new objective kind

1. Add a variant to `Objective` in [types.rs](types.rs).
2. If the new objective progresses via an event, add a `fn advance_*` system
   in [progress.rs](progress.rs) that reads that event, matches on the new
   variant, updates the `MissionLog`, and emits `MissionCompleted` when done.
3. Register the system in [mod.rs](mod.rs).
4. Extend `format_objective` in [ui.rs](ui.rs) to display the new objective.

### Add kill-ship objectives later

Preferred pattern (keeps weapons/ship code clean):

1. Add an `Objective::DestroyShip { target_tag }` variant.
2. When the mission starts, spawn the target ship with a mission-owned marker
   component `MissionTarget { mission_id, tag }` (define this component in
   `missions/`).
3. Emit a general `ShipDestroyed { entity }` event from the existing death
   path in `ship.rs::apply_damage` (one line).
4. A new `advance_destroy_objectives` system reads `ShipDestroyed`, checks the
   `MissionTarget` component on the entity, and completes the matching
   mission. No mission concept appears in weapons/ship/ai code.

### Add faction / loyalty preconditions

1. Add a `Precondition::FactionStandingAtLeast { faction, value }` variant.
2. Extend `preconditions_met` in [progress.rs](progress.rs) to read the
   player's faction-standing resource.
3. The faction system emits `FactionStandingChanged` already (or will); run
   `update_locked_to_available` on that event.

### Add new effects

New `StartEffect` / `CompletionEffect` variants: add the variant, extend the
match in `apply_start_effects` / `finalize_completions`. Examples: `PayCredits`,
`UnlockOutfitter`, `SpawnMarkedShip`, `AdjustFactionStanding`.

## The rule

The rest of the codebase emits events about what happened. The missions
module listens and decides what they mean. If you find yourself reaching for
`Res<MissionLog>` from outside `missions/`, stop and emit an event instead.
