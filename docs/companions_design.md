# Companions — loyal friends & hired escorts (design)

**Status: BUILT (src/companions.rs)** — registry, GrantCompanion + the
permadeath ledger, the six friend arcs, the bar's wingman desk (hire /
dismiss / rejoin), temperaments, rate-limited chatter, and surface avatars
(friends walk with the player when landed). Remaining polish: escort-panel
portraits, mechanic escort repair, mournful-bartender lines.

The escort-persistence substrate is BUILT (src/carrier.rs): the
`EscortRoster` session resource persists escorts across jumps, landings and
saves, with `EscortKind::Companion { name }` reserved for exactly this
feature — a wingman with no carrier bay who flies with the player until they
die. What remains is everything that makes a companion a *person*: how
they're earned or hired, who they are, and how they talk.

The guiding principle stays the codebase's own: state is data, consequences
are derived. A companion is one roster entry plus one registry record;
chatter, combat temperament, and availability all derive from data that
mostly already exists (npcs.yaml, ships.yaml, factions, standing).

---

## 1. The registry: companions.yaml

One file, keyed like everything else:

```yaml
vex_marlowe:
  name: "Vex Marlowe"
  npc: vex_marlowe            # appearance/portrait from assets/npcs.yaml
  ship_type: fed_patrol
  home_planet: kepler_22b     # where they return when dismissed
  temperament: aggressive     # see §4
  bio: "Cashiered Federation ace. Flies angry, drinks angrier."
  chatter:
    kill:        ["Tally one. You're welcome.", "That's how it's done."]
    player_hit:  ["Break left! I've got them."]
    jump_in:     ["Still with you. Regrettably."]
    idle:        ["My old squadron patrols this lane. Wave."]
```

`EscortKind::Companion { name }` becomes `Companion { companion: String }`
(the registry key) when this ships — a save-format note, not a problem:
serde-tag the enum variant fields with defaults so old saves parse.

Validators (test-enforced, the usual pattern): companion npc/ship/planet
exist; every companion is obtainable (granted by some mission OR hireable);
chatter keys are known events.

## 2. Loyal friends — earned, never bought

`CompletionEffect::GrantCompanion { companion }` — applied by the carrier
module watching `MissionCompleted` (the AdjustStanding pattern; missions
stay decoupled). The mission that grants a friend should be the *end of a
short personal arc* (2–3 missions), and the arc should explain why this
person would strap into a cockpit for you.

Friends are permadeath-weighty: if they die, they're gone — no re-earn.
That's what makes flying with them mean something. (The grant effect checks
"was this companion ever enrolled AND is now dead" via a small
`CompanionLedger` set in the roster save-blob.)

### The six friends (v1 cast)

| Name | Ship | Arc hook | Temperament / voice |
|---|---|---|---|
| **Vex Marlowe** | fed_patrol | Freed from a penal transport in the bounty arc — the warrant was political. | *Aggressive.* Dry, insubordinate: "I outrank you. I just don't care." |
| **Okonkwo "Oak" Adaora** | frontier_sailtender | Their homestead survives your hearts-and-minds relief run on a contested world. | *Protective.* Steady, rural: "Right beside you. Like a fencepost." |
| **Sable Dune** | rebel_gunboat | Smuggler you extract in the espionage arc; owes you a ledger she intends to pay in escort-hours. | *Cautious.* Opportunist: "We can outrun this. I vote we outrun this." |
| **Brother Cassian** | order_acolyte | Excommunicated for helping you recover the relic the Order wanted buried. | *Fatalistic-calm.* "The relic hums. Or that's the reactor. Either way." |
| **Tinny** | helios_drone | A combat drone you salvage and rebuild in the mining arc; imprinted on your hull. | *Literal.* ALL-CAPS deadpan: "TARGET REDUCED TO COMPONENTS. COMPONENTS ARE FREE." |
| **Capt. Yara Brakespear** | bastion_lance | Retired lance sergeant; joins after the Bastion arc if standing ≥ +40 — she follows *conduct*, not pay. | *Disciplined.* Gruff honor: "Shields to the front. Always to the front." |

Each friend's arc slots into an existing storyline's tail (bounty,
espionage, frontier, order, mining, bastion) — no new mission machinery,
just 2–3 defs per arc and the grant effect.

## 3. Hired escorts — bought, replaceable, faceless-ish

Offered at the **bar** (where freelance work already lives), as a "Pilots
for hire" section in the bar window — not NPC-walkers, a list:

* Availability derives from the planet: hulls drawn from the local
  shipyard's derived catalog (fighter-personality, tech ≤ planet tech),
  pilots named from per-faction name pools (three archetype flavors:
  Veteran / Rookie / Mercenary — small chatter pools each).
* Price: one-time hire fee ≈ 30% of hull price (rebalanced-economy scale:
  a corvette wingman ≈ 5.4k, serious muscle 20k+). No retainer in v1 —
  they fly until they die or you dismiss them. Death = money gone; hire
  another. That asymmetry (friends irreplaceable / hires replaceable) is
  the emotional spine of the system.
* Standing-gated like everything else: below-neutral standing and the
  bar won't broker (reuses `offers_allowed`).
* Cap: total companions (friends + hires) limited by... nothing physical.
  Soft-cap 3 in v1 (a UI limit) so formations, chatter and escort orders
  stay readable. Revisit after playtesting.

## 4. Personality that costs one enum

`temperament` maps onto knobs the escort AI already has:

* **aggressive** — engage radius ×1.5, chases kills past the leash.
* **protective** — stays close, prioritizes targets attacking the player.
* **cautious** — disengages below 40% hull, returns when repaired
  (companions can't dock, so "returns" = resumes formation).

Chatter is a tiny system: on matching events (kill by companion, player
damaged, system entered, long idle) roll a line from the companion's pool
into the existing comms ticker, rate-limited (≥45s between lines, one
speaker at a time). Friends have bespoke pools; hires get archetype pools.

## 5. Dismissal, death, and the ledger

* **Dismiss** (bar, or escort panel): hires vanish (fee sunk); friends *go
  home* — their roster entry converts to "at home_planet", and you can
  re-recruit them there for free (a chat, not a fee). Friends only truly
  leave via death.
* **Death**: roster entry retires (already built); friends additionally
  write into the `CompanionLedger` (never re-grantable, and their arc's
  bar NPC gets one mournful line if you return).
* Repair: companions arrive with persisted damage (built). The mechanic
  gains a "repair escorts" line item — hull price fraction, same formula
  as the player's.

## 6. Build order

1. **Registry + grant**: companions.yaml, `GrantCompanion` effect,
   ledger, validators + tests. Companion respawn already works.
2. **Hire UI**: bar section, derived hire pool, fee, dismiss. Tests on the
   derivation (faction/tech/standing gates).
3. **Temperament + chatter**: enum → AI knobs; comms-ticker lines with
   rate limiting. Tests: temperament maps to knobs; chatter rate-limits.
4. **Friend arcs**: 6 × (2–3 defs) appended to existing storylines with
   `GrantCompanion` tails; flow-chart regeneration; mission-graph
   validators catch wiring mistakes.
5. **Polish**: escort panel names/portraits, mechanic escort repair,
   mournful-bartender lines.

Phases 1–2 make the system playable; 3–4 make it worth caring about.
