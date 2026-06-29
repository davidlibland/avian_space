# Faction progression redesign (2026-06)

Splits the four "neutral faction" tech trees (Helios, Free Frontier, Bastion,
Order) and the Rebel arc so that **combat ships gate behind individual combat
missions** instead of everything unlocking from the first mission. Also fixes the
dead `rebel_carrier_license` and the "buy-the-ship-but-not-its-weapon" gaps.

## Principles
- **Mission 1** of each faction grants the faction's *base* licence, which now
  unlocks only **merchant/miner ships** (Trader/Miner personalities) plus the
  faction's **basic weapon** (the one those ships ship with).
- Each **combat ship** gets its **own licence**, granted by its own mission,
  unlocked **smallest → largest** by price.
- The **first combat ship to introduce a new weapon** unlocks that weapon at the
  outfitter under the **same licence** (so owning the ship ⇒ buying the weapon).
- Validated by `scripts/validate_progression.py` (mission-graph aware).

## Decisions (user-approved)
- Combat ladder: **each combat ship = its own licence + mission**. Helios/Frontier/
  Bastion get a new **5th** mission (they had only 3 post-intro); Order already had
  4 post-intro missions so it needs none.
- `rebel_carrier_license`: **add a capstone rebel mission** (east_wind_7) that grants it.
- Reused (non-combat) missions keep their existing objective; only the new 5th
  missions are authored as combat. The rebel javelin mission is combat.

## Licence map (new)

### Helios — missions helios_1..5 (add helios_5)
| licence | mission | ships | weapons |
|---|---|---|---|
| helios_license (base) | helios_1 | helios_courier, helios_freighter, helios_hauler | helios_lance |
| helios_drone_license | helios_2 | helios_drone | — |
| helios_enforcer_license | helios_3 | helios_enforcer | seeker |
| helios_overseer_license | helios_4 | helios_overseer | drone_bay |
| helios_titan_license | helios_5 (new) | helios_titan | — |

### Free Frontier — missions frontier_1..5 (add frontier_5)
| licence | mission | ships | weapons |
|---|---|---|---|
| free_frontier_license (base) | frontier_1 | frontier_prospector, frontier_dredger | flak_gun |
| frontier_skiff_license | frontier_2 | frontier_skiff | — |
| frontier_sailtender_license | frontier_3 | frontier_sailtender | prox_mine |
| frontier_harvester_license | frontier_4 | frontier_harvester | mass_driver |
| frontier_monitor_license | frontier_5 (new) | frontier_monitor | flak_turret |

### Bastion — missions bastion_1..5 (add bastion_5)
| licence | mission | ships | weapons |
|---|---|---|---|
| bastion_license (base) | bastion_1 | bastion_dredger, bastion_collier | chaingun |
| bastion_guard_license | bastion_2 | bastion_guard | — |
| bastion_lance_license | bastion_3 | bastion_lance | chaingun_turret |
| siege_monitor_license | bastion_4 | siege_monitor | siege_cannon |
| iron_dreadnought_license | bastion_5 (new) | iron_dreadnought | — |

### Order — missions order_1..5 (no new mission)
| licence | mission | ships | weapons |
|---|---|---|---|
| artifact_order_license (base) | order_1 | order_quarryman, order_almoner | relic_lance |
| order_acolyte_license | order_2 | order_acolyte | — |
| order_censer_license | order_3 | order_censer | censer_charge |
| order_reliquary_license | order_4 | order_reliquary | — |
| order_cathedral_license (+precursor_tech_license) | order_5 | order_cathedral | relic_turret |

### Rebel — east_wind arc (insert javelin mission, renumber, add carrier capstone)
Final order: 1 meet → 2 deliver → 3 destroy (rebel_ship_license) →
**4 NEW destroy (rebel_weapons_license / javelin)** → 5 (was 4) meet
(rebel_capital_license) → 6 (was 5) destroy (federation_ship_license) →
**7 NEW capstone destroy (rebel_carrier_license + turret_license)**.
- rebel_frigate (rebel_capital_license) carries `javelin` → now granted one step
  earlier, at east_wind_4.
- rebel_carrier (rebel_carrier_license) carries `laser_turret` → east_wind_7 also
  grants `turret_license` so the turret is buyable.

## Invariant fixes beyond the four factions (buy-ship ⇒ buy-its-weapons)
The "every weapon a buyable ship carries is buyable too" rule was already broken
for pre-existing Federation/pirate carriers. Minimal fixes:
- `federation_ship_license` missions (weight_of_order_2, east_wind_6) now also grant
  `proton_beam_license` + `federation_weapons_license` — a Federation ship licence
  comes with the basic Federation gun and IR missiles (fed_patrol / fed_destroyer /
  fed_missile_cruiser carry them).
- Every carrier-unlocking mission now grants `turret_license` (carriers ship with
  turrets): weight_of_order_5 (fed_carrier), no_flag_5 / invisible_hand_5 / rift_final
  (carrier_license), east_wind_7 (rebel_carrier).
- `pirate_carrier` had **no** `required_unlocks` (buyable by anyone) → now gated by
  `carrier_license` like `surplus_carrier`.
- `fed_missile_cruiser` carried a `fighter_bay` (needs carrier_license) on an
  otherwise federation_ship_license hull — anomalous for a *missile* cruiser and
  unsatisfiable without coupling carrier tech into the basic Fed line, so the
  fighter_bay was removed (it keeps proton_beam + ir_missile).

## Validation
- `scripts/validate_progression.py` — standalone, mission-graph-aware report (0 violations).
- `src/validate_assets.rs::check_ship_weapons_buyable` — the same invariant enforced at
  game launch and in CI via `test_asset_validators_pass` (alongside the existing
  unlock-obtainability + mission-DAG checks). Both pass.

## Files touched
- assets/ships.yaml — per-ship `required_unlocks` (+ pirate_carrier gate, fed_missile_cruiser bay).
- assets/outfitter_items.yaml — per-weapon `required_unlocks`.
- assets/missions.yaml — grants, preconditions, briefing/success text, 5 new missions, cross-arc co-grants.
- scripts/validate_progression.py — mission-graph reachability check (new).
- src/validate_assets.rs — launch-time buy-ship⇒buy-weapon check (new).
