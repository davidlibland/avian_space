# Fuel rescue — design

Closes the "stranded with a dry tank in a planetless system" hole. Two
independent safety nets plus a validator so the hole can't reopen.

## 1. Fuel as a scoopable commodity
- New `fuel` commodity (`assets/commodities.yaml`, cyan). Never listed on
  planets — it fills the jump tank, not the hold.
- `Ship::receive_pickup(commodity, qty)` (src/ship.rs) is the single
  routing point: `fuel` → tank (capped at `fuel_capacity`), everything
  else → cargo. `collect_pickups` calls it instead of poking `ship.cargo`
  directly, so scooping fuel refuels automatically.

## 2. Fuel asteroids (per-rock, not per-field)
The old model drops pickups by *field* commodity weight, so "fuel
asteroid" wasn't a real thing. Now it is:
- `Asteroid.fuel: bool`, rolled at spawn from the field's `fuel` weight.
- Fuel is REMOVED from the generic shatter roll — an iron rock never
  surprise-drops fuel; only a fuel rock does, and always.
- Fuel rocks carry `FuelShimmer`; so do the fuel pickups they drop. One
  marker, one shimmer system (cyan sparks via `explosions::spawn_spark`),
  so rock and pickup share the exact effect — the player learns "shoot
  the shimmering rock."

## 3. Distress call (the floor)
For the pacifist / weaponless / empty-field case:
- When the player is Flying with `fuel == 0`, a comms line + a small egui
  button offer a distress call.
- Accepting spawns a **Relief Tanker** (Merchant hull, distinct name) at
  a near jump-in radius with the hyperspace flash. `RescueTanker` state
  machine: **Approach** (steer to player) → **Service** (brief pause,
  sparks bridge the hulls, tank filled to full) → **Depart**
  (`JumpingOut` + flash).
- Billed at ~4× the fuel-station rate for a full tank; if the player
  can't pay it goes to debt (negative credits) — consequence without a
  dead end. One rescue at a time.

## Validator
`check_refuel_coverage`: every non-training system must have a landable
planet OR a fuel-bearing asteroid field, else a dry tank strands the
player forever. Test-enforced; six planetless systems gained `fuel`
weights on a field.

## Reuse / cleanliness
- `Ship::receive_pickup` unifies the two "add a scooped thing" paths.
- `Ship::fuel_price_per_unit` shared by the fuel station and the tanker.
- `explosions::spawn_spark` factored out of `trigger_explosions`, shared
  by the fuel shimmer.
- Tanker nav is bespoke (steer-to-point), NOT the Escort/formation code,
  which is roster-coupled and would be dirtier to bend here.

## Status
Shipped. The `relief_tanker` reuses the hauler sprite + wireframe and is
exempt from the "sold nowhere" validator (INTERNAL_SHIPS). The help
screen documents fuel, scooping, and the distress call.

## Deferred
Bespoke 3D-baked tanker hull (using the Merchant hauler sprite for now
— a Blender bake can't be visually QA'd headless). Would slot in by
baking `sprites/ships/atlas/relief_tanker.png` + a wireframe and
pointing the ships.yaml entry at them.
