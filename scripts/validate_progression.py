#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml"]
# ///
"""
validate_progression.py — sanity-check the ship / weapon / licence progression.

Primary invariant (requested):
  If a player can BUY a ship, they can also BUY every weapon that ship spawns
  with, from the outfitter. Concretely, for each ship S with required_unlocks
  U_S and each built-in weapon W, the outfitter must sell W and the unlocks it
  needs (U_W) must be reachable by the time S is buyable — i.e. U_W ⊆ U_S, OR
  every unlock in U_W is granted no later than the unlocks in U_S along the
  mission graph.

Also reports, as information:
  * built-in weapons that are NEVER sold at the outfitter,
  * every licence/unlock: which mission(s) grant it, which ships/weapons it gates,
  * licences that are never granted by any mission.

Run:  uv run scripts/validate_progression.py
Exit: non-zero if any hard violation of the buy-the-ship-buy-its-weapon invariant.
"""
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "assets")


# ── YAML loader that tolerates the custom !PrimaryWeapon / !Ammo / ... tags ──
class Loader(yaml.SafeLoader):
    pass


def _tagged(loader, tag_suffix, node):
    if isinstance(node, yaml.MappingNode):
        val = loader.construct_mapping(node, deep=True)
    elif isinstance(node, yaml.SequenceNode):
        val = loader.construct_sequence(node, deep=True)
    else:
        val = loader.construct_scalar(node)
    if isinstance(val, dict):
        val["__tag__"] = tag_suffix
    return val


Loader.add_multi_constructor("!", _tagged)


def load(name):
    with open(os.path.join(A, name)) as f:
        return yaml.load(f, Loader=Loader) or {}


def main():
    ships = load("ships.yaml")
    outfitter = load("outfitter_items.yaml")
    missions = load("missions.yaml")

    # weapon_type -> required_unlocks (what the outfitter needs to sell it)
    sold = {}          # weapon_type -> set(unlocks)
    for _item, spec in outfitter.items():
        if not isinstance(spec, dict):
            continue
        wt = spec.get("weapon_type")
        if wt is None:
            continue
        sold[wt] = set(spec.get("required_unlocks", []) or [])

    # ship -> (unlocks, built-in weapons)
    ship_unlocks = {}
    ship_weapons = {}
    for sid, spec in ships.items():
        if not isinstance(spec, dict):
            continue
        ship_unlocks[sid] = set(spec.get("required_unlocks", []) or [])
        ship_weapons[sid] = list((spec.get("base_weapons") or {}).keys())

    # mission -> unlocks it grants ; mission -> prerequisite missions
    grants = {}
    prereqs = {}
    granted_by = {}
    for mid, spec in missions.items():
        if not isinstance(spec, dict):
            continue
        g = set()
        for key in ("completion_effects", "start_effects"):
            for eff in spec.get(key, []) or []:
                if isinstance(eff, dict) and eff.get("kind") == "grant_unlock":
                    g.add(eff["name"])
                    granted_by.setdefault(eff["name"], []).append(
                        mid + (" (start)" if key == "start_effects" else ""))
        grants[mid] = g
        prereqs[mid] = [p["mission"] for p in (spec.get("preconditions", []) or [])
                        if isinstance(p, dict) and p.get("kind") == "completed"
                        and p.get("mission") in missions]

    # held_after[M] = unlocks guaranteed to be owned once M is completed
    # (fixpoint over the precondition graph).
    held_after = {m: set(grants[m]) for m in missions if isinstance(missions[m], dict)}
    for _ in range(len(held_after) + 2):
        changed = False
        for m in held_after:
            new = set(grants[m])
            for p in prereqs[m]:
                new |= held_after.get(p, set())
            if new != held_after[m]:
                held_after[m] = new
                changed = True
        if not changed:
            break

    # reach(u) = unlocks guaranteed already owned whenever u is owned =
    # intersection over every mission that grants u of held_after[mission].
    def reach(u):
        gm = [m for m in missions if isinstance(missions[m], dict) and u in grants[m]]
        if not gm:
            return None  # never granted
        acc = None
        for m in gm:
            acc = held_after[m] if acc is None else (acc & held_after[m])
        return acc

    reach_cache = {}

    def held(unlock_set):
        """Unlocks guaranteed owned once every unlock in unlock_set is owned."""
        out = set(unlock_set)
        for u in unlock_set:
            if u not in reach_cache:
                reach_cache[u] = reach(u)
            r = reach_cache[u]
            if r:
                out |= r
        return out

    all_unlocks = set()
    for u in ship_unlocks.values():
        all_unlocks |= u
    for u in sold.values():
        all_unlocks |= u

    # ── HARD CHECK: buy-the-ship ⇒ buy-its-weapons (mission-graph aware) ──
    violations = []      # (ship, weapon, reason)
    for sid, weps in ship_weapons.items():
        have = held(ship_unlocks[sid])
        for w in weps:
            if w not in sold:
                violations.append((sid, w, "weapon is NOT sold at the outfitter"))
                continue
            missing = sold[w] - have
            if missing:
                violations.append(
                    (sid, w,
                     f"outfitter needs {sorted(sold[w])}; owning {sid} only "
                     f"guarantees {sorted(have)} (missing {sorted(missing)})"))

    # ── REPORT ──
    print("=" * 78)
    print("LICENCES: granted-by → gates")
    print("=" * 78)
    for lic in sorted(all_unlocks):
        g = granted_by.get(lic, [])
        gships = sorted(s for s, u in ship_unlocks.items() if lic in u)
        gweps = sorted(w for w, u in sold.items() if lic in u)
        flag = "" if g else "   ⚠ NEVER GRANTED"
        print(f"\n{lic}{flag}")
        print(f"  granted by : {g or '—'}")
        print(f"  gates ships: {gships or '—'}")
        print(f"  gates weaps: {gweps or '—'}")

    print("\n" + "=" * 78)
    print("BUILT-IN WEAPONS NEVER SOLD AT THE OUTFITTER")
    print("=" * 78)
    never = sorted({w for weps in ship_weapons.values() for w in weps} - set(sold))
    for w in never:
        owners = sorted(s for s, weps in ship_weapons.items() if w in weps)
        print(f"  {w:24} (built into: {', '.join(owners)})")
    if not never:
        print("  none")

    print("\n" + "=" * 78)
    print(f"INVARIANT VIOLATIONS (buy ship ⇒ buy its weapons): {len(violations)}")
    print("=" * 78)
    for sid, w, reason in violations:
        print(f"  [{sid}] weapon '{w}': {reason}")
    if not violations:
        print("  none — every buyable ship's built-in weapons are buyable too ✓")

    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
