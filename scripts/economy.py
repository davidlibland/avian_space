#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pyyaml",
# ]
# ///
"""Procedural commodity-price economy for avian_space.

The idea (see design discussion): instead of hand-authoring ~200 per-planet
prices, declare what each place *produces* and *consumes*, then derive prices
everywhere by diffusing supply and demand across the jump/trade graph.

Model
-----
Nodes        = planets + asteroid fields (~50).
Edges        = same-system (cheap, complete graph) + jump connections (costly).
Per commodity c, with production vector s_c and consumption vector d_c:

    availability  A_c = (I - alpha*P)^-1 s_c     # supply diffused from sources
    demand        D_c = (I - alpha*P)^-1 d_c     # demand diffused from sinks
    price_c(node) = base_c * (Dn / An) ** beta   # scarce + wanted => dear

where P is the row-normalised trade graph and An/Dn are mean-normalised fields.
alpha is "trade efficiency" (how far goods spread before scarcity bites).

Industries are *endogenous*: a factory's output is throttled by the local
availability of its recipe inputs, so we solve raw materials first and iterate a
few passes up the production chain (ore -> chemicals -> electronics -> ...).

Tuning
------
Three knobs are autotuned to gameplay-meaningful targets (see AUTOTUNER):
  ALPHA       → a good haul is ~ROUTE_LEN jumps (price correlation length).
  BETA        → price/base contrast fills the clamp band (CLAMP_FILL_*).
  PRICE_SCALE → the median merchant-ship rung costs ~JUMPS_PER_SHIP jumps to
                afford (read from ships.yaml). Sets progression pace.

Run:  .venv/bin/python scripts/economy.py            # print price report
      .venv/bin/python scripts/economy.py --tune     # solve ALPHA/BETA/PRICE_SCALE, bake in
      .venv/bin/python scripts/economy.py --tune --write   # tune, then regenerate prices
      .venv/bin/python scripts/economy.py --write    # regenerate prices with current knobs
"""
import argparse
import os
import re
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
STAR_SYSTEMS = os.path.join(HERE, "..", "assets", "star_systems.yaml")

# ── tunables ────────────────────────────────────────────────────────────────
# ALPHA, BETA and PRICE_SCALE are AUTOTUNED by `--tune` (see the AUTOTUNER
# section below); the values here are the last tuned result. W_SYS / W_JUMP /
# PRICE_CLAMP are hand-set and left alone by the tuner.
ALPHA = 0.4500      # trade efficiency: higher = goods spread further, flatter prices
BETA = 0.2531       # price sensitivity to the demand/supply ratio
PRICE_SCALE = 0.1109 # global multiplier on commodity base prices (sets progression pace)
W_SYS = 1.0       # edge weight between nodes in the same system
W_JUMP = 0.30     # edge weight across a jump connection (costlier than in-system)
PRICE_CLAMP = (0.30, 4.0)   # price as a multiple of (scaled) base, clamped to this band
INDUSTRY_PASSES = 6         # fixed-point iterations up the production chain

# ── autotune targets (what `--tune` solves the three knobs to hit) ──────────
# Each knob maps to ONE well-posed, gameplay-meaningful target:
#   ALPHA       → ROUTE_LEN: a good haul should be ~this many jumps. Formally the
#                 price "correlation length" — the jump-distance over which prices
#                 decorrelate. Travelling much farther than this buys little extra
#                 margin, so it sets how far a trader flies for a deal (geography).
#   BETA        → CLAMP_FILL: the CLAMP_FILL_PCTL-th percentile of price/base
#                 should sit near CLAMP_FILL_TARGET, so the clamp band is well used
#                 but not pinned at the 4× rail (cross-commodity contrast).
#   PRICE_SCALE → JUMPS_PER_SHIP: the MEDIAN rung of the merchant-ship ladder (read
#                 from ships.yaml) should take ~this many jumps of trading profit to
#                 afford — i.e. how fast the player progresses (pace).
# ALPHA/BETA shape what trade *feels* like; PRICE_SCALE sets how *fast* you climb
# the ship ladder — keeping the two concerns separate.
ROUTE_LEN = 2.5          # ALPHA target: good haul ≈ 2.5 jumps (sweet spot for this galaxy)
CLAMP_FILL_PCTL = 90     # BETA target: this percentile of price/base ...
CLAMP_FILL_TARGET = 3.0  #   ... should land near here (clamp ceiling is 4×)
JUMPS_PER_SHIP = 50      # PRICE_SCALE target: median merchant rung ≈ 50 jumps
HAUL_MAX = 5             # the simulated trader won't haul farther than this (jumps)
TRADER_PERSONALITY = "Trader"   # fallback: ships of this personality form the ladder
# The "bulk merchant" ladder used for progression pacing — the ships a trader
# buys to haul progressively MORE cargo. Excludes specialty traders like the
# courier (fast, long-range, but small-hold), which would distort the rungs. If
# empty, falls back to every TRADER_PERSONALITY ship. Order is irrelevant
# (sorted by price); only ships that exist in ships.yaml are used.
MERCHANT_LADDER = ["shuttle", "cargo_transport", "freighter", "hauler", "bulk_carrier"]

# ── commodity taxonomy ──────────────────────────────────────────────────────
# cat: "raw" (extracted) | "industry" (made from a recipe) | "special".
# recipe: input units consumed per unit of output.
COMMODITIES = {
    "water":            dict(base=18,  cat="raw"),
    "oxygen":           dict(base=22,  cat="raw"),
    "iron":             dict(base=30,  cat="raw"),
    "silicates":        dict(base=34,  cat="raw"),
    "titanium":         dict(base=110, cat="raw"),
    "gold":             dict(base=190, cat="raw"),
    "uranium":          dict(base=240, cat="raw"),
    "exotic_matter":    dict(base=620, cat="raw"),
    "helium3":          dict(base=200, cat="raw"),
    "food":             dict(base=40,  cat="industry", recipe={"water": 0.4}),
    "chemicals":        dict(base=80,  cat="industry", recipe={"water": 0.3, "silicates": 0.3}),
    "polymers":         dict(base=120, cat="industry", recipe={"chemicals": 0.4, "oxygen": 0.2}),
    "electronics":      dict(base=220, cat="industry", recipe={"silicates": 0.5, "iron": 0.3, "gold": 0.05, "titanium": 0.1}),
    "robotics":         dict(base=420, cat="industry", recipe={"electronics": 0.5, "polymers": 0.2, "titanium": 0.15}),
    "medical_supplies": dict(base=180, cat="industry", recipe={"chemicals": 0.5, "water": 0.3, "polymers": 0.1}),
    "cooling_gel":      dict(base=150, cat="industry", recipe={"chemicals": 0.6, "exotic_matter": 0.03}),
    "fuel_cells":       dict(base=160, cat="industry", recipe={"uranium": 0.2, "chemicals": 0.3, "helium3": 0.2}),
    "weapons_parts":    dict(base=260, cat="industry", recipe={"iron": 0.4, "electronics": 0.3, "titanium": 0.15}),
    "weapons_tech":     dict(base=520, cat="industry", recipe={"weapons_parts": 0.5, "robotics": 0.2, "exotic_matter": 0.08}),
}
# Order industries up the chain so each pass sees fresher inputs.
INDUSTRY_ORDER = ["food", "chemicals", "polymers", "electronics", "robotics",
                  "medical_supplies", "cooling_gel", "fuel_cells",
                  "weapons_parts", "weapons_tech"]

# ── planet-type archetypes ──────────────────────────────────────────────────
# extract: raw production weights. industry: factory output capacities.
# pop: population scale (× radius/30) → consumes life-support + finished goods.
ARCHETYPES = {
    "rocky":     dict(extract={"iron": 1.0, "silicates": 0.7},                       industry={"electronics": 0.25}, pop=0.4),
    "desert":    dict(extract={"iron": 0.8, "silicates": 0.6, "uranium": 0.4},        industry={},                    pop=0.3),
    "habitable": dict(extract={},                                                     industry={"electronics": 1.0, "robotics": 0.5, "medical_supplies": 0.8, "chemicals": 0.6, "weapons_parts": 0.4, "food": 1.2}, pop=2.5),
    "cloud":     dict(extract={"oxygen": 0.6},                                        industry={"chemicals": 1.0, "polymers": 0.7, "cooling_gel": 0.5, "fuel_cells": 0.4}, pop=0.5),
    "gas_giant": dict(extract={"oxygen": 1.0, "helium3": 0.7, "exotic_matter": 0.15}, industry={"fuel_cells": 0.5},   pop=0.1),
    "ice_giant": dict(extract={"water": 1.0, "oxygen": 0.5, "helium3": 0.4},          industry={},                    pop=0.1),
    "icy_dwarf": dict(extract={"water": 1.0},                                         industry={},                    pop=0.15),
}
# What a unit of population consumes per turn.
POP_CONSUME = {"food": 1.0, "water": 0.8, "oxygen": 0.7, "medical_supplies": 0.25, "electronics": 0.15, "robotics": 0.05}


def load_nodes():
    """Return (nodes, systems) where nodes is a list of dicts describing each
    planet / asteroid field and its base production & population."""
    data = yaml.safe_load(open(STAR_SYSTEMS))
    nodes = []
    sys_nodes = {}  # system -> [node indices]
    for sysname, sysd in data.items():
        sys_nodes.setdefault(sysname, [])
        for pname, pd in (sysd.get("planets") or {}).items():
            arch = ARCHETYPES.get(pd.get("planet_type", "rocky"), ARCHETYPES["rocky"])
            radius = float(pd.get("radius", 30.0))
            # Uncolonized worlds (e.g. gas giants) can't be landed on / have no
            # market, but still feed the system via automated extraction. So they
            # stay as supply sources with no population and no factories.
            uncol = bool(pd.get("uncolonized", False))
            nodes.append(dict(
                id=f"{sysname}/{pname}", system=sysname, kind="planet",
                ptype=pd.get("planet_type", "rocky"), uncolonized=uncol,
                extract=dict(arch["extract"]),
                industry={} if uncol else dict(arch.get("industry", {})),
                pop=0.0 if uncol else arch["pop"] * radius / 30.0,
            ))
            sys_nodes[sysname].append(len(nodes) - 1)
        for i, f in enumerate(sysd.get("astroid_fields") or []):
            comm = f.get("commodities") or {"iron": 1.0}
            scale = float(f.get("number", 10)) / 10.0
            nodes.append(dict(
                id=f"{sysname}/field{i}", system=sysname, kind="field", ptype="field",
                uncolonized=False,
                extract={k: v * scale for k, v in comm.items()},
                industry={}, pop=0.0,
            ))
            sys_nodes[sysname].append(len(nodes) - 1)
    return nodes, data, sys_nodes


def build_transition(nodes, data, sys_nodes):
    """Row-normalised transition matrix P for the trade graph."""
    n = len(nodes)
    W = np.zeros((n, n))
    # intra-system: complete graph
    for idxs in sys_nodes.values():
        for a in idxs:
            for b in idxs:
                if a != b:
                    W[a, b] += W_SYS
    # inter-system: connect all nodes across each jump connection
    for sysname, sysd in data.items():
        for other in sysd.get("connections", []) or []:
            if other not in sys_nodes:
                continue
            for a in sys_nodes[sysname]:
                for b in sys_nodes[other]:
                    W[a, b] += W_JUMP
    rowsum = W.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    return W / rowsum


def diffuse(P, source, alpha):
    """Availability field = (I - alpha P)^-1 source (supply/demand diffusion)."""
    n = P.shape[0]
    return np.linalg.solve(np.eye(n) - alpha * P, source)


def solve_economy(nodes, P, alpha=ALPHA):
    """Iterate production up the chain, returning per-commodity production,
    consumption, availability, and demand vectors."""
    n = len(nodes)
    pop = np.array([nd["pop"] for nd in nodes])

    # raw production is fixed (extraction + fields)
    prod = {c: np.zeros(n) for c in COMMODITIES}
    for j, nd in enumerate(nodes):
        for c, w in nd["extract"].items():
            if c in prod:
                prod[c][j] += w

    avail = {c: diffuse(P, prod[c], alpha) for c in COMMODITIES}

    for _ in range(INDUSTRY_PASSES):
        cons = {c: np.zeros(n) for c in COMMODITIES}
        # population demand
        for c, rate in POP_CONSUME.items():
            cons[c] += pop * rate
        # industry: output throttled by local input availability; charges inputs
        for c in INDUSTRY_ORDER:
            recipe = COMMODITIES[c].get("recipe", {})
            out = np.zeros(n)
            for j, nd in enumerate(nodes):
                cap = nd["industry"].get(c, 0.0)
                if cap <= 0:
                    continue
                # limiter in [0,1]: how well the local market can feed the recipe
                limit = 1.0
                for inp, coeff in recipe.items():
                    a = avail[inp][j]
                    ref = avail[inp].mean() or 1.0
                    limit = min(limit, a / (ref + 1e-9))
                out[j] = cap * min(limit, 1.0)
            prod[c] = out
            for inp, coeff in recipe.items():
                cons[inp] += out * coeff
            avail[c] = diffuse(P, out, alpha)  # refresh so downstream sees it
        # refresh raw availability too (cheap)
        for c in COMMODITIES:
            avail[c] = diffuse(P, prod[c], alpha)
        demand = {c: diffuse(P, cons[c], alpha) for c in COMMODITIES}
    return prod, cons, avail, demand


def price_field(avail, demand, beta=BETA, scale=PRICE_SCALE):
    """price[c] (array over nodes), only for commodities that have a source.

    price = scale * base * clip((demand/supply)^beta, lo, hi). The clamp acts on
    the price/base *factor* (so it stays a 0.3–4× band regardless of `scale`);
    `scale` then lifts every price uniformly to set the progression pace.
    """
    prices = {}
    lo, hi = PRICE_CLAMP
    for c, meta in COMMODITIES.items():
        a, d = avail[c], demand[c]
        if a.sum() <= 1e-9:
            continue  # nothing produces it anywhere
        an = a / (a.mean() + 1e-9)
        dn = d / (d.mean() + 1e-9) if d.sum() > 1e-9 else np.ones_like(d)
        ratio = dn / (an + 1e-9)
        clip = np.clip(np.power(ratio, beta), lo, hi)
        prices[c] = scale * meta["base"] * clip
    return prices


def report(nodes, prices):
    cols = [c for c in COMMODITIES if c in prices]
    # Only colonized planets are real markets — restrict extremes to those.
    trad = np.array([j for j, nd in enumerate(nodes)
                     if nd["kind"] == "planet" and not nd["uncolonized"]])

    def lohi(p):
        sub = p[trad]
        return trad[sub.argmin()], trad[sub.argmax()]

    print(f"\n{'commodity':18s}  {'cheapest @':28s} {'dearest @':28s}  spread")
    print("-" * 96)
    for c in cols:
        p = prices[c]
        lo, hi = lohi(p)
        spread = p[hi] / max(p[lo], 1e-9)
        print(f"{c:18s}  {nodes[lo]['id']:20s}{p[lo]:7.0f}  "
              f"{nodes[hi]['id']:20s}{p[hi]:7.0f}  {spread:5.1f}x")

    print("\n=== best single-commodity trade routes (buy → sell) ===")
    routes = []
    for c in cols:
        p = prices[c]
        lo, hi = lohi(p)
        routes.append((p[hi] - p[lo], c, nodes[lo]['id'], p[lo], nodes[hi]['id'], p[hi]))
    for margin, c, src, ps, dst, pd in sorted(routes, reverse=True)[:12]:
        print(f"  {c:16s} {src:22s}{ps:6.0f}  ->  {dst:22s}{pd:6.0f}   +{margin:.0f}/unit")


# Commodities every market stocks, so the player can always buy basics.
STAPLES = {"food", "water"}


def planet_basket(c_order, prices, prod, availn, demandn, j):
    """Which commodities a planet trades: anything it produces, or for which it
    is a notably cheap (locally abundant) buy point or dear (in-demand) sell
    point — plus staples. Abundance/demand are mean-normalised fields."""
    out = {}
    for c in c_order:
        if (prod[c][j] > 1e-3 or availn[c][j] >= 1.2 or demandn[c][j] >= 1.2
                or c in STAPLES):
            out[c] = max(1, int(round(prices[c][j])))
    return out


def write_prices(nodes, prices, prod, avail, demand, path):
    availn = {c: avail[c] / (avail[c].mean() + 1e-9) for c in avail}
    demandn = {c: demand[c] / (demand[c].mean() + 1e-9) for c in demand}
    """Surgically replace each planet's `commodities:` block in-place, leaving
    all comments / structure / asteroid-field drop tables untouched."""
    # Only colonized planets get a market; uncolonized worlds must stay empty
    # (asset validation requires it) but still act as sources in the model.
    idx = {nd["id"]: j for j, nd in enumerate(nodes)
           if nd["kind"] == "planet" and not nd["uncolonized"]}
    c_order = sorted((c for c in COMMODITIES if c in prices),
                     key=lambda c: COMMODITIES[c]["base"])
    lines = open(path).read().split("\n")
    out, cur_sys, cur_planet = [], None, None
    seen, missing = set(), []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if re.match(r"^[a-z_][a-z0-9_]*:\s*$", ln):              # system (col 0)
            cur_sys, cur_planet = ln[:-1], None
        elif re.match(r"^  [a-z_][a-z0-9_]*:", ln):              # 2-space key → leaves planet body
            cur_planet = None
        elif (m := re.match(r"^    ([a-z_][a-z0-9_]*):\s*$", ln)):  # planet name (4-space)
            cur_planet = m.group(1)
        elif re.match(r"^      commodities:\s*$", ln) and cur_sys and cur_planet:
            nid = f"{cur_sys}/{cur_planet}"
            if nid in idx:
                seen.add(nid)
                out.append(ln)
                for c, price in planet_basket(c_order, prices, prod, availn, demandn, idx[nid]).items():
                    out.append(f"        {c}: {price}")
                i += 1
                while i < len(lines) and re.match(r"^        \S", lines[i]):
                    i += 1
                continue
        out.append(ln)
        i += 1
    for nid in idx:
        if nid not in seen:
            missing.append(nid)
    open(path, "w").write("\n".join(out))
    print(f"\nwrote prices for {len(seen)}/{len(idx)} planets → {os.path.relpath(path)}")
    if missing:
        print(f"  ⚠ no commodities: block found for: {', '.join(missing)}")


# ════════════════════════════════════════════════════════════════════════════
#  AUTOTUNER  (`--tune`)
#
#  Solves the three knobs to their targets (see the tunables section):
#    ALPHA       → ROUTE_LEN       (price correlation length, in jumps)
#    BETA        → CLAMP_FILL      (price/base percentile fills the clamp band)
#    PRICE_SCALE → JUMPS_PER_SHIP  (median merchant rung costs ~N jumps)
#
#  ALPHA and BETA both shape the price field, so they're solved by alternating
#  monotonic bisection to convergence. PRICE_SCALE is linear in price, so once
#  ALPHA/BETA are fixed it closes form from the simulated trader profit rate.
# ════════════════════════════════════════════════════════════════════════════


def system_jump_distances(data):
    """BFS on the system graph → {(sysA, sysB): jumps}. Same system = 0 jumps."""
    dist = {}
    for s in data:
        seen, frontier = {s: 0}, [s]
        while frontier:
            nxt = []
            for u in frontier:
                for v in (data[u].get("connections") or []):
                    if v in data and v not in seen:
                        seen[v] = seen[u] + 1
                        nxt.append(v)
            frontier = nxt
        for t, d in seen.items():
            dist[(s, t)] = d
    return dist


def system_markets(nodes, prices, prod, avail, demand):
    """Per system, the best buy (min) / sell (max) price of each commodity across
    its colonized planets — but only commodities a planet actually *stocks*
    (its market basket); you can't trade what isn't on the shelf."""
    availn = {c: avail[c] / (avail[c].mean() + 1e-9) for c in avail}
    demandn = {c: demand[c] / (demand[c].mean() + 1e-9) for c in demand}
    c_order = [c for c in COMMODITIES if c in prices]
    buy, sell = {}, {}
    for j, nd in enumerate(nodes):
        if nd["kind"] != "planet" or nd["uncolonized"]:
            continue
        s = nd["system"]
        for c, p in planet_basket(c_order, prices, prod, availn, demandn, j).items():
            buy.setdefault(s, {})
            sell.setdefault(s, {})
            buy[s][c] = min(buy[s].get(c, p), p)
            sell[s][c] = max(sell[s].get(c, p), p)
    return buy, sell


def trader_profit_per_jump(buy, sell, jd, haul_max=HAUL_MAX):
    """Greedy system-to-system trader: from the current system, take the
    (destination ≤ haul_max jumps, commodity) with the best profit-per-jump
    (margin ÷ jump-distance), bank it, repeat — averaged over all start systems.
    Returns (credits per cargo-unit per jump, mean haul length in jumps)."""
    systems = [s for s in buy]
    if not systems:
        return 0.0, 0.0
    total_profit = total_jumps = 0.0
    hauls = []
    for start in systems:
        s = start
        for _ in range(2 * len(systems)):
            best = None  # (yield, margin, d, dest)
            for t in systems:
                d = jd.get((s, t), 99)
                if not 1 <= d <= haul_max:
                    continue
                for c, sp in sell[t].items():
                    bp = buy[s].get(c)
                    if bp is None or sp <= bp:
                        continue
                    y = (sp - bp) / d
                    if best is None or y > best[0]:
                        best = (y, sp - bp, d, t)
            if best is None:
                break
            _, margin, d, t = best
            total_profit += margin
            total_jumps += d
            hauls.append(d)
            s = t
    if total_jumps == 0:
        return 0.0, 0.0
    return total_profit / total_jumps, float(np.mean(hauls))


def correlation_length(nodes, prices, jd):
    """Price 'correlation length' in jumps: the jump-distance at which the
    price/base field decorrelates (variogram reaches 1−1/e of its sill),
    variance-weighted across commodities. Larger = prices only differ over long
    distances (high ALPHA); smaller = sharp local gradients (low ALPHA)."""
    pl = [(j, nd["system"]) for j, nd in enumerate(nodes)
          if nd["kind"] == "planet" and not nd["uncolonized"]]
    maxd = max(jd.values()) if jd else 1
    lengths = []
    for c in prices:
        base = COMMODITIES[c]["base"]
        r = np.array([prices[c][j] / base for j, _ in pl])
        if r.std() < 1e-6:
            continue
        gamma = np.zeros(maxd + 1)
        cnt = np.zeros(maxd + 1)
        for a in range(len(pl)):
            for b in range(a + 1, len(pl)):
                d = jd.get((pl[a][1], pl[b][1]))
                if d is None:
                    continue
                gamma[d] += (r[a] - r[b]) ** 2
                cnt[d] += 1
        thresh = (1 - 1 / np.e) * 2 * r.var()   # variogram sill = 2·Var
        L = maxd
        for d in range(1, maxd + 1):
            if cnt[d] > 0 and gamma[d] / cnt[d] >= thresh:
                L = d
                break
        lengths.append((L, r.var()))
    if not lengths:
        return 0.0
    Ls = np.array([L for L, _ in lengths], float)
    w = np.array([v for _, v in lengths])
    return float((Ls * w).sum() / w.sum())


def bisect(f, lo, hi, target, iters=22, tol=1e-3):
    """Root-find x in [lo, hi] with f(x) ≈ target, assuming f is monotone ↑."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm - target) < tol:
            return mid
        lo, hi = (mid, hi) if fm < target else (lo, mid)
    return 0.5 * (lo + hi)


def merchant_line(ships_path):
    """The bulk-merchant ladder from ships.yaml, sorted by price: [(name, price,
    cargo)]. Uses MERCHANT_LADDER if set (excludes specialty traders like the
    courier), else every TRADER_PERSONALITY ship."""
    s = yaml.safe_load(open(ships_path))
    names = ([n for n in MERCHANT_LADDER if n in s] if MERCHANT_LADDER
             else [n for n, d in s.items() if d.get("personality") == TRADER_PERSONALITY])
    line = [(n, s[n]["price"], s[n]["cargo_space"]) for n in names]
    line.sort(key=lambda t: t[1])
    return line


def autotune(nodes, P, jd, line):
    """Solve (ALPHA, BETA, PRICE_SCALE) for their targets. ALPHA/BETA are coupled
    (both reshape the field) so we alternate their bisections; PRICE_SCALE is
    linear in price so it closes form last. Returns (alpha, beta, scale, diag)."""

    def field(alpha, beta):
        prod, _, avail, demand = solve_economy(nodes, P, alpha)
        return price_field(avail, demand, beta, scale=1.0), prod, avail, demand

    def clamp_pctl(prices):
        allr = np.concatenate([prices[c] / COMMODITIES[c]["base"] for c in prices])
        return float(np.percentile(allr, CLAMP_FILL_PCTL))

    def trader_haul(alpha, beta):
        prices, prod, av, dm = field(alpha, beta)
        buy, sell = system_markets(nodes, prices, prod, av, dm)
        return trader_profit_per_jump(buy, sell, jd)[1]

    alpha, beta = ALPHA, BETA
    for _ in range(3):  # alternate α and β to convergence (both reshape the field)
        # ALPHA → the optimal trader's MEAN HAUL ≈ ROUTE_LEN jumps. The haul vs α
        # curve is a noisy step function (the best route switches discretely), so
        # we GRID-SCAN α and take the value whose haul is closest to the target,
        # rather than bisecting (which a non-monotone curve would break).
        grid = np.linspace(0.45, 0.975, 12)
        hauls = [trader_haul(a, beta) for a in grid]
        alpha = float(grid[int(np.argmin(np.abs(np.array(hauls) - ROUTE_LEN)))])
        # BETA → clamp-fill is smooth & monotone in β, so bisect it.
        beta = bisect(lambda b: clamp_pctl(field(alpha, b)[0]),
                      0.15, 1.80, CLAMP_FILL_TARGET, tol=0.02)

    # Profit rate of the tuned field (credits per cargo-unit per jump, at scale=1).
    prices, prod, avail, demand = field(alpha, beta)
    buy, sell = system_markets(nodes, prices, prod, avail, demand)
    m1, mean_haul = trader_profit_per_jump(buy, sell, jd)

    # EVEN RUNGS: a single PRICE_SCALE can only fix the *median* rung, so to make
    # every merchant rung cost ~JUMPS_PER_SHIP jumps we also set ship prices.
    # Anchor the cheapest + priciest merchant ships (keep the player's price range)
    # and distribute the middle so each price gap ∝ the cargo you haul to earn it —
    # which makes jumps_i = gap_i/(cargo_i·m) equal across rungs.
    names = [n for n, _, _ in line]
    p_lo, p_hi = line[0][1], line[-1][1]
    cargos = [c for _, _, c in line[:-1]]      # cargo of the ship flown on each rung
    span, tot = p_hi - p_lo, sum(cargos)
    new_prices, acc = {names[0]: p_lo}, p_lo
    for i, c in enumerate(cargos):
        acc += span * c / tot
        new_prices[names[i + 1]] = int(round(acc))
    new_prices[names[-1]] = p_hi
    # each rung then needs span/tot credits of profit; at N jumps, m = (span/tot)/N.
    scale = ((span / tot) / JUMPS_PER_SHIP) / m1 if m1 > 0 else 1.0

    diag = dict(alpha=alpha, beta=beta, scale=scale, m1=m1, mean_haul=mean_haul,
                corr_len=correlation_length(nodes, prices, jd),
                clamp_pctl=clamp_pctl(prices), line=line, new_prices=new_prices)
    return alpha, beta, scale, diag


def autotune_report(diag):
    print("\n══════════════════════ AUTOTUNE ══════════════════════")
    print(f"  ALPHA       = {diag['alpha']:.4f}   trader mean haul {diag['mean_haul']:.2f} "
          f"≈ target {ROUTE_LEN:.1f} jumps  (price correlation length {diag['corr_len']:.1f})")
    print(f"  BETA        = {diag['beta']:.4f}   {CLAMP_FILL_PCTL}th pct price/base "
          f"{diag['clamp_pctl']:.2f}× ≈ target {CLAMP_FILL_TARGET:.1f}×")
    print(f"  PRICE_SCALE = {diag['scale']:.4f}   profit ≈ "
          f"{diag['m1'] * diag['scale']:.1f} cr / cargo-unit / jump")
    m = diag["m1"] * diag["scale"]
    line, np_ = diag["line"], diag["new_prices"]
    print(f"\n  merchant progression — proposed even ship prices ({JUMPS_PER_SHIP} jumps/rung,"
          f" endpoints kept):")
    print(f"    {'ship':13s} {'old price':>10} {'new price':>10}   rung → jumps")
    for i, (n, p, c) in enumerate(line):
        newp = np_[n]
        if i < len(line) - 1:
            gap = np_[line[i + 1][0]] - newp
            jumps = gap / (c * m) if m > 0 else float("inf")
            tail = f"   gap {gap:7d} → {jumps:4.0f} jumps"
        else:
            tail = ""
        flag = "" if newp == p else "  *changed"
        print(f"    {n:13s} {p:10d} {newp:10d}{tail}{flag}")


def write_ship_prices(new_prices, path):
    """Surgically update `  price:` for the named merchant ships in ships.yaml."""
    lines = open(path).read().split("\n")
    out, cur = [], None
    changed = 0
    for ln in lines:
        m = re.match(r"^([a-z_][a-z0-9_]*):\s*$", ln)
        if m:
            cur = m.group(1)
        elif cur in new_prices and re.match(r"^  price:\s*-?\d+", ln):
            ln = f"  price: {new_prices[cur]}"
            changed += 1
        out.append(ln)
    open(path, "w").write("\n".join(out))
    print(f"  ↳ updated {changed} merchant ship prices in {os.path.relpath(path)}")


def apply_constants(alpha, beta, scale, path):
    """Bake the tuned ALPHA / BETA / PRICE_SCALE back into this script's constants."""
    src = open(path).read()
    for name, val in [("ALPHA", alpha), ("BETA", beta), ("PRICE_SCALE", scale)]:
        src = re.sub(rf"(?m)^{name} = [0-9.]+", f"{name} = {val:.4f}", src, count=1)
    open(path, "w").write(src)
    print(f"\n  ↳ baked tuned values into {os.path.relpath(path)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="rewrite each planet's commodities: block in star_systems.yaml")
    ap.add_argument("--tune", action="store_true",
                    help="autotune ALPHA/BETA/PRICE_SCALE to their targets and bake them in")
    args = ap.parse_args()
    nodes, data, sys_nodes = load_nodes()
    P = build_transition(nodes, data, sys_nodes)
    jd = system_jump_distances(data)
    global ALPHA, BETA, PRICE_SCALE
    if args.tune:
        line = merchant_line(os.path.join(HERE, "..", "assets", "ships.yaml"))
        ALPHA, BETA, PRICE_SCALE, diag = autotune(nodes, P, jd, line)
        autotune_report(diag)
        apply_constants(ALPHA, BETA, PRICE_SCALE, os.path.abspath(__file__))
    prod, cons, avail, demand = solve_economy(nodes, P, ALPHA)
    prices = price_field(avail, demand, BETA, PRICE_SCALE)
    print(f"\nloaded {len(nodes)} nodes "
          f"({sum(nd['kind']=='planet' for nd in nodes)} planets, "
          f"{sum(nd['kind']=='field' for nd in nodes)} fields)  "
          f"alpha={ALPHA:.3f} beta={BETA:.3f} scale={PRICE_SCALE:.3f}")
    report(nodes, prices)
    if args.write:
        write_prices(nodes, prices, prod, avail, demand, os.path.abspath(STAR_SYSTEMS))
        if args.tune:
            write_ship_prices(diag["new_prices"],
                              os.path.join(HERE, "..", "assets", "ships.yaml"))


if __name__ == "__main__":
    main()
