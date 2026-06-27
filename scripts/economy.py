#!/usr/bin/env python3
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

Run:  .venv/bin/python scripts/economy.py            # print price report
      .venv/bin/python scripts/economy.py --write    # (todo) write back to yaml
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
ALPHA = 0.82      # trade efficiency: higher = goods spread further, flatter prices
BETA = 0.55       # price sensitivity to the demand/supply ratio
W_SYS = 1.0       # edge weight between nodes in the same system
W_JUMP = 0.30     # edge weight across a jump connection (costlier than in-system)
PRICE_CLAMP = (0.30, 4.0)   # price as a multiple of base, clamped to this band
INDUSTRY_PASSES = 6         # fixed-point iterations up the production chain

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
    "food":             dict(base=40,  cat="industry", recipe={"water": 0.4}),
    "chemicals":        dict(base=80,  cat="industry", recipe={"water": 0.3, "silicates": 0.3}),
    "electronics":      dict(base=220, cat="industry", recipe={"silicates": 0.5, "iron": 0.3, "gold": 0.05, "titanium": 0.1}),
    "medical_supplies": dict(base=180, cat="industry", recipe={"chemicals": 0.5, "water": 0.3}),
    "cooling_gel":      dict(base=150, cat="industry", recipe={"chemicals": 0.6, "exotic_matter": 0.03}),
    "fuel_cells":       dict(base=160, cat="industry", recipe={"uranium": 0.2, "chemicals": 0.3}),
    "weapons_parts":    dict(base=260, cat="industry", recipe={"iron": 0.4, "electronics": 0.3, "titanium": 0.15}),
    "weapons_tech":     dict(base=520, cat="industry", recipe={"weapons_parts": 0.5, "electronics": 0.3, "exotic_matter": 0.08}),
}
# Order industries up the chain so each pass sees fresher inputs.
INDUSTRY_ORDER = ["food", "chemicals", "electronics", "medical_supplies",
                  "cooling_gel", "fuel_cells", "weapons_parts", "weapons_tech"]

# ── planet-type archetypes ──────────────────────────────────────────────────
# extract: raw production weights. industry: factory output capacities.
# pop: population scale (× radius/30) → consumes life-support + finished goods.
ARCHETYPES = {
    "rocky":     dict(extract={"iron": 1.0, "silicates": 0.7},                       industry={"electronics": 0.25}, pop=0.4),
    "desert":    dict(extract={"iron": 0.8, "silicates": 0.6, "uranium": 0.4},        industry={},                    pop=0.3),
    "habitable": dict(extract={},                                                     industry={"electronics": 1.0, "medical_supplies": 0.8, "chemicals": 0.6, "weapons_parts": 0.4, "food": 1.2}, pop=2.5),
    "cloud":     dict(extract={"oxygen": 0.6},                                        industry={"chemicals": 1.0, "cooling_gel": 0.5, "fuel_cells": 0.4}, pop=0.5),
    "gas_giant": dict(extract={"oxygen": 1.0, "exotic_matter": 0.15},                 industry={"fuel_cells": 0.5},   pop=0.1),
    "ice_giant": dict(extract={"water": 1.0, "oxygen": 0.5},                          industry={},                    pop=0.1),
    "icy_dwarf": dict(extract={"water": 1.0},                                         industry={},                    pop=0.15),
}
# What a unit of population consumes per turn.
POP_CONSUME = {"food": 1.0, "water": 0.8, "oxygen": 0.7, "medical_supplies": 0.25, "electronics": 0.15}


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


def diffuse(P, source):
    """Availability field = (I - alpha P)^-1 source (supply/demand diffusion)."""
    n = P.shape[0]
    return np.linalg.solve(np.eye(n) - ALPHA * P, source)


def solve_economy(nodes, P):
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

    avail = {c: diffuse(P, prod[c]) for c in COMMODITIES}

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
            avail[c] = diffuse(P, out)  # refresh so downstream sees it
        # refresh raw availability too (cheap)
        for c in COMMODITIES:
            avail[c] = diffuse(P, prod[c])
        demand = {c: diffuse(P, cons[c]) for c in COMMODITIES}
    return prod, cons, avail, demand


def price_field(avail, demand):
    """price[c] (array over nodes), only for commodities that have a source."""
    prices = {}
    for c, meta in COMMODITIES.items():
        a, d = avail[c], demand[c]
        if a.sum() <= 1e-9:
            continue  # nothing produces it anywhere
        an = a / (a.mean() + 1e-9)
        dn = d / (d.mean() + 1e-9) if d.sum() > 1e-9 else np.ones_like(d)
        ratio = dn / (an + 1e-9)
        p = meta["base"] * np.power(ratio, BETA)
        lo, hi = PRICE_CLAMP
        prices[c] = np.clip(p, meta["base"] * lo, meta["base"] * hi)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="rewrite each planet's commodities: block in star_systems.yaml")
    args = ap.parse_args()
    nodes, data, sys_nodes = load_nodes()
    P = build_transition(nodes, data, sys_nodes)
    prod, cons, avail, demand = solve_economy(nodes, P)
    prices = price_field(avail, demand)
    print(f"loaded {len(nodes)} nodes "
          f"({sum(nd['kind']=='planet' for nd in nodes)} planets, "
          f"{sum(nd['kind']=='field' for nd in nodes)} fields), "
          f"alpha={ALPHA} beta={BETA}")
    report(nodes, prices)
    if args.write:
        write_prices(nodes, prices, prod, avail, demand, os.path.abspath(STAR_SYSTEMS))


if __name__ == "__main__":
    main()
