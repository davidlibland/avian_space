#!/usr/bin/env python3
"""Distilled status summary for the SFT->RL handoff experiment (run_4+).

Focuses on the three questions that matter for this experiment, rather than the
full dashboard (see monitor_training.py for that):

  1. ANCHORING  — does the BC-cloned nav/targeting survive RL, or collapse?
                  (prior un-anchored run collapsed nav bc_agreement 0.70 -> 0.26)
  2. CAPACITY   — does the bigger value net (hidden=128) raise explained
                  variance above the hidden=64 baseline (TOTAL ~0.20)?
  3. BEHAVIORS  — do landing / cargo / combat rewards survive into trained RL?

Reads the highest-numbered experiments/run_*/tb. Run via the analysis venv:
    .venv/bin/python scripts/experiment_summary.py
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
from tbparse import SummaryReader

PERSONALITIES = ["fighter", "miner", "trader"]
HEADS = ["turn", "thrust", "fire_primary", "fire_secondary", "nav", "wep"]
# Baseline (value hidden=64, run_2 mature ~cycle 2500).
BASELINE_TOTAL_EV = 0.20
VALUE_HEADS = ["landing", "cargo_sold", "ship_hit", "asteroid_hit",
               "pickup", "health_gated", "damage"]


def latest_tb(base="experiments"):
    runs = [(int(m.group(1)), n) for n in os.listdir(base)
            if (m := re.match(r"run_(\d+)$", n))]
    if not runs:
        sys.exit("no runs")
    runs.sort()
    return runs[-1][1], os.path.join(base, runs[-1][1], "tb")


def main():
    run, tb = latest_tb()
    if not os.path.isdir(tb):
        print(f"{run}: no tb dir yet (BC stage writes no TensorBoard).")
        return
    df = SummaryReader(tb).scalars

    def g(tag):
        return df[df["tag"] == tag]["value"].values

    n = len(g("reward/mean_per_step"))
    print(f"=== {run}  (BC-anchored multi-world RL, value hidden=128)  cycle={n} ===")

    # Training health / burn-in.
    ent = g("train/entropy")
    clip = g("train/frac_clipped")
    burn = g("train/value_burnin")
    ev = g("train/explained_variance")
    if len(burn) >= 1:
        recent = burn[-min(len(burn), 60):]
        bpct = 100.0 * float(np.mean(recent))
        print(f"  health: burn-in~{bpct:.0f}% (policy trains ~{100-bpct:.0f}%)  "
              f"entropy={np.mean(ent[-10:]):.2f}  clip={np.mean(clip[-10:]):.3f}"
              if len(ent) else f"  burn-in~{bpct:.0f}%")

    # 1. ANCHORING — nav/targeting BC-agreement: does it hold?
    print("  [1] ANCHORING (BC survival; target: nav holds ~0.70, not ->0.26):")
    nav = g("policy/bc_agreement/nav")
    if len(nav) == 0:
        print("      policy frozen at BC clone (no policy updates logged yet — value burn-in)")
    else:
        early = float(np.mean(nav[:5]))
        now = float(np.mean(nav[-5:]))
        verdict = "HOLDING" if now >= 0.55 else "eroding" if now < early - 0.1 else "mid"
        print(f"      nav  early={early:.2f} -> now={now:.2f}  [{verdict}]")
        for h in ["fire_primary", "wep"]:
            v = g(f"policy/bc_agreement/{h}")
            if len(v):
                print(f"      {h:13} now={np.mean(v[-5:]):.2f}")

    # 2. CAPACITY — value EV vs baseline.
    print(f"  [2] CAPACITY (value hidden=128 vs baseline TOTAL EV~{BASELINE_TOTAL_EV}):")
    if len(ev):
        t = float(np.mean(ev[-10:]))
        flag = "ABOVE baseline" if t > BASELINE_TOTAL_EV else "below baseline (may be immature)"
        print(f"      TOTAL EV={t:.3f}  [{flag}]")
    per = []
    for h in VALUE_HEADS:
        v = g(f"value_head/{h}/explained_variance")
        if len(v):
            per.append(f"{h}={np.mean(v[-10:]):.2f}")
    if per:
        print("      per-head: " + "  ".join(per))

    # 3. BEHAVIORS — do landing/cargo/combat survive (reward/step)?
    print("  [3] BEHAVIORS survive? (reward/step, last 10):")
    for label, tag in [
        ("trader landing", "reward_per_step/trader/landing"),
        ("trader cargo_sold", "reward_per_step/trader/cargo_sold"),
        ("fighter ship_hit", "reward_per_step/fighter/ship_hit"),
        ("miner asteroid_hit", "reward_per_step/miner/asteroid_hit"),
    ]:
        v = g(tag)
        val = float(np.mean(v[-10:])) if len(v) else 0.0
        print(f"      {label:20} {val:.5f}")

    # Cooperation (shared reward).
    af = g("reward_shared_fraction/all")
    if len(af):
        parts = [f"all={np.mean(af[-10:]):.1%}"]
        for p in PERSONALITIES:
            v = g(f"reward_shared_fraction/{p}")
            if len(v):
                parts.append(f"{p}={np.mean(v[-10:]):.1%}")
        print("  shared reward: " + "  ".join(parts))


if __name__ == "__main__":
    main()
