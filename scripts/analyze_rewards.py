#!/usr/bin/env python3
"""Analyze per-personality × reward-type breakdowns from TensorBoard logs.

Reads the event files produced by the PPO training thread and prints
summary tables of reward totals, per-step averages, and value-head
explained variance.

Usage:
    python scripts/analyze_rewards.py [RUN_DIR]

    RUN_DIR defaults to the highest-numbered experiments/run_*/tb/ directory.

Examples:
    python scripts/analyze_rewards.py                       # latest run
    python scripts/analyze_rewards.py experiments/run_3/tb  # specific run

Requirements:
    pip install -r scripts/analysis_requirements.txt
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
from tbparse import SummaryReader

PERSONALITIES = ["fighter", "miner", "trader"]
REWARD_TYPES = ["landing", "cargo_sold", "ship_hit", "asteroid_hit", "pickup", "health_gated", "health_raw", "damage"]


def find_latest_tb_dir(base: str = "experiments") -> str:
    """Return the tb/ subdirectory of the highest-numbered run."""
    runs = []
    for name in os.listdir(base):
        m = re.match(r"run_(\d+)", name)
        if m:
            runs.append((int(m.group(1)), name))
    if not runs:
        sys.exit(f"No run directories found under {base}/")
    runs.sort()
    return os.path.join(base, runs[-1][1], "tb")


def print_table(title: str, row_labels: list[str], col_labels: list[str],
                data: list[list[float]], fmt: str = ".5f"):
    """Pretty-print a 2-D table."""
    col_w = max(10, max(len(c) for c in col_labels) + 2)
    row_w = max(len(r) for r in row_labels) + 2
    print(f"\n{title}")
    header = " " * row_w + "".join(c.rjust(col_w) for c in col_labels)
    print(header)
    print("-" * len(header))
    for label, row in zip(row_labels, data):
        cells = "".join(f"{v:{col_w}{fmt}}" for v in row)
        print(f"{label:>{row_w}}{cells}")


def main():
    if len(sys.argv) > 1:
        tb_dir = sys.argv[1]
    else:
        tb_dir = find_latest_tb_dir()

    print(f"Reading TensorBoard logs from: {tb_dir}")
    reader = SummaryReader(tb_dir)
    df = reader.scalars
    tags = set(df["tag"].unique())

    # ── Per-step reward matrix ────────────────────────────────────────────
    per_step = []
    for pers in PERSONALITIES:
        row = []
        for rtype in REWARD_TYPES:
            tag = f"reward_per_step/{pers}/{rtype}"
            vals = df[df["tag"] == tag]["value"].values
            row.append(float(np.mean(vals)) if len(vals) > 0 else 0.0)
        per_step.append(row)
    print_table("Mean reward per step (averaged across all cycles)",
                PERSONALITIES, REWARD_TYPES, per_step)

    # ── Total reward matrix ───────────────────────────────────────────────
    totals = []
    for pers in PERSONALITIES:
        row = []
        for rtype in REWARD_TYPES:
            tag = f"reward_total/{pers}/{rtype}"
            vals = df[df["tag"] == tag]["value"].values
            row.append(float(np.mean(vals)) if len(vals) > 0 else 0.0)
        totals.append(row)
    print_table("Mean reward total per cycle (averaged across all cycles)",
                PERSONALITIES, REWARD_TYPES, totals, fmt=".3f")

    # ── Value-head explained variance (last 5 cycles) ────────────────────
    print("\nValue-head explained variance (last 5 cycles):")
    for rtype in REWARD_TYPES:
        tag = f"value_head/{rtype}/explained_variance"
        vals = df[df["tag"] == tag]["value"].values
        last5 = vals[-5:] if len(vals) >= 5 else vals
        formatted = [f"{v:.4f}" for v in last5]
        print(f"  {rtype:>12}: {formatted}")

    if "train/explained_variance" in tags:
        ev = df[df["tag"] == "train/explained_variance"]["value"].values
        print(f"  {'TOTAL':>12}: {[f'{v:.4f}' for v in ev[-5:]]}")

    # ── Overall training stats ────────────────────────────────────────────
    print("\nTraining stats (last 5 cycles):")
    for tag_name in ["train/policy_loss", "train/value_loss", "train/entropy",
                     "train/frac_clipped", "train/value_burnin"]:
        if tag_name in tags:
            vals = df[df["tag"] == tag_name]["value"].values[-5:]
            print(f"  {tag_name:>30}: {[f'{v:.4f}' for v in vals]}")

    n_cycles = len(df[df["tag"] == "reward/mean_per_step"])
    print(f"\nTotal update cycles: {n_cycles}")


if __name__ == "__main__":
    main()
