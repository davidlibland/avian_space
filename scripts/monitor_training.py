#!/usr/bin/env python3
"""Real-time training monitor — prints a comprehensive dashboard.

Usage:
    python scripts/monitor_training.py [RUN_DIR]

    RUN_DIR defaults to the highest-numbered experiments/run_*/tb/ directory.

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
REWARD_TYPES = ["health", "weapon_hit", "landing", "cargo_sold", "pickup", "goal_target"]
REWARD_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1]  # keep in sync with consts.rs


def find_latest_tb_dir(base: str = "experiments") -> str:
    runs = []
    for name in os.listdir(base):
        m = re.match(r"run_(\d+)", name)
        if m:
            runs.append((int(m.group(1)), name))
    if not runs:
        sys.exit(f"No run directories found under {base}/")
    runs.sort()
    return os.path.join(base, runs[-1][1], "tb")


def get(df, tag):
    return df[df["tag"] == tag]["value"].values


def print_section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def main():
    tb_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_tb_dir()
    print(f"Reading: {tb_dir}")
    reader = SummaryReader(tb_dir)
    df = reader.scalars
    n = len(get(df, "reward/mean_per_step"))
    print(f"Total update cycles: {n}")

    # ── Throughput ────────────────────────────────────────────────────────
    print_section("THROUGHPUT")
    for tag, label in [
        ("throughput/wait_secs", "Wait (data collection)"),
        ("throughput/train_secs", "Train (gradient steps)"),
        ("throughput/cycle_secs", "Total cycle time"),
        ("throughput/segments", "Segments per cycle"),
        ("throughput/train_steps_per_sec", "Train steps/sec (GPU)"),
    ]:
        vals = get(df, tag)
        if len(vals) >= 5:
            last5 = vals[-5:]
            print(f"  {label:30s}  last5: [{', '.join(f'{v:.1f}' for v in last5)}]  "
                  f"mean={np.mean(vals[-10:]):.1f}")

    wait = get(df, "throughput/wait_secs")
    train = get(df, "throughput/train_secs")
    if len(wait) >= 5 and len(train) >= 5:
        recent_wait = np.mean(wait[-5:])
        recent_train = np.mean(train[-5:])
        total = recent_wait + recent_train
        if total > 0:
            print(f"\n  Bottleneck: {'DATA' if recent_wait > recent_train else 'TRAINING'}"
                  f"  (wait={recent_wait:.1f}s, train={recent_train:.1f}s)")

    # ── Effective reward per step (after weights) ─────────────────────────
    print_section("EFFECTIVE REWARD PER STEP (raw × weight)")
    w_map = dict(zip(REWARD_TYPES, REWARD_WEIGHTS))
    header = f"{'':>12}" + "".join(f"{r:>12}" for r in REWARD_TYPES) + "     TOTAL"
    print(header)
    print("-" * len(header))
    for pers in PERSONALITIES:
        row_vals = []
        for rtype in REWARD_TYPES:
            vals = get(df, f"reward_per_step/{pers}/{rtype}")
            raw = float(np.mean(vals[-10:])) if len(vals) >= 5 else 0.0
            row_vals.append(raw * w_map[rtype])
        total = sum(row_vals)
        cells = "".join(f"{v:12.5f}" for v in row_vals)
        print(f"{pers:>12}{cells}  {total:8.5f}")

    # ── Reward evolution ──────────────────────────────────────────────────
    print_section("REWARD TREND (first quarter → last quarter, % change)")
    for pers in PERSONALITIES:
        changes = []
        for rtype in REWARD_TYPES:
            vals = get(df, f"reward_per_step/{pers}/{rtype}")
            if len(vals) >= 8:
                q = len(vals) // 4
                early = np.mean(vals[:q])
                late = np.mean(vals[-q:])
                pct = (late - early) / (abs(early) + 1e-8) * 100
                changes.append(f"{rtype}={pct:+.0f}%")
        if changes:
            print(f"  {pers:>10}: {', '.join(changes)}")

    # ── Training health ───────────────────────────────────────────────────
    print_section("TRAINING HEALTH (last 5 cycles)")
    for tag, label in [
        ("train/policy_loss", "Policy loss"),
        ("train/value_loss", "Value loss"),
        ("train/entropy", "Entropy"),
        ("train/frac_clipped", "Clip fraction"),
        ("train/explained_variance", "Explained var (total)"),
        ("train/advantage_std", "Advantage std"),
    ]:
        vals = get(df, tag)
        if len(vals) >= 5:
            last5 = vals[-5:]
            print(f"  {label:25s}  [{', '.join(f'{v:.4f}' for v in last5)}]")

    # ── Per-head value diagnostics ────────────────────────────────────────
    print_section("VALUE HEAD EXPLAINED VARIANCE (last 5)")
    for rtype in REWARD_TYPES:
        vals = get(df, f"value_head/{rtype}/explained_variance")
        if len(vals) >= 5:
            last5 = vals[-5:]
            print(f"  {rtype:>12}: [{', '.join(f'{v:.3f}' for v in last5)}]")
    vals = get(df, "train/explained_variance")
    if len(vals) >= 5:
        print(f"  {'TOTAL':>12}: [{', '.join(f'{v:.3f}' for v in vals[-5:])}]")

    # ── Warnings ──────────────────────────────────────────────────────────
    print_section("DIAGNOSTICS")
    segs = get(df, "throughput/segments")
    if len(segs) >= 3 and segs[-1] > segs[-2] > segs[-3]:
        print("  ⚠  Segments per cycle growing — batch size runaway!")
        print(f"     Last 5: {[f'{v:.0f}' for v in segs[-5:]]}")

    clip = get(df, "train/frac_clipped")
    if len(clip) >= 3:
        recent_clip = np.mean(clip[-3:])
        if recent_clip > 0.4:
            print(f"  ⚠  High clip fraction ({recent_clip:.2f}) — policy changing too fast")

    ent = get(df, "train/entropy")
    ent_active = ent[ent > 0]
    if len(ent_active) >= 5:
        recent_ent = np.mean(ent_active[-5:])
        if recent_ent < 0.2:
            print(f"  ⚠  Low entropy ({recent_ent:.3f}) — policy may be collapsing")

    ploss = get(df, "train/policy_loss")
    if len(ploss) >= 3 and all(ploss[-3:] > 0.1):
        print(f"  ⚠  High policy loss ({ploss[-1]:.3f}) — may indicate instability")

    print()


if __name__ == "__main__":
    main()
