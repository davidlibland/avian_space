#!/usr/bin/env python3
"""Hourly overnight monitor — appends snapshots to a markdown log file.

Usage:
    python scripts/overnight_monitor.py [RUN_DIR] [LOG_FILE] [HOURS]

    Defaults: latest run, docs/overnight_run16.md, 8 hours
"""

from __future__ import annotations
import os, re, sys, time
from datetime import datetime

import numpy as np
from tbparse import SummaryReader

PERSONALITIES = ["fighter", "miner", "trader"]
REWARD_TYPES = ["landing", "cargo_sold", "ship_hit", "asteroid_hit", "pickup", "health_gated", "health_raw", "damage"]
REWARD_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]


def find_latest_tb_dir(base="experiments"):
    runs = []
    for name in os.listdir(base):
        m = re.match(r"run_(\d+)", name)
        if m:
            runs.append((int(m.group(1)), name))
    runs.sort()
    return os.path.join(base, runs[-1][1], "tb") if runs else None


def get(df, tag):
    return df[df["tag"] == tag]["value"].values


def snapshot(tb_dir):
    reader = SummaryReader(tb_dir)
    df = reader.scalars
    n = len(get(df, "reward/mean_per_step"))

    lines = []
    lines.append(f"### Cycle {n} — {datetime.now().strftime('%H:%M')}")
    lines.append("")

    # Training health
    ploss = get(df, "train/policy_loss")
    ent = get(df, "train/entropy")
    clip = get(df, "train/frac_clipped")
    ev = get(df, "train/explained_variance")
    n_nan = int(np.sum(np.isnan(ploss)))
    lines.append(f"**Training:** policy_loss={np.nanmean(ploss[-5:]):.4f}, "
                 f"entropy={np.nanmean(ent[-5:]):.3f}, "
                 f"clip={np.nanmean(clip[-5:]):.3f}, "
                 f"EV={np.nanmean(ev[-5:]):.3f}, "
                 f"NaN={n_nan}")
    lines.append("")

    # Effective reward per step (after weights)
    w_map = dict(zip(REWARD_TYPES, REWARD_WEIGHTS))
    lines.append("**Effective reward/step (raw × weight):**")
    lines.append("")
    header = "| | " + " | ".join(REWARD_TYPES) + " | TOTAL |"
    sep = "|---|" + "|".join(["---"] * len(REWARD_TYPES)) + "|---|"
    lines.append(header)
    lines.append(sep)
    for pers in PERSONALITIES:
        row = []
        for rtype in REWARD_TYPES:
            vals = get(df, f"reward_per_step/{pers}/{rtype}")
            raw = float(np.mean(vals[-10:])) if len(vals) >= 5 else 0.0
            row.append(raw * w_map[rtype])
        total = sum(row)
        cells = " | ".join(f"{v:.5f}" for v in row)
        lines.append(f"| {pers} | {cells} | {total:.5f} |")
    lines.append("")

    # Reward trends
    lines.append("**Reward trends (Q1 → Q4):**")
    lines.append("")
    for pers in PERSONALITIES:
        changes = []
        for rtype in REWARD_TYPES:
            vals = get(df, f"reward_per_step/{pers}/{rtype}")
            if len(vals) >= 8:
                q = len(vals) // 4
                early = np.mean(vals[:q])
                late = np.mean(vals[-q:])
                pct = (late - early) / (abs(early) + 1e-8) * 100
                if abs(pct) > 5:
                    changes.append(f"{rtype}={pct:+.0f}%")
        if changes:
            lines.append(f"- **{pers}**: {', '.join(changes)}")
    lines.append("")

    # Per-head EV
    lines.append("**Value head EV (last 5):**")
    lines.append("")
    lines.append("| head | EV |")
    lines.append("|---|---|")
    for rtype in REWARD_TYPES:
        ev_vals = get(df, f"value_head/{rtype}/explained_variance")
        if len(ev_vals) >= 5:
            ev_str = ", ".join(f"{v:.3f}" for v in ev_vals[-5:])
            lines.append(f"| {rtype} | [{ev_str}] |")
    total_ev = get(df, "train/explained_variance")
    if len(total_ev) >= 5:
        ev_str = ", ".join(f"{v:.3f}" for v in total_ev[-5:])
        lines.append(f"| **TOTAL** | [{ev_str}] |")
    lines.append("")

    # Landing counts
    landing_counts = {}
    for pers in PERSONALITIES:
        for rt in ["landing", "cargo_sold"]:
            vals = get(df, f"reward_total/{pers}/{rt}")
            nz = int(np.count_nonzero(vals))
            landing_counts[f"{pers}/{rt}"] = f"{nz}/{len(vals)}"

    lines.append(f"**Landing events:** " + ", ".join(
        f"{k}={v}" for k, v in landing_counts.items()
    ))
    lines.append("")

    # Nav target selection
    lines.append("**Nav target selection (last 5 avg):**")
    lines.append("")
    nav_types = ["ship", "asteroid", "planet", "pickup", "none"]
    header = "| | " + " | ".join(nav_types) + " |"
    sep = "|---|" + "|".join(["---"] * len(nav_types)) + "|"
    lines.append(header)
    lines.append(sep)
    for pers in PERSONALITIES:
        row = []
        for tt in nav_types:
            vals = get(df, f"nav_target_type/{pers}/{tt}")
            row.append(float(np.mean(vals[-5:])) if len(vals) >= 5 else 0.0)
        cells = " | ".join(f"{v:.3f}" for v in row)
        lines.append(f"| {pers} | {cells} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def main():
    tb_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_tb_dir()
    log_file = sys.argv[2] if len(sys.argv) > 2 else "docs/overnight_run16.md"
    hours = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    print(f"Monitoring {tb_dir} → {log_file} for {hours} hours")

    for i in range(hours):
        try:
            snap = snapshot(tb_dir)
            with open(log_file, "a") as f:
                f.write(snap)
            print(f"[{datetime.now().strftime('%H:%M')}] Snapshot {i+1}/{hours} written")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M')}] Error: {e}")

        if i < hours - 1:
            time.sleep(3600)

    # Final summary
    print("Monitoring complete.")


if __name__ == "__main__":
    main()
