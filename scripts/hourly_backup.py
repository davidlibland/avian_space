#!/usr/bin/env python3
"""Hourly backup + monitoring script.

Creates a timestamped backup of the checkpoint files and appends a training
snapshot to the overnight log. Designed to be called by a cron job.

Usage:
    python scripts/hourly_backup.py [RUN_DIR] [LOG_FILE]
"""

import os
import re
import shutil
import sys
from datetime import datetime

import numpy as np
from tbparse import SummaryReader

PERSONALITIES = ["fighter", "miner", "trader"]
REWARD_TYPES = ["health", "weapon_hit", "landing", "cargo_sold", "pickup",
                "nav_target", "weapons_target", "movement"]


def find_latest_run(base="experiments"):
    runs = []
    for name in os.listdir(base):
        m = re.match(r"run_(\d+)", name)
        if m:
            runs.append((int(m.group(1)), name))
    runs.sort()
    return os.path.join(base, runs[-1][1]) if runs else None


def backup_checkpoint(run_dir):
    """Copy checkpoint files to a timestamped backup directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    backup_dir = os.path.join(run_dir, "backups", ts)
    os.makedirs(backup_dir, exist_ok=True)

    files = ["policy.bin", "value.bin", "policy_optim.bin", "value_optim.bin",
             "step_counter.bin", "rl_buffer.bin"]
    copied = 0
    for f in files:
        src = os.path.join(run_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(backup_dir, f))
            copied += 1

    return backup_dir, copied


def get(df, tag):
    return df[df["tag"] == tag]["value"].values


def snapshot(tb_dir):
    reader = SummaryReader(tb_dir)
    df = reader.scalars
    n = len(get(df, "reward/mean_per_step"))

    lines = []
    ts = datetime.now().strftime("%H:%M")
    lines.append(f"### Cycle {n} — {ts}")
    lines.append("")

    # NaN check
    ploss = get(df, "train/policy_loss")
    n_nan = int(np.sum(np.isnan(ploss)))
    if n_nan > 0:
        lines.append(f"**⚠ NaN DETECTED: {n_nan}/{len(ploss)} policy loss values are NaN!**")
        lines.append("")

    # Training health
    ent = get(df, "train/entropy")
    clip = get(df, "train/frac_clipped")
    ev = get(df, "train/explained_variance")
    lines.append(f"**Training:** policy_loss={np.nanmean(ploss[-5:]):.4f}, "
                 f"entropy={np.nanmean(ent[-5:]):.3f}, "
                 f"clip={np.nanmean(clip[-5:]):.3f}, "
                 f"EV={np.nanmean(ev[-5:]):.3f}, NaN={n_nan}")
    lines.append("")

    # Landing counts
    landing_lines = []
    total_landings = 0
    for pers in PERSONALITIES:
        vals = get(df, f"reward_total/{pers}/landing")
        nz = int(np.count_nonzero(vals))
        total_landings += nz
        landing_lines.append(f"{pers}={nz}/{len(vals)}")
    lines.append(f"**Landing:** {', '.join(landing_lines)} (total={total_landings})")
    lines.append("")

    # Landing EV
    lev = get(df, "value_head/landing/explained_variance")
    if len(lev) >= 5:
        lines.append(f"**Landing EV:** [{', '.join(f'{v:.3f}' for v in lev[-5:])}]")
        lines.append("")

    # Effective reward per step
    weights = dict(zip(REWARD_TYPES, [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1]))
    lines.append("**Effective reward/step (last 20):**")
    lines.append("")
    for pers in PERSONALITIES:
        parts = []
        for rt in REWARD_TYPES:
            vals = get(df, f"reward_per_step/{pers}/{rt}")
            raw = float(np.mean(vals[-20:])) if len(vals) >= 20 else 0.0
            eff = raw * weights[rt]
            if eff > 0.0005:
                parts.append(f"{rt}={eff:.4f}")
        lines.append(f"- **{pers}**: {', '.join(parts)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_run()
    log_file = sys.argv[2] if len(sys.argv) > 2 else "docs/overnight_run16.md"

    if not run_dir:
        print("No run directory found")
        return

    tb_dir = os.path.join(run_dir, "tb")
    ts = datetime.now().strftime("%H:%M")

    # Backup checkpoint
    backup_dir, n_copied = backup_checkpoint(run_dir)
    print(f"[{ts}] Backed up {n_copied} files to {backup_dir}")

    # Training snapshot
    try:
        snap = snapshot(tb_dir)
        with open(log_file, "a") as f:
            f.write(snap)
        print(f"[{ts}] Snapshot appended to {log_file}")
    except Exception as e:
        print(f"[{ts}] Snapshot error: {e}")


if __name__ == "__main__":
    main()
