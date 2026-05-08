#!/usr/bin/env python3
"""Per-personality × reward-type progress and cooperation analysis.

Unlike `analyze_rewards.py`, which averages over the entire training run,
this script:

  * Compares an EARLY post-warmup window against the RECENT tail of the run.
  * Reports block-bootstrap standard errors (cycles are autocorrelated, so
    pretending each cycle is i.i.d. understates uncertainty).
  * Flags changes that exceed 2 std-errors as "significant" (★).
  * Adds a dedicated cooperation section that exploits the reward-sharing
    mechanism: a fighter's `cargo_sold` reward must come from *visible
    allied traders*, etc. Off-diagonal cells therefore measure how often
    different roles are co-located.

Usage:
    python scripts/analyze_progress.py [RUN_DIR]   # tb dir or run dir
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
from tbparse import SummaryReader

PERSONALITIES = ["fighter", "miner", "trader"]
REWARD_TYPES = ["landing", "cargo_sold", "ship_hit", "asteroid_hit",
                "pickup", "health_gated", "damage"]
# health_raw is dominated by the "alive" baseline; uninformative for learning.

# Reward-sharing fractions from src/consts.rs.
SHARE_FRAC = {"fighter": 0.30, "miner": 0.05, "trader": 0.05}

# A ship can earn each reward type "natively" only for some role(s). Anything
# else showing up is shared from a visible ally.
NATIVE_SOURCES = {
    "landing":      {"fighter", "miner", "trader"},  # any ship can land
    "cargo_sold":   {"trader", "miner"},              # both haul cargo
    "ship_hit":     {"fighter"},                       # fighters fight
    "asteroid_hit": {"miner"},
    "pickup":       {"miner"},                         # primarily miners
    "health_gated": {"fighter", "miner", "trader"},
    "damage":       {"fighter", "miner", "trader"},
}

EARLY_LO_FRAC, EARLY_HI_FRAC = 0.05, 0.15  # 5–15 % of run = post-warmup baseline
RECENT_TAIL_FRAC = 0.10                     # last 10 % = "recent"
BLOCK = 50                                  # bootstrap block (cycles)


# ────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ────────────────────────────────────────────────────────────────────────────


def block_mean_se(x: np.ndarray, block: int = BLOCK) -> tuple[float, float]:
    """Block-mean estimate and SE, accounting for autocorrelation."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    if n < block * 2:
        return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(max(n, 1)))
    nb = n // block
    blocks = x[: nb * block].reshape(nb, block).mean(axis=1)
    return float(np.mean(blocks)), float(np.std(blocks, ddof=1) / np.sqrt(nb))


def windowed_compare(values: np.ndarray, n: int):
    """Return (early_mean, early_se, recent_mean, recent_se, delta, se_diff)."""
    early_lo = int(EARLY_LO_FRAC * n)
    early_hi = int(EARLY_HI_FRAC * n)
    recent_lo = int((1 - RECENT_TAIL_FRAC) * n)
    early = values[early_lo:early_hi]
    recent = values[recent_lo:]
    em, es = block_mean_se(early)
    rm, rs = block_mean_se(recent)
    delta = rm - em
    se_diff = float(np.sqrt(es * es + rs * rs))
    return em, es, rm, rs, delta, se_diff


def fmt_cell(em, es, rm, rs, delta, se_diff, fmt=".4f"):
    if not np.isfinite(em) or not np.isfinite(rm):
        return "—"
    sig = "★" if (np.isfinite(se_diff) and se_diff > 0
                  and abs(delta) > 2 * se_diff) else " "
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "·")
    return (f"{em:{fmt}}±{es:.1e} → {rm:{fmt}}±{rs:.1e} "
            f"Δ={delta:+{fmt}}±{se_diff:.1e} {arrow}{sig}")


# ────────────────────────────────────────────────────────────────────────────
# Tag access
# ────────────────────────────────────────────────────────────────────────────


def find_run(path: str) -> str:
    if path.endswith("/tb") or path.endswith("/tb/"):
        return path
    if os.path.isdir(os.path.join(path, "tb")):
        return os.path.join(path, "tb")
    return path


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


def get(df, tag: str) -> np.ndarray:
    return df[df["tag"] == tag]["value"].to_numpy()


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


def print_progress_table(df, tag_prefix: str, title: str, fmt: str = ".4f"):
    n_cycles = len(get(df, "reward/mean_per_step"))
    if n_cycles < 200:
        print(f"\n{title}: not enough cycles ({n_cycles})")
        return

    print(f"\n{title}")
    print(f"  early window = cycles "
          f"[{int(EARLY_LO_FRAC*n_cycles)}, {int(EARLY_HI_FRAC*n_cycles)})  "
          f"|  recent = last {int(RECENT_TAIL_FRAC*n_cycles)} cycles  "
          f"(★ = |Δ| > 2·SE)")
    print(f"  {'(rtype)':>14}  {'fighter':<55}  {'miner':<55}  {'trader':<55}")
    for rtype in REWARD_TYPES:
        cells = []
        for pers in PERSONALITIES:
            vals = get(df, f"{tag_prefix}/{pers}/{rtype}")
            if len(vals) == 0:
                cells.append("—")
                continue
            cells.append(fmt_cell(*windowed_compare(vals, len(vals)), fmt=fmt))
        print(f"  {rtype:>14}  " + "  ".join(f"{c:<55}" for c in cells))


def cooperation_analysis(df):
    """Look at off-diagonal cells (rewards that can only arrive via sharing)."""
    n_cycles = len(get(df, "reward/mean_per_step"))
    print("\n" + "=" * 80)
    print("COOPERATION ANALYSIS")
    print("=" * 80)
    print("""
A fighter cannot earn `cargo_sold` reward natively (they don't haul cargo);
any positive value must arrive via the reward-sharing mechanism — i.e. a
visible allied trader/miner sold cargo while the fighter could see them.
Same for `asteroid_hit`/`pickup` (from miners). Conversely, a miner's or
trader's `ship_hit` reward must come from an allied fighter scoring a hit
nearby. Growth in these "cross-shared" cells over training = increased
co-location of different roles, the prerequisite for cooperative behavior.
""")

    # 1. Cross-shared (cooperation) rewards over time.
    cross_shared_pairs = []
    for pers in PERSONALITIES:
        for rtype in REWARD_TYPES:
            if pers in NATIVE_SOURCES[rtype]:
                continue  # native — not necessarily cooperation
            cross_shared_pairs.append((pers, rtype))

    print("Cross-shared reward signals (per-step; non-native cells only):\n")
    print(f"  {'(pers/rtype)':>22}  {'window comparison':<60}")
    for pers, rtype in cross_shared_pairs:
        vals = get(df, f"reward_per_step/{pers}/{rtype}")
        if len(vals) == 0:
            continue
        em, es, rm, rs, delta, se_diff = windowed_compare(vals, len(vals))
        # Drop the noisy zero-baseline rewards that never moved.
        if abs(rm) < 1e-7 and abs(em) < 1e-7:
            continue
        cell = fmt_cell(em, es, rm, rs, delta, se_diff, fmt=".4f")
        print(f"  {pers + '/' + rtype:>22}  {cell}")

    # 2. Cooperation index: shared / total reward, per personality, over time.
    print("\nCooperation index (sum-of-shared / sum-of-|all| reward), per personality:")
    print("  Higher = more of the personality's reward signal is coming from allies.\n")
    print(f"  {'(personality)':>14}  {'early':>20}  {'recent':>20}  {'Δ':>20}")
    for pers in PERSONALITIES:
        own_sum = np.zeros(n_cycles)
        shared_sum = np.zeros(n_cycles)
        for rtype in REWARD_TYPES:
            v = get(df, f"reward_per_step/{pers}/{rtype}")
            if len(v) != n_cycles:
                continue
            if pers in NATIVE_SOURCES[rtype]:
                own_sum += np.abs(v)
            else:
                shared_sum += np.abs(v)
        denom = own_sum + shared_sum
        denom = np.where(denom > 1e-12, denom, np.nan)
        ratio = shared_sum / denom
        em, es, rm, rs, delta, se_diff = windowed_compare(ratio, n_cycles)
        sig = "★" if (np.isfinite(se_diff) and se_diff > 0
                       and abs(delta) > 2 * se_diff) else " "
        print(f"  {pers:>14}  "
              f"{em:.4f}±{es:.1e}".ljust(28)
              + f"{rm:.4f}±{rs:.1e}".ljust(22)
              + f"{delta:+.4f}±{se_diff:.1e} {sig}")

    # 3. Co-occurrence correlation: smooth each series and correlate per-cycle.
    print("\nPer-cycle reward correlations (smoothed; recent window only):")
    print("  ρ > 0 means rewards rise/fall together → roles co-active in time.\n")

    def smooth(x, w=50):
        if len(x) < w:
            return x
        c = np.cumsum(np.insert(x, 0, 0.0))
        return (c[w:] - c[:-w]) / w

    recent_lo = int((1 - RECENT_TAIL_FRAC) * n_cycles)

    def recent_smoothed(tag):
        v = get(df, tag)
        if len(v) == 0:
            return None
        v = v[recent_lo:]
        return smooth(v)

    pairs = [
        # Fighter rewarded for hitting a hostile while miner is mining → escort signal
        ("reward_per_step/fighter/ship_hit", "reward_per_step/miner/asteroid_hit",
         "fighter:ship_hit  ↔  miner:asteroid_hit  (escort-while-mining)"),
        ("reward_per_step/fighter/ship_hit", "reward_per_step/trader/cargo_sold",
         "fighter:ship_hit  ↔  trader:cargo_sold  (escort-while-trading)"),
        # Damage taken should anti-correlate with fighter ship_hit if escorting works
        ("reward_per_step/fighter/ship_hit", "reward_per_step/miner/damage",
         "fighter:ship_hit  ↔  miner:damage     (negative = protection)"),
        ("reward_per_step/fighter/ship_hit", "reward_per_step/trader/damage",
         "fighter:ship_hit  ↔  trader:damage    (negative = protection)"),
        # Ally-shared reward bleed: if fighter is escorting traders, fighter/cargo_sold
        # tracks trader/cargo_sold.
        ("reward_per_step/fighter/cargo_sold", "reward_per_step/trader/cargo_sold",
         "fighter:cargo_sold ↔  trader:cargo_sold (shared-bleed; escorting traders)"),
        ("reward_per_step/fighter/asteroid_hit", "reward_per_step/miner/asteroid_hit",
         "fighter:asteroid_hit ↔ miner:asteroid_hit (shared-bleed; escorting miners)"),
    ]

    for ta, tb, label in pairs:
        a = recent_smoothed(ta)
        b = recent_smoothed(tb)
        if a is None or b is None or len(a) < 50 or len(b) < 50:
            continue
        m = min(len(a), len(b))
        a, b = a[:m], b[:m]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            print(f"  {label:<55}  ρ = (flat)")
            continue
        rho = float(np.corrcoef(a, b)[0, 1])
        # Approximate SE under independence: 1/sqrt(N - 3).
        se_rho = 1.0 / np.sqrt(max(m - 3, 1))
        sig = "★" if abs(rho) > 2 * se_rho else " "
        print(f"  {label:<60}  ρ = {rho:+.3f} ± {se_rho:.3f} {sig}")


def diagnostics(df):
    n = len(get(df, "reward/mean_per_step"))
    print("\nRun-level diagnostics (recent window):")
    for tag in ["train/policy_loss", "train/value_loss", "train/entropy",
                "train/frac_clipped", "train/explained_variance",
                "throughput/wait_fraction"]:
        v = get(df, tag)
        if len(v) == 0:
            continue
        em, es, rm, rs, delta, se_diff = windowed_compare(v, n)
        sig = "★" if (np.isfinite(se_diff) and se_diff > 0
                      and abs(delta) > 2 * se_diff) else " "
        print(f"  {tag:>32}: early {em:+.4f}±{es:.1e}  →  "
              f"recent {rm:+.4f}±{rs:.1e}  Δ={delta:+.4f} {sig}")


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else None
    tb_dir = find_run(base) if base else find_latest_tb_dir()
    print(f"Reading TensorBoard logs from: {tb_dir}")
    reader = SummaryReader(tb_dir)
    df = reader.scalars
    n_cycles = len(get(df, "reward/mean_per_step"))
    print(f"Total update cycles: {n_cycles}")

    print_progress_table(df, "reward_per_step",
                         "Per-step reward — early vs recent window",
                         fmt=".4f")
    cooperation_analysis(df)
    diagnostics(df)


if __name__ == "__main__":
    main()
