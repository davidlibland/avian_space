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
REWARD_TYPES = ["landing", "cargo_sold", "ship_hit", "asteroid_hit", "pickup", "health_gated", "health_raw", "damage"]
REWARD_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]  # keep in sync with consts.rs


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


def mean_se(vals, k=10):
    """Return (mean, standard_error_of_mean) over the last k samples.

    Returns (nan, nan) if fewer than 2 samples available.
    """
    if len(vals) < 2:
        return float("nan"), float("nan")
    window = vals[-k:] if len(vals) >= k else vals
    n = len(window)
    m = float(np.mean(window))
    sd = float(np.std(window, ddof=1)) if n > 1 else 0.0
    return m, sd / np.sqrt(n)


def windowed_z(vals, frac=0.25):
    """Compare last `frac` of `vals` against first `frac`. Returns
    (delta, pooled_se, z, n_each). z is `delta / pooled_se`.
    """
    if len(vals) < 8:
        return None
    q = max(2, len(vals) // int(1 / frac))
    early = np.array(vals[:q])
    late = np.array(vals[-q:])
    m_e, m_l = float(np.mean(early)), float(np.mean(late))
    se_e = float(np.std(early, ddof=1)) / np.sqrt(len(early)) if len(early) > 1 else 0.0
    se_l = float(np.std(late, ddof=1)) / np.sqrt(len(late)) if len(late) > 1 else 0.0
    pooled_se = float(np.sqrt(se_e ** 2 + se_l ** 2))
    delta = m_l - m_e
    z = delta / pooled_se if pooled_se > 0 else 0.0
    return delta, pooled_se, z, len(early)


def print_section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def main():
    # Support: monitor_training.py [tb_dir] [--as-of "YYYY-MM-DD HH:MM"]
    args = sys.argv[1:]
    as_of = None
    if "--as-of" in args:
        i = args.index("--as-of")
        as_of = args[i + 1]
        args = args[:i] + args[i + 2:]
    tb_dir = args[0] if args else find_latest_tb_dir()
    print(f"Reading: {tb_dir}")
    reader = SummaryReader(tb_dir, extra_columns={"wall_time"})
    df = reader.scalars
    if as_of is not None:
        import datetime as _dt
        cutoff = _dt.datetime.strptime(as_of, "%Y-%m-%d %H:%M").timestamp()
        df = df[df["wall_time"] <= cutoff].reset_index(drop=True)
        print(f"Filtered to wall_time <= {as_of} ({len(df)} scalar events).")
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

    # ── Reward evolution: z-scored first-quarter vs last-quarter ──────────
    print_section("REWARD TREND (last-quarter vs first-quarter, |z|>2 = significant)")
    print("  Δ = late_mean - early_mean. z = Δ / pooled_se. Flagged: ↑ z>2, ↓ z<-2, ~ otherwise.")
    for pers in PERSONALITIES:
        signif, mild = [], []
        for rtype in REWARD_TYPES:
            vals = get(df, f"reward_per_step/{pers}/{rtype}")
            r = windowed_z(vals)
            if r is None:
                continue
            delta, _se, z, _n = r
            tag = "↑" if z > 2 else "↓" if z < -2 else "~"
            entry = f"{rtype}{tag}{delta:+.2g}(z={z:+.1f})"
            (signif if abs(z) > 2 else mild).append(entry)
        if signif:
            print(f"  {pers:>10}  significant: {', '.join(signif)}")
        if mild:
            print(f"  {pers:>10}  noise:       {', '.join(mild)}")

    # ── Training health (last-10 mean ± stderr; also raw last-5) ──────────
    print_section("TRAINING HEALTH (mean±se over last 10, raw last 5)")
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
            m, se = mean_se(vals, 10)
            last5 = vals[-5:]
            print(
                f"  {label:25s}  {m:.4f}±{se:.4f}  "
                f"[{', '.join(f'{v:.4f}' for v in last5)}]"
            )

    # ── Policy concentration + BC agreement ──────────────────────────────
    # max_prob: mean of max(softmax) per action head, averaged over last cycles.
    #   ≈ 1/K uniform (entropy bonus dominant); ≈ 0.5 bimodal (PPO/BC conflict);
    #   → 1.0 concentrated (policy converged).
    # bc_agreement: fraction of steps where sampled action == BC expert action.
    #   ≈ 1/K random; → 1.0 policy tracks expert.
    head_names = ["turn", "thrust", "fire_primary", "fire_secondary", "nav", "wep"]
    head_k = [3, 2, 2, 2, 13, 13]  # num classes per head (for "uniform" baseline)
    if any(len(get(df, f"policy/max_prob/{h}")) > 0 for h in head_names):
        print_section("POLICY CONCENTRATION & BC AGREEMENT (last-10 mean±se)")
        print(f"  {'head':>16}  {'max_prob':>16}  {'bc_agreement':>16}   (interp per head)")
        for h, k in zip(head_names, head_k):
            mp = get(df, f"policy/max_prob/{h}")
            ag = get(df, f"policy/bc_agreement/{h}")
            if len(mp) < 2 or len(ag) < 2:
                continue
            m_mp, se_mp = mean_se(mp, 10)
            m_ag, se_ag = mean_se(ag, 10)
            # Flag interpretation per head.
            uniform = 1.0 / k
            flag = ""
            if m_mp < uniform + 0.1:
                flag = " ⚠ near-uniform"
            elif 0.35 < m_mp < 0.65 and k > 2:
                flag = " ⚠ bimodal-ish"
            elif m_mp > 0.85:
                flag = " concentrated"
            print(
                f"  {h:>16}  {m_mp:.3f}±{se_mp:.3f}        {m_ag:.3f}±{se_ag:.3f}"
                f"   (uniform={uniform:.2f}){flag}"
            )

    # ── Per-head value diagnostics ────────────────────────────────────────
    print_section("VALUE HEAD DIAGNOSTICS (last 5)")
    header_ev = f"{'':>16} {'expl_var':>40}  {'mean_abs_td':>40}"
    print(header_ev)
    print("-" * len(header_ev))
    for rtype in REWARD_TYPES:
        ev_vals = get(df, f"value_head/{rtype}/explained_variance")
        td_vals = get(df, f"value_head/{rtype}/mean_abs_td_error")
        ev_str = f"[{', '.join(f'{v:.3f}' for v in ev_vals[-5:])}]" if len(ev_vals) >= 5 else "n/a"
        td_str = f"[{', '.join(f'{v:.4f}' for v in td_vals[-5:])}]" if len(td_vals) >= 5 else "n/a"
        print(f"  {rtype:>14}: {ev_str:>40}  {td_str:>40}")
    vals = get(df, "train/explained_variance")
    if len(vals) >= 5:
        print(f"  {'TOTAL':>14}: [{', '.join(f'{v:.3f}' for v in vals[-5:])}]")

    # ── Raw reward totals per cycle (recent) ──────────────────────────────
    print_section("RAW REWARD TOTAL PER CYCLE (last 10 avg)")
    header_raw = f"{'':>12}" + "".join(f"{r:>12}" for r in REWARD_TYPES)
    print(header_raw)
    print("-" * len(header_raw))
    for pers in PERSONALITIES:
        row = []
        for rtype in REWARD_TYPES:
            vals = get(df, f"reward_total/{pers}/{rtype}")
            row.append(float(np.mean(vals[-10:])) if len(vals) >= 10 else 0.0)
        cells = "".join(f"{v:12.2f}" for v in row)
        print(f"{pers:>12}{cells}")

    # ── Target selection matrices ───────────────────────────────────────
    def print_target_matrix(title, prefix, type_names):
        has_data = any(
            len(get(df, f"{prefix}/{PERSONALITIES[0]}/{tt}")) > 0
            for tt in type_names
        )
        if not has_data:
            return
        print_section(title)
        header = f"{'':>12}" + "".join(f"{t:>14}" for t in type_names)
        print(header)
        print("-" * len(header))
        for pers in PERSONALITIES:
            row = []
            for tt in type_names:
                vals = get(df, f"{prefix}/{pers}/{tt}")
                row.append(float(np.mean(vals[-5:])) if len(vals) >= 5 else 0.0)
            cells = "".join(f"{v:14.3f}" for v in row)
            print(f"{pers:>12}{cells}")

    print_target_matrix(
        "NAV TARGET SELECTION (fraction, last 5 avg)",
        "nav_target_type",
        ["ship", "asteroid", "planet", "pickup", "none"],
    )
    print_target_matrix(
        "WEAPONS TARGET SELECTION (fraction, last 5 avg)",
        "wep_target_type",
        ["ship_engage", "ship_hostile", "ship_neutral", "asteroid", "none"],
    )
    print_target_matrix(
        "MEAN ENTITY SLOT COUNTS (per step, last 5 avg)",
        "slot_counts",
        ["ship", "asteroid", "planet", "pickup"],
    )

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
