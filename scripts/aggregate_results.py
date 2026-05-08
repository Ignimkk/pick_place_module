#!/usr/bin/env python3
# =============================================================
# aggregate_results.py
#
# 2026 KIIE conference table generator.
#
# Input:
#   cumulative.csv produced by experiment_runner.py — must contain
#   (run_id, scenario, mode, trial_idx, seed, pick_*, place_*)
#   plus the 31-column ExperimentRecord columns from pick_place_node.
#
# Outputs (next to the input CSV unless --out-dir is given):
#   table4_main.csv         — main mode comparison
#   table4_main.tex         — LaTeX (booktabs) of Table 4
#   table5_stats.csv        — paired Mann-Whitney U: B vs C, C vs C1, C vs C2
#   table5_stats.tex        — LaTeX of Table 5
#
# Usage:
#   python3 aggregate_results.py cumulative.csv
#   python3 aggregate_results.py cumulative.csv --out-dir tables/ --step pre_grasp
# =============================================================

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Column order in cumulative.csv:
#  prefix:  run_id, scenario, mode, trial_idx, seed,
#           pick_x..z, place_x..z
#  body  :  trial_id, step_name, experiment_mode, success, fallback_used,
#           ik_time_sec, rrt_planning_sec, shortcut_time_sec,
#           initial_guess_time_sec, solve_time_sec, total_compute_sec, ...

DEFAULT_MODE_ORDER = ("A", "B", "C", "C1", "C2")

# Metrics → (csv column, label, unit, decimals)
METRICS: List[Tuple[str, str, str, int]] = [
    ("total_compute_sec",        "Solver time",          "s",        3),
    ("rrt_planning_sec",         "RRT plan time",        "s",        3),
    ("solve_time_sec",           "NLP solve time",       "s",        3),
    ("trajectory_duration_sec",  "Traj duration",        "s",        3),
    ("joint_path_length",        "Joint path length",    "rad",      3),
    ("max_joint_velocity",       "Max joint vel",        "rad/s",    3),
    ("max_joint_acceleration",   "Max joint accel",      "rad/s$^2$",3),
    ("max_joint_jerk",           "Max joint jerk",       "rad/s$^3$",3),
    ("mean_torque",              "Mean torque",          "N$\\cdot$m",3),
    ("max_torque",               "Max torque",           "N$\\cdot$m",3),
    ("max_torque_rate",          "Max torque rate",      "N$\\cdot$m/s",3),
    ("max_constraint_violation", "Max constraint viol.", "",         5),
    ("final_cost",               "Final cost $J$",       "",         4),
]

# Statistical comparisons for Table 5
COMPARISONS = [
    ("B",  "C",  "B vs C  (TrajOpt-only vs RRT+TrajOpt)"),
    ("C",  "C1", "C vs C1 (reduced vs full SLSQP)"),
    ("C",  "C2", "C vs C2 (free-T vs fixed-T)"),
]

# Solver-time column for Figure 4 / Table 4
SOLVER_TIME_COL = "total_compute_sec"


# ---------------------------------------------------------------------------
# Loading + filtering
# ---------------------------------------------------------------------------

def load_cumulative(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"scenario", "mode", "success"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[aggregate] missing required columns: {missing}")
    return df


def filter_df(df: pd.DataFrame,
              step: Optional[str],
              successful_only: bool) -> pd.DataFrame:
    out = df
    if step and "step_name" in out.columns:
        out = out[out["step_name"] == step]
    if successful_only and "success" in out.columns:
        out = out[out["success"].astype(int) == 1]
    return out.copy()


# ---------------------------------------------------------------------------
# Table 4: main comparison
# ---------------------------------------------------------------------------

def fmt_mean_std(series: pd.Series, decimals: int) -> str:
    s = series.dropna()
    if len(s) == 0:
        return "—"
    return f"{s.mean():.{decimals}f} $\\pm$ {s.std():.{decimals}f}"


def build_table4(df: pd.DataFrame,
                 mode_order: Iterable[str]) -> pd.DataFrame:
    rows = []
    scenarios = sorted(df["scenario"].dropna().unique())
    modes_present = [m for m in mode_order if m in df["mode"].unique()]

    for scen in scenarios:
        scen_df = df[df["scenario"] == scen]
        # Success rate row
        row = {"scenario": scen, "metric": "Success rate", "unit": ""}
        for m in modes_present:
            sub = scen_df[scen_df["mode"] == m]
            if len(sub) == 0:
                row[m] = "—"
            else:
                rate = sub["success"].astype(int).mean()
                n_succ = int(sub["success"].astype(int).sum())
                row[m] = f"{n_succ}/{len(sub)} ({rate:.1%})"
        rows.append(row)
        # Per-metric rows (success-only subset)
        succ_df = scen_df[scen_df["success"].astype(int) == 1]
        for col, label, unit, decimals in METRICS:
            if col not in df.columns:
                continue
            row = {"scenario": scen, "metric": label, "unit": unit}
            for m in modes_present:
                sub = succ_df[succ_df["mode"] == m]
                row[m] = fmt_mean_std(sub[col], decimals)
            rows.append(row)
    return pd.DataFrame(rows, columns=["scenario", "metric", "unit", *modes_present])


def table_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    # Tabular spec: scenario, metric, unit, then modes — left + center
    align = "l l l " + " ".join(["c"] * (len(cols) - 3))
    out = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append("\\small")
    out.append(f"\\begin{{tabular}}{{{align}}}")
    out.append("\\toprule")
    header = " & ".join(c.replace("_", "\\_") for c in cols) + " \\\\"
    out.append(header)
    out.append("\\midrule")
    last_scen = None
    for _, r in df.iterrows():
        scen = r["scenario"]
        if last_scen is not None and scen != last_scen:
            out.append("\\midrule")
        last_scen = scen
        cells = [str(r[c]) for c in cols]
        out.append(" & ".join(cells) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Table 5: paired statistical tests
# ---------------------------------------------------------------------------

def paired_align(df: pd.DataFrame, mode_a: str, mode_b: str,
                 col: str) -> Tuple[np.ndarray, np.ndarray]:
    """Pair rows by (scenario, trial_idx, step_name) so identical seeds
    are compared. Returns aligned (a_values, b_values)."""
    keys = ["scenario", "trial_idx"]
    if "step_name" in df.columns:
        keys.append("step_name")
    a = df[df["mode"] == mode_a].set_index(keys)[col]
    b = df[df["mode"] == mode_b].set_index(keys)[col]
    common = a.index.intersection(b.index)
    return a.loc[common].to_numpy(), b.loc[common].to_numpy()


def build_table5(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    test_metrics = [
        ("total_compute_sec",        "Solver time"),
        ("trajectory_duration_sec",  "Traj duration"),
        ("max_joint_acceleration",   "Max joint accel"),
        ("max_joint_jerk",           "Max joint jerk"),
        ("max_torque",               "Max torque"),
        ("final_cost",               "Final cost"),
    ]
    for mode_a, mode_b, label in COMPARISONS:
        for col, metric_label in test_metrics:
            if col not in df.columns:
                continue
            a_vals, b_vals = paired_align(
                df[df["success"].astype(int) == 1], mode_a, mode_b, col
            )
            n = min(len(a_vals), len(b_vals))
            if n < 3:
                rows.append({
                    "comparison": label, "metric": metric_label, "n": n,
                    "median_a": float("nan"), "median_b": float("nan"),
                    "U": float("nan"), "p_value": float("nan"),
                })
                continue
            if HAS_SCIPY:
                u_stat, p = mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            else:
                u_stat, p = float("nan"), float("nan")
            rows.append({
                "comparison": label,
                "metric": metric_label,
                "n": n,
                "median_a": float(np.median(a_vals)),
                "median_b": float(np.median(b_vals)),
                "U": float(u_stat),
                "p_value": float(p),
            })
    return pd.DataFrame(rows)


def table5_to_latex(df: pd.DataFrame) -> str:
    out = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Paired Mann--Whitney $U$ tests across modes (per-trial-seed pairs).}",
        "\\label{tab:stats}",
        "\\small",
        "\\begin{tabular}{l l c r r r r}",
        "\\toprule",
        "Comparison & Metric & $n$ & median$_a$ & median$_b$ & $U$ & $p$ \\\\",
        "\\midrule",
    ]
    last_cmp = None
    for _, r in df.iterrows():
        if last_cmp is not None and r["comparison"] != last_cmp:
            out.append("\\midrule")
        last_cmp = r["comparison"]
        cmp_cell = r["comparison"].replace("_", "\\_") if r["comparison"] != last_cmp else ""
        cmp_cell = r["comparison"].replace("_", "\\_")
        med_a = "—" if math.isnan(r["median_a"]) else f"{r['median_a']:.3f}"
        med_b = "—" if math.isnan(r["median_b"]) else f"{r['median_b']:.3f}"
        u_cell = "—" if math.isnan(r["U"]) else f"{r['U']:.1f}"
        if math.isnan(r["p_value"]):
            p_cell = "—"
        else:
            p = r["p_value"]
            p_cell = f"{p:.4f}" if p >= 1e-4 else f"{p:.2e}"
            if p < 0.05:
                p_cell = f"\\textbf{{{p_cell}}}"
        out.append(
            f"{cmp_cell} & {r['metric']} & {int(r['n'])} & "
            f"{med_a} & {med_b} & {u_cell} & {p_cell} \\\\"
        )
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Aggregate KIIE 2026 experiment CSV")
    p.add_argument("csv", help="cumulative.csv from experiment_runner.py")
    p.add_argument("--out-dir", default=None,
                   help="output directory (default: same as input)")
    p.add_argument("--step", default=None,
                   help="filter step_name (e.g. pre_grasp / pre_place / approach / retreat)")
    p.add_argument("--mode-order", default=",".join(DEFAULT_MODE_ORDER),
                   help="comma-separated mode order")
    args = p.parse_args(argv)

    in_path = Path(args.csv).expanduser()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_cumulative(in_path)
    print(f"[aggregate] loaded {len(df)} rows from {in_path}")
    if args.step:
        df = filter_df(df, step=args.step, successful_only=False)
        print(f"[aggregate] after step={args.step!r}: {len(df)} rows")

    mode_order = [m.strip() for m in args.mode_order.split(",") if m.strip()]

    # Table 4 — built from filtered df (handles success-rate + mean±std internally)
    t4 = build_table4(df, mode_order)
    t4_csv = out_dir / "table4_main.csv"
    t4_tex = out_dir / "table4_main.tex"
    t4.to_csv(t4_csv, index=False)
    t4_tex.write_text(table_to_latex(
        t4,
        caption=("Main comparison across scenarios (S1 obstacle-free, "
                 "S2 with obstacle) and modes A/B/C/C1/C2. "
                 "Mean $\\pm$ std over successful trials."),
        label="tab:main",
    ))
    print(f"[aggregate] wrote {t4_csv}")
    print(f"[aggregate] wrote {t4_tex}")

    # Table 5 — paired stats
    if not HAS_SCIPY:
        print("[aggregate] WARNING: scipy not available — Table 5 will have NaN p-values")
    t5 = build_table5(df)
    t5_csv = out_dir / "table5_stats.csv"
    t5_tex = out_dir / "table5_stats.tex"
    t5.to_csv(t5_csv, index=False)
    t5_tex.write_text(table5_to_latex(t5))
    print(f"[aggregate] wrote {t5_csv}")
    print(f"[aggregate] wrote {t5_tex}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
