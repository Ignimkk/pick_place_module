#!/usr/bin/env python3
"""
analyze_results.py — Experiment comparison analysis for pick-place trials.

Usage:
    python3 analyze_results.py <results.csv>
    python3 analyze_results.py <results.csv> --plot
    python3 analyze_results.py ~/.ros/pick_place_exp/  # loads all CSVs in dir

Columns expected in CSV (31-column ExperimentRecord layout):
    trial_id, timestamp, experiment_mode, step_name, success,
    ik_time_sec, rrt_planning_sec, trajopt_total_sec,
    shortcut_time_sec, guess_time_sec, solve_time_sec,
    num_rrt_waypoints, num_shortcut_waypoints, num_opt_points,
    traj_num_points, traj_duration_sec, traj_joint_path_length,
    traj_mean_vel, traj_max_vel, traj_mean_accel, traj_max_accel,
    traj_mean_jerk, traj_max_jerk,
    mean_torque, max_torque, mean_torque_rate, max_torque_rate,
    final_cost, max_constraint_violation,
    t_opt_sec, fallback_used
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir():
        csvs = sorted(p.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in {p}")
        print(f"Loading {len(csvs)} CSV file(s) from {p}")
        return pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    return pd.read_csv(p)


def _fmt(v, decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "  N/A  "
    return f"{v:>{decimals + 5}.{decimals}f}"


def _stats(series: pd.Series) -> dict:
    clean = series.dropna()
    if len(clean) == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {
        "n":    len(clean),
        "mean": float(clean.mean()),
        "std":  float(clean.std()),
        "min":  float(clean.min()),
        "max":  float(clean.max()),
    }


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

MODES = ("rrt_only", "trajopt_only", "rrt_trajopt")

METRIC_DEFS = [
    # (column, label, unit)
    ("traj_duration_sec",       "Traj duration",      "s"),
    ("traj_joint_path_length",  "Joint path length",  "rad"),
    ("traj_mean_vel",           "Mean joint vel",     "rad/s"),
    ("traj_max_vel",            "Max  joint vel",     "rad/s"),
    ("traj_mean_accel",         "Mean joint accel",   "rad/s²"),
    ("traj_max_accel",          "Max  joint accel",   "rad/s²"),
    ("traj_mean_jerk",          "Mean joint jerk",    "rad/s³"),
    ("traj_max_jerk",           "Max  joint jerk",    "rad/s³"),
    ("mean_torque",             "Mean |τ|",           "N·m"),
    ("max_torque",              "Max  |τ|",           "N·m"),
    ("mean_torque_rate",        "Mean |dτ/dt|",       "N·m/s"),
    ("max_torque_rate",         "Max  |dτ/dt|",       "N·m/s"),
    ("final_cost",              "TrajOpt cost J",     ""),
    ("max_constraint_violation","Max constraint viol",""),
    ("t_opt_sec",               "T_opt",              "s"),
    ("ik_time_sec",             "IK time",            "s"),
    ("rrt_planning_sec",        "RRT plan time",      "s"),
    ("trajopt_total_sec",       "TrajOpt total time", "s"),
    ("solve_time_sec",          "NLP solve time",     "s"),
]


def print_summary(df: pd.DataFrame) -> None:
    modes_present = [m for m in MODES if m in df["experiment_mode"].unique()]

    # ── Success rates ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUCCESS RATES")
    print("=" * 72)
    print(f"  {'Mode':<20}  {'Success':>8}  {'Total':>6}  {'Rate':>7}")
    print("-" * 72)
    for mode in modes_present:
        sub = df[df["experiment_mode"] == mode]
        n_total   = len(sub)
        n_success = int(sub["success"].sum()) if "success" in sub.columns else 0
        rate = n_success / n_total if n_total > 0 else float("nan")
        print(f"  {mode:<20}  {n_success:>8}  {n_total:>6}  {rate:>6.1%}")

    # ── Per-metric comparison ─────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  METRIC COMPARISON  (mean ± std)")
    print("=" * 72)
    header = f"  {'Metric':<28}  {'Unit':<8}" + "".join(
        f"  {m:<22}" for m in modes_present
    )
    print(header)
    print("-" * len(header))

    for col, label, unit in METRIC_DEFS:
        if col not in df.columns:
            continue
        row = f"  {label:<28}  {unit:<8}"
        for mode in modes_present:
            sub = df[df["experiment_mode"] == mode]
            if col not in sub.columns:
                row += f"  {'—':<22}"
                continue
            s = _stats(sub[col])
            if s["n"] == 0:
                row += f"  {'—':<22}"
            else:
                row += f"  {s['mean']:>8.3f} ±{s['std']:>7.3f}  "
        print(row)

    # ── Per-step breakdown (rrt_trajopt only) ─────────────────────────────
    if "step_name" in df.columns and "rrt_trajopt" in modes_present:
        sub = df[df["experiment_mode"] == "rrt_trajopt"]
        steps = sorted(sub["step_name"].unique())
        if len(steps) > 1:
            print("\n" + "=" * 72)
            print("  STEP BREAKDOWN (rrt_trajopt mode)")
            print("=" * 72)
            print(f"  {'Step':<20}  {'N':>4}  {'Duration mean':>14}  "
                  f"{'TrajOpt time':>14}  {'J mean':>10}")
            print("-" * 72)
            for step in steps:
                ss = sub[sub["step_name"] == step]
                n  = len(ss)
                dur = _stats(ss.get("traj_duration_sec", pd.Series(dtype=float)))
                tot = _stats(ss.get("trajopt_total_sec", pd.Series(dtype=float)))
                j   = _stats(ss.get("final_cost", pd.Series(dtype=float)))
                print(f"  {step:<20}  {n:>4}  {_fmt(dur['mean']):>14}  "
                      f"{_fmt(tot['mean']):>14}  {_fmt(j['mean']):>10}")

    # ── Fallback stats (rrt_trajopt) ──────────────────────────────────────
    if "fallback_used" in df.columns and "rrt_trajopt" in modes_present:
        sub = df[df["experiment_mode"] == "rrt_trajopt"]
        n_fb = int(sub["fallback_used"].sum())
        n_total = len(sub)
        if n_total > 0:
            print(f"\n  [rrt_trajopt] Fallback to MoveIt2 execute: "
                  f"{n_fb}/{n_total} ({n_fb/n_total:.1%})")

    print("=" * 72)


# ---------------------------------------------------------------------------
# Optional plotting
# ---------------------------------------------------------------------------

def plot_comparison(df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    modes_present = [m for m in MODES if m in df["experiment_mode"].unique()]
    colors = {"rrt_only": "steelblue", "trajopt_only": "darkorange",
              "rrt_trajopt": "green"}

    plot_metrics = [
        ("traj_duration_sec",    "Trajectory Duration [s]"),
        ("traj_joint_path_length","Joint Path Length [rad]"),
        ("traj_max_vel",         "Max Joint Velocity [rad/s]"),
        ("traj_max_accel",       "Max Joint Acceleration [rad/s²]"),
        ("mean_torque",          "Mean |τ| [N·m]"),
        ("final_cost",           "TrajOpt Cost J"),
    ]
    n_plots = sum(1 for col, _ in plot_metrics if col in df.columns)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()
    ax_idx = 0

    for col, ylabel in plot_metrics:
        if col not in df.columns:
            continue
        ax = axes_flat[ax_idx]
        ax_idx += 1
        data = [df[df["experiment_mode"] == m][col].dropna().values
                for m in modes_present]
        bp = ax.boxplot(data, patch_artist=True, labels=modes_present)
        for patch, mode in zip(bp["boxes"], modes_present):
            patch.set_facecolor(colors.get(mode, "gray"))
            patch.set_alpha(0.7)
        ax.set_title(ylabel, fontsize=10)
        ax.set_xticklabels(modes_present, rotation=15, fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(ax_idx, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle("Pick-Place Experiment Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse pick-place experiment CSV results."
    )
    parser.add_argument("path", help="CSV file or directory of CSV files")
    parser.add_argument("--plot", action="store_true",
                        help="Show box plots (requires matplotlib)")
    parser.add_argument("--step", default=None,
                        help="Filter to a specific step_name")
    parser.add_argument("--mode", default=None,
                        help="Filter to a specific experiment_mode")
    args = parser.parse_args()

    df = _load(args.path)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    if args.step:
        df = df[df["step_name"] == args.step]
        print(f"Filtered to step '{args.step}': {len(df)} rows")
    if args.mode:
        df = df[df["experiment_mode"] == args.mode]
        print(f"Filtered to mode '{args.mode}': {len(df)} rows")

    if df.empty:
        print("No data after filtering — nothing to report.")
        sys.exit(0)

    print_summary(df)

    if args.plot:
        plot_comparison(df)


if __name__ == "__main__":
    main()
