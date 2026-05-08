#!/usr/bin/env python3
# =============================================================
# plot_results.py
#
# 2026 KIIE conference figure generator.
#
# Produces:
#   Figure 4 — solver_time boxplot, 2 scenarios × 5 modes
#   Figure 5 — 6-joint acceleration profile of a representative trial,
#              overlaid for modes A/B/C
#
# Both figures are saved as both PDF and PNG (200 dpi). figsize is tuned
# for the KIIE two-column layout (~3.5 in column width).
#
# Inputs:
#   cumulative.csv   — runner output (Figure 4)
#   --traj-dir DIR   — directory with per-trial JointTrajectory recordings
#                       written by experiment_runner.py
#                       (Figure 5; skipped if missing)
#
# Usage:
#   python3 plot_results.py cumulative.csv --out-dir figs/
#   python3 plot_results.py cumulative.csv --out-dir figs/ \
#       --traj-dir ~/.ros/pick_place_exp/trajectories
# =============================================================

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

DEFAULT_MODES = ("A", "B", "C", "C1", "C2")
SCENARIOS = ("S1", "S2")

# KIIE two-column page width in inches. Single column ≈ 3.5 in.
SINGLE_COL_INCH = 3.5
DOUBLE_COL_INCH = 7.16

MODE_COLORS = {
    "A":  "#1f77b4",
    "B":  "#ff7f0e",
    "C":  "#2ca02c",
    "C1": "#9467bd",
    "C2": "#8c564b",
}

JOINT_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1",      "wrist_2",       "wrist_3",
)


def save_fig(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    pdf = out_base.with_suffix(".pdf")
    png = out_base.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    print(f"[plot] wrote {pdf}")
    print(f"[plot] wrote {png}")


# ---------------------------------------------------------------------------
# Figure 4 — solver-time boxplot
# ---------------------------------------------------------------------------

def figure4_solver_time(df: pd.DataFrame, out_dir: Path,
                        modes: Tuple[str, ...]) -> None:
    if "total_compute_sec" not in df.columns:
        print("[plot] figure 4: total_compute_sec missing — skipped")
        return
    df = df[df["success"].astype(int) == 1].copy()
    df["solver_time_ms"] = df["total_compute_sec"].astype(float) * 1000.0

    scenarios_present = [s for s in SCENARIOS if s in df["scenario"].unique()]
    modes_present = [m for m in modes if m in df["mode"].unique()]
    if not scenarios_present or not modes_present:
        print("[plot] figure 4: no data — skipped")
        return

    fig, axes = plt.subplots(
        1, len(scenarios_present),
        figsize=(DOUBLE_COL_INCH, 2.6),
        sharey=True,
    )
    if len(scenarios_present) == 1:
        axes = [axes]

    for ax, scen in zip(axes, scenarios_present):
        scen_df = df[df["scenario"] == scen]
        data = [scen_df[scen_df["mode"] == m]["solver_time_ms"].dropna().to_numpy()
                for m in modes_present]
        bp = ax.boxplot(
            data, patch_artist=True,
            tick_labels=list(modes_present),
            widths=0.6, showfliers=True,
            medianprops=dict(color="black", linewidth=1.2),
        )
        for patch, m in zip(bp["boxes"], modes_present):
            patch.set_facecolor(MODE_COLORS.get(m, "gray"))
            patch.set_alpha(0.65)
        ax.set_title(f"Scenario {scen}", fontsize=10)
        ax.set_xlabel("Mode", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.tick_params(axis="both", labelsize=8)
    axes[0].set_ylabel("Solver time [ms]", fontsize=9)

    fig.suptitle("Total compute time per mode (successful trials)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir / "figure4_solver_time")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — 6-joint acceleration profile (representative trial, A/B/C)
# ---------------------------------------------------------------------------

def _load_traj_csv(path: Path) -> Optional[pd.DataFrame]:
    """Trajectory CSV columns: time, p1..p6, v1..v6 (optional), a1..a6 (optional)."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[plot] failed to load {path}: {exc}")
        return None
    if "time" not in df.columns:
        print(f"[plot] {path}: missing 'time' column")
        return None
    return df


def _accel_profile(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return (time, accel[N,6]). Use stored accelerations if present;
    otherwise finite-diff positions twice."""
    t = df["time"].to_numpy(dtype=float)
    accel_cols = [f"a{i}" for i in range(1, 7)]
    if all(c in df.columns for c in accel_cols):
        a = df[accel_cols].to_numpy(dtype=float)
        return t, a
    pos_cols = [f"p{i}" for i in range(1, 7)]
    if not all(c in df.columns for c in pos_cols):
        return t, np.zeros((len(t), 6))
    p = df[pos_cols].to_numpy(dtype=float)
    # central-diff velocity
    v = np.zeros_like(p)
    if len(t) > 2:
        v[1:-1] = (p[2:] - p[:-2]) / (t[2:] - t[:-2]).reshape(-1, 1)
    a = np.zeros_like(p)
    if len(t) > 4:
        a[2:-2] = (v[3:-1] - v[1:-3]) / (t[3:-1] - t[1:-3]).reshape(-1, 1)
    return t, a


def _pick_representative(df: pd.DataFrame, mode: str,
                         scenario: str, step: Optional[str]) -> Optional[pd.Series]:
    sub = df[(df["mode"] == mode) & (df["scenario"] == scenario) &
             (df["success"].astype(int) == 1)]
    if step and "step_name" in sub.columns:
        sub = sub[sub["step_name"] == step]
    if len(sub) == 0:
        return None
    if "total_compute_sec" not in sub.columns:
        return sub.iloc[0]
    median_val = sub["total_compute_sec"].median()
    idx = (sub["total_compute_sec"] - median_val).abs().idxmin()
    return sub.loc[idx]


def figure5_accel_profile(df: pd.DataFrame, traj_dir: Optional[Path],
                          out_dir: Path,
                          scenario: str = "S1",
                          modes: Tuple[str, ...] = ("A", "B", "C"),
                          step: Optional[str] = None) -> None:
    if traj_dir is None or not traj_dir.exists():
        print("[plot] figure 5: --traj-dir missing or empty — skipped")
        return

    fig, axes = plt.subplots(
        2, 3,
        figsize=(DOUBLE_COL_INCH, 4.2),
        sharex=True,
    )
    axes = axes.flatten()
    plotted_any = False

    for mode in modes:
        rep = _pick_representative(df, mode, scenario, step)
        if rep is None:
            print(f"[plot] figure 5: no successful {scenario}/{mode} trial — skipping")
            continue
        run_id    = rep["run_id"]
        trial_idx = int(rep["trial_idx"])
        step_name = rep.get("step_name", step or "")
        # File naming convention from runner:
        # <run_id>_<scenario>_<mode>_<trial_idx>_<step>.csv
        candidate = traj_dir / f"{run_id}_{scenario}_{mode}_{trial_idx}_{step_name}.csv"
        if not candidate.exists():
            # Fallback: any file matching prefix
            matches = sorted(traj_dir.glob(
                f"{run_id}_{scenario}_{mode}_{trial_idx}*.csv"))
            if not matches:
                print(f"[plot] figure 5: no trajectory file for "
                      f"{run_id}/{scenario}/{mode}/{trial_idx}")
                continue
            candidate = matches[0]
        traj_df = _load_traj_csv(candidate)
        if traj_df is None:
            continue
        t, a = _accel_profile(traj_df)
        for j in range(6):
            axes[j].plot(t, a[:, j],
                         label=f"Mode {mode}",
                         color=MODE_COLORS.get(mode, "gray"),
                         linewidth=1.2)
            axes[j].set_title(JOINT_NAMES[j], fontsize=9)
            axes[j].grid(True, linestyle="--", alpha=0.4)
            axes[j].tick_params(axis="both", labelsize=7)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print("[plot] figure 5: no overlay produced")
        return

    for j in range(6):
        axes[j].set_xlabel("time [s]", fontsize=8)
        axes[j].set_ylabel(r"$\ddot q$ [rad/s$^2$]", fontsize=8)
    axes[0].legend(fontsize=7, loc="best", framealpha=0.85)
    fig.suptitle(f"Joint acceleration profiles — Scenario {scenario}, modes A/B/C",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir / "figure5_accel_profile")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="KIIE 2026 figure generator")
    p.add_argument("csv", help="cumulative.csv from experiment_runner.py")
    p.add_argument("--out-dir", default="figs",
                   help="figure output directory (default: ./figs)")
    p.add_argument("--traj-dir", default=None,
                   help="directory with per-trial JointTrajectory recordings "
                        "(Figure 5)")
    p.add_argument("--mode-order", default=",".join(DEFAULT_MODES),
                   help="comma-separated mode order")
    p.add_argument("--fig5-scenario", default="S1",
                   help="scenario for Figure 5 (default: S1)")
    p.add_argument("--fig5-modes", default="A,B,C",
                   help="comma-separated mode list overlaid in Figure 5")
    p.add_argument("--fig5-step", default=None,
                   help="restrict Figure 5 trial selection to a step_name")
    args = p.parse_args(argv)

    in_path = Path(args.csv).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    df = pd.read_csv(in_path)
    print(f"[plot] loaded {len(df)} rows from {in_path}")

    modes = tuple(m.strip() for m in args.mode_order.split(",") if m.strip())
    fig5_modes = tuple(m.strip() for m in args.fig5_modes.split(",") if m.strip())
    traj_dir = Path(args.traj_dir).expanduser() if args.traj_dir else None

    figure4_solver_time(df, out_dir, modes)
    figure5_accel_profile(
        df, traj_dir, out_dir,
        scenario=args.fig5_scenario,
        modes=fig5_modes,
        step=args.fig5_step,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
