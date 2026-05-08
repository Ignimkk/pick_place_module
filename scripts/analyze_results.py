#!/usr/bin/env python3
# =============================================================
# analyze_results.py
#
# 6개 실험 CSV 를 분석하여 (mode × has_obstacle) 비교 테이블과
# 박스플롯을 출력한다.
#
# 사용법:
#   python3 analyze_results.py results/              # 디렉터리 내 전체 CSV 합산
#   python3 analyze_results.py results/ --plot        # 그래프 포함
#   python3 analyze_results.py results/01_rrt_only_no_obs.csv  # 단일 파일
#   python3 analyze_results.py results/ --step pre_pick        # step 필터
#   python3 analyze_results.py results/ --mode rrt_only        # mode 필터
#   python3 analyze_results.py results/ --obstacle yes         # 장애물 필터
#
# 기대 컬럼 (trial_runner.py 출력 + 노드 CSV 컬럼):
#   [메타] run_id, timestamp, mode, has_obstacle, trial_idx, seed,
#          pick_x, pick_y, pick_z, place_x, place_y, place_z
#   [노드] trial_id, step_name, experiment_mode, success, fallback_used,
#          ik_time_sec, rrt_planning_sec, shortcut_time_sec,
#          initial_guess_time_sec, solve_time_sec, total_compute_sec,
#          exec_wait_sec, num_rrt_points, num_shortcut_waypoints,
#          num_optimized_points, trajectory_duration_sec,
#          joint_path_length, mean_joint_velocity, max_joint_velocity,
#          mean_joint_acceleration, max_joint_acceleration,
#          mean_joint_jerk, max_joint_jerk,
#          mean_torque, max_torque, mean_torque_rate, max_torque_rate,
#          max_constraint_violation, final_cost, solver_status, message
# =============================================================

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# =============================================================
# 상수
# =============================================================

MODES    = ("rrt_only", "trajopt_only", "rrt_trajopt")
OBSTACLES = ("no", "yes")   # has_obstacle 컬럼 값

# (컬럼명, 표시 레이블, 단위)
METRIC_DEFS: List[tuple] = [
    ("trajectory_duration_sec",   "Traj duration",        "s"),
    ("joint_path_length",         "Joint path length",    "rad"),
    ("mean_joint_velocity",       "Mean joint vel",       "rad/s"),
    ("max_joint_velocity",        "Max joint vel",        "rad/s"),
    ("mean_joint_acceleration",   "Mean joint accel",     "rad/s²"),
    ("max_joint_acceleration",    "Max joint accel",      "rad/s²"),
    ("mean_joint_jerk",           "Mean joint jerk",      "rad/s³"),
    ("max_joint_jerk",            "Max joint jerk",       "rad/s³"),
    ("mean_torque",               "Mean |τ|",             "N·m"),
    ("max_torque",                "Max |τ|",              "N·m"),
    ("mean_torque_rate",          "Mean |dτ/dt|",         "N·m/s"),
    ("max_torque_rate",           "Max |dτ/dt|",          "N·m/s"),
    ("final_cost",                "TrajOpt cost J",       ""),
    ("max_constraint_violation",  "Max constr viol",      ""),
    ("solve_time_sec",            "NLP solve time",       "s"),
    ("total_compute_sec",         "Total compute time",   "s"),
    ("ik_time_sec",               "IK time",              "s"),
    ("rrt_planning_sec",          "RRT plan time",        "s"),
    ("initial_guess_time_sec",    "Initial guess time",   "s"),
    ("shortcut_time_sec",         "Shortcut time",        "s"),
    ("exec_wait_sec",             "Exec wait time",       "s"),
    ("num_rrt_points",            "# RRT waypoints",      ""),
    ("num_shortcut_waypoints",    "# Shortcut waypoints", ""),
    ("num_optimized_points",      "# Opt waypoints",      ""),
]


# =============================================================
# 로드 헬퍼
# =============================================================

def _load(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir():
        csvs = sorted(p.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files in {p}")
        print(f"[load] {len(csvs)} file(s): {[f.name for f in csvs]}")
        frames = []
        for f in csvs:
            try:
                df = pd.read_csv(f)
                df["_source"] = f.name
                frames.append(df)
            except Exception as e:
                print(f"[warn] {f.name}: {e}")
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df = pd.read_csv(p)
    df["_source"] = Path(p).name
    return df


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼 타입 정규화 및 누락 파생 컬럼 보완."""
    # success 컬럼을 bool 로
    if "success" in df.columns:
        df["success"] = df["success"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
    # has_obstacle 컬럼 정규화
    if "has_obstacle" in df.columns:
        df["has_obstacle"] = df["has_obstacle"].astype(str).str.lower()
    # numeric 컬럼 강제 변환
    numeric_cols = [c for c, _, _ in METRIC_DEFS]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# =============================================================
# 통계 헬퍼
# =============================================================

def _stats(s: pd.Series) -> dict:
    clean = s.dropna()
    if len(clean) == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {
        "n":    int(len(clean)),
        "mean": float(clean.mean()),
        "std":  float(clean.std()),
        "min":  float(clean.min()),
        "max":  float(clean.max()),
    }


def _fmt(v: float, decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return " N/A "
    return f"{v:.{decimals}f}"


# =============================================================
# 보고서 출력
# =============================================================

def print_summary(df: pd.DataFrame) -> None:
    modes_present = [m for m in MODES if m in df["mode"].unique()] \
        if "mode" in df.columns else []
    obs_present   = sorted(df["has_obstacle"].unique()) \
        if "has_obstacle" in df.columns else []

    # ── 성공률 (mode × has_obstacle) ─────────────────────────────────
    print("\n" + "=" * 76)
    print("  성공률 (SUCCESS RATE)")
    print("=" * 76)
    header = f"  {'Mode':<22}  {'Obstacle':<10}  {'Success':>7}  {'Total':>6}  {'Rate':>7}"
    print(header)
    print("-" * 76)
    for mode in (modes_present or df["mode"].unique()):
        for obs in obs_present:
            cond = (df["mode"] == mode) & (df["has_obstacle"] == obs)
            sub  = df[cond]
            if sub.empty:
                continue
            n_total   = len(sub)
            n_success = int(sub["success"].sum()) if "success" in sub.columns else 0
            rate      = n_success / n_total if n_total > 0 else float("nan")
            print(f"  {mode:<22}  {obs:<10}  {n_success:>7}  {n_total:>6}  {rate:>6.1%}")

    # ── 메트릭 비교 테이블 ───────────────────────────────────────────
    groups = [(m, o) for m in (modes_present or df.get("mode", pd.Series()).unique())
                     for o in obs_present
                     if not df[(df["mode"] == m) & (df["has_obstacle"] == o)].empty]
    if not groups:
        return

    col_w = 22
    print("\n" + "=" * (36 + col_w * len(groups)))
    print("  메트릭 비교  (mean ± std)")
    print("=" * (36 + col_w * len(groups)))
    hdr = f"  {'Metric':<22}  {'Unit':<9}"
    for m, o in groups:
        lbl = f"{m[:14]}/{'obs' if o=='yes' else 'no_obs'}"
        hdr += f"  {lbl:<{col_w}}"
    print(hdr)
    print("-" * len(hdr))

    for col, label, unit in METRIC_DEFS:
        if col not in df.columns:
            continue
        row = f"  {label:<22}  {unit:<9}"
        any_data = False
        for m, o in groups:
            sub = df[(df["mode"] == m) & (df["has_obstacle"] == o)]
            s   = _stats(sub[col]) if col in sub.columns else {"n": 0}
            if s["n"] == 0:
                row += f"  {'—':<{col_w}}"
            else:
                any_data = True
                cell = f"{s['mean']:>8.3f}±{s['std']:>6.3f}"
                row += f"  {cell:<{col_w}}"
        if any_data:
            print(row)

    # ── step별 breakdown ─────────────────────────────────────────────
    if "step_name" in df.columns:
        steps = sorted(df["step_name"].dropna().unique())
        if len(steps) > 1:
            print("\n" + "=" * 76)
            print("  STEP BREAKDOWN  (mode × step  /  trajectory_duration_sec)")
            print("=" * 76)
            for mode in (modes_present or df["mode"].unique()):
                sub_m = df[df["mode"] == mode]
                if sub_m.empty:
                    continue
                print(f"\n  [{mode}]")
                print(f"  {'Step':<25}  {'N':>5}  {'Duration mean':>14}  {'Compute time':>13}")
                print("  " + "-" * 62)
                for step in steps:
                    ss  = sub_m[sub_m["step_name"] == step]
                    if ss.empty:
                        continue
                    dur = _stats(ss.get("trajectory_duration_sec",
                                        pd.Series(dtype=float)))
                    tot = _stats(ss.get("total_compute_sec",
                                        pd.Series(dtype=float)))
                    print(f"  {step:<25}  {dur['n']:>5}  "
                          f"{_fmt(dur['mean']):>14}  "
                          f"{_fmt(tot['mean']):>13}")

    # ── fallback 통계 ─────────────────────────────────────────────
    if "fallback_used" in df.columns:
        sub_rrt = df[df["mode"] == "rrt_trajopt"] if "mode" in df.columns else df
        if not sub_rrt.empty:
            n_fb = int(sub_rrt["fallback_used"].astype(str)
                                               .str.lower()
                                               .isin(["true","1","yes"]).sum())
            n_t  = len(sub_rrt)
            print(f"\n  [rrt_trajopt] MoveIt fallback: {n_fb}/{n_t} "
                  f"({n_fb/n_t:.1%})" if n_t else "")

    print("=" * 76)


# =============================================================
# 그래프
# =============================================================

def plot_comparison(df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib 없음 — 그래프 건너뜀")
        return

    modes_present = [m for m in MODES if m in df.get("mode", pd.Series()).unique()]
    obs_present   = sorted(df["has_obstacle"].unique()) \
        if "has_obstacle" in df.columns else ["no"]

    colors = {
        "rrt_only":     {"no": "#5B9BD5", "yes": "#1F4E79"},
        "trajopt_only": {"no": "#ED7D31", "yes": "#843C0C"},
        "rrt_trajopt":  {"no": "#70AD47", "yes": "#375623"},
    }

    plot_metrics = [
        ("trajectory_duration_sec",  "Trajectory Duration [s]"),
        ("joint_path_length",        "Joint Path Length [rad]"),
        ("max_joint_velocity",       "Max Joint Velocity [rad/s]"),
        ("max_joint_acceleration",   "Max Joint Acceleration [rad/s²]"),
        ("mean_torque",              "Mean |τ| [N·m]"),
        ("final_cost",               "TrajOpt Cost J"),
    ]

    avail = [(col, lbl) for col, lbl in plot_metrics if col in df.columns]
    if not avail:
        print("[warn] 그래프에 사용할 수치 컬럼이 없습니다.")
        return

    ncols = 3
    nrows = math.ceil(len(avail) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axes_flat = axes.flatten() if nrows * ncols > 1 else [axes]

    for ax_idx, (col, ylabel) in enumerate(avail):
        ax = axes_flat[ax_idx]
        data   = []
        labels = []
        clrs   = []
        for mode in modes_present:
            for obs in obs_present:
                sub = df[(df["mode"] == mode) & (df["has_obstacle"] == obs)]
                if sub.empty:
                    continue
                data.append(sub[col].dropna().values)
                labels.append(f"{mode[:8]}\n({'obs' if obs=='yes' else 'no'})")
                clrs.append(colors.get(mode, {}).get(obs, "gray"))

        if not data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(data, patch_artist=True, labels=labels)
        for patch, c in zip(bp["boxes"], clrs):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        ax.set_title(ylabel, fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    for i in range(len(avail), len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle("Pick-Place Experiment: Mode × Obstacle 비교",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results_comparison.png", dpi=150)
    print("[그래프] results_comparison.png 저장됨")
    plt.show()


# =============================================================
# 진입점
# =============================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="pick-place 실험 CSV 분석 (mode × obstacle 비교)"
    )
    p.add_argument("path",
                   help="CSV 파일 또는 결과 디렉터리")
    p.add_argument("--plot",     action="store_true",
                   help="박스플롯 표시 (matplotlib 필요)")
    p.add_argument("--step",     default=None,
                   help="step_name 필터 (예: pre_pick, pick, place, pre_place)")
    p.add_argument("--mode",     default=None,
                   help="mode 필터 (예: rrt_only)")
    p.add_argument("--obstacle", default=None, choices=["yes", "no"],
                   help="has_obstacle 필터")
    args = p.parse_args()

    df = _load(args.path)
    if df.empty:
        print("[오류] 데이터 없음")
        sys.exit(1)

    df = _normalise(df)
    print(f"[load] {len(df)} 행  /  컬럼: {list(df.columns)}")

    if args.step:
        df = df[df["step_name"] == args.step]
        print(f"[filter] step='{args.step}': {len(df)} 행")
    if args.mode:
        df = df[df["mode"] == args.mode]
        print(f"[filter] mode='{args.mode}': {len(df)} 행")
    if args.obstacle:
        df = df[df["has_obstacle"] == args.obstacle]
        print(f"[filter] obstacle='{args.obstacle}': {len(df)} 행")

    if df.empty:
        print("[오류] 필터 후 데이터 없음")
        sys.exit(0)

    print_summary(df)

    if args.plot:
        plot_comparison(df)


if __name__ == "__main__":
    main()
