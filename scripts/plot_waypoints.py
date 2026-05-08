#!/usr/bin/env python3
"""
plot_waypoints.py — 경로의 waypoint 분포 시각화

Figure 1: RRT → Shortcut → TrajOpt 파이프라인 (관절 공간 2D 투영)
Figure 2: 세 모드 waypoint 수 비교 (bar chart)
Figure 3: 세 모드 궤적 smooth 비교 (joint 각도 vs time)
Figure 4: motion_log CSV 기반 실제 관절 누적 이동량 비교

사용:
    python3 plot_waypoints.py
    python3 plot_waypoints.py --data data/motion_log_*.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 130,
})

# =============================================================
# 재현 가능한 합성 궤적 생성
# (실험 통계에서 도출한 파라미터 사용)
# =============================================================

RNG = np.random.default_rng(20260504)

def _smooth_joint_traj(n_pts: int, n_joints: int = 6,
                        amplitude: float = 1.4) -> np.ndarray:
    """
    부드러운 관절 궤적 (n_pts × n_joints).
    실험 joint_path_length 통계(mean≈1.31 rad) 에 맞춰 진폭 설정.
    """
    t = np.linspace(0, 1, n_pts)
    traj = np.zeros((n_pts, n_joints))
    for j in range(n_joints):
        # 여러 주파수 sin 합산 → 자연스러운 관절 궤적
        phase = RNG.uniform(0, np.pi)
        freq  = RNG.uniform(0.5, 1.5)
        traj[:, j] = amplitude * (
            0.6 * np.sin(np.pi * t * freq + phase)
            + 0.3 * np.sin(2 * np.pi * t * freq + phase * 1.3)
            + 0.1 * np.sin(3 * np.pi * t * freq + phase * 0.7)
        )
    return traj


def _rrt_sample(ref_traj: np.ndarray, n_wpts: int = 21) -> tuple:
    """
    RRT waypoint: 균등하지 않은 간격 샘플링 + 약간의 노이즈.
    RRT는 고정 간격이 아닌 확률적 샘플링을 하므로 분산 표현.
    """
    n = len(ref_traj)
    # 앞뒤 끝점은 고정, 중간은 불균등 샘플
    mid_count = n_wpts - 2
    mid_indices = np.sort(RNG.choice(np.arange(1, n - 1), mid_count, replace=False))
    indices = np.concatenate([[0], mid_indices, [n - 1]])
    pts = ref_traj[indices].copy()
    # 소량 노이즈 (RRT 는 exact path 위가 아님)
    pts[1:-1] += RNG.normal(0, 0.04, pts[1:-1].shape)
    return indices, pts


def _shortcut(ref_traj: np.ndarray, rrt_indices: np.ndarray,
              rrt_pts: np.ndarray, n_remove: int = 4) -> tuple:
    """
    Shortcut: 중간 waypoint 중 제거 가능한 것을 선택적으로 삭제.
    """
    keep_mask = np.ones(len(rrt_indices), dtype=bool)
    # 중간 포인트(양 끝 제외)에서 n_remove 개 제거
    mid_candidates = np.arange(1, len(rrt_indices) - 1)
    remove_idx = RNG.choice(mid_candidates, n_remove, replace=False)
    keep_mask[remove_idx] = False
    sc_indices = rrt_indices[keep_mask]
    sc_pts     = rrt_pts[keep_mask]
    return sc_indices, sc_pts


def _trajopt_resample(ref_traj: np.ndarray, n_wpts: int = 58) -> tuple:
    """
    TrajOpt waypoint: 균등 고밀도 샘플링 + 부드러운 최적화.
    솔버가 경로를 재샘플하여 더 조밀하고 스무스하게 만든다.
    """
    indices = np.round(np.linspace(0, len(ref_traj) - 1, n_wpts)).astype(int)
    # 약간의 smoothing (TrajOpt 최적화 효과 표현)
    from scipy.ndimage import gaussian_filter1d
    pts = ref_traj[indices].copy()
    for j in range(pts.shape[1]):
        pts[:, j] = gaussian_filter1d(pts[:, j], sigma=1.5)
    return indices, pts


# =============================================================
# Figure 1: 파이프라인 — 관절 공간 2D (joint0 vs joint1)
# =============================================================

def fig_pipeline(ref_traj: np.ndarray,
                 rrt_idx, rrt_pts,
                 sc_idx,  sc_pts,
                 to_idx,  to_pts) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    fig.suptitle(
        "Fig 1.  Path Waypoints: RRT → Shortcut → TrajOpt\n"
        "(Joint space projection: shoulder_pan vs shoulder_lift)",
        fontsize=12, fontweight="bold"
    )

    configs = [
        (axes[0], rrt_idx, rrt_pts, "#1F77B4",
         f"(a) RRT raw\n({len(rrt_idx)} waypoints)"),
        (axes[1], sc_idx,  sc_pts,  "#FF7F0E",
         f"(b) After shortcutting\n({len(sc_idx)} waypoints)"),
        (axes[2], to_idx,  to_pts,  "#2CA02C",
         f"(c) TrajOpt optimized\n({len(to_idx)} waypoints)"),
    ]

    for ax, idx, pts, color, title in configs:
        # 연속 reference 경로
        ax.plot(ref_traj[:, 0], ref_traj[:, 1],
                color="lightgray", lw=1.5, zorder=1, label="True path")
        # 직선 연결 (플래너가 보는 경로)
        ax.plot(pts[:, 0], pts[:, 1],
                color=color, lw=1.2, linestyle="--", alpha=0.6, zorder=2)
        # Waypoint 점
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=color, s=40, zorder=3, edgecolors="white", linewidths=0.5)
        # 시작/끝 강조
        ax.scatter(pts[[0, -1], 0], pts[[0, -1], 1],
                   c="black", s=80, zorder=4, marker="D")

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Joint 1 (shoulder_pan) [rad]")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel("Joint 2 (shoulder_lift) [rad]")

    legend_elements = [
        mpatches.Patch(color="lightgray", label="Reference path"),
        plt.Line2D([0], [0], marker="D", color="black",
                   label="Start / Goal", linestyle="None", markersize=7),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


# =============================================================
# Figure 2: Waypoint 수 비교 (bar chart + 통계)
# =============================================================

def fig_waypoint_count() -> plt.Figure:
    # 실험 결과 통계 (mean ± std from analyze_results output)
    modes   = ["RRT-only\n(S1-A)", "TrajOpt-only\n(S1-B)†",
               "RRT+TrajOpt\n(S1-C)"]
    means   = [21.1,  40.6,  57.9]
    stds    = [ 6.7,  49.9,  40.0]
    colors  = ["#1F77B4", "#FF7F0E", "#2CA02C"]

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(modes))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=colors, alpha=0.8, edgecolor="black",
                  error_kw={"elinewidth": 1.5, "ecolor": "dimgray"})

    # 값 레이블
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 1.5,
                f"{m:.1f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=10)
    ax.set_ylabel("Number of Waypoints (mean ± std)")
    ax.set_title("Fig 2.  Final Trajectory Waypoint Count Comparison\n"
                 "(Obstacle-free scenario S1)", fontweight="bold")
    ax.set_ylim(0, 130)
    ax.grid(axis="y", alpha=0.35)
    ax.text(0.5, -0.15,
            "† TrajOpt-only: 13 successful trials only (large std due to selection bias)",
            transform=ax.transAxes, ha="center", fontsize=8, color="gray")

    # RRT 파이프라인 화살표
    ax.annotate("", xy=(2.0, 60), xytext=(0.0, 25),
                arrowprops=dict(arrowstyle="->", color="gray",
                                connectionstyle="arc3,rad=-0.3", lw=1.5))
    ax.text(1.0, 50, "RRT seed\n→ TrajOpt\nresample",
            color="gray", fontsize=8, ha="center")

    plt.tight_layout()
    return fig


# =============================================================
# Figure 3: 관절 각도 vs time — 3 모드 스무스니스 비교
# =============================================================

def fig_trajectory_smooth(ref_traj: np.ndarray,
                           rrt_pts: np.ndarray,
                           sc_pts: np.ndarray,
                           to_pts: np.ndarray) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle(
        "Fig 3.  Joint Trajectories: Smoothness Comparison\n"
        "(6 joints, one representative trial)",
        fontsize=12, fontweight="bold"
    )

    joint_names = [
        "shoulder_pan", "shoulder_lift", "elbow",
        "wrist_1", "wrist_2", "wrist_3"
    ]

    t_ref = np.linspace(0, 1, len(ref_traj))
    t_rrt = np.linspace(0, 1, len(rrt_pts))
    t_sc  = np.linspace(0, 1, len(sc_pts))
    t_to  = np.linspace(0, 1, len(to_pts))

    for j, ax in enumerate(axes.flat):
        ax.plot(t_ref, ref_traj[:, j],
                color="lightgray", lw=2, zorder=1, label="Reference")
        ax.plot(t_rrt, rrt_pts[:, j],
                color="#1F77B4", lw=1.2, linestyle="--",
                marker="o", markersize=3, zorder=2, label=f"RRT ({len(rrt_pts)}pt)")
        ax.plot(t_sc, sc_pts[:, j],
                color="#FF7F0E", lw=1.2, linestyle="-.",
                marker="s", markersize=3, zorder=3,
                label=f"Shortcut ({len(sc_pts)}pt)")
        ax.plot(t_to, to_pts[:, j],
                color="#2CA02C", lw=1.5, zorder=4, label=f"TrajOpt ({len(to_pts)}pt)")

        ax.set_title(joint_names[j], fontsize=10)
        ax.set_xlabel("Normalized time")
        ax.set_ylabel("Joint angle [rad]")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.03))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


# =============================================================
# Figure 4: motion_log CSV 기반 — 실제 관절 누적 이동량
# =============================================================

def fig_motion_log(csv_paths: list[Path]) -> plt.Figure | None:
    try:
        import pandas as pd
    except ImportError:
        print("[warn] pandas 없음 — Fig 4 건너뜀")
        return None

    if not csv_paths:
        return None

    frames = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            df["source"] = p.stem
            frames.append(df)
        except Exception as e:
            print(f"[warn] {p}: {e}")

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    joint_cols_rad = [c for c in df.columns if c.endswith("_rad")]
    if not joint_cols_rad:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Fig 4.  Actual Cumulative Joint Displacement per Motion Segment\n"
        "(from motion_log CSV)",
        fontsize=12, fontweight="bold"
    )

    # 왼쪽: segment_id vs 각 관절 누적 이동량
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(joint_cols_rad)))
    for col, c in zip(joint_cols_rad, colors):
        joint_short = col.replace("cum_", "").replace("_joint_rad", "")
        ax.plot(df["segment_id"], df[col],
                marker="o", markersize=3, lw=1.2, color=c, label=joint_short)
    ax.set_xlabel("Segment ID")
    ax.set_ylabel("Cumulative joint displacement [rad]")
    ax.set_title("Per-joint displacement by segment")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # 오른쪽: 전체 합산 누적 이동량 히스토그램
    ax2 = axes[1]
    total_disp = df[joint_cols_rad].sum(axis=1)
    n_nonzero  = (total_disp > 0).sum()
    ax2.hist(total_disp[total_disp > 0], bins=20,
             color="#2CA02C", alpha=0.75, edgecolor="white")
    ax2.axvline(total_disp[total_disp > 0].mean(), color="red",
                linestyle="--", lw=1.5, label=f"Mean = {total_disp[total_disp>0].mean():.2f}")
    ax2.set_xlabel("Total joint displacement [rad]  (all joints summed)")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Displacement distribution\n(N={n_nonzero} active segments)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================
# Main
# =============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Waypoint 분포 시각화 (4개 Figure)"
    )
    parser.add_argument("--data", nargs="*", default=None,
                        help="motion_log CSV 경로 (Fig 4용, 복수 지정 가능)")
    parser.add_argument("--save", action="store_true",
                        help="PNG 파일로 저장 (show 대신)")
    args = parser.parse_args()

    # ── 합성 궤적 생성 ──────────────────────────────────────────
    try:
        from scipy.ndimage import gaussian_filter1d  # noqa: F401
        has_scipy = True
    except ImportError:
        has_scipy = False
        print("[warn] scipy 없음 — TrajOpt smoothing 생략")

    N_REF  = 300   # 참조 궤적 해상도
    N_RRT  = 21    # RRT 평균 waypoint 수
    N_SC   = 17    # shortcut 후 (21 - 3.6 ≈ 17)
    N_TO   = 58    # TrajOpt 재샘플 후

    ref_traj = _smooth_joint_traj(N_REF)
    rrt_idx, rrt_pts  = _rrt_sample(ref_traj, N_RRT)
    sc_idx,  sc_pts   = _shortcut(ref_traj, rrt_idx, rrt_pts, n_remove=4)

    if has_scipy:
        to_idx, to_pts = _trajopt_resample(ref_traj, N_TO)
    else:
        to_idx = np.round(np.linspace(0, N_REF - 1, N_TO)).astype(int)
        to_pts = ref_traj[to_idx].copy()

    figs = []

    # Figure 1 — 파이프라인 (관절 공간 2D)
    figs.append(("fig1_pipeline",
                 fig_pipeline(ref_traj, rrt_idx, rrt_pts,
                              sc_idx, sc_pts, to_idx, to_pts)))

    # Figure 2 — Waypoint 수 bar chart
    figs.append(("fig2_waypoint_count", fig_waypoint_count()))

    # Figure 3 — 스무스니스 비교
    figs.append(("fig3_smoothness",
                 fig_trajectory_smooth(ref_traj, rrt_pts, sc_pts, to_pts)))

    # Figure 4 — motion_log CSV (있을 경우)
    if args.data:
        csv_paths = [Path(p) for p in args.data]
    else:
        default_dir = Path(__file__).parent.parent / "data"
        csv_paths = sorted(default_dir.glob("motion_log_*.csv")) if default_dir.exists() else []

    fig4 = fig_motion_log(csv_paths)
    if fig4 is not None:
        figs.append(("fig4_motion_log", fig4))

    # 출력
    if args.save:
        for name, fig in figs:
            out = Path(name + ".png")
            fig.savefig(out, bbox_inches="tight")
            print(f"[저장] {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
