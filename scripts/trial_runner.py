#!/usr/bin/env python3
# =============================================================
# trial_runner.py
#
# 사용자가 직접 관리하는 것:
#   - pick_place.launch.py 실행 (experiment_mode 포함)
#   - 장애물 추가 / 제거
#
# 이 스크립트가 하는 것:
#   - N회 pick/place 동작 전송 및 완료 대기
#   - 노드가 기록한 CSV 에서 신규 행 읽기
#   - run_id, mode, has_obstacle, trial_idx, seed, pose 메타데이터를
#     접두어 컬럼으로 추가하여 결과 CSV 에 누적 저장
#
# 호출 예시:
#   python3 trial_runner.py \
#       --mode rrt_only --obstacle no \
#       --trials 20 \
#       --output results/rrt_only_no_obs.csv \
#       --node-csv ~/.ros/pick_place_exp/node_data.csv
# =============================================================

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped

# =============================================================
# Default pose-sampling bounds (scenarios.yaml S1/S2 공통)
# =============================================================
DEFAULT_PICK_X  = [0.30,  0.50]
DEFAULT_PICK_Y  = [0.20,  0.50]
DEFAULT_PICK_Z  = [0.30,  0.50]
DEFAULT_PLACE_X = [-0.50, -0.30]
DEFAULT_PLACE_Y = [ 0.30,  0.50]
DEFAULT_PLACE_Z = [ 0.30,  0.50]

DEFAULT_GRASP_ORIENT = {"x": 0.0, "y": 1.0, "z": 0.0, "w": 0.0}
DEFAULT_NODE_CSV     = Path.home() / ".ros" / "pick_place_exp" / "node_data.csv"
DEFAULT_SEED         = 20260504


# =============================================================
# CSV I/O helpers
# =============================================================

# 결과 CSV 앞에 붙는 메타데이터 컬럼 목록
META_FIELDS = (
    "run_id",
    "timestamp",
    "mode",
    "has_obstacle",
    "trial_idx",
    "seed",
    "pick_x", "pick_y", "pick_z",
    "place_x", "place_y", "place_z",
)


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _read_new_rows(
    node_csv: Path,
    byte_offset: int,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """byte_offset 이후 node_csv 에 추가된 행과 헤더를 반환한다."""
    if not node_csv.exists():
        return [], []
    with node_csv.open("r", newline="") as f:
        f.seek(0)
        header = [h.strip() for h in f.readline().strip().split(",")]
        f.seek(byte_offset)
        new_text = f.read()
    rows: List[Dict[str, str]] = []
    if not new_text.strip():
        return rows, header
    lines = [ln for ln in new_text.splitlines() if ln.strip()]
    for row_dict in csv.DictReader(lines, fieldnames=header):
        if row_dict.get("trial_id") == "trial_id":   # 중복 헤더 행 무시
            continue
        rows.append(row_dict)
    return rows, header


def _append_rows(
    out_csv: Path,
    rows: List[Dict[str, str]],
    header_fields: List[str],
    meta: Dict[str, Any],
) -> None:
    """메타데이터 + 노드 CSV 행을 out_csv 에 누적 추가한다."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(META_FIELDS) + list(header_fields)
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({**meta, **r})


# =============================================================
# ROS2 노드 — pick/place 액션 전송만 담당
# =============================================================

class TrialRunner(Node):
    def __init__(self) -> None:
        super().__init__("trial_runner")

        self._pick_pub  = self.create_publisher(PoseStamped, "/pick_goal",  10)
        self._place_pub = self.create_publisher(PoseStamped, "/place_goal", 10)

        from pick_place_module.action import Pick, Place
        self._pick_action  = ActionClient(self, Pick,  "pick")
        self._place_action = ActionClient(self, Place, "place")

    # ----------------------------------------------------------
    def make_pose_stamped(
        self,
        xyz: Tuple[float, float, float],
        orient: Dict[str, float],
    ) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id  = "base_link"
        ps.header.stamp     = self.get_clock().now().to_msg()
        ps.pose.position.x  = float(xyz[0])
        ps.pose.position.y  = float(xyz[1])
        ps.pose.position.z  = float(xyz[2])
        ps.pose.orientation.x = float(orient["x"])
        ps.pose.orientation.y = float(orient["y"])
        ps.pose.orientation.z = float(orient["z"])
        ps.pose.orientation.w = float(orient["w"])
        return ps

    # ----------------------------------------------------------
    def send_pick(
        self, ps: PoseStamped, timeout_sec: float
    ) -> Tuple[bool, str]:
        from pick_place_module.action import Pick
        if not self._pick_action.wait_for_server(timeout_sec=10.0):
            return False, "pick action server unavailable"
        goal = Pick.Goal()
        goal.pick_pose = ps.pose
        return self._await_action(self._pick_action, goal, timeout_sec)

    def send_place(
        self, ps: PoseStamped, timeout_sec: float
    ) -> Tuple[bool, str]:
        from pick_place_module.action import Place
        if not self._place_action.wait_for_server(timeout_sec=10.0):
            return False, "place action server unavailable"
        goal = Place.Goal()
        goal.place_pose = ps.pose
        return self._await_action(self._place_action, goal, timeout_sec)

    def _await_action(
        self,
        client: ActionClient,
        goal: Any,
        timeout_sec: float,
    ) -> Tuple[bool, str]:
        fut = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=15.0)
        if not fut.done() or fut.result() is None:
            return False, "goal send timed out"
        gh = fut.result()
        if not gh.accepted:
            return False, "goal rejected"
        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=timeout_sec)
        if not res_fut.done() or res_fut.result() is None:
            return False, "result timed out"
        result = res_fut.result().result
        return bool(result.success), str(getattr(result, "message", ""))


# =============================================================
# 실험 루프
# =============================================================

def _sample(rng: random.Random, lo: float, hi: float) -> float:
    return rng.uniform(lo, hi)


def run_trials(args: argparse.Namespace) -> int:
    node_csv  = Path(args.node_csv).expanduser()
    out_csv   = Path(args.output).expanduser()
    pick_to   = float(args.pick_timeout)
    place_to  = float(args.place_timeout)
    n_trials  = int(args.trials)
    master_seed = int(args.seed)
    run_id    = args.run_id or f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    print("=" * 60)
    print(f"  trial_runner  run_id={run_id}")
    print(f"  mode         : {args.mode}")
    print(f"  has_obstacle : {args.obstacle}")
    print(f"  trials       : {n_trials}")
    print(f"  output CSV   : {out_csv}")
    print(f"  node CSV     : {node_csv}")
    print("=" * 60)

    rclpy.init()
    runner = TrialRunner()

    completed = failures = 0
    t_start   = time.time()

    try:
        for t in range(n_trials):
            # 재현 가능한 seed (run마다 동일 pose 시퀀스)
            seed = master_seed + t
            rng  = random.Random(seed)

            pick_xyz = (
                _sample(rng, *DEFAULT_PICK_X),
                _sample(rng, *DEFAULT_PICK_Y),
                _sample(rng, *DEFAULT_PICK_Z),
            )
            place_xyz = (
                _sample(rng, *DEFAULT_PLACE_X),
                _sample(rng, *DEFAULT_PLACE_Y),
                _sample(rng, *DEFAULT_PLACE_Z),
            )

            meta: Dict[str, Any] = {
                "run_id":       run_id,
                "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
                "mode":         args.mode,
                "has_obstacle": args.obstacle,
                "trial_idx":    t,
                "seed":         seed,
                "pick_x":       f"{pick_xyz[0]:.6f}",
                "pick_y":       f"{pick_xyz[1]:.6f}",
                "pick_z":       f"{pick_xyz[2]:.6f}",
                "place_x":      f"{place_xyz[0]:.6f}",
                "place_y":      f"{place_xyz[1]:.6f}",
                "place_z":      f"{place_xyz[2]:.6f}",
            }

            pre_offset = _file_size(node_csv)

            pick_ps  = runner.make_pose_stamped(pick_xyz,  DEFAULT_GRASP_ORIENT)
            place_ps = runner.make_pose_stamped(place_xyz, DEFAULT_GRASP_ORIENT)

            print(f"[trial {t+1:3d}/{n_trials}]  pick={pick_xyz}  place={place_xyz}")

            pick_ok,  pick_msg  = runner.send_pick(pick_ps,   pick_to)
            place_ok, place_msg = (runner.send_place(place_ps, place_to)
                                   if pick_ok else (False, "skipped (pick failed)"))

            if not (pick_ok and place_ok):
                failures += 1
                print(f"  [FAIL] pick={pick_ok}({pick_msg!r})  "
                      f"place={place_ok}({place_msg!r})")

            # 노드가 CSV 를 flush 할 때까지 짧게 대기
            time.sleep(0.5)

            new_rows, header = _read_new_rows(node_csv, pre_offset)
            if not new_rows:
                print(f"  [WARN] trial {t}: node CSV 에 새 행 없음")

            _append_rows(out_csv, new_rows, header, meta)

            completed += 1
            elapsed = time.time() - t_start
            rate = completed / elapsed if elapsed > 0 else 0.0
            eta  = (n_trials - completed) / rate if rate > 0 else float("inf")
            print(f"  [진행] {completed}/{n_trials}  failures={failures}  "
                  f"ETA {eta/60.0:.1f}min")

    finally:
        runner.destroy_node()
        rclpy.shutdown()

    print(f"\n[완료] completed={completed}  failures={failures}  "
          f"elapsed={(time.time()-t_start)/60.0:.1f}min")
    print(f"[출력] {out_csv}")
    return 0 if failures == 0 else 1


# =============================================================
# CLI
# =============================================================

def _parse(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="pick-place 단독 trial 실행기 (모드/장애물은 사용자가 직접 설정)"
    )
    p.add_argument("--mode", required=True,
                   choices=["rrt_only", "trajopt_only", "rrt_trajopt"],
                   help="레이블 전용 — pick_place.launch.py 설정과 일치해야 함")
    p.add_argument("--obstacle", required=True,
                   choices=["yes", "no"],
                   help="레이블 전용 — 실제 장애물은 사용자가 직접 관리")
    p.add_argument("--trials",  type=int, default=20,
                   help="실행할 trial 횟수 (기본: 20)")
    p.add_argument("--output",  required=True,
                   help="결과 CSV 경로 (누적 추가)")
    p.add_argument("--node-csv", default=str(DEFAULT_NODE_CSV),
                   help=f"pick_place_node 가 쓰는 CSV (기본: {DEFAULT_NODE_CSV})")
    p.add_argument("--seed",    type=int, default=DEFAULT_SEED,
                   help=f"마스터 RNG seed (기본: {DEFAULT_SEED})")
    p.add_argument("--run-id",  default=None,
                   help="run 식별자 (기본: 자동 생성)")
    p.add_argument("--pick-timeout",  type=float, default=120.0,
                   help="pick 액션 timeout 초 (기본: 120)")
    p.add_argument("--place-timeout", type=float, default=120.0,
                   help="place 액션 timeout 초 (기본: 120)")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run_trials(_parse()))
