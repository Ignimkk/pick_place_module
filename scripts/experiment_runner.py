#!/usr/bin/env python3
# =============================================================
# experiment_runner.py
#
# 2026 KIIE conference automation runner.
#
# For each (scenario S1/S2) × (mode A/B/C/C1/C2) × (20 trials):
#   1. Set ROS 2 parameters on the running pick_place_node.
#   2. Spawn / despawn the testbed obstacle (Gazebo + MoveIt PlanningScene).
#   3. Sample randomized pick/place poses with a deterministic seed.
#   4. Send /pick_goal and /place_goal goals via PoseStamped topics.
#      (goal_relay_node forwards them to the Pick / Place actions.)
#   5. Wait for completion, then read newly appended rows from the
#      pick_place_node CSV.
#   6. Augment each row with (run_id, scenario, mode, trial_idx, seed)
#      and append to a single cumulative CSV.
#
# Pre-conditions
#   - ur_setup_bringup launch is up      (Gazebo + move_group + controllers)
#   - trajopt_server_node is running     (ros2 run trajopt_validation trajopt_server_node)
#   - pick_place_module launch is up     (pick_place_node + goal_relay_node)
#       launched with experiment_csv_path:=<EXPERIMENT_CSV>  AND
#                     return_home_after_place:=true
#
# Example (one shot, 200 trials):
#   ros2 run pick_place_module experiment_runner.py \
#       --scenarios-file $(ros2 pkg prefix pick_place_module)/share/pick_place_module/config/scenarios.yaml \
#       --node-csv ~/.ros/pick_place_exp/run_node.csv \
#       --out ~/.ros/pick_place_exp/cumulative.csv
# =============================================================

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile

from geometry_msgs.msg import Pose, PoseStamped
from rcl_interfaces.msg import ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from trajectory_msgs.msg import JointTrajectory

# Optional MoveIt collision messaging (graceful fallback if unavailable)
try:
    from moveit_msgs.msg import CollisionObject, PlanningScene
    from moveit_msgs.srv import ApplyPlanningScene
    from shape_msgs.msg import SolidPrimitive
    HAS_MOVEIT_MSGS = True
except Exception:
    HAS_MOVEIT_MSGS = False


# ---------------------------------------------------------------------------
# Mode → parameter mapping is read from scenarios.yaml; nothing hard-coded here.
# ---------------------------------------------------------------------------

PICK_PLACE_NODE = "/pick_place_node"


# ---------------------------------------------------------------------------
# CSV row augmentation helpers
# ---------------------------------------------------------------------------

PREFIX_FIELDS = (
    "run_id",
    "scenario",
    "mode",
    "trial_idx",
    "seed",
    "pick_x", "pick_y", "pick_z",
    "place_x", "place_y", "place_z",
)


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def read_new_rows(node_csv: Path, byte_offset: int) -> Tuple[List[Dict[str, str]], List[str]]:
    """Return (rows_appended_since_offset, header_fields)."""
    if not node_csv.exists():
        return [], []
    with node_csv.open("r", newline="") as f:
        # Always read the header first
        f.seek(0)
        header_line = f.readline()
        header = [h.strip() for h in header_line.strip().split(",")]
        # Now jump past the previously-seen content
        f.seek(byte_offset)
        new_text = f.read()
    rows: List[Dict[str, str]] = []
    if not new_text.strip():
        return rows, header
    # If the offset landed mid-line (e.g. before any rows were ever written),
    # csv.DictReader will misalign. Splitlines and skip blanks defensively.
    lines = [ln for ln in new_text.splitlines() if ln.strip()]
    reader = csv.DictReader(lines, fieldnames=header)
    for r in reader:
        # Skip a duplicated header row (happens when offset was 0).
        if r.get("trial_id") == "trial_id":
            continue
        rows.append(r)
    return rows, header


def write_trajectory_csv(out_path: Path, jt) -> None:
    """Write a JointTrajectory message to CSV with columns:
       time, p1..pN, v1..vN, a1..aN  (velocities/accelerations may be 0
       if the controller did not populate them)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_points = len(jt.points)
    if n_points == 0:
        return
    n_joints = len(jt.points[0].positions)
    fields = ["time"]
    fields += [f"p{i+1}" for i in range(n_joints)]
    fields += [f"v{i+1}" for i in range(n_joints)]
    fields += [f"a{i+1}" for i in range(n_joints)]
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for pt in jt.points:
            t = float(pt.time_from_start.sec) + float(pt.time_from_start.nanosec) * 1e-9
            pos = list(pt.positions) if pt.positions else [0.0] * n_joints
            vel = list(pt.velocities) if pt.velocities else [0.0] * n_joints
            acc = list(pt.accelerations) if pt.accelerations else [0.0] * n_joints
            w.writerow([f"{t:.6f}"] + [f"{x:.6f}" for x in pos + vel + acc])


def write_cumulative_rows(
    out_csv: Path,
    new_rows: List[Dict[str, str]],
    header_fields: List[str],
    meta: Dict[str, Any],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(PREFIX_FIELDS) + list(header_fields)
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in new_rows:
            row_out = {**meta, **r}
            w.writerow(row_out)


# ---------------------------------------------------------------------------
# rclpy node — encapsulates parameter, publisher, action wiring
# ---------------------------------------------------------------------------

class ExperimentRunner(Node):
    def __init__(self):
        super().__init__("experiment_runner")

        self._set_param_cli = self.create_client(
            SetParameters, f"{PICK_PLACE_NODE}/set_parameters"
        )

        self._pick_pub = self.create_publisher(PoseStamped, "/pick_goal", 10)
        self._place_pub = self.create_publisher(PoseStamped, "/place_goal", 10)

        # /apply_planning_scene 서비스 클라이언트 (토픽 발행보다 신뢰성 높음)
        if HAS_MOVEIT_MSGS:
            self._apply_scene_cli = self.create_client(
                ApplyPlanningScene, "/apply_planning_scene"
            )
        else:
            self._apply_scene_cli = None

        # Action clients for /pick and /place — used to await completion.
        from pick_place_module.action import Pick, Place
        self._pick_action = ActionClient(self, Pick, "pick")
        self._place_action = ActionClient(self, Place, "place")

        # ── Trajectory recorder ────────────────────────────────────────
        # pick_place_node publishes the executed JointTrajectory on
        # /joint_trajectory_controller/joint_trajectory. Each header.frame_id
        # is set to the step_name, so we can route each captured trajectory
        # to the right (mode, trial, step) file.
        self._traj_buffer: List[Tuple[str, JointTrajectory]] = []
        self._traj_sub = self.create_subscription(
            JointTrajectory,
            "/joint_trajectory_controller/joint_trajectory",
            self._on_trajectory,
            10,
        )

    def _on_trajectory(self, msg: JointTrajectory) -> None:
        # Tag with step_name (encoded by pick_place_node into header.frame_id)
        step = msg.header.frame_id or "unknown"
        self._traj_buffer.append((step, msg))

    def consume_trajectories(self) -> List[Tuple[str, JointTrajectory]]:
        out = self._traj_buffer
        self._traj_buffer = []
        return out

    # ------------------------------------------------------------------
    # Parameter setting
    # ------------------------------------------------------------------

    def set_node_parameters(
        self,
        params: Dict[str, Any],
        timeout_sec: float = 5.0,
        max_retries: int = 3,
        retry_delay_sec: float = 2.0,
    ) -> bool:
        """
        /pick_place_node/set_parameters 서비스를 호출하여 파라미터를 일괄 설정한다.

        재시도 이유:
          spawn_obstacle() 의 subprocess (ros2 run ros_gz_sim create) 가 종료될 때
          DDS discovery 이벤트가 발생하고, 서비스 엔드포인트가 일시적으로 불안정해질 수
          있다. wait_for_service() 는 True 를 반환하더라도 첫 응답이 "not declared"로
          돌아오는 race condition 을 재시도로 흡수한다.
        """
        for attempt in range(max_retries):
            if not self._set_param_cli.wait_for_service(timeout_sec=timeout_sec):
                self.get_logger().error(
                    f"set_parameters service for {PICK_PLACE_NODE} not available "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay_sec)
                continue

            ros_params: List[Parameter] = []
            for k, v in params.items():
                ros_params.append(Parameter(name=k, value=v))

            req = SetParameters.Request()
            req.parameters = [p.to_parameter_msg() for p in ros_params]

            future = self._set_param_cli.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)

            if not future.done() or future.result() is None:
                self.get_logger().error(
                    f"set_parameters call did not complete "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay_sec)
                continue

            ok = all(r.successful for r in future.result().results)
            if not ok:
                for k, r in zip(params.keys(), future.result().results):
                    if not r.successful:
                        self.get_logger().error(
                            f"  param {k} failed: {r.reason}"
                        )
                if attempt < max_retries - 1:
                    self.get_logger().warn(
                        f"  param set failed, retrying in {retry_delay_sec}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(retry_delay_sec)
                    continue

            return ok

        return False

    # ------------------------------------------------------------------
    # Pose publishing + action awaiting
    # ------------------------------------------------------------------

    def make_pose_stamped(self, xyz: Tuple[float, float, float],
                          orient: Dict[str, float]) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = "base_link"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(xyz[0])
        ps.pose.position.y = float(xyz[1])
        ps.pose.position.z = float(xyz[2])
        ps.pose.orientation.x = float(orient["x"])
        ps.pose.orientation.y = float(orient["y"])
        ps.pose.orientation.z = float(orient["z"])
        ps.pose.orientation.w = float(orient["w"])
        return ps

    def send_pick(self, pose_stamped: PoseStamped, timeout_sec: float) -> Tuple[bool, str]:
        from pick_place_module.action import Pick
        if not self._pick_action.wait_for_server(timeout_sec=10.0):
            return False, "pick action server not available"
        goal = Pick.Goal()
        goal.pick_pose = pose_stamped.pose
        return self._await_action(self._pick_action, goal, timeout_sec)

    def send_place(self, pose_stamped: PoseStamped, timeout_sec: float) -> Tuple[bool, str]:
        from pick_place_module.action import Place
        if not self._place_action.wait_for_server(timeout_sec=10.0):
            return False, "place action server not available"
        goal = Place.Goal()
        goal.place_pose = pose_stamped.pose
        return self._await_action(self._place_action, goal, timeout_sec)

    def _await_action(self, client: ActionClient, goal, timeout_sec: float) -> Tuple[bool, str]:
        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        if not future.done() or future.result() is None:
            return False, "goal send timed out"
        gh = future.result()
        if not gh.accepted:
            return False, "goal rejected"
        result_future = gh.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout_sec)
        if not result_future.done() or result_future.result() is None:
            return False, "result timed out"
        result_wrapper = result_future.result()
        result = result_wrapper.result
        return bool(result.success), str(getattr(result, "message", ""))

    # ------------------------------------------------------------------
    # Obstacle (Gazebo + PlanningScene)
    # ------------------------------------------------------------------

    def spawn_obstacle(self, obs_cfg: Dict[str, Any]) -> bool:
        sdf_pkg = obs_cfg["sdf_pkg"]
        sdf_relpath = obs_cfg["sdf_relpath"]
        name = obs_cfg["name"]
        x, y, z = obs_cfg["spawn_xyz"]

        try:
            prefix = subprocess.check_output(
                ["ros2", "pkg", "prefix", sdf_pkg], text=True
            ).strip()
        except subprocess.CalledProcessError:
            self.get_logger().error(f"package {sdf_pkg} not found")
            return False
        sdf_path = Path(prefix) / "share" / sdf_pkg / sdf_relpath
        if not sdf_path.exists():
            self.get_logger().warn(f"obstacle sdf not found at {sdf_path} — skipping Gazebo spawn")
        else:
            cmd = [
                "ros2", "run", "ros_gz_sim", "create",
                "-file", str(sdf_path),
                "-name", name,
                "-x", str(x), "-y", str(y), "-z", str(z),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=20)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                self.get_logger().warn(
                    f"ros_gz_sim create failed (already spawned?): {e}"
                )

        col_cfg = obs_cfg.get("collision")
        if col_cfg:
            self._apply_collision_object(name, col_cfg, op="ADD")
        return True

    def remove_obstacle(self, obs_cfg: Dict[str, Any]) -> None:
        name = obs_cfg["name"]
        cmd = ["ros2", "service", "call", "/world/default/remove",
               "ros_gz_interfaces/srv/DeleteEntity",
               f"{{entity: {{name: '{name}', type: 2}}}}"]
        try:
            subprocess.run(cmd, check=False, capture_output=True, timeout=10)
        except subprocess.TimeoutExpired:
            pass
        col_cfg = obs_cfg.get("collision")
        if col_cfg:
            self._apply_collision_object(name, col_cfg, op="REMOVE")

    def _apply_collision_object(
        self,
        name: str,
        col_cfg: Dict[str, Any],
        op: str,
        timeout_sec: float = 10.0,
    ) -> bool:
        """
        /apply_planning_scene 서비스를 사용하여 MoveIt PlanningScene 에 충돌 객체를
        추가(ADD) 또는 제거(REMOVE)한다.

        토픽 발행 대신 서비스를 사용하는 이유:
          - 토픽 방식은 QoS·타이밍 문제로 move_group 이 메시지를 놓칠 수 있다.
          - 서비스 방식은 request-response 로 씬 적용이 보장된다.
          - 서비스 응답 후에는 pick_place_node PSM 이 즉시 최신 씬을 조회할 수 있다.
        """
        if not HAS_MOVEIT_MSGS or self._apply_scene_cli is None:
            self.get_logger().warn(
                "moveit_msgs 없음 또는 서비스 클라이언트 미초기화 — 충돌 객체 적용 건너뜀"
            )
            return False

        if not self._apply_scene_cli.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error(
                f"/apply_planning_scene 서비스 없음 (timeout={timeout_sec}s) — "
                "장애물이 MoveIt 씬에 반영되지 않습니다"
            )
            return False

        # CollisionObject 구성
        co = CollisionObject()
        co.header.frame_id = col_cfg["frame_id"]
        co.header.stamp = self.get_clock().now().to_msg()
        co.id = name

        if op == "ADD":
            prim = SolidPrimitive()
            prim.type = SolidPrimitive.BOX
            prim.dimensions = list(col_cfg["size"])
            co.primitives = [prim]
            pose = Pose()
            px, py, pz = col_cfg["position"]
            pose.position.x, pose.position.y, pose.position.z = px, py, pz
            pose.orientation.w = 1.0
            co.primitive_poses = [pose]
            co.operation = CollisionObject.ADD
        else:
            co.operation = CollisionObject.REMOVE

        # PlanningScene diff 구성
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(co)

        req = ApplyPlanningScene.Request()
        req.scene = scene

        future = self._apply_scene_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)

        if not future.done() or future.result() is None:
            self.get_logger().error(
                f"/apply_planning_scene 호출 미완료 (timeout={timeout_sec}s)"
            )
            return False

        ok = bool(future.result().success)
        if ok:
            self.get_logger().info(
                f"[obstacle] {op} '{name}' → MoveIt PlanningScene 적용 완료"
            )
        else:
            self.get_logger().warn(
                f"[obstacle] {op} '{name}' → /apply_planning_scene 반환 success=False"
            )
        return ok


# ---------------------------------------------------------------------------
# Trial orchestration
# ---------------------------------------------------------------------------

@dataclass
class TrialMeta:
    run_id: str
    scenario: str
    mode: str
    trial_idx: int
    seed: int
    pick_x: float
    pick_y: float
    pick_z: float
    place_x: float
    place_y: float
    place_z: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_id":   self.run_id,
            "scenario": self.scenario,
            "mode":     self.mode,
            "trial_idx": self.trial_idx,
            "seed":     self.seed,
            "pick_x":   f"{self.pick_x:.6f}",
            "pick_y":   f"{self.pick_y:.6f}",
            "pick_z":   f"{self.pick_z:.6f}",
            "place_x":  f"{self.place_x:.6f}",
            "place_y":  f"{self.place_y:.6f}",
            "place_z":  f"{self.place_z:.6f}",
        }


def sample_pose(rng: random.Random,
                x_range: List[float],
                y_range: List[float],
                z_range: List[float]) -> Tuple[float, float, float]:
    return (
        rng.uniform(*x_range),
        rng.uniform(*y_range),
        rng.uniform(*z_range),
    )


def trial_seed(master: int, scenario_idx: int, mode_idx: int, trial_idx: int) -> int:
    return int(master) + scenario_idx * 1000 + mode_idx * 100 + trial_idx


def run_all(args: argparse.Namespace) -> int:
    cfg_path = Path(args.scenarios_file).expanduser()
    cfg = yaml.safe_load(cfg_path.read_text())

    scenarios = cfg["scenarios"]
    modes = cfg["modes"]
    n_trials = int(cfg["trials_per_cell"])
    master_seed = int(cfg["master_seed"])
    grasp_orient = cfg["grasp_orientation"]
    timeouts = cfg.get("timeouts", {})
    pick_to = float(timeouts.get("pick_action_sec", 120.0))
    place_to = float(timeouts.get("place_action_sec", 120.0))
    param_to = float(timeouts.get("param_set_sec", 5.0))

    if args.scenarios:
        scen_keys = [s for s in args.scenarios.split(",") if s.strip() in scenarios]
    else:
        scen_keys = list(scenarios.keys())
    if args.modes:
        mode_keys = [m for m in args.modes.split(",") if m.strip() in modes]
    else:
        mode_keys = list(modes.keys())

    if args.trials_per_cell is not None:
        n_trials = int(args.trials_per_cell)

    run_id = args.run_id or f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    node_csv = Path(args.node_csv).expanduser()
    out_csv = Path(args.out).expanduser()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    traj_dir = Path(args.traj_dir).expanduser() if args.traj_dir else None
    if traj_dir is not None:
        traj_dir.mkdir(parents=True, exist_ok=True)

    print(f"[runner] run_id      = {run_id}")
    print(f"[runner] scenarios   = {scen_keys}")
    print(f"[runner] modes       = {mode_keys}")
    print(f"[runner] trials/cell = {n_trials}")
    print(f"[runner] node CSV    = {node_csv}")
    print(f"[runner] out  CSV    = {out_csv}")

    rclpy.init()
    runner = ExperimentRunner()

    total_trials = len(scen_keys) * len(mode_keys) * n_trials
    completed = 0
    failures = 0
    t_start = time.time()

    try:
        for s_idx, scen_key in enumerate(scen_keys):
            scen = scenarios[scen_key]
            obstacle_enabled = bool(scen.get("obstacle_enabled", False))
            if obstacle_enabled and "obstacle" in scen:
                print(f"[runner][{scen_key}] spawning obstacle")
                runner.spawn_obstacle(scen["obstacle"])
                # ros_gz_sim create subprocess 가 종료될 때 DDS discovery 이벤트가
                # 발생하여 서비스 엔드포인트가 일시적으로 불안정해질 수 있다.
                # 2초 대기로 DDS 재탐색이 안정화되도록 한다.
                time.sleep(2.0)

            for m_idx, mode_key in enumerate(mode_keys):
                mode_cfg = modes[mode_key]
                params = mode_cfg["params"]
                mode_label = mode_cfg["label"]

                print(f"\n[runner][{scen_key}/{mode_key}] setting parameters: {params}")
                if not runner.set_node_parameters(params, timeout_sec=param_to):
                    print(f"[runner][{scen_key}/{mode_key}] param set FAILED — skipping cell")
                    failures += n_trials
                    continue

                # Allow a brief settle after parameter changes
                time.sleep(0.5)

                for t in range(n_trials):
                    seed = trial_seed(master_seed, s_idx, m_idx, t)
                    rng = random.Random(seed)
                    pick_xyz  = sample_pose(rng,
                                            scen["pick_x_range"],
                                            scen["pick_y_range"],
                                            scen["pick_z_range"])
                    place_xyz = sample_pose(rng,
                                            scen["place_x_range"],
                                            scen["place_y_range"],
                                            scen["place_z_range"])

                    meta = TrialMeta(
                        run_id=run_id, scenario=scen_key, mode=mode_key,
                        trial_idx=t, seed=seed,
                        pick_x=pick_xyz[0], pick_y=pick_xyz[1], pick_z=pick_xyz[2],
                        place_x=place_xyz[0], place_y=place_xyz[1], place_z=place_xyz[2],
                    )

                    pre_offset = file_size(node_csv)

                    pick_ps  = runner.make_pose_stamped(pick_xyz,  grasp_orient)
                    place_ps = runner.make_pose_stamped(place_xyz, grasp_orient)

                    print(f"[runner][{scen_key}/{mode_key}] trial {t+1}/{n_trials}  "
                          f"seed={seed}  pick={pick_xyz}  place={place_xyz}")
                    pick_ok, pick_msg = runner.send_pick(pick_ps, pick_to)
                    place_ok = False
                    place_msg = "skipped (pick failed)"
                    if pick_ok:
                        place_ok, place_msg = runner.send_place(place_ps, place_to)
                    if not (pick_ok and place_ok):
                        failures += 1
                        print(f"[runner][{scen_key}/{mode_key}] trial {t}: "
                              f"pick={pick_ok}({pick_msg!r}) place={place_ok}({place_msg!r})")

                    # Brief settle so the C++ node finishes flushing CSV
                    time.sleep(0.5)

                    # Persist captured JointTrajectory messages for this trial
                    if traj_dir is not None:
                        for step_name, jt in runner.consume_trajectories():
                            out_path = traj_dir / (
                                f"{run_id}_{scen_key}_{mode_key}_{t}_{step_name}.csv"
                            )
                            write_trajectory_csv(out_path, jt)
                    else:
                        runner.consume_trajectories()  # discard

                    new_rows, header = read_new_rows(node_csv, pre_offset)
                    if not new_rows:
                        print(f"[runner][{scen_key}/{mode_key}] trial {t}: "
                              f"WARNING — no new CSV rows appended")
                    write_cumulative_rows(out_csv, new_rows, header, meta.as_dict())

                    completed += 1
                    elapsed = time.time() - t_start
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    eta = (total_trials - completed) / rate if rate > 0 else float("inf")
                    print(f"[runner] progress {completed}/{total_trials}  "
                          f"failures={failures}  ETA {eta/60.0:.1f} min")

            if obstacle_enabled and "obstacle" in scen:
                print(f"[runner][{scen_key}] removing obstacle")
                runner.remove_obstacle(scen["obstacle"])

    finally:
        runner.destroy_node()
        rclpy.shutdown()

    print(f"\n[runner] DONE  completed={completed}  failures={failures}  "
          f"elapsed={(time.time()-t_start)/60.0:.1f} min")
    print(f"[runner] cumulative CSV: {out_csv}")
    return 0 if failures == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="2026 KIIE pick-place experiment automation runner",
    )
    default_cfg = "/home/mk/dev_ws/robot_arm/UR/ur_setup_ws/src/pick_place_module/config/scenarios.yaml"
    p.add_argument("--scenarios-file", default=default_cfg,
                   help="path to scenarios.yaml")
    p.add_argument("--node-csv", required=True,
                   help="CSV file pick_place_node was launched with "
                        "(experiment_csv_path) — runner reads new rows from this")
    p.add_argument("--out", required=True,
                   help="cumulative output CSV (one row per appended pick_place_node row, "
                        "augmented with run_id/scenario/mode/trial_idx/seed/pose)")
    p.add_argument("--traj-dir", default=None,
                   help="directory to persist per-trial JointTrajectory recordings "
                        "(needed by plot_results.py Figure 5). "
                        "Default: skip trajectory recording.")
    p.add_argument("--scenarios", default=None,
                   help="comma-separated scenario keys to run (default: all)")
    p.add_argument("--modes", default=None,
                   help="comma-separated mode keys to run (default: all)")
    p.add_argument("--trials-per-cell", type=int, default=None,
                   help="override trials per (scenario, mode); default from yaml")
    p.add_argument("--run-id", default=None,
                   help="override run_id; default = timestamp + uuid")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return run_all(args)


if __name__ == "__main__":
    sys.exit(main())
