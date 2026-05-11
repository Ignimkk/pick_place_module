#!/usr/bin/env python3
"""Shared ROS2/Gazebo KPI experiment runner for pick_place_node.

This module assumes Gazebo/MoveIt, pick_place_node, and trajopt_server_node are
already running. It sets pick_place_node runtime parameters for one experiment
mode, sends deterministic pick/place action goals, and copies the KPI rows that
pick_place_node appends to its experiment CSV into a mode-specific result CSV
with trial metadata.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rcl_interfaces.srv import GetParameters, ListParameters, SetParameters
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    ROS_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:
    rclpy = None
    PoseStamped = Any
    GetParameters = None
    ListParameters = None
    SetParameters = None
    ActionClient = Any
    Parameter = None
    Node = object
    ROS_IMPORT_ERROR = exc


DEFAULT_PICK_X = (0.30, 0.50)
DEFAULT_PICK_Y = (0.20, 0.50)
DEFAULT_PICK_Z = (0.30, 0.50)
DEFAULT_PLACE_X = (-0.50, -0.30)
DEFAULT_PLACE_Y = (0.30, 0.50)
DEFAULT_PLACE_Z = (0.30, 0.50)
DEFAULT_ORIENTATION = {"x": 0.0, "y": 1.0, "z": 0.0, "w": 0.0}
DEFAULT_NODE_CSV = Path.home() / ".ros" / "pick_place_exp" / "node_data.csv"
DEFAULT_SEED = 20260504

META_FIELDS = [
    "run_id",
    "timestamp",
    "mode",
    "scenario",
    "has_obstacle",
    "trial_idx",
    "seed",
    "pick_x",
    "pick_y",
    "pick_z",
    "place_x",
    "place_y",
    "place_z",
    "trajopt_N",
    "trajopt_use_reduced",
    "trajopt_use_free_t",
    "t_init_sec",
]

ACTION_FIELDS = [
    "trial_action_success",
    "pick_action_success",
    "pick_action_message",
    "place_action_success",
    "place_action_message",
    "new_node_rows",
]

DEFAULT_NODE_FIELDS = [
    "trial_id",
    "step_name",
    "experiment_mode",
    "planner_id",
    "success",
    "fallback_used",
    "ik_time_sec",
    "rrt_planning_sec",
    "shortcut_time_sec",
    "initial_guess_time_sec",
    "solve_time_sec",
    "total_compute_sec",
    "exec_wait_sec",
    "num_rrt_points",
    "num_shortcut_waypoints",
    "num_optimized_points",
    "trajectory_duration_sec",
    "joint_path_length",
    "mean_joint_velocity",
    "max_joint_velocity",
    "mean_joint_acceleration",
    "max_joint_acceleration",
    "mean_joint_jerk",
    "max_joint_jerk",
    "mean_torque",
    "max_torque",
    "mean_torque_rate",
    "max_torque_rate",
    "max_constraint_violation",
    "final_cost",
    "solver_status",
    "message",
]

REQUIRED_PICK_PLACE_PARAMS = [
    "experiment_csv_path",
    "experiment_mode",
    "use_trajopt",
    "trajopt_use_reduced",
    "trajopt_use_free_t",
    "trajopt_N",
    "t_init_sec",
]


def parse_range(text: str) -> tuple[float, float]:
    parts = [float(p.strip()) for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected 'lo,hi', got: {text}")
    lo, hi = parts
    if lo > hi:
        raise argparse.ArgumentTypeError(f"range low > high: {text}")
    return lo, hi


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _same_path(a: Path, b: Path) -> bool:
    return a.expanduser().resolve(strict=False) == b.expanduser().resolve(strict=False)


def read_new_rows(node_csv: Path, byte_offset: int) -> tuple[list[dict[str, str]], list[str]]:
    if not node_csv.exists():
        return [], []

    with node_csv.open("r", newline="") as f:
        header_line = f.readline().strip()
        header = [h.strip() for h in header_line.split(",") if h.strip()]
        f.seek(byte_offset)
        new_text = f.read()

    if not header:
        return [], []

    rows: list[dict[str, str]] = []
    for row in csv.DictReader(
        [ln for ln in new_text.splitlines() if ln.strip()],
        fieldnames=header,
    ):
        if row.get("trial_id") == "trial_id":
            continue
        rows.append(row)
    return rows, header


def append_rows(
    out_csv: Path,
    rows: list[dict[str, str]],
    node_header: list[str],
    meta: dict[str, Any],
    action_meta: dict[str, Any],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = node_header if node_header else DEFAULT_NODE_FIELDS
    fieldnames = META_FIELDS + ACTION_FIELDS + header
    write_header = not out_csv.exists() or _file_size(out_csv) == 0

    with out_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        if rows:
            for row in rows:
                writer.writerow({**meta, **action_meta, **row})
            return

        fallback = {name: "" for name in header}
        fallback.update({
            "experiment_mode": meta["mode"],
            "success": "False",
            "solver_status": "no_node_csv_rows",
            "message": "pick_place_node did not append KPI rows for this trial",
        })
        writer.writerow({**meta, **action_meta, **fallback})


class GazeboExperimentNode(Node):
    def __init__(self, target_node: str) -> None:
        super().__init__("gazebo_kpi_experiment_runner")
        self.target_node = target_node.rstrip("/")
        self._set_param_cli = self.create_client(
            SetParameters, f"{self.target_node}/set_parameters")
        self._get_param_cli = self.create_client(
            GetParameters, f"{self.target_node}/get_parameters")
        self._list_param_cli = self.create_client(
            ListParameters, f"{self.target_node}/list_parameters")

        from pick_place_module.action import Pick, Place

        self._pick_action = ActionClient(self, Pick, "pick")
        self._place_action = ActionClient(self, Place, "place")

    def wait_until_ready(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            names = set(self.get_node_names())
            if self.target_node.lstrip("/") in names or self.target_node in names:
                break
            rclpy.spin_once(self, timeout_sec=0.2)
        else:
            self.get_logger().error(f"{self.target_node} is not visible")
            return False

        remaining = max(0.1, deadline - time.monotonic())
        if not self._set_param_cli.wait_for_service(timeout_sec=remaining):
            self.get_logger().error(f"{self.target_node}/set_parameters unavailable")
            return False
        if not self._get_param_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(f"{self.target_node}/get_parameters unavailable")
        if not self._list_param_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(f"{self.target_node}/list_parameters unavailable")

        if not self._pick_action.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("pick action server unavailable")
            return False
        if not self._place_action.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("place action server unavailable")
            return False
        return True

    def list_parameters(self) -> list[str]:
        req = ListParameters.Request()
        req.depth = 0
        fut = self._list_param_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
        if fut.result() is None:
            return []
        return list(fut.result().result.names)

    def get_string_parameter(self, name: str) -> str:
        req = GetParameters.Request()
        req.names = [name]
        fut = self._get_param_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
        if fut.result() is None or not fut.result().values:
            return ""
        return fut.result().values[0].string_value

    def set_parameters_checked(self, params: dict[str, Any], timeout_sec: float) -> bool:
        req = SetParameters.Request()
        req.parameters = [
            Parameter(name=name, value=value).to_parameter_msg()
            for name, value in params.items()
        ]
        fut = self._set_param_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout_sec)
        if fut.result() is None:
            self.get_logger().error("set_parameters timed out")
            return False

        ok = True
        for name, result in zip(params.keys(), fut.result().results):
            if not result.successful:
                self.get_logger().error(
                    f"set parameter failed: {name}={params[name]!r}: {result.reason}")
                ok = False
        return ok

    def make_pose(
        self,
        xyz: tuple[float, float, float],
        orientation: dict[str, float],
    ) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(xyz[0])
        pose.pose.position.y = float(xyz[1])
        pose.pose.position.z = float(xyz[2])
        pose.pose.orientation.x = float(orientation["x"])
        pose.pose.orientation.y = float(orientation["y"])
        pose.pose.orientation.z = float(orientation["z"])
        pose.pose.orientation.w = float(orientation["w"])
        return pose

    def send_pick(self, pose: PoseStamped, timeout_sec: float) -> tuple[bool, str]:
        from pick_place_module.action import Pick

        goal = Pick.Goal()
        goal.pick_pose = pose.pose
        return self._send_goal(self._pick_action, goal, timeout_sec)

    def send_place(self, pose: PoseStamped, timeout_sec: float) -> tuple[bool, str]:
        from pick_place_module.action import Place

        goal = Place.Goal()
        goal.place_pose = pose.pose
        return self._send_goal(self._place_action, goal, timeout_sec)

    def _send_goal(self, client: ActionClient, goal: Any, timeout_sec: float) -> tuple[bool, str]:
        send_fut = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut, timeout_sec=15.0)
        if send_fut.result() is None:
            return False, "goal send timed out"
        goal_handle = send_fut.result()
        if not goal_handle.accepted:
            return False, "goal rejected"

        result_fut = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_fut, timeout_sec=timeout_sec)
        if result_fut.result() is None:
            return False, "result timed out"

        result = result_fut.result().result
        return bool(result.success), str(getattr(result, "message", ""))


def build_common_parser(mode: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--append", action="store_true", help="append to --output instead of replacing it")
    parser.add_argument("--node-csv", default=str(DEFAULT_NODE_CSV))
    parser.add_argument("--target-node", default="/pick_place_node")
    parser.add_argument(
        "--launch-params-only",
        action="store_true",
        help=(
            "do not get/set pick_place_node parameters; trust the values passed "
            "to pick_place.launch.py"
        ),
    )
    parser.add_argument("--skip-node-csv-check", action="store_true")
    parser.add_argument("--ready-timeout", type=float, default=20.0)
    parser.add_argument("--param-timeout", type=float, default=5.0)
    parser.add_argument("--pick-timeout", type=float, default=120.0)
    parser.add_argument("--place-timeout", type=float, default=120.0)
    parser.add_argument("--settle-sec", type=float, default=1.0)
    parser.add_argument("--scenario", default="S1")
    parser.add_argument("--obstacle", choices=["yes", "no"], default="no")
    parser.add_argument("--pick-x-range", type=parse_range, default=DEFAULT_PICK_X)
    parser.add_argument("--pick-y-range", type=parse_range, default=DEFAULT_PICK_Y)
    parser.add_argument("--pick-z-range", type=parse_range, default=DEFAULT_PICK_Z)
    parser.add_argument("--place-x-range", type=parse_range, default=DEFAULT_PLACE_X)
    parser.add_argument("--place-y-range", type=parse_range, default=DEFAULT_PLACE_Y)
    parser.add_argument("--place-z-range", type=parse_range, default=DEFAULT_PLACE_Z)
    parser.add_argument("--trajopt-N", type=int, default=6)
    parser.add_argument("--t-init-sec", type=float, default=3.0)
    parser.add_argument("--full", action="store_true", help="use full TrajOpt variables instead of reduced")
    parser.add_argument("--fixed-t", action="store_true", help="disable free final-time optimization")
    parser.add_argument("--planner-id", default="", help="optional MoveIt planner_id for rrt_trajopt")
    parser.add_argument("--planning-time", type=float, default=0.0, help="optional MoveIt planning_time override")
    parser.add_argument("--num-planning-attempts", type=int, default=0)
    parser.set_defaults(mode=mode)
    return parser


def _sample_xyz(
    rng: random.Random,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
) -> tuple[float, float, float]:
    return (
        rng.uniform(*x_range),
        rng.uniform(*y_range),
        rng.uniform(*z_range),
    )


def _mode_params(args: argparse.Namespace) -> dict[str, Any]:
    params: dict[str, Any] = {
        "experiment_mode": args.mode,
        "use_trajopt": True,
        "trajopt_use_reduced": not args.full,
        "trajopt_use_free_t": not args.fixed_t,
        "trajopt_N": int(args.trajopt_N),
        "t_init_sec": float(args.t_init_sec),
    }
    if args.planner_id:
        params["planner_id"] = args.planner_id
    if args.planning_time > 0.0:
        params["planning_time"] = float(args.planning_time)
    if args.num_planning_attempts > 0:
        params["num_planning_attempts"] = int(args.num_planning_attempts)
    return params


def _default_output(mode: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parents[1] / "results" / f"{stamp}_{mode}_gazebo_kpi.csv"


def run_gazebo_experiment(args: argparse.Namespace) -> int:
    if rclpy is None:
        print(
            "[ERROR] ROS2 Python modules are not available. Source ROS2 and "
            "the project workspaces first, for example:\n"
            "  source /opt/ros/humble/setup.bash\n"
            "  source ur_setup_ws/install/setup.bash\n"
            "  source trajopt_ws/install/setup.bash\n"
            f"Original import error: {ROS_IMPORT_ERROR}",
            file=sys.stderr,
        )
        return 10

    if args.trials <= 0:
        raise ValueError("--trials must be positive")

    node_csv = Path(args.node_csv).expanduser()
    out_csv = Path(args.output).expanduser() if args.output else _default_output(args.mode)
    if out_csv.exists() and not args.append:
        out_csv.unlink()

    run_id = args.run_id or f"{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    params = _mode_params(args)

    print("=" * 72)
    print(f"  Gazebo/ROS2 KPI experiment: {args.mode}")
    print(f"  run_id     : {run_id}")
    print(f"  trials     : {args.trials}")
    print(f"  output CSV : {out_csv}")
    print(f"  node CSV   : {node_csv}")
    print("=" * 72)

    rclpy.init()
    node = GazeboExperimentNode(args.target_node)
    failures = 0

    try:
        if not node.wait_until_ready(args.ready_timeout):
            return 2

        if args.launch_params_only:
            print("[INFO] launch-params-only: skipping pick_place_node get/set parameter calls")
        elif not args.skip_node_csv_check:
            declared = set(node.list_parameters())
            missing = [name for name in REQUIRED_PICK_PLACE_PARAMS if name not in declared]
            if missing:
                print(
                    "[ERROR] /pick_place_node is running, but it does not expose "
                    "the experiment parameters required by this runner.\n"
                    f"  missing: {', '.join(missing)}\n"
                    "This usually means an old pick_place_node is still running "
                    "or the launch terminal did not source the rebuilt "
                    "ur_setup_ws/install/setup.bash.\n"
                    "Restart the launch after killing old ROS/Gazebo processes, then check:\n"
                    "  ros2 pkg prefix pick_place_module\n"
                    "  ros2 param list /pick_place_node | grep -E "
                    "'experiment_mode|use_trajopt|trajopt_N|t_init_sec'",
                    file=sys.stderr,
                )
                return 6

            actual_csv = node.get_string_parameter("experiment_csv_path")
            if not actual_csv:
                print(
                    "[ERROR] /pick_place_node experiment_csv_path is empty. "
                    "Launch pick_place_node with "
                    f"experiment_csv_path:={node_csv}",
                    file=sys.stderr,
                )
                return 3
            if not _same_path(Path(actual_csv), node_csv):
                print(
                    "[ERROR] --node-csv does not match /pick_place_node "
                    f"experiment_csv_path\n  node param: {actual_csv}\n"
                    f"  --node-csv : {node_csv}",
                    file=sys.stderr,
                )
                return 4

        if not args.launch_params_only and not node.set_parameters_checked(params, args.param_timeout):
            return 5
        time.sleep(max(0.0, args.settle_sec))

        started = time.monotonic()
        for trial_idx in range(1, args.trials + 1):
            seed = args.seed + trial_idx - 1
            rng = random.Random(seed)
            pick_xyz = _sample_xyz(rng, args.pick_x_range, args.pick_y_range, args.pick_z_range)
            place_xyz = _sample_xyz(rng, args.place_x_range, args.place_y_range, args.place_z_range)

            meta = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "mode": args.mode,
                "scenario": args.scenario,
                "has_obstacle": args.obstacle,
                "trial_idx": trial_idx,
                "seed": seed,
                "pick_x": f"{pick_xyz[0]:.6f}",
                "pick_y": f"{pick_xyz[1]:.6f}",
                "pick_z": f"{pick_xyz[2]:.6f}",
                "place_x": f"{place_xyz[0]:.6f}",
                "place_y": f"{place_xyz[1]:.6f}",
                "place_z": f"{place_xyz[2]:.6f}",
                "trajopt_N": args.trajopt_N,
                "trajopt_use_reduced": str(not args.full),
                "trajopt_use_free_t": str(not args.fixed_t),
                "t_init_sec": f"{args.t_init_sec:.6f}",
            }

            offset = _file_size(node_csv)
            pick_pose = node.make_pose(pick_xyz, DEFAULT_ORIENTATION)
            place_pose = node.make_pose(place_xyz, DEFAULT_ORIENTATION)

            print(
                f"[{args.mode}] trial {trial_idx}/{args.trials} "
                f"seed={seed} pick={tuple(round(v, 4) for v in pick_xyz)} "
                f"place={tuple(round(v, 4) for v in place_xyz)}"
            )

            pick_ok, pick_msg = node.send_pick(pick_pose, args.pick_timeout)
            if pick_ok:
                place_ok, place_msg = node.send_place(place_pose, args.place_timeout)
            else:
                place_ok, place_msg = False, "skipped because pick failed"

            time.sleep(0.5)
            new_rows, header = read_new_rows(node_csv, offset)
            action_meta = {
                "trial_action_success": str(pick_ok and place_ok),
                "pick_action_success": str(pick_ok),
                "pick_action_message": pick_msg,
                "place_action_success": str(place_ok),
                "place_action_message": place_msg,
                "new_node_rows": len(new_rows),
            }
            append_rows(out_csv, new_rows, header, meta, action_meta)

            if not (pick_ok and place_ok):
                failures += 1
            elapsed = time.monotonic() - started
            rate = trial_idx / elapsed if elapsed > 0.0 else 0.0
            eta = (args.trials - trial_idx) / rate if rate > 0.0 else 0.0
            print(
                f"  rows={len(new_rows)} pick={pick_ok} place={place_ok} "
                f"failures={failures} ETA={eta / 60.0:.1f}min"
            )
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print(f"[DONE] {args.mode}: failures={failures}/{args.trials}")
    print(f"[CSV] {out_csv}")
    return 0 if failures == 0 else 1
