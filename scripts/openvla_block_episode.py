#!/usr/bin/env python3
"""
Run one hard-coded colored-block pick-and-place episode for OpenVLA data capture.

Prerequisites:
  1. Start Gazebo / MoveIt:
     ros2 launch ur_setup_bringup ur_sim_moveit_robotiq_ur16e.launch.py \
       object_type:=colored_blocks \
       enable_wrist_camera_image:=false \
       enable_third_person_camera:=true

  2. Start the pick/place action server:
     ros2 launch pick_place_module pick_place.launch.py \
       use_sim_time:=true trigger_mode:=0 enable_logger:=true \
       experiment_mode:=rrt_only

  3. Run this script:
     ros2 run pick_place_module openvla_block_episode.py --block yellow

The script records camera frames plus robot state snapshots in a folder under
~/.ros/openvla_block_episodes by default. Images are saved as binary PPM files
to avoid adding OpenCV / PIL as runtime dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Empty
from tf2_ros import Buffer, TransformException, TransformListener

from pick_place_module.action import Pick, Place


DOWNWARD_QUAT = (0.0, 1.0, 0.0, 0.0)  # x, y, z, w

BLOCK_PRESETS = {
    # Coordinates are expressed in base_link frame. base_link shares world x/y,
    # and is mounted 0.9 m above world z. The z value is tool0 target height.
    "red": {
        "model": "block_red",
        "pick_xyz": (0.42, 0.37, 0.22),
        "place_xyz": (-0.70, 0.35, 0.25),
        "instruction": "Pick up the red block from the green plate and place it on the blue pallet.",
    },
    "yellow": {
        "model": "block_yellow",
        "pick_xyz": (0.66, 0.37, 0.22),
        "place_xyz": (-0.50, 0.35, 0.25),
        "instruction": "Pick up the yellow block from the green plate and place it on the blue pallet.",
    },
}


def _stamp_to_float(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _pose_dict(pose: Pose) -> Dict[str, Any]:
    return {
        "position": {
            "x": pose.position.x,
            "y": pose.position.y,
            "z": pose.position.z,
        },
        "orientation_xyzw": {
            "x": pose.orientation.x,
            "y": pose.orientation.y,
            "z": pose.orientation.z,
            "w": pose.orientation.w,
        },
    }


def _make_pose(xyz: Tuple[float, float, float], quat_xyzw: Tuple[float, float, float, float]) -> Pose:
    pose = Pose()
    pose.position.x = float(xyz[0])
    pose.position.y = float(xyz[1])
    pose.position.z = float(xyz[2])
    pose.orientation.x = float(quat_xyzw[0])
    pose.orientation.y = float(quat_xyzw[1])
    pose.orientation.z = float(quat_xyzw[2])
    pose.orientation.w = float(quat_xyzw[3])
    return pose


def _save_ppm(path: Path, msg: Image) -> bool:
    """Save common ROS Image encodings as binary PPM (P6)."""
    enc = msg.encoding.lower()
    width = int(msg.width)
    height = int(msg.height)
    step = int(msg.step)

    if width <= 0 or height <= 0:
        return False

    data = bytes(msg.data)
    rows: List[bytes] = []

    if enc in ("rgb8", "bgr8"):
        channels = 3
        row_len = width * channels
        for y in range(height):
            row = data[y * step:y * step + row_len]
            if len(row) < row_len:
                return False
            if enc == "bgr8":
                row = b"".join(row[i + 2:i + 3] + row[i + 1:i + 2] + row[i:i + 1]
                               for i in range(0, len(row), 3))
            rows.append(row)
    elif enc in ("rgba8", "bgra8"):
        row_len = width * 4
        for y in range(height):
            row = data[y * step:y * step + row_len]
            if len(row) < row_len:
                return False
            if enc == "rgba8":
                rows.append(b"".join(row[i:i + 3] for i in range(0, len(row), 4)))
            else:
                rows.append(b"".join(row[i + 2:i + 3] + row[i + 1:i + 2] + row[i:i + 1]
                                     for i in range(0, len(row), 4)))
    elif enc in ("mono8", "8uc1"):
        row_len = width
        for y in range(height):
            row = data[y * step:y * step + row_len]
            if len(row) < row_len:
                return False
            rows.append(b"".join(bytes((v, v, v)) for v in row))
    else:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        for row in rows:
            f.write(row)
    return True


class OpenVLABlockEpisode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("openvla_block_episode")
        self.args = args
        self.phase = "init"
        self.latest_image: Optional[Image] = None
        self.latest_joint_state: Optional[JointState] = None
        self.samples: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.frame_idx = 0

        self.episode_id = args.episode_id or f"episode_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.out_dir = Path(args.output_dir).expanduser() / self.episode_id
        self.frames_dir = self.out_dir / "frames"

        self.pick_client = ActionClient(self, Pick, "pick")
        self.place_client = ActionClient(self, Place, "place")
        self.attach_pub = self.create_publisher(Empty, f"/grasp/attach/{args.model_name}", 10)
        self.collision_pub = self.create_publisher(CollisionObject, "/collision_object", 10)

        self.image_sub = self.create_subscription(
            Image, args.image_topic, self._on_image, 10)
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self._on_joint_state, 50)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        period = 1.0 / max(0.1, float(args.sample_hz))
        self.sample_timer = self.create_timer(period, self._record_sample)

    def _on_image(self, msg: Image) -> None:
        self.latest_image = msg

    def _on_joint_state(self, msg: JointState) -> None:
        self.latest_joint_state = msg

    def wait_ready(self) -> bool:
        deadline = time.monotonic() + float(self.args.ready_timeout)
        while time.monotonic() < deadline:
            if self.pick_client.wait_for_server(timeout_sec=0.2) and \
               self.place_client.wait_for_server(timeout_sec=0.2) and \
               self.latest_image is not None and \
               self.latest_joint_state is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def event(self, name: str, **data: Any) -> None:
        row = {
            "wall_time": time.time(),
            "ros_time": _stamp_to_float(self.get_clock().now().to_msg()),
            "event": name,
            **data,
        }
        self.events.append(row)
        self.get_logger().info(f"[event] {name} {data}")

    def _lookup_ee_pose(self) -> Optional[Dict[str, Any]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.args.base_frame,
                self.args.ee_frame,
                Time(),
                timeout=rclpy.duration.Duration(seconds=0.02),
            )
        except (TransformException, Exception):
            return None

        tr = tf.transform.translation
        qr = tf.transform.rotation
        return {
            "frame_id": self.args.base_frame,
            "child_frame_id": self.args.ee_frame,
            "position": {"x": tr.x, "y": tr.y, "z": tr.z},
            "orientation_xyzw": {"x": qr.x, "y": qr.y, "z": qr.z, "w": qr.w},
        }

    def _joint_state_dict(self) -> Dict[str, Any]:
        msg = self.latest_joint_state
        if msg is None:
            return {}
        names = list(msg.name)
        return {
            "stamp": _stamp_to_float(msg.header.stamp),
            "position": {n: float(v) for n, v in zip(names, msg.position)},
            "velocity": {n: float(v) for n, v in zip(names, msg.velocity)},
        }

    def _record_sample(self) -> None:
        if self.phase == "done":
            return

        image_rel = ""
        image_stamp = None
        msg = self.latest_image
        if self.args.record_images and msg is not None:
            image_rel = f"frames/{self.frame_idx:06d}.ppm"
            image_path = self.out_dir / image_rel
            if _save_ppm(image_path, msg):
                image_stamp = _stamp_to_float(msg.header.stamp)
                self.frame_idx += 1
            else:
                image_rel = ""

        sample = {
            "sample_index": len(self.samples),
            "wall_time": time.time(),
            "ros_time": _stamp_to_float(self.get_clock().now().to_msg()),
            "phase": self.phase,
            "instruction": self.args.instruction,
            "image": image_rel,
            "image_topic": self.args.image_topic,
            "image_stamp": image_stamp,
            "ee_pose": self._lookup_ee_pose(),
            "joint_state": self._joint_state_dict(),
        }
        self.samples.append(sample)

    def send_pick(self, pose: Pose) -> Tuple[bool, str]:
        goal = Pick.Goal()
        goal.pick_pose = pose
        return self._send_goal(self.pick_client, goal, self.args.pick_timeout)

    def send_place(self, pose: Pose) -> Tuple[bool, str]:
        goal = Place.Goal()
        goal.place_pose = pose
        return self._send_goal(self.place_client, goal, self.args.place_timeout)

    def _send_goal(self, client: ActionClient, goal: Any, timeout_sec: float) -> Tuple[bool, str]:
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

    def publish_attach_hint(self) -> None:
        if not self.args.publish_attach_hint:
            return
        msg = Empty()
        for _ in range(3):
            self.attach_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)
        self.event("attach_hint_published", model_name=self.args.model_name)

    def remove_grasped_collision_object(self) -> None:
        if not self.args.remove_collision_object:
            return
        ids = [self.args.model_name]
        sdf_model_name = f"colored_block_{self.args.block}"
        if sdf_model_name not in ids:
            ids.append(sdf_model_name)
        for obj_id in ids:
            msg = CollisionObject()
            msg.header.frame_id = self.args.base_frame
            msg.id = obj_id
            msg.operation = CollisionObject.REMOVE
            for _ in range(3):
                self.collision_pub.publish(msg)
                rclpy.spin_once(self, timeout_sec=0.05)
        self.event("collision_object_remove_published", ids=ids)

    def finalize(self, success: bool, message: str, pick_pose: Pose, place_pose: Pose) -> None:
        self.phase = "done"
        self._fill_next_actions()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "episode_id": self.episode_id,
            "success": bool(success),
            "message": message,
            "instruction": self.args.instruction,
            "object": {
                "color": self.args.block,
                "model_name": self.args.model_name,
            },
            "pick_pose_base_link": _pose_dict(pick_pose),
            "place_pose_base_link": _pose_dict(place_pose),
            "image_topic": self.args.image_topic,
            "sample_hz": self.args.sample_hz,
            "events": self.events,
            "format_note": (
                "action_delta_ee_xyz is the next-sample minus current-sample "
                "tool0 position in base_link frame. Convert as needed for OpenVLA."
            ),
        }
        with (self.out_dir / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        with (self.out_dir / "samples.jsonl").open("w") as f:
            for row in self.samples:
                f.write(json.dumps(row) + "\n")

        self.get_logger().info(
            f"episode saved: {self.out_dir} samples={len(self.samples)} frames={self.frame_idx}")

    def _fill_next_actions(self) -> None:
        for idx, row in enumerate(self.samples):
            cur = row.get("ee_pose")
            nxt = self.samples[idx + 1].get("ee_pose") if idx + 1 < len(self.samples) else None
            if not cur or not nxt:
                row["action_delta_ee_xyz"] = None
                row["action_next_ee_pose"] = None
                continue
            cp = cur["position"]
            np = nxt["position"]
            row["action_delta_ee_xyz"] = {
                "dx": np["x"] - cp["x"],
                "dy": np["y"] - cp["y"],
                "dz": np["z"] - cp["z"],
            }
            row["action_next_ee_pose"] = nxt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hard-coded OpenVLA block pick-and-place episode runner.")
    parser.add_argument("--block", choices=sorted(BLOCK_PRESETS), default="yellow")
    parser.add_argument("--model-name", default="", help="Gazebo model name; defaults from --block")
    parser.add_argument("--image-topic", default="/openvla_camera/image_raw")
    parser.add_argument("--output-dir", default=str(Path.home() / ".ros" / "openvla_block_episodes"))
    parser.add_argument("--episode-id", default="")
    parser.add_argument("--sample-hz", type=float, default=2.0)
    parser.add_argument("--record-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ready-timeout", type=float, default=30.0)
    parser.add_argument("--pick-timeout", type=float, default=180.0)
    parser.add_argument("--place-timeout", type=float, default=180.0)
    parser.add_argument("--settle-sec", type=float, default=1.0)
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--ee-frame", default="tool0")
    parser.add_argument("--publish-attach-hint", action=argparse.BooleanOptionalAction, default=True,
                        help="publish /grasp/attach/<model> after pick if a bridge/plugin exists")
    parser.add_argument("--remove-collision-object", action=argparse.BooleanOptionalAction, default=True,
                        help="publish CollisionObject.REMOVE after pick so the grasped object no longer blocks place planning")

    parser.add_argument("--pick-x", type=float, default=math.nan)
    parser.add_argument("--pick-y", type=float, default=math.nan)
    parser.add_argument("--pick-z", type=float, default=math.nan)
    parser.add_argument("--place-x", type=float, default=math.nan)
    parser.add_argument("--place-y", type=float, default=math.nan)
    parser.add_argument("--place-z", type=float, default=math.nan)
    parser.add_argument("--qx", type=float, default=DOWNWARD_QUAT[0])
    parser.add_argument("--qy", type=float, default=DOWNWARD_QUAT[1])
    parser.add_argument("--qz", type=float, default=DOWNWARD_QUAT[2])
    parser.add_argument("--qw", type=float, default=DOWNWARD_QUAT[3])
    parser.add_argument("--instruction", default="")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    preset = BLOCK_PRESETS[args.block]
    args.model_name = args.model_name or preset["model"]
    args.instruction = args.instruction or preset["instruction"]

    pick_xyz = list(preset["pick_xyz"])
    if not math.isnan(args.pick_x):
        pick_xyz[0] = args.pick_x
    if not math.isnan(args.pick_y):
        pick_xyz[1] = args.pick_y
    if not math.isnan(args.pick_z):
        pick_xyz[2] = args.pick_z
    place_xyz = list(preset["place_xyz"])
    if not math.isnan(args.place_x):
        place_xyz[0] = args.place_x
    if not math.isnan(args.place_y):
        place_xyz[1] = args.place_y
    if not math.isnan(args.place_z):
        place_xyz[2] = args.place_z
    place_xyz = tuple(place_xyz)
    quat = (args.qx, args.qy, args.qz, args.qw)

    rclpy.init()
    node = OpenVLABlockEpisode(args)
    success = False
    message = ""
    pick_pose = _make_pose(tuple(pick_xyz), quat)
    place_pose = _make_pose(place_xyz, quat)

    try:
        node.event("episode_start", pick_xyz=pick_xyz, place_xyz=place_xyz)
        if not node.wait_ready():
            message = "required action servers, image topic, or joint_states not ready"
            node.get_logger().error(message)
            return 2

        node.phase = "before_pick"
        end = time.monotonic() + max(0.0, args.settle_sec)
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)

        node.phase = "pick"
        node.event("pick_goal_sent")
        pick_ok, pick_msg = node.send_pick(pick_pose)
        node.event("pick_result", success=pick_ok, message=pick_msg)
        if not pick_ok:
            message = f"pick failed: {pick_msg}"
            return 3

        node.phase = "between_pick_place"
        node.publish_attach_hint()
        node.remove_grasped_collision_object()
        end = time.monotonic() + max(0.0, args.settle_sec)
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)

        node.phase = "place"
        node.event("place_goal_sent")
        place_ok, place_msg = node.send_place(place_pose)
        node.event("place_result", success=place_ok, message=place_msg)
        if not place_ok:
            message = f"place failed: {place_msg}"
            return 4

        success = True
        message = "episode completed"
        return 0
    finally:
        node.event("episode_end", success=success, message=message)
        node.finalize(success, message, pick_pose, place_pose)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
