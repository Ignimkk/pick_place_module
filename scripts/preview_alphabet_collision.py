#!/usr/bin/env python3
"""
preview_alphabet_collision.py — alphabet 의 collision box 분해를 RViz 에서 시각화.

각 letter 에 대해 2 가지 marker 를 발행:
  (1) visual:    원본 STL mesh (반투명 회색)
  (2) collision: box 분해 (반투명 빨강)

8 글자를 격자로 배치 (x 축 따라 일렬). 사용자는 RViz 에서
  MarkerArray Display → /alphabet_collision_preview 추가 후
  '회색 STL' 안에 '빨간 box' 들이 잘 들어맞는지 확인.

설계 의도:
  - 실제 sim 의 letter 위치와 무관 — 검토 전용
  - sys.path 를 거쳐 ur_setup_ws/models/_alphabet_boxes.py 의 데이터 직접 import
  - mesh URI 는 ROS package:// scheme 으로 변환 (RViz 가 인식)

실행:
  ros2 run pick_place_module preview_alphabet_collision.py
  rviz2 → Fixed Frame = world (또는 base_link) → Add MarkerArray
         → Topic = /alphabet_collision_preview
"""

import math
import os
import sys

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Vector3
from std_msgs.msg import ColorRGBA

# ── ALPHABET_BOXES 데이터 import ───────────────────────────────────────────
# models/_alphabet_boxes.py 는 ur_setup_bringup 패키지 share/models/ 에 설치됨.
def _load_alphabet_boxes():
    candidates = []
    try:
        share = get_package_share_directory("ur_setup_bringup")
        candidates.append(os.path.join(share, "models"))
    except Exception:
        pass
    # 소스 트리 fallback (개발 시).
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.normpath(
        os.path.join(here, "..", "..", "ur_setup_ws", "models")))
    for d in candidates:
        f = os.path.join(d, "_alphabet_boxes.py")
        if os.path.isfile(f):
            sys.path.insert(0, d)
            from _alphabet_boxes import ALPHABET_BOXES   # type: ignore
            return ALPHABET_BOXES
    raise RuntimeError(
        f"_alphabet_boxes.py not found in any of: {candidates}")

ALPHABET_BOXES = _load_alphabet_boxes()


def yaw_to_quat(yaw):
    """단순 z-axis yaw → (x, y, z, w) quaternion."""
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


class AlphabetCollisionPreview(Node):
    def __init__(self):
        super().__init__("alphabet_collision_preview")
        self.declare_parameter("frame_id", "world")
        self.declare_parameter("origin_x", 1.5)       # 작업 영역 밖
        self.declare_parameter("origin_y", -0.5)
        self.declare_parameter("origin_z", 1.0)
        self.declare_parameter("pitch_x",  0.20)      # letter 간 가로 간격
        self.declare_parameter("publish_period_sec", 1.0)

        self.pub = self.create_publisher(
            MarkerArray, "alphabet_collision_preview", 1)
        period = self.get_parameter("publish_period_sec").value
        self.create_timer(period, self.publish_markers)

        self.frame_id = self.get_parameter("frame_id").value
        self.ox = self.get_parameter("origin_x").value
        self.oy = self.get_parameter("origin_y").value
        self.oz = self.get_parameter("origin_z").value
        self.px = self.get_parameter("pitch_x").value
        self.get_logger().info(
            f"preview frame_id='{self.frame_id}' origin=({self.ox:.2f}, "
            f"{self.oy:.2f}, {self.oz:.2f}) pitch_x={self.px:.3f}")

    def publish_markers(self):
        ma = MarkerArray()
        marker_id = 0
        stamp = self.get_clock().now().to_msg()

        for col, (letter, boxes) in enumerate(ALPHABET_BOXES.items()):
            letter_x = self.ox + col * self.px
            letter_y = self.oy
            letter_z = self.oz

            # ── (1) STL visual marker (반투명 회색) ──────────────────
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = stamp
            m.ns = f"visual_{letter}"
            m.id = marker_id; marker_id += 1
            m.type = Marker.MESH_RESOURCE
            m.action = Marker.ADD
            m.pose.position.x = letter_x
            m.pose.position.y = letter_y
            m.pose.position.z = letter_z
            m.pose.orientation.w = 1.0
            m.scale = Vector3(x=0.001, y=0.001, z=0.001)  # STL 단위 mm
            m.color = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.35)
            m.mesh_resource = (
                f"package://ur_setup_bringup/meshes/alphabet/TA-{letter}.stl")
            m.mesh_use_embedded_materials = False
            ma.markers.append(m)

            # ── (2) collision box markers (반투명 빨강) ──────────────
            for i, (cx, cy, cz, sx, sy, sz, yaw) in enumerate(boxes):
                qx, qy, qz, qw = yaw_to_quat(yaw)
                m = Marker()
                m.header.frame_id = self.frame_id
                m.header.stamp = stamp
                m.ns = f"collision_{letter}"
                m.id = marker_id; marker_id += 1
                m.type = Marker.CUBE
                m.action = Marker.ADD
                m.pose.position.x = letter_x + cx
                m.pose.position.y = letter_y + cy
                m.pose.position.z = letter_z + cz
                m.pose.orientation.x = qx
                m.pose.orientation.y = qy
                m.pose.orientation.z = qz
                m.pose.orientation.w = qw
                m.scale = Vector3(x=sx, y=sy, z=sz)
                m.color = ColorRGBA(r=0.9, g=0.15, b=0.15, a=0.45)
                ma.markers.append(m)

            # ── 라벨 (글자명) ───────────────────────────────────────
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = stamp
            m.ns = f"label_{letter}"
            m.id = marker_id; marker_id += 1
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.pose.position.x = letter_x
            m.pose.position.y = letter_y
            m.pose.position.z = letter_z + 0.10
            m.pose.orientation.w = 1.0
            m.scale.z = 0.03
            m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            m.text = f"{letter} ({len(boxes)} boxes)"
            ma.markers.append(m)

        self.pub.publish(ma)


def main():
    rclpy.init()
    node = AlphabetCollisionPreview()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
