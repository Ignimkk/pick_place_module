"""
pick_place.launch.py

pick_place_module 의 노드를 기동:
  - pick_place_node    : MoveIt2 기반 pick / place action server
  - goal_relay_node    : pick_goal / place_goal 토픽 → action 변환
  - motion_logger_node : EE 이동 거리 + 관절 이동량 기록 (선택, 기본 활성)

전제조건:
  ur_setup_bringup 의 ur_sim_moveit_robotiq_ur16e.launch.py 가 먼저 실행되어
  move_group, robot_state_publisher, controller_manager 가 구동 중이어야 함.

런치 인수:
  trigger_mode    : 0(기본) = 토픽 하나만 받아도 즉시 해당 action 실행
                    1        = pick_goal + place_goal 모두 수신 후 순차 실행
  use_sim_time    : true(기본) / false
  enable_logger   : true(기본) = motion_logger_node 활성화
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ur_moveit_config.launch_common import load_yaml


def generate_launch_description():
    pkg_share     = get_package_share_directory("pick_place_module")
    bringup_share = get_package_share_directory("ur_setup_bringup")

    config     = os.path.join(pkg_share,     "config", "pick_place_params.yaml")
    kinematics = os.path.join(bringup_share, "config", "kinematics.yaml")

    trigger_mode   = LaunchConfiguration("trigger_mode")
    use_sim_time   = LaunchConfiguration("use_sim_time")
    enable_logger  = LaunchConfiguration("enable_logger")

    return LaunchDescription([
        # --------------------------------------------------------
        # 런치 인수 선언
        # --------------------------------------------------------
        DeclareLaunchArgument(
            "trigger_mode",
            default_value="0",
            description=(
                "실행 트리거 모드: "
                "0=pick_goal 또는 place_goal 중 하나만 받아도 즉시 해당 action 실행, "
                "1=pick_goal + place_goal 모두 수신 후 pick → place 순차 실행"
            ),
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="시뮬레이션 시간 사용 여부 (Gazebo 연동 시 true)",
        ),
        DeclareLaunchArgument(
            "enable_logger",
            default_value="true",
            description="motion_logger_node 활성화 여부 (EE 이동 거리 + 관절 이동량 기록)",
        ),
        DeclareLaunchArgument(
            "experiment_mode",
            default_value="rrt_trajopt",
            description="실험 모드: rrt_only / trajopt_only / rrt_trajopt",
        ),
        DeclareLaunchArgument(
            "experiment_csv_path",
            default_value="",
            description="실험 결과 CSV 저장 경로",
        ),

        # --------------------------------------------------------
        # pick_place_node
        # MoveIt2 RRTConnect + Cartesian 경로 기반 action server
        #
        # kinematics.yaml 로드 이유:
        #   MoveGroupInterface 가 IK 플러그인(KDLKinematicsPlugin)을 인식하고
        #   Cartesian pose 계획을 실행하기 위해 필수.
        #   누락 시: "No kinematics plugins defined" 경고 → planning abort.
        # --------------------------------------------------------
        Node(
            package="pick_place_module",
            executable="pick_place_node",
            name="pick_place_node",
            output="screen",
            parameters=[
                config,
                kinematics,
                {
                    "use_sim_time": use_sim_time,
                    "experiment_mode": LaunchConfiguration("experiment_mode"),
                    "experiment_csv_path": LaunchConfiguration("experiment_csv_path"),
                }
            ],
        ),

        # --------------------------------------------------------
        # goal_relay_node
        # pick_goal / place_goal 토픽 수신 → pick / place action goal 전송
        # trigger_mode 는 런치 인수로 오버라이드 (YAML 기본값보다 우선)
        # --------------------------------------------------------
        Node(
            package="pick_place_module",
            executable="goal_relay_node",
            name="goal_relay_node",
            output="screen",
            parameters=[
                config,
                {
                    "trigger_mode": trigger_mode,
                    "use_sim_time": use_sim_time,
                },
            ],
        ),

        # --------------------------------------------------------
        # motion_logger_node
        # /joint_states + TF2(base_link → tool0) 기반
        # EE 이동 거리 및 관절 이동량을 CSV로 기록
        # enable_logger:=false 로 비활성화 가능
        # --------------------------------------------------------
        Node(
            package="pick_place_module",
            executable="motion_logger_node",
            name="motion_logger_node",
            output="screen",
            condition=IfCondition(enable_logger),
            parameters=[
                config,
                {"use_sim_time": use_sim_time},
            ],
        ),
    ])
