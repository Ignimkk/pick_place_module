"""
pick_place.launch.py

pick_place_module 의 노드를 기동:
  - pick_place_node    : MoveIt2 기반 pick / place action server
  - goal_relay_node    : pick_goal / place_goal 토픽 → action 변환
  - motion_logger_node : EE 이동 거리 + 관절 이동량 기록 (선택, 기본 활성)

전제조건:
  ur_setup_bringup 의 다음 중 하나가 먼저 실행되어
  move_group, robot_state_publisher, controller_manager 가 구동 중이어야 함:
    - ur_sim_moveit_robotiq_ur16e.launch.py   (use_sim:=true,  기본)
    - ur_real_moveit_robotiq_ur16e.launch.py  (use_sim:=false, 실물)

런치 인수:
  use_sim         : true(기본)=Gazebo + use_sim_time=true,
                    false=실물 UR16e + use_sim_time=false
  trigger_mode    : 0(기본)=토픽 하나만 받아도 즉시 해당 action 실행
                    1=pick_goal + place_goal 모두 수신 후 순차 실행
  enable_logger   : true(기본)=motion_logger_node 활성화
  launch_pick_place_node :
                    true(기본)=pick_place_node 실행.
                    false=노드 미실행 → MoveIt RViz panel 로 수동 plan/execute 가능
                    (자동 모션 없는 dry-run 검증용).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share     = get_package_share_directory("pick_place_module")
    bringup_share = get_package_share_directory("ur_setup_bringup")

    config     = os.path.join(pkg_share,     "config", "pick_place_params.yaml")
    kinematics = os.path.join(bringup_share, "config", "kinematics.yaml")

    use_sim                = LaunchConfiguration("use_sim")
    trigger_mode           = LaunchConfiguration("trigger_mode")
    enable_logger          = LaunchConfiguration("enable_logger")
    launch_pick_place_node = LaunchConfiguration("launch_pick_place_node")

    # use_sim → use_sim_time 매핑 (string).
    # PythonExpression 으로 "true"/"false" 문자열을 그대로 use_sim_time 으로 전달.
    use_sim_time = PythonExpression([
        "'true' if '", use_sim, "' == 'true' else 'false'"
    ])

    return LaunchDescription([
        # --------------------------------------------------------
        # 런치 인수
        # --------------------------------------------------------
        DeclareLaunchArgument(
            "use_sim", default_value="true",
            description="true: Gazebo 시뮬레이션 (use_sim_time=true), "
                        "false: 실물 UR16e + Robotiq (use_sim_time=false).",
        ),
        DeclareLaunchArgument(
            "trigger_mode", default_value="0",
            description="0=pick_goal/place_goal 단일 트리거 즉시 실행, "
                        "1=둘 다 수신 후 순차 실행.",
        ),
        DeclareLaunchArgument(
            "enable_logger", default_value="true",
            description="motion_logger_node 활성화 여부.",
        ),
        DeclareLaunchArgument(
            "launch_pick_place_node", default_value="true",
            description="false 로 두면 pick_place_node 미실행. "
                        "실물에서 자동 모션 없이 MoveIt RViz panel 로 수동 plan/execute 검증할 때 사용.",
        ),
        DeclareLaunchArgument(
            "experiment_mode", default_value="rrt_trajopt",
            description="실험 모드: rrt_only / trajopt_only / rrt_trajopt",
        ),
        DeclareLaunchArgument(
            "experiment_csv_path", default_value="",
            description="실험 결과 CSV 저장 경로",
        ),

        # --------------------------------------------------------
        # pick_place_node
        # --------------------------------------------------------
        Node(
            package="pick_place_module",
            executable="pick_place_node",
            output="screen",
            condition=IfCondition(launch_pick_place_node),
            parameters=[
                config,
                kinematics,
                {
                    "use_sim_time":         use_sim_time,
                    "experiment_mode":      LaunchConfiguration("experiment_mode"),
                    "experiment_csv_path":  LaunchConfiguration("experiment_csv_path"),
                },
            ],
        ),

        # --------------------------------------------------------
        # goal_relay_node
        # --------------------------------------------------------
        Node(
            package="pick_place_module",
            executable="goal_relay_node",
            name="goal_relay_node",
            output="screen",
            condition=IfCondition(launch_pick_place_node),
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
