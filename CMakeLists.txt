cmake_minimum_required(VERSION 3.8)
project(pick_place_module)

# ---------------------------------------------------------------
# 컴파일러 경고 옵션
# ---------------------------------------------------------------
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

# ---------------------------------------------------------------
# 데이터 저장 경로 (소스 디렉터리 기준)
# motion_logger_node / rrt_path_recorder_node 양쪽이 동일 경로를 사용.
# CMAKE_CURRENT_SOURCE_DIR = .../src/pick_place_module
# ---------------------------------------------------------------
set(PICK_PLACE_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")

# ---------------------------------------------------------------
# 의존성 찾기
# ---------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# ---------------------------------------------------------------
# Action 인터페이스 생성
# Pick.action / Place.action 분리 정의
# geometry_msgs 를 DEPENDENCIES 에 명시 — Pose 메시지 참조를 위해 필요
# ---------------------------------------------------------------
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/Pick.action"
  "action/Place.action"
  DEPENDENCIES geometry_msgs
)

# ---------------------------------------------------------------
# 생성된 C++ 타입서포트 타겟 획득
# rosidl_generate_interfaces 완료 후에 사용 가능
# ---------------------------------------------------------------
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} rosidl_typesupport_cpp)

# ---------------------------------------------------------------
# goal_relay_node
# 역할: pick_goal / place_goal 토픽 → pick / place action goal 변환 전송
#       mode 0: 토픽 하나만 받아도 즉시 해당 action 실행
#       mode 1: 두 토픽 모두 수신 후 pick → place 순차 실행
# ---------------------------------------------------------------
add_executable(goal_relay_node src/goal_relay_node.cpp)
ament_target_dependencies(goal_relay_node
  rclcpp
  rclcpp_action
  geometry_msgs
)
target_link_libraries(goal_relay_node "${cpp_typesupport_target}")

# ---------------------------------------------------------------
# pick_place_node
# 역할: MoveIt2 기반 pick / place action server
#       - pick action: RRTConnect + Cartesian 직선 접근
#       - place action: RRTConnect + Cartesian 직선 접근
# ---------------------------------------------------------------
add_executable(pick_place_node src/pick_place_node.cpp)
ament_target_dependencies(pick_place_node
  rclcpp
  rclcpp_action
  std_msgs
  nav_msgs
  geometry_msgs
  control_msgs
  moveit_ros_planning_interface
  moveit_core
  moveit_msgs
)
target_link_libraries(pick_place_node "${cpp_typesupport_target}")

# ---------------------------------------------------------------
# motion_logger_node
# 역할: /joint_states + TF2 기반 EE 이동 거리 및 관절 이동량 기록
#       CSV 파일 저장 + 주기적 콘솔 요약 출력
# ---------------------------------------------------------------
add_executable(motion_logger_node src/motion_logger_node.cpp)
ament_target_dependencies(motion_logger_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  tf2_ros
  tf2_geometry_msgs
)
target_compile_definitions(motion_logger_node PRIVATE
  PICK_PLACE_DATA_DIR="${PICK_PLACE_DATA_DIR}"
)

# ---------------------------------------------------------------
# ee_pose_publisher
# 현재 엔드이펙터 pose 를 TF2 로 조회하여 /ee_pose 발행
# ---------------------------------------------------------------
add_executable(ee_pose_publisher src/ee_pose_publisher.cpp)
ament_target_dependencies(ee_pose_publisher
  rclcpp
  geometry_msgs
  tf2_ros
  tf2_geometry_msgs
)

# ---------------------------------------------------------------
# ee_path_visualizer_node
# 역할: EE 궤적 RViz 시각화
#   /ee_path/planned : RRT 계획 경로 (FK 변환, Method B)
#   /ee_path/actual  : 실제 실행 경로 (TF 폴링, Method C)
# ---------------------------------------------------------------
add_executable(ee_path_visualizer_node src/ee_path_visualizer_node.cpp)
ament_target_dependencies(ee_path_visualizer_node
  rclcpp
  nav_msgs
  std_msgs
  geometry_msgs
  tf2_ros
  tf2_geometry_msgs
)

# ---------------------------------------------------------------
# rrt_path_recorder_node
# 역할: /pick_place/rrt_trajectory 구독 → CSV 파일 저장
#       pick_place_node 가 RRTConnect 계획 성공 직후 publish 하는
#       moveit_msgs/RobotTrajectory 를 수신하여 waypoints 를 기록.
# ---------------------------------------------------------------
add_executable(rrt_path_recorder_node src/rrt_path_recorder_node.cpp)
ament_target_dependencies(rrt_path_recorder_node
  rclcpp
  moveit_msgs
)
target_compile_definitions(rrt_path_recorder_node PRIVATE
  PICK_PLACE_DATA_DIR="${PICK_PLACE_DATA_DIR}"
)

# ---------------------------------------------------------------
# 설치
# ---------------------------------------------------------------
install(
  TARGETS
    goal_relay_node
    pick_place_node
    motion_logger_node
    ee_pose_publisher
    rrt_path_recorder_node
    ee_path_visualizer_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY
    launch
    config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
