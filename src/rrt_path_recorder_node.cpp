/**
 * rrt_path_recorder_node.cpp
 *
 * 역할
 * ────────────────────────────────────────────────────────────────
 * pick_place_node 가 RRTConnect 계획 성공 직후 publish 하는
 * /pick_place/rrt_trajectory 토픽을 구독하여 CSV 파일로 저장한다.
 *
 * 저장 형식
 * ────────────────────────────────────────────────────────────────
 * 파일명 : <output_dir>/rrt_<step_name>_<YYYYMMDD_HHMMSS>.csv
 * 헤더행 : time_sec,<joint_name_0>,...,<joint_name_N-1>
 * 데이터행: <time_from_start[s]>,<q0>,...,<qN-1>
 *   - time_sec : trajectory point 의 time_from_start (초 단위 실수)
 *   - qN       : N번째 관절의 위치 [rad]
 *
 * 프로토콜
 * ────────────────────────────────────────────────────────────────
 * joint_trajectory.header.frame_id = step_name (pick_place_node 가 태깅)
 *   예) "pre_grasp", "pre_place"
 * 이 필드는 TF/MoveIt 이 소비하지 않는 내부 전용 태그이다.
 *
 * 파라미터 (YAML 또는 CLI --ros-args -p 로 오버라이드 가능)
 * ────────────────────────────────────────────────────────────────
 *   output_dir (string, default: "<source_dir>/data", CMake 컴파일 타임 고정)
 *     CSV 파일이 저장될 디렉터리. 없으면 자동 생성.
 *     --ros-args -p output_dir:=/other/path 로 런타임 오버라이드 가능.
 */

#include <rclcpp/rclcpp.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

class RrtPathRecorderNode : public rclcpp::Node
{
public:
  RrtPathRecorderNode()
  : Node("rrt_path_recorder_node")
  {
    // ── 기본 저장 경로: <source_dir>/data ────────────────────────
    // PICK_PLACE_DATA_DIR 은 CMakeLists.txt 의 CMAKE_CURRENT_SOURCE_DIR/data
    // 로 컴파일 타임에 고정된다. motion_logger_node 와 동일 디렉터리 사용.
    declare_parameter<std::string>("output_dir", PICK_PLACE_DATA_DIR);
    output_dir_ = get_parameter("output_dir").as_string();

    // 출력 디렉터리 생성 (이미 존재하면 무시)
    std::error_code ec;
    std::filesystem::create_directories(output_dir_, ec);
    if (ec) {
      RCLCPP_ERROR(get_logger(),
        "Cannot create output directory '%s': %s",
        output_dir_.c_str(), ec.message().c_str());
    }

    sub_ = create_subscription<moveit_msgs::msg::RobotTrajectory>(
      "/pick_place/rrt_trajectory",
      10,
      std::bind(&RrtPathRecorderNode::onTrajectory, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(),
      "RrtPathRecorderNode ready — output_dir: %s", output_dir_.c_str());
  }

private:
  // ── 궤적 수신 콜백 ──────────────────────────────────────────
  void onTrajectory(const moveit_msgs::msg::RobotTrajectory::SharedPtr msg)
  {
    const auto & jt = msg->joint_trajectory;

    // step_name: pick_place_node 가 frame_id 에 태깅한 값
    const std::string step_name =
      jt.header.frame_id.empty() ? "unknown" : jt.header.frame_id;

    if (jt.points.empty()) {
      RCLCPP_WARN(get_logger(),
        "[%s] Received empty trajectory — skipping", step_name.c_str());
      return;
    }

    // ── CSV 파일 경로 생성 ───────────────────────────────────
    const std::string filename = buildFilePath(step_name);

    std::ofstream file(filename);
    if (!file.is_open()) {
      RCLCPP_ERROR(get_logger(),
        "[%s] Cannot open file: %s", step_name.c_str(), filename.c_str());
      return;
    }

    // ── 헤더 행 ──────────────────────────────────────────────
    file << "time_sec";
    for (const auto & jn : jt.joint_names) {
      file << "," << jn;
    }
    file << "\n";

    // ── 데이터 행 ─────────────────────────────────────────────
    for (const auto & pt : jt.points) {
      const double t_sec =
        static_cast<double>(pt.time_from_start.sec) +
        static_cast<double>(pt.time_from_start.nanosec) * 1e-9;

      file << std::fixed << std::setprecision(6) << t_sec;
      for (const double q : pt.positions) {
        file << "," << std::fixed << std::setprecision(6) << q;
      }
      file << "\n";
    }

    file.close();

    RCLCPP_INFO(get_logger(),
      "[%s] Saved %zu waypoints (%zu joints) → %s",
      step_name.c_str(),
      jt.points.size(),
      jt.joint_names.size(),
      filename.c_str());
  }

  // ── 파일 경로 생성 헬퍼 ──────────────────────────────────────
  // 형식: <output_dir>/rrt_<step_name>_<YYYYMMDD_HHMMSS>.csv
  std::string buildFilePath(const std::string & step_name) const
  {
    const auto now_tp  = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now_tp);

    std::ostringstream ts;
    ts << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S");

    return output_dir_ + "/rrt_" + step_name + "_" + ts.str() + ".csv";
  }

  // ── 멤버 변수 ─────────────────────────────────────────────────
  rclcpp::Subscription<moveit_msgs::msg::RobotTrajectory>::SharedPtr sub_;
  std::string output_dir_;
};

// ================================================================
// main
// ================================================================
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RrtPathRecorderNode>());
  rclcpp::shutdown();
  return 0;
}
