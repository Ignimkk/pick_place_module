/**
 * motion_logger_node.cpp
 *
 * 동작 방식 (트리거 기반):
 *   /motion_logger/record (std_msgs/Bool) 수신으로 제어
 *
 *   true  → 누적 값 초기화 + 기록 시작 (새 세그먼트)
 *   false → 기록 중단 + 최종 요약 1회 콘솔 출력 + CSV flush
 *
 * 기록 대상:
 *   pick_place_node 가 RRTConnect 계획을 실행하는 구간에서만 트리거됨.
 *   현재 두 구간:
 *     - 세그먼트 1: pre_grasp 이동 (executePick Step 2)
 *     - 세그먼트 2: pre_place 이동 (executePlace Step 1)
 *
 * 기록 지표:
 *   ① EE 이동 거리 [m] : base_link → tool0, Euclidean 거리 누적
 *   ② 관절 이동량 [rad]: |q_new - q_old| 누적 (UR 6-DOF, 그리퍼 제외)
 *
 * 출력:
 *   - CSV 파일  : log_dir/motion_log_YYYYMMDD_HHMMSS.csv
 *                 (segment_id 열로 구간 구분)
 *   - 콘솔 요약 : 각 세그먼트 종료 시 1회 출력
 *
 * 파라미터 (pick_place_params.yaml):
 *   base_frame  : EE 기준 프레임  (기본 "base_link")
 *   ee_frame    : EE 링크명       (기본 "tool0")
 *   min_delta_m   : EE 노이즈 임계값   [m]   (기본 0.0001)
 *   min_delta_rad : 관절 노이즈 임계값 [rad] (기본 0.0001)
 *   log_dir     : CSV 저장 디렉토리   (기본 "<source_dir>/data", CMake 컴파일 타임 고정)
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

class MotionLoggerNode : public rclcpp::Node
{
public:
  static constexpr size_t ARM_DOF = 6;
  const std::array<std::string, ARM_DOF> ARM_JOINTS = {
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
  };

  MotionLoggerNode()
  : Node("motion_logger_node")
  {
    // ── 기본 저장 경로: <source_dir>/data ────────────────────────
    // PICK_PLACE_DATA_DIR 은 CMakeLists.txt 의 CMAKE_CURRENT_SOURCE_DIR/data
    // 로 컴파일 타임에 고정된다. install/ 경로가 아닌 소스 경로에 저장.
    declare_parameter<std::string>("base_frame",    "base_link");
    declare_parameter<std::string>("ee_frame",      "tool0");
    declare_parameter<double>     ("min_delta_m",   1e-4);
    declare_parameter<double>     ("min_delta_rad", 1e-4);
    declare_parameter<std::string>("log_dir",       PICK_PLACE_DATA_DIR);

    base_frame_    = get_parameter("base_frame").as_string();
    ee_frame_      = get_parameter("ee_frame").as_string();
    min_delta_m_   = get_parameter("min_delta_m").as_double();
    min_delta_rad_ = get_parameter("min_delta_rad").as_double();

    // ── TF2 ──────────────────────────────────────────────────────
    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // ── 구독: joint_states ────────────────────────────────────────
    joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "joint_states", 10,
      [this](sensor_msgs::msg::JointState::SharedPtr msg) {
        onJointState(msg);
      });

    // ── 구독: 트리거 토픽 ─────────────────────────────────────────
    // true  → 초기화 + 기록 시작
    // false → 기록 중단 + 요약 출력
    record_sub_ = create_subscription<std_msgs::msg::Bool>(
      "/motion_logger/record", 10,
      [this](std_msgs::msg::Bool::SharedPtr msg) {
        onRecord(msg);
      });

    // ── CSV 파일 열기 ──────────────────────────────────────────────
    openCsvFile();

    RCLCPP_INFO(get_logger(), "MotionLoggerNode ready");
    RCLCPP_INFO(get_logger(), "  base_frame : %s", base_frame_.c_str());
    RCLCPP_INFO(get_logger(), "  ee_frame   : %s", ee_frame_.c_str());
    RCLCPP_INFO(get_logger(), "  log file   : %s", csv_path_.c_str());
    RCLCPP_INFO(get_logger(), "  trigger    : /motion_logger/record (Bool)");
  }

  ~MotionLoggerNode()
  {
    if (csv_file_.is_open()) {
      csv_file_.flush();
      csv_file_.close();
    }
    RCLCPP_INFO(get_logger(), "CSV saved → %s", csv_path_.c_str());
  }

private:
  // ──────────────────────────────────────────────────────────────
  // 트리거 콜백
  // ──────────────────────────────────────────────────────────────
  void onRecord(const std_msgs::msg::Bool::SharedPtr msg)
  {
    if (msg->data) {
      // ── 기록 시작: 누적 값 초기화 ─────────────────────────────
      segment_id_++;
      resetAccumulators();
      recording_ = true;
      RCLCPP_INFO(get_logger(),
        "━━━ [Segment %d] Recording STARTED ━━━", segment_id_);
    } else {
      // ── 기록 중단: 요약 1회 출력 ──────────────────────────────
      if (!recording_) { return; }
      recording_ = false;
      printSummary();
      if (csv_file_.is_open()) { csv_file_.flush(); }
    }
  }

  // ──────────────────────────────────────────────────────────────
  // 누적 값 초기화
  //   first_reading_ = true 로 설정하여 다음 수신 시
  //   이전 값 기준점을 재설정함
  // ──────────────────────────────────────────────────────────────
  void resetAccumulators()
  {
    first_reading_  = true;
    ee_cum_dist_    = 0.0;
    joint_cum_disp_.fill(0.0);
    prev_joints_.fill(std::numeric_limits<double>::quiet_NaN());
  }

  // ──────────────────────────────────────────────────────────────
  // CSV 파일 초기화
  // ──────────────────────────────────────────────────────────────
  void openCsvFile()
  {
    const std::string dir = get_parameter("log_dir").as_string();

    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ss;
    ss << dir << "/motion_log_"
       << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S")
       << ".csv";
    csv_path_ = ss.str();

    // 저장 디렉토리가 없으면 자동 생성
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
      RCLCPP_ERROR(get_logger(),
        "Cannot create log dir '%s': %s", dir.c_str(), ec.message().c_str());
      return;
    }

    csv_file_.open(csv_path_);
    if (!csv_file_.is_open()) {
      RCLCPP_ERROR(get_logger(), "Cannot open CSV: %s", csv_path_.c_str());
      return;
    }

    // CSV 헤더 (세그먼트 종료 시 요약 1행만 기록):
    //   segment_id | cum_ee_m
    //   | cum_qi_rad×6 | cum_qi_deg×6
    csv_file_ << "segment_id,cum_ee_m,";
    for (const auto & n : ARM_JOINTS) { csv_file_ << "cum_" << n << "_rad,"; }
    for (const auto & n : ARM_JOINTS) { csv_file_ << "cum_" << n << "_deg,"; }
    csv_file_ << "\n";
    csv_file_.flush();
  }

  // ──────────────────────────────────────────────────────────────
  // joint_states 콜백 — recording_ == true 일 때만 동작
  // ──────────────────────────────────────────────────────────────
  void onJointState(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    if (!recording_) { return; }

    // ── 1. ARM 관절 위치 추출 (이름 기반 매핑) ──────────────────
    std::array<double, ARM_DOF> cur_q;
    for (size_t i = 0; i < ARM_DOF; ++i) {
      auto it = std::find(msg->name.begin(), msg->name.end(), ARM_JOINTS[i]);
      if (it == msg->name.end()) { return; }
      cur_q[i] = msg->position[std::distance(msg->name.begin(), it)];
    }

    // ── 2. EE 위치 TF2 조회 ────────────────────────────────────
    geometry_msgs::msg::TransformStamped tf_stamped;
    try {
      tf_stamped = tf_buffer_->lookupTransform(
        base_frame_, ee_frame_, tf2::TimePointZero);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
        "TF %s → %s: %s", base_frame_.c_str(), ee_frame_.c_str(), ex.what());
      return;
    }

    const double ex_ = tf_stamped.transform.translation.x;
    const double ey_ = tf_stamped.transform.translation.y;
    const double ez_ = tf_stamped.transform.translation.z;

    // ── 3. 첫 수신: 기준점 설정 후 스킵 ────────────────────────
    //    기준점 없이 delta 계산하면 초기 큰 오차 발생
    if (first_reading_) {
      prev_ee_x_   = ex_;
      prev_ee_y_   = ey_;
      prev_ee_z_   = ez_;
      prev_joints_ = cur_q;
      first_reading_ = false;
      return;
    }

    // ── 4. 변화량 계산 ──────────────────────────────────────────
    const double dx = ex_ - prev_ee_x_;
    const double dy = ey_ - prev_ee_y_;
    const double dz = ez_ - prev_ee_z_;
    const double delta_ee = std::sqrt(dx*dx + dy*dy + dz*dz);

    std::array<double, ARM_DOF> delta_q;
    for (size_t i = 0; i < ARM_DOF; ++i) {
      delta_q[i] = std::abs(cur_q[i] - prev_joints_[i]);
    }

    // ── 5. 노이즈 필터 (임계값 미만이면 무시) ───────────────────
    const bool ee_moved =
      delta_ee > min_delta_m_;
    const bool joint_moved =
      std::any_of(delta_q.begin(), delta_q.end(),
        [this](double d) { return d > min_delta_rad_; });

    if (!ee_moved && !joint_moved) { return; }

    // ── 6. 누적 ──────────────────────────────────────────────────
    ee_cum_dist_ += delta_ee;
    for (size_t i = 0; i < ARM_DOF; ++i) {
      joint_cum_disp_[i] += delta_q[i];
    }

    // ── 7. 이전 값 갱신 ─────────────────────────────────────────
    prev_ee_x_   = ex_;
    prev_ee_y_   = ey_;
    prev_ee_z_   = ez_;
    prev_joints_ = cur_q;
  }

  // ──────────────────────────────────────────────────────────────
  // 세그먼트 종료 시 1회 콘솔 출력
  // ──────────────────────────────────────────────────────────────
  void printSummary()
  {
    RCLCPP_INFO(get_logger(),
      "━━━ [Segment %d] Recording STOPPED — Summary ━━━", segment_id_);

    if (first_reading_) {
      RCLCPP_INFO(get_logger(), "  (no data recorded in this segment)");
      RCLCPP_INFO(get_logger(),
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
      return;
    }

    // ── 콘솔 출력 ────────────────────────────────────────────────
    RCLCPP_INFO(get_logger(),
      "  EE total path length : %.4f m", ee_cum_dist_);
    for (size_t i = 0; i < ARM_DOF; ++i) {
      RCLCPP_INFO(get_logger(),
        "  %-26s : %7.4f rad  (%7.2f deg)",
        ARM_JOINTS[i].c_str(),
        joint_cum_disp_[i],
        joint_cum_disp_[i] * 180.0 / M_PI);
    }

    // ── CSV: 세그먼트 요약 1행 기록 ──────────────────────────────
    // 형식: segment_id, cum_ee_m, cum_qi_rad×6, cum_qi_deg×6
    if (csv_file_.is_open()) {
      csv_file_ << std::fixed << std::setprecision(6)
                << segment_id_ << ","
                << ee_cum_dist_ << ",";
      for (double c : joint_cum_disp_) { csv_file_ << c << ","; }
      for (double c : joint_cum_disp_) {
        csv_file_ << c * 180.0 / M_PI << ",";
      }
      csv_file_ << "\n";
      csv_file_.flush();
    }

    RCLCPP_INFO(get_logger(), "  CSV → %s", csv_path_.c_str());
    RCLCPP_INFO(get_logger(),
      "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  }

  // ── members ───────────────────────────────────────────────────
  std::string base_frame_;
  std::string ee_frame_;
  double      min_delta_m_;
  double      min_delta_rad_;

  std::shared_ptr<tf2_ros::Buffer>            tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr           record_sub_;

  // 상태
  bool   recording_{false};
  int    segment_id_{0};
  bool   first_reading_{true};
  double prev_ee_x_{0.0}, prev_ee_y_{0.0}, prev_ee_z_{0.0};
  std::array<double, ARM_DOF> prev_joints_{};

  // 누적 지표
  double ee_cum_dist_{0.0};
  std::array<double, ARM_DOF> joint_cum_disp_{};

  // CSV
  std::ofstream csv_file_;
  std::string   csv_path_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MotionLoggerNode>());
  rclcpp::shutdown();
  return 0;
}
