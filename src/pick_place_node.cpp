/**
 * pick_place_node.cpp
 *
 * ── Pick 시퀀스 (Cartesian 의존성 제거) ───────────────────────────
 *   Step 1: grasp IK 검증              ← computeIKForPose(grasp, seed=current)
 *   Step 2: pre_grasp IK 검증          ← computeIKForPose(pre_grasp, seed=grasp)
 *   Step 3: 그리퍼 열기
 *   Step 4: pre_grasp 이동             ← planAndExecuteToJoints (RRT/TrajOpt)
 *   Step 5: grasp 접근                  ← approach_strategy:
 *                                          "vertical_cartesian" (default):
 *                                            x/y/orientation 고정, z 만 단조
 *                                            감소하는 N 개 Cartesian waypoint
 *                                          "joint": planAndExecuteToJoints (RRT/TrajOpt)
 *                                          "cartesian": 단일 computeCartesianPath
 *   Step 6: 그리퍼 닫기
 *   Step 7: 후퇴 (best-effort)         ← approach_strategy 와 동일 (역방향):
 *                                          "vertical_cartesian": z 단조 증가
 *
 * ── Place 시퀀스 ───────────────────────────────────────────────────
 *   동일 구조: place IK → pre_place IK(seed=place) → pre_place plan/exec
 *              → approach (joint/cartesian) → release →
 *              retreat (joint/cartesian, best-effort)
 *
 * ── 핵심 설계 ──────────────────────────────────────────────────────
 *   - 최종 grasp pose 의 IK 를 가장 먼저 검증한다.
 *   - pre_grasp IK 는 grasp solution 을 seed 로 사용 → 같은 IK 분기 보존
 *     (검색 시작 상태 + L2 비용 함수 모두 seed 기준)
 *   - pre_grasp → grasp 접근에서 computeCartesianPath 실패 의존 제거.
 *     기본 "joint" 모드는 grasp_solution 으로 RRT/TrajOpt plan 을 실행,
 *     충돌 인지 plan 으로 실패 가능성을 줄인다.
 *   - 실패 라벨 분리: grasp IK / pre_grasp IK / pre_grasp planning /
 *                    pre_grasp execution / approach planning /
 *                    approach execution.
 *
 * ── Trajectory Optimization 통합 (use_trajopt=true) ───────────────
 *   use_trajopt=false (기본): 기존 MoveIt2 execute 동작 유지
 *   use_trajopt=true:
 *     RRT 계획 완료 → TrajOpt Action goal 전송 → 최적화 trajectory 수신
 *     → /joint_trajectory_controller/joint_trajectory 직접 발행
 *     → T_opt + traj_exec_margin_sec 대기
 *     → trajopt 실패 시 MoveIt2 execute 로 폴백
 *
 * ── 파라미터 (config/pick_place_params.yaml) ───────────────────────
 *   arm_group, pre_grasp_offset, gripper_*, velocity_scaling,
 *   acceleration_scaling, planning_time, num_planning_attempts,
 *   cartesian_eef_step, cartesian_min_fraction, grasp_orientation_*,
 *   ik_timeout, ik_cost_weight_l2,
 *   use_trajopt, trajopt_server_timeout_sec, T_init_sec,
 *   trajopt_N, traj_exec_margin_sec
 */

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/kinematics_base/kinematics_base.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/collision_detection/collision_common.h>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <nav_msgs/msg/path.hpp>
#include <control_msgs/action/gripper_command.hpp>
#include <std_msgs/msg/bool.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include <pick_place_module/action/pick.hpp>
#include <pick_place_module/action/place.hpp>
#include <pick_place_module/action/traj_opt.hpp>

#include <Eigen/Dense>
#include <cctype>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ── IK 비용 함수 헬퍼 ─────────────────────────────────────────────────
static double computeL2Norm(const std::vector<double> & solution,
                             const std::vector<double> & seed)
{
  double sum = 0.0;
  for (size_t i = 0; i < solution.size(); ++i) {
    const double d = solution[i] - seed[i];
    sum += d * d;
  }
  return sum;
}

// ─────────────────────────────────────────────────────────────────────────

// UR16e 관절 이름 (ros2_control 컨트롤러 순서)
static const std::vector<std::string> UR_JOINT_NAMES = {
  "shoulder_pan_joint",
  "shoulder_lift_joint",
  "elbow_joint",
  "wrist_1_joint",
  "wrist_2_joint",
  "wrist_3_joint",
};

// MoveIt JointModelGroup 변수 순서 → UR 컨트롤러 순서(UR_JOINT_NAMES)로 재정렬.
//
// MoveIt의 copyJointGroupPositions()는 jmg->getVariableNames() 순서(알파벳 등)로
// 반환하므로, UR 하드웨어 순서와 다를 수 있다.
// trajopt_only 모드에서 q_start / q_end 를 TrajOpt 서버에 보내기 전에 반드시 사용.
static std::vector<double> reorderToUrOrder(
  const std::vector<double>           & q_moveit,
  const moveit::core::JointModelGroup * jmg)
{
  const auto & var_names = jmg->getVariableNames();
  std::vector<double> q_ur(UR_JOINT_NAMES.size(), 0.0);
  for (size_t j = 0; j < UR_JOINT_NAMES.size(); ++j) {
    for (size_t k = 0; k < var_names.size(); ++k) {
      if (var_names[k] == UR_JOINT_NAMES[j] && k < q_moveit.size()) {
        q_ur[j] = q_moveit[k];
        break;
      }
    }
  }
  return q_ur;
}

// ─────────────────────────────────────────────────────────────────────────
// 실험 프레임워크 — 구조체 및 헬퍼
// ─────────────────────────────────────────────────────────────────────────

static double durationSec(
  const std::chrono::steady_clock::time_point & a,
  const std::chrono::steady_clock::time_point & b)
{
  return std::chrono::duration<double>(b - a).count();
}

static std::string trimCopy(const std::string & s)
{
  size_t first = 0;
  while (first < s.size() &&
         std::isspace(static_cast<unsigned char>(s[first])))
  {
    ++first;
  }
  size_t last = s.size();
  while (last > first &&
         std::isspace(static_cast<unsigned char>(s[last - 1])))
  {
    --last;
  }
  return s.substr(first, last - first);
}

struct TrajectoryMetrics {
  int    num_points        = 0;
  double duration_sec      = 0.0;
  double joint_path_length = 0.0;   // sum ||Δq||_2
  double mean_vel          = 0.0;
  double max_vel           = 0.0;
  double mean_accel        = 0.0;
  double max_accel         = 0.0;
  double mean_jerk         = 0.0;
  double max_jerk          = 0.0;
};

struct ExperimentRecord {
  int         trial_id           = 0;
  std::string step_name;
  std::string experiment_mode;
  bool        success            = false;
  bool        fallback_used      = false;
  double      ik_time_sec        = 0.0;
  double      rrt_planning_sec   = 0.0;
  double      shortcut_time_sec  = 0.0;
  double      guess_time_sec     = 0.0;
  double      solve_time_sec     = 0.0;
  double      total_compute_sec  = 0.0;
  double      exec_wait_sec      = 0.0;
  int         num_rrt_points     = 0;
  int         num_shortcut_pts   = 0;
  int         num_opt_points     = 0;
  TrajectoryMetrics traj;
  double      mean_torque        = 0.0;
  double      max_torque         = 0.0;
  double      mean_torque_rate   = 0.0;
  double      max_torque_rate    = 0.0;
  double      max_constr_viol    = 0.0;
  double      final_cost         = 0.0;
  std::string solver_status;
  std::string message;
};

static TrajectoryMetrics computeTrajectoryMetrics(
  const trajectory_msgs::msg::JointTrajectory & jt)
{
  TrajectoryMetrics m;
  const int K = static_cast<int>(jt.points.size());
  if (K < 2) return m;
  m.num_points = K;

  auto toSec = [](const builtin_interfaces::msg::Duration & d) {
    return static_cast<double>(d.sec) + d.nanosec * 1e-9;
  };

  m.duration_sec = toSec(jt.points.back().time_from_start);
  const int nj   = static_cast<int>(jt.points[0].positions.size());

  // path length
  double path = 0.0;
  for (int k = 1; k < K; ++k) {
    double sq = 0.0;
    for (int j = 0; j < nj; ++j) {
      double d = jt.points[k].positions[j] - jt.points[k-1].positions[j];
      sq += d * d;
    }
    path += std::sqrt(sq);
  }
  m.joint_path_length = path;

  // get velocity at (k, j) — use stored or finite-diff
  auto getVel = [&](int k, int j) -> double {
    if (!jt.points[k].velocities.empty())
      return jt.points[k].velocities[j];
    if (k == 0 || k == K - 1) return 0.0;
    double dt = toSec(jt.points[k+1].time_from_start)
              - toSec(jt.points[k-1].time_from_start);
    if (dt < 1e-9) return 0.0;
    return (jt.points[k+1].positions[j] - jt.points[k-1].positions[j]) / dt;
  };

  // get acceleration at (k, j)
  auto getAccel = [&](int k, int j) -> double {
    if (!jt.points[k].accelerations.empty())
      return jt.points[k].accelerations[j];
    if (k == 0 || k == K - 1) return 0.0;
    double dt = toSec(jt.points[k+1].time_from_start)
              - toSec(jt.points[k-1].time_from_start);
    if (dt < 1e-9) return 0.0;
    return (getVel(k+1, j) - getVel(k-1, j)) / dt;
  };

  double sv = 0.0, sa = 0.0, sj = 0.0;
  double mv = 0.0, ma = 0.0, mj = 0.0;
  int    cv = 0,   ca = 0,   cj = 0;

  for (int k = 0; k < K; ++k) {
    for (int j = 0; j < nj; ++j) {
      double v = std::abs(getVel(k, j));
      sv += v; mv = std::max(mv, v); ++cv;
      double a = std::abs(getAccel(k, j));
      sa += a; ma = std::max(ma, a); ++ca;
    }
  }
  // jerk: central-diff of acceleration
  for (int k = 1; k < K - 1; ++k) {
    double dt = toSec(jt.points[k+1].time_from_start)
              - toSec(jt.points[k-1].time_from_start);
    if (dt < 1e-9) continue;
    for (int j = 0; j < nj; ++j) {
      double jerk = std::abs((getAccel(k+1, j) - getAccel(k-1, j)) / dt);
      sj += jerk; mj = std::max(mj, jerk); ++cj;
    }
  }

  m.mean_vel   = (cv > 0) ? sv / cv : 0.0;
  m.max_vel    = mv;
  m.mean_accel = (ca > 0) ? sa / ca : 0.0;
  m.max_accel  = ma;
  m.mean_jerk  = (cj > 0) ? sj / cj : 0.0;
  m.max_jerk   = mj;
  return m;
}

class CsvLogger
{
public:
  explicit CsvLogger(const std::string & path)
  : path_(path)
  {
    namespace fs = std::filesystem;
    fs::create_directories(fs::path(path_).parent_path());
    const bool exists = fs::exists(path_);
    ofs_.open(path_, std::ios::app);
    if (!exists) writeHeader();
  }

  void write(const ExperimentRecord & r)
  {
    if (!ofs_.is_open()) return;
    auto f = [&](double v) { return std::isnan(v) ? "nan" : std::to_string(v); };
    ofs_ << r.trial_id                << ","
         << r.step_name               << ","
         << r.experiment_mode         << ","
         << r.success                 << ","
         << r.fallback_used           << ","
         << f(r.ik_time_sec)          << ","
         << f(r.rrt_planning_sec)     << ","
         << f(r.shortcut_time_sec)    << ","
         << f(r.guess_time_sec)       << ","
         << f(r.solve_time_sec)       << ","
         << f(r.total_compute_sec)    << ","
         << f(r.exec_wait_sec)        << ","
         << r.num_rrt_points          << ","
         << r.num_shortcut_pts        << ","
         << r.num_opt_points          << ","
         << f(r.traj.duration_sec)    << ","
         << f(r.traj.joint_path_length)<< ","
         << f(r.traj.mean_vel)        << ","
         << f(r.traj.max_vel)         << ","
         << f(r.traj.mean_accel)      << ","
         << f(r.traj.max_accel)       << ","
         << f(r.traj.mean_jerk)       << ","
         << f(r.traj.max_jerk)        << ","
         << f(r.mean_torque)          << ","
         << f(r.max_torque)           << ","
         << f(r.mean_torque_rate)     << ","
         << f(r.max_torque_rate)      << ","
         << f(r.max_constr_viol)      << ","
         << f(r.final_cost)           << ","
         << r.solver_status           << ","
         << r.message                 << "\n";
    ofs_.flush();
  }

private:
  void writeHeader() {
    ofs_ << "trial_id,step_name,experiment_mode,success,fallback_used,"
            "ik_time_sec,rrt_planning_sec,shortcut_time_sec,"
            "initial_guess_time_sec,solve_time_sec,total_compute_sec,"
            "exec_wait_sec,num_rrt_points,num_shortcut_waypoints,"
            "num_optimized_points,trajectory_duration_sec,"
            "joint_path_length,mean_joint_velocity,max_joint_velocity,"
            "mean_joint_acceleration,max_joint_acceleration,"
            "mean_joint_jerk,max_joint_jerk,"
            "mean_torque,max_torque,mean_torque_rate,max_torque_rate,"
            "max_constraint_violation,final_cost,solver_status,message\n";
  }
  std::string   path_;
  std::ofstream ofs_;
};

static std::string defaultCsvPath()
{
  const char * home = std::getenv("HOME");
  std::string dir   = home ? std::string(home) + "/.ros/pick_place_exp" : "/tmp/pick_place_exp";
  std::time_t t     = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&t));
  return dir + "/" + std::string(buf) + "_results.csv";
}

// ---------------------------------------------------------------------------

class PickPlaceNode : public rclcpp::Node
{
public:
  using Pick            = pick_place_module::action::Pick;
  using Place           = pick_place_module::action::Place;
  using TrajOpt         = pick_place_module::action::TrajOpt;
  using PickGoalHandle  = rclcpp_action::ServerGoalHandle<Pick>;
  using PlaceGoalHandle = rclcpp_action::ServerGoalHandle<Place>;
  using GripperCommand  = control_msgs::action::GripperCommand;
  using MoveGroupIface  = moveit::planning_interface::MoveGroupInterface;

  explicit PickPlaceNode(std::shared_ptr<rclcpp::Node> move_group_node)
  : Node("pick_place_node")
  {
    // ── 파라미터 선언 ─────────────────────────────────────────────
    declare_parameter<std::string>("arm_group",            "ur_manipulator");
    declare_parameter<double>("pre_grasp_offset",          0.10);
    declare_parameter<double>("gripper_open_pos",          0.0);
    declare_parameter<double>("gripper_close_pos",         0.8);
    declare_parameter<double>("velocity_scaling",          0.3);
    declare_parameter<double>("acceleration_scaling",      0.3);
    declare_parameter<double>("planning_time",             10.0);
    declare_parameter<int>   ("num_planning_attempts",     20);
    declare_parameter<double>("gripper_max_effort",        50.0);
    declare_parameter<double>("gripper_timeout_sec",       10.0);
    declare_parameter<double>("cartesian_eef_step",        0.01);
    declare_parameter<double>("cartesian_min_fraction",    0.95);
    declare_parameter<double>("grasp_orientation_x",  0.0);
    declare_parameter<double>("grasp_orientation_y",  1.0);
    declare_parameter<double>("grasp_orientation_z",  0.0);
    declare_parameter<double>("grasp_orientation_w",  0.0);
    declare_parameter<double>("ik_timeout",           0.5);
    declare_parameter<double>("ik_cost_weight_l2",    0.05);
    declare_parameter<bool>("return_home_after_place", true);
    declare_parameter<std::string>("initial_positions_path", INITIAL_POSITIONS_FILE);

    // ── TrajOpt 통합 파라미터 ────────────────────────────────────
    declare_parameter<bool>  ("use_trajopt",               false);
    declare_parameter<double>("trajopt_server_timeout_sec", 2.0);
    declare_parameter<double>("t_init_sec",                 3.0);
    declare_parameter<int>   ("trajopt_N",                  6);
    declare_parameter<double>("traj_exec_margin_sec",       1.5);
    declare_parameter<bool>  ("trajopt_use_reduced",        true);
    declare_parameter<bool>  ("trajopt_use_free_t",         true);

    // ── 실험 모드 파라미터 ────────────────────────────────────
    // "rrt_only"    : RRTConnect → MoveIt2 execute  (기준선)
    // "trajopt_only": IK → TrajOpt (RRT 경로 미사용, Hermite 초기 추정)
    // "rrt_trajopt" : RRTConnect → TrajOpt → publish  (기본)
    declare_parameter<std::string>("experiment_mode", "rrt_trajopt");
    // CSV 저장 경로 (비워두면 ~/.ros/pick_place_exp/TIMESTAMP_results.csv)
    declare_parameter<std::string>("experiment_csv_path", "");

    // ── pre_grasp → grasp 접근 전략 ───────────────────────────
    // "joint"              : grasp IK 로 관절 plan(RRT/TrajOpt) 실행
    //                         (충돌 인지, 그러나 EE 경로가 수직 보장 안 됨)
    // "cartesian"          : computeCartesianPath 단일 waypoint (레거시)
    // "vertical_cartesian" : pre_pose ↔ final_pose 사이를 N 개 Cartesian
    //                         waypoint 로 분할, x/y/orientation 고정,
    //                         z 만 단조 변화 → 수직 강하/후퇴 보장 (기본)
    declare_parameter<std::string>("approach_strategy", "vertical_cartesian");
    // vertical_cartesian 분할 waypoint 수 (5–20 권장)
    declare_parameter<int>("vertical_cartesian_waypoints", 10);

    // ── 강하 경로(pre_pose → final_pose) 충돌 검증 파라미터 ─────
    // 보간 단계 수: 10–20 이 권장 (높을수록 정확하지만 비용 증가)
    declare_parameter<int>("descent_check_steps",        15);
    // pre_pose IK 재시도 최대 횟수 (각 시도마다 다른 seed/random restart)
    // 너무 크면 응답 지연, 작으면 회피 실패. 3–5 권장.
    declare_parameter<int>("descent_max_ik_retries",     4);

    // ── MoveGroupInterface 초기화 ────────────────────────────────
    const std::string arm = get_parameter("arm_group").as_string();
    move_group_ = std::make_shared<MoveGroupIface>(move_group_node, arm);
    move_group_->setPlannerId("RRTConnectkConfigDefault");
    move_group_->setMaxVelocityScalingFactor(get_parameter("velocity_scaling").as_double());
    move_group_->setMaxAccelerationScalingFactor(get_parameter("acceleration_scaling").as_double());
    move_group_->setPlanningTime(get_parameter("planning_time").as_double());
    move_group_->setNumPlanningAttempts(get_parameter("num_planning_attempts").as_int());
    move_group_->setWorkspace(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0);
    move_group_->setPoseReferenceFrame("base_link");

    RCLCPP_INFO(get_logger(), "Planning group      : %s", arm.c_str());
    RCLCPP_INFO(get_logger(), "Planning frame      : %s", move_group_->getPlanningFrame().c_str());
    RCLCPP_INFO(get_logger(), "End effector link   : %s", move_group_->getEndEffectorLink().c_str());
    RCLCPP_INFO(get_logger(), "use_trajopt         : %s",
      get_parameter("use_trajopt").as_bool() ? "true" : "false");
    RCLCPP_INFO(get_logger(), "experiment_mode     : %s",
      get_parameter("experiment_mode").as_string().c_str());

    // ── callback groups ───────────────────────────────────────────
    gripper_cbg_  = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    pick_cbg_     = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    place_cbg_    = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    trajopt_cbg_  = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    // ── gripper action client ─────────────────────────────────────
    gripper_client_ = rclcpp_action::create_client<GripperCommand>(
      this, "robotiq_gripper_controller/gripper_cmd", gripper_cbg_);

    // ── TrajOpt action client ─────────────────────────────────────
    trajopt_client_ = rclcpp_action::create_client<TrajOpt>(
      this, "trajopt", trajopt_cbg_);

    // ── JointTrajectory 직접 발행 (TrajOpt 결과 실행용) ──────────
    traj_pub_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "/joint_trajectory_controller/joint_trajectory", 10);

    // ── pick action server ────────────────────────────────────────
    pick_server_ = rclcpp_action::create_server<Pick>(
      this, "pick",
      [this](const rclcpp_action::GoalUUID &, std::shared_ptr<const Pick::Goal>) {
        RCLCPP_INFO(get_logger(), "[pick] Goal received — accepted");
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
      },
      [this](const std::shared_ptr<PickGoalHandle>) {
        RCLCPP_WARN(get_logger(), "[pick] Cancel requested");
        move_group_->stop();
        return rclcpp_action::CancelResponse::ACCEPT;
      },
      [this](const std::shared_ptr<PickGoalHandle> gh) {
        std::thread{[this, gh]() { executePick(gh); }}.detach();
      },
      rcl_action_server_get_default_options(), pick_cbg_);

    // ── place action server ───────────────────────────────────────
    place_server_ = rclcpp_action::create_server<Place>(
      this, "place",
      [this](const rclcpp_action::GoalUUID &, std::shared_ptr<const Place::Goal>) {
        RCLCPP_INFO(get_logger(), "[place] Goal received — accepted");
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
      },
      [this](const std::shared_ptr<PlaceGoalHandle>) {
        RCLCPP_WARN(get_logger(), "[place] Cancel requested");
        move_group_->stop();
        return rclcpp_action::CancelResponse::ACCEPT;
      },
      [this](const std::shared_ptr<PlaceGoalHandle> gh) {
        std::thread{[this, gh]() { executePlace(gh); }}.detach();
      },
      rcl_action_server_get_default_options(), place_cbg_);

    // ── 기타 publishers ───────────────────────────────────────────
    motion_log_pub_ = create_publisher<std_msgs::msg::Bool>("/motion_logger/record", 10);
    rrt_traj_pub_   = create_publisher<moveit_msgs::msg::RobotTrajectory>(
      "/pick_place/rrt_trajectory", 10);
    ee_path_planned_pub_ = create_publisher<nav_msgs::msg::Path>(
      "/ee_path/planned", rclcpp::QoS(1).transient_local());

    // ── CSV 로거 초기화 ───────────────────────────────────────────
    std::string csv_path = get_parameter("experiment_csv_path").as_string();
    if (csv_path.empty()) csv_path = defaultCsvPath();
    csv_logger_ = std::make_shared<CsvLogger>(csv_path);
    RCLCPP_INFO(get_logger(), "Experiment CSV      : %s", csv_path.c_str());

    // ── PlanningSceneMonitor (충돌 검사용) ──────────────────────
    // move_group_node 가 robot_description 을 보유하므로 이를 재사용한다.
    // monitored_planning_scene 토픽을 구독하여 move_group 이 갖는 동일한
    // 충돌 정보(self-collision matrix, world geometry) 를 공유한다.
    psm_ = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(
      move_group_node, "robot_description");
    if (psm_->getPlanningScene()) {
      psm_->startSceneMonitor("/monitored_planning_scene");
      psm_->startWorldGeometryMonitor();
      psm_->startStateMonitor("/joint_states");
      psm_->requestPlanningSceneState("/get_planning_scene");
      RCLCPP_INFO(get_logger(),
        "PlanningSceneMonitor active — descent collision checks enabled "
        "(check_steps=%ld, max_retries=%ld)",
        get_parameter("descent_check_steps").as_int(),
        get_parameter("descent_max_ik_retries").as_int());
    } else {
      psm_.reset();
      RCLCPP_WARN(get_logger(),
        "PlanningSceneMonitor failed to initialize — descent collision "
        "checks DISABLED (will use single IK without validation)");
    }

    RCLCPP_INFO(get_logger(), "PickPlaceNode ready  |  /pick  /place");
  }

private:
  // ── 현재 EE yaw 에 가장 가까운 하향 orientation 적용 ─────────────
  //
  // 입력 pose 의 position 은 유지한다. orientation 은 현재 엔드이펙터의
  // 수평 방향을 최대한 보존하되, tool Z 축은 base_link 기준 -Z 방향으로
  // 향하게 만든다. 현재 상태를 읽지 못하면 기존 고정 quaternion 파라미터로
  // 폴백한다.
  geometry_msgs::msg::Pose applyDownwardOrientation(
    const geometry_msgs::msg::Pose & input) const
  {
    geometry_msgs::msg::Pose out = input;

    auto apply_fixed_fallback = [&]() {
      out.orientation.x = get_parameter("grasp_orientation_x").as_double();
      out.orientation.y = get_parameter("grasp_orientation_y").as_double();
      out.orientation.z = get_parameter("grasp_orientation_z").as_double();
      out.orientation.w = get_parameter("grasp_orientation_w").as_double();
    };

    auto rs = move_group_->getCurrentState(2.0);
    if (!rs) {
      RCLCPP_WARN(get_logger(),
        "getCurrentState failed while computing downward orientation — using fixed fallback");
      apply_fixed_fallback();
      return out;
    }

    const std::string ee_link = move_group_->getEndEffectorLink();
    if (ee_link.empty()) {
      RCLCPP_WARN(get_logger(),
        "End effector link is empty — using fixed downward orientation fallback");
      apply_fixed_fallback();
      return out;
    }

    const Eigen::Isometry3d world_T_base =
      rs->getGlobalLinkTransform("base_link");
    const Eigen::Isometry3d world_T_ee =
      rs->getGlobalLinkTransform(ee_link);
    const Eigen::Matrix3d base_R_ee =
      world_T_base.linear().transpose() * world_T_ee.linear();

    const Eigen::Vector3d z_axis(0.0, 0.0, -1.0);
    Eigen::Vector3d x_axis(base_R_ee(0, 0), base_R_ee(1, 0), 0.0);
    Eigen::Vector3d y_axis;
    const char * ref_axis = "tool_x";

    if (x_axis.norm() > 1e-6) {
      x_axis.normalize();
      y_axis = z_axis.cross(x_axis);
      y_axis.normalize();
    } else {
      y_axis = Eigen::Vector3d(base_R_ee(0, 1), base_R_ee(1, 1), 0.0);
      if (y_axis.norm() <= 1e-6) {
        RCLCPP_WARN(get_logger(),
          "Current EE horizontal axes are degenerate — using fixed downward orientation fallback");
        apply_fixed_fallback();
        return out;
      }
      ref_axis = "tool_y";
      y_axis.normalize();
      x_axis = y_axis.cross(z_axis);
      x_axis.normalize();
    }

    Eigen::Matrix3d base_R_target;
    base_R_target.col(0) = x_axis;
    base_R_target.col(1) = y_axis;
    base_R_target.col(2) = z_axis;

    Eigen::Quaterniond q(base_R_target);
    q.normalize();
    out.orientation.w = q.w();
    out.orientation.x = q.x();
    out.orientation.y = q.y();
    out.orientation.z = q.z();

    const double yaw_like_deg =
      std::atan2(x_axis.y(), x_axis.x()) * 180.0 / 3.14159265358979323846;
    RCLCPP_INFO(get_logger(),
      "Downward orientation from current EE %s projection: x_axis_yaw=%.1f deg, "
      "q=(x=%.4f, y=%.4f, z=%.4f, w=%.4f)",
      ref_axis, yaw_like_deg,
      out.orientation.x, out.orientation.y,
      out.orientation.z, out.orientation.w);
    return out;
  }

  // ── 그리퍼 제어 ───────────────────────────────────────────────
  bool controlGripper(double position, double max_effort, double timeout_sec)
  {
    if (!gripper_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_logger(), "Gripper action server not available");
      return false;
    }
    auto promise = std::make_shared<std::promise<bool>>();
    auto future  = promise->get_future();
    GripperCommand::Goal goal{};
    goal.command.position   = position;
    goal.command.max_effort = max_effort;
    auto opts = rclcpp_action::Client<GripperCommand>::SendGoalOptions{};
    opts.goal_response_callback =
      [promise](const rclcpp_action::ClientGoalHandle<GripperCommand>::SharedPtr & gh) {
        if (!gh) { promise->set_value(false); }
      };
    opts.result_callback =
      [promise](const rclcpp_action::ClientGoalHandle<GripperCommand>::WrappedResult & res) {
        promise->set_value(res.code == rclcpp_action::ResultCode::SUCCEEDED);
      };
    gripper_client_->async_send_goal(goal, opts);
    if (future.wait_for(std::chrono::duration<double>(timeout_sec)) ==
        std::future_status::timeout)
    {
      RCLCPP_ERROR(get_logger(), "Gripper timed out (%.1f s)", timeout_sec);
      return false;
    }
    return future.get();
  }

  // ── motion_logger 트리거 ─────────────────────────────────────
  void triggerMotionLog(bool start)
  {
    std_msgs::msg::Bool msg;
    msg.data = start;
    motion_log_pub_->publish(msg);
  }

  bool loadInitialJointTarget(std::vector<double> & joint_target)
  {
    const std::string path = get_parameter("initial_positions_path").as_string();
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
      RCLCPP_ERROR(get_logger(),
        "Failed to open initial_positions.yaml: %s", path.c_str());
      return false;
    }

    std::map<std::string, double> values;
    std::string line;
    int line_no = 0;
    while (std::getline(ifs, line)) {
      ++line_no;
      const size_t comment_pos = line.find('#');
      if (comment_pos != std::string::npos) {
        line = line.substr(0, comment_pos);
      }
      line = trimCopy(line);
      if (line.empty()) continue;

      const size_t colon_pos = line.find(':');
      if (colon_pos == std::string::npos) {
        RCLCPP_WARN(get_logger(),
          "Skipping malformed initial_positions line %d: %s",
          line_no, line.c_str());
        continue;
      }

      const std::string name = trimCopy(line.substr(0, colon_pos));
      const std::string value_text = trimCopy(line.substr(colon_pos + 1));
      if (name.empty() || value_text.empty()) {
        RCLCPP_WARN(get_logger(),
          "Skipping incomplete initial_positions line %d: %s",
          line_no, line.c_str());
        continue;
      }

      try {
        values[name] = std::stod(value_text);
      } catch (const std::exception & e) {
        RCLCPP_ERROR(get_logger(),
          "Invalid joint value in %s line %d (%s: %s): %s",
          path.c_str(), line_no, name.c_str(), value_text.c_str(), e.what());
        return false;
      }
    }

    joint_target.clear();
    joint_target.reserve(UR_JOINT_NAMES.size());
    for (const auto & joint_name : UR_JOINT_NAMES) {
      const auto it = values.find(joint_name);
      if (it == values.end()) {
        RCLCPP_ERROR(get_logger(),
          "initial_positions.yaml missing required joint: %s",
          joint_name.c_str());
        return false;
      }
      joint_target.push_back(it->second);
    }

    RCLCPP_INFO(get_logger(),
      "Loaded initial joint target from %s", path.c_str());
    return true;
  }

  // ── TrajOpt Action client (rrt_trajopt 전용) ────────────────────
  //
  // use_rrt_waypoints=true : RRT plan 의 waypoint 를 TrajOpt goal 에 패킹.
  //                          rrt_trajopt 모드에서만 호출됨.
  //
  // trajopt_only 모드는 runTrajoptOnly() 를 사용한다.
  // 이 함수에서 use_rrt_waypoints=false 경로는 사용되지 않는다.
  //
  // 성공 시 rec 에 서버 메트릭을 기록하고 optimized_trajectory 를 발행 후
  // T_opt + traj_exec_margin_sec 대기. 실패 시 false 반환 (caller 가 RRT 폴백).
  bool runWithTrajopt(
    const MoveGroupIface::Plan & plan,
    bool                         use_rrt_waypoints,
    const std::vector<double>  & q_start_joints,
    const std::vector<double>  & q_end_joints,
    const std::string          & step_name,
    ExperimentRecord           & rec)
  {
    const double server_timeout = get_parameter("trajopt_server_timeout_sec").as_double();
    const double t_init         = get_parameter("t_init_sec").as_double();
    const int    N_cheb         = get_parameter("trajopt_N").as_int();
    const double exec_margin    = get_parameter("traj_exec_margin_sec").as_double();
    const bool   use_reduced    = get_parameter("trajopt_use_reduced").as_bool();
    const bool   use_free_t     = get_parameter("trajopt_use_free_t").as_bool();

    // ── 서버 연결 확인 ──────────────────────────────────────────
    if (!trajopt_client_->wait_for_action_server(
          std::chrono::duration<double>(server_timeout)))
    {
      RCLCPP_WARN(get_logger(),
        "[%s] TrajOpt server not available (timeout=%.1f s) — MoveIt2 폴백",
        step_name.c_str(), server_timeout);
      return false;
    }

    // ── joint 순서 인덱스 맵 (UR 순서 기준) ────────────────────
    // RRT plan 의 joint_names 에서 UR 순서 인덱스를 구성한다.
    // rrt_trajopt 에서 호출되므로 jt.joint_names 는 항상 비어 있지 않다.
    const auto & jt = plan.trajectory_.joint_trajectory;
    std::vector<int> joint_idx(6, -1);
    if (!jt.joint_names.empty()) {
      for (size_t j = 0; j < UR_JOINT_NAMES.size(); ++j) {
        for (size_t k = 0; k < jt.joint_names.size(); ++k) {
          if (jt.joint_names[k] == UR_JOINT_NAMES[j]) {
            joint_idx[j] = static_cast<int>(k);
            break;
          }
        }
      }
    } else {
      // 안전 폴백: RRT plan 에 joint_names 없을 경우 identity 매핑
      for (int j = 0; j < 6; ++j) joint_idx[j] = j;
    }

    // ── TrajOpt goal 구성 ────────────────────────────────────────
    TrajOpt::Goal trajopt_goal;
    trajopt_goal.t_init      = t_init;
    trajopt_goal.cheb_degree = static_cast<int32_t>(N_cheb);
    trajopt_goal.use_reduced = use_reduced;
    trajopt_goal.use_free_t  = use_free_t;

    if (use_rrt_waypoints) {
      const size_t K = jt.points.size();
      if (K < 2) {
        RCLCPP_WARN(get_logger(),
          "[%s] RRT plan has < 2 points — TrajOpt 불가", step_name.c_str());
        return false;
      }
      trajopt_goal.num_waypoints = static_cast<int32_t>(K);
      trajopt_goal.timestamps.reserve(K);
      for (const auto & pt : jt.points) {
        double t = static_cast<double>(pt.time_from_start.sec)
                 + static_cast<double>(pt.time_from_start.nanosec) * 1e-9;
        trajopt_goal.timestamps.push_back(t);
      }
      trajopt_goal.q_waypoints.reserve(K * 6);
      for (const auto & pt : jt.points) {
        for (int j = 0; j < 6; ++j) {
          int k = joint_idx[j];
          trajopt_goal.q_waypoints.push_back(
            (k >= 0 && static_cast<size_t>(k) < pt.positions.size())
              ? pt.positions[k] : 0.0);
        }
      }
    } else {
      // 비정상: use_rrt_waypoints=false 는 이 함수에서 사용하지 않음
      trajopt_goal.num_waypoints = 0;
    }

    // q_start / q_end (RRT plan 의 joint_names 기반 UR 순서로 재정렬)
    trajopt_goal.q_start.resize(6, 0.0);
    trajopt_goal.q_end.resize(6, 0.0);
    for (int j = 0; j < 6; ++j) {
      int k = joint_idx[j];
      if (k >= 0) {
        if (static_cast<size_t>(k) < q_start_joints.size())
          trajopt_goal.q_start[j] = q_start_joints[k];
        if (static_cast<size_t>(k) < q_end_joints.size())
          trajopt_goal.q_end[j] = q_end_joints[k];
      }
    }

    RCLCPP_INFO(get_logger(),
      "[%s][rrt_trajopt] TrajOpt goal 전송 — K=%d, t_init=%.2fs, N=%d",
      step_name.c_str(), trajopt_goal.num_waypoints,
      t_init, N_cheb);

    // ── Goal 전송 + 결과 대기 ────────────────────────────────────
    auto promise_result = std::make_shared<
      std::promise<rclcpp_action::ClientGoalHandle<TrajOpt>::WrappedResult>>();
    auto future_result = promise_result->get_future();
    auto promise_goal  = std::make_shared<std::promise<bool>>();
    auto future_goal   = promise_goal->get_future();

    auto opts = rclcpp_action::Client<TrajOpt>::SendGoalOptions{};
    opts.goal_response_callback =
      [this, &step_name, promise_goal](
        const rclcpp_action::ClientGoalHandle<TrajOpt>::SharedPtr & gh)
      {
        if (!gh) {
          RCLCPP_ERROR(get_logger(), "[%s] TrajOpt goal rejected", step_name.c_str());
          promise_goal->set_value(false);
        } else {
          RCLCPP_INFO(get_logger(), "[%s] TrajOpt goal accepted", step_name.c_str());
          promise_goal->set_value(true);
        }
      };
    opts.feedback_callback =
      [this, &step_name](
        rclcpp_action::ClientGoalHandle<TrajOpt>::SharedPtr,
        const std::shared_ptr<const TrajOpt::Feedback> fb)
      {
        RCLCPP_INFO(get_logger(),
          "[%s] TrajOpt [%5.1f%%] %s (%.1fs)",
          step_name.c_str(), fb->progress * 100.0f,
          fb->status.c_str(), fb->elapsed_sec);
      };
    opts.result_callback =
      [promise_result](
        const rclcpp_action::ClientGoalHandle<TrajOpt>::WrappedResult & res)
      {
        promise_result->set_value(res);
      };

    trajopt_client_->async_send_goal(trajopt_goal, opts);

    // goal 수락 대기 (최대 5s)
    if (future_goal.wait_for(std::chrono::seconds(5)) != std::future_status::ready
        || !future_goal.get())
    {
      RCLCPP_ERROR(get_logger(),
        "[%s] TrajOpt goal 수락 실패 — MoveIt2 폴백", step_name.c_str());
      return false;
    }

    // 결과 대기 (최대 T_init*3 + 5s)
    const double result_timeout = t_init * 3.0 + 5.0;
    if (future_result.wait_for(std::chrono::duration<double>(result_timeout))
        != std::future_status::ready)
    {
      RCLCPP_ERROR(get_logger(),
        "[%s] TrajOpt result 타임아웃 (%.1fs) — MoveIt2 폴백",
        step_name.c_str(), result_timeout);
      return false;
    }

    auto wrapped = future_result.get();
    if (wrapped.code != rclcpp_action::ResultCode::SUCCEEDED) {
      RCLCPP_WARN(get_logger(),
        "[%s] TrajOpt Action code=%d — MoveIt2 폴백",
        step_name.c_str(), static_cast<int>(wrapped.code));
      return false;
    }

    const auto & res = wrapped.result;
    if (!res->success) {
      RCLCPP_WARN(get_logger(),
        "[%s] TrajOpt NLP 미수렴: %s — 현재 추정치로 계속 진행",
        step_name.c_str(), res->message.c_str());
    }

    // ── 서버 메트릭 추출 → ExperimentRecord ─────────────────────
    rec.shortcut_time_sec  = res->shortcut_time_sec;
    rec.guess_time_sec     = res->initial_guess_time_sec;
    rec.solve_time_sec     = res->solve_time_sec;
    rec.num_shortcut_pts   = res->num_shortcut_waypoints;
    rec.num_opt_points     = static_cast<int>(
                               res->optimized_trajectory.points.size());
    rec.final_cost         = res->cost;
    rec.max_constr_viol    = res->max_constraint_violation;
    rec.mean_torque        = res->mean_torque;
    rec.max_torque         = res->max_torque;
    rec.mean_torque_rate   = res->mean_torque_rate;
    rec.max_torque_rate    = res->max_torque_rate;
    rec.solver_status      = res->success ? "converged" : "partial";
    rec.traj               = computeTrajectoryMetrics(res->optimized_trajectory);

    RCLCPP_INFO(get_logger(),
      "[%s] TrajOpt — t_opt=%.3fs, J=%.4f, τ_max=%.1fNm, sc=%d, opt=%d pts",
      step_name.c_str(), res->t_opt, res->cost,
      res->max_torque, res->num_shortcut_waypoints,
      rec.num_opt_points);

    // ── 최적화 궤적 발행 + 실행 대기 ────────────────────────────
    auto t_exec = std::chrono::steady_clock::now();
    traj_pub_->publish(res->optimized_trajectory);
    rclcpp::sleep_for(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(res->t_opt + exec_margin)));
    rec.exec_wait_sec = durationSec(t_exec, std::chrono::steady_clock::now());

    RCLCPP_INFO(get_logger(), "[%s] 실행 완료 (%.2fs 대기)", step_name.c_str(), rec.exec_wait_sec);
    return true;
  }

  // ── trajopt_only 전용: RRT 없이 q_start/q_end → TrajOpt 서버 직송 ──
  //
  // q_start_ur, q_end_ur: reorderToUrOrder() 로 UR 컨트롤러 순서가 보장된 벡터.
  // num_waypoints=0 → 서버가 Cubic Hermite 초기 추정을 사용 (K=0 경로).
  // runWithTrajopt 와 달리 MoveGroupIface::Plan 에 의존하지 않는다.
  bool runTrajoptOnly(
    const std::vector<double> & q_start_ur,
    const std::vector<double> & q_end_ur,
    const std::string          & step_name,
    ExperimentRecord           & rec)
  {
    const double server_timeout = get_parameter("trajopt_server_timeout_sec").as_double();
    const double t_init         = get_parameter("t_init_sec").as_double();
    const int    N_cheb         = get_parameter("trajopt_N").as_int();
    const double exec_margin    = get_parameter("traj_exec_margin_sec").as_double();
    const bool   use_reduced    = get_parameter("trajopt_use_reduced").as_bool();
    const bool   use_free_t     = get_parameter("trajopt_use_free_t").as_bool();

    // ── 서버 연결 확인 ─────────────────────────────────────────────
    if (!trajopt_client_->wait_for_action_server(
          std::chrono::duration<double>(server_timeout)))
    {
      rec.message = "TrajOpt server not available";
      RCLCPP_ERROR(get_logger(),
        "[%s][trajopt_only] TrajOpt 서버 연결 실패 (timeout=%.1fs) — 중단",
        step_name.c_str(), server_timeout);
      return false;
    }

    // ── Goal 구성 (K=0, Hermite 초기 추정) ─────────────────────────
    TrajOpt::Goal trajopt_goal;
    trajopt_goal.t_init        = t_init;
    trajopt_goal.cheb_degree   = static_cast<int32_t>(N_cheb);
    trajopt_goal.use_reduced   = use_reduced;
    trajopt_goal.use_free_t    = use_free_t;
    trajopt_goal.num_waypoints = 0;  // RRT 경로 없음

    trajopt_goal.q_start.assign(q_start_ur.begin(), q_start_ur.end());
    trajopt_goal.q_end.assign(q_end_ur.begin(), q_end_ur.end());

    RCLCPP_INFO(get_logger(),
      "[%s][trajopt_only] TrajOpt goal 전송 — K=0 (Hermite), t_init=%.2fs, N=%d",
      step_name.c_str(), t_init, N_cheb);

    // ── Goal 전송 + 결과 대기 ──────────────────────────────────────
    auto promise_result = std::make_shared<
      std::promise<rclcpp_action::ClientGoalHandle<TrajOpt>::WrappedResult>>();
    auto future_result = promise_result->get_future();
    auto promise_goal  = std::make_shared<std::promise<bool>>();
    auto future_goal   = promise_goal->get_future();

    auto opts = rclcpp_action::Client<TrajOpt>::SendGoalOptions{};
    opts.goal_response_callback =
      [this, &step_name, promise_goal](
        const rclcpp_action::ClientGoalHandle<TrajOpt>::SharedPtr & gh)
      {
        if (!gh) {
          RCLCPP_ERROR(get_logger(),
            "[%s][trajopt_only] TrajOpt goal 거부", step_name.c_str());
          promise_goal->set_value(false);
        } else {
          RCLCPP_INFO(get_logger(),
            "[%s][trajopt_only] TrajOpt goal 수락", step_name.c_str());
          promise_goal->set_value(true);
        }
      };
    opts.feedback_callback =
      [this, &step_name](
        rclcpp_action::ClientGoalHandle<TrajOpt>::SharedPtr,
        const std::shared_ptr<const TrajOpt::Feedback> fb)
      {
        RCLCPP_INFO(get_logger(),
          "[%s][trajopt_only] [%5.1f%%] %s (%.1fs)",
          step_name.c_str(), fb->progress * 100.0f,
          fb->status.c_str(), fb->elapsed_sec);
      };
    opts.result_callback =
      [promise_result](
        const rclcpp_action::ClientGoalHandle<TrajOpt>::WrappedResult & res)
      {
        promise_result->set_value(res);
      };

    trajopt_client_->async_send_goal(trajopt_goal, opts);

    // goal 수락 대기 (최대 5s)
    if (future_goal.wait_for(std::chrono::seconds(5)) != std::future_status::ready
        || !future_goal.get())
    {
      rec.message = "TrajOpt goal rejected or timeout";
      RCLCPP_ERROR(get_logger(),
        "[%s][trajopt_only] TrajOpt goal 수락 실패 — 중단", step_name.c_str());
      return false;
    }

    // 결과 대기 (최대 T_init*3 + 5s)
    const double result_timeout = t_init * 3.0 + 5.0;
    if (future_result.wait_for(std::chrono::duration<double>(result_timeout))
        != std::future_status::ready)
    {
      rec.message = "TrajOpt result timeout";
      RCLCPP_ERROR(get_logger(),
        "[%s][trajopt_only] TrajOpt result 타임아웃 (%.1fs) — 중단",
        step_name.c_str(), result_timeout);
      return false;
    }

    auto wrapped = future_result.get();
    if (wrapped.code != rclcpp_action::ResultCode::SUCCEEDED) {
      rec.message = "TrajOpt action failed";
      RCLCPP_ERROR(get_logger(),
        "[%s][trajopt_only] TrajOpt Action FAILED (code=%d) — 중단",
        step_name.c_str(), static_cast<int>(wrapped.code));
      return false;
    }

    const auto & res = wrapped.result;
    if (!res->success) {
      RCLCPP_WARN(get_logger(),
        "[%s][trajopt_only] TrajOpt NLP 미수렴: %s — 현재 추정치로 계속 진행",
        step_name.c_str(), res->message.c_str());
    }

    // ── 서버 메트릭 기록 ────────────────────────────────────────────
    rec.shortcut_time_sec  = res->shortcut_time_sec;
    rec.guess_time_sec     = res->initial_guess_time_sec;
    rec.solve_time_sec     = res->solve_time_sec;
    rec.num_shortcut_pts   = res->num_shortcut_waypoints;
    rec.num_opt_points     = static_cast<int>(res->optimized_trajectory.points.size());
    rec.final_cost         = res->cost;
    rec.max_constr_viol    = res->max_constraint_violation;
    rec.mean_torque        = res->mean_torque;
    rec.max_torque         = res->max_torque;
    rec.mean_torque_rate   = res->mean_torque_rate;
    rec.max_torque_rate    = res->max_torque_rate;
    rec.solver_status      = res->success ? "converged" : "partial";
    rec.traj               = computeTrajectoryMetrics(res->optimized_trajectory);

    RCLCPP_INFO(get_logger(),
      "[%s][trajopt_only] TrajOpt — t_opt=%.3fs, J=%.4f, τ_max=%.1fNm, opt=%d pts",
      step_name.c_str(), res->t_opt, res->cost, res->max_torque, rec.num_opt_points);

    // ── 최적화 궤적 발행 + 실행 대기 ────────────────────────────────
    auto t_exec = std::chrono::steady_clock::now();
    traj_pub_->publish(res->optimized_trajectory);
    rclcpp::sleep_for(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(res->t_opt + exec_margin)));
    rec.exec_wait_sec = durationSec(t_exec, std::chrono::steady_clock::now());

    RCLCPP_INFO(get_logger(),
      "[%s][trajopt_only] 실행 완료 (%.2fs 대기)", step_name.c_str(), rec.exec_wait_sec);
    return true;
  }

  // ── IK 헬퍼 ───────────────────────────────────────────────────
  //
  // target_in_base : 입력 pose (base_link 기준)
  // seed_state     : (1) IK 검색 시작 상태, (2) L2 비용 함수 기준
  //                  → 같은 IK 분기로 수렴하도록 유도
  // max_attempts   : 내부 재시도 횟수 (1차는 seed, 2차+는 random restart)
  // 반환: 성공 여부. solution / ik_time_sec 채움.
  bool computeIKForPose(
    const geometry_msgs::msg::Pose   & target_in_base,
    const std::vector<double>        & seed_state,
    std::vector<double>              & solution,
    double                           & ik_time_sec,
    const std::string                & label,
    int                                max_attempts = 3)
  {
    auto rs = move_group_->getCurrentState(2.0);
    if (!rs) {
      RCLCPP_ERROR(get_logger(), "[%s] getCurrentState failed", label.c_str());
      return false;
    }
    const auto * jmg = rs->getJointModelGroup(get_parameter("arm_group").as_string());

    // base_link → world 변환
    const Eigen::Isometry3d world_T_base =
      rs->getGlobalLinkTransform("base_link");
    Eigen::Isometry3d target_in_base_eig = Eigen::Isometry3d::Identity();
    target_in_base_eig.translation() = Eigen::Vector3d(
      target_in_base.position.x, target_in_base.position.y, target_in_base.position.z);
    target_in_base_eig.linear() = Eigen::Quaterniond(
      target_in_base.orientation.w, target_in_base.orientation.x,
      target_in_base.orientation.y, target_in_base.orientation.z).toRotationMatrix();
    const Eigen::Isometry3d target_in_world = world_T_base * target_in_base_eig;

    geometry_msgs::msg::Pose target_world;
    target_world.position.x = target_in_world.translation().x();
    target_world.position.y = target_in_world.translation().y();
    target_world.position.z = target_in_world.translation().z();
    const Eigen::Quaterniond q_world(target_in_world.linear());
    target_world.orientation.w = q_world.w();
    target_world.orientation.x = q_world.x();
    target_world.orientation.y = q_world.y();
    target_world.orientation.z = q_world.z();

    kinematics::KinematicsQueryOptions ik_opts;
    // 기존 동작 호환: KDL 이 exact 수렴에 실패해도 근사해를 받아들임
    // (그렇지 않으면 ik_timeout 동안 수렴 못 한 모든 케이스가 ABORT 로 빠짐)
    ik_opts.return_approximate_solution = true;
    const double ik_timeout = get_parameter("ik_timeout").as_double();
    const double w_l2       = get_parameter("ik_cost_weight_l2").as_double();

    const auto cost_fn =
      [w_l2, seed_state](
        const geometry_msgs::msg::Pose &,
        const moveit::core::RobotState & sol_state,
        const moveit::core::JointModelGroup * jmg_arg,
        const std::vector<double> &) -> double
    {
      std::vector<double> prop;
      sol_state.copyJointGroupPositions(jmg_arg, prop);
      return w_l2 * computeL2Norm(prop, seed_state);
    };
    moveit::core::GroupStateValidityCallbackFn cb_fn;

    bool found = false;
    auto t0 = std::chrono::steady_clock::now();
    for (int att = 0; att < max_attempts && !found; ++att) {
      auto st = std::make_shared<moveit::core::RobotState>(*rs);
      if (att == 0) {
        // 1차: seed_state 부터 검색 시작 → 같은 IK 분기 선호
        st->setJointGroupPositions(jmg, seed_state);
      } else {
        st->setToRandomPositions(jmg);
      }
      found = st->setFromIK(jmg, target_world, ik_timeout, cb_fn, ik_opts, cost_fn);
      if (found) st->copyJointGroupPositions(jmg, solution);
    }
    ik_time_sec = durationSec(t0, std::chrono::steady_clock::now());

    if (!found) {
      RCLCPP_ERROR(get_logger(), "[%s] IK FAILED (%.3fs)", label.c_str(), ik_time_sec);
      return false;
    }
    RCLCPP_INFO(get_logger(),
      "[%s] IK OK (%.3fs, target world=(%.3f,%.3f,%.3f), L2=%.4f vs seed)",
      label.c_str(), ik_time_sec,
      target_world.position.x, target_world.position.y, target_world.position.z,
      computeL2Norm(solution, seed_state));
    return true;
  }

  // ── 충돌 검사 헬퍼 ────────────────────────────────────────────
  //
  // joints 를 적용한 상태가 PlanningScene 기준 충돌(self + world)인지 반환.
  // PSM 미초기화 시: false 반환 (검사 비활성, IK 차단하지 않음).
  bool inCollision(const std::vector<double> & joints,
                   const std::string         & arm_group_name)
  {
    if (!psm_) return false;
    planning_scene_monitor::LockedPlanningSceneRO ps(psm_);
    if (!ps) return false;

    moveit::core::RobotState state = ps->getCurrentState();
    const auto * jmg = state.getJointModelGroup(arm_group_name);
    if (!jmg) return false;
    state.setJointGroupPositions(jmg, joints);
    state.update();

    collision_detection::CollisionRequest req;
    req.group_name = arm_group_name;
    req.contacts   = false;     // 빠른 응답 (접촉 정보 미수집)
    collision_detection::CollisionResult res;
    ps->checkCollision(req, res, state);
    return res.collision;
  }

  // ── 강하 경로 충돌 검증 ───────────────────────────────────────
  //
  // start ↔ end 사이를 joint-space 에서 균등 보간하여 각 단계마다
  // 충돌 여부를 확인. 한 곳이라도 충돌 시 false 반환.
  // n_steps = 0 이면 검사를 스킵.
  bool validateJointSpaceDescent(
    const std::vector<double> & start,
    const std::vector<double> & end,
    int                          n_steps,
    const std::string          & label)
  {
    if (!psm_)        return true;     // PSM 없음 → 검사 불가, 통과 처리
    if (n_steps <= 0) return true;
    if (start.size() != end.size() || start.empty()) return false;

    const std::string arm_group_name = get_parameter("arm_group").as_string();

    for (int k = 0; k <= n_steps; ++k) {
      const double alpha = static_cast<double>(k) / n_steps;
      std::vector<double> q(start.size());
      for (size_t j = 0; j < start.size(); ++j) {
        q[j] = (1.0 - alpha) * start[j] + alpha * end[j];
      }
      if (inCollision(q, arm_group_name)) {
        const char * loc =
          (k == 0)        ? "START (candidate IK itself)" :
          (k == n_steps)  ? "END (anchor IK itself)" :
                            "intermediate";
        RCLCPP_WARN(get_logger(),
          "[%s] descent COLLISION at step %d/%d (alpha=%.2f, %s) — IK rejected",
          label.c_str(), k, n_steps, alpha, loc);
        return false;
      }
    }
    RCLCPP_INFO(get_logger(),
      "[%s] descent path OK (%d intermediate states checked)",
      label.c_str(), n_steps + 1);
    return true;
  }

  // ── 충돌-free IK (anchor 용) ───────────────────────────────────
  //
  // grasp_pose / place_pose 같은 "최종 목표 pose" 의 IK 를 구하면서,
  // 그 IK 자체가 self/world 충돌 상태인지 검증. 충돌이면 random seed
  // 로 재시도. computeIKForPose 만으로는 IK 수렴 여부만 보고 충돌은
  // 보지 못하므로, 콜리딩 분기를 그대로 grasp/place 실행에 넘기게
  // 되어 pre-approach 단계에서 무한 실패하는 문제를 막는다.
  //
  //   try 0          : seed_state 그대로 (현재 상태 우선)
  //   try 1 ~ N-1    : random restart (다른 분기 탐색)
  //
  // PSM 미초기화 시 첫 IK 결과를 그대로 채택 (검사 비활성).
  bool computeCollisionFreeIK(
    const geometry_msgs::msg::Pose   & target_in_base,
    const std::vector<double>        & seed_state,
    int                                max_attempts,
    std::vector<double>              & solution,
    double                           & ik_time_sec_total,
    const std::string                & label)
  {
    ik_time_sec_total = 0.0;

    auto rs = move_group_->getCurrentState(2.0);
    if (!rs) {
      RCLCPP_ERROR(get_logger(), "[%s] getCurrentState failed", label.c_str());
      return false;
    }
    const std::string arm_group_name = get_parameter("arm_group").as_string();
    const auto * jmg = rs->getJointModelGroup(arm_group_name);

    for (int outer = 0; outer < max_attempts; ++outer) {
      std::vector<double> seed = seed_state;
      if (outer > 0) {
        auto st = std::make_shared<moveit::core::RobotState>(*rs);
        st->setToRandomPositions(jmg);
        st->copyJointGroupPositions(jmg, seed);
      }

      const std::string sub_label =
        label + "[try" + std::to_string(outer) + "]";

      std::vector<double> candidate;
      double dt = 0.0;
      const bool ik_ok = computeIKForPose(
        target_in_base, seed, candidate, dt, sub_label, /*max_attempts=*/1);
      ik_time_sec_total += dt;
      if (!ik_ok) continue;

      // IK 자체가 충돌이면 거부
      if (psm_ && inCollision(candidate, arm_group_name)) {
        RCLCPP_WARN(get_logger(),
          "[%s] IK solution is in COLLISION — rejecting, retry with new seed",
          sub_label.c_str());
        continue;
      }

      solution = candidate;
      RCLCPP_INFO(get_logger(),
        "[%s] collision-free IK at attempt %d (IK time %.3fs)",
        label.c_str(), outer, ik_time_sec_total);
      return true;
    }

    RCLCPP_ERROR(get_logger(),
      "[%s] no collision-free IK after %d attempts (IK time %.3fs)",
      label.c_str(), max_attempts, ik_time_sec_total);
    return false;
  }

  // ── 충돌-free pre-approach IK 탐색 ────────────────────────────
  //
  // pre_pose 의 IK 를 여러 번 시도하면서 anchor_solution 까지의
  // 강하 경로가 충돌하지 않는 해를 찾는다.
  //
  //   try 0          : seed = anchor_solution (같은 IK 분기 우선)
  //   try 1 ~ N-1    : random restart (다른 분기 탐색)
  //
  // 각 candidate 마다 validateJointSpaceDescent 로 강하 경로 검증.
  // 통과 시 즉시 반환. RRT plan 검증은 호출하지 않음 — 본 단계에서는
  // 빠른 self/world 충돌 검사로 후보를 거른 뒤, 실제 plan 은 다음
  // planAndExecuteToJoints 단계에서 수행한다.
  //
  // 반환: 첫 번째로 통과한 IK 가 pre_solution 에 채워짐.
  //       모든 attempt 실패 시 false.
  bool findCollisionFreePreApproachIK(
    const geometry_msgs::msg::Pose   & pre_pose,
    const std::vector<double>        & anchor_solution,
    int                                max_outer_attempts,
    int                                n_descent_steps,
    std::vector<double>              & pre_solution,
    double                           & ik_time_sec_total,
    const std::string                & label)
  {
    ik_time_sec_total = 0.0;

    auto rs = move_group_->getCurrentState(2.0);
    if (!rs) {
      RCLCPP_ERROR(get_logger(), "[%s] getCurrentState failed", label.c_str());
      return false;
    }
    const std::string arm_group_name = get_parameter("arm_group").as_string();
    const auto * jmg = rs->getJointModelGroup(arm_group_name);

    // Anchor 자체 충돌 여부 사전 점검 (defensive — anchor 가 콜리딩이면
    // 모든 pre-approach 후보가 step N (alpha=1.0) 에서 실패하므로 무의미)
    if (psm_ && inCollision(anchor_solution, arm_group_name)) {
      RCLCPP_ERROR(get_logger(),
        "[%s] anchor solution is in COLLISION — cannot find valid pre-approach. "
        "Likely cause: grasp/place IK landed on a colliding branch.",
        label.c_str());
      return false;
    }

    for (int outer = 0; outer < max_outer_attempts; ++outer) {
      // outer 0: anchor_solution 에서 출발 (같은 분기 선호)
      // outer 1+: 무작위 seed (다른 분기 탐색)
      std::vector<double> seed = anchor_solution;
      if (outer > 0) {
        auto st = std::make_shared<moveit::core::RobotState>(*rs);
        st->setToRandomPositions(jmg);
        st->copyJointGroupPositions(jmg, seed);
      }

      const std::string sub_label =
        label + "[try" + std::to_string(outer) + "]";

      // bio_ik 1회만 호출 (max_attempts=1) → 외부 outer loop 가
      // random restart 를 통제. 총 IK 호출 수 = max_outer_attempts.
      std::vector<double> candidate;
      double dt = 0.0;
      const bool ik_ok = computeIKForPose(
        pre_pose, seed, candidate, dt, sub_label, /*max_attempts=*/1);
      ik_time_sec_total += dt;

      if (!ik_ok) continue;

      // 강하 경로 충돌 검증
      if (validateJointSpaceDescent(candidate, anchor_solution,
                                     n_descent_steps, sub_label))
      {
        pre_solution = candidate;
        RCLCPP_INFO(get_logger(),
          "[%s] valid IK + collision-free descent at attempt %d "
          "(IK time %.3fs, L2 vs anchor=%.4f)",
          label.c_str(), outer, ik_time_sec_total,
          computeL2Norm(candidate, anchor_solution));
        return true;
      }
    }

    RCLCPP_ERROR(get_logger(),
      "[%s] no collision-free IK found after %d attempts (IK time %.3fs)",
      label.c_str(), max_outer_attempts, ik_time_sec_total);
    return false;
  }

  // ── joint_target 으로 plan + execute (experiment_mode 분기) ────
  //
  // 사전에 IK 로 얻은 joint 해를 그대로 사용한다 (IK 재실행 없음).
  //
  // experiment_mode:
  //   "trajopt_only" → IK q_start/q_end 를 UR 순서로 재정렬 → runTrajoptOnly()
  //                    RRT 완전 미사용. move_group_->plan() 호출 없음.
  //   "rrt_only"     → setJointValueTarget → RRTConnect → execute
  //   "rrt_trajopt"  → RRTConnect → TrajOpt → publish
  //                    (TrajOpt 실패 시 MoveIt2 폴백)
  //
  // 결과는 rec 에 누적 (caller 가 IK time / total compute time 등 추가 후 CSV 기록).
  bool planAndExecuteToJoints(
    const std::vector<double> & joint_target,
    const std::string         & step_name,
    ExperimentRecord          & rec)
  {
    const std::string mode = get_parameter("experiment_mode").as_string();
    rec.experiment_mode = mode;

    // 현재 robot state 취득
    auto rs = move_group_->getCurrentState(2.0);
    if (!rs) {
      rec.message = "get_current_state failed";
      RCLCPP_ERROR(get_logger(), "[%s] %s", step_name.c_str(), rec.message.c_str());
      return false;
    }
    const auto * jmg = rs->getJointModelGroup(get_parameter("arm_group").as_string());

    // ── trajopt_only: RRT 없이 IK 결과를 직접 TrajOpt 서버로 전송 ──────────
    // MoveIt 내부 변수 순서(jmg->getVariableNames())와 UR 하드웨어 순서가
    // 다를 수 있으므로 reorderToUrOrder() 로 joint 이름 기반 명시적 재정렬.
    // move_group_->setJointValueTarget(), plan() 모두 호출하지 않음.
    if (mode == "trajopt_only") {
      rec.num_rrt_points = 0;

      std::vector<double> q_moveit_start;
      rs->copyJointGroupPositions(jmg, q_moveit_start);

      const std::vector<double> q_start_ur = reorderToUrOrder(q_moveit_start, jmg);
      const std::vector<double> q_end_ur   = reorderToUrOrder(joint_target,   jmg);

      RCLCPP_INFO(get_logger(),
        "[%s][trajopt_only] q_start/q_end → UR 순서 재정렬 완료, TrajOpt 서버로 전송",
        step_name.c_str());

      const bool exec_ok = runTrajoptOnly(q_start_ur, q_end_ur, step_name, rec);
      if (!exec_ok && rec.message.empty())
        rec.message = "execution failed (trajopt_only)";
      rec.success = exec_ok;
      return exec_ok;
    }

    // ── RRT 계획 (rrt_only / rrt_trajopt) ──────────────────────────────────
    std::vector<double> q_start_moveit;
    rs->copyJointGroupPositions(jmg, q_start_moveit);

    MoveGroupIface::Plan plan;
    move_group_->setJointValueTarget(joint_target);
    auto t_rrt = std::chrono::steady_clock::now();
    const auto plan_res = move_group_->plan(plan);
    rec.rrt_planning_sec = durationSec(t_rrt, std::chrono::steady_clock::now());
    rec.num_rrt_points   = static_cast<int>(
                             plan.trajectory_.joint_trajectory.points.size());

    if (!plan_res) {
      rec.message = "planning failed (RRTConnect)";
      RCLCPP_ERROR(get_logger(), "[%s] RRTConnect FAILED (%.3fs)",
        step_name.c_str(), rec.rrt_planning_sec);
      return false;
    }
    RCLCPP_INFO(get_logger(), "[%s] RRT OK (%.3fs, %d pts)",
      step_name.c_str(), rec.rrt_planning_sec, rec.num_rrt_points);

    auto pub_traj = plan.trajectory_;
    pub_traj.joint_trajectory.header.frame_id = step_name;
    pub_traj.joint_trajectory.header.stamp    = this->now();
    rrt_traj_pub_->publish(pub_traj);
    publishEePath(plan, step_name);

    // ── 실행 분기 ──
    bool exec_ok = false;

    if (mode == "rrt_only") {
      auto t_exec = std::chrono::steady_clock::now();
      const auto er = move_group_->execute(plan);
      rec.exec_wait_sec = durationSec(t_exec, std::chrono::steady_clock::now());
      exec_ok = static_cast<bool>(er);
      if (exec_ok) {
        rec.traj           = computeTrajectoryMetrics(plan.trajectory_.joint_trajectory);
        rec.num_opt_points = rec.num_rrt_points;
        rec.solver_status  = "rrt_only";
      } else {
        rec.message = "execution failed (MoveIt2 execute)";
        RCLCPP_ERROR(get_logger(), "[%s] %s", step_name.c_str(), rec.message.c_str());
      }

    } else {
      // rrt_trajopt: RRT 경로를 TrajOpt 초기 추정으로 사용
      exec_ok = runWithTrajopt(
        plan, /*use_rrt_waypoints=*/true,
        q_start_moveit, joint_target, step_name, rec);
      if (!exec_ok) {
        rec.fallback_used = true;
        RCLCPP_WARN(get_logger(), "[%s] TrajOpt 실패 → MoveIt2 폴백", step_name.c_str());
        auto t_exec = std::chrono::steady_clock::now();
        const auto er = move_group_->execute(plan);
        rec.exec_wait_sec = durationSec(t_exec, std::chrono::steady_clock::now());
        exec_ok = static_cast<bool>(er);
        if (exec_ok) {
          rec.traj           = computeTrajectoryMetrics(plan.trajectory_.joint_trajectory);
          rec.num_opt_points = rec.num_rrt_points;
          rec.solver_status  = "fallback";
        } else {
          rec.message = "execution failed (fallback execute)";
          RCLCPP_ERROR(get_logger(), "[%s] %s", step_name.c_str(), rec.message.c_str());
        }
      }
    }

    rec.success = exec_ok;
    return exec_ok;
  }

  // ── FK → /ee_path/planned publish ───────────────────────────
  void publishEePath(const MoveGroupIface::Plan & plan,
                     const std::string & step_name)
  {
    const auto robot_model = move_group_->getRobotModel();
    if (!robot_model) {
      RCLCPP_WARN(get_logger(),
        "[%s] Robot model unavailable — /ee_path/planned skipped", step_name.c_str());
      return;
    }
    moveit::core::RobotState fk_state(robot_model);
    fk_state.setToDefaultValues();
    const std::string ee_link = move_group_->getEndEffectorLink();
    const auto & jt           = plan.trajectory_.joint_trajectory;

    nav_msgs::msg::Path path;
    path.header.frame_id = move_group_->getPlanningFrame();
    path.header.stamp    = this->now();

    for (const auto & pt : jt.points) {
      std::map<std::string, double> joint_map;
      for (size_t i = 0;
           i < jt.joint_names.size() && i < pt.positions.size(); ++i)
      {
        joint_map[jt.joint_names[i]] = pt.positions[i];
      }
      fk_state.setVariablePositions(joint_map);
      fk_state.updateLinkTransforms();
      const Eigen::Isometry3d & ee_tf = fk_state.getGlobalLinkTransform(ee_link);
      const Eigen::Quaterniond   q(ee_tf.linear());

      geometry_msgs::msg::PoseStamped ps;
      ps.header             = path.header;
      ps.pose.position.x    = ee_tf.translation().x();
      ps.pose.position.y    = ee_tf.translation().y();
      ps.pose.position.z    = ee_tf.translation().z();
      ps.pose.orientation.w = q.w();
      ps.pose.orientation.x = q.x();
      ps.pose.orientation.y = q.y();
      ps.pose.orientation.z = q.z();
      path.poses.push_back(ps);
    }
    ee_path_planned_pub_->publish(path);
    RCLCPP_INFO(get_logger(),
      "[%s] EE planned path: %zu poses → /ee_path/planned",
      step_name.c_str(), path.poses.size());
  }

  // ── Cartesian 직선 경로 계획 + 실행 ──────────────────────────
  bool cartesianMove(const geometry_msgs::msg::Pose & target,
                     const std::string & step_name)
  {
    const double eef_step     = get_parameter("cartesian_eef_step").as_double();
    const double min_fraction = get_parameter("cartesian_min_fraction").as_double();

    std::vector<geometry_msgs::msg::Pose> waypoints = {target};
    moveit_msgs::msg::RobotTrajectory trajectory;
    const double fraction = move_group_->computeCartesianPath(
      waypoints, eef_step, /*jump_threshold=*/0.0, trajectory);

    if (fraction < min_fraction) {
      RCLCPP_ERROR(get_logger(),
        "[%s] Cartesian path incomplete: %.1f%% (required >= %.1f%%)",
        step_name.c_str(), fraction * 100.0, min_fraction * 100.0);
      return false;
    }
    RCLCPP_INFO(get_logger(),
      "[%s] Cartesian path %.1f%% (%zu pts) — executing",
      step_name.c_str(),
      fraction * 100.0,
      trajectory.joint_trajectory.points.size());

    const auto exec_result = move_group_->execute(trajectory);
    if (!exec_result) {
      RCLCPP_ERROR(get_logger(),
        "[%s] Cartesian execution FAILED (code=%d)", step_name.c_str(), exec_result.val);
      return false;
    }
    RCLCPP_INFO(get_logger(), "[%s] Done", step_name.c_str());
    return true;
  }

  // ── 수직 Cartesian 강하/후퇴 ──────────────────────────────────
  //
  // start_pose ↔ end_pose 사이를 z 축 방향으로만 보간하며 N 개의
  // Cartesian waypoint 를 생성, computeCartesianPath 로 실행한다.
  //   - x, y, orientation 은 end_pose 값으로 고정 (start.x/y 가 다르면 경고)
  //   - z 는 (start_z → end_z) 균등 보간 (단조 감소 또는 증가)
  //
  // 의도:
  //   pre_grasp → grasp / pre_place → place 강하 동안 EE 경로가
  //   joint planning 으로 인해 휘거나 기울어지지 않도록 Cartesian
  //   수직 경로를 강제. 후퇴는 같은 함수에 start/end 만 swap.
  //
  // 반환: false 시 cartesian_min_fraction 미달 또는 execute 실패.
  bool verticalCartesianMove(
    const geometry_msgs::msg::Pose & start_pose,
    const geometry_msgs::msg::Pose & end_pose,
    int                              n_waypoints,
    const std::string              & step_name)
  {
    if (n_waypoints < 1) n_waypoints = 1;

    const double eef_step     = get_parameter("cartesian_eef_step").as_double();
    const double min_fraction = get_parameter("cartesian_min_fraction").as_double();

    // x, y, orientation 일관성 점검 (수직 강하의 전제)
    const double dx = std::abs(start_pose.position.x - end_pose.position.x);
    const double dy = std::abs(start_pose.position.y - end_pose.position.y);
    if (dx > 1e-3 || dy > 1e-3) {
      RCLCPP_WARN(get_logger(),
        "[%s] vertical_cartesian: start/end x,y mismatch "
        "(dx=%.4fm, dy=%.4fm) — using end pose x,y for waypoints",
        step_name.c_str(), dx, dy);
    }

    // waypoint 생성: alpha = k/N, k = 1..N
    //   wp.x/y/orientation = end (고정)
    //   wp.z = (1-alpha)*start_z + alpha*end_z
    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.reserve(n_waypoints);
    for (int k = 1; k <= n_waypoints; ++k) {
      const double alpha = static_cast<double>(k) / n_waypoints;
      geometry_msgs::msg::Pose wp = end_pose;
      wp.position.z = (1.0 - alpha) * start_pose.position.z
                    +        alpha  * end_pose.position.z;
      waypoints.push_back(wp);
    }

    moveit_msgs::msg::RobotTrajectory trajectory;
    const double fraction = move_group_->computeCartesianPath(
      waypoints, eef_step, /*jump_threshold=*/0.0, trajectory);

    const double dz = std::abs(end_pose.position.z - start_pose.position.z);

    if (fraction < min_fraction) {
      RCLCPP_ERROR(get_logger(),
        "[%s] vertical_cartesian path incomplete: %.1f%% "
        "(required >= %.1f%%, %d waypoints, dz=%.4fm)",
        step_name.c_str(), fraction * 100.0, min_fraction * 100.0,
        n_waypoints, dz);
      return false;
    }
    RCLCPP_INFO(get_logger(),
      "[%s] vertical_cartesian path %.1f%% (%d waypoints → %zu pts, dz=%.4fm)",
      step_name.c_str(), fraction * 100.0,
      n_waypoints, trajectory.joint_trajectory.points.size(), dz);

    const auto er = move_group_->execute(trajectory);
    if (!er) {
      RCLCPP_ERROR(get_logger(),
        "[%s] vertical_cartesian execution FAILED (code=%d)",
        step_name.c_str(), er.val);
      return false;
    }
    RCLCPP_INFO(get_logger(), "[%s] Done", step_name.c_str());
    return true;
  }

  // ── Pick 실행 ────────────────────────────────────────────────
  //
  // 새 시퀀스 (Cartesian 의존성 제거):
  //   1) grasp IK 검증 (실패 → "grasp IK failed")
  //   2) pre_grasp IK 검증 — seed = grasp solution (같은 분기)
  //                          (실패 → "pre_grasp IK failed")
  //   3) gripper open
  //   4) plan & execute → pre_grasp (실패 → "pre_grasp planning failed" /
  //                                          "pre_grasp execution failed")
  //   5) approach_strategy:
  //      "joint"     → plan & execute → grasp (sees collision)
  //                                          (실패 → "approach planning failed" /
  //                                                  "approach execution failed")
  //      "cartesian" → computeCartesianPath (레거시)
  //                                          (실패 → "approach execution failed (cartesian)")
  //   6) gripper close
  //   7) retreat (Cartesian, best-effort)
  void executePick(const std::shared_ptr<PickGoalHandle> gh)
  {
    std::lock_guard<std::mutex> exec_lock(exec_mutex_);
    auto result = std::make_shared<Pick::Result>();

    const double offset       = get_parameter("pre_grasp_offset").as_double();
    const double grp_open     = get_parameter("gripper_open_pos").as_double();
    const double grp_close    = get_parameter("gripper_close_pos").as_double();
    const double max_effort   = get_parameter("gripper_max_effort").as_double();
    const double gripper_to   = get_parameter("gripper_timeout_sec").as_double();
    const std::string strat   = get_parameter("approach_strategy").as_string();

    auto abort_with = [&](const std::string & msg) {
      RCLCPP_ERROR(get_logger(), "[pick] ABORT: %s", msg.c_str());
      result->success = false; result->message = msg;
      gh->abort(result);
    };
    auto check_cancel = [&]() -> bool {
      if (gh->is_canceling()) {
        result->success = false; result->message = "Cancelled by client";
        gh->canceled(result);
        RCLCPP_WARN(get_logger(), "[pick] Cancelled by client");
        return true;
      }
      return false;
    };
    auto fb = [&](const std::string & status, float progress) {
      auto f = std::make_shared<Pick::Feedback>();
      f->status = status; f->progress = progress;
      gh->publish_feedback(f);
      RCLCPP_INFO(get_logger(), "[pick][%5.1f%%] %s", progress * 100.0f, status.c_str());
    };

    const geometry_msgs::msg::Pose grasp_pose =
      applyDownwardOrientation(gh->get_goal()->pick_pose);
    geometry_msgs::msg::Pose pre_grasp_pose = grasp_pose;
    pre_grasp_pose.position.z += offset;

    // ── 현재 상태(seed) 취득 ─────────────────────────────────
    std::vector<double> current_state;
    {
      auto rs = move_group_->getCurrentState(2.0);
      if (!rs) { abort_with("grasp IK failed: getCurrentState failed"); return; }
      const auto * jmg = rs->getJointModelGroup(get_parameter("arm_group").as_string());
      rs->copyJointGroupPositions(jmg, current_state);
    }

    // ── Step 1: grasp IK + 충돌 검증 ────────────────────────
    // computeCollisionFreeIK: IK 자체가 self/world 충돌 분기에 떨어지면
    // random restart 로 재시도. 콜리딩 anchor 가 그대로 pre_grasp 단계로
    // 넘어가 모든 후보가 alpha=1.0 에서 실패하는 문제를 차단.
    fb("Step1: Validating grasp IK", 0.05f);
    if (check_cancel()) return;
    std::vector<double> grasp_solution;
    double ik_grasp_sec = 0.0;
    const int n_retries_anchor =
      static_cast<int>(get_parameter("descent_max_ik_retries").as_int());
    if (!computeCollisionFreeIK(grasp_pose, current_state, n_retries_anchor,
                                  grasp_solution, ik_grasp_sec, "grasp_IK"))
    {
      abort_with("grasp IK failed (no collision-free solution)");
      return;
    }

    // ── Step 2: pre_grasp IK — seed = grasp solution + 강하 충돌 검증 ──
    // pre_grasp → grasp 경로를 joint-space 보간으로 미리 검사하여
    // self-collision 위험이 있는 IK 분기를 거부한다.
    fb("Step2: Computing pre-grasp IK + validating descent", 0.10f);
    if (check_cancel()) return;
    std::vector<double> pre_grasp_solution;
    double ik_pre_sec = 0.0;
    const int n_steps_pick   = static_cast<int>(get_parameter("descent_check_steps").as_int());
    const int n_retries_pick = static_cast<int>(get_parameter("descent_max_ik_retries").as_int());
    if (!findCollisionFreePreApproachIK(
          pre_grasp_pose, /*anchor=*/grasp_solution,
          n_retries_pick, n_steps_pick,
          pre_grasp_solution, ik_pre_sec, "pre_grasp_IK"))
    {
      abort_with("pre_grasp IK failed (no collision-free descent found)");
      return;
    }

    // ── Step 3: gripper open ────────────────────────────────
    fb("Step3: Opening gripper", 0.15f);
    if (check_cancel()) return;
    if (!controlGripper(grp_open, max_effort, gripper_to)) {
      abort_with("Failed to open gripper"); return;
    }

    // ── Step 4: plan + execute → pre_grasp ──────────────────
    fb("Step4: Moving to pre-grasp (joint plan)", 0.30f);
    if (check_cancel()) return;
    triggerMotionLog(true);
    ExperimentRecord pre_rec;
    pre_rec.trial_id   = ++trial_id_;
    pre_rec.step_name  = "pre_grasp";
    pre_rec.ik_time_sec = ik_pre_sec;
    auto t_pre = std::chrono::steady_clock::now();
    const bool pre_ok = planAndExecuteToJoints(pre_grasp_solution, "pre_grasp", pre_rec);
    pre_rec.total_compute_sec = durationSec(t_pre, std::chrono::steady_clock::now());
    triggerMotionLog(false);
    if (csv_logger_) csv_logger_->write(pre_rec);
    if (!pre_ok) {
      const bool was_planning = pre_rec.message.find("planning") != std::string::npos;
      abort_with(was_planning ? "pre_grasp planning failed: " + pre_rec.message
                              : "pre_grasp execution failed: " + pre_rec.message);
      return;
    }

    // ── Step 5: approach pre_grasp → grasp ──────────────────
    fb(std::string("Step5: Approaching grasp (") + strat + ")", 0.55f);
    if (check_cancel()) return;
    bool approach_ok = false;
    if (strat == "vertical_cartesian") {
      // 수직 강하: x/y/orientation 고정, z 만 단조 감소
      const int n_wp = static_cast<int>(
        get_parameter("vertical_cartesian_waypoints").as_int());
      approach_ok = verticalCartesianMove(
        pre_grasp_pose, grasp_pose, n_wp, "grasp_approach");
      if (!approach_ok) {
        abort_with("approach execution failed (vertical_cartesian path incomplete)");
        return;
      }
    } else if (strat == "cartesian") {
      // 레거시 직선 접근 — computeCartesianPath 단일 waypoint
      approach_ok = cartesianMove(grasp_pose, "grasp_approach");
      if (!approach_ok) {
        abort_with("approach execution failed (cartesian path incomplete)");
        return;
      }
    } else {
      // "joint": 사전 계산한 grasp_solution 으로 충돌 인지 plan/execute
      triggerMotionLog(true);
      ExperimentRecord ap_rec;
      ap_rec.trial_id   = ++trial_id_;
      ap_rec.step_name  = "grasp_approach";
      ap_rec.ik_time_sec = ik_grasp_sec;
      auto t_ap = std::chrono::steady_clock::now();
      approach_ok = planAndExecuteToJoints(grasp_solution, "grasp_approach", ap_rec);
      ap_rec.total_compute_sec = durationSec(t_ap, std::chrono::steady_clock::now());
      triggerMotionLog(false);
      if (csv_logger_) csv_logger_->write(ap_rec);
      if (!approach_ok) {
        const bool was_planning = ap_rec.message.find("planning") != std::string::npos;
        abort_with(was_planning ? "approach planning failed: " + ap_rec.message
                                : "approach execution failed: " + ap_rec.message);
        return;
      }
    }

    // ── Step 6: gripper close ───────────────────────────────
    fb("Step6: Closing gripper", 0.75f);
    if (check_cancel()) return;
    if (!controlGripper(grp_close, max_effort, gripper_to)) {
      abort_with("Failed to close gripper"); return;
    }

    // ── Step 7: retreat (best-effort, approach_strategy 동일 적용) ──
    // "vertical_cartesian": grasp → pre_grasp 수직 후퇴 (z 만 증가, 기본)
    // "joint":              pre_grasp_solution 으로 plan/exec
    // "cartesian":          단일 Cartesian waypoint (레거시)
    fb(std::string("Step7: Retreating (") + strat + ")", 0.90f);
    if (!check_cancel()) {
      bool retreat_ok = false;
      if (strat == "vertical_cartesian") {
        const int n_wp = static_cast<int>(
          get_parameter("vertical_cartesian_waypoints").as_int());
        retreat_ok = verticalCartesianMove(
          grasp_pose, pre_grasp_pose, n_wp, "grasp_retreat");
        if (!retreat_ok) {
          RCLCPP_WARN(get_logger(),
            "[pick] retreat execution failed (vertical_cartesian path incomplete) "
            "— object grasped, continuing");
        }
      } else if (strat == "cartesian") {
        retreat_ok = cartesianMove(pre_grasp_pose, "grasp_retreat");
        if (!retreat_ok) {
          RCLCPP_WARN(get_logger(),
            "[pick] retreat execution failed (cartesian path incomplete) "
            "— object grasped, continuing");
        }
      } else {
        // "joint"
        triggerMotionLog(true);
        ExperimentRecord ret_rec;
        ret_rec.trial_id    = ++trial_id_;
        ret_rec.step_name   = "grasp_retreat";
        ret_rec.ik_time_sec = ik_pre_sec;          // pre_grasp IK 재사용
        auto t_ret = std::chrono::steady_clock::now();
        retreat_ok = planAndExecuteToJoints(pre_grasp_solution, "grasp_retreat", ret_rec);
        ret_rec.total_compute_sec = durationSec(t_ret, std::chrono::steady_clock::now());
        triggerMotionLog(false);
        if (csv_logger_) csv_logger_->write(ret_rec);
        if (!retreat_ok) {
          const bool was_planning = ret_rec.message.find("planning") != std::string::npos;
          RCLCPP_WARN(get_logger(),
            "[pick] retreat %s failed: %s — object grasped, continuing",
            was_planning ? "planning" : "execution",
            ret_rec.message.c_str());
        }
      }
    }

    fb("Pick complete", 1.0f);
    result->success = true;
    result->message = "Pick completed successfully";
    gh->succeed(result);
    RCLCPP_INFO(get_logger(), "[pick] Task SUCCEEDED");
  }

  // ── Place 실행 ───────────────────────────────────────────────
  // executePick 과 동일 구조. place IK → pre_place IK(seed=place) →
  // pre_place plan/exec → approach (joint/cartesian) → release → retreat.
  void executePlace(const std::shared_ptr<PlaceGoalHandle> gh)
  {
    std::lock_guard<std::mutex> exec_lock(exec_mutex_);
    auto result = std::make_shared<Place::Result>();

    const double offset       = get_parameter("pre_grasp_offset").as_double();
    const double grp_open     = get_parameter("gripper_open_pos").as_double();
    const double max_effort   = get_parameter("gripper_max_effort").as_double();
    const double gripper_to   = get_parameter("gripper_timeout_sec").as_double();
    const std::string strat   = get_parameter("approach_strategy").as_string();

    auto abort_with = [&](const std::string & msg) {
      RCLCPP_ERROR(get_logger(), "[place] ABORT: %s", msg.c_str());
      result->success = false; result->message = msg;
      gh->abort(result);
    };
    auto check_cancel = [&]() -> bool {
      if (gh->is_canceling()) {
        result->success = false; result->message = "Cancelled by client";
        gh->canceled(result);
        RCLCPP_WARN(get_logger(), "[place] Cancelled by client");
        return true;
      }
      return false;
    };
    auto fb = [&](const std::string & status, float progress) {
      auto f = std::make_shared<Place::Feedback>();
      f->status = status; f->progress = progress;
      gh->publish_feedback(f);
      RCLCPP_INFO(get_logger(), "[place][%5.1f%%] %s", progress * 100.0f, status.c_str());
    };

    const geometry_msgs::msg::Pose place_pose =
      applyDownwardOrientation(gh->get_goal()->place_pose);
    geometry_msgs::msg::Pose pre_place_pose = place_pose;
    pre_place_pose.position.z += offset;

    std::vector<double> current_state;
    {
      auto rs = move_group_->getCurrentState(2.0);
      if (!rs) { abort_with("place IK failed: getCurrentState failed"); return; }
      const auto * jmg = rs->getJointModelGroup(get_parameter("arm_group").as_string());
      rs->copyJointGroupPositions(jmg, current_state);
    }

    // ── Step 1: place IK + 충돌 검증 ──
    // computeCollisionFreeIK: IK 자체가 self/world 충돌 분기에 떨어지면
    // random restart 로 재시도. 콜리딩 anchor 가 그대로 pre_place 단계로
    // 넘어가 모든 후보가 alpha=1.0 에서 실패하는 문제를 차단.
    fb("Step1: Validating place IK", 0.05f);
    if (check_cancel()) return;
    std::vector<double> place_solution;
    double ik_place_sec = 0.0;
    const int n_retries_anchor_place =
      static_cast<int>(get_parameter("descent_max_ik_retries").as_int());
    if (!computeCollisionFreeIK(place_pose, current_state, n_retries_anchor_place,
                                  place_solution, ik_place_sec, "place_IK"))
    {
      abort_with("place IK failed (no collision-free solution)");
      return;
    }

    // ── Step 2: pre_place IK — seed = place + 강하 충돌 검증 ──
    // pre_place → place 경로를 joint-space 보간으로 미리 검사하여
    // gripper ↔ upper_arm_link self-collision 같은 위험을 거부한다.
    fb("Step2: Computing pre-place IK + validating descent", 0.10f);
    if (check_cancel()) return;
    std::vector<double> pre_place_solution;
    double ik_pre_sec = 0.0;
    const int n_steps_place   = static_cast<int>(get_parameter("descent_check_steps").as_int());
    const int n_retries_place = static_cast<int>(get_parameter("descent_max_ik_retries").as_int());
    if (!findCollisionFreePreApproachIK(
          pre_place_pose, /*anchor=*/place_solution,
          n_retries_place, n_steps_place,
          pre_place_solution, ik_pre_sec, "pre_place_IK"))
    {
      abort_with("pre_place IK failed (no collision-free descent found)");
      return;
    }

    // ── Step 3: plan + execute → pre_place ──
    fb("Step3: Moving to pre-place (joint plan)", 0.25f);
    if (check_cancel()) return;
    triggerMotionLog(true);
    ExperimentRecord pre_rec;
    pre_rec.trial_id   = ++trial_id_;
    pre_rec.step_name  = "pre_place";
    pre_rec.ik_time_sec = ik_pre_sec;
    auto t_pre = std::chrono::steady_clock::now();
    const bool pre_ok = planAndExecuteToJoints(pre_place_solution, "pre_place", pre_rec);
    pre_rec.total_compute_sec = durationSec(t_pre, std::chrono::steady_clock::now());
    triggerMotionLog(false);
    if (csv_logger_) csv_logger_->write(pre_rec);
    if (!pre_ok) {
      const bool was_planning = pre_rec.message.find("planning") != std::string::npos;
      abort_with(was_planning ? "pre_place planning failed: " + pre_rec.message
                              : "pre_place execution failed: " + pre_rec.message);
      return;
    }

    // ── Step 4: approach pre_place → place ──
    fb(std::string("Step4: Approaching place (") + strat + ")", 0.50f);
    if (check_cancel()) return;
    bool approach_ok = false;
    if (strat == "vertical_cartesian") {
      // 수직 강하: x/y/orientation 고정, z 만 단조 감소
      const int n_wp = static_cast<int>(
        get_parameter("vertical_cartesian_waypoints").as_int());
      approach_ok = verticalCartesianMove(
        pre_place_pose, place_pose, n_wp, "place_approach");
      if (!approach_ok) {
        abort_with("approach execution failed (vertical_cartesian path incomplete)");
        return;
      }
    } else if (strat == "cartesian") {
      approach_ok = cartesianMove(place_pose, "place_approach");
      if (!approach_ok) {
        abort_with("approach execution failed (cartesian path incomplete)");
        return;
      }
    } else {
      // "joint"
      triggerMotionLog(true);
      ExperimentRecord ap_rec;
      ap_rec.trial_id   = ++trial_id_;
      ap_rec.step_name  = "place_approach";
      ap_rec.ik_time_sec = ik_place_sec;
      auto t_ap = std::chrono::steady_clock::now();
      approach_ok = planAndExecuteToJoints(place_solution, "place_approach", ap_rec);
      ap_rec.total_compute_sec = durationSec(t_ap, std::chrono::steady_clock::now());
      triggerMotionLog(false);
      if (csv_logger_) csv_logger_->write(ap_rec);
      if (!approach_ok) {
        const bool was_planning = ap_rec.message.find("planning") != std::string::npos;
        abort_with(was_planning ? "approach planning failed: " + ap_rec.message
                                : "approach execution failed: " + ap_rec.message);
        return;
      }
    }

    // ── Step 5: release ──
    fb("Step5: Releasing object", 0.75f);
    if (check_cancel()) return;
    if (!controlGripper(grp_open, max_effort, gripper_to)) {
      abort_with("Failed to open gripper at place"); return;
    }

    // ── Step 6: retreat (best-effort, approach_strategy 동일 적용) ──
    fb(std::string("Step6: Retreating (") + strat + ")", 0.90f);
    if (!check_cancel()) {
      bool retreat_ok = false;
      if (strat == "vertical_cartesian") {
        const int n_wp = static_cast<int>(
          get_parameter("vertical_cartesian_waypoints").as_int());
        retreat_ok = verticalCartesianMove(
          place_pose, pre_place_pose, n_wp, "place_retreat");
        if (!retreat_ok) {
          RCLCPP_WARN(get_logger(),
            "[place] retreat execution failed (vertical_cartesian path incomplete) "
            "— object released, continuing");
        }
      } else if (strat == "cartesian") {
        retreat_ok = cartesianMove(pre_place_pose, "place_retreat");
        if (!retreat_ok) {
          RCLCPP_WARN(get_logger(),
            "[place] retreat execution failed (cartesian path incomplete) "
            "— object released, continuing");
        }
      } else {
        // "joint"
        triggerMotionLog(true);
        ExperimentRecord ret_rec;
        ret_rec.trial_id    = ++trial_id_;
        ret_rec.step_name   = "place_retreat";
        ret_rec.ik_time_sec = ik_pre_sec;          // pre_place IK 재사용
        auto t_ret = std::chrono::steady_clock::now();
        retreat_ok = planAndExecuteToJoints(pre_place_solution, "place_retreat", ret_rec);
        ret_rec.total_compute_sec = durationSec(t_ret, std::chrono::steady_clock::now());
        triggerMotionLog(false);
        if (csv_logger_) csv_logger_->write(ret_rec);
        if (!retreat_ok) {
          const bool was_planning = ret_rec.message.find("planning") != std::string::npos;
          RCLCPP_WARN(get_logger(),
            "[place] retreat %s failed: %s — object released, continuing",
            was_planning ? "planning" : "execution",
            ret_rec.message.c_str());
        }
      }
    }

    if (get_parameter("return_home_after_place").as_bool()) {
      fb("Step7: Returning to initial pose", 0.97f);
      if (check_cancel()) return;

      std::vector<double> home_joints;
      if (!loadInitialJointTarget(home_joints)) {
        abort_with("return home failed: cannot load initial_positions.yaml");
        return;
      }

      triggerMotionLog(true);
      ExperimentRecord home_rec;
      home_rec.trial_id  = ++trial_id_;
      home_rec.step_name = "return_home";
      auto t_home = std::chrono::steady_clock::now();
      const bool home_ok =
        planAndExecuteToJoints(home_joints, "return_home", home_rec);
      home_rec.total_compute_sec =
        durationSec(t_home, std::chrono::steady_clock::now());
      triggerMotionLog(false);
      if (csv_logger_) csv_logger_->write(home_rec);

      if (!home_ok) {
        const bool was_planning =
          home_rec.message.find("planning") != std::string::npos;
        abort_with(was_planning ? "return home planning failed: " + home_rec.message
                                : "return home execution failed: " + home_rec.message);
        return;
      }
    }

    fb("Place complete", 1.0f);
    result->success = true;
    result->message = "Place completed successfully";
    gh->succeed(result);
    RCLCPP_INFO(get_logger(), "[place] Task SUCCEEDED");
  }

  // ── members ───────────────────────────────────────────────────
  std::shared_ptr<MoveGroupIface>                      move_group_;

  rclcpp_action::Server<Pick>::SharedPtr               pick_server_;
  rclcpp_action::Server<Place>::SharedPtr              place_server_;
  rclcpp_action::Client<GripperCommand>::SharedPtr     gripper_client_;
  rclcpp_action::Client<TrajOpt>::SharedPtr            trajopt_client_;

  // 최적화 궤적 직접 발행 (TrajOpt 사용 시)
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr traj_pub_;

  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr                   motion_log_pub_;
  rclcpp::Publisher<moveit_msgs::msg::RobotTrajectory>::SharedPtr     rrt_traj_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr                   ee_path_planned_pub_;

  rclcpp::CallbackGroup::SharedPtr pick_cbg_;
  rclcpp::CallbackGroup::SharedPtr place_cbg_;
  rclcpp::CallbackGroup::SharedPtr gripper_cbg_;
  rclcpp::CallbackGroup::SharedPtr trajopt_cbg_;

  std::mutex exec_mutex_;
  int trial_id_ = 0;
  std::shared_ptr<CsvLogger> csv_logger_;

  // PlanningSceneMonitor: 강하 경로 self/world 충돌 검사용.
  // null 일 수 있음 → 검사 스킵 처리.
  std::shared_ptr<planning_scene_monitor::PlanningSceneMonitor> psm_;
};

// ================================================================
// main
// ================================================================
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto move_group_node =
    rclcpp::Node::make_shared("pick_place_move_group_interface");

  rclcpp::executors::SingleThreadedExecutor mg_executor;
  mg_executor.add_node(move_group_node);
  auto mg_thread = std::thread([&mg_executor]() { mg_executor.spin(); });

  auto node = std::make_shared<PickPlaceNode>(move_group_node);

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  mg_executor.cancel();
  mg_thread.join();
  rclcpp::shutdown();
  return 0;
}
