/**
 * pick_place_node.cpp
 *
 * ── Pick 시퀀스 ────────────────────────────────────────────────────
 *   Step 1: 그리퍼 열기
 *   Step 2: pre-grasp 이동  (pick_pose + Z offset) ← setFromIK + RRTConnect
 *   Step 3: grasp 직선 접근 (pick_pose)            ← Cartesian
 *   Step 4: 그리퍼 닫기
 *   Step 5: 직선 후퇴       (pre-grasp 복귀)       ← Cartesian
 *
 * ── Place 시퀀스 ───────────────────────────────────────────────────
 *   Step 1: pre-place 이동  (place_pose + Z offset) ← setFromIK + RRTConnect
 *   Step 2: place 직선 접근 (place_pose)            ← Cartesian
 *   Step 3: 그리퍼 열기
 *   Step 4: 직선 후퇴       (pre-place 복귀)        ← Cartesian (best-effort)
 *
 * ── 모션 계획 전략 ─────────────────────────────────────────────────
 *   pre-grasp / pre-place (장거리):
 *     setFromIK(pick_ik global, L2 비용 함수) → setJointValueTarget → RRTConnect
 *     → [옵션] trajopt_server_node 로 trajectory optimization
 *     - fallback 없음: 최대 3회 재시도(2회차~는 랜덤 시드), 모두 실패 시 ABORT
 *   grasp / place 접근·후퇴 (단거리 직선):
 *     computeCartesianPath → execute
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
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
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

// ─────────────────────────────────────────────────────────────────────────
// 실험 프레임워크 — 구조체 및 헬퍼
// ─────────────────────────────────────────────────────────────────────────

static double durationSec(
  const std::chrono::steady_clock::time_point & a,
  const std::chrono::steady_clock::time_point & b)
{
  return std::chrono::duration<double>(b - a).count();
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
    declare_parameter<double>("ik_cost_weight_l2",    0.0002);

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

    RCLCPP_INFO(get_logger(), "PickPlaceNode ready  |  /pick  /place");
  }

private:
  // ── orientation 강제 ──────────────────────────────────────────
  geometry_msgs::msg::Pose applyDownwardOrientation(
    const geometry_msgs::msg::Pose & input) const
  {
    geometry_msgs::msg::Pose out = input;
    out.orientation.x = get_parameter("grasp_orientation_x").as_double();
    out.orientation.y = get_parameter("grasp_orientation_y").as_double();
    out.orientation.z = get_parameter("grasp_orientation_z").as_double();
    out.orientation.w = get_parameter("grasp_orientation_w").as_double();
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

  // ── TrajOpt Action client ─────────────────────────────────────
  //
  // use_rrt_waypoints=true : RRT plan 의 waypoint 를 TrajOpt goal 에 패킹
  // use_rrt_waypoints=false: K=0 전송 → 서버가 Cubic Hermite 초기 추정 사용
  //                           (trajopt_only 모드)
  //
  // 성공 시 rec 에 서버 메트릭을 기록하고 optimized_trajectory 를 발행 후
  // T_opt + traj_exec_margin_sec 대기. 실패 시 false 반환.
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
    // trajopt_only(K=0) 에서도 q_start/q_end 재정렬에 사용
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
      // q_start_joints 가 이미 UR 순서라고 가정 (trajopt_only)
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
      // trajopt_only: K=0 → 서버가 Cubic Hermite 초기 추정 사용
      trajopt_goal.num_waypoints = 0;
    }

    // q_start / q_end (UR 순서로 재정렬)
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
      "[%s] TrajOpt goal 전송 — K=%d (%s), t_init=%.2fs, N=%d",
      step_name.c_str(), trajopt_goal.num_waypoints,
      use_rrt_waypoints ? "rrt_trajopt" : "trajopt_only",
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

  // ── IK + experiment_mode 분기 실행 ──────────────────────────
  //
  // experiment_mode:
  //   "rrt_only"    → IK → RRTConnect → MoveIt2 execute
  //   "trajopt_only"→ IK → TrajOpt (K=0, Hermite 초기 추정)
  //   "rrt_trajopt" → IK → RRTConnect → TrajOpt → publish
  //                   (TrajOpt 실패 시 MoveIt2 폴백)
  //
  // 결과는 rec 에 기록 후 csv_logger_ 에 씀.
  bool planAndExecuteWithIK(const geometry_msgs::msg::Pose & target,
                             const std::string & step_name)
  {
    const std::string mode = get_parameter("experiment_mode").as_string();
    ExperimentRecord rec;
    rec.trial_id        = ++trial_id_;
    rec.step_name       = step_name;
    rec.experiment_mode = mode;

    auto t_total_start = std::chrono::steady_clock::now();

    // ── 1. 현재 로봇 상태 취득 ────────────────────────────────
    auto robot_state = move_group_->getCurrentState(2.0);
    if (!robot_state) {
      RCLCPP_ERROR(get_logger(), "[%s] Failed to get current robot state", step_name.c_str());
      rec.message = "get_current_state failed";
      if (csv_logger_) csv_logger_->write(rec);
      return false;
    }
    const auto * jmg = robot_state->getJointModelGroup(
      get_parameter("arm_group").as_string());

    std::vector<double> seed_state;
    robot_state->copyJointGroupPositions(jmg, seed_state);

    // ── 2. IK 옵션 ─────────────────────────────────────────────
    kinematics::KinematicsQueryOptions ik_opts;
    ik_opts.return_approximate_solution = true;
    const double ik_timeout = get_parameter("ik_timeout").as_double();
    const double w_l2       = get_parameter("ik_cost_weight_l2").as_double();

    // ── 3. base_link → world 변환 ──────────────────────────────
    const Eigen::Isometry3d world_T_base =
      robot_state->getGlobalLinkTransform("base_link");
    Eigen::Isometry3d target_in_base = Eigen::Isometry3d::Identity();
    target_in_base.translation() = Eigen::Vector3d(
      target.position.x, target.position.y, target.position.z);
    target_in_base.linear() = Eigen::Quaterniond(
      target.orientation.w, target.orientation.x,
      target.orientation.y, target.orientation.z).toRotationMatrix();
    const Eigen::Isometry3d target_in_world = world_T_base * target_in_base;

    geometry_msgs::msg::Pose target_world;
    target_world.position.x = target_in_world.translation().x();
    target_world.position.y = target_in_world.translation().y();
    target_world.position.z = target_in_world.translation().z();
    const Eigen::Quaterniond q_world(target_in_world.linear());
    target_world.orientation.w = q_world.w();
    target_world.orientation.x = q_world.x();
    target_world.orientation.y = q_world.y();
    target_world.orientation.z = q_world.z();

    RCLCPP_INFO(get_logger(),
      "[%s][%s] target world(%.3f, %.3f, %.3f)",
      step_name.c_str(), mode.c_str(),
      target_world.position.x, target_world.position.y, target_world.position.z);

    // ── 4. IK ──────────────────────────────────────────────────
    const auto ik_cost_fn =
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
    bool                found_ik = false;
    std::vector<double> solution;
    auto base_state = move_group_->getCurrentState(2.0);

    auto t_ik_start = std::chrono::steady_clock::now();
    for (int att = 0; att < 3 && !found_ik; ++att) {
      auto st = std::make_shared<moveit::core::RobotState>(*base_state);
      if (att > 0) st->setToRandomPositions(jmg);
      found_ik = st->setFromIK(jmg, target_world, ik_timeout, cb_fn, ik_opts, ik_cost_fn);
      if (found_ik) st->copyJointGroupPositions(jmg, solution);
    }
    rec.ik_time_sec = durationSec(t_ik_start, std::chrono::steady_clock::now());

    if (!found_ik) {
      RCLCPP_ERROR(get_logger(), "[%s] IK failed — ABORT", step_name.c_str());
      rec.message = "IK failed";
      rec.total_compute_sec = durationSec(t_total_start, std::chrono::steady_clock::now());
      if (csv_logger_) csv_logger_->write(rec);
      return false;
    }
    RCLCPP_INFO(get_logger(), "[%s] IK OK (%.3fs, L2=%.4f)",
      step_name.c_str(), rec.ik_time_sec, computeL2Norm(solution, seed_state));

    // ── 5. RRT 계획 (rrt_only / rrt_trajopt) ──────────────────
    MoveGroupIface::Plan plan;
    if (mode != "trajopt_only") {
      move_group_->setJointValueTarget(solution);
      auto t_rrt = std::chrono::steady_clock::now();
      const auto plan_res = move_group_->plan(plan);
      rec.rrt_planning_sec = durationSec(t_rrt, std::chrono::steady_clock::now());
      rec.num_rrt_points   = static_cast<int>(
                               plan.trajectory_.joint_trajectory.points.size());

      if (!plan_res) {
        RCLCPP_ERROR(get_logger(), "[%s] RRTConnect FAILED", step_name.c_str());
        rec.message = "RRT planning failed";
        rec.total_compute_sec = durationSec(t_total_start, std::chrono::steady_clock::now());
        if (csv_logger_) csv_logger_->write(rec);
        return false;
      }
      RCLCPP_INFO(get_logger(), "[%s] RRT OK (%.3fs, %d pts)",
        step_name.c_str(), rec.rrt_planning_sec, rec.num_rrt_points);

      // 기록용 토픽 발행
      auto pub_traj = plan.trajectory_;
      pub_traj.joint_trajectory.header.frame_id = step_name;
      pub_traj.joint_trajectory.header.stamp    = this->now();
      rrt_traj_pub_->publish(pub_traj);
      publishEePath(plan, step_name);
    }

    // ── 6. 실행 분기 ──────────────────────────────────────────
    bool exec_ok = false;

    if (mode == "rrt_only") {
      // ── 기준선: MoveIt2 execute ──
      auto t_exec = std::chrono::steady_clock::now();
      const auto er = move_group_->execute(plan);
      rec.exec_wait_sec = durationSec(t_exec, std::chrono::steady_clock::now());
      exec_ok = static_cast<bool>(er);
      if (exec_ok) {
        rec.traj = computeTrajectoryMetrics(plan.trajectory_.joint_trajectory);
        rec.num_opt_points = rec.num_rrt_points;
        rec.solver_status  = "rrt_only";
      } else {
        RCLCPP_ERROR(get_logger(), "[%s] MoveIt2 execute FAILED", step_name.c_str());
        rec.message = "execute failed";
      }

    } else if (mode == "trajopt_only") {
      // ── TrajOpt only (no RRT waypoints) ──
      rec.num_rrt_points = 0;
      exec_ok = runWithTrajopt(
        plan, /*use_rrt_waypoints=*/false,
        seed_state, solution, step_name, rec);
      if (!exec_ok) rec.message = "trajopt_only failed";

    } else {
      // ── rrt_trajopt (기본) ──
      exec_ok = runWithTrajopt(
        plan, /*use_rrt_waypoints=*/true,
        seed_state, solution, step_name, rec);
      if (!exec_ok) {
        rec.fallback_used = true;
        RCLCPP_WARN(get_logger(), "[%s] TrajOpt 실패 → MoveIt2 폴백", step_name.c_str());
        auto t_exec = std::chrono::steady_clock::now();
        const auto er = move_group_->execute(plan);
        rec.exec_wait_sec = durationSec(t_exec, std::chrono::steady_clock::now());
        exec_ok = static_cast<bool>(er);
        if (exec_ok) {
          rec.traj = computeTrajectoryMetrics(plan.trajectory_.joint_trajectory);
          rec.num_opt_points = rec.num_rrt_points;
          rec.solver_status  = "fallback";
        } else {
          rec.message = "fallback execute failed";
        }
      }
    }

    rec.success          = exec_ok;
    rec.total_compute_sec = durationSec(t_total_start, std::chrono::steady_clock::now());
    if (csv_logger_) csv_logger_->write(rec);

    RCLCPP_INFO(get_logger(), "[%s] Done — mode=%s success=%s (%.2fs)",
      step_name.c_str(), mode.c_str(), exec_ok ? "true" : "false",
      rec.total_compute_sec);
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

  // ── Pick 실행 ────────────────────────────────────────────────
  void executePick(const std::shared_ptr<PickGoalHandle> gh)
  {
    std::lock_guard<std::mutex> exec_lock(exec_mutex_);
    auto result = std::make_shared<Pick::Result>();

    const double offset     = get_parameter("pre_grasp_offset").as_double();
    const double grp_open   = get_parameter("gripper_open_pos").as_double();
    const double grp_close  = get_parameter("gripper_close_pos").as_double();
    const double max_effort = get_parameter("gripper_max_effort").as_double();
    const double gripper_to = get_parameter("gripper_timeout_sec").as_double();

    auto abort = [&](const std::string & msg) {
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

    const geometry_msgs::msg::Pose pick_pose =
      applyDownwardOrientation(gh->get_goal()->pick_pose);
    geometry_msgs::msg::Pose pre_grasp = pick_pose;
    pre_grasp.position.z += offset;

    fb("Step1: Opening gripper", 0.05f);
    if (check_cancel()) { return; }
    if (!controlGripper(grp_open, max_effort, gripper_to)) {
      abort("Failed to open gripper"); return;
    }

    fb("Step2: Moving to pre-grasp (IK + RRTConnect [+ TrajOpt])", 0.20f);
    if (check_cancel()) { return; }
    triggerMotionLog(true);
    const bool pre_grasp_ok = planAndExecuteWithIK(pre_grasp, "pre_grasp");
    triggerMotionLog(false);
    if (!pre_grasp_ok) {
      abort("Failed to reach pre-grasp pose"); return;
    }

    fb("Step3: Approaching grasp (Cartesian)", 0.45f);
    if (check_cancel()) { return; }
    if (!cartesianMove(pick_pose, "grasp_approach")) {
      abort("Failed to approach grasp pose"); return;
    }

    fb("Step4: Closing gripper", 0.65f);
    if (check_cancel()) { return; }
    if (!controlGripper(grp_close, max_effort, gripper_to)) {
      abort("Failed to close gripper"); return;
    }

    fb("Step5: Retreating to pre-grasp (Cartesian)", 0.85f);
    if (!check_cancel()) {
      if (!cartesianMove(pre_grasp, "grasp_retreat")) {
        RCLCPP_WARN(get_logger(),
          "[pick] Retreat failed — object grasped, continuing");
      }
    }

    fb("Pick complete", 1.0f);
    result->success = true;
    result->message = "Pick completed successfully";
    gh->succeed(result);
    RCLCPP_INFO(get_logger(), "[pick] Task SUCCEEDED");
  }

  // ── Place 실행 ───────────────────────────────────────────────
  void executePlace(const std::shared_ptr<PlaceGoalHandle> gh)
  {
    std::lock_guard<std::mutex> exec_lock(exec_mutex_);
    auto result = std::make_shared<Place::Result>();

    const double offset     = get_parameter("pre_grasp_offset").as_double();
    const double grp_open   = get_parameter("gripper_open_pos").as_double();
    const double max_effort = get_parameter("gripper_max_effort").as_double();
    const double gripper_to = get_parameter("gripper_timeout_sec").as_double();

    auto abort = [&](const std::string & msg) {
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
    geometry_msgs::msg::Pose pre_place = place_pose;
    pre_place.position.z += offset;

    fb("Step1: Moving to pre-place (IK + RRTConnect [+ TrajOpt])", 0.15f);
    if (check_cancel()) { return; }
    triggerMotionLog(true);
    const bool pre_place_ok = planAndExecuteWithIK(pre_place, "pre_place");
    triggerMotionLog(false);
    if (!pre_place_ok) {
      abort("Failed to reach pre-place pose"); return;
    }

    fb("Step2: Approaching place (Cartesian)", 0.40f);
    if (check_cancel()) { return; }
    if (!cartesianMove(place_pose, "place_approach")) {
      abort("Failed to approach place pose"); return;
    }

    fb("Step3: Releasing object", 0.70f);
    if (check_cancel()) { return; }
    if (!controlGripper(grp_open, max_effort, gripper_to)) {
      abort("Failed to open gripper at place"); return;
    }

    fb("Step4: Retreating from place (Cartesian)", 0.90f);
    if (!check_cancel()) {
      cartesianMove(pre_place, "place_retreat");
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
