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
#include <future>
#include <memory>
#include <mutex>
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
    // use_trajopt=false: 기존 MoveIt2 execute 동작 (기본)
    // use_trajopt=true : TrajOpt Action server 로 최적화 후 직접 발행
    declare_parameter<bool>  ("use_trajopt",               false);
    declare_parameter<double>("trajopt_server_timeout_sec", 2.0);   // 서버 연결 대기 [s]
    declare_parameter<double>("t_init_sec",                 3.0);   // 초기 시간 추정 [s]
    declare_parameter<int>   ("trajopt_N",                  6);     // Chebyshev 차수
    declare_parameter<double>("traj_exec_margin_sec",       1.5);   // 실행 완료 여유 [s]
    declare_parameter<bool>  ("trajopt_use_reduced",        true);  // reduced-space 솔버
    declare_parameter<bool>  ("trajopt_use_free_t",         true);  // free-T 최적화

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

  // ── TrajOpt Action client: RRT 계획 → 최적화 궤적 수신 ───────
  //
  // plan 의 waypoint 를 TrajOpt goal 에 패킹하여 전송한다.
  // 성공 시 optimized_trajectory 를 /joint_trajectory_controller 에 발행하고
  // T_opt + traj_exec_margin_sec 동안 대기한다.
  // 실패 또는 서버 미연결 시 false 반환 → 호출자가 폴백 결정.
  bool runWithTrajopt(
    const MoveGroupIface::Plan & plan,
    const std::vector<double>  & q_start_joints,
    const std::vector<double>  & q_end_joints,
    const std::string          & step_name)
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

    const auto & jt = plan.trajectory_.joint_trajectory;
    const size_t K  = jt.points.size();
    if (K < 2) {
      RCLCPP_WARN(get_logger(),
        "[%s] RRT plan has < 2 points — MoveIt2 폴백", step_name.c_str());
      return false;
    }

    // ── joint 순서 인덱스 맵 구성 ───────────────────────────────
    // jt.joint_names 와 UR_JOINT_NAMES 순서가 다를 수 있으므로 재정렬
    std::vector<int> joint_idx(6, -1);
    for (size_t j = 0; j < UR_JOINT_NAMES.size(); ++j) {
      for (size_t k = 0; k < jt.joint_names.size(); ++k) {
        if (jt.joint_names[k] == UR_JOINT_NAMES[j]) {
          joint_idx[j] = static_cast<int>(k);
          break;
        }
      }
    }

    // ── TrajOpt goal 구성 ────────────────────────────────────────
    TrajOpt::Goal trajopt_goal;
    trajopt_goal.num_waypoints = static_cast<int32_t>(K);
    trajopt_goal.t_init        = t_init;
    trajopt_goal.cheb_degree   = static_cast<int32_t>(N_cheb);
    trajopt_goal.use_reduced   = use_reduced;
    trajopt_goal.use_free_t    = use_free_t;

    // timestamps
    trajopt_goal.timestamps.reserve(K);
    for (const auto & pt : jt.points) {
      double t = static_cast<double>(pt.time_from_start.sec)
               + static_cast<double>(pt.time_from_start.nanosec) * 1e-9;
      trajopt_goal.timestamps.push_back(t);
    }

    // q_waypoints (K×6, row-major, UR 순서)
    trajopt_goal.q_waypoints.reserve(K * 6);
    for (const auto & pt : jt.points) {
      for (int j = 0; j < 6; ++j) {
        int k = joint_idx[j];
        trajopt_goal.q_waypoints.push_back(
          (k >= 0 && static_cast<size_t>(k) < pt.positions.size())
            ? pt.positions[k] : 0.0);
      }
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
      "[%s] TrajOpt goal 전송 — K=%zu, t_init=%.2fs, N=%d, reduced=%s, free_t=%s",
      step_name.c_str(), K, t_init, N_cheb,
      use_reduced ? "true" : "false",
      use_free_t  ? "true" : "false");

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
        "[%s] TrajOpt NLP 미수렴: %s — 결과 궤적으로 계속 진행",
        step_name.c_str(), res->message.c_str());
      // 미수렴이어도 현재 추정치를 실행 (SLSQP 은 partial solution 을 반환함)
    }

    RCLCPP_INFO(get_logger(),
      "[%s] TrajOpt 성공 — t_opt=%.3fs, J=%.4f",
      step_name.c_str(), res->t_opt, res->cost);

    // ── 최적화 궤적 발행 + 실행 대기 ────────────────────────────
    traj_pub_->publish(res->optimized_trajectory);
    RCLCPP_INFO(get_logger(),
      "[%s] 최적화 궤적 발행 (%zu pts) — %.2fs + %.2fs margin 대기",
      step_name.c_str(),
      res->optimized_trajectory.points.size(),
      res->t_opt, exec_margin);

    rclcpp::sleep_for(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(res->t_opt + exec_margin)));

    RCLCPP_INFO(get_logger(), "[%s] 실행 완료", step_name.c_str());
    return true;
  }

  // ── setFromIK(IKCostFn) + RRTConnect 계획 + [선택] TrajOpt + 실행 ──
  bool planAndExecuteWithIK(const geometry_msgs::msg::Pose & target,
                             const std::string & step_name)
  {
    // ── 1. 현재 로봇 상태 취득 ────────────────────────────────
    auto robot_state = move_group_->getCurrentState(2.0);
    if (!robot_state) {
      RCLCPP_ERROR(get_logger(),
        "[%s] Failed to get current robot state", step_name.c_str());
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

    // ── 3. target: base_link → world 변환 ──────────────────────
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
      "[%s] target base_link(%.3f, %.3f, %.3f) → world(%.3f, %.3f, %.3f)",
      step_name.c_str(),
      target.position.x, target.position.y, target.position.z,
      target_world.position.x, target_world.position.y, target_world.position.z);

    // ── 4. IK 비용 함수 ──────────────────────────────────────
    const auto ik_cost_fn =
      [w_l2, seed_state](
        const geometry_msgs::msg::Pose &,
        const moveit::core::RobotState & solution_state,
        const moveit::core::JointModelGroup * jmg_arg,
        const std::vector<double> &) -> double
    {
      std::vector<double> proposed;
      solution_state.copyJointGroupPositions(jmg_arg, proposed);
      return w_l2 * computeL2Norm(proposed, seed_state);
    };

    // ── 5. setFromIK: 최대 3회 시도 ───────────────────────────
    const int    max_ik_retries      = 3;
    moveit::core::GroupStateValidityCallbackFn callback_fn;
    bool                found_ik = false;
    std::vector<double> solution;
    auto base_state = move_group_->getCurrentState(2.0);

    for (int attempt = 0; attempt < max_ik_retries && !found_ik; ++attempt) {
      auto state_attempt = std::make_shared<moveit::core::RobotState>(*base_state);
      if (attempt > 0) {
        state_attempt->setToRandomPositions(jmg);
        RCLCPP_INFO(get_logger(),
          "[%s] IK attempt %d/%d — random seed",
          step_name.c_str(), attempt + 1, max_ik_retries);
      }
      found_ik = state_attempt->setFromIK(
        jmg, target_world, ik_timeout, callback_fn, ik_opts, ik_cost_fn);
      if (found_ik) {
        state_attempt->copyJointGroupPositions(jmg, solution);
      }
    }

    if (!found_ik) {
      RCLCPP_ERROR(get_logger(),
        "[%s] setFromIK failed after %d attempts — ABORT",
        step_name.c_str(), max_ik_retries);
      return false;
    }
    RCLCPP_INFO(get_logger(),
      "[%s] IK found — L2 cost: %.4f",
      step_name.c_str(), computeL2Norm(solution, seed_state));

    // ── 6. IK 해 → RRTConnect 계획 ────────────────────────────
    move_group_->setJointValueTarget(solution);
    MoveGroupIface::Plan plan;
    const auto plan_result = move_group_->plan(plan);
    if (!plan_result) {
      RCLCPP_ERROR(get_logger(),
        "[%s] RRTConnect planning FAILED (code=%d)",
        step_name.c_str(), plan_result.val);
      return false;
    }
    RCLCPP_INFO(get_logger(),
      "[%s] Plan OK (%.3f s, %zu pts)",
      step_name.c_str(),
      plan.planning_time_,
      plan.trajectory_.joint_trajectory.points.size());

    // ── 7. RRT 궤적 publish (기록용 — rrt_path_recorder_node) ──
    auto pub_traj = plan.trajectory_;
    pub_traj.joint_trajectory.header.frame_id = step_name;
    pub_traj.joint_trajectory.header.stamp    = this->now();
    rrt_traj_pub_->publish(pub_traj);

    // ── 8. FK → EE 경로 publish ──────────────────────────────
    publishEePath(plan, step_name);

    // ── 9. 실행: TrajOpt or MoveIt2 execute ──────────────────
    const bool use_trajopt = get_parameter("use_trajopt").as_bool();
    if (use_trajopt) {
      // q_start: seed_state (IK 기준 현재 위치)
      // q_end:   solution  (IK 해)
      const bool trajopt_ok = runWithTrajopt(plan, seed_state, solution, step_name);
      if (trajopt_ok) {
        return true;
      }
      RCLCPP_WARN(get_logger(),
        "[%s] TrajOpt 실패 — MoveIt2 execute 로 폴백", step_name.c_str());
    }

    // ── 10. MoveIt2 execute (기본 또는 폴백) ─────────────────
    const auto exec_result = move_group_->execute(plan);
    if (!exec_result) {
      RCLCPP_ERROR(get_logger(),
        "[%s] Execution FAILED (code=%d)", step_name.c_str(), exec_result.val);
      return false;
    }
    RCLCPP_INFO(get_logger(), "[%s] Done", step_name.c_str());
    return true;
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
