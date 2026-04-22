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
 *     - fallback 없음: 최대 3회 재시도(2회차~는 랜덤 시드), 모두 실패 시 ABORT
 *   grasp / place 접근·후퇴 (단거리 직선):
 *     computeCartesianPath → execute
 *
 * ── 파라미터 (config/pick_place_params.yaml) ───────────────────────
 *   arm_group, pre_grasp_offset, gripper_*, velocity_scaling,
 *   acceleration_scaling, planning_time, num_planning_attempts,
 *   cartesian_eef_step, cartesian_min_fraction, grasp_orientation_*,
 *   ik_timeout, ik_cost_weight_l2
 */

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/kinematics_base/kinematics_base.h>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <control_msgs/action/gripper_command.hpp>
#include <std_msgs/msg/bool.hpp>

#include <pick_place_module/action/pick.hpp>
#include <pick_place_module/action/place.hpp>

#include <Eigen/Dense>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ── IK 비용 함수 헬퍼 ─────────────────────────────────────────────────
// setFromIK 의 IKCostFn 인수로 전달되어, 여러 IK 해 중 최적 해 선택.
// L2 norm: 현재 관절값(seed)과 후보 해 사이의 유클리드 거리 제곱합.

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

class PickPlaceNode : public rclcpp::Node
{
public:
  using Pick            = pick_place_module::action::Pick;
  using Place           = pick_place_module::action::Place;
  using PickGoalHandle  = rclcpp_action::ServerGoalHandle<Pick>;
  using PlaceGoalHandle = rclcpp_action::ServerGoalHandle<Place>;
  using GripperCommand  = control_msgs::action::GripperCommand;
  using MoveGroupIface  = moveit::planning_interface::MoveGroupInterface;

  /**
   * @param move_group_node  "pick_place_move_group_interface" 전용 노드.
   *                         main() 에서 미리 생성하여 SingleThreadedExecutor 로 spin 중.
   *                         MoveGroupInterface 는 이 노드로 초기화되어
   *                         move_group 서버와 독립적으로 통신함.
   */
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

    // ── setFromIK + 비용 함수 관련 파라미터 ──────────────────────
    declare_parameter<double>("ik_timeout",           0.5);
    declare_parameter<double>("ik_cost_weight_l2",    0.0002);

    // ── MoveGroupInterface 초기화 ────────────────────────────
    // move_group_node 는 main() 의 SingleThreadedExecutor 가 별도 스레드에서 spin.
    // 이 구조로 MoveGroupInterface 가 robot state / planning scene 구독을 executor 간섭 없이 안정적으로 수신함.
    const std::string arm = get_parameter("arm_group").as_string();
    move_group_ = std::make_shared<MoveGroupIface>(move_group_node, arm);
    move_group_->setPlannerId("RRTConnectkConfigDefault"); //RRTstarkConfigDefault
    move_group_->setMaxVelocityScalingFactor(get_parameter("velocity_scaling").as_double());
    move_group_->setMaxAccelerationScalingFactor(get_parameter("acceleration_scaling").as_double());
    move_group_->setPlanningTime(get_parameter("planning_time").as_double());
    move_group_->setNumPlanningAttempts(get_parameter("num_planning_attempts").as_int());
    move_group_->setWorkspace(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0);

    // 기준 프레임을 base_link 로 설정. -> setFromIK 는 world 프레임 기준 포즈를 요구함. 그래서 이거 적용 안됨.
    move_group_->setPoseReferenceFrame("base_link");

    RCLCPP_INFO(get_logger(), "Planning group      : %s", arm.c_str());
    RCLCPP_INFO(get_logger(), "Planning frame      : %s", move_group_->getPlanningFrame().c_str());
    RCLCPP_INFO(get_logger(), "Pose reference frame: %s", move_group_->getPoseReferenceFrame().c_str());
    RCLCPP_INFO(get_logger(), "End effector link   : %s", move_group_->getEndEffectorLink().c_str());

    // ── callback groups ───────────────────────────────────────────
    gripper_cbg_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    pick_cbg_    = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    place_cbg_   = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    // ── gripper action client ─────────────────────────────────────
    gripper_client_ = rclcpp_action::create_client<GripperCommand>(
      this,
      "robotiq_gripper_controller/gripper_cmd",
      gripper_cbg_);

    // ── pick action server ────────────────────────────────────────
    pick_server_ = rclcpp_action::create_server<Pick>(
      this,
      "pick",
      [this](const rclcpp_action::GoalUUID &,
             std::shared_ptr<const Pick::Goal>) {
        RCLCPP_INFO(get_logger(), "[pick] Goal received — accepted");
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
      },
      [this](const std::shared_ptr<PickGoalHandle>) {
        RCLCPP_WARN(get_logger(), "[pick] Cancel requested");
        move_group_->stop();
        return rclcpp_action::CancelResponse::ACCEPT;
      },
      [this](const std::shared_ptr<PickGoalHandle> gh) {
        // 별도 스레드: action cbg 를 블로킹하지 않기 위함
        std::thread{[this, gh]() { executePick(gh); }}.detach();
      },
      rcl_action_server_get_default_options(),
      pick_cbg_);

    // ── place action server ───────────────────────────────────────
    place_server_ = rclcpp_action::create_server<Place>(
      this,
      "place",
      [this](const rclcpp_action::GoalUUID &,
             std::shared_ptr<const Place::Goal>) {
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
      rcl_action_server_get_default_options(),
      place_cbg_);

    // ── motion_logger 트리거 publisher ────────────────────────────
    // motion_logger_node 에 RRT 구간 시작/종료를 알림.
    // true  → 초기화 + 기록 시작
    // false → 기록 중단 + 요약 출력
    motion_log_pub_ = create_publisher<std_msgs::msg::Bool>(
      "/motion_logger/record", 10);

    // ── RRT 궤적 publisher ────────────────────────────────────────
    // rrt_path_recorder_node 에서 구독하여 CSV 로 저장.
    // joint_trajectory.header.frame_id = step_name (pre_grasp / pre_place).
    rrt_traj_pub_ = create_publisher<moveit_msgs::msg::RobotTrajectory>(
      "/pick_place/rrt_trajectory", 10);

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
      [promise](
        const rclcpp_action::ClientGoalHandle<GripperCommand>::SharedPtr & gh) {
        if (!gh) { promise->set_value(false); }
      };
    opts.result_callback =
      [promise](
        const rclcpp_action::ClientGoalHandle<GripperCommand>::WrappedResult & res) {
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

  // ── motion_logger 트리거 헬퍼 ────────────────────────────────
  void triggerMotionLog(bool start)
  {
    std_msgs::msg::Bool msg;
    msg.data = start;
    motion_log_pub_->publish(msg);
  }

  // ── setFromIK(IKCostFn) + RRTConnect 계획 + 실행 ────────────
  //
  // ① base_link 기준 target → world(planning frame) 변환
  // ② pick_ik global 모드로 IK 탐색 (L2 비용 함수: 현재 자세 기준 이동량 최소화)
  // ③ IK 해 → setJointValueTarget → RRTConnect 경로 계획 (내부 IK 호출 없음)
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

    // IK 비용 기준: 현재 자세 (재시도 시에도 동일 기준 유지)
    std::vector<double> seed_state;
    robot_state->copyJointGroupPositions(jmg, seed_state);

    // ── 2. IK 옵션 ─────────────────────────────────────────────
    kinematics::KinematicsQueryOptions ik_opts;
    ik_opts.return_approximate_solution = true;     // true: 수렴 허용 오차 내 근사 해 수용.

    const double ik_timeout = get_parameter("ik_timeout").as_double();
    const double w_l2       = get_parameter("ik_cost_weight_l2").as_double();

    // ── 3. 목표 포즈를 base_link → world(planning frame) 으로 변환 ─
    // setFromIK 는 항상 world(planning frame) 기준 포즈를 요구함.
    // getGlobalLinkTransform("base_link"): world 프레임 내 base_link 의 자세.
    const Eigen::Isometry3d world_T_base =
      robot_state->getGlobalLinkTransform("base_link");

    Eigen::Isometry3d target_in_base = Eigen::Isometry3d::Identity();
    target_in_base.translation() = Eigen::Vector3d(
      target.position.x, target.position.y, target.position.z);
    target_in_base.linear() = Eigen::Quaterniond(
      target.orientation.w,
      target.orientation.x,
      target.orientation.y,
      target.orientation.z).toRotationMatrix();

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

    // ── 4. 비용 함수 ──────────────────────────────────────────
    const auto ik_cost_fn =
      [w_l2, seed_state](
        const geometry_msgs::msg::Pose & /*goal_pose*/,
        const moveit::core::RobotState & solution_state,
        const moveit::core::JointModelGroup * jmg_arg,
        const std::vector<double> & /*seed*/) -> double
    {
      std::vector<double> proposed;
      solution_state.copyJointGroupPositions(jmg_arg, proposed);
      return w_l2 * computeL2Norm(proposed, seed_state);
    };

    // ── 5. setFromIK: 최대 max_ik_retries 회 시도 ──────────────
    const int    max_ik_retries      = 3;
    const double per_attempt_timeout = ik_timeout;   // 시도마다 전체 타임아웃 부여
    moveit::core::GroupStateValidityCallbackFn callback_fn;

    bool                found_ik = false;
    std::vector<double> solution;

    // 로봇 상태를 한 번만 취득 후 복사본으로 재시도 (중복 getCurrentState 호출 방지)
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
        jmg, target_world, per_attempt_timeout, callback_fn, ik_opts, ik_cost_fn);

      if (found_ik) {
        state_attempt->copyJointGroupPositions(jmg, solution);
      }
    }

    if (!found_ik) {
      RCLCPP_ERROR(get_logger(),
        "[%s] setFromIK failed after %d attempts (total %.2f s) — ABORT",
        step_name.c_str(), max_ik_retries, ik_timeout);
      return false;
    }

    RCLCPP_INFO(get_logger(),
      "[%s] IK found — L2 cost: %.4f",
      step_name.c_str(), computeL2Norm(solution, seed_state));

    // ── 6. IK 해 → 관절 목표 → RRTConnect 계획 ─────────────────
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

    // ── 7. RRT 궤적 publish (rrt_path_recorder_node 가 CSV 저장) ─
    // frame_id 에 step_name 을 임시 태깅 — 이 토픽은 TF/MoveIt 이 소비하지 않음.
    auto pub_traj = plan.trajectory_;
    pub_traj.joint_trajectory.header.frame_id = step_name;
    pub_traj.joint_trajectory.header.stamp    = this->now();
    rrt_traj_pub_->publish(pub_traj);

    // ── 9. 실행 ──────────────────────────────────────────────────
    const auto exec_result = move_group_->execute(plan);
    if (!exec_result) {
      RCLCPP_ERROR(get_logger(),
        "[%s] Execution FAILED (code=%d)", step_name.c_str(), exec_result.val);
      return false;
    }
    RCLCPP_INFO(get_logger(), "[%s] Done", step_name.c_str());
    return true;
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

  // ── Pick 실행 (별도 스레드에서 호출) ─────────────────────────
  //
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
      result->success = false;
      result->message = msg;
      gh->abort(result);
    };

    auto check_cancel = [&]() -> bool {
      if (gh->is_canceling()) {
        result->success = false;
        result->message = "Cancelled by client";
        gh->canceled(result);
        RCLCPP_WARN(get_logger(), "[pick] Cancelled by client");
        return true;
      }
      return false;
    };

    auto fb = [&](const std::string & status, float progress) {
      auto f      = std::make_shared<Pick::Feedback>();
      f->status   = status;
      f->progress = progress;
      gh->publish_feedback(f);
      RCLCPP_INFO(get_logger(),
        "[pick][%5.1f%%] %s", progress * 100.0f, status.c_str());
    };

    const geometry_msgs::msg::Pose pick_pose =
      applyDownwardOrientation(gh->get_goal()->pick_pose);
    geometry_msgs::msg::Pose pre_grasp = pick_pose;
    pre_grasp.position.z += offset;

    // Step 1: 그리퍼 열기
    fb("Step1: Opening gripper", 0.05f);
    if (check_cancel()) { return; }
    if (!controlGripper(grp_open, max_effort, gripper_to)) {
      abort("Failed to open gripper"); return;
    }

    // Step 2: pre-grasp 이동 (setFromIK cost function + RRTConnect
    fb("Step2: Moving to pre-grasp (IK cost fn + RRTConnect)", 0.20f);
    if (check_cancel()) { return; }
    triggerMotionLog(true);
    const bool pre_grasp_ok = planAndExecuteWithIK(pre_grasp, "pre_grasp"); // planAndExecuteWithIK: 비용 함수(이동량로 최적 IK 해 선택 후 계획/실행
    triggerMotionLog(false); // triggerMotionLog(true/false) 로 motion_logger_node 의 기록 구간 제어.
    if (!pre_grasp_ok) {
      abort("Failed to reach pre-grasp pose"); return;
    }

    // Step 3: grasp 직선 접근 (Cartesian — 물체 근처 직선 이동)
    fb("Step3: Approaching grasp (Cartesian)", 0.45f);
    if (check_cancel()) { return; }
    if (!cartesianMove(pick_pose, "grasp_approach")) {
      abort("Failed to approach grasp pose"); return;
    }

    // Step 4: 그리퍼 닫기
    fb("Step4: Closing gripper", 0.65f);
    if (check_cancel()) { return; }
    if (!controlGripper(grp_close, max_effort, gripper_to)) {
      abort("Failed to close gripper"); return;
    }

    // Step 5: 직선 후퇴 (Cartesian, best-effort)
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

  // ── Place 실행 (별도 스레드에서 호출) ────────────────────────
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
      result->success = false;
      result->message = msg;
      gh->abort(result);
    };

    auto check_cancel = [&]() -> bool {
      if (gh->is_canceling()) {
        result->success = false;
        result->message = "Cancelled by client";
        gh->canceled(result);
        RCLCPP_WARN(get_logger(), "[place] Cancelled by client");
        return true;
      }
      return false;
    };

    auto fb = [&](const std::string & status, float progress) {
      auto f      = std::make_shared<Place::Feedback>();
      f->status   = status;
      f->progress = progress;
      gh->publish_feedback(f);
      RCLCPP_INFO(get_logger(),
        "[place][%5.1f%%] %s", progress * 100.0f, status.c_str());
    };

    const geometry_msgs::msg::Pose place_pose =
      applyDownwardOrientation(gh->get_goal()->place_pose);
    geometry_msgs::msg::Pose pre_place = place_pose;
    pre_place.position.z += offset;

    // Step 1: pre-place 이동 (setFromIK cost function + RRTConnect)
    fb("Step1: Moving to pre-place (IK cost fn + RRTConnect)", 0.15f);
    if (check_cancel()) { return; }
    triggerMotionLog(true);
    const bool pre_place_ok = planAndExecuteWithIK(pre_place, "pre_place");
    triggerMotionLog(false);
    if (!pre_place_ok) {
      abort("Failed to reach pre-place pose"); return;
    }

    // Step 2: place 직선 접근 (Cartesian)
    fb("Step2: Approaching place (Cartesian)", 0.40f);
    if (check_cancel()) { return; }
    if (!cartesianMove(place_pose, "place_approach")) {
      abort("Failed to approach place pose"); return;
    }

    // Step 3: 그리퍼 열기 (해제)
    fb("Step3: Releasing object", 0.70f);
    if (check_cancel()) { return; }
    if (!controlGripper(grp_open, max_effort, gripper_to)) {
      abort("Failed to open gripper at place"); return;
    }

    // Step 4: 직선 후퇴 (Cartesian, best-effort)
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
  std::shared_ptr<MoveGroupIface>                  move_group_;

  rclcpp_action::Server<Pick>::SharedPtr           pick_server_;
  rclcpp_action::Server<Place>::SharedPtr          place_server_;
  rclcpp_action::Client<GripperCommand>::SharedPtr gripper_client_;

  // motion_logger_node 트리거: RRT 구간 시작(true)/종료(false)
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr motion_log_pub_;

  // rrt_path_recorder_node 용: RRT 궤적 publish
  rclcpp::Publisher<moveit_msgs::msg::RobotTrajectory>::SharedPtr rrt_traj_pub_;

  rclcpp::CallbackGroup::SharedPtr pick_cbg_;
  rclcpp::CallbackGroup::SharedPtr place_cbg_;
  rclcpp::CallbackGroup::SharedPtr gripper_cbg_;

  std::mutex exec_mutex_;   // pick / place 동시 실행 방지
};

// ================================================================
// main
// ================================================================
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  // ── move_group_node 생성 및 spin ────────────────────────────
  auto move_group_node =
    rclcpp::Node::make_shared("pick_place_move_group_interface");

  rclcpp::executors::SingleThreadedExecutor mg_executor;
  mg_executor.add_node(move_group_node);
  auto mg_thread = std::thread([&mg_executor]() { mg_executor.spin(); });

  // ── 메인 노드 생성 ──────────────────────────────────────────
  auto node = std::make_shared<PickPlaceNode>(move_group_node);

  // ── MultiThreadedExecutor ────────────────────────────────────
  // pick_cbg_ / place_cbg_  : action server 콜백
  // gripper_cbg_            : gripper result callback (promise.set_value)
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  // ── 정리 ────────────────────────────────────────────────────
  mg_executor.cancel();
  mg_thread.join();

  rclcpp::shutdown();
  return 0;
}
