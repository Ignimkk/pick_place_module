/**
 * edge_brain_dispatcher_node.cpp
 *
 * 역할:
 *   plate 위에 스폰된 9개 알파벳(A,B,D,E,E,G,I,N,R) 을 차례로 집어
 *   pallet 위에 "EDGE BRAIN" 순서대로 3x3 격자로 배치한다.
 *
 *   인터페이스는 *topic 기반*:
 *     - geometry_msgs/PoseStamped 를 /pick_goal · /place_goal 로 발행
 *     - std_msgs/Bool 을 /pickplace_done 로 구독 (goal_relay_node mode 1 이 발행)
 *
 *   동작 흐름 (각 알파벳마다):
 *     1) /pick_goal  PoseStamped  publish (world frame)
 *     2) /place_goal PoseStamped  publish (world frame)
 *        → goal_relay_node (trigger_mode=1) 가 두 토픽을 모두 받으면
 *          내부적으로 Pick action → Place action 을 순차 실행한다.
 *     3) /pickplace_done std_msgs/Bool 수신 대기 (성공/실패 결과)
 *        → success=true 면 다음 step 으로, false 면 시퀀스 중단
 *
 *   이 노드는 action client 를 직접 사용하지 않는다. action 호출 / 결과 처리는
 *   전적으로 goal_relay_node 가 담당한다. (테스트용 수동 명령
 *     `ros2 topic pub --once /pick_goal geometry_msgs/PoseStamped ...`
 *   와 동일한 인터페이스로 동작.)
 *
 * 좌표 산출 근거 (testbed.urdf.xacro / launch 파일):
 *   - plate  중심 (world) : (0.60, 0.25), 크기 0.60 x 0.60, 상단 z = 0.92
 *   - pallet 중심 (world) : (-0.60, 0.35), 크기 0.60 x 0.35, 상단 z = 0.92
 *
 *   - alphabet STL bbox (스케일 0.001 적용, [m]):
 *       A:(0.0991, 0.1196, 0.0300)  B:(0.0786, 0.1196, 0.0300)
 *       D:(0.1013, 0.1246, 0.0300)  E:(0.0718, 0.1196, 0.0300)
 *       G:(0.0926, 0.1207, 0.0300)  I:(0.0444, 0.1196, 0.0300)
 *       N:(0.0820, 0.1196, 0.0300)  R:(0.0923, 0.1196, 0.0300)
 *
 *   - 3x3 격자 (place 시 알파벳 yaw=+90°):
 *       x-pitch = 0.60 / 3 = 0.20 (>= STL max y-size 0.125 ✓)
 *       y-pitch = 0.35 / 3 ≈ 0.117 (>= STL max x-size 0.101 ✓)
 *
 *   - EDGE BRAIN 배치:
 *       row2(위)  : E  D  G    ← 첫 3 글자
 *       row1(중)  : E  B  R
 *       row0(아래): A  I  N    ← 마지막 3 글자
 */

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/empty.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace std::chrono_literals;

namespace {

struct Step
{
  std::string label;
  std::string model_name;           // Gazebo spawn name (= /grasp/attach/<...> suffix)
  double pick_x, pick_y;            // tool0 target xy (frame_id)
  double pick_yaw_deg;              // tool0 yaw at pick (= th_pick from user table)
  double place_center_x, place_center_y;  // desired letter center xy on pallet
  double place_x, place_y;                // tool0 target xy on pallet
  double place_yaw_deg;                   // tool0 yaw at place
};

// letter 마다의 picking 보정값 (사용자 제공 표 기준).
//   spawn_x, spawn_y          : launch 파일의 alphabet spawn 좌표 (world / base_link xy 동일)
//   pick_off_x, pick_off_y    : letter 중심 기준 pick 점 offset [m]
//                               (letter 가 world yaw=0 으로 스폰되므로
//                                local frame ≡ world frame, 단순 합산으로 적용)
//                               place 에서는 letter 중심을 pallet cell 에 맞추기 위해
//                               이 offset 을 다시 tool0 목표점에 반영한다.
//   pick_yaw_deg              : pick 시 tool0 의 yaw [deg]  (= th_pick)
struct LetterDef
{
  std::string name;
  double spawn_x, spawn_y;
  double pick_off_x, pick_off_y;
  double pick_yaw_deg;
};

constexpr double kPlateTopZ        = 0.02;
constexpr double kPalletTopZ       = 0.02;
constexpr double kAlphabetHeight   = 0.030;

// pick_place_node 가 사용하는 EE 링크 = tool0 (UR16e 손목 flange).
// Robotiq 2F-85 어댑터+gripper 구성에서 tool0 → finger tip 거리 ≈ 0.16 m.
// fingertip 을 letter mid-height(plate 위 ~0.015 m) 에 두려면
//   tool0_z = plate_top + (alphabet_height/2) + tool0_to_fingertip
//           ≈ 0.92 + 0.015 + 0.16 = 1.095
// 안전 마진 약 0.005 추가하여 기본값 1.10 사용.
constexpr double kTool0ToFingertip = 0.16;
constexpr double kDefaultGraspZ    = kPlateTopZ  + kAlphabetHeight / 2.0 + kTool0ToFingertip + 0.01;
constexpr double kDefaultPlaceZ    = kPalletTopZ + kAlphabetHeight / 2.0 + kTool0ToFingertip + 0.02;

constexpr double kPalletCenterX    = -0.60;
constexpr double kPalletCenterY    =  0.35;
constexpr double kPalletSizeX      =  0.60;
constexpr double kPalletSizeY      =  0.35;
constexpr double kPalletPitchX     = kPalletSizeX / 3.0;          // 0.20
constexpr double kPalletPitchY     = kPalletSizeY / 3.0;          // ≈ 0.11667

// EE tool0 가 -Z 를 가리키는 downward grasp 자세에 yaw 만 추가한 quaternion 생성.
// 기준 자세 q_base = (x,y,z,w) = (0, 1, 0, 0)  (pick_place_node 의 grasp_orientation_* 기본값)
//   q_result = q_yaw(yaw) ⊗ q_base
//            = (-sin(yaw/2), cos(yaw/2), 0, 0)
//
// 검증:
//   yaw =   0° → ( 0,        1,       0, 0)
//   yaw = +90° → (-√2/2,     √2/2,    0, 0)
//   yaw = -90° → ( √2/2,     √2/2,    0, 0)
inline geometry_msgs::msg::Quaternion downwardYawQuat(double yaw_rad)
{
  const double half = yaw_rad * 0.5;
  geometry_msgs::msg::Quaternion q;
  q.x = -std::sin(half);
  q.y =  std::cos(half);
  q.z =  0.0;
  q.w =  0.0;
  return q;
}

inline double deg2rad(double deg) { return deg * M_PI / 180.0; }

// place 시 letter 의 world yaw 목표.
//   3x3 격자에서 y-pitch 가 좁으므로 letter 를 90° 회전시켜 배치
//   (자세한 근거는 파일 상단 헤더 주석 참조).
constexpr double kLetterPlaceYawDeg = 90.0;

}  // namespace

class EdgeBrainDispatcher : public rclcpp::Node
{
public:
  EdgeBrainDispatcher()
  : Node("edge_brain_dispatcher")
  {
    // ── 파라미터 ────────────────────────────────────────────────
    // grasp_z / place_z 는 *tool0 (UR 손목 flange)* 의 world z 값.
    // 기본값은 fingertip 이 letter mid-height 에 도달하도록 산출.
    declare_parameter<double>("grasp_z", kDefaultGraspZ);
    declare_parameter<double>("place_z", kDefaultPlaceZ);
    // frame_id : pose 좌표가 표현되는 frame.
    //   base_link: kPlateTopZ / kPalletTopZ 등 상수가 base_link 기준으로 정의됨.
    //   world   : 위 상수를 world frame 값(예: 0.92)으로 다시 설정해야 함.
    declare_parameter<std::string>("frame_id", "base_link");
    // /pickplace_done 대기 timeout [s]. -1 이면 무제한 대기.
    declare_parameter<double>("done_timeout_sec", 100.0);
    // 첫 publish 전 subscriber 매칭 대기 timeout [s]
    declare_parameter<double>("subscriber_wait_sec", 15.0);
    // pick_goal 과 place_goal publish 사이 간격 [s] (mode 1 race 방지)
    declare_parameter<double>("publish_gap_sec", 0.2);
    // /pickplace_done 발행이 mode 1 시퀀스 종료보다 늦게 도착할 수 있으므로,
    // 받기 전에 publish 한 done 이 누락되지 않도록 latched-like 수신 윈도우.
    declare_parameter<bool>("autostart", true);
    // 실제 처리할 letter 개수.
    //   launch 의 `alphabet_count` 와 일치시켜 sim 에 없는 letter 까지
    //   pick 을 시도해 실패하는 것을 막는다. 기본값 9 = EDGE BRAIN 전체.
    declare_parameter<int>("letter_count", 9);

    // pick_place_node 에 강제 설정할 파라미터들 (시작 시 1회 적용).
    //   use_input_orientation : per-letter yaw 가 무시되지 않도록 true.
    //   gripper_timeout_sec   : 저 RTF 시뮬레이션에서 wall-clock 부족 방지.
    //   값을 -1 로 두면 해당 항목은 설정하지 않음 (기존 값 유지).
    declare_parameter<bool>("set_use_input_orientation", true);
    declare_parameter<double>("set_gripper_timeout_sec", 60.0);

    // ── pub/sub ──────────────────────────────────────────────
    pick_pub_  = create_publisher<geometry_msgs::msg::PoseStamped>("pick_goal",  10);
    place_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("place_goal", 10);
    done_sub_  = create_subscription<std_msgs::msg::Bool>(
      "pickplace_done", 10,
      [this](std_msgs::msg::Bool::SharedPtr msg) {
        onDone(msg->data);
      });

    // pick action 종료 신호 (goal_relay_node 가 발행).
    // success=true → 현재 step 의 letter 에 대해 /grasp/attach/<model> publish
    //   = DetachableJoint 가 fixed joint 생성 = contact 비용 회피.
    pick_phase_done_sub_ = create_subscription<std_msgs::msg::Bool>(
      "pick_phase_done", 10,
      [this](std_msgs::msg::Bool::SharedPtr msg) {
        onPickPhaseDone(msg->data);
      });

    steps_ = buildSteps();
    logPlanSummary();

    // ── 모든 attach publisher 를 미리 생성 (discovery race 방지) ──
    // 이전에는 step 진입 시점에 lazy 생성했는데, 새 publisher 가 만들어진
    // 직후 onPickPhaseDone 이 publish 하면 ros_gz_bridge 의 subscriber 매칭이
    // 아직 완료되지 않아 첫 메시지가 dropped 되는 경우가 있었음 (특히
    // sub 두세 단계 거치는 ROS→Ign bridge 의 경우 매칭에 수십 ms 소요).
    // → 9 letter 모두 시작 시 publisher 만들어 두면 runSequence (autostart
    //   2s 후) 시작 시점에 모두 매칭 완료 보장.
    for (const auto & st : steps_) {
      const std::string topic = "/grasp/attach/" + st.model_name;
      attach_pubs_.emplace(
        st.model_name,
        create_publisher<std_msgs::msg::Empty>(topic, 10));
    }
    RCLCPP_INFO(get_logger(),
      "[edge_brain] pre-created %zu attach publishers", attach_pubs_.size());

    if (get_parameter("autostart").as_bool()) {
      start_timer_ = create_wall_timer(
        2s,
        [this]() {
          start_timer_->cancel();
          std::thread([this]() { runSequence(); }).detach();
        });
    }
  }

private:
  // ───────────────────────────────────────────────────────────
  // EDGE BRAIN 배치 계획
  // ───────────────────────────────────────────────────────────
  std::vector<Step> buildSteps() const
  {
    // 사용자 제공 표:
    //   x_letter_to_pick / y_letter_to_pick / th_pick(=th_letter_to_pick)
    //   pick offset 은 letter 중심(=spawn 좌표) 기준이며,
    //   letter 가 world yaw=0 으로 스폰되므로 그대로 합산하면 world xy 가 된다.
    // 주의: 사용자 검증 결과 E ↔ D 의 th_pick 컨벤션이 반대였음.
    //   E1/E2 : -90° → 0°  (실제 grasp 자세는 그리퍼 초기 회전과 동일)
    //   D     :   0° → -90°(실제 grasp 자세는 E 가 쓰던 값)
    //   G/A/B/I/N/R 은 아직 동일하게 두고 실 동작 후 별도 검증.
    const LetterDef A   { "A",  0.42, 0.43,  0.034, -0.040,  -71.0 };
    const LetterDef B   { "B",  0.60, 0.43, -0.030, -0.020,   -90.0 };
    const LetterDef D   { "D",  0.78, 0.43, -0.040,  0.000, -90.0 };
    const LetterDef E1  { "E1", 0.42, 0.25,  0.010, -0.053,   0.0 };
    const LetterDef E2  { "E2", 0.60, 0.25,  0.010, -0.053,   0.0 };
    const LetterDef G   { "G",  0.78, 0.25,  0.015, -0.005, 0.0 };
    const LetterDef I   { "I",  0.42, 0.07,  0.000, -0.017,   -90.0 };
    const LetterDef N   { "N",  0.60, 0.07, -0.033, -0.045,   -90.0 };
    const LetterDef R   { "R",  0.78, 0.07, -0.040, -0.040,   -90.0 };

    auto cell_x = [](int col) { return kPalletCenterX + (col - 1) * kPalletPitchX; };
    auto cell_y = [](int row) { return kPalletCenterY + (row - 1) * kPalletPitchY; };

    struct Slot { LetterDef p; int col; int row; };
    const std::vector<Slot> full_plan = {
      { E1, 0, 2 }, { D,  1, 2 }, { G,  2, 2 },
      { E2, 0, 1 }, { B,  1, 1 }, { R,  2, 1 },
      { A,  0, 0 }, { I,  1, 0 }, { N,  2, 0 },
    };

    // letter_count 만큼만 처리. launch 의 alphabet_count 와 일치시킬 것.
    const int requested = static_cast<int>(get_parameter("letter_count").as_int());
    const int n = std::clamp(requested, 0, static_cast<int>(full_plan.size()));
    const std::vector<Slot> plan(full_plan.begin(), full_plan.begin() + n);

    std::vector<Step> steps;
    steps.reserve(plan.size());
    for (const auto & s : plan) {
      Step st;
      st.label        = s.p.name + " -> (col=" + std::to_string(s.col) +
                        ", row=" + std::to_string(s.row) + ")";
      // launch alphabet_specs_all 의 -name 인자와 동일해야 함.
      //   single-letter (A,B,D,G,I,N,R) : "alphabet_<X>"
      //   duplicate (E1, E2) : "alphabet_E1" / "alphabet_E2"
      st.model_name   = "alphabet_" + s.p.name;
      st.pick_x       = s.p.spawn_x + s.p.pick_off_x;
      st.pick_y       = s.p.spawn_y + s.p.pick_off_y;
      st.pick_yaw_deg = s.p.pick_yaw_deg;
      // place 시 gripper yaw = pick 시 yaw (수송 중 wrist 회전 없음).
      //   예) E : pick yaw = -90° → place yaw = -90° (그 자세 그대로)
      //       D : pick yaw =   0° → place yaw =   0° (회전 없이)
      //   결과적으로 letter 는 pallet 에서 world yaw = 0° (자연 자세) 로 안착.
      st.place_yaw_deg = s.p.pick_yaw_deg;

      st.place_center_x = cell_x(s.col);
      st.place_center_y = cell_y(s.row);

      // pick_off 는 "letter center -> tool0 grasp point" 벡터다.
      // pallet cell 은 letter center 목표이므로, place goal(tool0)은
      // center 에 grasp offset 을 더한 지점이어야 한다. 현재는 pick/place yaw 를
      // 같게 유지하지만, 향후 place yaw 를 바꿔도 offset 이 함께 회전되도록
      // yaw 변화량 기준으로 계산한다.
      const double yaw_delta = deg2rad(st.place_yaw_deg - st.pick_yaw_deg);
      const double c = std::cos(yaw_delta);
      const double sn = std::sin(yaw_delta);
      const double place_off_x = c * s.p.pick_off_x - sn * s.p.pick_off_y;
      const double place_off_y = sn * s.p.pick_off_x + c * s.p.pick_off_y;
      st.place_x = st.place_center_x + place_off_x;
      st.place_y = st.place_center_y + place_off_y;
      steps.push_back(st);
    }
    return steps;
  }

  void logPlanSummary() const
  {
    RCLCPP_INFO(get_logger(),
      "[edge_brain] interface = topic (publish /pick_goal & /place_goal, "
      "subscribe /pickplace_done). goal_relay_node trigger_mode=1 must be running.");
    RCLCPP_INFO(get_logger(),
      "[edge_brain] frame_id='%s'  grasp_z=%.4f  place_z=%.4f  letter_count=%zu",
      get_parameter("frame_id").as_string().c_str(),
      get_parameter("grasp_z").as_double(),
      get_parameter("place_z").as_double(),
      steps_.size());
    RCLCPP_INFO(get_logger(),
      "[edge_brain] pallet center=(%.3f, %.3f) size=(%.3f x %.3f), pitch=(%.4f, %.4f)",
      kPalletCenterX, kPalletCenterY, kPalletSizeX, kPalletSizeY,
      kPalletPitchX, kPalletPitchY);
    for (size_t i = 0; i < steps_.size(); ++i) {
      const auto & st = steps_[i];
      RCLCPP_INFO(get_logger(),
        "[edge_brain]  step %zu  %s  "
        "pick=(%.3f, %.3f) yaw=%+.1f°  "
        "place_tool=(%.3f, %.3f) center=(%.3f, %.3f) yaw=%+.1f°",
        i + 1, st.label.c_str(),
        st.pick_x,  st.pick_y,  st.pick_yaw_deg,
        st.place_x, st.place_y,
        st.place_center_x, st.place_center_y,
        st.place_yaw_deg);
    }
  }

  // ───────────────────────────────────────────────────────────
  // /pick_phase_done 콜백
  //   pick 성공 직후, 현재 letter 의 /grasp/attach/<model> 토픽으로
  //   trigger 메시지를 publish. DetachableJoint 가 tool0 ↔ letter
  //   fixed joint 를 생성하여, place 의 long-distance 이동 동안
  //   gripper-letter contact 해석 비용을 제거 (RTF 회복).
  //   pick 실패 시에는 attach 하지 않음.
  // ───────────────────────────────────────────────────────────
  void onPickPhaseDone(bool success)
  {
    if (!success) {
      RCLCPP_WARN(get_logger(),
        "[edge_brain] /pick_phase_done success=false — skip attach");
      return;
    }
    std::string letter;
    rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr pub;
    {
      std::lock_guard<std::mutex> lk(attach_mutex_);
      letter = current_letter_model_;
      pub    = current_attach_pub_;
    }
    if (letter.empty() || !pub) {
      RCLCPP_WARN(get_logger(),
        "[edge_brain] /pick_phase_done received but no current letter — ignored");
      return;
    }
    pub->publish(std_msgs::msg::Empty());
    RCLCPP_INFO(get_logger(),
      "[edge_brain] /grasp/attach/%s published (letter attached to tool0)",
      letter.c_str());
  }

  // ───────────────────────────────────────────────────────────
  // /pickplace_done 콜백
  // ───────────────────────────────────────────────────────────
  void onDone(bool success)
  {
    std::lock_guard<std::mutex> lock(done_mutex_);
    if (!done_pending_) {
      // pending 없는데 도착한 done (예: 초기 latched, 또는 외부 트리거).
      // 다음 wait 직전에 stale 로 무시되도록 stale_done_pending_ 으로만 표기.
      RCLCPP_DEBUG(get_logger(),
        "[edge_brain] /pickplace_done received but no pending wait (ignored)  success=%s",
        success ? "true" : "false");
      return;
    }
    done_pending_ = false;
    try {
      done_promise_.set_value(success);
    } catch (const std::future_error & e) {
      RCLCPP_WARN(get_logger(),
        "[edge_brain] double-set on done promise ignored (%s)", e.what());
    }
  }

  // ───────────────────────────────────────────────────────────
  // pick_place_node 에 강제 설정해야 하는 파라미터 적용 (시작 시 1회).
  //   - use_input_orientation = true  : per-letter yaw 가 무시되지 않게
  //   - gripper_timeout_sec   = N s   : 저 RTF sim 에서 wall-clock 부족 방지
  // SyncParametersClient 로 직접 set. set 값이 음수/false 이면 skip.
  // ───────────────────────────────────────────────────────────
  void applyExternalParams()
  {
    // 주의: 자신(this) 을 SyncParametersClient 에 넘기면, 내부에서 임시 executor 에
    //   add_node 를 시도하다가 "Node has already been added to an executor" 으로 abort 함
    //   (이 노드는 main 의 MultiThreadedExecutor 에 이미 등록되어 있음).
    // 회피: parameter 호출 전용 helper node 를 일회용으로 생성해서 사용.
    auto helper = rclcpp::Node::make_shared("edge_brain_param_helper");
    auto client = std::make_shared<rclcpp::SyncParametersClient>(
      helper, "/pick_place_node");
    if (!client->wait_for_service(std::chrono::seconds(10))) {
      RCLCPP_WARN(get_logger(),
        "[edge_brain] /pick_place_node parameter service not available — "
        "skipping auto parameter setup. Set use_input_orientation / "
        "gripper_timeout_sec manually if needed.");
      return;
    }

    std::vector<rclcpp::Parameter> params;
    if (get_parameter("set_use_input_orientation").as_bool()) {
      params.emplace_back("use_input_orientation", true);
    }
    const double gto = get_parameter("set_gripper_timeout_sec").as_double();
    if (gto > 0.0) {
      params.emplace_back("gripper_timeout_sec", gto);
    }
    if (params.empty()) return;

    const auto results = client->set_parameters(params);
    for (size_t i = 0; i < results.size(); ++i) {
      const auto & p = params[i];
      if (results[i].successful) {
        RCLCPP_INFO(get_logger(),
          "[edge_brain] /pick_place_node %s set OK", p.get_name().c_str());
      } else {
        RCLCPP_WARN(get_logger(),
          "[edge_brain] /pick_place_node %s set FAILED: %s",
          p.get_name().c_str(), results[i].reason.c_str());
      }
    }
  }

  // ───────────────────────────────────────────────────────────
  // 시퀀스 실행 (별도 스레드)
  // ───────────────────────────────────────────────────────────
  void runSequence()
  {
    applyExternalParams();

    if (!waitForSubscribers()) {
      RCLCPP_ERROR(get_logger(),
        "[edge_brain] no subscriber on /pick_goal or /place_goal — "
        "is goal_relay_node running with trigger_mode=1 ?");
      return;
    }

    const std::string frame_id = get_parameter("frame_id").as_string();
    const double grasp_z       = get_parameter("grasp_z").as_double();
    const double place_z       = get_parameter("place_z").as_double();
    const double done_to       = get_parameter("done_timeout_sec").as_double();
    const double pub_gap       = get_parameter("publish_gap_sec").as_double();

    // 실제 sim 에 스폰된 letter 만 처리 (launch 의 alphabet_count 와 맞춤).
    const int    raw_count     = get_parameter("letter_count").as_int();
    const size_t letter_count  = static_cast<size_t>(
      std::max(0, std::min(raw_count, static_cast<int>(steps_.size()))));
    RCLCPP_INFO(get_logger(),
      "[edge_brain] running first %zu / %zu steps (letter_count=%d)",
      letter_count, steps_.size(), raw_count);

    for (size_t i = 0; i < letter_count; ++i) {
      const auto & st = steps_[i];
      RCLCPP_INFO(get_logger(),
        "[edge_brain] === step %zu / %zu : %s ===",
        i + 1, steps_.size(), st.label.c_str());

      // ── pending done 준비 (publish 전에 promise 재설정) ──────
      {
        std::lock_guard<std::mutex> lock(done_mutex_);
        done_promise_ = std::promise<bool>{};
        done_future_  = done_promise_.get_future();
        done_pending_ = true;
      }

      // ── 이번 step letter 의 attach publisher 선택 ─────────────
      // publisher 들은 모두 생성자에서 사전 생성되어 있음 (discovery race
      // 회피용). 여기서는 lookup 만.
      {
        std::lock_guard<std::mutex> lk(attach_mutex_);
        current_letter_model_ = st.model_name;
        auto it = attach_pubs_.find(st.model_name);
        if (it == attach_pubs_.end()) {
          RCLCPP_ERROR(get_logger(),
            "[edge_brain] step %zu: attach publisher for '%s' missing — "
            "letter will not be attached. Skipping current step.",
            i + 1, st.model_name.c_str());
          current_attach_pub_ = nullptr;
        } else {
          current_attach_pub_ = it->second;
        }
      }

      // ── pick / place PoseStamped publish ────────────────────
      const auto stamp = now();
      const auto pick_q  = downwardYawQuat(deg2rad(st.pick_yaw_deg));
      const auto place_q = downwardYawQuat(deg2rad(st.place_yaw_deg));
      auto pick_msg  = makePoseStamped(stamp, frame_id,
                                       st.pick_x, st.pick_y, grasp_z, pick_q);
      auto place_msg = makePoseStamped(stamp, frame_id,
                                       st.place_x, st.place_y, place_z, place_q);

      pick_pub_->publish(pick_msg);
      RCLCPP_INFO(get_logger(),
        "[edge_brain] /pick_goal  published  pos=(%.3f, %.3f, %.3f)",
        st.pick_x, st.pick_y, grasp_z);

      // pub_gap > 0: goal_relay 가 pick_goal 수신을 먼저 처리하도록 약간 지연
      if (pub_gap > 0.0) {
        std::this_thread::sleep_for(std::chrono::duration<double>(pub_gap));
      }

      place_pub_->publish(place_msg);
      RCLCPP_INFO(get_logger(),
        "[edge_brain] /place_goal published  tool=(%.3f, %.3f, %.3f) "
        "letter_center=(%.3f, %.3f)",
        st.place_x, st.place_y, place_z,
        st.place_center_x, st.place_center_y);

      // ── /pickplace_done 대기 ────────────────────────────────
      RCLCPP_INFO(get_logger(),
        "[edge_brain] waiting for /pickplace_done (timeout %.0fs)",
        done_to);

      std::future_status fs;
      if (done_to > 0.0) {
        fs = done_future_.wait_for(std::chrono::duration<double>(done_to));
      } else {
        done_future_.wait();
        fs = std::future_status::ready;
      }

      if (fs != std::future_status::ready) {
        RCLCPP_ERROR(get_logger(),
          "[edge_brain] step %zu (%s) timed out waiting for /pickplace_done — aborting",
          i + 1, st.label.c_str());
        return;
      }
      const bool ok = done_future_.get();
      if (!ok) {
        RCLCPP_ERROR(get_logger(),
          "[edge_brain] step %zu (%s) reported failure — aborting sequence",
          i + 1, st.label.c_str());
        return;
      }
      RCLCPP_INFO(get_logger(),
        "[edge_brain] step %zu (%s) completed", i + 1, st.label.c_str());
    }

    RCLCPP_INFO(get_logger(),
      "[edge_brain] === ALL %zu pick-and-place steps completed ===",
      steps_.size());
  }

  bool waitForSubscribers()
  {
    const double t_max = get_parameter("subscriber_wait_sec").as_double();
    const auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::duration<double>(t_max);
    while (rclcpp::ok()) {
      const auto n_pick  = pick_pub_->get_subscription_count();
      const auto n_place = place_pub_->get_subscription_count();
      if (n_pick > 0 && n_place > 0) {
        RCLCPP_INFO(get_logger(),
          "[edge_brain] subscribers ready: pick_goal=%zu, place_goal=%zu",
          n_pick, n_place);
        return true;
      }
      if (std::chrono::steady_clock::now() > deadline) return false;
      std::this_thread::sleep_for(200ms);
    }
    return false;
  }

  static geometry_msgs::msg::PoseStamped makePoseStamped(
    const rclcpp::Time & stamp, const std::string & frame_id,
    double x, double y, double z,
    const geometry_msgs::msg::Quaternion & q)
  {
    geometry_msgs::msg::PoseStamped ps;
    ps.header.stamp = stamp;
    ps.header.frame_id = frame_id;
    ps.pose.position.x = x;
    ps.pose.position.y = y;
    ps.pose.position.z = z;
    ps.pose.orientation = q;
    return ps;
  }

  // ── 상태 ───────────────────────────────────────────────────
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr  pick_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr  place_pub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr           done_sub_;
  rclcpp::TimerBase::SharedPtr                                   start_timer_;
  std::vector<Step>                                              steps_;

  // done 대기용 promise/future (한 step 당 한 쌍, 재생성).
  std::mutex          done_mutex_;
  std::promise<bool>  done_promise_;
  std::future<bool>   done_future_;
  bool                done_pending_{false};

  // pick_phase_done 구독 + letter 별 attach publisher 캐시.
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr pick_phase_done_sub_;
  std::mutex          attach_mutex_;
  std::string         current_letter_model_;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr   current_attach_pub_;
  std::unordered_map<
    std::string,
    rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr> attach_pubs_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<EdgeBrainDispatcher>();
  // MultiThreadedExecutor: runSequence() 가 별도 스레드에서 future.wait_for()
  // 로 블로킹하는 동안 done_sub_ 콜백이 main executor 에서 처리되어야 한다.
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
