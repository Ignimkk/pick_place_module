/**
 * goal_relay_node.cpp
 *
 * 역할:
 *   - /pick_goal  (geometry_msgs/PoseStamped) 토픽 수신
 *   - /place_goal (geometry_msgs/PoseStamped) 토픽 수신
 *   - trigger_mode 에 따라 pick / place action goal 을 서버로 전송
 *
 * trigger_mode:
 *   0 → pick_goal 수신 시 즉시 Pick action 실행
 *         place_goal 수신 시 즉시 Place action 실행 (독립)
 *   1 → pick_goal + place_goal 모두 수신 후 pick → place 순차 실행
 *
 * ── 설계 / 안정성 ──────────────────────────────────────────────────
 *   - subscriptions: 기본 MutuallyExclusive 콜백 그룹
 *   - action clients: 별도 Reentrant 콜백 그룹 (client_cbg_)
 *       → subscription 과 action client 콜백이 같은 single thread 에서
 *         직렬화되어 result_callback 이 늦게 fire 되는 문제 방지
 *       → 여러 client 콜백(goal_response, feedback, result)이 동시 처리 가능
 *   - mode 1 의 동기 대기는 async_send_goal → goal_handle_future,
 *           이후 async_get_result → result_future 패턴으로 명확히 분리
 *   - 모든 future.wait_for() 에 timeout 적용 (deadlock 방지)
 *   - 결과 callback 은 atomic guard 로 1회만 set_value (이중 set 시 예외 방지)
 *   - res.result / res.code 모두 nullptr / 비정상 코드 방어
 */

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/callback_group.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <pick_place_module/action/pick.hpp>
#include <pick_place_module/action/place.hpp>

#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <thread>

using namespace std::chrono_literals;

class GoalRelayNode : public rclcpp::Node
{
public:
  using Pick            = pick_place_module::action::Pick;
  using Place           = pick_place_module::action::Place;
  using PickGoalHandle  = rclcpp_action::ClientGoalHandle<Pick>;
  using PlaceGoalHandle = rclcpp_action::ClientGoalHandle<Place>;

  GoalRelayNode()
  : Node("goal_relay_node")
  {
    declare_parameter<int>("trigger_mode", 0);
    // 단일 action 결과를 기다리는 최대 시간 [s]. 0 이하 = 무제한 (비추천).
    declare_parameter<double>("action_result_timeout_sec", 600.0);

    // ── 콜백 그룹 분리 ─────────────────────────────────────────────
    // subscription 은 default(MutuallyExclusive),
    // action client 는 Reentrant → result/feedback 가 sub 콜백에 막히지 않음
    client_cbg_ = create_callback_group(
      rclcpp::CallbackGroupType::Reentrant);

    rclcpp::SubscriptionOptions sub_opts;   // subscription 은 기본 그룹

    pick_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "pick_goal", 10,
      [this](geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        pick_pose_     = msg->pose;
        pick_received_ = true;
        RCLCPP_INFO(get_logger(),
          "[goal_relay] pick_goal received  pos=(%.3f, %.3f, %.3f)",
          msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
        onPickReceived();
      }, sub_opts);

    place_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "place_goal", 10,
      [this](geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        place_pose_     = msg->pose;
        place_received_ = true;
        RCLCPP_INFO(get_logger(),
          "[goal_relay] place_goal received  pos=(%.3f, %.3f, %.3f)",
          msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
        onPlaceReceived();
      }, sub_opts);

    // action client → 별도 Reentrant 그룹
    pick_client_  = rclcpp_action::create_client<Pick>(
      this, "pick", client_cbg_);
    place_client_ = rclcpp_action::create_client<Place>(
      this, "place", client_cbg_);

    RCLCPP_INFO(get_logger(),
      "[goal_relay] GoalRelayNode started  trigger_mode=%ld",
      get_parameter("trigger_mode").as_int());
  }

private:
  // ---------------------------------------------------------------
  // mutex_ 보유 상태에서 호출
  // ---------------------------------------------------------------
  void onPickReceived()
  {
    if (get_parameter("trigger_mode").as_int() == 0) {
      if (executing_pick_) {
        RCLCPP_WARN(get_logger(),
          "[goal_relay][mode0] Pick already executing — new pick_goal ignored");
        return;
      }
      sendPickAsync();
    } else {
      tryTriggerBoth();
    }
  }

  void onPlaceReceived()
  {
    if (get_parameter("trigger_mode").as_int() == 0) {
      if (executing_place_) {
        RCLCPP_WARN(get_logger(),
          "[goal_relay][mode0] Place already executing — new place_goal ignored");
        return;
      }
      sendPlaceAsync();
    } else {
      tryTriggerBoth();
    }
  }

  // ---------------------------------------------------------------
  // Mode 1: 두 토픽 모두 준비된 경우 순차 실행 스레드 기동
  // mutex_ 보유 상태에서 호출
  // ---------------------------------------------------------------
  void tryTriggerBoth()
  {
    if (!pick_received_ || !place_received_) {
      RCLCPP_INFO(get_logger(),
        "[goal_relay][mode1] Waiting for both goals  pick=%s  place=%s",
        pick_received_  ? "OK" : "waiting",
        place_received_ ? "OK" : "waiting");
      return;
    }
    if (sequence_running_.load()) {
      RCLCPP_WARN(get_logger(),
        "[goal_relay][mode1] Sequence already running — new goals ignored");
      return;
    }

    geometry_msgs::msg::Pose pick_p  = pick_pose_;
    geometry_msgs::msg::Pose place_p = place_pose_;
    pick_received_   = false;
    place_received_  = false;
    executing_pick_  = true;
    executing_place_ = true;
    sequence_running_.store(true);

    RCLCPP_INFO(get_logger(),
      "[goal_relay] trigger_mode=1: both goals received → pick→place sequence start");

    // 별도 스레드: future.get() 이 메인 executor 를 블로킹하지 않도록
    std::thread([this, pick_p, place_p]() {
      sequentialPickThenPlace(pick_p, place_p);
    }).detach();
  }

  // ---------------------------------------------------------------
  // Mode 1: pick → place 순차 실행 (별도 스레드)
  // ---------------------------------------------------------------
  void sequentialPickThenPlace(
    const geometry_msgs::msg::Pose & pick_p,
    const geometry_msgs::msg::Pose & place_p)
  {
    // ── Pick ────────────────────────────────────────────────────
    RCLCPP_INFO(get_logger(), "[goal_relay] sending pick goal");
    const bool pick_ok = sendPickSync(pick_p);

    if (!pick_ok) {
      RCLCPP_ERROR(get_logger(),
        "[goal_relay] pick failed — aborting place (sequence aborted)");
      resetExecutingFlags();
      return;
    }
    RCLCPP_INFO(get_logger(), "[goal_relay] pick succeeded");

    {
      std::lock_guard<std::mutex> lock(mutex_);
      executing_pick_ = false;
    }

    // ── Place ───────────────────────────────────────────────────
    RCLCPP_INFO(get_logger(), "[goal_relay] sending place goal");
    const bool place_ok = sendPlaceSync(place_p);

    if (place_ok) {
      RCLCPP_INFO(get_logger(), "[goal_relay] place succeeded");
      RCLCPP_INFO(get_logger(), "[goal_relay] sequence completed");
    } else {
      RCLCPP_ERROR(get_logger(),
        "[goal_relay] place failed — sequence ended with error");
    }

    resetExecutingFlags();
  }

  void resetExecutingFlags()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    executing_pick_  = false;
    executing_place_ = false;
    sequence_running_.store(false);
  }

  // ===============================================================
  // Pick / Place sync wrappers (mode 1 전용)
  // async_send_goal → goal_handle_future → async_get_result → result_future
  // 둘 다 timeout 적용. 비정상 응답(null result, 잘못된 code) 방어.
  // ===============================================================

  bool sendPickSync(const geometry_msgs::msg::Pose & pick_p)
  {
    if (!pick_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_logger(),
        "[goal_relay][mode1] Pick action server not available (5s timeout)");
      return false;
    }

    Pick::Goal goal{};
    goal.pick_pose = pick_p;

    auto opts = rclcpp_action::Client<Pick>::SendGoalOptions{};
    opts.feedback_callback =
      [this](PickGoalHandle::SharedPtr,
             const std::shared_ptr<const Pick::Feedback> fb) {
        RCLCPP_INFO(get_logger(),
          "[goal_relay][mode1][pick][%5.1f%%] %s",
          fb->progress * 100.0f, fb->status.c_str());
      };

    auto goal_handle_future = pick_client_->async_send_goal(goal, opts);

    // ── 1) 서버의 goal accept/reject 응답 대기 ─────────────────
    if (goal_handle_future.wait_for(std::chrono::seconds(10)) !=
        std::future_status::ready)
    {
      RCLCPP_ERROR(get_logger(),
        "[goal_relay][mode1] Pick goal_response timeout (10s)");
      return false;
    }
    auto goal_handle = goal_handle_future.get();
    if (!goal_handle) {
      RCLCPP_ERROR(get_logger(),
        "[goal_relay][mode1] Pick goal REJECTED by server");
      return false;
    }
    RCLCPP_INFO(get_logger(), "[goal_relay][mode1] Pick goal accepted");

    // ── 2) action 결과 대기 ───────────────────────────────────
    auto result_future = pick_client_->async_get_result(goal_handle);
    const double timeout_sec =
      get_parameter("action_result_timeout_sec").as_double();
    auto status = (timeout_sec > 0.0)
      ? result_future.wait_for(std::chrono::duration<double>(timeout_sec))
      : std::future_status::ready;

    if (timeout_sec > 0.0 && status != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(),
        "[goal_relay][mode1] Pick result timeout (%.0fs) — sequence aborted",
        timeout_sec);
      return false;
    }

    auto wrapped = result_future.get();
    if (wrapped.code != rclcpp_action::ResultCode::SUCCEEDED) {
      RCLCPP_ERROR(get_logger(),
        "[goal_relay][mode1] Pick action ended with code=%d",
        static_cast<int>(wrapped.code));
      return false;
    }
    if (!wrapped.result) {
      RCLCPP_ERROR(get_logger(),
        "[goal_relay][mode1] Pick result is null — treating as failure");
      return false;
    }
    if (!wrapped.result->success) {
      RCLCPP_ERROR(get_logger(),
        "[goal_relay][mode1] Pick result.success=false  msg='%s'",
        wrapped.result->message.c_str());
      return false;
    }
    return true;
  }

  bool sendPlaceSync(const geometry_msgs::msg::Pose & place_p)
  {
    if (!place_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_logger(),
        "[goal_relay][mode1] Place action server not available (5s timeout)");
      return false;
    }

    Place::Goal goal{};
    goal.place_pose = place_p;

    auto opts = rclcpp_action::Client<Place>::SendGoalOptions{};
    opts.feedback_callback =
      [this](PlaceGoalHandle::SharedPtr,
             const std::shared_ptr<const Place::Feedback> fb) {
        RCLCPP_INFO(get_logger(),
          "[goal_relay][mode1][place][%5.1f%%] %s",
          fb->progress * 100.0f, fb->status.c_str());
      };

    auto goal_handle_future = place_client_->async_send_goal(goal, opts);

    if (goal_handle_future.wait_for(std::chrono::seconds(10)) !=
        std::future_status::ready)
    {
      RCLCPP_ERROR(get_logger(),
        "[goal_relay][mode1] Place goal_response timeout (10s)");
      return false;
    }
    auto goal_handle = goal_handle_future.get();
    if (!goal_handle) {
      RCLCPP_ERROR(get_logger(),
        "[goal_relay][mode1] Place goal REJECTED by server");
      return false;
    }
    RCLCPP_INFO(get_logger(), "[goal_relay][mode1] Place goal accepted");

    auto result_future = place_client_->async_get_result(goal_handle);
    const double timeout_sec =
      get_parameter("action_result_timeout_sec").as_double();
    auto status = (timeout_sec > 0.0)
      ? result_future.wait_for(std::chrono::duration<double>(timeout_sec))
      : std::future_status::ready;

    if (timeout_sec > 0.0 && status != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(),
        "[goal_relay][mode1] Place result timeout (%.0fs)", timeout_sec);
      return false;
    }

    auto wrapped = result_future.get();
    if (wrapped.code != rclcpp_action::ResultCode::SUCCEEDED) {
      RCLCPP_ERROR(get_logger(),
        "[goal_relay][mode1] Place action ended with code=%d",
        static_cast<int>(wrapped.code));
      return false;
    }
    if (!wrapped.result) {
      RCLCPP_ERROR(get_logger(),
        "[goal_relay][mode1] Place result is null — treating as failure");
      return false;
    }
    if (!wrapped.result->success) {
      RCLCPP_ERROR(get_logger(),
        "[goal_relay][mode1] Place result.success=false  msg='%s'",
        wrapped.result->message.c_str());
      return false;
    }
    return true;
  }

  // ===============================================================
  // Mode 0: pick / place async (각자 독립 실행)
  // mutex_ 보유 상태에서 호출
  // ===============================================================

  void sendPickAsync()
  {
    if (!pick_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_logger(),
        "[goal_relay][mode0] Pick action server not available");
      return;
    }
    Pick::Goal goal{};
    goal.pick_pose  = pick_pose_;
    pick_received_  = false;
    executing_pick_ = true;

    auto opts = rclcpp_action::Client<Pick>::SendGoalOptions{};
    opts.goal_response_callback =
      [this](const PickGoalHandle::SharedPtr & gh) {
        if (!gh) {
          RCLCPP_ERROR(get_logger(),
            "[goal_relay][mode0][pick] Goal rejected by server");
          std::lock_guard<std::mutex> lock(mutex_);
          executing_pick_ = false;
        } else {
          RCLCPP_INFO(get_logger(),
            "[goal_relay][mode0][pick] Goal accepted");
        }
      };
    opts.feedback_callback =
      [this](PickGoalHandle::SharedPtr,
             const std::shared_ptr<const Pick::Feedback> fb) {
        RCLCPP_INFO(get_logger(),
          "[goal_relay][mode0][pick][%5.1f%%] %s",
          fb->progress * 100.0f, fb->status.c_str());
      };
    opts.result_callback =
      [this](const PickGoalHandle::WrappedResult & res) {
        const bool ok =
          (res.code == rclcpp_action::ResultCode::SUCCEEDED) &&
          res.result && res.result->success;
        if (ok) {
          RCLCPP_INFO(get_logger(),
            "[goal_relay][mode0][pick] SUCCESS: %s",
            res.result->message.c_str());
        } else {
          RCLCPP_ERROR(get_logger(),
            "[goal_relay][mode0][pick] FAILED  code=%d  msg='%s'",
            static_cast<int>(res.code),
            res.result ? res.result->message.c_str() : "(null)");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        executing_pick_ = false;
      };

    pick_client_->async_send_goal(goal, opts);
    RCLCPP_INFO(get_logger(), "[goal_relay][mode0] Pick goal sent");
  }

  void sendPlaceAsync()
  {
    if (!place_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_logger(),
        "[goal_relay][mode0] Place action server not available");
      return;
    }
    Place::Goal goal{};
    goal.place_pose  = place_pose_;
    place_received_  = false;
    executing_place_ = true;

    auto opts = rclcpp_action::Client<Place>::SendGoalOptions{};
    opts.goal_response_callback =
      [this](const PlaceGoalHandle::SharedPtr & gh) {
        if (!gh) {
          RCLCPP_ERROR(get_logger(),
            "[goal_relay][mode0][place] Goal rejected by server");
          std::lock_guard<std::mutex> lock(mutex_);
          executing_place_ = false;
        } else {
          RCLCPP_INFO(get_logger(),
            "[goal_relay][mode0][place] Goal accepted");
        }
      };
    opts.feedback_callback =
      [this](PlaceGoalHandle::SharedPtr,
             const std::shared_ptr<const Place::Feedback> fb) {
        RCLCPP_INFO(get_logger(),
          "[goal_relay][mode0][place][%5.1f%%] %s",
          fb->progress * 100.0f, fb->status.c_str());
      };
    opts.result_callback =
      [this](const PlaceGoalHandle::WrappedResult & res) {
        const bool ok =
          (res.code == rclcpp_action::ResultCode::SUCCEEDED) &&
          res.result && res.result->success;
        if (ok) {
          RCLCPP_INFO(get_logger(),
            "[goal_relay][mode0][place] SUCCESS: %s",
            res.result->message.c_str());
        } else {
          RCLCPP_ERROR(get_logger(),
            "[goal_relay][mode0][place] FAILED  code=%d  msg='%s'",
            static_cast<int>(res.code),
            res.result ? res.result->message.c_str() : "(null)");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        executing_place_ = false;
      };

    place_client_->async_send_goal(goal, opts);
    RCLCPP_INFO(get_logger(), "[goal_relay][mode0] Place goal sent");
  }

  // ---- subscriptions ----
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pick_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr place_sub_;

  // ---- action clients ----
  rclcpp::CallbackGroup::SharedPtr        client_cbg_;
  rclcpp_action::Client<Pick>::SharedPtr  pick_client_;
  rclcpp_action::Client<Place>::SharedPtr place_client_;

  // ---- state (mutex_ 로 보호) ----
  geometry_msgs::msg::Pose pick_pose_{};
  geometry_msgs::msg::Pose place_pose_{};
  bool pick_received_{false};
  bool place_received_{false};
  bool executing_pick_{false};
  bool executing_place_{false};
  std::mutex mutex_;

  // sequence guard (mode 1) — atomic 으로 단순 비교만 수행
  std::atomic<bool> sequence_running_{false};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  // MultiThreadedExecutor 사용 이유:
  //   별도 스레드의 future.get() 이 블로킹하는 동안 메인 executor 가
  //   action protocol 메시지(goal_response, feedback, result)를
  //   동시에 처리해야 한다. ReentrantCallbackGroup 과 결합하여
  //   subscription / client callback 의 직렬화 문제 제거.
  auto node = std::make_shared<GoalRelayNode>();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
