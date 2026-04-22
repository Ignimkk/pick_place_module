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
 *         place_goal 수신 시 즉시 Place action 실행
 *         (둘이 독립적으로 실행 — 단, 암은 exec_mutex_ 로 직렬화됨)
 *
 *   1 → pick_goal + place_goal 모두 수신 후 실행
 *         Pick action 완료 → Place action 순차 실행
 *         (별도 스레드에서 promise/future 패턴으로 동기화)
 *
 * 설계 노트:
 *   - mode 0: 각 토픽 수신 즉시 해당 action client 로 goal 전송 (비동기)
 *   - mode 1: 두 토픽 모두 수신 → 전용 스레드에서 pick 동기 완료 후 place 전송
 *             (전용 스레드 사용 이유: action result callback 내부에서
 *              wait_for_action_server 를 호출하면 executor 재진입 문제 발생 가능)
 *   - mutex_ 로 모든 공유 상태 보호
 *   - executing_pick_ / executing_place_ : 중복 실행 방지 플래그
 */

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <pick_place_module/action/pick.hpp>
#include <pick_place_module/action/place.hpp>

#include <future>
#include <mutex>
#include <thread>

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

    pick_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "pick_goal", 10,
      [this](geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        pick_pose_     = msg->pose;
        pick_received_ = true;
        RCLCPP_INFO(get_logger(),
          "pick_goal received  pos=(%.3f, %.3f, %.3f)",
          msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
        onPickReceived();
      });

    place_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "place_goal", 10,
      [this](geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        place_pose_     = msg->pose;
        place_received_ = true;
        RCLCPP_INFO(get_logger(),
          "place_goal received  pos=(%.3f, %.3f, %.3f)",
          msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
        onPlaceReceived();
      });

    pick_client_  = rclcpp_action::create_client<Pick>(this, "pick");
    place_client_ = rclcpp_action::create_client<Place>(this, "place");

    RCLCPP_INFO(get_logger(),
      "GoalRelayNode started  trigger_mode=%d",
      get_parameter("trigger_mode").as_int());
  }

private:
  // ---------------------------------------------------------------
  // Mode 0: pick_goal 수신 즉시 Pick action 전송
  // Mode 1: 두 토픽 모두 준비되면 tryTriggerBoth 로 위임
  // mutex_ 보유 상태에서 호출
  // ---------------------------------------------------------------
  void onPickReceived()
  {
    if (get_parameter("trigger_mode").as_int() == 0) {
      if (executing_pick_) {
        RCLCPP_WARN(get_logger(),
          "[mode0] Pick already executing — new pick_goal ignored");
        return;
      }
      sendPickAsync();
    } else {
      tryTriggerBoth();
    }
  }

  // ---------------------------------------------------------------
  // Mode 0: place_goal 수신 즉시 Place action 전송
  // Mode 1: tryTriggerBoth 로 위임
  // mutex_ 보유 상태에서 호출
  // ---------------------------------------------------------------
  void onPlaceReceived()
  {
    if (get_parameter("trigger_mode").as_int() == 0) {
      if (executing_place_) {
        RCLCPP_WARN(get_logger(),
          "[mode0] Place already executing — new place_goal ignored");
        return;
      }
      sendPlaceAsync();
    } else {
      tryTriggerBoth();
    }
  }

  // ---------------------------------------------------------------
  // Mode 1 전용: 두 토픽이 모두 준비된 경우 순차 실행 스레드 기동
  // mutex_ 보유 상태에서 호출
  // ---------------------------------------------------------------
  void tryTriggerBoth()
  {
    if (!pick_received_ || !place_received_) {
      RCLCPP_INFO(get_logger(),
        "[mode1] Waiting for both goals  pick=%s  place=%s",
        pick_received_  ? "OK" : "waiting",
        place_received_ ? "OK" : "waiting");
      return;
    }
    if (executing_pick_ || executing_place_) {
      RCLCPP_WARN(get_logger(),
        "[mode1] Already executing — new goals ignored until completion");
      return;
    }

    // 두 pose 복사 후 플래그 리셋 (중복 트리거 방지)
    geometry_msgs::msg::Pose pick_p  = pick_pose_;
    geometry_msgs::msg::Pose place_p = place_pose_;
    pick_received_   = false;
    place_received_  = false;
    executing_pick_  = true;
    executing_place_ = true;

    // 별도 스레드에서 pick → place 순차 실행
    // 이유: action result callback 내에서 wait_for_action_server + future.get()
    //       을 호출하면 executor 재진입 문제가 발생할 수 있으므로,
    //       독립된 스레드에서 동기 방식(promise/future)으로 처리
    std::thread([this, pick_p, place_p]() {
      sequentialPickThenPlace(pick_p, place_p);
    }).detach();

    RCLCPP_INFO(get_logger(), "[mode1] Sequential pick→place thread spawned");
  }

  // ---------------------------------------------------------------
  // Mode 0: Pick action 비동기 전송
  // mutex_ 보유 상태에서 호출
  // ---------------------------------------------------------------
  void sendPickAsync()
  {
    if (!pick_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_logger(), "[mode0] Pick action server not available");
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
          RCLCPP_ERROR(get_logger(), "[mode0][pick] Goal rejected by server");
          std::lock_guard<std::mutex> lock(mutex_);
          executing_pick_ = false;
        } else {
          RCLCPP_INFO(get_logger(), "[mode0][pick] Goal accepted");
        }
      };

    opts.feedback_callback =
      [this](PickGoalHandle::SharedPtr,
             const std::shared_ptr<const Pick::Feedback> fb) {
        RCLCPP_INFO(get_logger(),
          "[mode0][pick][%5.1f%%] %s",
          fb->progress * 100.0f, fb->status.c_str());
      };

    opts.result_callback =
      [this](const PickGoalHandle::WrappedResult & res) {
        if (res.result->success) {
          RCLCPP_INFO(get_logger(),
            "[mode0][pick] SUCCESS: %s", res.result->message.c_str());
        } else {
          RCLCPP_ERROR(get_logger(),
            "[mode0][pick] FAILED : %s", res.result->message.c_str());
        }
        std::lock_guard<std::mutex> lock(mutex_);
        executing_pick_ = false;
      };

    pick_client_->async_send_goal(goal, opts);
    RCLCPP_INFO(get_logger(), "[mode0] Pick goal sent");
  }

  // ---------------------------------------------------------------
  // Mode 0: Place action 비동기 전송
  // mutex_ 보유 상태에서 호출
  // ---------------------------------------------------------------
  void sendPlaceAsync()
  {
    if (!place_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_logger(), "[mode0] Place action server not available");
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
          RCLCPP_ERROR(get_logger(), "[mode0][place] Goal rejected by server");
          std::lock_guard<std::mutex> lock(mutex_);
          executing_place_ = false;
        } else {
          RCLCPP_INFO(get_logger(), "[mode0][place] Goal accepted");
        }
      };

    opts.feedback_callback =
      [this](PlaceGoalHandle::SharedPtr,
             const std::shared_ptr<const Place::Feedback> fb) {
        RCLCPP_INFO(get_logger(),
          "[mode0][place][%5.1f%%] %s",
          fb->progress * 100.0f, fb->status.c_str());
      };

    opts.result_callback =
      [this](const PlaceGoalHandle::WrappedResult & res) {
        if (res.result->success) {
          RCLCPP_INFO(get_logger(),
            "[mode0][place] SUCCESS: %s", res.result->message.c_str());
        } else {
          RCLCPP_ERROR(get_logger(),
            "[mode0][place] FAILED : %s", res.result->message.c_str());
        }
        std::lock_guard<std::mutex> lock(mutex_);
        executing_place_ = false;
      };

    place_client_->async_send_goal(goal, opts);
    RCLCPP_INFO(get_logger(), "[mode0] Place goal sent");
  }

  // ---------------------------------------------------------------
  // Mode 1 전용: 별도 스레드에서 pick → place 순차 실행
  // promise/future 로 각 action 완료를 동기 대기
  // (스레드가 future.get() 으로 블로킹하는 동안 메인 executor 는
  //  action result callback 을 처리하여 promise.set_value() 를 호출)
  // ---------------------------------------------------------------
  void sequentialPickThenPlace(
    const geometry_msgs::msg::Pose & pick_p,
    const geometry_msgs::msg::Pose & place_p)
  {
    // --- Pick 동기 실행 ---
    const bool pick_ok = sendPickSync(pick_p);

    if (!pick_ok) {
      RCLCPP_ERROR(get_logger(),
        "[mode1] Pick FAILED — aborting place");
      std::lock_guard<std::mutex> lock(mutex_);
      executing_pick_  = false;
      executing_place_ = false;
      return;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      executing_pick_ = false;
    }
    RCLCPP_INFO(get_logger(), "[mode1] Pick succeeded — starting place");

    // --- Place 동기 실행 ---
    const bool place_ok = sendPlaceSync(place_p);

    if (place_ok) {
      RCLCPP_INFO(get_logger(), "[mode1] Place succeeded — full PnP complete");
    } else {
      RCLCPP_ERROR(get_logger(), "[mode1] Place FAILED");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    executing_place_ = false;
  }

  // ---------------------------------------------------------------
  // Pick action 동기 전송 및 결과 대기 (promise/future 패턴)
  // 반환값: true = 성공, false = 실패 또는 서버 미연결
  // ---------------------------------------------------------------
  bool sendPickSync(const geometry_msgs::msg::Pose & pick_p)
  {
    if (!pick_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_logger(), "[mode1][pick] Action server not available");
      return false;
    }

    auto promise = std::make_shared<std::promise<bool>>();
    auto future  = promise->get_future();

    Pick::Goal goal{};
    goal.pick_pose = pick_p;

    auto opts = rclcpp_action::Client<Pick>::SendGoalOptions{};

    opts.goal_response_callback =
      [promise](const PickGoalHandle::SharedPtr & gh) {
        if (!gh) {
          promise->set_value(false);
        }
      };

    opts.feedback_callback =
      [this](PickGoalHandle::SharedPtr,
             const std::shared_ptr<const Pick::Feedback> fb) {
        RCLCPP_INFO(get_logger(),
          "[mode1][pick][%5.1f%%] %s",
          fb->progress * 100.0f, fb->status.c_str());
      };

    opts.result_callback =
      [promise](const PickGoalHandle::WrappedResult & res) {
        promise->set_value(res.result->success);
      };

    pick_client_->async_send_goal(goal, opts);
    RCLCPP_INFO(get_logger(), "[mode1] Pick goal sent, waiting...");

    return future.get();
  }

  // ---------------------------------------------------------------
  // Place action 동기 전송 및 결과 대기 (promise/future 패턴)
  // ---------------------------------------------------------------
  bool sendPlaceSync(const geometry_msgs::msg::Pose & place_p)
  {
    if (!place_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_logger(), "[mode1][place] Action server not available");
      return false;
    }

    auto promise = std::make_shared<std::promise<bool>>();
    auto future  = promise->get_future();

    Place::Goal goal{};
    goal.place_pose = place_p;

    auto opts = rclcpp_action::Client<Place>::SendGoalOptions{};

    opts.goal_response_callback =
      [promise](const PlaceGoalHandle::SharedPtr & gh) {
        if (!gh) {
          promise->set_value(false);
        }
      };

    opts.feedback_callback =
      [this](PlaceGoalHandle::SharedPtr,
             const std::shared_ptr<const Place::Feedback> fb) {
        RCLCPP_INFO(get_logger(),
          "[mode1][place][%5.1f%%] %s",
          fb->progress * 100.0f, fb->status.c_str());
      };

    opts.result_callback =
      [promise](const PlaceGoalHandle::WrappedResult & res) {
        promise->set_value(res.result->success);
      };

    place_client_->async_send_goal(goal, opts);
    RCLCPP_INFO(get_logger(), "[mode1] Place goal sent, waiting...");

    return future.get();
  }

  // ---- subscriptions ----
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pick_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr place_sub_;

  // ---- action clients ----
  rclcpp_action::Client<Pick>::SharedPtr  pick_client_;
  rclcpp_action::Client<Place>::SharedPtr place_client_;

  // ---- shared state (protected by mutex_) ----
  geometry_msgs::msg::Pose pick_pose_{};
  geometry_msgs::msg::Pose place_pose_{};
  bool pick_received_{false};
  bool place_received_{false};
  bool executing_pick_{false};
  bool executing_place_{false};

  std::mutex mutex_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  // MultiThreadedExecutor 사용 이유:
  //   mode 1 에서 별도 스레드의 future.get() 이 블로킹하는 동안
  //   메인 executor 가 action result callback 을 처리해야 하므로
  //   SingleThreadedExecutor 로는 deadlock 이 발생함.
  auto node = std::make_shared<GoalRelayNode>();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
