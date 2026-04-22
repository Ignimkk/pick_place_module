/**
 * ee_path_visualizer_node.cpp
 *
 * 역할
 * ────────────────────────────────────────────────────────────────
 * EE(End-Effector) 실행 궤적을 nav_msgs/Path 로 RViz 에 publish 한다.
 *
 * ── /ee_path/actual (Method C: TF 폴링) ─────────────────────────
 *   /motion_logger/record true  → actual_path 초기화 + TF 폴링 시작
 *   /motion_logger/record false → TF 폴링 중단 + 최종 Path publish
 *   tf_poll_hz 주파수로 TF(base_frame → ee_link) 를 폴링하여 누적.
 *   RViz 에서 선이 실시간으로 자라는 효과가 있다.
 *
 * ── /ee_path/planned (Method B: FK) ─────────────────────────────
 *   pick_place_node 가 RRTConnect 계획 성공 직후 FK 를 계산하여
 *   직접 publish 한다. (이 노드에서는 처리하지 않음)
 *   이유: pick_place_node 가 MoveGroupInterface 를 통해 robot model 을
 *         이미 보유하고 있어 RobotModelLoader 를 별도로 띄울 필요가 없음.
 *
 * RViz 설정
 * ────────────────────────────────────────────────────────────────
 *   Add → By display type → Path
 *     Topic  : /ee_path/planned   Color: 0, 0, 255 (파랑)  Line Width: 0.005
 *   Add → By display type → Path
 *     Topic  : /ee_path/actual    Color: 255, 128, 0 (주황) Line Width: 0.005
 *
 * 파라미터
 * ────────────────────────────────────────────────────────────────
 *   ee_link    (string, default: "tool0")
 *   base_frame (string, default: "base_link")
 *   tf_poll_hz (double, default: 20.0)
 */

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/bool.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <memory>
#include <string>

class EePathVisualizerNode : public rclcpp::Node
{
public:
  EePathVisualizerNode()
  : Node("ee_path_visualizer_node")
  {
    // ── 파라미터 ──────────────────────────────────────────────
    declare_parameter<std::string>("ee_link",     "tool0");
    declare_parameter<std::string>("base_frame",  "base_link");
    declare_parameter<double>     ("tf_poll_hz",  20.0);

    ee_link_    = get_parameter("ee_link").as_string();
    base_frame_ = get_parameter("base_frame").as_string();

    // ── TF2 ──────────────────────────────────────────────────
    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // ── Publisher ─────────────────────────────────────────────
    // transient_local: RViz 가 나중에 subscribe 해도 마지막 Path 를 수신.
    actual_pub_ = create_publisher<nav_msgs::msg::Path>(
      "/ee_path/actual", rclcpp::QoS(1).transient_local());

    // ── record 트리거 subscriber ──────────────────────────────
    record_sub_ = create_subscription<std_msgs::msg::Bool>(
      "/motion_logger/record", 10,
      std::bind(&EePathVisualizerNode::onRecord, this, std::placeholders::_1));

    // ── TF 폴링 타이머 ────────────────────────────────────────
    const double hz     = get_parameter("tf_poll_hz").as_double();
    const auto   period = std::chrono::duration<double>(1.0 / hz);
    tf_timer_ = create_wall_timer(period,
      std::bind(&EePathVisualizerNode::onTfTimer, this));

    RCLCPP_INFO(get_logger(),
      "EePathVisualizerNode ready — ee: %s  frame: %s  %.0f Hz",
      ee_link_.c_str(), base_frame_.c_str(), hz);
    RCLCPP_INFO(get_logger(),
      "  /ee_path/planned  ← pick_place_node (FK, published on plan)");
    RCLCPP_INFO(get_logger(),
      "  /ee_path/actual   ← this node (TF polling during execution)");
  }

private:
  // ── record 트리거: 기록 시작 / 중단 ──────────────────────────
  void onRecord(const std_msgs::msg::Bool::SharedPtr msg)
  {
    if (msg->data) {
      actual_path_.poses.clear();
      actual_path_.header.frame_id = base_frame_;
      actual_path_.header.stamp    = this->now();
      recording_ = true;
      RCLCPP_INFO(get_logger(), "[actual] Recording started");
    } else {
      if (!recording_) { return; }
      recording_ = false;
      actual_pub_->publish(actual_path_);
      RCLCPP_INFO(get_logger(),
        "[actual] Recording stopped — %zu poses published",
        actual_path_.poses.size());
    }
  }

  // ── TF 폴링: actual path 점진적 누적 + publish ───────────────
  void onTfTimer()
  {
    if (!recording_) { return; }

    geometry_msgs::msg::TransformStamped tf_stamped;
    try {
      tf_stamped = tf_buffer_->lookupTransform(
        base_frame_, ee_link_, tf2::TimePointZero);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
        "[actual] TF %s→%s: %s",
        base_frame_.c_str(), ee_link_.c_str(), ex.what());
      return;
    }

    geometry_msgs::msg::PoseStamped ps;
    ps.header.frame_id  = base_frame_;
    ps.header.stamp     = tf_stamped.header.stamp;
    ps.pose.position.x  = tf_stamped.transform.translation.x;
    ps.pose.position.y  = tf_stamped.transform.translation.y;
    ps.pose.position.z  = tf_stamped.transform.translation.z;
    ps.pose.orientation = tf_stamped.transform.rotation;

    actual_path_.poses.push_back(ps);
    actual_path_.header.stamp = ps.header.stamp;

    // 점진적 publish: RViz 에서 선이 실시간으로 자라는 효과
    actual_pub_->publish(actual_path_);
  }

  // ── 멤버 변수 ─────────────────────────────────────────────────
  std::string ee_link_;
  std::string base_frame_;

  std::shared_ptr<tf2_ros::Buffer>            tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr    actual_pub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr record_sub_;
  rclcpp::TimerBase::SharedPtr                         tf_timer_;

  nav_msgs::msg::Path actual_path_;
  bool                recording_{false};
};

// ================================================================
// main
// ================================================================
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EePathVisualizerNode>());
  rclcpp::shutdown();
  return 0;
}
