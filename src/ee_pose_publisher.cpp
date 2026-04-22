/**
 * ee_pose_publisher.cpp
 *
 * 역할:
 *   TF2 를 통해 현재 엔드이펙터(기본: tool0) 의 pose 를 주기적으로
 *   콘솔에 출력하고 /ee_pose 토픽으로 발행.
 *
 * 용도:
 *   - 현재 robot 위치에서의 xyz, quaternion 확인
 *   - pick_goal / place_goal 에 입력할 실제 좌표 조사
 *   - grasp_orientation 파라미터 튜닝용 참조값 획득
 *
 * 파라미터:
 *   base_frame   : 기준 frame (기본 "base_link")
 *   ee_frame     : 엔드이펙터 frame (기본 "tool0")
 *                  그리퍼 팁이 필요하면 "robotiq_85_base_link" 등으로 변경
 *   publish_rate : 발행 주파수 [Hz] (기본 2.0)
 *
 * 발행 토픽:
 *   /ee_pose  (geometry_msgs/PoseStamped)
 *
 * 실행:
 *   ros2 run pick_place_module ee_pose_publisher
 *   ros2 run pick_place_module ee_pose_publisher --ros-args \
 *     -p ee_frame:=robotiq_85_base_link -p publish_rate:=1.0
 */

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

class EePosePublisher : public rclcpp::Node
{
public:
  EePosePublisher()
  : Node("ee_pose_publisher")
  {
    declare_parameter<std::string>("base_frame",   "base_link");
    declare_parameter<std::string>("ee_frame",     "tool0");
    declare_parameter<double>     ("publish_rate", 2.0);

    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(
      *tf_buffer_, this, false);

    pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("ee_pose", 10);

    const double rate = get_parameter("publish_rate").as_double();
    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / rate),
      [this]() { timerCallback(); });

    RCLCPP_INFO(get_logger(), "=== EePosePublisher started ===");
    RCLCPP_INFO(get_logger(), "  base_frame  : %s",
      get_parameter("base_frame").as_string().c_str());
    RCLCPP_INFO(get_logger(), "  ee_frame    : %s",
      get_parameter("ee_frame").as_string().c_str());
    RCLCPP_INFO(get_logger(), "  publish_rate: %.1f Hz  -> /ee_pose",
      rate);
    RCLCPP_INFO(get_logger(), "================================");
  }

private:
  void timerCallback()
  {
    const std::string base = get_parameter("base_frame").as_string();
    const std::string ee   = get_parameter("ee_frame").as_string();

    geometry_msgs::msg::TransformStamped tf_stamped;
    try {
      // tf2::TimePointZero → 가장 최근 transform 사용
      tf_stamped = tf_buffer_->lookupTransform(
        base, ee, tf2::TimePointZero);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
        "TF lookup [%s -> %s] failed: %s", base.c_str(), ee.c_str(), ex.what());
      return;
    }

    // PoseStamped 구성
    geometry_msgs::msg::PoseStamped msg;
    msg.header              = tf_stamped.header;
    msg.pose.position.x     = tf_stamped.transform.translation.x;
    msg.pose.position.y     = tf_stamped.transform.translation.y;
    msg.pose.position.z     = tf_stamped.transform.translation.z;
    msg.pose.orientation    = tf_stamped.transform.rotation;

    pub_->publish(msg);

    // 콘솔 출력 (pick_goal 입력값으로 바로 복사 가능한 형식)
    RCLCPP_INFO(get_logger(),
      "\n"
      "  [%s -> %s]\n"
      "  position   : x=%.5f  y=%.5f  z=%.5f\n"
      "  orientation: x=%.5f  y=%.5f  z=%.5f  w=%.5f",
      base.c_str(), ee.c_str(),
      msg.pose.position.x,
      msg.pose.position.y,
      msg.pose.position.z,
      msg.pose.orientation.x,
      msg.pose.orientation.y,
      msg.pose.orientation.z,
      msg.pose.orientation.w);
  }

  std::shared_ptr<tf2_ros::Buffer>            tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr                timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EePosePublisher>());
  rclcpp::shutdown();
  return 0;
}
