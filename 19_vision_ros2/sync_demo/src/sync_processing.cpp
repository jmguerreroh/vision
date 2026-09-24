/**
 * @file sync_processing.cpp
 * @brief message_filters with ApproximateTime: one callback, two already paired images
 * @author José Miguel Guerrero Hernández
 *
 * Subscribes to /left/image and /right/image and displays the pair.
 *
 * Two cameras without a hardware trigger never stamp exactly the same time,
 * so ExactTime would leave the callback silent. ApproximateTime pairs the
 * closest messages within a tolerance, which is what makes a stereo pair
 * usable at all.
 */

#include "sync_demo/sync_processing.hpp"

#include <rclcpp/version.h>

namespace sync_demo
{

SyncProcessing::SyncProcessing()
: Node("sync_processing")
{
  // Sensor-data QoS (best effort), like the other nodes of the chapter. The
  // message_filters subscriber takes it as an rclcpp::QoS from Kilted on
  // (rclcpp 29), and as the older rmw_qos_profile_t up to Jazzy (rclcpp 28):
  // neither form compiles on the other side.
#if RCLCPP_VERSION_GTE(29, 0, 0)
  const rclcpp::QoS qos = rclcpp::SensorDataQoS();
#else
  const rmw_qos_profile_t qos = rmw_qos_profile_sensor_data;
#endif
  left_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this,
      "/left/image", qos);
  right_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this,
      "/right/image", qos);

  sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), *left_sub_,
      *right_sub_);
  sync_->registerCallback(std::bind(&SyncProcessing::image_callback, this, std::placeholders::_1,
      std::placeholders::_2));

  RCLCPP_INFO(this->get_logger(), "Synchronizer node initialized");
}

void SyncProcessing::image_callback(
  const sensor_msgs::msg::Image::ConstSharedPtr & left,
  const sensor_msgs::msg::Image::ConstSharedPtr & right)
{
  try {
    auto left_cv = cv_bridge::toCvShare(left, sensor_msgs::image_encodings::BGR8);
    auto right_cv = cv_bridge::toCvShare(right, sensor_msgs::image_encodings::BGR8);

    cv::imshow("Left Image", left_cv->image);
    cv::imshow("Right Image", right_cv->image);
    cv::waitKey(1);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
  }
}

}  // namespace sync_demo
