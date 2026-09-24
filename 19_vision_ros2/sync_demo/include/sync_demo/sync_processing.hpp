/**
 * @file sync_processing.hpp
 * @brief Declaration of SyncProcessing
 * @author José Miguel Guerrero Hernández
 *
 * Subscribes to /left/image and /right/image and displays the pair.
 *
 * Two cameras without a hardware trigger never stamp exactly the same time,
 * so ExactTime would leave the callback silent. ApproximateTime pairs the
 * closest messages within a tolerance, which is what makes a stereo pair
 * usable at all.
 */

#ifndef SYNC_DEMO__SYNC_PROCESSING_HPP_
#define SYNC_DEMO__SYNC_PROCESSING_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <message_filters/subscriber.hpp>
#include <message_filters/synchronizer.hpp>
#include <message_filters/sync_policies/approximate_time.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

namespace sync_demo
{

class SyncProcessing : public rclcpp::Node
{
public:
  SyncProcessing();

private:
  void image_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr & left,
    const sensor_msgs::msg::Image::ConstSharedPtr & right);

  // Message filter subscribers
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> left_sub_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> right_sub_;

  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::Image>;

  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
};

}  // namespace sync_demo

#endif  // SYNC_DEMO__SYNC_PROCESSING_HPP_
