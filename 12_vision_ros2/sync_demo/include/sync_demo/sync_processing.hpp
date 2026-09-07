#ifndef OPENCV_DEMO__STEREO_SYNC_HPP_
#define OPENCV_DEMO__STEREO_SYNC_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
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

#endif  // OPENCV_DEMO__STEREO_SYNC_HPP_
