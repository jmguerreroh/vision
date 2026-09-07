#include "sync_demo/sync_processing.hpp"

namespace sync_demo
{

SyncProcessing::SyncProcessing()
: Node("sync_processing")
{
  left_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this,
      "/left/image");
  right_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this,
      "/right/image");

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
