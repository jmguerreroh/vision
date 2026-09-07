#ifndef OPENCV_DEMO__OPENCV_PROCESSING_HPP_
#define OPENCV_DEMO__OPENCV_PROCESSING_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

namespace opencv_demo
{

class OpenCVProcessing : public rclcpp::Node
{
public:
  OpenCVProcessing();

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};

}  // namespace opencv_demo

#endif  // OPENCV_DEMO__OPENCV_PROCESSING_HPP_
