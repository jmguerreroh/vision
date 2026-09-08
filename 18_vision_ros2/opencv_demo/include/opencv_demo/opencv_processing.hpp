/**
 * @file opencv_processing.hpp
 * @brief Declaration of OpenCVProcessing
 * @author José Miguel Guerrero Hernández
 *
 * Subscribes to /color/image and publishes /image_processed.
 *
 * The point of the example is the pair toCvCopy() / toImageMsg(): everything
 * between them is ordinary OpenCV, the same code as in the earlier chapters.
 * The original header is carried over so the result keeps the timestamp and
 * frame of the image it came from, which is what lets it be related to any
 * other message of the system.
 */

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
