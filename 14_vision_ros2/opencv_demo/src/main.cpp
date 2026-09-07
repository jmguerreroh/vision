/**
 * @file main.cpp
 * @brief Entry point of the opencv_demo node
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

#include <rclcpp/rclcpp.hpp>
#include "opencv_demo/opencv_processing.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<opencv_demo::OpenCVProcessing>());
  rclcpp::shutdown();
  return 0;
}
