#include <rclcpp/rclcpp.hpp>
#include "opencv_demo/opencv_processing.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<opencv_demo::OpenCVProcessing>());
  rclcpp::shutdown();
  return 0;
}
