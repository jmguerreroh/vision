#include <rclcpp/rclcpp.hpp>
#include "pcl_demo/pcl_processing.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<pcl_demo::PCLProcessing>());
  rclcpp::shutdown();
  return 0;
}
