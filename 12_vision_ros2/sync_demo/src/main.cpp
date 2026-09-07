#include <rclcpp/rclcpp.hpp>
#include "sync_demo/sync_processing.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<sync_demo::SyncProcessing>());
  rclcpp::shutdown();
  return 0;
}
