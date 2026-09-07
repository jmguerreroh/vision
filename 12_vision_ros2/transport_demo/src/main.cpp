#include <rclcpp/rclcpp.hpp>
#include "transport_demo/transport_processing.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<transport_demo::TransportProcessing>();
  node->initialize();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
