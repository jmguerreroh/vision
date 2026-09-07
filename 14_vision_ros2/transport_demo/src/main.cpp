/**
 * @file main.cpp
 * @brief Entry point of the transport_demo node
 * @author José Miguel Guerrero Hernández
 *
 * Subscribes to /color/image and publishes image_processed, plus the
 * sub-topics that each transport plugin adds on its own.
 *
 * image_transport is declared after construction, in initialize(), because it
 * needs a shared_ptr to the node and that does not exist yet inside the
 * constructor.
 */

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
