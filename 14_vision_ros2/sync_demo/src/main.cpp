/**
 * @file main.cpp
 * @brief Entry point of the sync_demo node
 * @author José Miguel Guerrero Hernández
 *
 * Subscribes to /left/image and /right/image and displays the pair.
 *
 * Two cameras without a hardware trigger never stamp exactly the same time,
 * so ExactTime would leave the callback silent. ApproximateTime pairs the
 * closest messages within a tolerance, which is what makes a stereo pair
 * usable at all.
 */

#include <rclcpp/rclcpp.hpp>
#include "sync_demo/sync_processing.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<sync_demo::SyncProcessing>());
  rclcpp::shutdown();
  return 0;
}
