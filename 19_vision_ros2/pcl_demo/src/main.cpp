/**
 * @file main.cpp
 * @brief Entry point of the pcl_demo node
 * @author José Miguel Guerrero Hernández
 *
 * Subscribes to /stereo/points and publishes /pcl_processed.
 *
 * The gap between the two conversions is where a PCL algorithm of Chapter 14
 * goes. Everything else is the ROS 2 plumbing around it.
 */

#include <rclcpp/rclcpp.hpp>
#include "pcl_demo/pcl_processing.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<pcl_demo::PCLProcessing>());
  rclcpp::shutdown();
  return 0;
}
