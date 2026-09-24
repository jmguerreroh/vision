/**
 * @file pcl_processing.cpp
 * @brief pcl_conversions: PointCloud2 to pcl::PointCloud and back
 * @author José Miguel Guerrero Hernández
 *
 * Subscribes to /stereo/points and publishes /pcl_processed.
 *
 * The gap between the two conversions is where a PCL algorithm of Chapter 15
 * goes. Everything else is the ROS 2 plumbing around it.
 */

#include "pcl_demo/pcl_processing.hpp"

namespace pcl_demo
{

PCLProcessing::PCLProcessing()
: Node("pcl_processing")
{
  subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/stereo/points", rclcpp::SensorDataQoS(),
    std::bind(&PCLProcessing::pcl_callback, this, std::placeholders::_1));

  publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
    "/pcl_processed", rclcpp::SensorDataQoS());

  RCLCPP_INFO(this->get_logger(), "PCL processing node initialized");
}

void PCLProcessing::pcl_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  // ROS -> PCL
  pcl::PointCloud<pcl::PointXYZRGB> pointcloud;
  pcl::fromROSMsg(*msg, pointcloud);

  // Process the cloud here (filtering, segmentation, ICP... Chapter 15).
  // In this example it is republished unchanged.

  // PCL -> ROS
  sensor_msgs::msg::PointCloud2 output;
  pcl::toROSMsg(pointcloud, output);
  output.header = msg->header;  // Preserve the original header
  publisher_->publish(output);
}

}  // namespace pcl_demo
