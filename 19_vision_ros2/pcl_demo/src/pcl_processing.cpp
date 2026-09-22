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
  pcl::PointCloud<pcl::PointXYZRGB> pointcloud;
  pcl::fromROSMsg(*msg, pointcloud);

  // Process the point cloud (e.g., filtering, segmentation, etc.)
  pcl::PointCloud<pcl::PointXYZRGB> filtered_cloud;
  for (const auto & point : pointcloud.points) {
    if (point.z < 0.8) {  // Example condition: filter points with z < 0.8
      filtered_cloud.push_back(point);
    }
  }

  sensor_msgs::msg::PointCloud2 output;
  pcl::toROSMsg(filtered_cloud, output);
  output.header = msg->header;  // Preserve the original header
  publisher_->publish(output);
}

}  // namespace pcl_demo
