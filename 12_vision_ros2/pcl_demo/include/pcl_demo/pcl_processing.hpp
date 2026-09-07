#ifndef PCL_DEMO__PCL_PROCESSING_HPP_
#define PCL_DEMO__PCL_PROCESSING_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

namespace pcl_demo
{

class PCLProcessing : public rclcpp::Node
{
public:
  PCLProcessing();

private:
  void pcl_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
};

}  // namespace pcl_demo

#endif  // PCL_DEMO__PCL_PROCESSING_HPP_
