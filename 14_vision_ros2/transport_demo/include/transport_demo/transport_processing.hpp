/**
 * @file transport_processing.hpp
 * @brief Declaration of TransportProcessing
 * @author José Miguel Guerrero Hernández
 *
 * Subscribes to /color/image and publishes image_processed, plus the
 * sub-topics that each transport plugin adds on its own.
 *
 * image_transport is declared after construction, in initialize(), because it
 * needs a shared_ptr to the node and that does not exist yet inside the
 * constructor.
 */

#ifndef TRANSPORT_DEMO__TRANSPORT_PROCESSING_HPP_
#define TRANSPORT_DEMO__TRANSPORT_PROCESSING_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

namespace transport_demo
{

class TransportProcessing : public rclcpp::Node
{
public:
  TransportProcessing();
  void initialize();

private:
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);

  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
};

}  // namespace transport_demo

#endif  // TRANSPORT_DEMO__TRANSPORT_PROCESSING_HPP_
