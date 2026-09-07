#include "transport_demo/transport_processing.hpp"

namespace transport_demo
{

TransportProcessing::TransportProcessing()
: Node("transport_processing")
{
  // Constructor
  RCLCPP_INFO(this->get_logger(), "Transport node initialized");
}


void TransportProcessing::initialize()
{
  // Name of the input topic and the transport type to use (e.g., "raw", "compressed", etc.)
  std::string topic_name = "/color/image";
  std::string transport_name = "raw";
  // Get a shared pointer to the current node instance
  auto node = this->shared_from_this();

  // Create an ImageTransport object for handling image subscriptions and publications
  image_transport::ImageTransport it(node);
  try {
    // Create a subscription to the image topic with the specified transport
    // - node: the ROS 2 node used to create the subscription
    // - topic_name: the name of the topic to subscribe to
    // - transport_name: the transport to use ("raw", "compressed", etc.)
    // - image_callback: the callback function triggered when a new image is received
    // - rmw_qos_profile_sensor_data: recommended QoS for sensor data (low latency)
    // - SubscriptionOptions: optional settings like intra-process communication or custom callbacks
    image_sub_ = image_transport::create_subscription(
      node.get(),
      topic_name,
      std::bind(&TransportProcessing::image_callback, this, std::placeholders::_1),
      transport_name,
      rmw_qos_profile_sensor_data,
      rclcpp::SubscriptionOptions());
    RCLCPP_INFO(
      node->get_logger(), "Image received: unwrapping using '%s' transport.",
      image_sub_.getTransport().c_str());
  } catch (image_transport::TransportLoadException & e) {
    RCLCPP_ERROR(
      node->get_logger(), "Failed to create subscriber for topic %s: %s",
      topic_name.c_str(), e.what());
    return;
  }

  // Advertise a publisher for processed images on the "image_processed" topic
  // The number 1 is the publisher queue size
  image_pub_ = it.advertise("image_processed", 1);
}

void TransportProcessing::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  try {
    // Convert to cv::Mat
    cv::Mat frame = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8)->image;

    // Show received image
    cv::resize(frame, frame, cv::Size(frame.cols / 4, frame.rows / 4));
    cv::imshow("Received Image", frame);

    // Process: convert to grayscale
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::imshow("Grayscale Image", gray);
    cv::waitKey(1);

    // Convert to ROS message and publish
    std_msgs::msg::Header header = msg->header;
    sensor_msgs::msg::Image::SharedPtr output_msg =
      cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO8, gray).toImageMsg();
    image_pub_.publish(*output_msg);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
  }
}

}  // namespace transport_demo
