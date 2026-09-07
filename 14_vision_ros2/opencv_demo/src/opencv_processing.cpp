#include "opencv_demo/opencv_processing.hpp"

namespace opencv_demo
{

OpenCVProcessing::OpenCVProcessing()
: Node("image_processing")
{
  subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
    "/color/image", rclcpp::SensorDataQoS(),
    std::bind(&OpenCVProcessing::image_callback, this, std::placeholders::_1));

  publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
    "/image_processed", rclcpp::SensorDataQoS());

  RCLCPP_INFO(this->get_logger(), "Image processing node initialized");
}

void OpenCVProcessing::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  try {
    // Convert to cv::Mat
    cv::Mat frame = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;

    // Show received image
    cv::resize(frame, frame, cv::Size(frame.cols / 4, frame.rows / 4));
    cv::imshow("Received Image", frame);

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Show processed image
    cv::imshow("Grayscale Image", gray);
    cv::waitKey(1);

    // Convert back to ROS image message and publish
    std_msgs::msg::Header header = msg->header;
    sensor_msgs::msg::Image::SharedPtr output_msg =
      cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO8, gray).toImageMsg();

    publisher_->publish(*output_msg);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
  }
}

}  // namespace opencv_demo
