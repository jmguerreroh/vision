/**
 * @file opencv_processing.cpp
 * @brief The cv_bridge round trip: ROS image message to cv::Mat and back
 * @author José Miguel Guerrero Hernández
 *
 * Subscribes to /color/image and publishes /image_processed.
 *
 * The point of the example is the pair toCvCopy() / toImageMsg(): everything
 * between them is ordinary OpenCV, the same code as in the earlier chapters.
 * The original header is carried over so the result keeps the timestamp and
 * frame of the image it came from, which is what lets it be related to any
 * other message of the system.
 */

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
    // Convert to cv::Mat (own copy)
    cv::Mat frame = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;

    // Process with OpenCV: convert to grayscale, at full resolution, so the
    // result still matches the CameraInfo of the camera it came from
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Convert back to ROS image message, keeping the original header, and publish.
    // To look at the result: ros2 run rqt_image_view rqt_image_view /image_processed
    std_msgs::msg::Header header = msg->header;
    sensor_msgs::msg::Image::SharedPtr output_msg =
      cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO8, gray).toImageMsg();

    publisher_->publish(*output_msg);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
  }
}

}  // namespace opencv_demo
