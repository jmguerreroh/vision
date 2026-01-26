/**
 * @file main.cpp
 * @brief OpenCV video processing example
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates basic video processing with OpenCV:
 * - Opening video from file or camera
 * - Reading and displaying frames
 * - Basic frame processing (HSV color space)
 * - Writing video to file
 *
 * @note Make sure to have a camera connected or provide a video file path as an argument
 */

#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char ** argv)
{
  // Open video source: camera (0) or file
  cv::VideoCapture cap;

  if (argc > 1) {
    // Open video file
    cap.open(argv[1]);
    std::cout << "Opening video file: " << argv[1] << std::endl;
  } else {
    // Open default camera
    cap.open(0);
    std::cout << "Opening camera..." << std::endl;
  }

  // Check if video source was opened successfully
  if (!cap.isOpened()) {
    std::cerr << "Error: Could not open video source" << std::endl;
    return -1;
  }

  // Get video properties
  int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = cap.get(cv::CAP_PROP_FPS);

  // If FPS is 0 (camera), set default
  if (fps == 0) {fps = 30.0;}

  std::cout << "Video properties:" << std::endl;
  std::cout << "  Resolution: " << frame_width << "x" << frame_height << std::endl;
  std::cout << "  FPS: " << fps << std::endl;

  // Create VideoWriter to save processed video
  cv::VideoWriter writer("output.avi",
    cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
    fps,
    cv::Size(frame_width, frame_height));

  if (!writer.isOpened()) {
    std::cerr << "Warning: Could not create output video file" << std::endl;
  }

  // Create windows
  cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("HSV", cv::WINDOW_AUTOSIZE);

  cv::Mat frame, hsv;
  int frame_count = 0;

  std::cout << "\nPress 'q' to quit, 's' to save current frame" << std::endl;

  while (true) {
    // Capture frame
    cap >> frame;

    // Check if frame is empty (end of video)
    if (frame.empty()) {
      std::cout << "End of video stream" << std::endl;
      break;
    }

    frame_count++;

    // Convert to HSV color space
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    // Display frames
    cv::imshow("Original", frame);
    cv::imshow("HSV", hsv);

    // Write frame to output video
    if (writer.isOpened()) {
      writer.write(hsv);
    }

    // Handle keyboard input
    char key = static_cast<char>(cv::waitKey(1000 / static_cast<int>(fps)));

    if (key == 'q' || key == 'Q' || key == 27) {      // 'q' or ESC to quit
      std::cout << "User requested exit" << std::endl;
      break;
    } else if (key == 's' || key == 'S') {      // 's' to save frame
      std::string filename = "frame_" + std::to_string(frame_count) + ".jpg";
      cv::imwrite(filename, frame);
      std::cout << "Saved: " << filename << std::endl;
    }
  }

  std::cout << "Total frames processed: " << frame_count << std::endl;

  // Release resources
  cap.release();
  writer.release();
  cv::destroyAllWindows();

  return 0;
}
