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

#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>    // cvtColor
#include <opencv2/imgcodecs.hpp>  // imwrite
#include <opencv2/videoio.hpp>    // VideoCapture, VideoWriter
#include <opencv2/highgui.hpp>    // imshow, waitKey
#include <iostream>

int main(int argc, char ** argv)
{
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | | Video file; if omitted, the default camera is used}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  const std::string source = parser.get<std::string>("@input");

  // Without this, a malformed value (--frames=xyz) is reported by the parser
  // but the example carries on with the default, which is the hardest kind
  // of failure to diagnose. Note it validates values, not option names:
  // cv::CommandLineParser ignores an unknown option without complaining
  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }

  // Open video source: camera (0) or file
  cv::VideoCapture cap;

  if (!source.empty()) {
    // Open video file
    cap.open(source);
    std::cout << "Opening video file: " << source << std::endl;
  } else {
    // Open default camera
    cap.open(0);
    std::cout << "Opening camera..." << std::endl;
  }

  // Check if video source was opened successfully
  if (!cap.isOpened()) {
    std::cerr << "Error: Could not open video source" << std::endl;
    return EXIT_FAILURE;
  }

  // Get video properties
  int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = cap.get(cv::CAP_PROP_FPS);

  // Some cameras report 0 (or nonsense) FPS; clamp to a sane value so the
  // VideoWriter and the waitKey() delay below never divide by zero
  if (fps < 1.0) {fps = 30.0;}

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

    // Write the processed frame to the output video.
    // Note: the codec assumes the 3 channels are B,G,R, so the saved file
    // will show the HSV values *reinterpreted* as BGR colors (same effect
    // as displaying it with imshow). We save it anyway to demonstrate the
    // write pipeline; save 'frame' instead to store the original video.
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

  return EXIT_SUCCESS;
}
