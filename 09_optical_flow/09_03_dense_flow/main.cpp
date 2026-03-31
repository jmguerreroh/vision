/**
 * @file main.cpp
 * @brief Dense optical flow using Gunnar Farneback's algorithm
 * @author José Miguel Guerrero Hernández
 *
 * @details Computes optical flow for every pixel in the frame using Farneback's
 *          polynomial expansion method. Unlike Lucas-Kanade (sparse), this
 *          approach produces a complete motion field.
 *
 *          Visualization:
 *          - Flow direction → Hue channel (color wheel)
 *          - Flow magnitude → Value channel (brightness)
 *          - Saturation is set to 1.0 (fully saturated)
 *
 * @see https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
 */

#include <cstdlib>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

int main(int argc, char ** argv)
{
  // Command-line parser setup
  const std::string keys =
    "{ h help |      | print this help message }"
    "{ @image | ../../data/vtest.avi | path to image file }";
  cv::CommandLineParser parser(argc, argv, keys);

  std::string filename = cv::samples::findFile(parser.get<std::string>("@image"));
  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }

  // Open video file
  cv::VideoCapture capture(filename);
  if (!capture.isOpened()) {
    std::cerr << "Unable to open file!" << std::endl;
    return EXIT_FAILURE;
  }

  // Read first frame and convert to grayscale
  cv::Mat frame1, prvs;
  capture >> frame1;
  cv::cvtColor(frame1, prvs, cv::COLOR_BGR2GRAY);

  // Main processing loop
  while (true) {
    cv::Mat frame2, next;

    // Capture the next frame
    capture >> frame2;
    if (frame2.empty()) {
      break;
    }

    // Convert to grayscale
    cv::cvtColor(frame2, next, cv::COLOR_BGR2GRAY);

    // Compute dense optical flow using Farneback's algorithm
    cv::Mat flow(prvs.size(), CV_32FC2);
    cv::calcOpticalFlowFarneback(
      prvs,   // Previous grayscale frame
      next,   // Current grayscale frame
      flow,   // Output flow image (2-channel: u, v components)
      0.5,    // Pyramid scale (0.5 = classical pyramid, each level halves the resolution)
      3,      // Number of pyramid levels (more levels capture larger motions)
      15,     // Window size for averaging (larger = smoother but less precise)
      3,      // Number of iterations at each pyramid level
      5,      // Pixel neighborhood size for polynomial expansion
      1.2,    // Standard deviation of Gaussian for polynomial expansion smoothing
      0       // Flags (0 = default behavior)
    );

    // Split the optical flow into x and y components
    cv::Mat flow_parts[2];
    cv::split(flow, flow_parts);
    cv::Mat magnitude, angle, magn_norm;
    cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
    cv::normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);
    angle *= ((1.f / 360.f) * (180.f / 255.f));

    // Create an HSV image representation of optical flow
    // H (Hue):        flow direction → color indicates movement direction
    // S (Saturation): fixed at 1.0  → fully saturated colors
    // V (Value):      flow magnitude → brightness indicates movement speed
    cv::Mat hsv_channels[3], hsv_image, hsv8, bgr;
    hsv_channels[0] = angle;                                // H: direction of motion
    hsv_channels[1] = cv::Mat::ones(angle.size(), CV_32F);  // S: max saturation
    hsv_channels[2] = magn_norm;                            // V: magnitude (bright = fast)
    cv::merge(hsv_channels, 3, hsv_image);
    hsv_image.convertTo(hsv8, CV_8U, 255.0);
    cv::cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);

    // Display the optical flow visualization
    cv::imshow("frame2", bgr);

    // Wait for user input to continue or exit
    int keyboard = cv::waitKey(30);
    if (keyboard == 'q' || keyboard == 27) {
      break;
    }

    // Update previous frame for next iteration
    prvs = next;
  }

  return EXIT_SUCCESS;
}
