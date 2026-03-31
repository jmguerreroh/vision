/**
 * @file main.cpp
 * @brief Sparse optical flow using Lucas-Kanade pyramid method
 * @author José Miguel Guerrero Hernández
 *
 * @details Tracks feature points across video frames using the Lucas-Kanade
 *          optical flow algorithm with image pyramids for multi-scale support.
 *
 *          Algorithm:
 *          1. Detect Shi-Tomasi corner points in the first frame
 *          2. For each subsequent frame, track points using calcOpticalFlowPyrLK()
 *          3. Draw motion trails connecting previous and current positions
 *          4. Update tracked points for the next iteration
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
  const std::string about =
    "This sample demonstrates Lucas-Kanade Optical Flow calculation.\n";

  const std::string keys =
    "{ h help |      | print this help message }"
    "{ @image | ../../data/vtest.avi | path to image file }";
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about(about);
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }

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

  // Generate random colors for tracking points
  std::vector<cv::Scalar> colors;
  cv::RNG rng;
  for (int i = 0; i < 100; i++) {
    int r = rng.uniform(0, 256);
    int g = rng.uniform(0, 256);
    int b = rng.uniform(0, 256);
    colors.push_back(cv::Scalar(r, g, b));
  }

  // Capture first frame and detect good feature points
  cv::Mat old_frame, old_gray;
  std::vector<cv::Point2f> p0, p1;

  capture >> old_frame;
  cv::cvtColor(old_frame, old_gray, cv::COLOR_BGR2GRAY);

  // Detect Shi-Tomasi corners in the first frame
  cv::goodFeaturesToTrack(
    old_gray,   // Input grayscale image
    p0,         // Output vector of detected points
    100,        // Maximum number of corners to detect
    0.3,        // Quality level (minimum accepted corner quality)
    7,          // Minimum Euclidean distance between detected corners
    cv::Mat(),  // Mask (empty means full image)
    7,          // Block size for computing the gradient covariance matrix
    false,      // Use Harris detector (false = Shi-Tomasi method)
    0.04        // Harris detector free parameter (used only if Harris is true)
  );

  // Create a mask image for drawing the tracked points
  cv::Mat mask = cv::Mat::zeros(old_frame.size(), old_frame.type());

  // Main processing loop
  while (true) {
    cv::Mat frame, frame_gray;

    // Capture the next frame
    capture >> frame;
    if (frame.empty()) {
      break;
    }
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

    // Compute the optical flow using Lucas-Kanade method
    std::vector<uchar> status; // Status vector: 1 if found, 0 if lost
    std::vector<float> err;    // Error vector for each point
    cv::TermCriteria criteria = cv::TermCriteria(
      (cv::TermCriteria::COUNT) +(cv::TermCriteria::EPS), 10, 0.03);
    cv::calcOpticalFlowPyrLK(
      old_gray,         // Previous grayscale frame
      frame_gray,       // Current grayscale frame
      p0,               // Previous points (to track)
      p1,               // New tracked points (output)
      status,           // Status of each point (1 = found, 0 = lost)
      err,              // Error vector
      cv::Size(15, 15), // Size of search window at each pyramid level
      2,                // Number of pyramid levels (higher values capture larger motions)
      criteria          // Termination criteria: stop after 10 iterations or error < 0.03
    );

    std::vector<cv::Point2f> good_new;
    for (size_t i = 0; i < p0.size(); i++) {
      // Select good points where tracking was successful
      if (status[i] == 1) {
        good_new.push_back(p1[i]);
        // Draw the movement of tracked points
        cv::line(mask, p1[i], p0[i], colors[i], 2);
        cv::circle(frame, p1[i], 5, colors[i], -1);
      }
    }

    // Overlay tracking lines on the frame
    cv::Mat img;
    cv::add(frame, mask, img);

    // Display the frame
    cv::imshow("Frame", img);

    // Exit on 'q' or 'Esc' key press
    int keyboard = cv::waitKey(30);
    if (keyboard == 'q' || keyboard == 27) {
      break;
    }

    // Update previous frame and previous points for next iteration
    old_gray = frame_gray.clone();
    p0 = good_new;
  }

  return EXIT_SUCCESS;
}
