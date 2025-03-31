/**
 * Optical flow using Gunner Farneback's algorithm demo sample
 * @author José Miguel Guerrero
 *
 * https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
 *
 * Lucas-Kanade method computes optical flow for a sparse feature set (in our example, corners detected using Shi-Tomasi algorithm).
 * OpenCV provides another algorithm to find the dense optical flow. It computes the optical flow for all the points in the frame.
 * It is based on Gunner Farneback's algorithm which is explained in "Two-Frame Motion Estimation Based on Polynomial Expansion" by Gunner Farneback in 2003.
 *
 * Below sample shows how to find the dense optical flow using above algorithm. We get a 2-channel array with optical flow vectors, (u,v).
 * We find their magnitude and direction. We color code the result for better visualization. Direction corresponds to Hue value of the image.
 * Magnitude corresponds to Value plane
 */

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

int main(int argc, char ** argv)
{
  // Define command line arguments with default values
  const std::string keys =
    "{ h help |      | print this help message }"
    "{ @image | vtest.avi | path to image file }";
  cv::CommandLineParser parser(argc, argv, keys);

  // Parse the input file path
  std::string filename = cv::samples::findFile(parser.get<std::string>("@image"));
  if (!parser.check()) {
    parser.printErrors();
    return 0;
  }

  // Open the video file
  cv::VideoCapture capture(filename);
  if (!capture.isOpened()) {
    // Error handling if the video file cannot be opened
    std::cerr << "Unable to open file!" << std::endl;
    return 0;
  }

  // Read the first frame and convert it to grayscale
  cv::Mat frame1, prvs;
  capture >> frame1;
  cv::cvtColor(frame1, prvs, cv::COLOR_BGR2GRAY);

  while (true) {
    cv::Mat frame2, next;
    // Capture the next frame
    capture >> frame2;
    if (frame2.empty()) {
      break; // Break the loop if no more frames are available
    }
    // Convert to grayscale
    cv::cvtColor(frame2, next, cv::COLOR_BGR2GRAY);

    // Compute dense optical flow using Farneback's algorithm
    cv::Mat flow(prvs.size(), CV_32FC2);
    cv::calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

    // Split the optical flow into x and y components
    cv::Mat flow_parts[2];
    cv::split(flow, flow_parts);
    cv::Mat magnitude, angle, magn_norm;
    cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true); // Convert flow to polar coordinates
    cv::normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX); // Normalize magnitude
    angle *= ((1.f / 360.f) * (180.f / 255.f)); // Scale angle for HSV representation

    // Create an HSV image representation of optical flow
    cv::Mat _hsv[3], hsv, hsv8, bgr;
    _hsv[0] = angle; // Hue (direction of motion)
    _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magn_norm;
    cv::merge(_hsv, 3, hsv);
    hsv.convertTo(hsv8, CV_8U, 255.0); // Convert to 8-bit image
    cv::cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR); // Convert HSV to BGR for visualization

    // Display the optical flow visualization
    cv::imshow("frame2", bgr);

    // Wait for user input to continue or exit
    int keyboard = cv::waitKey(30);
    if (keyboard == 'q' || keyboard == 27) { // Press 'q' or 'ESC' to exit
      break;
    }

    // Update previous frame for next iteration
    prvs = next;
  }
}
