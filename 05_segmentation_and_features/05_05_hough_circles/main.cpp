/**
 * @file main.cpp
 * @brief Hough Circle Transform for circular object detection
 * @author José Miguel Guerrero Hernández
 *
 * @note The Hough Circle Transform detects circles using the gradient-based
 *          method (HOUGH_GRADIENT). Unlike standard Hough for lines (2D space),
 *          circles require 3D parameter space (center_x, center_y, radius).
 *
 *          Algorithm steps:
 *          1. Apply Canny edge detection internally
 *          2. For each edge point, vote for possible circle centers using gradient direction
 *          3. Find local maxima in accumulator (potential centers)
 *          4. For each center candidate, determine radius by voting
 *
 *          This approach is more efficient than 3D voting, reducing
 *          complexity from O(n³) to O(n²).
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <vector>
#include <iostream>

int main(int argc, char ** argv)
{
  // Load image from argument or use default
  const std::string filename = argc >= 2 ? argv[1] : "smarties.png";
  cv::Mat src = cv::imread(cv::samples::findFile(filename), cv::IMREAD_COLOR);

  if (src.empty()) {
    std::cerr << "Error: Could not open or find the image!" << std::endl;
    std::cerr << "Path: " << filename << std::endl;
    std::cerr << "Usage: " << argv[0] << " <Input image>" << std::endl;
    return -1;
  }

  // ========================================
  // Preprocessing
  // ========================================

  // Convert to grayscale (required for HoughCircles)
  cv::Mat gray;
  cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

  // Apply median blur to reduce noise while preserving edges
  // Kernel size 5 is effective for salt-and-pepper noise removal
  // Unlike Gaussian blur, median blur doesn't blur edges as much
  cv::Mat blurred;
  cv::medianBlur(gray, blurred, 5);

  // ========================================
  // Circle Detection with Hough Transform
  // ========================================

  // Detect circles using Hough Circle Transform
  // cv::HoughCircles(input, output, method, dp, minDist, param1, param2, minR, maxR)
  //
  // Key parameters tuning guide:
  //   dp = 1: Accumulator has same resolution as input image
  //           Higher values (2, 3) reduce accuracy but speed up detection
  //
  //   minDist = rows/16: Minimum distance between detected centers
  //           Too small → multiple detections for same circle
  //           Too large → nearby circles missed
  //
  //   param1 = 100: Upper threshold for internal Canny edge detector
  //           Lower threshold is automatically set to param1/2
  //           Higher values → fewer edges → fewer but cleaner circles
  //
  //   param2 = 30: Accumulator threshold for center detection
  //           Lower values → more circles detected (may include false positives)
  //           Higher values → fewer but more confident detections
  //
  //   minRadius, maxRadius: Filter circles by size
  //           Set based on expected object sizes in your image
  std::vector<cv::Vec3f> circles;
  cv::HoughCircles(
    blurred, circles, cv::HOUGH_GRADIENT, 1,
    blurred.rows / 16,
    100, 30,
    1, 30
  );

  // ========================================
  // Visualization: Draw detected circles
  // ========================================

  // Clone original to preserve it
  cv::Mat result = src.clone();

  // Draw detected circles on the result image
  // Each circle is stored as Vec3f: [center_x, center_y, radius]
  for (size_t i = 0; i < circles.size(); i++) {
    cv::Vec3i c = circles[i];
    cv::Point center = cv::Point(c[0], c[1]);
    int radius = c[2];

    // Draw small filled circle at center (yellow)
    cv::circle(result, center, 3, cv::Scalar(0, 100, 100), -1, cv::LINE_AA);

    // Draw circle perimeter (magenta)
    cv::circle(result, center, radius, cv::Scalar(255, 0, 255), 3, cv::LINE_AA);
  }

  // Display results
  cv::imshow("1. Original", src);
  cv::imshow("2. Grayscale", gray);
  cv::imshow("3. Blurred (Median)", blurred);
  cv::imshow("4. Detected Circles", result);

  // Wait for user input and exit
  cv::waitKey();

  return 0;
}
