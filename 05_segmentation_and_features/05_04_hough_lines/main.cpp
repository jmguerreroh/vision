/**
 * @file main.cpp
 * @brief Hough Line Transform detection using Standard and Probabilistic methods
 * @author José Miguel Guerrero Hernández
 * @note The Hough Transform is a feature extraction technique used to detect lines.
 *       It transforms points in (x,y) space to lines in (rho,theta) parameter space.
 *       Standard Hough: detects infinite lines using (rho, theta) representation
 *       Probabilistic Hough: more efficient, returns finite line segments
 */

#include <cstdlib>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>

int main(int argc, char ** argv)
{
  // Load input image
  const std::string image_path = argc >= 2 ? argv[1] : "chess.jpg";
  cv::Mat src = cv::imread(cv::samples::findFile(image_path), cv::IMREAD_COLOR);

  if (src.empty()) {
    std::cerr << "Error: Could not open or find the image!" << std::endl;
    std::cerr << "Path: " << image_path << std::endl;
    std::cerr << "Usage: " << argv[0] << " <Input image>" << std::endl;
    return EXIT_FAILURE;
  }

  // ========================================
  // Preprocessing
  // ========================================

  // Convert to grayscale (Canny requires single-channel image)
  cv::Mat gray;
  cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

  // Canny edge detection as preprocessing for Hough Transform
  // The Hough Transform works on binary edge images (white edges on black background)
  cv::Mat edges;
  cv::Canny(gray, edges, 50, 200, 3);

  // Convert edge image to BGR for colored line visualization
  cv::Mat standard_hough_result, probabilistic_hough_result;
  cv::cvtColor(edges, standard_hough_result, cv::COLOR_GRAY2BGR);
  probabilistic_hough_result = standard_hough_result.clone();

  // ========================================
  // Method 1: Standard Hough Line Transform
  // ========================================

  // cv::HoughLines(image, lines, rho, theta, threshold, srn, stn)
  // Detects infinite lines in parameter space (rho, theta)
  // Parameters:
  //   image: 8-bit, single-channel binary source image (edge map)
  //   lines: output vector of lines as (rho, theta) pairs
  //     rho: perpendicular distance from origin to the line
  //     theta: angle of the perpendicular in radians [0, π]
  //   rho: distance resolution in pixels (1 pixel)
  //   theta: angle resolution in radians (1° = CV_PI/180)
  //   threshold: accumulator threshold - minimum votes to detect line (200)
  //   srn, stn: for multi-scale Hough transform (0 = disabled)
  std::vector<cv::Vec2f> lines;
  cv::HoughLines(edges, lines, 1, CV_PI / 180, 200, 0, 0);

  // Draw detected lines on the image
  // Lines are in (rho, theta) polar form - convert to Cartesian endpoints
  // Line equation: x*cos(θ) + y*sin(θ) = ρ
  for (size_t i = 0; i < lines.size(); i++) {
    float rho = lines[i][0], theta = lines[i][1];
    cv::Point pt1, pt2;
    double a = std::cos(theta), b = std::sin(theta);
    double x0 = a * rho, y0 = b * rho;  // Point on the line closest to origin
    // Extend line to image boundaries (1000 pixels in each direction)
    pt1.x = cvRound(x0 + 1000 * (-b));
    pt1.y = cvRound(y0 + 1000 * (a));
    pt2.x = cvRound(x0 - 1000 * (-b));
    pt2.y = cvRound(y0 - 1000 * (a));
    cv::line(standard_hough_result, pt1, pt2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
  }

  // ========================================
  // Method 2: Probabilistic Hough Line Transform
  // ========================================

  // cv::HoughLinesP(image, lines, rho, theta, threshold, minLineLength, maxLineGap)
  // More efficient variant that returns finite line segments with endpoints
  // Parameters:
  //   lines: output vector of line segments as (x1, y1, x2, y2)
  //   threshold: accumulator threshold (50 votes)
  //   minLineLength: minimum length of line segment to detect (50 pixels)
  //   maxLineGap: maximum gap between points to join as single line (10 pixels)
  std::vector<cv::Vec4i> linesP;
  cv::HoughLinesP(edges, linesP, 1, CV_PI / 180, 50, 50, 10);

  // Draw detected line segments
  // Each line is already represented as (x1, y1, x2, y2) endpoints - no conversion needed
  for (size_t i = 0; i < linesP.size(); i++) {
    cv::Vec4i l = linesP[i];
    cv::line(probabilistic_hough_result, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]),
             cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
  }

  // Display results
  cv::imshow("1. Original", src);
  cv::imshow("2. Grayscale", gray);
  cv::imshow("3. Edges (Canny)", edges);
  cv::imshow("4. Standard Hough Lines", standard_hough_result);
  cv::imshow("5. Probabilistic Hough Lines", probabilistic_hough_result);

  // Wait for user input and exit
  cv::waitKey();
  return EXIT_SUCCESS;
}
