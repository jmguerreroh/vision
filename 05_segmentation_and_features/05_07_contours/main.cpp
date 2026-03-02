/**
 * @file main.cpp
 * @brief Contour detection and drawing using OpenCV findContours function
 * @author José Miguel Guerrero Hernández
 * @note This example demonstrates how to detect contours in a binary image
 *       using Canny edge detection and findContours, then draw them
 */

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

namespace Config
{
// Preprocessing parameters
constexpr int GAUSSIAN_KERNEL_SIZE = 5;
constexpr double GAUSSIAN_SIGMA = 0.0;

// Edge detection parameters
constexpr double CANNY_THRESHOLD_LOW = 50.0;
constexpr double CANNY_THRESHOLD_HIGH = 100.0;
constexpr int CANNY_APERTURE_SIZE = 3;

// Visualization parameters
constexpr int CONTOUR_THICKNESS = 2;
}

int main(int argc, char ** argv)
{
  // ========================================
  // Load Input Image
  // ========================================
  const std::string image_path = argc >= 2 ? argv[1] : "coins.jpg";
  const cv::Mat src = cv::imread(cv::samples::findFile(image_path), cv::IMREAD_COLOR);

  if (src.empty()) {
    std::cerr << "Error: Could not open or find the image!" << std::endl;
    std::cerr << "Path: " << image_path << std::endl;
    std::cerr << "Usage: " << argv[0] << " <Input image>" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Image loaded: " << src.cols << "x" << src.rows << " pixels\n" << std::endl;

  // ========================================
  // Preprocessing
  // ========================================
  cv::Mat gray, blurred;
  cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

  // Apply Gaussian blur to reduce noise
  cv::GaussianBlur(gray, blurred,
                   cv::Size(Config::GAUSSIAN_KERNEL_SIZE, Config::GAUSSIAN_KERNEL_SIZE),
                   Config::GAUSSIAN_SIGMA);

  // ========================================
  // Edge Detection
  // ========================================
  cv::Mat edges;
  cv::Canny(blurred, edges,
            Config::CANNY_THRESHOLD_LOW,
            Config::CANNY_THRESHOLD_HIGH,
            Config::CANNY_APERTURE_SIZE);

  // ========================================
  // Contour Detection
  // ========================================
  // cv::findContours finds contours in a binary image
  // - RETR_TREE: Retrieves full hierarchy (all relationships)
  //   Alternatives: RETR_EXTERNAL (outermost only), RETR_LIST (no hierarchy)
  // - CHAIN_APPROX_SIMPLE: Stores only endpoints of segments
  //   Alternative: CHAIN_APPROX_NONE (all boundary points)
  // - hierarchy[i] = [next, previous, first_child, parent]
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(edges, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

  std::cout << "Total contours found: " << contours.size() << "\n" << std::endl;

  // ========================================
  // Visualization: Draw Contours
  // ========================================
  // Create black image for drawing contours
  cv::Mat contours_only = cv::Mat::zeros(edges.size(), CV_8UC3);

  // cv::drawContours draws detected contours
  // Parameters:
  // - contourIdx: -1 draws all contours
  // - maxLevel: 1 draws contours + 1st level children
  cv::drawContours(contours_only, contours, -1, cv::Scalar(0, 0, 255),
                   Config::CONTOUR_THICKNESS, cv::LINE_8, hierarchy, 1);

  // Draw contours overlaid on original image
  cv::Mat contours_overlay = src.clone();
  cv::drawContours(contours_overlay, contours, -1, cv::Scalar(0, 255, 0),
                   Config::CONTOUR_THICKNESS, cv::LINE_8, hierarchy, 1);

  // ========================================
  // Display Results
  // ========================================
  cv::imshow("Original Image", src);
  cv::imshow("Grayscale", gray);
  cv::imshow("Gaussian Blur", blurred);
  cv::imshow("Canny Edges", edges);
  cv::imshow("Contours Only", contours_only);
  cv::imshow("Contours Overlay", contours_overlay);

  std::cout << "Press any key to exit..." << std::endl;
  cv::waitKey(0);
  cv::destroyAllWindows();

  return EXIT_SUCCESS;
}
