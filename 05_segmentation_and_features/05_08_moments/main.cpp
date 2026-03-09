/**
 * @file main.cpp
 * @brief Image moments calculation for contour analysis using OpenCV
 * @author José Miguel Guerrero Hernández
 * @note This example demonstrates how to compute spatial, central, and
 *       normalized central moments from contours, as well as area and perimeter
 * @see https://docs.opencv.org/3.4/d8/d23/classcv_1_1Moments.html
 */

#include <cstdlib>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <vector>

namespace Config
{
// Preprocessing parameters
constexpr int GAUSSIAN_KERNEL_SIZE = 5;
constexpr double GAUSSIAN_SIGMA = 0.0;

// Edge detection parameters
constexpr double CANNY_THRESHOLD_LOW = 50.0;
constexpr double CANNY_THRESHOLD_HIGH = 100.0;
constexpr int CANNY_APERTURE_SIZE = 3;

// Contour filtering
constexpr double MIN_CONTOUR_AREA = 100.0;

// Display parameters
constexpr size_t MAX_MOMENTS_DISPLAY = 3;
constexpr int CENTROID_RADIUS = 5;
constexpr int CONTOUR_THICKNESS = 2;
constexpr double TEXT_FONT_SCALE = 0.4;
constexpr int TEXT_THICKNESS = 1;
constexpr int TEXT_OFFSET_X = 10;
constexpr int TEXT_OFFSET_Y = -10;
}

/**
 * @brief Display all moment values for a given contour
 * @param moments Moments structure containing spatial, central, and normalized moments
 *
 * Moments are computed as:
 * - Spatial moments: m_ij = sum(I(x,y) * x^i * y^j)
 * - Central moments: mu_ij = sum(I(x,y) * (x - x_bar)^i * (y - y_bar)^j)
 * - Normalized central moments: nu_ij = mu_ij / m00^((i+j)/2 + 1)
 */
void displayMoments(const cv::Moments & moments)
{
  std::cout << "  Spatial moments:" << std::endl;
  std::cout << "    m00 = " << std::fixed << std::setprecision(2) << moments.m00
            << ", m10 = " << moments.m10 << ", m01 = " << moments.m01 << std::endl;
  std::cout << "    m20 = " << moments.m20 << ", m11 = " << moments.m11
            << ", m02 = " << moments.m02 << std::endl;
  std::cout << "    m30 = " << moments.m30 << ", m21 = " << moments.m21
            << ", m12 = " << moments.m12 << ", m03 = " << moments.m03 << std::endl;

  std::cout << "  Central moments:" << std::endl;
  std::cout << "    mu20 = " << moments.mu20 << ", mu11 = " << moments.mu11
            << ", mu02 = " << moments.mu02 << std::endl;
  std::cout << "    mu30 = " << moments.mu30 << ", mu21 = " << moments.mu21
            << ", mu12 = " << moments.mu12 << ", mu03 = " << moments.mu03 << std::endl;

  std::cout << "  Normalized central moments:" << std::endl;
  std::cout << "    nu20 = " << moments.nu20 << ", nu11 = " << moments.nu11
            << ", nu02 = " << moments.nu02 << std::endl;
  std::cout << "    nu30 = " << moments.nu30 << ", nu21 = " << moments.nu21
            << ", nu12 = " << moments.nu12 << ", nu03 = " << moments.nu03 << std::endl;
}

/**
 * @brief Calculate centroid from spatial moments
 * @param moments Moments structure
 * @return Centroid point (x̄, ȳ) = (m10/m00, m01/m00)
 */
cv::Point2f calculateCentroid(const cv::Moments & moments)
{
  if (moments.m00 != 0) {
    return cv::Point2f(
      static_cast<float>(moments.m10 / moments.m00),
      static_cast<float>(moments.m01 / moments.m00)
    );
  }
  return cv::Point2f(0.0f, 0.0f);
}

/**
 * @brief Filter contours by minimum area
 * @param contours Input contours
 * @param minArea Minimum area threshold
 * @return Filtered contours
 */
std::vector<std::vector<cv::Point>> filterContoursByArea(
  const std::vector<std::vector<cv::Point>> & contours,
  double minArea)
{
  std::vector<std::vector<cv::Point>> filtered;
  for (const auto & contour : contours) {
    if (cv::contourArea(contour) > minArea) {
      filtered.push_back(contour);
    }
  }
  return filtered;
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
  cv::GaussianBlur(gray, blurred,
                   cv::Size(Config::GAUSSIAN_KERNEL_SIZE, Config::GAUSSIAN_KERNEL_SIZE),
                   Config::GAUSSIAN_SIGMA);

  // ========================================
  // Edge Detection and Contour Finding
  // ========================================
  cv::Mat edges;
  cv::Canny(blurred, edges,
            Config::CANNY_THRESHOLD_LOW,
            Config::CANNY_THRESHOLD_HIGH,
            Config::CANNY_APERTURE_SIZE);

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(edges, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
  std::cout << "Total contours found: " << contours.size() << std::endl;

  // Filter out small contours to remove noise
  const auto filtered_contours = filterContoursByArea(contours, Config::MIN_CONTOUR_AREA);
  std::cout << "Contours after filtering (area > " << Config::MIN_CONTOUR_AREA << "): "
            << filtered_contours.size() << "\n" << std::endl;

  // ========================================
  // Calculate Moments and Centroids
  // ========================================
  // cv::moments calculates all spatial, central, and normalized moments up to 3rd order
  std::vector<cv::Moments> moments(filtered_contours.size());
  std::vector<cv::Point2f> centroids(filtered_contours.size());

  for (size_t i = 0; i < filtered_contours.size(); ++i) {
    moments[i] = cv::moments(filtered_contours[i]);
    centroids[i] = calculateCentroid(moments[i]);
  }

  // ========================================
  // Visualization
  // ========================================
  // Draw colored contours on black background
  cv::Mat contours_colored = cv::Mat::zeros(edges.size(), CV_8UC3);
  for (size_t i = 0; i < filtered_contours.size(); ++i) {
    const cv::Scalar color(rand() & 255, rand() & 255, rand() & 255);
    cv::drawContours(contours_colored, filtered_contours, static_cast<int>(i),
                     color, cv::FILLED, cv::LINE_8);
  }

  // Draw contours with centroids on original image
  cv::Mat image_with_centroids = src.clone();
  for (size_t i = 0; i < filtered_contours.size(); ++i) {
    // Draw contour outline in green
    cv::drawContours(image_with_centroids, filtered_contours, static_cast<int>(i),
                     cv::Scalar(0, 255, 0), Config::CONTOUR_THICKNESS, cv::LINE_8);

    // Draw centroid as red circle
    cv::circle(image_with_centroids, centroids[i],
               Config::CENTROID_RADIUS, cv::Scalar(0, 0, 255), cv::FILLED);

    // Add centroid coordinates as text
    const std::string coord_text = "(" +
      std::to_string(static_cast<int>(centroids[i].x)) + "," +
      std::to_string(static_cast<int>(centroids[i].y)) + ")";
    cv::putText(image_with_centroids, coord_text,
                cv::Point(centroids[i].x + Config::TEXT_OFFSET_X,
                         centroids[i].y + Config::TEXT_OFFSET_Y),
                cv::FONT_HERSHEY_SIMPLEX, Config::TEXT_FONT_SCALE,
                cv::Scalar(0, 0, 0), Config::TEXT_THICKNESS);
  }

  // ========================================
  // Console Output: Detailed Moments
  // ========================================
  const size_t num_to_display = std::min(Config::MAX_MOMENTS_DISPLAY, filtered_contours.size());

  std::cout << "========================================" << std::endl;
  std::cout << "Detailed Moments Information" << std::endl;
  std::cout << "========================================" << std::endl;

  for (size_t i = 0; i < num_to_display; ++i) {
    std::cout << "\nContour #" << i << ":" << std::endl;
    std::cout << "  Centroid: (" << std::fixed << std::setprecision(2)
              << centroids[i].x << ", " << centroids[i].y << ")" << std::endl;
    displayMoments(moments[i]);
  }

  if (filtered_contours.size() > num_to_display) {
    std::cout << "\n(Showing only first " << num_to_display
              << " of " << filtered_contours.size() << " contours)" << std::endl;
  }

  // ========================================
  // Console Output: Area and Perimeter
  // ========================================
  std::cout << "\n========================================" << std::endl;
  std::cout << "Area and Perimeter Summary" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << std::fixed << std::setprecision(2);

  for (size_t i = 0; i < filtered_contours.size(); ++i) {
    const double area_moment = moments[i].m00;
    const double area_opencv = cv::contourArea(filtered_contours[i]);
    const double perimeter = cv::arcLength(filtered_contours[i], true);

    std::cout << "Contour #" << std::setw(2) << i << " | "
              << "Area (m00): " << std::setw(8) << area_moment << " | "
              << "Area (cv): " << std::setw(8) << area_opencv << " | "
              << "Perimeter: " << std::setw(8) << perimeter << std::endl;
  }
  std::cout << "========================================\n" << std::endl;

  // ========================================
  // Display Results
  // ========================================
  cv::imshow("Original Image", src);
  cv::imshow("Grayscale", gray);
  cv::imshow("Gaussian Blur", blurred);
  cv::imshow("Canny Edges", edges);
  cv::imshow("Colored Contours", contours_colored);
  cv::imshow("Contours with Centroids", image_with_centroids);

  std::cout << "Press any key to exit..." << std::endl;
  cv::waitKey(0);
  cv::destroyAllWindows();

  return EXIT_SUCCESS;
}
