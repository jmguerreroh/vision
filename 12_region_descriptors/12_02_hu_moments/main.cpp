/**
 * @file main.cpp
 * @brief Hu invariant moments for shape recognition using OpenCV
 * @author José Miguel Guerrero Hernández
 * @note This example demonstrates how to compute Hu invariant moments from
 *       contours and use them to compare shape similarity via matchShapes.
 *       Hu moments are invariant to translation, scale, and rotation.
 */

#include <cstdlib>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>   // std::ostringstream (contour labels)
#include <vector>
#include <cmath>

namespace Config
{
// Preprocessing parameters
constexpr int GAUSSIAN_KERNEL_SIZE = 5;
constexpr double GAUSSIAN_SIGMA = 0.0;

// Edge detection parameters
constexpr double CANNY_THRESHOLD_LOW = 50.0;
constexpr double CANNY_THRESHOLD_HIGH = 150.0;
constexpr int CANNY_APERTURE_SIZE = 3;

// Contour filtering
constexpr double MIN_CONTOUR_AREA = 500.0;

// Display parameters
constexpr int CONTOUR_THICKNESS = 2;
constexpr int CENTROID_RADIUS = 5;
constexpr double TEXT_FONT_SCALE = 0.45;
constexpr int TEXT_THICKNESS = 1;
constexpr int TEXT_OFFSET_X = 10;
constexpr int TEXT_OFFSET_Y = -10;

// Image transformation parameters
constexpr double ROTATION_ANGLE = 90.0;
constexpr double RESIZE_SCALE = 0.25;
}

/**
 * @brief Display the 7 Hu invariant moments for a contour
 * @param hu Array of 7 Hu moment values
 * @param label Label to identify the contour
 *
 * Hu moments (h1–h7) are derived from normalized central moments
 * and are invariant to translation, scale, and rotation:
 *   h1 = nu20 + nu02
 *   h2 = (nu20 - nu02)^2 + 4*nu11^2
 *   h3 = (nu30 - 3*nu12)^2 + (3*nu21 - nu03)^2
 *   h4 = (nu30 + nu12)^2 + (nu21 + nu03)^2
 *   h5 = (nu30 - 3*nu12)(nu30 + nu12)[(nu30 + nu12)^2 - 3(nu21 + nu03)^2]
 *        + (3*nu21 - nu03)(nu21 + nu03)[3(nu30 + nu12)^2 - (nu21 + nu03)^2]
 *   h6 = (nu20 - nu02)[(nu30 + nu12)^2 - (nu21 + nu03)^2]
 *        + 4*nu11*(nu30 + nu12)(nu21 + nu03)
 *   h7 = (3*nu21 - nu03)(nu30 + nu12)[(nu30 + nu12)^2 - 3(nu21 + nu03)^2]
 *        - (nu30 - 3*nu12)(nu21 + nu03)[3(nu30 + nu12)^2 - (nu21 + nu03)^2]
 */
void displayHuMoments(const double hu[7], const std::string & label)
{
  std::cout << "  " << label << ":" << std::endl;
  for (int i = 0; i < 7; ++i) {
    // Log transform for readability: -sign(hu) * log10(|hu|).
    // The minus sign makes typical values positive (|hu| < 1 => negative log)
    const double log_hu = (hu[i] != 0.0) ? -std::copysign(1.0,
      hu[i]) * std::log10(std::abs(hu[i])) : 0.0;
    std::cout << "    h" << (i + 1)
              << " = " << std::scientific << std::setprecision(6) << hu[i]
              << "  (log: " << std::fixed << std::setprecision(4) << log_hu << ")"
              << std::endl;
  }
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
 * @return Filtered contours with area > minArea
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
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/shapes.png | Input file}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  const std::string image_path = parser.get<std::string>("@input");

  // Without this, a malformed value (--frames=xyz) is reported by the parser
  // but the example carries on with the default, which is the hardest kind
  // of failure to diagnose. Note it validates values, not option names:
  // cv::CommandLineParser ignores an unknown option without complaining
  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }
  const cv::Mat src = cv::imread(cv::samples::findFile(image_path, false), cv::IMREAD_COLOR);

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
  // Hu Moments from Full Image (BGR → Gray)
  // ========================================
  // cv::moments() also accepts a single-channel image directly,
  // treating pixel intensities as the "mass" distribution.
  // This computes a global shape descriptor for the entire image.
  const cv::Moments image_moments = cv::moments(gray, false);
  double hu_image[7];
  cv::HuMoments(image_moments, hu_image);

  // ========================================
  // Rotated + Resized Image for Comparison
  // ========================================
  // Rotate ROTATION_ANGLE around center and resize to RESIZE_SCALE to demonstrate
  // that Hu moments are invariant to rotation and scale.
  const cv::Point2f center(gray.cols / 2.0f, gray.rows / 2.0f);
  const cv::Mat rot_matrix = cv::getRotationMatrix2D(center, Config::ROTATION_ANGLE, 1.0);
  cv::Mat rotated;
  cv::warpAffine(gray, rotated, rot_matrix, gray.size());

  cv::Mat rotated_resized;
  cv::resize(rotated, rotated_resized, cv::Size(), Config::RESIZE_SCALE, Config::RESIZE_SCALE);

  const cv::Moments transformed_moments = cv::moments(rotated_resized, false);
  double hu_transformed[7];
  cv::HuMoments(transformed_moments, hu_transformed);

  // ========================================
  // Console Output: Compare Original vs Transformed
  // ========================================
  std::cout << "========================================" << std::endl;
  std::cout << "Hu Moments: Original vs Rotated+Resized" << std::endl;
  std::cout << "(Rotated " << Config::ROTATION_ANGLE << "° + Resized to "
            << static_cast<int>(Config::RESIZE_SCALE * 100) << "%)" << std::endl;
  std::cout << "========================================" << std::endl;

  displayHuMoments(hu_image, "Original image");
  std::cout << std::endl;
  displayHuMoments(hu_transformed, "Rotated + Resized");

  // Compare each Hu moment in log scale: -sign(h)*log10(|h|).
  // Small |diff| confirms invariance to rotation and scale.
  std::cout << "\n  Comparison (log-scale difference):" << std::endl;
  for (int i = 0; i < 7; ++i) {
    const double log_orig = (hu_image[i] != 0.0) ?
      -std::copysign(1.0, hu_image[i]) * std::log10(std::abs(hu_image[i])) : 0.0;
    const double log_trans = (hu_transformed[i] != 0.0) ?
      -std::copysign(1.0, hu_transformed[i]) * std::log10(std::abs(hu_transformed[i])) : 0.0;
    const double diff = std::abs(log_orig - log_trans);
    std::cout << "    h" << (i + 1)
              << ": orig=" << std::fixed << std::setprecision(4) << log_orig
              << "  trans=" << log_trans
              << "  |diff|=" << diff
              << (diff < 0.5 ? "  ✓ similar" : "  ✗ differs")
              << std::endl;
  }
  std::cout << std::endl;

  // cv::matchShapes also accepts grayscale images directly (not just contours).
  // It internally computes Hu moments for each image and compares them.
  // A value close to 0 means the two shapes are very similar.
  const double image_similarity = cv::matchShapes(
    gray, rotated_resized, cv::CONTOURS_MATCH_I1, 0.0);
  std::cout << "Image similarity (cv::matchShapes): "
            << std::fixed << std::setprecision(6) << std::setw(12)
            << image_similarity << std::endl;

  // Show the transformed image
  cv::imshow("Rotated + Resized (1/4)", rotated_resized);

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
  cv::findContours(edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  std::cout << "Total contours found: " << contours.size() << std::endl;

  // Filter out small contours to remove noise
  const auto filtered_contours = filterContoursByArea(contours, Config::MIN_CONTOUR_AREA);
  std::cout << "Contours after filtering (area > " << Config::MIN_CONTOUR_AREA << "): "
            << filtered_contours.size() << "\n" << std::endl;

  // ========================================
  // Calculate Moments and Hu Moments
  // ========================================
  // cv::moments computes spatial, central, and normalized central moments
  // cv::HuMoments derives the 7 invariant Hu moments from them
  std::vector<cv::Moments> moments(filtered_contours.size());
  std::vector<cv::Point2f> centroids(filtered_contours.size());

  // Store Hu moments for each contour (7 values each)
  std::vector<std::array<double, 7>> hu_moments(filtered_contours.size());

  for (size_t i = 0; i < filtered_contours.size(); ++i) {
    moments[i] = cv::moments(filtered_contours[i]);
    centroids[i] = calculateCentroid(moments[i]);

    // Compute the 7 Hu invariant moments
    double hu[7];
    cv::HuMoments(moments[i], hu);
    std::copy(hu, hu + 7, hu_moments[i].begin());
  }

  // ========================================
  // Console Output: Hu Moments per Contour
  // ========================================
  std::cout << "========================================" << std::endl;
  std::cout << "Hu Invariant Moments per Contour" << std::endl;
  std::cout << "========================================" << std::endl;

  for (size_t i = 0; i < filtered_contours.size(); ++i) {
    const double area = cv::contourArea(filtered_contours[i]);
    const double perimeter = cv::arcLength(filtered_contours[i], true);

    std::cout << "\nContour #" << i << ":" << std::endl;
    std::cout << "  Centroid: (" << std::fixed << std::setprecision(2)
              << centroids[i].x << ", " << centroids[i].y << ")" << std::endl;
    std::cout << "  Area: " << area << "  Perimeter: " << perimeter << std::endl;

    displayHuMoments(hu_moments[i].data(), "Hu moments");
  }

  // ========================================
  // Shape Comparison using matchShapes
  // ========================================
  // Compare every pair of contours using cv::matchShapes,
  // which internally uses Hu moments.
  // Lower values indicate more similar shapes.
  if (filtered_contours.size() >= 2) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Shape Similarity (cv::matchShapes)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Method: CONTOURS_MATCH_I1 (lower = more similar)\n" << std::endl;

    std::cout << std::setw(12) << " ";
    for (size_t j = 0; j < filtered_contours.size(); ++j) {
      std::cout << std::setw(12) << ("C#" + std::to_string(j));
    }
    std::cout << std::endl;

    for (size_t i = 0; i < filtered_contours.size(); ++i) {
      std::cout << std::setw(12) << ("C#" + std::to_string(i));
      for (size_t j = 0; j < filtered_contours.size(); ++j) {
        const double similarity = cv::matchShapes(
          filtered_contours[i], filtered_contours[j],
          cv::CONTOURS_MATCH_I1, 0.0);
        std::cout << std::fixed << std::setprecision(6) << std::setw(12) << similarity;
      }
      std::cout << std::endl;
    }

    std::cout << "\nNote: Diagonal is 0.0 (contour compared with itself)." << std::endl;
    std::cout << "========================================\n" << std::endl;
  }

  // ========================================
  // Visualization
  // ========================================
  // Draw contours with labels and centroids
  cv::Mat result = src.clone();

  for (size_t i = 0; i < filtered_contours.size(); ++i) {
    // Assign a distinct color per contour (deterministic based on index)
    const int hue = static_cast<int>(i * 180.0 / filtered_contours.size());
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    const cv::Scalar color(bgr.at<cv::Vec3b>(0, 0)[0],
      bgr.at<cv::Vec3b>(0, 0)[1],
      bgr.at<cv::Vec3b>(0, 0)[2]);

    // Draw contour outline
    cv::drawContours(result, filtered_contours, static_cast<int>(i),
                     color, Config::CONTOUR_THICKNESS, cv::LINE_AA);

    // Draw centroid as filled circle
    cv::circle(result, centroids[i],
               Config::CENTROID_RADIUS, cv::Scalar(0, 0, 255), cv::FILLED);

    // Label each contour with its index and log|h1| value
    const double log_h1 = (hu_moments[i][0] != 0.0) ?
      -std::copysign(1.0, hu_moments[i][0]) * std::log10(std::abs(hu_moments[i][0])) :
      0.0;

    std::ostringstream label;
    label << "#" << i << " h1=" << std::fixed << std::setprecision(2) << log_h1;

    cv::putText(result, label.str(),
                cv::Point(static_cast<int>(centroids[i].x) + Config::TEXT_OFFSET_X,
                          static_cast<int>(centroids[i].y) + Config::TEXT_OFFSET_Y),
                cv::FONT_HERSHEY_SIMPLEX, Config::TEXT_FONT_SCALE,
                cv::Scalar(0, 0, 0), Config::TEXT_THICKNESS + 1);
    cv::putText(result, label.str(),
                cv::Point(static_cast<int>(centroids[i].x) + Config::TEXT_OFFSET_X,
                          static_cast<int>(centroids[i].y) + Config::TEXT_OFFSET_Y),
                cv::FONT_HERSHEY_SIMPLEX, Config::TEXT_FONT_SCALE,
                cv::Scalar(255, 255, 255), Config::TEXT_THICKNESS);
  }

  // ========================================
  // Display Results
  // ========================================
  cv::imshow("Original Image", src);
  cv::imshow("Canny Edges", edges);
  cv::imshow("Hu Moments - Shape Analysis", result);

  std::cout << "Press any key to exit..." << std::endl;
  cv::waitKey(0);
  cv::destroyAllWindows();

  return EXIT_SUCCESS;
}
