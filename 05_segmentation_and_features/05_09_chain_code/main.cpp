/**
 * @file main.cpp
 * @brief Chain Code representation of contours (Freeman code)
 * @author José Miguel Guerrero Hernández
 * @note This example demonstrates the computation and visualization of
 *       chain codes (Freeman codes) for binary image contours using
 *       8-connectivity directions (0-7)
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// ============================================================================
// Constants
// ============================================================================

namespace Config
{
// Binary thresholding
constexpr int THRESH_VALUE = 0;
constexpr int THRESH_MAX_VALUE = 255;

// Contour filtering
constexpr size_t MIN_CONTOUR_POINTS = 5;

// Visualization parameters
constexpr int ARROW_LENGTH = 5;
constexpr double ARROW_TIP_LENGTH = 0.4;
constexpr double ARROW_THICKNESS = 1.5;
constexpr int STARTING_POINT_RADIUS = 6;
constexpr int CENTROID_RADIUS = 4;
constexpr int CONTOUR_THICKNESS = 2;
constexpr int CONTOUR_VIZ_THICKNESS = 1;
constexpr double TEXT_FONT_SCALE = 0.5;
constexpr int TEXT_THICKNESS = 2;

// Chain code display limits
constexpr size_t MAX_CHAIN_CODE_DISPLAY = 50;

// Adaptive arrow step sizes
constexpr size_t STEP_THRESHOLD_VERY_LARGE = 500;
constexpr size_t STEP_THRESHOLD_LARGE = 200;
constexpr size_t STEP_THRESHOLD_MEDIUM = 80;
constexpr int STEP_VERY_LARGE = 15;
constexpr int STEP_LARGE = 8;
constexpr int STEP_MEDIUM = 4;
constexpr int STEP_SMALL = 2;
}

/**
 * @brief Computes the 8-directional chain code for a contour
 * @param contour Vector of points representing the contour
 * @return Vector of integers (0-7) representing directions
 *
 * Freeman chain code directions (8-connectivity):
 *   3   2   1
 *    \  |  /
 *   4 --P-- 0
 *    /  |  \
 *   5   6   7
 *
 * 0 = East (right)
 * 1 = North-East (diagonal up-right)
 * 2 = North (up)
 * 3 = North-West (diagonal up-left)
 * 4 = West (left)
 * 5 = South-West (diagonal down-left)
 * 6 = South (down)
 * 7 = South-East (diagonal down-right)
 */
std::vector<int> computeChainCode(const std::vector<cv::Point> & contour)
{
  std::vector<int> chain_code;

  if (contour.size() < 2) {
    return chain_code;
  }

  for (size_t i = 0; i < contour.size() - 1; ++i) {
    const int dx = contour[i + 1].x - contour[i].x;
    const int dy = contour[i + 1].y - contour[i].y;

    // Map (dx, dy) to chain code direction (0-7)
    int code = -1;

    if (dx == 1 && dy == 0) {
      code = 0;  // Right
    } else if (dx == 1 && dy == -1) {
      code = 1;  // Up-Right
    } else if (dx == 0 && dy == -1) {
      code = 2;  // Up
    } else if (dx == -1 && dy == -1) {
      code = 3;  // Up-Left
    } else if (dx == -1 && dy == 0) {
      code = 4;  // Left
    } else if (dx == -1 && dy == 1) {
      code = 5;  // Down-Left
    } else if (dx == 0 && dy == 1) {
      code = 6;  // Down
    } else if (dx == 1 && dy == 1) {
      code = 7;  // Down-Right
    } else {
      // Non-unit step, normalize using angle
      const double angle = std::atan2(-dy, dx);  // Negative dy: y-axis points down
      code = static_cast<int>(std::round(angle / (CV_PI / 4.0)));
      // Ensure code is in range [0, 7]
      code = (code % 8 + 8) % 8;
    }

    chain_code.push_back(code);
  }

  return chain_code;
}

/**
 * @brief Draws directional arrows on contour to visualize chain code
 * @param image Image to draw on
 * @param contour Contour points
 * @param chainCode Chain code directions
 * @param color Arrow color
 * @param step Draw arrow every 'step' points (default 1)
 */
void drawChainCodeArrows(
  cv::Mat & image,
  const std::vector<cv::Point> & contour,
  const std::vector<int> & chain_code,
  const cv::Scalar & color,
  int step = 1)
{
  // Direction vectors for each code (8-connectivity)
  static const int dx[] = {1, 1, 0, -1, -1, -1, 0, 1};
  static const int dy[] = {0, -1, -1, -1, 0, 1, 1, 1};

  for (size_t i = 0; i < chain_code.size() && i < contour.size(); i += step) {
    const int code = chain_code[i];
    const cv::Point start = contour[i];
    const cv::Point end = start + cv::Point(dx[code] * Config::ARROW_LENGTH,
                                             dy[code] * Config::ARROW_LENGTH);

    cv::arrowedLine(image, start, end, color,
                   Config::ARROW_THICKNESS, cv::LINE_AA, 0,
                   Config::ARROW_TIP_LENGTH);
  }
}

/**
 * @brief Computes the first difference of a chain code
 * @param chainCode Original chain code
 * @return First difference chain code (rotation invariant)
 *
 * First difference = (code[i] - code[i-1]) mod 8
 * This representation is invariant to starting point rotation
 */
std::vector<int> computeFirstDifference(const std::vector<int> & chain_code)
{
  std::vector<int> diff;

  if (chain_code.size() < 2) {
    return diff;
  }

  for (size_t i = 1; i < chain_code.size(); ++i) {
    const int d = (chain_code[i] - chain_code[i - 1] + 8) % 8;
    diff.push_back(d);
  }

  return diff;
}

/**
 * @brief Determines adaptive step size for arrow visualization based on contour size
 * @param contour_size Number of points in contour
 * @return Step size for drawing arrows
 */
int getAdaptiveStepSize(size_t contour_size)
{
  if (contour_size > Config::STEP_THRESHOLD_VERY_LARGE) {
    return Config::STEP_VERY_LARGE;
  } else if (contour_size > Config::STEP_THRESHOLD_LARGE) {
    return Config::STEP_LARGE;
  } else if (contour_size > Config::STEP_THRESHOLD_MEDIUM) {
    return Config::STEP_MEDIUM;
  } else {
    return Config::STEP_SMALL;
  }
}

int main(int argc, char ** argv)
{
  // ========================================
  // Load Input Image
  // ========================================
  const std::string filename = argc >= 2 ? argv[1] : "../../data/shapes.png";

  std::cout << "========================================" << std::endl;
  std::cout << "Chain Code (Freeman Code) Computation" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Reading image: " << filename << std::endl;

  const cv::Mat src = cv::imread(filename, cv::IMREAD_COLOR);

  if (src.empty()) {
    std::cerr << "Error: Could not load image!" << std::endl;
    std::cerr << "Path: " << filename << std::endl;
    std::cerr << "Usage: " << argv[0] << " [image_path]" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Image loaded: " << src.cols << "x" << src.rows << " pixels\n" << std::endl;

  // ========================================
  // Convert to Binary Image
  // ========================================
  cv::Mat gray, binary;
  cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

  // Use Otsu's method for automatic threshold
  cv::threshold(gray, binary,
               Config::THRESH_VALUE,
               Config::THRESH_MAX_VALUE,
               cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

  // ========================================
  // Find Contours
  // ========================================
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;

  cv::findContours(binary, contours, hierarchy,
                   cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  std::cout << "Total contours found: " << contours.size() << "\n" << std::endl;

  // ========================================
  // Compute and Visualize Chain Codes
  // ========================================
  cv::Mat contours_colored = src.clone();
  cv::Mat chain_code_visualization = src.clone();

  // Draw all contours first
  cv::drawContours(contours_colored, contours, -1,
                  cv::Scalar(0, 255, 0), Config::CONTOUR_THICKNESS);
  cv::drawContours(chain_code_visualization, contours, -1,
                  cv::Scalar(0, 255, 0), Config::CONTOUR_VIZ_THICKNESS);

  for (size_t i = 0; i < contours.size(); ++i) {
    // Skip very small contours
    if (contours[i].size() < Config::MIN_CONTOUR_POINTS) {
      continue;
    }

    // Compute chain code and first difference
    const std::vector<int> chain_code = computeChainCode(contours[i]);
    const std::vector<int> first_diff = computeFirstDifference(chain_code);

    // Calculate contour statistics
    const cv::Moments m = cv::moments(contours[i]);
    const cv::Point2f centroid(static_cast<float>(m.m10 / m.m00),
      static_cast<float>(m.m01 / m.m00));
    const double area = cv::contourArea(contours[i]);

    std::cout << "--- Contour #" << (i + 1) << " ---" << std::endl;
    std::cout << "  Points: " << contours[i].size() << std::endl;
    std::cout << "  Area: " << std::fixed << std::setprecision(2)
              << area << " pixels²" << std::endl;
    std::cout << "  Chain Code Length: " << chain_code.size() << std::endl;

    // Print chain code (limited display)
    std::cout << "  Chain Code: ";
    const size_t max_display = std::min(chain_code.size(), Config::MAX_CHAIN_CODE_DISPLAY);
    for (size_t j = 0; j < max_display; ++j) {
      std::cout << chain_code[j];
    }
    if (chain_code.size() > Config::MAX_CHAIN_CODE_DISPLAY) {
      std::cout << "... (" << chain_code.size() << " total)";
    }
    std::cout << std::endl;

    // Print first difference (limited display)
    std::cout << "  First Diff: ";
    const size_t max_diff_display = std::min(first_diff.size(), Config::MAX_CHAIN_CODE_DISPLAY);
    for (size_t j = 0; j < max_diff_display; ++j) {
      std::cout << first_diff[j];
    }
    if (first_diff.size() > Config::MAX_CHAIN_CODE_DISPLAY) {
      std::cout << "... (" << first_diff.size() << " total)";
    }
    std::cout << "\n" << std::endl;

    // Draw chain code arrows with adaptive step size
    const int step = getAdaptiveStepSize(contours[i].size());
    drawChainCodeArrows(chain_code_visualization, contours[i], chain_code,
                       cv::Scalar(0, 0, 255), step);

    // Mark starting point (origin) in red
    cv::circle(contours_colored, contours[i][0],
              Config::STARTING_POINT_RADIUS, cv::Scalar(0, 0, 255), cv::FILLED);
    cv::circle(chain_code_visualization, contours[i][0],
              Config::STARTING_POINT_RADIUS, cv::Scalar(0, 0, 255), cv::FILLED);

    // Draw centroid in blue
    cv::circle(contours_colored, centroid,
              Config::CENTROID_RADIUS, cv::Scalar(255, 0, 0), cv::FILLED);
    cv::circle(chain_code_visualization, centroid,
              Config::CENTROID_RADIUS, cv::Scalar(255, 0, 0), cv::FILLED);

    // Add contour label
    const std::string label = "C" + std::to_string(i + 1);
    const cv::Point2f text_offset(10.0f, 0.0f);
    cv::putText(contours_colored, label, centroid + text_offset,
                cv::FONT_HERSHEY_SIMPLEX, Config::TEXT_FONT_SCALE,
                cv::Scalar(255, 255, 0), Config::TEXT_THICKNESS);
    cv::putText(chain_code_visualization, label, centroid + text_offset,
                cv::FONT_HERSHEY_SIMPLEX, Config::TEXT_FONT_SCALE,
                cv::Scalar(255, 255, 0), Config::TEXT_THICKNESS);
  }

  // ========================================
  // Display Results
  // ========================================
  cv::imshow("Original Image", src);
  cv::imshow("Grayscale", gray);
  cv::imshow("Binary (Otsu Threshold)", binary);
  cv::imshow("Contours with Landmarks", contours_colored);
  cv::imshow("Chain Code Directions", chain_code_visualization);

  // ========================================
  // Display Legend and Exit
  // ========================================
  std::cout << "========================================" << std::endl;
  std::cout << "Freeman Chain Code Direction Legend:" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "      3   2   1" << std::endl;
  std::cout << "       \\  |  /" << std::endl;
  std::cout << "       4--P--0" << std::endl;
  std::cout << "       /  |  \\" << std::endl;
  std::cout << "      5   6   7" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Visualization Legend:" << std::endl;
  std::cout << "  Red circle   = Starting point (origin)" << std::endl;
  std::cout << "  Blue circle  = Centroid" << std::endl;
  std::cout << "  Green line   = Contour outline" << std::endl;
  std::cout << "  Red arrows   = Chain code directions" << std::endl;
  std::cout << "                 (sampled adaptively)" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Press any key to exit..." << std::endl;

  cv::waitKey(0);
  cv::destroyAllWindows();

  return EXIT_SUCCESS;
}
