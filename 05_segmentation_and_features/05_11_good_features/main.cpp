/**
 * @file main.cpp
 * @brief Shi-Tomasi Corner Detector (Good Features to Track)
 * @author José Miguel Guerrero Hernández
 * @note This example demonstrates the Shi-Tomasi corner detection method,
 *       an improvement over Harris that uses Gaussian derivatives for
 *       finding the best trackable features (corners).
 *
 * Shi-Tomasi vs Harris:
 *   - Harris: Uses R = λ₁·λ₂ - k(λ₁+λ₂)² = det(M) - k·trace(M)²
 *   - Shi-Tomasi: Uses R = min(λ₁, λ₂)
 *   - Shi-Tomasi is better for tracking (selects more stable corners)
 *
 * Algorithm Steps:
 *   1. Compute image gradients (Ix, Iy) using Gaussian derivatives
 *   2. Build structure tensor M = [Ix², IxIy; IxIy, Iy²] (Gaussian weighted)
 *   3. Calculate eigenvalues (λ₁, λ₂) for each pixel
 *   4. Corner response = min(λ₁, λ₂)
 *   5. Apply non-maximum suppression and quality threshold
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

namespace Config
{
// Default detection parameters
constexpr int DEFAULT_MAX_CORNERS = 100;
constexpr double DEFAULT_QUALITY_LEVEL = 0.01;
constexpr int DEFAULT_MIN_DISTANCE = 10;
constexpr int DEFAULT_BLOCK_SIZE = 3;
constexpr double DEFAULT_K = 0.04;

// Trackbar limits
constexpr int MAX_CORNERS_LIMIT = 500;
constexpr int QUALITY_SCALE_FACTOR = 100;
constexpr int MAX_QUALITY_SCALED = 100;
constexpr int MAX_DISTANCE = 100;
constexpr int MAX_BLOCK_SIZE = 31;
constexpr int MIN_BLOCK_SIZE = 1;

// Visualization parameters
constexpr int MARKER_SIZE = 10;
constexpr int MARKER_THICKNESS = 2;
constexpr int CIRCLE_RADIUS = 5;
constexpr int CIRCLE_THICKNESS = 1;
constexpr int MAX_LABELED_CORNERS = 20;
constexpr double LABEL_FONT_SCALE = 0.4;
constexpr double LABEL_THICKNESS = 1.5;
constexpr int LABEL_OFFSET_X = 8;
constexpr int LABEL_OFFSET_Y = -8;

// Info text parameters
constexpr double INFO_FONT_SCALE = 0.4;
constexpr int INFO_THICKNESS = 1;
constexpr int INFO_TEXT_Y_1 = 25;
constexpr int INFO_TEXT_Y_2 = 55;
constexpr int INFO_TEXT_X = 10;
constexpr int INFO_MARGIN_X = 7;
constexpr int INFO_PADDING = 3;

// Key codes
constexpr int KEY_ESC = 27;
constexpr int KEY_HARRIS = 'h';
constexpr int KEY_SHITOMASI = 's';
constexpr int KEY_RESET = 'r';
constexpr int WAITKEY_DELAY = 50;
}

/**
 * @brief Application state for interactive corner detection
 */
struct ShiTomasiApp
{
  cv::Mat src;           // Original image
  cv::Mat gray;          // Grayscale image

  int max_corners = Config::DEFAULT_MAX_CORNERS;
  double quality_level = Config::DEFAULT_QUALITY_LEVEL;
  int min_distance = Config::DEFAULT_MIN_DISTANCE;
  int block_size = Config::DEFAULT_BLOCK_SIZE;
  bool use_harris = false;
  double k = Config::DEFAULT_K;

  const char * window_name = "Shi-Tomasi Corner Detection";
  const char * trackbar_corners = "Max Corners";
  const char * trackbar_quality = "Quality Level (x100)";
  const char * trackbar_distance = "Min Distance";
  const char * trackbar_block = "Block Size";
};

ShiTomasiApp app;

/**
 * @brief Detect and draw corners using current parameters
 */
void detectAndDrawCorners()
{
  // ========================================
  // Detect Corners
  // ========================================
  std::vector<cv::Point2f> corners;

  // cv::goodFeaturesToTrack: Shi-Tomasi corner detector
  // Parameters:
  //   - image: grayscale input
  //   - max_corners: maximum number to find (0 = unlimited)
  //   - quality_level: minimum quality (percentage of best corner, 0.0-1.0)
  //   - min_distance: minimum Euclidean distance between corners
  //   - mask: optional ROI
  //   - block_size: size of averaging neighborhood for gradients
  //   - use_harris_detector: false = Shi-Tomasi, true = Harris
  //   - k: Harris free parameter (only used if use_harris_detector=true)
  cv::goodFeaturesToTrack(
    app.gray,
    corners,
    app.max_corners,
    app.quality_level,
    app.min_distance,
    cv::Mat(),
    app.block_size,
    app.use_harris,
    app.k
  );

  // ========================================
  // Visualize Results
  // ========================================
  cv::Mat display = app.src.clone();

  // Draw each detected corner
  for (size_t i = 0; i < corners.size(); ++i) {
    const cv::Point2f pt = corners[i];

    // Draw cross marker
    cv::drawMarker(display, pt, cv::Scalar(0, 255, 0),
                   cv::MARKER_CROSS, Config::MARKER_SIZE,
                   Config::MARKER_THICKNESS, cv::LINE_AA);

    // Draw circle
    cv::circle(display, pt, Config::CIRCLE_RADIUS, cv::Scalar(0, 0, 255),
              Config::CIRCLE_THICKNESS, cv::LINE_AA);

    // Draw corner number (limited to avoid clutter)
    if (i < Config::MAX_LABELED_CORNERS) {
      const std::string label = std::to_string(i + 1);
      cv::putText(display, label,
                 pt + cv::Point2f(Config::LABEL_OFFSET_X, Config::LABEL_OFFSET_Y),
                 cv::FONT_HERSHEY_SIMPLEX, Config::LABEL_FONT_SCALE,
                 cv::Scalar(0, 10, 255), Config::LABEL_THICKNESS);
    }
  }

  // ========================================
  // Display Statistics
  // ========================================
  const std::string info = "Corners: " + std::to_string(corners.size()) + "/" +
    std::to_string(app.max_corners);
  const std::string method = app.use_harris ? "Harris" : "Shi-Tomasi";
  const std::string method_info = "Method: " + method;

  // Draw black background rectangles for text readability
  int baseline = 0;
  const cv::Size text_size_1 = cv::getTextSize(info, cv::FONT_HERSHEY_SIMPLEX,
                                               Config::INFO_FONT_SCALE,
                                               Config::INFO_THICKNESS, &baseline);
  const cv::Size text_size_2 = cv::getTextSize(method_info, cv::FONT_HERSHEY_SIMPLEX,
                                               Config::INFO_FONT_SCALE,
                                               Config::INFO_THICKNESS, &baseline);

  cv::rectangle(display,
               cv::Point(Config::INFO_MARGIN_X,
                        Config::INFO_TEXT_Y_1 - text_size_1.height - 2),
               cv::Point(Config::INFO_TEXT_X + text_size_1.width + Config::INFO_PADDING,
                        Config::INFO_TEXT_Y_1 + baseline + 2),
               cv::Scalar(0, 0, 0), cv::FILLED);

  cv::rectangle(display,
               cv::Point(Config::INFO_MARGIN_X,
                        Config::INFO_TEXT_Y_2 - text_size_2.height - 2),
               cv::Point(Config::INFO_TEXT_X + text_size_2.width + Config::INFO_PADDING,
                        Config::INFO_TEXT_Y_2 + baseline + 2),
               cv::Scalar(0, 0, 0), cv::FILLED);

  // Draw text over black background
  cv::putText(display, info, cv::Point(Config::INFO_TEXT_X, Config::INFO_TEXT_Y_1),
             cv::FONT_HERSHEY_SIMPLEX, Config::INFO_FONT_SCALE,
             cv::Scalar(0, 255, 0), Config::LABEL_THICKNESS);

  cv::putText(display, method_info, cv::Point(Config::INFO_TEXT_X, Config::INFO_TEXT_Y_2),
             cv::FONT_HERSHEY_SIMPLEX, Config::INFO_FONT_SCALE,
             cv::Scalar(0, 255, 0), Config::LABEL_THICKNESS);

  // ========================================
  // Display and Report
  // ========================================
  cv::imshow(app.window_name, display);

  // Print status to console
  std::cout << "\r" << method << " - Detected " << corners.size()
            << " corners (quality=" << app.quality_level
            << ", distance=" << app.min_distance
            << ", block=" << app.block_size << ")     " << std::flush;
}

/**
 * @brief Trackbar callback for max corners
 */
void onTrackbarCorners(int val, void *)
{
  app.max_corners = val;
  detectAndDrawCorners();
}

/**
 * @brief Trackbar callback for min distance
 */
void onTrackbarDistance(int val, void *)
{
  app.min_distance = val;
  detectAndDrawCorners();
}

/**
 * @brief Display help information and algorithm description
 */
void showHelp()
{
  std::cout << "========================================" << std::endl;
  std::cout << "Shi-Tomasi Corner Detection Demo" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Controls:" << std::endl;
  std::cout << "  Trackbars - Adjust detection parameters" << std::endl;
  std::cout << "  h/s       - Toggle Harris/Shi-Tomasi" << std::endl;
  std::cout << "  r         - Reset to defaults" << std::endl;
  std::cout << "  ESC       - Exit" << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Algorithm Steps:" << std::endl;
  std::cout << "  1. Compute Gaussian derivatives (Ix, Iy)" << std::endl;
  std::cout << "  2. Structure tensor M = G*[Ix², IxIy; IxIy, Iy²]" << std::endl;
  std::cout << "  3. Shi-Tomasi: R = min(λ₁, λ₂)" << std::endl;
  std::cout << "  4. Harris: R = det(M) - k·trace(M)²" << std::endl;
  std::cout << "  5. Non-maximum suppression" << std::endl;
  std::cout << "========================================\n" << std::endl;
}

int main(int argc, char ** argv)
{
  // ========================================
  // Load and Prepare Image
  // ========================================
  const std::string filename = argc >= 2 ? argv[1] : "checkerboard.png";

  app.src = cv::imread(cv::samples::findFile(filename), cv::IMREAD_COLOR);

  if (app.src.empty()) {
    std::cerr << "Error: Could not load image!" << std::endl;
    std::cerr << "Path: " << filename << std::endl;
    std::cerr << "Usage: " << argv[0] << " [image_path]" << std::endl;
    return EXIT_FAILURE;
  }

  cv::cvtColor(app.src, app.gray, cv::COLOR_BGR2GRAY);
  std::cout << "Image loaded: " << app.src.cols << "x" << app.src.rows
            << " pixels\n" << std::endl;

  showHelp();

  // ========================================
  // Create Interactive GUI
  // ========================================
  cv::namedWindow(app.window_name, cv::WINDOW_AUTOSIZE);

  // Create trackbars for parameter adjustment
  cv::createTrackbar(app.trackbar_corners, app.window_name,
                     nullptr, Config::MAX_CORNERS_LIMIT, onTrackbarCorners);
  cv::setTrackbarPos(app.trackbar_corners, app.window_name, app.max_corners);

  cv::createTrackbar(app.trackbar_quality, app.window_name,
                     nullptr, Config::MAX_QUALITY_SCALED,
    [](int val, void *) {
      app.quality_level = val / static_cast<double>(Config::QUALITY_SCALE_FACTOR);
      if (app.quality_level > 0) {  // Prevent invalid values
        detectAndDrawCorners();
      }
    });
  cv::setTrackbarPos(app.trackbar_quality, app.window_name,
                    static_cast<int>(app.quality_level * Config::QUALITY_SCALE_FACTOR));

  cv::createTrackbar(app.trackbar_distance, app.window_name,
                     nullptr, Config::MAX_DISTANCE, onTrackbarDistance);
  cv::setTrackbarPos(app.trackbar_distance, app.window_name, app.min_distance);

  cv::createTrackbar(app.trackbar_block, app.window_name,
                     nullptr, Config::MAX_BLOCK_SIZE,
    [](int val, void *) {
      // Block size must be odd and >= 1
      if (val < Config::MIN_BLOCK_SIZE) {
        val = Config::MIN_BLOCK_SIZE;
      }
      if (val % 2 == 0) {
        val += 1;
      }
      app.block_size = val;
      cv::setTrackbarPos(app.trackbar_block, app.window_name, val);
      detectAndDrawCorners();
    });
  cv::setTrackbarPos(app.trackbar_block, app.window_name, app.block_size);

  // ========================================
  // Execute Initial Detection
  // ========================================
  detectAndDrawCorners();

  // ========================================
  // Main Event Loop
  // ========================================
  while (true) {
    const int key = cv::waitKey(Config::WAITKEY_DELAY);

    if (key == Config::KEY_ESC) {
      break;
    }

    switch (key) {
      case Config::KEY_HARRIS:
      case Config::KEY_SHITOMASI:
        // Toggle between Harris and Shi-Tomasi
        app.use_harris = !app.use_harris;
        std::cout << "\nSwitched to " << (app.use_harris ? "Harris" : "Shi-Tomasi")
                  << " detector" << std::endl;
        detectAndDrawCorners();
        break;

      case Config::KEY_RESET:
        // Reset to default parameters
        app.max_corners = Config::DEFAULT_MAX_CORNERS;
        app.quality_level = Config::DEFAULT_QUALITY_LEVEL;
        app.min_distance = Config::DEFAULT_MIN_DISTANCE;
        app.block_size = Config::DEFAULT_BLOCK_SIZE;

        cv::setTrackbarPos(app.trackbar_corners, app.window_name, app.max_corners);
        cv::setTrackbarPos(app.trackbar_quality, app.window_name,
                          static_cast<int>(app.quality_level * Config::QUALITY_SCALE_FACTOR));
        cv::setTrackbarPos(app.trackbar_distance, app.window_name, app.min_distance);
        cv::setTrackbarPos(app.trackbar_block, app.window_name, app.block_size);

        std::cout << "\nReset to default parameters" << std::endl;
        detectAndDrawCorners();
        break;
    }
  }

  // ========================================
  // Cleanup and Exit
  // ========================================
  std::cout << "\n" << std::endl;
  cv::destroyAllWindows();

  return EXIT_SUCCESS;
}
