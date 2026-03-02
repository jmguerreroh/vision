/**
 * @file main.cpp
 * @brief ORB Feature Detector and Descriptor
 * @author José Miguel Guerrero Hernández
 * @note This example demonstrates ORB (Oriented FAST and Rotated BRIEF),
 *       a fast, rotation-invariant feature detector and binary descriptor.
 *
 * ORB (Oriented FAST and Rotated BRIEF):
 *   - FAST: Features from Accelerated Segment Test (keypoint detector)
 *   - BRIEF: Binary Robust Independent Elementary Features (descriptor)
 *   - Oriented: Computes orientation for rotation invariance
 *   - Free to use (patent-free, unlike SIFT/SURF)
 *
 * Why ORB?
 *   - Very fast (real-time applications)
 *   - Rotation invariant (handles rotated images)
 *   - Scale invariant (handles size changes)
 *   - Binary descriptor (fast matching with Hamming distance)
 *   - Good performance/speed tradeoff
 *
 * Use Cases:
 *   - Image matching and registration (RANSAC)
 *   - Object detection and tracking
 *   - SLAM (Simultaneous Localization and Mapping)
 *   - Panorama stitching
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

namespace Config
{
// Default ORB parameters
constexpr int DEFAULT_N_FEATURES = 500;
constexpr float DEFAULT_SCALE_FACTOR = 1.2f;
constexpr int DEFAULT_N_LEVELS = 8;
constexpr int DEFAULT_EDGE_THRESHOLD = 31;
constexpr int DEFAULT_FIRST_LEVEL = 0;
constexpr int DEFAULT_WTA_K = 2;
constexpr int DEFAULT_PATCH_SIZE = 31;
constexpr int DEFAULT_FAST_THRESHOLD = 20;

// Trackbar limits
constexpr int MAX_FEATURES_LIMIT = 2000;
constexpr int MAX_LEVELS_LIMIT = 16;
constexpr int MIN_FEATURES = 10;
constexpr int MIN_LEVELS = 1;

// Visualization parameters
constexpr int OVERLAY_X1 = 5;
constexpr int OVERLAY_Y1 = 5;
constexpr int OVERLAY_X2 = 400;
constexpr int OVERLAY_Y2 = 200;
constexpr double OVERLAY_ALPHA = 0.7;
constexpr double OVERLAY_BETA = 0.3;
constexpr int TEXT_START_Y = 25;
constexpr int TEXT_LINE_HEIGHT = 20;
constexpr int TEXT_EXTRA_SPACING = 5;
constexpr int TEXT_REDUCED_SPACING = -5;
constexpr double TEXT_FONT_SCALE_TITLE = 0.6;
constexpr double TEXT_FONT_SCALE_MAIN = 0.5;
constexpr double TEXT_FONT_SCALE_SMALL = 0.4;
constexpr int TEXT_THICKNESS_TITLE = 2;
constexpr int TEXT_THICKNESS_MAIN = 1;
constexpr int TEXT_MARGIN_X = 10;

// Analysis parameters
constexpr int MAX_LEVELS_TO_DISPLAY = 4;
constexpr int MAX_KEYPOINTS_INFO = 5;
constexpr int DESCRIPTOR_BYTES_TO_SHOW = 8;

// Key codes
constexpr int KEY_ESC = 27;
constexpr int KEY_RESET = 'r';
constexpr int KEY_INFO = 'i';
constexpr int WAITKEY_DELAY = 50;
}

/**
 * @brief Application state for interactive ORB detection
 */
struct ORBApp
{
  cv::Mat src;      // Original image
  cv::Mat gray;     // Grayscale image

  int n_features = Config::DEFAULT_N_FEATURES;
  float scale_factor = Config::DEFAULT_SCALE_FACTOR;
  int n_levels = Config::DEFAULT_N_LEVELS;
  int edge_threshold = Config::DEFAULT_EDGE_THRESHOLD;
  int patch_size = Config::DEFAULT_PATCH_SIZE;

  const char * window_name = "ORB Features";
  const char * trackbar_features = "Max Features";
  const char * trackbar_levels = "Pyramid Levels";
};

ORBApp app;

/**
 * @brief Detect and visualize ORB features
 */
void detectAndDrawORB()
{
  // ========================================
  // Create ORB Detector
  // ========================================
  // cv::ORB::create() creates an ORB feature detector/descriptor
  // Parameters:
  //   - n_features: Maximum number of features to retain (best N features)
  //   - scale_factor: Pyramid decimation ratio (>1, typically 1.2)
  //   - n_levels: Number of pyramid levels (scale invariance)
  //   - edge_threshold: Size of border ignored (should match patch_size)
  //   - first_level: Level of pyramid to start from (usually 0)
  //   - WTA_K: Number of points for descriptor (2, 3, or 4)
  //   - score_type: HARRIS_SCORE or FAST_SCORE
  //   - patch_size: Size of patch used by descriptor
  //   - fast_threshold: FAST detector threshold
  cv::Ptr<cv::ORB> orb = cv::ORB::create(
    app.n_features,
    app.scale_factor,
    app.n_levels,
    app.edge_threshold,
    Config::DEFAULT_FIRST_LEVEL,
    Config::DEFAULT_WTA_K,
    cv::ORB::HARRIS_SCORE,
    app.patch_size,
    Config::DEFAULT_FAST_THRESHOLD
  );

  // ========================================
  // Detect Keypoints and Compute Descriptors
  // ========================================
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;

  // detectAndCompute: Detect keypoints AND compute descriptors in one call
  // - Input: grayscale image, mask (empty = full image)
  // - Output: keypoints vector, descriptors matrix
  // Descriptors matrix: [N x 32] where N = number of keypoints
  //   - Each row is a 256-bit binary descriptor (32 bytes)
  //   - Binary format allows fast Hamming distance matching
  orb->detectAndCompute(app.gray, cv::Mat(), keypoints, descriptors);

  // ========================================
  // Analyze Keypoints
  // ========================================
  // Group keypoints by pyramid level
  std::vector<int> levels_count(app.n_levels, 0);
  for (const auto & kp : keypoints) {
    if (kp.octave >= 0 && kp.octave < app.n_levels) {
      levels_count[kp.octave]++;
    }
  }

  // Calculate statistics
  float avg_size = 0.0f;
  float avg_response = 0.0f;
  for (const auto & kp : keypoints) {
    avg_size += kp.size;
    avg_response += kp.response;
  }
  if (!keypoints.empty()) {
    avg_size /= static_cast<float>(keypoints.size());
    avg_response /= static_cast<float>(keypoints.size());
  }

  // ========================================
  // Visualize Results
  // ========================================
  cv::Mat output;

  // cv::drawKeypoints: Draw keypoints on image
  // Flags:
  //   - DRAW_RICH_KEYPOINTS: Draw size and orientation
  //   - DRAW_OVER_OUTIMG: Draw on existing image
  //   - DEFAULT: Just draw circles
  cv::drawKeypoints(app.src, keypoints, output,
                    cv::Scalar(0, 255, 0),
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  // ========================================
  // Display Statistics on Image
  // ========================================
  // Create semi-transparent black box for text
  cv::Mat overlay = output.clone();
  cv::rectangle(overlay,
               cv::Point(Config::OVERLAY_X1, Config::OVERLAY_Y1),
               cv::Point(Config::OVERLAY_X2, Config::OVERLAY_Y2),
               cv::Scalar(0, 0, 0), cv::FILLED);
  cv::addWeighted(overlay, Config::OVERLAY_ALPHA, output, Config::OVERLAY_BETA, 0, output);

  // Draw statistics
  int y = Config::TEXT_START_Y;
  const int dy = Config::TEXT_LINE_HEIGHT;
  const cv::Scalar text_color(0, 255, 0);

  cv::putText(output, "ORB Feature Detection",
             cv::Point(Config::TEXT_MARGIN_X, y),
             cv::FONT_HERSHEY_SIMPLEX, Config::TEXT_FONT_SCALE_TITLE,
             text_color, Config::TEXT_THICKNESS_TITLE);
  y += dy + Config::TEXT_EXTRA_SPACING;

  cv::putText(output, "Keypoints detected: " + std::to_string(keypoints.size()),
             cv::Point(Config::TEXT_MARGIN_X, y),
             cv::FONT_HERSHEY_SIMPLEX, Config::TEXT_FONT_SCALE_MAIN,
             text_color, Config::TEXT_THICKNESS_MAIN);
  y += dy;

  cv::putText(output, "Descriptor size: " +
             std::to_string(descriptors.cols) + " bytes (256 bits)",
             cv::Point(Config::TEXT_MARGIN_X, y),
             cv::FONT_HERSHEY_SIMPLEX, Config::TEXT_FONT_SCALE_MAIN,
             text_color, Config::TEXT_THICKNESS_MAIN);
  y += dy;

  cv::putText(output, "Avg keypoint size: " +
             std::to_string(static_cast<int>(avg_size)) + " pixels",
             cv::Point(Config::TEXT_MARGIN_X, y),
             cv::FONT_HERSHEY_SIMPLEX, Config::TEXT_FONT_SCALE_MAIN,
             text_color, Config::TEXT_THICKNESS_MAIN);
  y += dy;

  cv::putText(output, "Avg response: " +
             std::to_string(static_cast<int>(avg_response)),
             cv::Point(Config::TEXT_MARGIN_X, y),
             cv::FONT_HERSHEY_SIMPLEX, Config::TEXT_FONT_SCALE_MAIN,
             text_color, Config::TEXT_THICKNESS_MAIN);
  y += dy;

  cv::putText(output, "Pyramid levels: " + std::to_string(app.n_levels),
             cv::Point(Config::TEXT_MARGIN_X, y),
             cv::FONT_HERSHEY_SIMPLEX, Config::TEXT_FONT_SCALE_MAIN,
             text_color, Config::TEXT_THICKNESS_MAIN);
  y += dy;

  // Show distribution across pyramid levels
  cv::putText(output, "Distribution by level:",
             cv::Point(Config::TEXT_MARGIN_X, y),
             cv::FONT_HERSHEY_SIMPLEX, Config::TEXT_FONT_SCALE_MAIN,
             text_color, Config::TEXT_THICKNESS_MAIN);
  y += dy + Config::TEXT_REDUCED_SPACING;

  const int max_levels_display = std::min(app.n_levels, Config::MAX_LEVELS_TO_DISPLAY);
  for (int i = 0; i < max_levels_display; ++i) {
    const std::string level_text = "  L" + std::to_string(i) + ": " +
      std::to_string(levels_count[i]);
    cv::putText(output, level_text,
               cv::Point(Config::TEXT_MARGIN_X, y),
               cv::FONT_HERSHEY_SIMPLEX, Config::TEXT_FONT_SCALE_SMALL,
               text_color, Config::TEXT_THICKNESS_MAIN);
    y += dy + Config::TEXT_REDUCED_SPACING;
  }

  cv::imshow(app.window_name, output);

  // ========================================
  // Console Output
  // ========================================
  std::cout << "\r[ORB] Features: " << keypoints.size()
            << " | Levels: " << app.n_levels
            << " | Scale: " << app.scale_factor
            << "     " << std::flush;
}

/**
 * @brief Trackbar callback for parameter changes
 */
void onTrackbar(int, void *)
{
  // Get current values from trackbars
  app.n_features = cv::getTrackbarPos(app.trackbar_features, app.window_name);
  app.n_levels = cv::getTrackbarPos(app.trackbar_levels, app.window_name);

  // Ensure minimum values
  if (app.n_features < Config::MIN_FEATURES) {
    app.n_features = Config::MIN_FEATURES;
    cv::setTrackbarPos(app.trackbar_features, app.window_name, app.n_features);
  }
  if (app.n_levels < Config::MIN_LEVELS) {
    app.n_levels = Config::MIN_LEVELS;
    cv::setTrackbarPos(app.trackbar_levels, app.window_name, app.n_levels);
  }

  detectAndDrawORB();
}

/**
 * @brief Display help information and algorithm description
 */
void showHelp()
{
  std::cout << "========================================" << std::endl;
  std::cout << "ORB Feature Detector Demo" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "What is ORB?" << std::endl;
  std::cout << "  - Oriented FAST keypoint detector" << std::endl;
  std::cout << "  - Rotated BRIEF binary descriptor" << std::endl;
  std::cout << "  - Fast, rotation-invariant, patent-free" << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Descriptor Properties:" << std::endl;
  std::cout << "  - 256-bit binary vector (32 bytes)" << std::endl;
  std::cout << "  - Matching uses Hamming distance" << std::endl;
  std::cout << "  - Much faster than SIFT/SURF" << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Controls:" << std::endl;
  std::cout << "  Trackbars - Adjust detection parameters" << std::endl;
  std::cout << "  r         - Reset to defaults" << std::endl;
  std::cout << "  i         - Show descriptor info" << std::endl;
  std::cout << "  ESC       - Exit" << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Visualization:" << std::endl;
  std::cout << "  Green circles   = Keypoints" << std::endl;
  std::cout << "  Circle size     = Feature scale" << std::endl;
  std::cout << "  Line from center = Orientation" << std::endl;
  std::cout << "========================================\n" << std::endl;
}

/**
 * @brief Display detailed descriptor information for first keypoints
 */
void showDescriptorInfo()
{
  // Recompute for detailed info
  cv::Ptr<cv::ORB> orb = cv::ORB::create(app.n_features, app.scale_factor,
                                        app.n_levels, app.edge_threshold);
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  orb->detectAndCompute(app.gray, cv::Mat(), keypoints, descriptors);

  if (keypoints.empty()) {
    std::cout << "\nNo keypoints detected!" << std::endl;
    return;
  }

  std::cout << "\n========================================" << std::endl;
  std::cout << "Descriptor Information" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Total keypoints: " << keypoints.size() << std::endl;
  std::cout << "Descriptor matrix: " << descriptors.rows << " x "
            << descriptors.cols << " (CV_8U)" << std::endl;
  std::cout << "Each descriptor: 256 bits = 32 bytes\n" << std::endl;

  // Show first N keypoints
  const int n = std::min(Config::MAX_KEYPOINTS_INFO, static_cast<int>(keypoints.size()));
  for (int i = 0; i < n; ++i) {
    const auto & kp = keypoints[i];
    std::cout << "Keypoint #" << i << ":" << std::endl;
    std::cout << "  Position: (" << static_cast<int>(kp.pt.x) << ", "
              << static_cast<int>(kp.pt.y) << ")" << std::endl;
    std::cout << "  Size: " << kp.size << " px" << std::endl;
    std::cout << "  Angle: " << kp.angle << "°" << std::endl;
    std::cout << "  Response: " << kp.response << std::endl;
    std::cout << "  Octave (level): " << kp.octave << std::endl;

    // Show first bytes of descriptor in hex
    std::cout << "  Descriptor (first " << Config::DESCRIPTOR_BYTES_TO_SHOW
              << " bytes): ";
    for (int j = 0; j < Config::DESCRIPTOR_BYTES_TO_SHOW && j < descriptors.cols; ++j) {
      printf("%02x ", descriptors.at<uchar>(i, j));
    }
    std::cout << "..." << std::endl << std::endl;
  }
  std::cout << "========================================\n" << std::endl;
}

int main(int argc, char ** argv)
{
  // ========================================
  // Load and Prepare Image
  // ========================================
  const std::string filename = argc >= 2 ? argv[1] : "../../data/building.jpg";

  app.src = cv::imread(filename, cv::IMREAD_COLOR);

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
  cv::createTrackbar(app.trackbar_features, app.window_name,
                     nullptr, Config::MAX_FEATURES_LIMIT, onTrackbar);
  cv::setTrackbarPos(app.trackbar_features, app.window_name, app.n_features);

  cv::createTrackbar(app.trackbar_levels, app.window_name,
                     nullptr, Config::MAX_LEVELS_LIMIT, onTrackbar);
  cv::setTrackbarPos(app.trackbar_levels, app.window_name, app.n_levels);

  // ========================================
  // Execute Initial Detection
  // ========================================
  detectAndDrawORB();

  // ========================================
  // Main Event Loop
  // ========================================
  while (true) {
    const int key = cv::waitKey(Config::WAITKEY_DELAY);

    if (key == Config::KEY_ESC) {
      break;
    }

    switch (key) {
      case Config::KEY_RESET:
        // Reset to default parameters
        app.n_features = Config::DEFAULT_N_FEATURES;
        app.n_levels = Config::DEFAULT_N_LEVELS;

        cv::setTrackbarPos(app.trackbar_features, app.window_name, app.n_features);
        cv::setTrackbarPos(app.trackbar_levels, app.window_name, app.n_levels);

        std::cout << "\nReset to default parameters" << std::endl;
        detectAndDrawORB();
        break;

      case Config::KEY_INFO:
        // Show detailed descriptor information
        showDescriptorInfo();
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
