/**
 * @file main.cpp
 * @brief Connected Components Analysis - Automatic segmentation and labeling
 * @author José Miguel Guerrero Hernández
 * @note This example demonstrates connected component analysis using
 *       cv::connectedComponentsWithStats() to identify, label, and analyze
 *       all separate regions in a binary image automatically.
 *
 * Connected Components:
 *   - Identifies all separate regions (blobs) in binary image
 *   - Labels each region with unique integer ID
 *   - Computes statistics: area, centroid, bounding box
 *   - Supports 4-connectivity (cross) or 8-connectivity (square)
 *
 * Difference from FloodFill:
 *   - FloodFill: Interactive, fills ONE region you click
 *   - ConnectedComponents: Automatic, finds ALL regions at once
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

namespace Config
{
// Default parameters
constexpr int DEFAULT_MIN_AREA = 100;
constexpr int RNG_SEED_MULTIPLIER = 12345;

// Binary threshold parameters
constexpr int THRESHOLD_MAX_VALUE = 255;

// Connectivity options
constexpr int CONNECTIVITY_4 = 4;    // Cross (no diagonals)
constexpr int CONNECTIVITY_8 = 8;    // Square (includes diagonals)

// Visualization parameters
constexpr int CENTROID_RADIUS = 5;
constexpr int CENTROID_THICKNESS = -1;    // Filled
constexpr int BBOX_THICKNESS = 2;
constexpr double LABEL_FONT_SCALE = 0.5;
constexpr int LABEL_THICKNESS = 2;
constexpr int LABEL_OFFSET_X = 8;
constexpr int LABEL_OFFSET_Y = -8;
constexpr double AREA_FONT_SCALE = 0.4;
constexpr int AREA_THICKNESS = 1;
constexpr int AREA_OFFSET_Y = -5;

// Color normalization
constexpr int NORMALIZE_MIN = 0;
constexpr int NORMALIZE_MAX = 255;
}

/**
 * @brief Generate random color for each component
 * @param seed Seed value for random generator
 * @return Random BGR color
 */
cv::Scalar generateRandomColor(int seed)
{
  cv::RNG rng(seed);
  return cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
}

/**
 * @brief Colorize labeled components with random colors
 * @param labels Label matrix (each pixel has component ID)
 * @param num_labels Total number of labels (including background)
 * @return Color image with each component in different color
 */
cv::Mat colorizeComponents(const cv::Mat & labels, int num_labels)
{
  cv::Mat colored = cv::Mat::zeros(labels.size(), CV_8UC3);

  // Generate random color for each label (skip 0 = background)
  std::vector<cv::Vec3b> colors(num_labels);
  colors[0] = cv::Vec3b(0, 0, 0); // Background = black

  for (int i = 1; i < num_labels; ++i) {
    const cv::Scalar color = generateRandomColor(i * Config::RNG_SEED_MULTIPLIER);
    colors[i] = cv::Vec3b(static_cast<uchar>(color[0]),
                          static_cast<uchar>(color[1]),
                          static_cast<uchar>(color[2]));
  }

  // Apply colors to labeled image
  for (int y = 0; y < labels.rows; ++y) {
    for (int x = 0; x < labels.cols; ++x) {
      const int label = labels.at<int>(y, x);
      colored.at<cv::Vec3b>(y, x) = colors[label];
    }
  }

  return colored;
}

int main(int argc, char ** argv)
{
  // ========================================
  // Load and Validate Image
  // ========================================
  const std::string filename = argc >= 2 ? argv[1] : "../../data/shapes.png";
  const int min_area = argc >= 3 ? std::atoi(argv[2]) : Config::DEFAULT_MIN_AREA;

  std::cout << "========================================" << std::endl;
  std::cout << "Connected Components Analysis" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Reading image: " << filename << std::endl;
  std::cout << "Minimum area filter: " << min_area << " pixels" << std::endl;

  const cv::Mat src = cv::imread(filename, cv::IMREAD_COLOR);
  if (src.empty()) {
    std::cerr << "Error: Could not load image!" << std::endl;
    std::cerr << "Path: " << filename << std::endl;
    std::cerr << "Usage: " << argv[0] << " [image_path] [min_area]" << std::endl;
    return EXIT_FAILURE;
  }

  // ========================================
  // Convert to Binary Image
  // ========================================
  cv::Mat gray, binary;
  cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

  // Use Otsu's method for automatic threshold
  const double threshold_value = cv::threshold(gray, binary, 0,
                                              Config::THRESHOLD_MAX_VALUE,
                                              cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

  std::cout << "\nOtsu threshold value: " << threshold_value << std::endl;

  // ========================================
  // Connected Components Analysis
  // ========================================
  cv::Mat labels, stats, centroids;

  // cv::connectedComponentsWithStats:
  // - labels: Output matrix where each pixel value = component ID (0 = background)
  // - stats: Statistics matrix [num_labels x 5]:
  //     CC_STAT_LEFT   (0): Leftmost x coordinate
  //     CC_STAT_TOP    (1): Topmost y coordinate
  //     CC_STAT_WIDTH  (2): Bounding box width
  //     CC_STAT_HEIGHT (3): Bounding box height
  //     CC_STAT_AREA   (4): Number of pixels in component
  // - centroids: Matrix [num_labels x 2] with (cx, cy) coordinates
  // - connectivity: 4 or 8
  const int num_labels = cv::connectedComponentsWithStats(
    binary, labels, stats, centroids, Config::CONNECTIVITY_8, CV_32S);

  std::cout << "\nFound " << (num_labels - 1) << " components (excluding background)"
            << std::endl;

  // ========================================
  // Filter and Visualize Components
  // ========================================
  cv::Mat colored = colorizeComponents(labels, num_labels);
  cv::Mat overlay = src.clone();
  cv::Mat filtered = src.clone();

  int valid_components = 0;

  std::cout << "\n========================================" << std::endl;
  std::cout << "Component Statistics:" << std::endl;
  std::cout << "========================================" << std::endl;

  // Process each component (skip 0 = background)
  for (int i = 1; i < num_labels; ++i) {
    // Extract statistics
    const int area = stats.at<int>(i, cv::CC_STAT_AREA);
    const int left = stats.at<int>(i, cv::CC_STAT_LEFT);
    const int top = stats.at<int>(i, cv::CC_STAT_TOP);
    const int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
    const int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
    const double cx = centroids.at<double>(i, 0);
    const double cy = centroids.at<double>(i, 1);

    // Display statistics
    std::cout << "\nComponent #" << i << ":" << std::endl;
    std::cout << "  Area: " << area << " pixels" << std::endl;
    std::cout << "  Centroid: (" << static_cast<int>(cx) << ", "
              << static_cast<int>(cy) << ")" << std::endl;
    std::cout << "  Bounding Box: [" << left << ", " << top << ", "
              << width << " x " << height << "]" << std::endl;

    // Skip small components (noise)
    if (area < min_area) {
      std::cout << "  Status: FILTERED (too small)" << std::endl;
      continue;
    }

    ++valid_components;
    std::cout << "  Status: VALID" << std::endl;

    // Random color for this component
    const cv::Scalar color = generateRandomColor(i * Config::RNG_SEED_MULTIPLIER);

    // Draw bounding box on overlay
    const cv::Point bbox_tl(left, top);
    const cv::Point bbox_br(left + width, top + height);
    cv::rectangle(overlay, bbox_tl, bbox_br, color, Config::BBOX_THICKNESS);
    cv::rectangle(filtered, bbox_tl, bbox_br, color, Config::BBOX_THICKNESS);

    // Draw centroid
    const cv::Point centroid_pt(static_cast<int>(cx), static_cast<int>(cy));
    const cv::Scalar centroid_color_red(0, 0, 255);
    const cv::Scalar centroid_color_white(255, 255, 255);
    const cv::Scalar label_color_yellow(255, 255, 0);

    cv::circle(overlay, centroid_pt, Config::CENTROID_RADIUS,
              centroid_color_red, Config::CENTROID_THICKNESS);
    cv::circle(colored, centroid_pt, Config::CENTROID_RADIUS,
              centroid_color_white, Config::CENTROID_THICKNESS);
    cv::circle(filtered, centroid_pt, Config::CENTROID_RADIUS,
              centroid_color_red, Config::CENTROID_THICKNESS);

    // Add label near centroid
    const std::string label_text = "C" + std::to_string(i);
    const cv::Point label_offset(Config::LABEL_OFFSET_X, Config::LABEL_OFFSET_Y);

    cv::putText(overlay, label_text, centroid_pt + label_offset,
               cv::FONT_HERSHEY_SIMPLEX, Config::LABEL_FONT_SCALE,
               label_color_yellow, Config::LABEL_THICKNESS);
    cv::putText(colored, label_text, centroid_pt + label_offset,
               cv::FONT_HERSHEY_SIMPLEX, Config::LABEL_FONT_SCALE,
               centroid_color_white, Config::LABEL_THICKNESS);
    cv::putText(filtered, label_text, centroid_pt + label_offset,
               cv::FONT_HERSHEY_SIMPLEX, Config::LABEL_FONT_SCALE,
               label_color_yellow, Config::LABEL_THICKNESS);

    // Display area on bounding box
    const std::string area_label = std::to_string(area) + "px";
    cv::putText(overlay, area_label, cv::Point(left, top + Config::AREA_OFFSET_Y),
               cv::FONT_HERSHEY_SIMPLEX, Config::AREA_FONT_SCALE,
               color, Config::AREA_THICKNESS);
  }

  // ========================================
  // Create Label Visualization
  // ========================================
  // Normalize labels to 0-255 for visualization
  cv::Mat labels_normalized;
  cv::normalize(labels, labels_normalized, Config::NORMALIZE_MIN,
               Config::NORMALIZE_MAX, cv::NORM_MINMAX, CV_8U);
  cv::Mat labels_color;
  cv::applyColorMap(labels_normalized, labels_color, cv::COLORMAP_JET);
  // Set background to black
  labels_color.setTo(cv::Scalar(0, 0, 0), labels == 0);

  // ========================================
  // Summary Statistics
  // ========================================
  const int total_components = num_labels - 1;
  const int filtered_components = total_components - valid_components;

  std::cout << "\n========================================" << std::endl;
  std::cout << "Summary:" << std::endl;
  std::cout << "  Total components: " << total_components << std::endl;
  std::cout << "  Valid components (area >= " << min_area << "): "
            << valid_components << std::endl;
  std::cout << "  Filtered components: " << filtered_components << std::endl;
  std::cout << "========================================" << std::endl;

  // ========================================
  // Display Results
  // ========================================
  cv::imshow("1. Original Image", src);
  cv::imshow("2. Grayscale", gray);
  cv::imshow("3. Binary (Otsu)", binary);
  cv::imshow("4. Labeled Components", labels_color);
  cv::imshow("5. Colored Components", colored);
  cv::imshow("6. Bounding Boxes + Centroids", overlay);

  std::cout << "\nVisualization:" << std::endl;
  std::cout << "  Window 4: Label IDs (jet colormap)" << std::endl;
  std::cout << "  Window 5: Random colors per component" << std::endl;
  std::cout << "  Window 6: Bounding boxes + centroids + areas" << std::endl;
  std::cout << "  Red circles = Centroids" << std::endl;
  std::cout << "  Colored rectangles = Bounding boxes" << std::endl;
  std::cout << "\n========================================" << std::endl;
  std::cout << "Connectivity: " << Config::CONNECTIVITY_8
            << " (includes diagonals)" << std::endl;
  std::cout << "Press any key to exit..." << std::endl;
  std::cout << "========================================" << std::endl;

  cv::waitKey();

  // ========================================
  // Cleanup and Exit
  // ========================================
  cv::destroyAllWindows();

  return EXIT_SUCCESS;
}
