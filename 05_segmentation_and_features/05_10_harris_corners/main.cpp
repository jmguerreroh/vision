/**
 * @file main.cpp
 * @brief Harris corner detector with interactive threshold control
 * @author José Miguel Guerrero Hernández
 * @note This example demonstrates Harris corner detection algorithm with
 *       a trackbar to adjust the corner response threshold in real-time
 * @see https://docs.opencv.org/3.4/d4/d7d/tutorial_harris_detector.html
 */

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>

namespace Config
{
// Harris corner detection parameters
constexpr int BLOCK_SIZE = 2;             // Neighborhood size for derivative calculation
constexpr int APERTURE_SIZE = 3;          // Sobel operator aperture size (3, 5, or 7)
constexpr double K_VALUE = 0.04;          // Harris free parameter (typical: 0.04-0.06)

// Visualization parameters
constexpr int CORNER_RADIUS_RESPONSE = 4;
constexpr int CORNER_RADIUS_ORIGINAL = 5;
constexpr int CORNER_CENTER_RADIUS = 2;
constexpr int CIRCLE_THICKNESS = 2;
constexpr double TEXT_FONT_SCALE = 0.7;
constexpr int TEXT_THICKNESS = 2;
constexpr int TEXT_Y_OFFSET = 25;
constexpr int TEXT_X_MARGIN = 10;

// Trackbar parameters
constexpr int MAX_THRESHOLD = 255;
constexpr int DEFAULT_THRESHOLD = 200;
}

/**
 * @brief Structure to hold image data for trackbar callback
 */
struct HarrisData
{
  cv::Mat src;
  cv::Mat src_gray;
};

/**
 * @brief Callback function for Harris corner detection with adjustable threshold
 * @param thresh Threshold value from trackbar
 * @param userdata Pointer to HarrisData structure
 *
 * Harris corner detector algorithm:
 * 1. Compute image derivatives (Sobel)
 * 2. Construct structure tensor M from products of derivatives
 * 3. Compute corner response: R = det(M) - k·trace(M)²
 * 4. Threshold response to identify corners
 */
void cornerHarris_callback(int thresh, void * userdata)
{
  HarrisData * data = static_cast<HarrisData *>(userdata);

  // ========================================
  // Compute Corner Response
  // ========================================
  // cv::cornerHarris computes Harris corner response for each pixel
  // Output is a float matrix where higher values indicate stronger corners
  cv::Mat corner_response;
  cv::cornerHarris(data->src_gray, corner_response,
                   Config::BLOCK_SIZE, Config::APERTURE_SIZE, Config::K_VALUE);

  // ========================================
  // Normalize and Prepare for Visualization
  // ========================================
  // Normalize corner response to 0-255 range
  cv::Mat response_norm;
  cv::normalize(corner_response, response_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1);

  // Convert to 8-bit for display
  cv::Mat response_scaled;
  cv::convertScaleAbs(response_norm, response_scaled);

  // ========================================
  // Detect and Visualize Corners
  // ========================================
  cv::Mat corners_on_response = response_scaled.clone();
  cv::Mat corners_on_original = data->src.clone();

  int corner_count = 0;

  // Iterate through response map and mark corners above threshold
  for (int i = 0; i < response_norm.rows; ++i) {
    for (int j = 0; j < response_norm.cols; ++j) {
      if (static_cast<int>(response_norm.at<float>(i, j)) > thresh) {
        const cv::Point corner(j, i);

        // Draw on response map (black circles)
        cv::circle(corners_on_response, corner,
                  Config::CORNER_RADIUS_RESPONSE, cv::Scalar(0),
                  Config::CIRCLE_THICKNESS);

        // Draw on original image (green circles with red center)
        cv::circle(corners_on_original, corner,
                  Config::CORNER_RADIUS_ORIGINAL, cv::Scalar(0, 255, 0),
                  Config::CIRCLE_THICKNESS);
        cv::circle(corners_on_original, corner,
                  Config::CORNER_CENTER_RADIUS, cv::Scalar(0, 0, 255),
                  cv::FILLED);

        ++corner_count;
      }
    }
  }

  // ========================================
  // Display Information and Results
  // ========================================
  const std::string info = "Corners: " + std::to_string(corner_count) +
    " | Threshold: " + std::to_string(thresh);
  cv::putText(corners_on_original, info,
              cv::Point(Config::TEXT_X_MARGIN, Config::TEXT_Y_OFFSET),
              cv::FONT_HERSHEY_SIMPLEX, Config::TEXT_FONT_SCALE,
              cv::Scalar(0, 0, 255), Config::TEXT_THICKNESS);

  // Show results
  cv::imshow("Corner Response Map", response_scaled);
  cv::imshow("Detected Corners (on Response)", corners_on_response);
  cv::imshow("Detected Corners (on Original)", corners_on_original);
}

int main(int argc, char ** argv)
{
  // ========================================
  // Load Input Image
  // ========================================
  const std::string image_path = argc >= 2 ? argv[1] : "building.jpg";
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
  // Convert to grayscale (Harris detector requires single-channel image)
  cv::Mat src_gray;
  cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

  // Prepare data for callback
  HarrisData harris_data{src, src_gray};

  // ========================================
  // Interactive GUI Setup
  // ========================================
  const std::string control_window = "Harris Corner Detector - Controls";
  const std::string trackbar_name = "Threshold";

  cv::namedWindow(control_window);

  // Create trackbar for threshold adjustment
  // Moving the slider automatically calls cornerHarris_callback
  cv::createTrackbar(trackbar_name, control_window, nullptr,
                     Config::MAX_THRESHOLD, cornerHarris_callback, &harris_data);
  cv::setTrackbarPos(trackbar_name, control_window, Config::DEFAULT_THRESHOLD);

  // Show instructions in control window
  cv::Mat control_display = src.clone();
  cv::putText(control_display, "Adjust threshold to detect corners",
              cv::Point(Config::TEXT_X_MARGIN, 30),
              cv::FONT_HERSHEY_SIMPLEX, Config::TEXT_FONT_SCALE,
              cv::Scalar(0, 0, 255), Config::TEXT_THICKNESS);
  cv::imshow(control_window, control_display);

  // ========================================
  // Execute Initial Detection
  // ========================================
  cornerHarris_callback(Config::DEFAULT_THRESHOLD, &harris_data);

  // ========================================
  // Display Instructions and Wait
  // ========================================
  std::cout << "========================================" << std::endl;
  std::cout << "Harris Corner Detector" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Parameters:" << std::endl;
  std::cout << "  Block size: " << Config::BLOCK_SIZE << std::endl;
  std::cout << "  Aperture size: " << Config::APERTURE_SIZE << std::endl;
  std::cout << "  K value: " << Config::K_VALUE << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Instructions:" << std::endl;
  std::cout << "  Adjust the threshold slider to control" << std::endl;
  std::cout << "  corner sensitivity:" << std::endl;
  std::cout << "  - Higher threshold → fewer, stronger corners" << std::endl;
  std::cout << "  - Lower threshold → more corners (may include noise)" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Press any key to exit..." << std::endl;

  cv::waitKey(0);
  cv::destroyAllWindows();

  return EXIT_SUCCESS;
}
