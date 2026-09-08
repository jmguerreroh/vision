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

#include <string>
#include <algorithm>
#include <cstdlib>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>

namespace
{
// The images this example works on are about 1400 px on the long side, and
// several windows at that size do not fit on a normal screen. The processing
// always runs at full resolution: only the copy sent to the screen is reduced,
// with INTER_AREA, which is the interpolation meant for shrinking
constexpr int MAX_DISPLAY_SIDE = 800;

// Returns the copy that goes to the screen, already reduced. Whatever is drawn
// on the result keeps its size in screen pixels, so labels are written here and
// not on the full-resolution frame: drawn before the reduction they shrink with
// it and stop being readable
cv::Mat fitToScreen(const cv::Mat & image)
{
  const int side = std::max(image.cols, image.rows);
  if (side <= MAX_DISPLAY_SIDE || image.empty()) {
    return image.clone();
  }
  const double factor = static_cast<double>(MAX_DISPLAY_SIDE) / side;
  cv::Mat reduced;
  cv::resize(image, reduced, cv::Size(), factor, factor, cv::INTER_AREA);
  return reduced;
}

void showFit(const std::string & window, const cv::Mat & image)
{
  cv::imshow(window, fitToScreen(image));
}
}  // namespace

int main(int argc, char ** argv)
{
  // Load image from argument or use default
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/smarties.png | Input file}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  const std::string filename = parser.get<std::string>("@input");

  // Without this, a malformed value (--frames=xyz) is reported by the parser
  // but the example carries on with the default, which is the hardest kind
  // of failure to diagnose. Note it validates values, not option names:
  // cv::CommandLineParser ignores an unknown option without complaining
  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }
  cv::Mat src = cv::imread(cv::samples::findFile(filename, false), cv::IMREAD_COLOR);

  if (src.empty()) {
    std::cerr << "Error: Could not open or find the image!" << std::endl;
    std::cerr << "Path: " << filename << std::endl;
    std::cerr << "Usage: " << argv[0] << " <Input image>" << std::endl;
    return EXIT_FAILURE;
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
  // The radius bounds are expressed as a fraction of the image height, like
  // minDist above, instead of in absolute pixels. A fixed 1..30 range only
  // works for one resolution: on data/smarties.png, whose candies have a
  // radius of about 66 px, it finds nothing at all
  cv::HoughCircles(
    blurred, circles, cv::HOUGH_GRADIENT, 1,
    blurred.rows / 16,
    100, 30,
    blurred.rows / 40, blurred.rows / 12
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
  showFit("1. Original", src);
  showFit("2. Grayscale", gray);
  showFit("3. Blurred (Median)", blurred);
  showFit("4. Detected Circles", result);

  // Wait for user input and exit
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
