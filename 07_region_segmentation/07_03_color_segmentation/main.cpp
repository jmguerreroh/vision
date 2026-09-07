/**
 * @file main.cpp
 * @brief Color-based segmentation with cv::inRange in HSV space
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - Why HSV (seen in 02_02) is the right space for color segmentation
 * - Selecting pixels inside a color range with cv::inRange()
 * - Handling the red hue wrap-around (red spans both ends of the H axis)
 * - Extracting the selected pixels with bitwise_and (seen in 03_03)
 *
 * Why HSV and not BGR? In BGR, the "same" color under different lighting
 * produces very different (B,G,R) triplets, so a box in BGR space captures
 * lighting as much as color. In HSV the color identity lives almost entirely
 * in the H (hue) channel, while lighting changes move mostly V (and some S).
 * A wide S,V range with a narrow H range segments a color robustly.
 */

#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace Config
{
// Saturation/Value limits shared by all colors: excludes near-gray pixels
// (low S) and very dark pixels (low V), where hue is unreliable noise
constexpr int MIN_SATURATION = 80;
constexpr int MIN_VALUE = 60;
}

/**
 * @brief One color to segment: a name plus its hue interval
 *
 * OpenCV stores hue in [0, 179] (the 0-360 degree circle divided by 2 so it
 * fits in 8 bits). Approximate centers: red=0/180, yellow=30, green=60,
 * cyan=90, blue=120, magenta=150.
 */
struct ColorRange
{
  std::string name;
  int hue_min;
  int hue_max;
};

/**
 * @brief Builds the binary mask of pixels whose hue lies in [hue_min, hue_max]
 * @param hsv Image already converted to HSV
 * @param range Hue interval to select
 * @return CV_8U mask: 255 where the pixel belongs to the color, 0 elsewhere
 *
 * cv::inRange(src, lower, upper, mask) tests every pixel channel-wise:
 * the mask is 255 only where ALL channels fall inside [lower, upper].
 */
cv::Mat maskForColor(const cv::Mat & hsv, const ColorRange & range)
{
  cv::Mat mask;
  cv::inRange(hsv,
              cv::Scalar(range.hue_min, Config::MIN_SATURATION, Config::MIN_VALUE),
              cv::Scalar(range.hue_max, 255, 255),
              mask);
  return mask;
}

int main(int argc, char ** argv)
{
  // Load image
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/smarties.png | Input file}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  const std::string image_path = parser.get<std::string>("@input");
  const cv::Mat src = cv::imread(cv::samples::findFile(image_path, false), cv::IMREAD_COLOR);

  if (src.empty()) {
    std::cerr << "Error: Could not load image '" << image_path << "'" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "=== HSV Color Segmentation ===" << std::endl;
  std::cout << "Image: " << src.cols << "x" << src.rows << " pixels" << std::endl;

  // Convert BGR -> HSV (exactly as in 02_02)
  cv::Mat hsv;
  cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

  cv::imshow("Original", src);

  // ========================================
  // Segment three colors with a single hue interval each
  // ========================================
  const std::vector<ColorRange> colors = {
    {"Yellow", 20, 35},
    {"Green", 40, 80},
    {"Blue", 100, 130},
  };

  const double total_pixels = static_cast<double>(src.rows) * src.cols;

  for (const ColorRange & color : colors) {
    const cv::Mat mask = maskForColor(hsv, color);

    // Keep only the selected pixels: AND of the image with itself, limited
    // by the mask (the bitwise operations were introduced in 03_03)
    cv::Mat segmented;
    cv::bitwise_and(src, src, segmented, mask);

    const double percent = 100.0 * cv::countNonZero(mask) / total_pixels;
    std::cout << "  " << color.name << ": hue [" << color.hue_min << ", "
              << color.hue_max << "] -> " << percent << "% of the image" << std::endl;

    cv::imshow("Mask " + color.name, mask);
    cv::imshow("Segmented " + color.name, segmented);
  }

  // ========================================
  // Special case: red wraps around the hue circle
  // ========================================
  //
  // Hue is an ANGLE: red sits at 0 degrees, so "reddish" covers both the
  // start (H in [0, 10]) and the end (H in [170, 179]) of the axis. A single
  // inRange cannot express that interval -- build two masks and OR them.
  cv::Mat red_low = maskForColor(hsv, {"RedLow", 0, 10});
  cv::Mat red_high = maskForColor(hsv, {"RedHigh", 170, 179});

  cv::Mat red_mask;
  cv::bitwise_or(red_low, red_high, red_mask);

  cv::Mat red_segmented;
  cv::bitwise_and(src, src, red_segmented, red_mask);

  const double red_percent = 100.0 * cv::countNonZero(red_mask) / total_pixels;
  std::cout << "  Red: hue [0, 10] U [170, 179] (wrap-around!) -> "
            << red_percent << "% of the image" << std::endl;

  cv::imshow("Mask Red (two ranges OR-ed)", red_mask);
  cv::imshow("Segmented Red", red_segmented);

  std::cout << "\nNote: the masks still contain small speckles. Chapter 6" << std::endl;
  std::cout << "introduces the morphological operations (opening/closing)" << std::endl;
  std::cout << "that are the standard tool for cleaning them up." << std::endl;
  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
