/**
 * @file main.cpp
 * @brief Pixel-to-pixel transformations demonstration
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates point operations (pixel-to-pixel transformations)
 * where each output pixel depends only on the corresponding input pixel.
 *
 * Point operations covered:
 * - Inverse (negative): out = 255 - in
 * - Binary threshold: out = (in > T) ? 255 : 0
 *
 * These are the simplest image transformations, useful for:
 * - Contrast enhancement
 * - Image segmentation
 * - Preprocessing for other algorithms
 *
 * General form: g(x,y) = T[f(x,y)]
 * where T is a transformation function applied to each pixel independently.
 *
 * @note Uses ../../data/starry_night.png as input image
 */

#include <string>
#include <algorithm>
#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>

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

/**
 * @brief Displays usage information
 * @param argv Command line arguments
 */
void printHelp(char ** argv)
{
  std::cout << "\n"
            << "Pixel-to-Pixel Transformations Demo\n"
            << "===================================\n"
            << "This program demonstrates point operations where each output pixel\n"
            << "depends only on the corresponding input pixel.\n\n"
            << "Usage: " << argv[0] << " [image_path]\n"
            << "  image_path: Path to input image (default: starry_night.jpg)\n\n";
}

/**
 * @brief Apply inverse (negative) transformation to an image
 *
 * The inverse transformation maps each pixel value to its complement:
 *   output = 255 - input
 *
 * This is useful for:
 * - Enhancing white/gray details in dark regions
 * - Medical imaging (e.g., X-rays)
 * - Photographic negatives
 *
 * @param src Input grayscale image (CV_8UC1)
 * @return Output inverted image
 */
cv::Mat applyInverse(const cv::Mat & src)
{
  cv::Mat dst(src.rows, src.cols, src.type());

  for (int y = 0; y < src.rows; y++) {
    for (int x = 0; x < src.cols; x++) {
      // Inverse transformation: new_value = 255 - old_value
      dst.at<uchar>(y, x) = 255 - src.at<uchar>(y, x);
    }
  }

  return dst;
}

/**
 * @brief Apply binary threshold transformation to an image
 *
 * The threshold transformation creates a binary image:
 *   output = 255 if input > threshold, else 0
 *
 * This is useful for:
 * - Simple image segmentation
 * - Object detection preprocessing
 * - Document binarization
 *
 * @param src Input grayscale image (CV_8UC1)
 * @param threshold Threshold value (0-255)
 * @return Output binary image
 */
cv::Mat applyThreshold(const cv::Mat & src, int threshold)
{
  cv::Mat dst(src.rows, src.cols, src.type());

  for (int y = 0; y < src.rows; y++) {
    for (int x = 0; x < src.cols; x++) {
      // Binary threshold: above threshold -> white, below -> black
      if (src.at<uchar>(y, x) > threshold) {
        dst.at<uchar>(y, x) = 255;
      } else {
        dst.at<uchar>(y, x) = 0;
      }
    }
  }

  return dst;
}

int main(int argc, char ** argv)
{

  // Load image in grayscale
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/starry_night.png | Input file}");
  if (parser.has("help")) {
    parser.printMessage();
    printHelp(argv);
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
  cv::Mat src = cv::imread(cv::samples::findFile(filename, false), cv::IMREAD_GRAYSCALE);

  if (src.empty()) {
    std::cerr << "Error: Could not load image '" << filename << "'" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "=== Pixel-to-Pixel Transformations ===" << std::endl;
  std::cout << "Image size: " << src.cols << "x" << src.rows << std::endl;

  // Transformation 1: Inverse (Negative)
  // We implement it manually to make the pixel-wise nature explicit.
  // The one-line OpenCV equivalent is:  cv::Mat inverse = 255 - src;
  std::cout << "\n1. Applying inverse transformation..." << std::endl;
  std::cout << "   Formula: output = 255 - input" << std::endl;
  cv::Mat inverse = applyInverse(src);

  // Transformation 2: Binary Threshold
  const int THRESHOLD_VALUE = 128;
  std::cout << "\n2. Applying binary threshold..." << std::endl;
  std::cout << "   Threshold value: " << THRESHOLD_VALUE << std::endl;
  std::cout << "   Formula: output = (input > " << THRESHOLD_VALUE << ") ? 255 : 0" << std::endl;
  // OpenCV equivalent (signature detailed in chapter 3, used from chapter 9 onwards):
  //   cv::threshold(src, thresholded, THRESHOLD_VALUE, 255, cv::THRESH_BINARY);
  cv::Mat thresholded = applyThreshold(src, THRESHOLD_VALUE);

  // Display results
  showFit("Original", src);
  showFit("Inverse (Negative)", inverse);
  showFit("Binary Threshold", thresholded);

  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
