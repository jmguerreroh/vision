/**
 * @file main.cpp
 * @brief Smoothing/Blurring filters demonstration using OpenCV
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates various smoothing techniques:
 * - Homogeneous (Normalized Box) Filter: simple averaging
 * - Gaussian Filter: weighted average, reduces high-frequency noise
 * - Median Filter: replaces pixel with median of neighbors, good for salt-and-pepper noise
 * - Bilateral Filter: edge-preserving smoothing
 *
 * @note Each filter is applied with increasing kernel sizes to show the effect.
 * @see https://docs.opencv.org/3.4/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html
 */

#include <iostream>
#include <string>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

// Configuration constants
namespace Config
{
constexpr int DELAY_CAPTION = 1500;           // Delay for caption display (ms)
constexpr int DELAY_BLUR = 100;               // Delay between blur iterations (ms)
constexpr int MAX_KERNEL_LENGTH = 31;         // Maximum kernel size
const cv::Size IMAGE_SIZE(512, 512);          // Standard display size
const std::string WINDOW_NAME = "Smoothing Demo";

// Bilateral filter parameters
constexpr double BILATERAL_SIGMA_COLOR_MULTIPLIER = 2.0;    // Color space sigma multiplier
constexpr double BILATERAL_SIGMA_SPACE_DIVISOR = 2.0;       // Coordinate space sigma divisor
}

/**
 * @brief Displays a caption message on a black background
 * @param src Source image used to determine display size
 * @param caption Text message to display
 * @return true if user pressed a key (to exit), false otherwise
 */
bool displayCaption(const cv::Mat & src, const std::string & caption)
{
  cv::Mat display = cv::Mat::zeros(src.size(), src.type());
  cv::putText(display, caption,
              cv::Point(src.cols / 4, src.rows / 2),
              cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 255, 255));
  cv::imshow(Config::WINDOW_NAME, display);
  return cv::waitKey(Config::DELAY_CAPTION) >= 0;
}

/**
 * @brief Displays an image with optional kernel size information overlay
 * @param img Image to display
 * @param kernelSize Kernel size to show in overlay (0 to hide)
 * @return true if user pressed a key (to exit), false otherwise
 */
bool displayResult(const cv::Mat & img, int kernelSize = 0)
{
  cv::Mat display = img.clone();

  // Show kernel size in top-left corner
  if (kernelSize > 0) {
    std::string text = "Kernel: " + std::to_string(kernelSize) + "x" + std::to_string(kernelSize);
    cv::putText(display, text, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
  }

  cv::imshow(Config::WINDOW_NAME, display);
  return cv::waitKey(Config::DELAY_BLUR) >= 0;
}

/**
 * @brief Applies homogeneous (normalized box) blur with increasing kernel sizes
 * @param src Input source image to blur
 * @return true if user pressed a key (to exit), false otherwise
 */
bool demoHomogeneousBlur(const cv::Mat & src)
{
  if (displayCaption(src, "Homogeneous Blur")) {return true;}
  std::cout << "  Homogeneous Blur (normalized box filter)..." << std::endl;

  cv::Mat dst;
  for (int k = 1; k < Config::MAX_KERNEL_LENGTH; k += 2) {
    cv::blur(src, dst, cv::Size(k, k));
    if (displayResult(dst, k)) {return true;}
  }
  return false;
}

/**
 * @brief Applies Gaussian blur with increasing kernel sizes
 * @param src Input source image to blur
 * @return true if user pressed a key (to exit), false otherwise
 */
bool demoGaussianBlur(const cv::Mat & src)
{
  if (displayCaption(src, "Gaussian Blur")) {return true;}
  std::cout << "  Gaussian Blur (weighted average)..." << std::endl;

  cv::Mat dst;
  for (int k = 1; k < Config::MAX_KERNEL_LENGTH; k += 2) {
    // sigmaX=0, sigmaY=0: automatically calculated from kernel size
    cv::GaussianBlur(src, dst, cv::Size(k, k), 0, 0);
    if (displayResult(dst, k)) {return true;}
  }
  return false;
}

/**
 * @brief Applies median blur with increasing kernel sizes
 * @param src Input source image to blur
 * @return true if user pressed a key (to exit), false otherwise
 */
bool demoMedianBlur(const cv::Mat & src)
{
  if (displayCaption(src, "Median Blur")) {return true;}
  std::cout << "  Median Blur (good for salt-and-pepper noise)..." << std::endl;

  cv::Mat dst;
  for (int k = 1; k < Config::MAX_KERNEL_LENGTH; k += 2) {
    cv::medianBlur(src, dst, k);
    if (displayResult(dst, k)) {return true;}
  }
  return false;
}

/**
 * @brief Applies bilateral filter with increasing kernel sizes
 * @param src Input source image to filter
 * @return true if user pressed a key (to exit), false otherwise
 *
 * @note Bilateral filter parameters:
 *   - d: Diameter of pixel neighborhood (kernel size)
 *   - sigmaColor: Filter sigma in color space (larger = more colors mixed)
 *   - sigmaSpace: Filter sigma in coordinate space (larger = farther pixels influence)
 */
bool demoBilateralBlur(const cv::Mat & src)
{
  if (displayCaption(src, "Bilateral Filter")) {return true;}
  std::cout << "  Bilateral Filter (edge-preserving)..." << std::endl;

  cv::Mat dst;
  for (int k = 1; k < Config::MAX_KERNEL_LENGTH; k += 2) {
    // Bilateral filter smooths while preserving edges
    // sigmaColor scales with kernel size to maintain edge detection
    // sigmaSpace inversely scales to control spatial influence
    double sigmaColor = k * Config::BILATERAL_SIGMA_COLOR_MULTIPLIER;
    double sigmaSpace = k / Config::BILATERAL_SIGMA_SPACE_DIVISOR;
    cv::bilateralFilter(src, dst, k, sigmaColor, sigmaSpace);
    if (displayResult(dst, k)) {return true;}
  }
  return false;
}

int main(int argc, char ** argv)
{
  // Load image
  const std::string filename = (argc >= 2) ? argv[1] : "lena.jpg";
  cv::Mat src = cv::imread(cv::samples::findFile(filename), cv::IMREAD_COLOR);

  if (src.empty()) {
    std::cerr << "Error: Could not open image!" << std::endl;
    std::cerr << "Usage: " << argv[0] << " [image_path]" << std::endl;
    return -1;
  }

  // Resize to standard size for consistent display
  cv::resize(src, src, Config::IMAGE_SIZE);

  std::cout << "=== Smoothing Filters Demo ===" << std::endl;
  std::cout << "Image: " << filename << " (" << src.cols << "x" << src.rows << ")" << std::endl;
  std::cout << "Press any key to skip to next filter..." << std::endl;

  cv::namedWindow(Config::WINDOW_NAME, cv::WINDOW_AUTOSIZE);

  // Show original
  if (displayCaption(src, "Original Image")) {return 0;}
  if (displayResult(src)) {return 0;}

  // Run all blur demos
  if (demoHomogeneousBlur(src)) {return 0;}
  if (demoGaussianBlur(src)) {return 0;}
  if (demoMedianBlur(src)) {return 0;}
  if (demoBilateralBlur(src)) {return 0;}

  // Done
  displayCaption(src, "Done!");
  std::cout << "Demo completed." << std::endl;

  return 0;
}
