/**
 * @file main.cpp
 * @brief Canny edge detection demonstration using OpenCV
 * @author José Miguel Guerrero Hernández
 * @note The Canny algorithm is a multi-stage edge detector that uses:
 *       1) Gaussian filter for noise reduction
 *       2) Gradient calculation using Sobel operators
 *       3) Non-maximum suppression for edge thinning
 *       4) Hysteresis thresholding for edge tracking
 */

#include <cstdlib>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

int main(int argc, char ** argv)
{
  // Load input image in color
  const std::string image_path = argc >= 2 ? argv[1] : "lena.jpg";
  cv::Mat image = cv::imread(cv::samples::findFile(image_path), cv::IMREAD_COLOR);

  if (image.empty()) {
    std::cerr << "Error: Could not open or find the image!" << std::endl;
    std::cerr << "Path: " << image_path << std::endl;
    std::cerr << "Usage: " << argv[0] << " <Input image>" << std::endl;
    return EXIT_FAILURE;
  }

  // Convert to grayscale (Canny requires single-channel image)
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

  // ========================================
  // Canny edge detection - Version 1: Direct on grayscale
  // ========================================
  cv::Mat edges_direct;

  // cv::Canny(input, output, lowThreshold, highThreshold, apertureSize, L2gradient)
  // Parameters:
  //   input: 8-bit single-channel (grayscale) image
  //   output: binary edge map (0 or 255)
  //   lowThreshold: lower bound for hysteresis (50)
  //   highThreshold: upper bound for hysteresis (150)
  //     Recommended ratio: high/low = 2:1 or 3:1 (here 3:1)
  //   apertureSize: Sobel operator kernel size (3, 5, or 7)
  //   L2gradient: if true, uses more accurate L2 norm (sqrt(Gx² + Gy²))
  //
  // Hysteresis thresholding:
  //   Gradient > highThreshold → strong edge (always kept)
  //   Gradient < lowThreshold → discarded (not an edge)
  //   Gradient between thresholds → weak edge (kept only if connected to strong edge)
  cv::Canny(gray, edges_direct, 50, 150, 3, false);

  // ========================================
  // Canny edge detection - Version 2: With explicit Gaussian blur preprocessing
  // ========================================
  // Although Canny applies blur internally, explicit blur can improve noisy images
  cv::Mat blurred, edges_blurred;
  cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.4);
  cv::Canny(blurred, edges_blurred, 50, 150, 3, false);

  // Display results
  cv::imshow("1. Original (Color)", image);
  cv::imshow("2. Grayscale", gray);
  cv::imshow("3. Canny Edges (Direct)", edges_direct);
  cv::imshow("4. Blurred", blurred);
  cv::imshow("5. Canny Edges (After Blur)", edges_blurred);

  // Wait for user input and exit
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
