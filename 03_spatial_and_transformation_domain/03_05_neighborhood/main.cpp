/**
 * @file main.cpp
 * @brief Neighborhood transformations using convolution kernels
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates neighborhood operations where each output pixel
 * depends on a neighborhood of input pixels (spatial filtering).
 *
 * Convolution operation:
 *   g(x,y) = Σ Σ h(i,j) * f(x-i, y-j)
 *
 * Kernels demonstrated:
 * 1. Box filter (averaging): Blurs the image by averaging neighbors
 * 2. Sobel Y (horizontal edges): Detects horizontal gradients
 * 3. Sobel X (vertical edges): Detects vertical gradients
 *
 * Key concepts:
 * - Kernel/Mask: Small matrix defining the operation
 * - Convolution: Sliding the kernel over the image
 * - Edge detection: Using derivative approximations
 *
 * @note Uses ../../data/lena.jpg as input image
 */

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>

/**
 * @brief Displays usage information
 * @param argv Command line arguments
 */
void printHelp(char ** argv)
{
  std::cout << "\n"
            << "Neighborhood Transformations Demo\n"
            << "=================================\n"
            << "This program demonstrates spatial filtering using convolution kernels.\n"
            << "Each output pixel depends on a neighborhood of input pixels.\n\n"
            << "Usage: " << argv[0] << " [image_path]\n"
            << "  image_path: Path to input image (default: lena.jpg)\n\n";
}

/**
 * @brief Create a box filter kernel (averaging)
 *
 * Box filter averages all pixels in the neighborhood.
 * Useful for noise reduction and blurring.
 *
 * Kernel:  [1 1 1]
 *          [1 1 1]  * (1/9)
 *          [1 1 1]
 *
 * @return 3x3 box filter kernel
 */
cv::Mat createBoxKernel()
{
  cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1);
  return kernel / 9.0f;    // Normalize to preserve brightness
}

/**
 * @brief Create Sobel Y kernel (horizontal edge detection)
 *
 * Approximates vertical derivative (∂f/∂y).
 * Detects horizontal edges (changes in vertical direction).
 *
 * Kernel:  [ 1  2  1]
 *          [ 0  0  0]
 *          [-1 -2 -1]
 *
 * @return 3x3 Sobel Y kernel
 */
cv::Mat createSobelYKernel()
{
  return  cv::Mat_<float>(3, 3) <<
         1, 2, 1,
         0, 0, 0,
         -1, -2, -1;
}

/**
 * @brief Create Sobel X kernel (vertical edge detection)
 *
 * Approximates horizontal derivative (∂f/∂x).
 * Detects vertical edges (changes in horizontal direction).
 *
 * Kernel:  [ 1  0 -1]
 *          [ 2  0 -2]
 *          [ 1  0 -1]
 *
 * @return 3x3 Sobel X kernel
 */
cv::Mat createSobelXKernel()
{
  return  cv::Mat_<float>(3, 3) <<
         1, 0, -1,
         2, 0, -2,
         1, 0, -1;
}

int main(int argc, char ** argv)
{
  printHelp(argv);

  // Load image in grayscale
  const char * filename = argc >= 2 ? argv[1] : "lena.jpg";
  cv::Mat src = cv::imread(cv::samples::findFile(filename), cv::IMREAD_GRAYSCALE);

  if (src.empty()) {
    std::cerr << "Error: Could not load image '" << filename << "'" << std::endl;
    return EXIT_FAILURE;
  }

  // Convert to float for precise calculations
  src.convertTo(src, CV_32F, 1.0 / 255.0);

  std::cout << "=== Neighborhood Transformations (Spatial Filtering) ===" << std::endl;
  std::cout << "Image size: " << src.cols << "x" << src.rows << std::endl;

  // Create convolution kernels
  cv::Mat boxKernel = createBoxKernel();
  cv::Mat sobelY = createSobelYKernel();
  cv::Mat sobelX = createSobelXKernel();

  std::cout << "\nKernels applied:" << std::endl;
  std::cout << "1. Box filter (3x3 averaging) - Smoothing" << std::endl;
  std::cout << "2. Sobel Y - Horizontal edge detection" << std::endl;
  std::cout << "3. Sobel X - Vertical edge detection" << std::endl;

  // Apply convolution filters
  cv::Mat blurred, edgesY, edgesX;

  cv::filter2D(src, blurred, src.depth(), boxKernel);
  cv::filter2D(src, edgesY, src.depth(), sobelY);
  cv::filter2D(src, edgesX, src.depth(), sobelX);

  // Normalize edge images for better visualization
  // (edge values can be negative, shift to [0,1] range)
  cv::Mat edgesYDisplay, edgesXDisplay;
  edgesY.convertTo(edgesYDisplay, CV_32F, 0.5, 0.5);    // Scale and shift
  edgesX.convertTo(edgesXDisplay, CV_32F, 0.5, 0.5);

  // Display results
  cv::imshow("Original", src);
  cv::imshow("Box Filter (Blur)", blurred);
  cv::imshow("Sobel Y (Horizontal Edges)", edgesYDisplay);
  cv::imshow("Sobel X (Vertical Edges)", edgesXDisplay);

  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
