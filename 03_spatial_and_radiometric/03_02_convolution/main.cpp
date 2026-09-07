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
 * @note Uses ../../data/starry_night.png as input image
 */

#include <string>
#include <algorithm>
#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

namespace
{
// The images this example works on are about 1400 px on the long side, and
// several windows at that size do not fit on a normal screen. The processing
// always runs at full resolution: only the copy sent to the screen is reduced,
// with INTER_AREA, which is the interpolation meant for shrinking
constexpr int MAX_DISPLAY_SIDE = 800;

void showFit(const std::string & window, const cv::Mat & image)
{
  const int side = std::max(image.cols, image.rows);
  if (side <= MAX_DISPLAY_SIDE || image.empty()) {
    cv::imshow(window, image);
    return;
  }
  const double factor = static_cast<double>(MAX_DISPLAY_SIDE) / side;
  cv::Mat reduced;
  cv::resize(image, reduced, cv::Size(), factor, factor, cv::INTER_AREA);
  cv::imshow(window, reduced);
}
}  // namespace

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
            << "  image_path: Path to input image (default: starry_night.jpg)\n\n";
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
  return (cv::Mat_<float>(3, 3) <<
         1, 2, 1,
         0, 0, 0,
         -1, -2, -1);
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
  return (cv::Mat_<float>(3, 3) <<
         1, 0, -1,
         2, 0, -2,
         1, 0, -1);
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

  // Convert to float for precise calculations
  src.convertTo(src, CV_32F, 1.0 / 255.0);

  std::cout << "=== Neighborhood Transformations (Spatial Filtering) ===" << std::endl;
  std::cout << "Image size: " << src.cols << "x" << src.rows << std::endl;

  // Create convolution kernels
  cv::Mat box_kernel = createBoxKernel();
  cv::Mat sobel_y = createSobelYKernel();
  cv::Mat sobel_x = createSobelXKernel();

  std::cout << "\nKernels applied:" << std::endl;
  std::cout << "1. Box filter (3x3 averaging) - Smoothing" << std::endl;
  std::cout << "2. Sobel Y - Horizontal edge detection" << std::endl;
  std::cout << "3. Sobel X - Vertical edge detection" << std::endl;

  // Apply the spatial filters.
  //
  // Technical note: filter2D actually computes a CORRELATION -- it does NOT
  // flip the kernel, as a strict mathematical convolution would. For
  // symmetric kernels (like the box filter) the result is identical; for
  // antisymmetric kernels (like Sobel) only the sign of the response flips.
  // For a strict convolution, rotate the kernel 180 degrees first:
  //   cv::flip(kernel, kernel, -1);
  cv::Mat blurred, edges_y, edges_x;

  cv::filter2D(src, blurred, src.depth(), box_kernel);
  cv::filter2D(src, edges_y, src.depth(), sobel_y);
  cv::filter2D(src, edges_x, src.depth(), sobel_x);

  // Normalize edge images for better visualization
  // (edge values can be negative, shift to [0,1] range)
  cv::Mat edges_y_display, edges_x_display;
  edges_y.convertTo(edges_y_display, CV_32F, 0.5, 0.5);    // Scale and shift
  edges_x.convertTo(edges_x_display, CV_32F, 0.5, 0.5);

  // Display results
  showFit("Original", src);
  showFit("Box Filter (Blur)", blurred);
  showFit("Sobel Y (Horizontal Edges)", edges_y_display);
  showFit("Sobel X (Vertical Edges)", edges_x_display);

  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
