/**
 * @file main.cpp
 * @brief Gabor filter bank for oriented texture analysis
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - How to build Gabor kernels with cv::getGaborKernel()
 * - Applying a bank of filters at several orientations with filter2D (detailed in 03_02)
 * - Visualizing each kernel next to its filter response
 *
 * A Gabor filter is a sinusoid modulated by a Gaussian envelope. It responds
 * strongly where the image contains a specific FREQUENCY at a specific
 * ORIENTATION, which is why banks of Gabor filters are a classic tool for
 * texture description (and a good model of the receptive fields found in the
 * human visual cortex).
 *
 * @note This filter belongs to the transform-domain family of 04_01..04_03
 *       (it selects frequency content), and that is where the book places it,
 *       but in practice it is applied by SPATIAL convolution. The single call
 *       it needs, cv::filter2D, is used here as a black box and taken apart
 *       in 03_02, kernel by kernel.
 */

#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>

// Gabor kernel parameters (see comments in main for their meaning)
namespace Config
{
constexpr int KERNEL_SIZE = 31;      // Kernel width/height in pixels (odd)
constexpr double SIGMA = 4.0;        // Std-dev of the Gaussian envelope
constexpr double LAMBDA = 10.0;      // Wavelength of the sinusoid (px/cycle)
constexpr double GAMMA = 0.5;        // Spatial aspect ratio (<1 = elongated)
constexpr double PSI = 0.0;          // Phase offset of the sinusoid
constexpr int NUM_ORIENTATIONS = 4;  // Bank size: 0, 45, 90, 135 degrees
}

int main(int argc, char ** argv)
{
  // Load image in grayscale
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/starry_night.jpg | Input file}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  const std::string image_path = parser.get<std::string>("@input");
  cv::Mat src = cv::imread(cv::samples::findFile(image_path, false), cv::IMREAD_GRAYSCALE);

  if (src.empty()) {
    std::cerr << "Error: Could not load image '" << image_path << "'" << std::endl;
    return EXIT_FAILURE;
  }

  // Work in float [0,1], as in 03_02 (precise filtering, easy display)
  src.convertTo(src, CV_32F, 1.0 / 255.0);

  std::cout << "=== Gabor Filter Bank ===" << std::endl;
  std::cout << "Kernel: " << Config::KERNEL_SIZE << "x" << Config::KERNEL_SIZE
            << "  sigma=" << Config::SIGMA
            << "  lambda=" << Config::LAMBDA
            << "  gamma=" << Config::GAMMA << std::endl;
  std::cout << "Orientations: " << Config::NUM_ORIENTATIONS
            << " (0, 45, 90, 135 degrees)" << std::endl;

  cv::imshow("Original", src);

  // Build and apply one Gabor filter per orientation
  //
  // cv::getGaborKernel(ksize, sigma, theta, lambda, gamma, psi)
  //   sigma:  width of the Gaussian envelope -- how far the filter "sees"
  //   theta:  orientation of the sinusoid normal; the filter responds to
  //           edges/stripes PERPENDICULAR to this angle
  //   lambda: wavelength -- the stripe spacing the filter is tuned to
  //   gamma:  aspect ratio of the envelope (1 = circular, <1 = elongated
  //           along the stripes, making orientation selectivity sharper)
  //   psi:    phase (0 = even/cosine filter, pi/2 = odd/sine filter)
  for (int i = 0; i < Config::NUM_ORIENTATIONS; ++i) {
    const double theta = i * CV_PI / Config::NUM_ORIENTATIONS;
    const int degrees = i * 180 / Config::NUM_ORIENTATIONS;

    cv::Mat kernel = cv::getGaborKernel(
      cv::Size(Config::KERNEL_SIZE, Config::KERNEL_SIZE),
      Config::SIGMA, theta, Config::LAMBDA, Config::GAMMA, Config::PSI, CV_32F);

    // Same spatial convolution (correlation) taken apart in 03_02
    cv::Mat response;
    cv::filter2D(src, response, CV_32F, kernel);

    // --- Visualization ---
    // Kernel: tiny (31x31), so upscale with INTER_NEAREST to see its shape;
    // normalize because its values are centered around zero
    cv::Mat kernel_display;
    cv::normalize(kernel, kernel_display, 0, 1, cv::NORM_MINMAX);
    cv::resize(kernel_display, kernel_display, cv::Size(128, 128), 0, 0,
               cv::INTER_NEAREST);
    cv::imshow("Kernel " + std::to_string(degrees) + " deg", kernel_display);

    // Response: bright where the image matches the filter's orientation.
    // Shift/scale like the Sobel display of 03_02 (responses are signed)
    cv::Mat response_display;
    response.convertTo(response_display, CV_32F, 0.5, 0.5);
    cv::imshow("Response " + std::to_string(degrees) + " deg", response_display);

    std::cout << "  theta = " << degrees
              << " deg -> responds to structures oriented at "
              << (degrees + 90) % 180 << " deg" << std::endl;
  }

  std::cout << "\nObserve how each response highlights only the edges" << std::endl;
  std::cout << "aligned with its filter (e.g. the 0 deg filter answers to" << std::endl;
  std::cout << "vertical structures). A texture descriptor is built by" << std::endl;
  std::cout << "collecting the response energy of every filter in the bank." << std::endl;
  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
