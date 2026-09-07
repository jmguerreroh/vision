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

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

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

// Text drawn on the full-resolution image shrinks with the on-screen reduction.
// Raising the font by the same factor makes it land at the size it was written
// for. Labels anchored to a region cannot simply be drawn after the reduction,
// because their position comes from the coordinates of the full-size image
double fontFor(const cv::Mat & image, double base)
{
  const int side = std::max(image.cols, image.rows);
  return side <= MAX_DISPLAY_SIDE ? base : base * side / MAX_DISPLAY_SIDE;
}

void showFit(const std::string & window, const cv::Mat & image)
{
  cv::imshow(window, fitToScreen(image));
}
}  // namespace

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
              cv::FONT_HERSHEY_COMPLEX, fontFor(display, 1), cv::Scalar(255, 255, 255));
  showFit(Config::WINDOW_NAME, display);
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
                cv::FONT_HERSHEY_SIMPLEX, fontFor(display, 0.7), cv::Scalar(0, 255, 0), 2);
  }

  showFit(Config::WINDOW_NAME, display);
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
    double sigma_color = k * Config::BILATERAL_SIGMA_COLOR_MULTIPLIER;
    double sigma_space = k / Config::BILATERAL_SIGMA_SPACE_DIVISOR;
    cv::bilateralFilter(src, dst, k, sigma_color, sigma_space);
    if (displayResult(dst, k)) {return true;}
  }
  return false;
}

/**
 * @brief Adds salt-and-pepper noise to an image
 * @param src Input image
 * @param amount Fraction of pixels to corrupt (e.g. 0.05 = 5%)
 * @return Noisy copy of the input
 *
 * Salt-and-pepper noise sets random pixels to pure white or pure black,
 * simulating transmission errors or dead sensor pixels. It is the classic
 * noise type where the MEDIAN filter clearly outperforms averaging filters:
 * an extreme outlier barely affects the median of a neighborhood, but it
 * drags the mean towards itself.
 */
cv::Mat addSaltPepperNoise(const cv::Mat & src, double amount)
{
  cv::Mat noisy = src.clone();
  cv::RNG rng(12345);  // Fixed seed: same noise pattern on every run
  const int num_pixels = static_cast<int>(amount * src.rows * src.cols);
  for (int i = 0; i < num_pixels; ++i) {
    const int y = rng.uniform(0, src.rows);
    const int x = rng.uniform(0, src.cols);
    // Half salt (white), half pepper (black)
    noisy.at<cv::Vec3b>(y, x) = (i % 2 == 0) ? cv::Vec3b(255, 255, 255)
                                             : cv::Vec3b(0, 0, 0);
  }
  return noisy;
}

/**
 * @brief Side-by-side comparison: which filter removes salt-and-pepper noise?
 * @param src Clean source image
 *
 * Shows why the choice of filter must match the type of noise:
 * - Gaussian blur averages the outliers INTO the image (gray smudges remain)
 * - Median blur discards them entirely (outliers never win the median vote)
 */
void demoNoiseComparison(const cv::Mat & src)
{
  std::cout << "  Noise comparison (salt-and-pepper, 5% of pixels)..." << std::endl;

  const cv::Mat noisy = addSaltPepperNoise(src, 0.05);

  cv::Mat gaussian_result, median_result;
  cv::GaussianBlur(noisy, gaussian_result, cv::Size(5, 5), 0);
  cv::medianBlur(noisy, median_result, 5);

  showFit("Noisy (salt & pepper 5%)", noisy);
  showFit("Gaussian 5x5 on noisy (smudges remain)", gaussian_result);
  showFit("Median 5x5 on noisy (noise removed)", median_result);
}

int main(int argc, char ** argv)
{
  // Load image
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/starry_night.png | Input file}");
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
    std::cerr << "Error: Could not open image!" << std::endl;
    std::cerr << "Usage: " << argv[0] << " [image_path]" << std::endl;
    return EXIT_FAILURE;
  }

  // Resize to standard size for consistent display
  cv::resize(src, src, Config::IMAGE_SIZE);

  std::cout << "=== Smoothing Filters Demo ===" << std::endl;
  std::cout << "Image: " << filename << " (" << src.cols << "x" << src.rows << ")" << std::endl;
  std::cout << "Press any key to skip to next filter..." << std::endl;

  cv::namedWindow(Config::WINDOW_NAME, cv::WINDOW_AUTOSIZE);

  // Show original
  if (displayCaption(src, "Original Image")) {return EXIT_SUCCESS;}
  if (displayResult(src)) {return EXIT_SUCCESS;}

  // Run all blur demos
  if (demoHomogeneousBlur(src)) {return EXIT_SUCCESS;}
  if (demoGaussianBlur(src)) {return EXIT_SUCCESS;}
  if (demoMedianBlur(src)) {return EXIT_SUCCESS;}
  if (demoBilateralBlur(src)) {return EXIT_SUCCESS;}

  // Final lesson: same filters, radically different results depending on
  // the NOISE TYPE. These windows stay open until a key is pressed.
  demoNoiseComparison(src);
  displayCaption(src, "Done! Check the noise comparison windows");
  std::cout << "Demo completed. Press any key on an image window to exit." << std::endl;
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
