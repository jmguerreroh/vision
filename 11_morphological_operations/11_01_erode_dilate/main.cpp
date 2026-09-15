/**
 * @file main.cpp
 * @brief Erosion and Dilation - sample code demonstrating basic morphological operations
 * @author José Miguel Guerrero Hernández
 * @note This program demonstrates how to apply erosion and dilation morphological
 *       operations using different structuring elements (rectangular, cross, ellipse)
 *       with adjustable kernel sizes via trackbars.
 */

#include <string>
#include <algorithm>
#include <cstdlib>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
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


// Configuration constants
namespace Config
{
constexpr int MAX_OPERATOR = 1;
constexpr int MAX_ELEM = 2;
constexpr int MAX_KERNEL_SIZE = 21;
const char * WINDOW_NAME = "Erode and Dilate Demo";
const char * TRACKBAR_OPERATOR = "Operator: 0: Erode - 1: Dilate";
const char * TRACKBAR_ELEMENT = "Element: 0: Rect - 1: Cross - 2: Ellipse";
const char * TRACKBAR_KERNEL = "Kernel size: 2n +1";
}

/**
 * @brief Application state for morphological operations
 */
struct MorphApp
{
  cv::Mat src;   // Source image
  cv::Mat dst;   // Destination image
};

// Global app state (required for OpenCV callbacks)
MorphApp app;

/**
 * @brief Callback function for trackbar events - applies erosion or dilation
 *
 * @note The two parameters are imposed by the OpenCV trackbar callback
 *       signature and are deliberately left unnamed: there is nothing to
 *       document about them, and the state comes from the global app object.
 *
 * Erosion: Shrinks bright regions and removes small white noise
 * Dilation: Expands bright regions and fills small holes
 */
void erodeDilate(int, void *)
{
  // Get current trackbar positions
  int morph_operator = cv::getTrackbarPos(Config::TRACKBAR_OPERATOR, Config::WINDOW_NAME);
  int morph_elem = cv::getTrackbarPos(Config::TRACKBAR_ELEMENT, Config::WINDOW_NAME);
  int morph_size = cv::getTrackbarPos(Config::TRACKBAR_KERNEL, Config::WINDOW_NAME);

  // Create the structuring element
  // morph_elem: The shape of the structuring element:
  //    cv::MORPH_RECT (rectangular)
  //    cv::MORPH_ELLIPSE (elliptical)
  //    cv::MORPH_CROSS (cross-shaped)
  // Size: 2 * morph_size + 1 ensures an odd size for a well-defined center
  // Anchor: cv::Point(morph_size, morph_size) specifies the center of the kernel
  cv::Mat element = cv::getStructuringElement(
        morph_elem,
        cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
        cv::Point(morph_size, morph_size));

  // Apply the selected morphological operation
  // Erosion: Minimum filter - replaces each pixel with the minimum in the neighborhood
  // Dilation: Maximum filter - replaces each pixel with the maximum in the neighborhood
  if (morph_operator == 0) {
    cv::erode(app.src, app.dst, element);
  } else {
    cv::dilate(app.src, app.dst, element);
  }

  showFit(Config::WINDOW_NAME, app.dst);
}

int main(int argc, char ** argv)
{
  // Parse command line arguments
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/horse.png | Input file}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  app.src = cv::imread(
    cv::samples::findFile(parser.get<std::string>("@input"), false), cv::IMREAD_COLOR);

  // Without this, a malformed value (--frames=xyz) is reported by the parser
  // but the example carries on with the default, which is the hardest kind
  // of failure to diagnose. Note it validates values, not option names:
  // cv::CommandLineParser ignores an unknown option without complaining
  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }

  if (app.src.empty()) {
    std::cout << "Could not open or find the image!\n" << std::endl;
    std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
    return EXIT_FAILURE;
  }

  // Create the display window
  cv::namedWindow(Config::WINDOW_NAME, cv::WINDOW_AUTOSIZE);

  // Create trackbars for interactive control
  cv::createTrackbar(Config::TRACKBAR_OPERATOR, Config::WINDOW_NAME,
    nullptr, Config::MAX_OPERATOR, erodeDilate);
  cv::createTrackbar(Config::TRACKBAR_ELEMENT, Config::WINDOW_NAME,
    nullptr, Config::MAX_ELEM, erodeDilate);
  cv::createTrackbar(Config::TRACKBAR_KERNEL, Config::WINDOW_NAME,
    nullptr, Config::MAX_KERNEL_SIZE, erodeDilate);

  // Set initial kernel size to 1
  cv::setTrackbarPos(Config::TRACKBAR_KERNEL, Config::WINDOW_NAME, 1);

  // Apply initial operation
  erodeDilate(0, nullptr);

  cv::waitKey(0);
  return EXIT_SUCCESS;
}
