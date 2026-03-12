/**
 * @file main.cpp
 * @brief Morphological contours - sample code for extracting internal and external contours
 * @author José Miguel Guerrero Hernández
 * @note This program demonstrates how to extract contours using morphological operations.
 *       Internal contours: original - eroded image
 *       External contours: dilated image - original
 */

#include <cstdlib>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

// Configuration constants
namespace Config
{
constexpr int MAX_OPERATOR = 1;
constexpr int MAX_ELEM = 2;
constexpr int MAX_KERNEL_SIZE = 21;
const char * WINDOW_NAME = "Morphological Contours Demo";
const char * TRACKBAR_OPERATOR = "Operator: 0: In - 1: Out";
const char * TRACKBAR_ELEMENT = "Element: 0: Rect - 1: Cross - 2: Ellipse";
const char * TRACKBAR_KERNEL = "Kernel size: 2n +1";
}

/**
 * @brief Application state for morphological contour extraction
 */
struct MorphApp
{
  cv::Mat src;   // Source image
  cv::Mat dst;   // Destination image
};

// Global app state (required for OpenCV callbacks)
MorphApp app;

/**
 * @brief Callback function for trackbar events - extracts morphological contours
 * @param[in] Unused parameter required by OpenCV callback signature
 * @param[in] Unused user data pointer
 *
 * Morphological contour extraction:
 *   Internal contour: src - erode(src) -> shows the inner edge of objects
 *   External contour: dilate(src) - src -> shows the outer edge of objects
 *
 * This is equivalent to the morphological gradient but allows separate
 * visualization of internal and external boundaries.
 */
void morphological_contours(int, void *)
{
  // Get current trackbar positions
  int morph_operator = cv::getTrackbarPos(Config::TRACKBAR_OPERATOR, Config::WINDOW_NAME);
  int morph_elem = cv::getTrackbarPos(Config::TRACKBAR_ELEMENT, Config::WINDOW_NAME);
  int morph_size = cv::getTrackbarPos(Config::TRACKBAR_KERNEL, Config::WINDOW_NAME);

  // Create the structuring element
  cv::Mat element = cv::getStructuringElement(
        morph_elem,
        cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
        cv::Point(morph_size, morph_size));

  if (morph_operator == 0) {
    // Internal contours: original minus eroded
    // This highlights the pixels that would be removed by erosion
    cv::erode(app.src, app.dst, element);
    app.dst = app.src - app.dst;
  } else {
    // External contours: dilated minus original
    // This highlights the pixels that would be added by dilation
    cv::dilate(app.src, app.dst, element);
    app.dst = app.dst - app.src;
  }

  cv::imshow(Config::WINDOW_NAME, app.dst);
}

int main(int argc, char ** argv)
{
  // Parse command line arguments
  cv::CommandLineParser parser(argc, argv, "{@input | horse.png | input image}");
  app.src = cv::imread(cv::samples::findFile(parser.get<std::string>("@input")), cv::IMREAD_COLOR);

  if (app.src.empty()) {
    std::cout << "Could not open or find the image!\n" << std::endl;
    std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
    return EXIT_FAILURE;
  }

  // Create the display window
  cv::namedWindow(Config::WINDOW_NAME, cv::WINDOW_AUTOSIZE);

  // Create trackbars for interactive control
  cv::createTrackbar(Config::TRACKBAR_OPERATOR, Config::WINDOW_NAME,
    nullptr, Config::MAX_OPERATOR, morphological_contours);
  cv::createTrackbar(Config::TRACKBAR_ELEMENT, Config::WINDOW_NAME,
    nullptr, Config::MAX_ELEM, morphological_contours);
  cv::createTrackbar(Config::TRACKBAR_KERNEL, Config::WINDOW_NAME,
    nullptr, Config::MAX_KERNEL_SIZE, morphological_contours);

  // Set initial kernel size to 1
  cv::setTrackbarPos(Config::TRACKBAR_KERNEL, Config::WINDOW_NAME, 1);

  // Apply initial operation
  morphological_contours(0, nullptr);

  cv::waitKey(0);

  return EXIT_SUCCESS;
}
