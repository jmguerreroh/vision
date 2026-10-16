/**
 * @file main.cpp
 * @brief Morphological contours - internal, external and gradient, by hand
 * @author José Miguel Guerrero Hernández
 * @note This program demonstrates how to extract contours using morphological operations.
 *       Internal contours: original - eroded image
 *       External contours: dilated image - original
 *       Gradient: dilated - eroded, that is, the two contours together. It is
 *       computed by hand too, and checked against cv::MORPH_GRADIENT.
 */

#include <cstdlib>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

// Configuration constants
namespace Config
{
constexpr int MAX_OPERATOR = 2;
constexpr int MAX_ELEM = 2;
constexpr int MAX_KERNEL_SIZE = 21;
const char * WINDOW_NAME = "Morphological Contours Demo";
const char * TRACKBAR_OPERATOR = "Operator: 0: In - 1: Out - 2: Gradient";
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
 * @brief Number of pixels with at least one nonzero channel
 *
 * @note cv::countNonZero only accepts one channel, and the image is loaded
 *       in color: a pixel counts if any of its channels is set.
 */
int countPixels(const cv::Mat & img)
{
  cv::Mat zero;
  cv::inRange(img, cv::Scalar::all(0), cv::Scalar::all(0), zero);
  return static_cast<int>(img.total()) - cv::countNonZero(zero);
}

/**
 * @brief Callback function for trackbar events - extracts morphological contours
 *
 * @note The two parameters are imposed by the OpenCV trackbar callback
 *       signature and are deliberately left unnamed: there is nothing to
 *       document about them, and the state comes from the global app object.
 *
 * Morphological contour extraction:
 *   Internal contour: src - erode(src) -> shows the inner edge of objects
 *   External contour: dilate(src) - src -> shows the outer edge of objects
 *   Gradient: dilate(src) - erode(src) -> both edges at once
 *
 * The gradient is the sum of the other two, pixel by pixel and in every
 * channel. Their pixel counts only add up on a binary image, where the two
 * contours never share a pixel; this one has intermediate levels, so a
 * pixel can belong to both and the check compares values, not counts.
 */
void morphologicalContours(int, void *)
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

  cv::Mat eroded, dilated;
  cv::erode(app.src, eroded, element);
  cv::dilate(app.src, dilated, element);

  // Internal contour: the pixels that erosion would remove
  cv::Mat internal = app.src - eroded;
  // External contour: the pixels that dilation would add
  cv::Mat external = dilated - app.src;

  if (morph_operator == 0) {
    app.dst = internal;
  } else if (morph_operator == 1) {
    app.dst = external;
  } else {
    // Gradient by hand: dilated minus eroded, the two contours together
    app.dst = dilated - eroded;

    // Two checks, both must give 0: the gradient is the sum of the two
    // contours, and it is the same one OpenCV computes with MORPH_GRADIENT
    cv::Mat reference;
    cv::morphologyEx(app.src, reference, cv::MORPH_GRADIENT, element);
    std::cout << "gradient vs internal + external: "
              << countPixels(app.dst != internal + external) << " px differ; "
              << "gradient vs MORPH_GRADIENT: "
              << countPixels(app.dst != reference) << " px differ" << std::endl;
  }

  cv::imshow(Config::WINDOW_NAME, app.dst);
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
    nullptr, Config::MAX_OPERATOR, morphologicalContours);
  cv::createTrackbar(Config::TRACKBAR_ELEMENT, Config::WINDOW_NAME,
    nullptr, Config::MAX_ELEM, morphologicalContours);
  cv::createTrackbar(Config::TRACKBAR_KERNEL, Config::WINDOW_NAME,
    nullptr, Config::MAX_KERNEL_SIZE, morphologicalContours);

  // Set initial kernel size to 1
  cv::setTrackbarPos(Config::TRACKBAR_KERNEL, Config::WINDOW_NAME, 1);

  // Apply initial operation
  morphologicalContours(0, nullptr);

  cv::waitKey(0);

  return EXIT_SUCCESS;
}
