/**
 * @file main.cpp
 * @brief Hit-or-Miss transform - detecting an exact pixel configuration
 * @author José Miguel Guerrero Hernández
 *
 * @note The Hit-or-Miss transform looks for an EXACT local configuration:
 *          two disjoint structuring elements, B1 with the pixels that must
 *          belong to the object and B2 with the pixels that must belong to
 *          the background.
 *
 *              A (*) B = (A erode B1) AND (Ac erode B2)
 *
 *          OpenCV packs both into a SINGLE signed kernel:
 *             +1 -> must be object
 *             -1 -> must be background
 *              0 -> don't care
 *
 *          Same skeleton as 09_01_erode_dilate, only the operation changes.
 *
 *          The example detects the four right-angle corners of a shape and
 *          lets the image be rotated, which is the fastest way to see the
 *          weak point of the method: the mask is RIGID. On the unrotated
 *          square each mask fires exactly once, on the right corner. Rotate
 *          it and the count goes haywire: the rasterised edges become
 *          staircases, some of their steps look exactly like the pattern,
 *          and the real corner of the square is eventually lost.
 */

#include <cstdlib>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>

// Configuration constants
namespace Config
{
constexpr int MAX_PATTERN = 3;
constexpr int MAX_ANGLE = 45;
const char * WINDOW_NAME = "Hit-or-Miss Demo";
const char * TRACKBAR_PATTERN = "Corner: 0 TL - 1 TR - 2 BL - 3 BR";
const char * TRACKBAR_ANGLE = "Rotation (deg)";
}

/**
 * @brief The four corner masks
 *
 * Top-left corner: the three neighbours above and the three to the left must
 * be background, and the 2x2 block that starts at the pixel must be object.
 * The other three are the same mask rotated by 90 degrees.
 */
const std::vector<cv::Mat> CORNER_KERNELS = {
  (cv::Mat_<schar>(3, 3) << -1, -1, -1, -1, 1, 1, -1, 1, 1),   // top-left
  (cv::Mat_<schar>(3, 3) << -1, -1, -1, 1, 1, -1, 1, 1, -1),   // top-right
  (cv::Mat_<schar>(3, 3) << -1, 1, 1, -1, 1, 1, -1, -1, -1),   // bottom-left
  (cv::Mat_<schar>(3, 3) << 1, 1, -1, 1, 1, -1, -1, -1, -1)    // bottom-right
};

const char * CORNER_NAMES[] = {"top-left", "top-right", "bottom-left", "bottom-right"};

/**
 * @brief Application state
 */
struct HitMissApp
{
  cv::Mat binary;   // Binary image: object in white
};

// Global app state (required for OpenCV callbacks)
HitMissApp app;

/**
 * @brief Rotate a binary image without introducing intermediate values
 * @param src Binary image
 * @param degrees Rotation angle
 * @return Rotated binary image
 *
 * INTER_NEAREST is mandatory here: any interpolation would produce grey
 * pixels along the edges, and the image would stop being binary.
 */
cv::Mat rotateBinary(const cv::Mat & src, int degrees)
{
  if (degrees == 0) {
    return src.clone();
  }

  const cv::Point2f centre(src.cols / 2.0f, src.rows / 2.0f);
  const cv::Mat rotation = cv::getRotationMatrix2D(centre, degrees, 1.0);

  cv::Mat rotated;
  cv::warpAffine(src, rotated, rotation, src.size(), cv::INTER_NEAREST);
  return rotated;
}

/**
 * @brief Callback function for trackbar events - runs the Hit-or-Miss transform
 */
void hitOrMiss(int, void *)
{
  const int pattern = cv::getTrackbarPos(Config::TRACKBAR_PATTERN, Config::WINDOW_NAME);
  const int angle = cv::getTrackbarPos(Config::TRACKBAR_ANGLE, Config::WINDOW_NAME);

  const cv::Mat binary = rotateBinary(app.binary, angle);

  // The whole operation is a single call: OpenCV reads the +1 entries as B1
  // and the -1 entries as B2. Only CV_8UC1 binary images are supported
  cv::Mat found;
  cv::morphologyEx(binary, found, cv::MORPH_HITMISS, CORNER_KERNELS[pattern]);

  const int matches = cv::countNonZero(found);

  // Draw the matches: a single pixel would be invisible, so each one is
  // marked with a circle on top of the shape
  cv::Mat display;
  cv::cvtColor(binary, display, cv::COLOR_GRAY2BGR);
  for (int y = 0; y < found.rows; ++y) {
    for (int x = 0; x < found.cols; ++x) {
      if (found.at<uchar>(y, x) != 0) {
        cv::circle(display, cv::Point(x, y), 12, cv::Scalar(0, 0, 255), 2);
      }
    }
  }

  cv::putText(display, std::string(CORNER_NAMES[pattern]) + ": " +
    std::to_string(matches) + " match(es)", cv::Point(10, 30),
    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

  cv::imshow(Config::WINDOW_NAME, display);
  std::cout << CORNER_NAMES[pattern] << " corner, " << angle << " deg: "
            << matches << " match(es)" << std::endl;
}

int main(int argc, char ** argv)
{
  // Parse command line arguments
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/shapes.png | Input file}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  const cv::Mat src = cv::imread(
    cv::samples::findFile(parser.get<std::string>("@input"), false), cv::IMREAD_GRAYSCALE);

  if (src.empty()) {
    std::cout << "Could not open or find the image!\n" << std::endl;
    std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
    return EXIT_FAILURE;
  }

  // The shapes are dark on a light background, so the object is what falls
  // BELOW the threshold. Hit-or-Miss needs a binary CV_8UC1 image
  cv::threshold(src, app.binary, 127, 255, cv::THRESH_BINARY_INV);

  // Report what each mask finds on the unrotated image
  std::cout << "=== Hit-or-Miss: right-angle corners ===" << std::endl;
  for (size_t i = 0; i < CORNER_KERNELS.size(); ++i) {
    cv::Mat found;
    cv::morphologyEx(app.binary, found, cv::MORPH_HITMISS, CORNER_KERNELS[i]);
    std::cout << "  " << CORNER_NAMES[i] << ": " << cv::countNonZero(found)
              << " match(es)" << std::endl;
  }
  std::cout << "\nEvery match is on the square. The circle, the triangle and the"
            << std::endl;
  std::cout << "diamond have no axis-aligned right angle, so they score zero:"
            << std::endl;
  std::cout << "Hit-or-Miss detects one configuration, not a 'corner' in general."
            << std::endl;
  std::cout << "\nMove the rotation trackbar to see the other side of that"
            << std::endl;
  std::cout << "rigidity: rotating the image turns every edge into a staircase,"
            << std::endl;
  std::cout << "steps of the CIRCLE start matching the corner pattern, and past"
            << std::endl;
  std::cout << "about 15 degrees the real corner of the square is not found at all."
            << std::endl;

  // Create the display window
  cv::namedWindow(Config::WINDOW_NAME, cv::WINDOW_AUTOSIZE);

  // Create trackbars for interactive control
  cv::createTrackbar(Config::TRACKBAR_PATTERN, Config::WINDOW_NAME,
    nullptr, Config::MAX_PATTERN, hitOrMiss);
  cv::createTrackbar(Config::TRACKBAR_ANGLE, Config::WINDOW_NAME,
    nullptr, Config::MAX_ANGLE, hitOrMiss);

  // Apply initial operation
  hitOrMiss(0, nullptr);

  cv::waitKey(0);
  return EXIT_SUCCESS;
}
