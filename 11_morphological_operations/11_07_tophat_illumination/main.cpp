/**
 * @file main.cpp
 * @brief Top-hat / black-hat: separating an object from its illumination
 * @author José Miguel Guerrero Hernández
 *
 * @note The problem this solves: global thresholding (10_01_threshold)
 *          assumes the object and the background are separated by ONE
 *          intensity value for the whole image. With uneven lighting that is
 *          false, and Otsu binarises half the picture instead of the ink.
 *
 *          Grayscale morphology fixes it:
 *            opening  = background WITHOUT the bright details smaller than SE
 *            closing  = background WITHOUT the dark details smaller than SE
 *            top-hat   = original - opening  -> small BRIGHT details
 *            black-hat = closing  - original -> small DARK details
 *
 *          The result no longer carries the lighting, because the lighting
 *          IS what the opening/closing kept. A global threshold then works.
 *
 *          The structuring element sets what counts as a detail: it must be
 *          LARGER than the objects to extract and smaller than the scale on
 *          which the illumination changes.
 */

#include <cstdlib>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

// Configuration constants
namespace Config
{
// The grid puts several panels side by side, so it allows more width than a
// single image: at 800 px each panel would be unreadable
constexpr int MAX_GRID_WIDTH = 1200;
constexpr int MAX_OPERATOR = 1;
constexpr int MAX_KERNEL_SIZE = 30;
constexpr int DEFAULT_KERNEL_SIZE = 12;   // 2*12+1 = 25 pixels
const char * WINDOW_NAME = "Top-hat Illumination Demo";
const char * TRACKBAR_OPERATOR = "Op: 0 BlackHat (dark ink) - 1 TopHat (bright)";
const char * TRACKBAR_KERNEL = "SE size: 2n +1";
}

/**
 * @brief Application state
 */
struct TopHatApp
{
  cv::Mat gray;         // Source image in grayscale
  cv::Mat otsu_direct;  // Otsu applied straight to the source
  double marked_direct = 0.0;
};

// Global app state (required for OpenCV callbacks)
TopHatApp app;

/**
 * @brief Percentage of non-zero pixels of a binary image
 */
double markedPercentage(const cv::Mat & binary)
{
  return 100.0 * cv::countNonZero(binary) / (binary.rows * binary.cols);
}

/**
 * @brief Put a caption on a panel of the comparison grid
 */
void label(cv::Mat & panel, const std::string & text)
{
  cv::putText(panel, text, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX,
    0.6, cv::Scalar(0, 255, 0), 2);
}

/**
 * @brief Callback function for trackbar events
 */
void topHat(int, void *)
{
  const int morph_operator = cv::getTrackbarPos(Config::TRACKBAR_OPERATOR, Config::WINDOW_NAME);
  const int morph_size = cv::getTrackbarPos(Config::TRACKBAR_KERNEL, Config::WINDOW_NAME);

  // An ellipse is the usual choice here: it has no privileged direction, so
  // it does not favour horizontal or vertical strokes
  const cv::Mat element = cv::getStructuringElement(
    cv::MORPH_ELLIPSE,
    cv::Size(2 * morph_size + 1, 2 * morph_size + 1));

  const int operation = (morph_operator == 0) ? cv::MORPH_BLACKHAT : cv::MORPH_TOPHAT;
  const char * operation_name = (morph_operator == 0) ? "black-hat" : "top-hat";

  cv::Mat corrected;
  cv::morphologyEx(app.gray, corrected, operation, element);

  // Both results are BRIGHT on a dark background, whatever the sign of the
  // original detail, so the same threshold sense works for the two
  cv::Mat otsu_corrected;
  const double threshold_corrected = cv::threshold(corrected, otsu_corrected, 0, 255,
      cv::THRESH_BINARY | cv::THRESH_OTSU);

  const double marked_corrected = markedPercentage(otsu_corrected);

  // Build the 2x2 comparison grid
  cv::Mat panels[4] = {app.gray.clone(), app.otsu_direct.clone(),
    corrected.clone(), otsu_corrected.clone()};
  for (cv::Mat & panel : panels) {
    cv::cvtColor(panel, panel, cv::COLOR_GRAY2BGR);
  }

  std::ostringstream direct_caption, corrected_caption;
  direct_caption << "Otsu direct: " << std::fixed << std::setprecision(1)
                 << app.marked_direct << "% marked";
  corrected_caption << "Otsu after: " << std::fixed << std::setprecision(1)
                    << marked_corrected << "% marked";

  label(panels[0], "Original");
  label(panels[1], direct_caption.str());
  label(panels[2], std::string(operation_name) + ", SE " +
    std::to_string(2 * morph_size + 1));
  label(panels[3], corrected_caption.str());

  cv::Mat row1, row2, grid;
  cv::hconcat(panels[0], panels[1], row1);
  cv::hconcat(panels[2], panels[3], row2);
  cv::vconcat(row1, row2, grid);

  if (grid.cols > Config::MAX_GRID_WIDTH) {
    const double scale = static_cast<double>(Config::MAX_GRID_WIDTH) / grid.cols;
    cv::resize(grid, grid, cv::Size(), scale, scale);
  }

  cv::imshow(Config::WINDOW_NAME, grid);

  std::cout << operation_name << ", SE " << 2 * morph_size + 1 << ": Otsu at "
            << threshold_corrected << " marks " << std::fixed << std::setprecision(1)
            << marked_corrected << "% (direct Otsu marked " << app.marked_direct
            << "%)" << std::endl;
}

int main(int argc, char ** argv)
{
  // Parse command line arguments
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/page_uneven.png | Input file}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  app.gray = cv::imread(
    cv::samples::findFile(parser.get<std::string>("@input"), false), cv::IMREAD_GRAYSCALE);

  // Without this, a malformed value (--frames=xyz) is reported by the parser
  // but the example carries on with the default, which is the hardest kind
  // of failure to diagnose. Note it validates values, not option names:
  // cv::CommandLineParser ignores an unknown option without complaining
  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }

  if (app.gray.empty()) {
    std::cout << "Could not open or find the image!\n" << std::endl;
    std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
    return EXIT_FAILURE;
  }

  // Reference result: Otsu straight on the source. The ink is dark, so the
  // object is what falls BELOW the threshold
  const double threshold_direct = cv::threshold(app.gray, app.otsu_direct, 0, 255,
      cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
  app.marked_direct = markedPercentage(app.otsu_direct);

  std::cout << "=== Uneven illumination and global thresholding ===" << std::endl;
  std::cout << "Otsu on the original: threshold " << threshold_direct << ", marks "
            << std::fixed << std::setprecision(1) << app.marked_direct
            << "% of the image" << std::endl;
  std::cout << "Anything far above the real amount of ink means Otsu has"
            << std::endl;
  std::cout << "binarised the SHADOW, not the writing." << std::endl;

  // Create the display window
  cv::namedWindow(Config::WINDOW_NAME, cv::WINDOW_AUTOSIZE);

  // Create trackbars for interactive control
  cv::createTrackbar(Config::TRACKBAR_OPERATOR, Config::WINDOW_NAME,
    nullptr, Config::MAX_OPERATOR, topHat);
  cv::createTrackbar(Config::TRACKBAR_KERNEL, Config::WINDOW_NAME,
    nullptr, Config::MAX_KERNEL_SIZE, topHat);
  cv::setTrackbarPos(Config::TRACKBAR_KERNEL, Config::WINDOW_NAME,
    Config::DEFAULT_KERNEL_SIZE);

  // Apply initial operation
  topHat(0, nullptr);

  cv::waitKey(0);
  return EXIT_SUCCESS;
}
