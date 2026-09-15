/**
 * @file main.cpp
 * @brief Non-linear intensity transforms applied with a look-up table
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - Gamma correction: s = r^gamma, with r normalised to [0, 1]
 * - Logarithmic transform: s = log(1 + r), normalised so that s(1) = 1
 * - Look-up tables: why they are the way to apply any per-pixel function
 *
 * @note An 8-bit image has only 256 possible input values, so a per-pixel
 *       function can only return 256 different results. Evaluating it once per
 *       pixel repeats work: it is enough to evaluate it 256 times, store the
 *       results in a table and then replace each pixel by its entry. The cost
 *       stops depending on how expensive the function is, which is why the same
 *       code serves for a power and for a logarithm.
 *
 * @see https://docs.opencv.org/4.x/d2/de8/group__core__array.html
 */

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <functional>
#include <iostream>
#include <string>

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

namespace Config
{
// gamma < 1 opens up the shadows, gamma > 1 darkens the mid-tones
constexpr double GAMMA_BRIGHTEN = 0.45;
constexpr double GAMMA_DARKEN = 2.2;

constexpr int LEVELS = 256;        // 8-bit input: this is the whole domain
}

/**
 * @brief Builds a 1x256 look-up table by sampling a function of r in [0, 1]
 * @param curve Function to sample; it must map [0, 1] to [0, 1]
 * @return Table ready for cv::LUT
 *
 * saturate_cast keeps the result inside [0, 255] even if the curve overshoots.
 */
cv::Mat buildLut(const std::function<double(double)> & curve)
{
  cv::Mat lut(1, Config::LEVELS, CV_8U);
  for (int i = 0; i < Config::LEVELS; i++) {
    const double r = i / (Config::LEVELS - 1.0);
    lut.at<uchar>(i) = cv::saturate_cast<uchar>(curve(r) * 255.0);
  }
  return lut;
}

/**
 * @brief Applies a curve through a look-up table and shows the result
 * @param src Input image, 8-bit
 * @param curve Function to apply
 * @param title Window title
 */
void showTransform(
  const cv::Mat & src,
  const std::function<double(double)> & curve,
  const std::string & title)
{
  cv::Mat dst;
  cv::LUT(src, buildLut(curve), dst);
  showFit(title, dst);
}

/**
 * @brief Compares the cost of evaluating the curve per pixel and via a table
 * @param src Input image, 8-bit
 * @param gamma Exponent of the gamma correction used for the measurement
 *
 * Both produce the same image; what changes is how many times std::pow runs:
 * once per pixel, or once per intensity level.
 */
void compareCost(const cv::Mat & src, double gamma)
{
  cv::Mat per_pixel = src.clone();

  const double start_direct = static_cast<double>(cv::getTickCount());
  for (int y = 0; y < per_pixel.rows; y++) {
    uchar * row = per_pixel.ptr<uchar>(y);
    // Flat index over the row: it walks columns times channels, not columns
    for (int n = 0; n < per_pixel.cols * per_pixel.channels(); n++) {
      row[n] = cv::saturate_cast<uchar>(std::pow(row[n] / 255.0, gamma) * 255.0);
    }
  }
  const double direct_ms =
    (cv::getTickCount() - start_direct) / cv::getTickFrequency() * 1000.0;

  cv::Mat with_lut;
  const double start_lut = static_cast<double>(cv::getTickCount());
  cv::LUT(src, buildLut([gamma](double r) {return std::pow(r, gamma);}), with_lut);
  const double lut_ms =
    (cv::getTickCount() - start_lut) / cv::getTickFrequency() * 1000.0;

  const int pixels = src.rows * src.cols * src.channels();
  std::cout << "\n=== Cost of applying the same gamma ===" << std::endl;
  std::cout << "Pixel by pixel: " << pixels << " calls to std::pow, "
            << direct_ms << " ms" << std::endl;
  std::cout << "Look-up table:  " << Config::LEVELS << " calls to std::pow, "
            << lut_ms << " ms" << std::endl;
  std::cout << "Largest difference between both results: "
            << cv::norm(per_pixel, with_lut, cv::NORM_INF) << std::endl;
}

int main(int argc, char ** argv)
{
  // Load input image
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/building_facade.png | Input file}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  const std::string image_path = parser.get<std::string>("@input");

  // Without this, a malformed value (--frames=xyz) is reported by the parser
  // but the example carries on with the default, which is the hardest kind
  // of failure to diagnose. Note it validates values, not option names:
  // cv::CommandLineParser ignores an unknown option without complaining
  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }
  cv::Mat src = cv::imread(cv::samples::findFile(image_path, false), cv::IMREAD_COLOR);

  if (src.empty()) {
    std::cerr << "Error: Could not open or find the image!" << std::endl;
    std::cerr << "Path: " << image_path << std::endl;
    std::cerr << "Usage: " << argv[0] << " <Input image>" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "=== Intensity Transforms Demo ===" << std::endl;
  std::cout << "Image loaded: " << src.cols << "x" << src.rows << " pixels" << std::endl;

  showFit("Original", src);

  showTransform(src, [](double r) {return std::pow(r, Config::GAMMA_BRIGHTEN);},
    "Gamma 0.45 (brighter)");
  showTransform(src, [](double r) {return std::pow(r, Config::GAMMA_DARKEN);},
    "Gamma 2.2 (darker)");

  // log(1 + r) divided by log(2) so that the maximum stays at 1
  showTransform(src, [](double r) {return std::log1p(r) / std::log(2.0);},
    "Logarithmic");

  // the negative is also a per-pixel function, and the same table applies
  showTransform(src, [](double r) {return 1.0 - r;}, "Negative");

  compareCost(src, Config::GAMMA_BRIGHTEN);

  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
