/**
 * @file main.cpp
 * @brief Image thresholding comparison: fixed vs Otsu vs adaptive methods
 * @author José Miguel Guerrero Hernández
 *
 * @note Thresholding converts grayscale images to binary images.
 *          This example compares different thresholding methods:
 *
 *          Fixed threshold methods (require manual threshold selection):
 *            - THRESH_BINARY:     pixel > thresh ? maxval : 0
 *            - THRESH_BINARY_INV: pixel > thresh ? 0 : maxval
 *            - THRESH_TRUNC:      pixel > thresh ? thresh : pixel
 *            - THRESH_TOZERO:     pixel > thresh ? pixel : 0
 *            - THRESH_TOZERO_INV: pixel > thresh ? 0 : pixel
 *
 *          Automatic threshold methods:
 *            - THRESH_OTSU: Calculates optimal threshold by minimizing
 *              intra-class variance (assumes bimodal histogram)
 *            - THRESH_TRIANGLE: Uses triangle algorithm (good for
 *              unimodal histograms with a tail)
 *
 *          Note: THRESH_OTSU and THRESH_TRIANGLE are flags that can be
 *          combined with THRESH_BINARY or THRESH_BINARY_INV using OR (|).
 *          The threshold value passed is ignored; the computed value is returned.
 *
 *          Adaptive threshold (local method):
 *            - Unlike all methods above, which use ONE global threshold,
 *              cv::adaptiveThreshold computes a DIFFERENT threshold for each
 *              pixel from its local neighborhood. This is the tool of choice
 *              when illumination is not uniform across the image.
 */

#include <algorithm>
#include <cstdlib>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include <iostream>

namespace
{
// The images this example works on are about 1400 px on the long side, and
// several windows at that size do not fit on a normal screen. The processing
// always runs at full resolution: only the copy sent to the screen is reduced,
// with INTER_AREA, which is the interpolation meant for shrinking
constexpr int MAX_DISPLAY_SIDE = 800;
// The comparison grid puts several panels side by side, so it allows more
// width than a single image: at 800 px each panel would be unreadable
constexpr int MAX_GRID_WIDTH = 1600;

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

/**
 * @brief Applies threshold and adds text label with threshold value
 * @param src Source grayscale image
 * @param thresh Threshold value (ignored for OTSU/TRIANGLE)
 * @param type Threshold type flag
 * @param label Text label to display
 * @return Thresholded image with label and BGR format for display
 */
cv::Mat applyThreshold(const cv::Mat & src, double thresh, int type, const std::string & label)
{
  cv::Mat dst;
  double computed = cv::threshold(src, dst, thresh, 255, type);

  // Convert to BGR for colored text on binary image
  cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR);

  // Display method name and threshold value
  // Cast to int for cleaner display (e.g., "127" instead of "127.000000")
  std::string text = label + " (T=" + std::to_string(static_cast<int>(computed)) + ")";
  cv::putText(dst, text, cv::Point(10, 25),
              cv::FONT_HERSHEY_SIMPLEX, fontFor(dst, 0.6), cv::Scalar(0, 255, 0), 2);

  return dst;
}

int main(int argc, char ** argv)
{
  // Load input image
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/coins.png | Input file}");
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

  // ========================================
  // Preprocessing
  // ========================================

  // Convert to grayscale (thresholding requires single-channel image)
  cv::Mat gray;
  cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

  // ========================================
  // Apply Different Thresholding Methods
  // ========================================

  // Fixed threshold value for manual methods
  const int FIXED_THRESH = 127;

  // Apply different thresholding methods
  // Fixed threshold methods (all use the same threshold value: 127)
  cv::Mat binary = applyThreshold(gray, FIXED_THRESH, cv::THRESH_BINARY, "BINARY");
  cv::Mat binary_inv = applyThreshold(gray, FIXED_THRESH, cv::THRESH_BINARY_INV, "BINARY_INV");
  cv::Mat trunc = applyThreshold(gray, FIXED_THRESH, cv::THRESH_TRUNC, "TRUNC");
  cv::Mat to_zero = applyThreshold(gray, FIXED_THRESH, cv::THRESH_TOZERO, "TOZERO");
  cv::Mat to_zero_inv = applyThreshold(gray, FIXED_THRESH, cv::THRESH_TOZERO_INV, "TOZERO_INV");

  // Automatic threshold methods (threshold value is computed automatically)
  // OTSU: Best for bimodal histograms (two distinct peaks)
  cv::Mat otsu = applyThreshold(gray, 0, cv::THRESH_BINARY | cv::THRESH_OTSU, "OTSU");

  // TRIANGLE: Best for unimodal histograms with tail
  cv::Mat triangle = applyThreshold(gray, 0, cv::THRESH_BINARY | cv::THRESH_TRIANGLE, "TRIANGLE");

  // ADAPTIVE: local threshold, one value per pixel
  //
  // cv::adaptiveThreshold(src, dst, maxValue, adaptiveMethod, thresholdType,
  //                       blockSize, C)
  //   adaptiveMethod: how the local threshold is computed
  //     * ADAPTIVE_THRESH_MEAN_C:     mean of the blockSize x blockSize window
  //     * ADAPTIVE_THRESH_GAUSSIAN_C: Gaussian-weighted mean (smoother)
  //   blockSize: neighborhood size (odd, e.g. 11)
  //   C: constant subtracted from the local mean (fine-tunes sensitivity)
  //
  // With uneven illumination a global threshold loses entire regions to
  // black or white; the adaptive method keeps local structures everywhere.
  cv::Mat adaptive_raw;
  cv::adaptiveThreshold(gray, adaptive_raw, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv::THRESH_BINARY, 11, 2);
  cv::Mat adaptive;
  cv::cvtColor(adaptive_raw, adaptive, cv::COLOR_GRAY2BGR);
  cv::putText(adaptive, "ADAPTIVE (local)", cv::Point(10, 25),
              cv::FONT_HERSHEY_SIMPLEX, fontFor(adaptive, 0.6), cv::Scalar(0, 255, 0), 2);

  // ========================================
  // Visualization: Create Comparison Grid
  // ========================================

  // Create labeled original image for grid
  cv::Mat original_bgr;
  cv::resize(src, original_bgr, gray.size());
  cv::putText(original_bgr, "ORIGINAL", cv::Point(10, 25),
              cv::FONT_HERSHEY_SIMPLEX, fontFor(original_bgr, 0.6), cv::Scalar(0, 255, 0), 2);

  // Create comparison grid using hconcat/vconcat
  // hconcat: horizontal concatenation (places images side by side in a row)
  // vconcat: vertical concatenation (stacks rows on top of each other)
  // Row 1: Original | Binary  | Binary_Inv | Otsu     | Triangle
  // Row 2: Trunc    | ToZero  | ToZero_Inv | Adaptive | (Otsu again for reference)
  cv::Mat row1, row2, comparison;
  cv::hconcat(std::vector<cv::Mat>{original_bgr, binary, binary_inv, otsu, triangle}, row1);
  cv::hconcat(std::vector<cv::Mat>{trunc, to_zero, to_zero_inv, adaptive, otsu}, row2);
  cv::vconcat(row1, row2, comparison);

  // Resize for display if too large
  if (comparison.cols > MAX_GRID_WIDTH) {
    double scale = static_cast<double>(MAX_GRID_WIDTH) / comparison.cols;
    cv::resize(comparison, comparison, cv::Size(), scale, scale);
  }

  // Display results
  showFit("Threshold Methods Comparison (global + adaptive)", comparison);
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
