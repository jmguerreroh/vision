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
 */

#include <cstdlib>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include <iostream>

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
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

  return dst;
}

int main(int argc, char ** argv)
{
  // Load input image
  const std::string image_path = argc >= 2 ? argv[1] : "RGB.jpg";
  cv::Mat src = cv::imread(cv::samples::findFile(image_path), cv::IMREAD_COLOR);

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
  cv::Mat binaryInv = applyThreshold(gray, FIXED_THRESH, cv::THRESH_BINARY_INV, "BINARY_INV");
  cv::Mat trunc = applyThreshold(gray, FIXED_THRESH, cv::THRESH_TRUNC, "TRUNC");
  cv::Mat toZero = applyThreshold(gray, FIXED_THRESH, cv::THRESH_TOZERO, "TOZERO");
  cv::Mat toZeroInv = applyThreshold(gray, FIXED_THRESH, cv::THRESH_TOZERO_INV, "TOZERO_INV");

  // Automatic threshold methods (threshold value is computed automatically)
  // OTSU: Best for bimodal histograms (two distinct peaks)
  cv::Mat otsu = applyThreshold(gray, 0, cv::THRESH_BINARY | cv::THRESH_OTSU, "OTSU");

  // TRIANGLE: Best for unimodal histograms with tail
  cv::Mat triangle = applyThreshold(gray, 0, cv::THRESH_BINARY | cv::THRESH_TRIANGLE, "TRIANGLE");

  // ========================================
  // Visualization: Create Comparison Grid
  // ========================================

  // Create labeled original image for grid
  cv::Mat originalBGR;
  cv::resize(src, originalBGR, gray.size());
  cv::putText(originalBGR, "ORIGINAL", cv::Point(10, 25),
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

  // Create comparison grid using hconcat/vconcat
  // hconcat: horizontal concatenation (places images side by side in a row)
  // vconcat: vertical concatenation (stacks rows on top of each other)
  // Row 1: Original | Binary | Binary_Inv | Otsu
  // Row 2: Trunc    | ToZero | ToZero_Inv | Triangle
  cv::Mat row1, row2, comparison;
  cv::hconcat(std::vector<cv::Mat>{originalBGR, binary, binaryInv, otsu}, row1);
  cv::hconcat(std::vector<cv::Mat>{trunc, toZero, toZeroInv, triangle}, row2);
  cv::vconcat(row1, row2, comparison);

  // Resize for display if too large
  if (comparison.cols > 1600) {
    double scale = 1600.0 / comparison.cols;
    cv::resize(comparison, comparison, cv::Size(), scale, scale);
  }

  // Display results
  cv::imshow("Threshold Methods Comparison (8 methods)", comparison);
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
