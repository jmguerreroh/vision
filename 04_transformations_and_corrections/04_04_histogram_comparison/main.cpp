/**
 * @file main.cpp
 * @brief Histogram comparison demonstration using OpenCV
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates histogram comparison methods:
 * - Correlation: measures linear correlation (1 = perfect match)
 * - Chi-Square: measures statistical difference (0 = identical)
 * - Intersection: measures overlap (higher = more similar)
 * - Bhattacharyya: measures distribution distance (0 = identical)
 *
 * @see https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <string>

// Histogram configuration for HS (Hue-Saturation) comparison
namespace Config
{
// 2D Histogram (Hue x Saturation)
constexpr int H_BINS = 50;                // Number of Hue bins
constexpr int S_BINS = 60;                // Number of Saturation bins
constexpr float H_MIN = 0.0f;             // Hue minimum (OpenCV range: 0-179)
constexpr float H_MAX = 180.0f;           // Hue maximum
constexpr float S_MIN = 0.0f;             // Saturation minimum
constexpr float S_MAX = 256.0f;           // Saturation maximum
constexpr int HS_CHANNELS[] = {0, 1};     // H and S channels

// 1D Histogram for visualization
constexpr int HIST_1D_SIZE = 256;
constexpr int HIST_WIDTH = 256;
constexpr int HIST_HEIGHT = 200;
constexpr int HIST_TEXT_PADDING = 30;

// Hue histogram specific
constexpr int HUE_BINS = 180;             // Hue range in OpenCV (0-179)
constexpr int HUE_SATURATION = 255;       // Full saturation for color display
constexpr int HUE_VALUE = 255;            // Full value for color display
}

/**
 * @brief Get comparison method name
 * @param method Method index (0-3)
 * @return Method name string
 */
std::string getMethodName(int method)
{
  static const std::array<std::string, 4> METHOD_NAMES = {
    "Correlation",
    "Chi-Square",
    "Intersection",
    "Bhattacharyya"
  };
  return METHOD_NAMES[method];
}

/**
 * @brief Calculate normalized HS histogram from BGR image
 * @param bgrImage Input BGR image
 * @return Normalized 2D histogram (Hue x Saturation)
 */
cv::Mat calculateHSHistogram(const cv::Mat & bgrImage)
{
  cv::Mat hsv, hist;
  cv::cvtColor(bgrImage, hsv, cv::COLOR_BGR2HSV);

  // Define histogram parameters
  const int histSize[] = {Config::H_BINS, Config::S_BINS};
  const float hRange[] = {Config::H_MIN, Config::H_MAX};
  const float sRange[] = {Config::S_MIN, Config::S_MAX};
  const float * ranges[] = {hRange, sRange};

  // cv::calcHist(images, nimages, channels, mask, hist, dims, histSize, ranges, uniform, accumulate)
  //   dims = 2: 2D histogram (Hue x Saturation = 50x60 = 3000 bins)
  //   A 2D histogram captures the relationship between H and S,
  //   providing a better "color signature" than 1D histograms
  cv::calcHist(&hsv, 1, Config::HS_CHANNELS, cv::Mat(), hist, 2, histSize, ranges, true, false);
  cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
  return hist;
}

/**
 * @brief Draw 1D Hue histogram visualization
 * @param bgrImage Input BGR image
 * @param title Label to show on the histogram
 * @return Histogram visualization image
 */
cv::Mat drawHueHistogram(const cv::Mat & bgrImage, const std::string & title)
{
  cv::Mat hsv;
  cv::cvtColor(bgrImage, hsv, cv::COLOR_BGR2HSV);

  // Calculate Hue histogram
  const float hueRange[] = {Config::H_MIN, Config::H_MAX};
  const float * histRange = hueRange;
  const int channels[] = {0};  // Hue channel only

  cv::Mat hist;
  cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 1, &Config::HUE_BINS, &histRange);
  cv::normalize(hist, hist, 0, Config::HIST_HEIGHT, cv::NORM_MINMAX);

  // Create colored histogram image
  const int imgHeight = Config::HIST_HEIGHT + Config::HIST_TEXT_PADDING;
  cv::Mat histImage(imgHeight, Config::HIST_WIDTH, CV_8UC3, cv::Scalar(255, 255, 255));

  // Draw bars with Hue colors
  const int binWidth = Config::HIST_WIDTH / Config::HUE_BINS;

  for (int h = 0; h < Config::HUE_BINS; h++) {
    const int barHeight = cvRound(hist.at<float>(h));

    // Convert Hue to BGR color for visualization
    cv::Mat hsvColor(1, 1, CV_8UC3, cv::Scalar(h, Config::HUE_SATURATION, Config::HUE_VALUE));
    cv::Mat bgrColor;
    cv::cvtColor(hsvColor, bgrColor, cv::COLOR_HSV2BGR);
    const cv::Vec3b color = bgrColor.at<cv::Vec3b>(0, 0);

    cv::rectangle(histImage,
                  cv::Point(h * binWidth, Config::HIST_HEIGHT - barHeight),
                  cv::Point((h + 1) * binWidth, Config::HIST_HEIGHT),
                  cv::Scalar(color[0], color[1], color[2]), cv::FILLED);
  }

  // Add title
  cv::putText(histImage, title, cv::Point(5, Config::HIST_HEIGHT + 20),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

  return histImage;
}

/**
 * @brief Compare histogram against multiple reference histograms
 * @param hist Histogram to compare
 * @param references Vector of reference histograms
 * @param method Comparison method (cv::HISTCMP_*)
 * @return Vector of comparison scores
 */
std::vector<double> compareWithReferences(
  const cv::Mat & hist,
  const std::vector<cv::Mat> & references,
  int method)
{
  std::vector<double> scores;
  for (const cv::Mat & ref : references) {
    scores.push_back(cv::compareHist(hist, ref, method));
  }
  return scores;
}

int main(int argc, char ** argv)
{
  (void)argc;
  (void)argv;

  // Load test images
  const std::string basePath = "../../data/";
  const cv::Mat imgBase = cv::imread(basePath + "Histogram_Comparison_Source_0.jpg");
  const cv::Mat imgTest1 = cv::imread(basePath + "Histogram_Comparison_Source_1.jpg");
  const cv::Mat imgTest2 = cv::imread(basePath + "Histogram_Comparison_Source_2.jpg");

  if (imgBase.empty() || imgTest1.empty() || imgTest2.empty()) {
    std::cerr << "Error: Could not load test images" << std::endl;
    std::cerr << "Expected files in: " << basePath << std::endl;
    return -1;
  }

  std::cout << "=== Histogram Comparison Demo ===" << std::endl;

  // Calculate histograms for comparison
  const cv::Mat histBase = calculateHSHistogram(imgBase);
  const cv::Mat histTest1 = calculateHSHistogram(imgTest1);
  const cv::Mat histTest2 = calculateHSHistogram(imgTest2);

  // Create half-image histogram (lower half of base)
  const cv::Mat imgHalf = imgBase(cv::Range(imgBase.rows / 2, imgBase.rows), cv::Range::all());
  const cv::Mat histHalf = calculateHSHistogram(imgHalf);

  // Draw Hue histograms for visualization
  cv::Mat histVizBase = drawHueHistogram(imgBase, "Base");
  cv::Mat histVizHalf = drawHueHistogram(imgHalf, "Half (similar)");
  cv::Mat histVizTest1 = drawHueHistogram(imgTest1, "Test1");
  cv::Mat histVizTest2 = drawHueHistogram(imgTest2, "Test2 (different)");

  // Create combined histogram comparison view
  // cv::hconcat(src1, src2, dst): Horizontal concatenation (side by side)
  //   Joins matrices horizontally: [A | B] - requires same height
  // cv::vconcat(src1, src2, dst): Vertical concatenation (stacked)
  //   Joins matrices vertically: [A]  - requires same width
  //                              [B]
  // Result layout:
  //   +------+------+
  //   | Base | Half |  ← row1
  //   +------+------+
  //   |Test1 |Test2 |  ← row2
  //   +------+------+
  cv::Mat histComparison;
  cv::Mat row1, row2;
  cv::hconcat(histVizBase, histVizHalf, row1);
  cv::hconcat(histVizTest1, histVizTest2, row2);
  cv::vconcat(row1, row2, histComparison);

  // Display images
  cv::imshow("Base Image", imgBase);
  cv::imshow("Base - Lower Half", imgHalf);
  cv::imshow("Test Image 1", imgTest1);
  cv::imshow("Test Image 2", imgTest2);
  cv::imshow("Histogram Comparison (Hue)", histComparison);

  // Prepare references for comparison
  const std::vector<cv::Mat> references = {histBase, histHalf, histTest1, histTest2};
  const std::vector<std::string> refNames = {"Base (self)", "Half", "Test1", "Test2"};

  // Print comparison table
  constexpr int tableWidth = 65;
  std::cout << "\nComparison Results (Base vs Others):\n";
  std::cout << std::string(tableWidth, '-') << std::endl;
  std::cout << std::left << std::setw(15) << "Method"
            << std::setw(12) << "Self"
            << std::setw(12) << "Half"
            << std::setw(12) << "Test1"
            << std::setw(12) << "Test2" << std::endl;
  std::cout << std::string(tableWidth, '-') << std::endl;

  for (int method = 0; method < 4; method++) {
    const std::vector<double> scores = compareWithReferences(histBase, references, method);

    std::cout << std::left << std::setw(15) << getMethodName(method);
    for (double score : scores) {
      std::cout << std::setw(12) << std::fixed << std::setprecision(4) << score;
    }
    std::cout << std::endl;
  }

  std::cout << std::string(tableWidth, '-') << std::endl;
  std::cout << "\nInterpretation:" << std::endl;
  std::cout << "  Correlation & Intersection: Higher = Better match" << std::endl;
  std::cout << "  Chi-Square & Bhattacharyya: Lower = Better match" << std::endl;
  std::cout << "\nVisualization: Similar histograms have similar color distributions" << std::endl;

  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return 0;
}
