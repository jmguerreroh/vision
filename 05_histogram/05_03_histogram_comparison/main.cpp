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

#include <cstdlib>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
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

// Comparison methods paired with their display names. Using the named
// cv::HISTCMP_* constants (instead of looping over raw indices 0..3) makes
// the code robust to any change in the numeric values of the enum.
struct ComparisonMethod
{
  int id;             // cv::HISTCMP_* constant passed to compareHist()
  const char * name;  // Human-readable label for the results table
};

const std::array<ComparisonMethod, 4> COMPARISON_METHODS = {{
  {cv::HISTCMP_CORREL, "Correlation"},
  {cv::HISTCMP_CHISQR, "Chi-Square"},
  {cv::HISTCMP_INTERSECT, "Intersection"},
  {cv::HISTCMP_BHATTACHARYYA, "Bhattacharyya"}
}};

/**
 * @brief Calculate normalized HS histogram from BGR image
 * @param bgr_image Input BGR image
 * @return Normalized 2D histogram (Hue x Saturation)
 */
cv::Mat calculateHSHistogram(const cv::Mat & bgr_image)
{
  cv::Mat hsv, hist;
  cv::cvtColor(bgr_image, hsv, cv::COLOR_BGR2HSV);

  // Define histogram parameters
  const int hist_size[] = {Config::H_BINS, Config::S_BINS};
  const float h_range[] = {Config::H_MIN, Config::H_MAX};
  const float s_range[] = {Config::S_MIN, Config::S_MAX};
  const float * ranges[] = {h_range, s_range};

  // cv::calcHist(images, nimages, channels, mask, hist, dims, histSize,
  //              ranges, uniform, accumulate)
  //   dims = 2: 2D histogram (Hue x Saturation = 50x60 = 3000 bins)
  //   A 2D histogram captures the relationship between H and S,
  //   providing a better "color signature" than 1D histograms
  cv::calcHist(&hsv, 1, Config::HS_CHANNELS, cv::Mat(), hist, 2, hist_size, ranges, true, false);
  cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
  return hist;
}

/**
 * @brief Draw 1D Hue histogram visualization
 * @param bgr_image Input BGR image
 * @param title Label to show on the histogram
 * @return Histogram visualization image
 */
cv::Mat drawHueHistogram(const cv::Mat & bgr_image, const std::string & title)
{
  cv::Mat hsv;
  cv::cvtColor(bgr_image, hsv, cv::COLOR_BGR2HSV);

  // Calculate Hue histogram
  const float hue_range[] = {Config::H_MIN, Config::H_MAX};
  const float * hist_range = hue_range;
  const int channels[] = {0};  // Hue channel only

  cv::Mat hist;
  cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 1, &Config::HUE_BINS, &hist_range);
  cv::normalize(hist, hist, 0, Config::HIST_HEIGHT, cv::NORM_MINMAX);

  // Create colored histogram image
  const int img_height = Config::HIST_HEIGHT + Config::HIST_TEXT_PADDING;
  cv::Mat hist_image(img_height, Config::HIST_WIDTH, CV_8UC3, cv::Scalar(255, 255, 255));

  // Draw bars with Hue colors
  const int bin_width = Config::HIST_WIDTH / Config::HUE_BINS;

  for (int h = 0; h < Config::HUE_BINS; h++) {
    const int bar_height = cvRound(hist.at<float>(h));

    // Convert Hue to BGR color for visualization
    cv::Mat hsv_color(1, 1, CV_8UC3, cv::Scalar(h, Config::HUE_SATURATION, Config::HUE_VALUE));
    cv::Mat bgr_color;
    cv::cvtColor(hsv_color, bgr_color, cv::COLOR_HSV2BGR);
    const cv::Vec3b color = bgr_color.at<cv::Vec3b>(0, 0);

    cv::rectangle(hist_image,
                  cv::Point(h * bin_width, Config::HIST_HEIGHT - bar_height),
                  cv::Point((h + 1) * bin_width, Config::HIST_HEIGHT),
                  cv::Scalar(color[0], color[1], color[2]), cv::FILLED);
  }

  // Add title
  cv::putText(hist_image, title, cv::Point(5, Config::HIST_HEIGHT + 20),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

  return hist_image;
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
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@data | ../../data/ | Directory with the images to compare}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }

  // Load test images
  const std::string base_path = parser.get<std::string>("@data");

  // Without this, a malformed value (--frames=xyz) is reported by the parser
  // but the example carries on with the default, which is the hardest kind
  // of failure to diagnose. Note it validates values, not option names:
  // cv::CommandLineParser ignores an unknown option without complaining
  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }
  const cv::Mat img_base = cv::imread(
    cv::samples::findFile(base_path + "Histogram_Comparison_Source_0.jpg", false));
  const cv::Mat img_test1 = cv::imread(
    cv::samples::findFile(base_path + "Histogram_Comparison_Source_1.jpg", false));
  const cv::Mat img_test2 = cv::imread(
    cv::samples::findFile(base_path + "Histogram_Comparison_Source_2.jpg", false));

  if (img_base.empty() || img_test1.empty() || img_test2.empty()) {
    std::cerr << "Error: Could not load test images" << std::endl;
    std::cerr << "Expected files in: " << base_path << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "=== Histogram Comparison Demo ===" << std::endl;

  // Calculate histograms for comparison
  const cv::Mat hist_base = calculateHSHistogram(img_base);
  const cv::Mat hist_test1 = calculateHSHistogram(img_test1);
  const cv::Mat hist_test2 = calculateHSHistogram(img_test2);

  // Create half-image histogram (lower half of base)
  const cv::Mat img_half = img_base(cv::Range(img_base.rows / 2, img_base.rows), cv::Range::all());
  const cv::Mat hist_half = calculateHSHistogram(img_half);

  // Draw Hue histograms for visualization
  cv::Mat hist_viz_base = drawHueHistogram(img_base, "Base");
  cv::Mat hist_viz_half = drawHueHistogram(img_half, "Half (similar)");
  cv::Mat hist_viz_test1 = drawHueHistogram(img_test1, "Test1");
  cv::Mat hist_viz_test2 = drawHueHistogram(img_test2, "Test2 (different)");

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
  cv::hconcat(hist_viz_base, hist_viz_half, row1);
  cv::hconcat(hist_viz_test1, hist_viz_test2, row2);
  cv::vconcat(row1, row2, histComparison);

  // Display images
  cv::imshow("Base Image", img_base);
  cv::imshow("Base - Lower Half", img_half);
  cv::imshow("Test Image 1", img_test1);
  cv::imshow("Test Image 2", img_test2);
  cv::imshow("Histogram Comparison (Hue)", histComparison);

  // Prepare references for comparison
  const std::vector<cv::Mat> references = {hist_base, hist_half, hist_test1, hist_test2};
  const std::vector<std::string> ref_names = {"Base (self)", "Half", "Test1", "Test2"};

  // Print comparison table
  constexpr int table_width = 65;
  std::cout << "\nComparison Results (Base vs Others):\n";
  std::cout << std::string(table_width, '-') << std::endl;
  std::cout << std::left << std::setw(15) << "Method"
            << std::setw(12) << "Self"
            << std::setw(12) << "Half"
            << std::setw(12) << "Test1"
            << std::setw(12) << "Test2" << std::endl;
  std::cout << std::string(table_width, '-') << std::endl;

  for (const ComparisonMethod & method : COMPARISON_METHODS) {
    const std::vector<double> scores = compareWithReferences(hist_base, references, method.id);

    std::cout << std::left << std::setw(15) << method.name;
    for (double score : scores) {
      std::cout << std::setw(12) << std::fixed << std::setprecision(4) << score;
    }
    std::cout << std::endl;
  }

  std::cout << std::string(table_width, '-') << std::endl;
  std::cout << "\nInterpretation:" << std::endl;
  std::cout << "  Correlation & Intersection: Higher = Better match" << std::endl;
  std::cout << "  Chi-Square & Bhattacharyya: Lower = Better match" << std::endl;
  std::cout << "\nVisualization: Similar histograms have similar color distributions" << std::endl;

  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
