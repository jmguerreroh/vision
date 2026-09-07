/**
 * @file main.cpp
 * @brief Histogram matching (histogram specification) using OpenCV
 * @author José Miguel Guerrero Hernández
 *
 * Histogram matching transforms a source image so that its histogram
 * approximates the histogram of a reference image. It generalizes histogram
 * equalization: instead of aiming at a uniform distribution, it aims at the
 * distribution of another image.
 *
 * OpenCV has no matchHistograms() function, but building one takes three
 * steps that are already available:
 *   1. cv::calcHist  -> histogram of each image
 *   2. the CDFs      -> and, for every source level, the reference level whose
 *                       CDF is closest to it. That mapping is the LUT
 *   3. cv::LUT       -> apply it, one table lookup per pixel
 *
 * The quality of the result is measured with cv::compareHist: the four
 * metrics must all agree that the output is closer to the reference than the
 * input was.
 *
 * @see https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html
 */

#include <cstdlib>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <string>

namespace Config
{
constexpr int LEVELS = 256;               // 8-bit images
constexpr float RANGE_MIN = 0.0f;
constexpr float RANGE_MAX = 256.0f;
}

struct ComparisonMethod
{
  int id;
  const char * name;
  bool higher_is_better;
};

const std::array<ComparisonMethod, 4> COMPARISON_METHODS = {{
  {cv::HISTCMP_CORREL, "Correlation", true},
  {cv::HISTCMP_CHISQR, "Chi-Square", false},
  {cv::HISTCMP_INTERSECT, "Intersection", true},
  {cv::HISTCMP_BHATTACHARYYA, "Bhattacharyya", false}
}};

/**
 * @brief Normalized histogram of a single-channel 8-bit image
 * @param gray Input image
 * @return 256x1 CV_32F histogram whose bins add up to 1
 */
cv::Mat normalizedHistogram(const cv::Mat & gray)
{
  const float range[] = {Config::RANGE_MIN, Config::RANGE_MAX};
  const float * hist_range = range;
  const int channels = 0;

  cv::Mat hist;
  cv::calcHist(&gray, 1, &channels, cv::Mat(), hist, 1, &Config::LEVELS, &hist_range);
  hist /= static_cast<float>(gray.total());   // so images of different size compare
  return hist;
}

/**
 * @brief Cumulative distribution function of a normalized histogram
 * @param hist Normalized histogram (bins add up to 1)
 * @return Vector of 256 values growing from 0 to 1
 */
std::vector<double> cumulative(const cv::Mat & hist)
{
  std::vector<double> cdf(Config::LEVELS);
  double running = 0.0;
  for (int i = 0; i < Config::LEVELS; ++i) {
    running += hist.at<float>(i);
    cdf[i] = running;
  }
  return cdf;
}

/**
 * @brief Build the matching look-up table between two images
 * @param source Image to transform
 * @param reference Image whose tonal distribution is to be copied
 * @return 1x256 CV_8U table T such that T[r] is the reference level whose CDF
 *         is closest to the CDF of the source level r
 *
 * The continuous formulation is T = CDF_ref^-1 . CDF_src, but discrete CDFs
 * are step functions with no exact inverse, so the closest level is taken.
 */
cv::Mat buildMatchingLUT(const cv::Mat & source, const cv::Mat & reference)
{
  const std::vector<double> cdf_src = cumulative(normalizedHistogram(source));
  const std::vector<double> cdf_ref = cumulative(normalizedHistogram(reference));

  cv::Mat lut(1, Config::LEVELS, CV_8U);
  for (int r = 0; r < Config::LEVELS; ++r) {
    int best = 0;
    double best_distance = std::abs(cdf_ref[0] - cdf_src[r]);
    for (int z = 1; z < Config::LEVELS; ++z) {
      const double distance = std::abs(cdf_ref[z] - cdf_src[r]);
      if (distance < best_distance) {
        best_distance = distance;
        best = z;
      }
    }
    lut.at<uchar>(r) = static_cast<uchar>(best);
  }
  return lut;
}

/**
 * @brief Number of distinct grey levels present in an image
 *
 * The matching, like the equalization, merges levels: it can only reorder
 * what is already there, never create new values.
 */
int countLevels(const cv::Mat & gray)
{
  std::array<bool, Config::LEVELS> present = {};
  for (int y = 0; y < gray.rows; ++y) {
    const uchar * row = gray.ptr<uchar>(y);
    for (int x = 0; x < gray.cols; ++x) {present[row[x]] = true;}
  }
  int total = 0;
  for (bool seen : present) {total += seen ? 1 : 0;}
  return total;
}

int main(int argc, char ** argv)
{
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/aero1.jpg | Input file}"
    "{@reference | ../../data/home.jpg | Reference image whose histogram is matched}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  const std::string source_path = parser.get<std::string>("@input");
  const std::string reference_path = parser.get<std::string>("@reference");

  // Without this, a malformed value (--frames=xyz) is reported by the parser
  // but the example carries on with the default, which is the hardest kind
  // of failure to diagnose. Note it validates values, not option names:
  // cv::CommandLineParser ignores an unknown option without complaining
  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }

  const cv::Mat source = cv::imread(cv::samples::findFile(source_path, false),
      cv::IMREAD_GRAYSCALE);
  const cv::Mat reference = cv::imread(cv::samples::findFile(reference_path, false),
      cv::IMREAD_GRAYSCALE);

  if (source.empty() || reference.empty()) {
    std::cerr << "Error: Could not load images" << std::endl;
    std::cerr << "Usage: " << argv[0] << " [source] [reference]" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "=== Histogram Matching Demo ===" << std::endl;
  std::cout << "Source:    " << source_path << " (" << source.cols << "x"
            << source.rows << ")" << std::endl;
  std::cout << "Reference: " << reference_path << " (" << reference.cols << "x"
            << reference.rows << ")" << std::endl;

  // Build the table and apply it. cv::LUT does the whole image with one
  // lookup per pixel, so the cost does not depend on how the table was built.
  const cv::Mat lut = buildMatchingLUT(source, reference);
  cv::Mat result;
  cv::LUT(source, lut, result);

  cv::imshow("Source", source);
  cv::imshow("Reference", reference);
  cv::imshow("Result (matched to reference)", result);

  // Did it work? Every metric should improve, each in its own direction
  const cv::Mat hist_src = normalizedHistogram(source);
  const cv::Mat hist_ref = normalizedHistogram(reference);
  const cv::Mat hist_res = normalizedHistogram(result);

  constexpr int table_width = 58;
  std::cout << "\nDistance to the reference histogram:\n";
  std::cout << std::string(table_width, '-') << std::endl;
  std::cout << std::left << std::setw(16) << "Metric"
            << std::setw(14) << "Source"
            << std::setw(14) << "Result"
            << "Better" << std::endl;
  std::cout << std::string(table_width, '-') << std::endl;

  for (const ComparisonMethod & method : COMPARISON_METHODS) {
    const double before = cv::compareHist(hist_src, hist_ref, method.id);
    const double after = cv::compareHist(hist_res, hist_ref, method.id);
    std::cout << std::left << std::setw(16) << method.name
              << std::setw(14) << std::fixed << std::setprecision(4) << before
              << std::setw(14) << after
              << (method.higher_is_better ? "higher" : "lower") << std::endl;
  }
  std::cout << std::string(table_width, '-') << std::endl;

  // The price paid: levels are merged, never created
  std::cout << "\nDistinct grey levels: source " << countLevels(source)
            << " -> result " << countLevels(result)
            << " (reference has " << countLevels(reference) << ")" << std::endl;
  std::cout << "Matching redistributes the levels that exist, it does not "
            << "invent new ones." << std::endl;

  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
