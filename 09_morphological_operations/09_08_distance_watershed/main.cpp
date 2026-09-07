/**
 * @file main.cpp
 * @brief Distance transform + marker-based watershed segmentation
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - The distance transform: each pixel gets its distance to the background
 * - Extracting one marker per object from the distance-transform peaks
 * - cv::watershed(): flooding from the markers to separate TOUCHING objects
 *
 * The problem it solves: with the tools seen so far (threshold in 09_01,
 * connected components in 09_02), coins that touch each other become ONE
 * single blob. Watershed treats the image as a topographic relief and
 * "floods" it from one seed (marker) per object; where two floods meet, a
 * watershed line is drawn -- splitting the blob into its real objects.
 *
 * Pipeline (the classic recipe):
 *   1. Binarize (Otsu, 09_01)
 *   2. Clean noise with morphological opening (11_02)
 *   3. Sure BACKGROUND = dilation of the blob (11_01)
 *   4. Sure FOREGROUND = high peaks of the distance transform
 *   5. Unknown region = background minus foreground
 *   6. Label the sure regions (connectedComponents, 09_02) -> markers
 *   7. cv::watershed() resolves the unknown region
 */

#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>

namespace Config
{
constexpr int OPENING_ITERATIONS = 2;    // Noise removal strength
constexpr int DILATE_ITERATIONS = 3;     // Growth of the sure-background band
constexpr double FG_THRESHOLD = 0.5;     // Fraction of the max distance that
                                         // counts as "surely inside an object"
}

int main(int argc, char ** argv)
{
  // Load image
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/coins.jpg | Input file}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  const std::string image_path = parser.get<std::string>("@input");
  const cv::Mat src = cv::imread(cv::samples::findFile(image_path, false), cv::IMREAD_COLOR);

  if (src.empty()) {
    std::cerr << "Error: Could not load image '" << image_path << "'" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "=== Distance Transform + Watershed ===" << std::endl;

  // ========================================
  // Steps 1-2: binarize and clean
  // ========================================
  cv::Mat gray, binary;
  cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
  cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

  const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
  cv::Mat cleaned;
  cv::morphologyEx(binary, cleaned, cv::MORPH_OPEN, kernel,
                   cv::Point(-1, -1), Config::OPENING_ITERATIONS);

  // ========================================
  // Step 3: sure background
  // ========================================
  // Dilating the blob grows it outwards: everything OUTSIDE the dilated
  // region is guaranteed to be background
  cv::Mat sure_background;
  cv::dilate(cleaned, sure_background, kernel, cv::Point(-1, -1),
             Config::DILATE_ITERATIONS);

  // ========================================
  // Step 4: sure foreground via the distance transform
  // ========================================
  // distanceTransform replaces each white pixel with its Euclidean distance
  // to the nearest black pixel. Object CENTERS get the highest values --
  // even when two objects touch, each keeps its own peak. Thresholding at a
  // fraction of the maximum keeps one island per object.
  cv::Mat distance;
  cv::distanceTransform(cleaned, distance, cv::DIST_L2, 5);

  double max_distance;
  cv::minMaxLoc(distance, nullptr, &max_distance);

  cv::Mat sure_foreground;
  cv::threshold(distance, sure_foreground, Config::FG_THRESHOLD * max_distance,
                255, cv::THRESH_BINARY);
  sure_foreground.convertTo(sure_foreground, CV_8U);

  // ========================================
  // Step 5: unknown region (to be decided by watershed)
  // ========================================
  cv::Mat unknown;
  cv::subtract(sure_background, sure_foreground, unknown);

  // ========================================
  // Step 6: markers (one label per sure-foreground island)
  // ========================================
  // connectedComponents labels background as 0 and islands as 1..N.
  // watershed's convention differs: 0 means "unknown, decide for me", so we
  // shift every label +1 (background becomes 1) and then zero the unknown band.
  cv::Mat markers;
  const int num_objects = cv::connectedComponents(sure_foreground, markers) - 1;
  markers += 1;
  markers.setTo(0, unknown);

  std::cout << "Objects detected (foreground islands): " << num_objects << std::endl;

  // ========================================
  // Step 7: watershed
  // ========================================
  // Floods the "relief" of the color image from the markers; pixels where
  // two basins meet are set to -1 (the watershed lines between objects)
  cv::watershed(src, markers);

  // ========================================
  // Visualization
  // ========================================
  // Distance transform, normalized for display
  cv::Mat distance_display;
  cv::normalize(distance, distance_display, 0, 1, cv::NORM_MINMAX);

  // Paint each segmented object with a deterministic distinct hue
  cv::Mat result = cv::Mat::zeros(src.size(), CV_8UC3);
  std::vector<cv::Vec3b> palette(num_objects + 2);
  for (int i = 0; i < static_cast<int>(palette.size()); ++i) {
    const cv::Mat hsv_color(1, 1, CV_8UC3,
                            cv::Scalar(i * 180 / static_cast<int>(palette.size()), 200, 255));
    cv::Mat bgr_color;
    cv::cvtColor(hsv_color, bgr_color, cv::COLOR_HSV2BGR);
    palette[i] = bgr_color.at<cv::Vec3b>(0, 0);
  }

  cv::Mat boundaries = src.clone();
  for (int y = 0; y < markers.rows; ++y) {
    for (int x = 0; x < markers.cols; ++x) {
      const int label = markers.at<int>(y, x);
      if (label == -1) {
        // Watershed line: the frontier found between touching objects
        boundaries.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
      } else if (label > 1) {  // 1 is the background label
        result.at<cv::Vec3b>(y, x) = palette[label % palette.size()];
      }
    }
  }

  cv::imshow("1. Original", src);
  cv::imshow("2. Binary (Otsu) + opening", cleaned);
  cv::imshow("3. Distance transform", distance_display);
  cv::imshow("4. Sure foreground (markers source)", sure_foreground);
  cv::imshow("5. Watershed regions", result);
  cv::imshow("6. Watershed lines on original", boundaries);

  std::cout << "\nCompare windows 2 and 5: touching coins that formed a single"
            << std::endl;
  std::cout << "blob in the binary image are now separated objects." << std::endl;
  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
