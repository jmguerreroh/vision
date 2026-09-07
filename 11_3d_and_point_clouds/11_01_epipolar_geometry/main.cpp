/**
 * @file main.cpp
 * @brief Epipolar geometry: fundamental matrix, epipolar lines and epipole
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - Matching two views with ORB + ratio test (Chapter 8)
 * - cv::findFundamentalMat() with RANSAC: F estimated from correspondences
 *   alone, without knowing anything about the two cameras
 * - cv::computeCorrespondEpilines(): the line the homologous point must be on
 * - Measuring the constraint instead of trusting it: the distance from every
 *   inlier to its own epipolar line, and the value of x'^T F x
 *
 * This is the example of the epipolar geometry section, and it opens the
 * chapter because it justifies everything that comes after it: it is what
 * explains WHY the disparity search of 11_02 can afford to scan a single row.
 * Run it twice:
 *
 *   ./11_01_epipolar_geometry                          (unrectified pair)
 *   ./11_01_epipolar_geometry ../../data/aloeL.jpg ../../data/aloeR.jpg
 *
 * With the first pair the epipolar lines come out tilted and converge on the
 * epipole; with the second, already rectified, they come out horizontal and
 * the mean tilt collapses to about zero degrees. That difference, printed as a
 * number, is the whole point of rectifying.
 *
 * @note Plain OpenCV: no contrib modules needed.
 */

#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>   // findFundamentalMat, computeCorrespondEpilines
#include <opencv2/features2d.hpp>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace Config
{
constexpr int NUM_FEATURES = 4000;      // ORB keypoints per image
constexpr float RATIO = 0.75f;          // Lowe's ratio test
constexpr double RANSAC_THRESHOLD = 1.0;  // Max distance to the epiline, px
constexpr double RANSAC_CONFIDENCE = 0.99;
constexpr int NUM_DRAWN = 10;           // Correspondences drawn, out of all
const char * WINDOW_NAME = "Epipolar geometry: left | right";
}

/**
 * @brief Matches two images with ORB and the ratio test
 *
 * The ratio test keeps a match only when the best candidate is clearly better
 * than the second best. It is the cheapest way to throw away the matches that
 * come from repeated texture, which is exactly what would poison F.
 */
void matchPair(
  const cv::Mat & left, const cv::Mat & right,
  std::vector<cv::Point2f> & points_left,
  std::vector<cv::Point2f> & points_right)
{
  cv::Ptr<cv::ORB> orb = cv::ORB::create(Config::NUM_FEATURES);
  std::vector<cv::KeyPoint> keypoints_left, keypoints_right;
  cv::Mat descriptors_left, descriptors_right;
  orb->detectAndCompute(left, cv::noArray(), keypoints_left, descriptors_left);
  orb->detectAndCompute(right, cv::noArray(), keypoints_right, descriptors_right);

  // NORM_HAMMING because ORB descriptors are binary strings
  cv::BFMatcher matcher(cv::NORM_HAMMING);
  std::vector<std::vector<cv::DMatch>> knn;
  matcher.knnMatch(descriptors_left, descriptors_right, knn, 2);

  for (const std::vector<cv::DMatch> & pair : knn) {
    if (pair.size() == 2 && pair[0].distance < Config::RATIO * pair[1].distance) {
      points_left.push_back(keypoints_left[pair[0].queryIdx].pt);
      points_right.push_back(keypoints_right[pair[0].trainIdx].pt);
    }
  }
}

/**
 * @brief Distance from a point to a line given as (a, b, c), with a^2+b^2 = 1
 */
double distanceToLine(const cv::Vec3f & line, const cv::Point2f & point)
{
  const double norm = std::sqrt(line[0] * line[0] + line[1] * line[1]);
  return std::abs(line[0] * point.x + line[1] * point.y + line[2]) / norm;
}

/**
 * @brief The epipole of an image, as the null vector of F (or of F^T)
 *
 * Every epipolar line of the right image passes through e', and F e = 0 by
 * construction, so the epipole is the null space of the matrix. SVD gives it
 * directly: the last row of Vt is the right null vector.
 */
cv::Point2f epipoleOf(const cv::Mat & fundamental)
{
  cv::Mat w, u, vt;
  cv::SVD::compute(fundamental, w, u, vt);
  const cv::Mat null_vector = vt.row(2);          // smallest singular value
  const double x = null_vector.at<double>(0);
  const double y = null_vector.at<double>(1);
  const double z = null_vector.at<double>(2);     // homogeneous scale
  return cv::Point2f(static_cast<float>(x / z), static_cast<float>(y / z));
}

/**
 * @brief Draws one epipolar line across the whole width of an image
 */
void drawEpiline(cv::Mat & image, const cv::Vec3f & line, const cv::Scalar & color)
{
  // a*x + b*y + c = 0 evaluated at both borders of the image
  const int width = image.cols;
  if (std::abs(line[1]) < 1e-9) {           // vertical line: b = 0
    const int x = static_cast<int>(-line[2] / line[0]);
    cv::line(image, cv::Point(x, 0), cv::Point(x, image.rows), color, 1, cv::LINE_AA);
    return;
  }
  const cv::Point a(0, static_cast<int>(-line[2] / line[1]));
  const cv::Point b(width, static_cast<int>(-(line[2] + line[0] * width) / line[1]));
  cv::line(image, a, b, color, 1, cv::LINE_AA);
}

int main(int argc, char ** argv)
{
  const std::string keys =
    "{help h | | Show this help message}"
    "{@left  | ../../data/left.jpg  | Left image of the pair}"
    "{@right | ../../data/right.jpg | Right image of the pair}";
  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }

  const std::string left_path = parser.get<std::string>("@left");
  const std::string right_path = parser.get<std::string>("@right");

  // Without this, a malformed value (--frames=xyz) is reported by the parser
  // but the example carries on with the default, which is the hardest kind
  // of failure to diagnose. Note it validates values, not option names:
  // cv::CommandLineParser ignores an unknown option without complaining
  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }
  cv::Mat left = cv::imread(cv::samples::findFile(left_path, false), cv::IMREAD_COLOR);
  cv::Mat right = cv::imread(cv::samples::findFile(right_path, false), cv::IMREAD_COLOR);
  if (left.empty() || right.empty()) {
    std::cerr << "Error: could not load the pair (" << left_path << ", "
              << right_path << ")" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "=== Epipolar geometry ===" << std::endl;
  std::cout << "Pair: " << left_path << " / " << right_path << std::endl;

  // ========================================
  // Step 1: correspondences
  // ========================================
  std::vector<cv::Point2f> points_left, points_right;
  matchPair(left, right, points_left, points_right);
  std::cout << "Matches after the ratio test: " << points_left.size() << std::endl;
  if (points_left.size() < 8) {
    // Eight is the minimum for the linear algorithm; RANSAC needs more
    std::cerr << "Error: not enough matches to estimate F" << std::endl;
    return EXIT_FAILURE;
  }

  // ========================================
  // Step 2: the fundamental matrix
  // ========================================
  // RANSAC checks each hypothesis with the epipolar constraint itself: a match
  // is an inlier when its point falls closer than RANSAC_THRESHOLD pixels to
  // the epipolar line the hypothesis predicts for it
  cv::Mat inlier_mask;
  cv::Mat fundamental = cv::findFundamentalMat(
    points_left, points_right, cv::FM_RANSAC,
    Config::RANSAC_THRESHOLD, Config::RANSAC_CONFIDENCE, inlier_mask);
  if (fundamental.empty() || fundamental.rows != 3) {
    std::cerr << "Error: F could not be estimated (degenerate configuration?)"
              << std::endl;
    return EXIT_FAILURE;
  }

  // F is only defined up to scale (x'^T F x = 0 does not change if F is
  // multiplied by anything), and findFundamentalMat returns it with whatever
  // scale the estimation produced. Normalising makes the printed matrix and
  // the algebraic residual below comparable between one pair and the next.
  // Epipolar lines, distances and epipoles do not depend on this scale
  fundamental /= cv::norm(fundamental);

  std::vector<cv::Point2f> inliers_left, inliers_right;
  for (size_t i = 0; i < points_left.size(); ++i) {
    if (inlier_mask.at<uchar>(static_cast<int>(i))) {
      inliers_left.push_back(points_left[i]);
      inliers_right.push_back(points_right[i]);
    }
  }
  std::cout << "Inliers accepted by RANSAC: " << inliers_left.size()
            << " of " << points_left.size() << std::endl;
  std::cout << "F =\n" << fundamental << std::endl;
  // F must have rank 2, and the honest way to show it is the singular
  // values: the third one has to be negligible against the first
  cv::Mat singular_values;
  cv::SVD::compute(fundamental, singular_values);
  const double s1 = singular_values.at<double>(0);
  const double s3 = singular_values.at<double>(2);
  std::cout << "Singular values of F: " << s1 << ", "
            << singular_values.at<double>(1) << ", " << s3
            << "  (s3/s1 = " << s3 / s1 << ", so rank 2)" << std::endl;

  // ========================================
  // Step 3: epipolar lines
  // ========================================
  // whichImage = 1 -> the points are in image 1 and the lines belong to image 2
  std::vector<cv::Vec3f> lines_in_right, lines_in_left;
  cv::computeCorrespondEpilines(inliers_left, 1, fundamental, lines_in_right);
  cv::computeCorrespondEpilines(inliers_right, 2, fundamental, lines_in_left);

  // ========================================
  // Step 4: measure the constraint
  // ========================================
  double sum_distance = 0.0, sum_algebraic = 0.0, sum_tilt = 0.0;
  for (size_t i = 0; i < inliers_left.size(); ++i) {
    // Symmetric distance: the point of each image against the line the other
    // image dictates for it
    sum_distance += 0.5 * (distanceToLine(lines_in_right[i], inliers_right[i]) +
      distanceToLine(lines_in_left[i], inliers_left[i]));

    // The constraint itself, x'^T F x, which should be zero. It is an
    // ALGEBRAIC residual: it has no units and it only means something once F
    // is normalised, which is why the distance in pixels above is the number
    // to trust
    const cv::Mat x = (cv::Mat_<double>(3, 1) << inliers_left[i].x, inliers_left[i].y, 1.0);
    const cv::Mat xp = (cv::Mat_<double>(3, 1) << inliers_right[i].x, inliers_right[i].y, 1.0);
    sum_algebraic += std::abs(cv::Mat(xp.t() * fundamental * x).at<double>(0));

    // Tilt of the epipolar line, folded into [0, 90]: a line has no
    // direction, so 158 degrees and 22 degrees are the same tilt.
    // 0 degrees means the line IS a row of the image
    double tilt = std::atan2(-lines_in_right[i][0], lines_in_right[i][1]) *
      180.0 / CV_PI;
    tilt = std::abs(tilt);
    if (tilt > 90.0) {
      tilt = 180.0 - tilt;
    }
    sum_tilt += tilt;
  }
  const double n = static_cast<double>(inliers_left.size());
  std::cout << "Mean distance point-to-epiline: " << sum_distance / n << " px"
            << std::endl;
  std::cout << "Mean |x'^T F x|: " << sum_algebraic / n << std::endl;
  std::cout << "Mean tilt of the epipolar lines: " << sum_tilt / n << " deg"
            << std::endl;

  // ========================================
  // Step 5: the epipole
  // ========================================
  // F e = 0 gives the epipole of the LEFT image; F^T e' = 0, that of the right
  const cv::Point2f epipole_left = epipoleOf(fundamental);
  const cv::Point2f epipole_right = epipoleOf(fundamental.t());
  std::cout << "Left epipole:  (" << epipole_left.x << ", " << epipole_left.y << ")"
            << (cv::Rect(cv::Point(), left.size()).contains(epipole_left) ?
    "  inside the image" : "  OUTSIDE the image") << std::endl;
  std::cout << "Right epipole: (" << epipole_right.x << ", " << epipole_right.y << ")"
            << (cv::Rect(cv::Point(), right.size()).contains(epipole_right) ?
    "  inside the image" : "  OUTSIDE the image") << std::endl;
  std::cout << "Both epipoles sit on the direction of the baseline, far from "
    "the centre\nof the frame. Getting them near the middle of the image takes "
    "a very\nconvergent pair, with each camera looking towards the other."
            << std::endl;

  // ========================================
  // Visualization
  // ========================================
  // Same color for a point and for the line of its homologue, so the pairing
  // can be followed by eye from one image to the other
  cv::RNG rng(7);            // seeded: the colors are the same on every run
  const int step = std::max<int>(1, static_cast<int>(inliers_left.size()) / Config::NUM_DRAWN);
  for (size_t i = 0; i < inliers_left.size(); i += step) {
    const cv::Scalar color(rng.uniform(0, 200), rng.uniform(0, 200), rng.uniform(80, 255));
    drawEpiline(left, lines_in_left[i], color);
    drawEpiline(right, lines_in_right[i], color);
    cv::circle(left, inliers_left[i], 5, color, cv::FILLED, cv::LINE_AA);
    cv::circle(right, inliers_right[i], 5, color, cv::FILLED, cv::LINE_AA);
  }

  cv::Mat canvas(std::max(left.rows, right.rows), left.cols + right.cols,
    left.type(), cv::Scalar::all(0));
  left.copyTo(canvas(cv::Rect(0, 0, left.cols, left.rows)));
  right.copyTo(canvas(cv::Rect(left.cols, 0, right.cols, right.rows)));
  cv::imshow(Config::WINDOW_NAME, canvas);

  std::cout << "\nEvery point of the left image lies on the line of its own "
    "color\nin the right one, and the other way round.\nPress any key to exit..."
            << std::endl;
  cv::waitKey(0);
  return EXIT_SUCCESS;
}
