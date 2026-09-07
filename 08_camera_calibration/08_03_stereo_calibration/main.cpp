/**
 * @file main.cpp
 * @brief Stereo calibration and rectification of a camera pair
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - Calibrating each camera of a stereo pair (same procedure as 08_01)
 * - cv::stereoCalibrate(): the rigid transform (R, T) BETWEEN the two cameras
 * - cv::stereoRectify() + cv::remap(): reprojecting both images so that their
 *   rows are aligned
 *
 * Why rectify. The disparity algorithms of chapter 08 look for the match of a
 * pixel by scanning the SAME ROW of the other image. That only works if the
 * two image planes are coplanar and their rows correspond, which no real
 * mounting achieves. Rectification builds two virtual cameras that do satisfy
 * it, and remaps both images onto them.
 *
 * The example measures the gain instead of just showing it: the chessboard
 * corners are in known correspondence, so the average vertical distance
 * between the same corner in both images can be computed before and after
 * rectifying. That number is exactly what makes the 1D search legal.
 *
 * @note The square size of this dataset is not documented, so it is set to
 *       1.0 and every distance comes out in "square sides". Measure your own
 *       board and the baseline will come out in millimeters.
 */

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace Config
{
const cv::Size BOARD_SIZE(9, 6);          // Inner corners of the pattern
constexpr float SQUARE_SIZE = 1.0f;       // Side of a square (see note above)
constexpr int LINE_STEP = 60;             // Spacing of the guide lines drawn
}

/**
 * @brief Average vertical distance between corresponding corners
 * @param left Corners detected in the left image
 * @param right Corners detected in the right image
 * @return Mean |y_left - y_right| in pixels
 *
 * In a rectified pair this value is nearly zero: that is the whole point of
 * rectification, and what allows the disparity search to be one-dimensional.
 */
double rowMismatch(
  const std::vector<cv::Point2f> & left, const std::vector<cv::Point2f> & right)
{
  double total = 0.0;
  for (std::size_t i = 0; i < left.size(); i++) {
    total += std::abs(left[i].y - right[i].y);
  }
  return total / static_cast<double>(left.size());
}

/**
 * @brief Detect and refine the chessboard corners of one image
 * @return true if the whole pattern was found
 */
bool findCorners(const cv::Mat & gray, std::vector<cv::Point2f> & corners)
{
  if (!cv::findChessboardCorners(gray, Config::BOARD_SIZE, corners)) {
    return false;
  }
  cv::cornerSubPix(
    gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
    cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001));
  return true;
}

/**
 * @brief Draw the same horizontal lines over both images and join them
 */
cv::Mat drawPair(const cv::Mat & left, const cv::Mat & right)
{
  cv::Mat pair;
  cv::hconcat(left, right, pair);
  for (int y = Config::LINE_STEP; y < pair.rows; y += Config::LINE_STEP) {
    cv::line(pair, cv::Point(0, y), cv::Point(pair.cols, y),
             cv::Scalar(0, 140, 255), 1);
  }
  return pair;
}

int main(int argc, char ** argv)
{
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@images | ../../data/left??.jpg | Glob of the LEFT images (the right ones "
    "are found by replacing left with right)}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }

  // ========================================
  // Step 1: the same detection as 08_01, but on BOTH cameras
  // ========================================
  // A view is only useful if the pattern is complete in the two images: the
  // point of stereo calibration is to relate what one camera sees with what
  // the other sees AT THE SAME TIME.
  std::vector<std::string> left_files;
  cv::glob(parser.get<std::string>("@images"), left_files, false);
  if (left_files.empty()) {
    std::cerr << "No stereo images found" << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<cv::Point3f> board_3d;
  for (int i = 0; i < Config::BOARD_SIZE.height; i++) {
    for (int j = 0; j < Config::BOARD_SIZE.width; j++) {
      board_3d.push_back(
        cv::Point3f(j * Config::SQUARE_SIZE, i * Config::SQUARE_SIZE, 0));
    }
  }

  std::vector<std::vector<cv::Point3f>> object_points;
  std::vector<std::vector<cv::Point2f>> left_points, right_points;
  std::vector<std::string> used_left, used_right;
  cv::Size image_size;

  for (const std::string & left_file : left_files) {
    std::string right_file = left_file;
    const std::size_t pos = right_file.rfind("left");
    if (pos == std::string::npos) {
      continue;
    }
    right_file.replace(pos, 4, "right");

    const cv::Mat left_gray = cv::imread(left_file, cv::IMREAD_GRAYSCALE);
    const cv::Mat right_gray = cv::imread(right_file, cv::IMREAD_GRAYSCALE);
    if (left_gray.empty() || right_gray.empty()) {
      continue;
    }
    image_size = left_gray.size();

    std::vector<cv::Point2f> left_corners, right_corners;
    if (!findCorners(left_gray, left_corners) ||
      !findCorners(right_gray, right_corners))
    {
      std::cerr << "Pattern incomplete in the pair " << left_file << std::endl;
      continue;
    }

    object_points.push_back(board_3d);
    left_points.push_back(left_corners);
    right_points.push_back(right_corners);
    used_left.push_back(left_file);
    used_right.push_back(right_file);
  }

  std::cout << "=== Stereo calibration and rectification ===" << std::endl;
  std::cout << "Usable pairs: " << object_points.size() << std::endl;
  if (object_points.size() < 3) {
    std::cerr << "Not enough pairs" << std::endl;
    return EXIT_FAILURE;
  }

  // ========================================
  // Step 2: calibrate each camera on its own
  // ========================================
  // The intrinsics of each camera do not depend on the other one, so they are
  // estimated separately -- exactly as in 08_01. Doing it first lets the
  // stereo step concentrate on the only thing left: the transform between them.
  cv::Mat K1, D1, K2, D2;
  std::vector<cv::Mat> rvecs, tvecs;
  const double rms_left = cv::calibrateCamera(
    object_points, left_points, image_size, K1, D1, rvecs, tvecs);
  const double rms_right = cv::calibrateCamera(
    object_points, right_points, image_size, K2, D2, rvecs, tvecs);
  std::cout << "Reprojection error: left " << rms_left
            << " px, right " << rms_right << " px" << std::endl;

  // ========================================
  // Step 3: stereo calibration -- R and T between the cameras
  // ========================================
  // CALIB_FIX_INTRINSIC keeps K1, D1, K2 and D2 as they are and estimates
  // ONLY the relative pose. It is the recommended option when each camera has
  // already been calibrated well: fewer unknowns, more stable result.
  cv::Mat R, T, E, F;
  const double rms_stereo = cv::stereoCalibrate(
    object_points,          // 3D points of the board, per view
    left_points,            // Corners seen by the left camera
    right_points,           // Corners seen by the right camera
    K1, D1, K2, D2,         // Intrinsics of each camera
    image_size,
    R, T,                   // Output: rotation and translation between cameras
    E, F,                   // Output: essential and fundamental matrices
    cv::CALIB_FIX_INTRINSIC);

  // The norm of T is the BASELINE: how far apart the two cameras are. It is
  // the parameter that decides the trade-off of the pair -- a longer baseline
  // measures depth more accurately but shrinks the region both cameras see.
  const double baseline = cv::norm(T);
  // Angle of the relative rotation, from the trace of R
  const double angle = std::acos((cv::trace(R)[0] - 1.0) / 2.0) * 180.0 / CV_PI;

  std::cout << "Stereo reprojection error: " << rms_stereo << " px" << std::endl;
  std::cout << "Baseline |T| = " << baseline << " square sides" << std::endl;
  std::cout << "Relative rotation: " << angle << " degrees" << std::endl;
  std::cout << "T = " << T.t() << std::endl;

  // ========================================
  // Step 4: rectification
  // ========================================
  // stereoRectify does NOT touch the images: it computes the rotations R1 and
  // R2 that turn each camera into a virtual one, and the projection matrices
  // P1 and P2 of those virtual cameras. Q is the matrix that later turns a
  // disparity map into 3D coordinates (chapter 08).
  cv::Mat R1, R2, P1, P2, Q;
  cv::stereoRectify(
    K1, D1, K2, D2, image_size, R, T,
    R1, R2, P1, P2, Q,
    cv::CALIB_ZERO_DISPARITY,   // Keep the principal points aligned
    0);                         // alpha = 0: crop until no invalid pixel is left

  // The maps are computed once and reused on every frame, as in 08_01. Here
  // R is no longer the identity: it carries the rectifying rotation, so the
  // same remap corrects distortion AND rectifies in a single interpolation.
  cv::Mat map1x, map1y, map2x, map2y;
  cv::initUndistortRectifyMap(K1, D1, R1, P1, image_size, CV_32FC1, map1x, map1y);
  cv::initUndistortRectifyMap(K2, D2, R2, P2, image_size, CV_32FC1, map2x, map2y);

  // ========================================
  // Step 5: measure the gain
  // ========================================
  double before = 0.0;
  int counted = 0;
  for (std::size_t i = 0; i < left_points.size(); i++) {
    before += rowMismatch(left_points[i], right_points[i]);
  }
  before /= static_cast<double>(left_points.size());

  double after = 0.0;
  for (std::size_t i = 0; i < used_left.size(); i++) {
    cv::Mat left_gray = cv::imread(used_left[i], cv::IMREAD_GRAYSCALE);
    cv::Mat right_gray = cv::imread(used_right[i], cv::IMREAD_GRAYSCALE);
    cv::remap(left_gray, left_gray, map1x, map1y, cv::INTER_LINEAR);
    cv::remap(right_gray, right_gray, map2x, map2y, cv::INTER_LINEAR);

    std::vector<cv::Point2f> left_corners, right_corners;
    if (findCorners(left_gray, left_corners) &&
      findCorners(right_gray, right_corners))
    {
      after += rowMismatch(left_corners, right_corners);
      counted++;
    }
  }
  after /= std::max(counted, 1);

  std::cout << "\nVertical mismatch of the same corner in both images:"
            << std::endl;
  std::cout << "  before rectifying: " << before << " px" << std::endl;
  std::cout << "  after rectifying:  " << after << " px" << std::endl;
  std::cout << "That is what turns the disparity search into a 1D scan along"
            << std::endl;
  std::cout << "the row, which is what chapter 08 assumes." << std::endl;

  // ========================================
  // Visualization
  // ========================================
  cv::Mat left_img = cv::imread(used_left[0], cv::IMREAD_COLOR);
  cv::Mat right_img = cv::imread(used_right[0], cv::IMREAD_COLOR);
  cv::Mat left_rect, right_rect;
  cv::remap(left_img, left_rect, map1x, map1y, cv::INTER_LINEAR);
  cv::remap(right_img, right_rect, map2x, map2y, cv::INTER_LINEAR);

  cv::imshow("Original pair (lines do NOT match)", drawPair(left_img, right_img));
  cv::imshow("Rectified pair (lines match)", drawPair(left_rect, right_rect));

  // Save the calibration: this is what a stereo pipeline needs
  cv::FileStorage fs("stereo_calibration.yml", cv::FileStorage::WRITE);
  fs << "image_width" << image_size.width << "image_height" << image_size.height;
  fs << "K1" << K1 << "D1" << D1 << "K2" << K2 << "D2" << D2;
  fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2;
  fs << "P1" << P1 << "P2" << P2 << "Q" << Q;
  fs.release();
  std::cout << "\nWritten to stereo_calibration.yml" << std::endl;

  std::cout << "Press any key to exit..." << std::endl;
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
