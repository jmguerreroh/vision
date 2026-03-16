/**
 * @file main.cpp
 * @brief Camera calibration demo - computes intrinsic parameters and lens distortion correction
 * @author José Miguel Guerrero Hernández
 *
 * Images from: https://github.com/niconielsen32/ComputerVision
 */

#include <cstdlib>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

/**
 * @brief Displays side-by-side comparison of distorted and undistorted images
 * @param title Window title
 * @param distorted Original distorted image
 * @param undistorted Corrected undistorted image
 */
void compare_images(
  const std::string & title, const cv::Mat & distorted, const cv::Mat & undistorted)
{
  cv::Mat dist_copy, undist_copy;
  cv::resize(distorted, dist_copy, cv::Size(distorted.cols / 2, distorted.rows / 2));
  cv::putText(
    dist_copy, "Original", cv::Point(dist_copy.cols - 100, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5,
    cv::Scalar(0, 0, 0), 2);
  cv::resize(undistorted, undist_copy,
    cv::Size(undistorted.cols / 2, undistorted.rows / 2));
  cv::putText(
    undist_copy, "Undistorted",
    cv::Point(undist_copy.cols - 100, 15),
    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);

  cv::Mat concat;
  cv::hconcat(dist_copy, undist_copy, concat);
  cv::imshow(title, concat);
  cv::waitKey(400);
}

int main(int argc, char ** argv)
{
  (void)argc;
  (void)argv;

  // Get filenames
  std::vector<cv::String> file_names;
  cv::glob("../../data/calibration_images/Image*.png", file_names, false);

  if (file_names.empty()) {
    std::cerr << "No calibration images found" << std::endl;
    return EXIT_FAILURE;
  }

  // Pattern size inner corners (cols - width, rows - height)
  cv::Size chess_board_size(24, 17);
  cv::Size square_size(15, 15); // mm

  // Images data
  cv::Mat original, grayscale, undistorted;
  cv::Size frame_size;

  // Generate world coordinates for 3D points assuming Z=0. The board has 25 x 18 fields with a size of 15x15mm
  std::vector<cv::Point3f> chess_board_3d_corners;
  for (int i = 0; i < chess_board_size.height; i++) {
    for (int j = 0; j < chess_board_size.width; j++) {
      chess_board_3d_corners.push_back(
        cv::Point3f(j * square_size.width, i * square_size.height, 0));
    }
  }

  // 2D points (image points). Each image has a vector of 2D points
  std::vector<std::vector<cv::Point2f>> chess_board_2d;
  // 3D points (object points). Each image has a vector of 3D points
  std::vector<std::vector<cv::Point3f>> chess_board_3d;
  // File names for images where the chessboard pattern was successfully found
  std::vector<cv::String> calibrated_files;

  // Detect feature points
  for (auto const & f : file_names) {
    std::cout << std::string(f) << std::endl;

    // Read in the image
    original = cv::imread(f);
    if (original.empty()) {
      std::cerr << "Error: could not read " << f << std::endl;
      continue;
    }
    frame_size = original.size();
    cv::cvtColor(original, grayscale, cv::COLOR_BGR2GRAY);

    // Find chessboard corners
    std::vector<cv::Point2f> corners;
    bool pattern_found = cv::findChessboardCorners(
      grayscale,                      // Input: Grayscale image
      chess_board_size,               // Input: Size of the chessboard pattern (rows, cols)
      corners,                        // Output: Detected 2D corner points
      cv::CALIB_CB_ADAPTIVE_THRESH |  // Input: Optional flags for optimization
      cv::CALIB_CB_NORMALIZE_IMAGE |
      cv::CALIB_CB_FAST_CHECK
    );

    // If pattern found
    if (pattern_found) {
      // Refine corner accuracy
      cv::cornerSubPix(
        grayscale,        // Input grayscale image (single-channel)
        corners,          // Initial corner coordinates to refine
        cv::Size(11, 11), // Half-size of the search window (11x11 pixels)
        cv::Size(-1, -1), // Half-size of the dead region (-1,-1 means no dead region)
        cv::TermCriteria(
          cv::TermCriteria::EPS |
          cv::TermCriteria::MAX_ITER, // Stop when accuracy or max iterations reached
          30,             // Maximum number of iterations
          0.1             // Accuracy threshold for stopping
      ));

      // Save 2D and 3D points
      chess_board_2d.push_back(corners);
      chess_board_3d.push_back(chess_board_3d_corners);
      // Save file name for display and reprojection error analysis
      calibrated_files.push_back(f);

      // Display
      cv::drawChessboardCorners(original, chess_board_size, corners, pattern_found);
      cv::resize(original, original, cv::Size(original.cols / 2, original.rows / 2));
      cv::imshow("chessboard detection", original);
      cv::waitKey(400);
    }
  }

  // Intrinsic camera matrix
  //    fx: focal length in x direction
  //    fy: focal length in y direction
  //    cx: principal point x
  //    cy: principal point y
  cv::Matx33f K(cv::Matx33f::eye());

  // distortion coefficients (k1, k2, p1, p2, k3):
  //    k1: radial distortion coefficient first order
  //    k2: radial distortion coefficient second order
  //    p1: tangential distortion coefficient horizontal deviation
  //    p2: tangential distortion coefficient vertical deviation
  //    k3: radial distortion coefficient third order
  cv::Vec<float, 5> dist_coeffs(0, 0, 0, 0, 0);

  // rvects and tvects are the rotation and translation vectors for each view
  std::vector<cv::Mat> rvecs, tvecs;

  // Flags for calibration
  int flags = cv::CALIB_FIX_ASPECT_RATIO |    // Keeps the aspect ratio fixed
    cv::CALIB_FIX_K3 |                        // Fixes k3 distortion coefficient
    cv::CALIB_ZERO_TANGENT_DIST |             // Assumes zero tangential distortion
    cv::CALIB_FIX_PRINCIPAL_POINT;            // Fixes the principal point at the center

  if (chess_board_2d.size() < 2) {
    std::cerr << "Not enough images for calibration" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Calibrating..." << std::endl;
  // Calibrate the camera using the detected 2D and 3D points
  float error = cv::calibrateCamera(
    chess_board_3d,  // 3D points (object points) in the world coordinate system
    chess_board_2d,  // 2D points (in image plane) in the camera coordinate system
    frame_size,      // Image size (width, height)
    K,               // Output intrinsic matrix (3x3)
    dist_coeffs,     // Output distortion coefficients (radial & tangential)
    rvecs,           // Output rotation vectors (3x1) one for each view
    tvecs,           // Output translation vectors (3x1) one for each view
    flags            // Flags (optional)
  );

  std::cout << "Reprojection error = " << error << "\nK =\n"
            << K << "\ndist_coeffs =\n"
            << dist_coeffs << std::endl;

  // Get first image and apply lens correction using undistort
  // This method is not recommended for real-time applications
  cv::Mat first_img = cv::imread(calibrated_files[0], cv::IMREAD_COLOR);
  cv::undistort(first_img, undistorted, K, dist_coeffs);

  // Display
  compare_images("Comparison no RT", first_img, undistorted);

  // For real-time applications, we can use the remap method
  // Precompute lens correction interpolation
  cv::Mat map_x, map_y;
  cv::initUndistortRectifyMap(
    K,                  // Camera intrinsic matrix (3x3)
    dist_coeffs,        // Distortion coefficients (radial & tangential)
    cv::Matx33f::eye(), // Rectification matrix (identity if no rectification)
    K,                  // New intrinsic matrix after undistortion (can be modified)
    frame_size,         // Size of the output image (width, height)
    CV_32FC1,           // Type of the output maps (CV_32FC1 or CV_16SC2)
    map_x,              // Output map for x-coordinates
    map_y               // Output map for y-coordinates
  );

  // Show lens corrected images (only for images where the pattern was found)
  for (std::size_t i = 0; i < calibrated_files.size(); i++) {
    std::cout << std::string(calibrated_files[i]) << std::endl;

    original = cv::imread(calibrated_files[i], cv::IMREAD_COLOR);

    // Draw the 3D axes
    cv::drawFrameAxes(original, K, dist_coeffs, rvecs[i], tvecs[i], 120, 10);

    // Remap the image using the precomputed interpolation maps
    cv::remap(
      original,         // Input distorted image
      undistorted,      // Output undistorted image
      map_x,            // Precomputed x-coordinates map
      map_y,            // Precomputed y-coordinates map
      cv::INTER_LINEAR  // Interpolation method
    );

    // Display
    compare_images("Comparison RT", original, undistorted);
  }

  return EXIT_SUCCESS;
}
