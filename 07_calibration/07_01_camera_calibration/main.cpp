/**
 * Camera calibration demo sample
 * @author José Miguel Guerrero
 *
 * Images from: https://github.com/niconielsen32/ComputerVision
 */

#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void compare_images(std::string title, cv::Mat & distorted, cv::Mat & undistorted)
{
  cv::resize(distorted, distorted, cv::Size(distorted.cols / 2, distorted.rows / 2));
  cv::putText(
    distorted, "Original", cv::Point(distorted.cols - 100, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5,
    cv::Scalar(0, 0, 0), 2);
  cv::resize(undistorted, undistorted,
    cv::Size(undistorted.cols / 2, undistorted.rows / 2));
  cv::putText(
    undistorted, "Undistorted",
    cv::Point(undistorted.cols - 100, 15),
    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);

  cv::Mat concat;
  cv::hconcat(distorted, undistorted, concat);
  cv::imshow(title, concat);
  cv::waitKey(400);
}

int main(int argc, char ** argv)
{
  // Get filenames
  std::vector<cv::String> fileNames;
  cv::glob("../calibration_images/Image*.png", fileNames, false);

  // Pattern size inner corners (cols - width, rows - height)
  cv::Size chessBoardSize(24, 17);
  cv::Size squareSize(15, 15); // mm

  // Images data
  cv::Mat original, grayscale, undistorted;
  cv::Size frameSize;

  // Generate world coordinates for 3D points assuming Z=0. The board has 25 x 18 fields with a size of 15x15mm
  std::vector<cv::Point3f> chessBoard3Dcorners;
  for (int i = 0; i < chessBoardSize.height; i++) {
    for (int j = 0; j < chessBoardSize.width; j++) {
      chessBoard3Dcorners.push_back(cv::Point3f(j * squareSize.width, i * squareSize.height, 0));
    }
  }

  // 2D points (image points). Each image has a vector of 2D points
  std::vector<std::vector<cv::Point2f>> chessBoard2D;
  // 3D points (object points). Each image has a vector of 3D points
  std::vector<std::vector<cv::Point3f>> chessBoard3D;

  // Detect feature points
  std::size_t i = 0;
  for (auto const & f : fileNames) {
    std::cout << std::string(f) << std::endl;

    // Read in the image
    original = cv::imread(fileNames[i]);
    frameSize = original.size();
    cv::cvtColor(original, grayscale, cv::COLOR_RGB2GRAY);

    // Find chessboard corners
    std::vector<cv::Point2f> corners;
    bool patternFound = cv::findChessboardCorners(
      grayscale,                      // Input: Grayscale image
      chessBoardSize,                 // Input: Size of the chessboard pattern (rows, cols)
      corners,                // Output: Detected 2D corner points
      cv::CALIB_CB_ADAPTIVE_THRESH +  // Input: Optional flags for optimization
      cv::CALIB_CB_NORMALIZE_IMAGE +
      cv::CALIB_CB_FAST_CHECK
    );

    // If pattern found
    if (patternFound) {
      // Refine corner accuracy
      cv::cornerSubPix(
        grayscale,        // Input grayscale image (single-channel)
        corners,  // Initial corner coordinates to refine
        cv::Size(11, 11), // Half-size of the search window (11x11 pixels)
        cv::Size(-1, -1), // Half-size of the dead region (-1,-1 means no dead region)
        cv::TermCriteria(
          cv::TermCriteria::EPS +
          cv::TermCriteria::MAX_ITER, // Stop when accuracy or max iterations reached
          30,             // Maximum number of iterations
          0.1             // Accuracy threshold for stopping
      ));

      // Save 2D and 3D points
      chessBoard2D.push_back(corners);
      chessBoard3D.push_back(chessBoard3Dcorners);

      // Display
      cv::drawChessboardCorners(original, chessBoardSize, chessBoard2D[i], patternFound);
      cv::resize(original, original, cv::Size(original.cols / 2, original.rows / 2));
      cv::imshow("chessboard detection", original);
      cv::waitKey(400);
    }

    i++;
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
  cv::Vec<float, 5> distCoeffs(0, 0, 0, 0, 0);

  // rvects and tvects are the rotation and translation vectors for each view
  std::vector<cv::Mat> rvecs, tvecs;

  // Flags for calibration
  int flags = cv::CALIB_FIX_ASPECT_RATIO +    // Keeps the aspect ratio fixed
    cv::CALIB_FIX_K3 +                        // Fixes k3 distortion coefficient
    cv::CALIB_ZERO_TANGENT_DIST +             // Assumes zero tangential distortion
    cv::CALIB_FIX_PRINCIPAL_POINT;            // Fixes the principal point at the center

  if (chessBoard2D.size() < 2) {
    std::cerr << "Not enough images for calibration" << std::endl;
    return -1;
  }

  std::cout << "Calibrating..." << std::endl;
  // Calibrate the camera using the detected 2D and 3D points
  float error = cv::calibrateCamera(
    chessBoard3D,  // 3D points (object points) in the world coordinate system
    chessBoard2D,  // 2D points (in image plane) in the camera coordinate system
    frameSize,     // Image size (width, height)
    K,             // Output intrinsic matrix (3x3)
    distCoeffs,    // Output distortion coefficients (radial & tangential)
    rvecs,         // Output rotation vectors (3x1) one for each view
    tvecs,         // Output translation vectors (3x1) one for each view
    flags          // Flags (optional)
  );

  std::cout << "Reprojection error = " << error << "\nK =\n"
            << K << "\ndistCoeffs =\n"
            << distCoeffs << std::endl;

  // Get first image and apply lens correction using undistort
  // This method is not recommended for real-time applications
  cv::Mat firstImg = cv::imread(fileNames[0], cv::IMREAD_COLOR);
  cv::undistort(firstImg, undistorted, K, distCoeffs);

  // Display
  compare_images("Comparison no RT", firstImg, undistorted);

  // For real-time applications, we can use the remap method
  // Precompute lens correction interpolation
  cv::Mat mapX, mapY;
  cv::initUndistortRectifyMap(
    K,                  // Camera intrinsic matrix (3x3)
    distCoeffs,         // Distortion coefficients (radial & tangential)
    cv::Matx33f::eye(), // Rectification matrix (identity if no rectification)
    K,                  // New intrinsic matrix after undistortion (can be modified)
    frameSize,          // Size of the output image (width, height)
    CV_32FC1,           // Type of the output maps (CV_32FC1 or CV_16SC2)
    mapX,               // Output map for x-coordinates
    mapY                // Output map for y-coordinates
  );

  // Show lens corrected images
  for (auto const & f : fileNames) {
    std::cout << std::string(f) << std::endl;

    original = cv::imread(f, cv::IMREAD_COLOR);

    // Remap the image using the precomputed interpolation maps
    cv::remap(
      original,         // Input distorted image
      undistorted,      // Output undistorted image
      mapX,             // Precomputed x-coordinates map
      mapY,             // Precomputed y-coordinates map
      cv::INTER_LINEAR  // Interpolation method
    );

    // Display
    compare_images("Comparison RT", original, undistorted);
  }

  return 0;
}
