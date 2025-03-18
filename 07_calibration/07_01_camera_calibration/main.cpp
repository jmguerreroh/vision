/**
 * Camera calibration demo sample
 * @author José Miguel Guerrero
 *
 * Based on: https://github.com/niconielsen32/ComputerVision
 */

#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char ** argv)
{
  // Get filenames
  std::vector<cv::String> fileNames;
  cv::glob("../calibration_images/Image*.png", fileNames, false);
  cv::Size patternSize(25 - 1, 18 - 1);
  std::vector<std::vector<cv::Point2f>> q(fileNames.size());

  // 1. Generate checkerboard (world) coordinates Q. The board has 25 x 18 fields with a size of 15x15mm
  std::vector<std::vector<cv::Point3f>> Q;
  int checkerBoard[2] = {25, 18};

  // Defining the world coordinates for 3D points assuming z=0
  std::vector<cv::Point3f> objp;
  for (int i = 1; i < checkerBoard[1]; i++) {
    for (int j = 1; j < checkerBoard[0]; j++) {
      objp.push_back(cv::Point3f(j, i, 0));
    }
  }

  // Detect feature points
  std::size_t i = 0;
  for (auto const & f : fileNames) {
    std::cout << std::string(f) << std::endl;

    // 2. Read in the image an call cv::findChessboardCorners()
    cv::Mat img = cv::imread(fileNames[i]);
    cv::Mat gray;

    cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);

    bool patternFound = cv::findChessboardCorners(
      gray, patternSize, q[i],
      cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE +
      cv::CALIB_CB_FAST_CHECK);

    // 3. Use cv::cornerSubPix() to refine the found corner detections
    if (patternFound) {
      cv::cornerSubPix(
        gray, q[i], cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
      Q.push_back(objp);
    }

    // Display
    cv::drawChessboardCorners(img, patternSize, q[i], patternFound);
    cv::resize(img, img, cv::Size(img.cols / 2, img.rows / 2));
    cv::imshow("chessboard detection", img);
    cv::waitKey(800);

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

  // Frame size
  cv::Size frameSize(1440, 1080);

  std::cout << "Calibrating..." << std::endl;

  // 4. Call "float error = cv::calibrateCamera()" with the input coordinates and output parameters as declared above...
  float error = cv::calibrateCamera(Q, q, frameSize, K, distCoeffs, rvecs, tvecs, flags);

  std::cout << "Reprojection error = " << error << "\nK =\n"
            << K << "\ndistCoeffs =\n"
            << distCoeffs << std::endl;

  // Get first image and apply lens correction using undistort
  // This method is not recommended for real-time applications
  cv::Mat firstImg = cv::imread(fileNames[0], cv::IMREAD_COLOR);
  cv::Mat undistortedImg;
  cv::undistort(firstImg, undistortedImg, K, distCoeffs);

  // Display
  cv::resize(firstImg, firstImg, cv::Size(firstImg.cols / 2, firstImg.rows / 2));
  cv::putText(
    firstImg, "Original", cv::Point(firstImg.cols - 100, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5,
    cv::Scalar(0, 0, 0), 2);
  cv::resize(undistortedImg, undistortedImg,
    cv::Size(undistortedImg.cols / 2, undistortedImg.rows / 2));
  cv::putText(
    undistortedImg, "Undistorted",
    cv::Point(undistortedImg.cols - 100, 15),
    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);

  cv::Mat concat_nort;
  cv::hconcat(firstImg, undistortedImg, concat_nort);
  cv::imshow("Comparison no RT", concat_nort);
  cv::waitKey(0);

  // For real-time applications, we can use the remap method
  // Precompute lens correction interpolation
  cv::Mat mapX, mapY;
  cv::initUndistortRectifyMap(
    K, distCoeffs, cv::Matx33f::eye(), K, frameSize, CV_32FC1, mapX, mapY);

  // Show lens corrected images
  for (auto const & f : fileNames) {
    std::cout << std::string(f) << std::endl;

    cv::Mat img = cv::imread(f, cv::IMREAD_COLOR);

    cv::Mat imgUndistorted;
    // 5. Remap the image using the precomputed interpolation maps
    cv::remap(img, imgUndistorted, mapX, mapY, cv::INTER_LINEAR);

    // Display
    cv::resize(img, img, cv::Size(img.cols / 2, img.rows / 2));
    cv::putText(
      img, "Original", cv::Point(img.cols - 100, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5,
      cv::Scalar(0, 0, 0), 2);
    cv::resize(imgUndistorted, imgUndistorted,
      cv::Size(imgUndistorted.cols / 2, imgUndistorted.rows / 2));
    cv::putText(
      imgUndistorted, "Undistorted",
      cv::Point(imgUndistorted.cols - 100, 15),
      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);

    cv::Mat concat;
    cv::hconcat(img, imgUndistorted, concat);
    cv::imshow("Comparison RT", concat);
    cv::waitKey(0);
  }

  return 0;
}
