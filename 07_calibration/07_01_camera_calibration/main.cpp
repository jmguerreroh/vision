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

using namespace cv;
using namespace std;

int main(int argc, char ** argv)
{
  // Get filenames
  vector<String> fileNames;
  glob("../calibration_images/Image*.png", fileNames, false);
  Size patternSize(25 - 1, 18 - 1);
  vector<vector<Point2f>> q(fileNames.size());

  // 1. Generate checkerboard (world) coordinates Q. The board has 25 x 18 fields with a size of 15x15mm
  vector<vector<Point3f>> Q;
  int checkerBoard[2] = {25, 18};
  // Defining the world coordinates for 3D points
  vector<Point3f> objp;
  for (int i = 1; i < checkerBoard[1]; i++) {
    for (int j = 1; j < checkerBoard[0]; j++) {
      objp.push_back(Point3f(j, i, 0));
    }
  }

  vector<Point2f> imgPoint;
  // Detect feature points
  size_t i = 0;
  for (auto const & f : fileNames) {
    cout << string(f) << endl;

    // 2. Read in the image an call cv::findChessboardCorners()
    Mat img = imread(fileNames[i]);
    Mat gray;

    cvtColor(img, gray, COLOR_RGB2GRAY);

    bool patternFound = findChessboardCorners(
      gray, patternSize, q[i],
      CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE +
      CALIB_CB_FAST_CHECK);

    // 2. Use cv::cornerSubPix() to refine the found corner detections
    if (patternFound) {
      cornerSubPix(
        gray, q[i], Size(11, 11), Size(-1, -1),
        TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.1));
      Q.push_back(objp);
    }

    // Display
    drawChessboardCorners(img, patternSize, q[i], patternFound);
    resize(img, img, Size(img.cols / 2, img.rows / 2));
    imshow("chessboard detection", img);
    waitKey(800);

    i++;
  }

  Matx33f K(Matx33f::eye());    // intrinsic camera matrix
  Vec<float, 5> k(0, 0, 0, 0, 0);   // distortion coefficients

  vector<Mat> rvecs, tvecs;
  vector<double> stdIntrinsics, stdExtrinsics, perViewErrors;
  int flags = CALIB_FIX_ASPECT_RATIO + CALIB_FIX_K3 +
    CALIB_ZERO_TANGENT_DIST + CALIB_FIX_PRINCIPAL_POINT;
  Size frameSize(1440, 1080);

  cout << "Calibrating..." << endl;
  // 4. Call "float error = cv::calibrateCamera()" with the input coordinates and output parameters as declared above...

  float error = calibrateCamera(Q, q, frameSize, K, k, rvecs, tvecs, flags);

  cout << "Reprojection error = " << error << "\nK =\n"
       << K << "\nk=\n"
       << k << endl;

  // Precompute lens correction interpolation
  Mat mapX, mapY;
  initUndistortRectifyMap(K, k, Matx33f::eye(), K, frameSize, CV_32FC1, mapX, mapY);

  // Show lens corrected images
  for (auto const & f : fileNames) {
    cout << string(f) << endl;

    Mat img = imread(f, IMREAD_COLOR);

    Mat imgUndistorted;
    // 5. Remap the image using the precomputed interpolation maps
    remap(img, imgUndistorted, mapX, mapY, INTER_LINEAR);

    // Display
    resize(img, img, Size(img.cols / 2, img.rows / 2));
    putText(
      img, "Original", Point(img.cols - 100, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(
        0, 0,
        0), 2);
    resize(imgUndistorted, imgUndistorted, Size(imgUndistorted.cols / 2, imgUndistorted.rows / 2));
    putText(
      imgUndistorted, "Undistorted", Point(
        imgUndistorted.cols - 100,
        15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0),
      2);

    Mat concat;
    hconcat(img, imgUndistorted, concat);
    imshow("Comparison", concat);
    waitKey(0);
  }

  return 0;
}
