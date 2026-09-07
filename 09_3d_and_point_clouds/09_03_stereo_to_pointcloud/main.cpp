/**
 * @file main.cpp
 * @brief From a disparity map to a colored 3D point cloud
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - Computing a disparity map with StereoSGBM (same idea as 09_02)
 * - Turning disparity into 3D points with cv::reprojectImageTo3D()
 * - Saving the result as a PLY file you can open in MeshLab/CloudCompare
 *
 * This is the bridge between the 2D stereo world and the 3D examples that
 * follow: the equation presented in 09_02,
 *
 *     Z = f * b / disparity
 *
 * is applied here to EVERY pixel. reprojectImageTo3D does it through the
 * 4x4 reprojection matrix Q:
 *
 *     Q = [ 1  0   0  -cx ]        [X]   [x - cx   ]
 *         [ 0  1   0  -cy ]   =>   [Y] ~ [y - cy   ]
 *         [ 0  0   0   f  ]        [Z]   [f        ]
 *         [ 0  0  1/b  0  ]        [W]   [disp / b ]
 *
 * so (X/W, Y/W, Z/W) gives the metric 3D point; note Z/W = f*b/disp.
 * In a real rig, Q is produced by cv::stereoRectify() during stereo
 * calibration (Chapter 7); here the aloe pair comes pre-rectified without
 * its calibration, so we build a plausible Q by hand -- shapes are correct,
 * the absolute scale is not.
 *
 * @note Plain OpenCV: no opencv_contrib needed (09_02 does need ximgproc for
 *       the WLS filter, this one only uses StereoSGBM).
 *       Output: cloud.ply in the current directory (one vertex per valid
 *       pixel, so it is a big ASCII file).
 */

#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>  // StereoSGBM, reprojectImageTo3D
#include <fstream>
#include <iostream>
#include <vector>

namespace Config
{
constexpr int NUM_DISPARITIES = 160;   // Search range; must be multiple of 16
constexpr int BLOCK_SIZE = 5;          // SGBM matching window (odd)
// Assumed rig parameters for the hand-built Q (see file header):
constexpr double FOCAL_PX = 800.0;     // Focal length in pixels
constexpr double BASELINE_M = 0.06;    // Distance between cameras (6 cm)
constexpr float MAX_Z = 10.0f;         // Discard points further than this (m)
}

/**
 * @brief Saves a colored point cloud in ASCII PLY format
 * @param path Output file path
 * @param points CV_32FC3 matrix from reprojectImageTo3D (X, Y, Z per pixel)
 * @param colors BGR image aligned with 'points' (color source per pixel)
 * @param valid_mask CV_8U mask of pixels with a usable disparity
 * @return Number of points written
 *
 * PLY is the simplest standard 3D format: a small ASCII header declaring
 * the per-vertex properties, then one "x y z r g b" line per point. The
 * PCL examples later in this chapter use the equivalent PCD format through
 * the library; writing PLY by hand here shows there is no magic inside.
 */
int savePointCloudPLY(
  const std::string & path,
  const cv::Mat & points,
  const cv::Mat & colors,
  const cv::Mat & valid_mask)
{
  // First pass: collect the valid points (the header needs the total count)
  std::vector<cv::Vec3f> xyz;
  std::vector<cv::Vec3b> bgr;
  for (int y = 0; y < points.rows; ++y) {
    for (int x = 0; x < points.cols; ++x) {
      if (!valid_mask.at<uchar>(y, x)) {
        continue;
      }
      const cv::Vec3f p = points.at<cv::Vec3f>(y, x);
      // reprojectImageTo3D marks unmatched pixels with huge Z values
      if (!std::isfinite(p[2]) || std::abs(p[2]) > Config::MAX_Z) {
        continue;
      }
      xyz.push_back(p);
      bgr.push_back(colors.at<cv::Vec3b>(y, x));
    }
  }

  std::ofstream out(path);
  if (!out) {
    return 0;
  }

  // PLY header: format + vertex count + per-vertex properties, in order
  out << "ply\n"
      << "format ascii 1.0\n"
      << "element vertex " << xyz.size() << "\n"
      << "property float x\nproperty float y\nproperty float z\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
      << "end_header\n";

  for (size_t i = 0; i < xyz.size(); ++i) {
    out << xyz[i][0] << " " << xyz[i][1] << " " << xyz[i][2] << " "
        << static_cast<int>(bgr[i][2]) << " "   // PLY expects RGB order,
        << static_cast<int>(bgr[i][1]) << " "   // OpenCV stores BGR
        << static_cast<int>(bgr[i][0]) << "\n";
  }

  return static_cast<int>(xyz.size());
}

int main(int argc, char ** argv)
{
  // Load the rectified stereo pair (same images as 09_02)
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@left  | ../../data/aloeL.jpg | Left image of the stereo pair}"
    "{@right | ../../data/aloeR.jpg | Right image of the stereo pair}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  const std::string left_path = parser.get<std::string>("@left");
  const std::string right_path = parser.get<std::string>("@right");

  const cv::Mat left = cv::imread(cv::samples::findFile(left_path, false), cv::IMREAD_COLOR);
  const cv::Mat right = cv::imread(cv::samples::findFile(right_path, false), cv::IMREAD_COLOR);

  if (left.empty() || right.empty()) {
    std::cerr << "Error: could not load stereo pair (" << left_path << ", "
              << right_path << ")" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "=== Stereo Disparity to Point Cloud ===" << std::endl;
  std::cout << "Pair: " << left.cols << "x" << left.rows << std::endl;

  // ========================================
  // Step 1: disparity with SGBM
  // ========================================
  // Minimal SGBM setup (09_02 explores the parameters and post-filtering
  // in depth; here disparity is just the input of the 3D step)
  cv::Ptr<cv::StereoSGBM> matcher = cv::StereoSGBM::create(
    0, Config::NUM_DISPARITIES, Config::BLOCK_SIZE);
  matcher->setP1(8 * 3 * Config::BLOCK_SIZE * Config::BLOCK_SIZE);
  matcher->setP2(32 * 3 * Config::BLOCK_SIZE * Config::BLOCK_SIZE);
  matcher->setUniquenessRatio(10);
  matcher->setSpeckleWindowSize(100);
  matcher->setSpeckleRange(2);

  cv::Mat disparity_16s;
  matcher->compute(left, right, disparity_16s);

  // SGBM returns fixed-point disparities scaled by 16 (see 09_02):
  // convert to real float pixels before doing geometry with them
  cv::Mat disparity;
  disparity_16s.convertTo(disparity, CV_32F, 1.0 / 16.0);

  // Valid = pixels where a match was actually found (disparity > 0)
  cv::Mat valid_mask = disparity > 0.0f;

  // ========================================
  // Step 2: build Q and reproject to 3D
  // ========================================
  // See the file header: in a calibrated rig this matrix comes from
  // cv::stereoRectify(). cx, cy = image center (principal point).
  const double cx = left.cols / 2.0;
  const double cy = left.rows / 2.0;
  const cv::Mat Q = (cv::Mat_<double>(4, 4) <<
    1, 0, 0, -cx,
    0, 1, 0, -cy,
    0, 0, 0, Config::FOCAL_PX,
    // The entry stereoRectify writes here is -1/Tx, and for a left-right rig
    // Tx = -baseline, so in terms of the baseline the sign is POSITIVE.
    // With -1/b every W comes out negative and the whole cloud lands behind
    // the camera, mirrored through the origin
    0, 0, 1.0 / Config::BASELINE_M, 0);

  // One call: every pixel (x, y, disparity) -> metric point (X, Y, Z)
  cv::Mat points_3d;
  cv::reprojectImageTo3D(disparity, points_3d, Q, /*handleMissingValues=*/true);

  // ========================================
  // Step 3: save as PLY
  // ========================================
  const std::string output_path = "cloud.ply";
  const int saved = savePointCloudPLY(output_path, points_3d, left, valid_mask);

  std::cout << "Points written to " << output_path << ": " << saved << std::endl;
  std::cout << "Open it with MeshLab or CloudCompare to inspect the 3D shape."
            << std::endl;

  // ========================================
  // Visualization of the intermediate steps
  // ========================================
  cv::Mat disparity_display;
  cv::normalize(disparity, disparity_display, 0, 255, cv::NORM_MINMAX, CV_8U,
                valid_mask);

  cv::imshow("Left image", left);
  cv::imshow("Disparity (near = bright)", disparity_display);

  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
