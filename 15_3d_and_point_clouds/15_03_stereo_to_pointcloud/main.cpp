/**
 * @file main.cpp
 * @brief From a disparity map to a colored 3D point cloud
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - Computing a disparity map with StereoSGBM (same idea as 15_02)
 * - Turning disparity into 3D points with cv::reprojectImageTo3D()
 * - Saving the result as a PLY file you can open in MeshLab/CloudCompare
 *
 * This is the bridge between the 2D stereo world and the 3D examples that
 * follow: the equation presented in 15_02,
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
 * calibration (Chapter 14). This example accepts that file directly:
 *
 *     ./15_03_stereo_to_pointcloud left.png right.png --calib=stereo_calibration.yml
 *
 * where stereo_calibration.yml is what 14_03_stereo_calibration writes. That
 * is the whole point of calibrating: with it the cloud is metric, in the units
 * of the calibration pattern. Without it the example falls back to a Q built
 * from an assumed rig, and then the shapes are correct but the absolute scale
 * is not. The aloe pair shipped with the repository comes pre-rectified and
 * without its calibration, so it uses the fallback.
 *
 * @note Plain OpenCV: no opencv_contrib needed (15_02 does need ximgproc for
 *       the WLS filter, this one only uses StereoSGBM).
 *       Output: cloud.ply in the current directory, or wherever --out says.
 *       One vertex per valid pixel, so it is a big ASCII file.
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
// Default depth limit, overridable with --maxz. Its unit is whatever Q uses:
// metres with the assumed rig below, calibration-pattern units with --calib.
constexpr float MAX_Z = 10.0f;
}

/**
 * @brief Saves a colored point cloud in ASCII PLY format
 * @param path Output file path
 * @param points CV_32FC3 matrix from reprojectImageTo3D (X, Y, Z per pixel)
 * @param colors BGR image aligned with 'points' (color source per pixel)
 * @param valid_mask CV_8U mask of pixels with a usable disparity
 * @param max_z Discard points beyond this Z, in the units of Q
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
  const cv::Mat & valid_mask,
  float max_z)
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
      if (!std::isfinite(p[2]) || std::abs(p[2]) > max_z) {
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
  // Load the rectified stereo pair (same images as 15_02)
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@left  | ../../data/aloeL.jpg | Left image of the stereo pair}"
    "{@right | ../../data/aloeR.jpg | Right image of the stereo pair}"
    "{calib c | | Optional stereo_calibration.yml written by 14_03. With it the "
    "pair is rectified and the cloud is metric; without it the scale is arbitrary}"
    "{maxz | 10.0 | Discard points beyond this Z, in the units of Q: metres with "
    "the assumed rig, calibration-pattern units with --calib}"
    "{out o | cloud.ply | Where to write the resulting point cloud}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  const std::string left_path = parser.get<std::string>("@left");
  const std::string right_path = parser.get<std::string>("@right");
  const std::string calib_file = parser.get<std::string>("calib");
  const float max_z = parser.get<float>("maxz");
  const std::string output_path = parser.get<std::string>("out");

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
    std::cerr << "Error: could not load stereo pair (" << left_path << ", "
              << right_path << ")" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "=== Stereo Disparity to Point Cloud ===" << std::endl;
  std::cout << "Pair: " << left.cols << "x" << left.rows << std::endl;

  // ========================================
  // Step 1: the calibration, if there is one
  // ========================================
  // Q is what turns a disparity map into METRIC 3D points, and there are two
  // ways to get it. The good one is the Q that cv::stereoRectify() produced
  // during stereo calibration, which is exactly what 14_03_stereo_calibration
  // writes to stereo_calibration.yml. The other is to guess the rig and build
  // Q by hand, which gives the right SHAPES and an arbitrary SCALE.
  //
  // Passing --calib is what closes the loop between the calibration of chapter
  // 14 and the 3D of chapter 15, and it does two things, not one: it rectifies
  // the pair (disparity along a row only means something on rectified images)
  // and it supplies the Q that puts the result in real units.
  cv::Mat Q;

  if (!calib_file.empty()) {
    cv::FileStorage fs(calib_file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
      std::cerr << "Error: cannot open " << calib_file << std::endl;
      return EXIT_FAILURE;
    }
    cv::Mat K1, D1, K2, D2, R1, R2, P1, P2;
    int cw = 0, ch = 0;
    fs["Q"] >> Q;
    fs["K1"] >> K1; fs["D1"] >> D1; fs["K2"] >> K2; fs["D2"] >> D2;
    fs["R1"] >> R1; fs["R2"] >> R2; fs["P1"] >> P1; fs["P2"] >> P2;
    fs["image_width"] >> cw; fs["image_height"] >> ch;
    fs.release();

    if (Q.empty() || Q.rows != 4 || Q.cols != 4) {
      std::cerr << "Error: " << calib_file << " has no valid 4x4 Q matrix. "
                << "Generate it by running 14_03_stereo_calibration." << std::endl;
      return EXIT_FAILURE;
    }

    // A calibration only describes the rig it was computed on. Applying it to
    // images from another camera gives a cloud that is silently wrong, so the
    // size stored in the file is checked against the pair being used.
    if (cw > 0 && ch > 0 && (cw != left.cols || ch != left.rows)) {
      std::cerr << "Error: the calibration was computed on " << cw << "x" << ch
                << " images and this pair is " << left.cols << "x" << left.rows
                << ".\nThey are different rigs, so the result would be meaningless."
                << " Use the pair you\ncalibrated with, or drop --calib to fall back"
                << " to the assumed rig." << std::endl;
      return EXIT_FAILURE;
    }

    // Rectification: the same remap that 14_03 applies, so that corresponding
    // points land on the same row and the disparity search along it is valid.
    if (!K1.empty() && !R1.empty() && !P1.empty()) {
      const cv::Size size(left.cols, left.rows);
      cv::Mat m1x, m1y, m2x, m2y;
      cv::initUndistortRectifyMap(K1, D1, R1, P1, size, CV_32FC1, m1x, m1y);
      cv::initUndistortRectifyMap(K2, D2, R2, P2, size, CV_32FC1, m2x, m2y);
      cv::Mat lr, rr;
      cv::remap(left, lr, m1x, m1y, cv::INTER_LINEAR);
      cv::remap(right, rr, m2x, m2y, cv::INTER_LINEAR);
      left = lr;
      right = rr;
      std::cout << "Pair rectified with the maps of " << calib_file << std::endl;
    }

    // Q(2,3) is the focal length in pixels. Q(3,2) is -1/Tx, and for a
    // left-right rig Tx = -baseline, so the baseline is 1/Q(3,2). Both come in
    // the units of the calibration pattern, so if the squares were measured in
    // millimetres the cloud comes out in millimetres.
    std::cout << "Q read from " << calib_file << std::endl;
    std::cout << "  focal   = " << Q.at<double>(2, 3) << " px" << std::endl;
    std::cout << "  baseline= " << (Q.at<double>(3, 2) != 0.0 ?
      1.0 / Q.at<double>(3, 2) : 0.0) << " (calibration units)" << std::endl;
  } else {
    // No calibration: build a plausible Q. cx, cy = image center.
    const double cx = left.cols / 2.0;
    const double cy = left.rows / 2.0;
    Q = (cv::Mat_<double>(4, 4) <<
      1, 0, 0, -cx,
      0, 1, 0, -cy,
      0, 0, 0, Config::FOCAL_PX,
      // The entry stereoRectify writes here is -1/Tx, and for a left-right rig
      // Tx = -baseline, so in terms of the baseline the sign is POSITIVE.
      // With -1/b every W comes out negative and the whole cloud lands behind
      // the camera, mirrored through the origin
      0, 0, 1.0 / Config::BASELINE_M, 0);
    std::cout << "No --calib given: using an assumed rig (focal "
              << Config::FOCAL_PX << " px, baseline " << Config::BASELINE_M
              << " m).\n  Shapes are correct, absolute scale is not. Run "
              << "14_03_stereo_calibration and pass its\n  stereo_calibration.yml "
              << "with --calib to get a metric cloud." << std::endl;
  }

  // ========================================
  // Step 2: disparity with SGBM
  // ========================================
  // Minimal SGBM setup (15_02 explores the parameters and post-filtering
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

  // SGBM returns fixed-point disparities scaled by 16 (see 15_02):
  // convert to real float pixels before doing geometry with them
  cv::Mat disparity;
  disparity_16s.convertTo(disparity, CV_32F, 1.0 / 16.0);

  // Valid = pixels where a match was actually found (disparity > 0)
  cv::Mat valid_mask = disparity > 0.0f;

  // ========================================
  // Step 3: reproject to 3D
  // ========================================
  // One call: every pixel (x, y, disparity) -> metric point (X, Y, Z)
  cv::Mat points_3d;
  cv::reprojectImageTo3D(disparity, points_3d, Q, /*handleMissingValues=*/true);

  // ========================================
  // Step 4: save as PLY
  // ========================================
  
  const int saved = savePointCloudPLY(output_path, points_3d, left, valid_mask, max_z);
  if (saved == 0) {
    std::cout << "No point survived the filters. Every pixel was either unmatched "
              << "or beyond\nthe " << max_z << " limit of --maxz, "
              << "which is expressed in the units of Q: metres\nwith the assumed "
              << "rig, and whatever the calibration pattern used with --calib."
              << std::endl;
  }

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
