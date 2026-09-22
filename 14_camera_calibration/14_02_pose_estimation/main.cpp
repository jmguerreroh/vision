/**
 * @file main.cpp
 * @brief Camera pose estimation (PnP) from a ChArUco board
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - Loading a camera calibration (K + distortion) from a YAML file
 * - Detecting ArUco markers and interpolating ChArUco chessboard corners
 * - Solving the Perspective-n-Point (PnP) problem to get the camera pose
 * - Drawing the estimated 3D reference frame on the image
 *
 * The PnP problem: given (a) N points with KNOWN 3D coordinates in some
 * world frame, (b) their 2D projections in the image, and (c) the intrinsic
 * calibration K (obtained in 14_01), estimate the rotation and translation
 * that place the camera with respect to that world frame. This is the basis
 * of augmented reality, robot localization and camera-in-hand systems.
 *
 * A ChArUco board is a chessboard with ArUco markers printed inside the
 * white squares. The markers make each corner IDENTIFIABLE (the plain
 * chessboard of 14_01 requires seeing the whole pattern), so the pose can
 * be estimated even under partial occlusion.
 *
 * @note Uses the classic cv::aruco API (opencv_contrib, OpenCV <= 4.6).
 *       Data: image and calibration come from the official OpenCV tutorial.
 */

#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>       // drawFrameAxes, Rodrigues
#include <opencv2/aruco/charuco.hpp> // requires opencv_contrib
#include <cmath>
#include <iostream>
#include <vector>

namespace Config
{
// Geometry of the board shown in data/aruco/choriginal.jpg (the values used
// by the OpenCV tutorial that produced the image and the calibration file)
constexpr int SQUARES_X = 5;             // Chessboard squares horizontally
constexpr int SQUARES_Y = 7;             // Chessboard squares vertically
constexpr float SQUARE_LENGTH = 0.04f;   // Square side in meters
constexpr float MARKER_LENGTH = 0.02f;   // ArUco marker side in meters
constexpr float AXIS_LENGTH = 0.1f;      // Drawn axis length in meters
}

int main(int argc, char ** argv)
{
  // Load the input image and the camera calibration
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/aruco/choriginal.jpg | Input file}"
    "{@calibration | ../../data/aruco/tutorial_camera_charuco.yml | Camera "
    "calibration file (YAML)}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  const std::string image_path = parser.get<std::string>("@input");
  const std::string calib_path = parser.get<std::string>("@calibration");

  // Without this, a malformed value (--frames=xyz) is reported by the parser
  // but the example carries on with the default, which is the hardest kind
  // of failure to diagnose. Note it validates values, not option names:
  // cv::CommandLineParser ignores an unknown option without complaining
  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }

  const cv::Mat image = cv::imread(cv::samples::findFile(image_path, false), cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "Error: Could not load image '" << image_path << "'" << std::endl;
    return EXIT_FAILURE;
  }

  // The calibration file stores the intrinsic matrix and the distortion
  // coefficients -- exactly what 14_01 computed. cv::FileStorage reads the
  // YAML/XML format OpenCV uses to serialize matrices.
  cv::Mat camera_matrix, dist_coeffs;
  cv::FileStorage fs(calib_path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    std::cerr << "Error: Could not open calibration file '" << calib_path << "'"
              << std::endl;
    return EXIT_FAILURE;
  }
  fs["camera_matrix"] >> camera_matrix;
  fs["distortion_coefficients"] >> dist_coeffs;
  fs.release();

  std::cout << "=== ChArUco Pose Estimation (PnP) ===" << std::endl;
  std::cout << "K =\n" << camera_matrix << std::endl;

  // ========================================
  // Describe the board we expect to see
  // ========================================
  // The dictionary defines the family of marker patterns (6x6 bits, 250
  // different ids). The board object knows the 3D position of every corner
  // in the board's own reference frame -- these are the "known 3D points"
  // that PnP needs.
  cv::Ptr<cv::aruco::Dictionary> dictionary =
    cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
  cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(
    Config::SQUARES_X, Config::SQUARES_Y,
    Config::SQUARE_LENGTH, Config::MARKER_LENGTH, dictionary);

  // ========================================
  // Step 1: detect the ArUco markers
  // ========================================
  std::vector<int> marker_ids;
  std::vector<std::vector<cv::Point2f>> marker_corners;
  cv::aruco::detectMarkers(image, dictionary, marker_corners, marker_ids);

  std::cout << "ArUco markers detected: " << marker_ids.size() << std::endl;
  if (marker_ids.empty()) {
    std::cerr << "Error: no markers found - cannot estimate pose" << std::endl;
    return EXIT_FAILURE;
  }

  // ========================================
  // Step 2: interpolate the chessboard corners
  // ========================================
  // Each detected marker identifies its neighborhood of the board, letting
  // OpenCV locate the chessboard corners between markers with sub-pixel
  // accuracy (chessboard corners are more precise landmarks than the
  // marker corners themselves)
  std::vector<cv::Point2f> charuco_corners;
  std::vector<int> charuco_ids;
  cv::aruco::interpolateCornersCharuco(marker_corners, marker_ids, image, board,
                                       charuco_corners, charuco_ids,
                                       camera_matrix, dist_coeffs);

  std::cout << "ChArUco corners interpolated: " << charuco_ids.size() << std::endl;

  // ========================================
  // Step 3: solve PnP for the board pose
  // ========================================
  // estimatePoseCharucoBoard pairs each detected 2D corner with its known
  // 3D position on the board and solves PnP (internally cv::solvePnP).
  // The result is the board pose in the CAMERA frame:
  //   rvec: rotation in Rodrigues form (axis * angle, 3 values)
  //   tvec: translation in meters
  cv::Vec3d rvec, tvec;
  const bool pose_ok = cv::aruco::estimatePoseCharucoBoard(
    charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, rvec, tvec);

  if (!pose_ok) {
    std::cerr << "Error: not enough corners for a reliable pose" << std::endl;
    return EXIT_FAILURE;
  }

  // Rodrigues converts the compact rvec into the full 3x3 rotation matrix
  cv::Mat rotation;
  cv::Rodrigues(rvec, rotation);

  std::cout << "\nPose of the board in the camera frame:" << std::endl;
  std::cout << "  rvec (Rodrigues) = " << rvec << std::endl;
  std::cout << "  tvec (meters)    = " << tvec << std::endl;
  std::cout << "  Distance camera-board: " << cv::norm(tvec) << " m" << std::endl;
  std::cout << "\nR =\n" << rotation << std::endl;

  // ========================================
  // Step 4: check the pose by reprojecting
  // ========================================
  // A pose always comes out: what tells whether it can be trusted is the
  // reprojection error, the same measure used to judge a calibration. Every
  // detected corner is projected back with the estimated pose and compared
  // with where it was actually found.
  std::vector<cv::Point3f> object_points;
  for (int id : charuco_ids) {
    object_points.push_back(board->chessboardCorners[id]);
  }

  std::vector<cv::Point2f> reprojected;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs,
                    reprojected);
  const double reprojection_error =
    cv::norm(charuco_corners, reprojected, cv::NORM_L2) /
    std::sqrt(static_cast<double>(reprojected.size()));
  std::cout << "\nReprojection error of the pose: " << reprojection_error
            << " px" << std::endl;

  // The same correspondences solved with cv::solvePnP give the same pose:
  // estimatePoseCharucoBoard is a convenience wrapper, not a different
  // algorithm. Running both is the cheapest way to see it.
  cv::Vec3d rvec_pnp, tvec_pnp;
  cv::solvePnP(object_points, charuco_corners, camera_matrix, dist_coeffs,
               rvec_pnp, tvec_pnp);
  std::cout << "Same pose with cv::solvePnP: tvec = " << tvec_pnp
            << "  (difference " << cv::norm(tvec - tvec_pnp) << " m)"
            << std::endl;

  // ========================================
  // Visualization
  // ========================================
  cv::Mat display = image.clone();
  cv::aruco::drawDetectedMarkers(display, marker_corners, marker_ids);
  cv::aruco::drawDetectedCornersCharuco(display, charuco_corners, charuco_ids,
                                        cv::Scalar(255, 0, 0));
  // Axes drawn at the board origin: X red, Y green, Z blue (towards camera)
  cv::drawFrameAxes(display, camera_matrix, dist_coeffs, rvec, tvec,
                    Config::AXIS_LENGTH);

  cv::imshow("Original", image);
  cv::imshow("Detected markers + corners + pose axes", display);

  std::cout << "\nAxes: X = red, Y = green, Z = blue" << std::endl;
  std::cout << "Press any key to exit..." << std::endl;
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
