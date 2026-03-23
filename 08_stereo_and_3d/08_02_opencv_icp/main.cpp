/**
 * @file main.cpp
 * @brief Pure ICP (Iterative Closest Point) algorithm demonstration
 * @author José Miguel Guerrero Hernández
 *
 * ICP is a fundamental algorithm for aligning two point clouds.
 * It iteratively minimizes the distance between corresponding points
 * by finding the optimal rigid transformation (rotation + translation).
 *
 * Algorithm steps (per iteration):
 *  1. For each point in source, find closest point in target
 *  2. Estimate transformation that minimizes distances
 *  3. Apply transformation to source
 *  4. Repeat until convergence (error < threshold or max iterations)
 *
 * This example demonstrates ICP using OpenCV's surface_matching module.
 * For PCL-based ICP, see 08_07_PCL_ICP.
 *
 * Usage: ./icp <source.ply> <target.ply>
 *
 * @note PLY files must contain only vertices with normals (no faces).
 *       Format: x y z nx ny nz per line after header.
 */

#include <cstdlib>
#include <opencv2/surface_matching.hpp>
#include <opencv2/surface_matching/ppf_helpers.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/viz.hpp>
#include <iostream>
#include <iomanip>
#include <string>

/**
 * @brief Display point cloud in 3D visualization window
 * @param windowName Name of the visualization window
 * @param cloud Point cloud matrix (Nx6: x,y,z,nx,ny,nz)
 * @param color Color for the point cloud
 * @param waitKey If true, wait for key press (blocking)
 */
void showCloud(
  const std::string & windowName,
  const cv::Mat & cloud,
  const cv::viz::Color & color = cv::viz::Color::white(),
  bool waitKey = false)
{
  cv::viz::Viz3d window(windowName);
  window.setBackgroundColor(cv::viz::Color::black());
  window.showWidget("coords", cv::viz::WCoordinateSystem(50.0));

  // Extract XYZ from Nx6 matrix and reshape for viz (needs 1xN with 3 channels)
  cv::Mat xyz;
  cloud.colRange(0, 3).convertTo(xyz, CV_32F);
  xyz = xyz.reshape(3, 1);  // 1 row, N cols, 3 channels

  cv::viz::WCloud cloudWidget(xyz, color);
  cloudWidget.setRenderingProperty(cv::viz::POINT_SIZE, 2.0);
  window.showWidget("cloud", cloudWidget);

  if (waitKey) {
    window.spin();
  } else {
    window.spinOnce(1, true);
  }
}

/**
 * @brief Display two point clouds overlaid for comparison
 * @param windowName Window name
 * @param source Source point cloud (will be shown in red)
 * @param target Target point cloud (will be shown in green)
 * @param waitKey If true, block until key press
 */
void showComparison(
  const std::string & windowName,
  const cv::Mat & source,
  const cv::Mat & target,
  bool waitKey = true)
{
  cv::viz::Viz3d window(windowName);
  window.setBackgroundColor(cv::viz::Color::black());
  window.showWidget("coords", cv::viz::WCoordinateSystem(50.0));

  // Extract XYZ and reshape for viz
  cv::Mat srcXYZ, tgtXYZ;
  source.colRange(0, 3).convertTo(srcXYZ, CV_32F);
  target.colRange(0, 3).convertTo(tgtXYZ, CV_32F);
  srcXYZ = srcXYZ.reshape(3, 1);
  tgtXYZ = tgtXYZ.reshape(3, 1);

  cv::viz::WCloud srcCloud(srcXYZ, cv::viz::Color::red());
  cv::viz::WCloud tgtCloud(tgtXYZ, cv::viz::Color::green());

  srcCloud.setRenderingProperty(cv::viz::POINT_SIZE, 2.0);
  tgtCloud.setRenderingProperty(cv::viz::POINT_SIZE, 2.0);

  window.showWidget("source", srcCloud);
  window.showWidget("target", tgtCloud);

  // Add instruction text
  cv::viz::WText instruction("Red: Source | Green: Target", cv::Point(10, 30), 20,
    cv::viz::Color::white());
  cv::viz::WText instruction2("Close the window to continue", cv::Point(10, 60), 20,
    cv::viz::Color::white());
  window.showWidget("instruction", instruction);
  window.showWidget("instruction2", instruction2);

  if (waitKey) {
    window.spin();
  } else {
    window.spinOnce(1, true);
  }
}

/**
 * @brief Apply 4x4 transformation matrix to point cloud
 * @param cloud Input point cloud (Nx6)
 * @param pose 4x4 transformation matrix
 * @return Transformed point cloud
 */
cv::Mat transformCloud(const cv::Mat & cloud, const cv::Matx44d & pose)
{
  return cv::ppf_match_3d::transformPCPose(cloud, pose);
}

int main(int argc, char ** argv)
{
  std::cout << "========================================" << std::endl;
  std::cout << "  ICP (Iterative Closest Point) Demo" << std::endl;
  std::cout << "========================================" << std::endl;

  // Parse arguments or use defaults
  std::string source_file;
  std::string target_file;

  if (argc < 3) {
    // Use default files from data folder
    source_file = "../../data/parasaurolophus_source.ply";
    target_file = "../../data/parasaurolophus_target.ply";
    std::cout << "\nNo arguments provided. Using default files:" << std::endl;
    std::cout << "  Source: " << source_file << std::endl;
    std::cout << "  Target: " << target_file << std::endl;
  } else {
    source_file = argv[1];
    target_file = argv[2];
  }

  // Load point clouds
  // loadPLYSimple(filename, withNormals)
  //   withNormals=1: expects 6 values per vertex (x,y,z,nx,ny,nz)
  //   withNormals=0: expects 3 values per vertex (x,y,z)
  std::cout << "\nLoading point clouds..." << std::endl;
  cv::Mat source = cv::ppf_match_3d::loadPLYSimple(source_file.c_str(), 1);
  cv::Mat target = cv::ppf_match_3d::loadPLYSimple(target_file.c_str(), 1);

  if (source.empty() || target.empty()) {
    std::cerr << "Error: Could not load point clouds!" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Source: " << source.rows << " points" << std::endl;
  std::cout << "Target: " << target.rows << " points" << std::endl;

  // Show initial state (before ICP)
  std::cout << "\nShowing initial alignment (red=source, green=target)..." << std::endl;
  std::cout << "Close the window to continue." << std::endl;
  showComparison("Before ICP", source, target, true);

  // Configure ICP
  // cv::ppf_match_3d::ICP(iterations, tolerence, rejectionScale, numLevels)
  //   iterations: Maximum number of ICP iterations (100)
  //   tolerance: Convergence threshold - stops when error change < tolerance (0.005)
  //   rejectionScale: Points with distance > rejectionScale * stddev are rejected (2.5)
  //   numLevels: Number of pyramid levels for coarse-to-fine (8)
  std::cout << "\nConfiguring ICP..." << std::endl;
  std::cout << "  Max iterations: 100" << std::endl;
  std::cout << "  Tolerance: 0.005" << std::endl;
  std::cout << "  Rejection scale: 2.5" << std::endl;
  std::cout << "  Pyramid levels: 8" << std::endl;

  cv::ppf_match_3d::ICP icp(100, 0.005f, 2.5f, 8);

  // Run ICP
  // registerModelToScene(source, target, residual, pose)
  //   Returns the transformation that aligns source to target
  //   Note: parameter order is (src, dst, residual, pose)
  std::cout << "\nRunning ICP..." << std::endl;
  int64 t1 = cv::getTickCount();

  cv::Matx44d pose;
  double residual;
  int result = icp.registerModelToScene(source, target, residual, pose);

  int64 t2 = cv::getTickCount();
  double elapsed = static_cast<double>(t2 - t1) / cv::getTickFrequency();

  if (result < 0) {
    std::cerr << "ICP failed!" << std::endl;
    return EXIT_FAILURE;
  }

  // Print results
  std::cout << "\n=== ICP Results ===" << std::endl;
  std::cout << "Elapsed time: " << elapsed << " seconds" << std::endl;
  std::cout << "Residual error: " << residual << std::endl;
  std::cout << "\nTransformation matrix (4x4):" << std::endl;
  for (int i = 0; i < 4; i++) {
    std::cout << "  [";
    for (int j = 0; j < 4; j++) {
      std::cout << std::setw(10) << std::fixed << std::setprecision(4) << pose(i, j);
    }
    std::cout << " ]" << std::endl;
  }

  // Extract rotation and translation
  cv::Matx33d rotation;
  cv::Vec3d translation;
  for (int i = 0; i < 3; i++) {
    translation(i) = pose(i, 3);
    for (int j = 0; j < 3; j++) {
      rotation(i, j) = pose(i, j);
    }
  }
  std::cout << "\nTranslation: [" << translation(0) << ", "
            << translation(1) << ", " << translation(2) << "]" << std::endl;

  // Apply transformation to source
  cv::Mat source_aligned = transformCloud(source, pose);

  // Save aligned point cloud
  std::string output_file = "../../data/parasaurolophus_aligned_output.ply";
  cv::ppf_match_3d::writePLY(source_aligned, output_file.c_str());
  std::cout << "\nAligned point cloud saved to: " << output_file << std::endl;

  // Show final result
  std::cout << "\nShowing final alignment (red=aligned source, green=target)..." << std::endl;
  showComparison("After ICP", source_aligned, target, true);

  return EXIT_SUCCESS;
}
