/**
 * @file main.cpp
 * @brief Iterative Closest Point (ICP) algorithm demonstration with visualization
 * @author José Miguel Guerrero Hernández
 *
 * This example shows:
 * - Creating synthetic point clouds with random 3D coordinates
 * - Applying known geometric transformations (translation + rotation)
 * - Configuring and executing ICP alignment algorithm
 * - Extracting rotation matrix and translation vector from transformation
 * - Computing Euler angles (roll, pitch, yaw) from rotation matrix
 * - Multi-cloud visualization with color coding (blue=source, green=target, red=aligned)
 * - Analyzing convergence quality with fitness score
 *
 * ICP Algorithm Parameters:
 * - MaximumIterations:         Maximum number of alignment iterations (50)
 * - TransformationEpsilon:     Convergence threshold for transformation (1e-8)
 * - EuclideanFitnessEpsilon:   Convergence threshold for fitness score (1e-6)
 * - MaxCorrespondenceDistance: Maximum distance for point matching (1000.0)
 *
 * @see https://pointclouds.org/documentation/tutorials/iterative_closest_point.html
 *
 * Usage: ./icp_demo
 * Output: Interactive visualization showing source, target, and aligned clouds
 */

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <vtkObject.h>

/**
 * @brief Visualizes multiple point clouds in a PCL viewer with different colors
 * @param cloud1 First point cloud to display (blue)
 * @param id1 Identifier string for the first point cloud
 * @param cloud2 Second point cloud to display (green)
 * @param id2 Identifier string for the second point cloud
 * @param aligned_cloud Optional aligned point cloud to display (red with transparency)
 * @param id3 Identifier string for the aligned point cloud
 */
void visualizePointClouds(
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1, const std::string & id1,
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2, const std::string & id2,
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud = nullptr,
  const std::string & id3 = "")
{
  pcl::visualization::PCLVisualizer viewer("ICP Visualization");
  viewer.setBackgroundColor(0, 0, 0);

  // PointCloud 1 (original): blue
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud1_color(cloud1, 0, 0, 255);
  viewer.addPointCloud<pcl::PointXYZ>(cloud1, cloud1_color, id1);
  viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, id1);

  // PointCloud 2 (transformed): green
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud2_color(cloud2, 0, 255, 0);
  viewer.addPointCloud<pcl::PointXYZ>(cloud2, cloud2_color, id2);
  viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, id2);

  // PointCloud 3 (aligned): red (if exists)
  if (aligned_cloud) {
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> aligned_color(aligned_cloud,
      255, 0, 0);
    viewer.addPointCloud<pcl::PointXYZ>(aligned_cloud, aligned_color, id3);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, id3);
      // Set color with alpha channel
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, id3);
  }

  // Add coordinate system (axes) to the visualizer
  viewer.addCoordinateSystem(1.0);
  // Set the camera position and orientation
  viewer.setCameraPosition(0, 3, -4, 0, 0, 0, 0);

  // Display instructions
  std::cout << "\n=== PCL Visualizer Controls ===" << std::endl;
  std::cout << "  - Mouse wheel: Zoom in/out" << std::endl;
  std::cout << "  - Left click + drag: Rotate view" << std::endl;
  std::cout << "  - Middle click + drag: Pan view" << std::endl;
  std::cout << "  - Press 'q': Close window" << std::endl;
  std::cout << "==============================\n" << std::endl;

  // Keep the visualizer open
  while (!viewer.wasStopped()) {
    viewer.spinOnce();
  }
}

/**
 * @brief Prints a 4x4 transformation matrix in a readable format
 */
void printTransformationMatrix(const Eigen::Matrix4f & matrix)
{
  std::cout << std::fixed << std::setprecision(6);
  for (int i = 0; i < 4; i++) {
    std::cout << "  ";
    for (int j = 0; j < 4; j++) {
      std::cout << std::setw(12) << matrix(i, j);
    }
    std::cout << std::endl;
  }
}

/**
 * @brief Extracts and displays rotation and translation from transformation matrix
 */
void analyzeTransformation(const Eigen::Matrix4f & transformation)
{
  std::cout << "\n=== Transformation Analysis ===" << std::endl;

  // Extract translation
  Eigen::Vector3f translation = transformation.block<3, 1>(0, 3);
  std::cout << "Translation: " << std::endl;
  std::cout << "  tx = " << translation(0) << std::endl;
  std::cout << "  ty = " << translation(1) << std::endl;
  std::cout << "  tz = " << translation(2) << std::endl;

  // Extract rotation matrix
  Eigen::Matrix3f rotation = transformation.block<3, 3>(0, 0);
  std::cout << "\nRotation Matrix:" << std::endl;
  std::cout << std::fixed << std::setprecision(6);
  for (int i = 0; i < 3; i++) {
    std::cout << "  ";
    for (int j = 0; j < 3; j++) {
      std::cout << std::setw(12) << rotation(i, j);
    }
    std::cout << std::endl;
  }

  // Calculate Euler angles (approximation)
  float roll = atan2(rotation(2, 1), rotation(2, 2)) * 180.0 / M_PI;
  float pitch = atan2(-rotation(2, 0),
    sqrt(rotation(2, 1) * rotation(2, 1) + rotation(2, 2) * rotation(2, 2))) * 180.0 / M_PI;
  float yaw = atan2(rotation(1, 0), rotation(0, 0)) * 180.0 / M_PI;

  std::cout << "\nEuler Angles (degrees):" << std::endl;
  std::cout << "  Roll:  " << roll << "°" << std::endl;
  std::cout << "  Pitch: " << pitch << "°" << std::endl;
  std::cout << "  Yaw:   " << yaw << "°" << std::endl;
  std::cout << "==============================\n" << std::endl;
}

int main(int argc, char ** argv)
{
  (void)argc;
  (void)argv;
  // Disable VTK warning messages
  vtkObject::GlobalWarningDisplayOff();

  // Seed for reproducible random numbers
  srand(42);

  // Create source and target point clouds
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>(5, 1));
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target(new pcl::PointCloud<pcl::PointXYZ>);

  // Fill the source point cloud with random but structured data
  std::cout << "\n=== Generating Source Point Cloud ===" << std::endl;
  for (auto & point : *cloud_source) {
    point.x = 1024 * rand() / (RAND_MAX + 1.0f);
    point.y = 1024 * rand() / (RAND_MAX + 1.0f);
    point.z = 1024 * rand() / (RAND_MAX + 1.0f);
  }

  std::cout << "Source cloud has " << cloud_source->size() << " points:" << std::endl;
  for (size_t i = 0; i < cloud_source->size(); i++) {
    std::cout << "  Point " << i << ": " << (*cloud_source)[i] << std::endl;
  }

  // Define a known transformation (translation + small rotation)
  Eigen::Matrix4f known_transform = Eigen::Matrix4f::Identity();

  // Translation component
  float tx = 0.7f;
  float ty = 0.3f;
  float tz = 0.0f;

  known_transform(0, 3) = tx;
  known_transform(1, 3) = ty;
  known_transform(2, 3) = tz;

  // Optional: Add a small rotation (5 degrees around Z axis)
  float theta = 5.0 * M_PI / 180.0; // 5 degrees
  known_transform(0, 0) = cos(theta);
  known_transform(0, 1) = -sin(theta);
  known_transform(1, 0) = sin(theta);
  known_transform(1, 1) = cos(theta);

  std::cout << "\n=== Applying Known Transformation ===" << std::endl;
  std::cout << "Transformation Matrix:" << std::endl;
  printTransformationMatrix(known_transform);

  // Apply transformation to create target cloud
  pcl::transformPointCloud(*cloud_source, *cloud_target, known_transform);

  std::cout << "\nTarget cloud has " << cloud_target->size() << " points:" << std::endl;
  for (size_t i = 0; i < cloud_target->size(); i++) {
    std::cout << "  Point " << i << ": " << (*cloud_target)[i] << std::endl;
  }

  // Visualize original clouds before ICP
  std::cout << "\n=== Showing Initial Point Clouds ===" << std::endl;
  visualizePointClouds(cloud_source, "source", cloud_target, "target");

  // Configure and apply ICP
  std::cout << "\n=== Configuring ICP Algorithm ===" << std::endl;
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

  // Set input clouds
  icp.setInputSource(cloud_source);
  icp.setInputTarget(cloud_target);

  // Configure ICP parameters for better convergence
  icp.setMaximumIterations(50);                  // Maximum number of iterations
  icp.setTransformationEpsilon(1e-8);            // Convergence criterion (transformation)
  icp.setEuclideanFitnessEpsilon(1e-6);          // Convergence criterion (fitness score)
  icp.setMaxCorrespondenceDistance(1000.0);      // Maximum distance for point correspondences (relaxed for few points)

  std::cout << "ICP Parameters:" << std::endl;
  std::cout << "  Max iterations: " << icp.getMaximumIterations() << std::endl;
  std::cout << "  Transformation epsilon: " << icp.getTransformationEpsilon() << std::endl;
  std::cout << "  Euclidean fitness epsilon: " << icp.getEuclideanFitnessEpsilon() << std::endl;
  std::cout << "  Max correspondence distance: " << icp.getMaxCorrespondenceDistance() << std::endl;

  // Perform alignment
  std::cout << "\n=== Running ICP Alignment ===" << std::endl;
  pcl::PointCloud<pcl::PointXYZ> final_cloud;
  icp.align(final_cloud);

  // Check convergence and show results
  std::cout << "\n=== ICP Results ===" << std::endl;
  if (icp.hasConverged()) {
    std::cout << "  ICP has CONVERGED successfully!" << std::endl;
    std::cout << "  Fitness Score: " << icp.getFitnessScore() << std::endl;
    std::cout << "  (Lower fitness score indicates better alignment)" << std::endl;

    std::cout << "\nEstimated Transformation Matrix:" << std::endl;
    printTransformationMatrix(icp.getFinalTransformation());

    // Analyze the found transformation
    analyzeTransformation(icp.getFinalTransformation());

    // Compare with known transformation
    std::cout << "\n=== Comparing with Known Transformation ===" << std::endl;
    Eigen::Matrix4f inverse_transform = known_transform.inverse();
    std::cout << "Expected Inverse Transformation (to align target to source):" << std::endl;
    printTransformationMatrix(inverse_transform);

    std::cout << "\nTransformation Error:" << std::endl;
    Eigen::Matrix4f error = icp.getFinalTransformation() - inverse_transform;
    printTransformationMatrix(error);
  } else {
    std::cout << "✗ ICP did NOT converge!" << std::endl;
    std::cout << "  Consider adjusting ICP parameters or input data." << std::endl;
    return EXIT_FAILURE;
  }

  // Convert the aligned cloud to pointer format for visualization
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>(final_cloud));

  // Show aligned points
  std::cout << "\nAligned cloud has " << aligned_cloud->size() << " points:" << std::endl;
  for (size_t i = 0; i < aligned_cloud->size(); i++) {
    std::cout << "  Point " << i << ": " << (*aligned_cloud)[i] << std::endl;
  }

  // Visualize clouds after ICP
  std::cout << "\n=== Showing Final Alignment Result ===" << std::endl;
  std::cout << "Colors: BLUE = Source | GREEN = Target | RED = Aligned" << std::endl;
  visualizePointClouds(cloud_source, "source", cloud_target, "target", aligned_cloud, "aligned");

  std::cout << "\n=== ICP Demo Complete ===\n" << std::endl;
  return EXIT_SUCCESS;
}
