/**
 * @file main.cpp
 * @brief RANSAC (Random Sample Consensus) model fitting demonstration
 * @author José Miguel Guerrero Hernández
 *
 * This example shows:
 * - Generating synthetic point clouds with plane and sphere models
 * - Adding outlier noise to simulate real-world data
 * - Applying RANSAC algorithm for robust model fitting
 * - Extracting inliers that conform to the detected model
 * - Visualizing original point cloud vs. fitted model points
 * - Comparing plane and sphere model detection
 *
 * RANSAC Models:
 * - Plane Model:  Points following equation: z = -(x + y)
 * - Sphere Model: Points on unit sphere: x² + y² + z² = 1
 * - Outliers:     Random points added (20% for both models)
 *
 * RANSAC Parameters:
 * - Distance Threshold: 0.01 units (maximum point-to-model distance)
 * - Sample Size: Minimum points to define model (3 for plane, 4 for sphere)
 * - Iterations: Automatic based on probability of success
 *
 * @see https://pointclouds.org/documentation/tutorials/random_sample_consensus.html
 *
 * Usage: ./ransac_demo [option]
 * Options:
 *   (none)  - Show generated plane points
 *   -f      - Fit plane with RANSAC and show inliers
 *   -s      - Show generated sphere points
 *   -sf     - Fit sphere with RANSAC and show inliers
 *   -h      - Show help
 * Output: 3D visualization of point cloud (all points or inliers only)
 */

#include <cstdlib>
#include <iostream>
#include <thread>

#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/visualization/pcl_visualizer.h>

/**
 * @brief Creates a simple 3D visualizer to display point clouds
 * @param cloud Point cloud to visualize (constant pointer)
 * @return Shared pointer to the created visualizer object
 *
 * This function initializes a PCL visualizer with basic configuration:
 * - Black background for better contrast
 * - Adjusted point size for clear visualization
 * - Default camera parameters
 */
pcl::visualization::PCLVisualizer::Ptr simpleVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
  // Create 3D visualizer with titled window
  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));

  // Set black background for better visual contrast
  viewer->setBackgroundColor(0, 0, 0);

  // Add point cloud to visualizer with unique identifier
  viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");

  // Configure point size to improve visualization
  viewer->setPointCloudRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,
    "sample cloud");

  // Optional: add coordinate system for spatial reference
  //viewer->addCoordinateSystem(1.0, "global");

  // Initialize camera position and orientation
  viewer->initCameraParameters();

  // Set camera position further away for better initial view
  viewer->setCameraPosition(0, 0, -5,    // Camera position: 5 units back on negative z-axis
                            0, 0, 0,     // Focal point: looking at origin (0, 0, 0)
                            0, -1, 0);   // Up direction: positive y-axis

  return viewer;
}

/**
 * @brief Displays program help with all available options
 *
 * Prints information about command line options:
 * - Generation of synthetic models (plane or sphere)
 * - Application of RANSAC for robust model fitting
 * - Visualization of original points vs fitted points (inliers)
 */
void showHelp()
{
  std::cout << std::endl;
  std::cout << "**********************************************" << std::endl;
  std::cout << "*                                            *" << std::endl;
  std::cout << "*                    RANSAC                  *" << std::endl;
  std::cout << "*                                            *" << std::endl;
  std::cout << "**********************************************" << std::endl << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     (none)                  Show points generated to fit a plane." << std::endl;
  std::cout << "     -f:                     Compute RANSAC to fit a plane." << std::endl;
  std::cout << "     -s:                     Show points generated to fit a sphere." << std::endl;
  std::cout << "     -sf:                    Compute RANSAC to fit a sphere." << std::endl;
}

int main(int argc, char ** argv)
{
  // Show help if -h option is specified
  if (pcl::console::find_switch(argc, argv, "-h")) {
    showHelp();
    exit(0);
  }

  // Initialize point clouds:
  // - cloud: will contain synthetically generated points (inliers + outliers)
  // - final: will contain only inlier points detected by RANSAC
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr final (new pcl::PointCloud<pcl::PointXYZ>);

  // Configure point cloud structure
  cloud->width = 500;           // 500 points in total
  cloud->height = 1;            // Unorganized cloud (one-dimensional)
  cloud->is_dense = false;      // May contain invalid values (NaN/Inf)
  cloud->points.resize(cloud->width * cloud->height);

  // Generate synthetic points according to selected model
  for (int i = 0; i < static_cast<int>(cloud->size()); ++i) {
    // Create sphere model if -s or -sf was specified
    if (pcl::console::find_argument(
        argc, argv,
        "-s") >= 0 || pcl::console::find_argument(argc, argv, "-sf") >= 0)
    {
      // Generate random x, y coordinates in range [0, 1024]
      (*cloud)[i].x = 1024 * rand() / (RAND_MAX + 1.0);
      (*cloud)[i].y = 1024 * rand() / (RAND_MAX + 1.0);

      // Every 5th point (20%) is an outlier with random z coordinate
      if (i % 5 == 0) {
        (*cloud)[i].z = 1024 * rand() / (RAND_MAX + 1.0);
      }
      // Even points: upper hemisphere of sphere (positive z)
      else if (i % 2 == 0) {
        (*cloud)[i].z = sqrt(
          1 - ((*cloud)[i].x * (*cloud)[i].x) -
          ((*cloud)[i].y * (*cloud)[i].y));
      }
      // Odd points: lower hemisphere of sphere (negative z)
      else {
        (*cloud)[i].z = -sqrt(
          1 - ((*cloud)[i].x * (*cloud)[i].x) -
          ((*cloud)[i].y * (*cloud)[i].y));
      }
    }
    // Create plane model by default (without -s/-sf options)
    else {
      // Generate random x, y coordinates in range [0, 1024]
      (*cloud)[i].x = 1024 * rand() / (RAND_MAX + 1.0);
      (*cloud)[i].y = 1024 * rand() / (RAND_MAX + 1.0);

      // Even points (50%): outliers with random z coordinate
      if (i % 2 == 0) {
        (*cloud)[i].z = 1024 * rand() / (RAND_MAX + 1.0);
      }
      // Odd points (50%): inliers that satisfy plane equation z = -(x + y)
      else {
        (*cloud)[i].z = -1 * ((*cloud)[i].x + (*cloud)[i].y);
      }
    }
  }

  // Vector to store indices of inlier points (that fit the model)
  std::vector<int> inliers;

  // Create sample consensus models for RANSAC:
  // - model_s: sphere model (x² + y² + z² = r²)
  // - model_p: plane model (ax + by + cz + d = 0)
  pcl::SampleConsensusModelSphere<pcl::PointXYZ>::Ptr model_s(
    new pcl::SampleConsensusModelSphere<pcl::PointXYZ>(cloud));
  pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p(
    new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud));

  // Execute RANSAC to fit a plane model (option -f)
  if (pcl::console::find_argument(argc, argv, "-f") >= 0) {
    // Create RANSAC object with plane model
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_p);

    // Set distance threshold: points within 0.01 units are considered inliers
    ransac.setDistanceThreshold(.01);

    // Compute the best model that fits the data
    ransac.computeModel();

    // Get indices of points that fit the model (inliers)
    ransac.getInliers(inliers);
  }
  // Execute RANSAC to fit a sphere model (option -sf)
  else if (pcl::console::find_argument(argc, argv, "-sf") >= 0) {
    // Create RANSAC object with sphere model
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_s);

    // Set distance threshold: points within 0.01 units are considered inliers
    ransac.setDistanceThreshold(.01);

    // Compute the best sphere model that fits the data
    ransac.computeModel();

    // Get indices of points on the sphere surface (inliers)
    ransac.getInliers(inliers);
  }

  // Copy only inlier points (that fit the model) to final cloud
  // This extracts from original set only those points identified by RANSAC
  pcl::copyPointCloud(*cloud, inliers, *final);

  // Create visualizer and show appropriate points according to options:
  // - If -f or -sf was used: show only inliers (points that fit the model)
  // - Without options or only -s: show all points (inliers + outliers)
  pcl::visualization::PCLVisualizer::Ptr viewer;
  if (pcl::console::find_argument(
      argc, argv,
      "-f") >= 0 || pcl::console::find_argument(argc, argv, "-sf") >= 0)
  {
    viewer = simpleVis(final);  // Show only points that fit the model
  } else {
    viewer = simpleVis(cloud);  // Show all generated points
  }

  // Main visualization loop
  // Updates window every 100ms until user closes it
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);                                   // Process interface events
    std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Wait 100 milliseconds
  }

  return EXIT_SUCCESS;
}
