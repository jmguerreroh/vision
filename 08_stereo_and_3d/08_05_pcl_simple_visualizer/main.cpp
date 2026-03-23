/**
 * @file main.cpp
 * @brief Demonstrates basic point cloud visualization using PCL CloudViewer
 * @author José Miguel Guerrero Hernández
 *
 * This example shows:
 * - Loading point cloud data from PCD files
 * - Displaying point clouds using pcl::visualization::CloudViewer
 * - Computing and displaying cloud statistics (bounding box, centroid, dimensions)
 * - Interactive visualization with mouse and keyboard controls
 *
 * CloudViewer Features:
 * - Simple interface for quick point cloud visualization
 * - Automatic camera setup and rendering
 * - Built-in interaction controls (rotate, zoom, pan)
 * - Custom callbacks for enhanced functionality
 * - Background coordinate system display
 *
 * @see https://pointclouds.org/documentation/tutorials/cloud_viewer.html
 *
 * Usage: ./simple_visualizer [path_to_pcd_file]
 * Default: Loads ../../data/pcl_data/head.pcd if no argument provided
 */

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/common.h>
#include <thread>
#include <chrono>
#include <vtkObject.h>

/**
 * @brief Calculates and displays detailed statistics about a point cloud
 * @param cloud The input point cloud
 */
void displayCloudStatistics(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud)
{
  std::cout << "\n=== Point Cloud Statistics ===" << std::endl;
  std::cout << "Total points: " << cloud->size() << std::endl;
  std::cout << "Is organized: " << (cloud->isOrganized() ? "Yes" : "No") << std::endl;

  if (cloud->isOrganized()) {
    std::cout << "Width: " << cloud->width << std::endl;
    std::cout << "Height: " << cloud->height << std::endl;
  }

  if (cloud->empty()) {
    std::cout << "WARNING: Point cloud is empty!" << std::endl;
    return;
  }

  // Find min and max points
  pcl::PointXYZ min_pt, max_pt;
  pcl::getMinMax3D(*cloud, min_pt, max_pt);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "\nBounding Box:" << std::endl;
  std::cout << "  Min: (" << min_pt.x << ", " << min_pt.y << ", " << min_pt.z << ")" << std::endl;
  std::cout << "  Max: (" << max_pt.x << ", " << max_pt.y << ", " << max_pt.z << ")" << std::endl;

  // Calculate dimensions
  float dx = max_pt.x - min_pt.x;
  float dy = max_pt.y - min_pt.y;
  float dz = max_pt.z - min_pt.z;

  std::cout << "\nDimensions:" << std::endl;
  std::cout << "  Width (X):  " << dx << std::endl;
  std::cout << "  Depth (Y):  " << dy << std::endl;
  std::cout << "  Height (Z): " << dz << std::endl;

  // Calculate centroid
  float cx = (min_pt.x + max_pt.x) / 2.0f;
  float cy = (min_pt.y + max_pt.y) / 2.0f;
  float cz = (min_pt.z + max_pt.z) / 2.0f;

  std::cout << "\nCentroid: (" << cx << ", " << cy << ", " << cz << ")" << std::endl;
  std::cout << "==============================\n" << std::endl;
}

/**
 * @brief Displays usage instructions for the viewer
 */
void displayUsageInstructions()
{
  std::cout << "\n=== Cloud Viewer Controls ===" << std::endl;
  std::cout << "  Mouse Controls:" << std::endl;
  std::cout << "    - Left click + drag:   Rotate view" << std::endl;
  std::cout << "    - Right click + drag:  Zoom in/out" << std::endl;
  std::cout << "    - Mouse wheel:         Zoom in/out" << std::endl;
  std::cout << "    - Middle click + drag: Pan view" << std::endl;
  std::cout << "\n  Keyboard Commands:" << std::endl;
  std::cout << "    - r/R: Reset camera view" << std::endl;
  std::cout << "    - q/Q: Quit viewer" << std::endl;
  std::cout << "    - j/J: Take screenshot" << std::endl;
  std::cout << "    - g/G: Display/hide coordinate axes" << std::endl;
  std::cout << "    - +/-: Increase/decrease point size" << std::endl;
  std::cout << "============================\n" << std::endl;
}

/**
 * @brief Custom viewer callback to add extra features
 */
void viewerOneOff(pcl::visualization::PCLVisualizer & viewer)
{
  std::cout << "Setting up enhanced viewer..." << std::endl;
  // This function is called once when viewer is initialized
  viewer.setBackgroundColor(0.0, 0.0, 0.0);
  viewer.addCoordinateSystem(1.0);
}

int main(int argc, char ** argv)
{
  // Disable VTK warning messages
  vtkObject::GlobalWarningDisplayOff();

  std::cout << "\n=== PCL Simple Cloud Viewer ===" << std::endl;

  // Determine file path
  std::string filepath = "../../data/pcl_data/head.pcd";
  if (argc > 1) {
    filepath = argv[1];
    std::cout << "Using provided file: " << filepath << std::endl;
  } else {
    std::cout << "Using default file: " << filepath << std::endl;
    std::cout << "Usage: " << argv[0] << " [path_to_pcd_file]" << std::endl;
  }

  // Create point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  // Load PCD file
  std::cout << "\nLoading point cloud..." << std::endl;
  int load_result = pcl::io::loadPCDFile<pcl::PointXYZ>(filepath, *cloud);

  if (load_result == -1) {
    PCL_ERROR("Failed to read file: %s\n", filepath.c_str());
    std::cerr << "\nPossible reasons:" << std::endl;
    std::cerr << "  - File does not exist" << std::endl;
    std::cerr << "  - File path is incorrect" << std::endl;
    std::cerr << "  - File is corrupted or invalid PCD format" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Successfully loaded point cloud!" << std::endl;

  // Display cloud statistics
  displayCloudStatistics(cloud);

  // Display usage instructions
  displayUsageInstructions();

  // Create and configure viewer
  std::cout << "Initializing viewer..." << std::endl;
  pcl::visualization::CloudViewer viewer("PCL Simple Cloud Viewer");

  // Set up one-off callback (called once when viewer starts)
  viewer.runOnVisualizationThreadOnce(viewerOneOff);

  // Display the cloud
  viewer.showCloud(cloud);

  std::cout << "Viewer is ready!" << std::endl;
  std::cout << "\nViewing point cloud... (close window to exit)" << std::endl;

  // Keep viewer open until user closes window
  while (!viewer.wasStopped()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  std::cout << "\n=== Viewer Closed ===" << std::endl;
  std::cout << "Thank you for using PCL Simple Cloud Viewer!\n" << std::endl;

  return EXIT_SUCCESS;
}
