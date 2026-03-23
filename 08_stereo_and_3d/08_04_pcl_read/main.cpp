/**
 * @file main.cpp
 * @brief Demonstrates how to load and read point cloud data from PCD files
 * @author José Miguel Guerrero Hernández
 *
 * This example shows:
 * - Loading a PCD file using pcl::io::loadPCDFile()
 * - Iterating through point cloud data (range-based for and iterators)
 * - Accessing point coordinates (x, y, z)
 *
 * PCD File Format Fields:
 * - VERSION:   PCD file version
 * - FIELDS:    Name of each dimension (x, y, z, rgb, normal_x, etc.)
 * - SIZE:      Size of each dimension in bytes
 * - TYPE:      Type of each dimension (I=signed, U=unsigned, F=float)
 * - COUNT:     Number of elements per dimension (1 for xyz, 308 for VFH)
 * - WIDTH:     Number of points per row (total points if unorganized)
 * - HEIGHT:    Number of rows (1 if unorganized point cloud)
 * - VIEWPOINT: Acquisition viewpoint (tx ty tz qw qx qy qz)
 * - POINTS:    Total number of points
 * - DATA:      Data type (ascii or binary)
 *
 * @see https://pointclouds.org/documentation/tutorials/pcd_file_format.html
 *
 * Usage: ./pcd_read
 * Input: Reads test_pcd.pcd from pcl_data folder
 */

#include <cstdlib>
#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

/**
 * @brief Loads a PCD file and displays its contents
 * @return 0 on success, -1 on error
 */
int main(int argc, char ** argv)
{
  (void)argc;
  (void)argv;

  // Create point cloud pointer
  // Using Ptr (boost::shared_ptr) for automatic memory management
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  // Load PCD file
  // loadPCDFile() returns -1 on error, 0 on success
  // Other options: loadPCDFile() with Eigen::Vector4f for sensor origin
  const std::string input_file = "../../data/pcl_data/test_pcd.pcd";
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(input_file, *cloud) == -1) {
    PCL_ERROR("Couldn't read file %s\n", input_file.c_str());
    return EXIT_FAILURE;
  }

  // Display point cloud info
  std::cout << "Loaded " << cloud->size() << " points from " << input_file << std::endl;
  std::cout << "  Width:    " << cloud->width << std::endl;
  std::cout << "  Height:   " << cloud->height << std::endl;
  std::cout << "  Is dense: " << (cloud->is_dense ? "yes" : "no") << std::endl;

  // Method 1: Range-based for loop (C++11, recommended)
  std::cout << "\n=== Range-based for loop ===" << std::endl;
  for (const auto & point : *cloud) {
    std::cout << "  (" << point.x << ", " << point.y << ", " << point.z << ")" << std::endl;
  }

  // Method 2: Iterator-based loop (traditional)
  std::cout << "\n=== Iterator-based loop ===" << std::endl;
  for (auto it = cloud->cbegin(); it != cloud->cend(); ++it) {
    std::cout << "  (" << it->x << ", " << it->y << ", " << it->z << ")" << std::endl;
  }

  // Method 3: Index-based access
  std::cout << "\n=== Index-based access ===" << std::endl;
  for (size_t i = 0; i < cloud->size(); ++i) {
    const auto & point = cloud->points[i];
    std::cout << "  [" << i << "] (" << point.x << ", " << point.y << ", " << point.z << ")" <<
      std::endl;
  }

  return EXIT_SUCCESS;
}
