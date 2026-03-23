/**
 * @file main.cpp
 * @brief Demonstrates how to create and save point cloud data to PCD format
 * @author José Miguel Guerrero Hernández
 *
 * This example shows:
 * - Creating a pcl::PointCloud with random XYZ points
 * - Saving point cloud data to PCD (Point Cloud Data) format
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
 * Usage: ./pcd_write
 * Output: Creates test_pcd.pcd in pcl_data folder
 */

#include <cstdlib>
#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>


int main(int argc, char ** argv)
{
  (void)argc;
  (void)argv;

  // Create point cloud object
  // pcl::PointXYZ contains only x, y, z coordinates (no color/intensity)
  pcl::PointCloud<pcl::PointXYZ> cloud;

  // Configure point cloud structure
  constexpr int NUM_POINTS = 5;
  cloud.width = NUM_POINTS;     // Number of points (unorganized cloud)
  cloud.height = 1;             // Height=1 means unorganized (no grid structure)
  cloud.is_dense = false;       // May contain NaN/Inf values
  cloud.points.resize(cloud.width * cloud.height);

  // Fill with random points in range [0, 1024)
  for (auto & point : cloud) {
    point.x = 1024.0f * rand() / (RAND_MAX + 1.0f);
    point.y = 1024.0f * rand() / (RAND_MAX + 1.0f);
    point.z = 1024.0f * rand() / (RAND_MAX + 1.0f);
  }

  // Save to PCD file (ASCII format for human readability)
  // Other options: savePCDFileBinary() for smaller files
  //                savePCDFileBinaryCompressed() for smallest files
  const std::string output_file = "../../data/pcl_data/test_pcd.pcd";
  pcl::io::savePCDFileASCII(output_file, cloud);

  // Display saved points
  std::cout << "Saved " << cloud.size() << " points to " << output_file << std::endl;
  std::cout << "\nPoint coordinates:" << std::endl;
  for (const auto & point : cloud) {
    std::cout << "  (" << point.x << ", " << point.y << ", " << point.z << ")" << std::endl;
  }

  return EXIT_SUCCESS;
}
