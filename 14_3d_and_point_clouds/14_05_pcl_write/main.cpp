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
 * Usage: ./14_05_pcl_write [path_to_output_pcd]
 * Output: Creates test_pcd.pcd in pcl_data folder
 */

#include <cstdlib>
#include <iostream>
#include <random>

#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>


int main(int argc, char ** argv)
{
  // Command-line arguments; -h prints the usage. This is a PCL program, so
  // it uses PCL's own argument parser instead of cv::CommandLineParser
  if (pcl::console::find_switch(argc, argv, "-h") ||
    pcl::console::find_switch(argc, argv, "--help"))
  {
    std::cout << "Usage: " << argv[0] << " [path_to_output_pcd]" << std::endl;
    return EXIT_SUCCESS;
  }

  // Create point cloud object
  // pcl::PointXYZ contains only x, y, z coordinates (no color/intensity)
  pcl::PointCloud<pcl::PointXYZ> cloud;

  // Configure point cloud structure
  constexpr int NUM_POINTS = 5;
  cloud.width = NUM_POINTS;     // Number of points (unorganized cloud)
  cloud.height = 1;             // Height=1 means unorganized (no grid structure)
  cloud.is_dense = false;       // May contain NaN/Inf values
  cloud.points.resize(cloud.width * cloud.height);

  // Fill with pseudo-random points in range [0, 1024).
  // std::mt19937 with a fixed seed is used instead of rand() on purpose: the
  // generator is specified by the standard, so every machine produces the same
  // five points and therefore the same file. That is what lets the resulting
  // test_pcd.pcd be kept under version control, so 14_06_pcl_read works on a
  // fresh clone without having to run this example first
  std::mt19937 generator(42);
  const auto sample = [&generator]() {
      return 1024.0f * (generator() / static_cast<float>(std::mt19937::max()) );
    };
  for (auto & point : cloud) {
    point.x = sample();
    point.y = sample();
    point.z = sample();
  }

  // Save to PCD file (ASCII format for human readability)
  // Other options: savePCDFileBinary() for smaller files
  //                savePCDFileBinaryCompressed() for smallest files
  // Written into data/pcl_data on purpose: the NEXT example (14_06_pcl_read)
  // reads this exact file, regardless of which directory each one runs from.
  // The file is also under version control, so 14_06 does not depend on this
  // example having run: the seeded generator above reproduces it exactly
  const std::string output_file = argc > 1 ? argv[1] : "../../data/pcl_data/test_pcd.pcd";
  pcl::io::savePCDFileASCII(output_file, cloud);

  // Display saved points
  std::cout << "Saved " << cloud.size() << " points to " << output_file << std::endl;
  std::cout << "\nPoint coordinates:" << std::endl;
  for (const auto & point : cloud) {
    std::cout << "  (" << point.x << ", " << point.y << ", " << point.z << ")" << std::endl;
  }

  return EXIT_SUCCESS;
}
