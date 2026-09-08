/**
 * @file main.cpp
 * @brief The standard 3D perception pipeline: downsample, remove the dominant
 *        plane, cluster what is left
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - pcl::VoxelGrid: reducing a quarter of a million points to a workable number
 * - pcl::SACSegmentation: fitting the dominant plane with RANSAC
 * - pcl::ExtractIndices: splitting the cloud into "the plane" and "everything else"
 * - pcl::EuclideanClusterExtraction: separating the remainder into objects
 *
 * Why this order. In an indoor scene most of the points belong to one supporting
 * surface: a table, the floor, a wall. That surface is rarely what you are after,
 * and while it is there every object resting on it is connected to every other
 * one through it, so no clustering can tell them apart. Removing it first is what
 * makes the objects fall out as separate connected components.
 *
 * The numbers this program prints are the ones quoted in the book: on
 * milk_cartoon_all_small_clorox.pcd the table is around 90 % of the cloud and
 * six objects survive the minimum-size filter.
 */

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

namespace Config
{
constexpr float LEAF_SIZE = 0.01f;         // Voxel side, in meters
constexpr double PLANE_TOLERANCE = 0.015;  // How close a point must be to the plane
constexpr int PLANE_ITERATIONS = 500;      // RANSAC hypotheses
constexpr double CLUSTER_TOLERANCE = 0.03; // Max gap inside one cluster
constexpr int MIN_CLUSTER = 150;           // Below this, it is sensor noise
constexpr int MAX_CLUSTER = 100000;
}

/**
 * @brief A distinct colour per cluster, so the viewer shows them apart
 */
std::vector<std::array<double, 3>> palette()
{
  return {{0.90, 0.30, 0.24}, {0.18, 0.80, 0.44}, {0.20, 0.60, 0.86},
          {0.95, 0.61, 0.07}, {0.61, 0.35, 0.71}, {0.10, 0.74, 0.61},
          {0.83, 0.33, 0.60}, {0.58, 0.65, 0.65}};
}

int main(int argc, char ** argv)
{
  // Command-line arguments; -h prints the usage. This is a PCL program, so
  // it uses PCL's own argument parser instead of cv::CommandLineParser
  if (pcl::console::find_switch(argc, argv, "-h") ||
    pcl::console::find_switch(argc, argv, "--help"))
  {
    std::cout << "Usage: " << argv[0] << " [path_to_pcd_file]" << std::endl;
    std::cout << "  Default: ../../data/pcl_data/milk_cartoon_all_small_clorox.pcd"
              << std::endl;
    return EXIT_SUCCESS;
  }
  std::string input = "../../data/pcl_data/milk_cartoon_all_small_clorox.pcd";
  if (argc > 1 && argv[1][0] != '-') {
    input = argv[1];
  }

  // ========================================
  // Step 0: load
  // ========================================
  CloudT::Ptr cloud(new CloudT);
  if (pcl::io::loadPCDFile<PointT>(input, *cloud) == -1) {
    std::cerr << "Could not read " << input << std::endl;
    return EXIT_FAILURE;
  }
  // A capture straight from an RGB-D sensor has holes, stored as NaN. They must
  // go before anything else: a NaN poisons every centroid and every distance it
  // takes part in.
  std::vector<int> keep;
  CloudT::Ptr dense(new CloudT);
  pcl::removeNaNFromPointCloud(*cloud, *dense, keep);

  std::cout << "=== 3D perception pipeline ===" << std::endl;
  std::cout << "Loaded " << cloud->size() << " points, " << dense->size()
            << " with a valid measurement" << std::endl;

  // ========================================
  // Step 1: downsample
  // ========================================
  // One centroid per occupied voxel. Besides cutting the point count, this
  // evens out the density: without it the near part of the scene has many more
  // points per square centimeter than the far part, and every later step is
  // biased towards it.
  CloudT::Ptr small(new CloudT);
  pcl::VoxelGrid<PointT> voxel;
  voxel.setInputCloud(dense);
  voxel.setLeafSize(Config::LEAF_SIZE, Config::LEAF_SIZE, Config::LEAF_SIZE);
  voxel.filter(*small);
  std::cout << "Voxel grid of " << Config::LEAF_SIZE * 100 << " cm: "
            << dense->size() << " -> " << small->size() << " points ("
            << 100.0 * small->size() / dense->size() << " %)" << std::endl;

  // ========================================
  // Step 2: the dominant plane, with RANSAC
  // ========================================
  pcl::SACSegmentation<PointT> seg;
  seg.setOptimizeCoefficients(true);     // Refit on the inliers once found
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(Config::PLANE_TOLERANCE);
  seg.setMaxIterations(Config::PLANE_ITERATIONS);
  seg.setInputCloud(small);

  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  seg.segment(*inliers, *coefficients);
  if (inliers->indices.empty()) {
    std::cerr << "No plane found" << std::endl;
    return EXIT_FAILURE;
  }
  // coefficients = [a, b, c, d] of ax + by + cz + d = 0. (a, b, c) is the
  // normal, and checking where it points is how you tell the floor from a wall.
  std::cout << "Dominant plane: " << inliers->indices.size() << " points ("
            << 100.0 * inliers->indices.size() / small->size() << " % of the cloud)"
            << std::endl;
  std::cout << "  normal (" << coefficients->values[0] << ", "
            << coefficients->values[1] << ", " << coefficients->values[2] << ")"
            << std::endl;

  CloudT::Ptr plane(new CloudT), rest(new CloudT);
  pcl::ExtractIndices<PointT> extract;
  extract.setInputCloud(small);
  extract.setIndices(inliers);
  extract.setNegative(false);
  extract.filter(*plane);               // The supporting surface
  extract.setNegative(true);
  extract.filter(*rest);                // Everything standing on it
  std::cout << "Remaining after removing the plane: " << rest->size()
            << " points" << std::endl;

  // ========================================
  // Step 3: cluster what is left
  // ========================================
  // Two points belong to the same object if a chain of points joins them in
  // which no step is longer than the tolerance. Without the KdTree the
  // neighbour search would be quadratic and this step alone would dominate the
  // runtime.
  pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
  tree->setInputCloud(rest);

  std::vector<pcl::PointIndices> clusters;
  pcl::EuclideanClusterExtraction<PointT> ec;
  ec.setClusterTolerance(Config::CLUSTER_TOLERANCE);
  ec.setMinClusterSize(Config::MIN_CLUSTER);
  ec.setMaxClusterSize(Config::MAX_CLUSTER);
  ec.setSearchMethod(tree);
  ec.setInputCloud(rest);
  ec.extract(clusters);

  std::cout << "Clustering at " << Config::CLUSTER_TOLERANCE * 100 << " cm: "
            << clusters.size() << " objects above " << Config::MIN_CLUSTER
            << " points" << std::endl;
  for (std::size_t i = 0; i < clusters.size(); i++) {
    std::cout << "  object " << i + 1 << ": " << clusters[i].indices.size()
              << " points" << std::endl;
  }

  // ========================================
  // Visualization
  // ========================================
  pcl::visualization::PCLVisualizer viewer("3D perception pipeline");
  viewer.setBackgroundColor(1.0, 1.0, 1.0);
  viewer.addPointCloud<PointT>(
    plane, pcl::visualization::PointCloudColorHandlerCustom<PointT>(
      plane, 220, 220, 220), "plane");
  viewer.setPointCloudRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "plane");

  const auto colors = palette();
  for (std::size_t i = 0; i < clusters.size(); i++) {
    CloudT::Ptr object(new CloudT);
    for (int idx : clusters[i].indices) {
      object->push_back((*rest)[idx]);
    }
    const auto & c = colors[i % colors.size()];
    const std::string id = "object" + std::to_string(i);
    viewer.addPointCloud<PointT>(
      object, pcl::visualization::PointCloudColorHandlerCustom<PointT>(
        object, c[0] * 255, c[1] * 255, c[2] * 255), id);
    viewer.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, id);
  }
  viewer.addCoordinateSystem(0.1);
  viewer.initCameraParameters();

  std::cout << "\nClose the window to exit..." << std::endl;
  while (!viewer.wasStopped()) {
    viewer.spinOnce(100);
  }
  return EXIT_SUCCESS;
}
