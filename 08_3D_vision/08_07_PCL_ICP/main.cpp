/**
 * ICP using PCL demo sample with visualization
 * @author José Miguel Guerrero
 */

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>

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

  // Keep the visualizer open
  while (!viewer.wasStopped()) {
    viewer.spinOnce();
  }
}

int main(int argc, char **argv)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>(5, 1));
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);

  // Fill the original point cloud with random data
  for (auto & point : *cloud_in) {
    point.x = 1024 * rand() / (RAND_MAX + 1.0f);
    point.y = 1024 * rand() / (RAND_MAX + 1.0f);
    point.z = 1024 * rand() / (RAND_MAX + 1.0f);
  }

  std::cout << "Saved " << cloud_in->size() << " data points to input:" << std::endl;
  for (auto & point : *cloud_in) {
    std::cout << point << std::endl;
  }

  // Copy the point cloud and transform it
  *cloud_out = *cloud_in;
  for (auto & point : *cloud_out) {
    point.x += 0.7f;
    point.y += 0.3f;
  }

  std::cout << "Transformed " << cloud_out->size() << " data points:" << std::endl;
  for (auto & point : *cloud_out) {
    std::cout << point << std::endl;
  }

  // Visualize original clouds before ICP
  visualizePointClouds(cloud_in, "cloud_in", cloud_out, "cloud_out");

  // Apply ICP
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(cloud_in);
  icp.setInputTarget(cloud_out);

  pcl::PointCloud<pcl::PointXYZ> Final;
  icp.align(Final);

  // Show results
  std::cout << "Has converged: " << icp.hasConverged() <<
    " Score: " << icp.getFitnessScore() << std::endl;
  std::cout << icp.getFinalTransformation() << std::endl;

  // Convert the aligned cloud to pointer format for visualization
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>(Final));

  // Visualize clouds after ICP
  visualizePointClouds(cloud_in, "cloud_in", cloud_out, "cloud_out", aligned_cloud, "aligned");

  return 0;
}
