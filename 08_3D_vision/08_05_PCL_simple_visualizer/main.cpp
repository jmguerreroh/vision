/**
 * PCL visualizer demo sample
 * @author José Miguel Guerrero
 */

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace pcl::visualization;

int  main(int argc, char ** argv)
{
  // PointCloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  // Read pcd file
  int load = pcl::io::loadPCDFile<pcl::PointXYZ>("../../PCL_data/model.pcd", *cloud);
  if (argc > 1) {load = pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *cloud);}
  if (load == -1) { //* load the file
    PCL_ERROR("Couldn't read file test_pcd.pcd \n");
    return -1;
  }

  //... populate cloud
  CloudViewer viewer("Simple Cloud Viewer");
  viewer.showCloud(cloud);
  while (!viewer.wasStopped()) {}
  return 0;
}
