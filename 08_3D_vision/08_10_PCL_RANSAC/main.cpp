/**
 * PCL RANSAC demo sample
 * @author José Miguel Guerrero
 */

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

using namespace std::chrono_literals;

pcl::visualization::PCLVisualizer::Ptr simpleVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,
    "sample cloud");
  //viewer->addCoordinateSystem (1.0, "global");
  viewer->initCameraParameters();
  return viewer;
}

void showHelp(char * filename)
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

  //Show help
  if (pcl::console::find_switch(argc, argv, "-h")) {
    showHelp(argv[0]);
    exit(0);
  }

  // initialize PointClouds
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr final (new pcl::PointCloud<pcl::PointXYZ>);

  // Populate our PointCloud with points
  cloud->width = 500;
  cloud->height = 1;
  cloud->is_dense = false;
  cloud->points.resize(cloud->width * cloud->height);
  for (int i = 0; i < cloud->size(); ++i) {
    // Create sphere model
    if (pcl::console::find_argument(
        argc, argv,
        "-s") >= 0 || pcl::console::find_argument(argc, argv, "-sf") >= 0)
    {
      (*cloud)[i].x = 1024 * rand() / (RAND_MAX + 1.0);
      (*cloud)[i].y = 1024 * rand() / (RAND_MAX + 1.0);
      if (i % 5 == 0) {
        (*cloud)[i].z = 1024 * rand() / (RAND_MAX + 1.0);
      } else if (i % 2 == 0) {
        (*cloud)[i].z = sqrt(
          1 - ((*cloud)[i].x * (*cloud)[i].x) -
          ((*cloud)[i].y * (*cloud)[i].y));
      } else {
        (*cloud)[i].z = -sqrt(
          1 - ((*cloud)[i].x * (*cloud)[i].x) -
          ((*cloud)[i].y * (*cloud)[i].y));
      }
    }
    // Create plane model
    else {
      (*cloud)[i].x = 1024 * rand() / (RAND_MAX + 1.0);
      (*cloud)[i].y = 1024 * rand() / (RAND_MAX + 1.0);
      if (i % 2 == 0) {
        (*cloud)[i].z = 1024 * rand() / (RAND_MAX + 1.0);
      } else {
        (*cloud)[i].z = -1 * ((*cloud)[i].x + (*cloud)[i].y);
      }
    }
  }

  // Vector used to cointain the inliers indexes that fit the model
  std::vector<int> inliers;

  // Created RandomSampleConsensus object and compute the appropriated model
  pcl::SampleConsensusModelSphere<pcl::PointXYZ>::Ptr model_s(new pcl::SampleConsensusModelSphere<pcl::PointXYZ>(
      cloud));                                                                                                             // Sphere model
  pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p(new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(
      cloud));                                                                                                            // Plane model

  // Calculate RANSAC for plane
  if (pcl::console::find_argument(argc, argv, "-f") >= 0) {
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_p);
    ransac.setDistanceThreshold(.01);
    ransac.computeModel();
    ransac.getInliers(inliers);
  }
  // Calculate RANSAC for sphere
  else if (pcl::console::find_argument(argc, argv, "-sf") >= 0) {
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_s);
    ransac.setDistanceThreshold(.01);
    ransac.computeModel();
    ransac.getInliers(inliers);
  }

  // Copies all inliers of the model computed to another PointCloud
  pcl::copyPointCloud(*cloud, inliers, *final);

  // Creates the visualization object and adds either our original cloud or all of the inliers
  // depending on the command line arguments specified.
  pcl::visualization::PCLVisualizer::Ptr viewer;
  if (pcl::console::find_argument(
      argc, argv,
      "-f") >= 0 || pcl::console::find_argument(argc, argv, "-sf") >= 0)
  {
    viewer = simpleVis(final); // Points that fit the model
  } else {
    viewer = simpleVis(cloud); // Original points

  }
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
    std::this_thread::sleep_for(100ms);
  }
  return 0;
}
