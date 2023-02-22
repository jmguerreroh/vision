/**
 * PCL write demo sample
 * @author José Miguel Guerrero
 *
 * VERSION   - specifies the PCD file version
 * FIELDS    - specifies the name of each dimension/field that a point can have.
 * SIZE      - specifies the size of each dimension in bytes.
 * TYPE      - specifies the type of each dimension as a char.
 * COUNT     - specifies how many elements does each dimension have. For example, x data usually has 1 element, but a feature descriptor like the VFH has 308.
 *             Basically this is a way to introduce n-D histogram descriptors at each point, and treating them as a single contiguous block of memory.
 *             By default, if COUNT is not present, all dimensions’ count is set to 1.
 * WIDTH     - specifies the width of the point cloud dataset in the number of points.
 * HEIGHT    - specifies the height of the point cloud dataset in the number of points.
 * VIEWPOINT - specifies an acquisition viewpoint for the points in the dataset. This could potentially be later on used for building transforms
 *             between different coordinate systems, or for aiding with features such as surface normals, that need a consistent orientation.
 *
 * https://pointclouds.org/documentation/tutorials/pcd_file_format.html
 */

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

int  main(int argc, char ** argv)
{
  pcl::PointCloud<pcl::PointXYZ> cloud;

  // Fill in the cloud data
  cloud.width = 5;
  cloud.height = 1;
  cloud.is_dense = false;
  cloud.points.resize(cloud.width * cloud.height);

  for (auto & point: cloud) {
    point.x = 1024 * rand() / (RAND_MAX + 1.0f);
    point.y = 1024 * rand() / (RAND_MAX + 1.0f);
    point.z = 1024 * rand() / (RAND_MAX + 1.0f);
  }

  // Write data un pcd file
  pcl::io::savePCDFileASCII("../../PCL_data/test_pcd.pcd", cloud);
  std::cerr << "Saved " << cloud.size() << " data points to test_pcd.pcd." << std::endl;

  for (const auto & point: cloud) {
    std::cerr << "    " << point.x << " " << point.y << " " << point.z << std::endl;
  }

  return 0;
}
