/**
 * PCL read demo sample
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

int main()
{
  // PointCloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  // Read pcd file
  if (pcl::io::loadPCDFile<pcl::PointXYZ>("../../PCL_data/test_pcd.pcd", *cloud) == -1) { //* load the file
    PCL_ERROR("Couldn't read file test_pcd.pcd \n");
    return -1;
  }

  // Show data
  std::cout << "Loaded "
            << cloud->width * cloud->height
            << " data points from test_pcd.pcd with the following fields: "
            << std::endl;

  // Using auto
  std::cout << std::endl << "FOR AUTO:" << std::endl;
  for (const auto & point: *cloud) {
    std::cout << "    " << point.x
              << " " << point.y
              << " " << point.z << std::endl;
  }

  // Using iterator
  std::cout << std::endl << "FOR ITERATOR:" << std::endl;
  pcl::PointCloud<pcl::PointXYZ>::const_iterator it;
  for (it = cloud->begin(); it != cloud->end(); ++it) {
    std::cout << "    " << it->x
              << " " << it->y
              << " " << it->z << std::endl;
  }
  return 0;
}
