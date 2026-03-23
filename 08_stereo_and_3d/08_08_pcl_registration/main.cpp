/**
 * @file main.cpp
 * @brief Pairwise incremental point cloud registration with ICP Non-Linear
 * @author José Miguel Guerrero Hernández
 *
 * This example shows:
 * - Loading multiple PCD files for sequential registration
 * - Voxel grid downsampling for computational efficiency
 * - Surface normal and curvature computation with KdTree
 * - Custom point representation using 4D features (x, y, z, curvature)
 * - Non-linear ICP (ICP-NL) iterative alignment with visual feedback
 * - Dual viewport visualization (original clouds vs. alignment progress)
 * - Real-time iteration display with progress feedback in visualizer window
 * - Keyboard interaction (ENTER key) for workflow control
 * - Incremental global transformation accumulation
 * - Saving intermediate registration results to PCD files
 *
 * ICP-NL Configuration:
 * - Point Representation: 4D (x, y, z, curvature)
 * - Downsampling Leaf Size: 0.05 units
 * - Normal Estimation K: 30 neighbors
 * - Max Correspondence Distance: 0.1 units (10cm)
 * - Transformation Epsilon: 1e-6
 * - Iterations per cycle: 2 (×30 cycles = 60 total)
 *
 * @see https://pointclouds.org/documentation/tutorials/pairwise_incremental_registration.html
 *
 * Usage: ./pcl_inc_registration file1.pcd file2.pcd file3.pcd ...
 * Example: ./pcl_inc_registration ../../data/pcl_data/capture000*.pcd
 * Output: Dual viewport showing original (left) and aligned clouds (right) with curvature coloring
 */

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <algorithm>
#include <vector>
#include <limits>
#include <thread>
#include <chrono>
#include <atomic>
#include <glob.h>

#include <pcl/memory.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <vtkObject.h>

// Type definitions for convenience
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

// Global mutable state (required for PCL callbacks and multi-function visualization)
pcl::visualization::PCLVisualizer * p;
int vp_1, vp_2;

// Global flag for keyboard events (thread-safe)
std::atomic<bool> keyboard_enter_pressed(false);

/**
 * @brief Structure to handle point cloud data with filename
 */
struct PCD
{
  PointCloud::Ptr cloud;
  std::string f_name;

  PCD()
  : cloud(new PointCloud) {}
};

struct PCDComparator
{
  bool operator()(const PCD & p1, const PCD & p2)
  {
    return p1.f_name < p2.f_name;
  }
};

/**
 * @brief Keyboard event callback - detects ENTER key press in visualizer window
 */
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent & event, void * cookie)
{
  (void)cookie;
  if (event.getKeySym() == "Return" && event.keyDown()) {
    keyboard_enter_pressed = true;
  }
}

/**
 * @brief Custom 4D point representation: <x, y, z, curvature>
 *
 * Extends default representation to include surface curvature, allowing ICP
 * to consider both geometric position and local surface characteristics.
 */
class MyPointRepresentation : public pcl::PointRepresentation<PointNormalT>
{
  using pcl::PointRepresentation<PointNormalT>::nr_dimensions_;

public:
  MyPointRepresentation()
  {
    nr_dimensions_ = 4;
  }

  /**
   * @brief Copies point data to a float array for ICP processing
   * @param p Input point with normal and curvature information
   * @param out Output float array [x, y, z, curvature]
   */
  virtual void copyToFloatArray(const PointNormalT & p, float * out) const
  {
    out[0] = p.x;
    out[1] = p.y;
    out[2] = p.z;
    out[3] = p.curvature;
  }
};

/**
 * @brief Display source and target clouds on the left viewport
 * @param cloud_source Source point cloud to be aligned (displayed in red)
 * @param cloud_target Target point cloud reference (displayed in green)
 *
 * Shows the initial state of the two clouds before alignment begins.
 * Displays on-screen instructions and waits for ENTER key in the visualizer window.
 * Allows interactive visualization (zoom, rotate, pan) until user presses ENTER.
 */
void showCloudsLeft(const PointCloud::Ptr cloud_source, const PointCloud::Ptr cloud_target)
{
  p->removePointCloud("vp1_target");
  p->removePointCloud("vp1_source");

  pcl::visualization::PointCloudColorHandlerCustom<PointT> src_h(cloud_source, 255, 0, 0);
  pcl::visualization::PointCloudColorHandlerCustom<PointT> tgt_h(cloud_target, 0, 255, 0);
  p->addPointCloud(cloud_source, src_h, "vp1_source", vp_1);
  p->addPointCloud(cloud_target, tgt_h, "vp1_target", vp_1);

  p->removeShape("instruction_text_vp1");
  p->addText("Source (RED) vs Target (GREEN)\nPress ENTER to begin registration...",
             10, 80, 18, 1.0, 1.0, 0.0, "instruction_text_vp1", vp_1);

  keyboard_enter_pressed = false;
  p->registerKeyboardCallback(keyboardEventOccurred, nullptr);

  while (!keyboard_enter_pressed) {
    p->spinOnce(100);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  p->removeShape("instruction_text_vp1");
  p->addText("Starting registration...", 10, 80, 18, 0.0, 1.0, 0.0, "starting_text_vp1", vp_1);
  p->spinOnce(100);
  std::this_thread::sleep_for(std::chrono::milliseconds(800));
  p->removeShape("starting_text_vp1");
}

/**
 * @brief Display source and target clouds on the right viewport with curvature coloring
 * @param cloud_target Target point cloud with normals
 * @param cloud_source Source point cloud with normals
 *
 * Shows the alignment progress during ICP iterations. Points are colored
 * according to their surface curvature values for better visualization.
 */
void showCloudsRight(
  const PointCloudWithNormals::Ptr cloud_target,
  const PointCloudWithNormals::Ptr cloud_source)
{
  p->removePointCloud("source");
  p->removePointCloud("target");

  pcl::visualization::PointCloudColorHandlerGenericField<PointNormalT> tgt_color_handler(
    cloud_target, "curvature");
  if (!tgt_color_handler.isCapable()) {
    std::cerr << "[Warning] Cannot create curvature color handler for target cloud" << std::endl;
  }

  pcl::visualization::PointCloudColorHandlerGenericField<PointNormalT> src_color_handler(
    cloud_source, "curvature");
  if (!src_color_handler.isCapable()) {
    std::cerr << "[Warning] Cannot create curvature color handler for source cloud" << std::endl;
  }

  p->addPointCloud(cloud_target, tgt_color_handler, "target", vp_2);
  p->addPointCloud(cloud_source, src_color_handler, "source", vp_2);

  p->spinOnce();
}

/**
 * @brief Expand file pattern using glob and return list of matching files
 * @param pattern File pattern to expand (e.g., "../../data/pcl_data/capture000*")
 * @return Vector of matching file paths, sorted alphabetically
 */
std::vector<std::string> expandGlobPattern(const std::string & pattern)
{
  std::vector<std::string> files;
  glob_t glob_result;

  int ret = glob(pattern.c_str(), GLOB_TILDE, nullptr, &glob_result);

  if (ret == 0) {
    for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
      files.push_back(std::string(glob_result.gl_pathv[i]));
    }
  }

  globfree(&glob_result);

  // Sort files alphabetically to ensure consistent ordering
  std::sort(files.begin(), files.end());

  return files;
}

/**
 * @brief Load a set of PCD files from a list of filenames
 * @param filenames Vector of file paths to load
 * @param models Output vector of loaded point cloud datasets
 *
 * Loads all .pcd files specified in the filenames vector,
 * removes NaN points, and stores them in the models vector.
 */
void loadData(
  const std::vector<std::string> & filenames,
  std::vector<PCD, Eigen::aligned_allocator<PCD>> & models)
{
  std::string extension(".pcd");

  std::cout << "\n=== Loading PCD Files ===" << std::endl;

  for (const auto & filename : filenames) {
    std::string fname = filename;

    if (fname.size() <= extension.size()) {
      continue;
    }

    std::transform(fname.begin(), fname.end(), fname.begin(), (int (*)(int))tolower);

    if (fname.compare(fname.size() - extension.size(), extension.size(), extension) == 0) {
      PCD m;
      m.f_name = filename;

      std::cout << "  Loading: " << filename << "...";

      if (pcl::io::loadPCDFile(filename, *m.cloud) == -1) {
        std::cerr << " FAILED" << std::endl;
        continue;
      }

      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(*m.cloud, *m.cloud, indices);

      std::cout << "  (" << m.cloud->size() << " points)" << std::endl;
      models.push_back(m);
    }
  }

  std::cout << "\nTotal files loaded: " << models.size() << std::endl;
  std::cout << "========================\n" << std::endl;
}

/**
 * @brief Align a pair of point clouds using ICP Non-Linear algorithm
 * @param cloud_src Source point cloud to be aligned
 * @param cloud_tgt Target point cloud (reference)
 * @param output Resultant aligned and merged point cloud
 * @param final_transform Resultant 4x4 transformation matrix (target to source)
 * @param downsample Enable voxel grid downsampling for large datasets
 *
 * Performs iterative alignment using custom 4D point representation (x,y,z,curvature)
 * with visual feedback showing alignment progress in the visualizer window.
 * Displays iteration count and completion status on-screen during registration.
 */
void pairAlign(
  const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt,
  PointCloud::Ptr output, Eigen::Matrix4f & final_transform, bool downsample = false)
{
  PointCloud::Ptr src(new PointCloud);
  PointCloud::Ptr tgt(new PointCloud);
  pcl::VoxelGrid<PointT> grid;

  if (downsample) {
    std::cout << "  - Downsampling clouds (leaf size: 0.05)..." << std::flush;
    grid.setLeafSize(0.05, 0.05, 0.05);
    grid.setInputCloud(cloud_src);
    grid.filter(*src);

    grid.setInputCloud(cloud_tgt);
    grid.filter(*tgt);
    std::cout << " Done" << std::endl;
    std::cout << "    Source: " << cloud_src->size() << " - " << src->size() << " points" <<
      std::endl;
    std::cout << "    Target: " << cloud_tgt->size() << " - " << tgt->size() << " points" <<
      std::endl;
  } else {
    src = cloud_src;
    tgt = cloud_tgt;
  }

  std::cout << "  - Computing surface normals and curvature (K=30)..." << std::flush;

  PointCloudWithNormals::Ptr points_with_normals_src(new PointCloudWithNormals);
  PointCloudWithNormals::Ptr points_with_normals_tgt(new PointCloudWithNormals);

  pcl::NormalEstimation<PointT, PointNormalT> norm_est;
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
  norm_est.setSearchMethod(tree);
  norm_est.setKSearch(30);

  norm_est.setInputCloud(src);
  norm_est.compute(*points_with_normals_src);
  pcl::copyPointCloud(*src, *points_with_normals_src);

  norm_est.setInputCloud(tgt);
  norm_est.compute(*points_with_normals_tgt);
  pcl::copyPointCloud(*tgt, *points_with_normals_tgt);

  std::cout << " Done" << std::endl;

  MyPointRepresentation point_representation;
  float alpha[4] = {1.0, 1.0, 1.0, 1.0};
  point_representation.setRescaleValues(alpha);

  std::cout << "  - Configuring ICP-NL algorithm..." << std::endl;
  pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;
  reg.setTransformationEpsilon(1e-6);
  reg.setMaxCorrespondenceDistance(0.1);
  reg.setPointRepresentation(pcl::make_shared<const MyPointRepresentation>(point_representation));

  reg.setInputSource(points_with_normals_src);
  reg.setInputTarget(points_with_normals_tgt);

  std::cout << "    Transformation Epsilon: 1e-6" << std::endl;
  std::cout << "    Max Correspondence Distance: 0.1" << std::endl;

  p->removeShape("icp_status_text");
  p->addText("Starting ICP alignment (30 iterations)...", 10, 50, 16, 1.0, 0.8, 0.0,
    "icp_status_text", vp_2);
  p->spinOnce(100);

  Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity(), prev = Eigen::Matrix4f::Identity(), targetToSource;
  PointCloudWithNormals::Ptr reg_result = points_with_normals_src;
  reg.setMaximumIterations(2);

  for (int i = 0; i < 30; ++i) {
    std::stringstream progress_text;
    progress_text << "ICP Iteration " << (i + 1) << "/30";

    p->removeShape("icp_status_text");
    p->addText(progress_text.str(), 10, 50, 16, 1.0, 0.8, 0.0, "icp_status_text", vp_2);

    points_with_normals_src = reg_result;

    reg.setInputSource(points_with_normals_src);
    reg.align(*reg_result);

    Ti = reg.getFinalTransformation() * Ti;

    if (std::abs((reg.getLastIncrementalTransformation() - prev).sum()) <
      reg.getTransformationEpsilon())
    {
      reg.setMaxCorrespondenceDistance(reg.getMaxCorrespondenceDistance() - 0.001);
    }

    prev = reg.getLastIncrementalTransformation();

    showCloudsRight(points_with_normals_tgt, points_with_normals_src);
  }

  p->removeShape("icp_status_text");
  p->addText("ICP Iteration 30/30 - Complete!", 10, 50, 16, 0.0, 1.0, 0.0, "icp_status_text", vp_2);
  p->spinOnce(100);
  std::this_thread::sleep_for(std::chrono::milliseconds(800));
  p->removeShape("icp_status_text");

  targetToSource = Ti.inverse();

  std::cout << "  - Applying final transformation..." << std::flush;
  pcl::transformPointCloud(*cloud_tgt, *output, targetToSource);
  std::cout << " Done" << std::endl;

  p->removePointCloud("source");
  p->removePointCloud("target");

  pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_tgt_h(output, 0, 255, 0);
  pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_src_h(cloud_src, 255, 0, 0);
  p->addPointCloud(output, cloud_tgt_h, "target", vp_2);
  p->addPointCloud(cloud_src, cloud_src_h, "source", vp_2);

  p->removeShape("instruction_text_vp2");
  p->addText("Final alignment result\nPress ENTER to continue to next pair...",
             10, 80, 18, 1.0, 1.0, 0.0, "instruction_text_vp2", vp_2);

  keyboard_enter_pressed = false;

  while (!keyboard_enter_pressed) {
    p->spinOnce(100);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // Remove instruction and show "Continuing"
  p->removeShape("instruction_text_vp2");
  p->addText("Continuing...", 10, 80, 18, 0.0, 1.0, 0.0, "continuing_text_vp2", vp_2);
  p->spinOnce(100);
  std::this_thread::sleep_for(std::chrono::milliseconds(800));
  p->removeShape("continuing_text_vp2");

  p->removePointCloud("source");
  p->removePointCloud("target");

  *output += *cloud_src;

  final_transform = targetToSource;
}

/**
 * @brief Main function - Performs pairwise incremental registration
 */
int main(int argc, char ** argv)
{
  vtkObject::GlobalWarningDisplayOff();

  std::cout << "\n╔════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║    PCL Pairwise Incremental Registration - ICP-NL          ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════╝\n" << std::endl;

  // Collect filenames from command line or use default pattern
  std::vector<std::string> filenames;

  if (argc > 1) {
    // Use files from command line arguments
    for (int i = 1; i < argc; ++i) {
      filenames.push_back(argv[i]);
    }
  } else {
    // No arguments provided - use default pattern
    std::cout <<
      "No files specified, using default pattern: /pcl_data/capture000*\n" <<
      std::endl;
    filenames = expandGlobPattern("../../data/pcl_data/capture000*");

    if (filenames.empty()) {
      std::cerr << "[Warning] No files found matching default pattern!" << std::endl;
    }
  }

  std::vector<PCD, Eigen::aligned_allocator<PCD>> data;
  loadData(filenames, data);

  // Check user input
  if (data.empty()) {
    std::cerr << "\n[Error] No PCD files loaded!" << std::endl;
    std::cerr << "\nUsage: " << argv[0] << " file1.pcd file2.pcd file3.pcd ..." << std::endl;
    std::cerr << "Example: " << argv[0] << " ../../data/pcl_data/capture*.pcd" << std::endl;
    std::cerr
                                                                                                               <<
      "\nNote: Registration will be performed pairwise: (file1+file2), then (result+file3), etc.\n"
                                                                                                               <<
      std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Successfully loaded " << data.size() << " datasets" << std::endl;

  std::cout << "\nInitializing visualizer..." << std::endl;
  p = new pcl::visualization::PCLVisualizer(argc, argv, "PCL Pairwise Incremental Registration");
  p->createViewPort(0.0, 0, 0.5, 1.0, vp_1);
  p->createViewPort(0.5, 0, 1.0, 1.0, vp_2);
  p->setBackgroundColor(0.05, 0.05, 0.05, vp_1);
  p->setBackgroundColor(0.05, 0.05, 0.05, vp_2);
  p->addText("Original Clouds", 10, 10, 16, 1.0, 1.0, 1.0, "vp1_title", vp_1);
  p->addText("Alignment Progress (Curvature)", 10, 10, 16, 1.0, 1.0, 1.0, "vp2_title", vp_2);

  // Set initial camera position for better view
  p->initCameraParameters();
  p->setCameraPosition(0, 0, -6,    // Camera position: 6 units back on negative z-axis
                       0, 0, 0,     // Focal point: looking at origin (0, 0, 0)
                       0, 1, 0);    // Up vector: Y-axis pointing up (standard orientation)

  std::cout << "Visualizer ready!\n" << std::endl;

  PointCloud::Ptr result(new PointCloud), source, target;
  Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity(), pairTransform;

  for (std::size_t i = 1; i < data.size(); ++i) {
    source = data[i - 1].cloud;
    target = data[i].cloud;

    std::cout << "\n═══════════════════════════════════════════════════════════" << std::endl;
    std::cout << "Processing Pair " << i << " of " << (data.size() - 1) << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════" << std::endl;
    std::cout << "  Source: " << data[i - 1].f_name << " (" << source->size() << " points)" <<
      std::endl;
    std::cout << "  Target: " << data[i].f_name << " (" << target->size() << " points)" <<
      std::endl;

    showCloudsLeft(source, target);

    PointCloud::Ptr temp(new PointCloud);
    pairAlign(source, target, temp, pairTransform, true);

    pcl::transformPointCloud(*temp, *result, GlobalTransform);

    GlobalTransform *= pairTransform;

    std::stringstream ss;
    ss << "../../data/pcl_data/result_" << std::setfill('0') << std::setw(3) << i << ".pcd";
    pcl::io::savePCDFile(ss.str(), *result, true);
    std::cout << "\n   Saved: " << ss.str() << " (" << result->size() << " points)" << std::endl;
  }

  delete p;

  std::cout << "\n═══════════════════════════════════════════════════════════" << std::endl;
  std::cout << "Registration Complete!" << std::endl;
  std::cout << "Total pairs processed: " << (data.size() - 1) << std::endl;
  std::cout << "═══════════════════════════════════════════════════════════\n" << std::endl;

  return EXIT_SUCCESS;
}
