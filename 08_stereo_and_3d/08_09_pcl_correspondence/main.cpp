/**
 * @file main.cpp
 * @brief 3D object recognition using correspondence grouping and SHOT descriptors
 * @author José Miguel Guerrero Hernández
 *
 * This example shows:
 * - Loading model and scene point clouds from PCD files
 * - Computing surface normals with OpenMP acceleration
 * - Uniform sampling for keypoint extraction
 * - SHOT352 (Signature of Histograms of OrienTations) descriptor computation
 * - KdTree-based feature matching between model and scene
 * - Reference frame estimation with BOARD (BOARd of Reference Directions)
 * - Correspondence grouping using Hough3D or Geometric Consistency
 * - Multiple instance detection with 6-DOF pose estimation
 * - Interactive visualization with keyboard controls (k: keypoints, c: correspondences)
 *
 * Algorithm Parameters:
 * - model_ss:    Model uniform sampling radius (default: 0.01)
 * - scene_ss:    Scene uniform sampling radius (default: 0.03)
 * - rf_rad:      Reference frame radius (default: 0.015)
 * - descr_rad:   SHOT descriptor radius (default: 0.02)
 * - cg_size:     Clustering bin size (default: 0.01)
 * - cg_thresh:   Clustering threshold (default: 5.0)
 *
 * @see https://pointclouds.org/documentation/tutorials/correspondence_grouping.html
 *
 * Usage: ./correspondence_grouping model.pcd scene.pcd [options]
 * Options: -h (help), -k (show keypoints), -c (show correspondences), -r (use cloud resolution)
 *          --algorithm (Hough|GC), --model_ss, --scene_ss, --rf_rad, --descr_rad, --cg_size, --cg_thresh
 * Output: Visualization showing detected model instances in the scene with transformation matrices
 */

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <sstream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>

typedef pcl::PointXYZRGBA PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;

struct AppConfig {
  std::string model_filename;
  std::string scene_filename;
  bool show_keypoints = false;
  bool show_correspondences = false;
  bool use_cloud_resolution = false;
  bool use_hough = true;
  float model_ss = 0.01f;    // Model uniform sampling radius
  float scene_ss = 0.03f;    // Scene uniform sampling radius
  float rf_rad = 0.015f;     // Reference frame radius
  float descr_rad = 0.02f;   // SHOT descriptor radius
  float cg_size = 0.01f;     // Clustering bin size
  float cg_thresh = 5.0f;    // Clustering threshold
};

AppConfig config;

void showHelp(const char * filename)
{
  std::cout << "\n***************************************************************************\n";
  std::cout << "*                                                                         *\n";
  std::cout << "*          Correspondence Grouping - 3D Object Recognition                *\n";
  std::cout << "*                                                                         *\n";
  std::cout << "***************************************************************************\n\n";
  std::cout << "Usage: " << filename << " model.pcd scene.pcd [Options]\n\n";
  std::cout << "Options:\n";
  std::cout << "  -h                      Show this help\n";
  std::cout << "  -k                      Show keypoints initially\n";
  std::cout << "  -c                      Show correspondences initially\n";
  std::cout << "  -r                      Use cloud resolution scaling\n";
  std::cout << "  --algorithm (Hough|GC)  Clustering algorithm (default: Hough)\n";
  std::cout << "  --model_ss val          Model sampling radius (default: 0.01)\n";
  std::cout << "  --scene_ss val          Scene sampling radius (default: 0.03)\n";
  std::cout << "  --rf_rad val            Reference frame radius (default: 0.015)\n";
  std::cout << "  --descr_rad val         Descriptor radius (default: 0.02)\n";
  std::cout << "  --cg_size val           Clustering bin size (default: 0.01)\n";
  std::cout << "  --cg_thresh val         Clustering threshold (default: 5.0)\n\n";
  std::cout << "Interactive Controls (in visualizer window):\n";
  std::cout << "  k                       Toggle keypoints display\n";
  std::cout << "  c                       Toggle correspondences display\n";
  std::cout << "  q/ESC                   Quit\n\n";
}

void parseCommandLine(int argc, char * argv[])
{
  if (pcl::console::find_switch(argc, argv, "-h")) {
    showHelp(argv[0]);
    exit(0);
  }

  std::vector<int> filenames = pcl::console::parse_file_extension_argument(argc, argv, ".pcd");
  if (filenames.size() != 2) {
    if (filenames.size() == 0) {
      std::cout << "\n[Info] No input files provided, using default files:\n";
      config.model_filename = "../../data/pcl_data/milk.pcd";
      config.scene_filename = "../../data/pcl_data/milk_cartoon_all_small_clorox.pcd";
      std::cout << "  Model: " << config.model_filename << "\n";
      std::cout << "  Scene: " << config.scene_filename << "\n" << std::endl;
    } else {
      std::cerr << "\n[Error] Two PCD files required (model and scene)\n";
      showHelp(argv[0]);
      exit(-1);
    }
  } else {
    config.model_filename = argv[filenames[0]];
    config.scene_filename = argv[filenames[1]];
  }

  config.show_keypoints = pcl::console::find_switch(argc, argv, "-k");
  config.show_correspondences = pcl::console::find_switch(argc, argv, "-c");
  config.use_cloud_resolution = pcl::console::find_switch(argc, argv, "-r");

  std::string used_algorithm;
  if (pcl::console::parse_argument(argc, argv, "--algorithm", used_algorithm) != -1) {
    if (used_algorithm == "Hough") {
      config.use_hough = true;
    } else if (used_algorithm == "GC") {
      config.use_hough = false;
    } else {
      std::cerr << "\n[Error] Invalid algorithm. Use 'Hough' or 'GC'\n";
      showHelp(argv[0]);
      exit(-1);
    }
  }

  pcl::console::parse_argument(argc, argv, "--model_ss", config.model_ss);
  pcl::console::parse_argument(argc, argv, "--scene_ss", config.scene_ss);
  pcl::console::parse_argument(argc, argv, "--rf_rad", config.rf_rad);
  pcl::console::parse_argument(argc, argv, "--descr_rad", config.descr_rad);
  pcl::console::parse_argument(argc, argv, "--cg_size", config.cg_size);
  pcl::console::parse_argument(argc, argv, "--cg_thresh", config.cg_thresh);
}

/**
 * @brief Compute average cloud resolution (mean distance to nearest neighbor)
 * @param cloud Input point cloud
 * @return Average resolution in cloud units, or 0.0 if computation fails
 *
 * Used for automatic parameter scaling when -r flag is provided.
 */
double computeCloudResolution(const pcl::PointCloud<PointType>::ConstPtr & cloud)
{
  double resolution = 0.0;
  int valid_points = 0;
  std::vector<int> indices(2);
  std::vector<float> sqr_distances(2);
  pcl::search::KdTree<PointType> tree;
  tree.setInputCloud(cloud);

  for (std::size_t i = 0; i < cloud->size(); ++i) {
    if (!std::isfinite((*cloud)[i].x)) {
      continue;
    }

    int found = tree.nearestKSearch(i, 2, indices, sqr_distances);
    if (found == 2) {
      resolution += sqrt(sqr_distances[1]);
      ++valid_points;
    }
  }

  return (valid_points != 0) ? resolution / valid_points : 0.0;
}

/**
 * @brief Data structure for interactive visualization
 *
 * Contains all point clouds and recognition results needed for
 * real-time visualization updates when toggling display options.
 */
struct VisualizationData
{
  pcl::PointCloud<PointType>::Ptr scene;
  pcl::PointCloud<PointType>::Ptr scene_keypoints;
  pcl::PointCloud<PointType>::Ptr off_scene_model;
  pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> * rototranslations;
  std::vector<pcl::Correspondences> * clustered_corrs;
  pcl::PointCloud<PointType>::Ptr model;
  bool * show_keypoints_flag;
  bool * show_correspondences_flag;
};

/**
 * @brief Update visualizer with current display settings
 * @param viewer PCL visualizer instance
 * @param data Visualization data containing clouds and display flags
 *
 * Called when user toggles keypoints or correspondences display.
 */
void updateVisualization(
  pcl::visualization::PCLVisualizer & viewer,
  const VisualizationData & data)
{
  viewer.removeAllShapes();
  viewer.removeAllPointClouds();

  viewer.addPointCloud(data.scene, "scene_cloud");

  if (*data.show_keypoints_flag || *data.show_correspondences_flag) {
    pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler(
      data.off_scene_model, 255, 255, 128);
    viewer.addPointCloud(data.off_scene_model, off_scene_model_color_handler, "off_scene_model");
  }

  if (*data.show_keypoints_flag) {
    pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler(
      data.scene_keypoints, 0, 0, 255);
    viewer.addPointCloud(
      data.scene_keypoints, scene_keypoints_color_handler,
      "scene_keypoints");
    viewer.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5,
      "scene_keypoints");

    pcl::visualization::PointCloudColorHandlerCustom<PointType>
    off_scene_model_keypoints_color_handler(data.off_scene_model_keypoints, 0, 0, 255);
    viewer.addPointCloud(
      data.off_scene_model_keypoints, off_scene_model_keypoints_color_handler,
      "off_scene_model_keypoints");
    viewer.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5,
      "off_scene_model_keypoints");
  }

  for (std::size_t i = 0; i < data.rototranslations->size(); ++i) {
    auto rotated_model = pcl::make_shared<pcl::PointCloud<PointType>>();
    pcl::transformPointCloud(*data.model, *rotated_model, (*data.rototranslations)[i]);

    std::stringstream ss_cloud;
    ss_cloud << "instance" << i;

    pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler(
      rotated_model, 255, 0, 0);
    viewer.addPointCloud(rotated_model, rotated_model_color_handler, ss_cloud.str());

    if (*data.show_correspondences_flag) {
      for (std::size_t j = 0; j < (*data.clustered_corrs)[i].size(); ++j) {
        std::stringstream ss_line;
        ss_line << "correspondence_line" << i << "_" << j;
        const PointType & model_point = data.off_scene_model_keypoints->at(
          (*data.clustered_corrs)[i][j].index_query);
        const PointType & scene_point = data.scene_keypoints->at(
          (*data.clustered_corrs)[i][j].index_match);

        viewer.addLine<PointType, PointType>(model_point, scene_point, 0, 255, 0, ss_line.str());
      }
    }
  }

  viewer.addText(
    std::string("[K] Keypoints: ") + (*data.show_keypoints_flag ? "ON" : "OFF") +
    "  [C] Correspondences: " + (*data.show_correspondences_flag ? "ON" : "OFF"),
    10, 10, 14, 1.0, 1.0, 0.0, "status_text");
}

/**
 * @brief Callback data for keyboard events
 *
 * Wraps visualization data and viewer pointer for keyboard callback.
 */
struct CallbackData
{
  VisualizationData * vis_data;
  pcl::visualization::PCLVisualizer * viewer;
};

/**
 * @brief Keyboard event handler for interactive controls
 * @param event Keyboard event from PCL visualizer
 * @param callback_data_void Pointer to CallbackData structure
 *
 * Handles K/k for keypoints and C/c for correspondences toggle.
 */
void keyboardEventOccurred(
  const pcl::visualization::KeyboardEvent & event,
  void * callback_data_void)
{
  CallbackData * callback_data = static_cast<CallbackData *>(callback_data_void);
  VisualizationData * data = callback_data->vis_data;
  pcl::visualization::PCLVisualizer * viewer = callback_data->viewer;

  if (event.keyDown()) {
    if (event.getKeySym() == "k" || event.getKeySym() == "K") {
      *data->show_keypoints_flag = !(*data->show_keypoints_flag);
      std::cout << "Keypoints: " << (*data->show_keypoints_flag ? "ON" : "OFF") << std::endl;
      updateVisualization(*viewer, *data);
    } else if (event.getKeySym() == "c" || event.getKeySym() == "C") {
      *data->show_correspondences_flag = !(*data->show_correspondences_flag);
      std::cout << "Correspondences: " << (*data->show_correspondences_flag ? "ON" : "OFF") <<
        std::endl;
      updateVisualization(*viewer, *data);
    }
  }
}

int main(int argc, char ** argv)
{
  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

  parseCommandLine(argc, argv);

  std::cout << "\n=== 3D Object Recognition - Correspondence Grouping ===\n" << std::endl;

  auto model = pcl::make_shared<pcl::PointCloud<PointType>>();
  auto model_keypoints = pcl::make_shared<pcl::PointCloud<PointType>>();
  auto scene = pcl::make_shared<pcl::PointCloud<PointType>>();
  auto scene_keypoints = pcl::make_shared<pcl::PointCloud<PointType>>();
  auto model_normals = pcl::make_shared<pcl::PointCloud<NormalType>>();
  auto scene_normals = pcl::make_shared<pcl::PointCloud<NormalType>>();
  auto model_descriptors = pcl::make_shared<pcl::PointCloud<DescriptorType>>();
  auto scene_descriptors = pcl::make_shared<pcl::PointCloud<DescriptorType>>();

  std::cout << "Loading point clouds..." << std::endl;
  if (pcl::io::loadPCDFile(config.model_filename, *model) < 0) {
    std::cerr << "[Error] Cannot load model: " << config.model_filename << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "  Model: " << config.model_filename << " (" << model->size() << " points)" << std::endl;

  if (pcl::io::loadPCDFile(config.scene_filename, *scene) < 0) {
    std::cerr << "[Error] Cannot load scene: " << config.scene_filename << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "  Scene: " << config.scene_filename << " (" << scene->size() << " points)" << std::endl;

  if (config.use_cloud_resolution) {
    std::cout << "\nComputing cloud resolution..." << std::endl;
    float resolution = static_cast<float>(computeCloudResolution(model));
    if (resolution != 0.0f) {
      config.model_ss *= resolution;
      config.scene_ss *= resolution;
      config.rf_rad *= resolution;
      config.descr_rad *= resolution;
      config.cg_size *= resolution;
    }

    std::cout << "  Model resolution:       " << resolution << std::endl;
    std::cout << "  Model sampling size:    " << config.model_ss << std::endl;
    std::cout << "  Scene sampling size:    " << config.scene_ss << std::endl;
    std::cout << "  Reference frame radius: " << config.rf_rad << std::endl;
    std::cout << "  SHOT descriptor radius: " << config.descr_rad << std::endl;
    std::cout << "  Clustering bin size:    " << config.cg_size << std::endl;
  }

  std::cout << "\nComputing surface normals..." << std::flush;
  pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
  norm_est.setKSearch(10);
  norm_est.setInputCloud(model);
  norm_est.compute(*model_normals);

  norm_est.setInputCloud(scene);
  norm_est.compute(*scene_normals);
  std::cout << " Done" << std::endl;

  std::cout << "\nExtracting keypoints via uniform sampling..." << std::endl;
  pcl::UniformSampling<PointType> uniform_sampling;
  uniform_sampling.setInputCloud(model);
  uniform_sampling.setRadiusSearch(config.model_ss);
  uniform_sampling.filter(*model_keypoints);
  std::cout << "  Model keypoints: " << model_keypoints->size() << " / " << model->size() <<
    std::endl;

  uniform_sampling.setInputCloud(scene);
  uniform_sampling.setRadiusSearch(config.scene_ss);
  uniform_sampling.filter(*scene_keypoints);
  std::cout << "  Scene keypoints: " << scene_keypoints->size() << " / " << scene->size() <<
    std::endl;

  std::cout << "\nComputing SHOT352 descriptors..." << std::flush;
  pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
  descr_est.setRadiusSearch(config.descr_rad);

  descr_est.setInputCloud(model_keypoints);
  descr_est.setInputNormals(model_normals);
  descr_est.setSearchSurface(model);
  descr_est.compute(*model_descriptors);

  descr_est.setInputCloud(scene_keypoints);
  descr_est.setInputNormals(scene_normals);
  descr_est.setSearchSurface(scene);
  descr_est.compute(*scene_descriptors);
  std::cout << " Done" << std::endl;

  std::cout << "\nFinding correspondences with KdTree..." << std::flush;
  auto model_scene_corrs = pcl::make_shared<pcl::Correspondences>();

  pcl::KdTreeFLANN<DescriptorType> match_search;
  match_search.setInputCloud(model_descriptors);

  constexpr float MATCH_THRESHOLD = 0.25f;
  for (std::size_t i = 0; i < scene_descriptors->size(); ++i) {
    std::vector<int> neigh_indices(1);
    std::vector<float> neigh_sqr_dists(1);

    if (!std::isfinite(scene_descriptors->at(i).descriptor[0])) {
      continue;
    }

    int found = match_search.nearestKSearch(scene_descriptors->at(i), 1, neigh_indices,
      neigh_sqr_dists);
    if (found == 1 && neigh_sqr_dists[0] < MATCH_THRESHOLD) {
      model_scene_corrs->push_back(pcl::Correspondence(neigh_indices[0], static_cast<int>(i),
        neigh_sqr_dists[0]));
    }
  }
  std::cout << " " << model_scene_corrs->size() << " found" << std::endl;

  std::cout << "\nPerforming correspondence grouping..." << std::endl;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> rototranslations;
  std::vector<pcl::Correspondences> clustered_corrs;

  if (config.use_hough) {
    std::cout << "  Algorithm: Hough3D" << std::endl;
    std::cout << "  Computing reference frames (BOARD)..." << std::flush;

    auto model_rf = pcl::make_shared<pcl::PointCloud<RFType>>();
    auto scene_rf = pcl::make_shared<pcl::PointCloud<RFType>>();

    pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
    rf_est.setFindHoles(true);
    rf_est.setRadiusSearch(config.rf_rad);

    rf_est.setInputCloud(model_keypoints);
    rf_est.setInputNormals(model_normals);
    rf_est.setSearchSurface(model);
    rf_est.compute(*model_rf);

    rf_est.setInputCloud(scene_keypoints);
    rf_est.setInputNormals(scene_normals);
    rf_est.setSearchSurface(scene);
    rf_est.compute(*scene_rf);
    std::cout << " Done" << std::endl;

    std::cout << "  Clustering with Hough3D..." << std::flush;
    pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
    clusterer.setHoughBinSize(config.cg_size);
    clusterer.setHoughThreshold(config.cg_thresh);
    clusterer.setUseInterpolation(true);
    clusterer.setUseDistanceWeight(false);

    clusterer.setInputCloud(model_keypoints);
    clusterer.setInputRf(model_rf);
    clusterer.setSceneCloud(scene_keypoints);
    clusterer.setSceneRf(scene_rf);
    clusterer.setModelSceneCorrespondences(model_scene_corrs);

    clusterer.recognize(rototranslations, clustered_corrs);
    std::cout << " Done" << std::endl;
  } else {
    std::cout << "  Algorithm: Geometric Consistency" << std::endl;
    std::cout << "  Clustering..." << std::flush;

    pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer;
    gc_clusterer.setGCSize(config.cg_size);
    gc_clusterer.setGCThreshold(config.cg_thresh);

    gc_clusterer.setInputCloud(model_keypoints);
    gc_clusterer.setSceneCloud(scene_keypoints);
    gc_clusterer.setModelSceneCorrespondences(model_scene_corrs);

    gc_clusterer.recognize(rototranslations, clustered_corrs);
    std::cout << " Done" << std::endl;
  }

  std::cout << "\n=== Recognition Results ===" << std::endl;
  std::cout << "Model instances found: " << rototranslations.size() << "\n" << std::endl;

  if (rototranslations.empty()) {
    std::cout << "No instances detected.\n" << std::endl;
  }

  for (std::size_t i = 0; i < rototranslations.size(); ++i) {
    std::cout << "Instance " << (i + 1) << ":" << std::endl;
    std::cout << "  Correspondences: " << clustered_corrs[i].size() << std::endl;

    const Eigen::Matrix3f rotation = rototranslations[i].block<3, 3>(0, 0);
    const Eigen::Vector3f translation = rototranslations[i].block<3, 1>(0, 3);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n      | " << std::setw(6) << rotation(0, 0) << " "
              << std::setw(6) << rotation(0, 1) << " "
              << std::setw(6) << rotation(0, 2) << " |" << std::endl;
    std::cout << "  R = | " << std::setw(6) << rotation(1, 0) << " "
              << std::setw(6) << rotation(1, 1) << " "
              << std::setw(6) << rotation(1, 2) << " |" << std::endl;
    std::cout << "      | " << std::setw(6) << rotation(2, 0) << " "
              << std::setw(6) << rotation(2, 1) << " "
              << std::setw(6) << rotation(2, 2) << " |\n" << std::endl;
    std::cout << "  t = < " << translation(0) << ", "
              << translation(1) << ", " << translation(2) << " >\n" << std::endl;
  }

  if (rototranslations.empty()) {
    std::cout << "\n[Warning] No model instances detected in the scene." << std::endl;
    std::cout << "Try adjusting parameters: --cg_thresh, --model_ss, --scene_ss\n" << std::endl;
  }

  std::cout << "\nInitializing visualizer..." << std::endl;
  pcl::visualization::PCLVisualizer::Ptr viewer(
    new pcl::visualization::PCLVisualizer("3D Object Recognition - Correspondence Grouping"));

  auto off_scene_model = pcl::make_shared<pcl::PointCloud<PointType>>();
  auto off_scene_model_keypoints = pcl::make_shared<pcl::PointCloud<PointType>>();

  pcl::transformPointCloud(
    *model, *off_scene_model, Eigen::Vector3f(-1, 0, 0),
    Eigen::Quaternionf(1, 0, 0, 0));
  pcl::transformPointCloud(
    *model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f(-1, 0, 0),
    Eigen::Quaternionf(1, 0, 0, 0));

  VisualizationData vis_data;
  vis_data.scene = scene;
  vis_data.scene_keypoints = scene_keypoints;
  vis_data.off_scene_model = off_scene_model;
  vis_data.off_scene_model_keypoints = off_scene_model_keypoints;
  vis_data.rototranslations = &rototranslations;
  vis_data.clustered_corrs = &clustered_corrs;
  vis_data.model = model;
  vis_data.show_keypoints_flag = &config.show_keypoints;
  vis_data.show_correspondences_flag = &config.show_correspondences;

  CallbackData callback_data;
  callback_data.vis_data = &vis_data;
  callback_data.viewer = viewer.get();

  viewer->registerKeyboardCallback(keyboardEventOccurred, static_cast<void *>(&callback_data));

  updateVisualization(*viewer, vis_data);

  std::cout << "\nVisualization Controls:" << std::endl;
  std::cout << "  K - Toggle keypoints display" << std::endl;
  std::cout << "  C - Toggle correspondences display" << std::endl;
  std::cout << "  Q - Quit\n" << std::endl;

  while (!viewer->wasStopped()) {
    viewer->spinOnce();
  }

  return EXIT_SUCCESS;
}
