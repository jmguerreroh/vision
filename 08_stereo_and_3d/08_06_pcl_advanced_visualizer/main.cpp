/**
 * @file main.cpp
 * @brief Advanced PCL visualizer demonstrations with multiple rendering modes
 * @author José Miguel Guerrero Hernández
 *
 * This example shows:
 * - Multiple visualization modes (simple, RGB, custom color, normals, shapes)
 * - Dual viewport setup for side-by-side comparisons
 * - Interactive mouse and keyboard event handling
 * - Surface normal computation and visualization
 * - Geometric primitive rendering (spheres, planes, cones, lines)
 * - Custom color handlers for point clouds
 * - Dynamic point cloud generation (elliptical helix)
 *
 * Visualization Modes:
 * - Simple (-s):    Basic XYZ point cloud rendering
 * - RGB (-r):       Color gradient visualization (red-green-blue)
 * - Custom (-c):    Single color rendering (green)
 * - Normals (-n):   Surface normals as arrows with color by curvature
 * - Shapes (-a):    Geometric primitives overlay
 * - Viewports (-v): Dual viewport with different normal radii (0.01 vs 0.1)
 * - Interactive (-i): Click to add labels, 'r' to clear all
 *
 * @see https://pointclouds.org/documentation/tutorials/pcl_visualizer.html
 *
 * Usage: ./advance_visualizer [option]
 * Options: -h, -s, -r, -c, -n, -a, -v, -i
 * Example: ./advance_visualizer -n
 */

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>

#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <vtkObject.h>

using namespace std::chrono_literals;

/**
 * @brief Display statistics about the generated point cloud
 * @param cloud_ptr RGB point cloud
 * @param basic_cloud_ptr Basic XYZ point cloud
 */
void displayCloudInfo(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud_ptr,
  const pcl::PointCloud<pcl::PointXYZ>::Ptr & basic_cloud_ptr)
{
  std::cout << "\n=== Point Cloud Information ===" << std::endl;
  std::cout << "  Total points (XYZRGB): " << cloud_ptr->size() << std::endl;
  std::cout << "  Total points (XYZ):    " << basic_cloud_ptr->size() << std::endl;
  std::cout << "  Cloud width:           " << cloud_ptr->width << std::endl;
  std::cout << "  Cloud height:          " << cloud_ptr->height << std::endl;
  std::cout << "  Is organized:          " << (cloud_ptr->isOrganized() ? "Yes" : "No") <<
    std::endl;

  if (!cloud_ptr->empty()) {
    pcl::PointXYZRGB min_pt, max_pt;
    pcl::getMinMax3D(*cloud_ptr, min_pt, max_pt);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n  Bounding Box:" << std::endl;
    std::cout << "    Min: (" << min_pt.x << ", " << min_pt.y << ", " << min_pt.z << ")" <<
      std::endl;
    std::cout << "    Max: (" << max_pt.x << ", " << max_pt.y << ", " << max_pt.z << ")" <<
      std::endl;
  }
  std::cout << "===============================\n" << std::endl;
}

/**
 * @brief Prints usage information and available command line options
 * @param prog_name Name of the program executable
 */
void
printUsage(const char * prog_name)
{
  std::cout << "\n==============================================================\n"
            << "     PCL Advanced Visualizer - Demonstration Tool          \n"
            << "==============================================================\n\n"
            << "Usage: " << prog_name << " [option]\n\n"
            << "Available Options:\n"
            << "--------------------------------------------------------------\n"
            << "  -h    Show this help message\n"
            << "  -s    Simple visualization (basic point cloud)\n"
            << "  -r    RGB color visualization (colored points)\n"
            << "  -c    Custom color visualization (green points)\n"
            << "  -n    Normals visualization (surface normals display)\n"
            << "  -a    Shapes visualization (geometric primitives)\n"
            << "  -v    Viewports (dual view with different normals)\n"
            << "  -i    Interactive mode (mouse/keyboard callbacks)\n\n"
            << "Examples:\n"
            << "  " << prog_name << " -s    # Simple point cloud\n"
            << "  " << prog_name << " -n    # With surface normals\n"
            << "  " << prog_name << " -i    # Interactive mode\n\n"
            << "Viewer Controls:\n"
            << "  Mouse wheel / Right drag : Zoom in/out\n"
            << "  Left drag               : Rotate view\n"
            << "  Middle drag             : Pan view\n"
            << "  r/R                     : Reset camera\n"
            << "  q/Q                     : Quit\n"
            << "  g/G                     : Toggle axes\n"
            << "  +/-                     : Adjust point size\n\n";
}

/**
 * @brief Creates a simple 3D point cloud visualizer
 * @param cloud Input point cloud to visualize
 * @return Pointer to the PCL visualizer
 */
pcl::visualization::PCLVisualizer::Ptr simpleVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
  // Open 3D viewer and add point cloud
  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,
    "sample cloud");
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();
  return viewer;
}

/**
 * @brief Creates a 3D visualizer with RGB color visualization
 * @param cloud Input RGB point cloud to visualize
 * @return Pointer to the PCL visualizer
 */
pcl::visualization::PCLVisualizer::Ptr rgbVis(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  // Open 3D viewer and add point cloud
  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,
    "sample cloud");
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();
  return viewer;
}

/**
 * @brief Creates a 3D visualizer with custom color (green)
 * @param cloud Input point cloud to visualize
 * @return Pointer to the PCL visualizer
 */
pcl::visualization::PCLVisualizer::Ptr customColourVis(
  pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
  // Open 3D viewer and add point cloud
  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 255, 0);
  viewer->addPointCloud<pcl::PointXYZ>(cloud, single_color, "sample cloud");
  viewer->setPointCloudRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,
    "sample cloud");
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();
  return viewer;
}

/**
 * @brief Creates a 3D visualizer displaying point cloud with surface normals
 * @param cloud Input RGB point cloud to visualize
 * @param normals Surface normals to display
 * @return Pointer to the PCL visualizer
 */
pcl::visualization::PCLVisualizer::Ptr normalsVis(
  pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
  pcl::PointCloud<pcl::Normal>::ConstPtr normals)
{
  // Open 3D viewer and add point cloud and normals
  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,
    "sample cloud");
  viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud, normals, 10, 0.05, "normals");
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();
  return viewer;
}

/**
 * @brief Creates a 3D visualizer with geometric shapes (line, sphere, plane, cone)
 * @param cloud Input RGB point cloud to visualize
 * @return Pointer to the PCL visualizer
 */
pcl::visualization::PCLVisualizer::Ptr shapesVis(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  // Open 3D viewer and add point cloud
  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,
    "sample cloud");
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();

  //------------------------------------
  //-----Add shapes at cloud points-----
  //------------------------------------
  viewer->addLine<pcl::PointXYZRGB>(
    (*cloud)[0],
    (*cloud)[cloud->size() - 1], "line");
  viewer->addSphere((*cloud)[0], 0.2, 0.5, 0.5, 0.0, "sphere");

  //---------------------------------------
  //-----Add shapes at other locations-----
  //---------------------------------------
  pcl::ModelCoefficients coeffs;
  coeffs.values.push_back(0.0);
  coeffs.values.push_back(0.0);
  coeffs.values.push_back(1.0);
  coeffs.values.push_back(0.0);
  viewer->addPlane(coeffs, "plane");
  coeffs.values.clear();
  coeffs.values.push_back(0.3);
  coeffs.values.push_back(0.3);
  coeffs.values.push_back(0.0);
  coeffs.values.push_back(0.0);
  coeffs.values.push_back(1.0);
  coeffs.values.push_back(0.0);
  coeffs.values.push_back(5.0);
  viewer->addCone(coeffs, "cone");

  return viewer;
}

/**
 * @brief Creates a 3D visualizer with dual viewports showing different normal calculations
 * @param cloud Input RGB point cloud to visualize
 * @param normals1 Surface normals with small search radius
 * @param normals2 Surface normals with large search radius
 * @return Pointer to the PCL visualizer
 */
pcl::visualization::PCLVisualizer::Ptr viewportsVis(
  pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
  pcl::PointCloud<pcl::Normal>::ConstPtr normals1, pcl::PointCloud<pcl::Normal>::ConstPtr normals2)
{
  // Open 3D viewer and add point cloud and normals
  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->initCameraParameters();

  int v1(0);
  viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
  viewer->setBackgroundColor(0, 0, 0, v1);
  viewer->addText("Radius: 0.01", 10, 10, "v1 text", v1);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud1", v1);

  int v2(0);
  viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
  viewer->setBackgroundColor(0.3, 0.3, 0.3, v2);
  viewer->addText("Radius: 0.1", 10, 10, "v2 text", v2);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color(cloud, 0, 255, 0);
  viewer->addPointCloud<pcl::PointXYZRGB>(cloud, single_color, "sample cloud2", v2);

  viewer->setPointCloudRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,
    "sample cloud1");
  viewer->setPointCloudRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,
    "sample cloud2");
  viewer->addCoordinateSystem(1.0);

  viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(
    cloud, normals1, 10, 0.05, "normals1",
    v1);
  viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(
    cloud, normals2, 10, 0.05, "normals2",
    v2);

  return viewer;
}

// Global mutable state (required for PCL callbacks)
unsigned int text_id = 0;
/**
 * @brief Callback function for keyboard events in the visualizer
 * @param event Keyboard event containing key information
 * @param viewer_void Pointer to the PCL visualizer instance
 */
void keyboardEventOccurred(
  const pcl::visualization::KeyboardEvent & event,
  void * viewer_void)
{
  pcl::visualization::PCLVisualizer * viewer =
    static_cast<pcl::visualization::PCLVisualizer *>(viewer_void);
  if (event.getKeySym() == "r" && event.keyDown()) {
    std::cout << "\n[Keyboard Event] 'r' pressed" << std::endl;
    std::cout << "  - Removing " << text_id << " text labels..." << std::endl;

    for (unsigned int i = 0; i < text_id; ++i) {
      std::ostringstream oss;
      oss << "text#" << std::setfill('0') << std::setw(3) << i;
      viewer->removeShape(oss.str());
    }
    text_id = 0;
    std::cout << "  All text labels removed\n" << std::endl;
  }
}

/**
 * @brief Callback function for mouse events in the visualizer
 * @param event Mouse event containing button and position information
 * @param viewer_void Pointer to the PCL visualizer instance
 */
void mouseEventOccurred(
  const pcl::visualization::MouseEvent & event,
  void * viewer_void)
{
  pcl::visualization::PCLVisualizer * viewer =
    static_cast<pcl::visualization::PCLVisualizer *>(viewer_void);
  if (event.getButton() == pcl::visualization::MouseEvent::LeftButton &&
    event.getType() == pcl::visualization::MouseEvent::MouseButtonRelease)
  {
    std::cout << "[Mouse Event] Left click at (" << event.getX() << ", "
              << event.getY() << ") - Label #" << text_id << " added" << std::endl;

    std::ostringstream oss;
    oss << "text#" << std::setfill('0') << std::setw(3) << text_id++;
    viewer->addText("clicked here", event.getX(), event.getY(), oss.str());
  }
}

/**
 * @brief Creates a 3D visualizer with custom keyboard and mouse interaction
 * @return Pointer to the PCL visualizer with registered callbacks
 */
pcl::visualization::PCLVisualizer::Ptr interactionCustomizationVis()
{
  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(
    "Interactive Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  viewer->addCoordinateSystem(1.0);

  // Add instruction text to the viewer
  viewer->addText("Interactive Mode - Click to add labels, press 'r' to clear", 10, 50, 16, 1.0,
    1.0, 1.0, "instructions");
  viewer->addText("Left Click: Add label | R: Remove all labels | Q: Quit", 10, 30, 12, 0.7, 0.7,
    0.7, "controls");

  viewer->registerKeyboardCallback(keyboardEventOccurred, (void *)viewer.get());
  viewer->registerMouseCallback(mouseEventOccurred, (void *)viewer.get());

  std::cout << "\n=== Interactive Mode Activated ===" << std::endl;
  std::cout << "  • Click anywhere to add a text label\n";
  std::cout << "  • Press 'r' to remove all labels\n";
  std::cout << "  • Press 'q' to quit\n" << std::endl;

  return viewer;
}

// Main
int main(int argc, char ** argv)
{
  // Disable VTK warning messages
  vtkObject::GlobalWarningDisplayOff();

  std::cout << "\n==============================================================\n"
            << "        PCL Advanced Visualizer - Demo Application          \n"
            << "==============================================================\n" << std::endl;

  // Parse Command Line Arguments
  if (pcl::console::find_argument(argc, argv, "-h") >= 0) {
    printUsage(argv[0]);
    return EXIT_SUCCESS;
  }

  bool simple(false), rgb(false), custom_c(false), normals(false),
  shapes(false), viewports(false), interaction_customization(false);

  if (pcl::console::find_argument(argc, argv, "-s") >= 0) {
    simple = true;
    std::cout << "Mode: Simple Visualization" << std::endl;
    std::cout << "    Basic point cloud rendering\n" << std::endl;
  } else if (pcl::console::find_argument(argc, argv, "-c") >= 0) {
    custom_c = true;
    std::cout << "Mode: Custom Color Visualization" << std::endl;
    std::cout << "    Points rendered in custom green color\n" << std::endl;
  } else if (pcl::console::find_argument(argc, argv, "-r") >= 0) {
    rgb = true;
    std::cout << "Mode: RGB Color Visualization" << std::endl;
    std::cout << "    Points colored with gradient (red-green-blue)\n" << std::endl;
  } else if (pcl::console::find_argument(argc, argv, "-n") >= 0) {
    normals = true;
    std::cout << "Mode: Normals Visualization" << std::endl;
    std::cout << "    Surface normals displayed as arrows\n" << std::endl;
  } else if (pcl::console::find_argument(argc, argv, "-a") >= 0) {
    shapes = true;
    std::cout << "Mode: Shapes Visualization" << std::endl;
    std::cout << "    Geometric primitives (line, sphere, plane, cone)\n" << std::endl;
  } else if (pcl::console::find_argument(argc, argv, "-v") >= 0) {
    viewports = true;
    std::cout << "Mode: Dual Viewports" << std::endl;
    std::cout << "    Side-by-side comparison with different normal radii\n" << std::endl;
  } else if (pcl::console::find_argument(argc, argv, "-i") >= 0) {
    interaction_customization = true;
    std::cout << "Mode: Interactive Customization" << std::endl;
    std::cout << "    Mouse and keyboard event handling\n" << std::endl;
  } else {
    printUsage(argv[0]);
    return EXIT_SUCCESS;
  }

  // Create example point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

  std::cout << "Generating synthetic point cloud..." << std::endl;
  std::cout << "    Creating 3D elliptical helix" << std::endl;
  std::cout << "    Applying color gradient (R-G-B)" << std::endl;
  // We're going to make an ellipse extruded along the z-axis. The colour for
  // the XYZRGB cloud will gradually go from red to green to blue.
  std::uint8_t r(255), g(15), b(15);
  for (float z(-1.0); z <= 1.0; z += 0.05) {
    for (float angle(0.0); angle <= 360.0; angle += 5.0) {
      pcl::PointXYZ basic_point;
      basic_point.x = 0.5 * std::cos(pcl::deg2rad(angle));
      basic_point.y = sinf(pcl::deg2rad(angle));
      basic_point.z = z;
      basic_cloud_ptr->points.push_back(basic_point);

      pcl::PointXYZRGB point;
      point.x = basic_point.x;
      point.y = basic_point.y;
      point.z = basic_point.z;
      std::uint32_t rgb = (static_cast<std::uint32_t>(r) << 16 |
        static_cast<std::uint32_t>(g) << 8 | static_cast<std::uint32_t>(b));
      std::memcpy(&point.rgb, &rgb, sizeof(float));
      point_cloud_ptr->points.push_back(point);
    }
    if (z < 0.0) {
      r -= 12;
      g += 12;
    } else {
      g -= 12;
      b += 12;
    }
  }
  basic_cloud_ptr->width = basic_cloud_ptr->size();
  basic_cloud_ptr->height = 1;
  point_cloud_ptr->width = point_cloud_ptr->size();
  point_cloud_ptr->height = 1;

  std::cout << "  Point cloud generated successfully!" << std::endl;

  // Display cloud information
  displayCloudInfo(point_cloud_ptr, basic_cloud_ptr);

  // Calculate surface normals with a search radius of 0.05
  std::cout << "Computing surface normals..." << std::endl;
  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
  ne.setInputCloud(point_cloud_ptr);
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
  ne.setSearchMethod(tree);

  std::cout << "    Computing normals with radius 0.05..." << std::endl;
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals1(new pcl::PointCloud<pcl::Normal>);
  ne.setRadiusSearch(0.05);
  ne.compute(*cloud_normals1);

  std::cout << "    Computing normals with radius 0.10..." << std::endl;
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2(new pcl::PointCloud<pcl::Normal>);
  ne.setRadiusSearch(0.1);
  ne.compute(*cloud_normals2);

  std::cout << "  Normal computation complete!\n" << std::endl;

  // Create the appropriate visualizer
  std::cout << "Initializing visualizer..." << std::endl;
  pcl::visualization::PCLVisualizer::Ptr viewer;

  if (simple) {
    viewer = simpleVis(basic_cloud_ptr);
  } else if (rgb) {
    viewer = rgbVis(point_cloud_ptr);
  } else if (custom_c) {
    viewer = customColourVis(basic_cloud_ptr);
  } else if (normals) {
    viewer = normalsVis(point_cloud_ptr, cloud_normals2);
  } else if (shapes) {
    viewer = shapesVis(point_cloud_ptr);
  } else if (viewports) {
    viewer = viewportsVis(point_cloud_ptr, cloud_normals1, cloud_normals2);
  } else if (interaction_customization) {
    viewer = interactionCustomizationVis();
  }

  std::cout << "Visualizer ready!" << std::endl;
  std::cout << "\n- Viewer window opened. Close window to exit.\n" << std::endl;

  //--------------------
  // Main loop
  //--------------------
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
    std::this_thread::sleep_for(100ms);
  }

  std::cout << "\n=== Visualizer Closed ===" << std::endl;
  std::cout << "Thank you for using PCL Advanced Visualizer!\n" << std::endl;

  return EXIT_SUCCESS;
}
