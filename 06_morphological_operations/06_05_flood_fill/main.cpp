/**
 * @file main.cpp
 * @brief Flood Fill Demo - Interactive demonstration of cv::floodFill()
 * @author José Miguel Guerrero Hernández
 *
 * @note Demonstrates cv::floodFill() for filling connected regions based on
 *          color similarity. The function starts from a seed point and fills
 *          all connected pixels that match the color criteria.
 *
 *          Fill modes:
 *            - Simple (s): Only fills pixels exactly matching seed color
 *            - Fixed range (f): Fills pixels within absolute range [seed±tol]
 *            - Floating range (g): Fills pixels within relative range of neighbors
 *
 *          Connectivity:
 *            - 4-connectivity: Only horizontal/vertical neighbors (cross pattern)
 *            - 8-connectivity: Includes diagonal neighbors (square pattern)
 *
 *          Controls:
 *            Click: Fill region at mouse position with random color
 *            c: Toggle color/grayscale mode
 *            m: Toggle mask mode (visualize fill boundaries)
 *            r: Restore original image
 *            s/f/g: Change fill mode
 *            4/8: Change connectivity
 *            ESC: Exit
 */

#include <cstdlib>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

/**
 * @brief Structure to hold flood fill application state
 */
struct FloodFillApp
{
  cv::Mat original;     // Original image (preserved for reset)
  cv::Mat image;        // Working color image
  cv::Mat gray;         // Working grayscale image
  cv::Mat mask;         // Fill mask (2 pixels larger than image)

  int fill_mode = 1;     // 0=simple, 1=fixed range, 2=floating range
  int connectivity = 4;  // 4 or 8 connectivity
  bool is_color = true;  // Color or grayscale mode
  bool use_mask = false; // Show mask window
  int mask_value = 255;  // Value written to mask for filled pixels

  const char * window_name = "Flood Fill Demo";
  const char * trackbar_lo = "Lower tolerance";
  const char * trackbar_up = "Upper tolerance";

  /**
   * @brief Reset working images to original state
   */
  void reset()
  {
    original.copyTo(image);
    cv::cvtColor(original, gray, cv::COLOR_BGR2GRAY);
    mask = cv::Scalar::all(0);
  }

  /**
   * @brief Get current working image based on color mode
   */
  cv::Mat & currentImage()
  {
    return is_color ? image : gray;
  }
};

// Global app state (required for OpenCV callbacks)
FloodFillApp app;

/**
 * @brief Mouse callback - applies flood fill at clicked position
 * @param event Mouse event type (only responds to left button down)
 * @param x X coordinate of click
 * @param y Y coordinate of click
 */
static void onMouse(int event, int x, int y, int /*flags*/, void * /*userdata*/)
{
  if (event != cv::EVENT_LBUTTONDOWN) {
    return;
  }

  // Get tolerance from trackbars
  int lo_diff = cv::getTrackbarPos(app.trackbar_lo, app.window_name);
  int up_diff = cv::getTrackbarPos(app.trackbar_up, app.window_name);

  // Simple mode ignores tolerance (exact color match only)
  int lo = (app.fill_mode == 0) ? 0 : lo_diff;
  int up = (app.fill_mode == 0) ? 0 : up_diff;

  // Build flags: connectivity | (mask_value << 8) | fill_mode_flag
  // FLOODFILL_FIXED_RANGE: Compare to seed pixel (absolute)
  // Without flag: Compare to neighbor pixels (relative/floating)
  int flags = app.connectivity + (app.mask_value << 8) +
    (app.fill_mode == 1 ? cv::FLOODFILL_FIXED_RANGE : 0);

  // Generate random fill color
  cv::RNG & rng = cv::theRNG();
  int b = rng.uniform(0, 256);
  int g = rng.uniform(0, 256);
  int r = rng.uniform(0, 256);

  // For grayscale, convert RGB to luminance
  cv::Scalar newVal = app.is_color ?
    cv::Scalar(b, g, r) :
    cv::Scalar(0.299 * r + 0.587 * g + 0.114 * b);

  cv::Rect filled_rect;
  cv::Scalar lo_diff_scalar(lo, lo, lo);
  cv::Scalar up_diff_scalar(up, up, up);
  cv::Point seed(x, y);
  int area;

  // Apply flood fill
  // With mask: mask must be 2 pixels larger, filled regions are marked
  // Without mask: simpler call, no boundary tracking
  if (app.use_mask) {
    cv::threshold(app.mask, app.mask, 1, 128, cv::THRESH_BINARY);
    area = cv::floodFill(app.currentImage(), app.mask, seed, newVal,
                         &filled_rect, lo_diff_scalar, up_diff_scalar, flags);
    cv::imshow("mask", app.mask);
  } else {
    area = cv::floodFill(app.currentImage(), seed, newVal,
                         &filled_rect, lo_diff_scalar, up_diff_scalar, flags);
  }

  cv::imshow(app.window_name, app.currentImage());
  std::cout << "Filled " << area << " pixels at (" << x << ", " << y << ")" << std::endl;
}

/**
 * @brief Display help message with keyboard controls
 */
void showHelp()
{
  std::cout << "\n=== Flood Fill Demo ===\n"
            << "Click on image to fill connected region with random color.\n\n"
            << "Keyboard controls:\n"
            << "  ESC  - Exit program\n"
            << "  c    - Toggle color/grayscale mode\n"
            << "  m    - Toggle mask visualization\n"
            << "  r    - Restore original image\n"
            << "  s    - Simple mode (exact color match)\n"
            << "  f    - Fixed range mode (compare to seed)\n"
            << "  g    - Gradient mode (compare to neighbors)\n"
            << "  4    - Use 4-connectivity\n"
            << "  8    - Use 8-connectivity\n" << std::endl;
}

int main(int argc, char ** argv)
{
  // Parse command line
  cv::CommandLineParser parser(argc, argv,
    "{help h||Show help message}"
    "{@image|fruits.jpg|Input image}");

  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }

  // Load image
  std::string filename = parser.get<std::string>("@image");
  app.original = cv::imread(cv::samples::findFile(filename), cv::IMREAD_COLOR);

  if (app.original.empty()) {
    std::cerr << "Error: Cannot load image: " << filename << std::endl;
    return EXIT_FAILURE;
  }

  showHelp();

  // Initialize working images
  app.reset();

  // Mask must be 2 pixels larger than image (floodFill requirement)
  app.mask.create(app.original.rows + 2, app.original.cols + 2, CV_8UC1);
  app.mask = cv::Scalar::all(0);

  // Create window and UI
  cv::namedWindow(app.window_name, cv::WINDOW_AUTOSIZE);
  cv::createTrackbar(app.trackbar_lo, app.window_name, nullptr, 255, nullptr);
  cv::createTrackbar(app.trackbar_up, app.window_name, nullptr, 255, nullptr);
  cv::setTrackbarPos(app.trackbar_lo, app.window_name, 20);
  cv::setTrackbarPos(app.trackbar_up, app.window_name, 20);
  cv::setMouseCallback(app.window_name, onMouse, nullptr);

  // Main loop
  while (true) {
    cv::imshow(app.window_name, app.currentImage());
    char key = static_cast<char>(cv::waitKey(0));

    if (key == 27) {  // ESC
      break;
    }

    switch (key) {
      case 'c':  // Toggle color mode
        app.is_color = !app.is_color;
        std::cout << "Mode: " << (app.is_color ? "Color" : "Grayscale") << std::endl;
        app.reset();
        break;

      case 'm':  // Toggle mask
        app.use_mask = !app.use_mask;
        if (app.use_mask) {
          cv::namedWindow("mask", cv::WINDOW_AUTOSIZE);
          app.mask = cv::Scalar::all(0);
          cv::imshow("mask", app.mask);
        } else {
          cv::destroyWindow("mask");
        }
        std::cout << "Mask: " << (app.use_mask ? "ON" : "OFF") << std::endl;
        break;

      case 'r':  // Reset
        app.reset();
        std::cout << "Image restored" << std::endl;
        break;

      case 's':  // Simple mode
        app.fill_mode = 0;
        std::cout << "Fill mode: Simple (exact match)" << std::endl;
        break;

      case 'f':  // Fixed range
        app.fill_mode = 1;
        std::cout << "Fill mode: Fixed range (compare to seed)" << std::endl;
        break;

      case 'g':  // Gradient/floating range
        app.fill_mode = 2;
        std::cout << "Fill mode: Gradient (compare to neighbors)" << std::endl;
        break;

      case '4':  // 4-connectivity
        app.connectivity = 4;
        std::cout << "Connectivity: 4 (cross pattern)" << std::endl;
        break;

      case '8':  // 8-connectivity
        app.connectivity = 8;
        std::cout << "Connectivity: 8 (square pattern)" << std::endl;
        break;
    }
  }

  return EXIT_SUCCESS;
}
