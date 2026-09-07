/**
 * @file main.cpp
 * @brief Pixel access and manipulation in OpenCV
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - How to access individual pixel values in an image
 * - Three access methods: at<Vec3b>, split channels, and row pointers (ptr)
 * - A timing comparison between at<> and ptr<> for full-image traversal
 * - How to separate, visualize and merge color channels (BGR)
 *
 * @note OpenCV uses BGR color order, not RGB!
 *       Channel 0 = Blue, Channel 1 = Green, Channel 2 = Red
 */

#include <string>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

namespace
{
// The images this example works on are about 1400 px on the long side, and
// several windows at that size do not fit on a normal screen. The processing
// always runs at full resolution: only the copy sent to the screen is reduced,
// with INTER_AREA, which is the interpolation meant for shrinking
constexpr int MAX_DISPLAY_SIDE = 800;

void showFit(const std::string & window, const cv::Mat & image)
{
  const int side = std::max(image.cols, image.rows);
  if (side <= MAX_DISPLAY_SIDE || image.empty()) {
    cv::imshow(window, image);
    return;
  }
  const double factor = static_cast<double>(MAX_DISPLAY_SIDE) / side;
  cv::Mat reduced;
  cv::resize(image, reduced, cv::Size(), factor, factor, cv::INTER_AREA);
  cv::imshow(window, reduced);
}
}  // namespace

int main(int argc, char ** argv)
{
  // Load and display the image
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/starry_night.png | Input file}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  const std::string image_path = parser.get<std::string>("@input");

  // Without this, a malformed value (--frames=xyz) is reported by the parser
  // but the example carries on with the default, which is the hardest kind
  // of failure to diagnose. Note it validates values, not option names:
  // cv::CommandLineParser ignores an unknown option without complaining
  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }

  // Load image in BGR color format (default)
  cv::Mat image;
  image = cv::imread(cv::samples::findFile(image_path, false), cv::IMREAD_COLOR);

  // Verify that the image was loaded successfully
  // An empty image indicates an error (file not found, invalid format, etc.)
  if (image.empty()) {
    std::cerr << "Error: Could not load image from: "
              << image_path << std::endl;
    std::cerr << "Please verify the file exists and the path is correct." << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Image loaded: " << image.cols << "x" << image.rows << " pixels" << std::endl;
  std::cout << "Channels: " << image.channels() << " (BGR format)" << std::endl;

  cv::namedWindow("Pixel Demo", cv::WINDOW_AUTOSIZE);
  showFit("Pixel Demo", image);

  // ========================================
  // Method 1 - Direct pixel access using Vec3b
  // ========================================
  //
  // Vec3b is a vector of 3 unsigned chars (bytes), representing BGR values.
  // Access: image.at<Vec3b>(y, x)[channel], where y is the row and x the column.
  // Mind the order: a pixel position is written (x, y), but cv::Mat stores the
  // image row by row, so at() takes the row first.
  //   - [0] = Blue
  //   - [1] = Green
  //   - [2] = Red
  //
  // Note: This method accesses pixels one by one (slower for large images)
  std::cout << "\n--- Method 1: Direct access with Vec3b ---" << std::endl;
  std::cout << "First 5 pixels (B G R):" << std::endl;

  int pixel_count = 0;
  for (int y = 0; y < image.rows && pixel_count < 5; y++) {
    for (int x = 0; x < image.cols && pixel_count < 5; x++) {
      // Access BGR values using Vec3b
      cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
      std::cout << "  Pixel[" << x << "," << y << "]: "
                << static_cast<int>(pixel[0]) << " "         // Blue
                << static_cast<int>(pixel[1]) << " "         // Green
                << static_cast<int>(pixel[2]) << std::endl;  // Red
      pixel_count++;
    }
  }

  // ========================================
  // Method 2 - Channel separation using split()
  // ========================================
  //
  // split() separates a multi-channel image into individual single-channel images.
  // This is useful when you need to process each channel independently.
  std::cout << "\n--- Method 2: Split channels ---" << std::endl;

  std::vector<cv::Mat> channels;  // Will contain 3 grayscale images (B, G, R)
  cv::split(image, channels);

  std::cout << "Image split into " << channels.size() << " channels" << std::endl;

  // Display first 5 pixels from separated channels
  std::cout << "First 5 pixels (B G R) from split channels:" << std::endl;
  pixel_count = 0;
  for (int y = 0; y < image.rows && pixel_count < 5; y++) {
    for (int x = 0; x < image.cols && pixel_count < 5; x++) {
      // Access each channel as a separate grayscale image
      std::cout << "  Pixel[" << x << "," << y << "]: "
                << static_cast<int>(channels[0].at<uchar>(y, x)) << " "        // Blue channel
                << static_cast<int>(channels[1].at<uchar>(y, x)) << " "        // Green channel
                << static_cast<int>(channels[2].at<uchar>(y, x)) << std::endl; // Red channel
      pixel_count++;
    }
  }

  // Visualize individual channels
  //
  // Each channel is displayed as a grayscale image.
  // Brighter areas indicate higher intensity of that color.
  showFit("Blue Channel", channels[0]);
  showFit("Green Channel", channels[1]);
  showFit("Red Channel", channels[2]);

  // Merge channels back into a color image
  //
  // merge() combines single-channel images into a multi-channel image.
  // The order of channels matters: {Blue, Green, Red}
  cv::Mat reconstructed;
  cv::merge(channels, reconstructed);
  showFit("Reconstructed Image", reconstructed);

  // ========================================
  // Method 3 - Row pointers with ptr<>() (the efficient way)
  // ========================================
  //
  // at<>() checks the type and computes the element address on EVERY call.
  // For full-image traversals the idiomatic fast pattern is to fetch a raw
  // pointer to each ROW once and then walk it -- image rows are contiguous
  // in memory, so this compiles down to a simple pointer sweep.
  //
  // We compute the average brightness of the whole image with both methods
  // and time them, so the difference is measured instead of just claimed.
  std::cout << "\n--- Method 3: at<> vs ptr<> timing (full image sum) ---" << std::endl;

  using clock = std::chrono::high_resolution_clock;
  std::uint64_t sum_at = 0;

  auto t0 = clock::now();
  for (int y = 0; y < image.rows; y++) {
    for (int x = 0; x < image.cols; x++) {
      const cv::Vec3b & p = image.at<cv::Vec3b>(y, x);
      sum_at += p[0] + p[1] + p[2];
    }
  }
  auto t1 = clock::now();

  std::uint64_t sum_ptr = 0;
  auto t2 = clock::now();
  for (int y = 0; y < image.rows; y++) {
    // One address computation per ROW instead of one per PIXEL
    const cv::Vec3b * row_ptr = image.ptr<cv::Vec3b>(y);
    for (int x = 0; x < image.cols; x++) {
      sum_ptr += row_ptr[x][0] + row_ptr[x][1] + row_ptr[x][2];
    }
  }
  auto t3 = clock::now();

  const double ms_at = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double ms_ptr = std::chrono::duration<double, std::milli>(t3 - t2).count();

  std::cout << "  at<Vec3b>: " << ms_at << " ms" << std::endl;
  std::cout << "  ptr<Vec3b>: " << ms_ptr << " ms" << std::endl;
  std::cout << "  Same result? " << (sum_at == sum_ptr ? "yes" : "NO (bug!)") << std::endl;
  std::cout << "  Rule of thumb: at<> for isolated pixels (readability)," << std::endl;
  std::cout << "                 ptr<> for whole-image loops (performance)." << std::endl;

  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
