/**
 * @file main.cpp
 * @brief Basic example of image reading and display with OpenCV
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - How to read an image from disk using cv::imread()
 * - How to display an image in a window using cv::imshow()
 * - How to wait for user interaction with cv::waitKey()
 *
 * @note This file uses the explicit cv:: prefix for all OpenCV functions.
 *       An alternative is to use 'using namespace cv;' at the top.
 *
 *       Comparison of both approaches:
 *       +--------------------------------+---------------------------+
 *       | Explicit cv:: prefix           | using namespace cv        |
 *       +--------------------------------+---------------------------+
 *       | Avoids name conflicts          | Cleaner, shorter code     |
 *       | Clear function origin          | Less repetitive typing    |
 *       | Recommended for large projects | Good for small examples   |
 *       +--------------------------------+---------------------------+
 */

#include <opencv2/highgui.hpp>
#include <cstdlib>
#include <iostream>

int main(int argc, char ** argv)
{
  // Path to the image file (relative to the execution directory)
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/starry_night.jpg | Input file}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  const std::string image_path = parser.get<std::string>("@input");

  // cv::Mat is OpenCV's main structure for storing images
  // Mat = Matrix, represents an image as a matrix of pixels
  cv::Mat image;

  // cv::imread() loads an image from a file
  // Parameters:
  //   - File path (string)
  //   - Read mode:
  //     * cv::IMREAD_COLOR (1): Load color image in BGR format (default)
  //     * cv::IMREAD_GRAYSCALE (0): Load image in grayscale
  //     * cv::IMREAD_UNCHANGED (-1): Load image with alpha channel if present
  image = cv::imread(cv::samples::findFile(image_path, false), cv::IMREAD_COLOR);

  // Verify that the image was loaded successfully
  // An empty image indicates an error (file not found, invalid format, etc.)
  if (image.empty()) {
    std::cerr << "Error: Could not load image from: "
              << image_path << std::endl;
    std::cerr << "Please verify the file exists and the path is correct." << std::endl;
    return EXIT_FAILURE;
  }

  // Display basic information about the loaded image
  std::cout << "Image loaded successfully:" << std::endl;
  std::cout << "  - Dimensions: " << image.cols << " x " << image.rows << " pixels" << std::endl;
  std::cout << "  - Channels: " << image.channels() << " (BGR)" << std::endl;
  // cv::typeToString() decodes the type constant: CV_8UC3 means
  // 8-bit Unsigned integers, 3 Channels (one per B, G, R component)
  std::cout << "  - Data type: " << cv::typeToString(image.type()) << std::endl;

  // cv::imshow() displays an image in a window
  // Parameters:
  //   - Window name (string) - used as unique identifier
  //   - Image to display (cv::Mat)
  cv::imshow("Original Image - BGR", image);

  // cv::waitKey() waits for the user to press a key
  // Parameters:
  //   - Wait time in milliseconds (0 = wait indefinitely)
  // Returns: ASCII code of the pressed key, or -1 if timeout expires
  std::cout << "\nPress any key to close the window..." << std::endl;
  cv::waitKey(0);

  // Windows are automatically destroyed when the program ends
  // You can also use cv::destroyAllWindows() to close them explicitly
  return EXIT_SUCCESS;
}
