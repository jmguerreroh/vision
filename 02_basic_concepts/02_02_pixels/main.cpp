/**
 * @file main.cpp
 * @brief Pixel access and manipulation in OpenCV
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - How to access individual pixel values in an image
 * - Two methods for reading pixel data (Vec3b and split channels)
 * - How to separate and visualize individual color channels (BGR)
 * - How to merge channels back into a single image
 *
 * @note OpenCV uses BGR color order, not RGB!
 *       Channel 0 = Blue, Channel 1 = Green, Channel 2 = Red
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

int main(int argc, char ** argv)
{
  // Load and display the image
  const std::string imagePath = "../../data/lena.jpg";

  // Load image in BGR color format (default)
  cv::Mat image;
  if (argc > 1) {
    image = cv::imread(argv[1], cv::IMREAD_COLOR);
  } else {
    image = cv::imread(imagePath, cv::IMREAD_COLOR);
  }

  // Verify that the image was loaded successfully
  // An empty image indicates an error (file not found, invalid format, etc.)
  if (image.empty()) {
    std::cerr << "Error: Could not load image from: "
              << (argc > 1 ? argv[1] : imagePath) << std::endl;
    std::cerr << "Please verify the file exists and the path is correct." << std::endl;
    return -1;
  }

  std::cout << "Image loaded: " << image.cols << "x" << image.rows << " pixels" << std::endl;
  std::cout << "Channels: " << image.channels() << " (BGR format)" << std::endl;

  cv::namedWindow("Pixel Demo", cv::WINDOW_AUTOSIZE);
  cv::imshow("Pixel Demo", image);

  // Method 1 - Direct pixel access using Vec3b
  //
  // Vec3b is a vector of 3 unsigned chars (bytes), representing BGR values.
  // Access: image.at<Vec3b>(row, col)[channel]
  //   - [0] = Blue
  //   - [1] = Green
  //   - [2] = Red
  //
  // Note: This method accesses pixels one by one (slower for large images)
  std::cout << "\n--- Method 1: Direct access with Vec3b ---" << std::endl;
  std::cout << "First 5 pixels (B G R):" << std::endl;

  int pixelCount = 0;
  for (int row = 0; row < image.rows && pixelCount < 5; row++) {
    for (int col = 0; col < image.cols && pixelCount < 5; col++) {
      // Access BGR values using Vec3b
      cv::Vec3b pixel = image.at<cv::Vec3b>(row, col);
      std::cout << "  Pixel[" << row << "," << col << "]: "
                << (int)pixel[0] << " "         // Blue
                << (int)pixel[1] << " "         // Green
                << (int)pixel[2] << std::endl;  // Red
      pixelCount++;
    }
  }

  // Method 2 - Channel separation using split()
  //
  // split() separates a multi-channel image into individual single-channel images.
  // This is useful when you need to process each channel independently.
  std::cout << "\n--- Method 2: Split channels ---" << std::endl;

  std::vector<cv::Mat> channels;  // Will contain 3 grayscale images (B, G, R)
  cv::split(image, channels);

  std::cout << "Image split into " << channels.size() << " channels" << std::endl;

  // Display first 5 pixels from separated channels
  std::cout << "First 5 pixels (B G R) from split channels:" << std::endl;
  pixelCount = 0;
  for (int row = 0; row < image.rows && pixelCount < 5; row++) {
    for (int col = 0; col < image.cols && pixelCount < 5; col++) {
      // Access each channel as a separate grayscale image
      std::cout << "  Pixel[" << row << "," << col << "]: "
                << (int)channels[0].at<uchar>(row, col) << " "        // Blue channel
                << (int)channels[1].at<uchar>(row, col) << " "        // Green channel
                << (int)channels[2].at<uchar>(row, col) << std::endl; // Red channel
      pixelCount++;
    }
  }

  // Visualize individual channels
  //
  // Each channel is displayed as a grayscale image.
  // Brighter areas indicate higher intensity of that color.
  cv::imshow("Blue Channel", channels[0]);
  cv::imshow("Green Channel", channels[1]);
  cv::imshow("Red Channel", channels[2]);

  // Merge channels back into a color image
  //
  // merge() combines single-channel images into a multi-channel image.
  // The order of channels matters: {Blue, Green, Red}
  cv::Mat reconstructed;
  cv::merge(channels, reconstructed);
  cv::imshow("Reconstructed Image", reconstructed);

  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return 0;
}
