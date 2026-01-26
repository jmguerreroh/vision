/**
 * @file main.cpp
 * @brief Color spaces conversion in OpenCV
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - How to convert between different color spaces using cvtColor()
 * - BGR, HSV, LAB, YCrCb, and Grayscale color spaces
 *
 * @note Color Spaces in OpenCV:
 *       - BGR: Blue, Green, Red (default in OpenCV, not RGB!)
 *       - HSV: Hue, Saturation, Value (useful for color-based segmentation)
 *       - LAB: Lightness, A (green-red), B (blue-yellow) - perceptually uniform
 *       - YCrCb: Luminance (Y), Chrominance Red (Cr), Chrominance Blue (Cb)
 *       - Grayscale: Single channel intensity
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

int main(int argc, char ** argv)
{
  // Load the image
  const std::string imagePath = "../../data/RGB.jpg";

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
  std::cout << "Original color space: BGR (3 channels)" << std::endl;

  // Convert to different color spaces
  //
  // cvtColor() converts an image from one color space to another.
  // Syntax: cvtColor(input, output, conversion_code)

  // BGR to Grayscale
  // Grayscale uses a weighted sum: Y = 0.299*R + 0.587*G + 0.114*B
  cv::Mat grayscale;
  cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);

  // BGR to HSV
  // HSV is useful for color detection/segmentation:
  //   H (Hue): 0-179 (color type: 0=red, 60=green, 120=blue)
  //   S (Saturation): 0-255 (color purity)
  //   V (Value): 0-255 (brightness)
  cv::Mat hsv;
  cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

  // BGR to LAB
  // LAB is perceptually uniform (distances correlate with human perception):
  //   L (Lightness): 0-255
  //   A: Green (-) to Red (+)
  //   B: Blue (-) to Yellow (+)
  cv::Mat lab;
  cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);

  // BGR to YCrCb
  // Used in video compression (JPEG, MPEG):
  //   Y (Luma): Brightness information
  //   Cr: Red chrominance component
  //   Cb: Blue chrominance component
  cv::Mat ycrcb;
  cv::cvtColor(image, ycrcb, cv::COLOR_BGR2YCrCb);

  // Display all color spaces
  cv::imshow("Original (BGR)", image);
  cv::imshow("Grayscale", grayscale);
  cv::imshow("HSV", hsv);
  cv::imshow("LAB", lab);
  cv::imshow("YCrCb", ycrcb);

  // Split and display HSV channels
  //
  // Visualizing individual channels helps understand each color space.
  std::vector<cv::Mat> hsvChannels;
  cv::split(hsv, hsvChannels);

  cv::imshow("HSV - Hue", hsvChannels[0]);
  cv::imshow("HSV - Saturation", hsvChannels[1]);
  cv::imshow("HSV - Value", hsvChannels[2]);

  // Demonstrate reverse conversion
  //
  // You can convert back to BGR for display or further processing.
  cv::Mat bgrFromHsv;
  cv::cvtColor(hsv, bgrFromHsv, cv::COLOR_HSV2BGR);
  cv::imshow("BGR (converted back from HSV)", bgrFromHsv);

  std::cout << "\nDisplaying:" << std::endl;
  std::cout << "  - Original BGR image" << std::endl;
  std::cout << "  - Grayscale conversion" << std::endl;
  std::cout << "  - HSV color space (+ individual H, S, V channels)" << std::endl;
  std::cout << "  - LAB color space" << std::endl;
  std::cout << "  - YCrCb color space" << std::endl;
  std::cout << "  - BGR reconstructed from HSV" << std::endl;
  std::cout << "\nPress any key to exit..." << std::endl;

  cv::waitKey(0);
  return 0;
}
