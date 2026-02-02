/**
 * @file main.cpp
 * @brief Logical (bitwise) operations on images
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates bitwise logical operations on images:
 * - AND: Intersection of two images (pixels white only where both are white)
 * - OR: Union of two images (pixels white where either is white)
 * - XOR: Exclusive OR (pixels white where images differ)
 * - NOT: Inversion of pixel values
 *
 * Applications:
 * - Image masking and region extraction
 * - Motion detection (comparing frames)
 * - Image compositing
 * - Binary image processing
 *
 * @note Uses two generated binary images with overlapping circles
 */

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

/**
 * @brief Displays usage information
 * @param argv Command line arguments
 */
void printHelp(char ** argv)
{
  std::cout << "\n"
            << "Logical (Bitwise) Operations Demo\n"
            << "=================================\n"
            << "This program demonstrates bitwise operations on images.\n"
            << "Uses two overlapping circles to show AND, OR, XOR, and NOT operations.\n\n"
            << "Usage: " << argv[0] << "\n\n";
}

/**
 * @brief Create a binary image with a circle
 * @param size Image dimensions
 * @param center Circle center point
 * @param radius Circle radius
 * @return Binary image with white circle on black background
 */
cv::Mat createCircleImage(cv::Size size, cv::Point center, int radius)
{
  cv::Mat image = cv::Mat::zeros(size, CV_8UC1);
  cv::circle(image, center, radius, cv::Scalar(255), cv::FILLED);
  return image;
}

int main(int argc, char ** argv)
{
  (void)argc;    // Unused

  printHelp(argv);

  std::cout << "=== Logical (Bitwise) Operations ===" << std::endl;

  // Create two overlapping circles
  const cv::Size imageSize(400, 400);
  const int radius = 100;

  // Circle 1: centered
  cv::Mat circle1 = createCircleImage(imageSize, cv::Point(180, 200), radius);

  // Circle 2: offset by 40 pixels to the right
  cv::Mat circle2 = createCircleImage(imageSize, cv::Point(220, 200), radius);

  std::cout << "Created two overlapping circles (offset: 40px)" << std::endl;

  // Perform bitwise operations
  cv::Mat andResult, orResult, xorResult, notResult;

  // AND: Intersection - white only where BOTH images are white
  cv::bitwise_and(circle1, circle2, andResult);
  std::cout << "AND: Intersection of circles" << std::endl;

  // OR: Union - white where EITHER image is white
  cv::bitwise_or(circle1, circle2, orResult);
  std::cout << "OR: Union of circles" << std::endl;

  // XOR: Exclusive OR - white where images DIFFER
  cv::bitwise_xor(circle1, circle2, xorResult);
  std::cout << "XOR: Symmetric difference (non-overlapping regions)" << std::endl;

  // NOT: Inversion of first circle
  cv::bitwise_not(circle1, notResult);
  std::cout << "NOT: Inverted circle 1" << std::endl;

  // Display results
  cv::imshow("Circle 1", circle1);
  cv::imshow("Circle 2", circle2);
  cv::imshow("AND (Intersection)", andResult);
  cv::imshow("OR (Union)", orResult);
  cv::imshow("XOR (Difference)", xorResult);
  cv::imshow("NOT (Inverted)", notResult);

  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
