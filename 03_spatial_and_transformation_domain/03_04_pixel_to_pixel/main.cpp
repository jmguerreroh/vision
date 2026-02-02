/**
 * @file main.cpp
 * @brief Pixel-to-pixel transformations demonstration
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates point operations (pixel-to-pixel transformations)
 * where each output pixel depends only on the corresponding input pixel.
 *
 * Point operations covered:
 * - Inverse (negative): out = 255 - in
 * - Binary threshold: out = (in > T) ? 255 : 0
 *
 * These are the simplest image transformations, useful for:
 * - Contrast enhancement
 * - Image segmentation
 * - Preprocessing for other algorithms
 *
 * General form: g(x,y) = T[f(x,y)]
 * where T is a transformation function applied to each pixel independently.
 *
 * @note Uses ../../data/lena.jpg as input image
 */

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <cmath>

/**
 * @brief Displays usage information
 * @param argv Command line arguments
 */
void printHelp(char ** argv)
{
  std::cout << "\n"
            << "Pixel-to-Pixel Transformations Demo\n"
            << "===================================\n"
            << "This program demonstrates point operations where each output pixel\n"
            << "depends only on the corresponding input pixel.\n\n"
            << "Usage: " << argv[0] << " [image_path]\n"
            << "  image_path: Path to input image (default: lena.jpg)\n\n";
}

/**
 * @brief Apply inverse (negative) transformation to an image
 *
 * The inverse transformation maps each pixel value to its complement:
 *   output = 255 - input
 *
 * This is useful for:
 * - Enhancing white/gray details in dark regions
 * - Medical imaging (e.g., X-rays)
 * - Photographic negatives
 *
 * @param src Input grayscale image (CV_8UC1)
 * @return Output inverted image
 */
cv::Mat applyInverse(const cv::Mat & src)
{
  cv::Mat dst(src.rows, src.cols, src.type());

  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      // Inverse transformation: new_value = 255 - old_value
      dst.at<uchar>(i, j) = 255 - src.at<uchar>(i, j);
    }
  }

  return dst;
}

/**
 * @brief Apply binary threshold transformation to an image
 *
 * The threshold transformation creates a binary image:
 *   output = 255 if input > threshold, else 0
 *
 * This is useful for:
 * - Simple image segmentation
 * - Object detection preprocessing
 * - Document binarization
 *
 * @param src Input grayscale image (CV_8UC1)
 * @param threshold Threshold value (0-255)
 * @return Output binary image
 */
cv::Mat applyThreshold(const cv::Mat & src, int threshold)
{
  cv::Mat dst(src.rows, src.cols, src.type());

  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      // Binary threshold: above threshold -> white, below -> black
      if (src.at<uchar>(i, j) > threshold) {
        dst.at<uchar>(i, j) = 255;
      } else {
        dst.at<uchar>(i, j) = 0;
      }
    }
  }

  return dst;
}

int main(int argc, char ** argv)
{
  printHelp(argv);

  // Load image in grayscale
  const char * filename = argc >= 2 ? argv[1] : "lena.jpg";
  cv::Mat src = cv::imread(cv::samples::findFile(filename), cv::IMREAD_GRAYSCALE);

  if (src.empty()) {
    std::cerr << "Error: Could not load image '" << filename << "'" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "=== Pixel-to-Pixel Transformations ===" << std::endl;
  std::cout << "Image size: " << src.cols << "x" << src.rows << std::endl;

  // Transformation 1: Inverse (Negative)
  std::cout << "\n1. Applying inverse transformation..." << std::endl;
  std::cout << "   Formula: output = 255 - input" << std::endl;
  cv::Mat inverse = applyInverse(src);

  // Transformation 2: Binary Threshold
  const int THRESHOLD_VALUE = 128;
  std::cout << "\n2. Applying binary threshold..." << std::endl;
  std::cout << "   Threshold value: " << THRESHOLD_VALUE << std::endl;
  std::cout << "   Formula: output = (input > " << THRESHOLD_VALUE << ") ? 255 : 0" << std::endl;
  cv::Mat thresholded = applyThreshold(src, THRESHOLD_VALUE);

  // Display results
  cv::imshow("Original", src);
  cv::imshow("Inverse (Negative)", inverse);
  cv::imshow("Binary Threshold", thresholded);

  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
